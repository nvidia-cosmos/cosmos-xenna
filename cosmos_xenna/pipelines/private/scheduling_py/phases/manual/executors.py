# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-stage delete / grow executors for the manual phase.

Two stateless ``@attrs.frozen`` Strategy bundles:

- :class:`ManualDeleteExecutor` deletes surplus workers from any
  manual stage whose ``requested_num_workers`` is below the live
  count. Youngest workers are deleted first so long-lived warmed
  workers survive a manual shrink.
- :class:`ManualGrowExecutor` brings any manual stage whose
  ``requested_num_workers`` is above the live count up to the
  request through ``ctx.try_add_worker``; cluster placement
  exhaustion stops growth without raising and emits one WARN per
  affected stage. ``AllocationError`` absorption follows the same
  contract as the saturation / floor paths via the shared
  :func:`try_add_worker_with_defense` wrapper.

The two executors live next to :class:`ManualPhase` and are
injected as ``@attrs.frozen`` fields on the phase so the entry
point (``ManualPhase.run``) is a two-line orchestration shell.

"""

from typing import TYPE_CHECKING

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.allocation_failure_gate import (
    try_add_worker_with_defense,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state.allocation_failure_gate import AllocationFailureGate
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.utils import python_log as logger

if TYPE_CHECKING:
    from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.services import ManualServices


def _select_workers_to_delete_youngest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
) -> list[str]:
    """Pick ``delete_count`` workers to delete, youngest first.

    Sort key ``(age ASC, worker_id ASC)``; missing ages default
    to 0; the id tiebreaker keeps the choice deterministic.

    Args:
        worker_ids: Worker ids in the stage's snapshot.
        worker_ages: Cluster-wide worker ages.
        delete_count: Number to return; clamped to ``len(worker_ids)``.

    Returns:
        Up to ``delete_count`` worker ids, youngest first.

    """
    ranked = sorted(
        ((worker_ages.get(wid, 0), wid) for wid in worker_ids),
        key=lambda pair: (pair[0], pair[1]),
    )
    return [wid for _, wid in ranked[:delete_count]]


@attrs.frozen
class ManualDeleteExecutor:
    """Delete surplus workers on manual stages whose request is below current.

    Stateless ``@attrs.frozen`` strategy. The per-stage deletion
    selects youngest workers first via
    :func:`_select_workers_to_delete_youngest_first` so long-lived
    warmed workers survive a manual shrink.

    """

    def execute(self, *, cycle: AutoscaleCycle, services: "ManualServices") -> None:
        """Walk manual stages and delete every surplus worker.

        Raises:
            SchedulerInvariantError: ``ctx.try_remove_worker``
                returned ``False`` for a worker present in
                ``problem_state`` (planner snapshot inconsistency).

        """
        problem = services.pipeline.problem
        ctx = cycle.ctx
        problem_state = cycle.problem_state
        worker_ages = ctx.worker_ages()
        for stage_index, problem_stage in enumerate(problem.rust.stages):
            requested = problem_stage.requested_num_workers
            if requested is None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            current = len(runtime_stage.worker_groups)
            if current <= requested:
                continue
            delete_count = current - requested
            worker_ids = [w.id for w in runtime_stage.worker_groups]
            victims = _select_workers_to_delete_youngest_first(
                worker_ids=worker_ids,
                worker_ages=worker_ages,
                delete_count=delete_count,
            )
            for worker_id in victims:
                if not ctx.try_remove_worker(stage_index, worker_id):
                    msg = (
                        f"Manual-shrink: try_remove_worker(stage_index={stage_index}, "
                        f"worker_id={worker_id!r}) returned False on stage "
                        f"{problem_stage.name!r}; the worker was present in problem_state "
                        "but unknown to the planner - snapshot inconsistency."
                    )
                    raise SchedulerInvariantError(msg)


@attrs.frozen
class ManualGrowExecutor:
    """Grow manual stages whose ``requested_num_workers`` exceeds current.

    Stateless ``@attrs.frozen`` strategy. Manual stages do NOT
    donate, so there is no donor fallback - the executor wraps
    ``try_add_worker_with_defense`` and emits a WARN per stage on
    placement exhaustion. Allocation-error absorption follows the
    same contract as the floor / saturation paths; the gate is
    owned by the executor so the manual phase stays free of
    cross-cycle ledger access.

    Attributes:
        allocation_gate: Cross-cycle allocation-failure gate; reset
            at the top of every ``execute()`` call.

    """

    allocation_gate: AllocationFailureGate

    def execute(self, *, cycle: AutoscaleCycle, services: "ManualServices") -> None:
        """Walk manual stages and bring the worker count up to the request.

        ``AllocationError`` absorption: when
        ``skip_cycle_on_allocation_error=True`` and the per-stage
        gate trips, the executor stops processing manual stages
        for the rest of the cycle (the next cycle re-evaluates
        against any freed capacity).

        Raises:
            IndexError: Planner rejected a stage index.

        """
        pipeline = services.pipeline
        problem = pipeline.problem
        ctx = cycle.ctx
        problem_state = cycle.problem_state
        self.allocation_gate.reset()
        for stage_index, problem_stage in enumerate(problem.rust.stages):
            requested = problem_stage.requested_num_workers
            if requested is None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            current = len(runtime_stage.worker_groups)
            while current < requested:
                if (
                    try_add_worker_with_defense(
                        ctx=ctx,
                        stage_index=stage_index,
                        stage_name=problem_stage.name,
                        pipeline_name=services.pipeline_name,
                        skip_cycle_on_allocation_error=pipeline.config.skip_cycle_on_allocation_error,
                        gate=self.allocation_gate,
                    )
                    is None
                ):
                    if self.allocation_gate.aborted_cycle:
                        return
                    deficit = requested - current
                    logger.warning(
                        f"manual grow: stage {problem_stage.name!r} requested "
                        f"{requested} workers; cluster placement exhausted at "
                        f"{current} (deficit={deficit}); manual request remains "
                        "partially satisfied this cycle."
                    )
                    break
                current += 1


__all__ = ("ManualDeleteExecutor", "ManualGrowExecutor")
