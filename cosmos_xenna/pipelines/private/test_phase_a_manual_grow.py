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

"""Behaviour tests for the manual-stage grow path.

Pin the contract:

  1. A manual stage (``requested_num_workers is not None``) with
     ``current < requested`` grows to exactly ``requested`` when the
     cluster has room.
  2. When the working cluster has no remaining placement, growth
     stops for that stage in this cycle without raising, leaving the
     request partially satisfied.
  3. Finished stages, non-manual stages, and stages already at or
     above the request are not touched.
"""

from typing import cast
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def _make_cycle(
    ctx: data_structures.AutoscalePlanContext,
    problem_state: data_structures.ProblemState,
) -> AutoscaleCycle:
    """Build a minimal ``AutoscaleCycle`` for direct phase-method tests."""
    return AutoscaleCycle(
        ctx=ctx,
        problem_state=problem_state,
        time=0.0,
        cycle_counter=0,
        pipeline_name="",
    )


def _cluster(total_cpus: int = 16) -> resources.ClusterResources:
    """Single-node cluster sized to fit every test fixture."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=total_cpus, gpus=[], name="node-0"),
        },
    )


def _problem_with_requested(
    stage_specs: list[tuple[str, int | None]],
    *,
    total_cpus: int = 16,
) -> data_structures.Problem:
    """Build a ``Problem`` whose stages carry the given ``requested_num_workers``."""
    cluster = _cluster(total_cpus=total_cpus)
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=requested,
            over_provision_factor=None,
        )
        for name, requested in stage_specs
    ]
    return data_structures.Problem(cluster, stages)


def _make_workers(stage_name: str, count: int) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers on ``node-0`` named ``<stage>-w<i>``."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(name, num_workers, slots_per_worker, is_finished)``."""
    states = [
        data_structures.ProblemStageState(
            stage_name=name,
            workers=_make_workers(name, num_workers),
            slots_per_worker=slots,
            is_finished=finished,
        )
        for name, num_workers, slots, finished in stage_specs
    ]
    return data_structures.ProblemState(states)


class _RaisingAddContext:
    """Fake planner that surfaces a hard placement-context failure.

    Exposes the minimal surface ``ManualGrowExecutor.execute`` reaches:
    ``try_add_worker`` (the call site under test) and
    ``cluster_snapshot`` (consumed by the shared
    :mod:`allocation_failures` defense layer when it absorbs an
    :class:`resources.AllocationError`). The snapshot is fixed
    cluster geometry; the test only asserts on the absorb / re-raise
    contract, not on the snapshot content.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[int] = []
        self._exc = exc or RuntimeError("planner context is drained")

    def try_add_worker(self, stage_index: int) -> data_structures.ProblemWorkerGroupState | None:
        """Raise the same exception type a corrupted planner context would raise."""
        self.calls.append(stage_index)
        raise self._exc

    def cluster_snapshot(self) -> resources.ClusterResources:
        """Return a stable cluster snapshot for the absorb path's diagnostic log."""
        return _cluster()


class TestManualGrowScheduler:
    """End-to-end manual grow via ``SaturationAwareScheduler.autoscale``."""

    def test_manual_grow_emits_new_workers_for_shortfall(self) -> None:
        """1-worker manual stage with ``requested=4`` grows to 4 (3 new workers)."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 4)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 1, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 3
        assert solution.stages[0].deleted_workers == []

    def test_grow_from_zero_workers(self) -> None:
        """A cold-start manual stage with ``current=0`` is brought up to ``requested``."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 3)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 3

    def test_finished_stages_are_skipped(self) -> None:
        """A stage flagged finished is not grown regardless of request shortfall."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 4)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 1, 1, True)]),
        )
        assert solution.stages[0].new_workers == []

    def test_non_manual_stages_are_skipped(self) -> None:
        """``requested_num_workers=None`` makes the stage non-manual; manual grow skips it.

        Start at ``current=2`` so the stage is already above the implicit
        Phase B floor; this isolates the manual-grow contract from the
        floor enforcement that runs later in the same cycle.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 2, 1, False)]),
        )
        assert solution.stages[0].new_workers == []

    def test_no_op_when_current_equals_requested(self) -> None:
        """``current == requested`` produces no delta on a manual stage."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 3)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False)]),
        )
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_no_op_when_current_above_requested(self) -> None:
        """``current > requested`` is a manual-shrink case; the grow path is a no-op."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 1)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False)]),
        )
        # Shrink path produces deletes; grow path adds nothing.
        assert solution.stages[0].new_workers == []

    def test_grow_stops_at_capacity_without_raising(self) -> None:
        """A request that exceeds cluster capacity stops short, no exception raised."""
        # Cluster has room for 4 1-CPU workers; stage already has 1; requested=10.
        # Growth should add at most (4 - 1) = 3 new workers, then stop.
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 10)], total_cpus=4))
        with patch("cosmos_xenna.pipelines.private.scheduling_py.phases.manual.executors.logger.warning") as warning:
            solution = scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 1, 1, False)]),
            )
        assert len(solution.stages[0].new_workers) == 3
        assert solution.stages[0].deleted_workers == []
        warning.assert_called_once_with(
            "manual grow: stage 'A' requested 10 workers; cluster placement exhausted at 4 "
            "(deficit=6); manual request remains partially satisfied this cycle."
        )

    def test_partial_grow_when_other_stages_consume_capacity(self) -> None:
        """Two manual stages compete for capacity; first stage fully grows, second partially."""
        # Cluster sized to 6 CPUs; stage A asks for 4 (currently 0); stage B asks for 4 (currently 0).
        # Expected: A grows to 4 first (slots are claimed in stage order), B can only add 2.
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 4), ("B", 4)], total_cpus=6))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False), ("B", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 4
        assert len(solution.stages[1].new_workers) == 2

    def test_only_manual_stage_grows_in_mixed_pipeline(self) -> None:
        """A pipeline with one manual + one auto stage grows only the manual one."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("manual", 3), ("auto", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state(
                [("manual", 1, 1, False), ("auto", 1, 1, False)],
            ),
        )
        assert len(solution.stages[0].new_workers) == 2
        assert solution.stages[1].new_workers == []

    def test_requested_zero_is_a_no_op_when_current_is_zero(self) -> None:
        """``requested=0`` and ``current=0`` produce no delta."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 0)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_shrink_frees_capacity_for_subsequent_grow_in_same_cycle(self) -> None:
        """Stage A's shrink releases capacity that stage B's grow consumes in the same cycle.

        Cluster has room for 5 1-CPU workers. A starts at 4 workers and is
        shrunk to ``requested=1`` (frees 3 CPUs). B starts at 1 worker and
        wants ``requested=4`` (needs +3 CPUs). The delete-before-grow ordering
        inside ``autoscale`` makes the freed CPUs available to B in the same
        cycle; without that ordering, B would see 0 free CPUs and partially
        fail.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 1), ("B", 4)], total_cpus=5))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False), ("B", 1, 1, False)]),
        )
        assert len(solution.stages[0].deleted_workers) == 3
        assert len(solution.stages[1].new_workers) == 3

    def test_planner_exception_propagates(self) -> None:
        """Hard planner failures are not downgraded to best-effort capacity exhaustion."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 2)]))
        ctx = _RaisingAddContext()

        with pytest.raises(RuntimeError, match="planner context is drained"):
            scheduler.runner.manual_services.grow_executor.execute(
                cycle=_make_cycle(
                    cast(data_structures.AutoscalePlanContext, ctx),
                    _problem_state([("A", 0, 1, False)]),
                ),
                services=scheduler.runner.manual_services,
            )

        assert ctx.calls == [0]

    def test_planner_index_error_propagates(self) -> None:
        """Planner index errors are not downgraded to capacity exhaustion."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 2)]))
        ctx = _RaisingAddContext(IndexError("stage_index 0 out of range"))

        with pytest.raises(IndexError, match="stage_index 0 out of range"):
            scheduler.runner.manual_services.grow_executor.execute(
                cycle=_make_cycle(
                    cast(data_structures.AutoscalePlanContext, ctx),
                    _problem_state([("A", 0, 1, False)]),
                ),
                services=scheduler.runner.manual_services,
            )

        assert ctx.calls == [0]


class TestManualGrowAllocationErrorDefense:
    """Pin the Phase A side of the shared allocation-failure defense layer.

    The wrapper in
    :func:`cosmos_xenna.pipelines.private.scheduling_py.cluster.allocation_failures.try_add_worker_with_defense`
    catches only :class:`resources.AllocationError`, leaving every
    other exception type to propagate (validated by
    ``test_planner_exception_propagates`` and
    ``test_planner_index_error_propagates`` in the suite above).
    These tests pin Phase A's contract for the absorb-vs-re-raise
    branch and the early-return semantics that protect subsequent
    manual stages from a cycle-wide cluster outage.
    """

    def test_allocation_error_is_absorbed_with_default_skip_switch(self) -> None:
        """An ``AllocationError`` is absorbed when ``skip_cycle_on_allocation_error`` is on (default).

        The wrapper logs the failure, increments the per-stage
        counter, sets
        :attr:`ManualServices.manual_allocation.aborted_cycle`,
        and Phase A returns without re-raising so the rest of the
        autoscale cycle can run.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 2)]))
        ctx = _RaisingAddContext(resources.AllocationError("synthetic placement failure"))

        scheduler.runner.manual_services.grow_executor.execute(
            cycle=_make_cycle(
                cast(data_structures.AutoscalePlanContext, ctx),
                _problem_state([("A", 0, 1, False)]),
            ),
            services=scheduler.runner.manual_services,
        )

        assert ctx.calls == [0]
        assert scheduler.runner.manual_services.grow_executor.allocation_gate.aborted_cycle is True

    def test_allocation_error_propagates_when_skip_switch_off(self) -> None:
        """``skip_cycle_on_allocation_error=False`` re-raises the ``AllocationError``.

        The wrapper still emits the ERROR log + counter increment
        before re-raising, but it does NOT set the cycle-skip flag
        (which is reserved for the absorb path); the autoscaler
        thread sees the original exception.
        """
        cfg = SaturationAwareConfig(skip_cycle_on_allocation_error=False)
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem_with_requested([("A", 2)]))
        ctx = _RaisingAddContext(resources.AllocationError("synthetic placement failure"))

        with pytest.raises(resources.AllocationError, match="synthetic placement failure"):
            scheduler.runner.manual_services.grow_executor.execute(
                cycle=_make_cycle(
                    cast(data_structures.AutoscalePlanContext, ctx),
                    _problem_state([("A", 0, 1, False)]),
                ),
                services=scheduler.runner.manual_services,
            )

        assert ctx.calls == [0]
        assert scheduler.runner.manual_services.grow_executor.allocation_gate.aborted_cycle is False

    def test_allocation_error_absorb_skips_subsequent_manual_stages(self) -> None:
        """Absorbing on stage A short-circuits manual grow for the rest of the cycle.

        With two manual stages requesting workers and the planner
        raising ``AllocationError`` on every call, the wrapper
        absorbs the first stage's failure and Phase A returns
        before iterating to the second stage; the planner is
        therefore called exactly once.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 2), ("B", 2)]))
        ctx = _RaisingAddContext(resources.AllocationError("synthetic placement failure"))

        scheduler.runner.manual_services.grow_executor.execute(
            cycle=_make_cycle(
                cast(data_structures.AutoscalePlanContext, ctx),
                _problem_state([("A", 0, 1, False), ("B", 0, 1, False)]),
            ),
            services=scheduler.runner.manual_services,
        )

        assert ctx.calls == [0], (
            "Phase A must short-circuit after absorbing the first AllocationError "
            "rather than iterating into stage B's grow loop"
        )
        assert scheduler.runner.manual_services.grow_executor.allocation_gate.aborted_cycle is True
