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

"""Phase B: per-stage minimum-worker floor enforcement.

Brings every non-manual, non-finished stage up to its
``target_min`` floor via ``ctx.try_add_worker``, falling back to
the cross-stage donor driven by ``DonorCoordinator`` under a
``FloorPolicy`` when the cluster is full. No-donor misses
accumulate a stuck-cycle counter; ``RuntimeError`` fires only
past ``floor_stuck_grace_cycles``. See
``docs/scheduler/saturation-aware/`` for the algorithm.
"""

from typing import assert_never

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.executor import (
    AllocationAborted,
    DirectAddSucceeded,
    DonorCommitted,
    PlacementExhausted,
    ProbeFailedAtCommit,
)
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorPlan
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.services import FloorServices
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class FloorPhase:
    """Per-stage minimum-worker floor enforcement with donor fallback.

    Each non-manual, non-finished stage is brought up to
    ``target_min`` via ``ctx.try_add_worker``; on capacity
    exhaustion the cross-stage donor fallback runs once per
    shortfall. No-donor misses accumulate a stuck-cycle counter
    that raises ``RuntimeError`` only past
    ``floor_stuck_grace_cycles``. The post-phase planner-state
    invariant is owned by ``CycleRunner`` via the
    ``PhaseInvariantSuite``.
    """

    def run(self, cycle: AutoscaleCycle, services: FloorServices) -> None:
        """Enforce the per-stage minimum-worker floor on non-manual stages.

        Brings each non-manual non-finished stage up to ``target_min``
        via ``ctx.try_add_worker``; on capacity exhaustion the
        shared :class:`DonorBackedAddExecutor` falls back to the
        floor-mode cross-stage donor once per shortfall. The runner
        invokes the post-phase planner invariant after this method
        returns.

        Raises:
            RuntimeError: Floor unsatisfied past
                ``floor_stuck_grace_cycles`` (operator action) or
                commit-time probe divergence (Floor treats as
                operator-actionable).
            SchedulerInvariantError: Surfaced by the executor when
                a probe-approved donor plan's post-commit receiver
                retry returned ``None`` (planner snapshot diverged
                after the atomic remove) or when the coordinator
                returned ``committed=True`` with a ``None`` plan.
            IndexError: Planner rejected the stage index.

        Allocation-error tolerance:
            The executor routes both ``try_add_worker`` calls
            (direct and post-donor retry) through the shared
            :func:`allocation_failures.try_add_worker_with_defense`
            wrapper. An :class:`resources.AllocationError` is
            absorbed when ``skip_cycle_on_allocation_error`` is
            ``True`` (the executor returns
            :class:`AllocationAborted` and Floor short-circuits the
            cycle); the kill switch
            ``skip_cycle_on_allocation_error=False`` keeps the
            original behaviour and propagates the
            ``AllocationError`` to the autoscaler thread after the
            ERROR log + counter increment are emitted.

        """
        pipeline = services.pipeline
        problem = pipeline.problem
        ctx = cycle.ctx
        executor = services.donor_executor
        # Per-cycle reset of the Floor-phase allocation-failure gate.
        # The gate now lives on the executor; ``reset_cycle`` clears
        # ``aborted_cycle`` so absorbed AllocationErrors from the
        # previous cycle do not bleed into this Floor run.
        executor.reset_cycle()
        num_nodes = problem.rust.cluster_resources.num_nodes()
        stage_floors = services.floors.compute(num_nodes)
        for stage_index, problem_stage in enumerate(problem.rust.stages):
            if problem_stage.requested_num_workers is not None:
                # Manual stages have their worker count owned by the manual-shrink/grow path.
                continue
            runtime_stage = cycle.problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            target_min = stage_floors[stage_index]
            current = len(ctx.worker_ids_by_stage()[stage_index])
            stage_cfg = pipeline.stage_config(problem_stage.name)
            made_progress = False
            stuck = False
            while current < target_min:
                # Floor rebuilds the donor planning context every time
                # the donor path is reached so the
                # ``worker_ids_by_stage`` / ``worker_ages`` snapshots
                # reflect every prior donation committed earlier in
                # the phase (the planner mutates between iterations).
                # Passing ``planning_context=None`` defers the build
                # into the executor's donor branch so direct-add
                # success pays nothing.
                outcome = executor.execute(
                    cycle=cycle,
                    stage_index=stage_index,
                    stage_name=problem_stage.name,
                    receiver_view=cycle.view_for(stage_index, services.stage_states),
                    receiver_intent=target_min - current,
                    stage_floors=stage_floors,
                    pipeline_name=services.pipeline_name,
                    skip_cycle_on_allocation_error=pipeline.config.skip_cycle_on_allocation_error,
                    planning_context=None,
                )
                match outcome:
                    case DirectAddSucceeded() | DonorCommitted():
                        current += 1
                        made_progress = True
                    case AllocationAborted():
                        # The wrapper absorbed an AllocationError; honour the
                        # configured cycle-skip contract and stop floor work
                        # for the rest of this cycle. The next cycle re-
                        # evaluates against the (possibly freed) cluster.
                        return
                    case PlacementExhausted():
                        # Partial progress (some workers added before donor
                        # exhaustion) is not a hard failure regardless of
                        # ``floor_stuck_grace_cycles``: when grace > 0 the
                        # operator sees a soft WARN; when grace == 0 the
                        # cycle silently accepts the partial commit. Only a
                        # fully stuck floor (no progress AND no donor plan)
                        # trips the stuck latch, which gates the long-running
                        # operator WARN / floor-unmet RuntimeError after
                        # ``floor_stuck_grace_cycles``.
                        if made_progress:
                            if pipeline.config.floor_stuck_grace_cycles > 0:
                                _warn_floor_partial_progress(
                                    stage_name=problem_stage.name,
                                    target_min=target_min,
                                    current=current,
                                )
                        else:
                            _on_floor_stuck(
                                services=services,
                                stage_name=problem_stage.name,
                                target_min=target_min,
                                current=current,
                                stage_cfg=stage_cfg,
                                num_nodes=num_nodes,
                            )
                            stuck = True
                        break
                    case ProbeFailedAtCommit(acquire_result=acquire_result):
                        # The probe approved the plan at selection time but
                        # the planner snapshot drifted between selection and
                        # commit (cluster mutated mid-cycle). Floor treats
                        # this as operator-actionable.
                        msg = _format_floor_unmet_message(
                            stage_name=problem_stage.name,
                            target_min=target_min,
                            current=current,
                            stage_cfg=stage_cfg,
                            num_nodes=num_nodes,
                            donor_attempted=True,
                            donor_plan=acquire_result.attempted_plan,
                            placement_reject_reason=acquire_result.placement_reject_reason,
                        )
                        raise RuntimeError(msg)
                    case _:
                        assert_never(outcome)
            if made_progress or not stuck:
                # Floor satisfied this cycle (either by direct add, by donation, or
                # because target_min was already met at entry), or the receiver made
                # partial progress; reset the stuck counter.
                services.floor_stuck_counters.reset_for(problem_stage.name)


def _warn_floor_partial_progress(
    *,
    stage_name: str,
    target_min: int,
    current: int,
) -> None:
    """Warn when a floor is still short but the receiver grew this cycle."""
    logger.warning(
        f"[scheduler] {stage_name!r}: minimum-worker floor partially satisfied "
        f"(target_min={target_min}, achieved={current}, no eligible cross-stage donor); "
        "stuck counter reset because the receiver made forward progress."
    )


def _on_floor_stuck(
    *,
    services: FloorServices,
    stage_name: str,
    target_min: int,
    current: int,
    stage_cfg: SaturationAwareStageConfig,
    num_nodes: int,
) -> None:
    """Account for a single stuck cycle and either raise or warn.

    Increments the per-stage stuck counter; raises
    ``RuntimeError`` past ``floor_stuck_grace_cycles``, otherwise
    emits one WARN per stuck cycle.

    Raises:
        RuntimeError: Counter exceeded the grace; message names
            the stage, floor, and the operator action.

    """
    counter = services.floor_stuck_counters.increment_stuck(stage_name)
    grace = services.pipeline.config.floor_stuck_grace_cycles
    if counter > grace:
        msg = _format_floor_unmet_message(
            stage_name=stage_name,
            target_min=target_min,
            current=current,
            stage_cfg=stage_cfg,
            num_nodes=num_nodes,
            donor_attempted=False,
            donor_plan=None,
        )
        raise RuntimeError(msg)
    remaining = grace - counter
    logger.warning(
        f"[scheduler] {stage_name!r}: minimum-worker floor stuck "
        f"({counter}/{grace} grace cycles); target_min={target_min}, "
        f"achieved={current}, no eligible cross-stage donor; will raise after "
        f"{remaining} more consecutive failed cycles."
    )


def _format_floor_unmet_message(
    *,
    stage_name: str,
    target_min: int,
    current: int,
    stage_cfg: SaturationAwareStageConfig,
    num_nodes: int,
    donor_attempted: bool,
    donor_plan: DonorPlan | None,
    placement_reject_reason: str = "",
) -> str:
    """Build the operator-actionable message for an unmet minimum-worker floor.

    Distinguishes the two failure modes (no eligible donor vs.
    donor plan selected but probe still returned infeasible).
    When a plan was selected, lists every donor worker so
    operators can correlate the failure with the donation log
    line; multi-worker plans are flattened to the same
    ``(stage_index, worker_id)`` list the planner contracts use.
    ``placement_reject_reason`` surfaces the planner-supplied
    textual reason from the commit-time probe
    (``worker_not_found`` / ``release_failed`` /
    ``no_placement``) so operators can correlate the failure
    with the coordinator's structured rejection log.
    """
    if donor_attempted:
        donor_label = (
            f" (removals={[(w.stage_index, w.worker_id) for w in donor_plan.removals]!r})"
            if donor_plan is not None
            else ""
        )
        reason_clause = (
            f"; planner placement_reject_reason={placement_reject_reason!r}" if placement_reject_reason else ""
        )
        donor_clause = (
            "donor fallback attempted but commit-time probe rejected the plan"
            f"{donor_label}{reason_clause}; the planner snapshot diverged "
            "between selection and commit"
        )
    else:
        donor_clause = "no eligible cross-stage donor (every other stage at its own floor)"
    return (
        f"Minimum-worker floor for stage {stage_name!r} cannot be satisfied: "
        f"target_min={target_min} (achieved={current}; from "
        f"min_workers={stage_cfg.min_workers}, "
        f"min_workers_per_node={stage_cfg.min_workers_per_node}, "
        f"num_nodes={num_nodes}). Cluster placement exhausted and "
        f"{donor_clause}. Reduce min_workers / min_workers_per_node, "
        "or scale up the cluster."
    )


__all__ = ["FloorPhase"]
