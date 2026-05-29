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

"""Phase C grow: stage positive intent deltas as planner adds.

For every non-finished stage with positive intent, calls
``ctx.try_add_worker`` up to ``intent`` times. Placement
exhaustion triggers the cross-stage donor fallback driven by
``DonorCoordinator`` under a ``SaturationPolicy`` - the policy
owns the four anti-flap / trust filters, the bounded multi-
donor resource-fit search, the throughput-first economic gate,
and the per-stage cooldown ledger update on commit. The
memory-pressure gate and per-stage ceilings can clamp or freeze
the grow path before the donor flow is consulted.
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
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.dag_priority import compute_grow_priority_order
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.services import GrowServices
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class SaturationGrowPhase:
    """Per-stage positive intent applied as planner adds with donor fallback.

    Walks stages in ``compute_grow_priority_order`` so the
    bottleneck stage grows first when engaged. Cluster placement
    exhaustion triggers the saturation-mode donor fallback; the
    memory-pressure gate and per-stage hard ceiling can clamp
    grow to zero. The runner captures the pre-Phase-C planner
    worker count before invoking this phase and runs the post-
    Phase-C planner invariant + classifier-state NaN gate after
    it returns; both responsibilities live in ``CycleRunner``.
    """

    def run(self, cycle: AutoscaleCycle, services: GrowServices) -> None:
        """Apply positive intent deltas as planner adds, DAG-priority order.

        Walks stages in ``compute_grow_priority_order`` (bottleneck-
        first when engaged) and applies each positive intent through
        the shared :class:`DonorBackedAddExecutor` (one frozen
        instance per scheduler, pre-wired with the saturation-mode
        policy). Placement exhaustion triggers the cross-stage donor
        fallback; on continued failure growth stops with one WARN.
        Advances ``ledgers.stuck_plan`` through its ``record`` entry
        point so the counter dict and the WARN-to-INFO detector stay
        in lockstep. The runner invokes the post-phase planner
        invariant and classifier NaN gate after this method returns.

        Raises:
            SchedulerInvariantError: Surfaced by the executor when a
                probe-approved donor plan's post-commit receiver
                retry returned ``None`` (planner snapshot diverged
                after the atomic remove), or when the coordinator
                returned ``committed=True`` with a ``None`` plan.
                Also raised by the bottleneck-snapshot read when
                Phase C runs before Phase BottleneckPhase.

        Allocation-error tolerance:
            The executor routes both ``try_add_worker`` calls
            (direct and post-donor retry) through the shared
            :func:`allocation_failures.try_add_worker_with_defense`
            wrapper. An :class:`resources.AllocationError` is
            absorbed when ``skip_cycle_on_allocation_error`` is
            ``True`` (the executor returns
            :class:`AllocationAborted` and Grow short-circuits the
            remaining receivers); the kill switch
            ``skip_cycle_on_allocation_error=False`` keeps the
            original behaviour and propagates the
            ``AllocationError`` to the autoscaler thread after the
            ERROR log + counter increment are emitted.

        """
        pipeline = services.pipeline
        problem = pipeline.problem
        ctx = cycle.ctx
        problem_state = cycle.problem_state
        executor = services.donor_executor
        executor.reset_cycle()

        # Cluster-wide memory-pressure kill switch. When the Ray
        # object-store ``used_fraction`` exceeds the configured
        # threshold, every stage's positive intent is frozen for the
        # cycle so the scheduler stops adding to a cluster already
        # approaching OOM. The preflight populates
        # ``cycle.is_memory_pressure_active`` once per cycle so this
        # branch reads a single boolean rather than re-evaluating the
        # config flag and monitor here. The stuck-plan counters are
        # reset to 0 on the freeze path so the post-Phase-D
        # monotonicity check treats this as a ``no-attempt`` cycle
        # (same branch as the existing ``intent <= 0`` reset path);
        # operators still see the cluster-wide pressure via the
        # ``xenna_scheduler_memory_pressure_active`` gauge. Phase A
        # (manual), Phase B (floor), and Phase D (shrink) keep running
        # because Phase B is the only recovery path for a stage at 0
        # workers and Phase D actively relieves pressure by shedding
        # workers.
        if cycle.is_memory_pressure_active:
            for runtime_stage in problem_state.rust.stages:
                if not runtime_stage.is_finished:
                    services.set_stuck_plan_counter(runtime_stage.stage_name, 0, last_intent=0)
            return

        # Defensive: the bottleneck phase is required to populate
        # ``cycle.bottleneck`` before Phase C runs. Accessing
        # ``cycle.bottleneck`` raises ``AttributeError`` when the
        # snapshot is missing, so a future reorder that drops the
        # bottleneck phase fails loud instead of silently making
        # the wrong decision.
        bottleneck = cycle.bottleneck
        d_k_now = bottleneck.d_k_now
        intent_deltas = cycle.intent.deltas
        bottleneck_engaged = pipeline.config.enable_bottleneck_priority_growth and bottleneck.identity.engaged
        stage_order = compute_grow_priority_order(
            problem,
            bottleneck_engaged=bottleneck_engaged,
            d_k_by_stage=d_k_now,
            enable_dag_priority=pipeline.config.enable_dag_priority_growth,
        )

        num_nodes = len(problem.rust.cluster_resources.nodes)
        stage_ceilings = services.ceilings.compute(num_nodes)
        stage_floors = services.floors.compute(num_nodes)
        worker_ids_by_stage = ctx.worker_ids_by_stage()

        for stage_index in stage_order:
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            # Manual stages are owned by Phase A (Manual); skip them so Grow never
            # pushes a saturated manual stage above its requested_num_workers pin
            # (mirrors Floor/Shrink). With the Intent guard this is belt-and-
            # braces - a manual stage already arrives with intent 0 - but the
            # explicit skip keeps the invariant local to Grow. The stuck-plan
            # counter is intentionally left untouched (no set_stuck_plan_counter
            # call): the post-Phase-D monotonicity check filters unchanged
            # counters by value-diff, so a skipped manual stage reads as a
            # no-attempt cycle (same contract as the other Grow early-returns).
            if problem.rust.stages[stage_index].requested_num_workers is not None:
                continue
            stage_name = runtime_stage.stage_name
            intent = intent_deltas.get(stage_name, 0)
            if intent <= 0:
                services.set_stuck_plan_counter(stage_name, 0, last_intent=0)
                continue
            # Hard worker cap: clamp the grow request to the headroom
            # left under ``ceiling = min(max_workers, max_workers_per_node * N)``.
            # The planner refuses excess ``try_add_worker`` calls beyond the
            # cap; an INFO log records the bound so operators can correlate
            # the suppressed growth with the configured cap.
            ceiling = stage_ceilings[stage_index]
            if ceiling is not None:
                current = len(worker_ids_by_stage[stage_index])
                headroom = max(0, ceiling - current)
                if intent > headroom:
                    logger.info(
                        f"saturation-aware scale-up: stage {stage_name!r} intent "
                        f"+{intent} workers; hard worker cap left {headroom} "
                        f"(current={current}, ceiling={ceiling})."
                    )
                    intent = headroom
                if intent <= 0:
                    services.set_stuck_plan_counter(stage_name, 0, last_intent=0)
                    continue
            added = 0
            while added < intent:
                # Grow rebuilds the donor planning context every time
                # the donor branch is reached (``planning_context=None``
                # below): each committed donation atomically mutates
                # ``problem_state.rust.stages`` / ``worker_ids_by_stage``
                # / ``worker_ages``, so a context built before the loop
                # would carry stale donor pools by the next receiver
                # iteration and risk selecting an already-removed
                # worker. The build is amortized: direct-add success
                # short-circuits before the build runs, and only
                # placement-exhausted iterations pay the rebuild cost.
                outcome = executor.execute(
                    cycle=cycle,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    receiver_view=cycle.view_for(stage_index, services.stage_states),
                    receiver_intent=intent - added,
                    stage_floors=stage_floors,
                    pipeline_name=services.pipeline_name,
                    skip_cycle_on_allocation_error=pipeline.config.skip_cycle_on_allocation_error,
                    planning_context=None,
                )
                match outcome:
                    case DirectAddSucceeded() | DonorCommitted():
                        added += 1
                    case AllocationAborted():
                        # The wrapper absorbed an AllocationError; honour
                        # the configured cycle-skip contract and stop
                        # processing every remaining receiver this cycle.
                        return
                    case PlacementExhausted():
                        # No donor plan was feasible (cluster
                        # placement exhausted). Grow treats this as
                        # recoverable: emit one WARN with the
                        # achieved deficit and let the stuck-plan
                        # counter tick.
                        deficit = intent - added
                        logger.warning(
                            f"saturation-aware scale-up: stage {stage_name!r} intent "
                            f"{intent} workers; cluster placement exhausted after "
                            f"{added} (deficit={deficit}); request remains partially "
                            "satisfied this cycle."
                        )
                        break
                    case ProbeFailedAtCommit(acquire_result=acquire_result):
                        # Probe approved the plan at selection time but
                        # the planner snapshot diverged between selection
                        # and commit. Grow treats this as a recoverable
                        # defensive event - one WARN with the planner's
                        # textual reason so operators can distinguish it
                        # from genuine placement exhaustion; the
                        # stuck-plan counter still ticks because the
                        # receiver did not progress this iteration.
                        deficit = intent - added
                        reject_reason = acquire_result.placement_reject_reason
                        logger.warning(
                            f"saturation-aware scale-up: stage {stage_name!r} intent "
                            f"{intent} workers; commit-time probe rejected the donor "
                            f"plan after {added} placement(s) (deficit={deficit}); "
                            f"reason: {reject_reason!r}; request remains partially "
                            "satisfied this cycle."
                        )
                        break
                    case _:
                        assert_never(outcome)
            if added < intent:
                next_count = services.stuck_plan_ledger.get_counter(stage_name) + 1
                services.set_stuck_plan_counter(stage_name, next_count, last_intent=intent)
            else:
                services.set_stuck_plan_counter(stage_name, 0, last_intent=intent)


__all__ = ["SaturationGrowPhase"]
