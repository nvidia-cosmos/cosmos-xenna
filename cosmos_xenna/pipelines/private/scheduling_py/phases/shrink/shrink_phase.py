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

"""Phase D shrink: remove workers for negative intent or hard-cap overflow.

Walks every non-finished, non-manual stage and computes the
per-cycle shrink as ``max(-intent, ceiling_excess)``, clamped by
floor, fraction cap, and donor-warmup grace. Manual stages are
skipped so Phase A stays the single source of truth for them.
See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

import math

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.gpu_fraction_map import (
    aggregate_host_gpu_used_fractions,
    project_stage_worker_fractions,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.scale_down import (
    select_workers_to_remove_oldest_first,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.services import ShrinkServices
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class SaturationShrinkPhase:
    """Per-cycle worker removal driven by negative intent and ceiling overflow.

    Skips manual stages so Phase A remains the single source of
    truth for them. The engaged bottleneck stage is protected from
    intent-driven shrink when ``enable_bottleneck_shrink_protection``
    is on; ceiling overflow always bypasses the gate.

    The runner captures the pre-Phase-D planner worker count before
    invoking this phase and runs the post-Phase-D planner invariant +
    floor invariant + executed-delta recording after it returns;
    each post-phase concern lives in ``CycleRunner``.
    """

    def run(self, cycle: AutoscaleCycle, services: ShrinkServices) -> None:
        """Remove workers via the planner to satisfy negative intent or hard-cap overflow.

        Per-stage shrink request is
        ``max(-intent if intent < 0 else 0, ceiling_excess)``,
        clamped by floor / fraction-cap / classifier magnitude cap.
        Manual stages skip Phase D. Bottleneck shrink protection
        skips the engaged bottleneck on intent-driven shrink;
        ceiling overflow always bypasses the gate. The runner
        invokes the post-phase planner / floor invariants and
        the executed-delta recording after this method returns.

        Raises:
            SchedulerInvariantError: Bottleneck phase did not populate
                ``cycle.bottleneck``; planner rejected a worker id from
                its own snapshot.

        """
        pipeline = services.pipeline
        problem = pipeline.problem

        # Defensive: the bottleneck phase is required to populate the
        # cycle's bottleneck snapshot before Phase D runs. Accessing
        # ``cycle.bottleneck`` raises ``AttributeError`` so a future
        # reorder that drops the bottleneck phase fails loud instead
        # of silently making the wrong decision.
        bottleneck = cycle.bottleneck
        bottleneck_meta = bottleneck.identity
        intent_deltas = cycle.intent.deltas
        donor_warmup_excluded_ids = cycle.donor_warmup_excluded_ids

        ctx = cycle.ctx
        problem_state = cycle.problem_state
        num_nodes = len(problem.rust.cluster_resources.nodes)
        stage_floors = services.floors.compute(num_nodes)
        stage_ceilings = services.ceilings.compute(num_nodes)
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        worker_ages = ctx.worker_ages()
        host_gpu_used_fractions = aggregate_host_gpu_used_fractions(problem_state)

        # Once-per-streak debounce for the Shrink bottleneck-protection
        # INFO log. ``services.bottleneck_protection`` owns the
        # previous-cycle snapshot; the phase reads it via ``maybe_log``
        # (which fires only on transition cycles) and replaces it at
        # the tail so a stage that drops out of protection and later
        # re-enters re-arms a fresh INFO log.
        bottleneck_protection = services.bottleneck_protection
        currently_protected: set[str] = set()

        for stage_index, problem_stage in enumerate(problem.rust.stages):
            if problem_stage.requested_num_workers is not None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            stage_name = problem_stage.name
            intent = intent_deltas.get(stage_name, 0)
            current = len(worker_ids_by_stage[stage_index])
            ceiling = stage_ceilings[stage_index]
            ceiling_excess = max(0, current - ceiling) if ceiling is not None else 0
            # Bottleneck shrink protection: an engaged bottleneck stage
            # whose intent is negative (transient idle from an upstream
            # stall, brief slot drop, or model reload) is NOT shrunk on
            # this cycle because re-growing it after recovery would pay
            # the full ``worker_warmup_measurement_grace_s`` window of
            # warmup, capping pipeline throughput during the ramp.
            # Ceiling overflow (``ceiling_excess > 0``) always bypasses
            # the gate; operator-driven shrink via
            # ``requested_num_workers`` is filtered out higher up.
            if (
                pipeline.config.enable_bottleneck_shrink_protection
                and bottleneck_meta.engaged
                and stage_name == bottleneck_meta.stage_name
                and intent < 0
                and ceiling_excess == 0
            ):
                # Log only on the cycle the stage transitions into the
                # protection set so steady-state heterogeneous workloads
                # see one INFO line per protection event, not one per
                # cycle.
                bottleneck_protection.maybe_log(
                    stage_name=stage_name,
                    intent=intent,
                    bottleneck_meta=bottleneck_meta,
                )
                currently_protected.add(stage_name)
                continue
            # Combine the two shrink drivers: negative intent and
            # hard-cap overflow. The cap dominates non-negative intent
            # (forced shrink); negative intent dominates a smaller cap
            # excess (operator wants more shrink than the cap alone).
            requested_remove = max(-intent if intent < 0 else 0, ceiling_excess)
            if requested_remove == 0:
                continue

            floor = stage_floors[stage_index]
            stage_cfg = pipeline.stage_config(stage_name)
            allowed_by_floor = max(0, current - floor)
            fraction_cap = (
                max(1, math.floor(current * stage_cfg.max_scale_down_fraction_per_cycle)) if current > 0 else 0
            )
            actual_remove = min(requested_remove, allowed_by_floor, fraction_cap)
            if actual_remove == 0:
                # The zero-removal log only fires when the operator-configured
                # cap was actively pressing (``ceiling_excess > 0``) and was
                # blocked by the floor or fraction cap. Pure intent-at-floor
                # is steady state and stays silent so the log stream does not
                # repeat once a stage settles at ``min_workers``.
                if ceiling_excess > 0:
                    _log_phase_d_shrink_outcome(
                        stage_name=stage_name,
                        intent=intent,
                        ceiling=ceiling,
                        ceiling_excess=ceiling_excess,
                        requested_remove=requested_remove,
                        actual_remove=actual_remove,
                        # actual_remove == 0 means the floor / fraction cap clamped before the
                        # warmup-grace filter would even run; nothing was effectively removed
                        # and warmup did not contribute to the deficit.
                        effective_remove=0,
                        warmup_excluded_count=0,
                        current=current,
                        floor=floor,
                        fraction_cap=fraction_cap,
                        allowed_by_floor=allowed_by_floor,
                        max_scale_down_fraction_per_cycle=stage_cfg.max_scale_down_fraction_per_cycle,
                    )
                continue

            worker_used_slots = {wg.id: wg.num_used_slots for wg in runtime_stage.worker_groups}
            worker_host_gpu_used_fractions = project_stage_worker_fractions(
                runtime_stage=runtime_stage,
                host_gpu_used_fractions=host_gpu_used_fractions,
            )
            stage_warmup_excluded = sum(
                1 for wid in worker_ids_by_stage[stage_index] if wid in donor_warmup_excluded_ids
            )
            victims = select_workers_to_remove_oldest_first(
                worker_ids=worker_ids_by_stage[stage_index],
                worker_ages=worker_ages,
                delete_count=actual_remove,
                worker_used_slots=worker_used_slots,
                worker_host_gpu_used_fractions=worker_host_gpu_used_fractions,
                excluded_worker_ids=donor_warmup_excluded_ids,
            )
            for victim_id in victims:
                if not ctx.try_remove_worker(stage_index, victim_id):
                    msg = (
                        f"Phase D shrink: stage {stage_name!r} planner refused removal of "
                        f"worker {victim_id!r} selected from its own snapshot. This is a "
                        "scheduler defect; the planner state and the runtime snapshot disagree."
                    )
                    raise SchedulerInvariantError(msg)
            # ``effective_remove`` is the deletion count actually applied to the planner.
            # When the warmup-grace filter removed candidates from the eligible pool the helper
            # returns fewer victims than ``actual_remove``; logging ``actual_remove`` in that
            # case would mislead operators about how many workers were really shrunk and
            # mis-attribute the deficit to floor / fraction caps that did not bind.
            effective_remove = len(victims)
            _log_phase_d_shrink_outcome(
                stage_name=stage_name,
                intent=intent,
                ceiling=ceiling,
                ceiling_excess=ceiling_excess,
                requested_remove=requested_remove,
                actual_remove=actual_remove,
                effective_remove=effective_remove,
                warmup_excluded_count=stage_warmup_excluded,
                current=current,
                floor=floor,
                fraction_cap=fraction_cap,
                allowed_by_floor=allowed_by_floor,
                max_scale_down_fraction_per_cycle=stage_cfg.max_scale_down_fraction_per_cycle,
            )

        # Replace the previous-cycle protection ledger with this cycle's
        # snapshot so a stage that drops out of protection on the next
        # cycle and later re-enters re-arms the once-per-streak INFO log.
        bottleneck_protection.replace_snapshot(currently_protected)


def _log_phase_d_shrink_outcome(
    *,
    stage_name: str,
    intent: int,
    ceiling: int | None,
    ceiling_excess: int,
    requested_remove: int,
    actual_remove: int,
    effective_remove: int,
    warmup_excluded_count: int,
    current: int,
    floor: int,
    fraction_cap: int,
    allowed_by_floor: int,
    max_scale_down_fraction_per_cycle: float,
) -> None:
    """Emit the per-cycle Phase D outcome log distinguishing the binding clamp.

    The preamble names the dominant driver (classifier intent vs
    hard-cap overflow); the trailing clause names the binding
    clamp (floor / fraction cap / donor warmup grace / no clamp).
    Floor wins ties against ``fraction_cap``.

    ``actual_remove`` is the post-clamp request before the
    warmup-grace filter; ``effective_remove`` is the count
    actually applied (``len(victims)``). When both clamps bind
    both branches log so the operator sees the full attribution;
    the cap-driven full-removal branch is suppressed in that case
    to avoid contradicting the deficit lines.
    """
    cap_driven = ceiling_excess > 0 and (intent >= 0 or ceiling_excess >= -intent)
    if cap_driven:
        preamble = (
            f"saturation-aware scale-down: stage {stage_name!r} hard worker cap "
            f"overflow requested {requested_remove} workers"
        )
        cap_kwargs = f", ceiling={ceiling}, intent={intent}"
    else:
        preamble = f"saturation-aware scale-down: stage {stage_name!r} intent -{requested_remove} workers"
        cap_kwargs = ""
    deficit_reported = False
    # Stage 1: actual_remove (post-clamp request) vs requested_remove (pre-clamp).
    # The "left N removed" count reports Stage 1's own output (actual_remove), so
    # the relation `deficit + left_removed == requested_remove` holds even when
    # Stage 2 (warmup grace) further shrinks actual_remove to effective_remove.
    if actual_remove < requested_remove:
        deficit = requested_remove - actual_remove
        fraction_bound = fraction_cap < allowed_by_floor and fraction_cap == actual_remove
        if fraction_bound:
            logger.info(
                f"{preamble}; per-cycle fraction cap left {actual_remove} removed "
                f"(deficit={deficit}, current={current}, "
                f"max_scale_down_fraction_per_cycle={max_scale_down_fraction_per_cycle}"
                f"{cap_kwargs})."
            )
        else:
            logger.info(
                f"{preamble}; floor cap left {actual_remove} removed "
                f"(deficit={deficit}, current={current}, floor={floor}{cap_kwargs})."
            )
        deficit_reported = True
    # Stage 2: warmup grace truncation reported even when Stage 1 also fired,
    # so the operator sees the full clamp chain instead of only the first binding clamp.
    if effective_remove < actual_remove:
        warmup_deficit = actual_remove - effective_remove
        logger.info(
            f"{preamble}; donor warmup grace left {effective_remove} removed "
            f"(deficit={warmup_deficit}, current={current}, "
            f"warmup_excluded={warmup_excluded_count}{cap_kwargs})."
        )
        deficit_reported = True
    # Stage 3: cap-driven full removal - only fires when no deficit branch fired,
    # because reporting "removed N workers" alongside a deficit message would be
    # contradictory.
    if cap_driven and not deficit_reported:
        logger.info(
            f"saturation-aware scale-down: stage {stage_name!r} hard worker cap "
            f"overflow removed {effective_remove} workers (current={current}, "
            f"ceiling={ceiling}, intent={intent})."
        )


__all__ = ["SaturationShrinkPhase"]
