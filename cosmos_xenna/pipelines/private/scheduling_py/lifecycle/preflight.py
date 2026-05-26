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

"""Factory that builds the per-cycle ``AutoscaleCycle`` working state.

``PreflightBuilder`` owns the ordered pre-phase setup that used to
live inline at the top of ``_autoscale_body``: cycle-counter
advance, warmup tracker refresh, regime detection + threshold
resolution, ``AutoscalePlanContext`` construction, donor warmup
exclusion build, and the snapshot of the stuck-plan counters
needed by the post-Phase-D monotonicity invariant.

Constructor dependencies are narrow value objects: the cross-cycle
``SchedulerLedgers``, the frozen ``RegimeController`` and
``ThresholdResolver``, and the post-setup ``PipelineModel`` that
captures the pipeline shape. The model is non-optional by
construction (built inside ``setup()``) so the builder body never
guards against ``problem is None``.

See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

import time as _time

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.invariants.checks import PhaseBoundary
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller import RegimeController
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver import ThresholdResolver


@attrs.frozen
class PreflightResult:
    """Output of ``PreflightBuilder.build``.

    Carries the cycle handed to the first phase plus a snapshot of
    the stuck-plan counters taken before any phase mutates the
    live map; the snapshot feeds the post-Phase-D monotonicity
    invariant ``check_stuck_plan_monotonicity``.

    """

    cycle: AutoscaleCycle
    prev_stuck_plan_counters: dict[str, int]


@attrs.frozen
class PreflightBuilder:
    """Builds a fully-populated ``AutoscaleCycle`` from one cycle's inputs.

    Args:
        ledgers: Cross-cycle ledger; mutated by
            ``cycle_counter += 1`` and by ``warmup.refresh``.
        regime: Regime controller; the per-cycle ``update`` step
            runs detection before threshold resolution so the
            classifier band is always derived under the freshest
            regime.
        threshold_resolver: Threshold resolver; lazily resolves
            per-stage classifier thresholds once the warmup has
            been refreshed.
        pipeline: ``PipelineModel`` captured by ``setup()``;
            provides the frozen problem, the canonical stage
            ordering, and the effective per-stage config lookup.
        pipeline_name: Pipeline tag carried onto the cycle (used
            by phase metrics and structured logs).

    """

    ledgers: SchedulerLedgers
    regime: RegimeController
    threshold_resolver: ThresholdResolver
    pipeline: PipelineModel
    pipeline_name: str

    def build(
        self,
        *,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> PreflightResult:
        """Run the pre-phase setup and return a cycle ready for Phase A.

        Runs the ordered setup that ``_autoscale_body`` used to
        do inline: shape check, cycle counter advance, warmup
        refresh, stuck-plan snapshot, regime detection,
        threshold resolution, planner context build, donor
        warmup exclusion build, ``AutoscaleCycle`` build.

        Raises:
            SchedulerInvariantError: ``problem_state`` shape
                disagrees with the cycle-start ``problem``.

        """
        problem = self.pipeline.problem
        _check_problem_state_shape_before_manual(problem, problem_state)
        self.ledgers.cycle_counter += 1
        # Refresh ready first-seen timestamps from this cycle's snapshot before
        # any phase reads them. The per-worker measurement grace consumes them
        # inside the classifier pipeline; Phase D shrink and the saturation-mode
        # cross-stage donor consult the resulting warmup-grace excluded set.
        self.ledgers.warmup.refresh(problem_state, now=time)
        # Snapshot the stuck-plan counters before Phase C mutates them so the
        # post-Phase-D monotonicity check can compare prev vs. curr without an
        # in-flight Phase C state. The same snapshot is also reused as the
        # caller-side filter: stages whose ``curr == prev`` were not touched
        # by ``StuckPlanLedger.record`` and are excluded from the assertion.
        prev_stuck_plan_counters = self.ledgers.stuck_plan.snapshot()
        self.regime.update(problem_state)
        self.threshold_resolver.ensure_resolved(problem_state)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            problem_state,
            worker_ages=_next_cycle_worker_ages(self.ledgers.worker_ages),
        )
        # Cache the per-cycle donor warmup excluded set immediately after
        # ``AutoscalePlanContext.from_problem_state`` seeds the planner from
        # ``problem_state``. ``ctx.worker_ids_by_stage()`` at this point
        # reflects the cycle-start snapshot (the live worker set as observed
        # by the actor pool); no phase has run yet, so the set stays a
        # cycle-start snapshot of "which observed workers are still in
        # warmup according to ``donor_warmup_grace_s``".
        donor_warmup_excluded_ids = self.ledgers.warmup.excluded_ids(
            ctx.worker_ids_by_stage(),
            self.pipeline.stage_names,
            self.pipeline.stage_config,
            now=time,
        )

        # Construct the cycle-scoped working state once per autoscale
        # call. Cross-cycle maps live on ``SchedulerLedgers`` and are
        # NOT mirrored here; phases read them through the per-phase
        # services. Snapshot fields are ``attrs.field(init=False)``;
        # phases assign them in declared order. Reading a snapshot
        # before its producing phase ran raises ``AttributeError``.
        cycle = AutoscaleCycle(
            ctx=ctx,
            problem_state=problem_state,
            time=time,
            cycle_counter=self.ledgers.cycle_counter,
            pipeline_name=self.pipeline_name,
        )
        cycle.donor_warmup_excluded_ids = donor_warmup_excluded_ids
        # Memory-pressure gating is precomputed here so the Grow phase
        # reads a single boolean instead of re-evaluating the config
        # flag and the monitor each cycle. The monitor needs a
        # monotonic timestamp for its TTL window; ``time.monotonic`` is
        # the same clock the monitor's poll path consumes elsewhere.
        cycle.is_memory_pressure_active = (
            self.pipeline.config.enable_memory_pressure_gate
            and self.ledgers.memory_pressure.is_pressure_active(_time.monotonic())
        )

        return PreflightResult(cycle=cycle, prev_stuck_plan_counters=prev_stuck_plan_counters)


def _check_problem_state_shape_before_manual(
    problem: data_structures.Problem,
    problem_state: data_structures.ProblemState,
) -> None:
    """Reject problem / problem_state shape drift before phase-specific indexing.

    The autoscale snapshot is corrupted whenever the runtime
    ``problem_state`` carries a different number of stages, or
    the stages disagree by name at any index. Raising before any
    phase runs makes the diagnosis attribute the failure to the
    correct boundary (``PhaseBoundary.MANUAL`` - the first phase
    in the per-cycle pipeline).
    """
    problem_stages = problem.rust.stages
    runtime_stages = problem_state.rust.stages
    if len(runtime_stages) != len(problem_stages):
        msg = (
            f"Before {PhaseBoundary.MANUAL}: problem_state has {len(runtime_stages)} stages "
            f"but problem has {len(problem_stages)}. The autoscale cycle snapshot is corrupted."
        )
        raise SchedulerInvariantError(msg)
    for stage_index, (problem_stage, runtime_stage) in enumerate(zip(problem_stages, runtime_stages, strict=True)):
        if runtime_stage.stage_name == problem_stage.name:
            continue
        msg = (
            f"Before {PhaseBoundary.MANUAL}: stage index {stage_index} has problem stage "
            f"{problem_stage.name!r} but problem_state stage {runtime_stage.stage_name!r}. "
            "The autoscale cycle snapshot is corrupted."
        )
        raise SchedulerInvariantError(msg)


def _next_cycle_worker_ages(worker_ages: dict[str, int]) -> dict[str, int]:
    """Build the planner's age seed for the next autoscale cycle.

    Ages count completed autoscale cycles. Surviving workers age by
    one when a new planning context is created; the Rust planner
    drops ids that are absent from the current ``ProblemState`` and
    assigns age 0 to newly observed workers.
    """
    return {worker_id: age + 1 for worker_id, age in worker_ages.items()}


__all__ = ("PreflightBuilder", "PreflightResult")
