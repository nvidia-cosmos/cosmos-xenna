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

"""Saturation-aware scheduler - the public class.

Implements the same public API as ``FragmentationBasedAutoscaler``
(``setup``, ``update_with_measurements``, ``autoscale``) so the two
can be swapped at ``Autoscaler.__init__`` via
``StreamingSpecificSpec.scheduler``.

``setup()`` seeds an empty per-stage ``_StageRuntimeState`` map.
Classifier thresholds are resolved lazily on the first
``autoscale()`` cycle: ``ProblemStageState.slots_per_worker``
supplies the M/M/c concurrency the resolver needs, and the
formula's ``K/sqrt(c)`` derivation is unstable until that value is
present. Once resolved, the values live on
``_StageRuntimeState.resolved_thresholds`` across ordinary cycles and
runtime ``slots_per_worker`` changes. A Halfin-Whitt regime transition
is the only re-resolution path: it clears the resolved thresholds and
threshold-relative classifier history before deriving the new band from
the updated effective aggressiveness.

``autoscale()`` constructs an ``AutoscalePlanContext`` per cycle so
per-stage decision logic can stage worker adds and removes against
a working cluster snapshot; the staged plan is frozen into a
``Solution`` via ``ctx.into_solution()``.
"""

import math

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    data_structures as rust_data_structures,
)
from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import (
    ResolvedThresholds,
    _resolve_auto_thresholds,
)
from cosmos_xenna.pipelines.private.scheduling_py.dag_priority import compute_dag_depth_order
from cosmos_xenna.pipelines.private.scheduling_py.donor import (
    DonorCandidate,
    find_saturation_donor,
    select_youngest_eligible_donor,
)
from cosmos_xenna.pipelines.private.scheduling_py.invariants import (
    PhaseBoundary,
    check_floor_after_phase_d,
    check_invariants_after_phase,
    check_no_nan_in_classifier_state,
    check_solution_shape,
    check_stuck_plan_monotonicity,
)
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import run_per_stage_pipeline
from cosmos_xenna.pipelines.private.scheduling_py.regime import (
    EXIT_BAND_MULTIPLIER,
    Regime,
    RegimeDetectorState,
    RegimeSignal,
    compute_regime_signal,
    update_regime_state,
)
from cosmos_xenna.pipelines.private.scheduling_py.scale_down import select_workers_to_remove_oldest_first
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


class SaturationAwareScheduler:
    """Pure-Python saturation-aware scheduler.

    Selected via ``StreamingSpecificSpec.scheduler == SchedulerKind.SATURATION_AWARE``.
    Replaces the Rust-backed ``FragmentationBasedAutoscaler`` at the
    algorithm layer; ``Autoscaler``-level concerns (deletion clamps,
    threading, measurement aggregation) remain owned by ``streaming.py``.

    Attributes:
        _config: Cluster-wide configuration. Per-stage configs are
            resolved on demand via
            ``SaturationAwareConfig.get_effective_stage_config``.
        _problem: Pipeline structure captured at ``setup()`` time. Held
            for use by the per-cycle decision pipeline (DAG depth,
            fixed stage indices); ``None`` until ``setup()`` is called.
        _stage_states: Per-stage runtime state, keyed by stage name.
            Populated at ``setup()``; mutated each cycle by the
            per-stage pipeline.
        _stage_names: Stage names in pipeline (DAG) order, captured at
            ``setup()``. Used to iterate stages deterministically.
        _regime_state: Cross-cycle hysteresis state for the
            Halfin-Whitt regime detector. Used to lift the effective
            ``saturation_aggressiveness`` when the cluster is packed
            close to capacity. Defaults to ``SUB_HALFIN_WHITT`` so
            cold-start cycles use the base aggressiveness.
        _worker_ages: Cross-cycle worker age snapshot keyed by worker
            id. Seeded into each planner context so manual shrink and
            floor donor fallback can prefer younger workers.
        _last_intent_deltas: Per-stage signed worker-count intent
            produced by the per-stage decision pipeline on the most
            recent ``autoscale()`` call. Saturation-driven scale-up
            (Phase C) and scale-down (Phase D) consume this map; the
            current iteration computes intent only and does not act
            on the values, so they are exposed for tests and
            observability. Reset on every ``autoscale()`` cycle.
            Finished stages are absent (skipped by
            ``_compute_intent_deltas``).
        _stuck_plan_counters: Per-stage count of consecutive
            ``autoscale()`` cycles where Phase C had a positive
            intent but could not place the full request. Increments
            when ``added < intent`` (cluster placement exhausted or
            a higher-priority downstream stage consumed the headroom
            first under DAG-priority ordering); resets to ``0`` on a
            cycle where the stage either fully met its intent or had
            no positive intent. The counter is the input the
            pipeline-level ``stuck_plan_detection_cycles`` watchdog
            (Phase 4) will read.
        _cycle_counter: Monotonic count of completed ``autoscale()``
            calls since ``setup()``. Used as the time index for the
            saturation-mode cross-stage donor anti-flap layers.
        _last_donation_cycle: Per-stage record of the cycle at which
            each stage most recently donated a worker through the
            saturation-mode cross-stage path. Drives both the
            ``cross_stage_donor_min_donation_interval_cycles`` donor
            cooldown and the ``cross_stage_donor_anti_flap_cycles``
            receiver-was-recent-donor block.
        _donations_received_this_cycle: Per-cycle bound on how many
            cross-stage donations a single receiver may absorb.
            Resets at the top of every ``autoscale()`` and is
            consulted against ``cross_stage_donor_max_per_cycle``.

    """

    def __init__(self, config: SaturationAwareConfig) -> None:
        """Initialize the scheduler.

        Args:
            config: Cluster-wide ``SaturationAwareConfig``. Stored by
                reference; per-stage configs are resolved lazily via
                ``config.get_effective_stage_config``.

        """
        self._config = config
        self._problem: data_structures.Problem | None = None
        self._stage_states: dict[str, _StageRuntimeState] = {}
        self._stage_names: list[str] = []
        self._regime_state: RegimeDetectorState = RegimeDetectorState()
        self._worker_ages: dict[str, int] = {}
        self._last_intent_deltas: dict[str, int] = {}
        self._stuck_plan_counters: dict[str, int] = {}
        self._floor_stuck_counters: dict[str, int] = {}
        # Saturation-mode cross-stage donor anti-flap state.
        # ``_cycle_counter`` is monotonic and increments at the top of every
        # ``autoscale()`` call. ``_last_donation_cycle`` records the cycle at
        # which each stage most recently donated; missing entries mean the
        # stage has never donated. ``_donations_received_this_cycle`` resets
        # on each ``autoscale()`` and bounds receiver donations under
        # ``cross_stage_donor_max_per_cycle``.
        self._cycle_counter: int = 0
        self._last_donation_cycle: dict[str, int] = {}
        self._donations_received_this_cycle: dict[str, int] = {}

    def setup(self, problem: data_structures.Problem) -> None:
        """Capture the pipeline shape and seed empty per-stage state.

        Args:
            problem: The frozen pipeline ``Problem``. The Python
                wrapper does not expose stages directly; iterating
                ``.rust.stages`` gives stage names ahead of any
                runtime state arriving in ``autoscale()``.

        """
        self._problem = problem
        self._stage_names = [stage.name for stage in problem.rust.stages]
        self._stage_states = {name: _StageRuntimeState(stage_name=name) for name in self._stage_names}
        self._regime_state = RegimeDetectorState()
        self._worker_ages = {}
        self._floor_stuck_counters = {}
        self._last_intent_deltas = {}
        self._stuck_plan_counters = {}
        self._cycle_counter = 0
        self._last_donation_cycle = {}
        self._donations_received_this_cycle = {}

    def _ensure_thresholds_resolved(self, problem_state: data_structures.ProblemState) -> None:
        """Lazily resolve per-stage classifier thresholds on the first cycle.

        Reads ``slots_per_worker`` from each stage's runtime
        ``ProblemStageState`` (the M/M/c concurrency), feeds it to the
        resolver with the regime-aware effective aggressiveness, stores
        the result on ``_StageRuntimeState``, and emits one INFO log
        line per stage. Stages whose ``resolved_thresholds`` is already
        populated short-circuit; mid-run ``slots_per_worker`` changes
        do not re-resolve. A regime transition (handled by
        ``_update_regime_aware_aggressiveness``) drops every stage's
        resolved thresholds and threshold-relative classifier history
        so this method re-derives them with the new effective
        aggressiveness on the same cycle.

        Raises:
            ValueError: If ``problem_state`` carries a stage name not
                present in the ``setup()`` state map (shape mismatch).

        """
        for stage in problem_state.rust.stages:
            runtime = self._stage_states.get(stage.stage_name)
            if runtime is None:
                msg = (
                    f"problem_state stage {stage.stage_name!r} not found in setup() "
                    f"state map (known: {sorted(self._stage_states)}); "
                    "problem and problem_state shapes disagree."
                )
                raise ValueError(msg)
            if runtime.resolved_thresholds is not None:
                continue
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=stage.stage_name,
                spec_override=None,
            )
            effective_aggressiveness = self._effective_aggressiveness(stage_cfg.saturation_aggressiveness)
            resolved = _resolve_auto_thresholds(
                stage_cfg,
                slots_per_actor=stage.slots_per_worker,
                aggressiveness_override=effective_aggressiveness,
            )
            runtime.resolved_thresholds = resolved
            self._log_resolved_thresholds(stage.stage_name, stage_cfg, resolved)

    def _effective_aggressiveness(self, base: float) -> float:
        """Apply the super-Halfin-Whitt lift to the base aggressiveness if active.

        Returns ``base`` unchanged when the cluster is in the
        sub-Halfin-Whitt regime or when the regime-aware lift is
        disabled. Returns ``base + super_halfin_whitt_aggressiveness_lift``
        when the cluster is in super-Halfin-Whitt and the lift is
        enabled.

        """
        if not self._config.enable_regime_aware_aggressiveness:
            return base
        if self._regime_state.current_regime is Regime.SUPER_HALFIN_WHITT:
            return base + self._config.super_halfin_whitt_aggressiveness_lift
        return base

    @staticmethod
    def _log_regime_transition(new_regime: Regime, signal: RegimeSignal, effective_aggressiveness: float) -> None:
        """Emit one INFO line per regime transition."""
        logger.info(
            f"scheduler regime transition: -> {new_regime.value} "
            f"(total_workers={signal.total_workers}, "
            f"cluster_idle_fraction={signal.cluster_idle_fraction:.4f}, "
            f"threshold={signal.threshold:.4f}, "
            f"effective_aggressiveness={effective_aggressiveness:.4f})"
        )

    @staticmethod
    def _log_resolved_thresholds(
        stage_name: str,
        stage_cfg: SaturationAwareStageConfig,
        resolved: ResolvedThresholds,
    ) -> None:
        """Emit one INFO line per stage describing the resolved thresholds."""
        sat_source = "manual override" if resolved.saturation_threshold_was_overridden else "auto"
        act_source = "manual override" if resolved.activation_threshold_was_overridden else "auto"
        logger.info(
            f"scheduler resolved auto thresholds for stage {stage_name!r}: "
            f"slots_per_actor={resolved.slots_per_actor}, "
            f"saturation_aggressiveness={resolved.saturation_aggressiveness}, "
            f"saturation_threshold={resolved.saturation_threshold:.6f} ({sat_source}), "
            f"activation_threshold={resolved.activation_threshold:.6f} ({act_source}), "
            f"over_provisioned_threshold={stage_cfg.over_provisioned_threshold} (config; not auto-derived)"
        )

    def update_with_measurements(
        self,
        time: float,
        measurements: data_structures.Measurements,
    ) -> None:
        """Ingest the latest measurement batch.

        The fragmentation-based scheduler uses this to feed its
        windowed throughput estimator. The saturation-aware scheduler
        is signal-driven from ``ProblemState.actor_pool_state``
        directly, so per-task measurements are not required for its
        own decisions. They are still accepted and discarded so the
        call sites in ``streaming.py`` need no special-case branch.

        Args:
            time: Current wall-clock time in seconds.
            measurements: Per-stage measurement batch since the previous
                cycle.

        """
        del time, measurements

    def autoscale(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Compute the autoscale plan for the current cycle.

        Args:
            time: Current wall-clock time in seconds.
            problem_state: Current per-stage runtime snapshot.

        Returns:
            A ``Solution`` with one ``StageSolution`` per stage in
            ``problem_state``, in the same order, carrying the existing
            ``slots_per_worker`` and any worker adds / removes staged
            on the cycle's ``AutoscalePlanContext``.

        Raises:
            RuntimeError: ``setup()`` was not called, planner context
                construction failed, or the non-manual worker floor
                cannot be satisfied by the current cluster.
            SchedulerInvariantError: A planner-context, classifier
                EWMA finiteness, Phase D floor / ceiling, stuck-plan
                monotonicity, or Solution-shape invariant failed
                between phases; the plan is corrupted and must not be
                applied. Caller should treat as a must-fail signal:
                log, surface, and refuse to apply the plan. See
                ``scheduling_py/invariants.py``.
            ValueError: ``problem`` and ``problem_state`` disagree on
                stage names or count.

        """
        del time
        if self._problem is None:
            msg = "SaturationAwareScheduler.autoscale() called before setup()"
            raise RuntimeError(msg)

        self._cycle_counter += 1
        self._donations_received_this_cycle = {}
        # Snapshot the stuck-plan counters before Phase C mutates them so the
        # post-Phase-D monotonicity check can compare prev vs. curr without an
        # in-flight Phase C state.
        prev_stuck_plan_counters = dict(self._stuck_plan_counters)
        self._update_regime_aware_aggressiveness(problem_state)
        self._ensure_thresholds_resolved(problem_state)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            self._problem,
            problem_state,
            worker_ages=self._next_cycle_worker_ages(),
        )
        self._run_phase_a_delete(ctx, problem_state)
        self._run_phase_a_grow(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_A, problem=self._problem, ctx=ctx)

        self._run_phase_b_floor(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_B, problem=self._problem, ctx=ctx)

        self._last_intent_deltas = self._compute_intent_deltas(ctx, problem_state)
        self._run_phase_c_grow(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_C, problem=self._problem, ctx=ctx)
        check_no_nan_in_classifier_state(
            phase_name=PhaseBoundary.PHASE_C,
            stage_runtime_states=self._stage_states,
        )

        # Capture pre-Phase-D worker counts BEFORE the shrink runs so the
        # post-Phase-D floor invariant can distinguish "Phase D reduced below
        # floor" (a defect) from "Phase B left the stage below floor" (a
        # grace-window scenario Phase D leaves untouched).
        pre_phase_d_worker_counts = {
            stage_index: len(worker_ids) for stage_index, worker_ids in enumerate(ctx.worker_ids_by_stage())
        }
        self._run_phase_d_shrink(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_D, problem=self._problem, ctx=ctx)
        # Recompute floors once for the Phase D invariant check; the helper is
        # O(num_stages) over an immutable Problem so the second call is cheap
        # and avoids threading a precomputed dict through
        # ``_run_phase_d_shrink``'s signature.
        num_nodes = len(self._problem.rust.cluster_resources.nodes)
        stage_floors = self._compute_stage_floors(num_nodes)
        check_floor_after_phase_d(
            phase_name=PhaseBoundary.PHASE_D,
            problem=self._problem,
            problem_state=problem_state,
            ctx=ctx,
            stage_floors=stage_floors,
            pre_phase_d_worker_counts=pre_phase_d_worker_counts,
        )
        # Stale counters for stages that finished mid-run would surface as
        # ``prev == curr`` transitions; filter both snapshots to the stages
        # Phase C actually touched (non-finished) so the monotonicity check
        # only validates transitions Phase C is responsible for.
        active_stage_names = {stage.stage_name for stage in problem_state.rust.stages if not stage.is_finished}
        check_stuck_plan_monotonicity(
            prev_counters={
                name: count for name, count in prev_stuck_plan_counters.items() if name in active_stage_names
            },
            curr_counters={
                name: count for name, count in self._stuck_plan_counters.items() if name in active_stage_names
            },
        )

        solution = ctx.into_solution()
        check_solution_shape(phase_name=PhaseBoundary.INTO_SOLUTION, problem=self._problem, solution=solution)
        self._persist_worker_ages(ctx)
        return solution

    def _next_cycle_worker_ages(self) -> dict[str, int]:
        """Build the planner's age seed for the next autoscale cycle.

        Ages count completed autoscale cycles. Surviving workers age by
        one when a new planning context is created; the Rust planner
        drops ids that are absent from the current ``ProblemState`` and
        assigns age 0 to newly observed workers.
        """
        return {worker_id: age + 1 for worker_id, age in self._worker_ages.items()}

    def _persist_worker_ages(self, ctx: data_structures.AutoscalePlanContext) -> None:
        """Persist live worker ages from a finalized planning context.

        Defensive against the Rust contract that worker ids returned by
        ``worker_ids_by_stage`` are also present in ``worker_ages``: a
        missing entry defaults to age 0 (treated as freshly observed)
        rather than raising ``KeyError`` mid-cycle. The fallback matches
        :func:`select_youngest_eligible_donor`'s missing-age semantics.
        """
        live_worker_ids = {worker_id for stage_ids in ctx.worker_ids_by_stage() for worker_id in stage_ids}
        worker_ages = ctx.worker_ages()
        self._worker_ages = {worker_id: worker_ages.get(worker_id, 0) for worker_id in live_worker_ids}

    def _compute_intent_deltas(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> dict[str, int]:
        """Compute the per-stage signed worker-count intent for this cycle.

        For each non-finished stage, calls
        :func:`run_per_stage_pipeline` with the live slot signals
        sourced from ``ProblemStageState`` and the post-Phase-B worker
        count read from the planner context. The returned delta is the
        algorithm's intent before any cluster-wide feasibility clamps.

        The intent values are exposed via :attr:`_last_intent_deltas`
        for tests and observability. Saturation-driven scale-up
        (Phase C) and scale-down (Phase D) consume these intents and
        stage worker adds / removes on the planner context.
        Finished stages are absent from the
        returned map: their classifier short-circuits upstream of
        ``run_per_stage_pipeline`` and any drain-state slot signal
        would only mutate per-stage EWMA state without affecting the
        plan.

        Args:
            ctx: The cycle's mutable planner context after Phase B.
                Read-only here (intent only); ``current_workers`` is
                sourced from ``ctx.worker_ids_by_stage()`` so the
                pipeline observes the post-Phase-B worker count.
            problem_state: The cycle's runtime snapshot. Provides the
                three slot signals (``num_used_slots``,
                ``num_empty_slots``, ``input_queue_depth``) populated
                in the streaming layer.

        Returns:
            Mapping of stage name -> signed intent. Positive values
            indicate scale-up intent, negative values scale-down,
            zero a no-op. Finished stages are absent.
        """
        intents: dict[str, int] = {}
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        for stage_index, runtime_stage in enumerate(problem_state.rust.stages):
            if runtime_stage.is_finished:
                continue
            stage_name = runtime_stage.stage_name
            stage_state = self._stage_states.get(stage_name)
            if stage_state is None:
                msg = (
                    f"problem_state stage {stage_name!r} not found in setup() "
                    f"state map (known: {sorted(self._stage_states)}); "
                    "problem and problem_state shapes disagree."
                )
                raise ValueError(msg)
            stage_cfg = self._config.get_effective_stage_config(stage_name=stage_name, spec_override=None)
            current_workers = len(worker_ids_by_stage[stage_index])
            delta = run_per_stage_pipeline(
                stage_state=stage_state,
                num_used_slots=runtime_stage.num_used_slots,
                num_empty_slots=runtime_stage.num_empty_slots,
                input_queue_depth=runtime_stage.input_queue_depth,
                current_workers=current_workers,
                config=stage_cfg,
            )
            intents[stage_name] = delta
        return intents

    def _run_phase_c_grow(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Apply positive intent deltas as planner adds, DAG-priority order.

        Walks stages in downstream-first order (greatest DAG depth
        first) so any free cluster capacity is spent on the stage
        most likely to bound pipeline throughput, and avoids the
        fragmentation-based scheduler behavior where one blocked
        bottleneck can stop growth attempts for other saturated
        stages: every stage with positive intent is attempted
        independently. For each
        non-finished stage whose :attr:`_last_intent_deltas` entry is
        positive, calls ``ctx.try_add_worker(stage_index)`` up to
        ``intent`` times. Cluster placement exhaustion (planner
        returns ``None``) is non-fatal: growth stops for that
        stage in the current cycle and a single WARNING per affected
        stage is emitted so operators retain visibility into
        partially-satisfied saturation-driven growth. Negative or
        zero intent (NORMAL, STARVED, OVER_PROVISIONED) is a no-op
        here; saturation-driven scale-down lands as Phase D.

        Iteration order is determined by
        :func:`compute_dag_depth_order` when
        ``config.enable_dag_priority_growth`` is True (the default);
        otherwise stages are walked in problem order. Every stage
        with a positive intent is attempted regardless of any
        earlier capacity exhaustion.

        Updates :attr:`_stuck_plan_counters`: per-stage count of
        consecutive cycles where ``added < intent``. Resets to ``0``
        on any cycle where the stage either fully met its intent or
        had no positive intent. The counter is the input the
        pipeline-level ``stuck_plan_detection_cycles`` watchdog
        (Phase 4) will read.

        Args:
            ctx: The cycle's mutable planner context. Mutated in place
                by ``try_add_worker``.
            problem_state: The cycle's runtime snapshot. Used to skip
                finished stages and to surface stage indices.

        """
        if self._problem is None:
            msg = "_run_phase_c_grow called before setup()"
            raise RuntimeError(msg)

        if self._config.enable_dag_priority_growth:
            stage_order = compute_dag_depth_order(self._problem)
        else:
            stage_order = list(range(len(problem_state.rust.stages)))

        num_nodes = len(self._problem.rust.cluster_resources.nodes)
        stage_ceilings = self._compute_stage_ceilings(num_nodes)
        worker_ids_by_stage = ctx.worker_ids_by_stage()

        for stage_index in stage_order:
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            stage_name = runtime_stage.stage_name
            intent = self._last_intent_deltas.get(stage_name, 0)
            if intent <= 0:
                self._stuck_plan_counters[stage_name] = 0
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
                    self._stuck_plan_counters[stage_name] = 0
                    continue
            added = 0
            while added < intent:
                if ctx.try_add_worker(stage_index) is not None:
                    added += 1
                    continue
                # Cluster placement exhausted. Try the saturation-mode
                # cross-stage donor fallback before giving up. On donor
                # success the planner has freed exactly one placement,
                # so the very next ``try_add_worker`` for this receiver
                # should succeed; if it does not (donor's freed slot
                # does not match the receiver's shape), give up for the
                # cycle without raising. The donor cooldown and
                # receiver-per-cycle counters are updated only on a
                # successfully completed donation+retry.
                if not self._attempt_cross_stage_donation(
                    ctx=ctx,
                    receiver_stage_index=stage_index,
                    receiver_stage_name=stage_name,
                ):
                    deficit = intent - added
                    logger.warning(
                        f"saturation-aware scale-up: stage {stage_name!r} intent "
                        f"{intent} workers; cluster placement exhausted after "
                        f"{added} (deficit={deficit}); request remains partially "
                        "satisfied this cycle."
                    )
                    break
                if ctx.try_add_worker(stage_index) is None:
                    deficit = intent - added
                    logger.warning(
                        f"saturation-aware scale-up: stage {stage_name!r} intent "
                        f"{intent} workers; cross-stage donation freed a slot but "
                        f"the post-donation retry returned no placement after "
                        f"{added} (deficit={deficit}); request remains partially "
                        "satisfied this cycle."
                    )
                    break
                added += 1
            if added < intent:
                self._stuck_plan_counters[stage_name] = self._stuck_plan_counters.get(stage_name, 0) + 1
            else:
                self._stuck_plan_counters[stage_name] = 0

    def _attempt_cross_stage_donation(
        self,
        *,
        ctx: data_structures.AutoscalePlanContext,
        receiver_stage_index: int,
        receiver_stage_name: str,
    ) -> bool:
        """Try to free a placement for a saturation-driven receiver.

        Selects an eligible donor via
        :func:`find_saturation_donor` (five anti-flap layers + strict
        upstream + master toggle), removes it from the planner, and
        updates the donor cooldown + receiver per-cycle counter. The
        caller is responsible for the immediate ``try_add_worker``
        retry after this method returns ``True``.

        Returns:
            True when a donor was selected, removed, and the donor /
            receiver counters were advanced. False when the master
            toggle is off, when no eligible donor exists, or when the
            planner refuses the selected donor (defensive guard).

        Raises:
            RuntimeError: The scheduler has not been set up.

        """
        if self._problem is None:
            msg = "_attempt_cross_stage_donation called before setup()"
            raise RuntimeError(msg)

        num_nodes = len(self._problem.rust.cluster_resources.nodes)
        stage_floors = self._compute_stage_floors(num_nodes)
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        worker_ages = ctx.worker_ages()
        stage_configs = {
            name: self._config.get_effective_stage_config(stage_name=name, spec_override=None)
            for name in self._stage_names
        }

        donor = find_saturation_donor(
            receiver_stage_index=receiver_stage_index,
            receiver_stage_name=receiver_stage_name,
            stage_names=self._stage_names,
            stage_floors=stage_floors,
            worker_ids_by_stage=worker_ids_by_stage,
            worker_ages=worker_ages,
            stage_states=self._stage_states,
            config=self._config,
            stage_configs=stage_configs,
            cycle=self._cycle_counter,
            last_donation_cycle=self._last_donation_cycle,
            donations_received_this_cycle=self._donations_received_this_cycle,
        )
        if donor is None:
            return False

        donor_stage_name = self._stage_names[donor.stage_index]
        if not ctx.try_remove_worker(donor.stage_index, donor.worker_id):
            logger.warning(
                f"[scheduler] saturation-mode donor: stage {donor_stage_name!r} "
                f"worker {donor.worker_id!r} selected by donor helper but planner "
                "refused removal; donation cancelled and receiver retry skipped."
            )
            return False

        self._last_donation_cycle[donor_stage_name] = self._cycle_counter
        self._donations_received_this_cycle[receiver_stage_name] = (
            self._donations_received_this_cycle.get(receiver_stage_name, 0) + 1
        )
        logger.info(
            f"[scheduler] saturation-mode donation: donor stage {donor_stage_name!r} "
            f"worker {donor.worker_id!r} (age={donor.age}) -> receiver stage "
            f"{receiver_stage_name!r} at cycle {self._cycle_counter}."
        )
        return True

    def _run_phase_d_shrink(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Remove workers via the planner to satisfy negative intent or hard-cap overflow.

        For each non-finished, non-manual stage, computes the
        per-stage requested shrink as the maximum of two drivers:

          * Negative classifier intent: when
            :attr:`_last_intent_deltas` is negative, the magnitude
            ``-intent`` is the saturation-driven shrink request.
          * Hard-cap overflow: when ``current_workers > ceiling`` (an
            operator just lowered the cap), the excess
            ``max(0, current - ceiling)`` is a forced shrink even if
            the classifier intent is non-negative.

        Three independent clamps then bound the per-cycle shrink:

          * The configured per-stage / per-node floor (a stage is
            never reduced below ``min_workers`` or
            ``min_workers_per_node * num_nodes``; the same target
            Phase B enforces). Floor wins over ceiling on
            misconfiguration; cross-field validation in
            ``SaturationAwareStageConfig`` already rejects
            ``min_workers > max_workers``.
          * The orchestrator-level fraction cap
            ``fraction_cap = max(1, floor(current *
            stage_cfg.max_scale_down_fraction_per_cycle))``
            (defense-in-depth against externally-injected intents
            that bypass ``compute_delta``).
          * The classifier-side magnitude cap (already applied by
            ``compute_delta._shrink_delta``).

        Manual stages (``requested_num_workers is not None``) are
        excluded so the operator-driven shrink path
        (``_run_phase_a_delete``) remains the single source of truth
        for those. Operators using manual mode are expected to set
        ``requested_num_workers`` consistently with any configured
        ``max_workers``.

        Selection key is
        ``(host_gpu_used_fraction ASC, idle DESC, age DESC, worker_id ASC)``
        where ``host_gpu_used_fraction`` is the total used fraction
        of the GPU each worker is placed on (summed across every
        stage that holds an allocation on that GPU) and ``idle =
        (num_used_slots == 0)``. The consolidation key is the
        primary sort: workers on the GPUs with the lowest total
        fraction are removed first, because those GPUs are most
        likely to become fully unallocated and recoverable for
        downstream whole-GPU stages. Per-worker ``num_used_slots``
        is sourced from
        ``runtime_stage.worker_groups[*].num_used_slots``; per-GPU
        used fractions are derived from
        ``runtime_stage.worker_groups[*].resources`` aggregated
        across all stages of ``problem_state``. For multi-GPU
        workers the per-worker fraction is the maximum across the
        worker's GPU allocations (the most-loaded GPU is the
        binding constraint for whole-GPU recovery).

        Args:
            ctx: The cycle's mutable planner context. Mutated in
                place by ``try_remove_worker``.
            problem_state: The cycle's runtime snapshot. Used to skip
                finished and manual stages, and to source per-worker
                idle signals for the selection helper.

        Raises:
            RuntimeError: The scheduler has not been set up, or the
                planner refuses a worker id selected from its own
                snapshot (defensive: the snapshot inconsistency is a
                scheduler defect, not an operator-config issue).

        """
        if self._problem is None:
            msg = "_run_phase_d_shrink called before setup()"
            raise RuntimeError(msg)

        num_nodes = len(self._problem.rust.cluster_resources.nodes)
        stage_floors = self._compute_stage_floors(num_nodes)
        stage_ceilings = self._compute_stage_ceilings(num_nodes)
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        worker_ages = ctx.worker_ages()
        host_gpu_used_fractions = self._compute_host_gpu_used_fractions(problem_state)

        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            if problem_stage.requested_num_workers is not None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            stage_name = problem_stage.name
            intent = self._last_intent_deltas.get(stage_name, 0)
            current = len(worker_ids_by_stage[stage_index])
            ceiling = stage_ceilings[stage_index]
            ceiling_excess = max(0, current - ceiling) if ceiling is not None else 0
            # Combine the two shrink drivers: negative intent and
            # hard-cap overflow. The cap dominates non-negative intent
            # (forced shrink); negative intent dominates a smaller cap
            # excess (operator wants more shrink than the cap alone).
            requested_remove = max(-intent if intent < 0 else 0, ceiling_excess)
            if requested_remove == 0:
                continue

            floor = stage_floors[stage_index]
            stage_cfg = self._config.get_effective_stage_config(stage_name=stage_name, spec_override=None)
            allowed_by_floor = max(0, current - floor)
            fraction_cap = (
                max(1, math.floor(current * stage_cfg.max_scale_down_fraction_per_cycle)) if current > 0 else 0
            )
            actual_remove = min(requested_remove, allowed_by_floor, fraction_cap)
            if actual_remove == 0:
                continue

            worker_used_slots = {wg.id: wg.num_used_slots for wg in runtime_stage.worker_groups}
            worker_host_gpu_used_fractions = self._extract_worker_host_gpu_used_fractions(
                runtime_stage=runtime_stage,
                host_gpu_used_fractions=host_gpu_used_fractions,
            )
            victims = select_workers_to_remove_oldest_first(
                worker_ids=worker_ids_by_stage[stage_index],
                worker_ages=worker_ages,
                delete_count=actual_remove,
                worker_used_slots=worker_used_slots,
                worker_host_gpu_used_fractions=worker_host_gpu_used_fractions,
            )
            for victim_id in victims:
                if not ctx.try_remove_worker(stage_index, victim_id):
                    msg = (
                        f"Phase D shrink: stage {stage_name!r} planner refused removal of "
                        f"worker {victim_id!r} selected from its own snapshot. This is a "
                        "scheduler defect; the planner state and the runtime snapshot disagree."
                    )
                    raise RuntimeError(msg)

            self._log_phase_d_shrink_outcome(
                stage_name=stage_name,
                intent=intent,
                ceiling=ceiling,
                ceiling_excess=ceiling_excess,
                requested_remove=requested_remove,
                actual_remove=actual_remove,
                current=current,
                floor=floor,
                fraction_cap=fraction_cap,
                allowed_by_floor=allowed_by_floor,
                max_scale_down_fraction_per_cycle=stage_cfg.max_scale_down_fraction_per_cycle,
            )

    @staticmethod
    def _log_phase_d_shrink_outcome(
        *,
        stage_name: str,
        intent: int,
        ceiling: int | None,
        ceiling_excess: int,
        requested_remove: int,
        actual_remove: int,
        current: int,
        floor: int,
        fraction_cap: int,
        allowed_by_floor: int,
        max_scale_down_fraction_per_cycle: float,
    ) -> None:
        """Emit the per-cycle Phase D outcome log distinguishing the binding clamp.

        The four message variants surface the operator-actionable
        cause of the achieved deletion count: the configured floor
        bound the shrink, the per-cycle fraction cap bound it, the
        request was satisfied in full but capped by an operator's
        hard worker cap, or the deletion was driven by hard-cap
        overflow rather than classifier intent. Ties between
        ``fraction_cap`` and ``allowed_by_floor`` resolve to floor for
        backward compatibility with the pre-2-vi log format.
        """
        cap_driven = ceiling_excess > 0 and (intent >= 0 or ceiling_excess >= -intent)
        if actual_remove < requested_remove:
            deficit = requested_remove - actual_remove
            fraction_bound = fraction_cap < allowed_by_floor and fraction_cap == actual_remove
            if fraction_bound:
                logger.info(
                    f"saturation-aware scale-down: stage {stage_name!r} intent "
                    f"-{requested_remove} workers; per-cycle fraction cap left "
                    f"{actual_remove} removed (deficit={deficit}, current={current}, "
                    f"max_scale_down_fraction_per_cycle={max_scale_down_fraction_per_cycle})."
                )
            else:
                logger.info(
                    f"saturation-aware scale-down: stage {stage_name!r} intent "
                    f"-{requested_remove} workers; floor cap left {actual_remove} "
                    f"removed (deficit={deficit}, current={current}, floor={floor})."
                )
        elif cap_driven:
            logger.info(
                f"saturation-aware scale-down: stage {stage_name!r} hard worker cap "
                f"overflow removed {actual_remove} workers (current={current}, "
                f"ceiling={ceiling}, intent={intent})."
            )

    @staticmethod
    def _compute_host_gpu_used_fractions(
        problem_state: data_structures.ProblemState,
    ) -> dict[tuple[str, int], float]:
        """Aggregate the cycle-start used fraction of every GPU in the cluster.

        Iterates the cycle-start snapshot of every worker group in
        every stage and sums each allocation's ``used_fraction`` by
        ``(node_name, gpu_offset)`` key. The result is the total
        used fraction of each GPU at cycle start, used by Phase D
        scale-down to prefer removing workers from GPUs whose total
        usage is lowest (most likely to become fully unallocated
        after deletion).

        This is a cycle-start approximation. Mutations from Phase
        A, Phase B, and Phase C may have shifted the live cluster
        fractions by the time Phase D runs. The consolidation key
        is a soft heuristic that converges over multiple cycles, so
        a per-cycle approximation is acceptable; a live-cluster
        accessor would tighten the signal at the cost of an
        additional Rust FFI hop.

        Args:
            problem_state: The cycle's runtime snapshot. Every
                stage's ``worker_groups[*].resources`` contributes
                to the aggregate.

        Returns:
            Mapping ``{(node_name, gpu_offset): total_used_fraction}``
            covering every GPU that has at least one worker group
            allocation at cycle start. GPUs with no allocations are
            absent from the map; callers must default to 0.0 for
            missing keys.

        """
        fraction_map: dict[tuple[str, int], float] = {}
        for stage_state in problem_state.rust.stages:
            for worker_group in stage_state.worker_groups:
                for resource in worker_group.resources:
                    node = resource.node
                    for gpu_alloc in resource.gpus:
                        key = (node, gpu_alloc.offset)
                        fraction_map[key] = fraction_map.get(key, 0.0) + float(gpu_alloc.used_fraction)
        return fraction_map

    @staticmethod
    def _extract_worker_host_gpu_used_fractions(
        *,
        runtime_stage: rust_data_structures.ProblemStageState,
        host_gpu_used_fractions: dict[tuple[str, int], float],
    ) -> dict[str, float]:
        """Project the cluster-wide GPU fraction map onto a single stage's workers.

        For each worker group in ``runtime_stage``, looks up the
        cycle-start used fraction of every GPU the worker is placed
        on and returns the maximum across those GPUs. The maximum
        captures the most-loaded GPU that the worker contributes
        to: that GPU is the binding constraint for whole-GPU
        recovery, since deleting the worker only frees the
        worker's contribution and the GPU stays loaded if other
        stages also occupy it.

        CPU-only workers (no GPU allocations) and workers whose
        GPUs have no aggregate fraction in
        ``host_gpu_used_fractions`` (allocations only on this
        worker but with ``used_fraction == 0``) map to 0.0.

        Args:
            runtime_stage: One stage's cycle-start snapshot. Only
                ``worker_groups[*].resources`` is read.
            host_gpu_used_fractions: Cluster-wide map produced by
                :meth:`_compute_host_gpu_used_fractions`.

        Returns:
            Mapping ``{worker_id: per_worker_max_host_gpu_used_fraction}``
            covering every worker group in ``runtime_stage``.

        """
        out: dict[str, float] = {}
        for worker_group in runtime_stage.worker_groups:
            fractions = [
                host_gpu_used_fractions.get((resource.node, gpu_alloc.offset), 0.0)
                for resource in worker_group.resources
                for gpu_alloc in resource.gpus
            ]
            out[worker_group.id] = max(fractions, default=0.0)
        return out

    def _run_phase_a_delete(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Shrink manual stages whose ``requested_num_workers`` is below current.

        Manual stages are those with ``requested_num_workers`` set.
        For unfinished manual stages whose current worker count exceeds
        the request, surplus workers are staged for deletion through
        ``ctx.try_remove_worker``.

        Raises:
            RuntimeError: ``try_remove_worker`` returned ``False`` for
                a worker that was present in ``problem_state``. This
                signals a planner-state inconsistency that must
                surface immediately.

        """
        if self._problem is None:
            msg = "_run_phase_a_delete called before setup()"
            raise RuntimeError(msg)
        worker_ages = ctx.worker_ages()
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
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
            # Delete youngest workers first so long-lived warmed workers survive manual shrink.
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
                    raise RuntimeError(msg)

    def _run_phase_a_grow(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Grow manual stages whose ``requested_num_workers`` is above current.

        Manual stages are those with ``requested_num_workers`` set.
        For unfinished manual stages whose current worker count is
        below the request, fresh workers are staged through
        ``ctx.try_add_worker`` until the request is met. When the
        working cluster has no remaining placement (returns ``None``),
        growth stops for that stage in this cycle without raising, and
        a single WARNING per affected stage is emitted so operators
        retain visibility into partially-satisfied manual requests.

        Raises:
            RuntimeError: The scheduler has not been set up, or the
                planner reports a corrupted/drained planning context.
            IndexError: The planner rejects the stage index.

        """
        if self._problem is None:
            msg = "_run_phase_a_grow called before setup()"
            raise RuntimeError(msg)
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            requested = problem_stage.requested_num_workers
            if requested is None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            current = len(runtime_stage.worker_groups)
            while current < requested:
                if ctx.try_add_worker(stage_index) is None:
                    deficit = requested - current
                    logger.warning(
                        f"manual grow: stage {problem_stage.name!r} requested "
                        f"{requested} workers; cluster placement exhausted at "
                        f"{current} (deficit={deficit}); manual request remains "
                        "partially satisfied this cycle."
                    )
                    break
                current += 1

    def _run_phase_b_floor(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Enforce the per-stage minimum-worker floor on non-manual stages.

        For each non-manual, non-finished stage, brings the current
        worker count up to ``_compute_stage_floors``'s ``target_min``.
        Each shortfall is filled by ``ctx.try_add_worker``; on
        capacity exhaustion the cross-stage donor fallback
        (:func:`select_youngest_eligible_donor`) reallocates one
        worker from another stage and the add is retried. A no-donor
        floor miss accumulates a per-stage stuck-cycle counter;
        ``RuntimeError`` is raised only when the counter exceeds
        ``floor_stuck_grace_cycles``. The counter resets on any cycle
        where the receiver makes forward progress. A post-donation
        retry miss raises immediately because the planner state already
        removed the donor and cannot be rolled back safely.

        Raises:
            RuntimeError: ``setup()`` was not called; OR the floor
                has stayed unsatisfied without progress for more than
                ``floor_stuck_grace_cycles`` consecutive cycles; OR a
                post-donation retry misses; OR the planner rejects a
                staged remove of a worker that is live in the snapshot
                (scheduler defect).
            IndexError: The planner rejects the stage index.

        """
        if self._problem is None:
            msg = "_run_phase_b_floor called before setup()"
            raise RuntimeError(msg)
        num_nodes = self._problem.rust.cluster_resources.num_nodes()
        stage_floors = self._compute_stage_floors(num_nodes)
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            if problem_stage.requested_num_workers is not None:
                # Manual stages have their worker count owned by the manual-shrink/grow path.
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            target_min = stage_floors[stage_index]
            # Read the receiver's count from the planner's live snapshot so any
            # mutation by an earlier-iteration donation is observed correctly.
            current = len(ctx.worker_ids_by_stage()[stage_index])
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=problem_stage.name,
                spec_override=None,
            )
            made_progress = False
            stuck = False
            while current < target_min:
                if ctx.try_add_worker(stage_index) is not None:
                    current += 1
                    made_progress = True
                    continue
                # Cluster is full for the receiver's shape; attempt the
                # cross-stage donor fallback before deciding whether to raise.
                donor = select_youngest_eligible_donor(
                    receiver_stage_index=stage_index,
                    stage_floors=stage_floors,
                    worker_ids_by_stage=ctx.worker_ids_by_stage(),
                    worker_ages=ctx.worker_ages(),
                )
                if donor is None:
                    if made_progress and self._config.floor_stuck_grace_cycles > 0:
                        self._warn_floor_partial_progress(
                            stage_name=problem_stage.name,
                            target_min=target_min,
                            current=current,
                        )
                    else:
                        self._on_floor_stuck(
                            stage_name=problem_stage.name,
                            target_min=target_min,
                            current=current,
                            stage_cfg=stage_cfg,
                            num_nodes=num_nodes,
                        )
                        stuck = True
                    break
                if not ctx.try_remove_worker(donor.stage_index, donor.worker_id):
                    msg = (
                        f"Cross-stage floor donor: planner snapshot inconsistency - "
                        f"try_remove_worker(stage_index={donor.stage_index}, "
                        f"worker_id={donor.worker_id!r}) returned False even though the "
                        "worker is present in the live snapshot. This is a scheduler "
                        "defect; report it with the autoscale cycle's problem_state."
                    )
                    raise RuntimeError(msg)
                if ctx.try_add_worker(stage_index) is None:
                    msg = self._format_floor_unmet_message(
                        stage_name=problem_stage.name,
                        target_min=target_min,
                        current=current,
                        stage_cfg=stage_cfg,
                        num_nodes=num_nodes,
                        donor_attempted=True,
                        donor=donor,
                    )
                    raise RuntimeError(msg)
                logger.info(
                    f"[scheduler] {problem_stage.name!r}: cross-stage minimum-floor donor "
                    f"accepted (donor_stage_index={donor.stage_index}, "
                    f"donor_worker_id={donor.worker_id!r}, donor_age={donor.age})"
                )
                current += 1
                made_progress = True
            if made_progress or not stuck:
                # Floor satisfied this cycle (either by direct add, by donation, or
                # because target_min was already met at entry), or the receiver made
                # partial progress; reset the stuck counter.
                self._floor_stuck_counters.pop(problem_stage.name, None)

    @staticmethod
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
        self,
        *,
        stage_name: str,
        target_min: int,
        current: int,
        stage_cfg: SaturationAwareStageConfig,
        num_nodes: int,
    ) -> None:
        """Account for a single stuck-cycle and either raise or warn.

        Increments the per-stage stuck counter, raises ``RuntimeError``
        when the counter exceeds ``floor_stuck_grace_cycles``, and
        otherwise emits a single WARNING per stuck cycle so operators
        can correlate transient pressure with eventual escalation.

        Args:
            stage_name: Receiver stage that did not reach its floor.
            target_min: Configured minimum for the stage.
            current: Worker count achieved before the failure.
            stage_cfg: Effective per-stage config (used for the
                operator-actionable error message).
            num_nodes: Cluster node count at this cycle.

        Raises:
            RuntimeError: The per-stage stuck counter has exceeded
                ``floor_stuck_grace_cycles``. Message identifies the
                stage, target floor, both source values, and tells
                the operator to reduce the floor or scale up the
                cluster.

        """
        counter = self._floor_stuck_counters.get(stage_name, 0) + 1
        grace = self._config.floor_stuck_grace_cycles
        if counter > grace:
            msg = self._format_floor_unmet_message(
                stage_name=stage_name,
                target_min=target_min,
                current=current,
                stage_cfg=stage_cfg,
                num_nodes=num_nodes,
                donor_attempted=False,
                donor=None,
            )
            raise RuntimeError(msg)
        self._floor_stuck_counters[stage_name] = counter
        remaining = grace - counter
        logger.warning(
            f"[scheduler] {stage_name!r}: minimum-worker floor stuck "
            f"({counter}/{grace} grace cycles); target_min={target_min}, "
            f"achieved={current}, no eligible cross-stage donor; will raise after "
            f"{remaining} more consecutive failed cycles."
        )

    def _compute_stage_floors(self, num_nodes: int) -> dict[int, int]:
        """Compute the donor / receiver floor for every stage in the problem.

        Returns ``target_min`` per stage (problem-stage index → floor)::

            target_min = max(
                stage_cfg.min_workers if stage_cfg.min_workers is not None else 1,
                (stage_cfg.min_workers_per_node * num_nodes) if stage_cfg.min_workers_per_node is not None else 0,
            )

        Applied uniformly to manual and non-manual stages. A manual
        stage may donate down to its configured ``min_workers`` floor;
        ``requested_num_workers`` is a target, not a hard lower bound.
        Operators who need a hard floor must set ``min_workers`` to the
        desired minimum.

        """
        if self._problem is None:
            msg = "_compute_stage_floors called before setup()"
            raise RuntimeError(msg)
        floors: dict[int, int] = {}
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=problem_stage.name,
                spec_override=None,
            )
            floors[stage_index] = max(
                stage_cfg.min_workers if stage_cfg.min_workers is not None else 1,
                stage_cfg.min_workers_per_node * num_nodes if stage_cfg.min_workers_per_node is not None else 0,
            )
        return floors

    def _compute_stage_ceilings(self, num_nodes: int) -> dict[int, int | None]:
        """Compute the hard upper bound on workers for every stage in the problem.

        Returns the effective ceiling per stage (problem-stage index ->
        ceiling, or ``None`` for unbounded)::

            ceiling = min(
                stage_cfg.max_workers if stage_cfg.max_workers is not None else +inf,
                (stage_cfg.max_workers_per_node * num_nodes) if stage_cfg.max_workers_per_node is not None else +inf,
            )

        Returns ``None`` when neither cap is configured (the common
        case in production pipelines that rely on saturation-driven
        sizing). When set, the cap is the final clamp on every
        scale-up path: Phase C drops excess ``try_add_worker`` calls
        and Phase D forces a shrink toward the cap when
        ``current > ceiling`` (operator-driven cap reduction). The
        ``min(...)`` semantics mirror the ``max(...)`` semantics of
        :meth:`_compute_stage_floors` so the two clamps compose
        symmetrically when both are configured.

        Args:
            num_nodes: Number of cluster nodes; multiplies
                ``max_workers_per_node`` to get the absolute per-node
                cap. Must be ``>= 1``; the caller already validates
                cluster shape via the planner.

        Returns:
            Mapping ``{stage_index: ceiling_or_None}`` with one entry
            per stage in :attr:`_problem.rust.stages`. ``None`` means
            no per-stage hard cap is configured and Phase C / Phase D
            should fall back to their pre-cap behavior for that stage.

        Raises:
            RuntimeError: The scheduler has not been set up.

        """
        if self._problem is None:
            msg = "_compute_stage_ceilings called before setup()"
            raise RuntimeError(msg)
        ceilings: dict[int, int | None] = {}
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=problem_stage.name,
                spec_override=None,
            )
            candidates: list[int] = []
            if stage_cfg.max_workers is not None:
                candidates.append(stage_cfg.max_workers)
            if stage_cfg.max_workers_per_node is not None:
                candidates.append(stage_cfg.max_workers_per_node * num_nodes)
            ceilings[stage_index] = min(candidates) if candidates else None
        return ceilings

    @staticmethod
    def _format_floor_unmet_message(
        *,
        stage_name: str,
        target_min: int,
        current: int,
        stage_cfg: SaturationAwareStageConfig,
        num_nodes: int,
        donor_attempted: bool,
        donor: DonorCandidate | None,
    ) -> str:
        """Build the operator-actionable message for an unmet minimum-worker floor.

        Distinguishes the two failure modes (no eligible donor vs.
        donor selected but retry still failed) and, when a donor was
        selected, names that donor so operators can correlate the
        failure with the donation log line.
        """
        if donor_attempted:
            donor_label = (
                f" (donor_stage_index={donor.stage_index}, donor_worker_id={donor.worker_id!r})"
                if donor is not None
                else ""
            )
            donor_clause = (
                "donor fallback attempted but post-donation retry returned no placement"
                f"{donor_label}; the donor's freed slot does not match the receiver's shape"
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

    def _update_regime_aware_aggressiveness(self, problem_state: data_structures.ProblemState) -> None:
        """Detect the cluster's Halfin-Whitt regime and re-resolve thresholds on transition.

        Computes the per-cycle regime signal, applies hysteresis via
        ``update_regime_state``, and on a regime transition drops
        every stage's ``resolved_thresholds`` and threshold-relative
        classifier history so the next call to
        ``_ensure_thresholds_resolved`` re-derives thresholds with the
        appropriate effective aggressiveness. Cycles whose signal is
        unavailable (some active stage has not populated
        ``num_used_slots`` / ``num_empty_slots`` yet) leave the regime
        state and resolved thresholds untouched. Respects the
        ``enable_regime_aware_aggressiveness`` flag.

        """
        if not self._config.enable_regime_aware_aggressiveness:
            return

        signal = _aggregate_cluster_regime_signal(problem_state)
        transitioned = update_regime_state(
            self._regime_state,
            signal,
            streak_cycles=self._config.regime_transition_streak_cycles,
            exit_band_multiplier=EXIT_BAND_MULTIPLIER,
        )
        if not transitioned:
            return

        effective = self._effective_aggressiveness(self._config.stage_defaults.saturation_aggressiveness)
        self._log_regime_transition(self._regime_state.current_regime, signal, effective)
        for runtime in self._stage_states.values():
            runtime.resolved_thresholds = None
            runtime.classifier_state = StageState.NORMAL
            runtime.classifier_streak = 0


def _select_workers_to_delete_youngest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
) -> list[str]:
    """Pick ``delete_count`` workers to delete, youngest first.

    Sort key: ``(age ASC, worker_id ASC)``. Workers missing from
    ``worker_ages`` are treated as age 0 (newly observed). The
    ``worker_id`` tiebreaker keeps the choice deterministic when
    every worker has the same age.

    Args:
        worker_ids: Worker ids in the stage's current snapshot.
        worker_ages: Cluster-wide worker ages.
        delete_count: Number of workers to return. Clamped to
            ``len(worker_ids)``.

    Returns:
        The first ``delete_count`` worker ids of the youngest-first
        ordering.

    """
    ranked = sorted(
        ((worker_ages.get(wid, 0), wid) for wid in worker_ids),
        key=lambda pair: (pair[0], pair[1]),
    )
    return [wid for _, wid in ranked[:delete_count]]


def _aggregate_cluster_regime_signal(problem_state: data_structures.ProblemState) -> RegimeSignal:
    """Aggregate the cluster's regime-detection inputs from per-stage state.

    Sums worker counts, used slots, and empty slots across every stage
    in ``problem_state``, then delegates to ``compute_regime_signal``.
    The returned signal is unavailable when any active worker stage
    still carries the ``0/0`` no-signal sentinel, or when no stage
    reports any slot occupancy at all.
    """
    total_workers = 0
    total_used = 0
    total_empty = 0
    for stage in problem_state.rust.stages:
        stage_workers = len(stage.worker_groups)
        stage_slots = stage.num_used_slots + stage.num_empty_slots
        if stage_workers > 0 and not stage.is_finished and stage_slots == 0:
            return compute_regime_signal(
                total_workers=total_workers + stage_workers,
                total_used_slots=0,
                total_empty_slots=0,
            )
        total_workers += stage_workers
        total_used += stage.num_used_slots
        total_empty += stage.num_empty_slots
    return compute_regime_signal(
        total_workers=total_workers,
        total_used_slots=total_used,
        total_empty_slots=total_empty,
    )
