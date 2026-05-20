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

Convergence is intentionally damped per cycle so the scheduler
tolerates noisy measurements without oscillating. Phase C grows
each stage by at most ``min(positive_intent, ceiling - current,
cluster_placement_capacity)`` workers per cycle and Phase D shrinks
each stage by at most ``min(max(-intent, ceiling_excess),
allowed_by_floor, fraction_cap)`` workers per cycle. The per-cycle
fraction cap (``max_scale_down_fraction_per_cycle``) bounds the
descent rate even when ``ceiling_excess`` or negative intent would
ask for a larger removal, trading absolute speed for stability under
noisy probes. Operators tune the cap per stage on the
``SaturationAwareStageConfig`` when faster reactions are required.
"""

import math
import types
from collections.abc import Mapping

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
from cosmos_xenna.pipelines.private.scheduling_py.errors import SchedulerInvariantError
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
from cosmos_xenna.pipelines.private.scheduling_py.stabilization import _RecommendationHistory
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
        _worker_ready_first_seen_at: Per-worker wall-clock timestamp
            of the first cycle in which the worker was observed in
            ``ProblemStageState.worker_groups`` (i.e. when the actor
            pool reported it as READY). Distinct from
            ``_worker_ages``: ``_worker_ages`` measures "cycles since
            planner add" and starts ticking the moment a fresh
            placement is staged, so a stage with a multi-minute model
            load already has a non-zero ``_worker_ages`` for the
            actor that has not yet finished ``stage_setup``.
            ``_worker_ready_first_seen_at`` only starts the clock
            once the actor reaches READY and contributes to slot
            signals, which is what the warmup grace mechanisms
            (per-worker measurement grace, donor warmup grace) need
            to suppress noise from freshly-warmed actors. Mutated
            each cycle by ``_refresh_worker_ready_first_seen`` from
            the live ``problem_state.rust.stages[*].worker_groups``
            snapshot; workers no longer observed are dropped so the
            map mirrors live READY state.
        _last_intent_deltas: Per-stage signed worker-count intent
            produced by the per-stage decision pipeline on the most
            recent ``autoscale()`` call. Saturation-driven scale-up
            consumes the positive intents in :meth:`_run_phase_c_grow`
            (capping each request at the available headroom and
            cluster placement budget) and saturation-driven scale-down
            consumes the negative intents in :meth:`_run_phase_d_shrink`
            (combining them with the shrink eligibility filter). The
            map is also surfaced unchanged for tests and observability.
            Reset on every ``autoscale()`` cycle. Finished stages are
            absent (skipped by ``_compute_intent_deltas``).
        _stage_spec_overrides: Per-stage config overrides sourced from
            ``StageSpec.saturation_aware`` and supplied to ``__init__``
            by ``streaming.Autoscaler``. These outrank named overrides
            and cluster defaults during runtime config resolution.
            Validated against the cluster ``config`` in ``__init__``
            and wrapped in a ``types.MappingProxyType`` so the stored
            view rejects post-construction mutation at runtime; the
            type is widened to ``Mapping`` to express the read-only
            contract through static analysis.
        _stuck_plan_counters: Per-stage count of consecutive
            ``autoscale()`` cycles where Phase C had a positive
            intent but could not place the full request. Increments
            when ``added < intent`` (cluster placement exhausted or
            a higher-priority downstream stage consumed the headroom
            first under DAG-priority ordering); resets to ``0`` on a
            cycle where the stage either fully met its intent or had
            no positive intent. The counter is the input the
            pipeline-level ``stuck_plan_detection_cycles`` watchdog
            consumes when it is wired in.
        _recommendation_histories: Per-stage asymmetric
            stabilization-window buffers, sized by the resolved
            ``stabilization_window_cycles_up`` /
            ``stabilization_window_cycles_down`` of each stage's
            effective config. Threaded into
            :func:`run_per_stage_pipeline` each cycle so the
            per-stage decision pipeline gates the raw delta against a
            consensus check before the growth-mode transition runs.
            Populated at ``setup()`` and re-built only when ``setup()``
            is called again; per-cycle config flux (regime-aware
            aggressiveness lift) does not affect window sizes.
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

    def __init__(
        self,
        config: SaturationAwareConfig,
        stage_spec_overrides: dict[str, SaturationAwareStageConfig] | None = None,
    ) -> None:
        """Initialize the scheduler.

        Constructor injection is the single entry point for installing
        per-stage overrides; the override map is cross-validated against
        ``config`` here so a misconfigured pipeline fails at build time,
        before ``setup()`` runs.

        Args:
            config: Cluster-wide ``SaturationAwareConfig``. Stored by
                reference; per-stage configs are resolved lazily via
                ``config.get_effective_stage_config``.
            stage_spec_overrides: Per-stage overrides sourced from
                ``StageSpec.saturation_aware`` and keyed by runtime
                stage name. These outrank named overrides
                (``config.per_stage_overrides``) and the cluster
                defaults (``config.stage_defaults``) when the runtime
                resolver consults them. ``None`` (the default) and an
                empty mapping behave identically: no overrides are
                installed and no validation is performed.

        Raises:
            ValueError: If any override config violates a cluster-wide
                guardrail (see
                ``SaturationAwareConfig.validate_effective_stage_configs``).
                The error fires synchronously from the constructor so
                misconfigurations surface during ``Autoscaler.__init__``
                rather than mid-``autoscale()``.

        """
        self._config = config
        self._problem: data_structures.Problem | None = None
        self._stage_states: dict[str, _StageRuntimeState] = {}
        self._stage_names: list[str] = []
        # ``stage_spec_overrides`` is validated eagerly so a misconfigured
        # pipeline cannot reach the per-cycle hot path with a weakened
        # cross-stage donor anti-flap guardrail. The empty-input fast path
        # skips the validator call to keep constructor cost negligible for
        # the common case where no spec-level overrides are present. The
        # caller's map is copied into a fresh ``dict`` and then wrapped in
        # a ``MappingProxyType`` so subsequent caller mutations cannot
        # leak in and post-construction mutation of the stored map fails
        # at runtime (not just by underscore-prefix convention).
        self._stage_spec_overrides: Mapping[str, SaturationAwareStageConfig] = types.MappingProxyType({})
        if stage_spec_overrides:
            self._config.validate_effective_stage_configs(tuple(stage_spec_overrides.values()))
            self._stage_spec_overrides = types.MappingProxyType(dict(stage_spec_overrides))
        self._regime_state: RegimeDetectorState = RegimeDetectorState()
        self._worker_ages: dict[str, int] = {}
        # Wall-clock first-seen timestamps are tracked separately from planner
        # ages because the warmup grace mechanisms need a clock that starts at
        # READY transition, not at planner add. A stage with a 10-minute model
        # load has its planner-age already at ~60 cycles by the time the actor
        # is ready; the donor and measurement graces would be silently
        # bypassed if they consumed planner ages. Storing wall-clock
        # timestamps (instead of incrementing cycle counts) lets the grace
        # comparison be a direct seconds-vs-seconds check and stays correct
        # even if the autoscale interval drifts (catch-up cycles, planner
        # stalls, or any operator-driven mid-run interval change).
        self._worker_ready_first_seen_at: dict[str, float] = {}
        # Per-cycle cache of the donor warmup excluded set. Populated at the
        # top of autoscale() from _build_donor_warmup_excluded_ids; consumed
        # by the saturation-mode cross-stage donor and Phase D shrink to skip
        # workers younger than donor_warmup_grace_s. Lives on the instance
        # rather than threading it through every call so all consumers see a
        # single consistent snapshot for the cycle.
        self._donor_warmup_excluded_ids: frozenset[str] = frozenset()
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
        # Per-stage stabilization-window buffer. Allocated at ``setup()`` once
        # the pipeline shape is known; the per-cycle pipeline reads this map
        # via :meth:`_compute_intent_deltas` and relies on the same
        # ``_RecommendationHistory`` instance across cycles so the buffer can
        # actually accumulate consensus.
        self._recommendation_histories: dict[str, _RecommendationHistory] = {}

    def _stage_cfg(self, stage_name: str) -> SaturationAwareStageConfig:
        """Resolve the effective stage config including ``StageSpec`` overrides."""
        return self._config.get_effective_stage_config(
            stage_name=stage_name,
            spec_override=self._stage_spec_overrides.get(stage_name),
        )

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
        self._worker_ready_first_seen_at = {}
        self._donor_warmup_excluded_ids = frozenset()
        self._floor_stuck_counters = {}
        self._last_intent_deltas = {}
        self._stuck_plan_counters = {}
        self._cycle_counter = 0
        self._last_donation_cycle = {}
        self._donations_received_this_cycle = {}
        # Build per-stage stabilization-window buffers from the resolved
        # effective config; both windows are config-time invariants
        # (cross-validated in ``SaturationAwareStageConfig.__attrs_post_init__``)
        # and do not flex during runtime, so the buffers can outlive every
        # ``autoscale()`` cycle without per-cycle re-allocation.
        self._recommendation_histories = {
            name: _RecommendationHistory(
                window_up=self._stage_cfg(name).stabilization_window_cycles_up,
                window_down=self._stage_cfg(name).stabilization_window_cycles_down,
            )
            for name in self._stage_names
        }

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
            stage_cfg = self._stage_cfg(stage.stage_name)
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

    def _check_problem_state_shape_before_phase_a(self, problem_state: data_structures.ProblemState) -> None:
        """Reject problem / problem_state shape drift before phase-specific indexing."""
        if self._problem is None:
            msg = "_check_problem_state_shape_before_phase_a called before setup()"
            raise RuntimeError(msg)
        problem_stages = self._problem.rust.stages
        runtime_stages = problem_state.rust.stages
        if len(runtime_stages) != len(problem_stages):
            msg = (
                f"Before {PhaseBoundary.PHASE_A}: problem_state has {len(runtime_stages)} stages "
                f"but problem has {len(problem_stages)}. The autoscale cycle snapshot is corrupted."
            )
            raise SchedulerInvariantError(msg)
        for stage_index, (problem_stage, runtime_stage) in enumerate(zip(problem_stages, runtime_stages, strict=True)):
            if runtime_stage.stage_name == problem_stage.name:
                continue
            msg = (
                f"Before {PhaseBoundary.PHASE_A}: stage index {stage_index} has problem stage "
                f"{problem_stage.name!r} but problem_state stage {runtime_stage.stage_name!r}. "
                "The autoscale cycle snapshot is corrupted."
            )
            raise SchedulerInvariantError(msg)

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
            time: Current wall-clock time in seconds. Threaded
                through the per-worker measurement grace and donor
                warmup grace helpers as the elapsed-time reference
                so a worker's "ready age" is real seconds since its
                first observation in ``worker_groups``, not an
                approximation derived from the cycle counter (which
                drifts when cycles are uneven, e.g. during
                catch-up loops or planner stalls).
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
        if self._problem is None:
            msg = "SaturationAwareScheduler.autoscale() called before setup()"
            raise RuntimeError(msg)

        self._check_problem_state_shape_before_phase_a(problem_state)
        self._cycle_counter += 1
        self._donations_received_this_cycle = {}
        # Refresh ready first-seen timestamps from this cycle's snapshot before
        # any phase reads them. The per-worker measurement grace consumes them
        # inside _compute_intent_deltas; Phase D shrink and the saturation-mode
        # cross-stage donor consult the resulting warmup-grace excluded set.
        self._refresh_worker_ready_first_seen(problem_state, now=time)
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
        # Cache the per-cycle donor warmup excluded set immediately after
        # AutoscalePlanContext.from_problem_state() seeds the planner from
        # ``problem_state``. ``ctx.worker_ids_by_stage()`` at this point
        # reflects the cycle-start snapshot (the live worker set as
        # observed by the actor pool); Phase A (delete + grow), Phase B
        # (floor), Phase C (grow), and Phase D (shrink) have NOT run yet
        # and have not mutated the planner. The cache is therefore a
        # cycle-start snapshot of "which observed workers are still in
        # warmup according to ``donor_warmup_grace_s``", and it stays
        # constant for the rest of the cycle so:
        #
        #   * Phase B floor donor selection
        #     (``select_youngest_eligible_donor``) intentionally bypasses
        #     this cache because the floor is a hard structural
        #     requirement; deadlocking on warmup-protected donors is
        #     worse than killing a young donor.
        #   * Phase C saturation-mode cross-stage donor selection
        #     (``find_saturation_donor``) consumes this cache so a
        #     receiver does not absorb a donor still in its own warmup.
        #   * Phase D shrink (``select_workers_to_remove_oldest_first``)
        #     consumes this cache so a saturation-driven shrink cannot
        #     delete a worker that has not yet contributed real
        #     measurements.
        #
        # Workers added during the current cycle are absent from this cache
        # because they were not in the cycle-start snapshot. Add paths:
        # Phase A grow (manual stage placement), Phase B floor grow / floor
        # donor, and Phase C saturation grow / saturation donor. Phase D
        # only removes workers, so it cannot contribute new ids to this
        # cache. Excluding intra-cycle additions is intentional - those
        # workers have no first-seen timestamp yet, so any warmup
        # decision would be vacuous. See ``_build_donor_warmup_excluded_ids``
        # for the per-stage filter that consumes this cache.
        self._donor_warmup_excluded_ids = self._build_donor_warmup_excluded_ids(
            ctx.worker_ids_by_stage(),
            now=time,
        )
        self._run_phase_a_delete(ctx, problem_state)
        self._run_phase_a_grow(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_A, problem=self._problem, ctx=ctx)

        self._run_phase_b_floor(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_B, problem=self._problem, ctx=ctx)

        self._last_intent_deltas = self._compute_intent_deltas(ctx, problem_state, now=time)
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
        # Filter ``curr_counters`` to only the stages Phase C touched
        # (non-finished); a stage that finished mid-run leaves its counter
        # untouched at the prior cycle's value, which would otherwise surface
        # as an illegal ``prev == curr`` transition. ``prev_counters`` is not
        # filtered because the helper only iterates ``curr_counters`` and
        # looks up ``prev_counters`` by key.
        active_stage_names = {stage.stage_name for stage in problem_state.rust.stages if not stage.is_finished}
        check_stuck_plan_monotonicity(
            prev_counters=prev_stuck_plan_counters,
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

    def _refresh_worker_ready_first_seen(
        self,
        problem_state: data_structures.ProblemState,
        now: float,
    ) -> None:
        """Update per-worker ready first-seen timestamps from this cycle's snapshot.

        For every worker observed in
        ``problem_state.rust.stages[*].worker_groups`` (workers the
        actor pool reports as READY), records ``now`` as the
        first-seen timestamp on first observation and carries an
        existing timestamp forward on subsequent cycles. Workers
        absent from the snapshot are dropped because their pool
        actor was lost between cycles, so any prior timestamp no
        longer corresponds to live measurement state.

        ::

           prev_seen = self._worker_ready_first_seen_at.get(wid)
           new map:
             wid first observed: prev_seen is None -> store now
             wid carried forward: prev_seen is set -> keep it
             wid disappeared:                       -> not added

        Drives the warmup grace mechanisms downstream:

          * :meth:`_aggregate_slot_signals_excluding_warmup` excludes
            workers younger than ``worker_warmup_measurement_grace_s``
            from EWMA contribution.
          * Phase D shrink and saturation-mode cross-stage donor
            selection exclude workers younger than
            ``donor_warmup_grace_s`` from victim / donor candidate
            pools.

        Args:
            problem_state: The cycle's runtime snapshot. Read-only.
            now: Wall-clock time at the top of the autoscale cycle,
                as passed to :meth:`autoscale`. New READY workers get
                this value as their first-seen timestamp.

        """
        new_seen: dict[str, float] = {}
        for stage in problem_state.rust.stages:
            for worker_group in stage.worker_groups:
                new_seen[worker_group.id] = self._worker_ready_first_seen_at.get(worker_group.id, now)
        self._worker_ready_first_seen_at = new_seen

    def _aggregate_slot_signals_excluding_warmup(
        self,
        runtime_stage: rust_data_structures.ProblemStageState,
        stage_cfg: SaturationAwareStageConfig,
        now: float,
    ) -> tuple[int, int]:
        """Re-aggregate per-worker slot signals, dropping workers in the warmup grace window.

        Cold-start readings from a freshly-ready worker pull the
        slots-empty ratio toward 1.0 because the dispatcher takes
        several cycles to fill the new actor's queue. EWMA-smoothing
        these readings drags the running average toward
        "over-provisioned" and risks a false Phase D shrink one cycle
        after a Phase B / Phase C grow. Excluding workers whose ready
        age (``now - first_seen_at``) is below
        ``worker_warmup_measurement_grace_s`` lets the EWMA see only
        steady-state samples from mature actors.

        ``input_queue_depth`` is a stage-level signal (not per-worker)
        and is therefore left untouched; the caller continues to read
        it directly from ``runtime_stage``.

        ::

           filter loop (per worker_group wg in stage):
             first_seen = self._worker_ready_first_seen_at.get(wg.id)
             if first_seen is None or (now - first_seen) < grace_s:
                 skip (warmup)
             else:
               mature_used  += wg.num_used_slots
               mature_empty += slots_per_worker - wg.num_used_slots

           all-warmup case (no mature workers): return (0, 0)
                       -> _resolve_classifier_signal observes
                          total_slots == 0 and carries forward the
                          last valid EWMA, holding the classifier
                          state constant until a worker matures.

        For non-SPMD stages each ``ProblemWorkerGroupState``
        corresponds to a single actor; ``num_used_slots`` is that
        actor's used-slot count and the per-group capacity is just
        ``slots_per_worker``. SPMD stages pack multiple actors into
        one worker_group and share a first-seen timestamp, so the
        warmup admission decision is at the group level (every
        actor in the group reached READY in the same cycle).

        Capacity arithmetic for SPMD groups:
        ``ActorPool.worker_group_num_used_slots`` sums occupied
        slots across all K actors of a worker_group. The per-group
        slot capacity is therefore ``slots_per_worker * K`` where
        ``K = len(worker_group.resources)`` (one ``WorkerResourcesInternal``
        per SPMD actor). Computing empties as
        ``slots_per_worker - num_used_slots`` (treating the group as
        a single actor) silently under-counts by ``(K - 1) *
        slots_per_worker`` and biases the classifier toward
        SATURATED. The fix uses the group-level capacity.

        Args:
            runtime_stage: The Rust ``ProblemStageState`` for the
                stage being filtered.
            stage_cfg: The stage's effective config; carries
                ``worker_warmup_measurement_grace_s``.
            now: Wall-clock time at the top of the autoscale cycle.
                Compared against each worker's first-seen timestamp
                to compute its ready age in seconds.

        Returns:
            ``(num_used_slots, num_empty_slots)`` after excluding
            warmup workers. When the configured grace is non-positive
            or the stage has no ``worker_groups`` to filter, returns
            the unfiltered totals from ``runtime_stage`` so existing
            behaviour is preserved.

        """
        grace_s = stage_cfg.worker_warmup_measurement_grace_s
        if grace_s <= 0 or not runtime_stage.worker_groups:
            return runtime_stage.num_used_slots, runtime_stage.num_empty_slots
        slots_per_worker = runtime_stage.slots_per_worker
        mature_used = 0
        mature_empty = 0
        for worker_group in runtime_stage.worker_groups:
            first_seen = self._worker_ready_first_seen_at.get(worker_group.id)
            if first_seen is None or (now - first_seen) < grace_s:
                continue
            # SPMD-aware capacity: per-group capacity is slots_per_worker * actor_count.
            # actor_count is len(worker_group.resources): one WorkerResourcesInternal per
            # SPMD actor (= 1 for non-SPMD groups, = K for K-way SPMD).
            actor_count = len(worker_group.resources)
            group_capacity = slots_per_worker * actor_count
            used = worker_group.num_used_slots
            mature_used += used
            mature_empty += max(0, group_capacity - used)
        return mature_used, mature_empty

    def _build_donor_warmup_excluded_ids(
        self,
        worker_ids_by_stage: list[list[str]],
        now: float,
    ) -> frozenset[str]:
        """Build the set of worker ids in donor warmup grace, indexed across all stages.

        Each stage contributes ids of workers whose ready age
        (``now - first_seen_at``) is below the stage's effective
        ``donor_warmup_grace_s``. Used by Phase D shrink (filters
        victim candidates) and the saturation-mode cross-stage donor
        (filters donor candidates) to leave freshly-warmed workers
        alone for at least one full warmup horizon. Floor donor
        selection (``select_youngest_eligible_donor``) does NOT
        consult this set because the floor is a hard structural
        requirement; deadlocking on warmup-protected donors is worse
        than killing a young donor.

        Args:
            worker_ids_by_stage: Per-stage live worker ids in problem
                order. Built from the planner's
                ``ctx.worker_ids_by_stage()``.
            now: Wall-clock time at the top of the autoscale cycle.

        Returns:
            A frozen set of worker ids. Empty when every stage's
            grace is configured to ``0`` or when no observed worker
            falls within its stage's grace window. A worker that is
            in ``worker_ids_by_stage`` but not in
            ``_worker_ready_first_seen_at`` is treated as warmup
            (defensive: it has not been observed in ``worker_groups``
            yet, so any prior knowledge of when it became READY is
            unavailable).

        """
        excluded: set[str] = set()
        for stage_index, worker_ids in enumerate(worker_ids_by_stage):
            stage_name = self._stage_names[stage_index]
            stage_cfg = self._stage_cfg(stage_name)
            grace_s = stage_cfg.donor_warmup_grace_s
            if grace_s <= 0:
                continue
            for worker_id in worker_ids:
                first_seen = self._worker_ready_first_seen_at.get(worker_id)
                if first_seen is None or (now - first_seen) < grace_s:
                    excluded.add(worker_id)
        return frozenset(excluded)

    def _compute_intent_deltas(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
        *,
        now: float,
    ) -> dict[str, int]:
        """Compute the per-stage signed worker-count intent for this cycle.

        For each non-finished stage, calls
        :func:`run_per_stage_pipeline` with the live slot signals
        sourced from ``ProblemStageState``, the post-Phase-B worker
        count read from the planner context, and the per-stage
        ``_RecommendationHistory`` allocated at ``setup()``. The
        pipeline records the raw delta into the history and replaces
        it with ``0`` if the asymmetric stabilization window has not
        yet seen a sustained recommendation in the same direction;
        the returned delta is therefore the post-stabilization-gate
        intent that Phase C / Phase D must execute.

        Per-worker measurement grace (gated by
        :attr:`SaturationAwareStageConfig.worker_warmup_measurement_grace_s`)
        re-aggregates the slot signals before the EWMA absorbs them:
        workers whose ready-age is below the configured grace are
        excluded from ``num_used_slots`` and ``num_empty_slots`` so a
        freshly-ready actor cannot drag the steady-state ratio toward
        "over-provisioned" while the dispatcher is still racing to
        fill its queue. ``input_queue_depth`` is unfiltered because
        it is a stage-level signal. When every observed worker is
        in warmup, the filter returns ``(0, 0)`` and
        ``_resolve_classifier_signal`` carries forward the last valid
        EWMA, holding the classifier state until at least one worker
        matures.

        Setup-phase quiescence (gated by
        :attr:`SaturationAwareStageConfig.setup_phase_quiescence_enabled`)
        intercepts two distinct half-initialised states before they
        feed the classifier or Phase C scale-up:

          1. **Cold-start** (``pending > 0`` and ``ready == 0``):
             every slot signal is zero by construction (no actor has
             ever processed a task), so the classifier would emit
             arbitrary noise. The pipeline is skipped entirely; the
             stage's recommendation history and runtime state are
             left untouched, and the intent map carries no entry.
             Phase C and Phase D both no-op for the stage this cycle
             via their ``intents.get(stage_name, 0)`` fallback.
          2. **Hot-pending** (``pending > 0`` and ``ready > 0``):
             ready actors are producing valid signal, so the
             classifier and the stabilization gate continue to run
             on real measurements. A positive intent (Phase C
             scale-up) is suppressed because adding another worker
             while a prior add has not finished setup would amplify
             cold-start noise and risk over-provisioning once the
             pending actor lands. Negative intents (Phase D
             scale-down) are preserved -- they only act on ready
             actors, and the still-pending actor's lifecycle is
             unaffected.

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
                ``num_empty_slots``, ``input_queue_depth``) and the
                ``num_pending_actors`` quiescence signal populated in
                the streaming layer.
            now: Wall-clock time at the top of the autoscale cycle,
                threaded into the per-worker measurement grace
                helper for elapsed-time comparisons against each
                worker's first-seen-READY timestamp.

        Returns:
            Mapping of stage name -> signed intent. Positive values
            indicate scale-up intent, negative values scale-down,
            zero a no-op. Finished stages and cold-start-quiescent
            stages are absent.
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
            stage_cfg = self._stage_cfg(stage_name)
            current_workers = len(worker_ids_by_stage[stage_index])
            pending_actors = runtime_stage.num_pending_actors
            # The quiescence check evaluates the snapshot's ready count
            # (workers visible to the actor pool at observation time), not
            # the post-Phase-B planner count. Phase B's floor enforcement
            # may have staged a fresh add for this same cycle, but that
            # add is not yet visible to the streaming snapshot's slot
            # signals - the per-worker measurements still come from the
            # ready actors that existed at observation time, of which
            # there are zero in the cold-start case.
            ready_at_snapshot = len(runtime_stage.worker_groups)
            quiescence_active = stage_cfg.setup_phase_quiescence_enabled and pending_actors > 0
            if quiescence_active and ready_at_snapshot == 0:
                # Cold-start: zero ready actors means every slot signal is
                # an artifact of the empty pool, not a real measurement.
                # Recording it would corrupt the classifier streak and the
                # stabilization-window history; better to wait until at
                # least one actor reaches ready and the signal is real.
                logger.debug(
                    f"saturation-aware: stage {stage_name!r} cold-start quiescent "
                    f"(pending={pending_actors}, ready=0); skipping intent pipeline."
                )
                continue
            # The stabilization-window buffer is allocated alongside the
            # runtime state in ``setup()``; a missing entry would mean the
            # ``problem`` -> ``problem_state`` shape contract above somehow
            # admitted a stage the runtime state map already rejected.
            history = self._recommendation_histories[stage_name]
            # Per-worker measurement grace: drop slot-signal contributions
            # from workers younger than ``worker_warmup_measurement_grace_s``
            # so the EWMA does not absorb cold-start noise from freshly-ready
            # actors. ``input_queue_depth`` is a stage-level signal (not
            # per-worker) and is therefore left unfiltered.
            num_used_slots, num_empty_slots = self._aggregate_slot_signals_excluding_warmup(
                runtime_stage=runtime_stage,
                stage_cfg=stage_cfg,
                now=now,
            )
            delta = run_per_stage_pipeline(
                stage_state=stage_state,
                num_used_slots=num_used_slots,
                num_empty_slots=num_empty_slots,
                input_queue_depth=runtime_stage.input_queue_depth,
                current_workers=current_workers,
                config=stage_cfg,
                recommendation_history=history,
            )
            if quiescence_active and delta > 0:
                # Hot-pending: ready actors observe real saturation, but a
                # prior scale-up has not finished setup yet. Suppress the
                # additional Phase C add to avoid amplifying cold-start
                # noise; the next cycle re-evaluates with the new ready
                # count. Phase D shrink (delta < 0) is left untouched
                # because the still-pending actor is independent of any
                # ready-actor removal Phase D might perform.
                logger.debug(
                    f"saturation-aware: stage {stage_name!r} hot-pending quiescent "
                    f"(pending={pending_actors}, ready={ready_at_snapshot}); "
                    f"clamping Phase C intent +{delta} -> 0."
                )
                delta = 0
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
        consumes when it is wired in.

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
                donor_stage_name = self._attempt_cross_stage_donation(
                    ctx=ctx,
                    receiver_stage_index=stage_index,
                    receiver_stage_name=stage_name,
                )
                if donor_stage_name is None:
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
                self._record_donation_success(
                    donor_stage_name=donor_stage_name,
                    receiver_stage_name=stage_name,
                )
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
    ) -> str | None:
        """Try to free a placement for a saturation-driven receiver.

        Selects an eligible donor via
        :func:`find_saturation_donor` (five anti-flap layers + strict
        upstream + master toggle) and removes it from the planner.
        The donor cooldown and receiver per-cycle counters are NOT
        updated here; the caller MUST advance them via
        :meth:`_record_donation_success` only after the immediate
        ``try_add_worker`` retry succeeds for the receiver. Recording
        cooldowns before retry success would penalise the donor for
        a transfer that never completed end-to-end.

        Returns:
            The donor stage name when a donor was selected and
            removed (the caller now owns the receiver retry and the
            cooldown bookkeeping). ``None`` when the master toggle is
            off, when no eligible donor exists, or when the planner
            refuses the selected donor (defensive guard).

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
        stage_configs = {name: self._stage_cfg(name) for name in self._stage_names}

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
            excluded_worker_ids=self._donor_warmup_excluded_ids,
        )
        if donor is None:
            return None

        donor_stage_name = self._stage_names[donor.stage_index]
        if not ctx.try_remove_worker(donor.stage_index, donor.worker_id):
            logger.warning(
                f"[scheduler] saturation-mode donor: stage {donor_stage_name!r} "
                f"worker {donor.worker_id!r} selected by donor helper but planner "
                "refused removal; donation cancelled and receiver retry skipped."
            )
            return None

        logger.info(
            f"[scheduler] saturation-mode donation: donor stage {donor_stage_name!r} "
            f"worker {donor.worker_id!r} (age={donor.age}) -> receiver stage "
            f"{receiver_stage_name!r} at cycle {self._cycle_counter} (pending retry)."
        )
        return donor_stage_name

    def _record_donation_success(
        self,
        *,
        donor_stage_name: str,
        receiver_stage_name: str,
    ) -> None:
        """Advance donor cooldown and receiver per-cycle counter on retry success.

        Called by the receiver-side caller after the post-donation
        ``try_add_worker`` retry completes successfully. Keeping the
        update split from
        :meth:`_attempt_cross_stage_donation` ensures cooldown state
        only reflects donations that placed a worker on the receiver,
        so a retry-failure path leaves the donor eligible to be
        revisited on the next planning cycle.
        """
        self._last_donation_cycle[donor_stage_name] = self._cycle_counter
        self._donations_received_this_cycle[receiver_stage_name] = (
            self._donations_received_this_cycle.get(receiver_stage_name, 0) + 1
        )

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
            stage_cfg = self._stage_cfg(stage_name)
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
                    self._log_phase_d_shrink_outcome(
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
            worker_host_gpu_used_fractions = self._extract_worker_host_gpu_used_fractions(
                runtime_stage=runtime_stage,
                host_gpu_used_fractions=host_gpu_used_fractions,
            )
            stage_warmup_excluded = sum(
                1 for wid in worker_ids_by_stage[stage_index] if wid in self._donor_warmup_excluded_ids
            )
            victims = select_workers_to_remove_oldest_first(
                worker_ids=worker_ids_by_stage[stage_index],
                worker_ages=worker_ages,
                delete_count=actual_remove,
                worker_used_slots=worker_used_slots,
                worker_host_gpu_used_fractions=worker_host_gpu_used_fractions,
                excluded_worker_ids=self._donor_warmup_excluded_ids,
            )
            for victim_id in victims:
                if not ctx.try_remove_worker(stage_index, victim_id):
                    msg = (
                        f"Phase D shrink: stage {stage_name!r} planner refused removal of "
                        f"worker {victim_id!r} selected from its own snapshot. This is a "
                        "scheduler defect; the planner state and the runtime snapshot disagree."
                    )
                    raise RuntimeError(msg)
            # ``effective_remove`` is the deletion count actually applied to the planner.
            # When the warmup-grace filter removed candidates from the eligible pool the helper
            # returns fewer victims than ``actual_remove``; logging ``actual_remove`` in that
            # case would mislead operators about how many workers were really shrunk and
            # mis-attribute the deficit to floor / fraction caps that did not bind.
            effective_remove = len(victims)
            self._log_phase_d_shrink_outcome(
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

    @staticmethod
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

        The message preamble names the dominant driver (classifier
        intent vs hard worker cap overflow) so operators can locate
        the source of the request without parsing the trailing
        kwargs. The trailing clause names the binding clamp:

          * floor cap (``allowed_by_floor`` was the smallest of the
            three),
          * per-cycle fraction cap (``fraction_cap`` was the
            smallest),
          * donor warmup grace (the floor / fraction caps did not
            bind but ``select_workers_to_remove_oldest_first``
            returned fewer workers than ``actual_remove`` because
            the per-cycle ``_donor_warmup_excluded_ids`` filter
            excluded eligible candidates), or
          * no clamp (full removal applied).

        Ties between ``fraction_cap`` and ``allowed_by_floor``
        resolve to the floor branch so the operator-configured
        floor is named in the log when both clamps would have
        produced the same deletion count.

        ``effective_remove`` is the count actually applied to the
        planner (``len(victims)`` after the warmup-grace filter).
        ``actual_remove`` is the post-clamp request before that
        filter ran. The two diverge only when the warmup grace
        excluded otherwise-eligible candidates.

        When both clamps bind in the same cycle (Stage 1 floor /
        fraction cap shrinks ``requested_remove`` to
        ``actual_remove`` AND Stage 2 warmup grace shrinks
        ``actual_remove`` to ``effective_remove``), both branches
        emit a log line so the operator can see the full deficit
        attribution. Only Stage 3 (cap-driven full removal) is
        suppressed when either deficit branch already fired,
        because Stage 3 reports the no-clamp success path.
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
        if actual_remove < requested_remove:
            deficit = requested_remove - actual_remove
            fraction_bound = fraction_cap < allowed_by_floor and fraction_cap == actual_remove
            if fraction_bound:
                logger.info(
                    f"{preamble}; per-cycle fraction cap left {effective_remove} removed "
                    f"(deficit={deficit}, current={current}, "
                    f"max_scale_down_fraction_per_cycle={max_scale_down_fraction_per_cycle}"
                    f"{cap_kwargs})."
                )
            else:
                logger.info(
                    f"{preamble}; floor cap left {effective_remove} removed "
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
        # Stage 3: cap-driven full removal -- only fires when no deficit branch fired,
        # because reporting "removed N workers" alongside a deficit message would be
        # contradictory.
        if cap_driven and not deficit_reported:
            logger.info(
                f"saturation-aware scale-down: stage {stage_name!r} hard worker cap "
                f"overflow removed {effective_remove} workers (current={current}, "
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
            stage_cfg = self._stage_cfg(problem_stage.name)
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
            stage_cfg = self._stage_cfg(problem_stage.name)
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
            stage_cfg = self._stage_cfg(problem_stage.name)
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
        every stage's ``resolved_thresholds``, threshold-relative
        classifier history, and stabilization-window recommendation
        buffer so the next call to ``_ensure_thresholds_resolved``
        re-derives thresholds with the appropriate effective
        aggressiveness and the post-transition cycles must rebuild
        gate consensus from scratch (pre-transition recommendations
        consumed a different threshold band and would otherwise let
        stale consensus leak into the new regime). Cycles whose
        signal is unavailable (some active stage has not populated
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
        for history in self._recommendation_histories.values():
            history.clear()


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
