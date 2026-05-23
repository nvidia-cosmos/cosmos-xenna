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
import threading
import time
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from ray.util.metrics import Histogram

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    data_structures as rust_data_structures,
)
from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.allocation_failures import emit_allocation_failure
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import (
    ResolvedThresholds,
    _resolve_auto_thresholds,
    derive_utilization_target,
)
from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import (
    BottleneckCycleContext,
    BottleneckEngagementState,
    BottleneckIdentity,
    HeterogeneityWarnState,
    compute_d_k,
    compute_heterogeneity_ratio,
    emit_bottleneck_score,
    identify_bottleneck,
    maybe_log_bottleneck_engagement,
)
from cosmos_xenna.pipelines.private.scheduling_py.dag_priority import compute_grow_priority_order
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
from cosmos_xenna.pipelines.private.scheduling_py.loop_watchdog import loop_watchdog
from cosmos_xenna.pipelines.private.scheduling_py.memory_pressure import MemoryPressureMonitor
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import (
    record_executed_delta,
    run_per_stage_pipeline,
)
from cosmos_xenna.pipelines.private.scheduling_py.pressure import (
    compute_capacity_target_workers,
)
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
from cosmos_xenna.pipelines.private.scheduling_py.stuck_plan import StuckPlanDetector
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger

# Module-level alias for ``time.monotonic`` so per-phase wrappers inside
# ``autoscale()`` can sample the wall-clock despite the method's
# ``time: float`` parameter shadowing the ``time`` module. Tests inject a
# deterministic clock via ``monkeypatch.setattr(saturation_aware, "_monotonic", fake)``.
_monotonic = time.monotonic

_PHASE_DURATION_HISTOGRAM = Histogram(
    name="xenna_scheduler_cycle_phase_duration_seconds",
    description=(
        "Wall-clock duration of one autoscale-cycle phase. "
        "Operators read this to locate which phase blew the cycle budget "
        "when the loop-watchdog WARN fires; the sum across phases "
        "approximates xenna_scheduler_cycle_duration_seconds."
    ),
    boundaries=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0],
    tag_keys=("phase", "pipeline"),
)


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
            ``autoscale()`` cycles where Phase C had a positive intent
            but could not place the full request. Increments when
            ``added < intent``; resets to ``0`` on any cycle where the
            stage met its intent or had none. Fed into
            ``StuckPlanDetector`` for the threshold-promoted INFO log
            and the per-stage Gauge / Counter. The post-Phase-D
            ``check_stuck_plan_monotonicity`` invariant filters out
            stages whose counter equals the prior cycle's snapshot
            (no observable transition), so a stage Phase C bailed
            before reaching the per-stage counter setter
            (allocation-failure absorption, donor-retry synthetic
            absorption) keeps its prior value without tripping the
            strict ``+1 / 0`` regression.
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
            saturation-mode cross-stage donor anti-flap layer.
        _last_donation_cycle: Per-stage record of the cycle at which
            each stage most recently donated a worker through the
            saturation-mode cross-stage path. Drives the
            ``cross_stage_donor_anti_flap_cycles``
            receiver-was-recent-donor block; missing entries mean the
            stage has never donated.

    """

    def __init__(
        self,
        config: SaturationAwareConfig,
        stage_spec_overrides: dict[str, SaturationAwareStageConfig] | None = None,
        *,
        pipeline_name: str = "",
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
            pipeline_name: Value of the ``pipeline`` Prometheus tag
                attached to every scheduler metric. Empty string when
                the caller has no job-level pipeline identifier.

        Raises:
            ValueError: If any override config violates a cluster-wide
                guardrail (see
                ``SaturationAwareConfig.validate_effective_stage_configs``).
                The error fires synchronously from the constructor so
                misconfigurations surface during ``Autoscaler.__init__``
                rather than mid-``autoscale()``.

        """
        self._config = config
        self._pipeline_name: str = pipeline_name
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
        # Cluster-wide kill switch on Phase C scale-up. Holds a polled
        # cache of Ray's object-store ``used_fraction`` so a single Ray
        # API call serves every scheduler cycle inside one
        # ``memory_pressure_polling_interval_s`` window. Reset by
        # ``setup()`` so a fresh pipeline starts with an empty cache.
        self._memory_pressure_monitor: MemoryPressureMonitor = MemoryPressureMonitor(
            polling_interval_s=config.memory_pressure_polling_interval_s,
            critical_threshold=config.memory_pressure_critical_threshold,
            pipeline_name=pipeline_name,
        )
        self._floor_stuck_counters: dict[str, int] = {}
        # Transient cycle-skip flag set by ``_try_add_worker_with_defense``
        # on absorbed exceptions; reset at the top of every Phase C grow.
        self._phase_c_allocation_failure: bool = False
        self._stuck_plan_detector: StuckPlanDetector = StuckPlanDetector()
        # Saturation-mode cross-stage donor anti-flap state.
        # ``_cycle_counter`` is monotonic and increments at the top of every
        # ``autoscale()`` call. ``_last_donation_cycle`` records the cycle
        # at which each stage most recently donated; missing entries mean
        # the stage has never donated. The dict is read only by the
        # receiver-was-recent-donor anti-flap gate; it is no longer a
        # donor-side cooldown.
        self._cycle_counter: int = 0
        self._last_donation_cycle: dict[str, int] = {}
        # Per-stage stabilization-window buffer. Allocated at ``setup()`` once
        # the pipeline shape is known; the per-cycle pipeline reads this map
        # via :meth:`_compute_intent_deltas` and relies on the same
        # ``_RecommendationHistory`` instance across cycles so the buffer can
        # actually accumulate consensus.
        self._recommendation_histories: dict[str, _RecommendationHistory] = {}
        # Measurement accumulators: ``update_with_measurements`` adds per
        # monitor tick, the per-cycle ``_consume_*`` helpers snapshot once
        # per autoscale cycle. The lock decouples the two cadences.
        # Two independent cumulative accumulators feed two independent
        # samplers:
        #   * ``_completed_counts`` + ``_last_throughput_sample`` -> rate
        #     (dcount / dt) consumed by the backlog-time pressure signal.
        #   * ``_completed_service_time_sums`` +
        #     ``_last_service_time_sum_sample`` -> mean per-task service
        #     time (dsum / dcount) consumed by the Forced-Flow-Law
        #     bottleneck score and cluster heterogeneity ratio.
        self._lock: threading.Lock = threading.Lock()
        self._completed_counts: dict[str, int] = {}
        self._last_throughput_sample: dict[str, tuple[int, float]] = {}
        self._completed_service_time_sums: dict[str, float] = {}
        # Service-time consumer keeps its OWN ``(count, sum)`` snapshot
        # so the call order with ``_consume_throughput_samples`` does
        # not matter; both helpers can run in any order in the same
        # autoscale cycle without their snapshots interfering.
        self._last_service_time_sample: dict[str, tuple[int, float]] = {}
        # Per-instance streak ledger for the cluster heterogeneity ratio
        # warn log. Owned at the instance level (not module-level)
        # because the scheduler may be re-instantiated across tests;
        # sharing a streak counter across instances would leak state
        # into a fresh scheduler. ``setup()`` calls ``.reset()`` so a
        # re-setup of the scheduler starts from a clean ledger.
        self._heterogeneity_state: HeterogeneityWarnState = HeterogeneityWarnState()
        # Bottleneck-decision integration state. All four are written
        # and read only from inside ``autoscale()`` running on the
        # streaming executor's single background thread, so no lock is
        # needed (verified call-graph: ``add_measurements`` -> main
        # thread; ``autoscale`` -> background ThreadPoolExecutor).
        # ``setup()`` resets these without acquiring the existing
        # ``self._lock`` because no consumer can be running at re-setup
        # time, mirroring the heterogeneity_state pattern above.
        # ``_s_k_ewma`` tracks the smoothed intrinsic per-task service
        # time S_k (consumed by the capacity sizer which divides by
        # the stage's slots/workers itself). ``_d_k_now`` and
        # ``_effective_capacities`` are the per-cycle actor-normalized
        # views computed in the bottleneck phase block; consumed by
        # ``identify_bottleneck``, the grow-priority order, the INFO
        # log, and the heterogeneity gauge so all four agree on the
        # same D_k definition.
        self._s_k_ewma: dict[str, float] = {}
        self._d_k_now: dict[str, float] = {}
        self._effective_capacities: dict[str, int] = {}
        self._last_bottleneck_meta: BottleneckIdentity | None = None
        self._bottleneck_engagement_state: BottleneckEngagementState = BottleneckEngagementState()
        # Per-stage latch for the Phase D bottleneck-protection INFO log.
        # The set holds the stage names that hit the protection branch on the
        # PREVIOUS cycle; on the current cycle, a stage logs only when it
        # transitions into the set (matches the once-per-streak pattern from
        # ``HeterogeneityWarnState.has_fired``). Cleared in ``_run_phase_d_shrink``
        # for any stage that is no longer protected this cycle, so a future
        # re-entry into protection re-arms a fresh INFO log.
        self._bottleneck_protected_stages_logged: set[str] = set()

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
        # ``setup()`` is pre-traffic: the streaming executor has not yet
        # wired ``update_with_measurements`` or ``autoscale``, so resetting
        # without ``self._lock`` is safe. The accumulator pair below is
        # still locked because re-``setup()`` of a recycled scheduler can
        # race the accumulator path in test fixtures.
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
        self._memory_pressure_monitor.reset()
        self._heterogeneity_state.reset()
        self._stuck_plan_detector.reset()
        # Bottleneck-decision state: NaN-seed every stage so the first
        # finite per-stage sample replaces (does not blend with) the
        # seed. ``setup()`` is pre-traffic so no lock is needed.
        # ``_d_k_now`` and ``_effective_capacities`` start empty; the
        # bottleneck phase block populates them on each cycle.
        self._s_k_ewma = {name: math.nan for name in self._stage_names}
        self._d_k_now = {}
        self._effective_capacities = {}
        self._last_bottleneck_meta = None
        self._bottleneck_engagement_state.reset()
        self._bottleneck_protected_stages_logged = set()
        self._cycle_counter = 0
        self._last_donation_cycle = {}
        # Measurement accumulator reset; locked because
        # ``update_with_measurements`` and the per-cycle ``_consume_*``
        # helpers also acquire ``self._lock``.
        with self._lock:
            self._completed_counts = {name: 0 for name in self._stage_names}
            self._last_throughput_sample = {}
            self._completed_service_time_sums = {name: 0.0 for name in self._stage_names}
            self._last_service_time_sample = {}
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

        Accumulates per-stage completed-task counts so :meth:`autoscale`
        can derive an ``observed_throughput`` sample without depending
        on a separate throughput estimator. ``streaming.Autoscaler``
        calls this on every monitor tick (cadence governed by the
        streaming loop, typically much faster than ``interval_s``);
        ``autoscale`` consumes the delta against the previously
        snapshotted ``(count, ts)`` and emits one throughput value
        per stage per cycle. The backlog-time pressure EWMA in
        ``pipeline.py`` absorbs the cycle-to-cycle noise.

        Counting policy: one tick per ``TaskMeasurement`` (i.e.
        ``len(stage_measurements.task_measurements)``). The matching
        drain-rate unit for ``input_queue_depth`` is "completed stage
        tasks per second", not "produced output items per second",
        so ``TaskMeasurement.num_returns`` is intentionally NOT
        summed here (a ``flat_map``-style stage that returns ``N``
        items per task should still count one queue drain per task).

        Args:
            time: Current wall-clock time in seconds. Currently
                unused - the autoscale-cycle timestamp is the
                canonical clock for the throughput delta. Accepted
                so the protocol signature in ``streaming.py``
                stays uniform across both scheduler kinds.
            measurements: Per-stage measurement batch since the
                previous tick. Iterated by position over
                ``measurements.rust.stages`` zipped with
                ``self._stage_names`` so the count attribution
                follows pipeline-DAG order rather than relying on
                stage-name hashing. Either empty (no measurements
                this tick, accepted as a no-op) or one entry per
                stage in DAG order.

        Raises:
            ValueError: ``measurements.rust.stages`` is non-empty
                and disagrees in length with ``self._stage_names``;
                a partial update would silently corrupt the
                bottleneck and throughput aggregates.

        """
        del time
        rust_stages = measurements.rust.stages
        # Shape validation runs BEFORE the lock so a corrupted
        # measurement batch cannot leave ``_completed_counts`` or
        # ``_completed_service_time_sums`` half-updated. An empty
        # ``rust_stages`` is the legitimate "no measurements this
        # tick" signal and is a no-op (the inner zip yields nothing).
        # Any non-empty list whose length disagrees with the
        # setup-time stage count is a Rust <-> Python boundary
        # violation: silent truncation here would corrupt the
        # bottleneck and throughput aggregates for every subsequent
        # cycle, so we fail loud with the same shape-mismatch
        # convention used by ``_resolve_thresholds``.
        if rust_stages and len(rust_stages) != len(self._stage_names):
            msg = (
                f"update_with_measurements shape mismatch: "
                f"measurements.rust.stages has {len(rust_stages)} entries but "
                f"setup() captured {len(self._stage_names)} stage names "
                f"(known: {sorted(self._stage_names)})"
            )
            raise ValueError(msg)
        with self._lock:
            for stage_name, stage_measurements in zip(self._stage_names, rust_stages, strict=False):
                task_measurements = stage_measurements.task_measurements
                count = len(task_measurements)
                if count == 0:
                    continue
                self._completed_counts[stage_name] = self._completed_counts.get(stage_name, 0) + count
                # Accumulate cumulative per-stage service-time sums in
                # the same loop so the bottleneck score / heterogeneity
                # ratio see the same per-cycle window the backlog-time
                # pressure throughput sample sees. ``duration()`` is the
                # Rust-side ``end - start`` accessor; each ``count``
                # contributes one sample, so ``mean = sum / count`` is a
                # straightforward Forced-Flow ``S_k`` estimate (V_k = 1
                # for Xenna's linear DAG, so D_k = S_k).
                sum_service = sum(tm.duration() for tm in task_measurements)
                self._completed_service_time_sums[stage_name] = (
                    self._completed_service_time_sums.get(stage_name, 0.0) + sum_service
                )

    def _consume_throughput_samples(self, now_ts: float) -> dict[str, float]:
        """Consume the per-stage completed-count delta and emit throughput samples.

        Called once per ``autoscale`` cycle. For each known stage, reads
        the running completed-count under ``self._lock``, computes
        ``dcount / dt`` against the previously snapshotted
        ``(count, ts)``, updates the snapshot, and returns the
        per-stage throughput sample. Stages that have never been
        sampled before return ``0.0`` (cold-start: the next cycle is
        the first sample with a valid ``dt``).

        Args:
            now_ts: Wall-clock seconds at the top of this autoscale
                cycle; used as the second component of the new
                ``(count, ts)`` snapshot.

        Returns:
            Mapping of stage name to ``observed_throughput_sample`` in
            tasks per second. Always contains an entry for every name
            in ``self._stage_names`` so the pipeline orchestrator can
            look up by stage name without a missing-key fallback.

        """
        samples: dict[str, float] = {}
        with self._lock:
            for stage_name in self._stage_names:
                now_count = self._completed_counts.get(stage_name, 0)
                prev = self._last_throughput_sample.get(stage_name)
                if prev is None:
                    samples[stage_name] = 0.0
                else:
                    prev_count, prev_ts = prev
                    dt = now_ts - prev_ts
                    dcount = max(0, now_count - prev_count)
                    samples[stage_name] = dcount / dt if dt > 0.0 else 0.0
                self._last_throughput_sample[stage_name] = (now_count, now_ts)
        return samples

    def _consume_service_time_samples(self) -> dict[str, float]:
        """Consume per-stage cumulative service-time deltas and emit mean ``S_k`` samples.

        Companion to :meth:`_consume_throughput_samples`. Computes the
        per-stage mean per-task service time over the in-cycle window
        as ``dsum / dcount`` against the previously snapshotted
        ``(count, sum)`` pair, then refreshes the snapshot for the
        next cycle.

        Cold-start contract: returns ``math.nan`` for any stage that
        has not yet observed a non-empty completed-task delta. The
        :func:`emit_bottleneck_score` and
        :func:`compute_heterogeneity_ratio` helpers fold ``math.nan``
        into their cold-start path (gauge observes ``NaN``, stage is
        excluded from the ``argmax_k D_k`` and from the heterogeneity
        ratio computation).

        The helper keeps its own ``(count, sum)`` snapshot in
        ``self._last_service_time_sample`` so the call order with
        :meth:`_consume_throughput_samples` does not matter - both
        helpers can run in any order in the same autoscale cycle
        without their snapshots interfering.

        Returns:
            Mapping of stage name to mean per-task service time in
            seconds (``S_k``), or ``math.nan`` for cold-start. Always
            contains an entry for every name in ``self._stage_names``
            so the autoscale call site can pass the dict to
            :func:`emit_bottleneck_score` directly.

        """
        samples: dict[str, float] = {}
        with self._lock:
            for stage_name in self._stage_names:
                now_count = self._completed_counts.get(stage_name, 0)
                now_sum = self._completed_service_time_sums.get(stage_name, 0.0)
                prev = self._last_service_time_sample.get(stage_name)
                if prev is None:
                    samples[stage_name] = math.nan
                else:
                    prev_count, prev_sum = prev
                    dcount = max(0, now_count - prev_count)
                    dsum = now_sum - prev_sum
                    if dcount > 0 and dsum > 0.0:
                        samples[stage_name] = dsum / dcount
                    else:
                        samples[stage_name] = math.nan
                self._last_service_time_sample[stage_name] = (now_count, now_sum)
        return samples

    def _update_s_k_ewma(self, service_times_s: Mapping[str, float]) -> None:
        """Apply one EWMA step to per-stage intrinsic ``S_k`` from this cycle's service times.

        The EWMA tracks the intrinsic per-task service time, NOT the
        actor-normalized ``D_k = S_k / c_k``. Smoothing only the
        per-task component prevents actor-count changes (Phase A
        grow / shrink, Phase D shrink) from leaking into the smooth
        signal: those transitions update the per-cycle ``c_k`` only,
        so the bottleneck path can react on the next cycle while the
        underlying service-time estimate stays stable.

        Args:
            service_times_s: Per-stage mean per-task service time
                produced by :meth:`_consume_service_time_samples`.
                ``math.nan`` means the stage produced no completed
                tasks this cycle.
        """
        alpha = self._config.bottleneck_d_k_smoothing_level
        for stage_name in self._stage_names:
            latest = service_times_s.get(stage_name, math.nan)
            prev = self._s_k_ewma.get(stage_name, math.nan)
            if not math.isfinite(latest) or latest <= 0.0:
                # Missed sample: preserve previous value (or seed NaN).
                if stage_name not in self._s_k_ewma:
                    self._s_k_ewma[stage_name] = math.nan
                continue
            if not math.isfinite(prev):
                # First-ever finite sample: replace the seed without
                # blending; otherwise alpha=0.2 would slow convergence
                # by ~5x for no benefit.
                self._s_k_ewma[stage_name] = latest
            else:
                self._s_k_ewma[stage_name] = prev * (1.0 - alpha) + latest * alpha

    @staticmethod
    def _effective_ready_capacity(runtime_stage: rust_data_structures.ProblemStageState) -> int:
        """Compute concurrent service channels at one stage.

        Effective ready capacity ``c_k`` is the number of concurrent
        service channels the stage has available right now: each ready
        worker contributes ``slots_per_worker`` channels, and an SPMD
        worker group with ``K`` allocations contributes
        ``slots_per_worker * K`` because all ``K`` allocations
        process tasks in parallel under one DDP unit. The plan
        documents the policy: count all ready workers, including
        those still inside ``worker_warmup_measurement_grace_s``,
        because ``D_k`` is an availability-adjusted demand signal
        and the classifier owns warmup-trust gating.

        Args:
            runtime_stage: Per-cycle runtime snapshot for one stage.

        Returns:
            Concurrent service channels available right now. Returns
            ``0`` for a stage with no ready worker groups; downstream
            :func:`compute_d_k` folds this into the cold-start
            sentinel ``math.nan``.
        """
        slots_per_worker = runtime_stage.slots_per_worker
        if slots_per_worker <= 0:
            return 0
        total_allocations = 0
        for worker_group in runtime_stage.worker_groups:
            allocation_count = max(1, len(worker_group.resources))
            total_allocations += allocation_count
        return slots_per_worker * total_allocations

    def autoscale(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Compute the autoscale plan for the current cycle.

        Wraps the per-cycle body with ``loop_watchdog`` so every cycle
        observes its wall-clock duration on
        ``xenna_scheduler_cycle_duration_seconds`` and a slow cycle
        emits a WARN log.

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
        with loop_watchdog(
            pipeline_name=self._pipeline_name,
            threshold_fraction=self._config.cycle_time_warn_threshold,
            interval_s=self._config.interval_s,
        ):
            return self._autoscale_body(time, problem_state)

    @contextmanager
    def _phase_timer(self, phase: str) -> Iterator[None]:
        """Bracket one phase of ``_autoscale_body`` with a duration timer.

        Records one observation on ``_PHASE_DURATION_HISTOGRAM`` tagged
        ``{"phase": phase, "pipeline": self._pipeline_name}``. The
        observation lives in ``finally`` so a phase that raises still
        records its duration.

        Args:
            phase: One of the 8 canonical phase labels pinned by
                ``test_every_phase_label_observed_once_per_cycle``.
        """
        start = _monotonic()
        try:
            yield
        finally:
            _PHASE_DURATION_HISTOGRAM.observe(
                _monotonic() - start,
                tags={"phase": phase, "pipeline": self._pipeline_name},
            )

    def _autoscale_body(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Run one autoscale cycle without the loop-watchdog wrap.

        ``autoscale()`` wraps this method with ``loop_watchdog`` so the
        per-cycle duration histogram and WARN log fire on every call,
        including paths that raise.
        """
        # Per-phase timing wrappers. Each phase block is bracketed with
        # ``_monotonic()`` samples in a ``try/finally`` so a phase that
        # raises (planner-context invariant violation, classifier NaN,
        # Phase D floor breach, stuck-plan monotonicity failure) still
        # records its duration on ``_PHASE_DURATION_HISTOGRAM`` before
        # the exception propagates. The 8 phase labels are pinned by
        # the test ``test_every_phase_label_observed_once_per_cycle``;
        # changing the label set requires updating both producer and
        # consumer dashboards (see docs/scheduler/saturation-aware/
        # 22-prometheus-metrics.md row
        # ``xenna_scheduler_cycle_phase_duration_seconds``).
        with self._phase_timer("pre_phase_setup"):
            if self._problem is None:
                msg = "SaturationAwareScheduler.autoscale() called before setup()"
                raise RuntimeError(msg)

            self._check_problem_state_shape_before_phase_a(problem_state)
            self._cycle_counter += 1
            # Refresh ready first-seen timestamps from this cycle's snapshot before
            # any phase reads them. The per-worker measurement grace consumes them
            # inside _compute_intent_deltas; Phase D shrink and the saturation-mode
            # cross-stage donor consult the resulting warmup-grace excluded set.
            self._refresh_worker_ready_first_seen(problem_state, now=time)
            # Snapshot the stuck-plan counters before Phase C mutates them so the
            # post-Phase-D monotonicity check can compare prev vs. curr without an
            # in-flight Phase C state. The same snapshot is also reused as the
            # caller-side filter: stages whose ``curr == prev`` were not touched
            # by ``_set_stuck_plan_counter`` and are excluded from the assertion.
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

        with self._phase_timer("phase_a"):
            self._run_phase_a_delete(ctx, problem_state)
            self._run_phase_a_grow(ctx, problem_state)
            check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_A, problem=self._problem, ctx=ctx)

        with self._phase_timer("phase_b"):
            self._run_phase_b_floor(ctx, problem_state)
            check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_B, problem=self._problem, ctx=ctx)

        # Bottleneck calculation block runs BEFORE the intent loop so the
        # per-stage decision pipeline observes a populated
        # ``cycle_bottleneck_context`` for the current cycle, and BEFORE
        # Phase C / Phase D so both phases see fresh
        # ``_last_bottleneck_meta`` / ``_s_k_ewma`` / ``_d_k_now``. The
        # block is wrapped in its own phase timer so the per-phase
        # Prometheus histogram accounts for its share of the cycle
        # duration; without the timer, ``loop_watchdog``'s wall-clock
        # measurement would diverge from the sum of per-phase histograms.
        # ``_d_k_now`` is recomputed each cycle from the current
        # intrinsic ``_s_k_ewma`` divided by the live effective
        # capacity, so all four downstream consumers (identity,
        # grow-priority, INFO log, heterogeneity gauge) see the same
        # actor-normalized D_k view.
        with self._phase_timer("bottleneck"):
            service_times_s = self._consume_service_time_samples()
            self._update_s_k_ewma(service_times_s)
            self._effective_capacities = {
                stage.stage_name: self._effective_ready_capacity(stage) for stage in problem_state.rust.stages
            }
            self._d_k_now = {
                name: compute_d_k(self._s_k_ewma.get(name, math.nan), self._effective_capacities.get(name, 0))
                for name in self._stage_names
            }
            self._last_bottleneck_meta = identify_bottleneck(
                self._d_k_now,
                heterogeneity_threshold=self._config.bottleneck_heterogeneity_threshold,
            )
            self._refresh_cycle_bottleneck_context()
            # Engagement log is silenced when both decision toggles are off:
            # disabling both decision paths must not introduce a new
            # operator log line. The EWMA state and meta snapshot are
            # still updated so re-enabling either toggle gets warm data
            # on the first cycle.
            if self._config.enable_bottleneck_priority_growth or self._config.enable_bottleneck_shrink_protection:
                maybe_log_bottleneck_engagement(
                    identity=self._last_bottleneck_meta,
                    state=self._bottleneck_engagement_state,
                    persistence_cycles=self._config.bottleneck_engagement_persistence_cycles,
                    pipeline_name=self._pipeline_name,
                )

        with self._phase_timer("intent"):
            # Sample throughput once per cycle BEFORE the intent loop so every
            # stage observes the same per-cycle delta even when a stage's
            # ``run_per_stage_pipeline`` raises (the lock is released first).
            throughput_samples = self._consume_throughput_samples(now_ts=time)
            self._last_intent_deltas = self._compute_intent_deltas(
                ctx,
                problem_state,
                now=time,
                throughput_samples=throughput_samples,
            )

        # Capture pre-Phase-C worker counts so we can compute the post-commit
        # executed delta per stage and feed it to the growth-mode state machine
        # via ``record_executed_delta`` after Phase D finishes. Recording the
        # post-commit delta (instead of the pre-Phase-C recommendation) means
        # hard caps, the fractional shrink clamp, and allocation failures all
        # reflect into the HOLD / ACQUIRING / TRACKING timers honestly.
        #
        # Keying by ``stage_name`` rather than the positional index keeps the
        # contract robust to any future planner change that reorders stages
        # between phases; ``stage_name`` is the canonical identifier already
        # used by ``_last_intent_deltas`` and ``_stage_states`` everywhere
        # else in the scheduler.
        pre_phase_c_worker_counts = self._worker_counts_by_stage_name(ctx, problem_state)

        # Memory-pressure gate semantics: when ``_run_phase_c_grow``
        # returns early because the cluster-wide pressure monitor is
        # active, the timer STILL records a (near-zero) duration. The
        # invariant + NaN checks below still run because they are
        # pass-through assertions when Phase C made no changes. This
        # pins the "always observe" guarantee for the per-phase
        # histogram contract.
        with self._phase_timer("phase_c"):
            self._run_phase_c_grow(ctx, problem_state)
            check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_C, problem=self._problem, ctx=ctx)
            check_no_nan_in_classifier_state(
                phase_name=PhaseBoundary.PHASE_C,
                stage_runtime_states=self._stage_states,
            )

        with self._phase_timer("phase_d"):
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

            self._record_post_commit_executed_deltas(
                ctx=ctx,
                problem_state=problem_state,
                pre_phase_c_worker_counts=pre_phase_c_worker_counts,
            )

        with self._phase_timer("invariants"):
            # Filter ``curr_counters`` to non-finished stages with an
            # observable transition this cycle:
            #   * ``active_stage_names`` excludes stages that finished mid-run.
            #   * ``curr != prev`` excludes stages Phase C did not touch
            #     (allocation-failure absorption, donor-retry synthetic
            #     absorption); since ``_set_stuck_plan_counter`` only ever
            #     writes ``0`` or ``prev + 1``, ``curr == prev > 0`` is
            #     unreachable through the funnel and ``curr == prev == 0``
            #     is a valid no-op reset that the strict rule would accept
            #     anyway, so excluding both is signal-preserving.
            # ``prev_counters`` is not filtered: the helper iterates
            # ``curr_counters`` only and looks up ``prev_counters`` by key,
            # so stale entries there are inert.
            active_stage_names = {stage.stage_name for stage in problem_state.rust.stages if not stage.is_finished}
            changed_counters = {
                name: curr
                for name, curr in self._stuck_plan_counters.items()
                if name in active_stage_names and curr != prev_stuck_plan_counters.get(name, 0)
            }
            check_stuck_plan_monotonicity(
                prev_counters=prev_stuck_plan_counters,
                curr_counters=changed_counters,
            )

        # Bottleneck score and heterogeneity emission live OUTSIDE the
        # per-phase wrappers above so the wrapper boundaries stay aligned
        # with the canonical phase labels in
        # docs/scheduler/saturation-aware/22-prometheus-metrics.md. Both
        # calls are sub-millisecond pure observability and consume the
        # actor-normalized ``self._d_k_now`` mapping computed inside the
        # bottleneck phase block; sharing the mapping with
        # ``identify_bottleneck`` and the grow-priority order guarantees
        # the operator-facing log and the heterogeneity gauge cannot
        # disagree with the planner decision in the same cycle.
        # Cold-start stages contribute ``math.nan`` so the helpers
        # preserve gauge cardinality and skip the ``argmax_k`` / log
        # line for any stage that has not produced a completed-task
        # sample yet.
        if self._last_bottleneck_meta is not None:
            emit_bottleneck_score(
                d_k_by_stage=self._d_k_now,
                bottleneck_identity=self._last_bottleneck_meta,
                pipeline_name=self._pipeline_name,
                effective_capacities=self._effective_capacities,
            )
        compute_heterogeneity_ratio(
            d_k_by_stage=self._d_k_now,
            pipeline_name=self._pipeline_name,
            state=self._heterogeneity_state,
            warn_threshold=self._config.cluster_heterogeneity_warn_threshold,
            warn_streak_cycles=self._config.cluster_heterogeneity_warn_streak,
        )

        with self._phase_timer("into_solution"):
            solution = ctx.into_solution()
            check_solution_shape(phase_name=PhaseBoundary.INTO_SOLUTION, problem=self._problem, solution=solution)
            self._persist_worker_ages(ctx)

        self._emit_cycle_summary()

        return solution

    def _emit_cycle_summary(self) -> None:
        """Emit one structured DEBUG summary line per cycle."""
        logger.debug(
            f"saturation-aware cycle {self._cycle_counter} summary: "
            f"regime={self._regime_state.current_regime.value}, "
            f"heterogeneity_streak={self._heterogeneity_state.streak_cycles}, "
            f"heterogeneity_fired={self._heterogeneity_state.has_fired}, "
            f"phase_c_allocation_failure={self._phase_c_allocation_failure}"
        )

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

    def _worker_counts_by_stage_name(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> dict[str, int]:
        """Return ``{stage_name: live_worker_count}`` from the planner snapshot.

        Pairs ``ctx.worker_ids_by_stage()`` (positional list ordered by
        ``Problem.stages``) with ``problem_state.rust.stages`` (the same
        ordering at construction time) to produce a name-keyed map. Used
        for pre/post snapshots so the executed-delta computation does
        not depend on stages staying in positional sync across phases.

        Args:
            ctx: Planner context whose live worker accessor we read.
            problem_state: Pipeline state whose ``stage_name`` field
                provides the canonical key.

        Returns:
            ``{stage_name: int}`` with one entry per stage in the
            current problem. Stages with no live workers map to ``0``.
        """
        worker_ids_by_index = ctx.worker_ids_by_stage()
        return {
            runtime_stage.stage_name: len(worker_ids)
            for runtime_stage, worker_ids in zip(problem_state.rust.stages, worker_ids_by_index, strict=True)
        }

    def _record_post_commit_executed_deltas(
        self,
        *,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
        pre_phase_c_worker_counts: dict[str, int],
    ) -> None:
        """Advance the growth-mode state machine using the post-commit delta.

        Called once per cycle, after Phase C (grow) and Phase D (shrink)
        commit their planner mutations. For every stage that
        participated in the intent computation, computes
        ``executed_delta = post_phase_d_count - pre_phase_c_count`` and
        calls :func:`record_executed_delta`. This pins the contract
        that the HOLD / ACQUIRING / TRACKING timers observe the
        post-commit delta, not the pre-commit recommendation, so a
        recommendation throttled by hard caps, the fractional shrink
        clamp, or an allocation failure cannot push the state machine
        toward ACQUIRING based on growth that never landed.

        Stages skipped by ``_compute_intent_deltas`` (finished or in
        cold-start quiescence) carry no entry in
        ``_last_intent_deltas`` and are intentionally not recorded
        here. Both ``pre_phase_c_worker_counts`` and the post-Phase-D
        snapshot are keyed by ``stage_name`` (the canonical identifier
        used by the rest of the scheduler), so the lookup is immune to
        any positional reordering that a future planner change might
        introduce.
        """
        post_counts = self._worker_counts_by_stage_name(ctx, problem_state)
        for stage_name in self._last_intent_deltas:
            stage_state = self._stage_states.get(stage_name)
            if stage_state is None:
                continue
            stage_cfg = self._stage_cfg(stage_name)
            pre = pre_phase_c_worker_counts.get(stage_name, 0)
            post = post_counts.get(stage_name, 0)
            record_executed_delta(
                stage_state=stage_state,
                delta_executed=post - pre,
                config=stage_cfg,
            )

    def _compute_intent_deltas(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
        *,
        now: float,
        throughput_samples: dict[str, float],
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
             scale-down) are preserved - they only act on ready
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
            throughput_samples: Per-stage tasks/sec from
                :meth:`_consume_throughput_samples`. Missing entries
                are treated as cold-start.

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
                #
                # Reset ``valid_signal_samples`` so any trust accumulated
                # before the quiescent gap is invalidated. Without this
                # reset, a post-quiescence cycle could fire a non-zero
                # recommendation off carry-forward EWMA the moment a
                # single ready actor emerges (the trust gate would still
                # see ``valid_signal_samples >= min_data_points`` from
                # the pre-gap streak). The reset mirrors the same
                # behaviour applied below in the trust-gate accounting
                # block for the all-warmup-gap case, keeping the
                # "consecutive valid samples" contract uniform across
                # both no-signal-gap paths (skip + in-pipeline).
                stage_state.valid_signal_samples = 0
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
            # Trust gate accounting: ``min_data_points`` defines how many
            # *strictly consecutive* warmup-excluded valid samples the
            # classifier output must accumulate before its non-zero
            # recommendations may drive Phase C/D. A sample is "valid"
            # if at least one ready actor contributed
            # (num_used_slots + num_empty_slots > 0).
            # Cycles where every ready actor was still within
            # ``worker_warmup_measurement_grace_s`` produce zeroed
            # signals; those RESET the counter to 0 (see ``else`` branch
            # below) so the gate is gated on *real, contiguous*
            # observations, not on actor existence. Phase B floor and
            # manual provisioning are evaluated outside this function,
            # so this gate cannot starve a zero-worker stage. The call
            # to ``run_per_stage_pipeline`` still runs even when the
            # gate is closed so the EWMA cache, classifier state, and
            # stabilization history keep tracking reality; only the
            # returned delta is clamped to zero while the gate is closed.
            cycle_has_fresh_sample = num_used_slots + num_empty_slots > 0
            if cycle_has_fresh_sample:
                stage_state.valid_signal_samples = min(stage_state.valid_signal_samples + 1, stage_cfg.min_data_points)
            else:
                # No-signal cycle: either every ready actor is still
                # inside ``worker_warmup_measurement_grace_s`` (the
                # warmup-excluding aggregator returned ``(0, 0)``) or
                # the stage has no ready actors at all (zero-worker
                # snapshot). Reset the consecutive-valid-sample
                # counter so the gate must rebuild from scratch
                # before the classifier can fire a non-zero
                # recommendation again. Without this reset, the gate
                # would reopen as soon as a single warmup-cleared
                # actor emerges, allowing the next cycle's
                # recommendation to be derived from the
                # ``last_valid_slots_empty_ratio_ewma`` carry-forward
                # of a worker mix that no longer reflects current
                # state.
                stage_state.valid_signal_samples = 0
            stage_state.capacity_target_workers = self._refresh_capacity_target_workers(
                stage_state=stage_state,
                stage_cfg=stage_cfg,
                input_queue_depth=runtime_stage.input_queue_depth,
                observed_throughput=throughput_samples.get(stage_name, 0.0),
                slots_per_worker=runtime_stage.slots_per_worker,
                stage_name=stage_name,
            )
            delta = run_per_stage_pipeline(
                stage_state=stage_state,
                num_used_slots=num_used_slots,
                num_empty_slots=num_empty_slots,
                input_queue_depth=runtime_stage.input_queue_depth,
                current_workers=current_workers,
                config=stage_cfg,
                recommendation_history=history,
                observed_throughput_sample=throughput_samples.get(stage_name, 0.0),
                pipeline_name=self._pipeline_name,
            )
            # Trust gate: a non-zero recommendation requires
            # ``valid_signal_samples >= min_data_points``. The
            # accounting block above resets ``valid_signal_samples``
            # to 0 on any no-signal cycle (all-warmup gap or
            # zero-worker snapshot), and the cold-start quiescent
            # skip branch above also resets, so any no-signal gap
            # forces the trust gate to rebuild from scratch before
            # the next non-zero recommendation may pass. This
            # intentionally penalises transient gaps - carry-forward
            # EWMA over warmup-churn windows was producing add/remove
            # decisions on stale worker mixes. The single freshness
            # leg is sufficient because the reset collapses the
            # "current cycle stale" case into the "counter below
            # threshold" case; the validator pins
            # ``min_data_points >= 1``, so ``valid_signal_samples = 0``
            # (post-reset) always trips this check.
            if delta != 0 and stage_state.valid_signal_samples < stage_cfg.min_data_points:
                logger.debug(
                    f"saturation-aware: stage {stage_name!r} valid samples "
                    f"{stage_state.valid_signal_samples}/{stage_cfg.min_data_points} - "
                    f"trust gate clamping recommendation {delta} to 0."
                )
                delta = 0
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

    def _refresh_capacity_target_workers(
        self,
        *,
        stage_state: _StageRuntimeState,
        stage_cfg: SaturationAwareStageConfig,
        input_queue_depth: int,
        observed_throughput: float,
        slots_per_worker: int,
        stage_name: str,
    ) -> int | None:
        """Return the capacity-sized worker target for one stage in this cycle.

        ``None`` if either ``D_k`` is unobservable (cold start) or
        ``resolved_thresholds`` is not yet populated; the caller stores
        the return value on ``stage_state`` and ``compute_delta`` uses
        ``None`` as the discrete-fallback sentinel.

        Args:
            stage_state: Per-stage runtime state, read for the
                resolved thresholds.
            stage_cfg: Per-stage saturation config.
            input_queue_depth: Tasks waiting upstream this cycle.
            observed_throughput: Per-cycle completed tasks/sec sample.
            slots_per_worker: Concurrent slots per worker for this stage.
            stage_name: Stage label used to look up the cached ``D_k``.

        Returns:
            Target worker count, or ``None`` when D_k is unobservable
            or thresholds have not resolved yet.
        """
        if stage_state.resolved_thresholds is None:
            return None
        d_k_seconds = self._s_k_ewma.get(stage_name, math.nan)
        return compute_capacity_target_workers(
            queue_depth=input_queue_depth,
            observed_throughput=observed_throughput,
            d_k_seconds=d_k_seconds,
            slots_per_worker=slots_per_worker,
            target_backlog_seconds=stage_cfg.target_backlog_seconds,
            utilization_target=derive_utilization_target(stage_state.resolved_thresholds),
        )

    def _refresh_cycle_bottleneck_context(self) -> None:
        """Overwrite every stage's ``cycle_bottleneck_context`` with the current cycle's identity.

        A disengaged ``_last_bottleneck_meta`` (or a bottleneck name
        no longer present in ``_stage_names``) is treated as no
        bottleneck: every stage receives the no-bottleneck default.
        """
        meta = self._last_bottleneck_meta
        if meta is None or not meta.engaged or meta.stage_name is None:
            for stage_state in self._stage_states.values():
                stage_state.cycle_bottleneck_context = BottleneckCycleContext()
            return
        try:
            bottleneck_index = self._stage_names.index(meta.stage_name)
        except ValueError:
            # Stale meta after a stage list change: fall through to the
            # no-bottleneck default so the per-stage context never
            # points at a missing stage.
            for stage_state in self._stage_states.values():
                stage_state.cycle_bottleneck_context = BottleneckCycleContext()
            return
        for stage_index, stage_name in enumerate(self._stage_names):
            looked_up_state = self._stage_states.get(stage_name)
            if looked_up_state is None:
                continue
            looked_up_state.cycle_bottleneck_context = BottleneckCycleContext(
                engaged=True,
                is_upstream_of_bottleneck=stage_index < bottleneck_index,
            )

    def _set_stuck_plan_counter(self, stage_name: str, value: int, *, last_intent: int) -> None:
        """Update ``_stuck_plan_counters[stage_name]`` and notify the detector."""
        self._stuck_plan_counters[stage_name] = value
        self._stuck_plan_detector.update(
            stage_name=stage_name,
            stuck_cycles=value,
            threshold_cycles=self._config.stuck_plan_detection_cycles,
            last_intent=last_intent,
            pipeline_name=self._pipeline_name,
        )

    def _try_add_worker_with_defense(
        self,
        ctx: data_structures.AutoscalePlanContext,
        stage_index: int,
        stage_name: str,
    ) -> data_structures.ProblemWorkerGroupState | None:
        """Call ``ctx.try_add_worker`` with the allocation-failure defense layer.

        Catches only ``AllocationError`` so scheduler bugs surfacing as
        ``SchedulerInvariantError``, ``KeyError``, ``IndexError``, etc.
        propagate to the autoscaler thread instead of being silently
        re-routed through the absorb path (which would mask the real
        defect).

        On absorbed allocation failures returns ``None`` and sets
        ``self._phase_c_allocation_failure`` so the caller aborts the
        Phase C loop. Re-raises when ``skip_cycle_on_allocation_error``
        is False.
        """
        try:
            return ctx.try_add_worker(stage_index)
        except resources.AllocationError as exc:
            self._absorb_allocation_failure(stage_name=stage_name, exc=exc)
            return None

    def _absorb_allocation_failure(self, *, stage_name: str, exc: BaseException) -> None:
        """Log the fragmentation snapshot, bump the counter, and raise or set the skip flag."""
        if self._problem is None:
            msg = "_absorb_allocation_failure called before setup()"
            raise RuntimeError(msg)
        emit_allocation_failure(
            stage_name=stage_name,
            pipeline_name=self._pipeline_name,
            cluster_resources=self._problem.rust.cluster_resources,
            exc=exc,
        )
        if not self._config.skip_cycle_on_allocation_error:
            raise exc
        self._phase_c_allocation_failure = True

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
        zero intent (NORMAL, OVER_PROVISIONED) is a no-op here;
        saturation-driven scale-down lands as Phase D.

        Iteration order is delegated to
        :func:`compute_grow_priority_order`. When the bottleneck gate
        is engaged for the cycle (``BottleneckIdentity.engaged`` AND
        ``config.enable_bottleneck_priority_growth``), stages sort by
        ``D_k`` descending with depth descending as the tiebreak.
        Otherwise ``config.enable_dag_priority_growth`` decides between
        depth-descending and problem order. Every stage with a positive
        intent is attempted regardless of any earlier capacity
        exhaustion.

        Updates :attr:`_stuck_plan_counters` (per-stage count of
        consecutive cycles where ``added < intent``; ``0`` when the
        stage met its intent or had none) and routes every mutation
        through :meth:`_set_stuck_plan_counter` so
        :class:`StuckPlanDetector` stays in lockstep. Bail paths
        (allocation-failure absorption, donor-retry synthetic
        absorption) intentionally do NOT call the funnel; the
        post-Phase-D :func:`check_stuck_plan_monotonicity` invariant
        excludes the resulting ``curr == prev`` no-op transitions
        from the strict +1/0 assertion at the caller side.

        Args:
            ctx: The cycle's mutable planner context. Mutated in place
                by ``try_add_worker``.
            problem_state: The cycle's runtime snapshot. Used to skip
                finished stages and to surface stage indices.

        """
        if self._problem is None:
            msg = "_run_phase_c_grow called before setup()"
            raise RuntimeError(msg)

        self._phase_c_allocation_failure = False

        # Cluster-wide memory-pressure kill switch. When the Ray object-store
        # ``used_fraction`` exceeds the configured threshold, every stage's
        # positive intent is frozen for the cycle so the scheduler stops
        # adding to a cluster already approaching OOM. The stuck-plan
        # counters are reset to 0 on the freeze path so the post-Phase-D
        # monotonicity check treats this as a ``no-attempt`` cycle (same
        # branch as the existing ``intent <= 0`` reset path); operators
        # still see the cluster-wide pressure via the
        # ``xenna_scheduler_memory_pressure_active`` gauge. Phase A (manual),
        # Phase B (floor), and Phase D (shrink) keep running because Phase B
        # is the only recovery path for a stage at 0 workers and Phase D
        # actively relieves pressure by shedding workers.
        if self._config.enable_memory_pressure_gate and self._memory_pressure_monitor.is_pressure_active(
            time.monotonic()
        ):
            for runtime_stage in problem_state.rust.stages:
                if not runtime_stage.is_finished:
                    self._set_stuck_plan_counter(runtime_stage.stage_name, 0, last_intent=0)
            return

        # Defensive: the bottleneck calculation block in ``autoscale()`` is
        # required to populate ``_last_bottleneck_meta`` before Phase C
        # runs. A future reorder that drops the calculation block would
        # leave Phase C reading stale prior-cycle data; raise a loud
        # ``RuntimeError`` instead of silently making the wrong decision.
        if self._last_bottleneck_meta is None:
            msg = "_last_bottleneck_meta is None; bottleneck calc block must run before phase C"
            raise RuntimeError(msg)
        bottleneck_meta = self._last_bottleneck_meta
        bottleneck_engaged = self._config.enable_bottleneck_priority_growth and bottleneck_meta.engaged
        stage_order = compute_grow_priority_order(
            self._problem,
            bottleneck_engaged=bottleneck_engaged,
            d_k_by_stage=self._d_k_now,
            enable_dag_priority=self._config.enable_dag_priority_growth,
        )

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
                self._set_stuck_plan_counter(stage_name, 0, last_intent=0)
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
                    self._set_stuck_plan_counter(stage_name, 0, last_intent=0)
                    continue
            added = 0
            while added < intent:
                if self._try_add_worker_with_defense(ctx, stage_index, stage_name) is not None:
                    added += 1
                    continue
                if self._phase_c_allocation_failure:
                    return
                # Cluster placement exhausted. Try the saturation-mode
                # cross-stage donor fallback before giving up. The donor
                # path runs a non-mutating probe before any removal, so
                # the post-donation ``try_add_worker`` is guaranteed to
                # succeed when the donor commit returns a stage name; a
                # failure there means the planner snapshot diverged
                # between probe and commit and surfaces as a
                # SchedulerInvariantError from
                # ``_attempt_cross_stage_donation`` rather than a silent
                # absorbed allocation failure. The anti-flap timestamp
                # is updated only on a successfully completed
                # donation+retry.
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
                if self._try_add_worker_with_defense(ctx, stage_index, stage_name) is None:
                    if self._phase_c_allocation_failure:
                        # Cluster mutated externally during the cycle so
                        # the post-probe commit can no longer place the
                        # receiver; honour the configured absorb path.
                        return
                    msg = (
                        f"saturation-mode donation: probe approved donor "
                        f"{donor_stage_name!r} for receiver {stage_name!r} but the "
                        f"post-removal try_add_worker returned None at intent "
                        f"{intent} after {added} adds; planner state diverged "
                        "between probe and commit."
                    )
                    raise SchedulerInvariantError(msg)
                self._record_donation_success(
                    donor_stage_name=donor_stage_name,
                    receiver_stage_name=stage_name,
                )
                added += 1
            if added < intent:
                next_count = self._stuck_plan_counters.get(stage_name, 0) + 1
                self._set_stuck_plan_counter(stage_name, next_count, last_intent=intent)
            else:
                self._set_stuck_plan_counter(stage_name, 0, last_intent=intent)

    def _attempt_cross_stage_donation(
        self,
        *,
        ctx: data_structures.AutoscalePlanContext,
        receiver_stage_index: int,
        receiver_stage_name: str,
    ) -> str | None:
        """Try to free a placement for a saturation-driven receiver.

        Selects an eligible donor via :func:`find_saturation_donor`
        (anti-flap + strict-upstream + master-toggle filters), proves
        the donor's release would unblock the receiver via
        ``ctx.probe_add_after_removals`` (no-op dry-run on a cloned
        cluster), then commits the removal atomically via
        ``ctx.remove_workers_atomically``. The
        ``_last_donation_cycle`` ledger is NOT updated here; the
        caller MUST advance it via :meth:`_record_donation_success`
        only after the immediate ``try_add_worker`` retry succeeds
        for the receiver, so a retry-failure path leaves the donor
        stage eligible to be revisited on the next planning cycle.

        Returns:
            The donor stage name when a donor was selected and
            removed (the caller now owns the receiver retry and the
            ledger bookkeeping). ``None`` when the master toggle is
            off, when no eligible donor exists, or when the planner
            cannot prove the receiver would fit after the removal.

        Raises:
            RuntimeError: The scheduler has not been set up.
            SchedulerInvariantError: ``remove_workers_atomically``
                returned ``False`` after pre-validation reported the
                donor as present, indicating mid-batch state
                divergence.

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
            excluded_worker_ids=self._donor_warmup_excluded_ids,
        )
        if donor is None:
            return None

        donor_stage_name = self._stage_names[donor.stage_index]
        removals = [(donor.stage_index, donor.worker_id)]
        probe = ctx.probe_add_after_removals(removals, receiver_stage_index)
        if not probe.feasible:
            logger.debug(
                f"[scheduler] saturation-mode donor probe rejected: "
                f"donor stage {donor_stage_name!r} worker {donor.worker_id!r} "
                f"-> receiver {receiver_stage_name!r} reject_reason="
                f"{probe.reject_reason!r}; donation skipped, receiver waits "
                "for next cycle."
            )
            return None

        if not ctx.remove_workers_atomically(removals):
            msg = (
                f"saturation-mode donation: pre-validated donor "
                f"({donor.stage_index}, {donor.worker_id!r}) for receiver "
                f"{receiver_stage_name!r} disappeared between probe and "
                "atomic-removal; planner snapshot diverged mid-cycle."
            )
            raise SchedulerInvariantError(msg)

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
        """Advance the donor-side anti-flap timestamp on retry success.

        Called by the receiver-side caller after the post-donation
        ``try_add_worker`` retry completes successfully. Keeping the
        update split from :meth:`_attempt_cross_stage_donation`
        ensures the anti-flap timestamp only reflects donations that
        placed a worker on the receiver; a retry-failure path leaves
        the donor stage eligible to be revisited on the next planning
        cycle.

        The ``receiver_stage_name`` argument is preserved for API
        symmetry with the multi-donor path (where every donor stage
        in the plan must be timestamped on success); it is unused by
        the single-donor wiring.
        """
        del receiver_stage_name
        self._last_donation_cycle[donor_stage_name] = self._cycle_counter

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

        Bottleneck shrink protection: when
        ``config.enable_bottleneck_shrink_protection`` is True and the
        cycle's ``BottleneckIdentity`` is engaged, the engaged bottleneck
        stage with negative intent and no ceiling overflow is skipped on
        the cycle. Ceiling overflow always bypasses the gate. See
        ``docs/scheduler/saturation-aware/25-bottleneck-decision-integration.md``.

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

        # Defensive: the bottleneck calculation block in ``autoscale()`` is
        # required to populate ``_last_bottleneck_meta`` before Phase D
        # runs. A future reorder that drops the calculation block would
        # leave Phase D reading stale prior-cycle data; raise a loud
        # ``RuntimeError`` instead of silently making the wrong decision.
        if self._last_bottleneck_meta is None:
            msg = "_last_bottleneck_meta is None; bottleneck calc block must run before phase D"
            raise RuntimeError(msg)
        bottleneck_meta = self._last_bottleneck_meta

        num_nodes = len(self._problem.rust.cluster_resources.nodes)
        stage_floors = self._compute_stage_floors(num_nodes)
        stage_ceilings = self._compute_stage_ceilings(num_nodes)
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        worker_ages = ctx.worker_ages()
        host_gpu_used_fractions = self._compute_host_gpu_used_fractions(problem_state)

        # Once-per-streak debounce ledger for the Phase D bottleneck-protection
        # INFO log. The previous-cycle snapshot tells us which stages were
        # already protected; a stage transitions into the set only on the
        # first cycle it enters protection (the log fires there). Any stage
        # that is no longer protected this cycle drops out of the new set,
        # so a future re-entry will re-arm a fresh INFO log.
        prev_protected_logged = self._bottleneck_protected_stages_logged
        currently_protected: set[str] = set()

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
                self._config.enable_bottleneck_shrink_protection
                and bottleneck_meta.engaged
                and stage_name == bottleneck_meta.stage_name
                and intent < 0
                and ceiling_excess == 0
            ):
                # Log only on the cycle the stage transitions into the
                # protection set so steady-state heterogeneous workloads
                # see one INFO line per protection event, not one per
                # cycle.
                if stage_name not in prev_protected_logged:
                    logger.info(
                        f"phase D bottleneck shrink protected: stage {stage_name!r} "
                        f"intent={intent} but D_k={bottleneck_meta.max_d_k:.2f}s is "
                        f"argmax (ratio={bottleneck_meta.heterogeneity_ratio:.2f}); "
                        "skipping shrink to preserve throughput across transient idle"
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

        # Replace the previous-cycle protection ledger with this cycle's
        # snapshot so a stage that drops out of protection on the next
        # cycle and later re-enters re-arms the once-per-streak INFO log.
        self._bottleneck_protected_stages_logged = currently_protected

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
                removals = [(donor.stage_index, donor.worker_id)]
                probe = ctx.probe_add_after_removals(removals, stage_index)
                if not probe.feasible:
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
                if not ctx.remove_workers_atomically(removals):
                    msg = (
                        f"Cross-stage floor donor: planner snapshot inconsistency - "
                        f"remove_workers_atomically(stage_index={donor.stage_index}, "
                        f"worker_id={donor.worker_id!r}) returned False even though the "
                        "worker was probe-validated as present. This is a scheduler "
                        "defect; report it with the autoscale cycle's problem_state."
                    )
                    raise RuntimeError(msg)
                if ctx.try_add_worker(stage_index) is None:
                    msg = (
                        f"Cross-stage floor donor: probe approved donor "
                        f"({donor.stage_index}, {donor.worker_id!r}) for receiver "
                        f"{problem_stage.name!r} but post-removal try_add_worker "
                        "returned None; planner state diverged between probe and commit."
                    )
                    raise SchedulerInvariantError(msg)
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
        classifier history, trust-gate counter
        (``valid_signal_samples``), and stabilization-window
        recommendation buffer so the next call to
        ``_ensure_thresholds_resolved`` re-derives thresholds with
        the appropriate effective aggressiveness and the
        post-transition cycles must rebuild both the trust gate and
        the stabilization consensus from scratch (pre-transition
        samples consumed a different threshold band and would
        otherwise let stale freshness or stale consensus leak into
        the new regime). Cycles whose signal is unavailable (some
        active stage has not populated ``num_used_slots`` /
        ``num_empty_slots`` yet) leave the regime state and resolved
        thresholds untouched. Respects the
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
            runtime.valid_signal_samples = 0
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
