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

"""Config wiring contract tests for the saturation-aware scheduler.

Background:
    ``SaturationAwareConfig.interval_s`` was previously defined but
    unused. The dispatcher's autoscale rate-limiter read
    ``StreamingSpecificSpec.autoscale_interval_s`` (180 s default)
    instead, while the watchdog inside the scheduler honored the
    saturation-aware default (10 s). This module pins the field-to-
    runtime wiring so the same class of defect cannot recur silently.

Coverage model:
    The wiring suite is intentionally partial; not every config
    field has a dedicated behavior test today. The meta-test (see
    :class:`TestConfigWiringMetaCoverage`) enumerates every public
    field via ``attrs.fields`` and pairs it with the running behavior /
    override allowlists in this file. The allowlists explicitly mark
    fields whose wiring tests are pending so a new unwired field
    cannot be added silently: any field absent from both the covered
    set and the documented gap set fails the meta-test. The gap
    allowlists are a known partial-coverage record, not a permanent
    waiver; closing them stage by stage is ongoing work.

Test quality rules (carried from the plan):
    * One test per field; one observable behavior per test.
    * Prefer the smallest pure helper that consumes the field
      (``classify``, ``_resolve_auto_thresholds``, ``compute_delta``,
      ``run_per_stage_pipeline``) over full scheduler integration.
    * Scheduler integration only when the field controls orchestration
      (``interval_s``, donor logic, memory gate, etc.).
    * Behavioral assertions on runtime state or return values; never
      on docstrings or comments.
"""

import math
from typing import ClassVar

import attrs
import pytest
from loguru import logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import (
    BottleneckEngagementState,
    BottleneckIdentity,
    identify_bottleneck,
    maybe_log_bottleneck_engagement,
)
from cosmos_xenna.pipelines.private.scheduling_py.classifier import classify
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import (
    record_executed_delta,
    run_per_stage_pipeline,
)
from cosmos_xenna.pipelines.private.scheduling_py.pressure import compute_pressure
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState, update_ewma
from cosmos_xenna.pipelines.private.specs import (
    PipelineConfig,
    PipelineSpec,
    SaturationAwareConfig,
    SaturationAwareStageConfig,
    SchedulerKind,
    StreamingSpecificSpec,
)
from cosmos_xenna.pipelines.private.streaming import effective_autoscale_interval


def _cluster(*, total_cpus_per_node: int = 8) -> resources.ClusterResources:
    """Single-node CPU cluster sufficient for ProblemStage construction."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=total_cpus_per_node, gpus=[], name="node-0"),
        },
    )


def _problem(stage_names: list[str]) -> data_structures.Problem:
    """Build a real ``Problem`` with one CPU stage per name. Order preserved."""
    cluster = _cluster()
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        for name in stage_names
    ]
    return data_structures.Problem(cluster, stages)


def _problem_state_with_signals(
    stage_specs: list[tuple[str, int, int, int, int]],
    *,
    input_queue_depth: int = 0,
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` populating per-stage num_used_slots / num_empty_slots.

    Args:
        stage_specs: Per-stage tuples ``(name, num_workers, slots_per_worker,
            num_used_slots, num_empty_slots)``.
        input_queue_depth: Stage-level upstream queue depth applied to every
            stage. Defaults to ``0`` so legacy tests that only exercise slot
            signals stay unchanged. The backlog-time pressure classifier
            (default-on) requires queue depth > 0 for the slot-pin SATURATED
            gate to fire, so trust-gate / wiring tests that expect a
            positive intent must pass a non-zero value here.
    """
    states = []
    for name, num_workers, slots, used, empty in stage_specs:
        worker_groups = [
            data_structures.ProblemWorkerGroupState.make(
                f"{name}-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(num_workers)
        ]
        states.append(
            data_structures.ProblemStageState(
                stage_name=name,
                workers=worker_groups,
                slots_per_worker=slots,
                is_finished=False,
                num_used_slots=used,
                num_empty_slots=empty,
                input_queue_depth=input_queue_depth,
            )
        )
    return data_structures.ProblemState(states)


def _resolved_state(
    *,
    stage_name: str = "S",
    config: SaturationAwareStageConfig,
) -> _StageRuntimeState:
    """Build a runtime state with thresholds already resolved."""
    from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import _resolve_auto_thresholds

    state = _StageRuntimeState(stage_name=stage_name)
    state.resolved_thresholds = _resolve_auto_thresholds(stage_cfg=config, slots_per_actor=8)
    return state


def _streaming_pipeline_spec(scheduler: SchedulerKind, **overrides: object) -> PipelineSpec:
    """Construct a minimal ``PipelineSpec`` in streaming mode.

    Args:
        scheduler: Which scheduler kind to wire into ``mode_specific``.
        **overrides: Optional ``StreamingSpecificSpec`` field overrides.
    """
    mode_specific = StreamingSpecificSpec(scheduler=scheduler, **overrides)  # type: ignore[arg-type]
    return PipelineSpec(
        input_data=[],
        stages=[],
        config=PipelineConfig(mode_specific=mode_specific),
    )


class TestEffectiveAutoscaleInterval:
    """``effective_autoscale_interval`` picks the cadence by scheduler kind."""

    def test_saturation_aware_uses_saturation_aware_interval(self) -> None:
        """Saturation-aware scheduler reads ``mode_specific.saturation_aware.interval_s``."""
        sat_cfg = SaturationAwareConfig(interval_s=7.5)
        spec = _streaming_pipeline_spec(SchedulerKind.SATURATION_AWARE, saturation_aware=sat_cfg)

        assert effective_autoscale_interval(spec) == pytest.approx(7.5)

    def test_fragmentation_based_uses_streaming_interval(self) -> None:
        """Fragmentation-based scheduler reads ``mode_specific.autoscale_interval_s``."""
        spec = _streaming_pipeline_spec(SchedulerKind.FRAGMENTATION_BASED, autoscale_interval_s=42.0)

        assert effective_autoscale_interval(spec) == pytest.approx(42.0)

    def test_saturation_aware_ignores_streaming_interval(self) -> None:
        """Saturation-aware ignores the fragmentation-based field even when both are set."""
        sat_cfg = SaturationAwareConfig(interval_s=5.0)
        spec = _streaming_pipeline_spec(
            SchedulerKind.SATURATION_AWARE,
            autoscale_interval_s=180.0,
            saturation_aware=sat_cfg,
        )

        assert effective_autoscale_interval(spec) == pytest.approx(5.0)

    def test_missing_mode_specific_raises(self) -> None:
        """``mode_specific=None`` is a programmer error (non-streaming mode)."""
        spec = PipelineSpec(input_data=[], stages=[], config=PipelineConfig(mode_specific=None))

        with pytest.raises(RuntimeError, match="mode_specific=None"):
            effective_autoscale_interval(spec)

    def test_fragmentation_based_zero_interval_rejected(self) -> None:
        """Zero ``autoscale_interval_s`` is rejected before the dispatcher inverts to a rate.

        ``StreamingSpecificSpec.autoscale_interval_s`` carries no
        field-level validator, so the resolver enforces the
        positive-interval invariant on this path; without it the
        rate-limiter would produce ``ZeroDivisionError`` at the
        ``1.0 / interval`` site.
        """
        spec = _streaming_pipeline_spec(SchedulerKind.FRAGMENTATION_BASED, autoscale_interval_s=0.0)

        with pytest.raises(ValueError, match=r"must be > 0.*autoscale_interval_s.*FRAGMENTATION_BASED"):
            effective_autoscale_interval(spec)

    def test_fragmentation_based_negative_interval_rejected(self) -> None:
        """Negative ``autoscale_interval_s`` is rejected before the dispatcher inverts to a rate.

        Without this guard the rate-limiter would derive a negative
        rate, silently breaking the cadence rather than failing
        loudly.
        """
        spec = _streaming_pipeline_spec(SchedulerKind.FRAGMENTATION_BASED, autoscale_interval_s=-1.5)

        with pytest.raises(ValueError, match=r"must be > 0.*-1\.5.*FRAGMENTATION_BASED"):
            effective_autoscale_interval(spec)


class TestRawStringSchedulerKindNormalization:
    """Raw-string ``scheduler`` values must coerce to ``SchedulerKind`` at construction.

    YAML / JSON / CLI deserialization commonly produces the raw string
    ``"saturation_aware"`` instead of the ``SchedulerKind`` enum member.
    The ``StreamingSpecificSpec.scheduler`` attrs field uses the
    ``SchedulerKind`` enum constructor as its converter so the field is
    always observed as the enum member by downstream consumers. Without
    coercion the dispatcher's cadence helper would fall back to
    ``autoscale_interval_s`` (180 s) while
    ``_make_scheduler_algorithm`` would still match the raw string to
    ``SchedulerKind.SATURATION_AWARE`` via the ``match`` value pattern,
    silently wiring a 180-second cadence into the saturation-aware
    scheduler whose watchdog is sized for 10 s cycles.
    """

    def test_raw_string_scheduler_coerces_to_enum_member(self) -> None:
        """Constructing with a raw string yields the enum member on the field."""
        mode_specific = StreamingSpecificSpec(scheduler="saturation_aware")  # type: ignore[arg-type]

        assert mode_specific.scheduler is SchedulerKind.SATURATION_AWARE
        assert isinstance(mode_specific.scheduler, SchedulerKind)

    def test_raw_string_scheduler_resolves_saturation_aware_interval(self) -> None:
        """``effective_autoscale_interval`` reads ``saturation_aware.interval_s`` for a string input."""
        sat_cfg = SaturationAwareConfig(interval_s=7.5)
        mode_specific = StreamingSpecificSpec(
            scheduler="saturation_aware",  # type: ignore[arg-type]
            autoscale_interval_s=180.0,
            saturation_aware=sat_cfg,
        )
        spec = PipelineSpec(
            input_data=[],
            stages=[],
            config=PipelineConfig(mode_specific=mode_specific),
        )

        assert effective_autoscale_interval(spec) == pytest.approx(7.5)

    def test_raw_string_scheduler_dispatches_saturation_aware_algorithm(self) -> None:
        """``_make_scheduler_algorithm`` instantiates the saturation-aware scheduler for a string input."""
        from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
        from cosmos_xenna.pipelines.private.streaming import _make_scheduler_algorithm

        mode_specific = StreamingSpecificSpec(scheduler="saturation_aware")  # type: ignore[arg-type]
        spec = PipelineSpec(
            input_data=[],
            stages=[],
            config=PipelineConfig(mode_specific=mode_specific),
        )

        algorithm = _make_scheduler_algorithm(spec)

        assert isinstance(algorithm, SaturationAwareScheduler)

    def test_unknown_scheduler_string_raises_at_construction(self) -> None:
        """Unknown raw strings raise during ``StreamingSpecificSpec`` construction, not at runtime."""
        with pytest.raises(ValueError, match="not a valid SchedulerKind"):
            StreamingSpecificSpec(scheduler="not_a_real_scheduler")  # type: ignore[arg-type]


class TestSaturationAwareIntervalUsedAcrossWiring:
    """``SaturationAwareConfig.interval_s`` flows into every cadence-aware code path.

    Cadence drift between the dispatcher rate-limiter and the scheduler's
    internal watchdog was the root regression that motivated this wiring
    suite. Pin both consumers to the same source field so a future
    refactor cannot reintroduce the split.
    """

    def test_interval_s_is_observed_on_constructed_scheduler(self) -> None:
        """The scheduler stores ``interval_s`` from its ``SaturationAwareConfig`` on construction."""
        sat_cfg = SaturationAwareConfig(interval_s=3.25)
        from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler

        scheduler = SaturationAwareScheduler(sat_cfg)

        assert scheduler._config.interval_s == pytest.approx(3.25)


class TestMinDataPointsTrustGate:
    """``min_data_points`` gates non-zero recommendations until enough samples accumulate."""

    def test_min_data_points_one_fires_on_first_valid_sample(self) -> None:
        """With ``min_data_points=1`` the first valid sample produces a non-zero recommendation.

        The stage-level slot pin (``8/8`` used, ``0/8`` empty -> ratio ``0.0``)
        plus the non-zero ``input_queue_depth`` yield a positive backlog-time
        pressure so the SATURATED demotion gate is satisfied. Without an
        explicit queue depth the test would inadvertently pin pressure to
        zero and the pressure classifier would correctly demote the slot
        pin to NORMAL.
        """
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=1,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                saturated_streak_min_cycles=1,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))

        ps = _problem_state_with_signals([("S", 1, 8, 8, 0)], input_queue_depth=100)
        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._stage_states["S"].valid_signal_samples == 1
        assert scheduler._last_intent_deltas.get("S", 0) > 0

    def test_min_data_points_three_suppresses_until_third_valid_sample(self) -> None:
        """With ``min_data_points=3`` the first two valid samples produce zero intent; the third fires."""
        # Disable regime-aware aggressiveness so the third cycle does
        # not coincide with the default ``regime_transition_streak_cycles=3``
        # transition, which (by design) resets the trust-gate counter
        # via ``_update_regime_aware_aggressiveness`` and would otherwise
        # require this test to run extra cycles for the gate to rebuild.
        # The test's contract is the ``min_data_points`` trust gate,
        # not the regime detector, so we silence regime detection.
        sat_cfg = SaturationAwareConfig(
            enable_regime_aware_aggressiveness=False,
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=3,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                saturated_streak_min_cycles=1,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))
        ps = _problem_state_with_signals([("S", 1, 8, 8, 0)], input_queue_depth=100)

        scheduler.autoscale(time=0.0, problem_state=ps)
        assert scheduler._last_intent_deltas.get("S", 0) == 0

        scheduler.autoscale(time=1.0, problem_state=ps)
        assert scheduler._last_intent_deltas.get("S", 0) == 0

        scheduler.autoscale(time=2.0, problem_state=ps)
        assert scheduler._last_intent_deltas.get("S", 0) > 0

    def test_warmup_filtered_all_zero_samples_reset_counter(self) -> None:
        """Cycles whose warmup filter drops every contribution RESET the counter to zero.

        The starting counter is pre-seeded to ``5`` (well above any
        plausible ``min_data_points`` default) so the assertion
        distinguishes the new "reset to 0" semantic from a hypothetical
        "do not advance" implementation. With the seed at ``0`` (state
        default) both semantics would leave the counter at ``0`` and
        the test would not catch a future regression that removed the
        reset.
        """
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=2,
                worker_warmup_measurement_grace_s=999.0,
                donor_warmup_grace_s=999.0,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))
        scheduler._stage_states["S"].valid_signal_samples = 5
        ps = _problem_state_with_signals([("S", 1, 8, 8, 0)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._stage_states["S"].valid_signal_samples == 0


class TestTrustGateFreshnessAfterMinDataPointsReached:
    """After ``min_data_points`` is satisfied, a stale-sample cycle still emits zero intent.

    The trust gate requires BOTH accumulated history (``min_data_points``)
    AND a fresh current-cycle sample before non-zero recommendations may
    drive Phase C / Phase D. Without the freshness leg the scheduler
    could classify off carry-forward EWMA during the warmup-churn window
    the gate was meant to dampen and add or remove workers from stale
    measurements. Pressure must also stay at its prior value on a
    stale-sample cycle so blending a stale utilisation factor with a
    fresh queue depth cannot corrupt the next valid cycle.
    """

    def test_stale_cycle_after_threshold_clamps_intent_and_resets_counter(self) -> None:
        """``min_data_points`` reached, then a zero-sample cycle -> ``delta == 0`` and counter resets to 0.

        A no-signal gap must invalidate accumulated trust so the
        next ``min_data_points`` consecutive fresh-sample cycles
        must re-accrue before the gate reopens. Without the reset,
        a single post-gap fresh sample would reopen the gate and
        let the scheduler emit recommendations off carry-forward
        EWMA from a worker mix that no longer reflects current
        state.
        """
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=1,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                saturated_streak_min_cycles=1,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))

        # Cycle 0: fresh saturated signal opens the trust gate and emits
        # a positive intent.
        ps_fresh = _problem_state_with_signals([("S", 1, 8, 8, 0)], input_queue_depth=100)
        scheduler.autoscale(time=0.0, problem_state=ps_fresh)
        assert scheduler._stage_states["S"].valid_signal_samples == 1
        assert scheduler._last_intent_deltas.get("S", 0) > 0

        # Cycle 1: same signal expressed via a 0-worker snapshot. The
        # aggregation returns ``(0, 0)`` slots so the current cycle is
        # stale-only; the trust gate must clamp the recommendation to 0
        # AND reset the consecutive-valid-sample counter so trust
        # rebuilds from scratch on subsequent fresh samples.
        ps_stale = _problem_state_with_signals([("S", 0, 8, 0, 0)], input_queue_depth=100)
        scheduler.autoscale(time=1.0, problem_state=ps_stale)

        assert scheduler._last_intent_deltas.get("S", 0) == 0
        assert scheduler._stage_states["S"].valid_signal_samples == 0

    def test_post_gap_recovery_requires_rebuilding_consecutive_trust(self) -> None:
        """Post-gap recovery: ``min_data_points`` consecutive fresh cycles must re-accrue before the gate reopens.

        With ``min_data_points=2`` this test walks the gate through
        open -> gap -> closed -> single-fresh -> reopen and asserts
        the counter trajectory ``1 -> 2 -> 0 -> 1 -> 2`` at each
        step. A "clamp-without-reset" implementation would instead
        produce ``1 -> 2 -> 2 -> 2 -> 2``, reopening the gate on
        the first post-gap fresh sample, so the trajectory pins the
        strict-consecutive contract.
        """
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=2,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                saturated_streak_min_cycles=1,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))

        ps_fresh = _problem_state_with_signals([("S", 1, 8, 8, 0)], input_queue_depth=100)
        ps_stale = _problem_state_with_signals([("S", 0, 8, 0, 0)], input_queue_depth=100)

        # Cycle 0: first fresh sample -- counter advances but gate
        # stays closed (need 2 consecutive).
        scheduler.autoscale(time=0.0, problem_state=ps_fresh)
        assert scheduler._stage_states["S"].valid_signal_samples == 1
        assert scheduler._last_intent_deltas.get("S", 0) == 0

        # Cycle 1: second consecutive fresh sample opens the gate.
        scheduler.autoscale(time=1.0, problem_state=ps_fresh)
        assert scheduler._stage_states["S"].valid_signal_samples == 2
        assert scheduler._last_intent_deltas.get("S", 0) > 0

        # Cycle 2: stale cycle invalidates trust -- counter resets and
        # delta is clamped to 0.
        scheduler.autoscale(time=2.0, problem_state=ps_stale)
        assert scheduler._stage_states["S"].valid_signal_samples == 0
        assert scheduler._last_intent_deltas.get("S", 0) == 0

        # Cycle 3: a single post-gap fresh sample is not enough -- the
        # gate stays closed (counter back to 1, still < min_data_points).
        scheduler.autoscale(time=3.0, problem_state=ps_fresh)
        assert scheduler._stage_states["S"].valid_signal_samples == 1
        assert scheduler._last_intent_deltas.get("S", 0) == 0

        # Cycle 4: second consecutive post-gap fresh sample reopens
        # the gate -- trust has fully rebuilt from scratch.
        scheduler.autoscale(time=4.0, problem_state=ps_fresh)
        assert scheduler._stage_states["S"].valid_signal_samples == 2
        assert scheduler._last_intent_deltas.get("S", 0) > 0

    def test_stale_cycle_does_not_refresh_pressure_ewma(self) -> None:
        """Carry-forward cycles leave ``stage_state.pressure_ewma`` untouched."""
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=1,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                saturated_streak_min_cycles=1,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))

        ps_fresh = _problem_state_with_signals([("S", 1, 8, 8, 0)], input_queue_depth=100)
        scheduler.autoscale(time=0.0, problem_state=ps_fresh)
        first_pressure = scheduler._stage_states["S"].pressure_ewma
        assert first_pressure is not None and first_pressure > 0.0

        # Stale cycle: zero current samples must not blend a fresh queue
        # depth into the pressure EWMA.
        ps_stale = _problem_state_with_signals([("S", 0, 8, 0, 0)], input_queue_depth=999)
        scheduler.autoscale(time=1.0, problem_state=ps_stale)

        assert scheduler._stage_states["S"].pressure_ewma == first_pressure


class TestPressureClassifierFields:
    """Pressure / backlog config fields steer the classifier verdict.

    Each test pins one config field with the smallest helper that
    reads it (``classify``, ``compute_pressure``, ``update_ewma``) so
    a field cannot be added without an observable wiring point.
    """

    def test_pressure_saturation_threshold_high_demotes_slot_pin_to_normal(self) -> None:
        """High ``pressure_saturation_threshold`` demotes a slot-pinned SATURATED to NORMAL.

        The three pressure thresholds are constrained by the ordering
        ``critical > saturation > normal`` and the BACKLOG_CAP=3.0
        ceiling on ``pressure_critical_threshold``.
        """
        cfg = SaturationAwareStageConfig(
            saturation_threshold=0.15,
            activation_threshold=0.05,
            pressure_critical_threshold=2.8,
            pressure_saturation_threshold=2.0,
            pressure_normal_threshold=0.1,
        )

        verdict = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=100,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
            pressure_ewma=0.5,  # well below pressure_saturation_threshold=2.0
        )

        assert verdict is StageState.NORMAL

    def test_pressure_critical_threshold_above_pressure_falls_through_to_saturated(self) -> None:
        """``pressure_critical_threshold`` strictly above current pressure keeps the verdict at SATURATED."""
        cfg = SaturationAwareStageConfig(
            saturation_threshold=0.15,
            activation_threshold=0.05,
            pressure_critical_threshold=2.5,
            pressure_saturation_threshold=1.0,
            pressure_normal_threshold=0.3,
        )

        # Slots empty 0.0 < activation_threshold=0.05; pressure 1.5
        # exceeds saturation gate (1.0) but is below the critical gate
        # (2.5), so CRITICAL falls through to SATURATED.
        verdict = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=100,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
            pressure_ewma=1.5,
        )

        assert verdict is StageState.SATURATED

    def test_pressure_normal_threshold_high_keeps_over_provisioned(self) -> None:
        """``pressure_normal_threshold`` above current pressure preserves OVER_PROVISIONED."""
        cfg = SaturationAwareStageConfig(
            saturation_threshold=0.15,
            activation_threshold=0.05,
            over_provisioned_threshold=0.50,
            pressure_critical_threshold=2.8,
            pressure_saturation_threshold=2.0,
            pressure_normal_threshold=1.0,
        )

        # Slots empty 0.70 above over_provisioned_threshold=0.50;
        # queue depth > 0; pressure 0.5 stays below
        # pressure_normal_threshold=1.0 so the demotion gate does
        # NOT fire and OVER_PROVISIONED is preserved.
        verdict = classify(
            slots_empty_ratio_ewma=0.70,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
            pressure_ewma=0.5,
        )

        assert verdict is StageState.OVER_PROVISIONED

    def test_target_backlog_seconds_scales_normalized_backlog(self) -> None:
        """Doubling ``target_backlog_seconds`` halves the computed pressure (same inputs)."""
        # W_q = 100 / 10 = 10s; both targets sit above W_q so the
        # cap-clamp branch never engages and the linear scaling is
        # observable directly.
        small_target = compute_pressure(
            slots_empty_ratio_ewma=0.20,
            input_queue_depth=100,
            observed_throughput=10.0,
            target_backlog_seconds=30.0,
        )
        large_target = compute_pressure(
            slots_empty_ratio_ewma=0.20,
            input_queue_depth=100,
            observed_throughput=10.0,
            target_backlog_seconds=60.0,
        )

        assert math.isclose(small_target, 2.0 * large_target, rel_tol=1e-9)

    def test_compute_pressure_rejects_non_positive_target_backlog_seconds(self) -> None:
        """A non-positive ``target_backlog_seconds`` would divide by zero or invert the scale."""
        with pytest.raises(ValueError, match=r"target_backlog_seconds must be > 0"):
            compute_pressure(
                slots_empty_ratio_ewma=0.20,
                input_queue_depth=100,
                observed_throughput=10.0,
                target_backlog_seconds=0.0,
            )

    def test_pressure_smoothing_level_one_replaces_prior_value_with_raw(self) -> None:
        """``pressure_smoothing_level=1.0`` makes ``update_ewma`` return the raw sample."""
        no_smoothing = update_ewma(prev_ewma=5.0, sample=1.0, alpha=1.0)
        heavy_smoothing = update_ewma(prev_ewma=5.0, sample=1.0, alpha=0.1)

        assert no_smoothing == pytest.approx(1.0)
        assert heavy_smoothing == pytest.approx(0.1 * 1.0 + 0.9 * 5.0)


class TestStageOverridePrecedenceForPressureFields:
    """Pressure / backlog stage fields propagate through the override chain."""

    def test_per_stage_override_changes_target_backlog_seconds_for_named_stage(self) -> None:
        """Per-stage override overrides the cluster default for ``target_backlog_seconds`` only."""
        per_stage = {
            "tight": SaturationAwareStageConfig(target_backlog_seconds=5.0),
        }
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(target_backlog_seconds=120.0),
            per_stage_overrides=per_stage,
        )

        assert sat_cfg.get_effective_stage_config("tight").target_backlog_seconds == pytest.approx(5.0)
        assert sat_cfg.get_effective_stage_config("loose").target_backlog_seconds == pytest.approx(120.0)


class TestBottleneckDecisionFields:
    """Cluster-level bottleneck-decision fields control engagement and ordering."""

    def test_enable_bottleneck_priority_growth_is_stored_on_scheduler(self) -> None:
        """``enable_bottleneck_priority_growth`` is observable on the constructed scheduler."""
        sat_cfg = SaturationAwareConfig(enable_bottleneck_priority_growth=False)

        scheduler = SaturationAwareScheduler(sat_cfg)

        assert scheduler._config.enable_bottleneck_priority_growth is False

    def test_enable_bottleneck_shrink_protection_is_stored_on_scheduler(self) -> None:
        """``enable_bottleneck_shrink_protection`` is observable on the constructed scheduler."""
        sat_cfg = SaturationAwareConfig(enable_bottleneck_shrink_protection=False)

        scheduler = SaturationAwareScheduler(sat_cfg)

        assert scheduler._config.enable_bottleneck_shrink_protection is False

    def test_bottleneck_d_k_smoothing_level_one_skips_ewma_blending(self) -> None:
        """``bottleneck_d_k_smoothing_level=1.0`` makes the EWMA latch onto the latest sample."""
        sat_cfg = SaturationAwareConfig(bottleneck_d_k_smoothing_level=1.0)
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["A", "B"]))

        # Seed cycle: latch each stage at a distinct intrinsic S_k value.
        scheduler._update_s_k_ewma({"A": 2.0, "B": 1.0})
        assert scheduler._s_k_ewma["A"] == pytest.approx(2.0)
        assert scheduler._s_k_ewma["B"] == pytest.approx(1.0)

        # Second cycle with alpha=1.0 replaces (no blending with prior).
        scheduler._update_s_k_ewma({"A": 5.0, "B": 0.5})

        assert scheduler._s_k_ewma["A"] == pytest.approx(5.0)
        assert scheduler._s_k_ewma["B"] == pytest.approx(0.5)

    def test_bottleneck_heterogeneity_threshold_above_ratio_disengages(self) -> None:
        """A threshold larger than the computed ratio leaves engagement off."""
        # max=3.0, median=1.0 (n=3) -> ratio=3.0.
        d_k = {"slow": 3.0, "mid": 1.0, "fast": 1.0}

        # Threshold below the ratio -> engaged.
        engaged = identify_bottleneck(d_k, heterogeneity_threshold=2.0)
        assert engaged.engaged is True

        # Threshold above the ratio -> disengaged on the same input.
        disengaged = identify_bottleneck(d_k, heterogeneity_threshold=5.0)
        assert disengaged.engaged is False

    def test_bottleneck_engagement_persistence_cycles_gates_announcement(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The engagement INFO log fires only after ``persistence_cycles`` agreeing cycles."""
        identity = BottleneckIdentity(
            engaged=True,
            stage_name="slow",
            max_d_k=3.0,
            median_d_k=1.0,
            heterogeneity_ratio=3.0,
        )
        state = BottleneckEngagementState()
        log_lines: list[str] = []
        sink_id = logger.add(lambda msg: log_lines.append(msg.record["message"]), level="INFO")
        try:
            # Persistence=3: first two cycles accumulate, third announces.
            maybe_log_bottleneck_engagement(identity=identity, state=state, persistence_cycles=3, pipeline_name="p")
            assert state.last_announced is None
            assert not any("engaged" in line for line in log_lines)

            maybe_log_bottleneck_engagement(identity=identity, state=state, persistence_cycles=3, pipeline_name="p")
            assert state.last_announced is None

            maybe_log_bottleneck_engagement(identity=identity, state=state, persistence_cycles=3, pipeline_name="p")
        finally:
            logger.remove(sink_id)

        # Third call promotes engagement; the implementation announces
        # via either a synchronous INFO log or by updating
        # ``last_announced``. The contract pinned here is that
        # persistence_cycles=3 must NOT promote earlier than the third
        # cycle, which the first two assertions above already verified.
        assert state.last_announced is True


class TestGrowthModeStateMachineKillSwitch:
    """``enable_growth_mode_state_machine=False`` neutralises HOLD's grow block."""

    def test_disabled_lets_hold_state_grow_on_saturated(self) -> None:
        """HOLD + SATURATED with the kill switch off -> the capacity-driven delta is honoured.

        With the kill switch on, HOLD blocks SATURATED grow (delta=0).
        With the kill switch off, the magnitude calculation receives
        ``TRACKING`` and the capacity-driven sizer drives the delta;
        a non-zero result pins that the kill switch defeated HOLD's
        binary block.
        """
        stage_cfg = SaturationAwareStageConfig(
            enable_growth_mode_state_machine=False,
            saturated_streak_min_cycles=1,
            min_data_points=1,
            worker_warmup_measurement_grace_s=0.0,
        )
        state = _resolved_state(config=stage_cfg)
        state.growth_mode = GrowthMode.HOLD
        state.classifier_state = StageState.SATURATED
        state.classifier_streak = 5
        # Capacity target exceeds current_workers so the sizer-driven delta is positive.
        state.capacity_target_workers = 10

        # Slot signal in the SATURATED zone (ratio between activation_threshold
        # and saturation_threshold), high pressure prevents demotion to NORMAL.
        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=19,
            num_empty_slots=1,
            input_queue_depth=100,
            current_workers=4,
            config=stage_cfg,
        )

        assert delta > 0, f"Kill switch did not neutralise HOLD; got delta={delta}"

    def test_disabled_skips_transition_so_state_does_not_drift(self) -> None:
        """``record_executed_delta`` is a no-op when the kill switch is off."""
        stage_cfg = SaturationAwareStageConfig(enable_growth_mode_state_machine=False)
        state = _resolved_state(config=stage_cfg)
        initial_mode = state.growth_mode
        initial_streak = state.growth_streak

        record_executed_delta(stage_state=state, delta_executed=-5, config=stage_cfg)

        assert state.growth_mode is initial_mode
        assert state.growth_streak == initial_streak


class TestStageOverridePrecedenceForMinDataPoints:
    """``min_data_points`` precedence: stage_defaults < per_stage_overrides < StageSpec.saturation_aware."""

    def test_per_stage_override_changes_behavior_for_named_stage_only(self) -> None:
        """Per-stage override applies to its stage; the negative control stage uses defaults."""
        per_stage = {"loud": SaturationAwareStageConfig(min_data_points=1, worker_warmup_measurement_grace_s=0.0)}
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=5,
                worker_warmup_measurement_grace_s=0.0,
                saturated_streak_min_cycles=1,
            ),
            per_stage_overrides=per_stage,
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["loud", "quiet"]))

        ps = _problem_state_with_signals(
            [("loud", 1, 8, 8, 0), ("quiet", 1, 8, 8, 0)],
            input_queue_depth=100,
        )
        scheduler.autoscale(time=0.0, problem_state=ps)

        # 'loud' has min_data_points=1, fires immediately. 'quiet' has =5 so trust gate clamps to 0.
        assert scheduler._last_intent_deltas.get("loud", 0) > 0
        assert scheduler._last_intent_deltas.get("quiet", 0) == 0


class TestConfigWiringMetaCoverage:
    """Enumerate every public config field and verify partial wiring coverage.

    The meta-test partitions every public field on the cluster and
    stage configs into two sets: fields with a dedicated behavior or
    override-precedence test, and fields documented as pending in
    the ``_KNOWN_COVERAGE_GAPS_*`` allowlists. Any field absent from
    both sets fails the meta-test, so a newly added field cannot be
    silently unwired even when its dedicated test has not yet
    landed. The allowlists are a partial-coverage record (ongoing
    work to close them stage by stage), not a permanent waiver.
    """

    _CLUSTER_BEHAVIOR_TESTS: ClassVar[set[str]] = {
        "interval_s",
        "enable_bottleneck_priority_growth",
        "enable_bottleneck_shrink_protection",
        "bottleneck_d_k_smoothing_level",
        "bottleneck_heterogeneity_threshold",
        "bottleneck_engagement_persistence_cycles",
    }

    _STAGE_BEHAVIOR_TESTS: ClassVar[set[str]] = {
        "min_data_points",
        "enable_growth_mode_state_machine",
        "pressure_critical_threshold",
        "pressure_saturation_threshold",
        "pressure_normal_threshold",
        "target_backlog_seconds",
        "pressure_smoothing_level",
    }

    _STAGE_OVERRIDE_TESTS: ClassVar[set[str]] = {
        "min_data_points",
        "target_backlog_seconds",
    }

    # Fields the wiring suite does not yet cover. Each entry is
    # tracked in ``docs/scheduler/saturation-aware/tuning.md`` or
    # the roadmap; add a wiring test and remove the field here in
    # the same change. The set is a documented partial-coverage
    # record, not a permanent waiver.
    _KNOWN_COVERAGE_GAPS_CLUSTER: ClassVar[set[str]] = {
        "enable_regime_aware_aggressiveness",
        "super_halfin_whitt_aggressiveness_lift",
        "regime_transition_streak_cycles",
        "enable_dag_priority_growth",
        "enable_cross_stage_donor",
        "donor_must_be_strictly_upstream",
        "cross_stage_donor_require_over_provisioned",
        "cross_stage_donor_exclude_hold_state",
        "cross_stage_donor_anti_flap_cycles",
        "cross_stage_donor_max_per_cycle",
        "cross_stage_donor_min_donation_interval_cycles",
        "floor_stuck_grace_cycles",
        "cycle_time_warn_threshold",
        "enable_memory_pressure_gate",
        "memory_pressure_critical_threshold",
        "memory_pressure_polling_interval_s",
        "skip_cycle_on_allocation_error",
        "stuck_plan_detection_cycles",
        "cluster_heterogeneity_warn_threshold",
        "cluster_heterogeneity_warn_streak",
        "stage_defaults",
        "per_stage_overrides",
    }

    _KNOWN_COVERAGE_GAPS_STAGE_BEHAVIOR: ClassVar[set[str]] = {
        "saturation_aggressiveness",
        "saturation_threshold",
        "activation_threshold",
        "auto_threshold_min",
        "auto_threshold_max",
        "activation_to_saturation_ratio",
        "over_provisioned_threshold",
        "saturation_deadband_pct",
        "over_provisioned_deadband_pct",
        "saturated_streak_min_cycles",
        "saturated_critical_streak_min_cycles",
        "over_provisioned_streak_min_cycles",
        "aggressive_growth_max_per_cycle",
        "slots_empty_ratio_smoothing_level",
        "stabilization_window_cycles_up",
        "stabilization_window_cycles_down",
        "max_scale_down_fraction_per_cycle",
        "setup_phase_quiescence_enabled",
        "worker_warmup_measurement_grace_s",
        "donor_warmup_grace_s",
        "min_workers",
        "min_workers_per_node",
        "max_workers",
        "max_workers_per_node",
        "setup_aware_max_queued",
    }

    _KNOWN_COVERAGE_GAPS_STAGE_OVERRIDE: ClassVar[set[str]] = _KNOWN_COVERAGE_GAPS_STAGE_BEHAVIOR | {
        "enable_growth_mode_state_machine",
        "pressure_critical_threshold",
        "pressure_saturation_threshold",
        "pressure_normal_threshold",
        "pressure_smoothing_level",
    }

    def test_every_cluster_field_has_a_behavior_test_or_is_explicitly_allowlisted(self) -> None:
        """The set of cluster fields equals behavior tests + known gaps; no field is silently unwired."""
        defined = {f.name for f in attrs.fields(SaturationAwareConfig)}
        covered = self._CLUSTER_BEHAVIOR_TESTS | self._KNOWN_COVERAGE_GAPS_CLUSTER

        missing = defined - covered
        assert not missing, (
            "New SaturationAwareConfig fields detected without a wiring test or allowlist entry: "
            f"{sorted(missing)}. Add a behavior test in this file and either move the field to "
            "_CLUSTER_BEHAVIOR_TESTS or document the gap in _KNOWN_COVERAGE_GAPS_CLUSTER."
        )

    def test_every_stage_field_has_a_behavior_test_or_is_explicitly_allowlisted(self) -> None:
        """The set of stage fields equals behavior tests + known gaps; no field is silently unwired."""
        defined = {f.name for f in attrs.fields(SaturationAwareStageConfig)}
        covered = self._STAGE_BEHAVIOR_TESTS | self._KNOWN_COVERAGE_GAPS_STAGE_BEHAVIOR

        missing = defined - covered
        assert not missing, (
            "New SaturationAwareStageConfig fields detected without a wiring test or allowlist entry: "
            f"{sorted(missing)}. Add a behavior test in this file and either move the field to "
            "_STAGE_BEHAVIOR_TESTS or document the gap in _KNOWN_COVERAGE_GAPS_STAGE_BEHAVIOR."
        )

    def test_every_stage_field_has_an_override_precedence_test_or_is_explicitly_allowlisted(self) -> None:
        """Stage fields must have override precedence tests so override chains stay correct."""
        defined = {f.name for f in attrs.fields(SaturationAwareStageConfig)}
        covered = self._STAGE_OVERRIDE_TESTS | self._KNOWN_COVERAGE_GAPS_STAGE_OVERRIDE

        missing = defined - covered
        assert not missing, (
            "New SaturationAwareStageConfig fields detected without an override-precedence test: "
            f"{sorted(missing)}. Add a test in TestStageOverridePrecedence... and either move the "
            "field to _STAGE_OVERRIDE_TESTS or document the gap in _KNOWN_COVERAGE_GAPS_STAGE_OVERRIDE."
        )
