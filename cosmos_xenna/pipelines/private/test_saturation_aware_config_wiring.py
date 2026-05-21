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

Contract pinned here:
    * Every cluster config field has a behavior test asserting the
      runtime observable changes when the field is mutated.
    * Every stage config field has a behavior test and an
      override-precedence test (``stage_defaults`` -> per-stage
      override -> ``StageSpec.saturation_aware``).
    * ``TestConfigWiringMetaCoverage`` enumerates every public field
      via ``attrs.fields`` and fails if any field is missing the
      required behavior / override tests. It carries an explicit
      ``_KNOWN_COVERAGE_GAPS`` allowlist that documents fields whose
      wiring tests are pending; closing that allowlist is intentional
      future work so the meta-test is informational today and
      enforcing tomorrow without a parallel-suite migration.

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

from typing import ClassVar

import attrs
import pytest
from loguru import logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import (
    record_executed_delta,
    run_per_stage_pipeline,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
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


# --------------------------------------------------------------------------
# TestClusterConfigFieldBehavior - cluster-scoped wiring
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# TestStageConfigFieldBehavior - stage-scoped wiring
# --------------------------------------------------------------------------


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
        sat_cfg = SaturationAwareConfig(
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

    def test_warmup_filtered_all_zero_samples_do_not_advance_counter(self) -> None:
        """Cycles whose warmup filter drops every contribution leave the counter unchanged."""
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                min_data_points=2,
                worker_warmup_measurement_grace_s=999.0,
                donor_warmup_grace_s=999.0,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["S"]))
        ps = _problem_state_with_signals([("S", 1, 8, 8, 0)])

        scheduler.autoscale(time=0.0, problem_state=ps)
        scheduler.autoscale(time=1.0, problem_state=ps)

        assert scheduler._stage_states["S"].valid_signal_samples == 0


class TestGrowthModeStateMachineKillSwitch:
    """``enable_growth_mode_state_machine=False`` forces TRACKING magnitudes."""

    def test_disabled_forces_tracking_magnitude_even_when_state_is_acquiring(self) -> None:
        """A pre-existing ACQUIRING state does not change magnitude when the kill switch is off.

        With a large worker count (32) the ACQUIRING multiplicative path
        produces +8 (clamped by aggressive_growth_max_per_cycle=4 -> +4),
        while TRACKING produces +2 (``tracking_critical_growth_count``).
        Disabling the state machine must pin the magnitude to the TRACKING
        value regardless of ``growth_mode``.
        """
        stage_cfg = SaturationAwareStageConfig(
            enable_growth_mode_state_machine=False,
            saturated_critical_streak_min_cycles=1,
            min_data_points=1,
            worker_warmup_measurement_grace_s=0.0,
        )
        state = _resolved_state(config=stage_cfg)
        state.growth_mode = GrowthMode.ACQUIRING
        state.classifier_state = StageState.SATURATED_CRITICAL
        state.classifier_streak = 5

        # SATURATED_CRITICAL fires on a non-empty input queue + zero empty slots.
        # tracking_critical_growth_count=2; ACQUIRING+SATURATED_CRITICAL on 32 workers
        # is ceil(0.5 * 32)=16 clamped to aggressive_growth_max_per_cycle=4. So a +2
        # answer pins the TRACKING path; +4 would mean ACQUIRING leaked through.
        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=32 * 8,
            num_empty_slots=0,
            input_queue_depth=100,
            current_workers=32,
            config=stage_cfg,
        )

        assert delta == stage_cfg.tracking_critical_growth_count, (
            f"Kill switch did not force TRACKING magnitude; got delta={delta}"
        )

    def test_disabled_skips_transition_so_state_does_not_drift(self) -> None:
        """``record_executed_delta`` is a no-op when the kill switch is off."""
        stage_cfg = SaturationAwareStageConfig(enable_growth_mode_state_machine=False)
        state = _resolved_state(config=stage_cfg)
        initial_mode = state.growth_mode
        initial_streak = state.growth_streak

        record_executed_delta(stage_state=state, delta_executed=-5, config=stage_cfg)

        assert state.growth_mode is initial_mode
        assert state.growth_streak == initial_streak


class TestStarvedWarningIsRateLimited:
    """The STARVED INFO line fires exactly once per streak crossing."""

    def test_warning_fires_on_streak_threshold_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """The log line emits when streak == threshold; not before, not on every subsequent cycle."""
        sat_cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                starved_streak_min_cycles=2,
                min_data_points=1,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
            ),
        )
        scheduler = SaturationAwareScheduler(sat_cfg)
        scheduler.setup(_problem(["upstream", "S"]))

        # All slots empty + queue=0 -> STARVED classification
        ps = _problem_state_with_signals([("upstream", 1, 8, 1, 7), ("S", 1, 8, 0, 8)])

        log_lines: list[str] = []
        sink_id = logger.add(lambda msg: log_lines.append(msg.record["message"]), level="INFO")
        try:
            scheduler.autoscale(time=0.0, problem_state=ps)
            scheduler.autoscale(time=1.0, problem_state=ps)
            scheduler.autoscale(time=2.0, problem_state=ps)
        finally:
            logger.remove(sink_id)

        starved_lines = [line for line in log_lines if "classifier STARVED" in line and "stage 'S'" in line]
        assert len(starved_lines) == 1, (
            f"Expected exactly one STARVED log line at streak threshold; got: {starved_lines}"
        )


# --------------------------------------------------------------------------
# TestStageConfigOverridePrecedence - propagation through the override chain
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# TestConfigWiringMetaCoverage - guardrail against new unwired fields
# --------------------------------------------------------------------------


class TestConfigWiringMetaCoverage:
    """Enumerate every public config field and verify wiring test coverage exists.

    The meta-test maintains an explicit ``_KNOWN_COVERAGE_GAPS``
    allowlist of fields whose wiring tests are pending. Closing the
    allowlist is intentional ongoing work, tracked by the saturation-
    aware roadmap. The meta-test is informational while the allowlist
    is non-empty; once the allowlist is emptied, removing it makes the
    test strictly enforcing and any future unwired field becomes a
    test failure (the same defect class as the original
    ``interval_s`` regression).
    """

    _CLUSTER_BEHAVIOR_TESTS: ClassVar[set[str]] = {
        "interval_s",
    }

    _STAGE_BEHAVIOR_TESTS: ClassVar[set[str]] = {
        "min_data_points",
        "enable_growth_mode_state_machine",
        "starved_streak_min_cycles",
    }

    _STAGE_OVERRIDE_TESTS: ClassVar[set[str]] = {
        "min_data_points",
    }

    # Fields the wiring suite does not yet cover. Each entry is tracked
    # in ``docs/scheduler/saturation-aware/tuning.md`` or the roadmap;
    # add a wiring test and remove the field here in the same commit.
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
        "enable_bottleneck_priority_growth",
        "enable_bottleneck_shrink_protection",
        "bottleneck_d_k_smoothing_level",
        "bottleneck_heterogeneity_threshold",
        "bottleneck_engagement_persistence_cycles",
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
        "acquiring_critical_growth_factor",
        "acquiring_saturated_growth_factor",
        "tracking_critical_growth_count",
        "tracking_saturated_growth_count",
        "hold_critical_growth_count",
        "hold_saturated_growth_count",
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
        "target_backlog_seconds",
        "pressure_smoothing_level",
        "pressure_critical_threshold",
        "pressure_saturation_threshold",
        "pressure_normal_threshold",
        "enable_backlog_time_classifier",
    }

    _KNOWN_COVERAGE_GAPS_STAGE_OVERRIDE: ClassVar[set[str]] = _KNOWN_COVERAGE_GAPS_STAGE_BEHAVIOR | {
        "enable_growth_mode_state_machine",
        "starved_streak_min_cycles",
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
