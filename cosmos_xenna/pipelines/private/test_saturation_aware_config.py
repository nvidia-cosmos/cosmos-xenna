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

"""Validator and resolver tests for the saturation-aware scheduler config.

These tests pin the public contract of ``SaturationAwareStageConfig`` and
``SaturationAwareConfig`` so a future refactor cannot silently weaken
either the per-field validators or the cross-field invariants. The
three-tier resolver behaviour is also pinned because downstream phases
rely on its precedence semantics.
"""

from typing import cast

import attrs
import pytest

from cosmos_xenna.pipelines.private.specs import (
    SaturationAwareConfig,
    SaturationAwareStageConfig,
    SchedulerKind,
    StreamingSpecificSpec,
)


class TestSchedulerKind:
    """Pin the enum values that operators select via spec / CLI."""

    def test_default_scheduler_is_fragmentation_based(self) -> None:
        """Default selection is the legacy scheduler (no behaviour change for existing pipelines)."""
        spec = StreamingSpecificSpec()
        assert spec.scheduler is SchedulerKind.FRAGMENTATION_BASED

    def test_string_values_are_purpose_based(self) -> None:
        """Enum values are purpose-based (not language-based) so YAML overrides are stable."""
        assert SchedulerKind.FRAGMENTATION_BASED.value == "fragmentation_based"
        assert SchedulerKind.SATURATION_AWARE.value == "saturation_aware"


class TestSaturationAwareStageConfigFieldValidators:
    """Single-field validators reject invalid values at __init__ time.

    One test per validator predicate so a future change that loosens any
    individual constraint surfaces as a precise test failure.
    """

    def test_default_construction_succeeds(self) -> None:
        """All defaults are mutually consistent.

        ``saturation_threshold`` and ``activation_threshold`` default to
        ``None`` - the resolver derives them lazily on the first
        ``autoscale()`` cycle from ``saturation_aggressiveness`` and
        the stage's runtime ``slots_per_worker``.
        """
        cfg = SaturationAwareStageConfig()
        assert cfg.min_data_points == 5
        assert cfg.saturation_aggressiveness == 0.30
        assert cfg.saturation_threshold is None
        assert cfg.activation_threshold is None

    def test_min_data_points_zero_is_rejected(self) -> None:
        """``validate_positive_int`` rejects values below 1."""
        with pytest.raises(ValueError, match="min_data_points must be >= 1"):
            SaturationAwareStageConfig(min_data_points=0)

    def test_saturation_threshold_above_one_is_rejected(self) -> None:
        """Fraction must lie in [0, 1]."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(saturation_threshold=1.1)

    def test_saturation_threshold_negative_is_rejected(self) -> None:
        """Fraction must lie in [0, 1]."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(saturation_threshold=-0.1)

    def test_slots_empty_ratio_smoothing_level_zero_is_rejected(self) -> None:
        """Smoothing level must be > 0 (a level of 0 freezes the EWMA forever)."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(slots_empty_ratio_smoothing_level=0.0)

    def test_acquiring_growth_factor_zero_is_rejected(self) -> None:
        """Multiplicative growth factor must be > 0; zero would never grow."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(acquiring_critical_growth_factor=0.0)

    def test_min_workers_zero_is_rejected(self) -> None:
        """``validate_optional_positive_int`` rejects 0 (cannot disable the implicit floor)."""
        with pytest.raises(ValueError, match="must be >= 1"):
            SaturationAwareStageConfig(min_workers=0)

    def test_min_workers_none_is_accepted(self) -> None:
        """``None`` means 'use the implicit one-worker floor'."""
        cfg = SaturationAwareStageConfig(min_workers=None)
        assert cfg.min_workers is None

    def test_max_workers_per_node_negative_is_rejected(self) -> None:
        """``validate_optional_positive_int`` rejects negatives."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(max_workers_per_node=-1)

    def test_donor_warmup_grace_negative_is_rejected(self) -> None:
        """Grace window cannot be negative."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(donor_warmup_grace_s=-1.0)


class TestPressureFieldValidators:
    """Single-field validators for the backlog-time pressure classifier knobs."""

    @pytest.fixture(
        params=[
            "target_backlog_seconds",
            "pressure_saturation_threshold",
            "pressure_normal_threshold",
        ]
    )
    def positive_float_field(self, request: pytest.FixtureRequest) -> str:
        return cast(str, request.param)

    def test_zero_is_rejected(self, positive_float_field: str) -> None:
        """Strictly-positive validator forbids zero values."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(**{positive_float_field: 0.0})  # type: ignore[arg-type]

    def test_negative_is_rejected(self, positive_float_field: str) -> None:
        """Strictly-positive validator forbids negatives."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(**{positive_float_field: -0.1})  # type: ignore[arg-type]

    def test_pressure_critical_threshold_negative_is_rejected(self) -> None:
        """The CRITICAL gate must be strictly positive (otherwise the slot pin always fires)."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(pressure_critical_threshold=-0.1)

    def test_pressure_critical_threshold_zero_is_rejected(self) -> None:
        """Zero would let any positive pressure fire CRITICAL on a slot pin."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(pressure_critical_threshold=0.0)

    def test_pressure_smoothing_level_zero_is_rejected(self) -> None:
        """Smoothing level of 0 would freeze the pressure EWMA forever."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(pressure_smoothing_level=0.0)

    def test_pressure_smoothing_level_above_one_is_rejected(self) -> None:
        """Smoothing level is bounded at 1.0 (alpha == 1.0 disables smoothing)."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(pressure_smoothing_level=1.01)

    def test_enable_backlog_time_classifier_default_is_true(self) -> None:
        """The backlog-time pressure classifier is enabled by default."""
        cfg = SaturationAwareStageConfig()
        assert cfg.enable_backlog_time_classifier is True

    def test_enable_backlog_time_classifier_must_be_bool(self) -> None:
        """Truthy strings are rejected so an operator typo cannot silently disable the gate."""
        with pytest.raises(TypeError):
            SaturationAwareStageConfig(enable_backlog_time_classifier=cast(bool, "yes"))


class TestSaturationAwareStageConfigCrossFieldValidators:
    """Cross-field invariants enforced in __attrs_post_init__."""

    def test_threshold_ordering_violation_is_rejected(self) -> None:
        """activation < saturation < over_provisioned must hold."""
        # activation_threshold > saturation_threshold violates the chain.
        with pytest.raises(ValueError, match="Threshold ordering violated"):
            SaturationAwareStageConfig(activation_threshold=0.20, saturation_threshold=0.15)

    def test_donor_grace_below_worker_grace_is_rejected(self) -> None:
        """Donor grace must cover at least the worker measurement grace."""
        with pytest.raises(ValueError, match="donor_warmup_grace_s"):
            SaturationAwareStageConfig(
                worker_warmup_measurement_grace_s=120.0,
                donor_warmup_grace_s=60.0,
            )

    def test_over_provisioned_streak_not_dominating_saturated_streak_is_rejected(self) -> None:
        """Asymmetric stabilization requires the shrink streak to dominate the growth streak."""
        with pytest.raises(ValueError, match="strictly > saturated_streak_min_cycles"):
            SaturationAwareStageConfig(
                saturated_streak_min_cycles=5,
                over_provisioned_streak_min_cycles=5,
            )

    def test_stabilization_window_down_not_dominating_up_is_rejected(self) -> None:
        """Asymmetric stabilization requires the down window to be larger than the up window."""
        with pytest.raises(ValueError, match="stabilization_window_cycles_down"):
            SaturationAwareStageConfig(
                stabilization_window_cycles_up=5,
                stabilization_window_cycles_down=5,
            )

    def test_min_workers_above_max_workers_is_rejected(self) -> None:
        """Per-stage cluster-wide floor cannot exceed cluster-wide cap."""
        with pytest.raises(ValueError, match=r"min_workers .* must be <= max_workers"):
            SaturationAwareStageConfig(min_workers=8, max_workers=4)

    def test_min_workers_per_node_above_max_workers_per_node_is_rejected(self) -> None:
        """Per-stage per-node floor cannot exceed per-stage per-node cap."""
        with pytest.raises(ValueError, match="min_workers_per_node"):
            SaturationAwareStageConfig(min_workers_per_node=4, max_workers_per_node=2)

    def test_min_workers_only_set_does_not_trigger_max_check(self) -> None:
        """Cross-field check only fires when both sides are set."""
        cfg = SaturationAwareStageConfig(min_workers=4, max_workers=None)
        assert cfg.min_workers == 4
        assert cfg.max_workers is None

    def test_pressure_critical_below_saturation_is_rejected(self) -> None:
        """CRITICAL must require a strictly larger pressure than SATURATED."""
        with pytest.raises(ValueError, match="pressure_critical_threshold"):
            SaturationAwareStageConfig(
                pressure_critical_threshold=0.5,
                pressure_saturation_threshold=1.0,
            )

    def test_pressure_critical_equals_saturation_is_rejected(self) -> None:
        """The strict-inequality variant is enforced (no implicit hysteresis between gates)."""
        with pytest.raises(ValueError, match="pressure_critical_threshold"):
            SaturationAwareStageConfig(
                pressure_critical_threshold=1.0,
                pressure_saturation_threshold=1.0,
            )

    def test_pressure_saturation_below_normal_is_rejected(self) -> None:
        """SATURATED must require strictly larger pressure than the OVER_PROVISIONED demotion gate."""
        with pytest.raises(ValueError, match="pressure_saturation_threshold"):
            SaturationAwareStageConfig(
                pressure_saturation_threshold=0.2,
                pressure_normal_threshold=0.3,
            )

    def test_pressure_critical_above_cap_is_rejected(self) -> None:
        """A CRITICAL threshold above the backlog cap can never fire."""
        with pytest.raises(ValueError, match="backlog cap"):
            SaturationAwareStageConfig(pressure_critical_threshold=3.5)


class TestSaturationAwareConfigClusterValidators:
    """Cluster-level cross-field validators on ``SaturationAwareConfig``."""

    def test_default_construction_succeeds(self) -> None:
        """Defaults are mutually consistent (anti-flap dominates default streak)."""
        cfg = SaturationAwareConfig()
        assert cfg.cross_stage_donor_anti_flap_cycles >= cfg.stage_defaults.over_provisioned_streak_min_cycles

    def test_anti_flap_below_longest_stage_streak_is_rejected(self) -> None:
        """Anti-flap must dominate the longest shrink streak across stage_defaults + per_stage_overrides."""
        # Build an override with a longer shrink streak than the cluster-level anti-flap window.
        long_shrink = attrs.evolve(
            SaturationAwareStageConfig(),
            saturated_streak_min_cycles=5,
            over_provisioned_streak_min_cycles=60,
        )
        with pytest.raises(ValueError, match="cross_stage_donor_anti_flap_cycles"):
            SaturationAwareConfig(
                cross_stage_donor_anti_flap_cycles=30,
                per_stage_overrides={"slow_stage": long_shrink},
            )

    def test_interval_s_zero_is_rejected(self) -> None:
        """Cycle period must be strictly positive."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(interval_s=0.0)

    def test_floor_stuck_grace_cycles_negative_is_rejected(self) -> None:
        """Floor stuck grace is a non-negative cycle count."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(floor_stuck_grace_cycles=-1)

    def test_memory_pressure_threshold_above_one_is_rejected(self) -> None:
        """Threshold is a fraction in (0, 1]."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(memory_pressure_critical_threshold=1.5)


class TestRegimeAwareFields:
    """Single-field validators for the regime-aware aggressiveness lift."""

    def test_enable_regime_aware_aggressiveness_default_is_true(self) -> None:
        """The lift is enabled by default."""
        assert SaturationAwareConfig().enable_regime_aware_aggressiveness is True

    def test_enable_regime_aware_aggressiveness_must_be_bool(self) -> None:
        """String values are rejected instead of being treated as truthy."""
        with pytest.raises(TypeError):
            SaturationAwareConfig(enable_regime_aware_aggressiveness=cast(bool, "false"))

    def test_super_halfin_whitt_aggressiveness_lift_default(self) -> None:
        """Default lift value matches the documented Halfin-Whitt-derived recommendation."""
        assert SaturationAwareConfig().super_halfin_whitt_aggressiveness_lift == pytest.approx(0.15)

    def test_super_halfin_whitt_aggressiveness_lift_negative_is_rejected(self) -> None:
        """The lift must be non-negative."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(super_halfin_whitt_aggressiveness_lift=-0.01)

    def test_super_halfin_whitt_aggressiveness_lift_above_max_is_rejected(self) -> None:
        """The lift is bounded at 0.5 to prevent runaway aggressiveness."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(super_halfin_whitt_aggressiveness_lift=0.51)

    def test_regime_transition_streak_cycles_zero_is_rejected(self) -> None:
        """A streak of 0 cycles would commit a transition on the first cycle (no hysteresis)."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(regime_transition_streak_cycles=0)

    def test_regime_transition_streak_cycles_negative_is_rejected(self) -> None:
        """A negative streak length is meaningless."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(regime_transition_streak_cycles=-1)


class TestSaturationAwareConfigPositiveIntValidators:
    """Per-field positive-int validators reject zero / negative values.

    Each field below uses ``attrs_utils.validate_positive_int``; a regression
    that drops the validator (e.g. switching to a plain ``attrs.field``) would
    silently accept zero or negative cycle counts and break anti-flap or
    heterogeneity guardrails. Parameterised so adding a new positive-int
    field requires only a new tuple in ``params``.
    """

    @pytest.fixture(
        params=[
            "cross_stage_donor_anti_flap_cycles",
            "cross_stage_donor_max_per_cycle",
            "cross_stage_donor_min_donation_interval_cycles",
            "stuck_plan_detection_cycles",
            "cluster_heterogeneity_warn_streak",
        ]
    )
    def positive_int_field(self, request: pytest.FixtureRequest) -> str:
        return cast(str, request.param)

    def test_zero_is_rejected(self, positive_int_field: str) -> None:
        """``positive_int`` validator forbids zero -- a zero-cycle window disables the guardrail."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(**{positive_int_field: 0})  # type: ignore[arg-type]

    def test_negative_is_rejected(self, positive_int_field: str) -> None:
        """A negative cycle count is structurally meaningless."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(**{positive_int_field: -1})  # type: ignore[arg-type]


class TestSaturationAwareConfigBoundedFractionValidators:
    """Validators that bound a knob to ``(0, 1]`` or ``> 1.0`` reject out-of-range values.

    These knobs are operator-tunable; without the validator a misconfigured
    value would slip into the watchdog / heterogeneity-warning path and
    silently disable the alert.
    """

    def test_cycle_time_warn_threshold_zero_is_rejected(self) -> None:
        """Threshold must be strictly positive (zero would warn every cycle)."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(cycle_time_warn_threshold=0.0)

    def test_cycle_time_warn_threshold_above_one_is_rejected(self) -> None:
        """Threshold is a fraction of ``interval_s``; values above 1.0 are nonsensical."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(cycle_time_warn_threshold=1.01)

    def test_memory_pressure_polling_interval_zero_is_rejected(self) -> None:
        """Polling interval must be strictly positive (zero would tight-loop)."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(memory_pressure_polling_interval_s=0.0)

    def test_cluster_heterogeneity_warn_threshold_at_floor_is_rejected(self) -> None:
        """Threshold must be strictly greater than 1.0 -- a homogeneous pipeline is ratio 1.0."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(cluster_heterogeneity_warn_threshold=1.0)


class TestBottleneckDecisionFields:
    """Pin defaults and validators for the four bottleneck-decision knobs."""

    def test_defaults_enable_both_decision_paths(self) -> None:
        """Aggressive rollout: both Phase C and Phase D bottleneck gates default on."""
        cfg = SaturationAwareConfig()
        assert cfg.enable_bottleneck_priority_growth is True
        assert cfg.enable_bottleneck_shrink_protection is True

    def test_default_smoothing_level_is_within_open_unit_interval(self) -> None:
        """Default EWMA alpha is a sensible 0.20 (slow enough to ignore single-cycle noise)."""
        cfg = SaturationAwareConfig()
        assert cfg.bottleneck_d_k_smoothing_level == pytest.approx(0.20)

    def test_default_heterogeneity_threshold_is_2x(self) -> None:
        """Default heterogeneity threshold matches the design doc."""
        cfg = SaturationAwareConfig()
        assert cfg.bottleneck_heterogeneity_threshold == pytest.approx(2.0)

    def test_default_engagement_persistence_is_two_cycles(self) -> None:
        """Default debounces a single noisy cycle from flipping the engagement log."""
        cfg = SaturationAwareConfig()
        assert cfg.bottleneck_engagement_persistence_cycles == 2

    def test_smoothing_level_zero_is_rejected(self) -> None:
        """``alpha=0`` would freeze the EWMA forever -- not allowed."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(bottleneck_d_k_smoothing_level=0.0)

    def test_smoothing_level_above_one_is_rejected(self) -> None:
        """``alpha>1`` is geometrically nonsensical for an EWMA."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(bottleneck_d_k_smoothing_level=1.01)

    def test_smoothing_level_at_one_is_allowed(self) -> None:
        """``alpha=1`` (no smoothing) is the upper bound of the valid range."""
        cfg = SaturationAwareConfig(bottleneck_d_k_smoothing_level=1.0)
        assert cfg.bottleneck_d_k_smoothing_level == pytest.approx(1.0)

    def test_heterogeneity_threshold_at_one_is_rejected(self) -> None:
        """A homogeneous cluster has ratio 1.0; the threshold must be strictly greater."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(bottleneck_heterogeneity_threshold=1.0)

    def test_engagement_persistence_zero_is_rejected(self) -> None:
        """Persistence must be a positive int (zero would log on every flip)."""
        with pytest.raises(ValueError):
            SaturationAwareConfig(bottleneck_engagement_persistence_cycles=0)


class TestThreeTierResolver:
    """``get_effective_stage_config`` resolves overrides by precedence.

    Precedence (highest first):
      1. ``StageSpec.saturation_aware`` - passed in as ``spec_override``
      2. ``SaturationAwareConfig.per_stage_overrides[stage_name]``
      3. ``SaturationAwareConfig.stage_defaults``
    """

    def test_no_override_falls_back_to_stage_defaults(self) -> None:
        """A stage with no override anywhere uses the cluster defaults."""
        cfg = SaturationAwareConfig()
        resolved = cfg.get_effective_stage_config("StageA", spec_override=None)
        assert resolved is cfg.stage_defaults

    def test_per_stage_overrides_dict_wins_over_defaults(self) -> None:
        """A stage named in the per_stage_overrides dict uses that override."""
        custom = attrs.evolve(SaturationAwareStageConfig(), min_workers=4)
        cfg = SaturationAwareConfig(per_stage_overrides={"FastStage": custom})
        resolved = cfg.get_effective_stage_config("FastStage", spec_override=None)
        assert resolved is custom

    def test_stage_spec_override_wins_over_per_stage_overrides(self) -> None:
        """When both spec_override AND per_stage_overrides are present, spec_override wins."""
        from_dict = attrs.evolve(SaturationAwareStageConfig(), min_workers=4)
        from_spec = attrs.evolve(SaturationAwareStageConfig(), min_workers=8)
        cfg = SaturationAwareConfig(per_stage_overrides={"StageA": from_dict})
        resolved = cfg.get_effective_stage_config("StageA", spec_override=from_spec)
        assert resolved is from_spec
        assert resolved.min_workers == 8

    def test_unrelated_stage_name_falls_through_to_defaults(self) -> None:
        """A stage NOT in per_stage_overrides resolves to defaults even when other stages are overridden."""
        custom = attrs.evolve(SaturationAwareStageConfig(), min_workers=4)
        cfg = SaturationAwareConfig(per_stage_overrides={"OtherStage": custom})
        resolved = cfg.get_effective_stage_config("StageA", spec_override=None)
        assert resolved is cfg.stage_defaults
