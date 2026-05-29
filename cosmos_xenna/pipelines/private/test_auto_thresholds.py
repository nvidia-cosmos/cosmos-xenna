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

"""Behaviour tests for the auto-derived classifier thresholds.

Pin the operator contract: ``saturation_aggressiveness`` (the
``K`` in ``K/sqrt(c)``) is the single primary knob; the per-threshold
overrides on ``SaturationAwareStageConfig`` opt out of auto-derivation;
the resolved values flow through scheduler ``setup()`` onto
``StageRuntimeState.classifier.resolved_thresholds`` for the lifetime of the run.

Each test verifies one slice of the resolution contract so a future
formula tweak surfaces as a precise test failure instead of a vague
classifier-zone regression.
"""

import math

import pytest
from attrs.exceptions import FrozenInstanceError

from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import (
    ResolvedThresholds,
    _resolve_auto_thresholds,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


class TestFormulaCorrectness:
    """Auto-derived ``saturation_threshold`` matches ``K / sqrt(c)`` to within 1 percent."""

    @pytest.mark.parametrize("slots_per_actor", [1, 2, 4, 8, 16, 32, 64])
    def test_default_aggressiveness_matches_halfin_whitt_formula(
        self,
        slots_per_actor: int,
    ) -> None:
        """``0.30 / sqrt(c)`` for c in {1, 2, 4, 8, 16, 32, 64}, post-clamp."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.30)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=slots_per_actor)

        expected_raw = cfg.saturation_aggressiveness / math.sqrt(slots_per_actor)
        expected_clamped = max(cfg.auto_threshold_min, min(expected_raw, cfg.auto_threshold_max))
        assert resolved.saturation_threshold == pytest.approx(expected_clamped, rel=0.01)

    @pytest.mark.parametrize("aggressiveness", [0.20, 0.30, 0.45, 0.60])
    def test_aggressiveness_scales_threshold_linearly(self, aggressiveness: float) -> None:
        """At fixed ``c`` the formula is linear in ``saturation_aggressiveness``."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=aggressiveness)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
        assert resolved.saturation_threshold == pytest.approx(aggressiveness / math.sqrt(8), rel=0.01)

    def test_default_activation_is_one_third_of_saturation(self) -> None:
        """``activation = saturation * activation_to_saturation_ratio`` (default 0.33)."""
        cfg = SaturationAwareStageConfig()
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
        assert resolved.activation_threshold == pytest.approx(
            resolved.saturation_threshold * 0.33,
            rel=1e-6,
        )

    @pytest.mark.parametrize("ratio", [0.20, 0.50, 0.75])
    def test_activation_to_saturation_ratio_is_applied_to_auto_activation(self, ratio: float) -> None:
        """Auto activation = resolved saturation * ``activation_to_saturation_ratio``."""
        cfg = SaturationAwareStageConfig(activation_to_saturation_ratio=ratio)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
        assert resolved.activation_threshold == pytest.approx(
            resolved.saturation_threshold * ratio,
            rel=1e-6,
        )

    def test_resolved_thresholds_record_aggressiveness_and_slots(self) -> None:
        """The resolved record exposes the formula inputs that produced it."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.45)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=16)
        assert resolved.saturation_aggressiveness == 0.45
        assert resolved.slots_per_actor == 16


class TestAggressivenessOverride:
    """``aggressiveness_override`` substitutes a runtime-adjusted ``K`` without mutating the config."""

    def test_override_replaces_config_aggressiveness_in_formula(self) -> None:
        """Override flows into the K/sqrt(c) numerator instead of cfg.saturation_aggressiveness."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.30)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=4, aggressiveness_override=0.45)
        # 0.45 / sqrt(4) = 0.225, comfortably inside [0.02, 0.45].
        assert resolved.saturation_threshold == pytest.approx(0.225)

    def test_override_recorded_on_resolved_thresholds(self) -> None:
        """The override value, not the config value, is recorded as provenance."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.30)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=4, aggressiveness_override=0.45)
        assert resolved.saturation_aggressiveness == pytest.approx(0.45)

    def test_override_none_falls_back_to_config(self) -> None:
        """Default ``aggressiveness_override=None`` reads ``cfg.saturation_aggressiveness``."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.45)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=4)
        assert resolved.saturation_aggressiveness == pytest.approx(0.45)
        assert resolved.saturation_threshold == pytest.approx(0.225)

    def test_override_clamps_at_auto_threshold_max(self) -> None:
        """A high override value is still clamped by ``auto_threshold_max``."""
        cfg = SaturationAwareStageConfig()  # auto_threshold_max=0.45
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=1, aggressiveness_override=0.60)
        assert resolved.saturation_threshold == pytest.approx(cfg.auto_threshold_max)

    def test_override_does_not_affect_pinned_saturation_threshold(self) -> None:
        """When ``saturation_threshold`` is pinned the override is unused for that field."""
        cfg = SaturationAwareStageConfig(saturation_threshold=0.10)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=4, aggressiveness_override=0.45)
        assert resolved.saturation_threshold == pytest.approx(0.10)
        assert resolved.saturation_threshold_was_overridden is True


class TestClamps:
    """``auto_threshold_min`` and ``auto_threshold_max`` bound the auto-derived saturation threshold."""

    def test_large_c_clamps_at_auto_threshold_min(self) -> None:
        """At c=256 the raw formula 0.30/16 = 0.01875 falls below the 0.02 floor."""
        cfg = SaturationAwareStageConfig()
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=256)
        assert resolved.saturation_threshold == pytest.approx(cfg.auto_threshold_min)

    def test_small_c_does_not_exceed_auto_threshold_max(self) -> None:
        """At c=1 with aggressiveness=0.60 the raw formula equals 0.60; ``cfg.auto_threshold_max`` clamps it.

        Pair the upper clamp with a sufficiently-high
        ``over_provisioned_threshold`` so the resolved triple still
        satisfies the ``activation < saturation < over_provisioned``
        ordering enforced by the resolver.
        """
        cfg = SaturationAwareStageConfig(
            saturation_aggressiveness=0.60,
            over_provisioned_threshold=0.80,
        )
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=1)
        assert resolved.saturation_threshold == pytest.approx(cfg.auto_threshold_max)

    def test_clamp_preserves_below_max_value_unchanged(self) -> None:
        """A formula output strictly inside the clamp interval is returned unchanged."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.30)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=4)
        # 0.30 / sqrt(4) = 0.15, comfortably inside [auto_threshold_min, auto_threshold_max].
        assert resolved.saturation_threshold == pytest.approx(0.15)


class TestOverrideHierarchy:
    """An explicit numeric value on the config overrides the auto-derived value."""

    def test_saturation_only_override_derives_activation_from_override(self) -> None:
        """``saturation_threshold = 0.10`` -> activation = 0.10 * 0.33 = 0.033."""
        cfg = SaturationAwareStageConfig(saturation_threshold=0.10)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
        assert resolved.saturation_threshold == pytest.approx(0.10)
        assert resolved.activation_threshold == pytest.approx(0.10 * 0.33)
        assert resolved.saturation_threshold_was_overridden is True
        assert resolved.activation_threshold_was_overridden is False

    def test_activation_only_override_derives_saturation_from_formula(self) -> None:
        """``activation_threshold = 0.05`` -> saturation = K / sqrt(c) (formula)."""
        cfg = SaturationAwareStageConfig(saturation_aggressiveness=0.30, activation_threshold=0.05)
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=4)
        assert resolved.saturation_threshold == pytest.approx(cfg.saturation_aggressiveness / math.sqrt(4))
        assert resolved.activation_threshold == pytest.approx(0.05)
        assert resolved.saturation_threshold_was_overridden is False
        assert resolved.activation_threshold_was_overridden is True

    def test_both_overrides_bypass_aggressiveness_and_ratio(self) -> None:
        """Both pinned -> ``saturation_aggressiveness`` and the ratio are unused."""
        cfg = SaturationAwareStageConfig(
            saturation_threshold=0.10,
            activation_threshold=0.04,
            saturation_aggressiveness=0.60,
            activation_to_saturation_ratio=0.50,
        )
        resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
        assert resolved.saturation_threshold == pytest.approx(0.10)
        assert resolved.activation_threshold == pytest.approx(0.04)
        assert resolved.saturation_threshold_was_overridden is True
        assert resolved.activation_threshold_was_overridden is True

    def test_inconsistent_both_pinned_overrides_caught_by_attrs_post_init(self) -> None:
        """Both pinned with activation >= saturation -- ``__attrs_post_init__`` rejects."""
        # When both thresholds are explicit, the config-level cross-field
        # validator catches the ordering violation and the resolver never
        # sees the pathological config.
        with pytest.raises(ValueError, match="Threshold ordering violated"):
            SaturationAwareStageConfig(saturation_threshold=0.10, activation_threshold=0.10)

    def test_activation_pinned_high_with_auto_saturation_clamped_low_raises(self) -> None:
        """Activation pinned high, saturation auto, large ``c`` -> resolver detects ordering violation.

        The reachable pathological case: the operator pins
        ``activation_threshold = 0.40``, leaves ``saturation_threshold
        = None`` so it auto-derives, and the stage runs at ``c = 256``
        so the formula clamps saturation at the floor (0.02). Result:
        activation (0.40) >= saturation (0.02). The resolver's
        ordering check rejects this.
        """
        cfg = SaturationAwareStageConfig(activation_threshold=0.40)
        with pytest.raises(ValueError, match="resolved thresholds violate zone ordering"):
            _resolve_auto_thresholds(cfg, slots_per_actor=256)

    def test_activation_pinned_equal_to_auto_saturation_floor_raises(self) -> None:
        """Strict ``<`` boundary: activation == auto saturation must reject."""
        cfg = SaturationAwareStageConfig(activation_threshold=0.02)  # equals auto_threshold_min
        with pytest.raises(ValueError, match="resolved thresholds violate zone ordering"):
            _resolve_auto_thresholds(cfg, slots_per_actor=256)


class TestOverProvisionedOrderingGuard:
    """Cross-field validator forbids ``auto_threshold_max >= over_provisioned_threshold``.

    The auto-derived saturation can clamp at ``auto_threshold_max``;
    if that ceiling is at or above ``over_provisioned_threshold`` the
    SATURATED zone would touch or invert with OVER_PROVISIONED.
    Catching it at construction is fail-fast vs surfacing as a
    resolver runtime error on the first cycle that hits the clamp.
    """

    def test_low_over_provisioned_below_auto_threshold_max_rejected_at_construction(self) -> None:
        """over_provisioned=0.20 + default auto_threshold_max=0.45 -> construction raises."""
        with pytest.raises(ValueError, match="auto_threshold_max .* must be < over_provisioned_threshold"):
            SaturationAwareStageConfig(over_provisioned_threshold=0.20)

    def test_equal_auto_threshold_max_and_over_provisioned_rejected(self) -> None:
        """``auto_threshold_max == over_provisioned_threshold`` rejected (strict inequality)."""
        with pytest.raises(ValueError, match="auto_threshold_max .* must be < over_provisioned_threshold"):
            SaturationAwareStageConfig(auto_threshold_max=0.50, over_provisioned_threshold=0.50)


class TestInvalidInputs:
    """The resolver rejects nonsense inputs so caller bugs surface immediately."""

    def test_zero_slots_per_actor_raises(self) -> None:
        """``slots_per_actor=0`` would trigger a divide-by-zero in the formula."""
        cfg = SaturationAwareStageConfig()
        with pytest.raises(ValueError, match="slots_per_actor must be >= 1"):
            _resolve_auto_thresholds(cfg, slots_per_actor=0)

    def test_negative_slots_per_actor_raises(self) -> None:
        """``slots_per_actor`` is a count; negative values are nonsensical."""
        cfg = SaturationAwareStageConfig()
        with pytest.raises(ValueError, match="slots_per_actor must be >= 1"):
            _resolve_auto_thresholds(cfg, slots_per_actor=-1)


class TestConfigCrossFieldValidators:
    """The new aggressiveness / clamp / ratio fields fail-fast at config construction time."""

    def test_aggressiveness_below_range_is_rejected(self) -> None:
        """``saturation_aggressiveness`` must be at least 0.10 (lower bound on operator-facing range)."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(saturation_aggressiveness=0.05)

    def test_aggressiveness_above_range_is_rejected(self) -> None:
        """``saturation_aggressiveness`` must be at most 0.60 (upper bound on operator-facing range)."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(saturation_aggressiveness=0.70)

    def test_auto_threshold_min_above_max_is_rejected(self) -> None:
        """The clamp interval must be non-empty: ``auto_threshold_min < auto_threshold_max``."""
        with pytest.raises(ValueError, match="auto_threshold_min"):
            SaturationAwareStageConfig(auto_threshold_min=0.05, auto_threshold_max=0.04)

    def test_auto_threshold_min_zero_is_rejected(self) -> None:
        """The lower clamp must be strictly positive (zero would let the formula collapse)."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(auto_threshold_min=0.0)

    def test_activation_to_saturation_ratio_at_one_is_rejected(self) -> None:
        """ratio = 1 would collapse SATURATED into SATURATED_CRITICAL."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(activation_to_saturation_ratio=1.0)

    def test_activation_to_saturation_ratio_at_zero_is_rejected(self) -> None:
        """ratio = 0 would make SATURATED_CRITICAL unreachable."""
        with pytest.raises(ValueError):
            SaturationAwareStageConfig(activation_to_saturation_ratio=0.0)


class TestResolvedStateLifecycle:
    """``ResolvedThresholds`` is immutable (``attrs.frozen``)."""

    def test_resolved_thresholds_is_frozen(self) -> None:
        """A future caller cannot mutate a resolved record."""
        resolved = ResolvedThresholds(
            saturation_threshold=0.15,
            activation_threshold=0.05,
            saturation_aggressiveness=0.30,
            slots_per_actor=8,
            saturation_threshold_was_overridden=False,
            activation_threshold_was_overridden=False,
        )
        with pytest.raises(FrozenInstanceError):
            resolved.saturation_threshold = 0.20  # type: ignore[misc]
