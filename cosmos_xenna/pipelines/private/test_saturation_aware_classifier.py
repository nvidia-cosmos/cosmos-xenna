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

"""Behaviour tests for the 5-state classifier.

Each test pins exactly one zone or one hysteresis edge so a future
threshold tweak surfaces as a precise test failure instead of a
vague regression. Default-configured ``SaturationAwareStageConfig``
is used unless the test explicitly varies a knob.

Default thresholds at the time of writing:
  activation_threshold = 0.05
  saturation_threshold = 0.15
  saturation_deadband_pct = 0.15  -> exit-up boundary 0.1725
  over_provisioned_threshold = 0.50
  over_provisioned_deadband_pct = 0.30 -> exit-down boundary 0.35
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.classifier import classify
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Default per-stage config; thresholds and deadbands documented above."""
    return SaturationAwareStageConfig()


class TestSaturatedCriticalZone:
    """Empty-slot ratio below activation_threshold -> SATURATED_CRITICAL.

    This zone bypasses hysteresis: a true burst must be detected on
    the first cycle so the growth controller can fire its multi-step
    response.
    """

    def test_zero_ratio_classifies_as_critical(self, cfg: SaturationAwareStageConfig) -> None:
        """All slots full -> ratio 0.0 -> CRITICAL."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_just_below_activation_classifies_as_critical(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """activation_threshold is the upper bound of the CRITICAL zone."""
        result = classify(
            slots_empty_ratio_ewma=cfg.activation_threshold - 1e-6,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_critical_bypasses_hysteresis_from_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A drop straight from OVER_PROVISIONED to CRITICAL fires immediately, no deadband."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            prev_state=StageState.OVER_PROVISIONED,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_critical_bypasses_hysteresis_from_saturated(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A SATURATED stage that drops further into CRITICAL fires immediately, not held by deadband."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            prev_state=StageState.SATURATED,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL


class TestSaturatedZone:
    """Activation_threshold <= ratio < saturation_threshold -> SATURATED."""

    def test_at_activation_classifies_as_saturated(self, cfg: SaturationAwareStageConfig) -> None:
        """Exact equality with activation_threshold puts the stage in SATURATED, not CRITICAL."""
        result = classify(
            slots_empty_ratio_ewma=cfg.activation_threshold,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.SATURATED

    def test_just_below_saturation_classifies_as_saturated(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """saturation_threshold is the upper bound of the SATURATED zone."""
        result = classify(
            slots_empty_ratio_ewma=cfg.saturation_threshold - 1e-6,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.SATURATED

    def test_at_saturation_classifies_as_normal_when_no_prev_saturation(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Exact equality with saturation_threshold -- no hysteresis -> NORMAL."""
        result = classify(
            slots_empty_ratio_ewma=cfg.saturation_threshold,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.NORMAL


class TestSaturatedHysteresis:
    """Exit-up from SATURATED requires ratio > saturation_threshold * (1 + deadband_pct)."""

    def test_inside_deadband_holds_saturated(self, cfg: SaturationAwareStageConfig) -> None:
        """A previously SATURATED stage stays SATURATED inside the upper deadband."""
        # Threshold is 0.15; deadband 0.15 -> exit-up boundary 0.1725.
        ratio_inside_deadband = cfg.saturation_threshold * (1.0 + cfg.saturation_deadband_pct / 2.0)
        result = classify(
            slots_empty_ratio_ewma=ratio_inside_deadband,
            input_queue_depth=10,
            prev_state=StageState.SATURATED,
            config=cfg,
        )
        assert result is StageState.SATURATED

    def test_above_deadband_escapes_to_normal(self, cfg: SaturationAwareStageConfig) -> None:
        """Once ratio exceeds the deadband, the stage transitions out of SATURATED."""
        # Threshold is 0.15; deadband 0.15 -> exit-up boundary 0.1725.
        ratio_above_deadband = cfg.saturation_threshold * (1.0 + cfg.saturation_deadband_pct) + 1e-6
        result = classify(
            slots_empty_ratio_ewma=ratio_above_deadband,
            input_queue_depth=10,
            prev_state=StageState.SATURATED,
            config=cfg,
        )
        assert result is StageState.NORMAL

    def test_critical_prev_state_uses_same_hysteresis(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """SATURATED_CRITICAL prev_state shares the hysteresis exit-up boundary with SATURATED."""
        ratio_inside_deadband = cfg.saturation_threshold * (1.0 + cfg.saturation_deadband_pct / 2.0)
        result = classify(
            slots_empty_ratio_ewma=ratio_inside_deadband,
            input_queue_depth=10,
            prev_state=StageState.SATURATED_CRITICAL,
            config=cfg,
        )
        assert result is StageState.SATURATED

    def test_above_deadband_with_empty_queue_classifies_as_normal_not_starved(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Escape-up from SATURATED into the NORMAL band ignores queue depth.

        The SATURATED -> NORMAL transition does not pass through the
        OVER_PROVISIONED check, so an empty queue does not promote the
        result to STARVED. STARVED only fires from the upper zone.
        """
        ratio_above_deadband = cfg.saturation_threshold * (1.0 + cfg.saturation_deadband_pct) + 1e-6
        result = classify(
            slots_empty_ratio_ewma=ratio_above_deadband,
            input_queue_depth=0,
            prev_state=StageState.SATURATED,
            config=cfg,
        )
        assert result is StageState.NORMAL


class TestNormalZone:
    """saturation_threshold <= ratio < over_provisioned_threshold -> NORMAL."""

    def test_midband_ratio_classifies_as_normal(self, cfg: SaturationAwareStageConfig) -> None:
        """A ratio half-way between the two thresholds is unambiguously NORMAL."""
        midband = (cfg.saturation_threshold + cfg.over_provisioned_threshold) / 2.0
        result = classify(
            slots_empty_ratio_ewma=midband,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.NORMAL


class TestOverProvisionedAndStarvedZones:
    """ratio >= over_provisioned_threshold; queue depth disambiguates."""

    def test_high_ratio_with_queue_classifies_as_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Free slots + work upstream pending -> can shed actors."""
        result = classify(
            slots_empty_ratio_ewma=cfg.over_provisioned_threshold + 0.1,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED

    def test_high_ratio_with_empty_queue_classifies_as_starved(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Free slots + empty queue -> upstream is the bottleneck (no local action)."""
        result = classify(
            slots_empty_ratio_ewma=cfg.over_provisioned_threshold + 0.1,
            input_queue_depth=0,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.STARVED

    def test_full_ratio_with_queue_is_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """ratio == 1.0 with queue -> OVER_PROVISIONED (every slot free)."""
        result = classify(
            slots_empty_ratio_ewma=1.0,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED

    def test_at_over_provisioned_threshold_with_queue_is_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Boundary equality: ratio == op_threshold (no hysteresis) + queue > 0 -> OVER_PROVISIONED."""
        result = classify(
            slots_empty_ratio_ewma=cfg.over_provisioned_threshold,
            input_queue_depth=10,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED

    def test_at_over_provisioned_threshold_with_empty_queue_is_starved(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Boundary equality: ratio == op_threshold + queue == 0 -> STARVED.

        Confirms the queue-disambiguation branch fires at the boundary,
        not just strictly above it.
        """
        result = classify(
            slots_empty_ratio_ewma=cfg.over_provisioned_threshold,
            input_queue_depth=0,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.STARVED


class TestOverProvisionedHysteresis:
    """Exit-down from OVER_PROVISIONED requires ratio < threshold * (1 - deadband_pct)."""

    def test_inside_deadband_holds_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Previously OVER_PROVISIONED stage stays OVER_PROVISIONED inside the lower deadband."""
        # Threshold 0.50; deadband 0.30 -> exit-down boundary 0.35.
        ratio_inside_deadband = cfg.over_provisioned_threshold * (1.0 - cfg.over_provisioned_deadband_pct / 2.0)
        result = classify(
            slots_empty_ratio_ewma=ratio_inside_deadband,
            input_queue_depth=10,
            prev_state=StageState.OVER_PROVISIONED,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED

    def test_below_deadband_escapes_to_normal(self, cfg: SaturationAwareStageConfig) -> None:
        """Once ratio drops below the deadband, the stage transitions out of OVER_PROVISIONED."""
        ratio_below_deadband = cfg.over_provisioned_threshold * (1.0 - cfg.over_provisioned_deadband_pct) - 1e-6
        result = classify(
            slots_empty_ratio_ewma=ratio_below_deadband,
            input_queue_depth=10,
            prev_state=StageState.OVER_PROVISIONED,
            config=cfg,
        )
        assert result is StageState.NORMAL

    def test_hysteresis_preserves_starved_when_queue_empties_inside_deadband(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Queue drops to 0 while ratio is in the OVER_PROVISIONED deadband -> STARVED.

        prev_state == OVER_PROVISIONED keeps the lowered op_boundary,
        so the upper-zone classification still applies; the queue
        disambiguation then routes the result to STARVED rather than
        OVER_PROVISIONED.
        """
        ratio_inside_deadband = cfg.over_provisioned_threshold * (1.0 - cfg.over_provisioned_deadband_pct / 2.0)
        result = classify(
            slots_empty_ratio_ewma=ratio_inside_deadband,
            input_queue_depth=0,
            prev_state=StageState.OVER_PROVISIONED,
            config=cfg,
        )
        assert result is StageState.STARVED


class TestStarvedDoesNotApplyToBusyStages:
    """STARVED requires free slots; a fully busy stage with empty queue is not STARVED."""

    def test_zero_ratio_with_empty_queue_classifies_as_critical_not_starved(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A stage saturated on already-pulled work shows CRITICAL, not STARVED."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=0,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_normal_ratio_with_empty_queue_remains_normal(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A NORMAL-zone stage with empty queue stays NORMAL until ratio crosses op_threshold."""
        midband = (cfg.saturation_threshold + cfg.over_provisioned_threshold) / 2.0
        result = classify(
            slots_empty_ratio_ewma=midband,
            input_queue_depth=0,
            prev_state=StageState.NORMAL,
            config=cfg,
        )
        assert result is StageState.NORMAL


class TestCrossZoneTransitions:
    """A single cycle may move a stage across multiple zones.

    Hysteresis only applies to the boundary the stage is exiting; a
    sudden swing through the deadband all the way into the opposite
    zone must transition immediately, not be held by hysteresis.
    """

    def test_saturated_prev_can_jump_directly_to_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Sudden idle: SATURATED stage observes ratio in OVER_PROVISIONED zone -> OVER_PROVISIONED.

        The saturation hysteresis (sat_boundary expanded upward) does
        not gate the OVER_PROVISIONED check; the upper zone is reached
        as soon as ratio crosses op_threshold from below.
        """
        result = classify(
            slots_empty_ratio_ewma=0.80,
            input_queue_depth=10,
            prev_state=StageState.SATURATED,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED

    def test_over_provisioned_prev_can_jump_directly_to_saturated(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Sudden burst: OVER_PROVISIONED stage observes ratio in SATURATED zone -> SATURATED.

        The over-provisioned hysteresis (op_boundary lowered) does not
        gate the SATURATED check; the lower zone is reached as soon as
        ratio crosses saturation_threshold from above.
        """
        result = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=10,
            prev_state=StageState.OVER_PROVISIONED,
            config=cfg,
        )
        assert result is StageState.SATURATED

    def test_starved_prev_returns_to_normal_with_no_hysteresis(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """STARVED is not subject to hysteresis -- when input returns the stage drops to NORMAL immediately.

        Unlike OVER_PROVISIONED, STARVED does not lower the
        op_boundary on exit because STARVED is a transient
        upstream-bottleneck state, not a scaled-out state.
        """
        result = classify(
            slots_empty_ratio_ewma=cfg.over_provisioned_threshold - 1e-6,
            input_queue_depth=10,
            prev_state=StageState.STARVED,
            config=cfg,
        )
        assert result is StageState.NORMAL

    def test_starved_prev_with_falling_ratio_classifies_as_saturated_immediately(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """STARVED stage that fills up faster than input arrives drops straight to SATURATED.

        Ensures the STARVED -> SATURATED edge does not get held by
        any spurious hysteresis path.
        """
        result = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=10,
            prev_state=StageState.STARVED,
            config=cfg,
        )
        assert result is StageState.SATURATED
