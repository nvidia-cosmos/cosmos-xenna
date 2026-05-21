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

"""Backlog-time pressure demotion behaviour for the saturation-aware classifier.

These tests pin the slot-pin-gate-then-pressure-demotion contract that
``classify()`` introduces alongside the existing slot-ratio thresholds.
Every demotion branch (CRITICAL, SATURATED, OVER_PROVISIONED) plus the
missing-pressure fallback and the escape-hatch flag have a focused test
so a refactor cannot silently weaken the gate ordering.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.classifier import classify
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Pin slot-ratio AND pressure thresholds to anchor every test's math.

    Slot ratios: ``activation=0.05`` < ``saturation=0.15`` <
    ``over_provisioned=0.50``. Pressure thresholds: ``normal=0.30`` <
    ``saturation=1.0`` < ``critical=2.0``. These match the documented
    defaults so the test math stays close to the production calibration.
    """
    return SaturationAwareStageConfig(
        saturation_threshold=0.15,
        activation_threshold=0.05,
        pressure_normal_threshold=0.3,
        pressure_saturation_threshold=1.0,
        pressure_critical_threshold=2.0,
    )


class TestSaturatedCriticalPressureDemotion:
    """The CRITICAL slot-pin gate is gated by ``pressure_critical_threshold``."""

    def test_low_pressure_demotes_critical_slot_pin(self, cfg: SaturationAwareStageConfig) -> None:
        """Slot pin in CRITICAL band but pressure < critical threshold -> NORMAL via fall-through."""
        # Slot pin: activation = 0.05, ratio = 0.0 << activation.
        # Pressure 0.0 < pressure_critical (2.0) AND < pressure_saturation
        # (1.0) AND < pressure_normal (0.3), so SATURATED gate also
        # demotes -> NORMAL.
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            pressure_ewma=0.0,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.NORMAL

    def test_high_pressure_fires_critical(self, cfg: SaturationAwareStageConfig) -> None:
        """Slot pin AND pressure > critical threshold -> CRITICAL."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            pressure_ewma=2.5,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_missing_pressure_falls_back_to_critical(self, cfg: SaturationAwareStageConfig) -> None:
        """When ``pressure_ewma is None`` the slot-pin contract is preserved."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            pressure_ewma=None,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_critical_pressure_at_threshold_does_not_fire(self, cfg: SaturationAwareStageConfig) -> None:
        """The CRITICAL pressure gate is strict (``>`` not ``>=``)."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            pressure_ewma=2.0,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        # pressure == critical (2.0) -> does NOT exceed, fall through to SATURATED gate.
        # pressure (2.0) > saturation (1.0) -> SATURATED fires.
        assert result is StageState.SATURATED


class TestSaturatedPressureDemotion:
    """The SATURATED slot-pin gate is gated by ``pressure_saturation_threshold``."""

    def test_low_pressure_demotes_saturated_to_normal(self, cfg: SaturationAwareStageConfig) -> None:
        """Slot pin in SATURATED band but pressure < saturation threshold -> NORMAL."""
        result = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=10,
            pressure_ewma=0.5,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.NORMAL

    def test_high_pressure_fires_saturated(self, cfg: SaturationAwareStageConfig) -> None:
        """Slot pin AND pressure > saturation threshold -> SATURATED."""
        result = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=10,
            pressure_ewma=1.5,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.SATURATED

    def test_missing_pressure_falls_back_to_saturated(self, cfg: SaturationAwareStageConfig) -> None:
        """``pressure_ewma=None`` preserves slot-only behaviour for SATURATED."""
        result = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=10,
            pressure_ewma=None,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.SATURATED


class TestOverProvisionedPressureDemotion:
    """The OVER_PROVISIONED slot-idle branch is demoted to NORMAL when pressure is high."""

    def test_low_pressure_keeps_over_provisioned(self, cfg: SaturationAwareStageConfig) -> None:
        """Idle slots + queue > 0 + pressure <= normal threshold -> OVER_PROVISIONED."""
        # ratio = 0.6 > over_provisioned = 0.50; queue > 0; pressure (0.1) <= normal (0.3).
        result = classify(
            slots_empty_ratio_ewma=0.6,
            input_queue_depth=10,
            pressure_ewma=0.1,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED

    def test_high_pressure_demotes_to_normal(self, cfg: SaturationAwareStageConfig) -> None:
        """Idle slots + queue > 0 + pressure > normal threshold -> NORMAL (downstream stuck)."""
        result = classify(
            slots_empty_ratio_ewma=0.6,
            input_queue_depth=10,
            pressure_ewma=0.5,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.NORMAL

    def test_empty_queue_stays_starved_regardless_of_pressure(self, cfg: SaturationAwareStageConfig) -> None:
        """STARVED is structural (queue empty); pressure is irrelevant."""
        result = classify(
            slots_empty_ratio_ewma=0.6,
            input_queue_depth=0,
            pressure_ewma=2.5,  # would otherwise trigger every demotion gate
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.STARVED

    def test_missing_pressure_keeps_over_provisioned(self, cfg: SaturationAwareStageConfig) -> None:
        """``pressure_ewma=None`` preserves the legacy OVER_PROVISIONED behaviour."""
        result = classify(
            slots_empty_ratio_ewma=0.6,
            input_queue_depth=10,
            pressure_ewma=None,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg,
        )
        assert result is StageState.OVER_PROVISIONED


class TestEnableFlagOff:
    """When ``enable_backlog_time_classifier=False`` the classifier reverts to slot-only."""

    @pytest.fixture
    def cfg_disabled(self) -> SaturationAwareStageConfig:
        """Same thresholds as the active fixture but with the pressure gate disabled."""
        return SaturationAwareStageConfig(
            saturation_threshold=0.15,
            activation_threshold=0.05,
            enable_backlog_time_classifier=False,
        )

    def test_disabled_critical_ignores_low_pressure(self, cfg_disabled: SaturationAwareStageConfig) -> None:
        """Slot-pin CRITICAL fires on slot ratio alone -- pressure is ignored."""
        result = classify(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            pressure_ewma=0.0,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg_disabled,
        )
        assert result is StageState.SATURATED_CRITICAL

    def test_disabled_saturated_ignores_low_pressure(self, cfg_disabled: SaturationAwareStageConfig) -> None:
        """Slot-pin SATURATED fires on slot ratio alone -- low pressure does not demote."""
        result = classify(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=10,
            pressure_ewma=0.0,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg_disabled,
        )
        assert result is StageState.SATURATED

    def test_disabled_over_provisioned_ignores_high_pressure(self, cfg_disabled: SaturationAwareStageConfig) -> None:
        """Idle-slot OVER_PROVISIONED stays put -- high pressure does not demote."""
        result = classify(
            slots_empty_ratio_ewma=0.6,
            input_queue_depth=10,
            pressure_ewma=2.5,
            prev_state=StageState.NORMAL,
            saturation_threshold=0.15,
            activation_threshold=0.05,
            config=cfg_disabled,
        )
        assert result is StageState.OVER_PROVISIONED
