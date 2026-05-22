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

"""Behaviour tests for the per-cycle decision helpers.

Pins the contract of ``update_streak``, ``should_fire_action``, and
``compute_delta``. Each test verifies one specific behaviour so a
future tuning change surfaces as a precise test failure.

Default per-stage thresholds at the time of writing:
  saturated_critical_streak_min_cycles = 1
  saturated_streak_min_cycles = 2
  over_provisioned_streak_min_cycles = 30
  acquiring_critical_growth_factor = 0.5
  acquiring_saturated_growth_factor = 0.25
  tracking_critical_growth_count = 2
  tracking_saturated_growth_count = 1
  hold_critical_growth_count = 1
  hold_saturated_growth_count = 0
  aggressive_growth_max_per_cycle = 4
  max_scale_down_fraction_per_cycle = 0.05
"""

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.decisions import (
    compute_delta,
    should_fire_action,
    update_streak,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Default per-stage config; thresholds and counts documented in the module docstring."""
    return SaturationAwareStageConfig()


class TestUpdateStreak:
    """Streak increments while state holds; resets to 1 on transition."""

    def test_same_state_increments_streak(self) -> None:
        """Holding the same state increments the prior streak by exactly one."""
        result = update_streak(StageState.SATURATED, prev_streak=4, new_state=StageState.SATURATED)
        assert result == 5

    def test_state_transition_resets_to_one(self) -> None:
        """Any transition wipes the prior streak; the new state's streak starts at 1."""
        result = update_streak(StageState.SATURATED, prev_streak=10, new_state=StageState.NORMAL)
        assert result == 1

    def test_first_cycle_in_initial_state_returns_one(self) -> None:
        """Initial _StageRuntimeState has streak=0; first cycle in same state -> 1."""
        result = update_streak(StageState.NORMAL, prev_streak=0, new_state=StageState.NORMAL)
        assert result == 1

    def test_negative_prev_streak_is_rejected(self) -> None:
        """Defensive validation -- negative streak indicates a programmer bug upstream."""
        with pytest.raises(ValueError, match="prev_streak must be >= 0"):
            update_streak(StageState.NORMAL, prev_streak=-1, new_state=StageState.NORMAL)

    def test_cold_start_with_state_transition_returns_one(self) -> None:
        """Cold start that flips state on the first cycle -- streak resets to 1, not zero or two."""
        result = update_streak(StageState.NORMAL, prev_streak=0, new_state=StageState.SATURATED)
        assert result == 1


class TestShouldFireActionScaleUp:
    """Aggressive-side thresholds (CRITICAL = 1, SATURATED = 2 by default)."""

    def test_critical_fires_on_first_cycle(self, cfg: SaturationAwareStageConfig) -> None:
        """SATURATED_CRITICAL fires immediately (default streak threshold = 1)."""
        assert should_fire_action(StageState.SATURATED_CRITICAL, streak=1, config=cfg) is True

    def test_saturated_does_not_fire_on_first_cycle(self, cfg: SaturationAwareStageConfig) -> None:
        """SATURATED with streak 1 < min 2 -> not yet firing."""
        assert should_fire_action(StageState.SATURATED, streak=1, config=cfg) is False

    def test_saturated_fires_at_threshold(self, cfg: SaturationAwareStageConfig) -> None:
        """SATURATED at streak == min cycles is the trigger boundary."""
        assert should_fire_action(StageState.SATURATED, streak=2, config=cfg) is True

    def test_saturated_continues_firing_above_threshold(self, cfg: SaturationAwareStageConfig) -> None:
        """A streak past the threshold does not drop firing; predicate is monotone in streak."""
        assert should_fire_action(StageState.SATURATED, streak=10, config=cfg) is True


class TestShouldFireActionScaleDown:
    """Conservative scale-down threshold (OVER_PROVISIONED = 30 by default)."""

    def test_over_provisioned_does_not_fire_below_threshold(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """OVER_PROVISIONED at streak 29 < min 30 -> hold; protects against premature shrink."""
        assert should_fire_action(StageState.OVER_PROVISIONED, streak=29, config=cfg) is False

    def test_over_provisioned_fires_at_threshold(self, cfg: SaturationAwareStageConfig) -> None:
        """OVER_PROVISIONED at streak == min cycles is the trigger boundary."""
        assert should_fire_action(StageState.OVER_PROVISIONED, streak=30, config=cfg) is True


class TestShouldFireActionNormal:
    """NORMAL is the no-action zone; predicate is False regardless of streak."""

    def test_normal_never_fires(self, cfg: SaturationAwareStageConfig) -> None:
        """NORMAL at any streak length -> no action."""
        assert should_fire_action(StageState.NORMAL, streak=1, config=cfg) is False
        assert should_fire_action(StageState.NORMAL, streak=10**6, config=cfg) is False


class TestComputeDeltaCriticalScaleUp:
    """SATURATED_CRITICAL -> burst scale-up magnitude per growth mode."""

    def test_acquiring_uses_multiplicative_factor(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING mode: delta = ceil(0.5 * current_workers); current=4 -> +2."""
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.ACQUIRING, current_workers=4, config=cfg)
        assert result == 2

    def test_acquiring_uses_ceil_for_fractional_results(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING with current=3 -> ceil(0.5 * 3) = ceil(1.5) = 2."""
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.ACQUIRING, current_workers=3, config=cfg)
        assert result == 2

    def test_acquiring_clamped_by_aggressive_max_per_cycle(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING with current=100 -> ceil(50) capped at aggressive_growth_max_per_cycle (4)."""
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.ACQUIRING, current_workers=100, config=cfg)
        assert result == cfg.aggressive_growth_max_per_cycle

    def test_tracking_uses_absolute_count(self, cfg: SaturationAwareStageConfig) -> None:
        """TRACKING mode uses tracking_critical_growth_count regardless of current_workers."""
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.TRACKING, current_workers=100, config=cfg)
        assert result == cfg.tracking_critical_growth_count

    def test_hold_emits_minimal_burst_response(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD mode allows a minimal burst response on CRITICAL (default = 1)."""
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.HOLD, current_workers=10, config=cfg)
        assert result == cfg.hold_critical_growth_count

    def test_acquiring_with_zero_workers_returns_zero(self, cfg: SaturationAwareStageConfig) -> None:
        """current_workers=0 -> ceil(0.5 * 0) = 0; the worker-floor step independently bootstraps to one.

        Programmatic edge: the classifier returns SATURATED_CRITICAL for a
        zero-actor stage (slots_empty_ratio = 0), but compute_delta is the
        algorithm's intent and intentionally produces zero -- the implicit
        one-worker floor bootstrap is owned by a separate phase.
        """
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.ACQUIRING, current_workers=0, config=cfg)
        assert result == 0


class TestComputeDeltaSaturatedScaleUp:
    """SATURATED -> additive scale-up magnitude per growth mode."""

    def test_acquiring_uses_multiplicative_factor(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING mode: delta = ceil(0.25 * current_workers); current=4 -> +1."""
        result = compute_delta(StageState.SATURATED, GrowthMode.ACQUIRING, current_workers=4, config=cfg)
        assert result == 1

    def test_acquiring_clamped_by_aggressive_max_per_cycle(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING with current=100 -> ceil(25) capped at aggressive_growth_max_per_cycle (4)."""
        result = compute_delta(StageState.SATURATED, GrowthMode.ACQUIRING, current_workers=100, config=cfg)
        assert result == cfg.aggressive_growth_max_per_cycle

    def test_tracking_uses_absolute_count(self, cfg: SaturationAwareStageConfig) -> None:
        """TRACKING mode uses tracking_saturated_growth_count regardless of current_workers."""
        result = compute_delta(StageState.SATURATED, GrowthMode.TRACKING, current_workers=100, config=cfg)
        assert result == cfg.tracking_saturated_growth_count

    def test_hold_suppresses_non_critical_growth(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD mode blocks non-critical growth (default hold_saturated_growth_count = 0)."""
        result = compute_delta(StageState.SATURATED, GrowthMode.HOLD, current_workers=10, config=cfg)
        assert result == 0

    def test_acquiring_with_zero_workers_returns_zero(self, cfg: SaturationAwareStageConfig) -> None:
        """current_workers=0 in ACQUIRING -> ceil(0.25 * 0) = 0; mirror of the CRITICAL bootstrap edge."""
        result = compute_delta(StageState.SATURATED, GrowthMode.ACQUIRING, current_workers=0, config=cfg)
        assert result == 0


class TestComputeDeltaOverProvisionedScaleDown:
    """OVER_PROVISIONED -> shrink magnitude bounded by max_scale_down_fraction_per_cycle."""

    def test_large_fleet_shrinks_by_fraction(self, cfg: SaturationAwareStageConfig) -> None:
        """current=100, fraction=0.05 -> -5 (= floor(5.0))."""
        result = compute_delta(StageState.OVER_PROVISIONED, GrowthMode.TRACKING, current_workers=100, config=cfg)
        assert result == -5

    def test_small_fleet_shrinks_by_at_least_one(self, cfg: SaturationAwareStageConfig) -> None:
        """current=10, fraction=0.05 -> floor(0.5)=0, max(1, 0)=1 -> -1; never blocked by rounding."""
        result = compute_delta(StageState.OVER_PROVISIONED, GrowthMode.TRACKING, current_workers=10, config=cfg)
        assert result == -1

    def test_growth_mode_is_irrelevant_for_shrink(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING/TRACKING/HOLD all produce the same shrink delta in OVER_PROVISIONED."""
        acq = compute_delta(StageState.OVER_PROVISIONED, GrowthMode.ACQUIRING, current_workers=20, config=cfg)
        track = compute_delta(StageState.OVER_PROVISIONED, GrowthMode.TRACKING, current_workers=20, config=cfg)
        hold = compute_delta(StageState.OVER_PROVISIONED, GrowthMode.HOLD, current_workers=20, config=cfg)
        assert acq == track == hold

    def test_at_implicit_floor_returns_negative_one_for_caller_to_clamp(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """current_workers=1 -> -1 (intent); caller clamps to 0 to respect one-worker floor.

        compute_delta is the algorithm's intent, not the feasibility
        check. The caller is responsible for refusing to apply the
        delta when it would violate min_workers or the implicit
        one-worker floor.
        """
        result = compute_delta(StageState.OVER_PROVISIONED, GrowthMode.TRACKING, current_workers=1, config=cfg)
        assert result == -1


class TestComputeDeltaCapAppliesUniformly:
    """``aggressive_growth_max_per_cycle`` is the hard ceiling on every scale-up path.

    The cap protects the cluster from an operator who configures a
    large absolute count or a large multiplicative factor without
    realising the per-cycle blast radius. Each growth-mode path
    (ACQUIRING / TRACKING / HOLD) must be subject to the same cap so
    the design intent is preserved across config variations.
    """

    def test_tracking_critical_count_clamped_when_exceeds_cap(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A custom tracking_critical_growth_count above the cap is clamped to the cap."""
        cfg_loud = attrs.evolve(cfg, tracking_critical_growth_count=10)
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.TRACKING, current_workers=4, config=cfg_loud)
        assert result == cfg_loud.aggressive_growth_max_per_cycle

    def test_tracking_saturated_count_clamped_when_exceeds_cap(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A custom tracking_saturated_growth_count above the cap is clamped to the cap."""
        cfg_loud = attrs.evolve(cfg, tracking_saturated_growth_count=10)
        result = compute_delta(StageState.SATURATED, GrowthMode.TRACKING, current_workers=4, config=cfg_loud)
        assert result == cfg_loud.aggressive_growth_max_per_cycle

    def test_hold_critical_count_clamped_when_exceeds_cap(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A custom hold_critical_growth_count above the cap is clamped to the cap."""
        cfg_loud = attrs.evolve(cfg, hold_critical_growth_count=10)
        result = compute_delta(StageState.SATURATED_CRITICAL, GrowthMode.HOLD, current_workers=4, config=cfg_loud)
        assert result == cfg_loud.aggressive_growth_max_per_cycle


class TestComputeDeltaNoActionStates:
    """NORMAL produces zero delta regardless of growth mode."""

    def test_normal_returns_zero(self, cfg: SaturationAwareStageConfig) -> None:
        """NORMAL is the no-action zone."""
        result = compute_delta(StageState.NORMAL, GrowthMode.ACQUIRING, current_workers=10, config=cfg)
        assert result == 0


class TestComputeDeltaInputValidation:
    """Defensive validation against programmer error."""

    def test_negative_current_workers_is_rejected(self, cfg: SaturationAwareStageConfig) -> None:
        """Defensive: negative worker count is a programmer bug, not a runtime concern."""
        with pytest.raises(ValueError, match="current_workers must be >= 0"):
            compute_delta(StageState.SATURATED, GrowthMode.ACQUIRING, current_workers=-1, config=cfg)
