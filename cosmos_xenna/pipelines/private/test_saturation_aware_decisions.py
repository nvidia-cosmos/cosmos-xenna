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
  aggressive_growth_max_per_cycle = 4
  max_scale_down_fraction_per_cycle = 0.05
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.decisions import (
    compute_delta,
    should_fire_action,
    update_streak,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import GrowthMode, StageState
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
        """Initial StageRuntimeState has streak=0; first cycle in same state -> 1."""
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

    def test_over_provisioned_does_not_fire_below_threshold(self, cfg: SaturationAwareStageConfig) -> None:
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


class TestComputeDeltaCapacityDrivenGrow:
    """SATURATED / SATURATED_CRITICAL scale-up magnitude is the capacity gap."""

    def test_grow_returns_shortfall_to_capacity_target(self, cfg: SaturationAwareStageConfig) -> None:
        """current=4, target=7 -> shortfall +3, below the per-cycle cap of 4."""
        result = compute_delta(
            StageState.SATURATED_CRITICAL,
            GrowthMode.ACQUIRING,
            current_workers=4,
            capacity_target_workers=7,
            config=cfg,
        )
        assert result == 3

    def test_grow_clamped_by_aggressive_growth_max_per_cycle(self, cfg: SaturationAwareStageConfig) -> None:
        """current=4, target=100 -> shortfall 96, clamped to aggressive_growth_max_per_cycle (4)."""
        result = compute_delta(
            StageState.SATURATED_CRITICAL,
            GrowthMode.ACQUIRING,
            current_workers=4,
            capacity_target_workers=100,
            config=cfg,
        )
        assert result == cfg.aggressive_growth_max_per_cycle

    def test_grow_returns_zero_when_at_target(self, cfg: SaturationAwareStageConfig) -> None:
        """current == target -> no further grow."""
        result = compute_delta(
            StageState.SATURATED,
            GrowthMode.ACQUIRING,
            current_workers=8,
            capacity_target_workers=8,
            config=cfg,
        )
        assert result == 0

    def test_grow_returns_zero_when_above_target(self, cfg: SaturationAwareStageConfig) -> None:
        """current > target -> classifier said SATURATED but the sizer disagrees; do not grow."""
        result = compute_delta(
            StageState.SATURATED,
            GrowthMode.TRACKING,
            current_workers=10,
            capacity_target_workers=8,
            config=cfg,
        )
        assert result == 0

    def test_acquiring_and_tracking_produce_identical_magnitude(self, cfg: SaturationAwareStageConfig) -> None:
        """Capacity-driven sizing collapses ACQUIRING and TRACKING to the same delta."""
        acq = compute_delta(
            StageState.SATURATED,
            GrowthMode.ACQUIRING,
            current_workers=4,
            capacity_target_workers=6,
            config=cfg,
        )
        track = compute_delta(
            StageState.SATURATED,
            GrowthMode.TRACKING,
            current_workers=4,
            capacity_target_workers=6,
            config=cfg,
        )
        assert acq == track == 2


class TestComputeDeltaHoldGate:
    """HOLD blocks SATURATED grow but allows SATURATED_CRITICAL grow."""

    def test_hold_blocks_saturated_grow(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + SATURATED -> 0 even when the capacity target asks for more workers."""
        result = compute_delta(
            StageState.SATURATED,
            GrowthMode.HOLD,
            current_workers=4,
            capacity_target_workers=10,
            config=cfg,
        )
        assert result == 0

    def test_hold_allows_critical_grow(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + SATURATED_CRITICAL passes through; the capacity gap drives the magnitude."""
        result = compute_delta(
            StageState.SATURATED_CRITICAL,
            GrowthMode.HOLD,
            current_workers=4,
            capacity_target_workers=6,
            config=cfg,
        )
        assert result == 2


class TestComputeDeltaCapacityDrivenShrink:
    """OVER_PROVISIONED shrink magnitude is the capacity excess, capped by the fraction limit."""

    def test_shrink_returns_excess_below_fraction_cap(self, cfg: SaturationAwareStageConfig) -> None:
        """current=20, target=18 -> excess 2; fraction cap floor(20*0.05)=1; shrink by 1."""
        result = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=20,
            capacity_target_workers=18,
            config=cfg,
        )
        assert result == -1

    def test_shrink_capped_by_fraction(self, cfg: SaturationAwareStageConfig) -> None:
        """current=100, target=10 -> excess 90; fraction cap floor(100*0.05)=5; shrink by -5."""
        result = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=100,
            capacity_target_workers=10,
            config=cfg,
        )
        assert result == -5

    def test_shrink_floor_one_when_excess_present(self, cfg: SaturationAwareStageConfig) -> None:
        """current=10, target=9 -> excess 1; fraction floor=0 -> max(1, 0)=1; -1 (never blocked by rounding)."""
        result = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=10,
            capacity_target_workers=9,
            config=cfg,
        )
        assert result == -1

    def test_shrink_returns_zero_when_at_or_below_target(self, cfg: SaturationAwareStageConfig) -> None:
        """OVER_PROVISIONED but already at the capacity target -> hold; classifier and sizer agree."""
        result = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=10,
            capacity_target_workers=10,
            config=cfg,
        )
        assert result == 0

    def test_growth_mode_is_irrelevant_for_shrink(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING/TRACKING/HOLD all produce the same shrink delta in OVER_PROVISIONED."""
        acq = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.ACQUIRING,
            current_workers=20,
            capacity_target_workers=10,
            config=cfg,
        )
        track = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=20,
            capacity_target_workers=10,
            config=cfg,
        )
        hold = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.HOLD,
            current_workers=20,
            capacity_target_workers=10,
            config=cfg,
        )
        assert acq == track == hold


class TestComputeDeltaColdStartFallback:
    """When the capacity target is unobservable, fall back to discrete +/-1 sizing."""

    def test_saturated_critical_returns_plus_one(self, cfg: SaturationAwareStageConfig) -> None:
        """No D_k yet -> SATURATED_CRITICAL grows by exactly one worker."""
        result = compute_delta(
            StageState.SATURATED_CRITICAL,
            GrowthMode.ACQUIRING,
            current_workers=4,
            capacity_target_workers=None,
            config=cfg,
        )
        assert result == 1

    def test_saturated_returns_plus_one(self, cfg: SaturationAwareStageConfig) -> None:
        """No D_k yet -> SATURATED grows by exactly one worker."""
        result = compute_delta(
            StageState.SATURATED,
            GrowthMode.TRACKING,
            current_workers=4,
            capacity_target_workers=None,
            config=cfg,
        )
        assert result == 1

    def test_over_provisioned_returns_minus_one(self, cfg: SaturationAwareStageConfig) -> None:
        """No D_k yet -> OVER_PROVISIONED shrinks by exactly one worker."""
        result = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=4,
            capacity_target_workers=None,
            config=cfg,
        )
        assert result == -1

    def test_hold_blocks_cold_start_saturated(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + SATURATED with no D_k -> still blocked; the gate applies before the fallback."""
        result = compute_delta(
            StageState.SATURATED,
            GrowthMode.HOLD,
            current_workers=4,
            capacity_target_workers=None,
            config=cfg,
        )
        assert result == 0


class TestComputeDeltaNoActionStates:
    """NORMAL produces zero delta regardless of growth mode and capacity target."""

    def test_normal_returns_zero(self, cfg: SaturationAwareStageConfig) -> None:
        """NORMAL is the no-action zone; capacity target is ignored."""
        result = compute_delta(
            StageState.NORMAL,
            GrowthMode.ACQUIRING,
            current_workers=10,
            capacity_target_workers=20,
            config=cfg,
        )
        assert result == 0


class TestComputeDeltaInputValidation:
    """Defensive validation against programmer error."""

    def test_negative_current_workers_is_rejected(self, cfg: SaturationAwareStageConfig) -> None:
        """Defensive: negative worker count is a programmer bug, not a runtime concern."""
        with pytest.raises(ValueError, match="current_workers must be >= 0"):
            compute_delta(
                StageState.SATURATED,
                GrowthMode.ACQUIRING,
                current_workers=-1,
                capacity_target_workers=4,
                config=cfg,
            )
