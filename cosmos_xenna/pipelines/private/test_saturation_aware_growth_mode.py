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

"""Behaviour tests for the slow-start growth-mode state machine.

Each test pins exactly one transition or one boundary case so a
future change to the growth-mode rules surfaces as a precise test
failure.

Default at the time of writing:
  stabilization_window_cycles_down = 30
"""

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.growth_mode import compute_growth_mode_transition
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Default per-stage config; window is 30 cycles by default."""
    return SaturationAwareStageConfig()


class TestAcquiringTransitions:
    """ACQUIRING is the initial mode; only first executed shrink moves it forward."""

    def test_first_shrink_transitions_to_tracking(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING + delta<0 -> (TRACKING, 1) -- ceiling discovered."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.ACQUIRING,
            prev_streak=5,
            delta_executed=-1,
            config=cfg,
        )
        assert result == (GrowthMode.TRACKING, 1)

    def test_no_action_holds_acquiring_and_increments_streak(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING + delta=0 -> (ACQUIRING, prev_streak + 1) -- still in slow-start."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.ACQUIRING,
            prev_streak=5,
            delta_executed=0,
            config=cfg,
        )
        assert result == (GrowthMode.ACQUIRING, 6)

    def test_grow_holds_acquiring_and_increments_streak(self, cfg: SaturationAwareStageConfig) -> None:
        """ACQUIRING + delta>0 -> (ACQUIRING, prev_streak + 1) -- multiplicative growth, mode unchanged."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.ACQUIRING,
            prev_streak=3,
            delta_executed=2,
            config=cfg,
        )
        assert result == (GrowthMode.ACQUIRING, 4)


class TestTrackingTransitions:
    """TRACKING is the steady-state after ceiling discovery; shrink moves it to HOLD."""

    def test_shrink_transitions_to_hold(self, cfg: SaturationAwareStageConfig) -> None:
        """TRACKING + delta<0 -> (HOLD, 1) -- post-shrink stabilization begins."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.TRACKING,
            prev_streak=10,
            delta_executed=-1,
            config=cfg,
        )
        assert result == (GrowthMode.HOLD, 1)

    def test_no_action_holds_tracking_and_increments_streak(self, cfg: SaturationAwareStageConfig) -> None:
        """TRACKING + delta=0 -> (TRACKING, prev_streak + 1)."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.TRACKING,
            prev_streak=10,
            delta_executed=0,
            config=cfg,
        )
        assert result == (GrowthMode.TRACKING, 11)

    def test_grow_holds_tracking_and_increments_streak(self, cfg: SaturationAwareStageConfig) -> None:
        """TRACKING + delta>0 -> (TRACKING, prev_streak + 1) -- additive growth, mode unchanged."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.TRACKING,
            prev_streak=10,
            delta_executed=1,
            config=cfg,
        )
        assert result == (GrowthMode.TRACKING, 11)


class TestHoldTransitions:
    """HOLD is the post-shrink stabilization mode; exits on timer expiry or restarts on re-shrink."""

    def test_re_shrink_restarts_hold_timer(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + delta<0 -> (HOLD, 1) -- timer restarts so the stabilization window covers the new shrink."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=20,
            delta_executed=-1,
            config=cfg,
        )
        assert result == (GrowthMode.HOLD, 1)

    def test_no_action_below_window_holds_and_increments(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + delta=0 + streak < window -> (HOLD, prev_streak + 1)."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=10,
            delta_executed=0,
            config=cfg,
        )
        assert result == (GrowthMode.HOLD, 11)

    def test_no_action_at_window_minus_one_holds(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + delta=0 + streak == window-1 -> (HOLD, window) -- one cycle before timer fires."""
        boundary = cfg.stabilization_window_cycles_down - 1
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=boundary,
            delta_executed=0,
            config=cfg,
        )
        assert result == (GrowthMode.HOLD, cfg.stabilization_window_cycles_down)

    def test_no_action_at_window_transitions_to_tracking(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + delta=0 + streak == window -> (TRACKING, 1) -- timer expires."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=cfg.stabilization_window_cycles_down,
            delta_executed=0,
            config=cfg,
        )
        assert result == (GrowthMode.TRACKING, 1)

    def test_burst_growth_in_hold_does_not_transition(self, cfg: SaturationAwareStageConfig) -> None:
        """HOLD + delta>0 (burst-only) -> (HOLD, prev_streak + 1) -- burst growth keeps timer counting."""
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=10,
            delta_executed=1,
            config=cfg,
        )
        assert result == (GrowthMode.HOLD, 11)

    def test_shrink_at_timer_boundary_takes_precedence_over_timer_expiry(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """HOLD + prev_streak == window + delta<0 -> (HOLD, 1) -- shrink path wins, timer restarts.

        Both branches of the function would fire on this input. The
        ``delta_executed < 0`` check comes first in the function body
        precisely so the timer-expiry transition cannot mask a fresh
        shrink event; the new shrink resets the stabilization window
        instead of leaking into a TRACKING state that would let the
        next saturation event grow immediately.
        """
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=cfg.stabilization_window_cycles_down,
            delta_executed=-1,
            config=cfg,
        )
        assert result == (GrowthMode.HOLD, 1)

    def test_streak_far_above_window_still_exits_to_tracking(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """HOLD + delta=0 + streak >> window -> (TRACKING, 1) -- ``>=`` not ``==``.

        Defensive pin against a future refactor that accidentally
        narrows the timer-expiry condition to exact equality. If a
        stage somehow accumulates a streak far past the window
        (programming bug, or window reduced at runtime), the function
        must still exit HOLD on the next no-action cycle.
        """
        large_streak = cfg.stabilization_window_cycles_down * 4
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=large_streak,
            delta_executed=0,
            config=cfg,
        )
        assert result == (GrowthMode.TRACKING, 1)


class TestSmallestValidWindow:
    """Window = 1 is the smallest valid value -- HOLD exits on the very next no-action cycle."""

    def test_window_of_one_exits_immediately_after_entry_cycle(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """With window=1, HOLD lasts a single cycle: enter on cycle N, exit on cycle N+1.

        The cross-field validator on ``SaturationAwareConfig`` requires
        ``stabilization_window_cycles_down > stabilization_window_cycles_up``;
        with the up window default of 1, the smallest legal down window
        is 2. We therefore exercise window=2 here and verify the
        boundary semantics of the smallest legal value, not a
        pathological window=1.
        """
        small_window_cfg = attrs.evolve(cfg, stabilization_window_cycles_down=2)
        # Entry cycle: prev_streak=1.
        # Cycle 2 with prev_streak=1 + delta=0: not yet at window -> stay.
        cycle_two = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD, prev_streak=1, delta_executed=0, config=small_window_cfg
        )
        assert cycle_two == (GrowthMode.HOLD, 2)
        # Cycle 3 with prev_streak=2 + delta=0: window reached -> exit.
        cycle_three = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD, prev_streak=2, delta_executed=0, config=small_window_cfg
        )
        assert cycle_three == (GrowthMode.TRACKING, 1)


class TestFullLifecycleTrace:
    """Multi-cycle simulation pinning the state machine's sequencing.

    Single-step tests pin each transition individually; this trace
    walks a realistic stage from initial ACQUIRING through ceiling
    discovery, additive tracking, post-shrink stabilization, and
    timer expiry. Catches any transition rule that succeeds in
    isolation but composes incorrectly.
    """

    def test_acquiring_through_tracking_through_hold_through_tracking(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Full lifecycle: 4 modes visited in correct order with correct streak resets."""
        mode = GrowthMode.ACQUIRING
        streak = 1

        # ---- Phase 1: ACQUIRING with no signals (5 cycles).
        for _ in range(5):
            mode, streak = compute_growth_mode_transition(
                prev_mode=mode, prev_streak=streak, delta_executed=0, config=cfg
            )
        assert mode is GrowthMode.ACQUIRING
        assert streak == 6  # 1 + 5 increments

        # ---- Phase 2: ACQUIRING grows on saturation (3 successful adds).
        for _ in range(3):
            mode, streak = compute_growth_mode_transition(
                prev_mode=mode, prev_streak=streak, delta_executed=2, config=cfg
            )
        assert mode is GrowthMode.ACQUIRING
        assert streak == 9  # grows do not transition; streak keeps incrementing

        # ---- Phase 3: First shrink -- ceiling discovered.
        mode, streak = compute_growth_mode_transition(prev_mode=mode, prev_streak=streak, delta_executed=-1, config=cfg)
        assert mode is GrowthMode.TRACKING
        assert streak == 1

        # ---- Phase 4: TRACKING with steady additive growth (4 cycles).
        for _ in range(4):
            mode, streak = compute_growth_mode_transition(
                prev_mode=mode, prev_streak=streak, delta_executed=1, config=cfg
            )
        assert mode is GrowthMode.TRACKING
        assert streak == 5

        # ---- Phase 5: Second shrink -- enter HOLD.
        mode, streak = compute_growth_mode_transition(prev_mode=mode, prev_streak=streak, delta_executed=-1, config=cfg)
        assert mode is GrowthMode.HOLD
        assert streak == 1

        # ---- Phase 6: HOLD ticks down (window-1 quiet cycles).
        for _ in range(cfg.stabilization_window_cycles_down - 1):
            mode, streak = compute_growth_mode_transition(
                prev_mode=mode, prev_streak=streak, delta_executed=0, config=cfg
            )
        assert mode is GrowthMode.HOLD
        assert streak == cfg.stabilization_window_cycles_down

        # ---- Phase 7: One more quiet cycle -- timer fires, exit to TRACKING.
        mode, streak = compute_growth_mode_transition(prev_mode=mode, prev_streak=streak, delta_executed=0, config=cfg)
        assert mode is GrowthMode.TRACKING
        assert streak == 1

    def test_re_shrink_during_hold_resets_full_window(self, cfg: SaturationAwareStageConfig) -> None:
        """Trace: enter HOLD, sit half-way, re-shrink, then verify the FULL window is required again.

        Pins that a re-shrink does not leak the existing timer count
        forward; the second shrink demands a new full stabilization
        window before HOLD can exit.
        """
        # Enter HOLD via TRACKING + shrink.
        mode, streak = compute_growth_mode_transition(
            prev_mode=GrowthMode.TRACKING, prev_streak=10, delta_executed=-1, config=cfg
        )
        assert mode is GrowthMode.HOLD
        assert streak == 1

        # Sit half-way through window with no action.
        half_window = cfg.stabilization_window_cycles_down // 2
        for _ in range(half_window - 1):
            mode, streak = compute_growth_mode_transition(
                prev_mode=mode, prev_streak=streak, delta_executed=0, config=cfg
            )
        assert mode is GrowthMode.HOLD
        assert streak == half_window

        # Re-shrink: streak must reset to 1 even though mode does not change.
        mode, streak = compute_growth_mode_transition(prev_mode=mode, prev_streak=streak, delta_executed=-1, config=cfg)
        assert mode is GrowthMode.HOLD
        assert streak == 1

        # Quiet cycles: still need the FULL window (not what's left of the original).
        for _ in range(cfg.stabilization_window_cycles_down - 1):
            mode, streak = compute_growth_mode_transition(
                prev_mode=mode, prev_streak=streak, delta_executed=0, config=cfg
            )
        assert mode is GrowthMode.HOLD
        assert streak == cfg.stabilization_window_cycles_down

        # One more quiet cycle: timer fires.
        mode, streak = compute_growth_mode_transition(prev_mode=mode, prev_streak=streak, delta_executed=0, config=cfg)
        assert mode is GrowthMode.TRACKING
        assert streak == 1


class TestStreakSemantics:
    """Streak resets only on shrink event or mode transition; otherwise increments."""

    def test_streak_increments_when_mode_unchanged_and_no_shrink(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Both delta=0 and delta>0 path through the increment branch."""
        no_action = compute_growth_mode_transition(
            prev_mode=GrowthMode.TRACKING, prev_streak=5, delta_executed=0, config=cfg
        )
        grow = compute_growth_mode_transition(
            prev_mode=GrowthMode.TRACKING, prev_streak=5, delta_executed=2, config=cfg
        )
        assert no_action == (GrowthMode.TRACKING, 6)
        assert grow == (GrowthMode.TRACKING, 6)

    def test_streak_resets_on_any_mode_transition(self, cfg: SaturationAwareStageConfig) -> None:
        """Whether ACQUIRING->TRACKING or HOLD->TRACKING, streak is 1."""
        acq_to_track = compute_growth_mode_transition(
            prev_mode=GrowthMode.ACQUIRING, prev_streak=20, delta_executed=-1, config=cfg
        )
        hold_to_track = compute_growth_mode_transition(
            prev_mode=GrowthMode.HOLD,
            prev_streak=cfg.stabilization_window_cycles_down,
            delta_executed=0,
            config=cfg,
        )
        assert acq_to_track[1] == 1
        assert hold_to_track[1] == 1


class TestInputValidation:
    """Defensive validation against programmer error."""

    def test_negative_prev_streak_is_rejected(self, cfg: SaturationAwareStageConfig) -> None:
        """Negative prev_streak indicates a programmer bug upstream; zero is the cold-start sentinel."""
        with pytest.raises(ValueError, match="prev_streak must be >= 0"):
            compute_growth_mode_transition(prev_mode=GrowthMode.ACQUIRING, prev_streak=-1, delta_executed=0, config=cfg)

    def test_zero_prev_streak_is_accepted_as_cold_start(self, cfg: SaturationAwareStageConfig) -> None:
        """Cold-start: ``_StageRuntimeState.growth_streak`` defaults to 0 before the first cycle.

        The first autoscale call must not raise on this initial value;
        the fall-through path increments to 1 cleanly.
        """
        result = compute_growth_mode_transition(
            prev_mode=GrowthMode.ACQUIRING, prev_streak=0, delta_executed=0, config=cfg
        )
        assert result == (GrowthMode.ACQUIRING, 1)
