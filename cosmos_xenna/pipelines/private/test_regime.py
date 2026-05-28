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

"""Behaviour tests for the Halfin-Whitt regime detector.

Pin the per-cycle signal computation, the asymmetric hysteresis on
the entry / exit transitions, and the no-signal defensive guard so a
future tweak to any branch surfaces as a precise failure.
"""

import math

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.regime.regime import (
    Regime,
    RegimeDetectorState,
    compute_regime_signal,
    update_regime_state,
)


class TestComputeRegimeSignal:
    """Per-cycle signal computation: threshold, fraction, availability."""

    def test_threshold_matches_one_over_sqrt_workers(self) -> None:
        """``threshold = 1 / sqrt(total_workers)`` per Halfin-Whitt."""
        signal = compute_regime_signal(total_workers=100, total_used_slots=80, total_empty_slots=20)
        assert signal.threshold == pytest.approx(1.0 / math.sqrt(100))

    def test_idle_fraction_is_empty_over_total_slots(self) -> None:
        """``cluster_idle_fraction = empty / (used + empty)``."""
        signal = compute_regime_signal(total_workers=10, total_used_slots=80, total_empty_slots=20)
        assert signal.cluster_idle_fraction == pytest.approx(0.20)

    def test_idle_below_threshold_marks_signal_available(self) -> None:
        """Sustained busy cluster -> idle fraction < threshold; signal available."""
        signal = compute_regime_signal(total_workers=100, total_used_slots=99, total_empty_slots=1)
        assert signal.cluster_idle_fraction < signal.threshold
        assert signal.signal_available is True

    def test_idle_at_or_above_threshold_marks_signal_available(self) -> None:
        """Comfortable headroom -> idle fraction >= threshold; signal available."""
        signal = compute_regime_signal(total_workers=100, total_used_slots=50, total_empty_slots=50)
        assert signal.cluster_idle_fraction >= signal.threshold
        assert signal.signal_available is True

    def test_zero_workers_defaults_threshold_to_one(self) -> None:
        """Edge case: no workers degenerates the threshold to 1.0."""
        signal = compute_regime_signal(total_workers=0, total_used_slots=0, total_empty_slots=0)
        assert signal.threshold == 1.0
        assert signal.signal_available is False

    def test_one_worker_defaults_threshold_to_one(self) -> None:
        """``total_workers=1`` is degenerate: threshold = 1 / sqrt(1) = 1.0."""
        signal = compute_regime_signal(total_workers=1, total_used_slots=1, total_empty_slots=0)
        assert signal.threshold == 1.0

    def test_no_slot_signals_marks_signal_unavailable(self) -> None:
        """Used + empty == 0 -> signal_available False."""
        signal = compute_regime_signal(total_workers=64, total_used_slots=0, total_empty_slots=0)
        assert signal.signal_available is False
        assert signal.cluster_idle_fraction == 0.0

    def test_signal_available_with_only_empty_slots(self) -> None:
        """All slots empty -> idle fraction = 1.0 >= threshold."""
        signal = compute_regime_signal(total_workers=100, total_used_slots=0, total_empty_slots=200)
        assert signal.signal_available is True
        assert signal.cluster_idle_fraction == 1.0


class TestUpdateRegimeStateValidation:
    """Bad inputs to ``update_regime_state`` raise immediately."""

    def test_streak_cycles_below_one_raises(self) -> None:
        """``streak_cycles=0`` would always commit on cycle one; reject."""
        state = RegimeDetectorState()
        signal = compute_regime_signal(total_workers=100, total_used_slots=99, total_empty_slots=1)
        with pytest.raises(ValueError, match="streak_cycles must be >= 1"):
            update_regime_state(state, signal, streak_cycles=0)

    def test_exit_band_at_one_raises(self) -> None:
        """Exit band must be strictly > 1.0 so it is wider than entry threshold."""
        state = RegimeDetectorState()
        signal = compute_regime_signal(total_workers=100, total_used_slots=99, total_empty_slots=1)
        with pytest.raises(ValueError, match="exit_band_multiplier must be > 1.0"):
            update_regime_state(state, signal, streak_cycles=3, exit_band_multiplier=1.0)


class TestEnterSuperHwHysteresis:
    """Enter ``SUPER_HALFIN_WHITT`` only after the streak completes."""

    def test_single_cycle_below_threshold_does_not_transition(self) -> None:
        """One observation below threshold is below the streak floor."""
        state = RegimeDetectorState()
        signal = compute_regime_signal(total_workers=100, total_used_slots=99, total_empty_slots=1)
        transitioned = update_regime_state(state, signal, streak_cycles=3)
        assert transitioned is False
        assert state.current_regime is Regime.SUB_HALFIN_WHITT
        assert state.streak == 1

    def test_third_consecutive_below_threshold_transitions(self) -> None:
        """Three cycles below threshold commit the transition."""
        state = RegimeDetectorState()
        signal = compute_regime_signal(total_workers=100, total_used_slots=99, total_empty_slots=1)
        for _ in range(2):
            update_regime_state(state, signal, streak_cycles=3)
        transitioned = update_regime_state(state, signal, streak_cycles=3)
        assert transitioned is True
        assert state.current_regime is Regime.SUPER_HALFIN_WHITT
        assert state.streak == 0

    def test_streak_resets_on_above_threshold_cycle(self) -> None:
        """A single 'good' cycle resets the streak; oscillation does not flap regime."""
        state = RegimeDetectorState()
        below = compute_regime_signal(total_workers=100, total_used_slots=99, total_empty_slots=1)
        above = compute_regime_signal(total_workers=100, total_used_slots=50, total_empty_slots=50)
        update_regime_state(state, below, streak_cycles=3)
        update_regime_state(state, below, streak_cycles=3)
        update_regime_state(state, above, streak_cycles=3)
        update_regime_state(state, below, streak_cycles=3)
        # Streak rebuilt only once after the reset; no transition committed.
        assert state.current_regime is Regime.SUB_HALFIN_WHITT
        assert state.streak == 1


class TestExitSuperHwHysteresis:
    """Exit ``SUPER_HALFIN_WHITT`` after the streak above the wider exit band."""

    def test_idle_inside_exit_band_does_not_transition_back(self) -> None:
        """Idle above ``threshold`` but below ``threshold * 1.5`` holds super-HW."""
        state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=0)
        # threshold for N=100 = 0.10; exit band = 0.15. Use idle = 0.12.
        signal = compute_regime_signal(total_workers=100, total_used_slots=88, total_empty_slots=12)
        for _ in range(10):
            update_regime_state(state, signal, streak_cycles=3)
        assert state.current_regime is Regime.SUPER_HALFIN_WHITT
        assert state.streak == 0

    def test_third_consecutive_above_exit_band_transitions_to_sub_hw(self) -> None:
        """Three cycles at or above ``threshold * 1.5`` commit the back-transition."""
        state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=0)
        # threshold for N=100 = 0.10; exit band = 0.15. Use idle = 0.20.
        signal = compute_regime_signal(total_workers=100, total_used_slots=80, total_empty_slots=20)
        for _ in range(2):
            update_regime_state(state, signal, streak_cycles=3)
        transitioned = update_regime_state(state, signal, streak_cycles=3)
        assert transitioned is True
        assert state.current_regime is Regime.SUB_HALFIN_WHITT
        assert state.streak == 0


class TestNoSignalDefensiveGuard:
    """Cycles with no slot signals leave hysteresis state untouched."""

    def test_no_signal_does_not_advance_streak_in_sub_hw(self) -> None:
        """Sub-HW + no-signal cycle holds the streak; no spurious entry."""
        state = RegimeDetectorState(current_regime=Regime.SUB_HALFIN_WHITT, streak=2)
        no_signal = compute_regime_signal(total_workers=100, total_used_slots=0, total_empty_slots=0)
        transitioned = update_regime_state(state, no_signal, streak_cycles=3)
        assert transitioned is False
        assert state.current_regime is Regime.SUB_HALFIN_WHITT
        assert state.streak == 2

    def test_no_signal_does_not_advance_streak_in_super_hw(self) -> None:
        """Super-HW + no-signal cycle holds the streak; no spurious exit."""
        state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=2)
        no_signal = compute_regime_signal(total_workers=100, total_used_slots=0, total_empty_slots=0)
        transitioned = update_regime_state(state, no_signal, streak_cycles=3)
        assert transitioned is False
        assert state.current_regime is Regime.SUPER_HALFIN_WHITT
        assert state.streak == 2


class TestExitBandSmallClusterReachability:
    """Pin the ``min(1.0, threshold * 1.5)`` upper clamp on the exit band.

    Without the clamp, ``threshold * 1.5`` exceeds the maximum
    achievable ``cluster_idle_fraction`` (``1.0``) for tiny clusters,
    leaving a stage that briefly went busy permanently stuck in
    ``SUPER_HALFIN_WHITT``. The clamp makes the SUPER -> SUB exit
    reachable when ``cluster_idle_fraction == 1.0`` (every slot empty).
    """

    def test_single_worker_fully_idle_exits_after_streak(self) -> None:
        """``total_workers=1`` gives ``threshold=1.0`` so raw exit_band would be 1.5; clamp restores reachability."""
        state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=0)
        signal = compute_regime_signal(total_workers=1, total_used_slots=0, total_empty_slots=4)
        for _ in range(2):
            update_regime_state(state, signal, streak_cycles=3)
        transitioned = update_regime_state(state, signal, streak_cycles=3)
        assert transitioned is True
        assert state.current_regime is Regime.SUB_HALFIN_WHITT
        assert state.streak == 0

    def test_two_workers_fully_idle_exits_after_streak(self) -> None:
        """``total_workers=2`` gives raw exit_band ~1.06 so the clamp at 1.0 keeps the SUPER->SUB exit reachable."""
        state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=0)
        signal = compute_regime_signal(total_workers=2, total_used_slots=0, total_empty_slots=8)
        for _ in range(2):
            update_regime_state(state, signal, streak_cycles=3)
        transitioned = update_regime_state(state, signal, streak_cycles=3)
        assert transitioned is True
        assert state.current_regime is Regime.SUB_HALFIN_WHITT
        assert state.streak == 0

    def test_single_worker_partially_idle_holds_super_hw(self) -> None:
        """``cluster_idle_fraction < 1.0`` cannot meet the clamped exit band on a one-worker cluster."""
        state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=0)
        signal = compute_regime_signal(total_workers=1, total_used_slots=1, total_empty_slots=3)
        for _ in range(10):
            update_regime_state(state, signal, streak_cycles=3)
        assert state.current_regime is Regime.SUPER_HALFIN_WHITT
        assert state.streak == 0
