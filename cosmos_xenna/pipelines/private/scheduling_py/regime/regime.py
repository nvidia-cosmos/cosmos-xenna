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

"""Halfin-Whitt regime detector for the saturation-aware scheduler.

Two heavy-traffic regimes - sub-Halfin-Whitt (slack headroom; a
wrong scale-down recovers next cycle) and super-Halfin-Whitt
(packed; a freed slot is likely re-claimed before recovery,
costing up to ``log(N)`` cycles). Detected per cycle by comparing
cluster-wide aggregate empty-slot fraction against
``threshold = 1 / sqrt(total_workers)``. Hysteresis: enter super-HW
after ``streak_cycles`` below threshold, exit after the same
streak above ``min(1.0, threshold * EXIT_BAND_MULTIPLIER)``.
``RegimeDetectorState`` carries the cross-cycle streak.

Defined as pure functions so they are testable in isolation without
a scheduler context.
"""

import enum
import math
from typing import Final

import attrs

# Asymmetric hysteresis exit band. The exit threshold is
# ``threshold * EXIT_BAND_MULTIPLIER`` so a regime that has just
# entered super-Halfin-Whitt requires a sustained signal noticeably
# above the entry threshold to revert.
EXIT_BAND_MULTIPLIER: Final[float] = 1.5


class Regime(enum.StrEnum):
    """Heavy-traffic operating regime of the cluster.

    Attributes:
        SUB_HALFIN_WHITT: Cluster has slack; ``cluster_idle >=
            threshold``. Default aggressiveness is correct.
        SUPER_HALFIN_WHITT: Cluster is packed; ``cluster_idle <
            threshold``. Scale-up should be more responsive (the
            aggressiveness lift fires).

    """

    SUB_HALFIN_WHITT = "SUB_HALFIN_WHITT"
    SUPER_HALFIN_WHITT = "SUPER_HALFIN_WHITT"


@attrs.frozen
class RegimeSignal:
    """Per-cycle regime-detection inputs.

    Attributes:
        total_workers: Sum of worker counts across all stages.
        cluster_idle_fraction: Aggregate empty-slot fraction across
            the cluster: ``total_empty_slots / (total_used_slots +
            total_empty_slots)``.
        threshold: ``1 / sqrt(total_workers)``. Equals ``1.0`` when
            ``total_workers <= 1``.
        signal_available: ``False`` when no stage in the input had any
            slot signal populated (sum of used + empty slots is zero
            across the cluster). Callers leave hysteresis state
            unchanged when the signal is unavailable.

    """

    total_workers: int
    cluster_idle_fraction: float
    threshold: float
    signal_available: bool


@attrs.define
class RegimeDetectorState:
    """Cross-cycle hysteresis state for the regime detector.

    Attributes:
        current_regime: The regime currently in effect after hysteresis.
            Defaults to ``SUB_HALFIN_WHITT`` so a freshly-started cluster
            uses the base aggressiveness until enough evidence accumulates
            for a transition.
        streak: Consecutive cycles whose raw signal disagrees with
            ``current_regime``. Resets to zero on any cycle that agrees
            with ``current_regime``; unavailable signals leave the
            streak unchanged.

    """

    current_regime: Regime = Regime.SUB_HALFIN_WHITT
    streak: int = 0


def compute_regime_signal(
    *,
    total_workers: int,
    total_used_slots: int,
    total_empty_slots: int,
) -> RegimeSignal:
    """Compute the cluster's regime-detection signal for one cycle.

    Args:
        total_workers: Sum of worker counts across all stages.
        total_used_slots: Sum of occupied slots across all stages.
        total_empty_slots: Sum of free slots across all stages.

    Returns:
        A ``RegimeSignal`` carrying the inputs the downstream
        hysteresis update needs. ``signal_available`` is ``False``
        when ``total_used_slots + total_empty_slots == 0``; callers
        leave hysteresis state unchanged in that case.

    """
    total_slots = total_used_slots + total_empty_slots
    signal_available = total_slots > 0
    cluster_idle_fraction = (total_empty_slots / total_slots) if signal_available else 0.0

    if total_workers <= 1:
        threshold = 1.0
    else:
        threshold = 1.0 / math.sqrt(total_workers)

    return RegimeSignal(
        total_workers=total_workers,
        cluster_idle_fraction=cluster_idle_fraction,
        threshold=threshold,
        signal_available=signal_available,
    )


def update_regime_state(
    state: RegimeDetectorState,
    signal: RegimeSignal,
    *,
    streak_cycles: int,
    exit_band_multiplier: float = EXIT_BAND_MULTIPLIER,
) -> bool:
    """Apply hysteresis to ``state`` in place; return whether the regime transitioned.

    Enter ``SUPER_HALFIN_WHITT`` after ``streak_cycles`` below
    ``threshold``. Exit after the same streak at or above
    ``min(1.0, threshold * exit_band_multiplier)`` - asymmetric
    exit band kills boundary flap. Cycles without an available
    signal neither advance nor reset the streak.

    Raises:
        ValueError: ``streak_cycles < 1`` or
            ``exit_band_multiplier <= 1.0``.

    """
    if streak_cycles < 1:
        msg = f"streak_cycles must be >= 1, got {streak_cycles}"
        raise ValueError(msg)
    if exit_band_multiplier <= 1.0:
        msg = f"exit_band_multiplier must be > 1.0, got {exit_band_multiplier}"
        raise ValueError(msg)

    if not signal.signal_available:
        return False

    if state.current_regime is Regime.SUB_HALFIN_WHITT:
        if signal.cluster_idle_fraction < signal.threshold:
            state.streak += 1
            if state.streak >= streak_cycles:
                state.current_regime = Regime.SUPER_HALFIN_WHITT
                state.streak = 0
                return True
            return False
        state.streak = 0
        return False

    exit_threshold = min(1.0, signal.threshold * exit_band_multiplier)
    if signal.cluster_idle_fraction >= exit_threshold:
        state.streak += 1
        if state.streak >= streak_cycles:
            state.current_regime = Regime.SUB_HALFIN_WHITT
            state.streak = 0
            return True
        return False
    state.streak = 0
    return False
