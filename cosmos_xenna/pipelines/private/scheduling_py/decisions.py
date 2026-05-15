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

"""Per-cycle decision helpers for the saturation-aware scheduler.

Three pure functions consumed by the scheduler's per-cycle decision
pipeline, sitting between the classifier and the Solution-construction
step:

::

   classifier ---> update_streak ---> should_fire_action ---> compute_delta
                   (counts cycles)    (gate on threshold)     (signed worker count)

  - ``update_streak``: counts consecutive cycles in the same
    classifier state; resets to 1 on transitions.
  - ``should_fire_action``: predicate gating whether a sustained
    state has reached its configured action threshold (asymmetric:
    aggressive scale-up, conservative scale-down).
  - ``compute_delta``: produces the worker count delta given the
    current classifier state, growth mode, and worker count. The
    caller is responsible for clamping by per-stage min/max,
    per-node caps, and cluster capacity; this function only encodes
    the algorithm's intent.

All three are pure functions of their inputs so they can be tested
in isolation without scheduler context.
"""

import math

from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def update_streak(prev_state: StageState, prev_streak: int, new_state: StageState) -> int:
    """Return the new streak count.

    A streak is the number of consecutive cycles the classifier has
    emitted the same state. Resets to 1 on any state transition.

    Args:
        prev_state: Classifier output from the previous cycle.
        prev_streak: Streak count entering this cycle. Must be ``>= 0``.
        new_state: Classifier output from the current cycle.

    Returns:
        ``prev_streak + 1`` when the state holds, ``1`` on transition.

    Raises:
        ValueError: If ``prev_streak`` is negative.

    """
    if prev_streak < 0:
        msg = f"prev_streak must be >= 0, got {prev_streak}"
        raise ValueError(msg)
    if new_state == prev_state:
        return prev_streak + 1
    return 1


def should_fire_action(
    state: StageState,
    streak: int,
    config: SaturationAwareStageConfig,
) -> bool:
    """Return whether a sustained state has reached its action threshold.

    Streak thresholds are intentionally asymmetric: scale-up
    (SATURATED, SATURATED_CRITICAL) fires aggressively after few
    cycles; scale-down (OVER_PROVISIONED) requires a long sustained
    signal; STARVED waits long enough to log the upstream-bottleneck
    warning. NORMAL is the no-action zone and never fires.

    Args:
        state: Current classifier output.
        streak: Number of consecutive cycles in ``state``.
        config: Per-stage config carrying the streak thresholds.

    Returns:
        True if the configured threshold for ``state`` is reached.

    """
    if state == StageState.SATURATED_CRITICAL:
        return streak >= config.saturated_critical_streak_min_cycles
    if state == StageState.SATURATED:
        return streak >= config.saturated_streak_min_cycles
    if state == StageState.OVER_PROVISIONED:
        return streak >= config.over_provisioned_streak_min_cycles
    if state == StageState.STARVED:
        return streak >= config.starved_streak_min_cycles
    return False


def compute_delta(
    state: StageState,
    growth_mode: GrowthMode,
    current_workers: int,
    config: SaturationAwareStageConfig,
) -> int:
    """Return the unclamped worker count delta for this cycle.

    Positive = scale-up; negative = scale-down; zero = no scale
    action. The caller is responsible for clamping by per-stage
    min/max, per-node caps, and cluster capacity; this function only
    encodes the algorithm's intent, not the feasibility check.

    Args:
        state: Current classifier output. NORMAL and STARVED produce
            zero (no scale action).
        growth_mode: Per-stage growth controller mode (ACQUIRING,
            TRACKING, HOLD). Determines the magnitude of scale-up.
        current_workers: Worker count before this cycle's decision.
            Must be ``>= 0``. Used as a base for multiplicative
            growth in ACQUIRING mode and the fraction-based scale-down
            cap in OVER_PROVISIONED state.
        config: Per-stage config carrying growth factors, absolute
            counts, the per-cycle scale-up cap, and the scale-down
            fraction cap.

    Returns:
        Signed integer: positive to add workers, negative to remove,
        zero for NORMAL/STARVED.

    Raises:
        ValueError: If ``current_workers`` is negative.

    """
    if current_workers < 0:
        msg = f"current_workers must be >= 0, got {current_workers}"
        raise ValueError(msg)

    if state == StageState.SATURATED_CRITICAL:
        return _critical_delta(growth_mode, current_workers, config)
    if state == StageState.SATURATED:
        return _saturated_delta(growth_mode, current_workers, config)
    if state == StageState.OVER_PROVISIONED:
        return _shrink_delta(current_workers, config)
    return 0


def _critical_delta(
    growth_mode: GrowthMode,
    current_workers: int,
    config: SaturationAwareStageConfig,
) -> int:
    """Burst-zone scale-up magnitude, clamped by ``aggressive_growth_max_per_cycle``."""
    if growth_mode == GrowthMode.ACQUIRING:
        delta = math.ceil(config.acquiring_critical_growth_factor * current_workers)
    elif growth_mode == GrowthMode.TRACKING:
        delta = config.tracking_critical_growth_count
    else:
        delta = config.hold_critical_growth_count
    return min(delta, config.aggressive_growth_max_per_cycle)


def _saturated_delta(
    growth_mode: GrowthMode,
    current_workers: int,
    config: SaturationAwareStageConfig,
) -> int:
    """Sustained-saturation scale-up magnitude, clamped by ``aggressive_growth_max_per_cycle``."""
    if growth_mode == GrowthMode.ACQUIRING:
        delta = math.ceil(config.acquiring_saturated_growth_factor * current_workers)
    elif growth_mode == GrowthMode.TRACKING:
        delta = config.tracking_saturated_growth_count
    else:
        delta = config.hold_saturated_growth_count
    return min(delta, config.aggressive_growth_max_per_cycle)


def _shrink_delta(current_workers: int, config: SaturationAwareStageConfig) -> int:
    """Return scale-down magnitude (negative integer).

    Bounded by ``max_scale_down_fraction_per_cycle`` to prevent
    cliff scale-downs that starve downstream stages. Always returns
    at least ``-1`` once the OVER_PROVISIONED streak threshold is
    reached; the caller clamps to per-stage and one-worker floors.
    """
    fraction_cap = max(1, math.floor(current_workers * config.max_scale_down_fraction_per_cycle))
    return -fraction_cap
