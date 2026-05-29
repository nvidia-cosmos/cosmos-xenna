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

Three pure functions between classifier and Solution
construction: ``update_streak`` counts consecutive same-state
cycles; ``should_fire_action`` gates on the asymmetric action
threshold; ``compute_delta`` produces the worker count delta from
state + growth mode + capacity target. Callers own per-stage,
per-node, and cluster-cap clamping.
"""

import math

from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import GrowthMode, StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def update_streak(prev_state: StageState, prev_streak: int, new_state: StageState) -> int:
    """Return the new streak count.

    Consecutive same-state cycles; resets to 1 on transition.
    Returns ``prev_streak + 1`` when the state holds, ``1`` otherwise.

    Raises:
        ValueError: ``prev_streak`` is negative.

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

    Asymmetric thresholds: scale-up (SATURATED,
    SATURATED_CRITICAL) fires after few cycles; scale-down
    (OVER_PROVISIONED) requires a long sustained signal. NORMAL
    is the no-action zone.

    Raises:
        ValueError: ``streak`` is negative.

    """
    if streak < 0:
        msg = f"streak must be >= 0, got {streak} (state={state})"
        raise ValueError(msg)
    if state == StageState.SATURATED_CRITICAL:
        return streak >= config.saturated_critical_streak_min_cycles
    if state == StageState.SATURATED:
        return streak >= config.saturated_streak_min_cycles
    if state == StageState.OVER_PROVISIONED:
        return streak >= config.over_provisioned_streak_min_cycles
    return False


def compute_delta(
    state: StageState,
    growth_mode: GrowthMode,
    current_workers: int,
    capacity_target_workers: int | None,
    config: SaturationAwareStageConfig,
) -> int:
    """Return the unclamped worker count delta for this cycle.

    Capacity-driven: magnitude is the gap between
    ``current_workers`` and ``compute_capacity_target_workers``,
    clamped by per-cycle blast-radius caps. Growth mode is a binary
    gate - HOLD blocks ``SATURATED`` grow; ``SATURATED_CRITICAL``
    is never blocked. ``capacity_target_workers=None`` triggers the
    discrete +/-1 cold-start fallback.

    Raises:
        ValueError: ``current_workers`` is negative.

    """
    if current_workers < 0:
        msg = f"current_workers must be >= 0, got {current_workers}"
        raise ValueError(msg)

    if state == StageState.NORMAL:
        return 0

    if state == StageState.SATURATED and growth_mode == GrowthMode.HOLD:
        return 0

    if capacity_target_workers is None:
        return _cold_start_delta(state)

    if state == StageState.OVER_PROVISIONED:
        excess = current_workers - capacity_target_workers
        if excess <= 0:
            return 0
        fraction_cap = max(1, math.floor(current_workers * config.max_scale_down_fraction_per_cycle))
        return -min(excess, fraction_cap)

    shortfall = capacity_target_workers - current_workers
    if shortfall <= 0:
        return 0
    return min(shortfall, config.aggressive_growth_max_per_cycle)


def _cold_start_delta(state: StageState) -> int:
    """Return discrete +/-1 fallback when the capacity target is unobservable.

    SATURATED / SATURATED_CRITICAL grow by one; OVER_PROVISIONED
    shrinks by one. NORMAL and HOLD-blocked SATURATED never reach
    this branch.
    """
    if state == StageState.OVER_PROVISIONED:
        return -1
    return 1
