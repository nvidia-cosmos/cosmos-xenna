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

"""Per-stage runtime state types for the saturation-aware scheduler.

Three concerns live in this module:

  1. The discrete state enums (``StageState``, ``GrowthMode``) the
     scheduler reads and emits. Both are string-valued so they can be
     used directly as Prometheus label values.
  2. The per-stage runtime container (``_StageRuntimeState``) the
     scheduler keeps across cycles - streak counters, EWMA value,
     classifier output, growth-mode state.
  3. Two pure-function helpers used by the scheduler each cycle:
     ``compute_slots_empty_ratio`` (raw signal) and ``update_ewma``
     (smoothing). They are isolated so they can be unit-tested in
     pure form, free of scheduler context.
"""

import enum

import attrs


class StageState(str, enum.Enum):
    """Five-zone saturation classifier output for a single stage.

    The classifier maps the (smoothed slots-empty ratio, input queue
    depth) pair onto one of these five zones each cycle. A per-stage
    streak counter then converts a sustained zone into an actionable
    signal once the configured number of consecutive cycles is reached.

    Members
    -------

    NORMAL
        Operating within bounds; no scale action warranted.
    STARVED
        Free slots remain but the input queue is empty; upstream is
        the bottleneck, no local scale action would help.
    SATURATED
        Few free slots (below ``saturation_threshold``); ordinary
        scale-up signal once sustained.
    SATURATED_CRITICAL
        Effectively zero free slots (below ``activation_threshold``);
        burst signal that bypasses hysteresis.
    OVER_PROVISIONED
        Many free slots with input pending; sustained scale-down
        signal.

    """

    NORMAL = "NORMAL"
    STARVED = "STARVED"
    SATURATED = "SATURATED"
    SATURATED_CRITICAL = "SATURATED_CRITICAL"
    OVER_PROVISIONED = "OVER_PROVISIONED"


class GrowthMode(str, enum.Enum):
    """Slow-start growth controller mode for a stage.

    Three-regime state machine that biases scale-up aggressiveness
    based on history. The scheduler cycles a stage between these
    modes as it observes saturation, ceiling, and shrink edges.

    Members
    -------

    ACQUIRING
        New stage or recently shrank below the previous ceiling -
        grow multiplicatively on saturation signals (slow-start).
    TRACKING
        Operating near the previous saturated ceiling - grow
        additively (congestion-avoidance).
    HOLD
        Just shrank from over-provisioned; suppress non-critical
        growth for one or more cycles to stabilize.

    """

    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    HOLD = "HOLD"


@attrs.define
class _StageRuntimeState:
    """Per-stage runtime state held by the scheduler across cycles.

    One instance per stage, keyed by stage name. Mutated each cycle
    by the classifier, streak counter, and growth controller. None of
    these fields participate in the scheduler's public API; they are
    internal bookkeeping.

    Attributes:
        stage_name: The pipeline stage's logical name.
        slots_empty_ratio_ewma: Exponential moving average of the
            stage's slots-empty ratio. ``None`` until the first
            sample arrives so cold-start does not pay an EWMA-warmup
            tax.
        last_valid_slots_empty_ratio_ewma: Most-recent finite EWMA
            value, retained across cycles where the stage has zero
            ready actors so the classifier never reads NaN. ``None``
            until the first finite EWMA is observed.
        classifier_state: Output of the most recent classifier call.
            Initial value ``NORMAL`` so the first cycle's hysteresis
            logic has a defined baseline.
        classifier_streak: Consecutive cycles that ``classifier_state``
            has held its current value. Resets to 1 on transition.
        growth_mode: Slow-start regime for this stage.
        growth_streak: Consecutive cycles spent in ``growth_mode``.
        prev_workers: Worker count observed at the end of the
            previous cycle. Used by growth-mode transitions to
            detect grow / shrink edges.

    """

    stage_name: str
    slots_empty_ratio_ewma: float | None = None
    last_valid_slots_empty_ratio_ewma: float | None = None
    classifier_state: StageState = StageState.NORMAL
    classifier_streak: int = 0
    growth_mode: GrowthMode = GrowthMode.ACQUIRING
    growth_streak: int = 0
    prev_workers: int = 0


def compute_slots_empty_ratio(num_used_slots: int, num_empty_slots: int) -> float:
    """Compute slots-empty / total-slots for a stage.

    Args:
        num_used_slots: Slots currently occupied by in-flight tasks.
            Must be ``>= 0``.
        num_empty_slots: Slots currently free. Must be ``>= 0``.

    Returns:
        The empty-slot fraction in ``[0.0, 1.0]``. Returns ``0.0``
        when both inputs are zero (stage has no actors); the
        classifier treats this as "no capacity" which maps to
        SATURATED_CRITICAL, matching the behaviour the worker-floor
        step independently grows out of.

    Raises:
        ValueError: If either input is negative.

    """
    if num_used_slots < 0:
        msg = f"num_used_slots must be >= 0, got {num_used_slots}"
        raise ValueError(msg)
    if num_empty_slots < 0:
        msg = f"num_empty_slots must be >= 0, got {num_empty_slots}"
        raise ValueError(msg)
    total = num_used_slots + num_empty_slots
    if total == 0:
        return 0.0
    return num_empty_slots / total


def update_ewma(prev_ewma: float | None, sample: float, alpha: float) -> float:
    """Update the exponential moving average with a new sample.

    Standard EWMA: ``new = alpha * sample + (1 - alpha) * prev``.
    On cold start (``prev_ewma is None``) the result is the sample
    itself, so the first cycle's classifier sees the live signal
    rather than a zero-anchored half-step.

    Args:
        prev_ewma: Previous EWMA value, or ``None`` on cold start.
        sample: New raw observation.
        alpha: Smoothing factor in ``(0.0, 1.0]``. Higher = less
            smoothing.

    Returns:
        Updated EWMA value.

    Raises:
        ValueError: If ``alpha`` is outside ``(0.0, 1.0]``.

    """
    if not (0.0 < alpha <= 1.0):
        msg = f"alpha must be in (0.0, 1.0], got {alpha}"
        raise ValueError(msg)
    if prev_ewma is None:
        return sample
    return alpha * sample + (1.0 - alpha) * prev_ewma
