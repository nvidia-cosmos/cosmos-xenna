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

The container is split into three orthogonal sub-states wrapped by
``StageRuntimeState``:

- ``ClassifierState`` - written only by the classification /
  trust-gate path: discrete classifier zone, streak, EWMAs of the
  raw classifier signal, resolved auto thresholds, and the
  signal-noise EWMA used by donor signal-trust.
- ``GrowthState`` - written only by the capacity sizing and the
  post-Phase-D ``record_executed_delta`` call: growth-mode FSM
  cursor, growth streak, last-cycle worker count, and the
  capacity-target hint.
- ``PressureState`` - written only by the pressure / backlog
  signal path: smoothed pressure EWMA.

``StageRuntimeState`` owns the per-stage identity (``stage_name``)
plus the three orthogonal sub-states. The sub-states are mutable
``@attrs.define`` objects; the aggregate uses ``attrs.Factory``
defaults so a stage entering the cycle for the first time starts
with a clean sub-state. Per-cycle bottleneck-topology projections
are computed on-the-fly at the consumer call site - no per-stage
mirror lives on the runtime state.

Two pure-function helpers complete the module so the streaming
classifier path can be unit-tested without scheduler context:
``compute_slots_empty_ratio`` (raw signal) and ``update_ewma``
(EWMA smoothing).

"""

import enum

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import ResolvedThresholds


class StageState(enum.StrEnum):
    """Four-zone saturation classifier output for a single stage.

    Attributes:
        NORMAL: Operating within bounds; no scale action warranted.
        SATURATED: Few free slots (below ``saturation_threshold``);
            ordinary scale-up signal once sustained.
        SATURATED_CRITICAL: Effectively zero free slots (below
            ``activation_threshold``); burst signal that bypasses
            hysteresis.
        OVER_PROVISIONED: Sustained idle slots; scale-down candidate
            and cross-stage donor candidate. Captures both genuine
            over-provisioning and backpressure-induced idleness
            upstream / downstream of an engaged bottleneck.

    """

    NORMAL = "NORMAL"
    SATURATED = "SATURATED"
    SATURATED_CRITICAL = "SATURATED_CRITICAL"
    OVER_PROVISIONED = "OVER_PROVISIONED"


class GrowthMode(enum.StrEnum):
    """Per-stage lifecycle mode that gates HOLD entry.

    Captures "has this stage ever shrunk?" so the post-shrink HOLD
    stabilization window is entered exactly once per shrink event.

    Attributes:
        ACQUIRING: No shrink observed yet.
        TRACKING: At least one shrink observed; ceiling known.
        HOLD: Post-shrink stabilization window. Blocks ``SATURATED``
            grow; ``SATURATED_CRITICAL`` grow is always allowed.

    """

    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    HOLD = "HOLD"


@attrs.define
class ClassifierState:
    """Classification-path mutable state for one stage.

    Holds the discrete classifier zone, the streak counter, the
    raw classifier-signal EWMAs (live + carry-forward + noise),
    resolved auto thresholds, and the validity-sample counter.
    Mutated only by the per-stage decision pipeline's classifier
    branch.

    Attributes:
        state: Current classifier zone (``NORMAL``, ``SATURATED``,
            ``SATURATED_CRITICAL``, ``OVER_PROVISIONED``).
        streak: Consecutive cycles in ``state``; reset on
            transition.
        valid_signal_samples: Count of cycles with a fresh slot
            sample; gates fire-rate decisions.
        resolved_thresholds: Per-stage auto-resolved thresholds
            (saturation, activation, over-provisioned); ``None``
            before ``ThresholdResolver.ensure_resolved`` populates
            it on the first cycle.
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in
            ``[0.0, 1.0]``; ``None`` cold-start.
        last_valid_slots_empty_ratio_ewma: Most recent fresh-sample
            EWMA; used to carry forward across zero-actor cycles.
        signal_noise_ewma: EWMA of ``|new_ewma - prev_ewma|`` used
            by the donor signal-trust gate.

    """

    state: StageState = StageState.NORMAL
    streak: int = 0
    valid_signal_samples: int = 0
    resolved_thresholds: ResolvedThresholds | None = None
    slots_empty_ratio_ewma: float | None = None
    last_valid_slots_empty_ratio_ewma: float | None = None
    signal_noise_ewma: float | None = None


@attrs.define
class GrowthState:
    """Growth-mode FSM mutable state for one stage.

    Captures the ACQUIRING -> TRACKING -> HOLD lifecycle plus the
    streak and the last-cycle worker count. Mutated only by the
    capacity sizing and the post-Phase-D ``record_executed_delta``
    advancement.

    Attributes:
        mode: Current growth mode (``ACQUIRING`` / ``TRACKING`` /
            ``HOLD``).
        streak: Consecutive cycles in ``mode``; reset on
            transition.
        prev_workers: Worker count observed at the start of the
            previous cycle; used by capacity sizing.
        capacity_target_workers: Optional hint published by the
            capacity-target sizer for the grow path.

    """

    mode: GrowthMode = GrowthMode.ACQUIRING
    streak: int = 0
    prev_workers: int = 0
    capacity_target_workers: int | None = None


@attrs.define
class PressureState:
    """Backlog-pressure mutable state for one stage.

    Holds the smoothed pressure EWMA written by the pressure /
    backlog signal path. Carry-forward and cold-start semantics
    are owned by the per-stage decision pipeline.

    Attributes:
        ewma: Smoothed pressure scalar in
            ``[0.0, BACKLOG_CAP]``; ``None`` cold-start.

    """

    ewma: float | None = None


@attrs.define
class StageRuntimeState:
    """Per-stage cross-cycle runtime state aggregate.

    Owns per-stage identity (``stage_name``) and the three
    orthogonal mutable sub-states whose ownership is enforced by
    file boundaries:

    +---++
    | classifier      | classification + trust-gate path                |
    | growth          | capacity sizing + executed-delta recording      |
    | pressure        | pressure / backlog signal path                  |
    +---++

    One instance per stage keyed by stage name. The aggregate
    holds no new flat scalars - every new per-stage scalar lives
    on the appropriate sub-state. Per-cycle bottleneck topology
    is computed on-the-fly as :class:`StageTopologyContext` at
    the consumer call site - never stored here.

    """

    stage_name: str
    classifier: ClassifierState = attrs.Factory(ClassifierState)
    growth: GrowthState = attrs.Factory(GrowthState)
    pressure: PressureState = attrs.Factory(PressureState)


type StageStateMap = dict[str, StageRuntimeState]
"""Typed alias for the per-stage runtime-state map.

Phases (``FloorPhase``, ``IntentPhase``, ``SaturationGrowPhase``)
mutate the inner :class:`StageRuntimeState` sub-states directly, so
the alias exposes the live mutable dict instead of a wrapper. Used by
``AutoscaleCycle.view_for(stage_index, stage_states)`` and by every
``*Services`` value object that consumes per-stage runtime state.
"""


def compute_slots_empty_ratio(num_used_slots: int, num_empty_slots: int) -> float:
    """Compute slots-empty / total-slots for a stage.

    Args:
        num_used_slots: Slots currently occupied. Must be ``>= 0``.
        num_empty_slots: Slots currently free. Must be ``>= 0``.

    Returns:
        Empty-slot fraction in ``[0.0, 1.0]``; ``0.0`` when both
        inputs are zero (stage has no actors).

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

    ``new = alpha * sample + (1 - alpha) * prev``; cold start
    (``prev_ewma is None``) returns the sample itself.

    Raises:
        ValueError: ``alpha`` outside ``(0.0, 1.0]``.

    """
    if not (0.0 < alpha <= 1.0):
        msg = f"alpha must be in (0.0, 1.0], got {alpha}"
        raise ValueError(msg)
    if prev_ewma is None:
        return sample
    return alpha * sample + (1.0 - alpha) * prev_ewma


__all__ = (
    "ClassifierState",
    "GrowthMode",
    "GrowthState",
    "PressureState",
    "StageRuntimeState",
    "StageState",
    "StageStateMap",
    "compute_slots_empty_ratio",
    "update_ewma",
)
