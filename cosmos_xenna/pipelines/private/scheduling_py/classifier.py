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

"""Pure-function 4-state classifier for the saturation-aware scheduler.

Slot-pin gate selects the candidate zone from the smoothed
slots-empty ratio; the smoothed backlog-time pressure then demotes
the candidate when it disagrees with the queue evidence:

  - SATURATED_CRITICAL is confirmed only when pressure exceeds
    ``pressure_critical_threshold``.
  - SATURATED is confirmed only when pressure exceeds
    ``pressure_saturation_threshold``; otherwise the queue is
    draining and the classifier returns NORMAL.
  - OVER_PROVISIONED is demoted to NORMAL when pressure exceeds
    ``pressure_normal_threshold`` (the queue is stuck downstream and
    shrinking would worsen the stall).

A drained input queue with idle slots resolves to OVER_PROVISIONED
under the same pressure-demotion gate; topology context (whether
this stage is upstream or downstream of an engaged bottleneck) is
captured at the log layer for diagnostics.

See ``docs/scheduler/saturation-aware/05-state-classifier.md``
and ``27-topology-aware-classifier.md`` for the full rationale.
"""

import math

from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def classify(
    *,
    slots_empty_ratio_ewma: float,
    input_queue_depth: int,
    prev_state: StageState,
    saturation_threshold: float,
    activation_threshold: float,
    config: SaturationAwareStageConfig,
    pressure_ewma: float | None = None,
) -> StageState:
    """Classify a stage's current cycle.

    Args:
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in
            ``[0, 1]``.
        input_queue_depth: Tasks waiting upstream of this stage.
            Retained as a diagnostic input; no longer participates
            in the zone decision.
        pressure_ewma: Smoothed backlog-time pressure scalar or
            ``None`` (no pressure value yet -- preserves slot-only
            behaviour for the cycle).
        prev_state: Previous cycle's zone (drives hysteresis).
        saturation_threshold: Empty-slot fraction below which the
            slot-pin gate enters ``SATURATED``. Strictly positive
            and strictly less than ``config.over_provisioned_threshold``.
        activation_threshold: Empty-slot fraction below which the
            slot-pin gate enters ``SATURATED_CRITICAL``. Strictly
            less than ``saturation_threshold``.
        config: Per-stage configuration.

    Returns:
        The current cycle's classifier zone.

    Raises:
        ValueError: If ``slots_empty_ratio_ewma`` is non-finite, or
            if ``pressure_ewma`` is provided and non-finite. NaN
            comparisons would silently fall through to
            ``StageState.NORMAL`` (every ``<``/``>``/``>=`` returns
            ``False``); ``+/-Inf`` would steer the slot-pin gate
            into spurious CRITICAL or OVER_PROVISIONED branches.

    """
    del input_queue_depth  # diagnostic input only; preserved on the public signature.

    if not math.isfinite(slots_empty_ratio_ewma):
        msg = f"slots_empty_ratio_ewma must be finite, got {slots_empty_ratio_ewma!r}"
        raise ValueError(msg)
    if pressure_ewma is not None and not math.isfinite(pressure_ewma):
        msg = f"pressure_ewma must be finite when provided, got {pressure_ewma!r}"
        raise ValueError(msg)

    if slots_empty_ratio_ewma < activation_threshold:
        if pressure_ewma is None or pressure_ewma > config.pressure_critical_threshold:
            return StageState.SATURATED_CRITICAL

    saturation_boundary = saturation_threshold
    if prev_state in (StageState.SATURATED, StageState.SATURATED_CRITICAL):
        saturation_boundary *= 1.0 + config.saturation_deadband_pct
    if slots_empty_ratio_ewma < saturation_boundary:
        if pressure_ewma is None or pressure_ewma > config.pressure_saturation_threshold:
            return StageState.SATURATED
        return StageState.NORMAL

    over_provisioned_boundary = config.over_provisioned_threshold
    if prev_state == StageState.OVER_PROVISIONED:
        over_provisioned_boundary *= 1.0 - config.over_provisioned_deadband_pct
    if slots_empty_ratio_ewma >= over_provisioned_boundary:
        if pressure_ewma is not None and pressure_ewma > config.pressure_normal_threshold:
            return StageState.NORMAL
        return StageState.OVER_PROVISIONED

    return StageState.NORMAL
