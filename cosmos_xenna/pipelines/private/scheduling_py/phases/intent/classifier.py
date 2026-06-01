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
slots-empty ratio; the smoothed backlog-time pressure then
demotes the candidate when it disagrees with the queue evidence
(SATURATED / SATURATED_CRITICAL require matching pressure;
OVER_PROVISIONED demotes to NORMAL when pressure is high). An
engaged pipeline bottleneck overrides the low-pressure SATURATED
demotion and holds SATURATED. Drained queue + idle slots resolves
to OVER_PROVISIONED under the same demotion gate. See
``docs/scheduler/saturation-aware/`` for the algorithm.
"""

import math

from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def classify(
    *,
    slots_empty_ratio_ewma: float,
    prev_state: StageState,
    saturation_threshold: float,
    activation_threshold: float,
    config: SaturationAwareStageConfig,
    pressure_ewma: float | None = None,
    is_bottleneck: bool = False,
) -> StageState:
    """Classify a stage's current cycle.

    Returns the current cycle's zone from the slot-pin gate, with
    hysteresis driven by ``prev_state`` and an optional backlog
    pressure demotion (``pressure_ewma=None`` keeps slot-only
    behaviour). Threshold ordering must satisfy
    ``activation < saturation < config.over_provisioned_threshold``.

    Raises:
        ValueError: Non-finite ``slots_empty_ratio_ewma`` or
            ``pressure_ewma``. NaN/Inf inputs would otherwise
            silently fall through to ``NORMAL`` or spurious
            CRITICAL / OVER_PROVISIONED branches.

    """
    if not math.isfinite(slots_empty_ratio_ewma):
        msg = f"slots_empty_ratio_ewma must be finite, got {slots_empty_ratio_ewma!r}"
        raise ValueError(msg)
    if not 0.0 <= slots_empty_ratio_ewma <= 1.0:
        msg = f"slots_empty_ratio_ewma must be in [0, 1], got {slots_empty_ratio_ewma!r}"
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
        if is_bottleneck:
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
