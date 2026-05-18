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

"""Pure-function 5-state classifier for the saturation-aware scheduler.

Maps ``(slots_empty_ratio_ewma, input_queue_depth, prev_state)``
plus thresholds + deadband config to a ``StageState`` zone with
hysteresis. ``0.0`` means every slot busy; ``1.0`` means every slot
free. Defined as a pure function so it is testable in isolation
and reusable by simulators / replay harnesses.

Zones (for ``ratio = slots_empty_ratio_ewma``)::

    ratio < activation                  -> SATURATED_CRITICAL  (no hysteresis)
    ratio < saturation_boundary         -> SATURATED
    ratio >= over_provisioned_boundary  -> OVER_PROVISIONED if queue > 0 else STARVED
    otherwise                           -> NORMAL

Hysteresis applies only to the state being exited: from
``SATURATED`` / ``SATURATED_CRITICAL`` the saturation boundary is
inflated by ``saturation_deadband_pct``; from ``OVER_PROVISIONED``
the over-provisioned boundary is deflated by
``over_provisioned_deadband_pct``. ``STARVED`` has no hysteresis
because the queue-empty signal can flip immediately.
"""

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
) -> StageState:
    """Classify a stage's current cycle by its saturation signals.

    Args:
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in ``[0, 1]``.
        input_queue_depth: Tasks waiting upstream of this stage;
            disambiguates ``OVER_PROVISIONED`` from ``STARVED``.
        prev_state: The previous cycle's zone (drives hysteresis).
        saturation_threshold: Empty-slot fraction below which the
            stage is classified ``SATURATED``. Strictly positive and
            strictly less than ``config.over_provisioned_threshold``.
        activation_threshold: Empty-slot fraction below which the
            stage is classified ``SATURATED_CRITICAL``. Strictly less
            than ``saturation_threshold``.
        config: Per-stage configuration carrying the deadbands and
            the over-provisioned threshold.

    Returns:
        The current cycle's classifier zone.

    """
    if slots_empty_ratio_ewma < activation_threshold:
        return StageState.SATURATED_CRITICAL

    saturation_boundary = saturation_threshold
    if prev_state in (StageState.SATURATED, StageState.SATURATED_CRITICAL):
        saturation_boundary *= 1.0 + config.saturation_deadband_pct
    if slots_empty_ratio_ewma < saturation_boundary:
        return StageState.SATURATED

    over_provisioned_boundary = config.over_provisioned_threshold
    if prev_state == StageState.OVER_PROVISIONED:
        over_provisioned_boundary *= 1.0 - config.over_provisioned_deadband_pct
    if slots_empty_ratio_ewma >= over_provisioned_boundary:
        if input_queue_depth == 0:
            return StageState.STARVED
        return StageState.OVER_PROVISIONED

    return StageState.NORMAL
