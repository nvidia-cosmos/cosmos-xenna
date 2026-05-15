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

The classifier reads empty capacity: ``0.0`` means every slot is busy,
``1.0`` means every slot is free. It maps the smoothed empty-slot
ratio, input queue depth, and previous state to a ``StageState``.
Hysteresis keeps noisy samples near the saturation and over-provisioned
thresholds from flapping; the critical zone bypasses hysteresis so
severe saturation is detected immediately.

Default thresholds shown below are configurable via
``SaturationAwareStageConfig``:

::

    slots_empty_ratio_ewma
    0.0 = all slots busy                                1.0 = all slots free

      1.00  +--------------------------------------------------------------+
            | UPPER IDLE ZONE                                              |
            |                                                              |
            | ratio >= over_provisioned_boundary                           |
            |   queue > 0  : OVER_PROVISIONED  -> sustained: shed actors   |
            |   queue == 0 : STARVED           -> upstream bottleneck      |
      0.50  +--------------------------------------------------------------+  enter upper zone
            | OVER_PROVISIONED EXIT BAND                                   |
            |   prev OVER_PROVISIONED holds zone until ratio < 0.35        |
      0.35  +--------------------------------------------------------------+  exit upper zone
            | NORMAL                                                       |
            |   capacity and demand are roughly balanced                   |
      0.1725+--------------------------------------------------------------+  exit saturated zone
            | SATURATION EXIT BAND                                         |
            |   prev SATURATED/CRITICAL holds zone until ratio >= 0.1725   |
      0.15  +--------------------------------------------------------------+  enter saturated zone
            | SATURATED                                                    |
            |   sustained signal -> additive scale-up                      |
      0.05  +--------------------------------------------------------------+  enter critical zone
            | SATURATED_CRITICAL                                           |
            |   immediate signal -> multiplicative scale-up path           |
      0.00  +--------------------------------------------------------------+

Boundary rules:

  - ``ratio < activation_threshold`` -> ``SATURATED_CRITICAL``.
  - ``ratio < saturation_boundary`` -> ``SATURATED``.
  - ``ratio >= over_provisioned_boundary`` -> upper idle zone.
  - Everything between saturation and upper idle -> ``NORMAL``.

Hysteresis only changes the boundary for the state being exited:

  - ``SATURATED`` / ``SATURATED_CRITICAL``:
    ``saturation_boundary *= 1 + saturation_deadband_pct``.
  - ``OVER_PROVISIONED``:
    ``over_provisioned_boundary *= 1 - over_provisioned_deadband_pct``.
  - ``STARVED``: no hysteresis; queue state can change immediately.

Defined as a pure function so it can be tested in isolation,
exercised on synthetic samples, and reused by future operators
(simulator, replay harness) without coupling to scheduler state.
"""

from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def classify(
    *,
    slots_empty_ratio_ewma: float,
    input_queue_depth: int,
    prev_state: StageState,
    config: SaturationAwareStageConfig,
) -> StageState:
    """Classify a stage's current cycle by its saturation signals.

    Args:
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in
            ``[0.0, 1.0]``. ``0.0`` means every slot is busy;
            ``1.0`` means every slot is free.
        input_queue_depth: Number of tasks waiting upstream of this
            stage. Used to disambiguate ``OVER_PROVISIONED`` (free
            slots, work pending elsewhere) from ``STARVED`` (free
            slots, nothing to do).
        prev_state: The classifier output from the previous cycle.
            Drives the hysteresis bands so the result is stable
            against noisy samples on either side of a threshold.
        config: Per-stage configuration carrying the threshold and
            deadband values.

    Returns:
        The current cycle's classifier zone.

    """
    if slots_empty_ratio_ewma < config.activation_threshold:
        return StageState.SATURATED_CRITICAL

    saturation_boundary = config.saturation_threshold
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
