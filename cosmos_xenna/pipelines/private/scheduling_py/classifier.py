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

Maps ``(slots_empty_ratio_ewma, input_queue_depth, pressure_ewma,
prev_state)`` plus thresholds + deadband config to a ``StageState``
zone. Defined as a pure function so it is testable in isolation and
reusable by simulators / replay harnesses.

Two-layer decision (06-backlog-time-signal.md):

1.  **Slot-pin gate**: the existing utilisation thresholds
    (``activation_threshold``, ``saturation_threshold``,
    ``over_provisioned_threshold``) decide which branch the cycle
    enters. This preserves the historical zone shape and the
    asymmetric deadband for hysteresis.

2.  **Pressure demotion**: once a slot-pin branch is selected, the
    smoothed MFI pressure
    (``utilisation * normalized_backlog``) acts as a demotion gate:

    *   ``SATURATED_CRITICAL`` requires ``pressure_ewma > pressure_critical_threshold``;
        otherwise the slot pin is treated as a transient burst and
        the classifier falls through to the next branch.
    *   ``SATURATED`` requires ``pressure_ewma > pressure_saturation_threshold``;
        otherwise the queue is draining despite busy slots and the
        classifier returns ``NORMAL``.
    *   ``OVER_PROVISIONED`` is demoted to ``NORMAL`` when
        ``pressure_ewma > pressure_normal_threshold`` -- the queue is
        stuck elsewhere (downstream bottleneck) and shrinking would
        only make the downstream stall worse.

Hysteresis (saturation deadband, over-provisioned deadband) applies
to the slot-pin gate as before. Pressure thresholds carry no deadband
because the EWMA already smooths the raw composite signal.

The escape hatch ``enable_backlog_time_classifier=False`` reverts the
stage to the legacy slot-only behaviour for workloads pre-tuned
against the utilisation-only signal.
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
    pressure_ewma: float | None = None,
) -> StageState:
    """Classify a stage's current cycle by its saturation signals.

    Args:
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in
            ``[0, 1]``.
        input_queue_depth: Tasks waiting upstream of this stage;
            disambiguates ``OVER_PROVISIONED`` from ``STARVED``.
        pressure_ewma: Smoothed MFI pressure scalar in
            ``[0.0, BACKLOG_CAP]`` or ``None``. ``None`` means no
            pressure value is available yet (``run_per_stage_pipeline``
            short-circuited before computing it because the slot
            signal was unavailable, or the helper-direct test path
            chose to skip the helper). The classifier preserves
            slot-only behaviour in the missing-pressure case so a
            transient signal gap cannot suppress a saturation
            response.
        prev_state: The previous cycle's zone (drives the slot-pin
            hysteresis bands).
        saturation_threshold: Empty-slot fraction below which the
            stage's slot-pin gate enters ``SATURATED``. Strictly
            positive and strictly less than
            ``config.over_provisioned_threshold``.
        activation_threshold: Empty-slot fraction below which the
            stage's slot-pin gate enters ``SATURATED_CRITICAL``.
            Strictly less than ``saturation_threshold``.
        config: Per-stage configuration carrying the deadbands, the
            over-provisioned threshold, the three pressure thresholds
            (``pressure_critical_threshold``,
            ``pressure_saturation_threshold``,
            ``pressure_normal_threshold``), and the escape-hatch flag
            ``enable_backlog_time_classifier``.

    Returns:
        The current cycle's classifier zone.

    """
    if not config.enable_backlog_time_classifier:
        return _classify_slot_only(
            slots_empty_ratio_ewma=slots_empty_ratio_ewma,
            input_queue_depth=input_queue_depth,
            prev_state=prev_state,
            saturation_threshold=saturation_threshold,
            activation_threshold=activation_threshold,
            config=config,
        )

    if slots_empty_ratio_ewma < activation_threshold:
        if pressure_ewma is None:
            # Missing pressure -> preserve legacy CRITICAL behaviour. The
            # alternative (demote because we cannot prove the burst is
            # genuine) would silently delay scale-up and is the riskier
            # default for production.
            return StageState.SATURATED_CRITICAL
        if pressure_ewma > config.pressure_critical_threshold:
            return StageState.SATURATED_CRITICAL
        # Slot-pinned but pressure low -> transient burst. Fall through
        # to the SATURATED gate; the same pressure-demotion rule there
        # may demote further to NORMAL.

    saturation_boundary = saturation_threshold
    if prev_state in (StageState.SATURATED, StageState.SATURATED_CRITICAL):
        saturation_boundary *= 1.0 + config.saturation_deadband_pct
    if slots_empty_ratio_ewma < saturation_boundary:
        if pressure_ewma is None:
            return StageState.SATURATED
        if pressure_ewma > config.pressure_saturation_threshold:
            return StageState.SATURATED
        # Slot-pinned but pressure low -> queue drains despite busy
        # slots. Demote to NORMAL via the fall-through (no SATURATED
        # response, no scale-up).
        return StageState.NORMAL

    over_provisioned_boundary = config.over_provisioned_threshold
    if prev_state == StageState.OVER_PROVISIONED:
        over_provisioned_boundary *= 1.0 - config.over_provisioned_deadband_pct
    if slots_empty_ratio_ewma >= over_provisioned_boundary:
        if input_queue_depth == 0:
            return StageState.STARVED
        if pressure_ewma is not None and pressure_ewma > config.pressure_normal_threshold:
            # Idle slots BUT queue stuck elsewhere (downstream bottleneck);
            # shrinking this stage would worsen the downstream stall. The
            # pressure rises because utilisation drops while the queue
            # holds steady -- the product still exceeds the demotion gate.
            return StageState.NORMAL
        return StageState.OVER_PROVISIONED

    return StageState.NORMAL


def _classify_slot_only(
    *,
    slots_empty_ratio_ewma: float,
    input_queue_depth: int,
    prev_state: StageState,
    saturation_threshold: float,
    activation_threshold: float,
    config: SaturationAwareStageConfig,
) -> StageState:
    """Legacy utilisation-only classifier for the escape-hatch path.

    Reproduces the pre-MFI behaviour exactly. Used when
    ``config.enable_backlog_time_classifier`` is ``False`` to preserve
    a stable contract for workloads that were pre-tuned against the
    slot-ratio signal alone.
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
