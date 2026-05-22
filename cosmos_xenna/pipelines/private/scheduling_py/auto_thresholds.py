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

"""Auto-derived classifier thresholds for the saturation-aware scheduler.

The classifier consumes two thresholds whose correct values depend
on each stage's per-actor concurrency ``c = slots_per_actor``: an
Erlang-C M/M/c queue hits its response-time knee at a different
utilisation for ``c=1`` vs ``c=8`` vs ``c=64``. A single fixed
default would be silently wrong across stages with different ``c``.

A textbook M/M/c queue uses ``c = slots_per_actor * num_workers``
(total concurrent servers across the stage), but the threshold is
fixed at first-cycle resolution and must NOT shift as the autoscaler
adds or removes workers. Using ``slots_per_actor`` alone keeps the
threshold stable across autoscale cycles at the cost of calibrating
against per-actor (rather than per-stage) concurrency.

The Halfin-Whitt heavy-traffic formula::

    saturation := clamp(
        saturation_aggressiveness / sqrt(slots_per_actor),
        auto_threshold_min,
        auto_threshold_max,
    )
    activation := saturation * activation_to_saturation_ratio

The ``saturation_aggressiveness`` knob is the Halfin-Whitt
parameter ``beta``; higher values trigger scale-up sooner.

Resolution happens lazily on the first ``autoscale()`` cycle, where
each stage's runtime ``slots_per_worker`` from ``ProblemStageState``
becomes available. Subsequent cycles reuse the resolved value;
mid-run changes to ``Solution.slots_per_worker`` do NOT re-resolve
(re-resolution would invalidate the EWMA / streak history tuned to
the original threshold band). Operators who deliberately reshape a
stage and want new thresholds must restart the pipeline.
"""

import math

import attrs

from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@attrs.frozen
class ResolvedThresholds:
    """Per-stage classifier thresholds plus their resolution provenance.

    Attributes:
        saturation_threshold: Empty-slot fraction below which the stage
            is classified ``SATURATED``.
        activation_threshold: Empty-slot fraction below which the stage
            is classified ``SATURATED_CRITICAL``. Always strictly less
            than ``saturation_threshold``.
        saturation_aggressiveness: The Halfin-Whitt ``beta`` from the
            ``K/sqrt(c)`` formula at the time this record was built.
        slots_per_actor: The slot count used in the ``sqrt(c)``
            denominator at resolution time.
        saturation_threshold_was_overridden: ``True`` when the operator
            explicitly pinned ``saturation_threshold`` on the config
            (the formula and clamps were bypassed for that field).
        activation_threshold_was_overridden: ``True`` when the operator
            explicitly pinned ``activation_threshold`` on the config.

    """

    saturation_threshold: float
    activation_threshold: float
    saturation_aggressiveness: float
    slots_per_actor: int
    saturation_threshold_was_overridden: bool
    activation_threshold_was_overridden: bool


def _resolve_auto_thresholds(
    stage_cfg: SaturationAwareStageConfig,
    slots_per_actor: int,
    *,
    aggressiveness_override: float | None = None,
) -> ResolvedThresholds:
    """Resolve the classifier thresholds for one stage.

    Applies the override hierarchy: an explicit numeric value pins;
    ``None`` auto-derives via ``K/sqrt(c)`` clamped to
    ``[auto_threshold_min, auto_threshold_max]``.

    Args:
        stage_cfg: Per-stage configuration.
        slots_per_actor: M/M/c slot count for the stage. Must be
            ``>= 1``.
        aggressiveness_override: Effective ``K`` to use in place of
            ``stage_cfg.saturation_aggressiveness`` for the formula.
            ``None`` (the default) uses the config value. Callers that
            apply runtime adjustments (e.g. a regime-aware lift) pass
            the adjusted value here without mutating the config.

    Returns:
        Resolved thresholds plus the provenance needed to log them.

    Raises:
        ValueError: If ``slots_per_actor < 1``, or if the resolved
            triple does not satisfy
            ``activation < saturation < over_provisioned_threshold``.

    """
    if slots_per_actor < 1:
        msg = f"slots_per_actor must be >= 1, got {slots_per_actor}"
        raise ValueError(msg)

    aggressiveness = (
        aggressiveness_override if aggressiveness_override is not None else stage_cfg.saturation_aggressiveness
    )

    pinned_saturation = stage_cfg.saturation_threshold
    if pinned_saturation is not None:
        saturation = pinned_saturation
        saturation_was_overridden = True
    else:
        raw = aggressiveness / math.sqrt(slots_per_actor)
        saturation = max(stage_cfg.auto_threshold_min, min(raw, stage_cfg.auto_threshold_max))
        saturation_was_overridden = False

    pinned_activation = stage_cfg.activation_threshold
    if pinned_activation is not None:
        activation = pinned_activation
        activation_was_overridden = True
    else:
        activation = saturation * stage_cfg.activation_to_saturation_ratio
        activation_was_overridden = False

    over_provisioned = stage_cfg.over_provisioned_threshold
    if not (activation < saturation < over_provisioned):
        msg = (
            f"resolved thresholds violate zone ordering: "
            f"activation={activation} < saturation={saturation} < "
            f"over_provisioned={over_provisioned} required "
            f"(saturation_was_overridden={saturation_was_overridden}, "
            f"activation_was_overridden={activation_was_overridden}, "
            f"slots_per_actor={slots_per_actor})"
        )
        raise ValueError(msg)

    return ResolvedThresholds(
        saturation_threshold=saturation,
        activation_threshold=activation,
        saturation_aggressiveness=aggressiveness,
        slots_per_actor=slots_per_actor,
        saturation_threshold_was_overridden=saturation_was_overridden,
        activation_threshold_was_overridden=activation_was_overridden,
    )


def derive_utilization_target(resolved: ResolvedThresholds) -> float:
    """Return the operating utilisation target the capacity sizer drives toward.

    The classifier fires SATURATED when the empty-slot fraction drops
    below ``saturation_threshold``, so the matching utilisation
    boundary is ``1 - saturation_threshold``. Aiming the sizer at that
    boundary keeps the steady-state plan inside the NORMAL zone and
    avoids the classifier oscillating on the threshold.

    Args:
        resolved: Per-stage thresholds from ``_resolve_auto_thresholds``.

    Returns:
        Utilisation target in ``(0, 1)``.

    Raises:
        ValueError: If the derived value is not strictly inside ``(0, 1)``.

    """
    target = 1.0 - resolved.saturation_threshold
    if not (0.0 < target < 1.0):
        msg = (
            f"derived utilization_target {target} outside (0, 1) (saturation_threshold={resolved.saturation_threshold})"
        )
        raise ValueError(msg)
    return target
