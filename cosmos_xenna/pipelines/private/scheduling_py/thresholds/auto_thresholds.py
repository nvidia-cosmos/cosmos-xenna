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

Per-stage M/M/c thresholds depend on ``c = slots_per_actor``
(Halfin-Whitt heavy traffic) and are calibrated against per-actor
concurrency so they stay stable across autoscale cycles. Defaults
come from ``saturation_aggressiveness`` (Halfin-Whitt ``beta``)
clamped to ``[auto_threshold_min, auto_threshold_max]``.
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

    Carries the resolved triple plus the inputs (Halfin-Whitt
    ``beta``, ``slots_per_actor``) and per-field override flags
    so the operator log can render full provenance.
    ``activation_threshold < saturation_threshold`` always holds.

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

    Override hierarchy: explicit numeric value pins; ``None``
    auto-derives via ``K/sqrt(c)`` clamped to
    ``[auto_threshold_min, auto_threshold_max]``.
    ``aggressiveness_override`` lets callers apply a runtime ``K``
    lift (e.g. regime-aware) without mutating the config.

    Raises:
        ValueError: ``slots_per_actor < 1``, or the resolved
            triple fails ``activation < saturation <
            over_provisioned_threshold``.

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

    Computes ``1 - saturation_threshold`` so the sizer aims at the
    boundary that triggers ``SATURATED``; keeps the steady-state
    plan inside ``NORMAL`` and avoids classifier oscillation.

    Raises:
        ValueError: Derived value not strictly inside ``(0, 1)``.

    """
    target = 1.0 - resolved.saturation_threshold
    if not (0.0 < target < 1.0):
        msg = (
            f"derived utilization_target {target} outside (0, 1) (saturation_threshold={resolved.saturation_threshold})"
        )
        raise ValueError(msg)
    return target
