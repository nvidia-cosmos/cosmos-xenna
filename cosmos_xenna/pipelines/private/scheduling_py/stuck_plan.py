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


"""Stuck-plan detector for the saturation-aware Phase C grow path."""

import attrs
from ray.util.metrics import Counter, Gauge

from cosmos_xenna.utils import python_log as logger

STUCK_PLAN_ACTIVE_METRIC = "xenna_scheduler_stuck_plan_active"
STUCK_PLAN_CYCLES_TOTAL_METRIC = "xenna_scheduler_stuck_plan_cycles_total"

_STUCK_PLAN_ACTIVE_GAUGE = Gauge(
    STUCK_PLAN_ACTIVE_METRIC,
    description="1 when a stage's Phase C grow is stuck above the detection threshold; 0 otherwise.",
    tag_keys=("stage", "pipeline"),
)
_STUCK_PLAN_CYCLES_COUNTER = Counter(
    STUCK_PLAN_CYCLES_TOTAL_METRIC,
    description="Total cycles a stage has been stuck above the detection threshold.",
    tag_keys=("stage", "pipeline"),
)


@attrs.define
class StuckPlanDetector:
    """Per-stage latch promoting the stuck-plan WARN to a one-shot INFO."""

    _fired: dict[str, bool] = attrs.Factory(dict)

    def reset(self) -> None:
        """Drop the latch state."""
        self._fired.clear()

    def update(
        self,
        *,
        stage_name: str,
        stuck_cycles: int,
        threshold_cycles: int,
        last_intent: int,
        pipeline_name: str,
    ) -> None:
        """Reconcile the per-stage stuck state with the configured threshold.

        Args:
            stage_name: Stage being reported on.
            stuck_cycles: Current value of ``_stuck_plan_counters[stage_name]``.
            threshold_cycles: Detection threshold.
            last_intent: Last positive intent for the INFO log.
            pipeline_name: Value for the ``pipeline`` Prometheus tag.
        """
        tags = {"stage": stage_name, "pipeline": pipeline_name}
        is_stuck = stuck_cycles >= threshold_cycles
        was_fired = self._fired.get(stage_name, False)

        if is_stuck:
            _STUCK_PLAN_ACTIVE_GAUGE.set(1.0, tags=tags)
            _STUCK_PLAN_CYCLES_COUNTER.inc(tags=tags)
            if not was_fired:
                logger.info(
                    f"saturation-aware stuck plan: stage {stage_name!r} stuck "
                    f"for {stuck_cycles} cycles (threshold={threshold_cycles}, "
                    f"last_intent={last_intent}); growth blocked by cluster "
                    "placement and donor selection."
                )
                self._fired[stage_name] = True
            return

        _STUCK_PLAN_ACTIVE_GAUGE.set(0.0, tags=tags)
        if was_fired and stuck_cycles == 0:
            logger.info(f"saturation-aware stuck plan: stage {stage_name!r} recovered; growth is no longer blocked.")
            self._fired[stage_name] = False


__all__ = [
    "STUCK_PLAN_ACTIVE_METRIC",
    "STUCK_PLAN_CYCLES_TOTAL_METRIC",
    "StuckPlanDetector",
]
