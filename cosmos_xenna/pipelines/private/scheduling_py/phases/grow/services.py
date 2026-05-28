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

"""Service value object passed to :class:`SaturationGrowPhase`."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.executor import DonorBackedAddExecutor
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.capacity import (
    CeilingCalculator,
    FloorCalculator,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageStateMap
from cosmos_xenna.pipelines.private.scheduling_py.state.stuck_plan_ledger import StuckPlanLedger


@attrs.frozen
class GrowServices:
    """Service view + behaviour bundle consumed by :class:`SaturationGrowPhase`.

    The grow phase applies the per-stage positive intent as planner
    adds, falls back to the saturation-mode cross-stage donor
    coordinator on placement exhaustion, and advances the
    stuck-plan ledger per stage. ``set_stuck_plan_counter`` is a
    convenience wrapper around :meth:`StuckPlanLedger.record` that
    fills in the ``threshold_cycles`` and ``pipeline_name``
    parameters from this service's pipeline configuration; the
    Grow phase calls it with one ``(stage_name, value, last_intent)``
    triple per outcome.

    Attributes:
        pipeline: Immutable post-setup pipeline shape.
        pipeline_name: Pipeline tag for logs / labels.
        floors: Per-stage floor calculator (read for ``stage_floors``
            and the donor planning context).
        ceilings: Per-stage hard ceiling calculator (clamps positive
            intent to ``min(max_workers, max_workers_per_node * N)
            - current``).
        donor_executor: Shared direct-add -> donor-acquire ->
            retry transaction, pre-wired with the saturation-mode
            policy. Owns the grow allocation-failure gate.
        stuck_plan_ledger: Composite stuck-plan counter dict +
            structured-log detector; advanced via
            :meth:`set_stuck_plan_counter`.
        stage_states: Per-stage runtime-state map; passed to
            ``cycle.view_for(stage_index, stage_states)``.

    """

    pipeline: PipelineModel
    pipeline_name: str
    floors: FloorCalculator
    ceilings: CeilingCalculator
    donor_executor: DonorBackedAddExecutor
    stuck_plan_ledger: StuckPlanLedger
    stage_states: StageStateMap

    def set_stuck_plan_counter(self, stage_name: str, value: int, *, last_intent: int) -> None:
        """Advance the Grow stuck-plan ledger for a stage.

        Convenience wrapper that fills in ``threshold_cycles`` from
        ``self.pipeline.config.stuck_plan_detection_cycles`` and
        ``pipeline_name`` from ``self.pipeline_name`` so the Grow
        phase advances the stuck-plan ledger with one triple per
        outcome (zero on no-intent cycles, increment on stuck cycles,
        reset on successful cycles).

        Args:
            stage_name: The stage whose counter is being updated.
            value: The new counter value to record.
            last_intent: The per-stage signed intent that produced
                this update (forwarded to the detector so its
                structured log can distinguish a true stuck plan
                from an intent-driven reset).

        """
        self.stuck_plan_ledger.record(
            stage_name,
            value,
            last_intent=last_intent,
            threshold_cycles=self.pipeline.config.stuck_plan_detection_cycles,
            pipeline_name=self.pipeline_name,
        )


__all__ = ("GrowServices",)
