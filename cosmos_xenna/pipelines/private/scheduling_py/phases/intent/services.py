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

"""Service value object passed to :class:`IntentPhase`."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.cluster.measurements import MeasurementCollector
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.capacity import CapacityModel
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageStateMap
from cosmos_xenna.pipelines.private.scheduling_py.warmup.warmup import WarmupTracker


@attrs.frozen
class IntentServices:
    """Service view consumed by :class:`IntentPhase` (signed worker-count intent).

    The intent phase consumes throughput samples from the
    measurement collector, runs each non-finished stage through
    the per-stage classifier pipeline, and writes the resulting
    deltas onto the cycle. The per-stage capacity-sizing
    arithmetic is delegated to the injected :class:`CapacityModel`,
    the single owner of M/M/c capacity-target computation.

    Attributes:
        pipeline: Immutable post-setup pipeline shape.
        pipeline_name: Pipeline tag for logs / labels.
        capacity: M/M/c capacity-target sizer; constructed once
            per scheduler in ``setup()``.
        measurements: Thread-safe measurement collector.
        stage_states: Per-stage runtime-state map (live, mutated
            in place by the classifier / growth path).
        recommendation_histories: Per-stage stabilization-window
            buffers keyed by stage name.
        warmup: Wall-clock warmup-grace tracker; gates per-stage
            classifier samples against fresh-slot signals.

    """

    pipeline: PipelineModel
    pipeline_name: str
    capacity: CapacityModel
    measurements: MeasurementCollector
    stage_states: StageStateMap
    recommendation_histories: dict[str, RecommendationHistory]
    warmup: WarmupTracker


__all__ = ("IntentServices",)
