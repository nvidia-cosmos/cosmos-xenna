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

"""Service value object passed to :class:`BottleneckPhase`."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.cluster.measurements import MeasurementCollector
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_engagement_state import BottleneckEngagementState
from cosmos_xenna.pipelines.private.scheduling_py.state.sk_ewma_store import SkEwmaStore


@attrs.frozen
class BottleneckServices:
    """Service view consumed by :class:`BottleneckPhase` (per-cycle ``D_k`` refresh).

    The bottleneck phase reads service-time samples from the
    measurement collector, updates the cross-cycle ``S_k`` EWMA,
    recomputes ``D_k``, identifies the engaged bottleneck, and
    emits the engagement INFO log.

    Attributes:
        pipeline: Immutable post-setup pipeline shape.
        pipeline_name: Pipeline tag for logs / labels.
        measurements: Thread-safe measurement collector;
            ``consume_service_time_samples`` returns per-stage
            mean service times for this cycle.
        s_k_ewma: Per-stage intrinsic service-time EWMA store
            (mutated in place via :meth:`SkEwmaStore.update`).
        bottleneck_engagement_state: Streak-debounced engagement
            state; the INFO log fires once per persistent change.

    """

    pipeline: PipelineModel
    pipeline_name: str
    measurements: MeasurementCollector
    s_k_ewma: SkEwmaStore
    bottleneck_engagement_state: BottleneckEngagementState


__all__ = ("BottleneckServices",)
