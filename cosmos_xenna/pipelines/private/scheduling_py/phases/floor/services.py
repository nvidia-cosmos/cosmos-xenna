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

"""Service value object passed to :class:`FloorPhase`."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.executor import DonorBackedAddExecutor
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.capacity import FloorCalculator
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.floor_stuck_counters import FloorStuckCounters
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageStateMap


@attrs.frozen
class FloorServices:
    """Service view consumed by :class:`FloorPhase` (minimum-worker enforcement).

    The floor phase enforces ``target_min`` per stage and falls back
    to the cross-stage donor coordinator when the cluster is full;
    it needs the floor calculator to compute the per-stage target,
    the floor-stuck counter store for the grace gating, and the
    pipeline / pipeline-name basics shared by every phase. The
    donor-backed receiver-add transaction is constructed once per
    scheduler in ``setup()`` and injected here so the phase reuses
    one frozen instance across every cycle.

    Attributes:
        pipeline: Immutable post-setup pipeline shape.
        pipeline_name: Pipeline tag for logs / labels.
        floors: Per-stage floor calculator (``max(min_workers,
            min_workers_per_node * num_nodes)``).
        donor_executor: Shared direct-add -> donor-acquire ->
            retry transaction, pre-wired with the floor-mode
            policy. Owns the floor allocation-failure gate.
        floor_stuck_counters: Per-stage stuck-cycle counter store.
        stage_states: Per-stage runtime-state map; passed to
            ``cycle.view_for(stage_index, stage_states)``.

    """

    pipeline: PipelineModel
    pipeline_name: str
    floors: FloorCalculator
    donor_executor: DonorBackedAddExecutor
    floor_stuck_counters: FloorStuckCounters
    stage_states: StageStateMap


__all__ = ("FloorServices",)
