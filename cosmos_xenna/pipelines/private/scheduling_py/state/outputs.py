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

"""Cycle-scoped value objects produced by individual phases.

Phases publish their outputs as immutable snapshots written
directly onto the typed ``AutoscaleCycle`` fields
(``cycle.bottleneck``, ``cycle.intent``). Downstream phases
consume the snapshots through the typed field rather than
re-deriving values from ``problem_state``. The runner does not
assign return values.

See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

from collections.abc import Mapping

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_identity import BottleneckIdentity


@attrs.frozen
class BottleneckSnapshot:
    """Per-cycle bottleneck identification result.

    Attributes:
        identity: Bottleneck stage identity for this cycle.
        d_k_now: Per-stage ``D_k = S_k / c_k``.
        effective_capacities: Per-stage effective ready capacity
            in service channels.
        channels_per_worker_group: Per-stage channels-per-group
            multiplier captured at cycle start.
        balance_score_start: Cluster balance score at cycle start;
            ``math.nan`` on cold start.

    """

    identity: BottleneckIdentity
    d_k_now: Mapping[str, float]
    effective_capacities: Mapping[str, int]
    channels_per_worker_group: Mapping[str, int]
    balance_score_start: float


@attrs.frozen
class IntentPlan:
    """Per-cycle classifier-derived signed worker deltas per stage.

    Produced by the intent phase and consumed by the saturation
    grow and shrink phases. Positive entries request growth;
    negative entries request shrink; zero is hold.

    Attributes:
        deltas: ``stage_name -> signed delta`` mapping for every
            non-manual stage participating in the current cycle.

    """

    deltas: Mapping[str, int]
