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

"""Phase C grow-priority ordering for the saturation-aware scheduler.

Cosmos-Xenna pipelines are linear streaming chains: the stage list
in ``Problem.rust.stages`` is already in topological order with
``stages[0]`` upstream and ``stages[-1]`` downstream. DAG depth is
therefore identical to the stage's position in the list.

Pipeline throughput is bounded by the slowest stage in any cycle.
The grow-priority order picks the stage to widen first when the
autoscaler has multiple positive intents and limited cluster
headroom. The unified helper :func:`compute_grow_priority_order`
encodes this hierarchy in a single function:

  1. Bottleneck-aware (D_k descending, depth descending tiebreak)
     when the bottleneck gate is engaged for the cycle.
  2. DAG-depth descending (downstream-first) when the engaged path
     is unavailable but the legacy DAG toggle is on.
  3. Problem order otherwise.
"""

import math
from collections.abc import Mapping

from cosmos_xenna.pipelines.private import data_structures


def compute_grow_priority_order(
    problem: data_structures.Problem,
    *,
    bottleneck_engaged: bool,
    d_k_by_stage: Mapping[str, float],
    enable_dag_priority: bool,
) -> list[int]:
    """Return stage indices for Phase C growth in priority order.

    The hierarchy is intentionally explicit so callers do not have to
    re-implement it: bottleneck-engaged ordering wins when available;
    otherwise the legacy DAG-depth toggle decides; otherwise problem
    order. ``d_k_by_stage`` is consulted only when
    ``bottleneck_engaged`` is True. Stages without a finite ``D_k``
    sort last (depth-descending among themselves) so cold-start
    stages never preempt the bottleneck.

    Args:
        problem: The frozen pipeline problem. An empty stage list
            returns an empty list.
        bottleneck_engaged: True when the bottleneck-aware sort is
            active; the caller must AND its config toggle with the
            ``BottleneckIdentity.engaged`` verdict so cold-start
            cycles fall back automatically.
        d_k_by_stage: Mapping from stage name to EWMA-smoothed
            ``D_k`` in seconds. Missing entries and non-finite
            values are treated as cold-start (sort last).
        enable_dag_priority: Legacy DAG-depth toggle; only consulted
            when ``bottleneck_engaged`` is False.

    Returns:
        Stage indices in priority order. Each index is a valid
        offset into ``problem.rust.stages``.
    """
    num_stages = len(problem.rust.stages)
    if num_stages == 0:
        return []

    if bottleneck_engaged:
        stage_names = [stage.name for stage in problem.rust.stages]

        def sort_key(idx: int) -> tuple[float, int]:
            d_k = d_k_by_stage.get(stage_names[idx], math.nan)
            d_key = -d_k if math.isfinite(d_k) and d_k > 0.0 else math.inf
            return (d_key, -idx)

        return sorted(range(num_stages), key=sort_key)

    if enable_dag_priority:
        return list(range(num_stages - 1, -1, -1))

    return list(range(num_stages))
