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

Linear streaming chains - ``Problem.rust.stages`` is in
topological order, so DAG depth = list position. Throughput is
bounded by the slowest stage; the priority decides who widens
first when intents compete for cluster headroom.
``compute_grow_priority_order`` encodes the three-tier hierarchy:
bottleneck-aware (``D_k`` desc) when engaged; DAG-depth desc
(downstream-first) when the toggle is on; problem order otherwise.
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

    Three-tier hierarchy: bottleneck-engaged ordering wins when
    available; otherwise the DAG-depth toggle decides; otherwise
    problem order. ``d_k_by_stage`` is consulted only when
    ``bottleneck_engaged`` is True; cold-start stages (missing or
    non-finite ``D_k``) sort last so they never preempt the
    bottleneck.

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
