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

"""DAG-priority traversal helpers for the saturation-aware scheduler.

Cosmos-Xenna pipelines are linear streaming chains: the stage list
in ``Problem.rust.stages`` is already in topological order with
``stages[0]`` upstream and ``stages[-1]`` downstream. DAG depth is
therefore identical to the stage's position in the list.

Pipeline throughput is bounded by the tail stage; widening an
upstream stage only pushes the bottleneck downward unless the
downstream stage can absorb the new work. So when growth-driving
phases iterate stages, they walk the chain **downstream-first**
(deepest depth first) to spend any free capacity on the stage most
likely to be the bottleneck.
"""

from cosmos_xenna.pipelines.private import data_structures


def compute_dag_depth_order(problem: data_structures.Problem) -> list[int]:
    """Return the stage indices sorted by DAG depth descending.

    For Xenna's linear streaming pipelines, depth equals the stage's
    position in ``problem.rust.stages``. The returned list iterates
    from the deepest (last) stage down to the shallowest (first).

    Args:
        problem: The frozen pipeline problem. Must carry a non-empty
            ``rust.stages`` list; an empty list returns an empty list.

    Returns:
        Stage indices in downstream-first order. Each index is a
        valid offset into ``problem.rust.stages``.

    """
    num_stages = len(problem.rust.stages)
    return list(range(num_stages - 1, -1, -1))
