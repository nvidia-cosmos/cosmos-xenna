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

"""Cluster-wide GPU fraction aggregation helpers (``GpuFractionMap``).

Two pure helpers extracted from ``SaturationAwareScheduler`` so
the cluster-wide GPU-allocation arithmetic lives next to its
tests, not on the scheduler facade:

- ``aggregate_host_gpu_used_fractions`` - sums each allocation's
  ``used_fraction`` by ``(node_name, gpu_offset)`` across every
  stage's worker group. Phase D scale-down prefers removing
  workers from the least-loaded GPUs (most likely to become
  fully unallocated) and reads this map as the cluster-wide
  ground truth.

- ``project_stage_worker_fractions`` - projects the cluster-wide
  map onto a single stage's workers by taking the per-worker max
  across its allocations. The most-loaded GPU is the binding
  constraint for whole-GPU recovery, so the per-worker max is
  what donor selection consults when ranking eligible donors.

Both helpers are stateless functions, not methods on a class:
they have no cross-cycle state and no construction-time
dependencies, so wrapping them in an ``@attrs.frozen`` class
would only add noise.
"""

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    data_structures as rust_data_structures,
)
from cosmos_xenna.pipelines.private import data_structures


def aggregate_host_gpu_used_fractions(
    problem_state: data_structures.ProblemState,
) -> dict[tuple[str, int], float]:
    """Aggregate the cycle-start used fraction of every GPU in the cluster.

    Sums each allocation's ``used_fraction`` by
    ``(node_name, gpu_offset)``. Phase D scale-down prefers
    removing workers from the least-loaded GPUs (most likely to
    become fully unallocated). Cycle-start approximation:
    intra-cycle mutations are not reflected. Returns
    ``{(node_name, gpu_offset): total_used_fraction}``;
    unallocated GPUs are absent and callers default to 0.0.

    """
    fraction_map: dict[tuple[str, int], float] = {}
    for stage_state in problem_state.rust.stages:
        for worker_group in stage_state.worker_groups:
            for resource in worker_group.resources:
                node = resource.node
                for gpu_alloc in resource.gpus:
                    key = (node, gpu_alloc.offset)
                    fraction_map[key] = fraction_map.get(key, 0.0) + float(gpu_alloc.used_fraction)
    return fraction_map


def project_stage_worker_fractions(
    *,
    runtime_stage: rust_data_structures.ProblemStageState,
    host_gpu_used_fractions: dict[tuple[str, int], float],
) -> dict[str, float]:
    """Project the cluster-wide GPU fraction map onto a single stage's workers.

    Returns the per-worker max GPU fraction across the worker's
    allocations; the most-loaded GPU is the binding constraint
    for whole-GPU recovery. CPU-only workers and workers whose
    GPUs are absent from the input map default to 0.0.

    Returns:
        ``{worker_id: per_worker_max_host_gpu_used_fraction}``.

    """
    out: dict[str, float] = {}
    for worker_group in runtime_stage.worker_groups:
        fractions = [
            host_gpu_used_fractions.get((resource.node, gpu_alloc.offset), 0.0)
            for resource in worker_group.resources
            for gpu_alloc in resource.gpus
        ]
        out[worker_group.id] = max(fractions, default=0.0)
    return out


__all__ = (
    "aggregate_host_gpu_used_fractions",
    "project_stage_worker_fractions",
)
