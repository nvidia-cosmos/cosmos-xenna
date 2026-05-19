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

"""Worker selection helpers for the saturation-aware Phase D scale-down.

Phase D removes workers from stages whose per-stage classifier
output is OVER_PROVISIONED (negative intent from
``compute_delta``). Selection priority is age-descending: the
oldest worker is removed first. Older workers have had the most
opportunity to accumulate stale state (model weights drift,
allocator fragmentation), so removing them first is a low-risk
shrink ordering when no per-worker idle signal is yet available.

The full STORY-33 contract sorts by ``(host_gpu_used_fraction
ASC, idle_status DESC, age DESC)``: a per-worker idle signal
plus a host-GPU-loading signal are both required. Today
``ProblemWorkerGroupState`` exposes neither -- the shrink
ordering therefore degrades to age-only with the worker_id
tiebreaker for determinism. Adding the missing signals is a
separate iteration; until then the helper here is the canonical
shrink-selection contract for Phase D.
"""

from operator import itemgetter


def select_workers_to_remove_oldest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
) -> list[str]:
    """Pick ``delete_count`` workers to delete, oldest first.

    Sort key: ``(age DESC, worker_id ASC)``. Workers missing from
    ``worker_ages`` are treated as age 0 (newly observed). The
    ``worker_id`` tiebreaker keeps the choice deterministic when
    every worker has the same age. Mirrors the Phase A
    youngest-first helper but inverts the age key.

    Args:
        worker_ids: Worker ids in the stage's current snapshot.
        worker_ages: Cluster-wide worker ages (cycles since
            placement).
        delete_count: Number of workers to return. Clamped to
            ``len(worker_ids)``; non-positive values return ``[]``.

    Returns:
        The first ``delete_count`` worker ids of the oldest-first
        ordering.

    """
    if delete_count <= 0:
        return []
    ranked = sorted(
        ((worker_ages.get(wid, 0), wid) for wid in worker_ids),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return list(map(itemgetter(1), ranked[:delete_count]))
