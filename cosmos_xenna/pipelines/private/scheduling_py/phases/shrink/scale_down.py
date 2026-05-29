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

Selection priority for saturation-driven removal:
``(host_gpu_used_fraction ASC, idle DESC, age DESC, worker_id ASC)``.
GPU-fraction first maximises whole-GPU consolidation; idle next
avoids killing busy workers; oldest within the bucket retires
stale actor state; worker_id is the deterministic tiebreak.
The age key is INVERTED relative to operator-driven shrink (which
picks youngest first to reverse the most recent operator intent).

Independent worker-lifetime knobs (``worker_max_lifetime_m``,
``worker_restart_interval_m``) handle scheduled rotation regardless
of saturation; this helper only governs the order in which Phase D
selects victims among the workers eligible for removal this cycle.
"""

import math
from operator import itemgetter

from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError


def select_workers_to_remove_oldest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
    worker_used_slots: dict[str, int] | None = None,
    worker_host_gpu_used_fractions: dict[str, float] | None = None,
    excluded_worker_ids: frozenset[str] | None = None,
) -> list[str]:
    """Pick ``delete_count`` workers to delete using the consolidation-first sort.

    Sort key:
    ``(host_gpu_used_fraction ASC, idle DESC, age DESC, worker_id ASC)``
    where ``idle = (used_slots == 0)``. Missing maps degrade
    gracefully (age->0, slots->0, host_gpu->0.0). Workers in
    ``excluded_worker_ids`` are filtered out before sorting.
    Returns the first ``delete_count`` worker ids in the resulting
    ordering (or ``[]`` if the pool is empty post-filter).

    Raises:
        SchedulerInvariantError: A candidate worker (present in
            ``worker_ids`` and not excluded) carries a non-finite or
            negative ``host_gpu_used_fraction``. That value is
            corrupted runtime state, so it surfaces as a hard,
            non-absorbable scheduler-invariant failure (consistent
            with the rest of the saturation-aware error taxonomy)
            rather than a bare ``ValueError`` that an outer handler
            could mistake for a recoverable usage error.

    """
    if delete_count <= 0:
        return []
    used_slots = worker_used_slots or {}
    gpu_fractions = worker_host_gpu_used_fractions or {}
    excluded = excluded_worker_ids or frozenset()
    for wid in worker_ids:
        if wid in excluded:
            continue
        fraction = gpu_fractions.get(wid, 0.0)
        if math.isfinite(fraction) and fraction >= 0.0:
            continue
        msg = f"host_gpu_used_fraction for worker {wid!r} must be finite and >= 0, got {fraction!r}"
        raise SchedulerInvariantError(msg)
    ranked = sorted(
        (
            (
                gpu_fractions.get(wid, 0.0),
                used_slots.get(wid, 0) > 0,
                worker_ages.get(wid, 0),
                wid,
            )
            for wid in worker_ids
            if wid not in excluded
        ),
        key=lambda quad: (quad[0], quad[1], -quad[2], quad[3]),
    )
    return list(map(itemgetter(3), ranked[:delete_count]))
