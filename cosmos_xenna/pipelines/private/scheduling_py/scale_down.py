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
output is over-provisioned. Selection priority is, in order of
precedence:

  1. ``host_gpu_used_fraction`` ASC. Workers placed on GPUs whose
     total used fraction is lowest are removed first, because
     those GPUs are most likely to become fully unallocated after
     deletion. Freeing a whole GPU is worth more than preserving
     an arbitrary fractional placement, since downstream
     whole-GPU stages can then claim the freed hardware.
  2. ``idle`` DESC. Workers with zero in-flight task slots are
     preferred over busy workers within the same GPU-fraction
     bucket.
  3. ``age`` DESC. Within an idle and GPU-fraction bucket, the
     oldest workers are removed first to retire stale state such
     as model-cache drift, allocator fragmentation history, and
     leaked references.
  4. ``worker_id`` ASC. A deterministic tiebreaker so the
     selection is identical across processes given the same
     inputs.

The age key (3) is intentionally inverted relative to the
operator-driven shrink paths (manual delete and donor fallback
both pick the youngest eligible worker first):

  - Operator-driven removal targets workers the operator just
    added, so reversing the most recent intent is the cheapest
    correction.
  - Saturation-driven removal targets a stage that has been
    over-provisioned for many cycles, so retiring the oldest
    worker is the higher-value decision: long-running actors
    accumulate stale state and rotating them off improves cluster
    hygiene over time.

Independent worker-lifetime knobs (``worker_max_lifetime_m``,
``worker_restart_interval_m``) handle scheduled rotation regardless
of saturation; this helper only governs the order in which Phase D
selects victims among the workers eligible for removal this cycle.
"""

import math
from operator import itemgetter


def select_workers_to_remove_oldest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
    worker_used_slots: dict[str, int] | None = None,
    worker_host_gpu_used_fractions: dict[str, float] | None = None,
) -> list[str]:
    """Pick ``delete_count`` workers to delete using the consolidation-first sort.

    The sort key is
    ``(host_gpu_used_fraction ASC, idle DESC, age DESC, worker_id ASC)`` where
    ``idle = (used_slots == 0)``.

    Defaults for missing or omitted maps:

      - Workers missing from ``worker_ages`` are treated as age 0
        (newly observed; sort to the bottom of the age bucket).
      - Workers missing from ``worker_used_slots`` are treated as
        idle (zero used slots). When ``worker_used_slots`` is
        omitted entirely, every worker is treated as idle and the
        idle key collapses to a constant.
      - Workers missing from ``worker_host_gpu_used_fractions``
        are treated as host-GPU-used-fraction 0.0 (most
        consolidatable; sort to the top of the consolidation
        bucket). When ``worker_host_gpu_used_fractions`` is
        omitted entirely, the consolidation key collapses to a
        constant and the helper degrades to the prior
        ``(idle, age, worker_id)`` ordering. This matches the
        behavior expected by CPU-only stages (no GPU footprint
        means consolidation is irrelevant).

    Args:
        worker_ids: Worker ids in the stage's current snapshot.
        worker_ages: Cluster-wide worker ages (cycles since
            placement).
        delete_count: Number of workers to return. Clamped to
            ``len(worker_ids)``; non-positive values return ``[]``.
        worker_used_slots: Optional mapping ``{worker_id: used_slots}``.
            Idle workers (``used_slots == 0``) sort before busy ones.
        worker_host_gpu_used_fractions: Optional mapping
            ``{worker_id: host_gpu_used_fraction}`` where the value
            is the total used fraction (0.0 to 1.0+) of the GPU(s)
            the worker is placed on, summed across every stage that
            holds an allocation on those GPUs. For multi-GPU
            workers this is the maximum across the worker's GPU
            allocations (the GPU least likely to become fully
            unallocated dominates).

    Returns:
        The first ``delete_count`` worker ids of the
        consolidation-first ordering.

    """
    if delete_count <= 0:
        return []
    used_slots = worker_used_slots or {}
    gpu_fractions = worker_host_gpu_used_fractions or {}
    for worker_id, fraction in gpu_fractions.items():
        if math.isfinite(fraction) and fraction >= 0.0:
            continue
        msg = f"host_gpu_used_fraction for worker {worker_id!r} must be finite and >= 0, got {fraction!r}"
        raise ValueError(msg)
    ranked = sorted(
        (
            (
                gpu_fractions.get(wid, 0.0),
                used_slots.get(wid, 0) > 0,
                worker_ages.get(wid, 0),
                wid,
            )
            for wid in worker_ids
        ),
        key=lambda quad: (quad[0], quad[1], -quad[2], quad[3]),
    )
    return list(map(itemgetter(3), ranked[:delete_count]))
