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
output is over-provisioned. Selection priority is
**idle-first, oldest-first**: workers with zero used slots are
preferred over busy workers, and within the same idle bucket the
oldest workers are removed first. The ``worker_id`` tiebreaker keeps
the choice deterministic across cycles.

The age key is intentionally inverted relative to the operator-driven
shrink paths (manual delete and donor fallback both pick the
**youngest** eligible worker first):

  - Operator-driven removal targets workers the operator just added,
    so reversing the most recent intent is the cheapest correction.
  - Saturation-driven removal targets a stage that has been
    over-provisioned for many cycles, so retiring the **oldest**
    worker is the higher-value decision: long-running actors
    accumulate stale state (model-cache drift, allocator
    fragmentation, leaked references) and rotating them off improves
    cluster hygiene over time.

Independent worker-lifetime knobs (``worker_max_lifetime_m``,
``worker_restart_interval_m``) handle scheduled rotation regardless
of saturation; this helper only governs the order in which Phase D
selects victims among the workers eligible for removal this cycle.
"""

from operator import itemgetter


def select_workers_to_remove_oldest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
    worker_used_slots: dict[str, int] | None = None,
) -> list[str]:
    """Pick ``delete_count`` workers to delete, idle-first then oldest-first.

    Sort key: ``(idle DESC, age DESC, worker_id ASC)`` where
    ``idle = (used_slots == 0)``. Workers missing from
    ``worker_ages`` are treated as age 0 (newly observed); workers
    missing from ``worker_used_slots`` are treated as 0 used slots
    (i.e. idle). When ``worker_used_slots`` is omitted entirely, every
    worker is idle and the sort collapses to ``(age DESC, worker_id ASC)``.

    Args:
        worker_ids: Worker ids in the stage's current snapshot.
        worker_ages: Cluster-wide worker ages (cycles since
            placement).
        delete_count: Number of workers to return. Clamped to
            ``len(worker_ids)``; non-positive values return ``[]``.
        worker_used_slots: Optional mapping ``{worker_id: used_slots}``.
            Idle workers (``used_slots == 0``) sort before busy ones.

    Returns:
        The first ``delete_count`` worker ids of the
        idle-first oldest-first ordering.

    """
    if delete_count <= 0:
        return []
    used_slots = worker_used_slots or {}
    ranked = sorted(
        ((used_slots.get(wid, 0) > 0, worker_ages.get(wid, 0), wid) for wid in worker_ids),
        key=lambda triple: (triple[0], -triple[1], triple[2]),
    )
    return list(map(itemgetter(2), ranked[:delete_count]))
