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

"""Per-worker READY-state bookkeeping and warmup-derived signals.

Single owner of the first-seen-READY timestamp per worker plus the
two derived signals the scheduler reads each cycle:

  - Mature slot-signal aggregation that excludes workers still
    inside ``worker_warmup_measurement_grace_s`` so the classifier's
    EWMA sees only steady-state samples.
  - Donor warmup grace set that protects freshly-warmed workers
    from being yanked off their stage as cross-stage donors.

See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

from collections.abc import Callable, Sequence

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    data_structures as rust_data_structures,
)
from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


class WarmupTracker:
    """Per-worker READY-state bookkeeping for warmup-derived signals.

    Owns the per-worker first-seen-READY timestamp (worker id ->
    wall-clock seconds at which the worker first appeared in
    ``problem_state``). The tracker lives across cycles on the
    scheduler; ``reset()`` re-initialises it on every ``setup()``.

    The underlying timestamp map is intentionally private. Callers
    that need to inspect tracking state must go through the
    explicit query methods (``first_seen_for``, ``is_tracked``,
    ``tracked_ids``, ``tracked_count``, ``is_in_warmup_grace``)
    so the tracker is the unambiguous owner and a future move to
    a different storage shape does not break consumers.

    Consumers: scheduler pre-phase setup (``refresh``); the intent
    phase classifier pipeline (``filter_slot_signals``); the donor
    selection path (``excluded_ids``).
    """

    __slots__ = ("_first_seen_at",)

    def __init__(self) -> None:
        """Construct an empty tracker."""
        self._first_seen_at: dict[str, float] = {}

    def reset(self) -> None:
        """Reset all tracking to an empty state.

        Called from ``SaturationAwareScheduler.setup()`` so a new
        pipeline run starts with no stale warmup history.

        """
        self._first_seen_at = {}

    def first_seen_for(self, worker_id: str) -> float | None:
        """Return the first-seen-READY timestamp for ``worker_id`` or ``None``.

        ``None`` is the unambiguous "not tracked" sentinel; callers
        that need to distinguish "tracked but unset" from "untracked"
        cannot, because every successful ``refresh()`` writes a
        finite timestamp before exposing the id.
        """
        return self._first_seen_at.get(worker_id)

    def is_tracked(self, worker_id: str) -> bool:
        """Return True iff ``worker_id`` has a recorded first-seen-READY timestamp."""
        return worker_id in self._first_seen_at

    def tracked_ids(self) -> frozenset[str]:
        """Return a frozen snapshot of every tracked worker id."""
        return frozenset(self._first_seen_at)

    def tracked_count(self) -> int:
        """Return the number of tracked workers."""
        return len(self._first_seen_at)

    def is_in_warmup_grace(self, worker_id: str, *, grace_s: float, now: float) -> bool:
        """Return True iff ``worker_id`` is still inside its warmup grace window.

        A worker is in grace when the tracker has never seen it
        (``first_seen_for`` returns ``None``: treated as freshly
        observed) OR its observed age ``now - first_seen`` is below
        ``grace_s``. Negative or zero ``grace_s`` short-circuits to
        ``False`` so callers can pin "grace disabled" semantics on
        the gate.
        """
        if grace_s <= 0:
            return False
        first_seen = self._first_seen_at.get(worker_id)
        if first_seen is None:
            return True
        return (now - first_seen) < grace_s

    def refresh(
        self,
        problem_state: data_structures.ProblemState,
        *,
        now: float,
    ) -> None:
        """Stamp newly observed READY workers and drop disappeared ids.

        Carries existing timestamps forward; workers absent from the
        current snapshot are evicted because their pool actor no
        longer reports state. Eviction guarantees the dict stays
        bounded by the live worker set.

        """
        new_seen: dict[str, float] = {}
        for stage in problem_state.rust.stages:
            for worker_group in stage.worker_groups:
                new_seen[worker_group.id] = self._first_seen_at.get(worker_group.id, now)
        self._first_seen_at = new_seen

    def filter_slot_signals(
        self,
        runtime_stage: rust_data_structures.ProblemStageState,
        stage_cfg: SaturationAwareStageConfig,
        *,
        now: float,
    ) -> tuple[int, int]:
        """Re-aggregate slot signals dropping workers in the measurement-grace window.

        Excludes workers whose ready age is below
        ``worker_warmup_measurement_grace_s`` so the EWMA observes
        only steady-state samples. SPMD groups use
        ``slots_per_worker * len(worker_group.resources)`` as their
        per-group capacity to avoid a SATURATED-biased empties
        count. Returns the unfiltered totals when grace <= 0 or no
        ``worker_groups`` are present.

        """
        grace_s = stage_cfg.worker_warmup_measurement_grace_s
        if grace_s <= 0 or not runtime_stage.worker_groups:
            return runtime_stage.num_used_slots, runtime_stage.num_empty_slots
        slots_per_worker = runtime_stage.slots_per_worker
        mature_used = 0
        mature_empty = 0
        for worker_group in runtime_stage.worker_groups:
            first_seen = self._first_seen_at.get(worker_group.id)
            if first_seen is None or (now - first_seen) < grace_s:
                continue
            # SPMD-aware capacity: per-group capacity is slots_per_worker * actor_count.
            # actor_count is len(worker_group.resources): one WorkerResourcesInternal per
            # SPMD actor (= 1 for non-SPMD groups, = K for K-way SPMD).
            actor_count = len(worker_group.resources)
            group_capacity = slots_per_worker * actor_count
            used = worker_group.num_used_slots
            mature_used += used
            mature_empty += max(0, group_capacity - used)
        return mature_used, mature_empty

    def excluded_ids(
        self,
        worker_ids_by_stage: list[list[str]],
        stage_names: Sequence[str],
        resolve_stage_cfg: Callable[[str], SaturationAwareStageConfig],
        *,
        now: float,
    ) -> frozenset[str]:
        """Return the cluster-wide donor-warmup grace excluded id set.

        A worker is excluded when its ready age is below the stage's
        ``donor_warmup_grace_s``. Workers absent from the tracker
        are treated as in warmup (defensive). Floor-mode donor
        selection does NOT consult this set -- deadlocking on
        warmup-protected donors is worse than killing a young
        donor.

        """
        excluded: set[str] = set()
        for stage_index, worker_ids in enumerate(worker_ids_by_stage):
            stage_name = stage_names[stage_index]
            stage_cfg = resolve_stage_cfg(stage_name)
            grace_s = stage_cfg.donor_warmup_grace_s
            if grace_s <= 0:
                continue
            for worker_id in worker_ids:
                first_seen = self._first_seen_at.get(worker_id)
                if first_seen is None or (now - first_seen) < grace_s:
                    excluded.add(worker_id)
        return frozenset(excluded)


__all__ = ["WarmupTracker"]
