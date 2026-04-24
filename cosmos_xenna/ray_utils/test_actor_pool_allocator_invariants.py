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

"""Allocator-pool consistency invariants for ``ActorPool``.

1. **Allocator/pool keys stay in lockstep** through ``_add_worker_group``
   and ``_delete_worker_group``: divergence is the precondition for
   "allocator believes a GPU offset is free while the pool still has
   an actor on it".

2. **The kill path never touches the allocator.** Allocator releases go
   exclusively through ``_delete_worker_group``; ``_try_delete_*``
   helpers must not release slots directly.

3. **Continuous-mode ``rate_duration_s`` is per-actor, not per-slot.**
   The inter-completion interval stamped in ``_collect_continuous_async``
   yields a different rate-estimate semantics than batch mode (which
   leaves ``rate_duration_s = None`` and falls through to ``timing.process_dur``).
"""

import collections
from unittest.mock import MagicMock

import pytest

import cosmos_xenna.ray_utils.actor_pool as ap_module
from cosmos_xenna.utils import timing


def _make_mock_worker_group(worker_id: str, node_id: str = "node-A") -> MagicMock:
    """Return a minimal ``WorkerGroup`` mock that ``_add_worker_group`` can swallow.

    Only the attributes touched by the non-SPMD ``_add_worker_group``
    code path are populated: ``id``, ``allocations`` (single element
    with a ``node`` field for the kill-path test).
    """
    wg = MagicMock()
    wg.id = worker_id
    allocation = MagicMock()
    allocation.node = node_id
    wg.allocations = [allocation]
    return wg


class _TrackingAllocator:
    """In-process ``WorkerAllocator`` stand-in that records add/remove calls.

    Mirrors the contract relied on by ``_add_worker_group`` /
    ``_delete_worker_group`` but stores worker ids in a plain ``set``,
    so tests can assert the allocator's view in lockstep with
    ``ActorPool._worker_groups``.

    The class deliberately does NOT raise on duplicate ids or unknown
    ids - those error semantics are owned by the real Rust allocator
    and are exercised by ``test_adjust_actors_allocation_guard.py``.
    The contract here is purely "did the actor pool call us at all".
    """

    def __init__(self) -> None:
        self.added_ids: set[str] = set()
        self.removed_ids: list[str] = []
        self.add_calls: int = 0
        self.remove_calls: int = 0

    def add_worker(self, worker: MagicMock) -> None:
        """Record a worker addition by id."""
        self.add_calls += 1
        self.added_ids.add(worker.id)

    def remove_worker(self, worker_id: str) -> None:
        """Record a worker removal by id and discard from the live set."""
        self.remove_calls += 1
        self.removed_ids.append(worker_id)
        self.added_ids.discard(worker_id)

    @property
    def live_worker_ids(self) -> set[str]:
        """Worker ids currently held by the allocator (added minus removed)."""
        return set(self.added_ids)


def _make_pool_with_tracking_allocator(worker_groups: list[MagicMock]) -> MagicMock:
    """Build an ``ActorPool`` mock wired to a ``_TrackingAllocator``.

    Re-binds ``_add_worker_group`` / ``_adjust_actors`` from the real
    ``ActorPool`` so ``_worker_groups`` insertion runs end-to-end. The
    ``_create_actor_for_worker_group`` step stays mocked because it
    reaches into Ray; the invariants pinned here only care about the
    bookkeeping symmetry between the allocator and ``_worker_groups``.
    """
    pool = MagicMock(spec=ap_module.ActorPool)
    pool.name = "InvariantStage"
    pool._is_spmd = False
    pool._allocator = _TrackingAllocator()
    pool._worker_groups = {}
    pool._worker_groups_to_delete = collections.deque()
    pool._worker_groups_to_create = collections.deque(worker_groups)
    pool._adjust_actors = ap_module.ActorPool._adjust_actors.__get__(pool)
    pool._add_worker_group = ap_module.ActorPool._add_worker_group.__get__(pool)
    return pool


class TestAllocatorPoolKeySymmetry:
    """Invariant 1: allocator ids match ``_worker_groups`` keys at every step."""

    def test_keys_match_after_each_add(self) -> None:
        """After every successful ``_add_worker_group``, allocator and pool agree."""
        wg1 = _make_mock_worker_group("wg-1")
        wg2 = _make_mock_worker_group("wg-2")
        pool = _make_pool_with_tracking_allocator([wg1, wg2])

        assert pool._add_worker_group(wg1) is True
        assert set(pool._worker_groups.keys()) == pool._allocator.live_worker_ids == {"wg-1"}

        assert pool._add_worker_group(wg2) is True
        assert set(pool._worker_groups.keys()) == pool._allocator.live_worker_ids == {"wg-1", "wg-2"}

    def test_keys_match_after_rollback(self) -> None:
        """A post-allocator failure rolls back BOTH allocator and pool insertion."""
        wg = _make_mock_worker_group("wg-rollback")
        pool = _make_pool_with_tracking_allocator([wg])
        pool._create_actor_for_worker_group.side_effect = RuntimeError("simulated spawn failure")

        with pytest.raises(RuntimeError, match="simulated spawn failure"):
            pool._add_worker_group(wg)

        assert pool._worker_groups == {}
        assert pool._allocator.live_worker_ids == set()
        assert pool._allocator.remove_calls == 1
        assert pool._allocator.removed_ids == ["wg-rollback"]


class TestKillPathDoesNotTouchAllocator:
    """Invariant 2: the kill path goes through ``_delete_worker_group`` only.

    ``_try_delete_ready_actor`` is invoked from inside ``_delete_actor``
    AFTER ``_delete_worker_group`` has already released the allocator
    slot. If a future change made ``_try_delete_*`` release the allocator
    too, double-removal (or removal of a worker that was never added,
    when the pending->ready transition has not happened yet) becomes
    a real failure mode. This test pins the boundary.
    """

    def test_try_delete_ready_actor_does_not_touch_allocator(self) -> None:
        """Re-queueing in-flight tasks then killing the actor must not call the allocator."""
        actor = MagicMock()
        actor.slots = []
        actor.kill = MagicMock()

        pool = MagicMock(spec=ap_module.ActorPool)
        pool._allocator = _TrackingAllocator()
        pool._ready_actors = {"actor-1": actor}
        pool._task_queue = collections.deque()
        pool._try_delete_ready_actor = ap_module.ActorPool._try_delete_ready_actor.__get__(pool)

        assert pool._try_delete_ready_actor("actor-1") is True

        actor.kill.assert_called_once_with(graceful=False)
        assert pool._allocator.remove_calls == 0
        assert pool._allocator.add_calls == 0
        assert "actor-1" not in pool._ready_actors


class TestContinuousModeRateDurationSemantics:
    """Pin the per-actor (continuous) vs per-slot (batch) rate semantics.

    With ``slots_per_actor=N`` async tasks in parallel, the inter-completion
    interval stamped in ``_collect_continuous_async`` is approximately
    ``per_slot_service_time / N``, so ``RateEstimatorDuration.get_rate()``
    returns ``N x per_slot_rate``. Batch-mode workers leave
    ``rate_duration_s = None`` and the actor pool uses ``timing.process_dur``,
    matching the per-slot rate exactly.
    """

    def test_batch_mode_rate_equals_per_slot_service_rate(self) -> None:
        """Batch mode: ``rate = 1 / process_dur`` is the per-slot rate."""
        per_slot_service_time_s = 0.5
        estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=60.0)
        for i in range(10):
            estimator.update(per_slot_service_time_s, current_time=float(i))

        per_slot_rate_hz = 1.0 / per_slot_service_time_s
        assert estimator.get_rate(current_time=10.0) == pytest.approx(per_slot_rate_hz, rel=1e-6)

    def test_continuous_mode_rate_scales_with_slots(self) -> None:
        """Continuous mode: ``rate = 1 / inter_completion_interval`` scales with slots."""
        per_slot_service_time_s = 0.5
        slots_per_actor = 4
        inter_completion_interval_s = per_slot_service_time_s / slots_per_actor

        estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=60.0)
        for i in range(10):
            estimator.update(inter_completion_interval_s, current_time=float(i))

        per_actor_rate_hz = float(slots_per_actor) / per_slot_service_time_s
        assert estimator.get_rate(current_time=10.0) == pytest.approx(per_actor_rate_hz, rel=1e-6)
        assert estimator.get_rate(current_time=10.0) == pytest.approx(slots_per_actor * (1.0 / per_slot_service_time_s))
