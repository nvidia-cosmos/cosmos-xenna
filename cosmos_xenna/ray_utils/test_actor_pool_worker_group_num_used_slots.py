# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``ActorPool.worker_group_num_used_slots``.

The accessor is the producer side of the per-worker saturation
signal. It sums ``num_used_slots`` across every ready actor in each
worker group and skips actors that are not yet in ``_ready_actors``.
The streaming layer consumes this snapshot per autoscale cycle and
threads it through to Phase D scale-down via ``ProblemWorkerGroupState``.
"""

import collections
from typing import Any, cast

from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.ray_utils import actor_pool


def _slot(*, used: bool) -> actor_pool._Slot[object]:
    """Build a ready-actor slot with or without an assigned task."""
    task = cast(actor_pool._SlotData[object], object()) if used else None
    return actor_pool._Slot(task=task)


def _ready_actor(*, used_slots: int, empty_slots: int) -> actor_pool._ReadyActor[object]:
    """Build a ready actor with the requested used / empty slot counts."""
    slots = [_slot(used=True) for _ in range(used_slots)]
    slots.extend(_slot(used=False) for _ in range(empty_slots))
    return actor_pool._ReadyActor(
        metadata=resources.WorkerMetadata.make_dummy(),
        actor_ref=cast(Any, object()),
        start_time=0.0,
        slots=collections.deque(slots),
    )


def _worker_group(*, group_id: str, actor_ids: set[str]) -> actor_pool._WorkerGroup:
    """Build a ``_WorkerGroup`` shell with the given actor-id membership."""
    del group_id  # field signature does not include the id; the dict key carries it
    return actor_pool._WorkerGroup(
        worker_group=cast(resources.WorkerGroup, object()),
        actors=actor_ids,
        state=actor_pool._WorkerGroupState.READY,
        rendevous_params=None,
    )


def _make_pool(
    *,
    worker_groups: dict[str, actor_pool._WorkerGroup],
    ready_actors: dict[str, actor_pool._ReadyActor[object]],
) -> actor_pool.ActorPool[object, object]:
    """Build a bare ``ActorPool`` exposing only the fields the method reads."""
    pool = actor_pool.ActorPool.__new__(actor_pool.ActorPool)
    pool._worker_groups = worker_groups
    pool._ready_actors = ready_actors
    return cast(actor_pool.ActorPool[object, object], pool)


class TestWorkerGroupNumUsedSlots:
    """Pin the contract of ``ActorPool.worker_group_num_used_slots``."""

    def test_empty_pool_returns_empty_dict(self) -> None:
        """Cold start: no worker groups -> empty snapshot, not an error."""
        pool = _make_pool(worker_groups={}, ready_actors={})

        assert pool.worker_group_num_used_slots() == {}

    def test_worker_group_with_no_actors_reports_zero(self) -> None:
        """Group exists but has not been populated yet -> zero, not omitted."""
        pool = _make_pool(
            worker_groups={"wg-a": _worker_group(group_id="wg-a", actor_ids=set())},
            ready_actors={},
        )

        assert pool.worker_group_num_used_slots() == {"wg-a": 0}

    def test_pending_actors_are_skipped_not_counted(self) -> None:
        """An actor in ``worker_group.actors`` but not in ``_ready_actors`` skips the sum.

        Pins the pending-startup case: while the actor is in
        ``_pending_actors`` / node-setup waiting state, its
        ``num_used_slots`` is unknown. The accessor MUST skip rather
        than ``KeyError``.
        """
        pool = _make_pool(
            worker_groups={
                "wg-a": _worker_group(group_id="wg-a", actor_ids={"actor-ready", "actor-pending"}),
            },
            ready_actors={"actor-ready": _ready_actor(used_slots=2, empty_slots=1)},
        )

        assert pool.worker_group_num_used_slots() == {"wg-a": 2}

    def test_sums_across_multiple_actors_in_one_group(self) -> None:
        """SPMD groups have multiple actors; the sum aggregates correctly."""
        pool = _make_pool(
            worker_groups={
                "wg-spmd": _worker_group(
                    group_id="wg-spmd",
                    actor_ids={"actor-0", "actor-1", "actor-2"},
                ),
            },
            ready_actors={
                "actor-0": _ready_actor(used_slots=2, empty_slots=0),
                "actor-1": _ready_actor(used_slots=1, empty_slots=1),
                "actor-2": _ready_actor(used_slots=0, empty_slots=2),
            },
        )

        assert pool.worker_group_num_used_slots() == {"wg-spmd": 3}

    def test_per_group_isolation_across_multiple_groups(self) -> None:
        """Multiple worker groups produce per-group entries; no cross-bleed."""
        pool = _make_pool(
            worker_groups={
                "wg-a": _worker_group(group_id="wg-a", actor_ids={"a0"}),
                "wg-b": _worker_group(group_id="wg-b", actor_ids={"b0", "b1"}),
                "wg-c": _worker_group(group_id="wg-c", actor_ids=set()),
            },
            ready_actors={
                "a0": _ready_actor(used_slots=4, empty_slots=0),
                "b0": _ready_actor(used_slots=1, empty_slots=1),
                "b1": _ready_actor(used_slots=2, empty_slots=0),
            },
        )

        assert pool.worker_group_num_used_slots() == {"wg-a": 4, "wg-b": 3, "wg-c": 0}
