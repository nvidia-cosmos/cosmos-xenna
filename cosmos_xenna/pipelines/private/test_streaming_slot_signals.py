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


"""Tests for ``Autoscaler._make_problem_state`` slot-signal wiring.

The streaming layer must populate the per-stage saturation signals
(``num_used_slots``, ``num_empty_slots``, ``input_queue_depth``) on
``ProblemStageState`` from the live ``ActorPool`` snapshot so that the
saturation-aware scheduler can read them each autoscale cycle. It also
copies per-worker ``num_used_slots`` onto each
``ProblemWorkerGroupState`` so Phase D can prefer idle worker groups.

This module pins the wiring contract:

    * Active stages: signals match the ``ActorPool`` accessors
      (``num_used_slots``, ``num_empty_slots``, ``num_queued_tasks``).
    * Active workers: ``worker_group_num_used_slots()`` entries land on
      matching worker snapshots.
    * Finished stages: signals are zeroed even if the pool reports
      transient drain-state slot or queue counts.
    * Multi-stage problems preserve per-stage signal isolation.
"""

import collections
import math
from typing import Any, cast

import attrs
import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.streaming import Autoscaler
from cosmos_xenna.ray_utils import actor_pool


@attrs.define
class _FakeActorPool:
    """Minimal stand-in for ``ActorPool`` exposing only the fields ``_make_problem_state`` reads.

    Using a small attrs class instead of ``unittest.mock`` keeps the
    test contract obvious: each field has one purpose, and the test
    fails immediately if ``_make_problem_state`` reaches for a field
    the production code does not actually consume.
    """

    name: str
    slots_per_actor: int
    num_used_slots: int
    num_empty_slots: int
    num_queued_tasks: int
    num_pending_actors: int = 0
    worker_group_used_slots: dict[str, int] = attrs.Factory(dict)

    def worker_group_num_used_slots(self) -> dict[str, int]:
        """Stand-in for ``ActorPool.worker_group_num_used_slots``; defaults to no signal."""
        return dict(self.worker_group_used_slots)


@attrs.define
class _FakeAllocator:
    """Worker-allocator stand-in returning an empty worker list per stage.

    The slot-signal wiring is independent of the worker-group list; the
    fake defaults to the no-worker case so most tests focus on signal
    propagation and not on placement plumbing. Individual tests can
    inject workers to verify the new signal kwargs do not disturb the
    existing worker snapshot path.
    """

    workers_by_stage: dict[str, list[resources.WorkerGroup]] = attrs.Factory(dict)
    calls: list[str] = attrs.Factory(list)

    def get_workers_in_stage(self, name: str) -> list[resources.WorkerGroup]:
        self.calls.append(name)
        return list(self.workers_by_stage.get(name, []))


@attrs.define
class _FakeWorkerGroup:
    """Duck-typed worker snapshot exposing the fields consumed by streaming."""

    id: str
    allocations: list[resources.WorkerResourcesInternal]


def _worker_group(stage_name: str, worker_id: str) -> resources.WorkerGroup:
    """Build a one-CPU worker group for ``stage_name``."""
    del stage_name
    return cast(
        resources.WorkerGroup,
        _FakeWorkerGroup(
            id=worker_id,
            allocations=[resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        ),
    )


def _slot(*, used: bool) -> actor_pool._Slot[object]:
    """Build a ready-actor slot with or without an assigned task."""
    task = cast(actor_pool._SlotData[object], object()) if used else None
    return actor_pool._Slot(task=task)


def _ready_actor(*, used_slots: int, empty_slots: int) -> actor_pool._ReadyActor[object]:
    """Build a ready actor with the requested used/empty slot counts."""
    slots = [_slot(used=True) for _ in range(used_slots)]
    slots.extend(_slot(used=False) for _ in range(empty_slots))
    return actor_pool._ReadyActor(
        metadata=resources.WorkerMetadata.make_dummy(),
        actor_ref=cast(Any, object()),
        start_time=0.0,
        slots=collections.deque(slots),
    )


def _real_actor_pool(
    *,
    name: str,
    slots_per_actor: int,
    ready_slots: list[tuple[str, int, int]],
    queued_tasks: int = 0,
    worker_groups: dict[str, set[str]] | None = None,
) -> actor_pool.ActorPool[object, object]:
    """Build an ``ActorPool`` shell with real slot-count properties.

    The shell mirrors every actor-set attribute the production
    ``ActorPool`` properties read; ``num_pending_actors`` aggregates
    across ``_pending_actors``, ``_pending_node_actors`` and
    ``_actors_waiting_for_node_setup`` so each container must exist
    (as empty containers) even when the test only exercises the
    ready-actor path. Seeding the containers empty makes the default
    ``num_pending_actors`` zero, matching the test's "no pending
    actors" semantics; individual tests that need to verify pending-
    aware behaviour can populate them explicitly.
    """
    pool = actor_pool.ActorPool.__new__(actor_pool.ActorPool)
    pool._name = name
    pool._slots_per_actor = slots_per_actor
    pool._ready_actors = {
        actor_id: _ready_actor(used_slots=used_slots, empty_slots=empty_slots)
        for actor_id, used_slots, empty_slots in ready_slots
    }
    pool._pending_actors = cast(Any, collections.OrderedDict())
    pool._pending_node_actors = cast(Any, collections.OrderedDict())
    pool._actors_waiting_for_node_setup = cast(Any, {})
    pool._task_queue = cast(Any, collections.deque(object() for _ in range(queued_tasks)))
    pool._worker_groups = {
        worker_group_id: actor_pool._WorkerGroup(
            worker_group=_worker_group(name, worker_group_id),
            actors=set(actor_ids),
            state=actor_pool._WorkerGroupState.READY,
            rendevous_params=None,
        )
        for worker_group_id, actor_ids in (worker_groups or {}).items()
    }
    return cast(actor_pool.ActorPool[object, object], pool)


def _make_problem_state(
    pools: list[Any],
    is_dones: list[bool],
    *,
    allocator: _FakeAllocator | None = None,
    upstream_queue_lens: list[int] | None = None,
    stage_batch_sizes: list[int] | None = None,
) -> data_structures.ProblemState:
    """Invoke ``Autoscaler._make_problem_state`` with the fakes above.

    Defaults ``upstream_queue_lens`` to zeros and ``stage_batch_sizes``
    to ones so existing tests pin only the pool-side signals; tests
    exercising the upstream-aggregation path pass explicit values.
    """
    autoscaler = Autoscaler.__new__(Autoscaler)
    autoscaler._allocator = allocator or _FakeAllocator()  # type: ignore[attr-defined, assignment]
    upstream = upstream_queue_lens if upstream_queue_lens is not None else [0] * len(pools)
    batch_sizes = stage_batch_sizes if stage_batch_sizes is not None else [1] * len(pools)
    return autoscaler._make_problem_state(pools, is_dones, upstream, batch_sizes)  # type: ignore[arg-type]


class TestMakeProblemStateSlotSignals:
    """Pin the slot-signal wiring contract in ``Autoscaler._make_problem_state``."""

    def test_num_used_and_empty_slots_sum_ready_actor_slots(self) -> None:
        """Real ``ActorPool`` slot properties sum used and empty ready-actor slots."""
        pool = _real_actor_pool(
            name="pool",
            slots_per_actor=4,
            ready_slots=[
                ("actor-a", 2, 1),
                ("actor-b", 0, 3),
                ("actor-c", 1, 0),
            ],
        )

        assert pool.num_used_slots == 3
        assert pool.num_empty_slots == 4

    def test_num_used_and_empty_slots_ignore_pending_actors_and_task_queue(self) -> None:
        """Ready-actor slot counts do not include pending actors or queued tasks."""
        pool = _real_actor_pool(
            name="pool",
            slots_per_actor=4,
            ready_slots=[("ready", 1, 2)],
            queued_tasks=7,
        )

        assert pool.num_used_slots == 1
        assert pool.num_empty_slots == 2
        assert pool.num_queued_tasks == 7

    def test_worker_group_num_used_slots_sums_ready_actors_by_group(self) -> None:
        """Real ``ActorPool`` reports per-worker-group used slots from ready actors."""
        pool = _real_actor_pool(
            name="pool",
            slots_per_actor=4,
            ready_slots=[
                ("actor-a", 2, 1),
                ("actor-b", 1, 2),
                ("actor-c", 0, 3),
            ],
            worker_groups={
                "wg-hot": {"actor-a", "actor-b"},
                "wg-idle": {"actor-c"},
            },
        )

        assert pool.worker_group_num_used_slots() == {"wg-hot": 3, "wg-idle": 0}

    def test_worker_group_num_used_slots_defaults_unready_group_to_zero(self) -> None:
        """A worker group with no ready actors reports 0 used slots."""
        pool = _real_actor_pool(
            name="pool",
            slots_per_actor=4,
            ready_slots=[("ready", 1, 2)],
            worker_groups={
                "wg-ready": {"ready"},
                "wg-pending": {"pending-actor"},
            },
        )

        assert pool.worker_group_num_used_slots() == {"wg-ready": 1, "wg-pending": 0}

    def test_make_problem_state_uses_real_actor_pool_slot_properties(self) -> None:
        """``_make_problem_state`` consumes real ``ActorPool`` slot-property values."""
        pool = _real_actor_pool(
            name="real",
            slots_per_actor=4,
            ready_slots=[
                ("actor-a", 2, 1),
                ("actor-b", 1, 3),
            ],
            queued_tasks=9,
        )

        state = _make_problem_state([pool], [False])

        stage = state.rust.stages[0]
        assert stage.stage_name == "real"
        assert stage.slots_per_worker == 4
        assert stage.num_used_slots == 3
        assert stage.num_empty_slots == 4
        assert stage.input_queue_depth == 9

    def test_active_stage_propagates_pool_signals(self) -> None:
        """An active stage carries the live ``ActorPool`` slot signals."""
        pool = _FakeActorPool(
            name="ingest",
            slots_per_actor=4,
            num_used_slots=7,
            num_empty_slots=3,
            num_queued_tasks=11,
        )

        state = _make_problem_state([pool], [False])

        stage = state.rust.stages[0]
        assert stage.stage_name == "ingest"
        assert stage.num_used_slots == 7
        assert stage.num_empty_slots == 3
        assert stage.input_queue_depth == 11

    def test_active_stage_preserves_workers_and_slots_with_signals(self) -> None:
        """Signal kwargs do not disturb the existing worker snapshot fields."""
        allocator = _FakeAllocator(
            workers_by_stage={
                "active": [
                    _worker_group("active", "active-w0"),
                    _worker_group("active", "active-w1"),
                ],
            },
        )
        pool = _FakeActorPool(
            name="active",
            slots_per_actor=8,
            num_used_slots=13,
            num_empty_slots=3,
            num_queued_tasks=21,
        )

        state = _make_problem_state([pool], [False], allocator=allocator)

        stage = state.rust.stages[0]
        assert allocator.calls == ["active"]
        assert stage.slots_per_worker == 8
        assert [worker.id for worker in stage.worker_groups] == ["active-w0", "active-w1"]
        assert stage.num_used_slots == 13
        assert stage.num_empty_slots == 3
        assert stage.input_queue_depth == 21

    def test_active_stage_populates_per_worker_used_slots(self) -> None:
        """Active stages copy per-worker slot signals onto worker snapshots."""
        allocator = _FakeAllocator(
            workers_by_stage={
                "active": [
                    _worker_group("active", "active-w0"),
                    _worker_group("active", "active-w1"),
                ],
            },
        )
        pool = _FakeActorPool(
            name="active",
            slots_per_actor=4,
            num_used_slots=3,
            num_empty_slots=5,
            num_queued_tasks=0,
            worker_group_used_slots={"active-w0": 3, "stale-w9": 99},
        )

        state = _make_problem_state([pool], [False], allocator=allocator)

        workers = state.rust.stages[0].worker_groups
        assert [(worker.id, worker.num_used_slots) for worker in workers] == [
            ("active-w0", 3),
            ("active-w1", 0),
        ]

    def test_finished_stage_carries_zero_signals(self) -> None:
        """A finished stage zeroes all three signals regardless of live pool counts."""
        pool = _FakeActorPool(
            name="draining",
            slots_per_actor=4,
            num_used_slots=2,  # transient drain-state values
            num_empty_slots=6,
            num_queued_tasks=99,
        )

        state = _make_problem_state([pool], [True])

        stage = state.rust.stages[0]
        assert stage.is_finished is True
        assert stage.num_used_slots == 0
        assert stage.num_empty_slots == 0
        assert stage.input_queue_depth == 0

    def test_finished_stage_does_not_read_pool_signal_accessors(self) -> None:
        """Finished stages skip signal accessors before zeroing the fields."""

        class _RaisingSignalPool:
            """Pool whose signal properties fail if read."""

            name = "done"
            slots_per_actor = 1

            @property
            def num_used_slots(self) -> int:
                msg = "num_used_slots should not be read"
                raise RuntimeError(msg)

            @property
            def num_empty_slots(self) -> int:
                msg = "num_empty_slots should not be read"
                raise RuntimeError(msg)

            @property
            def num_queued_tasks(self) -> int:
                msg = "num_queued_tasks should not be read"
                raise RuntimeError(msg)

        state = _make_problem_state([_RaisingSignalPool()], [True])

        stage = state.rust.stages[0]
        assert stage.num_used_slots == 0
        assert stage.num_empty_slots == 0
        assert stage.input_queue_depth == 0

    def test_finished_stage_preserves_workers_and_slots_while_zeroing_signals(self) -> None:
        """Finished stages keep structural fields but suppress saturation signals."""
        allocator = _FakeAllocator(
            workers_by_stage={
                "draining": [_worker_group("draining", "draining-w0")],
            },
        )
        pool = _FakeActorPool(
            name="draining",
            slots_per_actor=6,
            num_used_slots=5,
            num_empty_slots=1,
            num_queued_tasks=34,
        )

        state = _make_problem_state([pool], [True], allocator=allocator)

        stage = state.rust.stages[0]
        assert allocator.calls == ["draining"]
        assert stage.slots_per_worker == 6
        assert [worker.id for worker in stage.worker_groups] == ["draining-w0"]
        assert stage.num_used_slots == 0
        assert stage.num_empty_slots == 0
        assert stage.input_queue_depth == 0

    def test_multi_stage_preserves_per_stage_isolation(self) -> None:
        """Multi-stage problems carry distinct signal values per stage."""
        pools = [
            _FakeActorPool("upstream", 1, 1, 0, 5),
            _FakeActorPool("midstream", 2, 0, 4, 0),
            _FakeActorPool("downstream", 4, 8, 8, 12),
        ]

        state = _make_problem_state(pools, [False, False, False])

        rust_stages = state.rust.stages
        assert [s.stage_name for s in rust_stages] == ["upstream", "midstream", "downstream"]
        assert [s.num_used_slots for s in rust_stages] == [1, 0, 8]
        assert [s.num_empty_slots for s in rust_stages] == [0, 4, 8]
        assert [s.input_queue_depth for s in rust_stages] == [5, 0, 12]

    def test_finished_stage_in_mixed_pipeline_still_zeroes(self) -> None:
        """A finished stage in the middle of an otherwise-active pipeline still zeroes."""
        pools = [
            _FakeActorPool("active_a", 1, 3, 1, 7),
            _FakeActorPool("finished_b", 1, 5, 5, 50),
            _FakeActorPool("active_c", 1, 2, 6, 1),
        ]

        state = _make_problem_state(pools, [False, True, False])

        rust_stages = state.rust.stages
        assert rust_stages[0].num_used_slots == 3
        assert rust_stages[0].input_queue_depth == 7
        assert rust_stages[1].num_used_slots == 0
        assert rust_stages[1].num_empty_slots == 0
        assert rust_stages[1].input_queue_depth == 0
        assert rust_stages[2].num_used_slots == 2
        assert rust_stages[2].input_queue_depth == 1

    def test_zero_signals_round_trip_unchanged(self) -> None:
        """A pool reporting zero on every signal produces a ``ProblemStageState`` with zeros.

        Pins the degenerate-but-legitimate cold-start case where a stage
        has no actors yet: zero is a meaningful value, not a missing
        signal, and must not be silently rewritten.
        """
        pool = _FakeActorPool("cold", 1, 0, 0, 0)

        state = _make_problem_state([pool], [False])

        stage = state.rust.stages[0]
        assert stage.num_used_slots == 0
        assert stage.num_empty_slots == 0
        assert stage.input_queue_depth == 0

    def test_empty_problem_produces_zero_stage_state(self) -> None:
        """No actor pools yields a ``ProblemState`` with zero stages."""
        state = _make_problem_state([], [])

        assert state.rust.stages == []

    def test_large_pipeline_preserves_signal_ordering(self) -> None:
        """A large pipeline keeps per-stage signal order without cross-stage bleed."""
        stage_count = 200
        pools = [
            _FakeActorPool(
                name=f"stage-{index}",
                slots_per_actor=(index % 8) + 1,
                num_used_slots=index,
                num_empty_slots=stage_count - index,
                num_queued_tasks=index * 2,
            )
            for index in range(stage_count)
        ]

        state = _make_problem_state(pools, [False] * stage_count)

        rust_stages = state.rust.stages
        assert len(rust_stages) == stage_count
        for index, stage in enumerate(rust_stages):
            assert stage.stage_name == f"stage-{index}"
            assert stage.slots_per_worker == (index % 8) + 1
            assert stage.num_used_slots == index
            assert stage.num_empty_slots == stage_count - index
            assert stage.input_queue_depth == index * 2

    def test_mismatched_pool_and_done_lengths_raise(self) -> None:
        """``zip(strict=True)`` rejects mismatched ``actor_pools`` and ``stages_is_dones`` lengths.

        A length mismatch is a programmer error in the caller (the
        streaming loop assembles both lists from the same pipeline
        spec); failing fast prevents a silent off-by-one that would
        bind the wrong ``is_done`` flag to the wrong stage. The error
        message must identify the mismatch so an operator hitting
        this in production can find the call site without attaching a
        debugger.
        """
        pool = _FakeActorPool("only", 1, 0, 1, 0)

        with pytest.raises(ValueError, match="zip"):
            _make_problem_state([pool], [False, True])

    def test_pool_property_exception_propagates(self) -> None:
        """An exception from a pool accessor propagates uncaught.

        Slot-signal accessors must NOT be silently swallowed: a future
        ``try/except Exception: pass`` wrapper would substitute zero
        signals for a real failure, and the saturation-aware classifier
        would then mis-identify a broken pool as a fully-idle stage.
        """

        class _RaisingPool:
            """Plain class (no attrs codegen) so the property can raise on read."""

            def __init__(self) -> None:
                self.name = "broken"
                self.slots_per_actor = 1
                self.num_empty_slots = 0
                self.num_queued_tasks = 0

            @property
            def num_used_slots(self) -> int:
                msg = "transient pool failure"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="transient pool failure"):
            _make_problem_state([_RaisingPool()], [False])  # type: ignore[list-item]

    @pytest.mark.parametrize(
        ("num_used_slots", "num_empty_slots", "num_queued_tasks"),
        [
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ],
    )
    def test_negative_signal_raises_contextual_value_error(
        self,
        num_used_slots: int,
        num_empty_slots: int,
        num_queued_tasks: int,
    ) -> None:
        """A negative pool-property value fails before the Rust ``usize`` cast."""
        pool = _FakeActorPool("buggy", 1, num_used_slots, num_empty_slots, num_queued_tasks)
        negative_field = next(
            field
            for field, value in {
                "num_used_slots": num_used_slots,
                "num_empty_slots": num_empty_slots,
                "input_queue_depth": num_queued_tasks,
            }.items()
            if value < 0
        )

        with pytest.raises(ValueError, match=rf"stage 'buggy'.*{negative_field} must be >= 0"):
            _make_problem_state([pool], [False])

    def test_repeated_calls_are_idempotent(self) -> None:
        """Two ``_make_problem_state`` calls on the same pools yield equal signal snapshots.

        Pins the no-side-effects contract: a future memoization layer
        that silently returned a cached ``ProblemState`` would feed the
        scheduler stale slot signals. Each call must read live pool
        state and emit fresh ``ProblemStageState`` objects.
        """
        pool = _FakeActorPool("steady", 1, 3, 5, 7)

        first = _make_problem_state([pool], [False])
        second = _make_problem_state([pool], [False])

        first_stage = first.rust.stages[0]
        second_stage = second.rust.stages[0]
        assert first_stage.num_used_slots == second_stage.num_used_slots == 3
        assert first_stage.num_empty_slots == second_stage.num_empty_slots == 5
        assert first_stage.input_queue_depth == second_stage.input_queue_depth == 7
        assert first is not second

    def test_repeated_calls_read_fresh_pool_signals(self) -> None:
        """A second call reads changed live pool values rather than cached state."""
        pool = _FakeActorPool("changing", 1, 1, 3, 5)
        first = _make_problem_state([pool], [False])

        pool.num_used_slots = 3
        pool.num_empty_slots = 1
        pool.num_queued_tasks = 9
        second = _make_problem_state([pool], [False])

        first_stage = first.rust.stages[0]
        second_stage = second.rust.stages[0]
        assert (first_stage.num_used_slots, first_stage.num_empty_slots, first_stage.input_queue_depth) == (1, 3, 5)
        assert (second_stage.num_used_slots, second_stage.num_empty_slots, second_stage.input_queue_depth) == (3, 1, 9)

    def test_all_finished_pipeline_zeroes_every_stage(self) -> None:
        """A pipeline with every stage finished zeroes every per-stage signal."""
        pools = [
            _FakeActorPool("a", 1, 9, 9, 9),
            _FakeActorPool("b", 1, 8, 8, 8),
        ]

        state = _make_problem_state(pools, [True, True])

        rust_stages = state.rust.stages
        for stage in rust_stages:
            assert stage.is_finished is True
            assert stage.num_used_slots == 0
            assert stage.num_empty_slots == 0
            assert stage.input_queue_depth == 0

    def test_signal_isolation_under_large_values(self) -> None:
        """Large signal values round-trip through the Rust boundary intact.

        Pins that signals up to a large but realistic upper bound (a
        100-actor stage with high concurrency could hit thousands of
        slots / queued tasks) are preserved without truncation.
        """
        pool = _FakeActorPool("hot", 64, 6_400, 3_200, 50_000)

        state = _make_problem_state([pool], [False])

        stage = state.rust.stages[0]
        assert stage.num_used_slots == 6_400
        assert stage.num_empty_slots == 3_200
        assert stage.input_queue_depth == 50_000

    def test_mismatched_pool_and_done_lengths_raise_when_done_list_is_shorter(self) -> None:
        """``zip(strict=True)`` also rejects missing ``is_done`` flags."""
        pools = [
            _FakeActorPool("a", 1, 0, 1, 0),
            _FakeActorPool("b", 1, 0, 1, 0),
        ]

        with pytest.raises(ValueError):
            _make_problem_state(pools, [False])


class TestMakeProblemStateInputQueueDepthAggregation:
    """Pin ``input_queue_depth`` aggregation of pool and upstream backlog.

    ``pool.num_queued_tasks`` is bounded by the streaming-mode
    ``compute_max_queued`` cap; backlog still sitting in the pipeline
    ``input_queue`` (stage 0) or ``queues[idx - 1]`` (downstream
    stages) would otherwise be invisible to the saturation-aware
    classifier and capacity sizer.
    """

    def test_unit_one_adds_upstream_lens_directly_when_batch_size_is_one(self) -> None:
        """``stage_batch_size = 1`` makes samples and tasks identical, so addition is direct."""
        pool = _FakeActorPool("stage-0", 1, 2, 1, 5)

        state = _make_problem_state(
            [pool],
            [False],
            upstream_queue_lens=[37],
            stage_batch_sizes=[1],
        )

        stage = state.rust.stages[0]
        assert stage.input_queue_depth == 5 + 37

    def test_upstream_samples_ceil_divide_by_batch_size_to_tasks(self) -> None:
        """Sample-denominated upstream lens convert to task-denominated via ceil-div."""
        pool = _FakeActorPool("batched", 4, 8, 0, 3)

        state = _make_problem_state(
            [pool],
            [False],
            upstream_queue_lens=[17],
            stage_batch_sizes=[8],
        )

        stage = state.rust.stages[0]
        assert stage.input_queue_depth == 3 + math.ceil(17 / 8)

    def test_upstream_zero_preserves_pool_only_signal(self) -> None:
        """An empty upstream queue yields ``input_queue_depth == pool.num_queued_tasks``."""
        pool = _FakeActorPool("empty-upstream", 1, 0, 1, 9)

        state = _make_problem_state(
            [pool],
            [False],
            upstream_queue_lens=[0],
            stage_batch_sizes=[1],
        )

        assert state.rust.stages[0].input_queue_depth == 9

    def test_done_stage_ignores_upstream_queue_lens(self) -> None:
        """Done stages emit zero on every signal; upstream lens must not leak."""
        pool = _FakeActorPool("drained", 1, 0, 0, 0)

        state = _make_problem_state(
            [pool],
            [True],
            upstream_queue_lens=[1234],
            stage_batch_sizes=[1],
        )

        assert state.rust.stages[0].input_queue_depth == 0

    def test_mismatched_upstream_queue_lens_length_raises(self) -> None:
        """``zip(strict=True)`` rejects upstream-queue-lens length mismatch.

        A length mismatch is a programmer error in the streaming
        loop (it assembles ``upstream_queue_lens`` from the same
        pipeline spec as ``actor_pools``); failing fast prevents the
        upstream backlog of one stage from being silently bound to
        another.
        """
        pool = _FakeActorPool("only", 1, 0, 1, 0)

        with pytest.raises(ValueError, match="zip"):
            _make_problem_state(
                [pool],
                [False],
                upstream_queue_lens=[],
                stage_batch_sizes=[1],
            )


class TestMakeProblemStatePerWorkerNumUsedSlots:
    """Pin per-worker ``num_used_slots`` propagation through ``_make_problem_state``."""

    @staticmethod
    def _per_worker_used_slots(
        state: data_structures.ProblemState,
        *,
        stage_index: int = 0,
    ) -> dict[str, int]:
        """Read back ``{worker_id: num_used_slots}`` from a ``ProblemState`` snapshot."""
        stage = state.rust.stages[stage_index]
        return {worker.id: worker.num_used_slots for worker in stage.worker_groups}

    def test_non_empty_pool_map_propagates_to_per_worker(self) -> None:
        """``pool.worker_group_num_used_slots()`` lands on the matching worker's field.

        Regression target: if the wiring keyed off ``pool.name`` instead
        of ``w.id``, or hard-coded ``num_used_slots=pool.num_used_slots``,
        the per-worker fields would not match the per-worker truth.
        """

        class _MapPool:
            """Pool returning a non-trivial per-worker idle map."""

            name = "active"
            slots_per_actor = 4
            num_used_slots = 99
            num_empty_slots = 1
            num_queued_tasks = 0
            num_pending_actors = 0

            def worker_group_num_used_slots(self) -> dict[str, int]:
                return {"active-w0": 3, "active-w1": 0}

        allocator = _FakeAllocator(
            workers_by_stage={
                "active": [
                    _worker_group("active", "active-w0"),
                    _worker_group("active", "active-w1"),
                ],
            },
        )

        state = _make_problem_state([_MapPool()], [False], allocator=allocator)  # type: ignore[list-item]

        assert self._per_worker_used_slots(state) == {"active-w0": 3, "active-w1": 0}

    def test_worker_missing_from_partial_fake_idle_map_defaults_to_zero(self) -> None:
        """A duck-typed pool without structural worker IDs can still omit a worker signal."""

        class _PartialMapPool:
            name = "ingest"
            slots_per_actor = 2
            num_used_slots = 5
            num_empty_slots = 3
            num_queued_tasks = 0
            num_pending_actors = 0

            def worker_group_num_used_slots(self) -> dict[str, int]:
                return {"ingest-w0": 2}

        allocator = _FakeAllocator(
            workers_by_stage={
                "ingest": [
                    _worker_group("ingest", "ingest-w0"),
                    _worker_group("ingest", "ingest-w1"),
                ],
            },
        )

        state = _make_problem_state([_PartialMapPool()], [False], allocator=allocator)  # type: ignore[list-item]

        assert self._per_worker_used_slots(state) == {"ingest-w0": 2, "ingest-w1": 0}

    def test_negative_worker_group_signal_raises_contextual_value_error(self) -> None:
        """A negative per-worker map value names the stage, worker, and field."""
        allocator = _FakeAllocator(
            workers_by_stage={"active": [_worker_group("active", "active-w0")]},
        )
        pool = _FakeActorPool(
            name="active",
            slots_per_actor=4,
            num_used_slots=0,
            num_empty_slots=4,
            num_queued_tasks=0,
            worker_group_used_slots={"active-w0": -1},
        )

        with pytest.raises(ValueError, match=r"stage 'active'.*worker 'active-w0'.*num_used_slots must be >= 0"):
            _make_problem_state([pool], [False], allocator=allocator)

    def test_real_pool_allocator_worker_mismatch_raises_contextual_value_error(self) -> None:
        """A stale allocator worker cannot be silently modelled as idle."""
        allocator = _FakeAllocator(
            workers_by_stage={
                "active": [
                    _worker_group("active", "active-w0"),
                    _worker_group("active", "stale-w1"),
                ],
            },
        )
        pool = _real_actor_pool(
            name="active",
            slots_per_actor=4,
            ready_slots=[("actor-a", 1, 3)],
            worker_groups={"active-w0": {"actor-a"}},
        )

        with pytest.raises(ValueError, match=r"stage 'active'.*allocator-only.*stale-w1"):
            _make_problem_state([pool], [False], allocator=allocator)

    def test_repeated_calls_read_fresh_per_worker_signals(self) -> None:
        """Per-worker map values are sampled fresh each cycle."""
        allocator = _FakeAllocator(
            workers_by_stage={
                "active": [
                    _worker_group("active", "active-w0"),
                    _worker_group("active", "active-w1"),
                ],
            },
        )
        pool = _FakeActorPool(
            name="active",
            slots_per_actor=4,
            num_used_slots=1,
            num_empty_slots=7,
            num_queued_tasks=0,
            worker_group_used_slots={"active-w0": 1, "active-w1": 0},
        )

        first = _make_problem_state([pool], [False], allocator=allocator)
        pool.worker_group_used_slots = {"active-w0": 0, "active-w1": 3}
        second = _make_problem_state([pool], [False], allocator=allocator)

        assert self._per_worker_used_slots(first) == {"active-w0": 1, "active-w1": 0}
        assert self._per_worker_used_slots(second) == {"active-w0": 0, "active-w1": 3}

    def test_pool_worker_group_num_used_slots_exception_propagates(self) -> None:
        """An exception from ``worker_group_num_used_slots()`` propagates uncaught.

        Pins the same contract as ``test_pool_property_exception_propagates``
        but for the per-worker accessor: a future ``try/except Exception:
        return {}`` wrapper would silently substitute all-idle signals
        for a real pool failure, and Phase D would happily shrink an
        all-busy stage.
        """

        class _RaisingMethodPool:
            name = "broken"
            slots_per_actor = 1
            num_used_slots = 0
            num_empty_slots = 1
            num_queued_tasks = 0
            num_pending_actors = 0

            def worker_group_num_used_slots(self) -> dict[str, int]:
                msg = "transient pool failure"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="transient pool failure"):
            _make_problem_state([_RaisingMethodPool()], [False])  # type: ignore[list-item]

    def test_finished_stage_zeroes_workers_without_reading_nonempty_map(self) -> None:
        """Finished workers get 0 used slots even if the pool has stale map entries."""
        allocator = _FakeAllocator(
            workers_by_stage={
                "done": [
                    _worker_group("done", "done-w0"),
                    _worker_group("done", "done-w1"),
                ],
            },
        )
        pool = _FakeActorPool(
            name="done",
            slots_per_actor=4,
            num_used_slots=8,
            num_empty_slots=0,
            num_queued_tasks=99,
            worker_group_used_slots={"done-w0": 4, "done-w1": 4},
        )

        state = _make_problem_state([pool], [True], allocator=allocator)

        assert self._per_worker_used_slots(state) == {"done-w0": 0, "done-w1": 0}

    def test_finished_stage_does_not_call_worker_group_num_used_slots(self) -> None:
        """Finished stages must not read the per-worker accessor (matches stage-signal skip)."""

        class _RaisingMethodPool:
            name = "done"
            slots_per_actor = 1
            num_used_slots = 0
            num_empty_slots = 1
            num_queued_tasks = 0

            def worker_group_num_used_slots(self) -> dict[str, int]:
                msg = "should not be called on a finished stage"
                raise RuntimeError(msg)

        allocator = _FakeAllocator(workers_by_stage={"done": [_worker_group("done", "done-w0")]})

        state = _make_problem_state([_RaisingMethodPool()], [True], allocator=allocator)  # type: ignore[list-item]

        stage = state.rust.stages[0]
        assert stage.is_finished is True
        assert self._per_worker_used_slots(state) == {"done-w0": 0}
