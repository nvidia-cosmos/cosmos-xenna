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

"""Unit tests for ``ActorPool._handle_actor_death``.

The recovery handler is invoked when ``ray.get`` on a slot's generator ref raises
one of ``_ACTOR_DEATH_ERRORS`` (OOM-kill, node loss, segfault, external ray.kill,
assertion in a Ray callback). It must:

  * Re-queue every unique in-flight ``Task`` (front of queue, matching the
    priority used by intentional-delete reclaim in ``_try_delete_ready_actor``).
  * Drop tasks that have already hit ``_MAX_ACTOR_DEATH_RETRIES`` so a poison
    input cannot loop forever, and count the drop.
  * Deduplicate by ``id(task)`` across SPMD sibling actors, since
    ``_schedule_task_on_worker_group`` dispatches the same ``Task`` instance to
    every rank's slot.
  * Clear all slots before delegating to ``_delete_worker_group`` so the
    existing reclaim in ``_try_delete_ready_actor`` becomes a no-op and does
    not defeat the dedupe.
  * Tear down the worker group through the normal teardown and re-enqueue its
    allocation onto ``_worker_groups_to_create`` so the next ``_adjust_actors``
    tick recreates it through the same code path autoscale uses.

These are pure unit tests over the recovery bookkeeping: ``_delete_worker_group``
is monkey-patched to record the call (the real implementation requires a Ray
cluster + allocator which is exercised in the integration test alongside
``test_segfaulting_pipeline.py``).
"""

from __future__ import annotations

import collections
from typing import Any
from unittest import mock

import attrs
import pytest
import ray.exceptions

from cosmos_xenna.ray_utils.actor_pool import (
    _ACTOR_DEATH_ERRORS,
    _MAX_ACTOR_DEATH_RETRIES,
    _MAX_CONSECUTIVE_WG_DEATHS,
    ActorPool,
    Task,
    _ReadyActor,
    _Slot,
    _SlotData,
    _WorkerGroup,
    _WorkerGroupState,
)
from cosmos_xenna.ray_utils.task_metadata import (
    FailureInfo,
    TaskDataInfo,
    TaskResultMetadata,
    TimingInfo,
)

# ---------------------------------------------------------------------------
# Test fixtures: minimal pool / worker-group / ready-actor builders.
# ---------------------------------------------------------------------------


@attrs.define
class _StubMetadata:
    """Minimal stand-in for ``resources.WorkerMetadata``.

    ``_handle_actor_death`` only reads ``worker_group_id`` off the actor's
    metadata (via the catch site in ``_process_completed_task``), so this stub
    exposes only that field. ``_ReadyActor`` itself does not introspect
    ``metadata``; it just stores the reference.
    """

    worker_id: str
    worker_group_id: str


def _make_task(node_id: str | None = None) -> Task[Any]:
    return Task(task_data=[], origin_node_id=node_id)


def _make_slot_with_task(task: Task[Any]) -> _Slot:
    """Build a ``_Slot`` that holds ``task`` but no real object ref."""
    slot: _Slot = _Slot(task=None)
    slot.task = _SlotData(task=task, scheduled_time=0.0, object_ref=mock.sentinel.object_ref)  # type: ignore[arg-type]
    return slot


def _make_empty_slot() -> _Slot:
    return _Slot(task=None)


def _make_ready_actor(actor_id: str, worker_group_id: str, slots: list[_Slot]) -> _ReadyActor:
    actor: _ReadyActor = object.__new__(_ReadyActor)
    actor.metadata = _StubMetadata(worker_id=actor_id, worker_group_id=worker_group_id)  # type: ignore[assignment]
    actor.slots = collections.deque(slots)
    actor.actor_ref = mock.sentinel.actor_ref  # type: ignore[assignment]
    actor.start_time = 0.0
    actor.last_became_idle_time = None
    # ``rate_estimator.update`` is called inside ``_process_completed_task`` on
    # the primary-rank success path. Tests that exercise the success path stub
    # it; tests that only exercise ``_handle_actor_death`` never touch it.
    actor.rate_estimator = mock.MagicMock()  # type: ignore[assignment]
    return actor


def _make_worker_group(wg_id: str, actor_ids: list[str]) -> _WorkerGroup:
    return _WorkerGroup(
        worker_group=mock.sentinel.worker_group_allocation,  # type: ignore[arg-type]
        actors=set(actor_ids),
        state=_WorkerGroupState.READY,
        rendevous_params=None,
        ready_at=0.0,
    )


def _make_pool() -> ActorPool:
    """Build a bare ``ActorPool`` populated with only the fields ``_handle_actor_death`` reads."""
    pool = object.__new__(ActorPool)
    pool._name = "test_stage"
    pool._worker_groups = {}
    pool._ready_actors = {}
    pool._pending_actors = collections.OrderedDict()
    pool._pending_node_actors = collections.OrderedDict()
    pool._actors_waiting_for_node_setup = collections.defaultdict(list)
    pool._task_queue = collections.deque()
    pool._worker_groups_to_create = collections.deque()
    pool._num_unexpected_actor_deaths = 0
    pool._num_tasks_dropped_on_actor_death = 0
    pool._num_worker_groups_abandoned = 0
    pool._consecutive_actor_deaths_by_wg_id = {}

    # Primary-rank success path state. Only consulted when a test drives
    # ``_process_completed_task`` with ``is_primary=True`` (e.g. the SPMD
    # post-success regression tests). Mocked metrics avoid pulling the real
    # Counter / metric backend into hermetic unit tests.
    pool._task_result_metadatas = collections.deque()
    pool._completed_tasks = collections.deque()
    pool._num_null_tasks = 0
    pool._num_completed_tasks = 0
    pool._num_dynamically_spawned_tasks = 0
    pool._params = mock.MagicMock(stage_batch_size=1)
    pool._update_task_metrics = mock.MagicMock()  # type: ignore[method-assign]

    # ``_delete_worker_group`` requires an allocator + port registry on the real
    # pool. Replace with a recorder so the unit tests stay hermetic; the
    # integration test in test_segfaulting_pipeline exercises the real path.
    # The mock mirrors the real method's mutation (pop from ``_worker_groups``)
    # so that follow-up calls observe an empty dict - critical for the same-tick
    # double-delete guard regression test.
    def _fake_delete(wg_id: str) -> Any:
        pool._worker_groups.pop(wg_id)
        return mock.sentinel.deleted_wg

    pool._delete_worker_group = mock.MagicMock(side_effect=_fake_delete)  # type: ignore[method-assign]
    return pool


def _install_worker_group(
    pool: ActorPool,
    wg_id: str,
    actor_ids: list[str],
    slots_per_actor: list[list[_Slot]],
) -> None:
    """Wire a worker group and its ready actors onto a bare pool."""
    assert len(actor_ids) == len(slots_per_actor)
    pool._worker_groups[wg_id] = _make_worker_group(wg_id, actor_ids)
    for actor_id, slots in zip(actor_ids, slots_per_actor, strict=True):
        pool._ready_actors[actor_id] = _make_ready_actor(actor_id, wg_id, slots)


# ---------------------------------------------------------------------------
# _ACTOR_DEATH_ERRORS sanity
# ---------------------------------------------------------------------------


def test_actor_death_errors_includes_expected_ray_exceptions() -> None:
    """Lock in the exception tuple so a Ray-side rename does not silently regress recovery.

    The docstring on ``_handle_actor_death`` promises coverage of OOM-kill, node
    loss, segfault, and external ``ray.kill``. The tuple must include every
    distinct ``RayError`` subclass needed for those scenarios; some of them
    (``NodeDiedError``, ``WorkerCrashedError``, ``OutOfMemoryError``) are NOT
    subclasses of ``RayActorError`` and must be listed individually.
    """
    assert ray.exceptions.RayActorError in _ACTOR_DEATH_ERRORS
    assert ray.exceptions.ActorDiedError in _ACTOR_DEATH_ERRORS
    assert ray.exceptions.ActorUnavailableError in _ACTOR_DEATH_ERRORS
    assert ray.exceptions.NodeDiedError in _ACTOR_DEATH_ERRORS
    assert ray.exceptions.WorkerCrashedError in _ACTOR_DEATH_ERRORS
    assert ray.exceptions.OutOfMemoryError in _ACTOR_DEATH_ERRORS


def test_actor_death_subclass_invariants_hold() -> None:
    """Lock the Ray exception hierarchy assumptions the tuple relies on.

    A future Ray refactor that demotes ``ActorDiedError`` / ``ActorUnavailableError``
    out of the ``RayActorError`` lineage, or promotes ``NodeDiedError`` /
    ``WorkerCrashedError`` into it, would silently change recovery coverage.
    This test makes that regression noisy.
    """
    assert issubclass(ray.exceptions.ActorDiedError, ray.exceptions.RayActorError)
    assert issubclass(ray.exceptions.ActorUnavailableError, ray.exceptions.RayActorError)
    assert not issubclass(ray.exceptions.NodeDiedError, ray.exceptions.RayActorError)
    assert not issubclass(ray.exceptions.WorkerCrashedError, ray.exceptions.RayActorError)
    assert not issubclass(ray.exceptions.OutOfMemoryError, ray.exceptions.RayActorError)
    # RayTaskError is intentionally NOT in the tuple: it wraps user-code
    # exceptions raised from inside ``process_data`` and must propagate as a
    # real bug, not trigger WG recovery.
    assert ray.exceptions.RayTaskError not in _ACTOR_DEATH_ERRORS


# ---------------------------------------------------------------------------
# _handle_actor_death: single-actor (non-SPMD) recovery
# ---------------------------------------------------------------------------


def test_handle_actor_death_requeues_single_in_flight_task() -> None:
    """One actor, one in-flight task: re-queued at front and worker group re-requested."""
    pool = _make_pool()
    task = _make_task()
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_slot_with_task(task)]])

    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("boom"))

    assert list(pool._task_queue) == [task]
    assert task.actor_death_retries == 1
    assert pool._num_unexpected_actor_deaths == 1
    assert pool._num_tasks_dropped_on_actor_death == 0
    pool._delete_worker_group.assert_called_once_with("wg-1")  # type: ignore[attr-defined]
    assert list(pool._worker_groups_to_create) == [mock.sentinel.worker_group_allocation]


def test_handle_actor_death_does_not_requeue_empty_slots() -> None:
    """Empty slots produce no queue entries and no spurious retry increments."""
    pool = _make_pool()
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot(), _make_empty_slot()]])

    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError())

    assert len(pool._task_queue) == 0
    assert pool._num_tasks_dropped_on_actor_death == 0
    assert pool._num_unexpected_actor_deaths == 1
    pool._delete_worker_group.assert_called_once_with("wg-1")  # type: ignore[attr-defined]


def test_handle_actor_death_requeues_multiple_distinct_tasks_in_order() -> None:
    """Distinct tasks across distinct slots are all re-queued at the front."""
    pool = _make_pool()
    task_a = _make_task("node-a")
    task_b = _make_task("node-b")
    _install_worker_group(
        pool,
        "wg-1",
        ["actor-1"],
        [[_make_slot_with_task(task_a), _make_slot_with_task(task_b)]],
    )

    pool._handle_actor_death("wg-1", ray.exceptions.OutOfMemoryError("oom"))

    # Both tasks live at the front; ``Task`` is mutable/unhashable so compare by id().
    queued_ids = {id(t) for t in pool._task_queue}
    assert queued_ids == {id(task_a), id(task_b)}
    assert len(pool._task_queue) == 2
    assert task_a.actor_death_retries == 1
    assert task_b.actor_death_retries == 1


# ---------------------------------------------------------------------------
# _handle_actor_death: SPMD dedupe (the critical correctness invariant)
# ---------------------------------------------------------------------------


def test_handle_actor_death_spmd_dedupes_same_task_across_siblings() -> None:
    """SPMD: the same ``Task`` instance lives on every rank actor's slot.

    Without dedupe via ``id(task)``, an N-rank WG death would N x duplicate every
    in-flight task in the queue. This is the worst-case regression the dedupe
    prevents.
    """
    pool = _make_pool()
    shared_task = _make_task()
    _install_worker_group(
        pool,
        "wg-spmd",
        ["rank-0", "rank-1", "rank-2", "rank-3"],
        [
            [_make_slot_with_task(shared_task)],
            [_make_slot_with_task(shared_task)],
            [_make_slot_with_task(shared_task)],
            [_make_slot_with_task(shared_task)],
        ],
    )

    pool._handle_actor_death("wg-spmd", ray.exceptions.RayActorError("nccl rank died"))

    assert list(pool._task_queue) == [shared_task]
    assert shared_task.actor_death_retries == 1
    assert pool._num_unexpected_actor_deaths == 1


def test_handle_actor_death_spmd_clears_all_sibling_slots() -> None:
    """All sibling slots must be cleared so the downstream reclaim is a no-op.

    Without this, ``_try_delete_ready_actor`` (invoked transitively from
    ``_delete_worker_group``) would re-enqueue the still-occupied slots and
    double-count against the retry cap.
    """
    pool = _make_pool()
    shared_task = _make_task()
    _install_worker_group(
        pool,
        "wg-spmd",
        ["rank-0", "rank-1"],
        [[_make_slot_with_task(shared_task)], [_make_slot_with_task(shared_task)]],
    )

    pool._handle_actor_death("wg-spmd", ray.exceptions.RayActorError("died"))

    for actor in pool._ready_actors.values():
        for slot in actor.slots:
            assert not slot.has_task, f"slot on {actor.metadata.worker_id} was not cleared"


def test_handle_actor_death_spmd_mixed_tasks_dedupes_only_duplicates() -> None:
    """Mixed SPMD scenario: shared task dedupes; an unrelated task on a sibling slot still re-queues."""
    pool = _make_pool()
    shared_task = _make_task()
    extra_task = _make_task()
    _install_worker_group(
        pool,
        "wg-spmd",
        ["rank-0", "rank-1"],
        [
            [_make_slot_with_task(shared_task), _make_slot_with_task(extra_task)],
            [_make_slot_with_task(shared_task)],
        ],
    )

    pool._handle_actor_death("wg-spmd", ray.exceptions.RayActorError())

    queued_ids = [id(t) for t in pool._task_queue]
    assert len(queued_ids) == 2
    assert id(shared_task) in queued_ids
    assert id(extra_task) in queued_ids
    assert shared_task.actor_death_retries == 1
    assert extra_task.actor_death_retries == 1


# ---------------------------------------------------------------------------
# _handle_actor_death: retry-cap and drop accounting
# ---------------------------------------------------------------------------


def test_handle_actor_death_drops_task_at_retry_cap() -> None:
    """A task that has hit ``_MAX_ACTOR_DEATH_RETRIES`` is dropped, not re-queued."""
    pool = _make_pool()
    poisoned = _make_task()
    poisoned.actor_death_retries = _MAX_ACTOR_DEATH_RETRIES
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_slot_with_task(poisoned)]])

    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died again"))

    assert list(pool._task_queue) == []
    assert poisoned.actor_death_retries == _MAX_ACTOR_DEATH_RETRIES, "retry counter must not advance past the cap"
    assert pool._num_tasks_dropped_on_actor_death == 1
    pool._delete_worker_group.assert_called_once_with("wg-1")  # type: ignore[attr-defined]


def test_handle_actor_death_retry_counter_increments_until_cap() -> None:
    """Successive deaths on the same task increment the counter and finally drop."""
    pool = _make_pool()
    task = _make_task()

    for attempt in range(1, _MAX_ACTOR_DEATH_RETRIES + 1):
        _install_worker_group(pool, f"wg-{attempt}", [f"actor-{attempt}"], [[_make_slot_with_task(task)]])
        pool._handle_actor_death(f"wg-{attempt}", ray.exceptions.RayActorError("retry"))
        # While we are under the cap the task is re-queued; pop it back out of the
        # queue so the next iteration can fill a fresh slot, simulating the next
        # scheduling round.
        assert list(pool._task_queue) == [task]
        pool._task_queue.popleft()

    # One more death after hitting the cap: the task is now dropped, not re-queued.
    _install_worker_group(pool, "wg-final", ["actor-final"], [[_make_slot_with_task(task)]])
    pool._handle_actor_death("wg-final", ray.exceptions.RayActorError("final"))

    assert task.actor_death_retries == _MAX_ACTOR_DEATH_RETRIES
    assert list(pool._task_queue) == []
    assert pool._num_tasks_dropped_on_actor_death == 1
    assert pool._num_unexpected_actor_deaths == _MAX_ACTOR_DEATH_RETRIES + 1


# ---------------------------------------------------------------------------
# _handle_actor_death: defensive / idempotency
# ---------------------------------------------------------------------------


def test_handle_actor_death_unknown_worker_group_is_a_noop() -> None:
    """If the WG is already torn down (e.g. handled this tick by another slot's death),
    the call must short-circuit without touching the queue or the create queue.

    The death counter is NOT incremented on the no-op path: it counts WGs that
    actually entered recovery, not raw call count. This keeps the eventual
    Prometheus counter free of duplicate spikes when ``deaths_seen`` only
    partially elides re-entry.
    """
    pool = _make_pool()

    pool._handle_actor_death("wg-nonexistent", ray.exceptions.RayActorError("died"))

    assert len(pool._task_queue) == 0
    assert len(pool._worker_groups_to_create) == 0
    pool._delete_worker_group.assert_not_called()  # type: ignore[attr-defined]
    assert pool._num_unexpected_actor_deaths == 0, (
        "no-op call (WG already torn down) must NOT advance the death counter"
    )
    assert pool._num_tasks_dropped_on_actor_death == 0


def test_handle_actor_death_schedules_replacement_after_teardown() -> None:
    """The recovered allocation must be appended to ``_worker_groups_to_create``
    so the next ``_adjust_actors`` tick recreates the worker group through the
    same code path autoscale uses."""
    pool = _make_pool()
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])

    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))

    pool._delete_worker_group.assert_called_once_with("wg-1")  # type: ignore[attr-defined]
    assert len(pool._worker_groups_to_create) == 1
    # The stub WG carries ``mock.sentinel.worker_group_allocation`` in its
    # ``worker_group`` field; the handler captures that *before* deletion runs.
    assert pool._worker_groups_to_create[0] is mock.sentinel.worker_group_allocation


# ---------------------------------------------------------------------------
# _handle_actor_death: per-WG consecutive-death cap (circuit breaker)
# ---------------------------------------------------------------------------


def test_handle_actor_death_under_cap_keeps_restarting_wg() -> None:
    """Below the cap, every death still re-pushes the allocation for restart and
    the consecutive-death counter accumulates against the WG id."""
    pool = _make_pool()
    # ``_MAX_CONSECUTIVE_WG_DEATHS - 1`` deaths must all restart.
    for _ in range(_MAX_CONSECUTIVE_WG_DEATHS - 1):
        _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])
        pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))

    assert pool._consecutive_actor_deaths_by_wg_id["wg-1"] == _MAX_CONSECUTIVE_WG_DEATHS - 1
    assert pool._num_worker_groups_abandoned == 0
    # Every death produced one re-push.
    assert len(pool._worker_groups_to_create) == _MAX_CONSECUTIVE_WG_DEATHS - 1
    assert all(alloc is mock.sentinel.worker_group_allocation for alloc in pool._worker_groups_to_create)


def test_handle_actor_death_at_cap_abandons_wg() -> None:
    """At the cap, the WG is torn down but NOT re-pushed onto the create queue;
    the abandonment counter increments and the consecutive-death tally is cleared
    so a future fresh allocation for the same id (rare; usually a new id) is not
    pre-penalized."""
    pool = _make_pool()
    # Drive the counter up to ``cap - 1`` via real death events on the WG id.
    for _ in range(_MAX_CONSECUTIVE_WG_DEATHS - 1):
        _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])
        pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))
    create_queue_len_before = len(pool._worker_groups_to_create)
    assert pool._consecutive_actor_deaths_by_wg_id["wg-1"] == _MAX_CONSECUTIVE_WG_DEATHS - 1

    # The cap-tripping death.
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])
    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("final straw"))

    assert pool._num_worker_groups_abandoned == 1
    # Tear-down still ran (the dead WG is gone), but no replacement was scheduled.
    assert len(pool._worker_groups_to_create) == create_queue_len_before, (
        "abandoned WG must NOT be re-pushed onto the create queue"
    )
    # Counter is cleared on abandonment so a future fresh WG (if the autoscaler
    # ever spawns one for this id) starts with a clean slate.
    assert "wg-1" not in pool._consecutive_actor_deaths_by_wg_id


def test_handle_actor_death_counter_resets_after_successful_completion() -> None:
    """A successful round-trip in ``_process_completed_task`` clears the
    consecutive-death counter so the next death restarts at 1, not at the
    accumulated tally. This lets a long-lived WG that finally OOMs after
    thousands of tasks get the full retry budget."""
    pool = _make_pool()

    # Two deaths bring the counter to 2.
    for _ in range(2):
        _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])
        pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))
    assert pool._consecutive_actor_deaths_by_wg_id["wg-1"] == 2

    # Drive the success path of ``_process_completed_task`` end-to-end with a
    # canned metadata response so the reset call site (not just the dict
    # operation in isolation) is exercised. ``is_primary=False`` skips metrics
    # bookkeeping that would require additional pool state.
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_slot_with_task(_make_task())]])
    actor = pool._ready_actors["actor-1"]
    metadata = TaskResultMetadata(
        timing=TimingInfo(),
        failure_info=FailureInfo(should_process_further=True, should_restart_worker=False),
        task_data_info=TaskDataInfo(serialized_input_size=0),
        num_returns=0,
    )
    metadata_ref = mock.sentinel.metadata_ref
    with mock.patch(
        "cosmos_xenna.ray_utils.actor_pool.ray.get",
        side_effect=[[metadata_ref], metadata],
    ):
        pool._process_completed_task(actor, slot_num=0, is_primary=False)

    assert "wg-1" not in pool._consecutive_actor_deaths_by_wg_id, (
        "successful completion must reset the consecutive-death tally"
    )

    # And a fresh death after the reset starts the counter back at 1, with the
    # WG re-pushed for restart (we are well under the cap again).
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])
    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died again"))
    assert pool._consecutive_actor_deaths_by_wg_id["wg-1"] == 1
    assert pool._num_worker_groups_abandoned == 0


def test_handle_actor_death_distinct_wgs_have_independent_counters() -> None:
    """Each WG id gets its own counter; one WG hitting the cap does NOT abandon
    a sibling WG that has died fewer times. This matters when an autoscaler has
    multiple WGs per stage and only one is in a bad slot/state."""
    pool = _make_pool()

    # WG A: cap-1 deaths, still restarting.
    for _ in range(_MAX_CONSECUTIVE_WG_DEATHS - 1):
        _install_worker_group(pool, "wg-a", ["actor-a"], [[_make_empty_slot()]])
        pool._handle_actor_death("wg-a", ray.exceptions.RayActorError())
    # WG B: one death, restarting.
    _install_worker_group(pool, "wg-b", ["actor-b"], [[_make_empty_slot()]])
    pool._handle_actor_death("wg-b", ray.exceptions.RayActorError())

    assert pool._consecutive_actor_deaths_by_wg_id["wg-a"] == _MAX_CONSECUTIVE_WG_DEATHS - 1
    assert pool._consecutive_actor_deaths_by_wg_id["wg-b"] == 1
    assert pool._num_worker_groups_abandoned == 0

    # WG A trips the cap; WG B's counter MUST be untouched.
    _install_worker_group(pool, "wg-a", ["actor-a"], [[_make_empty_slot()]])
    pool._handle_actor_death("wg-a", ray.exceptions.RayActorError("final straw"))

    assert pool._num_worker_groups_abandoned == 1
    assert "wg-a" not in pool._consecutive_actor_deaths_by_wg_id
    assert pool._consecutive_actor_deaths_by_wg_id["wg-b"] == 1


# ---------------------------------------------------------------------------
# Coordination invariant: per-task drop MUST fire at or before per-WG abandon
# ---------------------------------------------------------------------------


def test_cap_coordination_invariant_holds() -> None:
    """The per-WG cap MUST strictly exceed the per-task cap so that any task
    continuously cycling on a dying WG hits the per-task drop threshold no later
    than the WG is abandoned. Without this, a poison input on the only WG of a
    stage could remain queued at the exact moment we tear the WG down - with no
    worker to surface its next death and drop it - stranding the work and
    stalling the stream forever. Locking the invariant in code keeps a future
    tuner from accidentally setting these caps to the same value."""
    assert _MAX_CONSECUTIVE_WG_DEATHS > _MAX_ACTOR_DEATH_RETRIES, (
        f"coordination invariant broken: "
        f"_MAX_CONSECUTIVE_WG_DEATHS ({_MAX_CONSECUTIVE_WG_DEATHS}) must be strictly greater than "
        f"_MAX_ACTOR_DEATH_RETRIES ({_MAX_ACTOR_DEATH_RETRIES})"
    )


def test_single_poison_task_on_only_wg_does_not_strand_queue() -> None:
    """End-to-end regression for the 'stranded queue on single-WG abandonment' bug.

    Simulates the worst case the coordination invariant exists to handle:
      * exactly one WG on the stage
      * exactly one perpetually-poisoning task

    Each death rotates: install fresh WG -> death -> handler bumps task retries
    and WG death counter, requeues task, restarts WG. On the WG-cap-tripping
    death, the per-task cap MUST also fire (task dropped, NOT requeued), so the
    task queue is empty when the WG is finally abandoned. Otherwise the stream
    sits with queued work and no worker forever.
    """
    pool = _make_pool()
    poison = _make_task()

    # Pull the in-flight task back out of the queue between rounds (the real pool
    # would schedule it onto the freshly-recreated WG). The loop must continue up
    # to and INCLUDING the cap-tripping death (range is exclusive on the right).
    for _ in range(_MAX_CONSECUTIVE_WG_DEATHS):
        _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_slot_with_task(poison)]])
        pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))
        if pool._task_queue:
            popped = pool._task_queue.popleft()
            # Sanity: nothing else gets requeued by this fixture.
            assert popped is poison

    # Cap-tripping death must produce: (a) WG abandoned, (b) task dropped (not
    # requeued back into the queue), (c) drop counter incremented exactly once,
    # (d) the per-task retry counter pinned at the cap (never advanced past it).
    assert pool._num_worker_groups_abandoned == 1, "WG should be abandoned on cap-tripping death"
    assert len(pool._task_queue) == 0, (
        "task queue must be empty after the cap-tripping death; otherwise the "
        "stream is stranded with queued work and no worker to process it"
    )
    assert pool._num_tasks_dropped_on_actor_death == 1, "poison task must be dropped on the cap-tripping death"
    assert poison.actor_death_retries == _MAX_ACTOR_DEATH_RETRIES, (
        "per-task retry counter must not advance past the cap even when WG also hits its cap"
    )


def test_abandoning_last_wg_with_queued_work_raises_instead_of_stalling() -> None:
    """Queued work cannot be left behind when the WG circuit breaker trips.

    The cap coordination above handles the in-flight poison task. This covers
    the broader failure mode where unrelated work is already queued when the
    final capacity in the pool is abandoned. With no active, pending, or
    scheduled WG left, the only safe behavior is to fail loudly instead of
    leaving ``has_work_or_completed`` true forever.
    """
    pool = _make_pool()
    poison = _make_task()
    unrelated_queued_task = _make_task("upstream-node")

    for attempt in range(_MAX_CONSECUTIVE_WG_DEATHS):
        _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_slot_with_task(poison)]])

        if attempt == _MAX_CONSECUTIVE_WG_DEATHS - 1:
            pool._task_queue.append(unrelated_queued_task)
            with pytest.raises(RuntimeError, match="queued task"):
                pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("final straw"))
            break

        pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))
        assert pool._task_queue.popleft() is poison
        # Simulate the next update tick consuming the requested replacement.
        pool._worker_groups_to_create.clear()

    assert pool._num_worker_groups_abandoned == 1
    assert pool._num_tasks_dropped_on_actor_death == 1
    assert list(pool._task_queue) == [unrelated_queued_task]


# ---------------------------------------------------------------------------
# Double-delete guard: _kill_worker_groups_requested vs _handle_actor_death
# ---------------------------------------------------------------------------


def test_kill_worker_groups_requested_skips_already_torn_down_wg() -> None:
    """Same-tick race: ``_handle_actor_death`` already deleted ``wg-1`` (via an
    unexpected actor death on a sibling slot) by the time the
    ``should_restart_worker=True`` path tries to delete it again. The helper
    MUST skip the missing WG instead of raising ``KeyError`` on ``pop``."""
    pool = _make_pool()
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])

    pool._handle_actor_death("wg-1", ray.exceptions.RayActorError("died"))
    assert "wg-1" not in pool._worker_groups
    delete_calls_before = pool._delete_worker_group.call_count  # type: ignore[attr-defined]

    pool._kill_worker_groups_requested({"wg-1"})

    assert pool._delete_worker_group.call_count == delete_calls_before, (  # type: ignore[attr-defined]
        "helper must not re-delete a WG already torn down this tick"
    )


def test_kill_worker_groups_requested_deletes_present_wg() -> None:
    """When the WG is still present (no race), the helper deletes it normally."""
    pool = _make_pool()
    _install_worker_group(pool, "wg-1", ["actor-1"], [[_make_empty_slot()]])

    pool._kill_worker_groups_requested({"wg-1"})

    pool._delete_worker_group.assert_called_once_with("wg-1")  # type: ignore[attr-defined]
    assert "wg-1" not in pool._worker_groups


def test_kill_worker_groups_requested_handles_mixed_present_and_absent() -> None:
    """Mixed batch: one WG was torn down by death recovery, one wasn't. Only the
    present WG must be deleted; the missing one is silently skipped."""
    pool = _make_pool()
    _install_worker_group(pool, "wg-alive", ["actor-a"], [[_make_empty_slot()]])
    _install_worker_group(pool, "wg-dead", ["actor-d"], [[_make_empty_slot()]])

    pool._handle_actor_death("wg-dead", ray.exceptions.RayActorError("died"))
    delete_calls_after_death = pool._delete_worker_group.call_count  # type: ignore[attr-defined]

    pool._kill_worker_groups_requested({"wg-alive", "wg-dead"})

    # Exactly one additional delete: ``wg-alive``. ``wg-dead`` is skipped.
    assert pool._delete_worker_group.call_count == delete_calls_after_death + 1  # type: ignore[attr-defined]
    pool._delete_worker_group.assert_called_with("wg-alive")  # type: ignore[attr-defined]
    assert "wg-alive" not in pool._worker_groups


# ---------------------------------------------------------------------------
# SPMD same-tick race: primary success + sibling death must not double-emit
# ---------------------------------------------------------------------------


def _make_success_metadata() -> TaskResultMetadata:
    """Canonical primary-rank success metadata used by the same-tick race tests.

    ``should_process_further=False`` lets the test skip the
    ``_completed_tasks.append(Task(..., actor.metadata.allocation.node))`` path
    (which would need an ``allocation`` field on ``_StubMetadata``). The SPMD
    post-success dedupe-recording lives after both branches and still fires
    here, which is what the tests care about.
    """
    return TaskResultMetadata(
        timing=TimingInfo(),
        failure_info=FailureInfo(should_process_further=False, should_restart_worker=False),
        task_data_info=TaskDataInfo(serialized_input_size=0),
        num_returns=0,
    )


def _make_success_metadata_with_restart() -> TaskResultMetadata:
    """Primary-rank success metadata that also requests a worker restart.

    Same ``should_process_further=False`` rationale as ``_make_success_metadata``.
    """
    return TaskResultMetadata(
        timing=TimingInfo(),
        failure_info=FailureInfo(should_process_further=False, should_restart_worker=True),
        task_data_info=TaskDataInfo(serialized_input_size=0),
        num_returns=0,
    )


def test_spmd_primary_success_then_sibling_death_does_not_double_emit() -> None:
    """Regression for the SPMD post-success race.

    ``_schedule_task_on_worker_group`` dispatches the same ``Task`` instance to
    every rank's slot. If ``ray.wait`` surfaces a primary's success ref AND a
    sibling rank's death ref in the same ``_process_completed_tasks`` tick, and
    the primary is processed FIRST:

      1. Primary succeeds -> result appended to the completed queue, primary
         slot cleared.
      2. Sibling raises ``ActorDiedError`` -> ``_handle_actor_death`` walks
         ``wg.actors``; primary's slot is empty (just cleared), sibling's slot
         still holds the task.

    Without the post-success dedupe, the death handler would requeue the task,
    a new primary would process it again, and the result would be emitted a
    second time for the same logical input - a silent at-least-twice violation.

    The fix threads ``successful_task_ids_by_wg`` through ``_process_completed_task``
    and into ``_handle_actor_death`` so the sibling-death code path recognizes the
    task as already-emitted and skips the requeue.
    """
    pool = _make_pool()
    shared_task = _make_task()
    _install_worker_group(
        pool,
        "wg-spmd",
        ["rank-0", "rank-1"],
        [[_make_slot_with_task(shared_task)], [_make_slot_with_task(shared_task)]],
    )
    primary_actor = pool._ready_actors["rank-0"]
    sibling_actor = pool._ready_actors["rank-1"]

    deaths_seen: set[str] = set()
    successful_task_ids_by_wg: dict[str, set[int]] = {}

    # Step 1: primary processes successfully. Drives the real
    # ``_process_completed_task`` so the dedupe-record code path is exercised
    # end-to-end (not just the dict mutation in isolation).
    metadata = _make_success_metadata()
    with mock.patch(
        "cosmos_xenna.ray_utils.actor_pool.ray.get",
        side_effect=[[mock.sentinel.metadata_ref], metadata],
    ):
        pool._process_completed_task(
            primary_actor,
            slot_num=0,
            is_primary=True,
            deaths_seen=deaths_seen,
            successful_task_ids_by_wg=successful_task_ids_by_wg,
        )

    assert not primary_actor.slots[0].has_task, "primary success must clear the slot"
    assert successful_task_ids_by_wg.get("wg-spmd") == {id(shared_task)}, (
        "primary success must record id(task) so a same-tick sibling death can dedupe"
    )

    # Step 2: sibling raises ActorDiedError on the same logical task. The death
    # handler is invoked through ``_process_completed_task`` so the kwarg
    # threading is exercised end-to-end.
    with mock.patch(
        "cosmos_xenna.ray_utils.actor_pool.ray.get",
        side_effect=ray.exceptions.ActorDiedError(),
    ):
        pool._process_completed_task(
            sibling_actor,
            slot_num=0,
            is_primary=False,
            deaths_seen=deaths_seen,
            successful_task_ids_by_wg=successful_task_ids_by_wg,
        )

    # Death handler ran and tore down the WG, but it MUST NOT have requeued the
    # task - that would let a new primary re-emit the same logical result.
    assert "wg-spmd" in deaths_seen
    assert "wg-spmd" not in pool._worker_groups, "death handler must tear down the dead WG"
    assert list(pool._task_queue) == [], (
        "task must not be requeued (primary's result is canonical; requeue would double-emit)"
    )
    assert pool._num_tasks_dropped_on_actor_death == 0, "post-success skip is not a drop"
    assert shared_task.actor_death_retries == 0, "skipped-task retry counter must NOT advance"
    assert pool._num_unexpected_actor_deaths == 1


def test_spmd_sibling_death_then_primary_success_short_circuits() -> None:
    """The reverse iteration order is already safe via the existing
    ``if not slot.has_task: return False`` short-circuit at the top of
    ``_process_completed_task``. This locks that invariant in.

    Order: sibling-death first -> death handler clears every slot on the WG,
    tears down the WG, and re-schedules. Then primary's success iteration sees
    its own slot already empty and returns False without calling ``ray.get``.
    Net effect: task is requeued exactly once by the death handler; primary's
    successful result is not consulted (it lives in Ray's object store and is
    GC'd). No duplicate.
    """
    pool = _make_pool()
    shared_task = _make_task()
    _install_worker_group(
        pool,
        "wg-spmd",
        ["rank-0", "rank-1"],
        [[_make_slot_with_task(shared_task)], [_make_slot_with_task(shared_task)]],
    )
    primary_actor = pool._ready_actors["rank-0"]
    sibling_actor = pool._ready_actors["rank-1"]

    deaths_seen: set[str] = set()
    successful_task_ids_by_wg: dict[str, set[int]] = {}

    # Step 1: sibling dies first.
    with mock.patch(
        "cosmos_xenna.ray_utils.actor_pool.ray.get",
        side_effect=ray.exceptions.ActorDiedError(),
    ):
        pool._process_completed_task(
            sibling_actor,
            slot_num=0,
            is_primary=False,
            deaths_seen=deaths_seen,
            successful_task_ids_by_wg=successful_task_ids_by_wg,
        )

    assert "wg-spmd" not in pool._worker_groups
    # Death handler requeued exactly once and bumped the retry counter.
    assert list(pool._task_queue) == [shared_task]
    assert shared_task.actor_death_retries == 1
    # Primary's slot was cleared by the death handler.
    assert not primary_actor.slots[0].has_task

    # Step 2: primary's "success" iteration short-circuits. The mock must NOT
    # be consulted (the slot is already empty) - if it is, the call_count
    # check will catch it.
    fake_get = mock.MagicMock()
    with mock.patch("cosmos_xenna.ray_utils.actor_pool.ray.get", new=fake_get):
        result = pool._process_completed_task(
            primary_actor,
            slot_num=0,
            is_primary=True,
            deaths_seen=deaths_seen,
            successful_task_ids_by_wg=successful_task_ids_by_wg,
        )

    assert result is False
    assert fake_get.call_count == 0, "short-circuit must NOT consult ray.get on an empty slot"
    # No second requeue, no extra retry bump, no spurious entry in the dedupe map.
    assert list(pool._task_queue) == [shared_task]
    assert shared_task.actor_death_retries == 1
    assert "wg-spmd" not in successful_task_ids_by_wg


def test_spmd_mixed_tasks_post_success_dedupe_is_per_task_not_per_wg() -> None:
    """Post-success dedupe is keyed on ``id(task)``, NOT on WG id.

    Setup: primary succeeded on task A this tick, but sibling slot still holds
    a DIFFERENT task B (e.g. multiple slots per actor with distinct tasks).
    When the sibling dies, A must NOT be requeued (primary emitted it) but B
    MUST be requeued (no result was ever produced for B).
    """
    pool = _make_pool()
    task_a = _make_task("node-a")
    task_b = _make_task("node-b")
    _install_worker_group(
        pool,
        "wg-mixed",
        ["rank-0", "rank-1"],
        [
            [_make_slot_with_task(task_a)],
            [_make_slot_with_task(task_a), _make_slot_with_task(task_b)],
        ],
    )

    # Primary already succeeded on task_a this tick.
    already_succeeded = {id(task_a)}

    pool._handle_actor_death("wg-mixed", ray.exceptions.RayActorError(), already_succeeded_task_ids=already_succeeded)

    queued_ids = [id(t) for t in pool._task_queue]
    assert queued_ids == [id(task_b)], (
        f"only task_b should be requeued; task_a was already emitted by primary. got={queued_ids}"
    )
    assert task_a.actor_death_retries == 0, "skipped (already-emitted) task must NOT advance retry counter"
    assert task_b.actor_death_retries == 1, "unrelated task must follow the normal requeue path"
    assert pool._num_tasks_dropped_on_actor_death == 0


# ---------------------------------------------------------------------------
# should_restart_worker=True same-tick race with death recovery
# ---------------------------------------------------------------------------


def test_should_restart_worker_race_with_sibling_death_does_not_double_delete() -> None:
    """End-to-end same-tick race: primary returns ``should_restart_worker=True``
    (so the caller plans to call ``_kill_worker_groups_requested({wg_id})``)
    AND a sibling rank on the same WG raises ``ActorDiedError`` in the same
    ``_process_completed_tasks`` tick (so ``_handle_actor_death`` has already
    torn down the WG by the time the helper runs).

    The double-delete guard in ``_kill_worker_groups_requested`` must skip the
    missing WG instead of raising ``KeyError`` on ``self._worker_groups.pop``.
    This complements ``test_kill_worker_groups_requested_skips_already_torn_down_wg``
    (which exercises the helper in isolation) by driving the actual same-tick
    sequence through ``_process_completed_task``.
    """
    pool = _make_pool()
    shared_task = _make_task()
    _install_worker_group(
        pool,
        "wg-race",
        ["rank-0", "rank-1"],
        [[_make_slot_with_task(shared_task)], [_make_slot_with_task(shared_task)]],
    )
    primary_actor = pool._ready_actors["rank-0"]
    sibling_actor = pool._ready_actors["rank-1"]

    deaths_seen: set[str] = set()
    successful_task_ids_by_wg: dict[str, set[int]] = {}
    wgs_to_kill: set[str] = set()

    # Step 1: primary succeeds with should_restart_worker=True.
    metadata = _make_success_metadata_with_restart()
    with mock.patch(
        "cosmos_xenna.ray_utils.actor_pool.ray.get",
        side_effect=[[mock.sentinel.metadata_ref], metadata],
    ):
        should_kill = pool._process_completed_task(
            primary_actor,
            slot_num=0,
            is_primary=True,
            deaths_seen=deaths_seen,
            successful_task_ids_by_wg=successful_task_ids_by_wg,
        )
    assert should_kill is True, "primary returned should_restart_worker=True"
    wgs_to_kill.add("wg-race")

    # Step 2: sibling dies; death handler tears down the WG.
    with mock.patch(
        "cosmos_xenna.ray_utils.actor_pool.ray.get",
        side_effect=ray.exceptions.ActorDiedError(),
    ):
        pool._process_completed_task(
            sibling_actor,
            slot_num=0,
            is_primary=False,
            deaths_seen=deaths_seen,
            successful_task_ids_by_wg=successful_task_ids_by_wg,
        )
    assert "wg-race" not in pool._worker_groups, "death handler must have torn down the WG"

    # Step 3: the helper runs with wg-race in wgs_to_kill but the WG is gone.
    # The guard must skip cleanly instead of raising KeyError on pop.
    pool._kill_worker_groups_requested(wgs_to_kill)

    # No extra delete beyond what the death handler already did. The death
    # handler called _delete_worker_group exactly once for wg-race.
    pool._delete_worker_group.assert_called_once_with("wg-race")  # type: ignore[attr-defined]
    # And no double-emission of the task: primary's success stands, sibling
    # death's requeue was suppressed by the post-success dedupe.
    assert list(pool._task_queue) == []
    assert shared_task.actor_death_retries == 0
