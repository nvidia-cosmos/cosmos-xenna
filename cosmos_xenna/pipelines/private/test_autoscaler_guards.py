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

"""Tests for the queue-aware and rate-limited scale-down guards in Autoscaler.apply_autoscale_result_if_ready."""

import collections
import concurrent.futures
from unittest.mock import MagicMock

from cosmos_xenna.pipelines.private.streaming import Autoscaler, Queue


def _make_mock_worker(worker_id: str = "w1") -> MagicMock:
    """Create a mock ProblemWorkerGroupState with a to_worker_group method."""
    worker = MagicMock()
    worker.to_worker_group.return_value = f"worker_group_{worker_id}"
    return worker


def _make_mock_stage_result(num_new: int = 0, num_delete: int = 0, slots_per_worker: int = 2) -> MagicMock:
    """Create a mock StageSolution with controllable new/deleted worker lists.

    Returns a mock whose deleted_workers and new_workers are plain lists
    (not properties), so the guard logic can snapshot and filter them.
    """
    result = MagicMock()
    result.slots_per_worker = slots_per_worker
    result.deleted_workers = [_make_mock_worker(f"del_{i}") for i in range(num_delete)]
    result.new_workers = [_make_mock_worker(f"new_{i}") for i in range(num_new)]
    return result


def _make_mock_solution(stage_results: list[MagicMock]) -> MagicMock:
    """Create a mock Solution wrapping a list of StageSolution mocks."""
    solution = MagicMock()
    solution.stages = stage_results
    return solution


def _make_mock_pool(
    name: str = "Stage 00",
    num_ready_actors: int = 10,
    task_queue_size: int = 0,
    slots_per_actor: int = 2,
) -> MagicMock:
    """Create a mock ActorPool with controllable queue and actor counts."""
    pool = MagicMock()
    pool.name = name
    pool.num_ready_actors = num_ready_actors
    pool.slots_per_actor = slots_per_actor
    pool._task_queue = collections.deque(range(task_queue_size))
    pool.add_actor_to_create = MagicMock()
    pool.add_actor_to_delete = MagicMock()
    pool.set_num_slots_per_actor = MagicMock()
    return pool


def _make_queue_with_items(num_items: int) -> Queue:
    """Create a Queue and populate it with dummy items."""
    q = Queue()
    if num_items > 0:
        q.by_node_id[None] = collections.deque(range(num_items))
    return q


def _make_autoscaler(max_scale_down_fraction: float = 0.5) -> Autoscaler:
    """Create an Autoscaler instance bypassing __init__, setting only
    the attributes needed for apply_autoscale_result_if_ready."""
    autoscaler = object.__new__(Autoscaler)
    autoscaler._autoscale_future = None
    autoscaler._autoscale_start_time = 0.0
    autoscaler._max_scale_down_fraction = max_scale_down_fraction
    autoscaler._verbosity_level = 0
    return autoscaler


def _set_ready_future(autoscaler: Autoscaler, solution: MagicMock) -> None:
    """Set a completed future on the autoscaler with the given solution."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(solution)
    autoscaler._autoscale_future = future
    autoscaler._autoscale_start_time = 0.0


# ---------------------------------------------------------------------------
# Guard 1: Queue-aware scale-down protection
# ---------------------------------------------------------------------------


class TestQueueAwareGuard:
    """Guard 1 blocks deletions when upstream or pool queues have backlog."""

    def test_blocks_deletions_when_upstream_queue_has_items(self):
        """Stage 0 should not have workers deleted when input_queue is non-empty."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=18)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(10)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        pool.add_actor_to_delete.assert_not_called()

    def test_blocks_deletions_when_pool_task_queue_has_items(self):
        """Deletions should be blocked when the pool's internal task queue has work."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        pool = _make_mock_pool("Stage 01", num_ready_actors=10, task_queue_size=5)
        stage_result = _make_mock_stage_result(num_delete=8)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        pool.add_actor_to_delete.assert_not_called()

    def test_blocks_deletions_when_inter_stage_queue_has_items(self):
        """Stage 1 deletions blocked when queues[0] (its upstream) has items."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        pool0 = _make_mock_pool("Stage 00", num_ready_actors=1, task_queue_size=0)
        pool1 = _make_mock_pool("Stage 01", num_ready_actors=10, task_queue_size=0)
        stage_result_0 = _make_mock_stage_result(num_delete=0)
        stage_result_1 = _make_mock_stage_result(num_delete=8)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result_0, stage_result_1]))

        input_queue = _make_queue_with_items(0)
        upstream_queue = _make_queue_with_items(15)
        queues = [upstream_queue, _make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool0, pool1], queues, input_queue)

        pool1.add_actor_to_delete.assert_not_called()

    def test_allows_deletions_when_queues_are_empty(self):
        """When both upstream and pool queues are empty, deletions proceed."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=5)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_delete.call_count == 5

    def test_new_workers_still_created_when_deletions_blocked(self):
        """Guard 1 only blocks deletions, not worker creation."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        pool = _make_mock_pool("Stage 00", num_ready_actors=5, task_queue_size=10)
        stage_result = _make_mock_stage_result(num_new=3, num_delete=4)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_create.call_count == 3
        pool.add_actor_to_delete.assert_not_called()


# ---------------------------------------------------------------------------
# Guard 2: Rate-limited scale-down
# ---------------------------------------------------------------------------


class TestRateLimitGuard:
    """Guard 2 caps deletions to max_scale_down_fraction of current actors."""

    def test_clamps_deletions_to_fraction(self):
        """With 20 actors and fraction=0.5, at most 10 can be deleted."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=18)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_delete.call_count == 10

    def test_allows_all_deletions_when_below_threshold(self):
        """Deletions at or below the threshold are not clamped."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=5)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_delete.call_count == 5

    def test_always_allows_at_least_one_deletion(self):
        """Even with a very small fraction, at least 1 deletion is allowed."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.01)
        pool = _make_mock_pool("Stage 00", num_ready_actors=3, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=3)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_delete.call_count == 1

    def test_disabled_when_fraction_is_one(self):
        """Setting fraction=1.0 effectively disables the rate limiter."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=18)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_delete.call_count == 18

    def test_gradual_scaledown_trajectory(self):
        """Verify the expected multi-cycle trajectory: 20->10->5->3->2->1."""
        trajectory = []
        current_actors = 20

        for _ in range(6):
            autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
            pool = _make_mock_pool("Stage 00", num_ready_actors=current_actors, task_queue_size=0)
            stage_result = _make_mock_stage_result(num_delete=current_actors - 1)
            _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

            input_queue = _make_queue_with_items(0)
            queues = [_make_queue_with_items(0)]

            autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

            deleted = pool.add_actor_to_delete.call_count
            current_actors -= deleted
            trajectory.append(current_actors)
            if current_actors <= 1:
                break

        assert trajectory == [10, 5, 3, 2, 1]


# ---------------------------------------------------------------------------
# Both guards combined
# ---------------------------------------------------------------------------


class TestCombinedGuards:
    """Test the interaction of both guards applied sequentially."""

    def test_queue_guard_takes_precedence_over_rate_limit(self):
        """If Guard 1 blocks all deletions, Guard 2 has nothing to do."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=5)
        stage_result = _make_mock_stage_result(num_delete=18)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        pool.add_actor_to_delete.assert_not_called()

    def test_rate_limit_applied_after_queue_guard_allows(self):
        """When queues are empty but rate limit applies, deletions are clamped."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        stage_result = _make_mock_stage_result(num_delete=18)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        assert pool.add_actor_to_delete.call_count == 10

    def test_multi_stage_pipeline_independent_guards(self):
        """Each stage is guarded independently -- stage 0 may be blocked
        while stage 1 gets clamped."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)

        pool0 = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=0)
        pool1 = _make_mock_pool("Stage 01", num_ready_actors=10, task_queue_size=0)

        stage_result_0 = _make_mock_stage_result(num_delete=18)
        stage_result_1 = _make_mock_stage_result(num_delete=8)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result_0, stage_result_1]))

        input_queue = _make_queue_with_items(30)
        upstream_for_1 = _make_queue_with_items(0)
        queues = [upstream_for_1, _make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool0, pool1], queues, input_queue)

        # Stage 0: queue-aware guard blocks (input_queue=30)
        pool0.add_actor_to_delete.assert_not_called()
        # Stage 1: queues empty, rate-limit clamps 8 -> 5
        assert pool1.add_actor_to_delete.call_count == 5

    def test_no_future_does_nothing(self):
        """When no autoscale future is pending, the method is a no-op."""
        autoscaler = _make_autoscaler()
        pool = _make_mock_pool("Stage 00")

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        pool.add_actor_to_create.assert_not_called()
        pool.add_actor_to_delete.assert_not_called()
        pool.set_num_slots_per_actor.assert_not_called()

    def test_slots_per_actor_always_applied(self):
        """Slot configuration is always applied regardless of guard outcomes."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        pool = _make_mock_pool("Stage 00", num_ready_actors=20, task_queue_size=10)
        stage_result = _make_mock_stage_result(num_delete=18, slots_per_worker=4)
        _set_ready_future(autoscaler, _make_mock_solution([stage_result]))

        input_queue = _make_queue_with_items(0)
        queues = [_make_queue_with_items(0)]

        autoscaler.apply_autoscale_result_if_ready([pool], queues, input_queue)

        pool.set_num_slots_per_actor.assert_called_once_with(4)
