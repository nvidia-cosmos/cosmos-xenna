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

"""Unit tests for the backlog-aware scale-down guard.

CPU-only tests for ``_required_workers_for_stage`` (algorithmic core) and
``Autoscaler.apply_autoscale_result_if_ready`` (deletion clamping).
"""

import concurrent.futures
import logging
import math
from collections.abc import Iterator
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private.streaming import (
    Autoscaler,
    _required_workers_for_stage,
)


def _make_mock_pool(
    *,
    name: str = "stage",
    num_ready_actors: int,
    num_used_slots: int,
    slots_per_actor: int,
    num_queued_tasks: int,
) -> MagicMock:
    """Build a lightweight mock ActorPool exposing only what the guard reads."""
    pool = MagicMock(name=f"pool-{name}")
    pool.name = name
    pool.num_ready_actors = num_ready_actors
    pool.num_used_slots = num_used_slots
    pool.slots_per_actor = slots_per_actor
    pool.num_queued_tasks = num_queued_tasks
    return pool


def _make_solution_stage(
    *,
    new_workers: Optional[list[Any]] = None,
    deleted_workers: Optional[list[Any]] = None,
    slots_per_worker: int = 1,
) -> MagicMock:
    """Build a mock ``StageSolution`` matching what Rust would return."""
    stage = MagicMock(name="stage-solution")
    stage.new_workers = new_workers or []
    stage.deleted_workers = deleted_workers or []
    stage.slots_per_worker = slots_per_worker
    return stage


def _make_solution(stage_solutions: list[MagicMock]) -> MagicMock:
    """Build a mock ``Solution`` object with the given stage solutions."""
    solution = MagicMock(name="solution")
    solution.stages = stage_solutions
    return solution


def _make_deleted_worker(worker_id: str = "w") -> MagicMock:
    """Build a mock ``ProblemWorkerGroupState`` suitable for deletion tracking."""
    w = MagicMock(name=f"deleted-{worker_id}")
    w.to_worker_group.return_value = MagicMock(name=f"worker-group-{worker_id}")
    return w


def _make_autoscaler_with_solution(solution: MagicMock) -> Autoscaler:
    """Construct an ``Autoscaler`` bypassing ``__init__`` and pre-loaded solution."""
    autoscaler = object.__new__(Autoscaler)
    future: concurrent.futures.Future[MagicMock] = concurrent.futures.Future()
    future.set_result(solution)
    autoscaler._autoscale_future = future
    autoscaler._autoscale_start_time = 0.0
    return autoscaler


def test_helper_floor_when_no_inflight_no_backlog() -> None:
    """Return MIN_WORKERS_PER_STAGE when nothing is in-flight and nothing is queued."""
    required = _required_workers_for_stage(
        slots_per_actor=4,
        stage_batch_size=1,
        inflight_slots=0,
        backlog_samples=0,
    )
    assert required == Autoscaler.MIN_WORKERS_PER_STAGE


def test_helper_uses_max_of_inflight_and_backlog() -> None:
    """Pick max(workers_for_inflight, workers_for_backlog) when both apply."""
    # slots_per_actor=2, batch=1 -> capacity_per_actor=2
    # inflight=4  -> workers_for_inflight = ceil(4/2) = 2
    # backlog=12  -> workers_for_backlog  = ceil(12/2) = 6
    required = _required_workers_for_stage(
        slots_per_actor=2,
        stage_batch_size=1,
        inflight_slots=4,
        backlog_samples=12,
    )
    assert required == 6


def test_helper_inflight_only_path() -> None:
    """With no backlog, required workers == ceil(inflight/slots_per_actor)."""
    required = _required_workers_for_stage(
        slots_per_actor=3,
        stage_batch_size=4,
        inflight_slots=7,
        backlog_samples=0,
    )
    assert required == math.ceil(7 / 3)


def test_helper_backlog_only_path() -> None:
    """With no in-flight work, required workers == ceil(backlog/capacity_per_actor)."""
    # capacity_per_actor = slots_per_actor * batch = 2 * 4 = 8
    required = _required_workers_for_stage(
        slots_per_actor=2,
        stage_batch_size=4,
        inflight_slots=0,
        backlog_samples=20,
    )
    assert required == math.ceil(20 / 8)


def test_helper_raises_on_zero_slots_per_actor() -> None:
    """Invalid slots_per_actor triggers ValueError, never a ZeroDivisionError."""
    with pytest.raises(ValueError, match="slots_per_actor"):
        _required_workers_for_stage(
            slots_per_actor=0,
            stage_batch_size=1,
            inflight_slots=0,
            backlog_samples=0,
        )


def test_helper_raises_on_zero_stage_batch_size() -> None:
    """Invalid stage_batch_size triggers ValueError, never a ZeroDivisionError."""
    with pytest.raises(ValueError, match="stage_batch_size"):
        _required_workers_for_stage(
            slots_per_actor=1,
            stage_batch_size=0,
            inflight_slots=0,
            backlog_samples=0,
        )


def test_blocks_deletions_when_backlog_exists() -> None:
    """Clamp deletions when the pool has a heavy backlog."""
    # 20 actors, 2 slots each, pool_q=36, no in-flight -> required = ceil(36/2)=18
    # max_safe_deletions = 20-18=2
    pool = _make_mock_pool(
        name="VideoFrameExtractionStage",
        num_ready_actors=20,
        num_used_slots=0,
        slots_per_actor=2,
        num_queued_tasks=36,
    )
    solution = _make_solution(
        [_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(18)])]
    )
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 2


def test_allows_deletions_when_no_backlog_no_inflight() -> None:
    """When there is nothing to protect, all proposed deletions pass through."""
    pool = _make_mock_pool(
        num_ready_actors=20,
        num_used_slots=0,
        slots_per_actor=2,
        num_queued_tasks=0,
    )
    solution = _make_solution(
        [_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(18)])]
    )
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    # 20 actors, required floor = MIN_WORKERS_PER_STAGE = 1, so deletions capped at 19.
    assert pool.add_actor_to_delete.call_count == min(18, 20 - Autoscaler.MIN_WORKERS_PER_STAGE)


def test_enforces_min_workers_floor_on_drain_tail() -> None:
    """Guard keeps MIN_WORKERS_PER_STAGE actors alive even when Rust proposes full deletion.

    Drain-tail scenario: upstream has gone silent, the pool's queue is empty, and
    nothing is in flight, so Rust (throughput-blind) proposes deleting every
    remaining actor. Fully applying that proposal would cost an autoscaler cycle
    to cold-start a worker when fresh work finally arrives. The guard must leave
    ``MIN_WORKERS_PER_STAGE`` workers alive on the active-stage path; actual
    end-of-pipeline teardown happens via the pipeline context exit, not here.
    """
    # current=2, required = max(MIN_WORKERS_PER_STAGE=1, 0, 0) = 1
    # max_safe = max(0, 2 - 1) = 1; Rust proposed 2 -> allowed = 1.
    pool = _make_mock_pool(
        num_ready_actors=2,
        num_used_slots=0,
        slots_per_actor=1,
        num_queued_tasks=0,
    )
    solution = _make_solution([_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(2)])])
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 2 - Autoscaler.MIN_WORKERS_PER_STAGE


def test_preserves_workers_for_inflight_only() -> None:
    """In-flight slots dominate the floor when there is no queued backlog."""
    # slots_per_actor=2, inflight=8 -> required = ceil(8/2) = 4
    # current=10 -> max_safe = 6
    pool = _make_mock_pool(
        num_ready_actors=10,
        num_used_slots=8,
        slots_per_actor=2,
        num_queued_tasks=0,
    )
    solution = _make_solution([_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(9)])])
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 6


def test_preserves_workers_for_upstream_backlog() -> None:
    """Upstream queue backlog counts the same as the pool's own queue backlog."""
    # slots_per_actor=2, batch=1, capacity=2, upstream=20, pool_q=0 -> required=ceil(20/2)=10
    # current=10 -> max_safe = 0
    pool = _make_mock_pool(
        num_ready_actors=10,
        num_used_slots=0,
        slots_per_actor=2,
        num_queued_tasks=0,
    )
    solution = _make_solution([_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(5)])])
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[20],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 0


def test_does_not_amplify_rust_proposal() -> None:
    """Guard never deletes more workers than Rust proposed."""
    # max_safe_deletions = 10-1 = 9, but Rust only proposed 3 -> applied=3
    pool = _make_mock_pool(
        num_ready_actors=10,
        num_used_slots=0,
        slots_per_actor=2,
        num_queued_tasks=0,
    )
    solution = _make_solution([_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(3)])])
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 3


def test_first_stage_uses_input_queue_length() -> None:
    """Stage index 0 reads ``upstream_queue_lens[0]`` (the pipeline input queue)."""
    pool = _make_mock_pool(
        num_ready_actors=5,
        num_used_slots=0,
        slots_per_actor=1,
        num_queued_tasks=0,
    )
    solution = _make_solution([_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(4)])])
    autoscaler = _make_autoscaler_with_solution(solution)

    # upstream=5 tasks, batch=1, slots=1 -> capacity=1 -> required=5 -> max_safe=0
    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[5],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 0


def test_non_first_stage_uses_its_own_upstream_queue_entry() -> None:
    """Stage index 1 reads ``upstream_queue_lens[1]`` (not stage 0's entry)."""
    pool0 = _make_mock_pool(
        name="s0",
        num_ready_actors=5,
        num_used_slots=0,
        slots_per_actor=1,
        num_queued_tasks=0,
    )
    pool1 = _make_mock_pool(
        name="s1",
        num_ready_actors=5,
        num_used_slots=0,
        slots_per_actor=1,
        num_queued_tasks=0,
    )
    solution = _make_solution(
        [
            _make_solution_stage(deleted_workers=[_make_deleted_worker(f"s0-w{i}") for i in range(4)]),
            _make_solution_stage(deleted_workers=[_make_deleted_worker(f"s1-w{i}") for i in range(4)]),
        ]
    )
    autoscaler = _make_autoscaler_with_solution(solution)

    # Stage 0 has no upstream backlog (=0) -> allowed to scale down to floor.
    # Stage 1 sees 5 tasks in its upstream -> required=5 -> max_safe=0.
    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool0, pool1],
        upstream_queue_lens=[0, 5],
        stage_batch_sizes=[1, 1],
    )

    assert pool0.add_actor_to_delete.call_count == 4
    assert pool1.add_actor_to_delete.call_count == 0


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    ``cosmos_xenna.utils.python_log`` routes logging through loguru, which does
    not propagate to the stdlib ``logging`` module by default, so ``caplog`` is
    otherwise blind to ``logger.info(...)`` calls. The bridge re-emits every
    loguru record through a stdlib logger named ``"loguru"``. The sink is torn
    down automatically at the end of the test.
    """
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        level=0,
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


def test_logs_clamping_at_info_level(loguru_caplog: pytest.LogCaptureFixture) -> None:
    """INFO log is emitted with stage name + counts when the guard clamps deletions."""
    # num_queued=20, slots=1, batch=1 -> required = max(1, 0, ceil(20/1)) = 20
    # current=13, required=20 -> max_safe = max(0, 13-20) = 0, so allowed = 0.
    pool = _make_mock_pool(
        name="ClipTranscodingStage",
        num_ready_actors=13,
        num_used_slots=0,
        slots_per_actor=1,
        num_queued_tasks=20,
    )
    solution = _make_solution(
        [_make_solution_stage(deleted_workers=[_make_deleted_worker(f"w{i}") for i in range(11)])]
    )
    autoscaler = _make_autoscaler_with_solution(solution)

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    clamp_messages = [rec.getMessage() for rec in loguru_caplog.records if "Clamped scale-down" in rec.getMessage()]
    assert len(clamp_messages) == 1
    message = clamp_messages[0]
    assert "ClipTranscodingStage" in message
    assert "current=13" in message
    assert "required=20" in message
    assert "proposed_delete=11" in message
    assert "allowed_delete=0" in message
    assert "pool_q=20" in message


def test_no_future_does_nothing() -> None:
    """Method is a safe no-op when no autoscale future is pending."""
    autoscaler = object.__new__(Autoscaler)
    autoscaler._autoscale_future = None
    autoscaler._autoscale_start_time = 0.0

    pool = _make_mock_pool(
        num_ready_actors=5,
        num_used_slots=0,
        slots_per_actor=1,
        num_queued_tasks=0,
    )

    autoscaler.apply_autoscale_result_if_ready(
        pools=[pool],
        upstream_queue_lens=[0],
        stage_batch_sizes=[1],
    )

    assert pool.add_actor_to_delete.call_count == 0
    assert pool.add_actor_to_create.call_count == 0
