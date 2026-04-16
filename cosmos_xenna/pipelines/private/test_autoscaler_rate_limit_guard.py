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

"""Tests for the rate-limited scale-down guard in Autoscaler.apply_autoscale_result_if_ready.

Verifies that the guard caps deletions per autoscale cycle to
``max_scale_down_fraction`` of current ready actors, preventing
aggressive cliff-effect scale-downs (e.g. 20 -> 2 in one cycle).
"""

import concurrent.futures
from unittest.mock import MagicMock

import pytest

from cosmos_xenna.pipelines.private.specs import StreamingSpecificSpec
from cosmos_xenna.pipelines.private.streaming import Autoscaler


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


def _make_mock_pool(name: str = "TestStage", num_ready_actors: int = 20) -> MagicMock:
    """Create a mock ActorPool with controllable num_ready_actors."""
    pool = MagicMock()
    pool.name = name
    pool.num_ready_actors = num_ready_actors
    return pool


def _make_autoscaler(max_scale_down_fraction: float = 0.5) -> Autoscaler:
    """Create an Autoscaler bypassing __init__ for test isolation.

    Must manually set all attributes read by apply_autoscale_result_if_ready:
    _autoscale_future, _autoscale_start_time, _verbosity_level,
    _max_scale_down_fraction.  Keep in sync with Autoscaler.__init__.
    """
    autoscaler = object.__new__(Autoscaler)
    autoscaler._autoscale_future = None
    autoscaler._autoscale_start_time = 0.0
    autoscaler._verbosity_level = 0
    autoscaler._max_scale_down_fraction = max_scale_down_fraction
    return autoscaler


def _inject_solution(autoscaler: Autoscaler, solution: MagicMock) -> None:
    """Inject a completed future with the given solution into the autoscaler."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(solution)
    autoscaler._autoscale_future = future


class TestRateLimitGuard:
    """Rate-limit guard caps deletions to max_scale_down_fraction of current actors."""

    def test_clamps_deletions_to_fraction(self) -> None:
        """20 actors, 18 deletions requested, fraction=0.5 -> only 10 applied."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        stage_result = _make_mock_stage_result(num_delete=18)
        solution = _make_mock_solution([stage_result])
        pool = _make_mock_pool(num_ready_actors=20)
        _inject_solution(autoscaler, solution)

        autoscaler.apply_autoscale_result_if_ready([pool])

        assert pool.add_actor_to_delete.call_count == 10

    def test_allows_all_deletions_when_below_threshold(self) -> None:
        """20 actors, 5 deletions requested, fraction=0.5 -> all 5 applied (5 <= 10)."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        stage_result = _make_mock_stage_result(num_delete=5)
        solution = _make_mock_solution([stage_result])
        pool = _make_mock_pool(num_ready_actors=20)
        _inject_solution(autoscaler, solution)

        autoscaler.apply_autoscale_result_if_ready([pool])

        assert pool.add_actor_to_delete.call_count == 5

    def test_always_allows_at_least_one_deletion(self) -> None:
        """100 actors, 50 deletions, fraction=0.01 -> max(1, int(1.0)) = 1 applied."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.01)
        stage_result = _make_mock_stage_result(num_delete=50)
        solution = _make_mock_solution([stage_result])
        pool = _make_mock_pool(num_ready_actors=100)
        _inject_solution(autoscaler, solution)

        autoscaler.apply_autoscale_result_if_ready([pool])

        assert pool.add_actor_to_delete.call_count == 1

    def test_disabled_when_fraction_is_one(self) -> None:
        """Setting fraction=1.0 effectively disables the rate limiter."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=1.0)
        stage_result = _make_mock_stage_result(num_delete=18)
        solution = _make_mock_solution([stage_result])
        pool = _make_mock_pool(num_ready_actors=20)
        _inject_solution(autoscaler, solution)

        autoscaler.apply_autoscale_result_if_ready([pool])

        assert pool.add_actor_to_delete.call_count == 18

    def test_gradual_scaledown_trajectory(self) -> None:
        """Simulates the 20 -> 10 -> 5 -> 3 -> 2 -> 1 trajectory over 5 cycles.

        Each cycle, the autoscaler requests deleting all actors. The guard
        clamps to 50%, producing exponential decay to 1.
        """
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        current_actors = 20
        trajectory = [current_actors]

        for _ in range(5):
            stage_result = _make_mock_stage_result(num_delete=current_actors)
            solution = _make_mock_solution([stage_result])
            pool = _make_mock_pool(num_ready_actors=current_actors)
            _inject_solution(autoscaler, solution)

            autoscaler.apply_autoscale_result_if_ready([pool])

            deleted = pool.add_actor_to_delete.call_count
            current_actors -= deleted
            trajectory.append(current_actors)

        assert trajectory == [20, 10, 5, 3, 2, 1]

    def test_zero_ready_actors_skips_clamping(self) -> None:
        """When num_ready_actors=0, all deletions pass through (target pending actors)."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        stage_result = _make_mock_stage_result(num_delete=5)
        solution = _make_mock_solution([stage_result])
        pool = _make_mock_pool(num_ready_actors=0)
        _inject_solution(autoscaler, solution)

        autoscaler.apply_autoscale_result_if_ready([pool])

        assert pool.add_actor_to_delete.call_count == 5

    def test_new_workers_still_created(self) -> None:
        """Rate-limit guard only affects deletions; creations are unaffected."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        stage_result = _make_mock_stage_result(num_new=3, num_delete=18)
        solution = _make_mock_solution([stage_result])
        pool = _make_mock_pool(num_ready_actors=20)
        _inject_solution(autoscaler, solution)

        autoscaler.apply_autoscale_result_if_ready([pool])

        assert pool.add_actor_to_create.call_count == 3
        assert pool.add_actor_to_delete.call_count == 10

    def test_no_future_does_nothing(self) -> None:
        """When no autoscale calculation is pending, nothing happens."""
        autoscaler = _make_autoscaler(max_scale_down_fraction=0.5)
        pool = _make_mock_pool()

        autoscaler.apply_autoscale_result_if_ready([pool])

        pool.add_actor_to_delete.assert_not_called()
        pool.add_actor_to_create.assert_not_called()


class TestConfigValidation:
    """StreamingSpecificSpec.autoscale_max_scale_down_fraction validators."""

    def test_rejects_zero(self) -> None:
        """fraction=0 violates gt(0) validator."""
        with pytest.raises(ValueError, match="greater than 0"):
            StreamingSpecificSpec(autoscale_max_scale_down_fraction=0)

    def test_rejects_negative(self) -> None:
        """fraction=-0.5 violates gt(0) validator."""
        with pytest.raises(ValueError, match="greater than 0"):
            StreamingSpecificSpec(autoscale_max_scale_down_fraction=-0.5)

    def test_rejects_above_one(self) -> None:
        """fraction=1.5 violates le(1.0) validator."""
        with pytest.raises(ValueError, match=r"less than or equal to 1\.0"):
            StreamingSpecificSpec(autoscale_max_scale_down_fraction=1.5)
