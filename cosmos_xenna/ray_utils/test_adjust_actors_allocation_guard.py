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

"""Tests for the allocation-failure guard in ActorPool._adjust_actors().

Verifies that:
- Rust WorkerAllocator ``ValueError``s with "Allocation error" are caught,
  logged, and skipped (pipeline continues).
- Non-allocation ``ValueError``s are re-raised (fail-fast preserved).
- Successful allocations in the same batch are not affected by a later failure.
"""

import collections
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_worker_group(worker_id: str = "wg-1") -> MagicMock:
    """Create a minimal mock of ``resources.WorkerGroup`` with an ``id`` property."""
    wg = MagicMock()
    wg.id = worker_id
    return wg


def _make_pool_with_queue(worker_groups: list[MagicMock]) -> MagicMock:
    """Build a lightweight ``ActorPool`` mock that delegates to the real ``_adjust_actors``.

    We patch the pool's internal deques and override ``_add_worker_group``
    per-test to control which allocations succeed/fail, while calling the
    real ``_adjust_actors`` implementation.
    """
    from cosmos_xenna.ray_utils.actor_pool import ActorPool

    pool = MagicMock(spec=ActorPool)
    pool.name = "TestStage"
    pool._worker_groups_to_delete = collections.deque()
    pool._worker_groups_to_create = collections.deque(worker_groups)
    pool._adjust_actors = ActorPool._adjust_actors.__get__(pool)
    return pool


class TestAllocationGuard:
    """Tests for the try/except ValueError guard in _adjust_actors."""

    def test_allocation_error_is_caught_and_logged(self) -> None:
        """An 'Allocation error' ValueError from _add_worker_group is caught."""
        wg = _make_mock_worker_group("wg-fail")
        pool = _make_pool_with_queue([wg])
        pool._add_worker_group.side_effect = ValueError(
            "Allocation error: Not enough resources on node abc. "
            "Requested: PoolOfResources { cpus: 4.0 }, available: PoolOfResources { cpus: 1.0 }"
        )

        pool._adjust_actors()

        pool._add_worker_group.assert_called_once_with(wg)
        assert len(pool._worker_groups_to_create) == 0

    def test_non_allocation_valueerror_is_reraised(self) -> None:
        """A ValueError that is NOT an allocation error must propagate."""
        wg = _make_mock_worker_group("wg-bad")
        pool = _make_pool_with_queue([wg])
        pool._add_worker_group.side_effect = ValueError("Invalid config: something else")

        with pytest.raises(ValueError, match="Invalid config"):
            pool._adjust_actors()

    def test_successful_allocations_not_affected_by_later_failure(self) -> None:
        """If the first allocation succeeds but the second fails, the first is kept."""
        wg_ok = _make_mock_worker_group("wg-ok")
        wg_fail = _make_mock_worker_group("wg-fail")
        pool = _make_pool_with_queue([wg_ok, wg_fail])

        call_count = 0

        def _side_effect(wg: MagicMock) -> None:
            nonlocal call_count
            call_count += 1
            if wg.id == "wg-ok":
                return
            raise ValueError("Allocation error: Not enough resources on node xyz")

        pool._add_worker_group.side_effect = _side_effect

        pool._adjust_actors()

        assert call_count == 2
        assert len(pool._worker_groups_to_create) == 0

    def test_all_allocations_succeed(self) -> None:
        """When all allocations succeed, no exception handling is triggered."""
        wg1 = _make_mock_worker_group("wg-1")
        wg2 = _make_mock_worker_group("wg-2")
        pool = _make_pool_with_queue([wg1, wg2])

        pool._adjust_actors()

        assert pool._add_worker_group.call_count == 2
        assert len(pool._worker_groups_to_create) == 0

    def test_empty_create_queue_is_noop(self) -> None:
        """No errors when there are no workers to create."""
        pool = _make_pool_with_queue([])

        pool._adjust_actors()

        pool._add_worker_group.assert_not_called()

    def test_warning_includes_stage_name_and_worker_id(self) -> None:
        """The warning log message contains the stage name and worker ID."""
        wg = _make_mock_worker_group("wg-abc")
        pool = _make_pool_with_queue([wg])
        pool.name = "MyTestStage"
        pool._add_worker_group.side_effect = ValueError(
            "Allocation error: Not enough resources on node xyz. "
            "Requested: PoolOfResources { cpus: 8.0 }, available: PoolOfResources { cpus: 2.0 }"
        )

        with patch("cosmos_xenna.ray_utils.actor_pool.logger") as mock_logger:
            pool._adjust_actors()
            mock_logger.warning.assert_called_once()
            log_msg = mock_logger.warning.call_args[0][0]
            assert "MyTestStage" in log_msg
            assert "wg-abc" in log_msg
            assert "Allocation error" in log_msg
