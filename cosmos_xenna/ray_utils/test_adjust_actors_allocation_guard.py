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

"""Tests for the allocation-failure guard in ActorPool._add_worker_group().

The guard catches ``AllocationError`` (a ``ValueError`` subclass defined
in Rust via PyO3) from the ``WorkerAllocator`` when a TOCTOU race makes
a resource snapshot stale.  Caught errors are logged and skipped so the
pipeline continues; non-allocation exceptions propagate normally.
"""

import collections
from unittest.mock import MagicMock, patch

import pytest


class _AllocationError(ValueError):
    """Test stand-in for the Rust-defined AllocationError.

    The real class is created by PyO3 ``create_exception!`` and inherits
    from ``ValueError``.  This mirror lets tests run without the compiled
    Rust extension while preserving the same MRO that production code relies on.
    """


def _make_mock_worker_group(worker_id: str = "wg-1") -> MagicMock:
    """Create a minimal mock of ``resources.WorkerGroup``.

    Sets ``id`` and ``allocations`` (single element for non-SPMD path).
    """
    wg = MagicMock()
    wg.id = worker_id
    wg.allocations = [MagicMock()]
    return wg


def _make_pool_with_queue(worker_groups: list[MagicMock]) -> MagicMock:
    """Build a lightweight ``ActorPool`` mock that exercises the real guard logic.

    Binds ``_adjust_actors`` and ``_add_worker_group`` from the real
    ``ActorPool`` so the allocation guard in ``_add_worker_group`` is
    exercised end-to-end.  The Rust allocator (``_allocator.add_worker``)
    remains a mock so tests can control success/failure.

    Patches ``actor_pool.AllocationError`` with the local test stand-in
    so the ``except AllocationError`` clause matches without the Rust extension.

    Attributes set to satisfy _add_worker_group's non-SPMD path:
    _is_spmd, _allocator, _worker_groups, _create_actor_for_worker_group,
    name, _worker_groups_to_delete, _worker_groups_to_create.
    """
    import cosmos_xenna.ray_utils.actor_pool as ap_module

    ap_module.AllocationError = _AllocationError  # type: ignore[misc]

    pool = MagicMock(spec=ap_module.ActorPool)
    pool.name = "TestStage"
    pool._is_spmd = False
    pool._allocator = MagicMock()
    pool._worker_groups = {}
    pool._worker_groups_to_delete = collections.deque()
    pool._worker_groups_to_create = collections.deque(worker_groups)
    pool._adjust_actors = ap_module.ActorPool._adjust_actors.__get__(pool)
    pool._add_worker_group = ap_module.ActorPool._add_worker_group.__get__(pool)
    return pool


class TestAllocationGuard:
    """Tests for the AllocationError guard in _add_worker_group."""

    def test_allocation_error_is_caught_and_skipped(self) -> None:
        """An AllocationError from the Rust allocator is caught and skipped."""
        wg = _make_mock_worker_group("wg-fail")
        pool = _make_pool_with_queue([wg])
        pool._allocator.add_worker.side_effect = _AllocationError(
            "Allocation error: Not enough resources on node abc. "
            "Requested: PoolOfResources { cpus: 4.0 }, available: PoolOfResources { cpus: 1.0 }"
        )

        pool._adjust_actors()

        pool._allocator.add_worker.assert_called_once_with(wg)
        assert len(pool._worker_groups_to_create) == 0
        pool._create_actor_for_worker_group.assert_not_called()

    def test_plain_valueerror_is_not_caught(self) -> None:
        """A plain ValueError (not AllocationError) must propagate."""
        wg = _make_mock_worker_group("wg-bad")
        pool = _make_pool_with_queue([wg])
        pool._allocator.add_worker.side_effect = ValueError("Invalid config: something else")

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
            raise _AllocationError("Allocation error: Not enough resources on node xyz")

        pool._allocator.add_worker.side_effect = _side_effect

        pool._adjust_actors()

        assert call_count == 2
        assert len(pool._worker_groups_to_create) == 0
        assert pool._create_actor_for_worker_group.call_count == 1

    def test_all_allocations_succeed(self) -> None:
        """When all allocations succeed, no exception handling is triggered."""
        wg1 = _make_mock_worker_group("wg-1")
        wg2 = _make_mock_worker_group("wg-2")
        pool = _make_pool_with_queue([wg1, wg2])

        pool._adjust_actors()

        assert pool._allocator.add_worker.call_count == 2
        assert pool._create_actor_for_worker_group.call_count == 2
        assert len(pool._worker_groups_to_create) == 0

    def test_empty_create_queue_is_noop(self) -> None:
        """No errors when there are no workers to create."""
        pool = _make_pool_with_queue([])

        pool._adjust_actors()

        pool._allocator.add_worker.assert_not_called()

    def test_non_valueerror_is_not_caught(self) -> None:
        """Non-ValueError exceptions propagate unmodified."""
        wg = _make_mock_worker_group("wg-crash")
        pool = _make_pool_with_queue([wg])
        pool._allocator.add_worker.side_effect = RuntimeError("Something broke")

        with pytest.raises(RuntimeError, match="Something broke"):
            pool._adjust_actors()

    def test_warning_includes_stage_name_and_worker_id(self) -> None:
        """The warning log message contains the stage name and worker ID."""
        wg = _make_mock_worker_group("wg-abc")
        pool = _make_pool_with_queue([wg])
        pool.name = "MyTestStage"
        pool._allocator.add_worker.side_effect = _AllocationError(
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

    def test_fragmentation_message_is_forwarded_in_warning(self) -> None:
        """Fragmentation diagnostics from the Rust layer propagate verbatim into the WARNING log.

        The enriched ``AllocationError::NotEnoughResources`` Display format
        includes the per-GPU free-fraction vector and the
        ``fragmentation_suspected`` classifier. The Python guard forwards
        ``str(e)`` unchanged into the ``logger.warning`` call, so operators
        can distinguish the "4 half-used GPUs" fragmentation pattern from
        plain CPU/GPU under-provisioning without any extra Python plumbing.
        This test asserts that forwarding behaviour with a mock side-effect
        that mirrors the real Rust ``Display`` string; it does not require
        the compiled Rust extension to be loaded.
        """
        wg = _make_mock_worker_group("wg-frag")
        pool = _make_pool_with_queue([wg])
        pool.name = "FragStage"
        pool._allocator.add_worker.side_effect = _AllocationError(
            "Allocation error: Not enough resources on node node-42. "
            "Requested: PoolOfResources { cpus: 1.0, gpus: 1.0 }, "
            "available: PoolOfResources { cpus: 1.0, gpus: 2.0 }, "
            "per_gpu_free: [0.5, 0.5, 0.5, 0.5], "
            "fragmentation_suspected: true"
        )

        with patch("cosmos_xenna.ray_utils.actor_pool.logger") as mock_logger:
            pool._adjust_actors()

            mock_logger.warning.assert_called_once()
            log_msg = mock_logger.warning.call_args[0][0]
            assert "per_gpu_free: [0.5, 0.5, 0.5, 0.5]" in log_msg
            assert "fragmentation_suspected: true" in log_msg
            assert "FragStage" in log_msg
            assert "wg-frag" in log_msg
