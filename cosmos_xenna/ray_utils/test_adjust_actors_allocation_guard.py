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


def _make_spmd_pool(worker_groups: list[MagicMock]) -> MagicMock:
    """Build an ``ActorPool`` mock configured for the SPMD code path.

    Same pattern as ``_make_pool_with_queue`` but flips ``_is_spmd`` and
    wires a mock ``_cluster_port_registry`` so SPMD-specific side effects
    (port registration, rendezvous lookup) are observable in assertions.

    ``_find_rendevous_params_for_worker_group`` and
    ``_create_actor_for_worker_group`` stay as auto-spec MagicMocks so the
    test exercises only the ordering logic in ``_add_worker_group``,
    not the rendezvous RPC or actor-spawn machinery.
    """
    import cosmos_xenna.ray_utils.actor_pool as ap_module

    ap_module.AllocationError = _AllocationError  # type: ignore[misc]

    pool = MagicMock(spec=ap_module.ActorPool)
    pool.name = "SpmdStage"
    pool._is_spmd = True
    pool._allocator = MagicMock()
    pool._cluster_port_registry = MagicMock()
    pool._worker_groups = {}
    pool._worker_groups_to_delete = collections.deque()
    pool._worker_groups_to_create = collections.deque(worker_groups)
    pool._find_rendevous_params_for_worker_group = MagicMock(return_value=MagicMock(master_port=29500))
    pool._adjust_actors = ap_module.ActorPool._adjust_actors.__get__(pool)
    pool._add_worker_group = ap_module.ActorPool._add_worker_group.__get__(pool)
    return pool


def _make_spmd_worker_group(worker_id: str = "wg-spmd-1", num_gpus: int = 2) -> MagicMock:
    """Create a minimal SPMD ``WorkerGroup`` mock.

    ``split_allocation_per_gpu`` returns a list of ``num_gpus`` mock
    allocations so the SPMD per-GPU actor-creation loop runs that many
    iterations.
    """
    wg = MagicMock()
    wg.id = worker_id
    allocation = MagicMock()
    allocation.node = "node-A"
    wg.allocations = [allocation]
    splits = []
    for _ in range(num_gpus):
        split = MagicMock()
        split.node = "node-A"
        splits.append(split)
    wg.split_allocation_per_gpu = MagicMock(return_value=splits)
    return wg


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


class TestSpmdAllocationOrdering:
    """Ordering invariant for SPMD ``_add_worker_group``.

    The Rust allocator gate must run before the rendezvous-port lookup
    so that an ``AllocationError`` cannot leak a port entry into the
    ``ClusterPortRegistry`` (only cleared in ``_delete_worker_group``,
    which is unreachable for a worker that was never added).
    """

    def test_spmd_allocation_failure_does_not_register_port(self) -> None:
        """SPMD ``AllocationError`` must skip rendezvous-port registration entirely.

        The mocked rendezvous helper mirrors production by calling
        ``_cluster_port_registry.register_port`` so the assertion that
        ``register_port`` is never called is a real regression check
        (not a tautology that passes only because the rendezvous helper
        itself was mocked out).
        """
        wg = _make_spmd_worker_group("wg-spmd-fail")
        pool = _make_spmd_pool([wg])
        pool._allocator.add_worker.side_effect = _AllocationError("Allocation error: Not enough GPUs on node-A")

        def _rendezvous_with_port_registration(w: MagicMock) -> MagicMock:
            pool._cluster_port_registry.register_port(w.allocations[0].node, w.id, 29500)
            return MagicMock(master_port=29500)

        pool._find_rendevous_params_for_worker_group = MagicMock(side_effect=_rendezvous_with_port_registration)

        result = pool._add_worker_group(wg)

        assert result is False
        pool._find_rendevous_params_for_worker_group.assert_not_called()
        pool._cluster_port_registry.register_port.assert_not_called()
        assert pool._worker_groups == {}
        pool._create_actor_for_worker_group.assert_not_called()

    def test_spmd_allocation_success_invokes_rendezvous_and_creates_actors(self) -> None:
        """On allocator success, rendezvous lookup runs and SPMD per-GPU actors are created.

        Also verifies the rollback path is NOT triggered on the happy path
        (no spurious allocator removal or port-registry cleanup).
        """
        wg = _make_spmd_worker_group("wg-spmd-ok", num_gpus=2)
        pool = _make_spmd_pool([wg])

        result = pool._add_worker_group(wg)

        assert result is True
        pool._find_rendevous_params_for_worker_group.assert_called_once_with(wg)
        assert pool._create_actor_for_worker_group.call_count == 2
        assert "wg-spmd-ok" in pool._worker_groups
        pool._allocator.remove_worker.assert_not_called()
        pool._cluster_port_registry.clear_port.assert_not_called()

    def test_allocator_is_called_before_rendezvous_lookup(self) -> None:
        """Defensive ordering check: allocator gate runs strictly before rendezvous lookup."""
        wg = _make_spmd_worker_group("wg-spmd-order")
        pool = _make_spmd_pool([wg])

        parent = MagicMock()
        parent.attach_mock(pool._allocator.add_worker, "add_worker")
        parent.attach_mock(pool._find_rendevous_params_for_worker_group, "find_rendezvous")

        pool._add_worker_group(wg)

        method_call_order = [call[0] for call in parent.mock_calls]
        assert method_call_order.index("add_worker") < method_call_order.index("find_rendezvous")

    def test_post_allocator_failure_rolls_back_allocator_and_port(self) -> None:
        """If a post-allocator step raises, both allocator and registry are rolled back."""
        wg = _make_spmd_worker_group("wg-spmd-rollback")
        pool = _make_spmd_pool([wg])

        def _rendezvous_then_fail(w: MagicMock) -> MagicMock:
            pool._cluster_port_registry.register_port(w.allocations[0].node, w.id, 29500)
            raise RuntimeError("simulated ray.get RPC failure")

        pool._find_rendevous_params_for_worker_group = MagicMock(side_effect=_rendezvous_then_fail)

        with pytest.raises(RuntimeError, match="simulated ray.get RPC failure"):
            pool._add_worker_group(wg)

        pool._allocator.remove_worker.assert_called_once_with("wg-spmd-rollback")
        pool._cluster_port_registry.clear_port.assert_called_once_with("node-A", "wg-spmd-rollback")
        assert pool._worker_groups == {}
