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

import cosmos_xenna.ray_utils.actor_pool as ap_module
from cosmos_xenna.pipelines.private import allocator as alloc_mod
from cosmos_xenna.pipelines.private import data_structures, resources


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

    The swap of ``actor_pool.AllocationError`` with the local test stand-in
    is performed by an autouse fixture on the enclosing test class
    (``_patch_allocation_error``), so the patch is scoped per-test and
    auto-reverted.

    Attributes set to satisfy _add_worker_group's non-SPMD path:
    _is_spmd, _allocator, _worker_groups, _create_actor_for_worker_group,
    name, _worker_groups_to_delete, _worker_groups_to_create.
    """
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

    The ``actor_pool.AllocationError`` swap is performed by an autouse
    fixture on the enclosing test class (``_patch_allocation_error``).

    ``_find_rendevous_params_for_worker_group`` and
    ``_create_actor_for_worker_group`` stay as auto-spec MagicMocks so the
    test exercises only the ordering logic in ``_add_worker_group``,
    not the rendezvous RPC or actor-spawn machinery.
    """
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

    @pytest.fixture(autouse=True)
    def _patch_allocation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-test patch of ``actor_pool.AllocationError`` to the local stand-in.

        Scoped via autouse so individual tests stay parameter-free; ``monkeypatch``
        reverts the swap at teardown so no global module state leaks across tests.
        """
        monkeypatch.setattr(ap_module, "AllocationError", _AllocationError)

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

    def test_non_spmd_post_allocator_failure_rolls_back_allocator_only(self) -> None:
        """Non-SPMD rollback path: ``clear_port`` MUST be skipped (``if self._is_spmd:`` guard).

        Coverage rationale: the asymmetric port-registry guard inside the
        rollback ``finally`` is the only thing keeping non-SPMD pools (which
        do not register any ports) from issuing a spurious ``clear_port``
        call into a registry that was never populated. A future refactor that
        drops the ``if self._is_spmd:`` check would silently break non-SPMD
        rollback - this test pins the guard.

        Implementation note: ``MagicMock(spec=ActorPool)`` does not
        auto-create ``_cluster_port_registry`` (it is set in ``__init__``,
        not declared at class level), so any unguarded access from the
        non-SPMD ``finally`` would raise ``AttributeError``. Attaching an
        explicit sentinel here lets the assertion read positively
        ("clear_port was never called") instead of relying on the
        absence of an AttributeError.
        """
        wg = _make_mock_worker_group("wg-nonspmd-rollback")
        pool = _make_pool_with_queue([wg])
        pool._cluster_port_registry = MagicMock()
        pool._create_actor_for_worker_group.side_effect = RuntimeError("simulated actor-spawn failure")

        with pytest.raises(RuntimeError, match="simulated actor-spawn failure"):
            pool._add_worker_group(wg)

        pool._allocator.remove_worker.assert_called_once_with("wg-nonspmd-rollback")
        pool._cluster_port_registry.clear_port.assert_not_called()
        assert pool._worker_groups == {}


class TestSpmdAllocationOrdering:
    """Ordering invariant for SPMD ``_add_worker_group``.

    The Rust allocator gate must run before the rendezvous-port lookup
    so that an ``AllocationError`` cannot leak a port entry into the
    ``ClusterPortRegistry`` (only cleared in ``_delete_worker_group``,
    which is unreachable for a worker that was never added).
    """

    @pytest.fixture(autouse=True)
    def _patch_allocation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-test patch of ``actor_pool.AllocationError`` to the local stand-in.

        Scoped via autouse so individual tests stay parameter-free; ``monkeypatch``
        reverts the swap at teardown so no global module state leaks across tests.
        """
        monkeypatch.setattr(ap_module, "AllocationError", _AllocationError)

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

    def test_spmd_actor_creation_failure_after_rendezvous_rolls_back_both(self) -> None:
        """A failure in the per-GPU SPMD loop (after rendezvous succeeded) still triggers full rollback.

        Coverage rationale: ``test_post_allocator_failure_rolls_back_allocator_and_port``
        injects the failure inside ``_find_rendevous_params_for_worker_group``,
        so the per-GPU actor-creation loop is never reached. This test
        injects the failure on the second per-GPU iteration to cover the
        case where rendezvous succeeded AND one actor was already spawned -
        the rollback contract still requires both allocator and port-registry
        cleanup, otherwise SPMD pools would leak both on partial failures.
        """
        wg = _make_spmd_worker_group("wg-spmd-actor-fail", num_gpus=2)
        pool = _make_spmd_pool([wg])

        def _rendezvous_with_port_registration(w: MagicMock) -> MagicMock:
            pool._cluster_port_registry.register_port(w.allocations[0].node, w.id, 29500)
            return MagicMock(master_port=29500)

        pool._find_rendevous_params_for_worker_group = MagicMock(side_effect=_rendezvous_with_port_registration)
        pool._create_actor_for_worker_group.side_effect = [None, RuntimeError("simulated GPU-2 spawn failure")]

        with pytest.raises(RuntimeError, match="simulated GPU-2 spawn failure"):
            pool._add_worker_group(wg)

        assert pool._create_actor_for_worker_group.call_count == 2
        pool._cluster_port_registry.register_port.assert_called_once()
        pool._allocator.remove_worker.assert_called_once_with("wg-spmd-actor-fail")
        pool._cluster_port_registry.clear_port.assert_called_once_with("node-A", "wg-spmd-actor-fail")
        assert pool._worker_groups == {}

    def test_allocator_rollback_failure_is_logged_and_does_not_mask_original(self) -> None:
        """If ``_allocator.remove_worker`` itself raises during rollback:

        - the original exception still propagates (no masking by the cleanup error)
        - ``_cluster_port_registry.clear_port`` is still attempted (rollback legs are independent)
        - the rollback failure is logged at ERROR with ``exc_info=True``
        """
        wg = _make_spmd_worker_group("wg-spmd-alloc-rb-fail")
        pool = _make_spmd_pool([wg])

        def _rendezvous_then_fail(w: MagicMock) -> MagicMock:
            pool._cluster_port_registry.register_port(w.allocations[0].node, w.id, 29500)
            raise RuntimeError("original failure")

        pool._find_rendevous_params_for_worker_group = MagicMock(side_effect=_rendezvous_then_fail)
        pool._allocator.remove_worker.side_effect = RuntimeError("rollback alloc broke")

        with patch("cosmos_xenna.ray_utils.actor_pool.logger") as mock_logger:
            with pytest.raises(RuntimeError, match="original failure"):
                pool._add_worker_group(wg)

            pool._cluster_port_registry.clear_port.assert_called_once_with("node-A", "wg-spmd-alloc-rb-fail")
            error_calls = [c for c in mock_logger.error.call_args_list if "Allocator rollback failed" in c[0][0]]
            assert len(error_calls) == 1
            assert error_calls[0].kwargs.get("exc_info") is True
        assert pool._worker_groups == {}

    def test_port_registry_rollback_failure_is_logged_and_does_not_mask_original(self) -> None:
        """If ``_cluster_port_registry.clear_port`` raises during rollback:

        - the original exception still propagates
        - ``_allocator.remove_worker`` already ran successfully (it is invoked first)
        - the port-registry rollback failure is logged at ERROR with ``exc_info=True``
        """
        wg = _make_spmd_worker_group("wg-spmd-port-rb-fail")
        pool = _make_spmd_pool([wg])

        def _rendezvous_then_fail(w: MagicMock) -> MagicMock:
            pool._cluster_port_registry.register_port(w.allocations[0].node, w.id, 29500)
            raise RuntimeError("original failure")

        pool._find_rendevous_params_for_worker_group = MagicMock(side_effect=_rendezvous_then_fail)
        pool._cluster_port_registry.clear_port.side_effect = RuntimeError("rollback port broke")

        with patch("cosmos_xenna.ray_utils.actor_pool.logger") as mock_logger:
            with pytest.raises(RuntimeError, match="original failure"):
                pool._add_worker_group(wg)

            pool._allocator.remove_worker.assert_called_once_with("wg-spmd-port-rb-fail")
            error_calls = [c for c in mock_logger.error.call_args_list if "Port registry rollback failed" in c[0][0]]
            assert len(error_calls) == 1
            assert error_calls[0].kwargs.get("exc_info") is True
        assert pool._worker_groups == {}


class TestRustAllocationErrorContract:
    """Validate the design contract for Rust-defined exceptions.

    The pure-Python ``_AllocationError(ValueError)`` stand-in used by the
    other test classes in this module is faithful only if the real PyO3
    exceptions actually inherit from ``ValueError`` and are catchable by
    type. This class exercises the compiled extension end-to-end to lock
    that contract in CI: a regression that demoted the Rust base class
    away from ``ValueError`` would silently break the
    ``except AllocationError`` guard in ``actor_pool._add_worker_group``
    even though every mock-based test would still pass.
    """

    @staticmethod
    def _make_single_node_cluster(num_cpus: int) -> resources.ClusterResources:
        """Build a minimal CPU-only single-node cluster for allocator tests."""
        return resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(
                    used_cpus=0,
                    total_cpus=num_cpus,
                    gpus=[],
                    name="node-0",
                ),
            },
        )

    @staticmethod
    def _make_worker(worker_id: str, cpus: float) -> resources.WorkerGroup:
        """Build a CPU-only WorkerGroup that requests ``cpus`` from ``node-0``.

        Constructed via ``ProblemWorkerGroupState.make().to_worker_group()``
        because the Rust extension does not expose a direct ``WorkerGroup``
        constructor; this mirrors the production construction path used
        by the autoscaler in ``streaming.py``.
        """
        state = data_structures.ProblemWorkerGroupState.make(
            worker_id,
            [resources.WorkerResourcesInternal(node="node-0", cpus=cpus, gpus=[])],
        )
        return state.to_worker_group("TestStage")

    def test_real_allocation_error_inherits_from_value_error(self) -> None:
        """Over-allocating the real Rust allocator must raise an AllocationError that ``isinstance``s as ValueError."""
        real_allocator = alloc_mod.WorkerAllocator.make(self._make_single_node_cluster(num_cpus=2))
        too_big = self._make_worker("wg-too-big", cpus=4.0)

        with pytest.raises(alloc_mod.AllocationError) as excinfo:
            real_allocator.add_worker(too_big)

        assert isinstance(excinfo.value, ValueError)

    def test_real_duplicate_worker_id_error_inherits_from_value_error(self) -> None:
        """Re-adding the same worker id must raise a DuplicateWorkerIdError that ``isinstance``s as ValueError."""
        real_allocator = alloc_mod.WorkerAllocator.make(self._make_single_node_cluster(num_cpus=8))
        real_allocator.add_worker(self._make_worker("wg-dup", cpus=1.0))

        with pytest.raises(alloc_mod.DuplicateWorkerIdError) as excinfo:
            real_allocator.add_worker(self._make_worker("wg-dup", cpus=1.0))

        assert isinstance(excinfo.value, ValueError)
