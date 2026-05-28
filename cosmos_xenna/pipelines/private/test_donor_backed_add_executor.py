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

"""Contract tests for :class:`DonorBackedAddExecutor`.

Pin the typed-outcome contract of the shared receiver-add
transaction: direct add success, donor commit success, placement
exhaustion, commit-time probe rejection, allocation-error
absorption, and the post-donor retry divergence invariant. Each
test verifies one observable outcome through the public
``execute(...)`` API; stubs are injected via ``attrs.evolve`` per
``prefer-oop.mdc`` Section 38.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.donor.executor import (
    AllocationAborted,
    DirectAddSucceeded,
    DonorBackedAddExecutor,
    DonorCommitted,
    PlacementExhausted,
    ProbeFailedAtCommit,
)
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.policy import DonorPolicy
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import (
    DonorAcquireResult,
    DonorPlan,
    DonorWorker,
    RejectReason,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.allocation_failure_gate import AllocationFailureGate
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle, StageCycleView
from cosmos_xenna.pipelines.private.scheduling_py.state.sk_ewma_store import SkEwmaStore

_DEFENSE_PATH = "cosmos_xenna.pipelines.private.scheduling_py.donor.executor.try_add_worker_with_defense"


def _stub_cycle() -> AutoscaleCycle:
    """Sentinel cycle with the minimum attribute surface the executor reads."""
    cycle = MagicMock(spec=AutoscaleCycle)
    cycle.ctx = MagicMock()
    cycle.cycle_counter = 1
    return cast(AutoscaleCycle, cycle)


def _stub_worker_group(worker_id: str = "w1") -> data_structures.ProblemWorkerGroupState:
    """Sentinel ``ProblemWorkerGroupState`` for direct-add / post-donor returns."""
    wg = MagicMock(spec=data_structures.ProblemWorkerGroupState)
    wg.id = worker_id
    return cast(data_structures.ProblemWorkerGroupState, wg)


def _stub_planning_context() -> DonorPlanningContext:
    """Sentinel ``DonorPlanningContext`` used to bypass the executor's internal rebuild."""
    return cast(DonorPlanningContext, MagicMock(spec=DonorPlanningContext))


def _stub_executor(
    *,
    coordinator: DonorCoordinator | None = None,
    gate: AllocationFailureGate | None = None,
) -> DonorBackedAddExecutor:
    """Build an executor with mock collaborators for direct-outcome tests."""
    return DonorBackedAddExecutor(
        coordinator=coordinator or cast(DonorCoordinator, MagicMock(spec=DonorCoordinator)),
        policy=cast(DonorPolicy, MagicMock(spec=DonorPolicy)),
        pipeline=cast(PipelineModel, MagicMock(spec=PipelineModel)),
        allocation_gate=gate or AllocationFailureGate(),
        stage_states={},
        last_donation_cycle={},
        s_k_ewma=SkEwmaStore(),
        planning_mode="saturation",
    )


def _execute_args(
    *,
    cycle: AutoscaleCycle | None = None,
    planning_context: DonorPlanningContext | None = None,
) -> dict[str, Any]:
    """Keyword args common to every ``execute(...)`` call in this file.

    Values are intentionally heterogeneous test fixtures; ``Any`` lets the
    caller forward the bundle via ``executor.execute(**args)`` without
    fighting mypy on the per-key value type.
    """
    return {
        "cycle": cycle or _stub_cycle(),
        "stage_index": 0,
        "stage_name": "rx",
        "receiver_view": cast(StageCycleView, MagicMock(spec=StageCycleView)),
        "receiver_intent": 1,
        "stage_floors": {},
        "pipeline_name": "p",
        "skip_cycle_on_allocation_error": True,
        "planning_context": planning_context,
    }


class TestDirectAddSucceeded:
    """Direct ``try_add_worker_with_defense`` returns a worker; coordinator never runs."""

    def test_direct_add_short_circuits_before_coordinator(self) -> None:
        """A successful direct add returns :class:`DirectAddSucceeded` without consulting the coordinator."""
        wg = _stub_worker_group()
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        executor = _stub_executor(coordinator=coord)
        with patch(_DEFENSE_PATH, return_value=wg) as defense:
            outcome = executor.execute(**_execute_args())
        assert outcome == DirectAddSucceeded(worker_group=wg)
        defense.assert_called_once()
        cast(MagicMock, coord.acquire).assert_not_called()


class TestAllocationAborted:
    """Allocation gate absorbs an ``AllocationError`` before the coordinator runs."""

    def test_direct_returns_none_with_aborted_gate(self) -> None:
        """Direct add ``None`` + ``gate.aborted_cycle=True`` surfaces :class:`AllocationAborted`."""
        gate = AllocationFailureGate()
        gate.aborted_cycle = True
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        executor = _stub_executor(coordinator=coord, gate=gate)
        with patch(_DEFENSE_PATH, return_value=None):
            outcome = executor.execute(**_execute_args())
        assert isinstance(outcome, AllocationAborted)
        cast(MagicMock, coord.acquire).assert_not_called()


class TestPlacementExhausted:
    """Coordinator returns ``committed=False`` without a commit-time probe failure."""

    def test_no_candidates_surfaces_placement_exhausted(self) -> None:
        """``NO_CANDIDATES`` reject surfaces :class:`PlacementExhausted` with the acquire result."""
        acquire = DonorAcquireResult(
            plan=None,
            attempted_plan=None,
            reject_reason=RejectReason.NO_CANDIDATES,
            placement_reject_reason="",
            gate_result=None,
        )
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        cast(MagicMock, coord.acquire).return_value = acquire
        executor = _stub_executor(coordinator=coord)
        with patch(_DEFENSE_PATH, return_value=None):
            outcome = executor.execute(**_execute_args(planning_context=_stub_planning_context()))
        assert outcome == PlacementExhausted(acquire_result=acquire)


class TestProbeFailedAtCommit:
    """Coordinator returns ``RESOURCE_FIT`` with a populated planner reject reason."""

    def test_commit_time_probe_failure_surfaces_probe_failed_at_commit(self) -> None:
        """``RESOURCE_FIT`` + ``placement_reject_reason`` surfaces :class:`ProbeFailedAtCommit`."""
        acquire = DonorAcquireResult(
            plan=None,
            attempted_plan=DonorPlan(
                removals=(DonorWorker(stage_index=1, worker_id="d1", age=10),),
                receiver_stage_index=0,
            ),
            reject_reason=RejectReason.RESOURCE_FIT,
            placement_reject_reason="no_placement",
            gate_result=None,
        )
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        cast(MagicMock, coord.acquire).return_value = acquire
        executor = _stub_executor(coordinator=coord)
        with patch(_DEFENSE_PATH, return_value=None):
            outcome = executor.execute(**_execute_args(planning_context=_stub_planning_context()))
        assert outcome == ProbeFailedAtCommit(acquire_result=acquire)


class TestDonorCommitted:
    """Coordinator commits; post-donor receiver retry succeeds."""

    def test_donor_commit_returns_donor_committed_with_plan_and_worker_group(self) -> None:
        """Successful donor commit + retry surfaces :class:`DonorCommitted` with the plan and retry worker group."""
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=1, worker_id="d1", age=10),),
            receiver_stage_index=0,
        )
        acquire = DonorAcquireResult(
            plan=plan,
            attempted_plan=plan,
            reject_reason=None,
            placement_reject_reason="",
            gate_result=None,
        )
        wg = _stub_worker_group("retry-w")
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        cast(MagicMock, coord.acquire).return_value = acquire
        executor = _stub_executor(coordinator=coord)
        with patch(_DEFENSE_PATH, side_effect=[None, wg]):
            outcome = executor.execute(**_execute_args(planning_context=_stub_planning_context()))
        assert outcome == DonorCommitted(plan=plan, worker_group=wg)


class TestPostDonorRetryDivergence:
    """Post-donor retry returning ``None`` without an abort flag is a planner defect."""

    def test_retry_none_without_aborted_gate_raises_scheduler_invariant_error(self) -> None:
        """Atomic remove succeeded but the receiver retry returned ``None``: invariant violation."""
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=1, worker_id="d1", age=10),),
            receiver_stage_index=0,
        )
        acquire = DonorAcquireResult(
            plan=plan,
            attempted_plan=plan,
            reject_reason=None,
            placement_reject_reason="",
            gate_result=None,
        )
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        cast(MagicMock, coord.acquire).return_value = acquire
        executor = _stub_executor(coordinator=coord)
        with (
            patch(_DEFENSE_PATH, side_effect=[None, None]),
            pytest.raises(SchedulerInvariantError, match="planner snapshot diverged"),
        ):
            executor.execute(**_execute_args(planning_context=_stub_planning_context()))


class TestCoordinatorContractGuard:
    """``committed=True`` with ``plan=None`` is a coordinator contract violation."""

    def test_committed_true_with_none_plan_raises_scheduler_invariant_error(self) -> None:
        """``DonorAcquireResult.committed`` and ``plan is None`` disagree -> invariant violation."""
        wg = _stub_worker_group()
        # Build a result whose ``committed`` property returns True despite ``plan=None``.
        acquire = MagicMock(spec=DonorAcquireResult)
        acquire.committed = True
        acquire.probe_failed_at_commit = False
        acquire.plan = None
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        cast(MagicMock, coord.acquire).return_value = acquire
        executor = _stub_executor(coordinator=coord)
        with (
            patch(_DEFENSE_PATH, side_effect=[None, wg]),
            pytest.raises(SchedulerInvariantError, match="coordinator contract violation"),
        ):
            executor.execute(**_execute_args(planning_context=_stub_planning_context()))


class TestPrecomputedPlanningContextBypass:
    """Caller-supplied ``planning_context`` is forwarded verbatim to the coordinator."""

    def test_precomputed_context_is_forwarded_to_coordinator(self) -> None:
        """A non-``None`` ``planning_context`` is passed verbatim as ``coordinator.acquire(context=...)``."""
        ctx = _stub_planning_context()
        acquire = DonorAcquireResult(
            plan=None,
            attempted_plan=None,
            reject_reason=RejectReason.NO_CANDIDATES,
            placement_reject_reason="",
            gate_result=None,
        )
        coord = cast(DonorCoordinator, MagicMock(spec=DonorCoordinator))
        cast(MagicMock, coord.acquire).return_value = acquire
        executor = _stub_executor(coordinator=coord)
        with patch(_DEFENSE_PATH, return_value=None):
            executor.execute(**_execute_args(planning_context=ctx))
        assert cast(MagicMock, coord.acquire).call_args.kwargs["context"] is ctx


class TestResetCycle:
    """``reset_cycle`` clears the allocation-failure gate state."""

    def test_reset_cycle_clears_aborted_flag(self) -> None:
        """``reset_cycle`` invokes ``AllocationFailureGate.reset`` exactly once."""
        gate = AllocationFailureGate()
        gate.aborted_cycle = True
        executor = _stub_executor(gate=gate)
        executor.reset_cycle()
        assert gate.aborted_cycle is False
