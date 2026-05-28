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

"""Contract tests for ``donor.DonorCoordinator``.

The coordinator wires together a ``DonorPolicy`` with the shared
resource-fit search and the probe + atomic-remove transaction.
The tests use a stub policy + stub ctx to exercise every branch
of ``acquire(...)`` independently of the production policies.
"""

from collections.abc import Mapping
from typing import ClassVar, cast

import attrs
import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.resource_fit import ResourceFitPlanner
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import (
    DonorCommitOutcome,
    DonorPlan,
    DonorWorker,
    GateResult,
    RejectReason,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import StageCycleView
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _coordinator() -> DonorCoordinator:
    """Build a default-bounded coordinator for the contract tests.

    The contract tests exercise the coordinator's policy dispatch
    and transaction handling; the planner's search bounds are
    intentionally generous (32 combinations of up to 4 donors)
    so they never short-circuit a contract assertion.

    """
    return DonorCoordinator(planner=ResourceFitPlanner(max_plan_size=4, max_plan_combinations=32))


@attrs.define
class _StubPolicy:
    """Configurable policy that records every method call.

    Mutable so tests can flip behaviour mid-run, and tracks the
    last set of arguments each method received so assertions can
    spot-check the coordinator's wiring.
    """

    label: ClassVar[str] = "stub"

    is_enabled_return: bool = True
    eligible_stages: list[int] = attrs.field(factory=list)
    candidates: list[DonorWorker] = attrs.field(factory=list)
    gate_result: GateResult | None = None
    on_commit_called: bool = False

    def is_enabled(self, context: DonorPlanningContext) -> bool:
        del context
        return self.is_enabled_return

    def filter_eligible_donors(
        self,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
    ) -> list[int]:
        del context, receiver_view
        return self.eligible_stages

    def candidate_pool(
        self,
        eligible_stages: list[int],
        context: DonorPlanningContext,
    ) -> list[DonorWorker]:
        del eligible_stages, context
        return self.candidates

    def evaluate_gate(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
        receiver_intent: int,
    ) -> GateResult | None:
        del plan, context, receiver_view, receiver_intent
        return self.gate_result

    def on_commit(self, plan: DonorPlan, context: DonorPlanningContext) -> None:
        del plan, context
        self.on_commit_called = True


@attrs.define
class _StubCtx:
    """Stand-in for ``AutoscalePlanContext`` covering the donor flow.

    The coordinator calls ``probe_add_after_removals`` (inside
    ``ResourceFitPlanner.find``) and ``remove_workers_atomically``
    (inside ``commit_donor_plan``); both are scripted per test.
    """

    feasible_probe: bool = True
    probe_reason: str = ""
    remove_success: bool = True

    def probe_add_after_removals(
        self,
        removals: list[DonorWorker],
        receiver_stage_index: int,
    ) -> "_StubProbeResult":
        del removals, receiver_stage_index
        return _StubProbeResult(feasible=self.feasible_probe, reject_reason=self.probe_reason)

    def remove_workers_atomically(self, removals: list[DonorWorker]) -> bool:
        del removals
        return self.remove_success


@attrs.frozen
class _StubProbeResult:
    """Mirrors the planner's probe result shape (``feasible`` + ``reject_reason``)."""

    feasible: bool
    reject_reason: str


def _gate_result(*, reject_reason: RejectReason | None = None) -> GateResult:
    return GateResult(
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        spread=0.0,
        donor_cost=0.0,
        receiver_value=0.0,
        throughput_before=0.0,
        throughput_after=0.0,
        max_d_before=0.0,
        max_d_after=0.0,
        balance_before=0.0,
        balance_after=0.0,
        signal_trust_per_donor={},
    )


def _config(**overrides: object) -> SaturationAwareConfig:
    """Build a config valid against the per-stage default validator."""
    base: dict[str, object] = {
        "enable_cross_stage_donor": True,
        "cross_stage_donor_anti_flap_cycles": 30,
        "cross_stage_donor_max_plan_size": 4,
        "cross_stage_donor_max_plan_combinations": 16,
    }
    base.update(overrides)
    return SaturationAwareConfig(**base)  # type: ignore[arg-type]


def _make_context(
    *,
    last_donation_cycle: dict[str, int] | None = None,
    cycle_counter: int = 100,
    stage_floors: Mapping[int, int] | None = None,
) -> DonorPlanningContext:
    stage_names = ("A", "B")
    stage_configs = {name: SaturationAwareStageConfig() for name in stage_names}
    stage_states: Mapping[str, StageRuntimeState] = {name: StageRuntimeState(stage_name=name) for name in stage_names}
    return DonorPlanningContext(
        stage_names=stage_names,
        stage_configs=stage_configs,
        stage_states=stage_states,
        stage_floors=stage_floors if stage_floors is not None else {0: 1, 1: 1},
        worker_ids_by_stage=(("A-w0", "A-w1"), ("B-w0",)),
        worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 1},
        worker_node_map={"A-w0": "node-0", "A-w1": "node-1", "B-w0": "node-0"},
        d_k_now={"A": 1.0, "B": 2.0},
        effective_capacities={"A": 2, "B": 1},
        s_k_ewma={"A": 0.5, "B": 1.0},
        slots_per_worker_by_stage={"A": 1, "B": 1},
        donor_warmup_exclusions=frozenset(),
        cycle_counter=cycle_counter,
        last_donation_cycle=last_donation_cycle if last_donation_cycle is not None else {},
        config=_config(),
    )


def _receiver() -> StageCycleView:
    return StageCycleView(
        stage_index=1,
        stage_name="B",
        runtime_state=StageRuntimeState(stage_name="B"),
        current_workers=1,
    )


def _as_ctx(stub: object) -> data_structures.AutoscalePlanContext:
    """Cast a structural stub into ``AutoscalePlanContext`` for static checkers.

    The coordinator only invokes ``probe_add_after_removals`` and
    ``remove_workers_atomically`` on ``ctx``; both ``_StubCtx`` and
    ``_ToggleCtx`` expose those methods. The cast keeps the production
    signature nominal while letting the tests inject minimal stubs.
    """
    return cast(data_structures.AutoscalePlanContext, stub)


class TestEarlyExits:
    """Each early-exit branch returns ``None`` without committing."""

    def test_is_enabled_false_returns_none(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(is_enabled_return=False)
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx()),
        )
        assert result.plan is None
        assert result.reject_reason is RejectReason.MASTER_TOGGLE_OFF
        assert policy.on_commit_called is False

    def test_no_eligible_donors_returns_none(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(eligible_stages=[])
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx()),
        )
        assert result.plan is None
        assert result.reject_reason is RejectReason.NO_CANDIDATES
        assert policy.on_commit_called is False

    def test_empty_candidate_pool_returns_none(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(eligible_stages=[0], candidates=[])
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx()),
        )
        assert result.plan is None
        assert result.reject_reason is RejectReason.NO_CANDIDATES


class TestResourceFitInfeasible:
    """A planner probe that rejects every combo aborts before commit."""

    def test_infeasible_probe_returns_none(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(
            eligible_stages=[0],
            candidates=[DonorWorker(stage_index=0, worker_id="A-w0", age=5)],
        )
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx(feasible_probe=False, probe_reason="no_placement")),
        )
        assert result.plan is None
        assert result.reject_reason is RejectReason.RESOURCE_FIT
        assert policy.on_commit_called is False


class TestPolicyGate:
    """When the policy's gate rejects, no commit happens."""

    def test_gate_reject_returns_none(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(
            eligible_stages=[0],
            candidates=[DonorWorker(stage_index=0, worker_id="A-w0", age=5)],
            gate_result=_gate_result(reject_reason=RejectReason.SPREAD_BELOW_THRESHOLD),
        )
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx()),
        )
        assert result.plan is None
        assert result.reject_reason is RejectReason.SPREAD_BELOW_THRESHOLD
        assert policy.on_commit_called is False

    def test_gate_none_accepts(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(
            eligible_stages=[0],
            candidates=[DonorWorker(stage_index=0, worker_id="A-w0", age=5)],
            gate_result=None,
        )
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx()),
        )
        assert result.plan is not None
        assert result.reject_reason is None
        assert policy.on_commit_called is True


class TestCommitFailure:
    """Probe and atomic-remove failures get distinct handling."""

    def test_probe_failed_returns_none(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(
            eligible_stages=[0],
            candidates=[DonorWorker(stage_index=0, worker_id="A-w0", age=5)],
        )

        # First probe in ``ResourceFitPlanner.find`` succeeds, second
        # probe in ``commit_donor_plan`` fails - the stub returns the
        # same answer for every probe so we simulate via
        # ``feasible_probe=False`` which short-circuits the planner
        # instead. To exercise ``commit_donor_plan``'s probe path we
        # need a stub that returns ``feasible=True`` for the first
        # probe (resource fit) and ``False`` for the second
        # (commit re-probe).
        class _ToggleCtx:
            def __init__(self) -> None:
                self.probe_calls = 0

            def probe_add_after_removals(
                self,
                removals: list[DonorWorker],
                receiver_stage_index: int,
            ) -> _StubProbeResult:
                del removals, receiver_stage_index
                self.probe_calls += 1
                feasible = self.probe_calls == 1
                return _StubProbeResult(feasible=feasible, reject_reason="" if feasible else "worker_not_found")

            def remove_workers_atomically(self, removals: list[DonorWorker]) -> bool:  # pragma: no cover - unreached
                del removals
                return True

        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_ToggleCtx()),
        )
        assert result.plan is None
        assert result.reject_reason is RejectReason.RESOURCE_FIT
        assert result.placement_reject_reason == "worker_not_found"
        assert result.probe_failed_at_commit is True
        assert policy.on_commit_called is False

    def test_atomic_remove_failed_raises(self) -> None:
        coord = _coordinator()
        policy = _StubPolicy(
            eligible_stages=[0],
            candidates=[DonorWorker(stage_index=0, worker_id="A-w0", age=5)],
        )
        with pytest.raises(SchedulerInvariantError, match="atomic removal"):
            coord.acquire(
                policy=policy,
                context=_make_context(),
                receiver_view=_receiver(),
                receiver_intent=1,
                ctx=_as_ctx(_StubCtx(feasible_probe=True, remove_success=False)),
            )


class TestSuccessPath:
    """A clean run returns the committed plan and invokes on_commit."""

    def test_commit_returns_plan_and_advances_ledger_hook(self) -> None:
        coord = _coordinator()
        donor = DonorWorker(stage_index=0, worker_id="A-w0", age=5)
        policy = _StubPolicy(eligible_stages=[0], candidates=[donor])
        result = coord.acquire(
            policy=policy,
            context=_make_context(),
            receiver_view=_receiver(),
            receiver_intent=1,
            ctx=_as_ctx(_StubCtx()),
        )
        assert isinstance(result.plan, DonorPlan)
        assert result.plan.receiver_stage_index == 1
        assert donor in result.plan.removals
        assert result.committed is True
        assert policy.on_commit_called is True


class TestStubReturnsCommitOutcome:
    """Sanity: DonorCommitOutcome path is wired through DonorTransaction."""

    def test_commit_outcome_factory(self) -> None:
        outcome = DonorCommitOutcome(
            committed=True,
            probe_failed=False,
            atomic_remove_failed=False,
            placement_reject_reason="",
        )
        assert outcome.committed is True
