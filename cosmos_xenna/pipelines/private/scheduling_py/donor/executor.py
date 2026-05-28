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

"""Donor-backed receiver-add transaction.

:class:`DonorBackedAddExecutor` runs the receiver-add transaction:
direct planner add, fallback to the donor coordinator, and post-donor
retry::

    +-------------+   +---------+   +-----------+
    | direct add  |-->| coord.  |-->| post-add  |
    +-------------+   | acquire |   |  retry    |
         |            +---------+   +-----------+
         v                |               |
    AllocationAborted     v               v
                    PlacementExh./  ProbeFailedAt
                    DonorCommitted   Commit

The executor returns a :data:`ReceiverAddOutcome` typed union; the
caller pattern-matches on the outcome and applies the
phase-specific bookkeeping it owns.

The ``planning_mode`` field selects between a floor configuration
(donor planning context built from empty saturation-only fields) and
a saturation configuration (context built from the per-cycle
``cycle.bottleneck`` snapshot). The executor owns the
``AllocationFailureGate`` and the cross-cycle donor planning inputs
so callers do not reach into cross-cycle ledger state.
"""

from typing import Literal

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.donor.context_factory import build_donor_planning_context
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.policy import DonorPolicy
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorAcquireResult, DonorPlan
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.allocation_failure_gate import (
    try_add_worker_with_defense,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.allocation_failure_gate import AllocationFailureGate
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle, StageCycleView
from cosmos_xenna.pipelines.private.scheduling_py.state.sk_ewma_store import SkEwmaStore
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageStateMap


@attrs.frozen
class DirectAddSucceeded:
    """Direct ``try_add_worker_with_defense`` placed one receiver worker.

    Attributes:
        worker_group: The receiver worker group snapshot returned by
            ``ctx.try_add_worker`` (stable worker id, resource
            allocations, and per-worker bookkeeping). Callers that
            only need to count the placement can ignore the
            snapshot; callers that need to inspect placement detail
            without re-issuing ``try_add_worker`` read it here.

    """

    worker_group: data_structures.ProblemWorkerGroupState


@attrs.frozen
class DonorCommitted:
    """Donor coordinator committed a plan and the post-donor retry succeeded.

    Attributes:
        plan: The committed donor plan (donations the planner
            atomically applied).
        worker_group: The receiver worker group snapshot returned by
            the post-donor ``try_add_worker_with_defense`` retry.
            Mirrors the field on :class:`DirectAddSucceeded` so the
            caller can dispatch uniformly across the direct-success
            and donor-success outcomes.

    """

    plan: DonorPlan
    worker_group: data_structures.ProblemWorkerGroupState


@attrs.frozen
class AllocationAborted:
    """``AllocationError`` absorbed by the allocation-failure gate.

    The caller MUST stop processing the current cycle when this
    outcome surfaces. ``AllocationFailureGate.aborted_cycle`` is
    already set; the cycle summary observes the asymmetric
    outcome via the gate's per-stage counter.

    """


@attrs.frozen
class PlacementExhausted:
    """Direct add failed and no donor plan was feasible.

    Carries the full :class:`DonorAcquireResult` so the caller
    can dispatch on the named reject reason (Floor treats a
    missing donor as a stuck-grace miss; Grow treats it as a
    recoverable WARN + stuck-plan counter tick).

    Attributes:
        acquire_result: The coordinator's verdict, including
            the named reject reason and any partial plan that
            failed selection-time gates.

    """

    acquire_result: DonorAcquireResult


@attrs.frozen
class ProbeFailedAtCommit:
    """Probe approved a plan at selection time but commit-time probe rejected it.

    The planner snapshot diverged between selection and commit
    (cluster mutated mid-cycle). Floor treats this as
    operator-actionable and raises ``RuntimeError`` with the
    planner-supplied textual reason; Grow treats it as a
    recoverable defensive event (one WARN, no raise).

    Attributes:
        acquire_result: The coordinator's verdict; carries the
            attempted plan and the planner's
            ``placement_reject_reason``.

    """

    acquire_result: DonorAcquireResult


type ReceiverAddOutcome = (
    DirectAddSucceeded | DonorCommitted | AllocationAborted | PlacementExhausted | ProbeFailedAtCommit
)


@attrs.frozen
class DonorBackedAddExecutor:
    """Direct-add + donor-fallback transaction owner.

    Stateless ``@attrs.frozen`` Strategy that runs the receiver-add
    spine. Phase-specific bookkeeping (floor stuck counters,
    stuck-plan ledger, DAG-priority ordering, partial-progress WARN
    text) lives at the caller and is dispatched via a ``match`` on
    the typed :data:`ReceiverAddOutcome` union.

    Attributes:
        coordinator: Select-probe-commit transaction coordinator.
        policy: Donor strategy (:class:`FloorPolicy` or
            :class:`SaturationPolicy`).
        pipeline: Immutable post-setup pipeline shape; read for
            ``stage_config(name)`` and ``config`` when building
            the planning context.
        allocation_gate: Cross-cycle allocation-failure gate
            reset by :meth:`reset_cycle` and consulted by every
            ``try_add_worker_with_defense`` call.
        stage_states: Cross-cycle per-stage runtime state map;
            forwarded into the planning context.
        last_donation_cycle: Cross-cycle anti-flap ledger;
            forwarded live (the saturation policy advances it on
            commit).
        s_k_ewma: Per-stage intrinsic service-time store;
            forwarded as a read-only view in saturation mode and
            ignored (empty) in floor mode.
        planning_mode: ``"floor"`` (skip ``cycle.bottleneck``
            access; pass empty saturation-only fields) or
            ``"saturation"`` (read ``cycle.bottleneck`` and the
            EWMA store).

    """

    coordinator: DonorCoordinator
    policy: DonorPolicy
    pipeline: PipelineModel
    allocation_gate: AllocationFailureGate
    stage_states: StageStateMap
    last_donation_cycle: dict[str, int]
    s_k_ewma: SkEwmaStore
    planning_mode: Literal["floor", "saturation"]

    def reset_cycle(self) -> None:
        """Reset the cross-cycle allocation-failure gate.

        Called once per cycle by ``FloorPhase.run`` and
        ``SaturationGrowPhase.run`` before the per-stage loop, so
        absorbed ``AllocationError`` flags from a prior cycle do
        not bleed into the current one.
        """
        self.allocation_gate.reset()

    def execute(
        self,
        *,
        cycle: AutoscaleCycle,
        stage_index: int,
        stage_name: str,
        receiver_view: StageCycleView,
        receiver_intent: int,
        stage_floors: dict[int, int],
        pipeline_name: str,
        skip_cycle_on_allocation_error: bool,
        planning_context: DonorPlanningContext | None = None,
    ) -> ReceiverAddOutcome:
        """Run one direct-add + donor-fallback transaction.

        Pass ``planning_context=None`` (the production policy) to let
        the donor branch lazily build the context against the current
        ``cycle.bottleneck`` / cross-cycle stores; planner mutations
        from prior same-cycle commits are reflected automatically.
        Direct-add success short-circuits before the build, so the
        cost is paid only when the donor fallback runs. Passing a
        precomputed ``DonorPlanningContext`` skips the rebuild and
        is safe only when planner state is guaranteed not to mutate
        between the build and this call.

        Returns a typed outcome describing what happened; the caller
        pattern-matches and applies its own phase-specific bookkeeping.

        Raises:
            SchedulerInvariantError: A donor plan was committed
                atomically but the post-commit receiver retry
                returned ``None`` (planner snapshot diverged
                after the atomic remove); or the coordinator
                returned ``committed=True`` with ``plan=None``
                (``DonorAcquireResult`` contract violation).

        """
        ctx = cycle.ctx
        direct = try_add_worker_with_defense(
            ctx=ctx,
            stage_index=stage_index,
            stage_name=stage_name,
            pipeline_name=pipeline_name,
            skip_cycle_on_allocation_error=skip_cycle_on_allocation_error,
            gate=self.allocation_gate,
        )
        if direct is not None:
            return DirectAddSucceeded(worker_group=direct)
        if self.allocation_gate.aborted_cycle:
            return AllocationAborted()
        context = (
            planning_context
            if planning_context is not None
            else self.build_planning_context(
                cycle=cycle,
                stage_floors=stage_floors,
            )
        )
        acquire_result = self.coordinator.acquire(
            policy=self.policy,
            context=context,
            receiver_view=receiver_view,
            receiver_intent=receiver_intent,
            ctx=ctx,
        )
        if not acquire_result.committed:
            if acquire_result.probe_failed_at_commit:
                return ProbeFailedAtCommit(acquire_result=acquire_result)
            return PlacementExhausted(acquire_result=acquire_result)
        # Post-donor receiver retry through the same defense wrapper:
        # an ``AllocationError`` is absorbed (the donor removals already
        # committed atomically); a ``None`` return after probe-approval
        # and atomic remove is the
        # ``DonorAcquireResult.committed -> plan is not None`` contract
        # violation and surfaces as ``SchedulerInvariantError``.
        retry = try_add_worker_with_defense(
            ctx=ctx,
            stage_index=stage_index,
            stage_name=stage_name,
            pipeline_name=pipeline_name,
            skip_cycle_on_allocation_error=skip_cycle_on_allocation_error,
            gate=self.allocation_gate,
        )
        if retry is None:
            if self.allocation_gate.aborted_cycle:
                return AllocationAborted()
            msg = (
                f"DonorBackedAddExecutor: post-commit retry returned None for "
                f"receiver {stage_name!r}; planner snapshot diverged after the "
                "atomic donor remove."
            )
            raise SchedulerInvariantError(msg)
        plan = acquire_result.plan
        if plan is None:
            # ``DonorAcquireResult.committed`` is defined as ``plan is not None``
            # in ``donor/types.py``; a ``committed=True`` paired with ``plan=None``
            # is a coordinator contract violation. Raise rather than rely on a
            # ``-O``-stripped ``assert``.
            msg = (
                "DonorBackedAddExecutor: DonorAcquireResult.committed=True "
                "with plan=None - coordinator contract violation."
            )
            raise SchedulerInvariantError(msg)
        return DonorCommitted(plan=plan, worker_group=retry)

    def build_planning_context(
        self,
        *,
        cycle: AutoscaleCycle,
        stage_floors: dict[int, int],
    ) -> DonorPlanningContext:
        """Materialize a ``DonorPlanningContext`` for the current cycle.

        Floor mode passes empty values for the saturation-only fields
        (``d_k_now``, ``effective_capacities``, ``s_k_ewma``,
        ``slots_per_worker_by_stage``, ``donor_warmup_exclusions``)
        because the floor donor flow does not consult them. Saturation
        mode reads them from the per-cycle ``cycle.bottleneck``
        snapshot and the cross-cycle EWMA store. Both branches
        delegate the per-cycle bundle construction to
        :func:`build_donor_planning_context`.
        """
        worker_ids_by_stage = tuple(tuple(workers) for workers in cycle.ctx.worker_ids_by_stage())
        stage_configs = {name: self.pipeline.stage_config(name) for name in self.pipeline.stage_names}
        if self.planning_mode == "floor":
            return build_donor_planning_context(
                problem_state=cycle.problem_state,
                worker_ids_by_stage=worker_ids_by_stage,
                stage_states=self.stage_states,
                stage_configs=stage_configs,
                stage_floors=stage_floors,
                worker_ages=cycle.ctx.worker_ages(),
                d_k_now={},
                effective_capacities={},
                s_k_ewma={},
                slots_per_worker_by_stage={},
                donor_warmup_exclusions=frozenset(),
                cycle_counter=cycle.cycle_counter,
                last_donation_cycle=self.last_donation_cycle,
                config=self.pipeline.config,
            )
        bottleneck = cycle.bottleneck
        return build_donor_planning_context(
            problem_state=cycle.problem_state,
            worker_ids_by_stage=worker_ids_by_stage,
            stage_states=self.stage_states,
            stage_configs=stage_configs,
            stage_floors=stage_floors,
            worker_ages=cycle.ctx.worker_ages(),
            d_k_now=bottleneck.d_k_now,
            effective_capacities=bottleneck.effective_capacities,
            s_k_ewma=self.s_k_ewma.view(),
            slots_per_worker_by_stage=bottleneck.channels_per_worker_group,
            donor_warmup_exclusions=cycle.donor_warmup_excluded_ids,
            cycle_counter=cycle.cycle_counter,
            last_donation_cycle=self.last_donation_cycle,
            config=self.pipeline.config,
        )


__all__ = (
    "AllocationAborted",
    "DirectAddSucceeded",
    "DonorBackedAddExecutor",
    "DonorCommitted",
    "PlacementExhausted",
    "ProbeFailedAtCommit",
    "ReceiverAddOutcome",
)
