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

"""Template Method coordinator for the donor select-probe-commit flow.

``DonorCoordinator`` owns the mechanics shared by both Phase B
floor enforcement and Phase C saturation grow: the resource-fit
search, the probe + atomic-remove transaction, the receiver-anti-
flap ledger advance hook. The five behavioural seams that differ
by mode live on the injected ``DonorPolicy``.

The class is ``@attrs.frozen`` and stateless after construction;
one instance per scheduler is reused across cycles.

See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

from typing import TYPE_CHECKING

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.policy import DonorPolicy
from cosmos_xenna.pipelines.private.scheduling_py.donor.resource_fit import ResourceFitPlanner
from cosmos_xenna.pipelines.private.scheduling_py.donor.transaction import DonorTransaction
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorAcquireResult, DonorPlan, RejectReason
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import StageCycleView
from cosmos_xenna.utils import python_log as logger

if TYPE_CHECKING:
    from cosmos_xenna.pipelines.private.data_structures import AutoscalePlanContext


@attrs.frozen
class DonorCoordinator:
    """Drives the donor select-probe-commit transaction under a policy.

    One instance per scheduler; reused across cycles. Phases call
    ``acquire(...)`` with the appropriate policy; the coordinator
    runs every shared step and dispatches to the policy at each
    of the five behavioural seams.

    Attributes:
        pipeline_name: Pipeline tag attached to every emitted
            structured log line so multi-pipeline deployments stay
            distinguishable.
        planner: Bounded multi-donor resource-fit search; injected
            by ``setup()`` with the configured ``max_plan_size``
            and ``max_plan_combinations`` bounds.
        transaction: Probe + atomic-remove transaction value object.

    """

    planner: ResourceFitPlanner
    pipeline_name: str = ""
    transaction: DonorTransaction = attrs.field(factory=DonorTransaction)

    def acquire(
        self,
        *,
        policy: DonorPolicy,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
        receiver_intent: int,
        ctx: "AutoscalePlanContext",
    ) -> DonorAcquireResult:
        """Run the policy-driven donor transaction once for ``receiver_view``.

        Returns a ``DonorAcquireResult`` whose ``plan`` is the
        committed donor plan on success or ``None`` when any
        gate rejects. ``reject_reason`` names the failing gate so
        callers can dispatch on mode-specific semantics
        (Phase B treats a commit-time probe failure as
        operator-actionable; Phase C as a recoverable defensive
        event). Atomic-remove failure after a positive probe is a
        scheduler invariant violation and raises rather than
        returning.

        Raises:
            SchedulerInvariantError: The probe-approved donor plan
                failed atomic removal (planner snapshot diverged
                mid-cycle).

        """
        if not policy.is_enabled(context):
            self._log_reject(policy=policy, receiver_view=receiver_view, reason=RejectReason.MASTER_TOGGLE_OFF)
            return DonorAcquireResult(
                plan=None,
                attempted_plan=None,
                reject_reason=RejectReason.MASTER_TOGGLE_OFF,
                placement_reject_reason="",
                gate_result=None,
            )

        eligible = policy.filter_eligible_donors(context, receiver_view)
        if not eligible:
            reason = (
                RejectReason.RECEIVER_ANTI_FLAP
                if self._receiver_in_cooldown(context, receiver_view)
                else RejectReason.NO_CANDIDATES
            )
            self._log_reject(policy=policy, receiver_view=receiver_view, reason=reason)
            return DonorAcquireResult(
                plan=None,
                attempted_plan=None,
                reject_reason=reason,
                placement_reject_reason="",
                gate_result=None,
            )

        candidates = policy.candidate_pool(eligible, context)
        if not candidates:
            self._log_reject(policy=policy, receiver_view=receiver_view, reason=RejectReason.NO_CANDIDATES)
            return DonorAcquireResult(
                plan=None,
                attempted_plan=None,
                reject_reason=RejectReason.NO_CANDIDATES,
                placement_reject_reason="",
                gate_result=None,
            )

        removable_by_stage = {
            donor_index: max(
                0, len(context.worker_ids_by_stage[donor_index]) - context.stage_floors.get(donor_index, 1)
            )
            for donor_index in eligible
        }
        plan = self.planner.find(
            receiver_stage_index=receiver_view.stage_index,
            candidates=candidates,
            worker_nodes=context.worker_node_map,
            ctx=ctx,
            removable_by_stage=removable_by_stage,
        )
        if plan is None:
            self._log_reject(policy=policy, receiver_view=receiver_view, reason=RejectReason.RESOURCE_FIT)
            return DonorAcquireResult(
                plan=None,
                attempted_plan=None,
                reject_reason=RejectReason.RESOURCE_FIT,
                placement_reject_reason="",
                gate_result=None,
            )

        gate_result = policy.evaluate_gate(plan, context, receiver_view, receiver_intent)
        if gate_result is not None and gate_result.reject_reason is not None:
            self._log_reject(
                policy=policy,
                receiver_view=receiver_view,
                reason=gate_result.reject_reason,
            )
            return DonorAcquireResult(
                plan=None,
                attempted_plan=plan,
                reject_reason=gate_result.reject_reason,
                placement_reject_reason="",
                gate_result=gate_result,
            )

        outcome = self.transaction.commit(plan=plan, ctx=ctx)
        if outcome.probe_failed:
            self._log_reject(
                policy=policy,
                receiver_view=receiver_view,
                reason=RejectReason.RESOURCE_FIT,
                placement_reject_reason=outcome.placement_reject_reason,
            )
            return DonorAcquireResult(
                plan=None,
                attempted_plan=plan,
                reject_reason=RejectReason.RESOURCE_FIT,
                placement_reject_reason=outcome.placement_reject_reason,
                gate_result=gate_result,
            )
        if outcome.atomic_remove_failed:
            msg = (
                f"DonorCoordinator.acquire: probe-approved donor plan failed atomic removal "
                f"for receiver {receiver_view.stage_name!r} (planner snapshot diverged mid-cycle)"
            )
            raise SchedulerInvariantError(msg)

        policy.on_commit(plan, context)
        self._log_commit(policy=policy, receiver_view=receiver_view, plan=plan)
        return DonorAcquireResult(
            plan=plan,
            attempted_plan=plan,
            reject_reason=None,
            placement_reject_reason="",
            gate_result=gate_result,
        )

    @staticmethod
    def _receiver_in_cooldown(
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
    ) -> bool:
        """Return True when the receiver is inside its anti-flap cooldown."""
        last_donated = context.last_donation_cycle.get(receiver_view.stage_name)
        if last_donated is None:
            return False
        return context.cycle_counter - last_donated < context.config.cross_stage_donor_anti_flap_cycles

    def _log_reject(
        self,
        *,
        policy: DonorPolicy,
        receiver_view: StageCycleView,
        reason: RejectReason,
        placement_reject_reason: str = "",
    ) -> None:
        """Emit a structured DEBUG line for a donor rejection."""
        logger.bind(
            decision="donor_reject",
            policy=policy.label,
            receiver=receiver_view.stage_name,
            receiver_index=receiver_view.stage_index,
            reject_reason=reason.value,
            placement_reject_reason=placement_reject_reason,
            pipeline=self.pipeline_name,
        ).debug("donor decision rejected")

    def _log_commit(
        self,
        *,
        policy: DonorPolicy,
        receiver_view: StageCycleView,
        plan: DonorPlan,
    ) -> None:
        """Emit a structured INFO line for a successful donor commit."""
        logger.bind(
            decision="donor_commit",
            policy=policy.label,
            receiver=receiver_view.stage_name,
            receiver_index=receiver_view.stage_index,
            removals=[(worker.stage_index, worker.worker_id) for worker in plan.removals],
            pipeline=self.pipeline_name,
        ).info("donor decision committed")


__all__ = ("DonorCoordinator",)
