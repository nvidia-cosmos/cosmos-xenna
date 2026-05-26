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

"""Donor selection strategies for the saturation-aware scheduler.

Two modes share the same select -> probe -> commit transaction
but differ in five behavioural seams: master toggle, eligibility
filter, candidate exclusion, economic gate, ledger advance on
commit. ``DonorPolicy`` (Protocol) is the Strategy interface;
``FloorPolicy`` and ``SaturationPolicy`` are the two concrete
strategies. The shared transaction lives in ``DonorCoordinator``.
See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

import operator
from typing import ClassVar, Protocol

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.economic_gate import EconomicGate, signal_trust
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import (
    DonorPlan,
    DonorWorker,
    GateResult,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import StageCycleView
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import GrowthMode, StageState


class DonorPolicy(Protocol):
    """Strategy that customises the donor select-then-commit flow.

    Concrete implementations: :class:`FloorPolicy` (Phase B floor
    enforcement, no economic gate, no warmup exclusion, no
    anti-flap) and :class:`SaturationPolicy` (Phase C saturation
    grow, full economic gate, anti-flap, warmup exclusion). The
    coordinator owns the shared select-probe-commit mechanics; the
    policy owns the five behaviour seams below.

    """

    label: ClassVar[str]

    def is_enabled(self, context: DonorPlanningContext) -> bool:
        """Whether this policy is enabled for the current cycle."""
        ...

    def filter_eligible_donors(
        self,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
    ) -> list[int]:
        """Stage indices that may donate to ``receiver_view``."""
        ...

    def candidate_pool(
        self,
        eligible_stages: list[int],
        context: DonorPlanningContext,
    ) -> list[DonorWorker]:
        """Worker-level candidates after policy-specific exclusion."""
        ...

    def evaluate_gate(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
        receiver_intent: int,
    ) -> GateResult | None:
        """Run the policy's economic gate; ``None`` skips evaluation."""
        ...

    def on_commit(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
    ) -> None:
        """Hook fired after a successful commit + receiver retry."""
        ...


def _build_candidate(
    *,
    donor_index: int,
    worker_id: str,
    context: DonorPlanningContext,
) -> DonorWorker:
    """Build a ``DonorWorker`` carrying the age tiebreaker."""
    return DonorWorker(
        stage_index=donor_index,
        worker_id=worker_id,
        age=context.worker_ages.get(worker_id, 0),
    )


_CANDIDATE_SORT_KEY = operator.attrgetter("age", "worker_id", "stage_index")


@attrs.frozen
class FloorPolicy:
    """Phase B donor strategy: structural floor enforcement.

    No master toggle (floor is structural), no economic gate, no
    warmup exclusion, no anti-flap ledger. Eligible donors are
    every non-receiver stage whose post-donation worker count
    stays at or above its own floor; upstream preferred with
    downstream as fallback. The non-negotiable donor-floor rule
    is the only filter.

    """

    label: ClassVar[str] = "floor"

    def is_enabled(self, context: DonorPlanningContext) -> bool:
        """Always enabled - floor enforcement is structural."""
        del context
        return True

    def filter_eligible_donors(
        self,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
    ) -> list[int]:
        """Return non-receiver stages that can spare a worker, upstream first."""
        upstream: list[int] = []
        downstream: list[int] = []
        for donor_index, worker_ids in enumerate(context.worker_ids_by_stage):
            if donor_index == receiver_view.stage_index:
                continue
            if len(worker_ids) - 1 < context.stage_floors.get(donor_index, 1):
                continue
            if donor_index < receiver_view.stage_index:
                upstream.append(donor_index)
            else:
                downstream.append(donor_index)
        return upstream + downstream

    def candidate_pool(
        self,
        eligible_stages: list[int],
        context: DonorPlanningContext,
    ) -> list[DonorWorker]:
        """Expand eligible stages to workers and sort age-ascending.

        Floor mode does NOT exclude workers in
        ``context.donor_warmup_exclusions`` because deadlocking the
        floor on warmup-protected donors is a worse failure mode
        than killing a young donor.

        """
        candidates: list[DonorWorker] = []
        for donor_index in eligible_stages:
            for worker_id in context.worker_ids_by_stage[donor_index]:
                candidates.append(_build_candidate(donor_index=donor_index, worker_id=worker_id, context=context))
        candidates.sort(key=_CANDIDATE_SORT_KEY)
        return candidates

    def evaluate_gate(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
        receiver_intent: int,
    ) -> GateResult | None:
        """Floor mode has no economic gate."""
        del plan, context, receiver_view, receiver_intent
        return None

    def on_commit(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
    ) -> None:
        """Floor mode does not maintain an anti-flap ledger."""
        del plan, context
        return None


@attrs.frozen
class SaturationPolicy:
    """Phase C donor strategy: saturation-driven cross-stage donation.

    Filters by four anti-flap / trust gates beyond the floor rule:
    classifier OVER_PROVISIONED + streak, growth mode != HOLD,
    receiver-recent-donor cooldown, donor signal trust. Excludes
    workers inside the donor-warmup grace window from the
    candidate pool. Runs the throughput-first economic gate
    (``EconomicGate.evaluate``) on the chosen plan. Advances the
    anti-flap ledger after a successful commit so the donor stage
    cannot re-donate within ``cross_stage_donor_anti_flap_cycles``.

    Attributes:
        gate: Throughput-first economic gate; injected by
            ``setup()`` with the scheduler's ``SaturationAwareConfig``
            so every receiver in the same cycle reads the same
            threshold values.

    """

    gate: EconomicGate
    label: ClassVar[str] = "saturation"

    def is_enabled(self, context: DonorPlanningContext) -> bool:
        """Gated by ``enable_cross_stage_donor``."""
        return context.config.enable_cross_stage_donor

    def filter_eligible_donors(
        self,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
    ) -> list[int]:
        """Apply receiver-anti-flap then per-donor classifier/streak/trust gates."""
        cooldown = context.config.cross_stage_donor_anti_flap_cycles
        last_donated = context.last_donation_cycle.get(receiver_view.stage_name)
        if last_donated is not None and context.cycle_counter - last_donated < cooldown:
            return []

        eligible: list[int] = []
        require_op = context.config.cross_stage_donor_require_over_provisioned
        exclude_hold = context.config.cross_stage_donor_exclude_hold_state
        trust_cap = context.config.cross_stage_donor_trust_streak_cap
        min_trust = context.config.cross_stage_donor_min_trust
        for donor_index, worker_ids in enumerate(context.worker_ids_by_stage):
            if donor_index == receiver_view.stage_index:
                continue
            if context.config.donor_must_be_strictly_upstream and donor_index >= receiver_view.stage_index:
                continue
            if len(worker_ids) - 1 < context.stage_floors.get(donor_index, 1):
                continue

            donor_name = context.stage_names[donor_index]
            donor_state = context.stage_states.get(donor_name)
            if donor_state is None:
                continue
            if require_op:
                donor_cfg = context.stage_configs.get(donor_name)
                if donor_cfg is None:
                    continue
                if donor_state.classifier.state is not StageState.OVER_PROVISIONED:
                    continue
                if donor_state.classifier.streak < donor_cfg.over_provisioned_streak_min_cycles:
                    continue
            if exclude_hold and donor_state.growth.mode is GrowthMode.HOLD:
                continue
            if signal_trust(donor_state, trust_streak_cap=trust_cap) < min_trust:
                continue
            eligible.append(donor_index)
        return eligible

    def candidate_pool(
        self,
        eligible_stages: list[int],
        context: DonorPlanningContext,
    ) -> list[DonorWorker]:
        """Expand eligible stages to workers, dropping warmup-protected ids."""
        excluded = context.donor_warmup_exclusions
        candidates: list[DonorWorker] = []
        for donor_index in eligible_stages:
            for worker_id in context.worker_ids_by_stage[donor_index]:
                if worker_id in excluded:
                    continue
                candidates.append(_build_candidate(donor_index=donor_index, worker_id=worker_id, context=context))
        candidates.sort(key=_CANDIDATE_SORT_KEY)
        return candidates

    def evaluate_gate(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
        receiver_view: StageCycleView,
        receiver_intent: int,
    ) -> GateResult | None:
        """Delegate to the throughput-first economic gate."""
        del receiver_view
        return self.gate.evaluate(
            plan=plan,
            stage_names=list(context.stage_names),
            stage_states=dict(context.stage_states),
            receiver_intent=receiver_intent,
            d_k_now=context.d_k_now,
            effective_capacities=context.effective_capacities,
            s_k_ewma=context.s_k_ewma,
            slots_per_worker_by_stage=context.slots_per_worker_by_stage,
        )

    def on_commit(
        self,
        plan: DonorPlan,
        context: DonorPlanningContext,
    ) -> None:
        """Advance ``last_donation_cycle`` for every distinct donor stage."""
        for donor_index in {worker.stage_index for worker in plan.removals}:
            donor_name = context.stage_names[donor_index]
            context.last_donation_cycle[donor_name] = context.cycle_counter


__all__ = ("DonorPolicy", "FloorPolicy", "SaturationPolicy")
