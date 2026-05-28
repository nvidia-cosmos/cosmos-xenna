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


"""Integration contracts for the donor coordinator that span Phase B and Phase C.

Two architectural contracts live here:

*   Multi-donor anti-flap accounting: ``SaturationPolicy.on_commit``
    advances ``last_donation_cycle`` for every **distinct** donor
    stage in ``DonorPlan.removals``. The contract matters when a
    multi-worker plan draws workers from the same donor stage twice
    (the timestamp is set exactly once per stage) and when it draws
    from multiple distinct stages (every stage's timestamp advances
    in lock step).
*   Floor vs Saturation policy asymmetry through
    ``DonorCoordinator.acquire``: a donor that ``SaturationPolicy``
    rejects on the signal-trust eligibility filter is still admitted
    by ``FloorPolicy``. Floor enforcement is structural and bypasses
    the trust / economic gates that saturation mode applies.

Helper-level coverage of ``ResourceFitPlanner``, the economic gate,
and ``DonorCoordinator`` failure modes lives in
``test_donor_resource_fit.py``, ``test_donor_economics.py``, and
``test_donor_coordinator.py``; this file does not duplicate those.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.donor.economic_gate import EconomicGate
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.policy import FloorPolicy, SaturationPolicy
from cosmos_xenna.pipelines.private.scheduling_py.donor.resource_fit import ResourceFitPlanner
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorPlan, DonorWorker, RejectReason
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import StageCycleView
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    ClassifierState,
    GrowthMode,
    GrowthState,
    StageRuntimeState,
    StageState,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _saturation_policy(config: SaturationAwareConfig | None = None) -> SaturationPolicy:
    """Build a ``SaturationPolicy`` with an injected economic gate.

    Mirrors the helper in ``test_donor_policy.py`` so the
    integration tests build the policy through the same path as
    the unit tests; the production scheduler wires the gate in
    ``setup()``.

    """
    return SaturationPolicy(gate=EconomicGate(config=config if config is not None else SaturationAwareConfig()))


def _coordinator() -> DonorCoordinator:
    """Build a default-bounded coordinator for the integration tests.

    Generous search bounds keep the tests focused on the policy
    asymmetry and on-commit accounting; the production scheduler
    configures these bounds in ``setup()``.

    """
    return DonorCoordinator(planner=ResourceFitPlanner(max_plan_size=4, max_plan_combinations=32))


@attrs.frozen
class _AlwaysFeasibleProbeResult:
    """Probe stub that always reports the candidate plan as feasible."""

    feasible: bool = True
    reject_reason: str = ""


@attrs.define
class _AlwaysFeasibleCtx:
    """Fake autoscale planner context for unit-level coordinator integration.

    ``probe_add_after_removals`` returns feasible unconditionally so
    the commit-time probe never blocks the test. ``remove_workers_atomically``
    is a noop returning True so ``DonorTransaction.commit`` reaches
    the success branch and the coordinator invokes
    ``policy.on_commit``.

    """

    def probe_add_after_removals(
        self,
        removals: list[tuple[int, str]],
        add_stage_index: int,
    ) -> _AlwaysFeasibleProbeResult:
        """Always-feasible probe stub; the coordinator never sees a probe rejection."""
        del removals, add_stage_index
        return _AlwaysFeasibleProbeResult()

    def remove_workers_atomically(self, removals: list[tuple[int, str]]) -> bool:
        """Atomic-remove stub; the coordinator never sees a removal failure."""
        del removals
        return True


def _cluster(*, total_cpus_per_node: int = 4) -> resources.ClusterResources:
    """Single-node CPU cluster used by every fixture in this file."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus_per_node,
                gpus=[],
                name="node-0",
            ),
        },
    )


def _problem(stage_names: list[str]) -> data_structures.Problem:
    """Build a problem with one CPU stage per name."""
    cluster = _cluster()
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    return data_structures.Problem(
        cluster,
        [
            data_structures.ProblemStage(
                name=name,
                stage_batch_size=1,
                worker_shape=cpu_shape,
                requested_num_workers=None,
                over_provision_factor=None,
            )
            for name in stage_names
        ],
    )


def _saturated_state(name: str, *, streak: int = 99) -> StageRuntimeState:
    """Runtime state representing a steady SATURATED stage."""
    return StageRuntimeState(
        stage_name=name,
        classifier=ClassifierState(state=StageState.SATURATED, streak=streak),
        growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
    )


def _config(**overrides: object) -> SaturationAwareConfig:
    """Build a default cluster-wide config; ``overrides`` flips toggles."""
    base: dict[str, object] = {
        "enable_cross_stage_donor": True,
        "donor_must_be_strictly_upstream": False,
        "cross_stage_donor_require_over_provisioned": True,
        "cross_stage_donor_exclude_hold_state": True,
        "cross_stage_donor_anti_flap_cycles": 30,
        "stage_defaults": SaturationAwareStageConfig(
            min_workers=1,
            over_provisioned_streak_min_cycles=3,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
        ),
    }
    base.update(overrides)
    return SaturationAwareConfig(**base)  # type: ignore[arg-type]


def _make_donor_context(scheduler: SaturationAwareScheduler, stage_names: list[str]) -> DonorPlanningContext:
    """Build a minimal ``DonorPlanningContext`` bound to the scheduler's live ledger.

    Carries only the fields ``SaturationPolicy.on_commit`` consults:
    the stage-name tuple (for the stage_index -> name translation)
    and the live ``last_donation_cycle`` reference so the policy's
    in-place mutation flows back to the scheduler's cross-cycle
    ledger. Other planning-context fields default to empty;
    on_commit does not read them.

    """
    return DonorPlanningContext(
        stage_names=tuple(stage_names),
        stage_configs={},
        stage_states={},
        stage_floors={},
        worker_ids_by_stage=(),
        worker_ages={},
        worker_node_map={},
        d_k_now={},
        effective_capacities={},
        s_k_ewma={},
        slots_per_worker_by_stage={},
        donor_warmup_exclusions=frozenset(),
        cycle_counter=scheduler.ledgers.cycle_counter,
        last_donation_cycle=scheduler.ledgers.last_donation_cycle,
        config=scheduler._config,
    )


class TestSaturationPolicyOnCommitAntiFlapPerDistinctStage:
    """``SaturationPolicy.on_commit`` advances ``last_donation_cycle`` for every distinct donor stage."""

    def test_two_workers_from_same_stage_advance_one_timestamp(self) -> None:
        """Two removals from the same donor stage produce one ledger entry, not two."""
        scheduler = SaturationAwareScheduler(_config())
        scheduler.setup(_problem(["A", "B", "C"]))
        scheduler.ledgers.cycle_counter = 7
        plan = DonorPlan(
            removals=(
                DonorWorker(stage_index=0, worker_id="A-w0", age=1),
                DonorWorker(stage_index=0, worker_id="A-w1", age=2),
            ),
            receiver_stage_index=2,
        )

        _saturation_policy().on_commit(plan, _make_donor_context(scheduler, ["A", "B", "C"]))

        assert scheduler.ledgers.last_donation_cycle == {"A": 7}

    def test_two_workers_from_distinct_stages_advance_each_timestamp(self) -> None:
        """A multi-stage plan timestamps every distinct donor stage at the current cycle."""
        scheduler = SaturationAwareScheduler(_config())
        scheduler.setup(_problem(["A", "B", "C"]))
        scheduler.ledgers.cycle_counter = 11
        plan = DonorPlan(
            removals=(
                DonorWorker(stage_index=0, worker_id="A-w0", age=1),
                DonorWorker(stage_index=1, worker_id="B-w0", age=2),
            ),
            receiver_stage_index=2,
        )

        _saturation_policy().on_commit(plan, _make_donor_context(scheduler, ["A", "B", "C"]))

        assert scheduler.ledgers.last_donation_cycle == {"A": 11, "B": 11}


class TestPolicyAsymmetryFloorBypassesSignalTrust:
    """Floor mode admits a donor that saturation mode rejects on the signal-trust filter.

    Pins the architectural contract that ``FloorPolicy.filter_eligible_donors``
    does NOT consult the signal-trust gate; floor enforcement is
    structural. The same donor pool is rejected by
    ``SaturationPolicy`` because ``SaturationPolicy.filter_eligible_donors``
    drops donors whose ``signal_trust < cross_stage_donor_min_trust``.
    The asymmetry is observed end-to-end through ``DonorCoordinator.acquire``:
    one policy returns ``NO_CANDIDATES``, the other returns a
    committed plan.

    """

    def test_low_trust_donor_rejected_by_saturation_admitted_by_floor(self) -> None:
        """Same donor pool: saturation -> NO_CANDIDATES, floor -> committed plan."""
        # Donor A has classifier_streak=3 (clears the layer-1
        # over_provisioned + streak gate) and noise_ewma=0 ->
        # signal_trust = min(3, 60) / (1 + 0) = 3.0. Configure
        # ``cross_stage_donor_min_trust=10.0`` so saturation mode
        # drops A on the eligibility filter. Floor mode does not
        # consult the trust gate; it must still admit A.
        config = _config(cross_stage_donor_min_trust=10.0)
        stage_names = ["A", "B"]
        stage_states = {
            "A": StageRuntimeState(
                stage_name="A",
                classifier=ClassifierState(state=StageState.OVER_PROVISIONED, streak=3),
                growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
            ),
            "B": _saturated_state("B"),
        }
        stage_configs = {
            name: SaturationAwareStageConfig(
                min_workers=1,
                # streak=3 satisfies layer 1; saturated_streak_min_cycles
                # defaults to 2 so the asymmetric-stabilization
                # validator (oop > sat) is honoured.
                over_provisioned_streak_min_cycles=3,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
            )
            for name in stage_names
        }

        # Shared planning context. ``last_donation_cycle`` is the
        # only mutable field; the saturation path may write to it on
        # commit, the floor path leaves it alone.
        last_donation_cycle: dict[str, int] = {}
        receiver_view = StageCycleView(
            stage_index=1,
            stage_name="B",
            runtime_state=stage_states["B"],
            current_workers=1,
        )
        ctx = _AlwaysFeasibleCtx()
        coordinator = _coordinator()

        def _build_context(*, exclude_warmup: bool = False) -> DonorPlanningContext:
            return DonorPlanningContext(
                stage_names=tuple(stage_names),
                stage_configs=stage_configs,
                stage_states=stage_states,
                stage_floors={0: 1, 1: 1},
                worker_ids_by_stage=(("A-w0", "A-w1"), ("B-w0",)),
                worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
                worker_node_map={},
                d_k_now={},
                effective_capacities={},
                s_k_ewma={},
                slots_per_worker_by_stage={},
                donor_warmup_exclusions=frozenset(),
                cycle_counter=100,
                last_donation_cycle=last_donation_cycle,
                config=config,
            )

        # Saturation mode drops the low-trust donor from eligibility
        # so no candidates remain.
        saturation_result = coordinator.acquire(
            policy=_saturation_policy(config),
            context=_build_context(),
            receiver_view=receiver_view,
            receiver_intent=1,
            ctx=ctx,  # type: ignore[arg-type]
        )

        assert saturation_result.committed is False
        assert saturation_result.reject_reason is RejectReason.NO_CANDIDATES, (
            "Saturation mode must reject the low-trust donor on the eligibility filter"
        )

        # Floor mode admits the same donor and commits the plan.
        floor_result = coordinator.acquire(
            policy=FloorPolicy(),
            context=_build_context(),
            receiver_view=receiver_view,
            receiver_intent=1,
            ctx=ctx,  # type: ignore[arg-type]
        )

        assert floor_result.committed is True, (
            "Floor mode must admit the same donor; signal-trust does not gate floor enforcement"
        )
        assert floor_result.plan is not None
        assert floor_result.plan.removals[0].stage_index == 0
