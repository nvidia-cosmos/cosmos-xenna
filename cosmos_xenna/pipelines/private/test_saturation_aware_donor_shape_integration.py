# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration contracts for the donor planner that span Phase B and Phase C.

Two architectural contracts live here:

*   Multi-donor anti-flap accounting: ``_record_donation_success``
    advances ``_last_donation_cycle`` for every **distinct** donor
    stage in ``DonorPlan.removals``. The contract matters when a
    multi-worker plan draws workers from the same donor stage twice
    (the timestamp is set exactly once per stage) and when it draws
    from multiple distinct stages (every stage's timestamp advances
    in lock step).
*   Phase B vs Phase C economic-gate split:
    ``select_youngest_eligible_donor`` (Phase B floor) accepts a
    donor that ``find_saturation_donor`` (Phase C saturation) would
    reject on the signal-trust gate. Floor enforcement is non-
    negotiable and bypasses every economic / trust gate; saturation
    mode applies all of them.

Helper-level coverage of ``_resource_fit_plan``, the economic gate,
the decision-log schema, mid-plan atomicity failures, and probe-vs-
actual disagreements lives in ``test_donor_resource_fit.py``,
``test_donor_economics.py``, ``test_saturation_aware_donor.py``,
``test_saturation_aware_donor_saturation.py``, and
``test_saturation_aware_allocation_error.py``; this file does not
duplicate those.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor import (
    DonorPlan,
    DonorWorker,
    find_saturation_donor,
    select_youngest_eligible_donor,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@attrs.frozen
class _AlwaysFeasibleProbeResult:
    feasible: bool = True
    reject_reason: str = ""


@attrs.define
class _AlwaysFeasibleCtx:
    """Fake autoscale planner context whose probe always reports feasible."""

    def probe_add_after_removals(
        self,
        removals: list[tuple[int, str]],
        add_stage_index: int,
    ) -> _AlwaysFeasibleProbeResult:
        del removals, add_stage_index
        return _AlwaysFeasibleProbeResult()


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


def _over_provisioned_state(name: str, *, streak: int = 30) -> _StageRuntimeState:
    """Runtime state representing a steady OVER_PROVISIONED stage."""
    return _StageRuntimeState(
        stage_name=name,
        classifier_state=StageState.OVER_PROVISIONED,
        classifier_streak=streak,
        growth_mode=GrowthMode.TRACKING,
        growth_streak=10,
    )


def _saturated_state(name: str, *, streak: int = 99) -> _StageRuntimeState:
    """Runtime state representing a steady SATURATED stage."""
    return _StageRuntimeState(
        stage_name=name,
        classifier_state=StageState.SATURATED,
        classifier_streak=streak,
        growth_mode=GrowthMode.TRACKING,
        growth_streak=10,
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


class TestRecordDonationSuccessAntiFlapPerDistinctStage:
    """``_record_donation_success`` advances ``_last_donation_cycle`` for every distinct donor stage."""

    def test_two_workers_from_same_stage_advance_one_timestamp(self) -> None:
        """Two removals from the same donor stage produce one ledger entry, not two."""
        scheduler = SaturationAwareScheduler(_config())
        scheduler.setup(_problem(["A", "B", "C"]))
        scheduler._cycle_counter = 7
        plan = DonorPlan(
            removals=(
                DonorWorker(stage_index=0, worker_id="A-w0", age=1),
                DonorWorker(stage_index=0, worker_id="A-w1", age=2),
            ),
            receiver_stage_index=2,
        )

        scheduler._record_donation_success(plan=plan)

        assert scheduler._last_donation_cycle == {"A": 7}

    def test_two_workers_from_distinct_stages_advance_each_timestamp(self) -> None:
        """A multi-stage plan timestamps every distinct donor stage at the current cycle."""
        scheduler = SaturationAwareScheduler(_config())
        scheduler.setup(_problem(["A", "B", "C"]))
        scheduler._cycle_counter = 11
        plan = DonorPlan(
            removals=(
                DonorWorker(stage_index=0, worker_id="A-w0", age=1),
                DonorWorker(stage_index=1, worker_id="B-w0", age=2),
            ),
            receiver_stage_index=2,
        )

        scheduler._record_donation_success(plan=plan)

        assert scheduler._last_donation_cycle == {"A": 11, "B": 11}


class TestPhaseBFloorBypassesSignalTrust:
    """Floor mode (Phase B) admits donors that saturation mode rejects on signal-trust.

    Pins the architectural contract that
    ``select_youngest_eligible_donor`` does not consult the
    signal-trust / spread / throughput / donor-flip / balance gates;
    floor enforcement is non-negotiable. ``find_saturation_donor``
    on the same input rejects because the donor's clamped streak
    fails the ``cross_stage_donor_min_trust`` threshold.
    """

    def test_low_trust_donor_rejected_by_saturation_admitted_by_floor(self) -> None:
        """The same donor pool yields ``None`` for saturation mode and a plan for floor mode."""
        # Donor A has classifier_streak=3 (clears the layer-1
        # over_provisioned + streak gate) and noise_ewma=0 ->
        # signal_trust = min(3, 60) / (1 + 0) = 3.0. Configure
        # ``cross_stage_donor_min_trust=10.0`` so saturation mode
        # rejects on layer 4 specifically. Floor mode does not consult
        # the trust gate; it must still admit A.
        config = _config(cross_stage_donor_min_trust=10.0)
        stage_states = {
            "A": _StageRuntimeState(
                stage_name="A",
                classifier_state=StageState.OVER_PROVISIONED,
                classifier_streak=3,  # trust = 3.0; below min_trust = 10.0.
                growth_mode=GrowthMode.TRACKING,
                growth_streak=10,
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
            for name in ("A", "B")
        }

        # Saturation mode rejects on signal-trust.
        saturation_decision = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            worker_nodes={},
            stage_states=stage_states,
            config=config,
            stage_configs=stage_configs,
            cycle=100,
            last_donation_cycle={},
            ctx=_AlwaysFeasibleCtx(),  # type: ignore[arg-type]
            receiver_intent=1,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
        )

        assert saturation_decision is None, "Saturation mode must reject the low-trust donor on the signal-trust gate"

        # Floor mode admits the same donor on the same pool.
        floor_plan = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            worker_nodes={},
            ctx=_AlwaysFeasibleCtx(),  # type: ignore[arg-type]
            max_plan_size=4,
            max_plan_combinations=32,
        )

        assert floor_plan is not None, (
            "Floor mode must admit the same donor; signal-trust does not gate floor enforcement"
        )
        assert floor_plan.removals[0].stage_index == 0
