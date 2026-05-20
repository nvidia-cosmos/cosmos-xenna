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


"""Tests for ``find_saturation_donor`` and the Phase C cross-stage donor fallback.

The helper layer is exercised in isolation to pin each of the five
anti-flap layers, the master toggle, and the strict-upstream filter.
The integration layer is exercised through
``SaturationAwareScheduler.autoscale`` to pin the user-stuck-cluster
regression: a receiver that wants to grow because it is SATURATED
sees a successful donation when a valid donor exists and a clean
non-fatal log when every other stage is at its own floor.

The five anti-flap layers under test:

    1. Donor must be OVER_PROVISIONED with full streak.
    2. Donor must not be in HOLD growth mode.
    3. Receiver-was-recent-donor cooldown
       (``cross_stage_donor_anti_flap_cycles``).
    4. Receiver per-cycle absorption cap
       (``cross_stage_donor_max_per_cycle``).
    5. Donor between-donations cooldown
       (``cross_stage_donor_min_donation_interval_cycles``).
"""

from unittest.mock import patch

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor import DonorCandidate, find_saturation_donor
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _saturated_state(name: str, *, streak: int = 99) -> _StageRuntimeState:
    """Build a runtime state representing a steady SATURATED stage."""
    return _StageRuntimeState(
        stage_name=name,
        classifier_state=StageState.SATURATED,
        classifier_streak=streak,
        growth_mode=GrowthMode.TRACKING,
        growth_streak=10,
    )


def _over_provisioned_state(
    name: str,
    *,
    streak: int = 99,
    growth_mode: GrowthMode = GrowthMode.TRACKING,
) -> _StageRuntimeState:
    """Build a runtime state representing a steady OVER_PROVISIONED stage."""
    return _StageRuntimeState(
        stage_name=name,
        classifier_state=StageState.OVER_PROVISIONED,
        classifier_streak=streak,
        growth_mode=growth_mode,
        growth_streak=10,
    )


def _stage_cfg(*, over_provisioned_streak: int = 30) -> SaturationAwareStageConfig:
    """Per-stage config with a customisable OVER_PROVISIONED streak threshold."""
    return SaturationAwareStageConfig(
        min_workers=1,
        over_provisioned_streak_min_cycles=over_provisioned_streak,
        # Saturation donor tests focus on the five-layer anti-flap and
        # cooldown contract, not on the donor warmup grace. Pin both
        # warmup graces to zero so victim and donor candidate pools are
        # unfiltered and the legacy contract remains observable.
        worker_warmup_measurement_grace_s=0.0,
        donor_warmup_grace_s=0.0,
    )


def _config(**overrides: object) -> SaturationAwareConfig:
    """Build a default cluster-wide config; ``overrides`` flips the anti-flap toggles."""
    base: dict[str, object] = {
        "floor_stuck_grace_cycles": 0,
        "enable_dag_priority_growth": True,
        "enable_cross_stage_donor": True,
        "donor_must_be_strictly_upstream": True,
        "cross_stage_donor_require_over_provisioned": True,
        "cross_stage_donor_exclude_hold_state": True,
        "cross_stage_donor_anti_flap_cycles": 30,
        "cross_stage_donor_max_per_cycle": 1,
        "cross_stage_donor_min_donation_interval_cycles": 30,
    }
    base.update(overrides)
    return SaturationAwareConfig(**base)  # type: ignore[arg-type]


def _find_donor(
    *,
    receiver_stage_index: int = 1,
    receiver_stage_name: str = "B",
    stage_names: list[str] | None = None,
    stage_floors: dict[int, int] | None = None,
    worker_ids_by_stage: list[list[str]] | None = None,
    worker_ages: dict[str, int] | None = None,
    stage_states: dict[str, _StageRuntimeState] | None = None,
    config: SaturationAwareConfig | None = None,
    stage_configs: dict[str, SaturationAwareStageConfig] | None = None,
    cycle: int = 100,
    last_donation_cycle: dict[str, int] | None = None,
    donations_received_this_cycle: dict[str, int] | None = None,
) -> DonorCandidate | None:
    """Call ``find_saturation_donor`` with a two-stage eligible default."""
    if stage_names is None:
        stage_names = ["A", "B"]
    if stage_floors is None:
        stage_floors = {0: 1, 1: 1}
    if worker_ids_by_stage is None:
        worker_ids_by_stage = [["A-w0", "A-w1"], ["B-w0"]]
    if worker_ages is None:
        worker_ages = {"A-w0": 5, "A-w1": 3, "B-w0": 2}
    if stage_states is None:
        stage_states = {"A": _over_provisioned_state("A"), "B": _saturated_state("B")}
    if config is None:
        config = _config()
    if stage_configs is None:
        stage_configs = {name: _stage_cfg() for name in stage_names}
    if last_donation_cycle is None:
        last_donation_cycle = {}
    if donations_received_this_cycle is None:
        donations_received_this_cycle = {}
    return find_saturation_donor(
        receiver_stage_index=receiver_stage_index,
        receiver_stage_name=receiver_stage_name,
        stage_names=stage_names,
        stage_floors=stage_floors,
        worker_ids_by_stage=worker_ids_by_stage,
        worker_ages=worker_ages,
        stage_states=stage_states,
        config=config,
        stage_configs=stage_configs,
        cycle=cycle,
        last_donation_cycle=last_donation_cycle,
        donations_received_this_cycle=donations_received_this_cycle,
    )


def _cluster(*, total_cpus_per_node: int = 4) -> resources.ClusterResources:
    """Build the CPU cluster used by orchestrator fixtures."""
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


def _problem(stage_names: list[str], *, total_cpus_per_node: int = 4) -> data_structures.Problem:
    """Build a one-node CPU problem with one-CPU worker shapes."""
    cluster = _cluster(total_cpus_per_node=total_cpus_per_node)
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


def _workers(stage_name: str, count: int) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` one-CPU workers for a stage."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _problem_state(stage_worker_counts: list[tuple[str, int]]) -> data_structures.ProblemState:
    """Build a runtime snapshot from ``(stage_name, worker_count)`` rows."""
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(
                stage_name=name,
                workers=_workers(name, count),
                slots_per_worker=1,
                is_finished=False,
            )
            for name, count in stage_worker_counts
        ],
    )


def _scheduler_for_donation(
    *,
    max_per_cycle: int = 1,
    min_interval_cycles: int = 30,
    total_cpus_per_node: int = 4,
) -> SaturationAwareScheduler:
    """Build a setup-completed scheduler with A eligible to donate to B."""
    cfg = _config(
        cross_stage_donor_max_per_cycle=max_per_cycle,
        cross_stage_donor_min_donation_interval_cycles=min_interval_cycles,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=1,
            over_provisioned_streak_min_cycles=3,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(["A", "B"], total_cpus_per_node=total_cpus_per_node))
    scheduler._stage_states["A"] = _over_provisioned_state("A", streak=3)
    scheduler._stage_states["B"] = _saturated_state("B")
    return scheduler


def _autoscale_with_intents(
    scheduler: SaturationAwareScheduler,
    state: data_structures.ProblemState,
    intents: dict[str, int],
) -> data_structures.Solution:
    """Run autoscale with injected intent deltas."""

    def _inject(_ctx: object, _state: object, **_kwargs: object) -> dict[str, int]:
        return dict(intents)

    with (
        patch.object(scheduler, "_compute_intent_deltas", side_effect=_inject),
        patch.object(scheduler, "_run_phase_d_shrink", return_value=None),
    ):
        return scheduler.autoscale(time=0.0, problem_state=state)


class TestFindSaturationDonorMasterToggle:
    """The master toggle short-circuits the helper to ``None``."""

    def test_master_toggle_disabled_returns_none(self) -> None:
        """``enable_cross_stage_donor=False`` rejects every candidate."""
        config = _config(enable_cross_stage_donor=False)
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _saturated_state("B"),
            },
            config=config,
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is None


class TestFindSaturationDonorClassifierLayer:
    """Layer 1: donor must be OVER_PROVISIONED with full streak."""

    def test_normal_classifier_state_is_filtered(self) -> None:
        """A donor with classifier_state=NORMAL is rejected when require_over_provisioned=True."""
        donor_state = _StageRuntimeState(
            stage_name="A",
            classifier_state=StageState.NORMAL,
            classifier_streak=99,
            growth_mode=GrowthMode.TRACKING,
            growth_streak=10,
        )
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={"A": donor_state, "B": _saturated_state("B")},
            config=_config(),
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is None

    def test_over_provisioned_with_short_streak_is_filtered(self) -> None:
        """A donor with classifier_state=OVER_PROVISIONED but streak < threshold is rejected."""
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A", streak=10),
                "B": _saturated_state("B"),
            },
            config=_config(),
            stage_configs={"A": _stage_cfg(over_provisioned_streak=30), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is None

    def test_over_provisioned_with_full_streak_is_eligible(self) -> None:
        """A donor with classifier_state=OVER_PROVISIONED and full streak is selected."""
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A", streak=30),
                "B": _saturated_state("B"),
            },
            config=_config(),
            stage_configs={"A": _stage_cfg(over_provisioned_streak=30), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is not None
        assert donor.stage_index == 0
        assert donor.worker_id == "A-w1"


class TestFindSaturationDonorHoldLayer:
    """Layer 2: donor in HOLD growth mode is excluded."""

    def test_donor_in_hold_growth_mode_is_filtered(self) -> None:
        """``cross_stage_donor_exclude_hold_state=True`` rejects HOLD donors."""
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A", growth_mode=GrowthMode.HOLD),
                "B": _saturated_state("B"),
            },
            config=_config(),
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is None


class TestFindSaturationDonorReceiverAntiFlap:
    """Layer 3: receiver was a recent donor and cannot receive yet."""

    def test_receiver_recent_donor_within_anti_flap_window_is_blocked(self) -> None:
        """A receiver that donated less than ``anti_flap_cycles`` ago cannot receive."""
        config = _config(cross_stage_donor_anti_flap_cycles=30)
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _saturated_state("B"),
            },
            config=config,
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=120,
            last_donation_cycle={"B": 100},  # B donated at cycle 100; window of 30 spans through 130
            donations_received_this_cycle={},
        )

        assert donor is None

    def test_receiver_recent_donor_outside_anti_flap_window_is_allowed(self) -> None:
        """A receiver whose last donation is older than ``anti_flap_cycles`` is no longer blocked."""
        config = _config(cross_stage_donor_anti_flap_cycles=30)
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _saturated_state("B"),
            },
            config=config,
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=131,  # 31 cycles since B's last donation; window cleared
            last_donation_cycle={"B": 100},
            donations_received_this_cycle={},
        )

        assert donor is not None


class TestFindSaturationDonorReceiverPerCycleCap:
    """Layer 4: receiver has already absorbed ``max_per_cycle`` donations this cycle."""

    def test_receiver_at_per_cycle_cap_is_blocked(self) -> None:
        """A receiver that already absorbed the configured per-cycle cap is rejected."""
        config = _config(cross_stage_donor_max_per_cycle=1)
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _saturated_state("B"),
            },
            config=config,
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={"B": 1},
        )

        assert donor is None


class TestFindSaturationDonorDonorCooldown:
    """Layer 5: donor donated recently and is on cooldown."""

    def test_donor_within_min_interval_is_filtered(self) -> None:
        """A donor whose ``last_donation_cycle`` is too recent is rejected."""
        config = _config(cross_stage_donor_min_donation_interval_cycles=30)
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _saturated_state("B"),
            },
            config=config,
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=120,
            last_donation_cycle={"A": 100},
            donations_received_this_cycle={},
        )

        assert donor is None


class TestFindSaturationDonorStrictUpstream:
    """``donor_must_be_strictly_upstream=True`` keeps donors strictly upstream of the receiver."""

    def test_downstream_donor_is_filtered_when_strict_upstream(self) -> None:
        """A donor downstream of the receiver is rejected when the strict-upstream flag is on."""
        donor = find_saturation_donor(
            receiver_stage_index=0,  # receiver is the most upstream stage
            receiver_stage_name="A",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0"], ["B-w0", "B-w1"]],
            worker_ages={"A-w0": 5, "B-w0": 3, "B-w1": 2},
            stage_states={
                "A": _saturated_state("A"),
                "B": _over_provisioned_state("B"),  # eligible classifier-wise
            },
            config=_config(donor_must_be_strictly_upstream=True),
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is None

    def test_downstream_donor_allowed_when_strict_upstream_off(self) -> None:
        """With ``donor_must_be_strictly_upstream=False`` a downstream donor is eligible."""
        donor = find_saturation_donor(
            receiver_stage_index=0,
            receiver_stage_name="A",
            stage_names=["A", "B"],
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0"], ["B-w0", "B-w1"]],
            worker_ages={"A-w0": 5, "B-w0": 3, "B-w1": 2},
            stage_states={
                "A": _saturated_state("A"),
                "B": _over_provisioned_state("B"),
            },
            config=_config(donor_must_be_strictly_upstream=False),
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is not None
        assert donor.stage_index == 1
        assert donor.worker_id == "B-w1"  # younger of B-w0(age=3) and B-w1(age=2)


class TestFindSaturationDonorFloorPreservation:
    """The non-negotiable donor-floor rule: a donor at its floor cannot donate."""

    def test_donor_at_floor_is_filtered(self) -> None:
        """A stage with ``len(workers) - 1 < floor`` is filtered regardless of classifier."""
        donor = find_saturation_donor(
            receiver_stage_index=1,
            receiver_stage_name="B",
            stage_names=["A", "B"],
            stage_floors={0: 2, 1: 1},  # A's floor is 2; A has only 2 workers
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3, "B-w0": 2},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _saturated_state("B"),
            },
            config=_config(),
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is None


class TestFindSaturationDonorYoungestSelection:
    """Among eligible donors, the youngest worker (with worker_id tiebreaker) wins."""

    def test_youngest_worker_across_eligible_donors_wins(self) -> None:
        """Two eligible donor stages: the youngest worker overall is selected."""
        donor = find_saturation_donor(
            receiver_stage_index=2,
            receiver_stage_name="C",
            stage_names=["A", "B", "C"],
            stage_floors={0: 1, 1: 1, 2: 1},
            worker_ids_by_stage=[
                ["A-w0", "A-w1"],
                ["B-w0", "B-w1"],
                ["C-w0"],
            ],
            worker_ages={"A-w0": 5, "A-w1": 4, "B-w0": 2, "B-w1": 3, "C-w0": 1},
            stage_states={
                "A": _over_provisioned_state("A"),
                "B": _over_provisioned_state("B"),
                "C": _saturated_state("C"),
            },
            config=_config(),
            stage_configs={"A": _stage_cfg(), "B": _stage_cfg(), "C": _stage_cfg()},
            cycle=100,
            last_donation_cycle={},
            donations_received_this_cycle={},
        )

        assert donor is not None
        assert donor.stage_index == 1
        assert donor.worker_id == "B-w0"
        assert donor.age == 2

    def test_equal_worker_ages_choose_lexicographically_smallest_worker_id(self) -> None:
        """Equal-age candidates fall back to worker id for deterministic output."""
        donor = _find_donor(
            worker_ids_by_stage=[["A-w2", "A-w1", "A-w0"], ["B-w0"]],
            worker_ages={"A-w0": 7, "A-w1": 7, "A-w2": 7, "B-w0": 1},
        )

        assert donor is not None
        assert donor.worker_id == "A-w0"


class TestFindSaturationDonorBoundaryAndToggleCases:
    """Pin exact cooldown boundaries and individually disabled layers."""

    def test_receiver_anti_flap_allows_at_exact_window_boundary(self) -> None:
        """Layer 3 blocks ``< window`` only; equality is eligible."""
        donor = _find_donor(
            config=_config(cross_stage_donor_anti_flap_cycles=30),
            cycle=130,
            last_donation_cycle={"B": 100},
        )

        assert donor is not None

    def test_donor_min_interval_allows_at_exact_window_boundary(self) -> None:
        """Layer 5 blocks ``< interval`` only; equality is eligible."""
        donor = _find_donor(
            config=_config(cross_stage_donor_min_donation_interval_cycles=30),
            cycle=130,
            last_donation_cycle={"A": 100},
        )

        assert donor is not None

    def test_non_over_provisioned_donor_allowed_when_requirement_disabled(self) -> None:
        """Disabling layer 1 permits a NORMAL donor that still satisfies floor rules."""
        donor = _find_donor(
            config=_config(cross_stage_donor_require_over_provisioned=False),
            stage_states={
                "A": _StageRuntimeState(
                    stage_name="A",
                    classifier_state=StageState.NORMAL,
                    classifier_streak=0,
                    growth_mode=GrowthMode.TRACKING,
                    growth_streak=0,
                ),
                "B": _saturated_state("B"),
            },
        )

        assert donor is not None
        assert donor.stage_index == 0

    def test_hold_donor_allowed_when_hold_exclusion_disabled(self) -> None:
        """Disabling layer 2 permits a HOLD donor that otherwise qualifies."""
        donor = _find_donor(
            config=_config(cross_stage_donor_exclude_hold_state=False),
            stage_states={
                "A": _over_provisioned_state("A", growth_mode=GrowthMode.HOLD),
                "B": _saturated_state("B"),
            },
        )

        assert donor is not None
        assert donor.stage_index == 0


class TestFindSaturationDonorMalformedInternalSnapshots:
    """Document fail-closed behavior for scheduler-internal snapshot gaps."""

    def test_missing_stage_state_filters_candidate_without_crashing(self) -> None:
        """A missing donor runtime state is treated as ineligible."""
        donor = _find_donor(stage_states={"B": _saturated_state("B")})

        assert donor is None

    def test_missing_stage_config_filters_candidate_when_required(self) -> None:
        """A missing donor config is fail-closed when layer 1 is enabled."""
        donor = _find_donor(stage_configs={"B": _stage_cfg()})

        assert donor is None

    def test_missing_stage_floor_defaults_to_one_worker_floor(self) -> None:
        """A donor missing from ``stage_floors`` can donate only above the implicit floor."""
        donor = _find_donor(stage_floors={1: 1})

        assert donor is not None
        assert donor.stage_index == 0

    def test_large_pipeline_selection_is_deterministic(self) -> None:
        """A large candidate set still picks the globally youngest eligible worker."""
        stage_count = 100
        workers_per_stage = 10
        stage_names = [f"S{stage_index:03d}" for stage_index in range(stage_count)]
        receiver_stage_index = stage_count - 1
        receiver_name = stage_names[receiver_stage_index]
        worker_ids_by_stage = [
            [f"{stage_name}-w{worker_index:02d}" for worker_index in range(workers_per_stage)]
            for stage_name in stage_names
        ]
        worker_ids_by_stage[receiver_stage_index] = [f"{receiver_name}-w00"]
        winner = "S042-w05"
        worker_ages = {
            worker_id: 10_000 + stage_index * workers_per_stage + worker_index
            for stage_index, worker_ids in enumerate(worker_ids_by_stage)
            for worker_index, worker_id in enumerate(worker_ids)
        }
        worker_ages[winner] = 0
        stage_states = {
            stage_name: _over_provisioned_state(stage_name) for stage_name in stage_names[:receiver_stage_index]
        }
        stage_states[receiver_name] = _saturated_state(receiver_name)
        stage_configs = {stage_name: _stage_cfg() for stage_name in stage_names}

        first = _find_donor(
            receiver_stage_index=receiver_stage_index,
            receiver_stage_name=receiver_name,
            stage_names=stage_names,
            stage_floors={stage_index: 1 for stage_index in range(stage_count)},
            worker_ids_by_stage=worker_ids_by_stage,
            worker_ages=worker_ages,
            stage_states=stage_states,
            stage_configs=stage_configs,
        )
        second = _find_donor(
            receiver_stage_index=receiver_stage_index,
            receiver_stage_name=receiver_name,
            stage_names=stage_names,
            stage_floors={stage_index: 1 for stage_index in range(stage_count)},
            worker_ids_by_stage=[list(reversed(worker_ids)) for worker_ids in worker_ids_by_stage],
            worker_ages=worker_ages,
            stage_states=stage_states,
            stage_configs=stage_configs,
        )

        assert first == second == DonorCandidate(stage_index=42, worker_id=winner, age=0)


class TestSaturationDonorPhaseCOrchestration:
    """Exercise Phase C donation through ``SaturationAwareScheduler.autoscale``."""

    def test_successful_donation_removes_donor_and_adds_receiver(self) -> None:
        """A full cluster can rebalance from over-provisioned A to saturated B."""
        scheduler = _scheduler_for_donation()
        state = _problem_state([("A", 3), ("B", 1)])

        solution = _autoscale_with_intents(scheduler, state, {"B": 1})

        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["A-w0"]
        assert len(solution.stages[1].new_workers) == 1
        assert scheduler._last_donation_cycle == {"A": 1}
        assert scheduler._donations_received_this_cycle == {"B": 1}
        assert scheduler._stuck_plan_counters["B"] == 0

    def test_no_eligible_donor_sticks_without_mutating_donor_state(self) -> None:
        """No donor is non-fatal but increments the receiver stuck counter."""
        scheduler = _scheduler_for_donation()
        state = _problem_state([("A", 1), ("B", 3)])

        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning") as warning:
            solution = _autoscale_with_intents(scheduler, state, {"B": 1})

        assert solution.stages[0].deleted_workers == []
        assert solution.stages[1].new_workers == []
        assert scheduler._last_donation_cycle == {}
        assert scheduler._donations_received_this_cycle == {}
        assert scheduler._stuck_plan_counters["B"] == 1
        assert any("cluster placement exhausted" in call.args[0] for call in warning.call_args_list)

    def test_planner_refuses_selected_donor_without_success_state(self) -> None:
        """A stale selected donor emits warnings but does not record a donation."""
        scheduler = _scheduler_for_donation()
        state = _problem_state([("A", 3), ("B", 1)])

        with (
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning",
            ) as warning,
            patch.object(data_structures.AutoscalePlanContext, "try_remove_worker", return_value=False),
        ):
            solution = _autoscale_with_intents(scheduler, state, {"B": 1})

        assert solution.stages[0].deleted_workers == []
        assert solution.stages[1].new_workers == []
        assert scheduler._last_donation_cycle == {}
        assert scheduler._donations_received_this_cycle == {}
        assert scheduler._stuck_plan_counters["B"] == 1
        assert any("planner refused removal" in call.args[0] for call in warning.call_args_list)

    def test_receiver_per_cycle_cap_blocks_second_donation_for_same_intent(self) -> None:
        """A receiver may absorb only the configured number of donations per cycle."""
        scheduler = _scheduler_for_donation(max_per_cycle=1)
        state = _problem_state([("A", 3), ("B", 1)])

        solution = _autoscale_with_intents(scheduler, state, {"B": 2})

        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1
        assert scheduler._donations_received_this_cycle == {"B": 1}
        assert scheduler._stuck_plan_counters["B"] == 1

    def test_donor_cooldown_blocks_repeat_donation_across_cycles(self) -> None:
        """A donor cannot donate again before ``min_interval`` cycles elapse."""
        scheduler = _scheduler_for_donation(min_interval_cycles=30)
        first_state = _problem_state([("A", 3), ("B", 1)])
        second_state = _problem_state([("A", 2), ("B", 2)])

        first_solution = _autoscale_with_intents(scheduler, first_state, {"B": 1})
        second_solution = _autoscale_with_intents(scheduler, second_state, {"B": 1})

        assert len(first_solution.stages[0].deleted_workers) == 1
        assert len(first_solution.stages[1].new_workers) == 1
        assert second_solution.stages[0].deleted_workers == []
        assert second_solution.stages[1].new_workers == []
        assert scheduler._last_donation_cycle == {"A": 1}
        assert scheduler._donations_received_this_cycle == {}
        assert scheduler._stuck_plan_counters["B"] == 1

    def test_setup_resets_cross_stage_donor_state(self) -> None:
        """``setup`` clears donation history and the per-cycle receiver cap."""
        scheduler = _scheduler_for_donation()
        _autoscale_with_intents(scheduler, _problem_state([("A", 3), ("B", 1)]), {"B": 1})

        scheduler.setup(_problem(["A", "B"]))

        assert scheduler._cycle_counter == 0
        assert scheduler._last_donation_cycle == {}
        assert scheduler._donations_received_this_cycle == {}
        assert scheduler._stuck_plan_counters == {}

    def test_post_donation_retry_failure_does_not_record_success_cooldowns(self) -> None:
        """A failed post-donation retry must not record success-only cooldown state."""
        scheduler = _scheduler_for_donation()
        state = _problem_state([("A", 3), ("B", 1)])

        with patch.object(data_structures.AutoscalePlanContext, "try_add_worker", return_value=None):
            _autoscale_with_intents(scheduler, state, {"B": 1})

        assert scheduler._last_donation_cycle == {}
        assert scheduler._donations_received_this_cycle == {}

    def test_post_donation_retry_failure_still_emits_donor_removal(self) -> None:
        """Retry failure leaves donor cooldown clean while still surfacing the donor removal."""
        scheduler = _scheduler_for_donation()
        state = _problem_state([("A", 3), ("B", 1)])

        with patch.object(data_structures.AutoscalePlanContext, "try_add_worker", return_value=None):
            solution = _autoscale_with_intents(scheduler, state, {"B": 1})

        assert len(solution.stages[0].deleted_workers) == 1
        assert solution.stages[1].new_workers == []
        assert scheduler._last_donation_cycle == {}
        assert scheduler._donations_received_this_cycle == {}
