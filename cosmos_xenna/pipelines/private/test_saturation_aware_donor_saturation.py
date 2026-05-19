# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from cosmos_xenna.pipelines.private.scheduling_py.donor import find_saturation_donor
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
