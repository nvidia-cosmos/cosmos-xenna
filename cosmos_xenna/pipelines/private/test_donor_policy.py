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

"""Contract tests for ``donor.FloorPolicy`` and ``donor.SaturationPolicy``.

The tests exercise the five strategy methods on each policy:
``is_enabled``, ``filter_eligible_donors``, ``candidate_pool``,
``evaluate_gate`` (FloorPolicy returns ``None``;
SaturationPolicy delegates to the economic gate),
``on_commit`` (FloorPolicy no-op; SaturationPolicy advances
``last_donation_cycle``).
"""

from collections.abc import Mapping

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.donor.economic_gate import EconomicGate
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.donor.policy import FloorPolicy, SaturationPolicy
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorPlan, DonorWorker
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

    Most policy tests focus on eligibility filters and the
    candidate pool rather than the gate output, so the default
    ``SaturationAwareConfig`` defaults suffice. Tests that need a
    specific threshold value pass a customised config.

    """
    return SaturationPolicy(gate=EconomicGate(config=config if config is not None else SaturationAwareConfig()))


def _state(
    name: str,
    *,
    classifier_state: StageState = StageState.OVER_PROVISIONED,
    classifier_streak: int = 100,
    growth_mode: GrowthMode = GrowthMode.TRACKING,
    classifier_signal_noise_ewma: float | None = 0.0,
) -> StageRuntimeState:
    return StageRuntimeState(
        stage_name=name,
        classifier=ClassifierState(
            state=classifier_state, streak=classifier_streak, signal_noise_ewma=classifier_signal_noise_ewma
        ),
        growth=GrowthState(mode=growth_mode),
    )


def _config(**overrides: object) -> SaturationAwareConfig:
    """Build a config with permissive defaults for policy filter tests.

    ``cross_stage_donor_anti_flap_cycles`` defaults to 30 so that
    the cluster guardrail dominates the per-stage default of 30
    cycles for ``over_provisioned_streak_min_cycles``; the validator
    in ``SaturationAwareConfig`` rejects anything lower.

    """
    base: dict[str, object] = {
        "enable_cross_stage_donor": True,
        "cross_stage_donor_anti_flap_cycles": 30,
        "cross_stage_donor_require_over_provisioned": True,
        "cross_stage_donor_exclude_hold_state": True,
        "cross_stage_donor_trust_streak_cap": 100,
        "cross_stage_donor_min_trust": 0.0,
        "donor_must_be_strictly_upstream": False,
    }
    base.update(overrides)
    return SaturationAwareConfig(**base)  # type: ignore[arg-type]


def _make_context(
    *,
    config: SaturationAwareConfig | None = None,
    stage_states: Mapping[str, StageRuntimeState] | None = None,
    stage_configs: Mapping[str, SaturationAwareStageConfig] | None = None,
    worker_ids_by_stage: tuple[tuple[str, ...], ...] = (
        ("A-w0", "A-w1", "A-w2"),
        ("B-w0",),
        ("C-w0", "C-w1"),
    ),
    worker_ages: Mapping[str, int] | None = None,
    stage_floors: Mapping[int, int] | None = None,
    donor_warmup_exclusions: frozenset[str] = frozenset(),
    cycle_counter: int = 100,
    last_donation_cycle: dict[str, int] | None = None,
) -> DonorPlanningContext:
    stage_names = ("A", "B", "C")
    states = stage_states if stage_states is not None else {name: _state(name) for name in stage_names}
    configs = (
        stage_configs if stage_configs is not None else {name: SaturationAwareStageConfig() for name in stage_names}
    )
    ages: Mapping[str, int] = (
        worker_ages
        if worker_ages is not None
        else {wid: i for stage_workers in worker_ids_by_stage for i, wid in enumerate(stage_workers)}
    )
    floors: Mapping[int, int] = stage_floors if stage_floors is not None else {0: 1, 1: 1, 2: 1}
    return DonorPlanningContext(
        stage_names=stage_names,
        stage_configs=configs,
        stage_states=states,
        stage_floors=floors,
        worker_ids_by_stage=worker_ids_by_stage,
        worker_ages=ages,
        worker_node_map={},
        d_k_now={"A": 0.5, "B": 1.0, "C": 0.25},
        effective_capacities={"A": 3, "B": 1, "C": 2},
        s_k_ewma={"A": 0.1, "B": 0.2, "C": 0.05},
        slots_per_worker_by_stage={"A": 1, "B": 1, "C": 1},
        donor_warmup_exclusions=donor_warmup_exclusions,
        cycle_counter=cycle_counter,
        last_donation_cycle=last_donation_cycle if last_donation_cycle is not None else {},
        config=config if config is not None else _config(),
    )


def _receiver(stage_index: int, stage_name: str) -> StageCycleView:
    return StageCycleView(
        stage_index=stage_index,
        stage_name=stage_name,
        runtime_state=_state(stage_name),
        current_workers=1,
    )


class TestFloorPolicyEnable:
    """``FloorPolicy.is_enabled`` is unconditionally True."""

    def test_is_enabled_independent_of_config(self) -> None:
        policy = FloorPolicy()
        context = _make_context(config=_config(enable_cross_stage_donor=False))
        assert policy.is_enabled(context) is True


class TestFloorPolicyEligibility:
    """``FloorPolicy.filter_eligible_donors`` prefers upstream, falls back to downstream."""

    def test_upstream_preferred_over_downstream(self) -> None:
        policy = FloorPolicy()
        context = _make_context()
        # Receiver is the middle stage; A is upstream, C is downstream.
        result = policy.filter_eligible_donors(context, _receiver(1, "B"))
        assert result == [0, 2]  # upstream A first, downstream C after

    def test_floor_eligibility_excludes_donor_at_floor(self) -> None:
        policy = FloorPolicy()
        # B only has 1 worker; floor=1 -> cannot donate.
        context = _make_context(stage_floors={0: 1, 1: 1, 2: 1})
        result = policy.filter_eligible_donors(context, _receiver(2, "C"))
        assert 1 not in result  # B is at floor

    def test_self_donation_excluded(self) -> None:
        policy = FloorPolicy()
        context = _make_context()
        result = policy.filter_eligible_donors(context, _receiver(0, "A"))
        assert 0 not in result


class TestFloorPolicyCandidatePool:
    """``FloorPolicy.candidate_pool`` ignores warmup exclusions."""

    def test_warmup_exclusions_not_applied(self) -> None:
        policy = FloorPolicy()
        # A-w0 is warmup-protected; floor mode must still include it.
        context = _make_context(donor_warmup_exclusions=frozenset({"A-w0"}))
        candidates = policy.candidate_pool([0], context)
        assert "A-w0" in {c.worker_id for c in candidates}

    def test_candidates_sorted_age_ascending(self) -> None:
        policy = FloorPolicy()
        context = _make_context(worker_ages={"A-w0": 5, "A-w1": 2, "A-w2": 8})
        candidates = policy.candidate_pool([0], context)
        assert [c.worker_id for c in candidates] == ["A-w1", "A-w0", "A-w2"]


class TestFloorPolicyGateAndCommit:
    """FloorPolicy short-circuits evaluate_gate and on_commit."""

    def test_evaluate_gate_returns_none(self) -> None:
        policy = FloorPolicy()
        context = _make_context()
        plan = DonorPlan(removals=(DonorWorker(0, "A-w0", 0),), receiver_stage_index=1)
        assert policy.evaluate_gate(plan, context, _receiver(1, "B"), receiver_intent=1) is None

    def test_on_commit_does_not_advance_ledger(self) -> None:
        policy = FloorPolicy()
        ledger: dict[str, int] = {}
        context = _make_context(cycle_counter=42, last_donation_cycle=ledger)
        plan = DonorPlan(removals=(DonorWorker(0, "A-w0", 0),), receiver_stage_index=1)
        policy.on_commit(plan, context)
        assert ledger == {}


class TestSaturationPolicyEnable:
    """``SaturationPolicy.is_enabled`` honours the master toggle."""

    def test_master_toggle_off(self) -> None:
        policy = _saturation_policy()
        context = _make_context(config=_config(enable_cross_stage_donor=False))
        assert policy.is_enabled(context) is False

    def test_master_toggle_on(self) -> None:
        policy = _saturation_policy()
        context = _make_context(config=_config(enable_cross_stage_donor=True))
        assert policy.is_enabled(context) is True


class TestSaturationPolicyEligibilityFilters:
    """Each filter gate is checked in isolation."""

    def test_receiver_anti_flap_blocks_donation(self) -> None:
        policy = _saturation_policy()
        # B was a donor at cycle 95; cooldown 30 blocks receiving through cycle 124.
        context = _make_context(
            config=_config(cross_stage_donor_anti_flap_cycles=30),
            cycle_counter=100,
            last_donation_cycle={"B": 95},
        )
        assert policy.filter_eligible_donors(context, _receiver(1, "B")) == []

    def test_classifier_streak_gate_excludes_short_streak(self) -> None:
        policy = _saturation_policy()
        stage_cfg = SaturationAwareStageConfig(over_provisioned_streak_min_cycles=10)
        states = {
            "A": _state("A", classifier_streak=3),  # too short
            "B": _state("B"),
            "C": _state("C"),
        }
        context = _make_context(
            stage_states=states,
            stage_configs={"A": stage_cfg, "B": stage_cfg, "C": stage_cfg},
        )
        result = policy.filter_eligible_donors(context, _receiver(1, "B"))
        assert 0 not in result

    def test_hold_growth_mode_excluded(self) -> None:
        policy = _saturation_policy()
        states = {
            "A": _state("A", growth_mode=GrowthMode.HOLD),
            "B": _state("B"),
            "C": _state("C"),
        }
        context = _make_context(stage_states=states)
        result = policy.filter_eligible_donors(context, _receiver(1, "B"))
        assert 0 not in result

    def test_strict_upstream_excludes_downstream(self) -> None:
        policy = _saturation_policy()
        context = _make_context(config=_config(donor_must_be_strictly_upstream=True))
        result = policy.filter_eligible_donors(context, _receiver(1, "B"))
        assert all(idx < 1 for idx in result)


class TestSaturationPolicyCandidatePool:
    """``SaturationPolicy.candidate_pool`` enforces warmup exclusion."""

    def test_warmup_exclusions_applied(self) -> None:
        policy = _saturation_policy()
        context = _make_context(donor_warmup_exclusions=frozenset({"A-w0"}))
        candidates = policy.candidate_pool([0], context)
        assert "A-w0" not in {c.worker_id for c in candidates}
        assert {"A-w1", "A-w2"} <= {c.worker_id for c in candidates}


class TestSaturationPolicyOnCommit:
    """``SaturationPolicy.on_commit`` advances the ledger per donor stage."""

    def test_single_donor_advances_ledger(self) -> None:
        policy = _saturation_policy()
        ledger: dict[str, int] = {}
        context = _make_context(cycle_counter=42, last_donation_cycle=ledger)
        plan = DonorPlan(removals=(DonorWorker(0, "A-w0", 0),), receiver_stage_index=1)
        policy.on_commit(plan, context)
        assert ledger == {"A": 42}

    def test_multi_donor_advances_each_distinct_stage_once(self) -> None:
        policy = _saturation_policy()
        ledger: dict[str, int] = {}
        context = _make_context(cycle_counter=7, last_donation_cycle=ledger)
        plan = DonorPlan(
            removals=(
                DonorWorker(0, "A-w0", 0),
                DonorWorker(0, "A-w1", 1),  # same donor stage; one ledger entry
                DonorWorker(2, "C-w0", 0),
            ),
            receiver_stage_index=1,
        )
        policy.on_commit(plan, context)
        assert ledger == {"A": 7, "C": 7}


@pytest.mark.parametrize("policy_factory", [FloorPolicy, _saturation_policy])
class TestSharedSelfExclusion:
    """Both policies refuse to nominate the receiver as a donor."""

    def test_receiver_never_in_eligible_pool(self, policy_factory: type) -> None:
        policy = policy_factory()
        context = _make_context()
        receiver_index = 1
        result = policy.filter_eligible_donors(context, _receiver(receiver_index, "B"))
        assert receiver_index not in result
