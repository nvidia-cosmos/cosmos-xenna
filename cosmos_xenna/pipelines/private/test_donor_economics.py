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

"""Donor-economics helpers and the throughput-first commit gate.

Five concerns ship from this module:

*   ``_donor_cost`` - marginal cost of removing N workers from a donor
    stage; combines slots-empty-ratio EWMA with a streak-bonus discount.
*   ``_receiver_value`` - marginal value of adding N workers to the
    receiver; combines pressure EWMA with a bottleneck-severity term
    and an intent term.
*   ``_signal_trust`` - per-stage Sharpe-style trust metric used by
    the layer-4 pre-filter.
*   ``_compute_post_plan_d_k`` - post-plan ``D_k`` simulator that
    holds ``S_k`` fixed and recomputes effective capacity per affected
    stage.
*   ``_evaluate_economic_gate`` - the throughput-first commit gate
    composer that returns a ``_GateResult`` carrying every metric the
    decision log emits.

Each test pins one contract; the gate's reject-reason values are
asserted via ``RejectReason`` to keep log strings stable.
"""

import math

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.donor import (
    DonorPlan,
    DonorWorker,
    RejectReason,
    _compute_post_plan_d_k,
    _donor_cost,
    _evaluate_economic_gate,
    _format_donor_decision_log,
    _receiver_value,
    _signal_trust,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def _state(
    name: str,
    *,
    classifier: StageState = StageState.OVER_PROVISIONED,
    streak: int = 30,
    slots_empty_ewma: float | None = 0.5,
    pressure_ewma: float | None = 0.0,
    noise_ewma: float | None = None,
) -> _StageRuntimeState:
    """Compact factory for runtime states used in the economics tests."""
    return _StageRuntimeState(
        stage_name=name,
        classifier_state=classifier,
        classifier_streak=streak,
        growth_mode=GrowthMode.TRACKING,
        growth_streak=10,
        slots_empty_ratio_ewma=slots_empty_ewma,
        pressure_ewma=pressure_ewma,
        classifier_signal_noise_ewma=noise_ewma,
    )


def _config(**overrides: object) -> SaturationAwareConfig:
    """``SaturationAwareConfig`` with overridable defaults for gate tests."""
    return SaturationAwareConfig(**overrides)  # type: ignore[arg-type]


class TestDonorCost:
    """Marginal cost combines slot-empty linear term with the streak discount."""

    def test_cost_scales_linearly_with_num_workers(self) -> None:
        """``cost = slots_empty_ratio_ewma * num_workers - streak_bonus * min(streak, cap)``."""
        stage = _state("A", slots_empty_ewma=0.4, streak=10)
        cost_one = _donor_cost(stage, num_workers=1, streak_bonus=0.05, streak_cap=60)
        cost_two = _donor_cost(stage, num_workers=2, streak_bonus=0.05, streak_cap=60)

        # Linear delta is the slot-empty term: 0.4 * (2 - 1) = 0.4.
        assert math.isclose(cost_two - cost_one, 0.4, rel_tol=1e-9)

    def test_streak_bonus_clamped_at_cap(self) -> None:
        """``min(streak, cap)`` prevents a runaway streak from dominating cost."""
        stage = _state("A", slots_empty_ewma=0.0, streak=10_000)
        cost = _donor_cost(stage, num_workers=1, streak_bonus=0.05, streak_cap=60)

        # base = 0; bonus = 0.05 * 60 = 3.0 -> cost = -3.0.
        assert math.isclose(cost, -3.0, rel_tol=1e-9)

    def test_cold_start_slots_empty_collapses_base_to_zero(self) -> None:
        """``slots_empty_ratio_ewma=None`` -> base term is 0 (streak bonus alone shapes)."""
        stage = _state("A", slots_empty_ewma=None, streak=10)
        cost = _donor_cost(stage, num_workers=5, streak_bonus=0.05, streak_cap=60)

        # base = None -> 0; bonus = 0.05 * 10 = 0.5; cost = -0.5.
        assert math.isclose(cost, -0.5, rel_tol=1e-9)


class TestReceiverValue:
    """Marginal value combines pressure linear term with bottleneck and intent terms."""

    def test_value_scales_linearly_with_num_workers(self) -> None:
        """``value = pressure_ewma * num_workers + bottleneck + intent``."""
        stage = _state("B", pressure_ewma=0.7)
        value = _receiver_value(
            stage,
            num_workers=3,
            d_k=0.0,
            median_d_k=0.0,
            intent=0,
            bottleneck_weight=1.0,
            intent_weight=0.5,
        )
        # bottleneck_term = 1.0 * (0 - 0) = 0; intent_term = 0.5 * 0 = 0.
        assert math.isclose(value, 2.1, rel_tol=1e-9)

    def test_bottleneck_term_uses_d_k_minus_median(self) -> None:
        """Severe bottleneck stages (d_k >> median) get higher value."""
        stage = _state("B", pressure_ewma=0.0)
        value = _receiver_value(
            stage,
            num_workers=1,
            d_k=5.0,
            median_d_k=1.0,
            intent=0,
            bottleneck_weight=2.0,
            intent_weight=0.0,
        )
        # bottleneck_term = 2.0 * (5 - 1) = 8.0.
        assert math.isclose(value, 8.0, rel_tol=1e-9)

    def test_non_finite_d_k_or_median_collapses_bottleneck_term(self) -> None:
        """Cold-start cycles (NaN d_k or median) should not poison the gate."""
        stage = _state("B", pressure_ewma=0.5)
        value = _receiver_value(
            stage,
            num_workers=1,
            d_k=math.nan,
            median_d_k=1.0,
            intent=0,
            bottleneck_weight=10.0,
            intent_weight=0.0,
        )
        # NaN d_k -> bottleneck_term collapses to 0.
        assert math.isclose(value, 0.5, rel_tol=1e-9)


class TestSignalTrust:
    """Trust = clamped streak / (1 + noise EWMA); noisy classifiers earn less trust."""

    def test_clean_signal_trust_equals_clamped_streak(self) -> None:
        """``classifier_signal_noise_ewma=None`` -> denominator = 1.0 -> trust = streak."""
        stage = _state("A", streak=30, noise_ewma=None)
        trust = _signal_trust(stage, trust_streak_cap=60)

        assert trust == 30.0

    def test_streak_clamped_at_trust_streak_cap(self) -> None:
        """``min(streak, cap)`` mirrors the donor-cost streak cap independently."""
        stage = _state("A", streak=10_000, noise_ewma=None)
        trust = _signal_trust(stage, trust_streak_cap=60)

        assert trust == 60.0

    def test_noise_attenuates_trust(self) -> None:
        """Larger noise EWMA reduces trust proportionally."""
        stage = _state("A", streak=30, noise_ewma=4.0)
        trust = _signal_trust(stage, trust_streak_cap=60)

        # min(30, 60) / (1 + 4) = 6.0.
        assert math.isclose(trust, 6.0, rel_tol=1e-9)


class TestComputePostPlanDk:
    """Post-plan simulator subtracts/adds capacity per affected stage."""

    def test_donor_loses_one_channel_receiver_gains_per_removal(self) -> None:
        """Capacity bookkeeping is per-removal: 2 removals -> +2 receiver, -2 across donors."""
        plan = DonorPlan(
            removals=(
                DonorWorker(stage_index=0, worker_id="a-w0", age=1),
                DonorWorker(stage_index=0, worker_id="a-w1", age=2),
            ),
            receiver_stage_index=1,
        )
        d_k_now = {"A": 1.0, "B": 4.0}
        effective_capacities = {"A": 4, "B": 1}
        s_k_ewma = {"A": 4.0, "B": 4.0}

        d_after = _compute_post_plan_d_k(
            d_k_now=d_k_now,
            plan=plan,
            stage_names=["A", "B"],
            effective_capacities=effective_capacities,
            s_k_ewma=s_k_ewma,
        )

        # A capacity: 4 - 2 = 2 -> S/c = 4/2 = 2.0.
        # B capacity: 1 + 2 = 3 -> S/c = 4/3 = 1.333...
        assert math.isclose(d_after["A"], 2.0, rel_tol=1e-9)
        assert math.isclose(d_after["B"], 4.0 / 3.0, rel_tol=1e-9)

    def test_output_keys_match_input_keys(self) -> None:
        """The result preserves ``d_k_now``'s keys exactly (no extras, no drops)."""
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=0),),
            receiver_stage_index=1,
        )
        d_k_now = {"A": 1.0, "B": 1.0, "C": 1.0}

        d_after = _compute_post_plan_d_k(
            d_k_now=d_k_now,
            plan=plan,
            stage_names=["A", "B", "C"],
            effective_capacities={"A": 2, "B": 1, "C": 1},
            s_k_ewma={"A": 2.0, "B": 1.0, "C": 1.0},
        )

        assert set(d_after) == {"A", "B", "C"}


class TestEconomicGateAccept:
    """Happy path: a feasible plan with healthy throughput passes every check."""

    def test_accepted_plan_when_throughput_improves(self) -> None:
        """Donor over-capacity, receiver under-capacity -> accept."""
        cfg = _config()
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.6, streak=60),
            "B": _state("B", classifier=StageState.SATURATED, pressure_ewma=2.0, streak=60),
        }
        d_k_now = {"A": 0.5, "B": 4.0}
        effective_capacities = {"A": 4, "B": 1}
        s_k_ewma = {"A": 2.0, "B": 4.0}

        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now=d_k_now,
            effective_capacities=effective_capacities,
            s_k_ewma=s_k_ewma,
            config=cfg,
        )

        assert result.accepted is True
        assert result.reject_reason is None


class TestEconomicGateRejections:
    """Each gate produces its own ``RejectReason`` value."""

    def test_low_signal_trust_rejects(self) -> None:
        """A donor with streak << min_trust never reaches the spread check."""
        cfg = _config(cross_stage_donor_min_trust=10.0)
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", streak=1),  # trust = 1.0; below min_trust 10.0.
            "B": _state("B", classifier=StageState.SATURATED),
        }

        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
            config=cfg,
        )

        assert result.reject_reason is RejectReason.SIGNAL_TRUST
        assert result.accepted is False

    def test_spread_below_threshold_rejects(self) -> None:
        """A weak receiver pressure drops spread below ``spread_threshold``.

        The donor's streak is bumped to a value that clears the trust
        gate so the test exercises the spread gate in isolation; both
        stages contribute zero economic terms (cost = 0, value ~ 0)
        so spread sits well below the configured 10.0 threshold.
        """
        cfg = _config(cross_stage_donor_spread_threshold=10.0)
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            # streak=99 clears trust; slots_empty=0 and streak_bonus default
            # 0.05 -> cost = 0 - 0.05 * 60 = -3.0.
            "A": _state("A", slots_empty_ewma=0.0, streak=99),
            # pressure=0 -> value = 0; spread = 0 - (-3.0) = 3.0 < 10.0.
            "B": _state("B", pressure_ewma=0.0, streak=99),
        }

        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=0,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
            config=cfg,
        )

        assert result.reject_reason is RejectReason.SPREAD_BELOW_THRESHOLD

    def test_throughput_regression_rejects(self) -> None:
        """Removing the donor's only channel doubles its post-plan ``D_k``."""
        cfg = _config()
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        # Donor "A" has only 2 channels for 4s of S_k -> D_k=2; receiver "B"
        # has 4 channels for 4s -> D_k=1. Donating leaves A=4s/1ch=4, B=4s/5ch=0.8;
        # max_d goes 2 -> 4, throughput drops 0.5 -> 0.25.
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", pressure_ewma=2.0, streak=60),
        }

        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": 2.0, "B": 1.0},
            effective_capacities={"A": 2, "B": 4},
            s_k_ewma={"A": 4.0, "B": 4.0},
            config=cfg,
        )

        assert result.reject_reason is RejectReason.THROUGHPUT_REGRESSION

    def test_donor_flip_guard_rejects(self) -> None:
        """A small donor flipping above the pre-plan max bottleneck fires the guard."""
        cfg = _config(cross_stage_donor_throughput_tolerance=10.0)
        # Throughput tolerance is wide so the throughput gate does not fire
        # first; donor-flip guard still must fire on the donor's post-plan D_k.
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", pressure_ewma=2.0, streak=60),
        }
        # A: 1 channel for 0.6s -> D=0.6 pre. After removal: 0 channels -> NaN.
        # Production replaces 0-cap with NaN, so this test uses a 2->1
        # donor flip: D=0.5 pre, D=1.0 post; max_d_before=1.0, donor_flip_tolerance default 0.10.
        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": 0.5, "B": 1.0},
            effective_capacities={"A": 2, "B": 1},
            s_k_ewma={"A": 1.0, "B": 1.0},
            config=cfg,
        )

        # A flips from 0.5 -> 1.0 (1.0/1=1.0); max_d_before is 1.0.
        # 1.0 > 1.0 + 0.10 = False; need a flip ABOVE max_d_before + tolerance.
        # With s_k_ewma="A"=2.0 instead the flip would be 2.0 > 1.10 -> trigger.
        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": 1.0, "B": 1.0},
            effective_capacities={"A": 2, "B": 1},
            s_k_ewma={"A": 2.0, "B": 1.0},
            config=cfg,
        )

        assert result.reject_reason is RejectReason.DONOR_FLIP_GUARD

    def test_balance_regression_rejects_only_when_throughput_tied(self) -> None:
        """Balance regression fires only when throughput is tied within tolerance."""
        cfg = _config(
            cross_stage_donor_throughput_tolerance=100.0,
            cross_stage_donor_donor_flip_tolerance=100.0,
            cross_stage_donor_balance_tolerance=0.0,
        )
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", pressure_ewma=2.0, streak=60),
        }
        # Pick s_k / capacities so throughput is unchanged but balance worsens:
        # before: A=1, B=1 -> max=min=1, ratio=1, balance=1.0
        # after donating from A (cap 4->3) to B (cap 1->2):
        #   A_after = 1*4/3 = 1.333, B_after = 1*1/2 = 0.5
        #   max_after=1.333, min_after=0.5 -> ratio=2.667, balance=1/2.667=0.375
        # max_d before = 1.0; max_d after = 1.333; throughput drops 1.0 -> 0.75.
        # That's a throughput regression. Need throughput preserved.
        # Alternative: design so the receiver's post-plan max D drops EQUAL to
        # the donor's post-plan rise. Symmetric S_k = 4, A goes 4->3 (D=1.0->1.333),
        # B goes 1->2 (D=4->2). max before = 4, max after = 2 -> throughput RISES.
        # Throughput up means abs(after - before) is large -> not tied -> balance gate skipped.
        #
        # To trigger balance gate exclusively: need throughput identical pre/post
        # while balance shifts. Symmetric capacity swap: A(4->3), B(3->4); s_k same.
        # max stays at S/3 in both cycles -> tied. min shifts: A_after=S/3, B_after=S/4
        # -> ratio = (S/3)/(S/4) = 4/3, balance = 1/(4/3) = 0.75.
        # min before: A=S/4, B=S/3 -> ratio = (S/3)/(S/4) = 4/3, balance = 0.75.
        # That ties -> not useful.
        #
        # For a real balance regression: need the swap to widen the spread.
        # Use 3 stages where the donor stage and receiver are NOT the bottleneck.
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states["C"] = _state("C", pressure_ewma=0.0, streak=60)

        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B", "C"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": 1.0, "B": 1.0, "C": 5.0},
            effective_capacities={"A": 4, "B": 4, "C": 1},
            s_k_ewma={"A": 4.0, "B": 4.0, "C": 5.0},
            config=cfg,
        )

        # max_d_before = 5 (C). After: A: 4/3 = 1.333, B: 4/5 = 0.8, C: 5.
        # max_d stays 5; throughput tied.
        # min before: 1; min after: 0.8. ratio_before = 5; ratio_after = 5/0.8=6.25.
        # balance_before = 1/5 = 0.2; balance_after = 1/6.25 = 0.16.
        # 0.16 < 0.2 - 0.0 -> regression -> reject.
        assert result.reject_reason is RejectReason.BALANCE_REGRESSION


class TestEconomicGateColdStart:
    """Cold-start cycles (empty / NaN ``d_k_now``) skip throughput / balance checks."""

    def test_empty_d_k_now_does_not_block_donation(self) -> None:
        """NaN comparisons short-circuit; only spread / signal-trust apply."""
        cfg = _config()
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", pressure_ewma=2.0, streak=60),
        }

        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
            config=cfg,
        )

        assert result.accepted is True


class TestRejectReasonEnumeration:
    """The ``RejectReason`` enum covers every advertised reason."""

    def test_all_reasons_round_trip_through_value(self) -> None:
        """Every ``RejectReason`` is a string-valued member with stable wire format."""
        # Snapshot the public reasons as the contract for log consumers.
        reasons = {member.value for member in RejectReason}
        assert reasons == {
            "master_toggle_off",
            "receiver_anti_flap",
            "no_candidates",
            "signal_trust",
            "resource_fit",
            "spread_below_threshold",
            "throughput_regression",
            "donor_flip_guard",
            "balance_regression",
        }


class TestGateResultStability:
    """``_GateResult`` carries every metric the decision log needs."""

    def test_gate_result_is_immutable_value(self) -> None:
        """``attrs.frozen`` rejects post-construction mutation."""
        cfg = _config()
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", streak=60),
            "B": _state("B", classifier=StageState.SATURATED),
        }
        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
            config=cfg,
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            result.accepted = False  # type: ignore[misc]


class TestDecisionLogSchema:
    """The structured decision-log line carries every advertised field in the same order."""

    def test_commit_line_starts_with_commit_marker(self) -> None:
        """INFO commit log starts with ``[scheduler] donor decision (commit):``."""
        cfg = _config()
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", classifier=StageState.SATURATED, pressure_ewma=2.0, streak=60),
        }
        gate = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
            config=cfg,
        )

        line = _format_donor_decision_log(
            receiver_name="B",
            receiver_state=states["B"],
            receiver_d_k=math.nan,
            receiver_intent=1,
            capacity_before=1,
            capacity_after=2,
            plan=plan,
            stage_names=["A", "B"],
            gate_result=gate,
            spread_threshold=cfg.cross_stage_donor_spread_threshold,
            reject_reason=None,
        )

        assert line.startswith("[scheduler] donor decision (commit):")

    def test_reject_line_starts_with_reject_marker(self) -> None:
        """DEBUG reject log starts with ``[scheduler] donor decision (reject):``."""
        line = _format_donor_decision_log(
            receiver_name="B",
            receiver_state=None,
            receiver_d_k=math.nan,
            receiver_intent=0,
            capacity_before=0,
            capacity_after=0,
            plan=None,
            stage_names=["A", "B"],
            gate_result=None,
            spread_threshold=0.5,
            reject_reason=RejectReason.NO_CANDIDATES.value,
        )

        assert line.startswith("[scheduler] donor decision (reject):")

    def test_commit_line_carries_every_documented_field(self) -> None:
        """The commit log carries every field the operator schema specifies."""
        cfg = _config()
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", classifier=StageState.SATURATED, pressure_ewma=2.0, streak=60),
        }
        gate = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={},
            effective_capacities={},
            s_k_ewma={},
            config=cfg,
        )

        line = _format_donor_decision_log(
            receiver_name="B",
            receiver_state=states["B"],
            receiver_d_k=4.0,
            receiver_intent=1,
            capacity_before=1,
            capacity_after=2,
            plan=plan,
            stage_names=["A", "B"],
            gate_result=gate,
            spread_threshold=cfg.cross_stage_donor_spread_threshold,
            reject_reason=None,
        )

        # Each field name MUST appear exactly as the operator schema documents.
        for field in (
            "receiver=",
            "classifier=",
            "pressure_ewma=",
            "slots_empty=",
            "D_k=",
            "capacity_before=",
            "capacity_after=",
            "intent=",
            "receiver_value=",
            "donor_plan=",
            "donor_cost=",
            "spread=",
            "spread_threshold=",
            "signal_trust=",
            "throughput_before=",
            "throughput_after=",
            "max_d_before=",
            "max_d_after=",
            "balance_score_before=",
            "balance_score_after=",
        ):
            assert field in line, f"missing field {field!r} in decision log: {line}"

    def test_reject_line_includes_reject_reason_and_placement_reject_reason(self) -> None:
        """DEBUG reject log carries both ``reject_reason`` and ``placement_reject_reason``."""
        line = _format_donor_decision_log(
            receiver_name="B",
            receiver_state=None,
            receiver_d_k=math.nan,
            receiver_intent=0,
            capacity_before=0,
            capacity_after=0,
            plan=None,
            stage_names=["A", "B"],
            gate_result=None,
            spread_threshold=0.5,
            reject_reason=RejectReason.RESOURCE_FIT.value,
            placement_reject_reason="no_placement",
        )

        assert "reject_reason='resource_fit'" in line
        assert "placement_reject_reason='no_placement'" in line

    def test_early_return_reject_line_uses_nan_for_unavailable_metrics(self) -> None:
        """Early-return paths (no plan / gate) render ``nan`` for missing floats."""
        line = _format_donor_decision_log(
            receiver_name="B",
            receiver_state=None,
            receiver_d_k=math.nan,
            receiver_intent=0,
            capacity_before=0,
            capacity_after=0,
            plan=None,
            stage_names=["A", "B"],
            gate_result=None,
            spread_threshold=0.5,
            reject_reason=RejectReason.MASTER_TOGGLE_OFF.value,
        )

        # NaN values carry the textual marker ``nan`` for log consumers
        # that key off it; donor_plan / signal_trust render as empty lists.
        assert "spread=nan" in line
        assert "throughput_before=nan" in line
        assert "donor_plan=[]" in line
        assert "signal_trust=[]" in line


class TestRejectionPathEmitsDebug:
    """Each layered rejection emits exactly one DEBUG line via ``_emit_reject``."""

    def test_master_toggle_off_emits_one_debug_line(self, caplog: pytest.LogCaptureFixture) -> None:
        """Disabling ``enable_cross_stage_donor`` emits one ``master_toggle_off`` DEBUG line."""
        from unittest.mock import patch

        from cosmos_xenna.pipelines.private.scheduling_py.donor import find_saturation_donor

        cfg = _config(enable_cross_stage_donor=False)
        states = {
            "A": _state("A", streak=60),
            "B": _state("B", classifier=StageState.SATURATED),
        }
        emitted: list[str] = []

        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.donor.logger.debug",
            lambda msg: emitted.append(msg),
        ):
            decision = find_saturation_donor(
                receiver_stage_index=1,
                receiver_stage_name="B",
                stage_names=["A", "B"],
                stage_floors={0: 1, 1: 1},
                worker_ids_by_stage=[["a-w0"], ["b-w0"]],
                worker_ages={"a-w0": 0, "b-w0": 0},
                worker_nodes={},
                stage_states=states,
                config=cfg,
                stage_configs={},
                cycle=0,
                last_donation_cycle={},
                ctx=None,  # type: ignore[arg-type]
                receiver_intent=1,
                d_k_now={},
                effective_capacities={},
                s_k_ewma={},
            )

        assert decision is None
        assert len(emitted) == 1
        assert "reject_reason='master_toggle_off'" in emitted[0]
