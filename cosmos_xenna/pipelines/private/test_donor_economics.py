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


"""Donor-economics helpers and the throughput-first commit gate.

Five concerns ship from this module:

*   ``_donor_cost`` - marginal cost of removing N workers from a donor
    stage; combines slots-empty-ratio EWMA with a streak-bonus discount.
*   ``_receiver_value`` - marginal value of adding N workers to the
    receiver; combines pressure EWMA with a bottleneck-severity term
    and an intent term.
*   ``signal_trust`` - per-stage Sharpe-style trust metric used by
    the layer-4 pre-filter.
*   ``_compute_post_plan_d_k`` - post-plan ``D_k`` simulator that
    holds ``S_k`` fixed and recomputes effective capacity per affected
    stage.
*   ``EconomicGate.evaluate`` - the throughput-first commit gate
    composer that returns a ``GateResult`` carrying every metric the
    decision log emits.

Each test pins one contract; the gate's reject-reason values are
asserted via ``RejectReason`` so the structured ``reject_reason``
field bound by ``logger.bind(...).debug(...)`` stays stable across
emit sites.
"""

import math
from collections.abc import Mapping

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.donor.economic_gate import (
    EconomicGate,
    _compute_post_plan_d_k,
    _donor_cost,
    _receiver_value,
    signal_trust,
)
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import (
    DonorPlan,
    DonorWorker,
    GateResult,
    RejectReason,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    ClassifierState,
    GrowthMode,
    GrowthState,
    PressureState,
    StageRuntimeState,
    StageState,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def _state(
    name: str,
    *,
    classifier: StageState = StageState.OVER_PROVISIONED,
    streak: int = 30,
    slots_empty_ewma: float | None = 0.5,
    pressure_ewma: float | None = 0.0,
    noise_ewma: float | None = None,
) -> StageRuntimeState:
    """Compact factory for runtime states used in the economics tests."""
    return StageRuntimeState(
        stage_name=name,
        classifier=ClassifierState(
            state=classifier, streak=streak, slots_empty_ratio_ewma=slots_empty_ewma, signal_noise_ewma=noise_ewma
        ),
        growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
        pressure=PressureState(ewma=pressure_ewma),
    )


def _config(**overrides: object) -> SaturationAwareConfig:
    """``SaturationAwareConfig`` with overridable defaults for gate tests."""
    return SaturationAwareConfig(**overrides)  # type: ignore[arg-type]


def _evaluate_economic_gate(
    *,
    plan: DonorPlan,
    stage_names: list[str],
    stage_states: dict[str, StageRuntimeState],
    receiver_intent: int,
    d_k_now: Mapping[str, float],
    effective_capacities: Mapping[str, int],
    s_k_ewma: Mapping[str, float],
    slots_per_worker_by_stage: Mapping[str, int],
    config: SaturationAwareConfig,
) -> GateResult:
    """Construct an ``EconomicGate`` and dispatch ``evaluate``.

    Tests call this helper once per assertion so each test exercises
    the public class API (``EconomicGate(config=...).evaluate(...)``)
    through a single call site; the production module exports
    ``EconomicGate`` only.

    """
    gate = EconomicGate(config=config)
    return gate.evaluate(
        plan=plan,
        stage_names=stage_names,
        stage_states=stage_states,
        receiver_intent=receiver_intent,
        d_k_now=d_k_now,
        effective_capacities=effective_capacities,
        s_k_ewma=s_k_ewma,
        slots_per_worker_by_stage=slots_per_worker_by_stage,
    )


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
        trust = signal_trust(stage, trust_streak_cap=60)

        assert trust == 30.0

    def test_streak_clamped_at_trust_streak_cap(self) -> None:
        """``min(streak, cap)`` mirrors the donor-cost streak cap independently."""
        stage = _state("A", streak=10_000, noise_ewma=None)
        trust = signal_trust(stage, trust_streak_cap=60)

        assert trust == 60.0

    def test_noise_attenuates_trust(self) -> None:
        """Larger noise EWMA reduces trust proportionally."""
        stage = _state("A", streak=30, noise_ewma=4.0)
        trust = signal_trust(stage, trust_streak_cap=60)

        # min(30, 60) / (1 + 4) = 6.0.
        assert math.isclose(trust, 6.0, rel_tol=1e-9)


class TestComputePostPlanDk:
    """Post-plan simulator subtracts/adds capacity per affected stage."""

    def test_donor_loses_slots_per_worker_per_removal_receiver_gains_once(self) -> None:
        """Capacity bookkeeping respects channel units.

        A donation cycle removes N donors and adds exactly ONE receiver
        worker (regardless of plan size) -- ``phases.phase_c.run`` calls
        ``try_add_worker`` once per ``DonorCoordinator.acquire``. With
        ``slots_per_worker=1`` for both stages, removing 2 donor
        workers drops donor capacity by 2 channels and gains the
        receiver 1 channel.
        """
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
            slots_per_worker_by_stage={"A": 1, "B": 1},
        )

        # A capacity: 4 - 2*slots_A = 4 - 2 = 2 -> S/c = 4/2 = 2.0.
        # B capacity: 1 + slots_B = 1 + 1 = 2 -> S/c = 4/2 = 2.0.
        assert math.isclose(d_after["A"], 2.0, rel_tol=1e-9)
        assert math.isclose(d_after["B"], 2.0, rel_tol=1e-9)

    def test_donor_loss_scales_with_slots_per_worker(self) -> None:
        """A stage with ``slots_per_worker=2`` releases 2 channels per worker removed."""
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        d_k_now = {"A": 1.0, "B": 1.0}
        effective_capacities = {"A": 4, "B": 4}
        s_k_ewma = {"A": 4.0, "B": 4.0}

        d_after = _compute_post_plan_d_k(
            d_k_now=d_k_now,
            plan=plan,
            stage_names=["A", "B"],
            effective_capacities=effective_capacities,
            s_k_ewma=s_k_ewma,
            slots_per_worker_by_stage={"A": 2, "B": 2},
        )

        # A capacity: 4 - 1*slots_A = 4 - 2 = 2 -> S/c = 4/2 = 2.0.
        # B capacity: 4 + slots_B = 4 + 2 = 6 -> S/c = 4/6 = 0.666...
        assert math.isclose(d_after["A"], 2.0, rel_tol=1e-9)
        assert math.isclose(d_after["B"], 4.0 / 6.0, rel_tol=1e-9)

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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
        )

        assert result.reject_reason is RejectReason.DONOR_FLIP_GUARD

    def test_balance_regression_rejects_only_when_throughput_tied(self) -> None:
        """Balance regression fires only when throughput is tied within tolerance.

        Three-stage fixture where the donor (A) and receiver (B) are
        not the cluster bottleneck (C is); the swap leaves ``max_d``
        unchanged at C (throughput tied) but widens ``max/min``
        because A's post-plan ``D_k`` shrinks below B's pre-plan
        value. Throughput / donor-flip tolerances are wide so only
        the balance gate can fire.
        """
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
            "C": _state("C", pressure_ewma=0.0, streak=60),
        }

        # Pre-plan:  A=4/4=1, B=4/4=1, C=5/1=5. max=5, min=1, balance=0.2.
        # Post-plan: A=4/3=1.33, B=4/5=0.8, C=5. max=5 (tied), min=0.8,
        # balance=1/(5/0.8)=0.16. Drop 0.04 > balance_tolerance 0 -> reject.
        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B", "C"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": 1.0, "B": 1.0, "C": 5.0},
            effective_capacities={"A": 4, "B": 4, "C": 1},
            s_k_ewma={"A": 4.0, "B": 4.0, "C": 5.0},
            config=cfg,
            slots_per_worker_by_stage={},
        )

        assert result.reject_reason is RejectReason.BALANCE_REGRESSION


class TestThroughputCheckRelativeTolerance:
    """Throughput gate uses a scale-free RELATIVE band on ``max_k D_k``.

    The gate must catch the same *percentage* throughput regression
    whether the bottleneck ``max_D`` is small or large. The legacy
    absolute band (``throughput_after < throughput_before - tol``) went
    inert at high ``max_D`` because ``1 / max_D`` shrinks toward zero,
    so the worst regressions (the most severe bottlenecks) slipped
    through. These tests pin the relative semantics and the
    high-``max_D`` fix.
    """

    @staticmethod
    def _isolated_config(*, throughput_tolerance: float) -> SaturationAwareConfig:
        """Config that lets ONLY ``ThroughputCheck`` decide the verdict.

        Spread is made scale-independent (``bottleneck_weight=0`` so the
        D_k values under test do not move spread, ``spread_threshold=0``)
        and the donor-flip / balance gates are widened out of the way, so
        an accept / reject is attributable to the throughput band alone.
        """
        return _config(
            cross_stage_donor_throughput_tolerance=throughput_tolerance,
            cross_stage_donor_spread_threshold=0.0,
            cross_stage_donor_bottleneck_weight=0.0,
            cross_stage_donor_donor_flip_tolerance=1.0e9,
            cross_stage_donor_balance_tolerance=1.0e9,
        )

    def _evaluate_donor_channel_halving(self, *, s_k: float) -> GateResult:
        """Donate one of donor A's two channels so its ``D_k`` doubles.

        Donor A holds 2 channels (``D = s_k / 2``); removing one leaves 1
        (``D = s_k`` -> the new ``max_D``). Receiver B gains a channel and
        drops. ``max_D`` therefore goes ``s_k / 2 -> s_k`` (a fixed 2x
        blow-up = a 50% throughput loss) at EVERY ``s_k``, so the only
        thing that changes with scale is the absolute ``1 / max_D`` size.
        """
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", pressure_ewma=2.0, streak=60),
        }
        return _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": s_k / 2.0, "B": s_k / 4.0},
            effective_capacities={"A": 2, "B": 4},
            s_k_ewma={"A": s_k, "B": s_k},
            config=self._isolated_config(throughput_tolerance=0.02),
            slots_per_worker_by_stage={},
        )

    @pytest.mark.parametrize("s_k", [4.0, 400.0])
    def test_relative_regression_rejects_at_any_scale(self, s_k: float) -> None:
        """A 2x ``max_D`` blow-up rejects whether ``max_D`` is ~2 or ~200.

        Fail-on-unfixed: the legacy absolute band is inert at ``s_k=400``
        (``throughput`` ~0.0025 vs the 0.02 tolerance, so the reject
        condition needs ``throughput_after < -0.015``), so the un-fixed
        gate ACCEPTS the high-scale regression -- exactly the bug this
        fix closes. The relative band rejects the identical *percentage*
        loss at both scales.
        """
        result = self._evaluate_donor_channel_halving(s_k=s_k)

        assert result.reject_reason is RejectReason.THROUGHPUT_REGRESSION

    def test_within_band_regression_is_tolerated_at_high_scale(self) -> None:
        """A sub-tolerance regression at high ``max_D`` passes (no over-firing).

        Donor A holds 100 channels at ``D=100`` (the bottleneck);
        donating one nudges it to ``D ~= 101.01`` (a ~1.01% rise, inside
        the 2% band), and receiver B only improves, so ``max_D`` moves
        within the relative band -> throughput tie -> the gate must not
        reject. Guards against the fix flipping from "inert at high
        ``max_D``" to "rejects every high-``max_D`` plan".
        """
        plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        states = {
            "A": _state("A", slots_empty_ewma=0.5, streak=60),
            "B": _state("B", pressure_ewma=2.0, streak=60),
        }
        # A: 100 channels, S_k=10000 -> D=100. Removing one leaves 99 ->
        # D=10000/99 ~= 101.01 (~1.01% rise). B: 8 channels, S_k=400 -> D=50,
        # gains a channel and drops, so A stays the max both before and after.
        result = _evaluate_economic_gate(
            plan=plan,
            stage_names=["A", "B"],
            stage_states=states,
            receiver_intent=1,
            d_k_now={"A": 100.0, "B": 50.0},
            effective_capacities={"A": 100, "B": 8},
            s_k_ewma={"A": 10000.0, "B": 400.0},
            config=self._isolated_config(throughput_tolerance=0.02),
            slots_per_worker_by_stage={},
        )

        # Genuine (small) regression, not an improvement...
        assert result.max_d_after > result.max_d_before
        # ...but inside the 2% relative band, so throughput tolerates it.
        assert result.max_d_after <= result.max_d_before * 1.02
        assert result.reject_reason is None


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
            slots_per_worker_by_stage={},
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
            slots_per_worker_by_stage={},
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            result.accepted = False  # type: ignore[misc]
