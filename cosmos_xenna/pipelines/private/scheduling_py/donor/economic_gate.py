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

"""Economic gate for saturation-mode donor decisions.

``EconomicGate`` evaluates a candidate ``DonorPlan`` against five
ordered gates:

  1. Donor signal trust (per donor stage).
  2. Spread = receiver_value - donor_cost.
  3. Throughput non-regression (``1 / max_k D_k``).
  4. Donor-flip guard (no donor's post-plan D may exceed
     pre-plan ``max_k D_k`` beyond tolerance).
  5. Balance regression (only on throughput tie).

The gate is throughput-first by design: a regression in pipeline
throughput is always rejected, balance is the tie-breaker.

The evaluation pipeline composes two stable parts:

    +---+         +-------+
    | EconomicGate     |  build  | DonorEconomicScore     |
    |  - config        +-->| - throughput / spread  |
    |  - checks (tuple)|         | - d_after / max-D      |
    +---+         | - signal_trust         |
            |               +-------+
            |                          |
            v                          v
    iterate self.checks (5 ordered strategies)
            |
            v
    +-------+
    | GateResult                  |
    |  - accepted / reject_reason |
    |  - score metrics            |
    +-------+

``signal_trust`` is the donor stage's Sharpe-style trust metric
(``min(streak, cap) / (1 + noise_ewma)``) - exported because the
saturation policy's eligibility filter consumes it too.
"""

import math
import statistics
from collections.abc import Mapping
from typing import ClassVar, Protocol

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.types import (
    DonorPlan,
    GateResult,
    RejectReason,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring import compute_balance_score, compute_d_k
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def signal_trust(stage: StageRuntimeState, *, trust_streak_cap: int) -> float:
    """Sharpe-style trust metric for the donor stage's classifier signal.

    ``signal_trust = min(classifier_streak, trust_streak_cap)
                   / (1.0 + classifier_signal_noise_ewma)``.

    Longer streaks raise trust; EWMA noise lowers it. Cold-start
    stages (``classifier_signal_noise_ewma is None``) get
    denominator ``1.0`` so trust collapses to the clamped streak;
    the ``min_trust`` gate decides whether that suffices.

    """
    streak = min(stage.classifier.streak, trust_streak_cap)
    noise = stage.classifier.signal_noise_ewma if stage.classifier.signal_noise_ewma is not None else 0.0
    return streak / (1.0 + noise)


def _donor_cost(
    stage: StageRuntimeState,
    *,
    num_workers: int,
    streak_bonus: float,
    streak_cap: int,
) -> float:
    """Marginal cost of removing ``num_workers`` from ``stage``.

    ``cost = slots_empty_ratio_ewma * num_workers
            - streak_bonus * min(streak, streak_cap)``.
    Linear in ``num_workers``; long OVER_PROVISIONED streaks earn
    a discount so stable donors are preferred. Cold-start
    (``slots_empty_ratio_ewma is None``) contributes ``0`` to the
    base term - the streak bonus alone shapes the choice.

    """
    base = (stage.classifier.slots_empty_ratio_ewma or 0.0) * num_workers
    bonus = streak_bonus * min(stage.classifier.streak, streak_cap)
    return base - bonus


def _receiver_value(
    stage: StageRuntimeState,
    *,
    num_workers: int,
    d_k: float,
    median_d_k: float,
    intent: int,
    bottleneck_weight: float,
    intent_weight: float,
) -> float:
    """Marginal value of adding ``num_workers`` to receiver ``stage``.

    ``value = pressure_ewma * num_workers
              + bottleneck_weight * (d_k - median_d_k)
              + intent_weight * intent``.

    The production gate always passes ``num_workers=1`` (one
    receiver worker per donation commit). The bottleneck term
    pulls toward stages above the cluster median ``D_k``; the
    intent term breaks ties between similarly severe receivers.
    Non-finite ``d_k`` or ``median_d_k`` collapse the bottleneck
    term to ``0`` so cold-start cycles do not confuse the gate.

    """
    pressure_term = (stage.pressure.ewma or 0.0) * num_workers
    if math.isfinite(d_k) and math.isfinite(median_d_k):
        bottleneck_term = bottleneck_weight * (d_k - median_d_k)
    else:
        bottleneck_term = 0.0
    intent_term = intent_weight * intent
    return pressure_term + bottleneck_term + intent_term


def _compute_post_plan_d_k(
    *,
    d_k_now: Mapping[str, float],
    plan: DonorPlan,
    stage_names: list[str],
    effective_capacities: Mapping[str, int],
    s_k_ewma: Mapping[str, float],
    slots_per_worker_by_stage: Mapping[str, int],
) -> dict[str, float]:
    """Simulate ``D_k`` after one donation cycle commits ``plan``.

    Capacity deltas are in service channels
    (``slots_per_worker * total_allocations``). Each donor
    removal releases ``slots_per_worker_by_stage[donor]``
    channels (SPMD under-counts; donor-flip guard stays
    conservative). The receiver gains
    ``slots_per_worker_by_stage[receiver]`` channels once per
    commit. ``S_k`` is held fixed - one cycle's plan cannot
    retroactively change measured service time. Returns
    ``{stage: post_plan_D_k}`` over the same keys as ``d_k_now``.

    """
    capacity_after = dict(effective_capacities)
    for worker in plan.removals:
        donor_name = stage_names[worker.stage_index]
        donor_slots = slots_per_worker_by_stage.get(donor_name, 1)
        capacity_after[donor_name] = capacity_after.get(donor_name, 0) - donor_slots
    receiver_name = stage_names[plan.receiver_stage_index]
    receiver_slots = slots_per_worker_by_stage.get(receiver_name, 1)
    capacity_after[receiver_name] = capacity_after.get(receiver_name, 0) + receiver_slots
    return {name: compute_d_k(s_k_ewma.get(name, math.nan), capacity_after.get(name, 0)) for name in d_k_now}


def _max_finite(d_k: Mapping[str, float]) -> float:
    """Return ``max`` over finite-positive values; ``math.nan`` when none qualify."""
    finite = [v for v in d_k.values() if math.isfinite(v) and v > 0.0]
    if not finite:
        return math.nan
    return max(finite)


@attrs.frozen
class DonorEconomicScore:
    """Per-plan economic snapshot consumed by every gate check.

    Computed once by :meth:`EconomicGate._build_score` and reused
    by every ordered :class:`DonorEconomicCheck` (and by the
    structured commit / reject log lines through the
    :class:`GateResult` projection).

    Holds three logical groupings:

    1. Scalar economics: ``spread``, ``donor_cost``,
       ``receiver_value``.
    2. Throughput / balance metrics derived from
       ``D_k`` before vs. after the plan commits:
       ``throughput_before / throughput_after``,
       ``max_d_before / max_d_after``,
       ``balance_before / balance_after``.
    3. Per-donor breakdowns used by the donor-flip and
       signal-trust gates:
       ``signal_trust_per_donor`` (donor name -> Sharpe trust),
       ``d_after_per_donor`` (donor name -> post-plan D_k).
    """

    spread: float
    donor_cost: float
    receiver_value: float
    throughput_before: float
    throughput_after: float
    max_d_before: float
    max_d_after: float
    balance_before: float
    balance_after: float
    signal_trust_per_donor: Mapping[str, float]
    d_after_per_donor: Mapping[str, float]


class DonorEconomicCheck(Protocol):
    """Ordered gate check strategy.

    One :class:`DonorEconomicCheck` evaluates a single gate
    against the precomputed :class:`DonorEconomicScore` and the
    configured thresholds. Returning ``None`` accepts the plan
    at this gate and the next check runs; returning a
    :class:`RejectReason` short-circuits the rest of the
    evaluation. ``label`` is the canonical name used in
    structured logs and tests.
    """

    label: ClassVar[str]

    def check(self, score: DonorEconomicScore, config: SaturationAwareConfig) -> RejectReason | None:
        """Evaluate this gate; return ``None`` to pass."""
        ...


@attrs.frozen
class SignalTrustCheck:
    """Reject when any donor stage's Sharpe trust is below the floor.

    Per-donor ``signal_trust = min(streak, cap) / (1 + noise_ewma)``.
    Cold-start donors with a low classifier streak fall below the
    floor and are rejected so donations are only committed when
    the donor's empty-slot signal is statistically credible.
    """

    label: ClassVar[str] = "signal_trust"

    def check(self, score: DonorEconomicScore, config: SaturationAwareConfig) -> RejectReason | None:
        """Return ``RejectReason.SIGNAL_TRUST`` if any donor's trust is below the floor."""
        if any(trust < config.cross_stage_donor_min_trust for trust in score.signal_trust_per_donor.values()):
            return RejectReason.SIGNAL_TRUST
        return None


@attrs.frozen
class SpreadCheck:
    """Reject when ``receiver_value - donor_cost`` is below the spread threshold.

    The spread is the net marginal benefit of the plan: the
    receiver's pressure-weighted gain minus the donors' empty-
    slot-weighted cost. Plans below the configured spread floor
    are rejected so donations are biased toward clearly net-
    positive movements.
    """

    label: ClassVar[str] = "spread"

    def check(self, score: DonorEconomicScore, config: SaturationAwareConfig) -> RejectReason | None:
        """Return ``RejectReason.SPREAD_BELOW_THRESHOLD`` when spread is below the floor."""
        if score.spread < config.cross_stage_donor_spread_threshold:
            return RejectReason.SPREAD_BELOW_THRESHOLD
        return None


@attrs.frozen
class ThroughputCheck:
    """Reject when pipeline throughput regresses past tolerance.

    Throughput is ``1 / max_k D_k``. The check fires only when
    both before / after values are finite (cold-start cycles with
    NaN ``D_k`` cannot regress, so they always pass). The
    tolerance is symmetric on the comparison: regressions strictly
    larger than the tolerance reject; ties (within tolerance) fall
    through to the balance check.
    """

    label: ClassVar[str] = "throughput"

    def check(self, score: DonorEconomicScore, config: SaturationAwareConfig) -> RejectReason | None:
        """Return ``RejectReason.THROUGHPUT_REGRESSION`` on a finite, beyond-tolerance regression."""
        if (
            math.isfinite(score.throughput_before)
            and math.isfinite(score.throughput_after)
            and score.throughput_after < score.throughput_before - config.cross_stage_donor_throughput_tolerance
        ):
            return RejectReason.THROUGHPUT_REGRESSION
        return None


@attrs.frozen
class DonorFlipCheck:
    """Reject when any donor's post-plan ``D`` exceeds pre-plan ``max_D``.

    A donor whose own ``D_after`` would exceed the cluster's pre-
    plan ``max_D`` (beyond tolerance) is at risk of becoming the
    next bottleneck. The guard prevents donations that would
    "flip" the bottleneck from the current receiver to a former
    donor. Skipped when ``max_d_before`` is non-finite (cold-start
    cycles have no defined cluster-wide max).
    """

    label: ClassVar[str] = "donor_flip"

    def check(self, score: DonorEconomicScore, config: SaturationAwareConfig) -> RejectReason | None:
        """Return ``RejectReason.DONOR_FLIP_GUARD`` on a donor whose D_after exceeds the tolerance."""
        if not math.isfinite(score.max_d_before):
            return None
        ceiling = score.max_d_before + config.cross_stage_donor_donor_flip_tolerance
        for d_donor_after in score.d_after_per_donor.values():
            if math.isfinite(d_donor_after) and d_donor_after > ceiling:
                return RejectReason.DONOR_FLIP_GUARD
        return None


@attrs.frozen
class BalanceCheck:
    """Reject on a balance regression when throughput is tied.

    Fires only on a throughput tie (``|after - before|`` <= the
    throughput tolerance) so balance is strictly a tie-breaker -
    throughput-improving plans always pass even if balance
    drops, and only same-throughput-but-worse-balance plans are
    rejected here. Skipped on cold-start (non-finite balance).
    """

    label: ClassVar[str] = "balance"

    def check(self, score: DonorEconomicScore, config: SaturationAwareConfig) -> RejectReason | None:
        """Return ``RejectReason.BALANCE_REGRESSION`` when throughput is tied and balance regresses."""
        throughput_tied = (
            math.isfinite(score.throughput_before)
            and math.isfinite(score.throughput_after)
            and abs(score.throughput_after - score.throughput_before) <= config.cross_stage_donor_throughput_tolerance
        )
        if (
            throughput_tied
            and math.isfinite(score.balance_before)
            and math.isfinite(score.balance_after)
            and score.balance_after < score.balance_before - config.cross_stage_donor_balance_tolerance
        ):
            return RejectReason.BALANCE_REGRESSION
        return None


def _default_checks() -> tuple[DonorEconomicCheck, ...]:
    """Build the five ordered checks for the throughput-first gate.

    Order is significant: signal trust gates cold-start donors
    out first, spread filters net-negative plans, throughput
    rejects pipeline-throughput regressions, donor-flip prevents
    bottleneck flips, balance is the final tie-breaker.
    """
    return (
        SignalTrustCheck(),
        SpreadCheck(),
        ThroughputCheck(),
        DonorFlipCheck(),
        BalanceCheck(),
    )


@attrs.frozen
class EconomicGate:
    """Throughput-first commit gate for a saturation-mode donor plan.

    Stateless ``@attrs.frozen`` behaviour bundle. One instance per
    donor coordinator. Holds the ``SaturationAwareConfig`` so the
    five thresholds (``cross_stage_donor_min_trust``,
    ``cross_stage_donor_spread_threshold``,
    ``cross_stage_donor_throughput_tolerance``,
    ``cross_stage_donor_donor_flip_tolerance``,
    ``cross_stage_donor_balance_tolerance``) and the trust /
    streak / weight knobs come from one source per cycle. The
    ordered list of :class:`DonorEconomicCheck` strategies is
    swappable for experimentation; the default is the five-gate
    chain :func:`_default_checks` returns.
    """

    config: SaturationAwareConfig
    checks: tuple[DonorEconomicCheck, ...] = attrs.field(factory=_default_checks)

    def evaluate(
        self,
        *,
        plan: DonorPlan,
        stage_names: list[str],
        stage_states: dict[str, StageRuntimeState],
        receiver_intent: int,
        d_k_now: Mapping[str, float],
        effective_capacities: Mapping[str, int],
        s_k_ewma: Mapping[str, float],
        slots_per_worker_by_stage: Mapping[str, int],
    ) -> GateResult:
        """Build the per-plan score and run the ordered gate chain.

        First call :meth:`_build_score` to compute every metric
        once; then iterate ``self.checks`` until either a check
        returns a :class:`RejectReason` (short-circuit) or every
        check passes (accept). The :class:`GateResult` carries
        the score metrics regardless of verdict so the structured
        log lines can attribute the decision.
        """
        score = self._build_score(
            plan=plan,
            stage_names=stage_names,
            stage_states=stage_states,
            receiver_intent=receiver_intent,
            d_k_now=d_k_now,
            effective_capacities=effective_capacities,
            s_k_ewma=s_k_ewma,
            slots_per_worker_by_stage=slots_per_worker_by_stage,
        )
        reason: RejectReason | None = None
        for check in self.checks:
            reason = check.check(score, self.config)
            if reason is not None:
                break
        return _build_gate_result(score=score, reject_reason=reason)

    def _build_score(
        self,
        *,
        plan: DonorPlan,
        stage_names: list[str],
        stage_states: dict[str, StageRuntimeState],
        receiver_intent: int,
        d_k_now: Mapping[str, float],
        effective_capacities: Mapping[str, int],
        s_k_ewma: Mapping[str, float],
        slots_per_worker_by_stage: Mapping[str, int],
    ) -> DonorEconomicScore:
        """Compute every economic metric exactly once for this plan.

        Aggregates the per-donor signal trust and cost
        contributions, projects ``D_k`` forward through one commit
        of ``plan``, and derives the throughput / balance scalars
        the ordered checks read. ``num_workers=1`` for the receiver
        value matches the per-commit semantics: Phase C / Phase B
        add exactly one receiver worker per donation commit
        regardless of how many donors the plan removes.
        """
        config = self.config
        receiver_name = stage_names[plan.receiver_stage_index]
        receiver_state = stage_states[receiver_name]

        workers_per_donor_stage: dict[int, int] = {}
        for worker in plan.removals:
            workers_per_donor_stage[worker.stage_index] = workers_per_donor_stage.get(worker.stage_index, 0) + 1

        signal_trust_per_donor: dict[str, float] = {}
        for stage_index in workers_per_donor_stage:
            donor_name = stage_names[stage_index]
            signal_trust_per_donor[donor_name] = signal_trust(
                stage_states[donor_name],
                trust_streak_cap=config.cross_stage_donor_trust_streak_cap,
            )

        donor_cost = sum(
            _donor_cost(
                stage_states[stage_names[stage_index]],
                num_workers=count,
                streak_bonus=config.cross_stage_donor_streak_bonus,
                streak_cap=config.cross_stage_donor_streak_cap,
            )
            for stage_index, count in workers_per_donor_stage.items()
        )

        finite_d_k = [v for v in d_k_now.values() if math.isfinite(v) and v > 0.0]
        # ``statistics.median`` keeps the gate's median definition
        # aligned with bottleneck.py on every length parity.
        median_d_k = statistics.median(finite_d_k) if finite_d_k else math.nan
        receiver_value = _receiver_value(
            receiver_state,
            num_workers=1,
            d_k=d_k_now.get(receiver_name, math.nan),
            median_d_k=median_d_k,
            intent=receiver_intent,
            bottleneck_weight=config.cross_stage_donor_bottleneck_weight,
            intent_weight=config.cross_stage_donor_intent_weight,
        )

        spread = receiver_value - donor_cost
        d_after = _compute_post_plan_d_k(
            d_k_now=d_k_now,
            plan=plan,
            stage_names=stage_names,
            effective_capacities=effective_capacities,
            s_k_ewma=s_k_ewma,
            slots_per_worker_by_stage=slots_per_worker_by_stage,
        )
        # Per-donor D_after projection keyed by stage name so the
        # donor-flip check can iterate without knowing about
        # ``stage_index``. Restricted to the donors actually named
        # in the plan; non-donors are not gate-relevant here.
        d_after_per_donor = {
            stage_names[idx]: d_after.get(stage_names[idx], math.nan) for idx in workers_per_donor_stage
        }

        max_d_before = _max_finite(d_k_now)
        max_d_after = _max_finite(d_after)
        throughput_before = 1.0 / max_d_before if math.isfinite(max_d_before) and max_d_before > 0.0 else math.nan
        throughput_after = 1.0 / max_d_after if math.isfinite(max_d_after) and max_d_after > 0.0 else math.nan
        balance_before = compute_balance_score(d_k_now)
        balance_after = compute_balance_score(d_after)

        return DonorEconomicScore(
            spread=spread,
            donor_cost=donor_cost,
            receiver_value=receiver_value,
            throughput_before=throughput_before,
            throughput_after=throughput_after,
            max_d_before=max_d_before,
            max_d_after=max_d_after,
            balance_before=balance_before,
            balance_after=balance_after,
            signal_trust_per_donor=signal_trust_per_donor,
            d_after_per_donor=d_after_per_donor,
        )


def _build_gate_result(*, score: DonorEconomicScore, reject_reason: RejectReason | None) -> GateResult:
    """Project the score into the public :class:`GateResult` shape.

    The :class:`GateResult` is the only value the structured log
    lines consume; ``signal_trust_per_donor`` is materialized as
    a plain ``dict`` so downstream log producers can mutate
    their own copy without affecting the gate's internal view.
    """
    return GateResult(
        accepted=reject_reason is None,
        reject_reason=reject_reason,
        spread=score.spread,
        donor_cost=score.donor_cost,
        receiver_value=score.receiver_value,
        throughput_before=score.throughput_before,
        throughput_after=score.throughput_after,
        max_d_before=score.max_d_before,
        max_d_after=score.max_d_after,
        balance_before=score.balance_before,
        balance_after=score.balance_after,
        signal_trust_per_donor=dict(score.signal_trust_per_donor),
    )


__all__ = (
    "BalanceCheck",
    "DonorEconomicCheck",
    "DonorEconomicScore",
    "DonorFlipCheck",
    "EconomicGate",
    "SignalTrustCheck",
    "SpreadCheck",
    "ThroughputCheck",
    "_compute_post_plan_d_k",
    "_donor_cost",
    "_receiver_value",
    "signal_trust",
)
