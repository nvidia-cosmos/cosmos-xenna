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

"""Cross-stage donor selection.

Two modes ship from this module.

Floor mode (``select_youngest_eligible_donor``) is used by Phase B
floor enforcement when the cluster is full and the receiver cannot
reach its minimum-worker floor through fresh placement. It picks the
youngest worker from any non-receiver stage that can spare one
without violating its own floor; upstream donors are preferred when
any are eligible.

Saturation mode (``find_saturation_donor``) is used by Phase C
saturation-driven scale-up when the cluster is full and the receiver
wants to grow because its classifier signals SATURATED. Selection
runs the gate-filter -> bounded resource-fit search -> economic
gate pipeline; rejected plans emit a single DEBUG log line naming
the failing reason.

The non-negotiable donor-floor preservation rule applies in both
modes: a stage whose live worker count minus one would drop below
its floor is filtered out, preventing a single donation from
cascading into another stage's bootstrap.
"""

import enum
import itertools
import math
import statistics
from collections.abc import Mapping
from typing import TYPE_CHECKING

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import compute_balance_score, compute_d_k
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger

if TYPE_CHECKING:
    from cosmos_xenna.pipelines.private.data_structures import AutoscalePlanContext


class RejectReason(str, enum.Enum):
    """Enumerated reasons for rejecting a donor plan in saturation mode.

    String-valued so reasons can be emitted directly into structured
    DEBUG log lines without ``.value`` boilerplate. Each reason maps
    to exactly one gate check inside :func:`find_saturation_donor`;
    expanding the enum is the contract for adding a new gate. Place-
    holder rust reasons (``worker_not_found`` / ``release_failed`` /
    ``no_placement``) live on ``rust_apc.PlacementProbeResult`` and
    are surfaced separately in ``placement_reject_reason``.

    """

    MASTER_TOGGLE_OFF = "master_toggle_off"
    RECEIVER_ANTI_FLAP = "receiver_anti_flap"
    NO_CANDIDATES = "no_candidates"
    SIGNAL_TRUST = "signal_trust"
    RESOURCE_FIT = "resource_fit"
    SPREAD_BELOW_THRESHOLD = "spread_below_threshold"
    THROUGHPUT_REGRESSION = "throughput_regression"
    DONOR_FLIP_GUARD = "donor_flip_guard"
    BALANCE_REGRESSION = "balance_regression"


@attrs.frozen
class DonorWorker:
    """One worker removal in a multi-donor plan.

    Carries the planner indices needed to translate the symbolic
    choice into ``ctx.probe_add_after_removals`` and
    ``ctx.remove_workers_atomically`` arguments, plus the
    deterministic age tiebreaker the resource-fit search shares
    with the floor selector.

    Attributes:
        stage_index: Position of the donor stage in problem order.
        worker_id: Planner-assigned worker identifier on that stage.
        age: Worker age in autoscale cycles. Older workers are
            preferred during deterministic tie-breaking; freshly
            observed workers default to ``0``.

    """

    stage_index: int
    worker_id: str
    age: int


@attrs.frozen
class DonorPlan:
    """A non-empty group of donor worker removals targeting one receiver.

    Phase B floor and Phase C saturation paths both consume this
    type. Single-worker plans are valid; the bounded multi-donor
    resource-fit search in :func:`_resource_fit_plan` may also emit
    multi-worker plans when no single donor's released resources
    fit the receiver's worker shape. Distinct donor stages in
    ``removals`` each have their anti-flap timestamps advanced on
    commit so a single multi-stage donation does not silently
    allow one stage to repeatedly donate within the anti-flap
    window.

    Attributes:
        removals: Workers to remove, in commit order. The tuple
            MUST be non-empty: ``None`` represents "no donation",
            an empty ``DonorPlan`` is never constructed.
        receiver_stage_index: Position of the receiver stage in
            problem order; carried alongside ``removals`` so the
            planner probe / commit path does not need to re-derive
            it from caller-side state.

    """

    removals: tuple[DonorWorker, ...]
    receiver_stage_index: int

    def __attrs_post_init__(self) -> None:
        if not self.removals:
            msg = "DonorPlan.removals must contain at least one DonorWorker; pass None to signal no donation"
            raise ValueError(msg)


def select_youngest_eligible_donor(
    *,
    receiver_stage_index: int,
    stage_floors: dict[int, int],
    worker_ids_by_stage: list[list[str]],
    worker_ages: dict[str, int],
    worker_nodes: Mapping[str, str],
    ctx: "AutoscalePlanContext",
    max_plan_size: int,
    max_plan_combinations: int,
) -> DonorPlan | None:
    """Pick a donor plan for Phase B floor enforcement.

    Eligibility rules (the only filters that apply -- floor
    enforcement is non-negotiable, so no classifier-streak,
    growth-mode, signal-trust, anti-flap, or economic gates run):

      - The donor stage must differ from ``receiver_stage_index``.
      - The donor stage's live worker count minus the number of
        workers donated to this receiver in the plan must be at
        least the donor's own floor (``stage_floors.get(idx, 1)``);
        prevents cascading rescue. The bounded resource-fit search
        in :func:`_resource_fit_plan` enforces the per-stage
        budget explicitly via the ``removable_by_stage`` parameter
        the caller computes here as
        ``len(workers) - stage_floors.get(i, 1)``; a multi-donor
        combo whose ``stage_index`` distribution exceeds any
        donor's budget is skipped before probing.
      - Upstream donors (stage index strictly less than
        ``receiver_stage_index``) are preferred. When no upstream
        donor is eligible, candidates from any non-receiver stage
        are considered.

    The candidate pool is then handed to :func:`_resource_fit_plan`,
    which uses ``ctx.probe_add_after_removals`` as the feasibility
    oracle and returns the smallest feasible plan; deterministic
    tiebreak inside a fixed plan_size is
    ``(age ASC, worker_id ASC, stage_index ASC)``. Floor mode
    deliberately bypasses the saturation-mode warmup-grace
    exclusion -- deadlocking the floor on warmup-protected donors
    is a worse failure mode than killing a young donor.

    Args:
        receiver_stage_index: Index of the stage that needs the
            extra worker.
        stage_floors: Per-stage donor floors. A missing entry
            defaults to ``1`` (the implicit one-worker floor).
        worker_ids_by_stage: Per-stage live worker ids in problem
            order. Each inner list is the snapshot of workers that
            stage currently holds in the planner's working state.
        worker_ages: Cluster-wide worker ages keyed by worker id.
            Missing entries default to age 0 (treated as freshly
            observed).
        worker_nodes: Mapping ``worker_id -> node_id`` driving the
            same-node-first iteration order in
            :func:`_resource_fit_plan`. An empty mapping degrades
            to flat iteration (no locality preference).
        ctx: Active autoscale planner context. Only
            ``probe_add_after_removals`` is invoked on it; the
            helper is non-mutating with respect to ``ctx``.
        max_plan_size: Bound on the multi-donor plan width.
            Mirrors ``cross_stage_donor_max_plan_size``.
        max_plan_combinations: Bound on probes evaluated per
            plan size. Mirrors
            ``cross_stage_donor_max_plan_combinations``.

    Returns:
        A ``DonorPlan`` (1..``max_plan_size`` workers) when the
        bounded resource-fit search finds a feasible plan;
        ``None`` when no stage can donate without violating its
        own floor or when no probe up to ``max_plan_size`` reports
        feasibility.

    """
    eligible_stages = [
        stage_index
        for stage_index, workers in enumerate(worker_ids_by_stage)
        if stage_index != receiver_stage_index and len(workers) - 1 >= stage_floors.get(stage_index, 1)
    ]
    if not eligible_stages:
        return None

    upstream = [s for s in eligible_stages if s < receiver_stage_index]
    pool = upstream if upstream else eligible_stages

    candidates = [
        DonorWorker(
            stage_index=stage_index,
            worker_id=wid,
            age=worker_ages.get(wid, 0),
        )
        for stage_index in pool
        for wid in worker_ids_by_stage[stage_index]
    ]
    if not candidates:
        return None

    # Per-stage donor budget: each donor stage can spare at most
    # ``len(workers) - floor`` workers in one combo. The pool-level
    # eligibility filter above only proves at least ONE worker is
    # spare (``len(workers) - 1 >= floor``); without an explicit
    # per-stage cap the bounded resource-fit search could pull
    # several workers from one donor and silently drop it below its
    # own floor.
    removable_by_stage = {
        stage_index: max(0, len(worker_ids_by_stage[stage_index]) - stage_floors.get(stage_index, 1))
        for stage_index in pool
    }

    return _resource_fit_plan(
        receiver_stage_index=receiver_stage_index,
        candidates=candidates,
        worker_nodes=worker_nodes,
        ctx=ctx,
        max_plan_size=max_plan_size,
        max_plan_combinations=max_plan_combinations,
        removable_by_stage=removable_by_stage,
    )


def _resource_fit_plan(
    *,
    receiver_stage_index: int,
    candidates: list[DonorWorker],
    worker_nodes: Mapping[str, str],
    ctx: "AutoscalePlanContext",
    max_plan_size: int,
    max_plan_combinations: int,
    removable_by_stage: Mapping[int, int],
) -> DonorPlan | None:
    """Bounded multi-donor search for the smallest feasible donor plan.

    Iterates ``plan_size`` from 1 to ``max_plan_size``. At each size,
    enumerates donor-worker combinations same-node first (using
    ``worker_nodes`` to group; an empty mapping collapses to flat
    iteration), then cross-node, capping evaluations at
    ``max_plan_combinations`` per ``plan_size``. Each candidate
    combination is filtered against the per-stage donor budget,
    then probed via ``ctx.probe_add_after_removals``; the first
    feasible probe wins. Within a fixed size the deterministic
    tiebreak is ``(age ASC, worker_id ASC, stage_index ASC)`` -
    ``itertools.combinations`` over a list pre-sorted on that key
    produces the lexicographically smallest combinations first.

    The Rust planner's probe is the source of truth for placement
    feasibility (FGD / SPMD reuse rules). The helper itself enforces
    the per-stage donor floor: previously the budget was implicit
    (the caller's ``len(workers) - 1 >= floor`` filter only proves
    that ONE worker can be spared, so multi-donor combos drawing
    several workers from the same stage could silently violate the
    floor), now an explicit ``removable_by_stage`` map caps the
    contribution of each donor stage to any combo.

    Args:
        receiver_stage_index: Position of the receiver stage in
            problem order; passed verbatim to
            ``ctx.probe_add_after_removals``.
        candidates: Already gate-filtered donor workers (anti-flap,
            classifier-streak, floor-eligibility, warmup-exclude
            checks already applied by the caller).
        worker_nodes: Mapping ``worker_id -> node_id``. Empty
            mapping -> all candidates collapse into a single
            "unknown" group and same-node iteration is a no-op.
        ctx: Active autoscale planner context. Only
            ``probe_add_after_removals`` is invoked; no mutation.
        max_plan_size: Maximum number of donor workers in a
            returned plan. Bounds the search width.
        max_plan_combinations: Per-``plan_size`` cap on probes
            evaluated. Bounds search depth on pathological clusters
            (e.g. dozens of fragmented donors).
        removable_by_stage: Per-stage donor budget keyed by donor
            ``stage_index``. The caller MUST populate this with
            ``max(0, len(worker_ids_by_stage[i]) - stage_floors.get(i, 1))``
            for every stage that contributes candidates; missing
            entries default to ``0`` (the stage is assumed unable
            to spare any worker, which collapses the search to
            single-stage combos drawn from explicitly budgeted
            stages). A combo whose ``stage_index`` distribution
            exceeds any stage's budget is skipped without probing.

    Returns:
        The first feasible ``DonorPlan`` whose removals satisfy the
        receiver's placement, or ``None`` when every probe up to
        ``max_plan_size`` returned infeasible (or the candidate
        list is empty / parameters are non-positive).

    """
    if not candidates:
        return None
    if max_plan_size < 1 or max_plan_combinations < 1:
        return None

    # Pre-sort to make ``itertools.combinations`` produce the
    # deterministic tiebreak order in lexicographic ASC.
    sorted_candidates = sorted(
        candidates,
        key=lambda w: (w.age, w.worker_id, w.stage_index),
    )

    # Group candidates by node for same-node-first iteration.
    # Insertion order of nodes follows first-occurrence in the
    # sorted candidate list so deterministic order survives the
    # grouping step. Workers whose node is unknown collapse into a
    # ``""`` bucket which is iterated like any other node.
    candidates_by_node: dict[str, list[DonorWorker]] = {}
    nodes_in_order: list[str] = []
    for worker in sorted_candidates:
        node = worker_nodes.get(worker.worker_id, "")
        bucket = candidates_by_node.get(node)
        if bucket is None:
            candidates_by_node[node] = [worker]
            nodes_in_order.append(node)
        else:
            bucket.append(worker)

    for plan_size in range(1, max_plan_size + 1):
        evaluations = 0
        # Dedup key uses the full planner identity ``(stage_index,
        # worker_id)`` rather than ``worker_id`` alone. While Ray
        # actor ids are globally unique today, the rest of the donor
        # path commits removals as ``(stage_index, worker_id)`` tuples
        # (``probe_add_after_removals`` / ``remove_workers_atomically``),
        # so the dedup key SHOULD use the same identity. Any future
        # change that introduces stage-scoped ids (e.g. a test fixture
        # that reuses ids across stages) would otherwise silently
        # collapse distinct combos and skew the per-cycle search.
        seen_combos: set[tuple[tuple[int, str], ...]] = set()

        # Phase 1: same-node combinations. Each node contributes its
        # own combinations; ordering within a node follows
        # ``sorted_candidates``. Nodes with fewer than ``plan_size``
        # candidates skip Phase 1 for this iteration.
        for node in nodes_in_order:
            same_node = candidates_by_node[node]
            if len(same_node) < plan_size:
                continue
            for combo in itertools.combinations(same_node, plan_size):
                if evaluations >= max_plan_combinations:
                    break
                if _combo_violates_stage_budget(combo, removable_by_stage):
                    # Cheaper than a probe round-trip and necessary
                    # for correctness: a multi-donor combo could
                    # otherwise pull every spare worker from one
                    # donor stage and drop it below floor. Skipped
                    # combos do not consume the evaluation cap so
                    # the search keeps making progress on combos
                    # that respect every stage's budget.
                    continue
                combo_key = tuple((w.stage_index, w.worker_id) for w in combo)
                seen_combos.add(combo_key)
                evaluations += 1
                removals = [(w.stage_index, w.worker_id) for w in combo]
                probe = ctx.probe_add_after_removals(removals, receiver_stage_index)
                if probe.feasible:
                    return DonorPlan(
                        removals=tuple(combo),
                        receiver_stage_index=receiver_stage_index,
                    )
            if evaluations >= max_plan_combinations:
                break

        # Phase 2: cross-node combinations. Skip any combo Phase 1
        # already evaluated so the cap counts unique probe calls only.
        if evaluations >= max_plan_combinations:
            continue
        for combo in itertools.combinations(sorted_candidates, plan_size):
            if evaluations >= max_plan_combinations:
                break
            if _combo_violates_stage_budget(combo, removable_by_stage):
                continue
            combo_key = tuple((w.stage_index, w.worker_id) for w in combo)
            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)
            evaluations += 1
            removals = [(w.stage_index, w.worker_id) for w in combo]
            probe = ctx.probe_add_after_removals(removals, receiver_stage_index)
            if probe.feasible:
                return DonorPlan(
                    removals=tuple(combo),
                    receiver_stage_index=receiver_stage_index,
                )

    return None


def _combo_violates_stage_budget(
    combo: tuple[DonorWorker, ...],
    removable_by_stage: Mapping[int, int],
) -> bool:
    """Return ``True`` when ``combo`` exceeds any donor stage's removable budget.

    A combo's ``stage_index`` distribution must respect every donor
    stage's per-cycle removable count
    (``len(workers) - stage_floors.get(stage_index, 1)``). Without
    this guard a multi-donor resource-fit search could draw several
    workers from a single donor stage and drop it below its own
    floor, cascading the rescue.

    Args:
        combo: Candidate donor workers under consideration this
            iteration. Ordering does not matter.
        removable_by_stage: Per-stage budget keyed by ``stage_index``;
            entries missing from the map default to ``0`` (no
            removable workers from that stage).

    Returns:
        ``True`` when at least one stage_index appears in ``combo``
        more times than ``removable_by_stage`` allows; ``False``
        when every stage's contribution fits its budget.
    """
    stage_counts: dict[int, int] = {}
    for worker in combo:
        stage_counts[worker.stage_index] = stage_counts.get(worker.stage_index, 0) + 1
    return any(count > removable_by_stage.get(stage_index, 0) for stage_index, count in stage_counts.items())


def _donor_cost(
    stage: _StageRuntimeState,
    *,
    num_workers: int,
    streak_bonus: float,
    streak_cap: int,
) -> float:
    """Marginal cost of removing ``num_workers`` from ``stage``.

    ``cost = slots_empty_ratio_ewma * num_workers - streak_bonus * min(streak, streak_cap)``.

    Larger ``num_workers`` increases cost linearly; a long
    OVER_PROVISIONED streak is rewarded with a discount so stable
    donors are preferred. Cold-start stages whose
    ``slots_empty_ratio_ewma`` is ``None`` contribute ``0`` to the
    base term, which means an unmeasured donor has no idle-capacity
    cost (the streak bonus alone shapes the choice).

    Args:
        stage: Per-stage runtime state for the donor stage.
        num_workers: Workers removed from this stage in the plan.
        streak_bonus: ``cross_stage_donor_streak_bonus`` weight on
            the streak discount; non-negative.
        streak_cap: ``cross_stage_donor_streak_cap`` upper bound on
            the streak term so a runaway streak cannot dominate.

    """
    base = (stage.slots_empty_ratio_ewma or 0.0) * num_workers
    bonus = streak_bonus * min(stage.classifier_streak, streak_cap)
    return base - bonus


def _receiver_value(
    stage: _StageRuntimeState,
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

    The pressure term scales with plan size so larger plans are
    rewarded proportional to the receiver's backlog; the bottleneck
    term pulls demand toward stages whose ``D_k`` exceeds the
    cluster median; the intent term breaks ties when two stages
    have similar bottleneck severity but differ in declared
    Phase C intent. Non-finite ``d_k`` or ``median_d_k`` collapse
    the bottleneck term to ``0`` so cold-start cycles do not
    confuse the gate.

    Args:
        stage: Per-stage runtime state for the receiver stage.
        num_workers: Workers added to this stage in the plan.
        d_k: Receiver's actor-normalized service demand this cycle.
        median_d_k: Median ``D_k`` across stages with finite
            ``D_k`` this cycle.
        intent: Receiver's Phase C grow intent (signed worker
            delta); only the positive part contributes meaningful
            value here.
        bottleneck_weight: ``cross_stage_donor_bottleneck_weight``
            applied to the ``(d_k - median_d_k)`` term.
        intent_weight: ``cross_stage_donor_intent_weight`` applied
            to the ``intent`` term.

    """
    pressure_term = (stage.pressure_ewma or 0.0) * num_workers
    if math.isfinite(d_k) and math.isfinite(median_d_k):
        bottleneck_term = bottleneck_weight * (d_k - median_d_k)
    else:
        bottleneck_term = 0.0
    intent_term = intent_weight * intent
    return pressure_term + bottleneck_term + intent_term


def _signal_trust(stage: _StageRuntimeState, *, trust_streak_cap: int) -> float:
    """Sharpe-style trust metric for the donor stage's classifier signal.

    ``signal_trust = min(classifier_streak, trust_streak_cap)
                   / (1.0 + classifier_signal_noise_ewma)``.

    The streak supplies the numerator (longer-running classifier
    state is more trustworthy); the EWMA over absolute deltas in
    ``slots_empty_ratio_ewma`` is the noise denominator (a
    flickering classifier earns less trust). Cold-start stages
    whose ``classifier_signal_noise_ewma`` is ``None`` get a
    denominator of ``1.0`` so the trust score equals the clamped
    streak; the trust gate's ``min_trust`` threshold then decides
    whether the cold-start signal is sufficient.

    Args:
        stage: Per-stage runtime state for the donor stage.
        trust_streak_cap: ``cross_stage_donor_trust_streak_cap``
            upper bound on the streak input.

    """
    streak = min(stage.classifier_streak, trust_streak_cap)
    noise = stage.classifier_signal_noise_ewma if stage.classifier_signal_noise_ewma is not None else 0.0
    return streak / (1.0 + noise)


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

    ``effective_capacities`` is in service channels
    (``slots_per_worker * total_allocations``), so capacity deltas
    must be in the same unit:

    *   Each donor worker removed releases
        ``slots_per_worker_by_stage[donor]`` channels (non-SPMD: one
        worker_group with one allocation; SPMD groups under-count
        their channel loss because the helper does not see per-group
        allocation counts -- this is an acceptable simplification
        because SPMD donor stages are rare and the gate's donor-flip
        guard remains conservative even when channel loss is
        understated).
    *   The receiver gains ``slots_per_worker_by_stage[receiver]``
        channels exactly once per donation commit, NOT per donor in
        the plan. ``_run_phase_c_grow`` calls ``try_add_worker``
        once per ``_attempt_cross_stage_donation`` invocation
        (regardless of ``len(plan.removals)``); subsequent receiver
        additions in the same cycle are separate donation commits
        with their own gate evaluations.

    The intrinsic ``S_k`` EWMA is held fixed across the simulation
    because a single cycle's donor plan cannot retroactively change
    measured per-task service time.

    Args:
        d_k_now: Current cycle's actor-normalized ``D_k`` mapping.
            Keys define the result mapping's keys.
        plan: Candidate donor plan to simulate.
        stage_names: Stage names in problem order; used to map
            ``DonorWorker.stage_index`` -> stage name.
        effective_capacities: Pre-plan effective capacity per stage
            in channels (``slots_per_worker * total_allocations``,
            per :meth:`SaturationAwareScheduler._effective_ready_capacity`).
        s_k_ewma: Intrinsic service-time EWMA per stage; held
            fixed across the simulation.
        slots_per_worker_by_stage: Per-stage ``slots_per_worker``
            from ``problem_state.rust.stages[*].slots_per_worker``.
            Used as the per-worker channel multiplier for both
            donor and receiver capacity deltas. Missing entries
            default to ``1`` (treats the stage as one channel per
            worker).

    Returns:
        ``{stage: post_plan_D_k}`` over the same keys as ``d_k_now``.
        ``compute_d_k`` handles non-finite or zero-capacity inputs
        by returning ``math.nan``, matching the production
        bottleneck-identification path.

    """
    capacity_after = dict(effective_capacities)
    for worker in plan.removals:
        donor_name = stage_names[worker.stage_index]
        donor_slots = slots_per_worker_by_stage.get(donor_name, 1)
        capacity_after[donor_name] = capacity_after.get(donor_name, 0) - donor_slots
    receiver_name = stage_names[plan.receiver_stage_index]
    receiver_slots = slots_per_worker_by_stage.get(receiver_name, 1)
    # Phase C adds exactly one receiver worker per donation commit,
    # regardless of plan size.
    capacity_after[receiver_name] = capacity_after.get(receiver_name, 0) + receiver_slots
    return {name: compute_d_k(s_k_ewma.get(name, math.nan), capacity_after.get(name, 0)) for name in d_k_now}


def _max_finite(d_k: Mapping[str, float]) -> float:
    """Return ``max`` over finite-positive values; ``math.nan`` when none qualify."""
    finite = [v for v in d_k.values() if math.isfinite(v) and v > 0.0]
    if not finite:
        return math.nan
    return max(finite)


@attrs.frozen
class _GateResult:
    """Outcome of the donor-plan economic gate evaluation."""

    accepted: bool
    reject_reason: RejectReason | None
    spread: float
    donor_cost: float
    receiver_value: float
    throughput_before: float
    throughput_after: float
    max_d_before: float
    max_d_after: float
    balance_before: float
    balance_after: float
    signal_trust_per_donor: dict[str, float]


@attrs.frozen
class DonorDecision:
    """Successful donor selection plus the gate metrics that justify it.

    Returned by :func:`find_saturation_donor` on the accept path so
    the caller can commit ``plan`` atomically and emit the structured
    INFO commit log without re-running the economic gate. ``plan``
    is the planner-validated donor plan; ``gate_result`` carries the
    full metric set the decision log surfaces (donor_cost, spread,
    signal_trust per donor, throughput / balance / max_d before-and-
    after). ``receiver_capacity_before`` records the receiver's
    effective ready capacity at the moment the gate fired so the
    log can compute ``capacity_after`` deterministically without
    re-querying the planner.

    Attributes:
        plan: The committed donor plan; non-empty by construction.
        gate_result: Full gate evaluation, including every metric
            the decision log reports.
        receiver_capacity_before: Receiver's effective ready
            capacity when the gate fired. Operators feed this into
            the log helper to render ``capacity_before`` /
            ``capacity_after`` symmetrically.

    """

    plan: DonorPlan
    gate_result: _GateResult
    receiver_capacity_before: int


def _evaluate_economic_gate(
    *,
    plan: DonorPlan,
    stage_names: list[str],
    stage_states: dict[str, _StageRuntimeState],
    receiver_intent: int,
    d_k_now: Mapping[str, float],
    effective_capacities: Mapping[str, int],
    s_k_ewma: Mapping[str, float],
    slots_per_worker_by_stage: Mapping[str, int],
    config: SaturationAwareConfig,
) -> _GateResult:
    """Apply the throughput-first commit gate to a candidate donor plan.

    The gate is throughput-first, balance-second: a plan is
    accepted only when (a) it does not regress the cluster
    throughput estimate beyond tolerance, (b) no donor stage flips
    above the pre-plan maximum ``D_k`` (a freshly created
    bottleneck is worse than the original imbalance), (c) the
    marginal-value spread clears the configured threshold, and (d)
    when throughput is exactly tied, the balance score does not
    regress beyond tolerance. Signal trust is enforced first so a
    noisy donor cannot drive any further computation.

    Order of checks:

      1. ``signal_trust`` per donor stage in ``plan`` must be at or
         above ``cross_stage_donor_min_trust``.
      2. ``spread = receiver_value - donor_cost`` must be at or
         above ``cross_stage_donor_spread_threshold``.
      3. ``throughput_after`` must not regress beyond
         ``cross_stage_donor_throughput_tolerance``.
      4. No donor stage's ``D_after`` may exceed the pre-plan
         ``max_d_before`` by more than
         ``cross_stage_donor_donor_flip_tolerance``.
      5. When throughput is tied within tolerance, ``balance_after``
         must not regress beyond
         ``cross_stage_donor_balance_tolerance``.

    Returns ``_GateResult`` carrying every metric the decision-log
    needs so the caller can emit a structured DEBUG line on
    rejection or INFO line on commit without re-deriving values.
    """
    receiver_name = stage_names[plan.receiver_stage_index]
    receiver_state = stage_states[receiver_name]

    # Per-donor-stage breakdown: how many workers come from each
    # distinct donor stage and that stage's runtime state.
    workers_per_donor_stage: dict[int, int] = {}
    for worker in plan.removals:
        workers_per_donor_stage[worker.stage_index] = workers_per_donor_stage.get(worker.stage_index, 0) + 1

    signal_trust_per_donor: dict[str, float] = {}
    for stage_index in workers_per_donor_stage:
        donor_name = stage_names[stage_index]
        signal_trust_per_donor[donor_name] = _signal_trust(
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
    # Use ``statistics.median`` so the gate's median definition matches
    # the rest of the scheduler (bottleneck.py) on every length parity.
    median_d_k = statistics.median(finite_d_k) if finite_d_k else math.nan
    receiver_value = _receiver_value(
        receiver_state,
        num_workers=len(plan.removals),
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
    max_d_before = _max_finite(d_k_now)
    max_d_after = _max_finite(d_after)
    throughput_before = 1.0 / max_d_before if math.isfinite(max_d_before) and max_d_before > 0.0 else math.nan
    throughput_after = 1.0 / max_d_after if math.isfinite(max_d_after) and max_d_after > 0.0 else math.nan
    balance_before = compute_balance_score(d_k_now)
    balance_after = compute_balance_score(d_after)

    def _result(reason: RejectReason | None) -> _GateResult:
        return _GateResult(
            accepted=reason is None,
            reject_reason=reason,
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
        )

    # Gate 1: signal trust per donor stage.
    if any(trust < config.cross_stage_donor_min_trust for trust in signal_trust_per_donor.values()):
        return _result(RejectReason.SIGNAL_TRUST)

    # Gate 2: spread threshold.
    if spread < config.cross_stage_donor_spread_threshold:
        return _result(RejectReason.SPREAD_BELOW_THRESHOLD)

    # Gate 3: throughput non-regression. NaN comparisons short-
    # circuit to False, so cold-start cycles never block.
    if (
        math.isfinite(throughput_before)
        and math.isfinite(throughput_after)
        and throughput_after < throughput_before - config.cross_stage_donor_throughput_tolerance
    ):
        return _result(RejectReason.THROUGHPUT_REGRESSION)

    # Gate 4: donor-flip guard. No donor stage may rise above the
    # cycle's pre-plan ``max_d_before`` (plus tolerance).
    if math.isfinite(max_d_before):
        for stage_index in workers_per_donor_stage:
            donor_name = stage_names[stage_index]
            d_donor_after = d_after.get(donor_name, math.nan)
            if (
                math.isfinite(d_donor_after)
                and d_donor_after > max_d_before + config.cross_stage_donor_donor_flip_tolerance
            ):
                return _result(RejectReason.DONOR_FLIP_GUARD)

    # Gate 5: balance regression, only when throughput is tied.
    throughput_tied = (
        math.isfinite(throughput_before)
        and math.isfinite(throughput_after)
        and abs(throughput_after - throughput_before) <= config.cross_stage_donor_throughput_tolerance
    )
    if (
        throughput_tied
        and math.isfinite(balance_before)
        and math.isfinite(balance_after)
        and balance_after < balance_before - config.cross_stage_donor_balance_tolerance
    ):
        return _result(RejectReason.BALANCE_REGRESSION)

    return _result(None)


def _format_donor_decision_log(
    *,
    receiver_name: str,
    receiver_state: _StageRuntimeState | None,
    receiver_d_k: float,
    receiver_intent: int,
    capacity_before: int,
    capacity_after: int,
    plan: DonorPlan | None,
    stage_names: list[str],
    gate_result: _GateResult | None,
    spread_threshold: float,
    reject_reason: str | None,
    placement_reject_reason: str = "",
) -> str:
    """Build the structured decision-log string for a donor commit or rejection.

    Common fields appear in the same order in both INFO commit and
    DEBUG reject variants. The DEBUG variant additionally prefixes
    two reason fields (``reject_reason``, ``placement_reject_reason``)
    immediately after the line marker; commit variants omit them.
    Missing inputs (early-return rejections that have no plan or gate
    result) are rendered as ``nan`` for floats, ``[]`` for lists, and
    the empty string for ``placement_reject_reason``; the common
    suffix stays stable so log consumers can split on field names
    without conditional parsing.

    Args:
        receiver_name: Receiver stage name.
        receiver_state: Runtime state for the receiver stage. ``None``
            when the helper is called from an early-return path before
            the receiver state lookup runs.
        receiver_d_k: Receiver's actor-normalized service demand for
            this cycle. ``math.nan`` is acceptable for cold-start.
        receiver_intent: Receiver's Phase C grow intent.
        capacity_before: Receiver's effective ready capacity before
            the plan would commit.
        capacity_after: Receiver's effective ready capacity after the
            plan commits (``capacity_before + len(plan.removals)``).
        plan: Candidate donor plan. ``None`` for early-return paths
            (no candidates, master toggle off, etc.).
        stage_names: Stage names in problem order; used to render
            donor stage names from ``DonorWorker.stage_index``.
        gate_result: Outcome of :func:`_evaluate_economic_gate`.
            ``None`` when the helper is called before the gate runs
            (early-return paths and resource-fit failures).
        spread_threshold: ``cross_stage_donor_spread_threshold`` value
            for this cycle, recorded so operators can correlate
            ``spread`` against the active threshold.
        reject_reason: ``RejectReason`` value emitted for DEBUG; pass
            ``None`` for the INFO commit form.
        placement_reject_reason: Rust placement-probe reason
            (``worker_not_found`` / ``release_failed`` /
            ``no_placement``) when the rejection came from a probe.
            Empty string for every other rejection.

    Returns:
        The formatted decision-log string. Caller chooses
        ``logger.info`` (commit) or ``logger.debug`` (reject).
    """
    if plan is not None:
        donor_pairs = [(stage_names[w.stage_index], w.worker_id) for w in plan.removals]
    else:
        donor_pairs = []

    if gate_result is not None:
        donor_cost = gate_result.donor_cost
        receiver_value = gate_result.receiver_value
        spread = gate_result.spread
        signal_trust_pairs = sorted(gate_result.signal_trust_per_donor.items())
        throughput_before = gate_result.throughput_before
        throughput_after = gate_result.throughput_after
        max_d_before = gate_result.max_d_before
        max_d_after = gate_result.max_d_after
        balance_before = gate_result.balance_before
        balance_after = gate_result.balance_after
    else:
        donor_cost = math.nan
        receiver_value = math.nan
        spread = math.nan
        signal_trust_pairs = []
        throughput_before = math.nan
        throughput_after = math.nan
        max_d_before = math.nan
        max_d_after = math.nan
        balance_before = math.nan
        balance_after = math.nan

    classifier_value = receiver_state.classifier_state.value if receiver_state is not None else ""
    pressure_ewma_value = (
        receiver_state.pressure_ewma
        if receiver_state is not None and receiver_state.pressure_ewma is not None
        else math.nan
    )
    slots_empty_value = (
        receiver_state.slots_empty_ratio_ewma
        if receiver_state is not None and receiver_state.slots_empty_ratio_ewma is not None
        else math.nan
    )

    line_kind = "commit" if reject_reason is None else "reject"
    fields: list[str] = [f"[scheduler] donor decision ({line_kind}):"]
    if reject_reason is not None:
        fields.append(f"reject_reason={reject_reason!r}")
        fields.append(f"placement_reject_reason={placement_reject_reason!r}")
    fields.extend(
        [
            f"receiver={receiver_name!r}",
            f"classifier={classifier_value!r}",
            f"pressure_ewma={pressure_ewma_value:.4f}",
            f"slots_empty={slots_empty_value:.4f}",
            f"D_k={receiver_d_k:.4f}",
            f"capacity_before={capacity_before}",
            f"capacity_after={capacity_after}",
            f"intent={receiver_intent}",
            f"receiver_value={receiver_value:.4f}",
            f"donor_plan={donor_pairs!r}",
            f"donor_cost={donor_cost:.4f}",
            f"spread={spread:.4f}",
            f"spread_threshold={spread_threshold:.4f}",
            f"signal_trust={signal_trust_pairs!r}",
            f"throughput_before={throughput_before:.4f}",
            f"throughput_after={throughput_after:.4f}",
            f"max_d_before={max_d_before:.4f}",
            f"max_d_after={max_d_after:.4f}",
            f"balance_score_before={balance_before:.4f}",
            f"balance_score_after={balance_after:.4f}",
        ]
    )
    return " ".join(fields)


def find_saturation_donor(
    *,
    receiver_stage_index: int,
    receiver_stage_name: str,
    stage_names: list[str],
    stage_floors: dict[int, int],
    worker_ids_by_stage: list[list[str]],
    worker_ages: dict[str, int],
    worker_nodes: Mapping[str, str],
    stage_states: dict[str, _StageRuntimeState],
    config: SaturationAwareConfig,
    stage_configs: dict[str, SaturationAwareStageConfig],
    cycle: int,
    last_donation_cycle: dict[str, int],
    ctx: "AutoscalePlanContext",
    receiver_intent: int,
    d_k_now: Mapping[str, float],
    effective_capacities: Mapping[str, int],
    s_k_ewma: Mapping[str, float],
    slots_per_worker_by_stage: Mapping[str, int],
    excluded_worker_ids: frozenset[str] | None = None,
) -> DonorDecision | None:
    """Pick a donor plan for saturation-driven Phase C growth.

    The non-negotiable donor-floor rule from
    :func:`select_youngest_eligible_donor` applies unchanged. On top
    of it, four anti-flap / trust layers gate the candidate pool:

      1. Donor classifier must be ``OVER_PROVISIONED`` with at least
         ``stage_cfg.over_provisioned_streak_min_cycles`` full
         streak (gated by
         ``config.cross_stage_donor_require_over_provisioned``).
      2. Donor growth mode must not be ``HOLD`` (gated by
         ``config.cross_stage_donor_exclude_hold_state``).
      3. Receiver must not have donated within the last
         ``config.cross_stage_donor_anti_flap_cycles`` (prevents
         donate-then-receive ping-pong).
      4. Donor stage's :func:`_signal_trust` must be at or above
         ``config.cross_stage_donor_min_trust`` - noisy classifier
         signals fail the trust gate even if the streak is long.

    Once the filtered candidate pool is built,
    :func:`_resource_fit_plan` performs a bounded multi-donor search
    (up to ``config.cross_stage_donor_max_plan_size`` workers,
    ``config.cross_stage_donor_max_plan_combinations`` probes per
    plan size) using ``ctx.probe_add_after_removals`` as the
    feasibility oracle. The first feasible plan is then run through
    the throughput-first economic gate
    (:func:`_evaluate_economic_gate`): a plan is committed only when
    its post-plan throughput does not regress, no donor stage flips
    into the new bottleneck, the spread clears
    ``cross_stage_donor_spread_threshold``, and (when throughput is
    tied) the balance score does not regress beyond tolerance. Each
    rejection emits one structured DEBUG log line carrying every
    metric the operator needs to diagnose the choice.

    The master toggle ``config.enable_cross_stage_donor`` short-
    circuits the whole helper to ``None`` when disabled. The
    ``config.donor_must_be_strictly_upstream`` flag restricts donors
    to stages with strictly smaller DAG depth when True. Per-cycle
    receiver absorption is naturally bounded by the receiver's
    Phase C intent (capped by ``aggressive_growth_max_per_cycle``);
    a separate cross-stage cap is redundant. Donor-side cooldown is
    subsumed by the OVER_PROVISIONED + streak gate above.

    Args:
        receiver_stage_index: Index of the stage that needs the
            extra worker.
        receiver_stage_name: Name of the receiver stage. Used for
            the receiver-side anti-flap lookup.
        stage_names: Stage names in problem order; the i-th entry
            is the name of the stage at index ``i``.
        stage_floors: Per-stage donor floors. A missing entry
            defaults to ``1``.
        worker_ids_by_stage: Per-stage live worker ids in problem
            order. Each inner list is the snapshot the planner
            currently holds for that stage.
        worker_ages: Cluster-wide worker ages keyed by worker id.
            Missing entries default to ``0``.
        worker_nodes: Mapping ``worker_id -> node_id`` driving the
            same-node-first iteration order in
            :func:`_resource_fit_plan`. An empty mapping degrades
            to flat iteration (no locality preference) so callers
            without per-worker node info still get a feasible
            search, just without locality-based heuristics.
        stage_states: Per-stage runtime state keyed by stage name.
            Drives the classifier and growth-mode checks.
        config: Cluster-wide configuration.
        stage_configs: Per-stage effective configs keyed by stage
            name. Drives the streak threshold.
        cycle: Current monotonic cycle number, against which the
            anti-flap cooldown is evaluated.
        last_donation_cycle: Per-stage record of the cycle at which
            each stage most recently donated. Read only by the
            receiver-was-recent-donor anti-flap gate.
        ctx: Active autoscale planner context. Only
            ``probe_add_after_removals`` is invoked on it; the
            helper is non-mutating with respect to ``ctx``.
        receiver_intent: Receiver stage's Phase C grow intent
            (signed worker delta) used by
            :func:`_receiver_value` to weight intent in the
            economic gate's spread computation.
        d_k_now: Cycle's actor-normalized service demand mapping
            keyed by stage name. Drives the bottleneck-severity
            term in :func:`_receiver_value`, the post-plan
            simulation in :func:`_compute_post_plan_d_k`, and the
            throughput / donor-flip / balance comparisons.
        effective_capacities: Per-stage effective ready capacity
            (counts non-warmup ready actors per
            :meth:`SaturationAwareScheduler._effective_ready_capacity`).
            Drives :func:`_compute_post_plan_d_k`'s capacity-after
            arithmetic.
        s_k_ewma: Per-stage intrinsic service-time EWMA. Held
            fixed across the post-plan simulation; the donor plan
            cannot retroactively change a stage's measured
            per-task service time within a single cycle.
        excluded_worker_ids: Optional set of worker ids to drop
            from the candidate pool before donor selection.
            Saturation-aware callers populate this with the donor
            warmup grace set so freshly-warmed workers are not
            yanked off their stage before they have had a chance to
            absorb load. The donor stage itself remains eligible if
            any of its other workers are mature; donor-stage
            elimination only happens when every one of the stage's
            workers is in warmup. ``None`` or an empty set leaves
            the candidate pool unfiltered.

    Returns:
        A ``DonorDecision`` carrying the accepted ``DonorPlan`` and
        the gate metrics that justified it; the caller commits
        ``decision.plan`` atomically and uses ``decision.gate_result``
        + ``decision.receiver_capacity_before`` to render the
        structured INFO commit log. ``None`` when the master toggle
        is disabled, the receiver is in cooldown, no donor stage
        passes every filter, every potential donor worker is in
        ``excluded_worker_ids``, no probe up to ``max_plan_size``
        reports feasibility, or the economic gate rejects the
        chosen plan; every rejection emits its own DEBUG line.
        Plans are emitted at the smallest feasible plan_size;
        multi-worker plans only appear when single-worker probes
        were all infeasible.

    Raises:
        ValueError: If ``stage_names`` and ``worker_ids_by_stage``
            disagree in length, or if ``receiver_stage_index`` is
            outside ``[0, len(worker_ids_by_stage))``. Either
            condition would otherwise surface as an ``IndexError``
            mid-loop or as silently wrong donor selection far from
            the misalignment site.

    """
    if len(stage_names) != len(worker_ids_by_stage):
        msg = (
            "stage_names and worker_ids_by_stage must align in length, "
            f"got len(stage_names)={len(stage_names)} vs "
            f"len(worker_ids_by_stage)={len(worker_ids_by_stage)}"
        )
        raise ValueError(msg)
    if not 0 <= receiver_stage_index < len(worker_ids_by_stage):
        msg = (
            f"receiver_stage_index={receiver_stage_index} is out of bounds for "
            f"len(worker_ids_by_stage)={len(worker_ids_by_stage)}"
        )
        raise ValueError(msg)

    # Common decision-log inputs for the early-return paths.
    receiver_state = stage_states.get(receiver_stage_name)
    receiver_d_k = d_k_now.get(receiver_stage_name, math.nan)
    capacity_before = effective_capacities.get(receiver_stage_name, 0)

    def _emit_reject(
        reason: RejectReason,
        *,
        plan: DonorPlan | None = None,
        gate_result: _GateResult | None = None,
        placement_reject_reason: str = "",
    ) -> None:
        capacity_after = capacity_before + (len(plan.removals) if plan is not None else 0)
        logger.debug(
            _format_donor_decision_log(
                receiver_name=receiver_stage_name,
                receiver_state=receiver_state,
                receiver_d_k=receiver_d_k,
                receiver_intent=receiver_intent,
                capacity_before=capacity_before,
                capacity_after=capacity_after,
                plan=plan,
                stage_names=stage_names,
                gate_result=gate_result,
                spread_threshold=config.cross_stage_donor_spread_threshold,
                reject_reason=reason.value,
                placement_reject_reason=placement_reject_reason,
            )
        )

    if not config.enable_cross_stage_donor:
        _emit_reject(RejectReason.MASTER_TOGGLE_OFF)
        return None

    receiver_anti_flap_cycle = last_donation_cycle.get(receiver_stage_name)
    if (
        receiver_anti_flap_cycle is not None
        and cycle - receiver_anti_flap_cycle < config.cross_stage_donor_anti_flap_cycles
    ):
        _emit_reject(RejectReason.RECEIVER_ANTI_FLAP)
        return None

    eligible_stages: list[int] = []
    for donor_index, donor_workers in enumerate(worker_ids_by_stage):
        if donor_index == receiver_stage_index:
            continue
        if config.donor_must_be_strictly_upstream and donor_index >= receiver_stage_index:
            continue
        if len(donor_workers) - 1 < stage_floors.get(donor_index, 1):
            continue

        donor_name = stage_names[donor_index]
        donor_state = stage_states.get(donor_name)
        if donor_state is None:
            continue
        if config.cross_stage_donor_require_over_provisioned:
            donor_cfg = stage_configs.get(donor_name)
            if donor_cfg is None:
                continue
            if donor_state.classifier_state is not StageState.OVER_PROVISIONED:
                continue
            if donor_state.classifier_streak < donor_cfg.over_provisioned_streak_min_cycles:
                continue
        if config.cross_stage_donor_exclude_hold_state and donor_state.growth_mode is GrowthMode.HOLD:
            continue

        # Layer 4: signal-trust pre-filter. Donor stages whose
        # classifier signal is too noisy fail the trust gate even if
        # they cleared every prior gate; a flickering OVER_PROVISIONED
        # signal is unreliable evidence of true idle capacity.
        trust = _signal_trust(
            donor_state,
            trust_streak_cap=config.cross_stage_donor_trust_streak_cap,
        )
        if trust < config.cross_stage_donor_min_trust:
            continue

        eligible_stages.append(donor_index)

    if not eligible_stages:
        _emit_reject(RejectReason.NO_CANDIDATES)
        return None

    excluded = excluded_worker_ids or frozenset()
    candidates = [
        DonorWorker(
            stage_index=donor_index,
            worker_id=wid,
            age=worker_ages.get(wid, 0),
        )
        for donor_index in eligible_stages
        for wid in worker_ids_by_stage[donor_index]
        if wid not in excluded
    ]
    if not candidates:
        _emit_reject(RejectReason.NO_CANDIDATES)
        return None

    # Per-stage donor budget. Mirrors the floor-mode caller's
    # convention (``len(workers) - floor``) so a multi-donor saturation
    # plan cannot pull more workers from one donor stage than that
    # stage can spare without violating its own floor. The budget is
    # computed from the planner's full live worker count for the
    # stage; ``excluded_worker_ids`` only narrows the candidate pool
    # (warmup grace), it does not raise the floor.
    removable_by_stage = {
        donor_index: max(0, len(worker_ids_by_stage[donor_index]) - stage_floors.get(donor_index, 1))
        for donor_index in eligible_stages
    }

    plan = _resource_fit_plan(
        receiver_stage_index=receiver_stage_index,
        candidates=candidates,
        worker_nodes=worker_nodes,
        ctx=ctx,
        max_plan_size=config.cross_stage_donor_max_plan_size,
        max_plan_combinations=config.cross_stage_donor_max_plan_combinations,
        removable_by_stage=removable_by_stage,
    )
    if plan is None:
        _emit_reject(RejectReason.RESOURCE_FIT)
        return None

    gate_result = _evaluate_economic_gate(
        plan=plan,
        stage_names=stage_names,
        stage_states=stage_states,
        receiver_intent=receiver_intent,
        d_k_now=d_k_now,
        effective_capacities=effective_capacities,
        s_k_ewma=s_k_ewma,
        slots_per_worker_by_stage=slots_per_worker_by_stage,
        config=config,
    )
    if gate_result.reject_reason is not None:
        # The structured INFO commit log fires from the saturation_aware
        # caller after the atomic commit succeeds; this DEBUG line is
        # the rejection counterpart that names the failing gate so
        # operators can correlate the reject reason with the metrics
        # that drove it.
        _emit_reject(
            gate_result.reject_reason,
            plan=plan,
            gate_result=gate_result,
        )
        return None

    return DonorDecision(
        plan=plan,
        gate_result=gate_result,
        receiver_capacity_before=capacity_before,
    )
