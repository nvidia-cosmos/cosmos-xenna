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

"""Pipeline throughput / capacity model for the saturation-aware scheduler.

Single source of truth for the scheduler's steady-state reasoning. For a
linear pipeline the sustainable throughput equals the slowest stage's
source-rate capacity ``bottleneck_rate = min_k cap_src[k]`` (the global
bottleneck), so no stage can usefully outrun ``bottleneck_rate``. From that one
fact this module derives two per-stage worker targets that the grow
(``sizing.py``) and shrink (``floor.py``) consumers share:

- ``w_sustain``: workers to sustain the smoothed ``bottleneck_rate`` (the
  scale-down hold target; no headroom).
- ``w_target``: workers worth asking the solver for this cycle. The bottleneck
  climbs toward ``next_bottleneck_rate`` - the second-slowest stage's capacity,
  i.e. the move that actually raises pipeline speed; every other stage,
  including the source, is bounded to ``bottleneck_rate * (1 +
  capacity_headroom)`` (a small read-ahead, never free growth).

Bottleneck identity is sticky (hysteresis) so a one-cycle ``cap_src`` dip from
a floor cut cannot drag every stage's target down. The smoothed arrival rate
``a_ewma`` and the bottleneck identity persist across cycles inside
:class:`CapacityModel`.
"""

import enum
import math
from collections.abc import Sequence
from typing import Any, Self, cast

import attrs


class FeederReason(enum.StrEnum):
    """Feeder-pressure reason values emitted in scheduler logs."""

    BOOSTED = "boosted"
    PENDING_CONFIRM = "pending-confirm"
    NO_BOOST_IMMINENT_ARRIVAL = "no-boost-imminent-arrival"
    NO_BOOST_GLOBAL_BOTTLENECK = "no-boost-global-bottleneck"
    NO_BOOST_MANUAL_FEEDER = "no-boost-manual-feeder"
    NO_BOOST_INVALID_SUPPLY = "no-boost-invalid-supply"
    NO_BOOST_FEEDER_SUFFICIENT = "no-boost-feeder-sufficient"
    CLEARED_LOCAL_INPUT = "cleared-local-input"
    CLEARED_NOT_WARM = "cleared-not-warm"


class FeederCandidateStatus(enum.StrEnum):
    """Selection status for one upstream feeder-pressure candidate."""

    ACTIONABLE = "actionable"
    IMMINENT = "imminent"
    GLOBAL_BOTTLENECK = "global-bottleneck"
    MANUAL = "manual"


@attrs.frozen
class FeederCandidate:
    """One upstream feeder candidate ranked by drain delay."""

    stage: int
    delay_s: float
    status: FeederCandidateStatus


@attrs.frozen
class CapacityParams:
    """Static tuning for the capacity model.

    Attributes:
        alpha_up: EWMA weight when the sustainable arrival rises (fast
            re-protect).
        alpha_down_cpu: EWMA weight when it falls, for CPU stages.
        alpha_down_gpu: EWMA weight when it falls, for GPU stages
            (smaller = slower release, since warmup is costly).
        capacity_headroom: Spare-capacity fraction added to the bottleneck
            rate for the bounded read-ahead / tie-break growth target.
        hysteresis_margin: A challenger must be at least this much slower
            (lower ``cap_src``) than the incumbent bottleneck to be eligible
            to take over.
        switch_confirm: Consecutive eligible cycles before the bottleneck
            identity switches to the challenger.
        feeder_pressure_confirm: Consecutive starved-warm cycles before a
            downstream stage may boost its binding feeder.
        feeder_arrival_horizon_s: Delay threshold below which upstream arrivals
            are considered imminent and no feeder boost is needed.
        feeder_boost_max_multiplier: Maximum feeder-pressure target multiplier
            relative to the normal capacity target.
        min_workers: Floor a stage's targets never fall below while active.
    """

    alpha_up: float
    alpha_down_cpu: float
    alpha_down_gpu: float
    capacity_headroom: float
    hysteresis_margin: float
    switch_confirm: int
    feeder_pressure_confirm: int = 2
    feeder_arrival_horizon_s: float = 10.0
    feeder_boost_max_multiplier: float = 2.0
    min_workers: int = 1


@attrs.frozen
class CapacityState:
    """Cross-cycle capacity state, indexed by stage.

    Attributes:
        a_ewma: Smoothed sustainable arrival per stage; ``None`` until the
            first observation (then initialized to that value).
        target_speed_ewma: Smoothed speed used for worker-target math; ``None``
            until a trusted speed has been observed.
        bottleneck: Incumbent global-bottleneck index, or ``-1`` when no
            stage is measured yet.
        bottleneck_streak: Consecutive cycles a margin-clearing challenger
            has beaten the incumbent (resets on any hold).
        feeder_pressure_streak: Consecutive delayed starved-warm cycles per
            downstream stage.
    """

    a_ewma: tuple[float | None, ...]
    target_speed_ewma: tuple[float | None, ...]
    bottleneck: int
    bottleneck_streak: int
    feeder_pressure_streak: tuple[int, ...]

    @classmethod
    def initial(cls, num_stages: int) -> Self:
        """Return empty state for a pipeline with ``num_stages`` stages."""
        return cls(
            a_ewma=(None,) * num_stages,
            target_speed_ewma=(None,) * num_stages,
            bottleneck=-1,
            bottleneck_streak=0,
            feeder_pressure_streak=(0,) * num_stages,
        )


@attrs.frozen
class CapacityInputs:
    """Per-cycle observed inputs for the capacity model, indexed by stage.

    Attributes:
        workers: Current (observed pre-solve) worker count per stage.
        speed: Trusted per-worker throughput per stage, in stage-input
            items/s; ``0.0`` for a cold / untrusted stage (excluded from the
            bottleneck).
        chain: Chain factors from :func:`chain.chain_factors`.
        is_gpu: Whether each stage holds GPU workers (selects the slower
            release alpha).
        is_manual: Whether each stage has an operator-pinned worker count.
        local_qin: Inter-stage input queue depth per stage, in stage-input
            samples.
        local_pending_depth: Local pending work per stage, excluding in-flight
            work, in stage-input samples.
        local_input_threshold: Per-stage local-input threshold, in stage-input
            samples.
        active_depth: Local active work per stage, including in-flight work, in
            stage-input samples.
        ready_workers: Workers not currently holding in-flight slots.
    """

    workers: tuple[int, ...]
    speed: tuple[float, ...]
    chain: tuple[float, ...]
    is_gpu: tuple[bool, ...]
    is_manual: tuple[bool, ...]
    local_qin: tuple[float, ...]
    local_pending_depth: tuple[float, ...]
    local_input_threshold: tuple[float, ...]
    active_depth: tuple[float, ...]
    ready_workers: tuple[int, ...]


@attrs.frozen
class StageCapacity:
    """One stage's capacity facts for this cycle.

    Attributes:
        speed: Trusted per-worker speed used this cycle (``0.0`` when cold);
            carried so the scheduler's logs stay self-contained.
        target_speed: Smoothed per-worker speed used for ``w_sustain``,
            ``w_target``, and solver demand sizing.
        cap_src: Source-rate capacity ``workers * speed / chain`` (``0.0``
            when cold).
        a_raw: Sustainable arrival before smoothing (``chain * bottleneck_rate``).
        a_ewma: Asymmetrically smoothed sustainable arrival used this cycle.
        w_sustain: Hold / scale-down target ``ceil(a_ewma / target_speed)``
            (matched to ``bottleneck_rate``, no headroom).
        w_target: Useful growth target this cycle (matched to
            ``next_bottleneck_rate`` for the bottleneck stage, to
            ``bottleneck_rate + headroom`` for every other stage), never below
            ``w_sustain``.
        starved_warm: Whether the stage has ready capacity but no local pending
            input.
        suppress_growth: Whether demand sizing must avoid growing this stage
            this cycle.
        binding_feeder: Upstream stage chosen as this stage's binding feeder,
            or ``-1`` when none was selected.
        feeder_path_delay_s: Estimated upstream delay for the selected feeder.
        blocked_feeder: Highest-signal unboostable upstream stage, or ``-1``.
        blocked_feeder_reason: Why ``blocked_feeder`` cannot be boosted.
        blocked_feeder_path_delay_s: Drain delay for ``blocked_feeder``.
        feeder_streak: Confirmation streak for this downstream request.
        feeder_required_workers: Uncapped worker target from feeder pressure
            (the max of the drain / demand / refill components below).
        feeder_drain_workers: Component sizing the feeder to drain its own
            active backlog within the effective horizon.
        feeder_demand_workers: Component sizing the feeder to match the
            downstream stage's ready-worker consumption rate.
        feeder_queue_refill_workers: Component sizing the feeder to refill the
            downstream input buffer to one batch per ready worker.
        downstream_buffer_deficit: Downstream input-buffer shortfall below one
            batch per ready worker, in stage-input samples.
        feeder_effective_horizon_s: Arrival horizon used for this downstream
            request; reduced while the downstream is under-buffered.
        feeder_boost_cap: Cap applied to the feeder-pressure target.
        feeder_boost: Additional target workers applied to this feeder.
        feeder_downstreams: Downstream stages whose requests were aggregated
            into this feeder's boost.
        feeder_candidates: Ranked upstream candidates considered for this
            downstream request.
        feeder_reason: Searchable feeder-pressure reason string for logs.
    """

    speed: float
    target_speed: float
    cap_src: float
    a_raw: float
    a_ewma: float
    w_sustain: int
    w_target: int
    starved_warm: bool = False
    suppress_growth: bool = False
    binding_feeder: int = -1
    feeder_path_delay_s: float = 0.0
    blocked_feeder: int = -1
    blocked_feeder_reason: str = ""
    blocked_feeder_path_delay_s: float = 0.0
    feeder_streak: int = 0
    feeder_required_workers: int = 0
    feeder_drain_workers: int = 0
    feeder_demand_workers: int = 0
    feeder_queue_refill_workers: int = 0
    downstream_buffer_deficit: float = 0.0
    feeder_effective_horizon_s: float = 0.0
    feeder_boost_cap: int = 0
    feeder_boost: int = 0
    feeder_downstreams: tuple[int, ...] = ()
    feeder_candidates: tuple[FeederCandidate, ...] = ()
    feeder_reason: str = ""


@attrs.frozen
class CapacityPlan:
    """Per-cycle capacity output consumed by sizing and the floor.

    Attributes:
        stages: One :class:`StageCapacity` per stage, in pipeline order.
        bottleneck_stage: Index of the sticky global bottleneck (identity
            only; may differ from the current argmin during a hold window), or
            ``-1`` when no stage is measured yet.
        bottleneck_rate: Pipeline throughput: the measured minimum ``cap_src``
            this cycle (``min_k cap_src[k]``), in source items/s. Only the
            bottleneck identity is sticky, not this rate.
        next_bottleneck_rate: The second-minimum ``cap_src`` (excluding the
            measured argmin) - the rate the bottleneck can usefully climb
            toward (falls back to ``bottleneck_rate`` when fewer than two
            stages are measured).
        bottleneck_streak: Current challenger confirmation streak.
        bottleneck_candidate: Current minimum-capacity measured stage.
        bottleneck_candidate_rate: Candidate stage's source-rate capacity.
    """

    stages: tuple[StageCapacity, ...]
    bottleneck_stage: int
    bottleneck_rate: float
    next_bottleneck_rate: float
    bottleneck_streak: int = 0
    bottleneck_candidate: int = -1
    bottleneck_candidate_rate: float = 0.0


@attrs.frozen
class CapacityResult:
    """A cycle's :class:`CapacityPlan` plus the state for the next cycle.

    Attributes:
        plan: This cycle's per-stage capacity and the bottleneck ladder.
        state: Updated :class:`CapacityState` to carry into the next cycle.
    """

    plan: CapacityPlan
    state: CapacityState


def asymmetric_ewma(prev: float | None, raw: float, alpha_up: float, alpha_down: float) -> float:
    """Smooth ``raw`` with a fast-up / slow-down EWMA.

    Initializes to ``raw`` on the first sample so cold start is not blunted.
    Rising values use ``alpha_up`` (re-protect quickly); falling values use
    ``alpha_down`` (release cautiously).
    """
    if prev is None:
        return raw
    alpha = alpha_up if raw >= prev else alpha_down
    return alpha * raw + (1.0 - alpha) * prev


def _source_capacities(workers: Sequence[int], speed: Sequence[float], chain: Sequence[float]) -> list[float]:
    """Return per-stage capacity ``w * s / k`` in source items/s (``0`` cold)."""
    return [
        worker_count * speed_k / factor if factor > 0.0 else 0.0
        for worker_count, speed_k, factor in zip(workers, speed, chain, strict=True)
    ]


def _select_bottleneck(
    cap_src: Sequence[float], prev_bn: int, prev_streak: int, margin: float, confirm: int
) -> tuple[int, int, int, float]:
    """Return the sticky bottleneck identity plus current challenger facts.

    Holds the incumbent unless a different measured stage is the minimum AND
    lower by ``margin`` for ``confirm`` consecutive cycles. Only the bottleneck
    IDENTITY is sticky; the caller derives the sizing rate from the measured-min
    candidate (``cap_src`` argmin), so a transient incumbent spike cannot inflate
    the pipeline rate.

    Returns:
        ``(selected_stage, streak, candidate_stage, candidate_cap)`` where
        ``selected_stage`` is the sticky incumbent identity and ``candidate_*``
        is the current measured argmin / minimum ``cap_src``.
    """
    measured = [(index, capacity) for index, capacity in enumerate(cap_src) if capacity > 0.0]
    if not measured:
        return -1, 0, -1, 0.0
    challenger, challenger_cap = min(measured, key=lambda item: item[1])
    if prev_bn < 0 or cap_src[prev_bn] <= 0.0:
        # No incumbent, or the incumbent went cold: adopt the current minimum.
        return challenger, 0, challenger, challenger_cap
    incumbent_cap = cap_src[prev_bn]
    decisive = challenger != prev_bn and challenger_cap < incumbent_cap * (1.0 - margin)
    if not decisive:
        # Incumbent still (near-)slowest: hold it and reset the streak.
        return prev_bn, 0, challenger, challenger_cap
    streak = prev_streak + 1
    if streak >= confirm:
        # A challenger has been decisively slower long enough: switch.
        return challenger, 0, challenger, challenger_cap
    # Decisive this cycle but not yet confirmed: hold, accruing confirmation.
    return prev_bn, streak, challenger, challenger_cap


def _second_min_capacity(cap_src: Sequence[float], exclude: int, fallback: float) -> float:
    """Return the smallest measured ``cap_src`` other than ``exclude``.

    Falls back to ``fallback`` (the bottleneck rate) when fewer than two stages
    are measured, so a single-stage pipeline has
    ``next_bottleneck_rate == bottleneck_rate``.
    """
    others = [capacity for index, capacity in enumerate(cap_src) if capacity > 0.0 and index != exclude]
    return min(others) if others else fallback


def _stage_completion_rate(workers: int, speed: float) -> float:
    """Return stage completion rate for trusted positive inputs."""
    if workers <= 0 or speed <= 0.0:
        return 0.0
    return workers * speed


def _drain_time(active_depth: float, workers: int, speed: float) -> float:
    """Return seconds needed to drain active depth at current capacity."""
    if active_depth <= 0.0:
        return 0.0
    completion_rate = _stage_completion_rate(workers, speed)
    if completion_rate <= 0.0:
        return math.inf
    return active_depth / completion_rate


def _classify_feeder(
    index: int,
    delay_s: float,
    effective_horizon_s: float,
    bottleneck_stage: int,
    inputs: CapacityInputs,
) -> FeederCandidateStatus:
    """Classify whether an upstream feeder can receive autoscaler pressure."""
    if delay_s <= effective_horizon_s:
        return FeederCandidateStatus.IMMINENT
    if index == bottleneck_stage:
        return FeederCandidateStatus.GLOBAL_BOTTLENECK
    if inputs.is_manual[index]:
        return FeederCandidateStatus.MANUAL
    return FeederCandidateStatus.ACTIONABLE


def _feeder_candidates(
    downstream: int,
    inputs: CapacityInputs,
    bottleneck_stage: int,
    effective_horizon_s: float,
) -> tuple[FeederCandidate, ...]:
    """Return upstream feeder candidates sorted by descending drain delay."""
    candidates: list[FeederCandidate] = []
    for index in range(downstream):
        if inputs.chain[index] <= 0.0 or inputs.active_depth[index] <= 0.0:
            continue
        delay = _drain_time(inputs.active_depth[index], inputs.workers[index], inputs.speed[index])
        if not math.isfinite(delay):
            continue
        status = _classify_feeder(index, delay, effective_horizon_s, bottleneck_stage, inputs)
        candidates.append(FeederCandidate(stage=index, delay_s=delay, status=status))
    return tuple(sorted(candidates, key=lambda candidate: candidate.delay_s, reverse=True))


def _select_feeder(candidates: Sequence[FeederCandidate]) -> tuple[FeederCandidate | None, FeederCandidate | None]:
    """Return the actionable feeder and the highest-delay blocked fallback."""
    selected = None
    blocked = None
    for candidate in candidates:
        if candidate.status is FeederCandidateStatus.ACTIONABLE:
            if selected is None:
                selected = candidate
            continue
        if blocked is None:
            blocked = candidate
    return selected, blocked


def _blocked_feeder_reason(status: FeederCandidateStatus) -> FeederReason:
    """Return the log reason for a blocked feeder candidate."""
    if status is FeederCandidateStatus.IMMINENT:
        return FeederReason.NO_BOOST_IMMINENT_ARRIVAL
    if status is FeederCandidateStatus.GLOBAL_BOTTLENECK:
        return FeederReason.NO_BOOST_GLOBAL_BOTTLENECK
    if status is FeederCandidateStatus.MANUAL:
        return FeederReason.NO_BOOST_MANUAL_FEEDER
    msg = f"actionable candidate is not blocked: {status}"
    raise ValueError(msg)


def _target_speed_for_cycle(
    prev_target_speed: float | None,
    raw_speed: float,
    alpha_up: float,
    alpha_down: float,
) -> tuple[float, float | None]:
    """Return the smoothed target speed and next persisted speed sample."""
    if raw_speed <= 0.0:
        return 0.0, prev_target_speed
    target_speed = asymmetric_ewma(prev_target_speed, raw_speed, alpha_up, alpha_down)
    return target_speed, target_speed


def _is_starved_warm(index: int, inputs: CapacityInputs, stages: Sequence[StageCapacity]) -> bool:
    """Return whether a locally dry stage has enough ready workers."""
    if inputs.workers[index] <= 0:
        return False
    ready_threshold = min(inputs.workers[index], stages[index].w_sustain)
    return (
        inputs.local_pending_depth[index] <= inputs.local_input_threshold[index]
        and inputs.ready_workers[index] >= ready_threshold
    )


def _feeder_boost_cap(base_target: int, multiplier: float) -> int:
    """Return the capped target allowed by feeder pressure."""
    return max(base_target + 1, math.ceil(base_target * multiplier))


def _required_feeder_workers(
    candidate: FeederCandidate,
    downstream: int,
    stages: Sequence[StageCapacity],
    inputs: CapacityInputs,
    horizon_s: float,
    buffer_deficit: float,
) -> tuple[int, int, int]:
    """Return feeder workers to (drain backlog, meet demand, refill buffer).

    The caller takes ``max`` of the three for the feeder target and passes the
    same ``buffer_deficit`` it logged, so the refill component cannot diverge
    from the reported deficit. Components are sized so a warm, locally dry
    downstream gets enough upstream supply to feed its ready workers and refill
    its dispatch buffer within ``horizon_s``; downstream item rates are converted
    to feeder item rates through chain factors. A non-positive divisor yields
    ``0`` for that component instead of raising.

    ::

        drain  = feeder backlog / horizon
        demand = downstream ready-worker rate -> feeder rate
        refill = downstream buffer deficit / horizon -> feeder rate

    Returns:
        ``(drain_workers, demand_workers, refill_workers)``.
    """
    feeder = candidate.stage
    feeder_speed = inputs.speed[feeder]
    if feeder_speed <= 0.0 or horizon_s <= 0.0:
        return 0, 0, 0

    drain_workers = math.ceil(inputs.active_depth[feeder] / (feeder_speed * horizon_s))

    feeder_chain = inputs.chain[feeder]
    downstream_chain = inputs.chain[downstream]
    if downstream_chain <= 0.0 or feeder_chain <= 0.0:
        return drain_workers, 0, 0

    # Downstream consumption the feeder must match right now, converted from the
    # downstream item rate back to the feeder item rate through chain factors.
    downstream_rate = inputs.ready_workers[downstream] * stages[downstream].target_speed
    feeder_rate_for_demand = downstream_rate / downstream_chain * feeder_chain
    demand_workers = math.ceil(feeder_rate_for_demand / feeder_speed)

    # Extra supply to rebuild the downstream buffer (one batch per ready worker)
    # so it does not immediately go dry again after consuming arriving supply.
    feeder_rate_for_refill = buffer_deficit / horizon_s / downstream_chain * feeder_chain
    refill_workers = math.ceil(feeder_rate_for_refill / feeder_speed)

    return drain_workers, demand_workers, refill_workers


def _mark_blocked_candidate(
    updates: dict[str, object],
    blocked: FeederCandidate,
    *,
    record_reason: bool,
) -> None:
    """Record a blocked feeder candidate for observability."""
    updates["blocked_feeder"] = blocked.stage
    updates["blocked_feeder_reason"] = blocked.status.value
    updates["blocked_feeder_path_delay_s"] = blocked.delay_s
    if record_reason:
        updates["feeder_reason"] = _blocked_feeder_reason(blocked.status).value


def _mark_selected_candidate(
    updates: dict[str, object],
    selected: FeederCandidate,
) -> None:
    """Record the selected actionable feeder for observability."""
    updates["binding_feeder"] = selected.stage
    updates["feeder_path_delay_s"] = selected.delay_s


def _mark_candidate_set(
    updates: dict[str, object],
    selected: FeederCandidate | None,
    blocked: FeederCandidate | None,
    candidates: tuple[FeederCandidate, ...],
) -> None:
    """Record selected or blocked feeder candidates on a downstream stage."""
    updates["feeder_candidates"] = candidates
    if selected is not None:
        _mark_selected_candidate(updates, selected)
    if blocked is not None:
        _mark_blocked_candidate(updates, blocked, record_reason=selected is None)


def _apply_feeder_pressure(
    stages: Sequence[StageCapacity],
    inputs: CapacityInputs,
    prev: CapacityState,
    params: CapacityParams,
    bottleneck_stage: int,
) -> tuple[tuple[StageCapacity, ...], tuple[int, ...]]:
    """Apply bounded feeder-pressure targets to warm, locally starved stages."""
    requested_targets = [stage.w_target for stage in stages]
    next_streak = [0] * len(stages)
    stage_updates: list[dict[str, object]] = [{} for _ in stages]
    downstreams_by_feeder: list[list[int]] = [[] for _ in stages]

    for downstream in range(len(stages)):
        if not _is_starved_warm(downstream, inputs, stages):
            if prev.feeder_pressure_streak[downstream] > 0:
                reason = (
                    FeederReason.CLEARED_LOCAL_INPUT
                    if inputs.local_pending_depth[downstream] > inputs.local_input_threshold[downstream]
                    else FeederReason.CLEARED_NOT_WARM
                )
                stage_updates[downstream]["feeder_reason"] = reason.value
            continue

        stage_updates[downstream]["starved_warm"] = True

        # Downstream-only demand signals, computed before feeder classification so
        # an under-buffered downstream uses a stricter (halved) arrival horizon.
        buffer_target = inputs.ready_workers[downstream] * inputs.local_input_threshold[downstream]
        buffer_deficit = max(buffer_target - inputs.local_pending_depth[downstream], 0.0)
        underbuffered = inputs.ready_workers[downstream] > 0 and buffer_deficit > 0.0
        effective_horizon_s = params.feeder_arrival_horizon_s * (0.5 if underbuffered else 1.0)
        stage_updates[downstream]["downstream_buffer_deficit"] = buffer_deficit
        stage_updates[downstream]["feeder_effective_horizon_s"] = effective_horizon_s

        candidates = _feeder_candidates(downstream, inputs, bottleneck_stage, effective_horizon_s)
        selected, blocked = _select_feeder(candidates)
        _mark_candidate_set(stage_updates[downstream], selected, blocked, candidates)
        if selected is None and blocked is None:
            stage_updates[downstream]["feeder_reason"] = FeederReason.NO_BOOST_INVALID_SUPPLY.value
            continue

        stage_updates[downstream]["suppress_growth"] = True
        candidate = selected if selected is not None else blocked
        if candidate is None:
            stage_updates[downstream]["feeder_reason"] = FeederReason.NO_BOOST_INVALID_SUPPLY.value
            continue
        if candidate.status is FeederCandidateStatus.IMMINENT:
            continue

        if selected is None:
            # Blocked-only (global bottleneck / manual): reason already recorded
            # by _mark_candidate_set; do not advance the confirmation streak.
            continue

        next_streak[downstream] = prev.feeder_pressure_streak[downstream] + 1
        stage_updates[downstream]["feeder_streak"] = next_streak[downstream]
        if next_streak[downstream] < params.feeder_pressure_confirm:
            stage_updates[downstream]["feeder_reason"] = FeederReason.PENDING_CONFIRM.value
            continue

        feeder = selected.stage
        base_target = stages[feeder].w_target
        drain_w, demand_w, refill_w = _required_feeder_workers(
            selected, downstream, stages, inputs, effective_horizon_s, buffer_deficit
        )
        required_workers = max(drain_w, demand_w, refill_w)
        boost_cap = _feeder_boost_cap(base_target, params.feeder_boost_max_multiplier)
        final_target = max(base_target, min(required_workers, boost_cap))
        requested_targets[feeder] = max(requested_targets[feeder], final_target)
        downstreams_by_feeder[feeder].append(downstream)
        stage_updates[downstream]["feeder_required_workers"] = required_workers
        stage_updates[downstream]["feeder_drain_workers"] = drain_w
        stage_updates[downstream]["feeder_demand_workers"] = demand_w
        stage_updates[downstream]["feeder_queue_refill_workers"] = refill_w
        stage_updates[downstream]["feeder_boost_cap"] = boost_cap
        stage_updates[downstream]["feeder_reason"] = (
            FeederReason.BOOSTED.value if final_target > base_target else FeederReason.NO_BOOST_FEEDER_SUFFICIENT.value
        )

    updated_stages: list[StageCapacity] = []
    for index, stage in enumerate(stages):
        updates = stage_updates[index]
        target = requested_targets[index]
        if target > stage.w_target:
            updates["w_target"] = target
            updates["feeder_boost"] = target - stage.w_target
            updates["feeder_downstreams"] = tuple(downstreams_by_feeder[index])
            updates["feeder_reason"] = FeederReason.BOOSTED.value
        updated_stages.append(attrs.evolve(stage, **cast(Any, updates)))
    return tuple(updated_stages), tuple(next_streak)


def compute_capacity(inputs: CapacityInputs, prev: CapacityState, params: CapacityParams) -> CapacityResult:
    """Compute the capacity plan for one cycle and the next-cycle state.

    Identifies the sticky bottleneck identity, derives the measured-min
    ``bottleneck_rate`` and the ``next_bottleneck_rate``, smooths each stage's
    bottleneck-matched arrival,
    and derives the hold target ``w_sustain`` and the growth target
    ``w_target``. Cold / untrusted stages (``speed <= 0`` or ``chain <= 0``)
    and an all-cold pipeline (``bottleneck_rate <= 0``) yield ``min_workers``
    targets and are excluded from the bottleneck, so the cold-start ramp keeps
    owning them.

    Args:
        inputs: Per-cycle observed per-stage inputs.
        prev: State carried from the previous cycle.
        params: Static smoothing / hysteresis / headroom tuning.

    Returns:
        The cycle's :class:`CapacityPlan` plus the :class:`CapacityState` to
        carry into the next cycle.

    Raises:
        ValueError: If the input tuples or the previous state differ in
            length from ``workers``.
    """
    num_stages = len(inputs.workers)
    if not (
        len(inputs.speed)
        == len(inputs.chain)
        == len(inputs.is_gpu)
        == len(inputs.is_manual)
        == len(inputs.local_qin)
        == len(inputs.local_pending_depth)
        == len(inputs.local_input_threshold)
        == len(inputs.active_depth)
        == len(inputs.ready_workers)
        == len(prev.a_ewma)
        == len(prev.target_speed_ewma)
        == len(prev.feeder_pressure_streak)
        == num_stages
    ):
        raise ValueError(
            "capacity inputs length mismatch: "
            f"workers={num_stages} speed={len(inputs.speed)} chain={len(inputs.chain)} "
            f"is_gpu={len(inputs.is_gpu)} is_manual={len(inputs.is_manual)} local_qin={len(inputs.local_qin)} "
            f"local_pending_depth={len(inputs.local_pending_depth)} "
            f"local_input_threshold={len(inputs.local_input_threshold)} active_depth={len(inputs.active_depth)} "
            f"ready_workers={len(inputs.ready_workers)} prev_a_ewma={len(prev.a_ewma)} "
            f"prev_target_speed_ewma={len(prev.target_speed_ewma)} "
            f"prev_feeder_pressure_streak={len(prev.feeder_pressure_streak)}"
        )

    cap_src = _source_capacities(inputs.workers, inputs.speed, inputs.chain)
    bottleneck_stage, bottleneck_streak, bottleneck_candidate, bottleneck_candidate_rate = _select_bottleneck(
        cap_src, prev.bottleneck, prev.bottleneck_streak, params.hysteresis_margin, params.switch_confirm
    )
    # The sizing rate is the TRUE measured-min cap_src (the module's definition of
    # the pipeline bottleneck), never the held incumbent's cap_src. A held incumbent
    # whose cap_src transiently spikes can no longer broadcast that spike to every
    # stage via a_raw. The sticky identity (bottleneck_stage) is consulted only for
    # the growth-owner branch (k == bottleneck_stage) below, so the growth target
    # still does not flap on a one-cycle dip.
    bottleneck_rate = bottleneck_candidate_rate
    next_bottleneck_rate = _second_min_capacity(cap_src, bottleneck_candidate, bottleneck_rate)
    headroom_rate = bottleneck_rate * (1.0 + params.capacity_headroom)

    stages: list[StageCapacity] = []
    next_ewma: list[float | None] = []
    next_target_speed_ewma: list[float | None] = []
    for k in range(num_stages):
        a_raw = inputs.chain[k] * bottleneck_rate
        alpha_down = params.alpha_down_gpu if inputs.is_gpu[k] else params.alpha_down_cpu
        a_ewma = asymmetric_ewma(prev.a_ewma[k], a_raw, params.alpha_up, alpha_down)
        speed_k = inputs.speed[k]
        target_speed, target_speed_state = _target_speed_for_cycle(
            prev.target_speed_ewma[k],
            speed_k,
            params.alpha_up,
            alpha_down,
        )
        if target_speed <= 0.0 or inputs.chain[k] <= 0.0 or bottleneck_rate <= 0.0:
            # Cold / untrusted stage, or no measured bottleneck yet: the
            # cold-start ramp owns spawning, so target only min_workers.
            w_sustain = params.min_workers
            w_target = params.min_workers
        else:
            w_sustain = math.ceil(a_ewma / target_speed)
            # Growing the bottleneck toward next_bottleneck_rate is the move
            # that raises pipeline speed; growing any other stage past
            # bottleneck_rate + headroom does not.
            target_rate = max(next_bottleneck_rate, headroom_rate) if k == bottleneck_stage else headroom_rate
            w_target = math.ceil(inputs.chain[k] * target_rate / target_speed)
            # Headroom lives in w_target; never target below the hold floor.
            w_target = max(w_target, w_sustain)
        stages.append(
            StageCapacity(
                speed=speed_k,
                target_speed=target_speed,
                cap_src=cap_src[k],
                a_raw=a_raw,
                a_ewma=a_ewma,
                w_sustain=w_sustain,
                w_target=w_target,
            )
        )
        next_ewma.append(a_ewma)
        next_target_speed_ewma.append(target_speed_state)

    final_stages, feeder_pressure_streak = _apply_feeder_pressure(stages, inputs, prev, params, bottleneck_stage)

    return CapacityResult(
        plan=CapacityPlan(
            stages=final_stages,
            bottleneck_stage=bottleneck_stage,
            bottleneck_rate=bottleneck_rate,
            next_bottleneck_rate=next_bottleneck_rate,
            bottleneck_streak=bottleneck_streak,
            bottleneck_candidate=bottleneck_candidate,
            bottleneck_candidate_rate=bottleneck_candidate_rate,
        ),
        state=CapacityState(
            a_ewma=tuple(next_ewma),
            target_speed_ewma=tuple(next_target_speed_ewma),
            bottleneck=bottleneck_stage,
            bottleneck_streak=bottleneck_streak,
            feeder_pressure_streak=feeder_pressure_streak,
        ),
    )


@attrs.define
class CapacityModel:
    """Owns the capacity model's cross-cycle state and static tuning.

    Wraps the pure :func:`compute_capacity` so the smoothed arrival and the
    sticky bottleneck identity persist inside the model across cycles, rather
    than being threaded through the scheduler.

    Attributes:
        params: Static smoothing / hysteresis / headroom tuning.
    """

    params: CapacityParams
    _state: CapacityState

    @classmethod
    def create(cls, num_stages: int, params: CapacityParams) -> Self:
        """Build a model with empty state for ``num_stages`` stages."""
        return cls(params, CapacityState.initial(num_stages))

    def plan(self, inputs: CapacityInputs) -> CapacityPlan:
        """Return this cycle's capacity plan and advance the internal state."""
        result = compute_capacity(inputs, self._state, self.params)
        self._state = result.state
        return result.plan
