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
linear pipeline the sustainable throughput equals the queue-cliff stage's
source-rate capacity (or the smoothed minimum capacity when no queue cliff is
visible), so no stage can usefully outrun ``bottleneck_rate``. From that one fact
this module derives two per-stage worker targets that the grow (``sizing.py``)
and shrink (``floor.py``) consumers share:

- ``w_sustain``: workers to sustain the smoothed ``bottleneck_rate`` (the
  scale-down hold target; no headroom).
- ``w_target``: workers worth asking the solver for this cycle. The bottleneck
  climbs toward ``next_bottleneck_rate`` - the second-slowest stage's capacity,
  i.e. the move that actually raises pipeline speed; every other stage,
  including the source, is bounded to ``bottleneck_rate * (1 +
  capacity_headroom)`` (a small read-ahead, never free growth). Growth is
  additionally bounded by each stage's best observed per-worker speed
  (``speed_peak``), so a stage whose per-worker speed collapses under
  contention cannot inflate its divisive target into a runaway request.

Bottleneck identity is sticky (hysteresis) so a one-cycle queue or ``cap_src``
dip cannot flap growth ownership. The smoothed arrival rate ``a_ewma`` and the
bottleneck identity persist across cycles inside :class:`CapacityModel`.
"""

import enum
import math
from collections.abc import Sequence
from typing import Self

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain


class StageQueueState(enum.StrEnum):
    """Queue-gradient state for one stage."""

    STARVED = "starved"
    BOTTLENECK = "bottleneck"
    BUFFERED = "buffered"
    BALANCED = "balanced"


@attrs.frozen
class CapacityParams:
    """Static tuning for the capacity model.

    Attributes:
        alpha_up: Arrival-rate EWMA weight when the sustainable arrival rises
            (fast re-protect).
        alpha_down: Arrival-rate EWMA weight when it falls, applied uniformly to
            every stage (smaller = slower release).
        speed_alpha_up: Per-worker speed EWMA weight when measured speed rises;
            modest so a transient fast task cannot collapse the worker target.
        speed_alpha_down: Per-worker speed EWMA weight when measured speed falls
            during normal completion variance; protective (small) so a one-cycle
            dip does not mislabel a capable stage as the bottleneck. A genuine
            stall (``rate_is_stale``) bypasses this damping and snaps down to the
            aged rate instead.
        capacity_headroom: Spare-capacity fraction added to the bottleneck
            rate for the bounded read-ahead / tie-break growth target.
        hysteresis_margin: In balanced fallback mode, a challenger must be at
            least this much slower (lower ``cap_src``) than the incumbent
            bottleneck to be eligible to take over.
        switch_confirm: Consecutive eligible cycles before the bottleneck
            identity switches to the challenger.
        min_workers: Floor a stage's targets never fall below while active.
        stale_growth_step: Maximum workers a stale-rate stage may grow in one
            cycle, so a stalled stage's collapsing ``target_speed`` cannot
            explode its divisive ``w_target`` into a node-filling request.
        speed_peak_decay: Per-cycle decay (0, 1] applied to a stage's
            high-water per-worker speed when the current sample is below it.
            Bounds ``w_target`` so a stage whose per-worker speed collapses
            under contention cannot inflate its divisive growth target; the
            slow decay lets a genuine, sustained per-task slowdown eventually
            lower the bound and re-enable growth.
    """

    alpha_up: float
    alpha_down: float
    speed_alpha_up: float
    speed_alpha_down: float
    capacity_headroom: float
    hysteresis_margin: float
    switch_confirm: int
    min_workers: int = 1
    stale_growth_step: int = 1
    speed_peak_decay: float = 0.999


@attrs.frozen
class CapacityState:
    """Cross-cycle capacity state, indexed by stage.

    Attributes:
        a_ewma: Smoothed sustainable arrival per stage; ``None`` until the
            first observation (then initialized to that value).
        target_speed_ewma: Smoothed speed used for worker-target math; ``None``
            until a trusted speed has been observed.
        speed_peak: Slowly decaying high-water per-worker speed per stage;
            ``None`` until a trusted speed has been observed. Bounds the
            divisive ``w_target`` so a contention-collapsed speed cannot inflate
            the growth target.
        bottleneck: Incumbent global-bottleneck index, or ``-1`` when no
            stage is measured yet.
        bottleneck_streak: Consecutive cycles a challenger has beaten the
            incumbent (resets on any hold, and on a regime flip - see
            ``bottleneck_from_queue``).
        bottleneck_from_queue: Whether the previous cycle selected from a queue
            cliff (``True``) or from the smoothed-``cap_src`` fallback
            (``False``). A confirmation streak only counts within one regime, so
            ``bottleneck_streak`` is discarded when this flips between cycles.
    """

    a_ewma: tuple[float | None, ...]
    target_speed_ewma: tuple[float | None, ...]
    speed_peak: tuple[float | None, ...]
    bottleneck: int
    bottleneck_streak: int
    bottleneck_from_queue: bool = False

    @classmethod
    def initial(cls, num_stages: int) -> Self:
        """Return empty state for a pipeline with ``num_stages`` stages."""
        return cls(
            a_ewma=(None,) * num_stages,
            target_speed_ewma=(None,) * num_stages,
            speed_peak=(None,) * num_stages,
            bottleneck=-1,
            bottleneck_streak=0,
            bottleneck_from_queue=False,
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
        rate_is_stale: Whether each stage is busy but overdue for a completion,
            so its windowed speed is no longer trustworthy for divisive sizing.
    """

    workers: tuple[int, ...]
    speed: tuple[float, ...]
    chain: tuple[float, ...]
    is_manual: tuple[bool, ...]
    local_qin: tuple[float, ...]
    local_pending_depth: tuple[float, ...]
    local_input_threshold: tuple[float, ...]
    active_depth: tuple[float, ...]
    ready_workers: tuple[int, ...]
    rate_is_stale: tuple[bool, ...]


@attrs.frozen
class StageCapacity:
    """One stage's capacity facts for this cycle.

    Attributes:
        speed: Trusted raw per-worker speed observed this cycle (``0.0`` when
            cold); carried for logs so it can be compared against the smoothed
            ``target_speed``. Sizing uses ``target_speed`` / ``cap_src``, not
            this raw value.
        target_speed: Smoothed per-worker speed used for ``cap_src``,
            balanced-fallback selection, ``w_sustain``, ``w_target``, and solver
            demand sizing.
        cap_src: Source-rate capacity ``workers * target_speed / chain``
            (``0.0`` when cold). Built from the smoothed ``target_speed`` (not
            raw ``speed``) for stable balanced-fallback selection and logs.
        a_raw: Sustainable arrival before smoothing (``chain * bottleneck_rate``).
        a_ewma: Asymmetrically smoothed sustainable arrival used this cycle.
        w_sustain: Hold / scale-down target ``ceil(a_ewma / target_speed)``
            (matched to ``bottleneck_rate``, no headroom).
        w_target: Useful growth target this cycle (matched to
            ``next_bottleneck_rate`` for the bottleneck stage, to
            ``bottleneck_rate + headroom`` for every other stage), never below
            ``w_sustain`` and never above the worker count that would meet that
            rate at the stage's best observed per-worker speed (so a
            contention-collapsed speed cannot inflate it).
        w_target_is_real: True when ``w_target`` is a real capacity-derived
            growth target, False when it is the ``min_workers`` placeholder used
            for a stage with no measurable demand this cycle (cold/untrusted
            speed, collapsed source fan-out ``chain == 0``, or no measured
            bottleneck). The cold-start ramp consults this flag as the single
            source of truth for whether to enforce ``w_target`` as a per-cycle
            growth ceiling: a placeholder is not a real target, so the ramp
            defers to the solver (uncapped) instead of pinning the stage to
            ``min_workers``.
        queue_state: Queue-gradient classification emitted for decision logs.
    """

    speed: float
    target_speed: float
    cap_src: float
    a_raw: float
    a_ewma: float
    w_sustain: int
    w_target: int
    w_target_is_real: bool
    queue_state: StageQueueState = StageQueueState.BALANCED


@attrs.frozen
class CapacityPlan:
    """Per-cycle capacity output consumed by sizing and the floor.

    Attributes:
        stages: One :class:`StageCapacity` per stage, in pipeline order.
        bottleneck_stage: Sticky growth-owner bottleneck identity, or ``-1``
            when no stage is measured yet.
        bottleneck_rate: Effective sizing rate for the whole pipeline this cycle,
            in source items/s. Always the slowest MEASURED ``cap_src`` (so a
            cold cliff or a balanced pipeline never collapses sizing to 0, and a
            transiently inflated candidate cannot exceed the physical minimum).
            Only growth ownership is sticky, not this rate.
        next_bottleneck_rate: The second-minimum ``cap_src`` (excluding the
            current candidate) - the rate the growth owner can usefully climb
            toward.
        bottleneck_streak: Current challenger confirmation streak.
        bottleneck_candidate: Current queue-cliff candidate, or smoothed-capacity
            fallback candidate when no queue cliff exists.
        bottleneck_candidate_rate: The candidate's own ``cap_src`` before the
            measured-min bound (or ``bottleneck_rate`` when the candidate is
            cold, ``cap_src == 0``). It exceeds ``bottleneck_rate`` whenever the
            candidate is not itself the slowest measured stage - a fast warm
            candidate fed by a slower upstream, or a transient chain-factor
            collapse.
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


def _source_capacities(workers: Sequence[int], speed: Sequence[float], chain_factors: Sequence[float]) -> list[float]:
    """Return per-stage source-rate capacity ``w * s / k`` (``0.0`` when cold).

    A chain factor below :data:`chain.MIN_CHAIN_FACTOR` is unusable (its
    reciprocal would explode) and contributes ``0.0``.
    """
    return [
        worker_count * speed_k / factor if factor >= chain.MIN_CHAIN_FACTOR else 0.0
        for worker_count, speed_k, factor in zip(workers, speed, chain_factors, strict=True)
    ]


def classify_stages(inputs: CapacityInputs) -> tuple[StageQueueState, ...]:
    """Classify stages from local queue gradients."""
    states: list[StageQueueState] = []
    last_stage = len(inputs.workers) - 1
    for index in range(len(inputs.workers)):
        populated = inputs.local_qin[index] >= inputs.local_input_threshold[index]
        if not populated:
            states.append(StageQueueState.STARVED)
            continue
        if index == last_stage:
            state = StageQueueState.BOTTLENECK if inputs.ready_workers[index] == 0 else StageQueueState.BALANCED
            states.append(state)
            continue
        if inputs.local_qin[index + 1] < inputs.local_input_threshold[index + 1]:
            states.append(StageQueueState.BOTTLENECK)
            continue
        states.append(StageQueueState.BUFFERED)
    return tuple(states)


def _select_bottleneck_by_queue(
    states: Sequence[StageQueueState],
    cap_src: Sequence[float],
    prev_bn: int,
    prev_streak: int,
    prev_from_queue: bool,
    margin: float,
    confirm: int,
) -> tuple[int, int, int, bool]:
    """Return sticky growth owner and current rate-source candidate."""
    candidates = [index for index, state in enumerate(states) if state is StageQueueState.BOTTLENECK]
    from_queue = bool(candidates)
    if from_queue != prev_from_queue:
        prev_streak = 0
    if candidates:
        challenger = max(candidates)
        incumbent_valid = (
            0 <= prev_bn < len(states) and states[prev_bn] is StageQueueState.BOTTLENECK and cap_src[prev_bn] > 0.0
        )
        if not incumbent_valid or challenger == prev_bn:
            return challenger, 0, challenger, True
        streak = prev_streak + 1
        if streak >= confirm:
            return challenger, 0, challenger, True
        return prev_bn, streak, challenger, True

    measured = [(index, capacity) for index, capacity in enumerate(cap_src) if capacity > 0.0]
    if not measured:
        return -1, 0, -1, False
    challenger, challenger_cap = min(measured, key=lambda item: item[1])
    if prev_bn < 0 or prev_bn >= len(cap_src) or cap_src[prev_bn] <= 0.0:
        return challenger, 0, challenger, False
    incumbent_cap = cap_src[prev_bn]
    decisive = challenger != prev_bn and challenger_cap < incumbent_cap * (1.0 - margin)
    if not decisive:
        return prev_bn, 0, challenger, False
    streak = prev_streak + 1
    if streak >= confirm:
        return challenger, 0, challenger, False
    return prev_bn, streak, challenger, False


def _second_min_capacity(cap_src: Sequence[float], exclude: int, fallback: float) -> float:
    """Return the smallest measured ``cap_src`` other than ``exclude``.

    Falls back to ``fallback`` (the bottleneck rate) when fewer than two stages
    are measured, so a single-stage pipeline has
    ``next_bottleneck_rate == bottleneck_rate``.
    """
    others = [capacity for index, capacity in enumerate(cap_src) if capacity > 0.0 and index != exclude]
    return min(others) if others else fallback


def _target_speed_for_cycle(
    prev_target_speed: float | None,
    raw_speed: float,
    alpha_up: float,
    alpha_down: float,
    is_stale: bool,
) -> tuple[float, float | None]:
    """Return the smoothed target speed and next persisted speed sample.

    Normal completion variance is smoothed with the protective fast-up /
    slow-down EWMA, so a single slow task cannot flap the worker target. A
    genuine stall (``is_stale``) instead snaps the target down to the aged raw
    rate, so a stalled feeder is recognized within one cycle rather than over
    the slow ``alpha_down`` decay.
    """
    if raw_speed <= 0.0:
        return 0.0, prev_target_speed
    if is_stale and prev_target_speed is not None and raw_speed < prev_target_speed:
        return raw_speed, raw_speed
    target_speed = asymmetric_ewma(prev_target_speed, raw_speed, alpha_up, alpha_down)
    return target_speed, target_speed


def _speed_peak_for_cycle(prev_peak: float | None, target_speed: float, decay: float) -> float | None:
    """Return the high-water per-worker speed, decayed when below the peak.

    A sample at or above the peak raises it immediately; a lower sample relaxes
    the peak geometrically by ``decay`` so a transient contention dip is ignored
    while a sustained genuine slowdown eventually lowers the bound. A cold
    sample (``target_speed <= 0``) carries the previous peak unchanged.
    """
    if target_speed <= 0.0:
        return prev_peak
    if prev_peak is None or target_speed >= prev_peak:
        return target_speed
    return max(target_speed, prev_peak * decay)


def compute_capacity(inputs: CapacityInputs, prev: CapacityState, params: CapacityParams) -> CapacityResult:
    """Compute the capacity plan for one cycle and the next-cycle state.

    Classifies the queue gradient, identifies the sticky growth-owner
    bottleneck, derives the current candidate's ``bottleneck_rate`` and
    ``next_bottleneck_rate``, smooths each stage's bottleneck-matched arrival,
    and derives the hold target ``w_sustain`` and growth target ``w_target``.
    ``w_target`` is bounded by each stage's decaying peak per-worker speed so a
    contention-collapsed speed cannot inflate the divisive target. Cold /
    untrusted stages (``speed <= 0`` or ``chain <= 0``) and an all-cold pipeline
    (``bottleneck_rate <= 0``) yield ``min_workers`` targets so the cold-start
    ramp keeps owning them.

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
        == len(inputs.is_manual)
        == len(inputs.local_qin)
        == len(inputs.local_pending_depth)
        == len(inputs.local_input_threshold)
        == len(inputs.active_depth)
        == len(inputs.ready_workers)
        == len(inputs.rate_is_stale)
        == len(prev.a_ewma)
        == len(prev.target_speed_ewma)
        == len(prev.speed_peak)
        == num_stages
    ):
        raise ValueError(
            "capacity inputs length mismatch: "
            f"workers={num_stages} speed={len(inputs.speed)} chain={len(inputs.chain)} "
            f"is_manual={len(inputs.is_manual)} local_qin={len(inputs.local_qin)} "
            f"local_pending_depth={len(inputs.local_pending_depth)} "
            f"local_input_threshold={len(inputs.local_input_threshold)} active_depth={len(inputs.active_depth)} "
            f"ready_workers={len(inputs.ready_workers)} rate_is_stale={len(inputs.rate_is_stale)} "
            f"prev_a_ewma={len(prev.a_ewma)} prev_target_speed_ewma={len(prev.target_speed_ewma)} "
            f"prev_speed_peak={len(prev.speed_peak)}"
        )

    # Smooth each stage's per-worker speed with the dedicated speed alphas
    # (modest up / protective down), independent of the arrival-rate alphas. A
    # stalled stage bypasses the protective damping and snaps down to its aged
    # rate so the feeder it is starving is grown without the slow decay delay.
    target_speeds: list[float] = []
    next_target_speed_ewma: list[float | None] = []
    speed_peaks: list[float | None] = []
    for k in range(num_stages):
        target_speed, target_speed_state = _target_speed_for_cycle(
            prev.target_speed_ewma[k],
            inputs.speed[k],
            params.speed_alpha_up,
            params.speed_alpha_down,
            inputs.rate_is_stale[k],
        )
        target_speeds.append(target_speed)
        next_target_speed_ewma.append(target_speed_state)
        speed_peaks.append(_speed_peak_for_cycle(prev.speed_peak[k], target_speed, params.speed_peak_decay))

    cap_src = _source_capacities(inputs.workers, target_speeds, inputs.chain)
    queue_states = classify_stages(inputs)
    bottleneck_stage, bottleneck_streak, bottleneck_candidate, has_queue_candidate = _select_bottleneck_by_queue(
        queue_states,
        cap_src,
        prev.bottleneck,
        prev.bottleneck_streak,
        prev.bottleneck_from_queue,
        params.hysteresis_margin,
        params.switch_confirm,
    )
    # One stable rate sizes every stage: the slowest MEASURED source capacity
    # cap_src. A serial pipeline's throughput is its minimum stage capacity, so
    # the sizing rate can never physically exceed that minimum.
    #
    # A warm queue-cliff candidate that is the genuine constraint already IS
    # this minimum, so the common case is unchanged. A chain-factor collapse can
    # only INFLATE a stage's cap_src (it is the reciprocal of a near-zero
    # fan-out), never deflate it, so the measured minimum is structurally immune
    # and clamps a transiently corrupted candidate before its impossible rate
    # poisons every stage's smoothed arrival a_ewma.
    #
    # A COLD candidate (full input queue but no trusted speed yet, still owned
    # by the cold-start ramp) has cap_src 0.0 and is excluded from the minimum.
    # This is the critical guard: a cold candidate's 0.0 rate must NOT become
    # bottleneck_rate, or every stage collapses to min_workers and the floor
    # tears down the trusted upstream feeders keeping the cold stage supplied.
    #
    # Bottleneck IDENTITY (bottleneck_stage) and the bottleneck's climb target
    # (next_bottleneck_rate) still own growth; only the rate MAGNITUDE is bounded
    # here. bottleneck_candidate_rate keeps the candidate's own (possibly
    # inflated) cap_src so a decision snapshot shows when the bound engaged.
    measured_caps = [cap for cap in cap_src if cap > 0.0]
    bottleneck_rate = min(measured_caps) if measured_caps else 0.0
    candidate_cap = cap_src[bottleneck_candidate] if 0 <= bottleneck_candidate < num_stages else 0.0
    bottleneck_candidate_rate = candidate_cap if candidate_cap > 0.0 else bottleneck_rate
    next_bottleneck_rate = _second_min_capacity(cap_src, bottleneck_candidate, bottleneck_rate)
    headroom_rate = bottleneck_rate * (1.0 + params.capacity_headroom)

    stages: list[StageCapacity] = []
    next_ewma: list[float | None] = []
    for k in range(num_stages):
        a_raw = inputs.chain[k] * bottleneck_rate
        a_ewma = asymmetric_ewma(prev.a_ewma[k], a_raw, params.alpha_up, params.alpha_down)
        target_speed = target_speeds[k]
        if target_speed <= 0.0 or inputs.chain[k] <= 0.0 or bottleneck_rate <= 0.0:
            # Cold / untrusted stage, collapsed source fan-out (chain == 0), or
            # no measured bottleneck yet: there is no source-normalized demand to
            # divide, so w_target cannot be computed. Emit the min_workers
            # placeholder and flag it as not-real so the cold-start ramp leaves
            # growth to the solver instead of pinning the stage to min_workers.
            w_sustain = params.min_workers
            w_target = params.min_workers
            w_target_is_real = False
        else:
            w_target_is_real = True
            w_sustain = math.ceil(a_ewma / target_speed)
            if inputs.rate_is_stale[k]:
                # A stalled stage's target_speed is collapsing toward zero, so
                # the divisive hold target is untrustworthy and would inflate;
                # hold at the current worker count instead, but never below the
                # min_workers floor (current workers can be below it mid-ramp).
                w_sustain = max(min(w_sustain, inputs.workers[k]), params.min_workers)
            # A manual stage must never be grown by the autoscaler while it is
            # the rate identity (sticky growth owner) OR the current rate-source
            # candidate during a switch confirmation; cap both its hold and
            # growth targets to the operator-pinned worker count.
            is_manual_rate_participant = inputs.is_manual[k] and (k == bottleneck_stage or k == bottleneck_candidate)
            if is_manual_rate_participant:
                w_sustain = min(w_sustain, inputs.workers[k])
            # Growing the bottleneck toward next_bottleneck_rate is the move
            # that raises pipeline speed; growing any other stage past
            # bottleneck_rate + headroom does not. While a switch is confirming,
            # the growth owner (held incumbent) can differ from the rate-source
            # candidate; next_bottleneck_rate excludes the candidate, so the
            # owner may briefly target near its own rate (a transient near no-op)
            # until the challenger confirms - this is intended, not a stall.
            target_rate = max(next_bottleneck_rate, headroom_rate) if k == bottleneck_stage else headroom_rate
            w_target = math.ceil(inputs.chain[k] * target_rate / target_speed)
            # Headroom lives in w_target; never target below the hold floor.
            w_target = max(w_target, w_sustain)
            # Bound growth by the worker count that would already meet
            # target_rate at this stage's best observed per-worker speed. Past
            # that knee more workers cannot raise throughput - the per-worker
            # speed only falls under contention, so the divisive w_target would
            # inflate without bound (a runaway request/contention feedback loop).
            # speed_peak is a slowly decaying high-water mark, so a genuine
            # sustained per-task slowdown (heavier work, not contention)
            # eventually lowers the bound and re-enables growth. Skip while
            # rate_is_stale: a stall is one long non-parallelizable task, not
            # contention, so the stale clamp below owns that growth path instead.
            stage_speed_peak = speed_peaks[k]
            if not inputs.rate_is_stale[k] and stage_speed_peak is not None and stage_speed_peak > 0.0:
                w_target_cap = math.ceil(inputs.chain[k] * target_rate / stage_speed_peak)
                w_target = max(min(w_target, w_target_cap), w_sustain)
            if inputs.rate_is_stale[k]:
                # A stalled stage's target_speed is collapsing toward zero, which
                # would inflate this divisive w_target into a node-filling burst.
                # Adding workers only drains the queued backlog (the single long
                # in-flight task is not parallelizable), so grow a bounded step
                # per cycle and let completions un-freeze the rate, never below
                # the hold floor.
                w_target = max(min(w_target, inputs.workers[k] + params.stale_growth_step), w_sustain)
            if is_manual_rate_participant:
                w_target = min(w_target, inputs.workers[k])
        stages.append(
            StageCapacity(
                speed=inputs.speed[k],
                target_speed=target_speed,
                cap_src=cap_src[k],
                a_raw=a_raw,
                a_ewma=a_ewma,
                w_sustain=w_sustain,
                w_target=w_target,
                w_target_is_real=w_target_is_real,
                queue_state=queue_states[k],
            )
        )
        next_ewma.append(a_ewma)

    return CapacityResult(
        plan=CapacityPlan(
            stages=tuple(stages),
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
            speed_peak=tuple(speed_peaks),
            bottleneck=bottleneck_stage,
            bottleneck_streak=bottleneck_streak,
            bottleneck_from_queue=has_queue_candidate,
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
