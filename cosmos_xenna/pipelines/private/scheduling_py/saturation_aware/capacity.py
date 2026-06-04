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
  climbs toward ``next_bottleneck_rate`` -- the second-slowest stage's capacity,
  i.e. the move that actually raises pipeline speed; every other stage,
  including the source, is bounded to ``bottleneck_rate * (1 +
  capacity_headroom)`` (a small read-ahead, never free growth).

Bottleneck identity is sticky (hysteresis) so a one-cycle ``cap_src`` dip from
a floor cut cannot drag every stage's target down. The smoothed arrival rate
``a_ewma`` and the bottleneck identity persist across cycles inside
:class:`CapacityModel`::

    cap_src[k] = workers[k]*speed[k]/chain[k]
       |
       v
    bottleneck_rate = sticky min(cap_src)
    next_bottleneck_rate = second-min(cap_src)
       |
       v
    w_sustain = ceil(chain*bottleneck_rate/speed) (hold)
    w_target  = ceil(chain*target_rate/speed)     (grow)
       where target_rate = max(next_bottleneck_rate, bottleneck_rate*(1+headroom))
                               for the bottleneck stage,
                           bottleneck_rate*(1+headroom)
                               for every other stage
"""

import math
from collections.abc import Sequence
from typing import Self

import attrs


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
        min_workers: Floor a stage's targets never fall below while active.
    """

    alpha_up: float
    alpha_down_cpu: float
    alpha_down_gpu: float
    capacity_headroom: float
    hysteresis_margin: float
    switch_confirm: int
    min_workers: int = 1


@attrs.frozen
class CapacityState:
    """Cross-cycle capacity state, indexed by stage.

    Attributes:
        a_ewma: Smoothed sustainable arrival per stage; ``None`` until the
            first observation (then initialized to that value).
        bottleneck: Incumbent global-bottleneck index, or ``-1`` when no
            stage is measured yet.
        bottleneck_streak: Consecutive cycles a margin-clearing challenger
            has beaten the incumbent (resets on any hold).
    """

    a_ewma: tuple[float | None, ...]
    bottleneck: int
    bottleneck_streak: int

    @classmethod
    def initial(cls, num_stages: int) -> Self:
        """Return empty state for a pipeline with ``num_stages`` stages."""
        return cls(a_ewma=(None,) * num_stages, bottleneck=-1, bottleneck_streak=0)


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
    """

    workers: tuple[int, ...]
    speed: tuple[float, ...]
    chain: tuple[float, ...]
    is_gpu: tuple[bool, ...]


@attrs.frozen
class StageCapacity:
    """One stage's capacity facts for this cycle.

    Attributes:
        speed: Trusted per-worker speed used this cycle (``0.0`` when cold);
            carried so the scheduler's logs stay self-contained.
        cap_src: Source-rate capacity ``workers * speed / chain`` (``0.0``
            when cold).
        a_raw: Sustainable arrival before smoothing (``chain * bottleneck_rate``).
        a_ewma: Asymmetrically smoothed sustainable arrival used this cycle.
        w_sustain: Hold / scale-down target ``ceil(a_ewma / speed)``
            (matched to ``bottleneck_rate``, no headroom).
        w_target: Useful growth target this cycle (matched to
            ``next_bottleneck_rate`` for the bottleneck stage, to
            ``bottleneck_rate + headroom`` for every other stage), never below
            ``w_sustain``.
    """

    speed: float
    cap_src: float
    a_raw: float
    a_ewma: float
    w_sustain: int
    w_target: int


@attrs.frozen
class CapacityPlan:
    """Per-cycle capacity output consumed by sizing and the floor.

    Attributes:
        stages: One :class:`StageCapacity` per stage, in pipeline order.
        bottleneck_stage: Index of the global bottleneck (sticky minimum
            ``cap_src``), or ``-1`` when no stage is measured yet.
        bottleneck_rate: Pipeline throughput: the (sticky) minimum ``cap_src``,
            in source items/s.
        next_bottleneck_rate: The second-minimum ``cap_src`` -- the rate the
            bottleneck can usefully climb toward (falls back to
            ``bottleneck_rate`` when fewer than two stages are measured).
    """

    stages: tuple[StageCapacity, ...]
    bottleneck_stage: int
    bottleneck_rate: float
    next_bottleneck_rate: float


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
) -> tuple[int, float, int]:
    """Return the sticky global bottleneck as ``(index, cap_src, streak)``.

    Holds the incumbent unless a different measured stage is the minimum AND
    lower by ``margin`` for ``confirm`` consecutive cycles. The returned
    capacity is always ``cap_src`` of the SELECTED stage (the incumbent while
    held), so the hold window uses a consistent bottleneck rate even if a
    just-shrunk stage transiently shows a lower ``cap_src``. ``(-1, 0.0, 0)``
    when no stage is measured.
    """
    measured = [(index, capacity) for index, capacity in enumerate(cap_src) if capacity > 0.0]
    if not measured:
        return -1, 0.0, 0
    challenger, challenger_cap = min(measured, key=lambda item: item[1])
    if prev_bn < 0 or cap_src[prev_bn] <= 0.0:
        # No incumbent, or the incumbent went cold: adopt the current minimum.
        return challenger, challenger_cap, 0
    incumbent_cap = cap_src[prev_bn]
    decisive = challenger != prev_bn and challenger_cap < incumbent_cap * (1.0 - margin)
    if not decisive:
        # Incumbent still (near-)slowest: hold it and reset the streak.
        return prev_bn, incumbent_cap, 0
    streak = prev_streak + 1
    if streak >= confirm:
        # A challenger has been decisively slower long enough: switch.
        return challenger, challenger_cap, 0
    # Decisive this cycle but not yet confirmed: hold, accruing confirmation.
    return prev_bn, incumbent_cap, streak


def _second_min_capacity(cap_src: Sequence[float], exclude: int, fallback: float) -> float:
    """Return the smallest measured ``cap_src`` other than ``exclude``.

    Falls back to ``fallback`` (the bottleneck rate) when fewer than two stages
    are measured, so a single-stage pipeline has
    ``next_bottleneck_rate == bottleneck_rate``.
    """
    others = [capacity for index, capacity in enumerate(cap_src) if capacity > 0.0 and index != exclude]
    return min(others) if others else fallback


def compute_capacity(inputs: CapacityInputs, prev: CapacityState, params: CapacityParams) -> CapacityResult:
    """Compute the capacity plan for one cycle and the next-cycle state.

    Identifies the sticky global ``bottleneck_rate`` and the
    ``next_bottleneck_rate``, smooths each stage's bottleneck-matched arrival,
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
    if not (len(inputs.speed) == len(inputs.chain) == len(inputs.is_gpu) == len(prev.a_ewma) == num_stages):
        raise ValueError(
            "capacity inputs length mismatch: "
            f"workers={num_stages} speed={len(inputs.speed)} chain={len(inputs.chain)} "
            f"is_gpu={len(inputs.is_gpu)} prev_a_ewma={len(prev.a_ewma)}"
        )

    cap_src = _source_capacities(inputs.workers, inputs.speed, inputs.chain)
    bottleneck_stage, bottleneck_rate, bottleneck_streak = _select_bottleneck(
        cap_src, prev.bottleneck, prev.bottleneck_streak, params.hysteresis_margin, params.switch_confirm
    )
    next_bottleneck_rate = _second_min_capacity(cap_src, bottleneck_stage, bottleneck_rate)
    headroom_rate = bottleneck_rate * (1.0 + params.capacity_headroom)

    stages: list[StageCapacity] = []
    next_ewma: list[float | None] = []
    for k in range(num_stages):
        a_raw = inputs.chain[k] * bottleneck_rate
        alpha_down = params.alpha_down_gpu if inputs.is_gpu[k] else params.alpha_down_cpu
        a_ewma = asymmetric_ewma(prev.a_ewma[k], a_raw, params.alpha_up, alpha_down)
        speed_k = inputs.speed[k]
        if speed_k <= 0.0 or inputs.chain[k] <= 0.0 or bottleneck_rate <= 0.0:
            # Cold / untrusted stage, or no measured bottleneck yet: the
            # cold-start ramp owns spawning, so target only min_workers.
            w_sustain = params.min_workers
            w_target = params.min_workers
        else:
            w_sustain = math.ceil(a_ewma / speed_k)
            # Growing the bottleneck toward next_bottleneck_rate is the move
            # that raises pipeline speed; growing any other stage past
            # bottleneck_rate + headroom does not.
            target_rate = max(next_bottleneck_rate, headroom_rate) if k == bottleneck_stage else headroom_rate
            w_target = math.ceil(inputs.chain[k] * target_rate / speed_k)
            # Headroom lives in w_target; never target below the hold floor.
            w_target = max(w_target, w_sustain)
        stages.append(
            StageCapacity(
                speed=speed_k,
                cap_src=cap_src[k],
                a_raw=a_raw,
                a_ewma=a_ewma,
                w_sustain=w_sustain,
                w_target=w_target,
            )
        )
        next_ewma.append(a_ewma)

    return CapacityResult(
        plan=CapacityPlan(
            stages=tuple(stages),
            bottleneck_stage=bottleneck_stage,
            bottleneck_rate=bottleneck_rate,
            next_bottleneck_rate=next_bottleneck_rate,
        ),
        state=CapacityState(a_ewma=tuple(next_ewma), bottleneck=bottleneck_stage, bottleneck_streak=bottleneck_streak),
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
