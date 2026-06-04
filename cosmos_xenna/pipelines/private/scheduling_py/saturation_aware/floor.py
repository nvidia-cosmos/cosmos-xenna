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

"""Pipeline-aware scale-down floor for the saturation-aware scheduler.

A fresh copy of the backlog-guard idea, generalized from a stage's own
immediate queue to the whole pipeline's GLOBAL bottleneck (the minimum
source-rate capacity over all measured stages). It protects an expensive
downstream stage from being shrunk while only *transiently* starved by the
bottleneck, yet lets it shrink to its sustainable size during a persistent
bottleneck and release fully once work drains. The floor is a per-stage
minimum worker count; it never demands growth (``floor <= current workers``),
so growth stays owned by the demand multiplier. Three regimes follow from one
capacity-based (no-derivative) formula::

    transient lull        -> bottleneck capacity (w*s) unchanged -> floor holds
    persistent bottleneck -> A decays (slow alpha_down)          -> shrink to sustainable
    genuine drain         -> stock gate clears                   -> release to MIN
"""

import math
from collections.abc import Sequence
from typing import Self

import attrs


@attrs.frozen
class FloorParams:
    """Smoothing and release tuning for the scale-down floor.

    Attributes:
        alpha_up: EWMA weight when the sustainable arrival rises
            (fast re-protect).
        alpha_down_cpu: EWMA weight when it falls, for CPU stages.
        alpha_down_gpu: EWMA weight when it falls, for GPU stages
            (smaller = slower release, since warmup is costly).
        release_confirm_cycles: Consecutive low-stock cycles before a
            stage releases to ``min_workers``.
        min_workers: Lower bound a stage is never shrunk below while active.
    """

    alpha_up: float
    alpha_down_cpu: float
    alpha_down_gpu: float
    release_confirm_cycles: int
    min_workers: int = 1


@attrs.frozen
class FloorState:
    """Cross-cycle floor state, indexed by stage.

    Attributes:
        a_ewma: Smoothed sustainable arrival per stage; ``None`` until the
            first observation (then initialized to that value).
        release_streak: Consecutive low-stock cycles per stage.
    """

    a_ewma: tuple[float | None, ...]
    release_streak: tuple[int, ...]

    @classmethod
    def initial(cls, num_stages: int) -> Self:
        """Return empty state for a pipeline with ``num_stages`` stages."""
        return cls(a_ewma=(None,) * num_stages, release_streak=(0,) * num_stages)


@attrs.frozen
class FloorInputs:
    """Per-cycle observed inputs for the floor, indexed by stage.

    Attributes:
        workers: Current (observed pre-solve) worker count per stage.
        speed: Per-worker throughput per stage, in stage-input items/s.
        chain: Chain factors from :func:`chain.chain_factors`.
        stock_src: Whole-chain at-or-upstream stock per stage, source units.
        batch_sizes: Per-stage input batch sizes.
        is_gpu: Whether each stage holds GPU workers.
    """

    workers: tuple[int, ...]
    speed: tuple[float, ...]
    chain: tuple[float, ...]
    stock_src: tuple[float, ...]
    batch_sizes: tuple[int, ...]
    is_gpu: tuple[bool, ...]


@attrs.frozen
class FloorDecision:
    """One stage's floor outcome plus the signals that produced it.

    Attributes:
        floor: Minimum worker count (lower bound) applied this cycle.
        cap_src: Source-rate capacity ``workers * speed / chain``.
        a_raw: Sustainable arrival before smoothing (``chain * bottleneck``).
        a_ewma: Asymmetrically smoothed sustainable arrival used this cycle.
        w_sustain: Capacity-matched worker count ``ceil(a_ewma / speed)``; the
            growth ceiling for a non-bottleneck stage.
    """

    floor: int
    cap_src: float
    a_raw: float
    a_ewma: float
    w_sustain: int


@attrs.frozen
class FloorPlan:
    """Per-cycle floor output: per-stage decisions plus the global bottleneck.

    Attributes:
        decisions: One :class:`FloorDecision` per stage, in pipeline order.
        bottleneck_stage: Index of the global bottleneck (minimum ``cap_src``
            over measured stages); ``-1`` when no stage is measured yet.
    """

    decisions: tuple[FloorDecision, ...]
    bottleneck_stage: int

    @property
    def floors(self) -> tuple[int, ...]:
        """Return the per-stage floor (lower bound), derived from decisions."""
        return tuple(decision.floor for decision in self.decisions)


@attrs.frozen
class FloorResult:
    """Computed per-stage decisions, the global bottleneck, and next-cycle state.

    Attributes:
        decisions: One :class:`FloorDecision` per stage, in pipeline order.
        state: Updated :class:`FloorState` to carry into the next cycle.
        bottleneck_stage: Index of the global bottleneck (minimum ``cap_src``
            over measured stages); ``-1`` when no stage is measured yet.
    """

    decisions: tuple[FloorDecision, ...]
    state: FloorState
    bottleneck_stage: int

    @property
    def floors(self) -> tuple[int, ...]:
        """Return the per-stage floor (lower bound), derived from decisions."""
        return tuple(decision.floor for decision in self.decisions)


def asymmetric_ewma(prev: float | None, raw: float, alpha_up: float, alpha_down: float) -> float:
    """Smooth ``raw`` with a fast-up / slow-down EWMA.

    Initializes to ``raw`` on the first sample so cold start is not
    blunted. Rising values use ``alpha_up`` (re-protect quickly); falling
    values use ``alpha_down`` (release cautiously).
    """
    if prev is None:
        return raw
    alpha = alpha_up if raw >= prev else alpha_down
    return alpha * raw + (1.0 - alpha) * prev


def compute_floors(inputs: FloorInputs, prev: FloorState, params: FloorParams) -> FloorResult:
    """Compute per-stage scale-down floors for one cycle.

    Sizes every stage to the GLOBAL pipeline bottleneck: the minimum
    source-rate capacity over all measured stages (``cap_src > 0``), since a
    linear pipeline's steady-state throughput equals that bottleneck. A stage
    fed faster than the bottleneck is therefore floored down (it cannot
    sustainably outrun the bottleneck) rather than pinned to its fast feeder.
    Stage 0 (the source) has no sustainable-arrival meaning, so its floor is
    ``min_workers`` (sizing is driven by the demand multiplier). The
    bottleneck-derived arrival is smoothed asymmetrically and gated by whether
    real work is still at-or-upstream.

    Args:
        inputs: Per-cycle observed per-stage inputs.
        prev: State carried from the previous cycle.
        params: Smoothing and release tuning.

    Returns:
        The per-stage decisions, next-cycle state, and global bottleneck index.
    """
    cap_src = _source_capacities(inputs.workers, inputs.speed, inputs.chain)
    bottleneck_stage, bottleneck = _global_bottleneck(cap_src)
    decisions: list[FloorDecision] = []
    next_ewma: list[float | None] = []
    next_streak: list[int] = []

    for k in range(len(inputs.workers)):
        if k == 0:
            # The source has no upstream, so no sustainable-arrival meaning:
            # w_sustain is a placeholder (min_workers), not a real ceiling.
            decisions.append(
                FloorDecision(
                    floor=params.min_workers,
                    cap_src=cap_src[0],
                    a_raw=0.0,
                    a_ewma=0.0,
                    w_sustain=params.min_workers,
                )
            )
            next_ewma.append(None)
            next_streak.append(0)
            continue

        a_raw = inputs.chain[k] * bottleneck
        alpha_down = params.alpha_down_gpu if inputs.is_gpu[k] else params.alpha_down_cpu
        a_ewma = asymmetric_ewma(prev.a_ewma[k], a_raw, params.alpha_up, alpha_down)

        speed_k = inputs.speed[k]
        w_sustain = math.ceil(a_ewma / speed_k) if speed_k > 0.0 else params.min_workers

        threshold = inputs.batch_sizes[k] / inputs.chain[k] if inputs.chain[k] > 0.0 else 0.0
        streak = 0 if inputs.stock_src[k] > threshold else prev.release_streak[k] + 1
        releasing = inputs.stock_src[k] <= threshold and streak >= params.release_confirm_cycles

        floor = params.min_workers if releasing else max(params.min_workers, min(w_sustain, inputs.workers[k]))
        decisions.append(
            FloorDecision(floor=floor, cap_src=cap_src[k], a_raw=a_raw, a_ewma=a_ewma, w_sustain=w_sustain)
        )
        next_ewma.append(a_ewma)
        next_streak.append(streak)

    return FloorResult(
        decisions=tuple(decisions),
        state=FloorState(a_ewma=tuple(next_ewma), release_streak=tuple(next_streak)),
        bottleneck_stage=bottleneck_stage,
    )


def _global_bottleneck(cap_src: Sequence[float]) -> tuple[int, float]:
    """Return the ``(index, capacity)`` of the slowest measured stage.

    Filters on ``cap_src > 0`` so cold and zero-worker stages neither become
    the bottleneck nor drag the floor to ``min_workers``. Returns ``(-1, 0.0)``
    when no stage is measured yet (the floor then falls back to ``min_workers``).
    """
    measured = [(index, capacity) for index, capacity in enumerate(cap_src) if capacity > 0.0]
    if not measured:
        return -1, 0.0
    return min(measured, key=lambda item: item[1])


def _source_capacities(workers: Sequence[int], speed: Sequence[float], chain: Sequence[float]) -> list[float]:
    """Return per-stage capacity ``w * s / k`` in source items/s."""
    return [
        worker_count * speed_k / factor if factor > 0.0 else 0.0
        for worker_count, speed_k, factor in zip(workers, speed, chain, strict=True)
    ]


@attrs.define
class ScaleDownFloorPolicy:
    """Owns the scale-down floor's cross-cycle state and release tuning.

    Wraps the pure :func:`compute_floors` so the smoothed sustainable arrival
    and release streaks persist inside the policy across cycles, rather than
    being threaded through the scheduler.

    Attributes:
        params: Smoothing and release tuning.
    """

    params: FloorParams
    _state: FloorState

    @classmethod
    def create(cls, num_stages: int, params: FloorParams) -> Self:
        """Build a policy with empty state for ``num_stages`` stages."""
        return cls(params, FloorState.initial(num_stages))

    def plan(self, inputs: FloorInputs) -> FloorPlan:
        """Return this cycle's floor plan and advance the internal state."""
        result = compute_floors(inputs, self._state, self.params)
        self._state = result.state
        return FloorPlan(decisions=result.decisions, bottleneck_stage=result.bottleneck_stage)
