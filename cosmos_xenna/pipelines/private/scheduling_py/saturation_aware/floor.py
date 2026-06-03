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
immediate queue to the whole upstream chain. It protects an expensive
downstream stage from being shrunk while only *transiently* starved by an
upstream bottleneck, yet lets it shrink to its sustainable size during a
persistent bottleneck and release fully once work drains. The floor is a
per-stage minimum worker count; it never demands growth
(``floor <= current workers``), so growth stays owned by the demand
multiplier. Three regimes follow from one capacity-based (no-derivative)
formula::

    transient lull        -> upstream capacity (w*s) unchanged -> floor holds
    persistent bottleneck -> A decays (slow alpha_down)        -> shrink to sustainable
    genuine drain         -> stock gate clears                 -> release to MIN
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
class FloorResult:
    """Computed floors plus the next-cycle state.

    Attributes:
        floors: Minimum worker count per stage for this cycle.
        state: Updated :class:`FloorState` to carry into the next cycle.
    """

    floors: tuple[int, ...]
    state: FloorState


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

    Stage 0 has no upstream and so cannot be upstream-starved; its floor
    is ``min_workers`` (sizing is driven by the demand multiplier). Each
    downstream stage is floored to the worker count that keeps up with the
    rate its upstream bottleneck can sustainably deliver, smoothed
    asymmetrically, gated by whether real work is still at-or-upstream.

    Args:
        inputs: Per-cycle observed per-stage inputs.
        prev: State carried from the previous cycle.
        params: Smoothing and release tuning.

    Returns:
        The per-stage floors and the next-cycle state.
    """
    cap_src = _source_capacities(inputs.workers, inputs.speed, inputs.chain)
    floors: list[int] = []
    next_ewma: list[float | None] = []
    next_streak: list[int] = []

    for k in range(len(inputs.workers)):
        if k == 0:
            floors.append(params.min_workers)
            next_ewma.append(None)
            next_streak.append(0)
            continue

        a_raw = inputs.chain[k] * min(cap_src[:k])
        alpha_down = params.alpha_down_gpu if inputs.is_gpu[k] else params.alpha_down_cpu
        a_ewma = asymmetric_ewma(prev.a_ewma[k], a_raw, params.alpha_up, alpha_down)

        speed_k = inputs.speed[k]
        w_sustain = math.ceil(a_ewma / speed_k) if speed_k > 0.0 else params.min_workers

        threshold = inputs.batch_sizes[k] / inputs.chain[k] if inputs.chain[k] > 0.0 else 0.0
        streak = 0 if inputs.stock_src[k] > threshold else prev.release_streak[k] + 1
        releasing = inputs.stock_src[k] <= threshold and streak >= params.release_confirm_cycles

        floor = params.min_workers if releasing else max(params.min_workers, min(w_sustain, inputs.workers[k]))
        floors.append(floor)
        next_ewma.append(a_ewma)
        next_streak.append(streak)

    return FloorResult(
        floors=tuple(floors),
        state=FloorState(a_ewma=tuple(next_ewma), release_streak=tuple(next_streak)),
    )


def _source_capacities(workers: Sequence[int], speed: Sequence[float], chain: Sequence[float]) -> list[float]:
    """Return per-stage capacity ``w * s / k`` in source items/s."""
    return [
        worker_count * speed_k / factor if factor > 0.0 else 0.0
        for worker_count, speed_k, factor in zip(workers, speed, chain, strict=True)
    ]
