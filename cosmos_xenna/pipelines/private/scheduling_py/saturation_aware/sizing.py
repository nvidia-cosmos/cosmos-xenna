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

"""Backlog demand sizing for the saturation-aware scheduler.

Turns a per-stage snapshot (workers, queue depth, measured speed, fan-out)
into a deflated solver speed plus a demand multiplier. The multiplier sizes a
stage to its arrival rate plus clearing the current backlog within one decision
interval, floored by the burst headroom and capped by ``max_backlog_boost``,
then EWMA-smoothed.
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig

_SOLVER_DEFAULT_SPEED = 1.0


@attrs.frozen
class StageSnapshot:
    """One stage's per-cycle inputs to demand sizing.

    Attributes:
        name: Stage name (estimator/state key).
        workers: Current pre-solve worker count.
        queue_depth: Current input queue depth, in stage-input items.
        speed: Measured per-worker speed (items/s), or ``None`` before any sample.
        num_returns: Measured fan-out (returns per batch), or ``None`` when unknown.
        batch_size: Stage input items consumed per batch.
        sample_count: Measured samples observed for this stage so far.
    """

    name: str
    workers: int
    queue_depth: float
    speed: float | None
    num_returns: float | None
    batch_size: int
    sample_count: int


@attrs.frozen
class DemandResult:
    """Demand-sizing outputs for one stage.

    Attributes:
        effective_speed: Per-worker speed fed to the solver estimate
            (``speed / multiplier`` when measured; the solver default at cold start).
        num_returns: Fan-out used for both the solver estimate and the chain factor.
        measured_speed_for_floor: Measured speed for the scale-down floor, or
            ``0.0`` at cold start (the floor must not protect an unestimated stage).
        multiplier: The demand multiplier (``>= 1``), for logging.
    """

    effective_speed: float
    num_returns: float
    measured_speed_for_floor: float
    multiplier: float


@attrs.define
class BacklogDemandPolicy:
    """Per-stage backlog demand sizing with EWMA-smoothed multipliers.

    Owns the small cross-cycle state (previous queue depth and previous
    multiplier per stage). :meth:`size` is called once per stage per cycle on
    the scheduler's single executor thread.

    Attributes:
        config: Operator tunables (interval, headroom, cap, smoothing, window).
    """

    config: SaturationAwareConfig
    _previous_queue: dict[str, float] = attrs.field(factory=dict)
    _previous_multiplier: dict[str, float] = attrs.field(factory=dict)

    def size(self, snapshot: StageSnapshot) -> DemandResult:
        """Size one stage from its snapshot.

        Cold start (no measured speed) returns the solver default speed, a unit
        multiplier, and a ``0.0`` floor speed. The previous queue depth is always
        recorded; the previous multiplier is recorded only on the measured path
        so the first measured cycle is unsmoothed (init-to-first).
        """
        if snapshot.num_returns is None:
            num_returns = float(snapshot.batch_size)
        elif snapshot.num_returns < 0.0:
            raise ValueError(f"num_returns for stage '{snapshot.name}' must be >= 0, got {snapshot.num_returns}")
        else:
            num_returns = snapshot.num_returns
        speed = snapshot.speed
        if speed is None or speed <= 0.0:
            result = DemandResult(
                effective_speed=_SOLVER_DEFAULT_SPEED,
                num_returns=num_returns,
                measured_speed_for_floor=0.0,
                multiplier=1.0,
            )
        else:
            multiplier = self._multiplier(snapshot, speed)
            result = DemandResult(
                effective_speed=speed / multiplier,
                num_returns=num_returns,
                measured_speed_for_floor=speed,
                multiplier=multiplier,
            )
        self._previous_queue[snapshot.name] = snapshot.queue_depth
        return result

    def _multiplier(self, snapshot: StageSnapshot, speed: float) -> float:
        """Clamp the desired-over-throughput ratio, then EWMA-smooth it."""
        config = self.config
        interval = config.interval_s
        queue_now = snapshot.queue_depth
        queue_prev = self._previous_queue.get(snapshot.name, queue_now)
        throughput = snapshot.workers * speed
        arrival_rate = max(0.0, (queue_now - queue_prev) / interval) + throughput
        desired = arrival_rate + queue_now / interval
        denominator = max(throughput, speed if queue_now > 0.0 else 0.0)
        headroom_floor = 1.0 + config.burst_headroom
        if denominator <= 0.0:
            raw = headroom_floor
        else:
            raw = min(max(desired / denominator, headroom_floor), config.max_backlog_boost)
        multiplier = self._smooth(snapshot, raw)
        self._previous_multiplier[snapshot.name] = multiplier
        return multiplier

    def _smooth(self, snapshot: StageSnapshot, raw: float) -> float:
        """EWMA-smooth the multiplier; pass raw through during warmup or first sight."""
        config = self.config
        alpha = config.backlog_smoothing
        warming = snapshot.sample_count < config.speed_estimation_min_data_points
        if alpha is None or snapshot.name not in self._previous_multiplier or warming:
            return raw
        return alpha * raw + (1.0 - alpha) * self._previous_multiplier[snapshot.name]
