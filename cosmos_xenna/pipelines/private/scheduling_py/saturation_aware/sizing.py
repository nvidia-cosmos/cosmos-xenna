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

"""Capacity-target demand sizing for the saturation-aware scheduler.

Turns a per-stage snapshot (workers, measured speed, fan-out) plus that
stage's capacity target into a deflated solver speed and a demand multiplier.
The growth target ``w_target`` already encodes the bounded read-ahead and the
bottleneck climb to the next bottleneck (see ``capacity.py``), so this stage
only translates "grow toward ``w_target``" into a multiplier the fragmentation
solver understands: a stage at or above its target asks for nothing extra
(multiplier ``1.0``), and a below-target stage with real whole-chain stock
asks for ``w_target / workers``. There is no local backlog math and no
cross-cycle state here -- the throughput smoothing lives in ``capacity.py``.
"""

from collections.abc import Callable, Sequence

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.capacity import CapacityPlan, StageCapacity

_SOLVER_DEFAULT_SPEED = 1.0


@attrs.frozen
class StageDemandSnapshot:
    """One stage's per-cycle inputs to demand sizing.

    Attributes:
        name: Stage name (used for error messages).
        workers: Current pre-solve worker count.
        speed: Trusted per-worker speed (items/s), or ``None`` before the
            stage clears the estimator trust threshold (cold).
        num_returns: Measured fan-out (returns per batch), or ``None`` when
            unknown.
        batch_size: Stage input items consumed per batch.
    """

    name: str
    workers: int
    speed: float | None
    num_returns: float | None
    batch_size: int


@attrs.frozen
class StageSizingResult:
    """Demand-sizing outputs for one stage.

    Attributes:
        effective_speed: Per-worker speed fed to the solver estimate
            (``speed / multiplier`` when measured; the solver default at cold
            start).
        num_returns: Fan-out used for both the solver estimate and the chain
            factor.
        multiplier: The demand multiplier (``>= 1``), for logging.
    """

    effective_speed: float
    num_returns: float
    multiplier: float


def resolve_num_returns(snapshot: StageDemandSnapshot) -> float:
    """Return the measured fan-out, else the batch size.

    The fan-out feeds both the solver estimate and the chain factor, so an
    unknown (uncold) measurement falls back to one output per input batch.

    Args:
        snapshot: One stage's per-cycle demand inputs.

    Returns:
        The measured fan-out when known and valid, else ``batch_size``.

    Raises:
        ValueError: If a measured fan-out is negative (an invalid signal that
            must not silently produce a wrong chain factor).
    """
    if snapshot.num_returns is None:
        return float(snapshot.batch_size)
    if snapshot.num_returns < 0.0:
        raise ValueError(f"num_returns for stage '{snapshot.name}' must be >= 0, got {snapshot.num_returns}")
    return snapshot.num_returns


def size_stage(snapshot: StageDemandSnapshot, capacity: StageCapacity, has_stock: bool) -> StageSizingResult:
    """Size one stage from its snapshot and capacity target.

    Cold start (no trusted speed) returns the solver default speed and a unit
    multiplier so the cold-start ramp owns the stage. Otherwise the multiplier
    grows the stage toward ``w_target`` only when it is below target AND real
    whole-chain stock exists; a stage at or above its target produces a unit
    multiplier (headroom already lives in ``w_target``, so the multiplier must
    never add extra growth pressure).

    Args:
        snapshot: One stage's per-cycle demand inputs.
        capacity: That stage's smoothed capacity target for this cycle.
        has_stock: Whether real whole-chain stock is waiting for the stage.

    Returns:
        The deflated solver speed, fan-out, and demand multiplier for the stage.
    """
    num_returns = resolve_num_returns(snapshot)
    speed = snapshot.speed
    if speed is None or speed <= 0.0:
        return StageSizingResult(effective_speed=_SOLVER_DEFAULT_SPEED, num_returns=num_returns, multiplier=1.0)
    if has_stock and capacity.w_target > snapshot.workers:
        multiplier = capacity.w_target / max(snapshot.workers, 1)
    else:
        multiplier = 1.0
    return StageSizingResult(effective_speed=speed / multiplier, num_returns=num_returns, multiplier=multiplier)


def size_pipeline(
    snapshots: Sequence[StageDemandSnapshot], capacity: CapacityPlan, has_stock: Callable[[int], bool]
) -> tuple[StageSizingResult, ...]:
    """Size every stage in pipeline order.

    Pairs each snapshot with its per-stage capacity target and the scheduler's
    once-per-cycle "has real whole-chain stock" predicate, so growth pressure is
    applied only to below-target stages that actually have work waiting.

    Args:
        snapshots: Per-stage demand snapshots, in pipeline order.
        capacity: This cycle's capacity plan (one ``StageCapacity`` per stage).
        has_stock: Predicate returning whether stage ``index`` has real
            whole-chain stock waiting.

    Returns:
        One :class:`StageSizingResult` per stage, in pipeline order.
    """
    return tuple(
        size_stage(snapshot, capacity.stages[index], has_stock(index)) for index, snapshot in enumerate(snapshots)
    )
