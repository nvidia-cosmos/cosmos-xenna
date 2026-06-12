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

"""Focused unit tests for capacity-target demand sizing (native-extension-free)."""

from collections.abc import Callable

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.capacity import CapacityPlan, StageCapacity
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.sizing import (
    StageDemandSnapshot,
    size_pipeline,
    size_stage,
)

type SnapshotFactory = Callable[..., StageDemandSnapshot]


@pytest.fixture
def make_snapshot() -> SnapshotFactory:
    """Return a snapshot factory with defaults that exercise the measured warm path."""

    def _make_snapshot(
        *,
        name: str = "s",
        workers: int = 2,
        speed: float | None = 2.0,
        num_returns: float | None = 1.0,
        batch_size: int = 1,
    ) -> StageDemandSnapshot:
        return StageDemandSnapshot(
            name=name,
            workers=workers,
            speed=speed,
            num_returns=num_returns,
            batch_size=batch_size,
        )

    return _make_snapshot


def _capacity(
    *,
    w_target: int,
    w_sustain: int = 1,
    speed: float = 2.0,
    target_speed: float | None = None,
    w_target_is_real: bool = True,
) -> StageCapacity:
    """A StageCapacity carrying only the fields demand sizing reads."""
    solver_speed = speed if target_speed is None else target_speed
    return StageCapacity(
        speed=speed,
        target_speed=solver_speed,
        cap_src=0.0,
        a_raw=0.0,
        a_ewma=0.0,
        w_sustain=w_sustain,
        w_target=w_target,
        w_target_is_real=w_target_is_real,
    )


def test_cold_start_without_speed_uses_solver_default(make_snapshot: SnapshotFactory) -> None:
    """No measured speed yields the solver default speed and a unit multiplier."""
    result = size_stage(make_snapshot(speed=None), _capacity(w_target=10), has_local_input=True)
    assert result.effective_speed == 1.0
    assert result.multiplier == 1.0


def test_non_positive_speed_is_treated_as_cold_start(make_snapshot: SnapshotFactory) -> None:
    """A zero speed is unestimated, so it falls back to the cold-start path."""
    result = size_stage(make_snapshot(speed=0.0), _capacity(w_target=10), has_local_input=True)
    assert result.effective_speed == 1.0
    assert result.multiplier == 1.0


def test_negative_speed_is_treated_as_cold_start(make_snapshot: SnapshotFactory) -> None:
    """A negative speed is an invalid estimate, so it falls back to the cold-start path."""
    result = size_stage(make_snapshot(speed=-1.0), _capacity(w_target=10), has_local_input=True)
    assert result.effective_speed == 1.0
    assert result.multiplier == 1.0


def test_at_or_above_target_does_not_grow_even_with_stock(make_snapshot: SnapshotFactory) -> None:
    """A stage at its target asks for nothing extra, even with whole-chain stock.

    Regression for run 9a14287f: a fast non-bottleneck stage's target equals its
    current size, so the multiplier stays 1.0 and the solver is never handed an
    inflated ask that could over-grow it.
    """
    result = size_stage(make_snapshot(workers=4, speed=2.0), _capacity(w_target=4), has_local_input=True)
    assert result.multiplier == 1.0
    assert result.effective_speed == pytest.approx(2.0)


def test_below_target_with_local_input_grows_toward_target(make_snapshot: SnapshotFactory) -> None:
    """A below-target stage with local input deflates speed by ``w_target / workers``."""
    result = size_stage(make_snapshot(workers=2, speed=2.0), _capacity(w_target=6), has_local_input=True)
    assert result.multiplier == pytest.approx(3.0)
    assert result.effective_speed == pytest.approx(2.0 / 3.0)


def test_solver_speed_uses_smoothed_capacity_target_speed(make_snapshot: SnapshotFactory) -> None:
    """Demand sizing deflates the smoothed capacity speed, not a one-cycle raw dip."""
    result = size_stage(
        make_snapshot(workers=2, speed=0.25),
        _capacity(w_target=6, speed=0.25, target_speed=1.5),
        has_local_input=True,
    )
    assert result.multiplier == pytest.approx(3.0)
    assert result.effective_speed == pytest.approx(1.5 / 3.0)


def test_below_target_without_local_input_does_not_grow(make_snapshot: SnapshotFactory) -> None:
    """Without local input the stage holds at a unit multiplier even below target."""
    result = size_stage(make_snapshot(workers=2, speed=2.0), _capacity(w_target=6), has_local_input=False)
    assert result.multiplier == 1.0
    assert result.effective_speed == pytest.approx(2.0)


def test_zero_workers_below_target_uses_one_as_divisor(make_snapshot: SnapshotFactory) -> None:
    """A stage with no current workers divides by one, not zero."""
    result = size_stage(make_snapshot(workers=0, speed=2.0), _capacity(w_target=3), has_local_input=True)
    assert result.multiplier == pytest.approx(3.0)


def test_num_returns_falls_back_to_batch_size_when_unmeasured(make_snapshot: SnapshotFactory) -> None:
    """Without a measured fan-out, the batch size seeds the chain factor."""
    result = size_stage(make_snapshot(num_returns=None, batch_size=4), _capacity(w_target=1), has_local_input=False)
    assert result.num_returns == 4.0


def test_zero_num_returns_is_passed_through(make_snapshot: SnapshotFactory) -> None:
    """A measured zero fan-out is valid and represents a fully dropping stage."""
    result = size_stage(make_snapshot(num_returns=0.0, batch_size=4), _capacity(w_target=1), has_local_input=False)
    assert result.num_returns == 0.0


def test_negative_num_returns_raises(make_snapshot: SnapshotFactory) -> None:
    """A negative measured fan-out is invalid and must not fall back to batch size."""
    with pytest.raises(ValueError, match=r"num_returns for stage 's' must be >= 0, got -1.0"):
        size_stage(make_snapshot(num_returns=-1.0, batch_size=4), _capacity(w_target=1), has_local_input=False)


def test_measured_num_returns_is_passed_through(make_snapshot: SnapshotFactory) -> None:
    """A measured fan-out is used verbatim."""
    result = size_stage(make_snapshot(num_returns=8.0), _capacity(w_target=1), has_local_input=False)
    assert result.num_returns == 8.0


def test_size_pipeline_pairs_each_stage_with_its_capacity_and_local_input(make_snapshot: SnapshotFactory) -> None:
    """``size_pipeline`` applies growth only to the below-target stage with local input.

    Stage 0 is below target with local input (grows), stage 1 is below target
    without local input (holds). The per-index predicate must select the right
    stage, proving the function pairs snapshots, capacities, and the gate by
    index.
    """
    snapshots = [make_snapshot(workers=2, speed=2.0), make_snapshot(workers=2, speed=2.0)]
    capacity = CapacityPlan(
        stages=(_capacity(w_target=6), _capacity(w_target=6)),
        bottleneck_stage=0,
        bottleneck_rate=1.0,
        next_bottleneck_rate=1.0,
        bottleneck_candidate=0,
        bottleneck_candidate_rate=1.0,
    )

    results = size_pipeline(snapshots, capacity, has_local_input=lambda index: index == 0)

    assert results[0].multiplier == pytest.approx(3.0)
    assert results[1].multiplier == 1.0


def test_size_pipeline_mismatched_lengths_raise(make_snapshot: SnapshotFactory) -> None:
    """More snapshots than capacity stages is a programming error and fails fast."""
    snapshots = [make_snapshot(), make_snapshot()]
    capacity = CapacityPlan(
        stages=(_capacity(w_target=1),),
        bottleneck_stage=0,
        bottleneck_rate=1.0,
        next_bottleneck_rate=1.0,
        bottleneck_candidate=0,
        bottleneck_candidate_rate=1.0,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        size_pipeline(snapshots, capacity, has_local_input=lambda _index: False)
