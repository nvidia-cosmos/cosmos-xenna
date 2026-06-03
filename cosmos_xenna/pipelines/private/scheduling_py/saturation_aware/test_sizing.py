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

"""Focused unit tests for backlog demand sizing (native-extension-free)."""

from collections.abc import Callable

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.sizing import (
    BacklogDemandPolicy,
    StageSnapshot,
)

type SnapshotFactory = Callable[..., StageSnapshot]


@pytest.fixture
def make_snapshot() -> SnapshotFactory:
    """Return a snapshot factory with defaults that exercise the measured warm path."""

    def _make_snapshot(
        *,
        name: str = "s",
        workers: int = 2,
        queue_depth: float = 0.0,
        speed: float | None = 2.0,
        num_returns: float | None = 1.0,
        batch_size: int = 1,
        sample_count: int = 10,
    ) -> StageSnapshot:
        return StageSnapshot(
            name=name,
            workers=workers,
            queue_depth=queue_depth,
            speed=speed,
            num_returns=num_returns,
            batch_size=batch_size,
            sample_count=sample_count,
        )

    return _make_snapshot


def test_cold_start_without_speed_uses_solver_default(make_snapshot: SnapshotFactory) -> None:
    """No measured speed yields the solver default speed and a unit multiplier."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(speed=None))
    assert result.effective_speed == 1.0
    assert result.multiplier == 1.0
    assert result.measured_speed_for_floor == 0.0


def test_non_positive_speed_is_treated_as_cold_start(make_snapshot: SnapshotFactory) -> None:
    """A zero speed is unestimated, so the floor speed must stay at zero."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(speed=0.0))
    assert result.effective_speed == 1.0
    assert result.measured_speed_for_floor == 0.0


def test_no_backlog_settles_at_burst_headroom_floor(make_snapshot: SnapshotFactory) -> None:
    """With an empty queue the multiplier rests at ``1 + burst_headroom``."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(workers=2, speed=2.0, queue_depth=0.0))
    assert result.multiplier == pytest.approx(1.1)
    assert result.effective_speed == pytest.approx(2.0 / 1.1)


def test_zero_workers_empty_queue_uses_burst_headroom_floor(make_snapshot: SnapshotFactory) -> None:
    """A stage with no current workers and no backlog still rests at headroom."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(workers=0, speed=2.0, queue_depth=0.0))
    assert result.multiplier == pytest.approx(1.1)
    assert result.effective_speed == pytest.approx(2.0 / 1.1)


def test_deep_backlog_caps_at_max_backlog_boost(make_snapshot: SnapshotFactory) -> None:
    """A very deep queue saturates the catch-up cap."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(
        make_snapshot(workers=1, speed=2.0, queue_depth=10_000.0)
    )
    assert result.multiplier == 8.0


def test_queue_growth_raises_the_multiplier(make_snapshot: SnapshotFactory) -> None:
    """A growing queue adds arrival pressure, lifting the multiplier."""
    policy = BacklogDemandPolicy(SaturationAwareConfig(backlog_smoothing=None))
    first = policy.size(make_snapshot(workers=2, speed=2.0, queue_depth=0.0))
    second = policy.size(make_snapshot(workers=2, speed=2.0, queue_depth=100.0))
    assert second.multiplier > first.multiplier


def test_num_returns_falls_back_to_batch_size_when_unmeasured(make_snapshot: SnapshotFactory) -> None:
    """Without a measured fan-out, the batch size seeds the chain factor."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(num_returns=None, batch_size=4))
    assert result.num_returns == 4.0


def test_zero_num_returns_is_passed_through(make_snapshot: SnapshotFactory) -> None:
    """A measured zero fan-out is valid and represents a fully dropping stage."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(num_returns=0.0, batch_size=4))
    assert result.num_returns == 0.0


def test_negative_num_returns_raises(make_snapshot: SnapshotFactory) -> None:
    """A negative measured fan-out is invalid and must not fall back to batch size."""
    with pytest.raises(ValueError, match=r"num_returns for stage 's' must be >= 0, got -1.0"):
        BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(num_returns=-1.0, batch_size=4))


def test_measured_num_returns_is_passed_through(make_snapshot: SnapshotFactory) -> None:
    """A measured fan-out is used verbatim."""
    result = BacklogDemandPolicy(SaturationAwareConfig()).size(make_snapshot(num_returns=8.0))
    assert result.num_returns == 8.0


def test_smoothing_is_skipped_during_warmup(make_snapshot: SnapshotFactory) -> None:
    """While the estimator is warming, the raw multiplier passes through."""
    policy = BacklogDemandPolicy(SaturationAwareConfig(speed_estimation_min_data_points=5))
    policy.size(make_snapshot(queue_depth=0.0, sample_count=1))
    second = policy.size(make_snapshot(workers=1, speed=2.0, queue_depth=10_000.0, sample_count=2))
    assert second.multiplier == 8.0


def test_smoothing_blends_with_the_previous_multiplier_after_warmup(make_snapshot: SnapshotFactory) -> None:
    """Once warm and seen before, the multiplier is an EWMA blend."""
    policy = BacklogDemandPolicy(SaturationAwareConfig(speed_estimation_min_data_points=1, backlog_smoothing=0.4))
    policy.size(make_snapshot(queue_depth=0.0, sample_count=5))
    second = policy.size(make_snapshot(workers=1, speed=2.0, queue_depth=10_000.0, sample_count=5))
    assert second.multiplier == pytest.approx(0.4 * 8.0 + 0.6 * 1.1)
