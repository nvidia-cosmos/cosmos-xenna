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

"""Pure-function tests for the capacity sizer.

Pins the contract of ``compute_capacity_target_workers`` and
``derive_utilization_target``: the closed-form helpers used by the
saturation-aware scheduler to translate queueing signals into a
worker count target.
"""

import math

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import (
    ResolvedThresholds,
    derive_utilization_target,
)
from cosmos_xenna.pipelines.private.scheduling_py.pressure import (
    compute_capacity_target_workers,
)


@pytest.fixture
def thresholds() -> ResolvedThresholds:
    """Default classifier thresholds; saturation_threshold=0.15 -> utilization_target=0.85."""
    return ResolvedThresholds(
        saturation_threshold=0.15,
        activation_threshold=0.05,
        saturation_aggressiveness=0.30,
        slots_per_actor=4,
        saturation_threshold_was_overridden=False,
        activation_threshold_was_overridden=False,
    )


class TestDeriveUtilizationTarget:
    """``utilization_target`` derives from ``1 - saturation_threshold``."""

    def test_returns_complement_of_saturation_threshold(self, thresholds: ResolvedThresholds) -> None:
        """sat_threshold=0.15 -> util_target=0.85."""
        assert derive_utilization_target(thresholds) == pytest.approx(0.85)

    def test_low_saturation_threshold_yields_high_utilization_target(self) -> None:
        """A small saturation_threshold (=tight slot-pin trigger) drives the target near 1.0."""
        tight = ResolvedThresholds(
            saturation_threshold=0.05,
            activation_threshold=0.025,
            saturation_aggressiveness=0.20,
            slots_per_actor=4,
            saturation_threshold_was_overridden=False,
            activation_threshold_was_overridden=False,
        )
        assert derive_utilization_target(tight) == pytest.approx(0.95)

    def test_zero_saturation_threshold_is_rejected(self) -> None:
        """sat_threshold=0 -> derived target=1.0, which would force perfect utilisation; reject."""
        bad = ResolvedThresholds(
            saturation_threshold=0.0,
            activation_threshold=0.0,
            saturation_aggressiveness=0.0,
            slots_per_actor=1,
            saturation_threshold_was_overridden=True,
            activation_threshold_was_overridden=True,
        )
        with pytest.raises(ValueError, match="utilization_target 1.0 outside"):
            derive_utilization_target(bad)

    def test_one_saturation_threshold_is_rejected(self) -> None:
        """sat_threshold=1.0 -> derived target=0.0, which would never plan workers; reject."""
        bad = ResolvedThresholds(
            saturation_threshold=1.0,
            activation_threshold=0.5,
            saturation_aggressiveness=10.0,
            slots_per_actor=1,
            saturation_threshold_was_overridden=True,
            activation_threshold_was_overridden=True,
        )
        with pytest.raises(ValueError, match="utilization_target 0.0 outside"):
            derive_utilization_target(bad)


class TestCapacityTargetSteadyState:
    """Steady-state throughput drives the target; the queue term is zero."""

    def test_arrival_rate_matched_by_one_worker(self) -> None:
        """rate=0.5 t/s, D_k=2 s, util=0.85, slots/worker=1 -> ceil(0.5*2/0.85)=ceil(1.18)=2 slots = 2 workers."""
        result = compute_capacity_target_workers(
            queue_depth=0,
            observed_throughput=0.5,
            d_k_seconds=2.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result == 2

    def test_packed_slots_per_worker_reduce_worker_count(self) -> None:
        """slots/worker=4 -> 2 slots become 1 worker (ceil)."""
        result = compute_capacity_target_workers(
            queue_depth=0,
            observed_throughput=0.5,
            d_k_seconds=2.0,
            slots_per_worker=4,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result == 1


class TestCapacityTargetBacklogCatchup:
    """Queue depth contributes an extra arrival-rate equivalent of ``queue / target_backlog_seconds``."""

    def test_large_queue_dominates_target(self) -> None:
        """The Stage 09 production scenario (queue=60, throughput=0.05, D_k=21s, target=30s, util=0.85)."""
        result = compute_capacity_target_workers(
            queue_depth=60,
            observed_throughput=0.05,
            d_k_seconds=21.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        # target_rate = 0.05 + 60/30 = 2.05 tasks/s
        # target_slots = ceil(2.05 * 21 / 0.85) = ceil(50.6) = 51
        # target_workers = ceil(51 / 1) = 51
        assert result == 51

    def test_smaller_target_backlog_seconds_demands_more_workers(self) -> None:
        """Halving target_backlog_seconds doubles the queue-drain contribution."""
        baseline = compute_capacity_target_workers(
            queue_depth=60,
            observed_throughput=0.0,
            d_k_seconds=10.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        tighter = compute_capacity_target_workers(
            queue_depth=60,
            observed_throughput=0.0,
            d_k_seconds=10.0,
            slots_per_worker=1,
            target_backlog_seconds=15.0,
            utilization_target=0.85,
        )
        assert tighter is not None
        assert baseline is not None
        assert tighter > baseline


class TestCapacityTargetUtilizationHeadroom:
    """Lower ``utilization_target`` (more headroom) demands more workers for the same load."""

    def test_lower_utilization_target_increases_worker_count(self) -> None:
        """util=0.5 needs ~2x the workers of util=1.0 for the same offered load."""
        loose = compute_capacity_target_workers(
            queue_depth=0,
            observed_throughput=1.0,
            d_k_seconds=4.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.5,
        )
        tight = compute_capacity_target_workers(
            queue_depth=0,
            observed_throughput=1.0,
            d_k_seconds=4.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=1.0,
        )
        assert loose is not None and tight is not None
        assert loose >= tight
        assert loose == 8
        assert tight == 4


class TestCapacityTargetColdStart:
    """Non-finite or non-positive D_k returns ``None`` so the caller falls back to discrete sizing."""

    def test_zero_d_k_returns_none(self) -> None:
        """D_k=0 -> cold start; no estimate possible."""
        assert (
            compute_capacity_target_workers(
                queue_depth=10,
                observed_throughput=0.5,
                d_k_seconds=0.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )
            is None
        )

    def test_nan_d_k_returns_none(self) -> None:
        """Cold-start sentinel is ``math.nan``."""
        assert (
            compute_capacity_target_workers(
                queue_depth=10,
                observed_throughput=0.5,
                d_k_seconds=math.nan,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )
            is None
        )

    def test_negative_d_k_returns_none(self) -> None:
        """Negative D_k is impossible in steady state but defensive check returns None instead of negative target."""
        assert (
            compute_capacity_target_workers(
                queue_depth=10,
                observed_throughput=0.5,
                d_k_seconds=-1.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )
            is None
        )


class TestCapacityTargetEmptyState:
    """Idle stage with no queue and no throughput needs zero workers (caller still applies floors)."""

    def test_empty_queue_zero_throughput_returns_zero(self) -> None:
        """No load + no backlog -> target=0; min_workers floor is applied by the caller, not here."""
        result = compute_capacity_target_workers(
            queue_depth=0,
            observed_throughput=0.0,
            d_k_seconds=5.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result == 0


class TestCapacityTargetInputValidation:
    """Defensive validation against programmer error and infeasible config."""

    def test_negative_queue_depth_is_rejected(self) -> None:
        """Defensive: negative queue depth indicates a bug upstream."""
        with pytest.raises(ValueError, match="queue_depth must be >= 0"):
            compute_capacity_target_workers(
                queue_depth=-1,
                observed_throughput=0.5,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )

    def test_negative_throughput_is_rejected(self) -> None:
        """Defensive: throughput is non-negative by construction."""
        with pytest.raises(ValueError, match="observed_throughput must be >= 0"):
            compute_capacity_target_workers(
                queue_depth=0,
                observed_throughput=-0.1,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )

    def test_nan_throughput_is_rejected(self) -> None:
        """NaN throughput would slip past ``< 0.0`` (False for NaN) into ``math.ceil`` as an indirect failure."""
        with pytest.raises(ValueError, match=r"observed_throughput must be finite"):
            compute_capacity_target_workers(
                queue_depth=0,
                observed_throughput=math.nan,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )

    def test_inf_throughput_is_rejected(self) -> None:
        """``+inf`` throughput would propagate into ``target_rate`` and trip ``math.ceil`` with ``OverflowError``."""
        with pytest.raises(ValueError, match=r"observed_throughput must be finite"):
            compute_capacity_target_workers(
                queue_depth=0,
                observed_throughput=math.inf,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )

    def test_zero_slots_per_worker_is_rejected(self) -> None:
        """slots_per_worker is the denominator at the worker boundary; must be >= 1."""
        with pytest.raises(ValueError, match="slots_per_worker must be >= 1"):
            compute_capacity_target_workers(
                queue_depth=0,
                observed_throughput=0.5,
                d_k_seconds=2.0,
                slots_per_worker=0,
                target_backlog_seconds=30.0,
                utilization_target=0.85,
            )

    def test_zero_target_backlog_seconds_is_rejected(self) -> None:
        """target_backlog_seconds is the denominator on the queue term; must be > 0."""
        with pytest.raises(ValueError, match="target_backlog_seconds must be > 0"):
            compute_capacity_target_workers(
                queue_depth=10,
                observed_throughput=0.5,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=0.0,
                utilization_target=0.85,
            )

    def test_utilization_target_above_one_is_rejected(self) -> None:
        """Cannot operate above 100% utilisation."""
        with pytest.raises(ValueError, match=r"utilization_target must be in \(0, 1\]"):
            compute_capacity_target_workers(
                queue_depth=0,
                observed_throughput=0.5,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=1.1,
            )

    def test_zero_utilization_target_is_rejected(self) -> None:
        """utilization_target=0 would divide by zero."""
        with pytest.raises(ValueError, match=r"utilization_target must be in \(0, 1\]"):
            compute_capacity_target_workers(
                queue_depth=0,
                observed_throughput=0.5,
                d_k_seconds=2.0,
                slots_per_worker=1,
                target_backlog_seconds=30.0,
                utilization_target=0.0,
            )


class TestCapacityTargetMonotonicity:
    """Adversarial: the target is monotone in each load-driving input."""

    @pytest.fixture
    def baseline(self) -> int:
        """Reference target for the monotonicity probes."""
        result = compute_capacity_target_workers(
            queue_depth=10,
            observed_throughput=0.5,
            d_k_seconds=2.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result is not None
        return result

    def test_higher_throughput_yields_at_least_baseline_target(self, baseline: int) -> None:
        """Doubling arrival rate cannot reduce the worker target."""
        result = compute_capacity_target_workers(
            queue_depth=10,
            observed_throughput=1.0,
            d_k_seconds=2.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result is not None
        assert result >= baseline

    def test_higher_d_k_yields_at_least_baseline_target(self, baseline: int) -> None:
        """Slower per-task service time cannot reduce the worker target."""
        result = compute_capacity_target_workers(
            queue_depth=10,
            observed_throughput=0.5,
            d_k_seconds=4.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result is not None
        assert result >= baseline

    def test_higher_queue_depth_yields_at_least_baseline_target(self, baseline: int) -> None:
        """Larger backlog cannot reduce the worker target."""
        result = compute_capacity_target_workers(
            queue_depth=100,
            observed_throughput=0.5,
            d_k_seconds=2.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result is not None
        assert result >= baseline


class TestCapacityTargetRoundingDiscipline:
    """A single ceil happens at the worker boundary so the slot count is rounded up exactly once."""

    def test_fractional_slots_round_up_exactly_once(self) -> None:
        """target_slots=2.6 rounds to 3 slots; with slots/worker=2, ceil(3/2)=2 workers."""
        result = compute_capacity_target_workers(
            queue_depth=0,
            observed_throughput=1.3,
            d_k_seconds=1.0,
            slots_per_worker=2,
            target_backlog_seconds=30.0,
            utilization_target=0.5,
        )
        # target_rate = 1.3
        # target_slots = ceil(1.3 * 1.0 / 0.5) = ceil(2.6) = 3
        # target_workers = ceil(3 / 2) = 2
        assert result == 2


@attrs.frozen
class _CapacitySizerSnapshot:
    """Convenience holder for an end-to-end signal trace assertion."""

    queue_depth: int
    throughput: float
    d_k: float
    expected_target: int


class TestCapacityTargetEndToEndSnapshot:
    """A multi-input snapshot pinning the closed-form formula against hand-derived numbers."""

    @pytest.mark.parametrize(
        "snapshot",
        [
            _CapacitySizerSnapshot(queue_depth=0, throughput=0.0, d_k=10.0, expected_target=0),
            _CapacitySizerSnapshot(queue_depth=10, throughput=0.0, d_k=3.0, expected_target=2),
            _CapacitySizerSnapshot(queue_depth=30, throughput=1.0, d_k=2.0, expected_target=5),
        ],
    )
    def test_matches_hand_derived_targets(self, snapshot: _CapacitySizerSnapshot) -> None:
        """Each row pins the formula at a reference point; the closed-form equality is the contract."""
        result = compute_capacity_target_workers(
            queue_depth=snapshot.queue_depth,
            observed_throughput=snapshot.throughput,
            d_k_seconds=snapshot.d_k,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert result == snapshot.expected_target
