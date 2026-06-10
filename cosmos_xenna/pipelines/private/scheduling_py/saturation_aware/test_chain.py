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

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain


def test_source_stage_factor_is_one() -> None:
    """Stage 0 has no upstream fan-out, so its chain factor is 1."""
    assert chain.chain_factors([1.0, 1.0], [1, 1])[0] == 1.0


def test_fan_out_accumulates_downstream() -> None:
    """A 1->8 fan-out at stage 0 makes stage 1 see 8 input items per source item."""
    assert chain.chain_factors([8.0, 1.0], [1, 1]) == [1.0, 8.0]


def test_chain_factor_is_a_running_product() -> None:
    """Successive fan-outs multiply: f0=8, f1=2 -> k = [1, 8, 16]."""
    assert chain.chain_factors([8.0, 2.0, 1.0], [1, 1, 1]) == [1.0, 8.0, 16.0]


def test_batch_size_divides_the_fan_out() -> None:
    """Fan-out per input item is num_returns / batch_size (8/4 = 2)."""
    assert chain.chain_factors([8.0, 1.0], [4, 1]) == [1.0, 2.0]


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        chain.chain_factors([1.0], [1, 1])


def test_non_positive_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        chain.chain_factors([1.0, 1.0], [1, 0])


def test_negative_num_returns_raises() -> None:
    with pytest.raises(ValueError, match=r"num_returns_per_batch\[1\] must be >= 0, got -1.0"):
        chain.chain_factors([1.0, -1.0], [1, 1])


def test_downstream_stage_sees_upstream_stock_in_source_units() -> None:
    """A starved caption stage still sees upstream backlog: 1000 videos -> 1000 source units."""
    factors = chain.chain_factors([8.0, 1.0], [1, 1])
    assert chain.whole_chain_stock([1000.0, 0.0], factors) == [1000.0, 1000.0]


def test_own_queue_is_normalized_into_source_units() -> None:
    """A stage's own queue is divided by its chain factor (80 clips / 8 = 10 source units)."""
    factors = chain.chain_factors([8.0, 1.0], [1, 1])
    assert chain.whole_chain_stock([0.0, 80.0], factors) == [0.0, 10.0]


def test_stock_is_cumulative_over_the_chain() -> None:
    """stock[k] sums every at-or-upstream queue, each in source units."""
    factors = chain.chain_factors([2.0, 2.0], [1, 1])  # k = [1, 2, 4]... here 2 stages -> [1, 2]
    assert chain.whole_chain_stock([4.0, 4.0], factors) == [4.0, 6.0]


def test_zero_fan_out_upstream_contributes_no_downstream_stock() -> None:
    """A fully dropping upstream (k=0 downstream) adds no stock past it."""
    factors = chain.chain_factors([0.0, 1.0], [1, 1])  # k = [1, 0]
    assert chain.whole_chain_stock([5.0, 7.0], factors) == [5.0, 5.0]


def test_negative_queue_depth_does_not_reduce_stock() -> None:
    """Transient negative queue readings are clamped before stock accumulation."""
    assert chain.whole_chain_stock([4.0, -4.0], [1.0, 2.0]) == [4.0, 4.0]


def test_source_stock_threshold_is_one_batch_in_source_units() -> None:
    """One batch of stage input is batch_size / chain_factor source items."""
    assert chain.source_stock_threshold(4, 8.0) == pytest.approx(0.5)


def test_source_stock_threshold_collapses_for_subminimum_chain_factor() -> None:
    """A chain factor below MIN_CHAIN_FACTOR collapses to 0, never a reciprocal blowup.

    Mirrors whole_chain_stock(), which omits the same factors: a degenerate tiny
    factor must not explode the threshold to a near-infinite value that would
    mark a busy stage as drained.
    """
    assert chain.source_stock_threshold(4, chain.MIN_CHAIN_FACTOR / 2) == 0.0


def test_source_stock_threshold_is_zero_for_non_positive_chain_factor() -> None:
    """A fully dropping stage (chain_factor <= 0) has no source-expressible work."""
    assert chain.source_stock_threshold(4, 0.0) == 0.0
    assert chain.source_stock_threshold(4, -1.0) == 0.0


def test_source_stock_threshold_is_finite_at_min_chain_factor() -> None:
    """Exactly MIN_CHAIN_FACTOR is the smallest usable factor (>= boundary)."""
    assert chain.source_stock_threshold(1, chain.MIN_CHAIN_FACTOR) == pytest.approx(1.0 / chain.MIN_CHAIN_FACTOR)
