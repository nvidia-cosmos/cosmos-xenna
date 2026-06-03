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

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain, floor

# A two-stage "clip-extract (CPU) -> caption (GPU)" pipeline, 1 video -> 8 clips.
# chain = [1, 8]; caption speed 0.5 clips/s/worker; upstream speed 0.1 videos/s/worker.
# So caption's sustainable arrival A = 8 * (w0 * 0.1) = 0.8 * w0 clips/s.
_CHAIN = (1.0, 8.0)
_BATCH = (1, 1)
_IS_GPU = (False, True)
_CAPTION_SPEED = 0.5
_UP_SPEED = 0.1
_DEEP_STOCK = (5000.0, 5000.0)  # plenty of upstream work, in source units
_EMPTY_STOCK = (0.0, 0.0)


def _params() -> floor.FloorParams:
    return floor.FloorParams(
        alpha_up=0.6,
        alpha_down_cpu=1.0 / 6.0,
        alpha_down_gpu=1.0 / 18.0,
        release_confirm_cycles=2,
        min_workers=1,
    )


def _inputs(*, upstream_workers: int, caption_workers: int, stock: tuple[float, float]) -> floor.FloorInputs:
    return floor.FloorInputs(
        workers=(upstream_workers, caption_workers),
        speed=(_UP_SPEED, _CAPTION_SPEED),
        chain=_CHAIN,
        stock_src=stock,
        batch_sizes=_BATCH,
        is_gpu=_IS_GPU,
    )


def _caption_floor_after_two_cycles(stock: tuple[float, float], params: floor.FloorParams) -> int:
    """Run the floor two cycles at fixed workers and return caption's floor.

    Two cycles is exactly ``release_confirm_cycles`` for ``_params()``, so a
    persistently low stock releases on the second cycle.
    """
    state = floor.FloorState.initial(2)
    first = floor.compute_floors(_inputs(upstream_workers=10, caption_workers=15, stock=stock), state, params)
    second = floor.compute_floors(_inputs(upstream_workers=10, caption_workers=15, stock=stock), first.state, params)
    return second.floors[1]


def test_asymmetric_ewma_initializes_to_first_sample() -> None:
    assert floor.asymmetric_ewma(None, 8.0, 0.6, 0.1) == 8.0


def test_asymmetric_ewma_is_fast_up_slow_down() -> None:
    up = floor.asymmetric_ewma(2.0, 10.0, 0.6, 0.1)
    down = floor.asymmetric_ewma(10.0, 2.0, 0.6, 0.1)
    # Rising moves most of the way (alpha_up); falling barely moves (alpha_down).
    assert up == pytest.approx(0.6 * 10.0 + 0.4 * 2.0)
    assert down == pytest.approx(0.1 * 2.0 + 0.9 * 10.0)
    assert (up - 2.0) > (10.0 - down)  # up step larger than down step


def test_stage_zero_floor_is_always_min() -> None:
    """The source stage has no upstream, so it is never upstream-starved."""
    result = floor.compute_floors(
        _inputs(upstream_workers=10, caption_workers=15, stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    assert result.floors[0] == 1


def test_transient_lull_holds_expensive_stage_warm() -> None:
    """Upstream capacity is intact (w*s high) and work is queued, so caption is held at 15."""
    result = floor.compute_floors(
        _inputs(upstream_workers=10, caption_workers=15, stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    # A = 8 * (10 * 0.1) = 8.0; w_sustain = ceil(8/0.5) = 16; floor = min(16, 15) = 15.
    assert result.floors[1] == 15


def test_persistent_bottleneck_shrinks_to_sustainable_size() -> None:
    """When upstream is genuinely slow, caption shrinks to the size its feed justifies (= borrow)."""
    converged = floor.FloorState(a_ewma=(None, 2.4), release_streak=(0, 0))
    result = floor.compute_floors(
        _inputs(upstream_workers=3, caption_workers=15, stock=_DEEP_STOCK),
        converged,
        _params(),
    )
    # A = 8 * (3 * 0.1) = 2.4; w_sustain = ceil(2.4/0.5) = 5; floor = min(5, 15) = 5.
    assert result.floors[1] == 5


def test_floor_never_exceeds_current_workers() -> None:
    """The floor is a shrink-veto, never a grow command."""
    result = floor.compute_floors(
        _inputs(upstream_workers=10, caption_workers=10, stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    # w_sustain = 16 but only 10 workers exist -> floor = 10, not 16.
    assert result.floors[1] == 10


def test_drain_releases_to_min_after_confirm_with_hysteresis() -> None:
    """With queues empty, the floor holds for release_confirm cycles, then releases to MIN."""
    params = _params()
    state = floor.FloorState.initial(2)

    # Cycle 1: queues just went empty; hysteresis keeps the stage held (capacity still high).
    result1 = floor.compute_floors(_inputs(upstream_workers=10, caption_workers=15, stock=_EMPTY_STOCK), state, params)
    assert result1.floors[1] == 15

    # Cycle 2: still empty -> release_confirm reached -> released to MIN.
    result2 = floor.compute_floors(
        _inputs(upstream_workers=10, caption_workers=15, stock=_EMPTY_STOCK), result1.state, params
    )
    assert result2.floors[1] == 1


def test_release_requires_low_stock_even_when_confirmation_is_zero() -> None:
    """A zero confirmation count must not release while upstream stock is present."""
    params = floor.FloorParams(
        alpha_up=0.6,
        alpha_down_cpu=1.0 / 6.0,
        alpha_down_gpu=1.0 / 18.0,
        release_confirm_cycles=0,
        min_workers=1,
    )
    result = floor.compute_floors(
        _inputs(upstream_workers=10, caption_workers=15, stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        params,
    )
    assert result.floors[1] == 15
    assert result.state.release_streak[1] == 0


def test_cold_start_without_speed_estimate_floors_at_min() -> None:
    """Before the speed estimate warms up, the floor cannot size and defaults to MIN."""
    inputs = floor.FloorInputs(
        workers=(10, 15),
        speed=(0.1, 0.0),  # caption speed not yet estimated
        chain=_CHAIN,
        stock_src=_DEEP_STOCK,
        batch_sizes=_BATCH,
        is_gpu=_IS_GPU,
    )
    result = floor.compute_floors(inputs, floor.FloorState.initial(2), _params())
    assert result.floors[1] == 1


def test_gpu_stage_releases_slower_than_cpu_stage() -> None:
    """A GPU stage uses the slower alpha_down, so its floor decays less per cycle on a capacity drop."""
    params = _params()
    high = floor.FloorState(a_ewma=(None, 8.0), release_streak=(0, 0))
    dropped = _inputs(upstream_workers=3, caption_workers=15, stock=_DEEP_STOCK)  # A drops 8 -> 2.4

    gpu_result = floor.compute_floors(dropped, high, params)
    cpu_inputs = floor.FloorInputs(
        workers=dropped.workers,
        speed=dropped.speed,
        chain=dropped.chain,
        stock_src=dropped.stock_src,
        batch_sizes=dropped.batch_sizes,
        is_gpu=(False, False),
    )
    cpu_result = floor.compute_floors(cpu_inputs, high, params)
    # Slower decay -> GPU floor stays >= CPU floor after the same single drop cycle.
    assert gpu_result.floors[1] >= cpu_result.floors[1]


def test_active_stock_blocks_release_that_queue_only_stock_would_trigger() -> None:
    """In-flight upstream work must keep caption warm when local queues are empty.

    Reproduction of the caption scale-down oscillation: at the bad moment both
    inter-stage queues read empty, so a queue-only stock (depths ``[0, 0]``)
    releases caption to MIN after the confirm window. An *active* stock that also
    counts clip-extraction's in-flight + pool-queued videos (10 used slots + 5
    pool-queued = 15 stage-0 input samples, depths ``[15, 0]``) keeps the release
    gate shut, so caption holds at 15.
    """
    params = _params()  # release_confirm_cycles = 2
    queue_only = chain.whole_chain_stock([0.0, 0.0], _CHAIN)
    active = chain.whole_chain_stock([15.0, 0.0], _CHAIN)

    assert _caption_floor_after_two_cycles((queue_only[0], queue_only[1]), params) == 1
    assert _caption_floor_after_two_cycles((active[0], active[1]), params) == 15
