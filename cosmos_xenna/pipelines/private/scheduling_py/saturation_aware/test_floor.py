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

"""Focused unit tests for the scale-down release gate (native-extension-free).

The throughput math (cap_src, bottleneck, w_sustain) now lives in
``capacity.py`` and is tested in ``test_capacity.py``; these tests cover only
the floor's release gate, which consumes a supplied ``w_sustain`` and decides
how far the solver may shrink each stage.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain, floor

# A two-stage "clip-extract (CPU) -> caption (GPU)" pipeline, 1 video -> 8 clips.
_CHAIN = (1.0, 8.0)
_BATCH = (1, 1)
_DEEP_STOCK = (5000.0, 5000.0)  # plenty of upstream work, in source units
_EMPTY_STOCK = (0.0, 0.0)


def _params(release_confirm_cycles: int = 2) -> floor.FloorParams:
    return floor.FloorParams(release_confirm_cycles=release_confirm_cycles, min_workers=1)


def _inputs(
    *,
    workers: tuple[int, int],
    w_sustain: tuple[int, int],
    stock: tuple[float, float],
    active: tuple[float, float] = (0.0, 0.0),
    chain_factors: tuple[float, float] = _CHAIN,
    protect_downstream_of: int = -1,
) -> floor.FloorInputs:
    # active_depths only affects the zero-fanout (chain <= 0) release branch.
    # _CHAIN is all-positive, so positive-chain tests are unaffected by it.
    return floor.FloorInputs(
        workers=workers,
        chain=chain_factors,
        stock_src=stock,
        active_depths=active,
        batch_sizes=_BATCH,
        w_sustain=w_sustain,
        protect_downstream_of=protect_downstream_of,
    )


def _caption_floor_after_two_cycles(stock: tuple[float, float], params: floor.FloorParams) -> int:
    """Run the floor two cycles at fixed inputs and return caption's floor.

    Two cycles is exactly ``release_confirm_cycles`` for ``_params()``, so a
    persistently low stock releases on the second cycle.
    """
    state = floor.FloorState.initial(2)
    args = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=stock)
    first = floor.compute_floors(args, state, params)
    second = floor.compute_floors(args, first.state, params)
    return second.plan.floors[1]


def test_holds_clamps_deletes_to_w_sustain() -> None:
    """While stock is present the floor clamps deletes to ``min(w_sustain, workers)``."""
    result = floor.compute_floors(
        _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    # w_sustain 5 < workers 15 -> the solver may shrink caption down to 5, no further.
    assert result.plan.floors[1] == 5


def test_floor_never_exceeds_current_workers() -> None:
    """The floor is a shrink-veto, never a grow command."""
    result = floor.compute_floors(
        _inputs(workers=(10, 10), w_sustain=(1, 18), stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    # w_sustain 18 > workers 10 -> floor = min(18, 10) = 10, not 18.
    assert result.plan.floors[1] == 10


def test_floor_never_exceeds_zero_current_workers() -> None:
    """A zero-worker stage gets a zero floor, not a growth request."""
    result = floor.compute_floors(
        _inputs(workers=(0, 0), w_sustain=(1, 1), stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    assert result.plan.floors == (0, 0)


def test_downstream_protection_holds_current_workers_while_stock_is_in_flight() -> None:
    """H1 blocks downstream shrink behind the current rate-source candidate."""
    protected = floor.compute_floors(
        _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_DEEP_STOCK, protect_downstream_of=0),
        floor.FloorState.initial(2),
        _params(),
    )
    drained = floor.compute_floors(
        _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_EMPTY_STOCK, protect_downstream_of=0),
        floor.FloorState.initial(2),
        _params(),
    )
    assert protected.plan.floors[1] == 15
    assert drained.plan.floors[1] == 5


def test_source_stage_is_clamped_like_any_other() -> None:
    """There is no source special case: stage 0 holds at ``min(w_sustain, workers)`` too."""
    result = floor.compute_floors(
        _inputs(workers=(10, 15), w_sustain=(4, 5), stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    assert result.plan.floors[0] == 4


def test_drain_releases_to_min_after_confirm() -> None:
    """With stock drained, the floor holds for the confirm window, then releases to MIN."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    args = _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_EMPTY_STOCK)

    first = floor.compute_floors(args, state, params)
    assert first.plan.floors[1] == 5  # cycle 1: still held at the sustain clamp (streak 1 < 2)

    second = floor.compute_floors(args, first.state, params)
    assert second.plan.floors[1] == 1  # cycle 2: confirm reached -> released to MIN


def test_release_requires_low_stock_even_when_confirmation_is_zero() -> None:
    """A zero confirmation count must not release while upstream stock is present."""
    params = _params(release_confirm_cycles=0)
    result = floor.compute_floors(
        _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        params,
    )
    assert result.plan.floors[1] == 5
    assert result.state.release_streak[1] == 0


def test_active_stock_blocks_release_that_queue_only_stock_would_trigger() -> None:
    """In-flight upstream work must keep caption warm when local queues are empty.

    At the bad moment both inter-stage queues read empty, so a queue-only stock
    (depths ``[0, 0]``) releases caption to MIN after the confirm window. An
    active stock that also counts clip-extraction's in-flight + pool-queued
    videos (depths ``[15, 0]``) keeps the release gate shut, so caption holds.
    """
    params = _params(release_confirm_cycles=2)
    queue_only = chain.whole_chain_stock([0.0, 0.0], _CHAIN)
    active = chain.whole_chain_stock([15.0, 0.0], _CHAIN)

    assert _caption_floor_after_two_cycles((queue_only[0], queue_only[1]), params) == 1
    assert _caption_floor_after_two_cycles((active[0], active[1]), params) == 15


def _zero_fanout_inputs(active_caption_depth: float) -> floor.FloorInputs:
    """A drop stage (chain[1] == 0): no source-normalized stock reaches stage 1."""
    return floor.FloorInputs(
        workers=(10, 15),
        chain=(1.0, 0.0),
        stock_src=(0.0, 0.0),
        active_depths=(0.0, active_caption_depth),
        batch_sizes=_BATCH,
        w_sustain=(1, 14),
    )


def test_zero_fanout_stage_holds_floor_while_local_work_remains() -> None:
    """A drop stage (chain == 0) is not released while it still owns admitted work.

    whole_chain_stock() cannot express a zero-fanout stage's own depth in source
    units, so stock_src reads 0 even with work in flight. Gating release on
    active_depths keeps the release streak at 0 (no premature release to MIN).
    """
    params = _params(release_confirm_cycles=2)
    busy = _zero_fanout_inputs(active_caption_depth=5.0)

    first = floor.compute_floors(busy, floor.FloorState.initial(2), params)
    second = floor.compute_floors(busy, first.state, params)

    # Held branch: floor = min(w_sustain 14, workers 15) = 14.
    assert first.plan.floors[1] == 14
    assert first.state.release_streak[1] == 0
    assert second.state.release_streak[1] == 0
    assert second.plan.floors[1] > params.min_workers


def test_zero_fanout_stage_releases_once_local_work_drains() -> None:
    """Once a drop stage's local work is gone, the normal low-stock release resumes."""
    params = _params(release_confirm_cycles=2)
    drained = _zero_fanout_inputs(active_caption_depth=0.0)

    first = floor.compute_floors(drained, floor.FloorState.initial(2), params)
    second = floor.compute_floors(drained, first.state, params)

    # Cycle 1: streak reaches 1 (< confirm=2); still held at the sustain clamp.
    assert first.state.release_streak[1] == 1
    # Cycle 2: streak reaches the confirm window -> released to MIN.
    assert second.plan.floors[1] == params.min_workers


def test_scale_down_floor_policy_carries_state_across_cycles() -> None:
    """The stateful policy holds for the confirm window, then releases to MIN.

    Verifies the release streak accumulates inside the policy across calls, so
    the scheduler no longer threads ``FloorState`` by hand.
    """
    policy = floor.ScaleDownFloorPolicy.create(2, _params(release_confirm_cycles=2))
    args = _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_EMPTY_STOCK)
    assert policy.plan(args).floors[1] == 5
    assert policy.plan(args).floors[1] == 1


def test_first_cycle_follows_desired_floor() -> None:
    """Initial state follows the capacity hold target immediately."""
    result = floor.compute_floors(
        _inputs(workers=(1, 2), w_sustain=(1, 1), stock=_DEEP_STOCK),
        floor.FloorState.initial(2),
        _params(),
    )
    assert result.plan.floors == (1, 1)
    assert not result.plan.decisions[1].shrink_deferred


def test_transient_dip_does_not_shrink() -> None:
    """A one-cycle lower hold target is deferred, then cleared on recovery."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=_DEEP_STOCK)
    dipped = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_DEEP_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(dipped, first.state, params)
    third = floor.compute_floors(steady, second.state, params)

    assert second.plan.floors[1] == 15
    assert second.plan.decisions[1].shrink_deferred
    assert second.plan.decisions[1].shrink_streak == 1
    assert third.plan.floors[1] == 15
    assert third.plan.decisions[1].shrink_streak == 0


def test_sustained_drop_shrinks_after_confirm() -> None:
    """A lower hold target applies after the confirmation window."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=_DEEP_STOCK)
    dipped = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_DEEP_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(dipped, first.state, params)
    third = floor.compute_floors(dipped, second.state, params)

    assert second.plan.floors[1] == 15
    assert second.plan.decisions[1].shrink_deferred
    assert third.plan.floors[1] == 14
    assert not third.plan.decisions[1].shrink_deferred


def test_deeper_one_cycle_dip_does_not_replace_pending_floor() -> None:
    """A confirmation window applies the conservative pending shrink floor."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=_DEEP_STOCK)
    shallow = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_DEEP_STOCK)
    deep = _inputs(workers=(10, 15), w_sustain=(1, 1), stock=_DEEP_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(shallow, first.state, params)
    third = floor.compute_floors(deep, second.state, params)

    assert second.plan.floors[1] == 15
    assert second.plan.decisions[1].pending_shrink_floor == 14
    assert third.plan.floors[1] == 14
    assert third.plan.decisions[1].pending_shrink_floor == 0


def test_pending_shrink_floor_never_exceeds_current_workers() -> None:
    """A carried pending floor is clamped when workers already dropped."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=_DEEP_STOCK)
    shallow = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_DEEP_STOCK)
    already_shrunk = _inputs(workers=(10, 13), w_sustain=(1, 1), stock=_DEEP_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(shallow, first.state, params)
    third = floor.compute_floors(already_shrunk, second.state, params)

    assert third.plan.floors[1] == 13
    assert third.plan.floors[1] <= already_shrunk.workers[1]


def test_stable_overprovision_unpins() -> None:
    """A persistent lower hold target shrinks instead of pinning a spare worker."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(1, 2), w_sustain=(1, 2), stock=_DEEP_STOCK)
    lower = _inputs(workers=(1, 2), w_sustain=(1, 1), stock=_DEEP_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(lower, first.state, params)
    third = floor.compute_floors(lower, second.state, params)

    assert second.plan.floors[1] == 2
    assert second.plan.decisions[1].shrink_deferred
    assert third.plan.floors[1] == 1


def test_floor_rises_immediately() -> None:
    """A higher hold target takes effect without waiting for confirmation."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    low = _inputs(workers=(10, 10), w_sustain=(1, 5), stock=_DEEP_STOCK)
    high = _inputs(workers=(10, 10), w_sustain=(1, 8), stock=_DEEP_STOCK)

    first = floor.compute_floors(low, state, params)
    second = floor.compute_floors(high, first.state, params)

    assert second.plan.floors[1] == 8
    assert second.plan.decisions[1].shrink_streak == 0


def test_active_stock_blocks_vllm_release_while_shrink_is_deferred() -> None:
    """Active upstream stock prevents release while a lower hold target confirms."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=_DEEP_STOCK)
    dipped = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_DEEP_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(dipped, first.state, params)

    assert second.plan.floors[1] > params.min_workers
    assert not second.plan.decisions[1].releasing
    assert second.plan.decisions[1].shrink_deferred


def test_confirmed_drain_releases_even_if_shrink_was_deferred() -> None:
    """A confirmed whole-chain drain releases even after a deferred shrink."""
    params = _params(release_confirm_cycles=2)
    state = floor.FloorState.initial(2)
    steady = _inputs(workers=(10, 15), w_sustain=(1, 15), stock=_DEEP_STOCK)
    dipped = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_DEEP_STOCK)
    drained = _inputs(workers=(10, 15), w_sustain=(1, 14), stock=_EMPTY_STOCK)

    first = floor.compute_floors(steady, state, params)
    second = floor.compute_floors(dipped, first.state, params)
    third = floor.compute_floors(drained, second.state, params)
    fourth = floor.compute_floors(drained, third.state, params)

    assert second.plan.decisions[1].shrink_deferred
    assert fourth.plan.floors[1] == params.min_workers
    assert fourth.plan.decisions[1].releasing
    assert not fourth.plan.decisions[1].shrink_deferred


def test_mismatched_input_length_raises() -> None:
    """A short floor-input tuple is a programming error."""
    short_w_sustain: tuple[int, ...] = (1,)
    mismatched = floor.FloorInputs(
        workers=(10, 15),
        chain=_CHAIN,
        stock_src=_DEEP_STOCK,
        active_depths=(0.0, 0.0),
        batch_sizes=_BATCH,
        w_sustain=short_w_sustain,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        floor.compute_floors(mismatched, floor.FloorState.initial(2), _params())
