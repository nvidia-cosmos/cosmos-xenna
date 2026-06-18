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


def _params(release_confirm_cycles: int = 2, reclaim_confirm_cycles: int = 2) -> floor.FloorParams:
    return floor.FloorParams(
        release_confirm_cycles=release_confirm_cycles,
        reclaim_confirm_cycles=reclaim_confirm_cycles,
        min_workers=1,
    )


def _inputs(
    *,
    workers: tuple[int, int],
    w_sustain: tuple[int, int],
    stock: tuple[float, float],
    active: tuple[float, float] = (0.0, 0.0),
    chain_factors: tuple[float, float] = _CHAIN,
    protect_downstream_of: int = -1,
    ready_workers: tuple[int, int] = (1, 1),
    local_pending: tuple[float, float] = (0.0, 0.0),
    is_manual: tuple[bool, bool] = (False, False),
    w_target_is_real: tuple[bool, bool] = (True, True),
    reclaim_beneficial: tuple[bool, bool] = (False, False),
) -> floor.FloorInputs:
    # active_depths only affects the zero-fanout (chain <= 0) release branch.
    # _CHAIN is all-positive, so positive-chain tests are unaffected by it.
    # ready_workers/local_pending default to "idle, no backlog" so the
    # saturation+backlog veto stays off unless a test opts into it.
    # reclaim_beneficial defaults to all-False so the downstream reclaim gate
    # stays shut unless a test opts in, preserving the plain warm-keeping pin.
    return floor.FloorInputs(
        workers=workers,
        chain=chain_factors,
        stock_src=stock,
        active_depths=active,
        batch_sizes=_BATCH,
        w_sustain=w_sustain,
        ready_workers=ready_workers,
        local_pending_depths=local_pending,
        is_manual=is_manual,
        w_target_is_real=w_target_is_real,
        reclaim_beneficial=reclaim_beneficial,
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
        ready_workers=(1, 1),
        local_pending_depths=(0.0, 0.0),
        is_manual=(False, False),
        w_target_is_real=(True, True),
        reclaim_beneficial=(False, False),
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


def test_stock_at_exactly_one_batch_is_held_not_released() -> None:
    """Stock at exactly one batch counts as work (>=), so the stage is not released.

    Matches the growth gate, which treats local_pending == batch_size as a usable
    batch: the floor must not release a stage sitting on exactly one batch.
    """
    params = _params(release_confirm_cycles=2)
    # caption threshold = batch / chain = 1 / 8 = 0.125 source units (one batch).
    at_threshold = _inputs(workers=(10, 15), w_sustain=(1, 5), stock=(0.0, 0.125))
    first = floor.compute_floors(at_threshold, floor.FloorState.initial(2), params)
    second = floor.compute_floors(at_threshold, first.state, params)
    assert second.state.release_streak[1] == 0
    assert not second.plan.decisions[1].releasing
    assert second.plan.floors[1] == 5


def test_stock_just_below_one_batch_releases() -> None:
    """Stock below one batch drains: the stage releases after the confirm window."""
    params = _params(release_confirm_cycles=2)
    below = _inputs(workers=(10, 15), w_sustain=(1, 5), stock=(0.0, 0.124))
    first = floor.compute_floors(below, floor.FloorState.initial(2), params)
    second = floor.compute_floors(below, first.state, params)
    assert second.plan.decisions[1].releasing
    assert second.plan.floors[1] == params.min_workers


def test_downstream_protection_holds_at_exactly_one_batch() -> None:
    """H1 protects a downstream stage holding exactly one batch of stock (>=)."""
    params = _params(release_confirm_cycles=2)
    at_threshold = _inputs(workers=(10, 15), w_sustain=(1, 5), stock=(0.0, 0.125), protect_downstream_of=0)
    result = floor.compute_floors(at_threshold, floor.FloorState.initial(2), params)
    assert result.plan.floors[1] == 15


def test_degenerate_chain_factor_does_not_explode_threshold() -> None:
    """A sub-MIN_CHAIN_FACTOR fan-out collapses the threshold to 0, not a giant value.

    The old reciprocal threshold (batch / 1e-9 ~= 1e9) marked a deeply backlogged
    stage as drained and released it. With the collapsed threshold, real upstream
    stock keeps the stage held.
    """
    params = _params(release_confirm_cycles=2)
    degenerate = _inputs(workers=(10, 15), w_sustain=(1, 5), stock=_DEEP_STOCK, chain_factors=(1.0, 1e-9))
    first = floor.compute_floors(degenerate, floor.FloorState.initial(2), params)
    second = floor.compute_floors(degenerate, first.state, params)
    assert second.state.release_streak[1] == 0
    assert second.plan.floors[1] == 5


def test_saturated_backlogged_stage_not_shrunk() -> None:
    """A fully utilized stage holding a queued batch is held against a decayed w_sustain.

    A transient global bottleneck_rate dip decays caption's w_sustain (5 < 7
    workers), but with no ready worker and one queued batch the stage is
    demonstrably under-provisioned, so the veto holds it at its current 7.
    """
    params = _params(release_confirm_cycles=2)
    busy = _inputs(
        workers=(10, 7),
        w_sustain=(1, 5),
        stock=_DEEP_STOCK,
        ready_workers=(1, 0),
        local_pending=(0.0, 1.0),
    )
    first = floor.compute_floors(busy, floor.FloorState.initial(2), params)
    second = floor.compute_floors(busy, first.state, params)

    assert first.plan.floors[1] == 7
    assert second.plan.floors[1] == 7
    assert second.plan.decisions[1].shrink_streak == 0
    assert not second.plan.decisions[1].shrink_deferred


def test_idle_stage_still_shrinks_despite_backlog() -> None:
    """A backlog alone does not veto a shrink; an idle worker means it is over-provisioned."""
    params = _params(release_confirm_cycles=2)
    over = _inputs(
        workers=(10, 7),
        w_sustain=(1, 5),
        stock=_DEEP_STOCK,
        ready_workers=(1, 1),
        local_pending=(0.0, 1.0),
    )
    first = floor.compute_floors(over, floor.FloorState.initial(2), params)
    second = floor.compute_floors(over, first.state, params)

    assert second.plan.floors[1] == 5


def test_saturated_but_no_batch_queued_still_shrinks() -> None:
    """Full utilization without a queued batch does not veto a shrink to w_sustain."""
    params = _params(release_confirm_cycles=2)
    drained_queue = _inputs(
        workers=(10, 7),
        w_sustain=(1, 5),
        stock=_DEEP_STOCK,
        ready_workers=(1, 0),
        local_pending=(0.0, 0.0),
    )
    first = floor.compute_floors(drained_queue, floor.FloorState.initial(2), params)
    second = floor.compute_floors(drained_queue, first.state, params)

    assert second.plan.floors[1] == 5


def test_release_overrides_saturation_veto() -> None:
    """A confirmed whole-chain drain releases to MIN even while the saturation veto holds.

    The veto sets the desired floor to current workers, but a confirmed
    upstream drain (stock below one batch for the confirm window) still
    releases the stage, so the veto cannot pin a stage whose feed has ended.
    """
    params = _params(release_confirm_cycles=2)
    drained = _inputs(
        workers=(10, 7),
        w_sustain=(1, 5),
        stock=_EMPTY_STOCK,
        ready_workers=(1, 0),
        local_pending=(0.0, 1.0),
    )
    first = floor.compute_floors(drained, floor.FloorState.initial(2), params)
    second = floor.compute_floors(drained, first.state, params)

    assert first.plan.floors[1] == 7
    assert not first.plan.decisions[1].releasing
    assert second.plan.decisions[1].releasing
    assert second.plan.floors[1] == params.min_workers


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
        ready_workers=(1, 1),
        local_pending_depths=(0.0, 0.0),
        is_manual=(False, False),
        w_target_is_real=(True, True),
        reclaim_beneficial=(False, False),
    )
    with pytest.raises(ValueError, match="length mismatch"):
        floor.compute_floors(mismatched, floor.FloorState.initial(2), _params())


def test_mismatched_reclaim_beneficial_length_raises() -> None:
    """A short ``reclaim_beneficial`` tuple is caught by the length check."""
    mismatched = floor.FloorInputs(
        workers=(10, 15),
        chain=_CHAIN,
        stock_src=_DEEP_STOCK,
        active_depths=(0.0, 0.0),
        batch_sizes=_BATCH,
        w_sustain=(1, 5),
        ready_workers=(1, 1),
        local_pending_depths=(0.0, 0.0),
        is_manual=(False, False),
        w_target_is_real=(True, True),
        reclaim_beneficial=(False,),
    )
    with pytest.raises(ValueError, match="length mismatch"):
        floor.compute_floors(mismatched, floor.FloorState.initial(2), _params())


def _floor_after_cycles(args: floor.FloorInputs, params: floor.FloorParams, cycles: int) -> floor.FloorPlan:
    """Run the floor for ``cycles`` cycles at fixed inputs and return the last plan."""
    state = floor.FloorState.initial(len(args.workers))
    result = floor.compute_floors(args, state, params)
    for _ in range(cycles - 1):
        result = floor.compute_floors(args, result.state, params)
    return result.plan


def _reclaimable_downstream(
    *,
    reclaim_beneficial: tuple[bool, bool] = (False, True),
    w_target_is_real: tuple[bool, bool] = (True, True),
    is_manual: tuple[bool, bool] = (False, False),
    ready_workers: tuple[int, int] = (1, 1),
) -> floor.FloorInputs:
    """Build a downstream stage that the protect pin would hold but reclaim could release.

    Stage 1 is downstream of the rate source (``protect_downstream_of=0``),
    over-provisioned (15 workers vs ``w_sustain`` 5), idle, and fed by deep
    upstream stock - so without a reclaim signal the floor pins it at 15.
    """
    return _inputs(
        workers=(10, 15),
        w_sustain=(1, 5),
        stock=_DEEP_STOCK,
        protect_downstream_of=0,
        reclaim_beneficial=reclaim_beneficial,
        w_target_is_real=w_target_is_real,
        is_manual=is_manual,
        ready_workers=ready_workers,
    )


def test_downstream_reclaim_releases_when_beneficial_and_confirmed() -> None:
    """A beneficial, idle, over-provisioned downstream stage releases its warm pin.

    With no reclaim signal the protect pin holds stage 1 at 15. Once it has been
    reclaimable for the confirm window the floor falls to ``min(w_sustain,
    workers)`` so the solver's deletes can free workers for the bottleneck.
    """
    params = _params(release_confirm_cycles=2, reclaim_confirm_cycles=2)
    reclaimable = _reclaimable_downstream()
    assert _floor_after_cycles(reclaimable, params, 1).floors[1] == 15
    assert _floor_after_cycles(reclaimable, params, 3).floors[1] == 5


def test_downstream_reclaim_holds_warm_when_not_beneficial() -> None:
    """An idle, over-provisioned downstream stage stays warm when reclaiming helps nothing."""
    params = _params(release_confirm_cycles=2, reclaim_confirm_cycles=2)
    not_beneficial = _reclaimable_downstream(reclaim_beneficial=(False, False))
    assert _floor_after_cycles(not_beneficial, params, 6).floors[1] == 15


def test_downstream_reclaim_skipped_for_cold_stage() -> None:
    """A cold stage (placeholder ``w_target``) is never released even when beneficial."""
    params = _params(release_confirm_cycles=2, reclaim_confirm_cycles=2)
    cold = _reclaimable_downstream(w_target_is_real=(True, False))
    assert _floor_after_cycles(cold, params, 6).floors[1] == 15


def test_downstream_reclaim_skipped_for_manual_stage() -> None:
    """An operator-pinned stage is never released even when beneficial."""
    params = _params(release_confirm_cycles=2, reclaim_confirm_cycles=2)
    manual = _reclaimable_downstream(is_manual=(False, True))
    assert _floor_after_cycles(manual, params, 6).floors[1] == 15


def test_downstream_reclaim_requires_idle_capacity() -> None:
    """A fully utilized downstream stage is not reclaimed even when beneficial."""
    params = _params(release_confirm_cycles=2, reclaim_confirm_cycles=2)
    busy = _reclaimable_downstream(ready_workers=(1, 0))
    assert _floor_after_cycles(busy, params, 6).floors[1] == 15


def test_downstream_reclaim_resets_on_transient_benefit_drop() -> None:
    """A benefit signal that flickers off resets the streak, so the pin is not released."""
    params = _params(release_confirm_cycles=2, reclaim_confirm_cycles=3)
    on = _reclaimable_downstream()
    off = _reclaimable_downstream(reclaim_beneficial=(False, False))
    state = floor.FloorState.initial(2)
    first = floor.compute_floors(on, state, params)
    second = floor.compute_floors(on, first.state, params)
    third = floor.compute_floors(off, second.state, params)
    fourth = floor.compute_floors(on, third.state, params)

    assert second.plan.decisions[1].benefit_streak == 2
    assert second.plan.floors[1] == 15
    assert third.plan.decisions[1].benefit_streak == 0
    assert fourth.plan.decisions[1].benefit_streak == 1
    assert fourth.plan.floors[1] == 15


def test_cold_stage_held_at_current_workers() -> None:
    """A cold stage (no trusted target) is held at its workers, not shrunk to the placeholder w_sustain.

    Capacity emits w_sustain=min_workers for a stage with no measured speed, so
    the baseline desired collapses to 1; the cold hold keeps the floor at the
    current 4 workers instead, leaving cold growth to the cold-start ramp.
    """
    result = floor.compute_floors(
        _inputs(workers=(10, 4), w_sustain=(1, 1), stock=_DEEP_STOCK, w_target_is_real=(True, False)),
        floor.FloorState.initial(2),
        _params(),
    )
    assert result.plan.floors[1] == 4  # held at workers, not min(w_sustain=1, workers=4)=1


def test_trusted_stage_still_shrinks_to_w_sustain() -> None:
    """The cold hold is scoped to untrusted stages: a trusted stage still clamps to min(w_sustain, workers)."""
    result = floor.compute_floors(
        _inputs(workers=(10, 4), w_sustain=(1, 2), stock=_DEEP_STOCK, w_target_is_real=(True, True)),
        floor.FloorState.initial(2),
        _params(),
    )
    assert result.plan.floors[1] == 2  # min(w_sustain=2, workers=4); cold hold does not apply


def test_cold_stage_with_drained_stock_still_releases_to_min() -> None:
    """The whole-chain drain override still releases a cold stage to min_workers after the confirm window.

    The cold hold pins a cold stage only while it still owns upstream work; once
    the whole-chain stock has drained for release_confirm_cycles cycles the
    release path overrides the hold, so a finished cold stage is not pinned.
    """
    params = _params(release_confirm_cycles=2)
    args = _inputs(workers=(10, 4), w_sustain=(1, 1), stock=_EMPTY_STOCK, w_target_is_real=(True, False))
    first = floor.compute_floors(args, floor.FloorState.initial(2), params)
    assert first.plan.floors[1] == 4  # cycle 1: cold hold at workers (release streak 1 < 2)
    second = floor.compute_floors(args, first.state, params)
    assert second.plan.floors[1] == 1  # cycle 2: confirmed drain overrides the hold -> MIN
