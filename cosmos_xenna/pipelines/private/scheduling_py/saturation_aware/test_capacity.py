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

"""Focused unit tests for the throughput / capacity model (native-extension-free).

Covers cap_src, the sticky bottleneck ladder (bottleneck_rate / next_bottleneck_rate),
asymmetric throughput smoothing, and the two per-stage targets (w_sustain hold,
w_target grow).
"""

import math

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import capacity


def _params(
    *,
    capacity_headroom: float = 0.10,
    hysteresis_margin: float = 0.15,
    switch_confirm: int = 2,
    feeder_pressure_confirm: int = 2,
    feeder_arrival_horizon_s: float = 10.0,
    feeder_boost_max_multiplier: float = 2.0,
) -> capacity.CapacityParams:
    # alpha_down_gpu = alpha_down_cpu / 4 mirrors the scheduler's GPU release slowdown.
    return capacity.CapacityParams(
        alpha_up=1.0,
        alpha_down_cpu=1.0 / 6.0,
        alpha_down_gpu=1.0 / 24.0,
        capacity_headroom=capacity_headroom,
        hysteresis_margin=hysteresis_margin,
        switch_confirm=switch_confirm,
        feeder_pressure_confirm=feeder_pressure_confirm,
        feeder_arrival_horizon_s=feeder_arrival_horizon_s,
        feeder_boost_max_multiplier=feeder_boost_max_multiplier,
        min_workers=1,
    )


def _inputs(
    *,
    workers: tuple[int, ...],
    speed: tuple[float, ...],
    chain: tuple[float, ...] | None = None,
    is_gpu: tuple[bool, ...] | None = None,
    is_manual: tuple[bool, ...] | None = None,
    local_qin: tuple[float, ...] | None = None,
    local_pending_depth: tuple[float, ...] | None = None,
    local_input_threshold: tuple[float, ...] | None = None,
    active_depth: tuple[float, ...] | None = None,
    ready_workers: tuple[int, ...] | None = None,
) -> capacity.CapacityInputs:
    num = len(workers)
    return capacity.CapacityInputs(
        workers=workers,
        speed=speed,
        chain=chain if chain is not None else (1.0,) * num,
        is_gpu=is_gpu if is_gpu is not None else (False,) * num,
        is_manual=is_manual if is_manual is not None else (False,) * num,
        local_qin=local_qin if local_qin is not None else (0.0,) * num,
        local_pending_depth=local_pending_depth if local_pending_depth is not None else (0.0,) * num,
        local_input_threshold=local_input_threshold if local_input_threshold is not None else (1.0,) * num,
        active_depth=active_depth if active_depth is not None else (0.0,) * num,
        ready_workers=ready_workers if ready_workers is not None else workers,
    )


def _state(
    *,
    a_ewma: tuple[float | None, ...],
    target_speed_ewma: tuple[float | None, ...] | None = None,
    bottleneck: int,
    bottleneck_streak: int = 0,
    feeder_pressure_streak: tuple[int, ...] | None = None,
) -> capacity.CapacityState:
    """Return capacity state with feeder-pressure streaks initialized."""
    return capacity.CapacityState(
        a_ewma=a_ewma,
        target_speed_ewma=target_speed_ewma if target_speed_ewma is not None else (None,) * len(a_ewma),
        bottleneck=bottleneck,
        bottleneck_streak=bottleneck_streak,
        feeder_pressure_streak=feeder_pressure_streak if feeder_pressure_streak is not None else (0,) * len(a_ewma),
    )


def test_asymmetric_ewma_initializes_to_first_sample() -> None:
    assert capacity.asymmetric_ewma(None, 8.0, 0.6, 0.1) == 8.0


def test_asymmetric_ewma_is_fast_up_slow_down() -> None:
    up = capacity.asymmetric_ewma(2.0, 10.0, 0.6, 0.1)
    down = capacity.asymmetric_ewma(10.0, 2.0, 0.6, 0.1)
    # Rising moves most of the way (alpha_up); falling barely moves (alpha_down).
    assert up == pytest.approx(0.6 * 10.0 + 0.4 * 2.0)
    assert down == pytest.approx(0.1 * 2.0 + 0.9 * 10.0)
    assert (up - 2.0) > (10.0 - down)  # up step larger than down step


def test_self_bottleneck_stage_sustains_its_current_size() -> None:
    """A stage that is its own bottleneck sustains exactly its current workers."""
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(10, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, True)))
    # cap_src = [1.0, 0.9375]; caption is the global bottleneck (0.9375).
    # a_raw = 8 * 0.9375 = 7.5; w_sustain = ceil(7.5 / 0.5) = 15.
    assert plan.bottleneck_stage == 1
    assert plan.stages[1].w_sustain == 15


def test_overfed_non_bottleneck_stage_sustain_matches_bottleneck() -> None:
    """A stage fed faster than the global bottleneck sustains down to the bottleneck rate."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4), is_gpu=(False, False, True)))
    # cap_src = [100, 60, 2.8]; bottleneck = stage 2 at 2.8.
    # Stage 1: a_raw = 1 * 2.8 = 2.8; w_sustain = ceil(2.8 / 1.0) = 3.
    assert plan.bottleneck_stage == 2
    assert plan.stages[1].w_sustain == 3


def test_bottleneck_stage_sustains_its_full_size() -> None:
    """The bottleneck's own capacity bounds its sustain target, so it is not shrunk."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4), is_gpu=(False, False, True)))
    # Stage 2 (bottleneck): a_raw = 2.8; w_sustain = ceil(2.8 / 0.4) = 7.
    assert plan.stages[2].w_sustain == 7


def test_persistent_upstream_bottleneck_shrinks_sustain_target() -> None:
    """When upstream is genuinely slow, a downstream stage's sustain target follows it down."""
    prev = _state(a_ewma=(None, 2.4), bottleneck=0)
    result = capacity.compute_capacity(
        _inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, True)), prev, _params()
    )
    # cap_src = [0.3, 0.9375]; bottleneck = stage 0 (0.3).
    # a_raw = 8 * 0.3 = 2.4; w_sustain = ceil(2.4 / 0.5) = 5.
    assert result.plan.bottleneck_stage == 0
    assert result.plan.stages[1].w_sustain == 5


def test_single_cycle_dip_decays_sustain_slowly() -> None:
    """A one-cycle bottleneck drop decays the sustain target slowly, not all at once."""
    prev = _state(a_ewma=(None, 8.0, None), bottleneck=2)
    result = capacity.compute_capacity(
        _inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4), is_gpu=(False, False, True)), prev, _params()
    )
    # a_raw drops to 2.8 but the CPU slow-down holds a_ewma ~7.13 -> w_sustain = 8,
    # well above the fully decayed bottleneck-matched size (3).
    assert result.plan.stages[1].w_sustain > 3


def test_gpu_stage_sustain_decays_slower_than_cpu() -> None:
    """A GPU stage uses the slower release alpha, so its sustain target decays less per drop."""
    params = _params()
    prev = _state(a_ewma=(None, 8.0), bottleneck=0)
    dropped = _inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, True))
    cpu = _inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, False))

    gpu_result = capacity.compute_capacity(dropped, prev, params)
    cpu_result = capacity.compute_capacity(cpu, prev, params)
    assert gpu_result.plan.stages[1].w_sustain >= cpu_result.plan.stages[1].w_sustain


def test_cold_stage_is_excluded_from_the_bottleneck() -> None:
    """A stage with no speed estimate neither becomes nor drags the bottleneck."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(10.0, 0.0, 0.4), is_gpu=(False, False, True)))
    # cap_src = [100, 0, 2.8]; the cold stage is skipped -> bottleneck = stage 2.
    assert plan.bottleneck_stage == 2
    assert plan.stages[1].w_sustain == 1
    assert plan.stages[1].w_target == 1


def test_all_cold_pipeline_has_no_bottleneck_and_min_targets() -> None:
    """With no measured stage, the model reports no bottleneck and min targets."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(0.0, 0.0, 0.0)))
    assert plan.bottleneck_stage == -1
    assert plan.bottleneck_rate == 0.0
    assert all(stage.w_sustain == 1 and stage.w_target == 1 for stage in plan.stages)


def test_bottleneck_reassigned_when_incumbent_goes_cold() -> None:
    """If the incumbent loses its speed estimate, the bottleneck is re-adopted."""
    state = _state(a_ewma=(None, None), bottleneck=0)
    result = capacity.compute_capacity(_inputs(workers=(10, 5), speed=(0.0, 1.0)), state, _params())
    # cap_src = [0, 5]; the cold incumbent is dropped -> adopt the only measured stage.
    assert result.plan.bottleneck_stage == 1


def test_non_bottleneck_target_is_bottleneck_rate_plus_headroom() -> None:
    """A fast non-bottleneck stage targets bottleneck_rate*(1+headroom), not its backlog.

    Regression for run 9a14287f (Stage 01 over-growth): the upstream stage's
    growth target is bounded by the bottleneck rate, NOT by how much work is
    queued, so it can never be inflated to dozens of workers.
    """
    model = capacity.CapacityModel.create(3, _params(capacity_headroom=0.10))
    plan = model.plan(_inputs(workers=(4, 4, 8), speed=(10.0, 8.0, 0.5), is_gpu=(False, False, True)))
    # cap_src = [40, 32, 4]; bottleneck = stage 2 (4); headroom_rate = 4.4.
    # Fast stage 0: w_target = ceil(1 * 4.4 / 10) = 1 (bounded).
    assert plan.bottleneck_stage == 2
    assert plan.stages[0].w_target == 1


def test_bottleneck_target_climbs_toward_next_bottleneck_rate() -> None:
    """The bottleneck grows toward the next bottleneck (the move that raises pipeline speed)."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(4, 4, 8), speed=(10.0, 8.0, 0.5), is_gpu=(False, False, True)))
    # next_bottleneck_rate = second-min(cap_src) = 32; w_target = ceil(1 * 32 / 0.5) = 64.
    assert plan.next_bottleneck_rate == pytest.approx(32.0)
    assert plan.stages[2].w_target == 64


def test_non_bottleneck_source_is_bounded_to_headroom() -> None:
    """The source (index 0), when not the bottleneck, is bounded to the read-ahead, not its size."""
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(10, 4), speed=(1.0, 0.5), is_gpu=(False, True)))
    # cap_src = [10, 2]; bottleneck = stage 1 (2); headroom_rate = 2.2.
    # Source w_target = ceil(1 * 2.2 / 1.0) = 3, far below its current 10 (no overproduction).
    assert plan.bottleneck_stage == 1
    assert plan.stages[0].w_target == 3


def test_source_grows_toward_next_bottleneck_when_it_is_the_bottleneck() -> None:
    """When the source is the global bottleneck, it climbs toward the next bottleneck."""
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(2, 10), speed=(0.5, 1.0)))
    # cap_src = [1, 10]; bottleneck = source (1); next_bottleneck_rate = 10.
    # Source w_target = ceil(1 * 10 / 0.5) = 20 (it is what limits the pipe).
    assert plan.bottleneck_stage == 0
    assert plan.stages[0].w_target == 20


def test_hold_window_sizes_to_measured_min_rate() -> None:
    """While holding the incumbent, the sizing rate is the measured-min cap_src, not the incumbent's.

    Identity stays sticky on stage 0, but a transiently slower challenger
    (stage 1) is the true minimum, so ``bottleneck_rate`` is the measured min
    (8) and ``next_bottleneck_rate`` is the second-min excluding that argmin
    (10). This prevents a held incumbent's cap_src from inflating every stage.
    """
    params = _params(hysteresis_margin=0.15, switch_confirm=2)
    state = _state(a_ewma=(None, None, None), bottleneck=0)
    decisive = _inputs(workers=(10, 8, 20), speed=(1.0, 1.0, 1.0))
    result = capacity.compute_capacity(decisive, state, params)
    assert result.plan.bottleneck_stage == 0  # identity still sticky (held)
    assert result.plan.bottleneck_rate == pytest.approx(8.0)  # measured min, not incumbent 10
    assert result.plan.next_bottleneck_rate == pytest.approx(10.0)  # second-min excluding argmin


def test_capacity_model_holds_incumbent_then_switches_across_cycles() -> None:
    """A challenger must be decisively slower for confirm cycles before it takes over.

    Verifies the bottleneck identity persists inside the model so a one-cycle dip
    cannot flap it, and a sustained, margin-clearing challenger eventually wins.
    """
    model = capacity.CapacityModel.create(3, _params(hysteresis_margin=0.15, switch_confirm=2))
    base = _inputs(workers=(10, 20, 30), speed=(1.0, 1.0, 1.0))  # cap_src = [10, 20, 30]
    assert model.plan(base).bottleneck_stage == 0  # incumbent established (global min)

    near = _inputs(workers=(10, 9, 30), speed=(1.0, 1.0, 1.0))  # challenger only 10% slower
    assert model.plan(near).bottleneck_stage == 0  # within margin -> held

    decisive = _inputs(workers=(10, 8, 30), speed=(1.0, 1.0, 1.0))  # challenger 20% slower
    assert model.plan(decisive).bottleneck_stage == 0  # cycle 1: holding (streak 1 < 2)
    assert model.plan(decisive).bottleneck_stage == 1  # cycle 2: confirmed -> switched


def test_bottleneck_hysteresis_fields_are_exposed() -> None:
    """The plan exposes challenger state without changing the held bottleneck."""
    params = _params(hysteresis_margin=0.15, switch_confirm=2)
    result = capacity.compute_capacity(
        _inputs(workers=(10, 8, 30), speed=(1.0, 1.0, 1.0)),
        _state(a_ewma=(None, None, None), bottleneck=0),
        params,
    )
    assert result.plan.bottleneck_stage == 0
    assert result.plan.bottleneck_streak == 1
    assert result.plan.bottleneck_candidate == 1
    assert result.plan.bottleneck_candidate_rate == pytest.approx(8.0)


def test_feeder_pressure_waits_for_confirmation() -> None:
    """A first delayed starved-warm cycle suppresses downstream growth but does not boost."""
    params = _params(switch_confirm=2)
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5, 5),
            speed=(5.0, 5.0, 1.0),
            local_pending_depth=(10.0, 0.0, 0.0),
            active_depth=(300.0, 0.0, 0.0),
            ready_workers=(5, 5, 5),
        ),
        capacity.CapacityState.initial(3),
        params,
    )
    downstream = result.plan.stages[2]
    feeder = result.plan.stages[0]
    assert downstream.starved_warm
    assert downstream.suppress_growth
    assert downstream.feeder_reason == capacity.FeederReason.PENDING_CONFIRM.value
    assert result.state.feeder_pressure_streak[2] == 1
    assert feeder.feeder_boost == 0


def test_feeder_pressure_does_not_fire_when_downstream_is_busy() -> None:
    """A dry queue with no ready workers means the downstream stage is busy, not starved."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5),
            speed=(5.0, 1.0),
            local_pending_depth=(10.0, 0.0),
            active_depth=(100.0, 5.0),
            ready_workers=(5, 0),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert not result.plan.stages[1].starved_warm
    assert not result.plan.stages[1].suppress_growth
    assert result.state.feeder_pressure_streak[1] == 0


def test_feeder_pressure_skips_imminent_arrival() -> None:
    """Short upstream drain time is treated as normal pipeline latency."""
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5),
            speed=(5.0, 1.0),
            local_pending_depth=(1.0, 0.0),
            active_depth=(5.0, 0.0),
            ready_workers=(5, 5),
        ),
        prev,
        _params(),
    )
    assert result.plan.stages[1].feeder_reason == capacity.FeederReason.NO_BOOST_IMMINENT_ARRIVAL.value
    assert result.state.feeder_pressure_streak[1] == 0
    assert result.plan.stages[0].feeder_boost == 0


def test_feeder_pressure_boosts_binding_non_bottleneck() -> None:
    """A confirmed delayed dry downstream boosts the slowest upstream non-bottleneck feeder."""
    params = _params()
    prev = _state(a_ewma=(None, None, None), bottleneck=2, feeder_pressure_streak=(0, 0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5, 20),
            speed=(5.0, 1.0, 0.1),
            local_pending_depth=(50.0, 20.0, 0.0),
            active_depth=(50.0, 100.0, 0.0),
            ready_workers=(5, 5, 20),
        ),
        prev,
        params,
    )
    feeder = result.plan.stages[1]
    downstream = result.plan.stages[2]
    assert result.plan.bottleneck_stage == 2
    assert downstream.binding_feeder == 1
    assert downstream.feeder_reason == capacity.FeederReason.BOOSTED.value
    assert feeder.feeder_boost > 0
    assert feeder.feeder_downstreams == (2,)


def test_feeder_pressure_does_not_boost_global_bottleneck() -> None:
    """When the binding feeder is already the bottleneck, normal capacity owns growth."""
    prev = _state(a_ewma=(None, None), bottleneck=0, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 20),
            speed=(0.1, 1.0),
            local_pending_depth=(20.0, 0.0),
            active_depth=(100.0, 0.0),
            ready_workers=(5, 20),
        ),
        prev,
        _params(),
    )
    assert result.plan.bottleneck_stage == 0
    assert result.plan.stages[1].feeder_reason == capacity.FeederReason.NO_BOOST_GLOBAL_BOTTLENECK.value
    assert result.plan.stages[0].feeder_boost == 0


def test_feeder_pressure_does_not_boost_manual_feeder() -> None:
    """A pinned feeder cannot receive autoscaler growth pressure."""
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 20),
            speed=(5.0, 0.1),
            is_manual=(True, False),
            local_pending_depth=(100.0, 0.0),
            active_depth=(300.0, 0.0),
            ready_workers=(5, 20),
        ),
        prev,
        _params(),
    )
    assert result.plan.stages[1].feeder_reason == capacity.FeederReason.NO_BOOST_MANUAL_FEEDER.value
    assert result.plan.stages[0].feeder_boost == 0


def test_feeder_pressure_skips_manual_source_and_boosts_actionable_feeder() -> None:
    """A pinned high-delay source does not hide a boostable downstream feeder."""
    prev = _state(a_ewma=(None, None, None), bottleneck=2, feeder_pressure_streak=(0, 0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 5, 20),
            speed=(1.0, 1.0, 0.1),
            is_manual=(True, False, False),
            local_pending_depth=(300.0, 30.0, 0.0),
            active_depth=(3000.0, 100.0, 0.0),
            ready_workers=(1, 5, 20),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[2]
    actionable_feeder = result.plan.stages[1]
    assert downstream.binding_feeder == 1
    assert downstream.blocked_feeder == 0
    assert downstream.blocked_feeder_reason == capacity.FeederCandidateStatus.MANUAL.value
    assert downstream.feeder_reason == capacity.FeederReason.BOOSTED.value
    assert tuple(candidate.status for candidate in downstream.feeder_candidates) == (
        capacity.FeederCandidateStatus.MANUAL,
        capacity.FeederCandidateStatus.ACTIONABLE,
    )
    assert actionable_feeder.feeder_boost > 0


def test_feeder_pressure_logs_blocked_manual_when_no_actionable_feeder_exists() -> None:
    """A pinned source is reported as blocked when no boostable feeder exists."""
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 20),
            speed=(1.0, 0.1),
            is_manual=(True, False),
            local_pending_depth=(300.0, 0.0),
            active_depth=(3000.0, 0.0),
            ready_workers=(1, 20),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[1]
    assert downstream.binding_feeder == -1
    assert downstream.blocked_feeder == 0
    assert downstream.blocked_feeder_reason == capacity.FeederCandidateStatus.MANUAL.value
    assert downstream.feeder_reason == capacity.FeederReason.NO_BOOST_MANUAL_FEEDER.value


def test_feeder_pressure_uses_max_not_sum_for_shared_feeder() -> None:
    """Two dry downstream stages sharing one feeder aggregate by max target."""
    prev = _state(a_ewma=(None, None, None), bottleneck=2, feeder_pressure_streak=(0, 1, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5, 20),
            speed=(5.0, 1.0, 0.1),
            local_pending_depth=(300.0, 0.0, 0.0),
            active_depth=(300.0, 0.0, 0.0),
            ready_workers=(5, 5, 20),
        ),
        prev,
        _params(),
    )
    feeder = result.plan.stages[0]
    assert feeder.feeder_downstreams == (1, 2)
    assert feeder.w_target == result.plan.stages[1].feeder_boost_cap


def test_feeder_pressure_caps_required_workers() -> None:
    """Extreme upstream stock cannot exceed the feeder boost multiplier."""
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 20),
            speed=(5.0, 0.1),
            local_pending_depth=(1000.0, 0.0),
            active_depth=(1000.0, 0.0),
            ready_workers=(5, 20),
        ),
        prev,
        _params(),
    )
    feeder = result.plan.stages[0]
    downstream = result.plan.stages[1]
    assert downstream.feeder_required_workers > downstream.feeder_boost_cap
    assert feeder.w_target <= downstream.feeder_boost_cap


def test_feeder_pressure_does_not_compound_across_cycles() -> None:
    """Identical boost inputs hold the feeder target at a fixed ceiling (no ratchet)."""
    # Each cycle recomputes the boost from the capacity-model base target, never
    # from the previously boosted value, so a sustained deficit cannot snowball.
    params = _params()
    args = _inputs(
        workers=(5, 5, 20),
        speed=(5.0, 1.0, 0.1),
        local_pending_depth=(50.0, 20.0, 0.0),
        active_depth=(50.0, 100.0, 0.0),
        ready_workers=(5, 5, 20),
    )
    prev = _state(a_ewma=(None, None, None), bottleneck=2, feeder_pressure_streak=(0, 0, 1))
    first = capacity.compute_capacity(args, prev, params)
    second = capacity.compute_capacity(args, first.state, params)
    assert first.plan.stages[1].feeder_boost > 0
    assert second.plan.stages[1].w_target == first.plan.stages[1].w_target
    assert second.plan.stages[1].feeder_boost == first.plan.stages[1].feeder_boost


def test_feeder_pressure_resets_when_local_input_recovers() -> None:
    """Recovered local pending work clears the starved downstream streak."""
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 2))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5),
            speed=(5.0, 1.0),
            local_pending_depth=(10.0, 2.0),
            active_depth=(100.0, 2.0),
            ready_workers=(5, 5),
        ),
        prev,
        _params(),
    )
    assert result.plan.stages[1].feeder_reason == capacity.FeederReason.CLEARED_LOCAL_INPUT.value
    assert result.state.feeder_pressure_streak[1] == 0


def test_feeder_pressure_clears_when_downstream_becomes_busy() -> None:
    """A still-dry downstream that fills its ready workers clears via not-warm."""
    # local_pending stays dry (<= threshold) but ready_workers (0) < w_sustain,
    # so the downstream is busy rather than starved-warm: the prior streak clears.
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5),
            speed=(5.0, 1.0),
            local_pending_depth=(10.0, 0.0),
            active_depth=(100.0, 5.0),
            ready_workers=(5, 0),
        ),
        prev,
        _params(),
    )
    assert result.plan.stages[1].feeder_reason == capacity.FeederReason.CLEARED_NOT_WARM.value
    assert result.state.feeder_pressure_streak[1] == 0


def test_starved_warm_when_current_workers_ready_but_sustain_target_is_higher() -> None:
    """A fully ready under-target stage is warm enough to request feeder pressure."""
    prev = _state(
        a_ewma=(None, 8.0),
        target_speed_ewma=(None, 1.0),
        bottleneck=1,
        feeder_pressure_streak=(0, 0),
    )
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 7),
            speed=(5.0, 1.0),
            local_pending_depth=(100.0, 0.0),
            active_depth=(300.0, 0.0),
            ready_workers=(5, 7),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[1]
    assert downstream.w_sustain > 7
    assert downstream.starved_warm
    assert downstream.feeder_reason == capacity.FeederReason.PENDING_CONFIRM.value


def test_feeder_pressure_skips_cold_or_invalid_feeder() -> None:
    """Cold or empty upstream stages cannot be selected as useful feeders."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 5),
            speed=(0.0, 1.0),
            local_pending_depth=(10.0, 0.0),
            active_depth=(100.0, 0.0),
            ready_workers=(5, 5),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert result.plan.stages[1].feeder_reason == capacity.FeederReason.NO_BOOST_INVALID_SUPPLY.value
    assert result.plan.stages[0].feeder_boost == 0


def test_feeder_pressure_demand_workers_drive_sizing() -> None:
    """Ready downstream workers can demand more feeder workers than backlog drain alone."""
    prev = _state(a_ewma=(None, None, None), bottleneck=0, feeder_pressure_streak=(0, 0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 2, 20),
            speed=(0.1, 1.0, 0.5),
            local_pending_depth=(0.0, 5.0, 0.0),
            active_depth=(0.0, 12.0, 0.0),
            ready_workers=(1, 2, 20),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[2]
    assert downstream.binding_feeder == 1
    assert downstream.feeder_effective_horizon_s == pytest.approx(5.0)
    assert downstream.feeder_drain_workers == 3
    assert downstream.feeder_demand_workers == 10
    assert downstream.feeder_queue_refill_workers == 4
    assert downstream.feeder_required_workers == 10


def test_feeder_pressure_refill_workers_drive_sizing() -> None:
    """An empty deep input buffer demands feeder workers to refill, beyond ready-worker demand."""
    prev = _state(a_ewma=(None, None, None), bottleneck=0, feeder_pressure_streak=(0, 0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 1, 20),
            speed=(0.1, 1.0, 0.1),
            local_pending_depth=(0.0, 5.0, 0.0),
            local_input_threshold=(1.0, 1.0, 5.0),
            active_depth=(0.0, 6.0, 0.0),
            ready_workers=(1, 1, 20),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[2]
    assert downstream.downstream_buffer_deficit == pytest.approx(100.0)
    assert downstream.feeder_effective_horizon_s == pytest.approx(5.0)
    assert downstream.feeder_drain_workers == 2
    assert downstream.feeder_demand_workers == 2
    assert downstream.feeder_queue_refill_workers == 20
    assert downstream.feeder_required_workers == 20


def test_feeder_pressure_buffered_downstream_keeps_full_horizon() -> None:
    """A downstream whose buffer already meets target keeps the full (unhalved) arrival horizon."""
    prev = _state(a_ewma=(None, None), bottleneck=1, feeder_pressure_streak=(0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 1),
            speed=(1.0, 0.5),
            local_pending_depth=(50.0, 2.0),
            local_input_threshold=(1.0, 2.0),
            active_depth=(6.0, 0.0),
            ready_workers=(1, 1),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[1]
    assert downstream.downstream_buffer_deficit == pytest.approx(0.0)
    assert downstream.feeder_effective_horizon_s == pytest.approx(10.0)
    assert downstream.feeder_reason == capacity.FeederReason.NO_BOOST_IMMINENT_ARRIVAL.value


def test_feeder_pressure_is_resource_agnostic() -> None:
    """Feeder sizing depends on demand, not on whether the stages hold GPUs."""

    def plan_for(is_gpu: tuple[bool, ...]) -> capacity.CapacityPlan:
        return capacity.compute_capacity(
            _inputs(
                workers=(1, 2, 20),
                speed=(0.1, 1.0, 0.5),
                is_gpu=is_gpu,
                local_pending_depth=(0.0, 5.0, 0.0),
                active_depth=(0.0, 12.0, 0.0),
                ready_workers=(1, 2, 20),
            ),
            _state(a_ewma=(None, None, None), bottleneck=0, feeder_pressure_streak=(0, 0, 1)),
            _params(),
        ).plan

    cpu = plan_for((False, False, False))
    gpu = plan_for((True, True, True))
    assert cpu.stages[2].feeder_required_workers == gpu.stages[2].feeder_required_workers
    assert cpu.stages[1].feeder_boost == gpu.stages[1].feeder_boost


def test_feeder_pressure_blocked_only_does_not_advance_streak() -> None:
    """A blocked-only feeder cycle resets the confirmation streak instead of advancing it."""
    prev = _state(a_ewma=(None, None), bottleneck=0, feeder_pressure_streak=(0, 3))
    result = capacity.compute_capacity(
        _inputs(
            workers=(5, 20),
            speed=(0.1, 1.0),
            local_pending_depth=(20.0, 0.0),
            active_depth=(100.0, 0.0),
            ready_workers=(5, 20),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[1]
    assert downstream.feeder_reason == capacity.FeederReason.NO_BOOST_GLOBAL_BOTTLENECK.value
    assert result.state.feeder_pressure_streak[1] == 0


def test_feeder_pressure_converts_demand_through_chain_factors() -> None:
    """Downstream demand is converted to the feeder item rate using the stages' chain factors."""
    prev = _state(a_ewma=(None, None, None), bottleneck=0, feeder_pressure_streak=(0, 0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 2, 20),
            speed=(0.1, 1.0, 0.5),
            chain=(1.0, 1.0, 2.0),
            local_pending_depth=(0.0, 5.0, 0.0),
            active_depth=(0.0, 12.0, 0.0),
            ready_workers=(1, 2, 20),
        ),
        prev,
        _params(),
    )
    downstream = result.plan.stages[2]
    # chain[downstream]=2 halves the per-feeder-item demand (chain=1 yields 10 / 4).
    assert downstream.feeder_demand_workers == 5
    assert downstream.feeder_queue_refill_workers == 2


def test_feeder_pressure_reports_sufficient_feeder_without_boosting() -> None:
    """When the feeder is already large enough, feeder pressure records sufficiency and adds no boost."""
    prev = _state(a_ewma=(None, None, None), bottleneck=0, feeder_pressure_streak=(0, 0, 1))
    result = capacity.compute_capacity(
        _inputs(
            workers=(1, 1, 10),
            speed=(0.6, 1.0, 0.5),
            local_pending_depth=(0.0, 5.0, 0.0),
            active_depth=(0.0, 6.0, 0.0),
            ready_workers=(1, 1, 2),
        ),
        prev,
        _params(capacity_headroom=1.0),
    )
    downstream = result.plan.stages[2]
    feeder = result.plan.stages[1]
    assert downstream.feeder_reason == capacity.FeederReason.NO_BOOST_FEEDER_SUFFICIENT.value
    assert downstream.feeder_required_workers == 2
    assert feeder.feeder_boost == 0
    assert feeder.w_target == 2


def test_transient_speed_drop_does_not_spike_target_to_raw_division_result() -> None:
    """Target sizing uses smoothed speed so one-cycle dips do not over-request."""
    prev = _state(
        a_ewma=(None, None),
        target_speed_ewma=(2.0, 2.0),
        bottleneck=1,
    )
    result = capacity.compute_capacity(
        _inputs(
            workers=(4, 4),
            speed=(2.0, 0.25),
            active_depth=(10.0, 10.0),
            ready_workers=(4, 4),
        ),
        prev,
        _params(),
    )
    dipped = result.plan.stages[1]
    raw_target = math.ceil(result.plan.next_bottleneck_rate / 0.25)
    assert dipped.target_speed > 0.25
    assert dipped.w_target < raw_target
    assert result.state.target_speed_ewma[1] == dipped.target_speed


def test_stage_capacity_echoes_trusted_speed() -> None:
    """Each StageCapacity carries the trusted speed used this cycle (0.0 when cold)."""
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(10, 5), speed=(2.0, 0.0)))
    assert plan.stages[0].speed == 2.0
    assert plan.stages[1].speed == 0.0


def test_mismatched_input_lengths_raise() -> None:
    """A short input tuple is a programming error and fails fast."""
    with pytest.raises(ValueError, match="length mismatch"):
        capacity.compute_capacity(
            capacity.CapacityInputs(
                workers=(1, 2),
                speed=(1.0,),
                chain=(1.0, 1.0),
                is_gpu=(False, False),
                is_manual=(False, False),
                local_qin=(0.0, 0.0),
                local_pending_depth=(0.0, 0.0),
                local_input_threshold=(1.0, 1.0),
                active_depth=(0.0, 0.0),
                ready_workers=(1, 2),
            ),
            capacity.CapacityState.initial(2),
            _params(),
        )


def test_mismatched_feeder_signal_lengths_raise() -> None:
    """A short feeder-pressure signal tuple is a programming error."""
    with pytest.raises(ValueError, match="length mismatch"):
        capacity.compute_capacity(
            _inputs(workers=(1, 2), speed=(1.0, 1.0), local_pending_depth=(0.0,)),
            capacity.CapacityState.initial(2),
            _params(),
        )


def test_mismatched_state_length_raises() -> None:
    """Previous state of the wrong length is a programming error and fails fast."""
    with pytest.raises(ValueError, match="length mismatch"):
        capacity.compute_capacity(
            _inputs(workers=(1, 2), speed=(1.0, 1.0)),
            capacity.CapacityState.initial(3),
            _params(),
        )
