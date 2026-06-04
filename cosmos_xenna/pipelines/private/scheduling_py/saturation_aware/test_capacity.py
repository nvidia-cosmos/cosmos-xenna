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

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import capacity


def _params(
    *,
    capacity_headroom: float = 0.10,
    hysteresis_margin: float = 0.15,
    switch_confirm: int = 2,
) -> capacity.CapacityParams:
    # alpha_down_gpu = alpha_down_cpu / 4 mirrors the scheduler's GPU release slowdown.
    return capacity.CapacityParams(
        alpha_up=1.0,
        alpha_down_cpu=1.0 / 6.0,
        alpha_down_gpu=1.0 / 24.0,
        capacity_headroom=capacity_headroom,
        hysteresis_margin=hysteresis_margin,
        switch_confirm=switch_confirm,
        min_workers=1,
    )


def _inputs(
    *,
    workers: tuple[int, ...],
    speed: tuple[float, ...],
    chain: tuple[float, ...] | None = None,
    is_gpu: tuple[bool, ...] | None = None,
) -> capacity.CapacityInputs:
    num = len(workers)
    return capacity.CapacityInputs(
        workers=workers,
        speed=speed,
        chain=chain if chain is not None else (1.0,) * num,
        is_gpu=is_gpu if is_gpu is not None else (False,) * num,
    )


# --- asymmetric throughput smoothing (moved here from the floor) ---


def test_asymmetric_ewma_initializes_to_first_sample() -> None:
    assert capacity.asymmetric_ewma(None, 8.0, 0.6, 0.1) == 8.0


def test_asymmetric_ewma_is_fast_up_slow_down() -> None:
    up = capacity.asymmetric_ewma(2.0, 10.0, 0.6, 0.1)
    down = capacity.asymmetric_ewma(10.0, 2.0, 0.6, 0.1)
    # Rising moves most of the way (alpha_up); falling barely moves (alpha_down).
    assert up == pytest.approx(0.6 * 10.0 + 0.4 * 2.0)
    assert down == pytest.approx(0.1 * 2.0 + 0.9 * 10.0)
    assert (up - 2.0) > (10.0 - down)  # up step larger than down step


# --- bottleneck identification and the w_sustain hold target ---


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
    prev = capacity.CapacityState(a_ewma=(None, 2.4), bottleneck=0, bottleneck_streak=0)
    result = capacity.compute_capacity(
        _inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, True)), prev, _params()
    )
    # cap_src = [0.3, 0.9375]; bottleneck = stage 0 (0.3).
    # a_raw = 8 * 0.3 = 2.4; w_sustain = ceil(2.4 / 0.5) = 5.
    assert result.plan.bottleneck_stage == 0
    assert result.plan.stages[1].w_sustain == 5


def test_single_cycle_dip_decays_sustain_slowly() -> None:
    """A one-cycle bottleneck drop decays the sustain target slowly, not all at once."""
    prev = capacity.CapacityState(a_ewma=(None, 8.0, None), bottleneck=2, bottleneck_streak=0)
    result = capacity.compute_capacity(
        _inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4), is_gpu=(False, False, True)), prev, _params()
    )
    # a_raw drops to 2.8 but the CPU slow-down holds a_ewma ~7.13 -> w_sustain = 8,
    # well above the fully decayed bottleneck-matched size (3).
    assert result.plan.stages[1].w_sustain > 3


def test_gpu_stage_sustain_decays_slower_than_cpu() -> None:
    """A GPU stage uses the slower release alpha, so its sustain target decays less per drop."""
    params = _params()
    prev = capacity.CapacityState(a_ewma=(None, 8.0), bottleneck=0, bottleneck_streak=0)
    dropped = _inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, True))
    cpu = _inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0), is_gpu=(False, False))

    gpu_result = capacity.compute_capacity(dropped, prev, params)
    cpu_result = capacity.compute_capacity(cpu, prev, params)
    assert gpu_result.plan.stages[1].w_sustain >= cpu_result.plan.stages[1].w_sustain


# --- cold-stage exclusion ---


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
    state = capacity.CapacityState(a_ewma=(None, None), bottleneck=0, bottleneck_streak=0)
    result = capacity.compute_capacity(_inputs(workers=(10, 5), speed=(0.0, 1.0)), state, _params())
    # cap_src = [0, 5]; the cold incumbent is dropped -> adopt the only measured stage.
    assert result.plan.bottleneck_stage == 1


# --- the w_target grow target ---


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


# --- sticky bottleneck hysteresis ---


def test_hold_window_can_have_next_rate_below_bottleneck_rate() -> None:
    """While holding the incumbent, a transiently slower challenger makes T1 < T0."""
    params = _params(hysteresis_margin=0.15, switch_confirm=2)
    state = capacity.CapacityState(a_ewma=(None, None, None), bottleneck=0, bottleneck_streak=0)
    decisive = _inputs(workers=(10, 8, 20), speed=(1.0, 1.0, 1.0))
    result = capacity.compute_capacity(decisive, state, params)
    # Holding stage 0 (streak 1 < confirm): bottleneck_rate = incumbent 10, but
    # the dipped challenger sets next_bottleneck_rate = 8 (< bottleneck_rate).
    assert result.plan.bottleneck_stage == 0
    assert result.plan.bottleneck_rate == pytest.approx(10.0)
    assert result.plan.next_bottleneck_rate == pytest.approx(8.0)


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


# --- StageCapacity.speed echo and input validation ---


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
            capacity.CapacityInputs(workers=(1, 2), speed=(1.0,), chain=(1.0, 1.0), is_gpu=(False, False)),
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
