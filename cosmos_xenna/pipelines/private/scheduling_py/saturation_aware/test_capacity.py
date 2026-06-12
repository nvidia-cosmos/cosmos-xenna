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
    # Arbitrary smoothing value chosen for legible per-cycle release math in the
    # tests; NOT the production default (which derives 1/24 from
    # scale_down_release_cycles * scale_down_release_slowdown).
    alpha_down: float = 1.0 / 6.0,
    capacity_headroom: float = 0.10,
    hysteresis_margin: float = 0.15,
    switch_confirm: int = 2,
    speed_alpha_up: float = 1.0,
    speed_alpha_down: float | None = None,
    stale_growth_step: int = 1,
) -> capacity.CapacityParams:
    # The speed EWMA defaults mirror the arrival alphas, so speed and arrival
    # smoothing behave identically unless a test opts into the split; tests that
    # need the production speed smoothing (0.3 / 0.1) pass the alphas explicitly.
    return capacity.CapacityParams(
        alpha_up=1.0,
        alpha_down=alpha_down,
        speed_alpha_up=speed_alpha_up,
        speed_alpha_down=speed_alpha_down if speed_alpha_down is not None else alpha_down,
        capacity_headroom=capacity_headroom,
        hysteresis_margin=hysteresis_margin,
        switch_confirm=switch_confirm,
        min_workers=1,
        stale_growth_step=stale_growth_step,
    )


def _inputs(
    *,
    workers: tuple[int, ...],
    speed: tuple[float, ...],
    chain: tuple[float, ...] | None = None,
    is_manual: tuple[bool, ...] | None = None,
    local_qin: tuple[float, ...] | None = None,
    local_pending_depth: tuple[float, ...] | None = None,
    local_input_threshold: tuple[float, ...] | None = None,
    active_depth: tuple[float, ...] | None = None,
    ready_workers: tuple[int, ...] | None = None,
    rate_is_stale: tuple[bool, ...] | None = None,
) -> capacity.CapacityInputs:
    num = len(workers)
    return capacity.CapacityInputs(
        workers=workers,
        speed=speed,
        chain=chain if chain is not None else (1.0,) * num,
        is_manual=is_manual if is_manual is not None else (False,) * num,
        local_qin=local_qin if local_qin is not None else (0.0,) * num,
        local_pending_depth=local_pending_depth if local_pending_depth is not None else (0.0,) * num,
        local_input_threshold=local_input_threshold if local_input_threshold is not None else (1.0,) * num,
        active_depth=active_depth if active_depth is not None else (0.0,) * num,
        ready_workers=ready_workers if ready_workers is not None else workers,
        rate_is_stale=rate_is_stale if rate_is_stale is not None else (False,) * num,
    )


def _state(
    *,
    a_ewma: tuple[float | None, ...],
    target_speed_ewma: tuple[float | None, ...] | None = None,
    speed_peak: tuple[float | None, ...] | None = None,
    bottleneck: int,
    bottleneck_streak: int = 0,
) -> capacity.CapacityState:
    """Return capacity state with optional smoothed speed and peak-speed state.

    ``speed_peak`` defaults to unset, so the first cycle initializes each peak to
    that cycle's ``target_speed`` and the growth bound is a no-op unless a test
    seeds a higher prior peak.
    """
    return capacity.CapacityState(
        a_ewma=a_ewma,
        target_speed_ewma=target_speed_ewma if target_speed_ewma is not None else (None,) * len(a_ewma),
        speed_peak=speed_peak if speed_peak is not None else (None,) * len(a_ewma),
        bottleneck=bottleneck,
        bottleneck_streak=bottleneck_streak,
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
    plan = model.plan(_inputs(workers=(10, 15), speed=(0.1, 0.5), chain=(1.0, 8.0)))
    # cap_src = [1.0, 0.9375]; caption is the global bottleneck (0.9375).
    # a_raw = 8 * 0.9375 = 7.5; w_sustain = ceil(7.5 / 0.5) = 15.
    assert plan.bottleneck_stage == 1
    assert plan.stages[1].w_sustain == 15


def test_overfed_non_bottleneck_stage_sustain_matches_bottleneck() -> None:
    """A stage fed faster than the global bottleneck sustains down to the bottleneck rate."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4)))
    # cap_src = [100, 60, 2.8]; bottleneck = stage 2 at 2.8.
    # Stage 1: a_raw = 1 * 2.8 = 2.8; w_sustain = ceil(2.8 / 1.0) = 3.
    assert plan.bottleneck_stage == 2
    assert plan.stages[1].w_sustain == 3


def test_bottleneck_stage_sustains_its_full_size() -> None:
    """The bottleneck's own capacity bounds its sustain target, so it is not shrunk."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4)))
    # Stage 2 (bottleneck): a_raw = 2.8; w_sustain = ceil(2.8 / 0.4) = 7.
    assert plan.stages[2].w_sustain == 7


def test_persistent_upstream_bottleneck_shrinks_sustain_target() -> None:
    """When upstream is genuinely slow, a downstream stage's sustain target follows it down."""
    prev = _state(a_ewma=(None, 2.4), bottleneck=0)
    result = capacity.compute_capacity(_inputs(workers=(3, 15), speed=(0.1, 0.5), chain=(1.0, 8.0)), prev, _params())
    # cap_src = [0.3, 0.9375]; bottleneck = stage 0 (0.3).
    # a_raw = 8 * 0.3 = 2.4; w_sustain = ceil(2.4 / 0.5) = 5.
    assert result.plan.bottleneck_stage == 0
    assert result.plan.stages[1].w_sustain == 5


def test_single_cycle_dip_decays_sustain_slowly() -> None:
    """A one-cycle bottleneck drop decays the sustain target slowly, not all at once."""
    prev = _state(a_ewma=(None, 8.0, None), bottleneck=2)
    result = capacity.compute_capacity(_inputs(workers=(10, 60, 7), speed=(10.0, 1.0, 0.4)), prev, _params())
    # a_raw drops to 2.8 but the slow-down release holds a_ewma ~7.13 -> w_sustain = 8,
    # well above the fully decayed bottleneck-matched size (3).
    assert result.plan.stages[1].w_sustain > 3


def test_cold_stage_is_excluded_from_the_bottleneck() -> None:
    """A stage with no speed estimate neither becomes nor drags the bottleneck."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(10, 60, 7), speed=(10.0, 0.0, 0.4)))
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
    plan = model.plan(_inputs(workers=(4, 4, 8), speed=(10.0, 8.0, 0.5)))
    # cap_src = [40, 32, 4]; bottleneck = stage 2 (4); headroom_rate = 4.4.
    # Fast stage 0: w_target = ceil(1 * 4.4 / 10) = 1 (bounded).
    assert plan.bottleneck_stage == 2
    assert plan.stages[0].w_target == 1


def test_bottleneck_target_climbs_toward_next_bottleneck_rate() -> None:
    """The bottleneck grows toward the next bottleneck (the move that raises pipeline speed)."""
    model = capacity.CapacityModel.create(3, _params())
    plan = model.plan(_inputs(workers=(4, 4, 8), speed=(10.0, 8.0, 0.5)))
    # next_bottleneck_rate = second-min(cap_src) = 32; w_target = ceil(1 * 32 / 0.5) = 64.
    assert plan.next_bottleneck_rate == pytest.approx(32.0)
    assert plan.stages[2].w_target == 64


def test_non_bottleneck_source_is_bounded_to_headroom() -> None:
    """The source (index 0), when not the bottleneck, is bounded to the read-ahead, not its size."""
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(10, 4), speed=(1.0, 0.5)))
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


def test_classify_stages_uses_queue_gradient() -> None:
    """A stage is bottleneck only when it has input and its consumer is under-fed."""
    states = capacity.classify_stages(
        _inputs(
            workers=(2, 2, 2, 2),
            speed=(1.0, 1.0, 1.0, 1.0),
            local_qin=(2.0, 0.0, 2.0, 2.0),
            local_input_threshold=(1.0, 1.0, 1.0, 1.0),
            ready_workers=(2, 2, 2, 2),
        )
    )
    assert states == (
        capacity.StageQueueState.BOTTLENECK,
        capacity.StageQueueState.STARVED,
        capacity.StageQueueState.BUFFERED,
        capacity.StageQueueState.BALANCED,
    )


def test_deepest_queue_bottleneck_candidate_wins() -> None:
    """A downstream queue cliff owns the rate source over a shallower cliff."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 2, 2, 2),
            speed=(10.0, 5.0, 1.0, 10.0),
            local_qin=(2.0, 0.0, 2.0, 0.0),
            local_input_threshold=(1.0, 1.0, 1.0, 1.0),
        ),
        capacity.CapacityState.initial(4),
        _params(),
    )
    assert result.plan.bottleneck_candidate == 2
    assert result.plan.bottleneck_stage == 2
    assert result.plan.bottleneck_rate == pytest.approx(2.0)


def test_pre_bottleneck_buffered_stage_is_not_selected() -> None:
    """A full producer with a full consumer buffer is buffered, not the bottleneck."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 2, 2),
            speed=(1.0, 0.2, 1.0),
            local_qin=(10.0, 10.0, 0.0),
            local_input_threshold=(1.0, 1.0, 1.0),
        ),
        capacity.CapacityState.initial(3),
        _params(),
    )
    assert result.plan.stages[0].queue_state is capacity.StageQueueState.BUFFERED
    assert result.plan.stages[1].queue_state is capacity.StageQueueState.BOTTLENECK
    assert result.plan.bottleneck_candidate == 1


def test_queue_candidate_uses_smoothed_capacity_for_rate() -> None:
    """A populated queue cliff is sized from the candidate's smoothed cap_src.

    Stage 0 sees its first sample, so its smoothed ``target_speed`` initializes
    to the raw speed and ``cap_src`` equals the raw source rate this cycle.
    """
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 10),
            speed=(1.0, 0.2),
            local_qin=(1.0, 0.0),
            local_input_threshold=(1.0, 1.0),
        ),
        _state(a_ewma=(None, None), target_speed_ewma=(None, 1.0), bottleneck=-1),
        _params(),
    )
    assert result.plan.bottleneck_candidate == 0
    assert result.plan.bottleneck_rate == pytest.approx(2.0)


def test_balanced_pipeline_falls_back_to_smoothed_capacity() -> None:
    """Without a queue cliff, speed smoothing protects selection from one raw dip."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 10),
            speed=(1.0, 0.1),
            local_qin=(2.0, 2.0),
            local_input_threshold=(1.0, 1.0),
            ready_workers=(2, 10),
        ),
        _state(a_ewma=(None, None), target_speed_ewma=(None, 1.0), bottleneck=-1),
        _params(),
    )
    assert result.plan.bottleneck_candidate == 0
    assert result.plan.bottleneck_rate == pytest.approx(2.0)


def test_warm_queue_cliff_follows_smoothed_capacity_not_raw_drop() -> None:
    """A warm cliff sizes from smoothed cap_src; a one-cycle raw drop is damped.

    With a stale-high smoothed speed and a collapsed raw speed, bottleneck_rate
    follows the protective smoothed cap_src instead of snapping to the raw rate.
    """
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 10),
            speed=(1.0, 0.2),
            local_qin=(1.0, 0.0),
            local_input_threshold=(1.0, 1.0),
        ),
        _state(a_ewma=(None, None), target_speed_ewma=(5.0, 1.0), bottleneck=0),
        _params(speed_alpha_up=0.3, speed_alpha_down=0.1),
    )
    assert result.plan.bottleneck_candidate == 0
    # Smoothed speed: 0.1 * 1.0 + 0.9 * 5.0 = 4.6 -> cap_src = 2 * 4.6 = 9.2,
    # not the raw 2 * 1.0 = 2.0.
    assert result.plan.bottleneck_rate == pytest.approx(9.2)


def test_modest_speed_alpha_up_damps_fast_sample_spikes() -> None:
    """Modest speed_alpha_up damps fast-sample spikes vs the no-smoothing baseline.

    A stage whose raw per-worker speed alternates fast/slow must not let a fast
    sample snap the smoothed sizing rate to the raw spike (the premature-shrink
    failure mode); a modest alpha_up keeps the peak well below it.
    """
    speeds = (0.2, 2.0, 0.2, 2.0, 0.2, 2.0)

    def peak_rate(speed_alpha_up: float) -> float:
        model = capacity.CapacityModel.create(1, _params(speed_alpha_up=speed_alpha_up, speed_alpha_down=0.1))
        return max(
            model.plan(_inputs(workers=(5,), speed=(s,), local_qin=(2.0,), ready_workers=(5,))).bottleneck_rate
            for s in speeds
        )

    # alpha_up=1.0 reproduces the old behavior: a fast sample snaps cap_src to
    # the raw spike (5 * 2.0 = 10.0). The modest 0.3 keeps the peak well below it.
    assert peak_rate(1.0) == pytest.approx(10.0)
    assert peak_rate(0.3) < 8.0


def test_cold_queue_cliff_candidate_keeps_trusted_upstream_sized() -> None:
    """A cold queue cliff must not collapse bottleneck_rate and strip upstream holds.

    Regression: a downstream stage that just (re)started has a full input queue
    (a queue cliff) but no trusted speed yet, so its ``cap_src`` is 0.0. The rate
    source must fall back to the slowest MEASURED ``cap_src`` so the trusted
    upstream feeder keeps its bottleneck-matched ``w_sustain`` instead of being
    sized down to ``min_workers`` (which the floor would later tear down).
    """
    result = capacity.compute_capacity(
        _inputs(
            # stage 0 trusted feeder; stages 1-2 just (re)started, no speed yet.
            workers=(4, 2, 2),
            speed=(1.0, 0.0, 0.0),
            # stage 1 has a full input queue, stage 2 is empty -> stage 1 is the
            # cliff, but it is cold so cap_src[1] == 0.0.
            local_qin=(2.0, 2.0, 0.0),
            local_input_threshold=(1.0, 1.0, 1.0),
        ),
        capacity.CapacityState.initial(3),
        _params(),
    )
    # The cold cliff owns growth identity, but its 0.0 cap_src must not size.
    assert result.plan.stages[1].queue_state is capacity.StageQueueState.BOTTLENECK
    assert result.plan.bottleneck_candidate == 1
    # Sizing falls back to the slowest measured cap_src (cap_src[0] = 4*1/1).
    assert result.plan.bottleneck_rate == pytest.approx(4.0)
    # A cold candidate has no own cap_src, so its candidate rate reports the bound.
    assert result.plan.bottleneck_candidate_rate == pytest.approx(4.0)
    # The trusted upstream feeder stays bottleneck-matched, not collapsed to 1.
    assert result.plan.stages[0].w_sustain == 4


def test_warm_candidate_rate_is_bounded_by_measured_min() -> None:
    """A warm queue-cliff candidate cannot set a sizing rate above the physical min.

    The candidate (last stage) is the queue cliff but has a higher ``cap_src``
    than a slower upstream feeder. Its identity still owns growth, but the
    pipeline-sizing ``bottleneck_rate`` is the slowest MEASURED ``cap_src`` (a
    serial pipeline cannot run faster than its slowest stage). The candidate's
    own rate stays visible on ``bottleneck_candidate_rate`` so the snapshot shows
    the bound engaged.
    """
    result = capacity.compute_capacity(
        # Feeder cap_src = 2*1/1 = 2.0 (the true min); candidate cap_src = 10.0.
        _inputs(
            workers=(2, 10),
            speed=(1.0, 1.0),
            local_qin=(0.0, 2.0),  # stage 0 starved, stage 1 is the warm cliff
            local_input_threshold=(1.0, 1.0),
            ready_workers=(2, 0),  # last stage drains -> bottleneck candidate
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert result.plan.bottleneck_candidate == 1
    assert result.plan.bottleneck_rate == pytest.approx(2.0)
    assert result.plan.bottleneck_candidate_rate == pytest.approx(10.0)


def test_corrupted_chain_candidate_does_not_poison_upstream_targets() -> None:
    """A chain-factor collapse on the candidate cannot inflate upstream targets.

    A near-zero (but above ``MIN_CHAIN_FACTOR``) fan-out makes the candidate's
    ``cap_src`` explode. Because sizing uses the slowest measured ``cap_src``,
    the upstream feeder's ``a_raw = chain * bottleneck_rate`` stays physical, so
    its ``w_sustain`` / ``w_target`` do not blow up.
    """
    tiny_chain = capacity.chain.MIN_CHAIN_FACTOR * 10.0
    result = capacity.compute_capacity(
        # Feeder cap_src = 2*1/1 = 2.0; candidate cap_src = 2*1/tiny ~ 200000.
        _inputs(
            workers=(2, 2),
            speed=(1.0, 1.0),
            chain=(1.0, tiny_chain),
            local_qin=(0.0, 2.0),
            local_input_threshold=(1.0, 1.0),
            ready_workers=(2, 0),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert result.plan.bottleneck_rate == pytest.approx(2.0)
    # Upstream feeder stays physically sized (single-digit), not 10^5+.
    assert result.plan.stages[0].w_sustain == 2
    assert result.plan.stages[0].w_target <= 3


def test_healthy_warm_bottleneck_rate_equals_candidate_cap_src() -> None:
    """When the candidate IS the slowest stage, the bound is a no-op.

    Guards against a regression that would under-size the normal warm-bottleneck
    case: the measured minimum equals the candidate's own ``cap_src``.
    """
    result = capacity.compute_capacity(
        # Candidate (stage 1) cap_src = 2*1/1 = 2.0 is the global min.
        _inputs(
            workers=(10, 2),
            speed=(1.0, 1.0),
            local_qin=(0.0, 2.0),
            local_input_threshold=(1.0, 1.0),
            ready_workers=(10, 0),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert result.plan.bottleneck_candidate == 1
    assert result.plan.bottleneck_rate == pytest.approx(2.0)
    assert result.plan.bottleneck_candidate_rate == pytest.approx(2.0)


def test_terminal_stage_becomes_candidate_only_when_no_ready_workers() -> None:
    """The last stage has no consumer queue, so readiness is its terminal drain signal."""
    blocked = capacity.compute_capacity(
        _inputs(
            workers=(2, 2),
            speed=(10.0, 1.0),
            local_qin=(2.0, 2.0),
            ready_workers=(2, 0),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    ready = capacity.compute_capacity(
        _inputs(
            workers=(2, 2),
            speed=(10.0, 1.0),
            local_qin=(2.0, 2.0),
            ready_workers=(2, 1),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert blocked.plan.stages[1].queue_state is capacity.StageQueueState.BOTTLENECK
    assert blocked.plan.bottleneck_candidate == 1
    assert ready.plan.stages[1].queue_state is capacity.StageQueueState.BALANCED
    assert ready.plan.bottleneck_candidate == 1


def test_queue_candidate_switch_confirm_holds_valid_incumbent() -> None:
    """A new deeper queue cliff must confirm while the incumbent remains a cliff."""
    prev = _state(a_ewma=(None, None, None), bottleneck=0, bottleneck_streak=0)
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 2, 2),
            speed=(1.0, 1.0, 1.0),
            local_qin=(2.0, 0.0, 2.0),
            local_input_threshold=(1.0, 1.0, 1.0),
            ready_workers=(2, 2, 0),
        ),
        prev,
        _params(switch_confirm=2),
    )
    assert result.plan.bottleneck_candidate == 2
    assert result.plan.bottleneck_stage == 0
    assert result.plan.bottleneck_streak == 1


def test_confirm_streak_resets_when_queue_regime_flips() -> None:
    """A streak accrued in the cap_src regime must not carry into the queue regime.

    The two confirm regimes (queue cliff vs smoothed-cap_src fallback) share one
    streak counter, so a streak from the opposite regime must be discarded on a
    flip; otherwise a switch could confirm faster than ``switch_confirm`` intends.
    """
    states = (
        capacity.StageQueueState.BOTTLENECK,
        capacity.StageQueueState.STARVED,
        capacity.StageQueueState.BOTTLENECK,
    )
    # Incumbent stage 0 is still a valid cliff; stage 2 is the deeper challenger.
    # A streak of 1 carried from the previous (fallback) regime must be dropped,
    # so the returned streak is the first confirm cycle (1), not 2.
    bottleneck, streak, candidate, from_queue = capacity._select_bottleneck_by_queue(
        states,
        cap_src=(4.0, 0.0, 2.0),
        prev_bn=0,
        prev_streak=1,
        prev_from_queue=False,
        margin=0.15,
        confirm=3,
    )
    assert from_queue is True
    assert candidate == 2
    assert bottleneck == 0  # incumbent held this cycle
    assert streak == 1  # reset on the flip, then +1 -- not the carried 2


def test_starved_incumbent_is_replaced_immediately() -> None:
    """A previous bottleneck with no input is no longer a valid incumbent."""
    prev = _state(a_ewma=(None, None, None), bottleneck=2, bottleneck_streak=0)
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 2, 2),
            speed=(1.0, 1.0, 1.0),
            local_qin=(2.0, 0.0, 0.0),
            local_input_threshold=(1.0, 1.0, 1.0),
        ),
        prev,
        _params(switch_confirm=2),
    )
    assert result.plan.stages[2].queue_state is capacity.StageQueueState.STARVED
    assert result.plan.bottleneck_stage == 0


def test_deadlock_regression_grows_full_input_producer_instead_of_starved_downstream() -> None:
    """A full producer before an empty downstream receives the growth target."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(4, 8),
            speed=(0.5, 0.1),
            local_qin=(10.0, 0.0),
            local_input_threshold=(1.0, 1.0),
        ),
        _state(a_ewma=(None, None), target_speed_ewma=(None, 1.0), bottleneck=1),
        _params(switch_confirm=2),
    )
    assert result.plan.stages[0].queue_state is capacity.StageQueueState.BOTTLENECK
    assert result.plan.stages[1].queue_state is capacity.StageQueueState.STARVED
    assert result.plan.bottleneck_stage == 0
    assert result.plan.stages[0].w_target > result.plan.stages[0].w_sustain


def test_manual_bottleneck_target_is_capped_at_current_workers() -> None:
    """A pinned bottleneck does not receive autoscaler growth pressure."""
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 10),
            speed=(1.0, 10.0),
            is_manual=(True, False),
            local_qin=(2.0, 0.0),
        ),
        capacity.CapacityState.initial(2),
        _params(),
    )
    assert result.plan.bottleneck_stage == 0
    assert result.plan.stages[0].w_target == 2


def test_manual_candidate_is_capped_during_switch_confirmation() -> None:
    """A pinned stage that is the rate-source candidate (mid switch-confirm) gets no growth.

    Stage 0 is the held incumbent bottleneck; a deeper queue cliff makes the
    pinned stage 2 the candidate for the confirm window. Capping must apply to
    the candidate too -- not only the confirmed bottleneck -- so the autoscaler
    never grows a pinned stage while it is winning the rate-source role.
    """
    prev = _state(a_ewma=(None, None, None), bottleneck=0, bottleneck_streak=0)
    result = capacity.compute_capacity(
        _inputs(
            workers=(2, 2, 2),
            speed=(1.0, 1.0, 0.1),
            is_manual=(False, False, True),
            local_qin=(2.0, 0.0, 2.0),
            local_input_threshold=(1.0, 1.0, 1.0),
            ready_workers=(2, 2, 0),
        ),
        prev,
        _params(switch_confirm=2),
    )
    assert result.plan.bottleneck_stage == 0  # incumbent held during confirm
    assert result.plan.bottleneck_candidate == 2  # pinned stage is the candidate
    assert result.plan.stages[2].w_target == 2  # capped to current workers, not grown
    assert result.plan.stages[2].w_sustain <= 2


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
                is_manual=(False, False),
                local_qin=(0.0, 0.0),
                local_pending_depth=(0.0, 0.0),
                local_input_threshold=(1.0, 1.0),
                active_depth=(0.0, 0.0),
                ready_workers=(1, 2),
                rate_is_stale=(False, False),
            ),
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


def test_source_capacities_excludes_degenerate_chain() -> None:
    """A sub-MIN_CHAIN_FACTOR chain factor yields 0.0 capacity instead of a reciprocal blowup."""
    caps = capacity._source_capacities((10, 10), (1.0, 1.0), (1.0, capacity.chain.MIN_CHAIN_FACTOR / 2))
    assert caps[0] == pytest.approx(10.0)
    assert caps[1] == 0.0


def test_under_fed_stage_is_not_a_false_bottleneck() -> None:
    """A stage whose raw speed collapsed below its smoothed target_speed is not the bottleneck."""
    prev = _state(a_ewma=(None, None), target_speed_ewma=(None, 1.0), bottleneck=-1)
    result = capacity.compute_capacity(
        _inputs(workers=(2, 10), speed=(1.0, 0.1)),
        prev,
        _params(),
    )
    underfed = result.plan.stages[1]
    # target_speed smooths the collapsed raw 0.1 back toward ~0.85, so cap_src uses
    # 10 * 0.85 = 8.5 (not raw 10 * 0.1 = 1.0); the genuinely slow stage 0 stays the min.
    assert underfed.speed == pytest.approx(0.1)
    assert underfed.target_speed > 0.5
    assert result.plan.bottleneck_stage == 0
    assert result.plan.bottleneck_candidate_rate == pytest.approx(2.0)


def test_stalled_stage_target_speed_snaps_down_to_aged_rate() -> None:
    """A stalled stage's collapsed raw speed snaps target_speed down, not damped.

    With ``rate_is_stale`` set, the protective alpha_down is bypassed so the
    stage's smoothed sizing speed follows the aged raw rate within one cycle.
    Contrast with ``test_under_fed_stage_is_not_a_false_bottleneck`` where a
    non-stale drop is damped back up.
    """
    prev = _state(a_ewma=(None, None), target_speed_ewma=(None, 1.0), bottleneck=-1)
    result = capacity.compute_capacity(
        _inputs(workers=(2, 10), speed=(1.0, 0.1), rate_is_stale=(False, True)),
        prev,
        _params(speed_alpha_up=0.3, speed_alpha_down=0.1),
    )
    stalled = result.plan.stages[1]
    assert stalled.target_speed == pytest.approx(0.1)  # snapped to the aged raw, not damped
    assert result.state.target_speed_ewma[1] == pytest.approx(0.1)


def test_stalled_stage_growth_is_bounded_to_step() -> None:
    """A stalled bottleneck's near-zero target_speed cannot explode w_target.

    The divisive ``w_target = ceil(chain * rate / target_speed)`` would request
    a node-filling burst as ``target_speed -> 0``; the stale clamp bounds growth
    to ``workers + stale_growth_step`` while keeping the hold floor.
    """
    prev = _state(a_ewma=(None, 5.0), target_speed_ewma=(2.0, 0.5), bottleneck=0)
    bounded = capacity.compute_capacity(
        _inputs(
            workers=(4, 4),
            speed=(2.0, 0.01),
            local_qin=(2.0, 2.0),
            ready_workers=(4, 4),
            rate_is_stale=(False, True),
        ),
        prev,
        _params(speed_alpha_up=0.3, speed_alpha_down=0.1, stale_growth_step=2),
    )
    stalled = bounded.plan.stages[1]
    assert stalled.w_target <= 4 + 2  # bounded step, not the divisive blowup
    assert stalled.w_target >= stalled.w_sustain  # never below the hold floor


def test_stale_growth_is_not_throttled_by_a_high_speed_peak() -> None:
    """A stall is one long task, not contention, so the peak bound must not cap its step.

    The contention peak bound and the stale growth step address different
    failure modes; with a high prior ``speed_peak`` and ``rate_is_stale`` set,
    growth follows ``workers + stale_growth_step``, not the peak-speed knee.
    """
    # speed_peak (2.0) is far above the stalled target_speed (0.01); without the
    # stale carve-out the peak knee would cap growth to ~5 instead of 4 + 2 = 6.
    # Stage 1 is the (stale) bottleneck so target_rate is next_bottleneck_rate.
    prev = _state(a_ewma=(None, 5.0), target_speed_ewma=(2.0, 2.0), speed_peak=(2.0, 2.0), bottleneck=1)
    result = capacity.compute_capacity(
        _inputs(
            workers=(4, 4),
            speed=(2.0, 0.01),
            local_qin=(2.0, 2.0),
            ready_workers=(4, 4),
            rate_is_stale=(False, True),
        ),
        prev,
        _params(speed_alpha_up=0.3, speed_alpha_down=0.1, stale_growth_step=2),
    )
    assert result.plan.stages[1].w_target == 4 + 2  # stale step, peak bound skipped


def test_stalled_stage_hold_never_drops_below_min_workers() -> None:
    """The stale hold cap to current workers must still respect the min_workers floor.

    A stage mid-ramp can hold fewer workers than min_workers; capping w_sustain
    to that count must not breach the floor invariant the cold branch enforces.
    """
    prev = _state(a_ewma=(None, 2.0), target_speed_ewma=(2.0, 0.5), bottleneck=0)
    result = capacity.compute_capacity(
        _inputs(workers=(4, 1), speed=(2.0, 0.01), rate_is_stale=(False, True)),
        prev,
        capacity.CapacityParams(
            alpha_up=1.0,
            alpha_down=1.0,
            speed_alpha_up=0.3,
            speed_alpha_down=0.1,
            capacity_headroom=0.1,
            hysteresis_margin=0.0,
            switch_confirm=2,
            min_workers=2,
        ),
    )
    stalled = result.plan.stages[1]
    assert stalled.w_sustain >= 2  # floored at min_workers, not the single current worker
    assert stalled.w_target >= stalled.w_sustain


def test_non_stale_drop_is_still_damped_not_snapped() -> None:
    """A non-stale one-cycle speed drop keeps the protective damping (no churn).

    Regression guard for the anti-churn invariant: only a genuine stall bypasses
    alpha_down, so normal completion variance does not snap target_speed down.
    """
    prev = _state(a_ewma=(None, None), target_speed_ewma=(None, 1.0), bottleneck=-1)
    result = capacity.compute_capacity(
        _inputs(workers=(2, 10), speed=(1.0, 0.1), rate_is_stale=(False, False)),
        prev,
        _params(speed_alpha_up=0.3, speed_alpha_down=0.1),
    )
    # 0.1 * 0.1 + 0.9 * 1.0 = 0.91 (damped), not the raw 0.1.
    assert result.plan.stages[1].target_speed == pytest.approx(0.91)


def test_w_target_is_real_when_bottleneck_measured() -> None:
    """Trusted stages with a measured bottleneck carry real growth targets.

    The cold-start ramp uses ``w_target_is_real`` to decide whether to enforce
    ``w_target`` as a ceiling; this locks the flag True for the normal path.
    """
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(10, 15), speed=(0.1, 0.5), chain=(1.0, 8.0)))
    assert plan.bottleneck_rate > 0.0
    assert all(stage.w_target_is_real for stage in plan.stages)


def test_w_target_not_real_at_cold_start_without_speed() -> None:
    """With no measured speed there is no bottleneck, so no target is real.

    The placeholder collapses to ``min_workers`` and is flagged not-real so the
    ramp leaves cold-start growth to the solver instead of pinning to one worker.
    """
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(0, 0), speed=(0.0, 0.0)))
    assert plan.bottleneck_rate == 0.0
    assert not any(stage.w_target_is_real for stage in plan.stages)
    assert all(stage.w_target == 1 for stage in plan.stages)  # min_workers placeholder


def test_w_target_not_real_for_collapsed_source_fanout() -> None:
    """A chain==0 downstream stage gets a placeholder even with a live bottleneck.

    This is the per-stage case a global ``bottleneck_rate > 0`` proxy would miss:
    the upstream stage keeps a real target while the stage whose source fan-out
    collapsed (chain factor 0, no source-normalized demand) is flagged not-real.
    """
    model = capacity.CapacityModel.create(2, _params())
    plan = model.plan(_inputs(workers=(10, 15), speed=(0.1, 0.5), chain=(1.0, 0.0)))
    assert plan.bottleneck_rate > 0.0  # upstream stage is measured
    assert plan.stages[0].w_target_is_real
    assert not plan.stages[1].w_target_is_real
    assert plan.stages[1].w_target == 1  # min_workers placeholder


def test_speed_peak_is_high_water_then_decays() -> None:
    """speed_peak rises immediately to a new high and relaxes geometrically below it."""
    assert capacity._speed_peak_for_cycle(None, 4.0, 0.9) == 4.0  # cold -> first sample
    assert capacity._speed_peak_for_cycle(4.0, 6.0, 0.9) == 6.0  # higher sample raises now
    assert capacity._speed_peak_for_cycle(6.0, 1.0, 0.9) == pytest.approx(5.4)  # decays by 0.9
    assert capacity._speed_peak_for_cycle(6.0, 5.9, 0.9) == pytest.approx(5.9)  # floored at sample
    assert capacity._speed_peak_for_cycle(6.0, 0.0, 0.9) == 6.0  # cold sample carries peak


def test_w_target_capped_by_peak_speed_under_contention() -> None:
    """A bottleneck whose per-worker speed collapses under contention cannot inflate w_target.

    The divisive ``w_target = ceil(chain * next_rate / target_speed)`` balloons
    as ``target_speed`` falls; the peak-speed bound holds it near the worker
    count that already meets the rate at the stage's best observed speed.
    """
    # Stage 1 is the bottleneck; its per-worker speed collapsed to 0.2 from a
    # peak of 1.0, while the feeder (stage 0) sets next_bottleneck_rate = 10.
    prev = _state(a_ewma=(None, 1.0), target_speed_ewma=(1.0, 1.0), speed_peak=(1.0, 1.0), bottleneck=1)
    result = capacity.compute_capacity(
        _inputs(workers=(10, 2), speed=(1.0, 0.2)),
        prev,
        _params(speed_alpha_up=1.0, speed_alpha_down=1.0),
    )
    stage = result.plan.stages[1]
    assert result.plan.bottleneck_stage == 1
    assert result.plan.next_bottleneck_rate == pytest.approx(10.0)
    # Uncapped divisive target would be ceil(10 / 0.2) = 50; the peak bound holds
    # it near ceil(10 / ~1.0) = 11.
    raw_divisive = math.ceil(result.plan.next_bottleneck_rate / 0.2)
    assert raw_divisive == 50
    assert stage.w_target < raw_divisive
    assert stage.w_target <= 12


def test_w_target_uncapped_when_speed_stable() -> None:
    """When per-worker speed equals its peak, the bound is a no-op (full growth)."""
    prev = _state(a_ewma=(None, 1.0), target_speed_ewma=(1.0, 1.0), speed_peak=(1.0, 1.0), bottleneck=1)
    result = capacity.compute_capacity(
        _inputs(workers=(10, 2), speed=(1.0, 1.0)),
        prev,
        _params(speed_alpha_up=1.0, speed_alpha_down=1.0),
    )
    # target_speed == peak == 1.0 -> w_target climbs to the full divisive target.
    assert result.plan.next_bottleneck_rate == pytest.approx(10.0)
    assert result.plan.stages[1].w_target == math.ceil(10.0 / 1.0)


def test_speed_peak_decay_re_enables_growth_after_sustained_slowdown() -> None:
    """A sustained genuine slowdown decays the peak, lifting the w_target cap.

    Once the peak has relaxed to the new (lower) sustained per-worker speed, the
    bound no longer holds growth back: w_target rises to the full divisive
    target so the stage can grow into a genuinely heavier workload.
    """
    # The peak has already decayed to the new sustained speed (0.2), matching
    # target_speed; the cap then equals the raw divisive target (no longer limiting).
    prev = _state(a_ewma=(None, 1.0), target_speed_ewma=(1.0, 0.2), speed_peak=(1.0, 0.2), bottleneck=1)
    result = capacity.compute_capacity(
        _inputs(workers=(10, 2), speed=(1.0, 0.2)),
        prev,
        _params(speed_alpha_up=1.0, speed_alpha_down=1.0),
    )
    assert result.plan.next_bottleneck_rate == pytest.approx(10.0)
    assert result.plan.stages[1].w_target == math.ceil(10.0 / 0.2)  # 50, fully re-enabled
