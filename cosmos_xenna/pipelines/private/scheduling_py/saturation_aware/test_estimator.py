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

"""Unit tests for the pure-measured per-stage rate estimator."""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import estimator

_WINDOW_S = 1000.0
_TASK_DURATION_S = 0.5


def _feed(est: estimator.PipelineRateEstimator, *, stage: str, count: int, returns: float) -> None:
    for i in range(1, count + 1):
        est.observe(stage, duration_s=_TASK_DURATION_S, num_returns=returns, now=float(i))


def test_cold_start_without_data_returns_none() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    assert est.speed("s", now=1.0, inflight=0) is None
    assert est.num_returns("s") is None


def test_unknown_stage_returns_none() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    _feed(est, stage="s", count=3, returns=8.0)
    assert est.speed("other", now=3.0, inflight=0) is None


def test_speed_is_measured_from_task_durations() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    _feed(est, stage="s", count=3, returns=8.0)
    # 3 tasks of 0.5 s each -> mean duration 0.5 s -> 2.0 tasks/s/worker.
    assert est.speed("s", now=3.0, inflight=0) == pytest.approx(2.0)


def test_num_returns_is_ewma_of_observed_returns() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    _feed(est, stage="s", count=4, returns=8.0)
    # Constant returns -> EWMA settles at that value.
    assert est.num_returns("s") == pytest.approx(8.0)


def test_empty_instant_skip_does_not_poison_speed() -> None:
    """An empty + instant skip must not enter speed or the trust count."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5, 1e-3)  # window, averaging_samples, eps=1ms
    for i in range(1, 4):  # 3 real tasks @ 5 s, 2 returns
        est.observe("s", duration_s=5.0, num_returns=2.0, now=float(i))
    est.observe("s", duration_s=1e-4, num_returns=0.0, now=4.0)  # empty + instant skip (~0.1 ms)
    assert est.speed("s", now=4.0, inflight=0) == pytest.approx(0.2)  # 1/5 s, not ~10000/s
    assert est.sample_count("s") == 3  # skip excluded from the trust count


def test_slow_zero_return_filter_is_kept() -> None:
    """A real filter/drop (zero returns but real duration >= eps) is still measured.

    Zero output is a legitimate result; a stage that does real work then returns
    nothing still has a real drain rate, so it must keep feeding the speed window.
    """
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5, 1e-3)
    for i in range(1, 6):
        est.observe("f", duration_s=2.0, num_returns=0.0, now=float(i))  # 0 returns, real 2 s
    assert est.speed("f", now=5.0, inflight=0) == pytest.approx(0.5)  # counted normally (1/2 s)
    assert est.sample_count("f") == 5


def test_default_eps_used_when_not_supplied() -> None:
    """The defaulted eps still excludes an instant empty when no value is supplied."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)  # no eps arg -> module default
    est.observe("s", duration_s=1e-6, num_returns=0.0, now=1.0)
    assert est.sample_count("s") == 0  # instant empty excluded from the trust count
    assert est.speed("s", now=1.0, inflight=0) is None  # below trust count -> cold


def test_averaging_depth_retains_samples_beyond_window() -> None:
    """averaging_samples retains old samples so one slow task cannot crater the rate.

    With a short window, a large averaging depth keeps the recent fast samples
    instead of letting the time window drop them, so a single slow task is
    averaged against many fast ones rather than dominating ``1/mean(duration)``.
    """
    window_s = 5.0
    est = estimator.PipelineRateEstimator(window_s, 20)  # averaging depth 20
    for i in range(1, 20):
        est.observe("s", duration_s=1.0, num_returns=1.0, now=float(i))
    est.observe("s", duration_s=135.0, num_returns=1.0, now=20.0)
    # All 20 samples are retained despite the 5 s window: mean = (19 + 135)/20 =
    # 7.7 s -> ~0.13/s, not the 1/135 ~0.0074/s a windowed-only estimate gives.
    assert est.speed("s", now=20.0, inflight=0) == pytest.approx(20.0 / 154.0)


def test_in_flight_ceiling_decays_speed_during_a_stall() -> None:
    """A busy stage with no recent completion has its rate aged toward 1/elapsed."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    for i in range(5):  # 5 tasks of 1 s -> last completion at t=5, windowed rate 1.0/s
        est.observe("s", duration_s=1.0, num_returns=1.0, now=float(i))
    assert est.speed("s", now=5.0, inflight=1) == pytest.approx(1.0)  # elapsed 0 -> no cap
    assert est.speed("s", now=105.0, inflight=1) == pytest.approx(1.0 / 100.0)  # aged to 1/elapsed


def test_idle_stage_keeps_its_measured_rate() -> None:
    """An idle (inflight==0) stage is starved, not stalled, so it keeps its rate."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    for i in range(5):
        est.observe("s", duration_s=1.0, num_returns=1.0, now=float(i))
    assert est.speed("s", now=105.0, inflight=0) == pytest.approx(1.0)


def test_fresh_completion_leaves_speed_uncapped() -> None:
    """Right after a completion the ceiling is large and does not bite."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    for i in range(5):
        est.observe("s", duration_s=1.0, num_returns=1.0, now=float(i))
    assert est.speed("s", now=5.5, inflight=1) == pytest.approx(1.0)  # ceiling 1/0.5 = 2.0 > 1.0


def test_rate_is_stale_only_when_busy_and_overdue() -> None:
    """rate_is_stale is true only for a busy stage overdue past stale_multiple * mean."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    for i in range(5):  # mean duration 1 s; last completion at t=5
        est.observe("s", duration_s=1.0, num_returns=1.0, now=float(i))
    assert est.rate_is_stale("s", now=105.0, inflight=0, stale_multiple=3.0) is False  # idle
    assert est.rate_is_stale("s", now=7.0, inflight=1, stale_multiple=3.0) is False  # elapsed 2 < 3
    assert est.rate_is_stale("s", now=105.0, inflight=1, stale_multiple=3.0) is True  # elapsed 100 > 3


def test_rate_is_stale_is_false_for_a_cold_stage() -> None:
    """A stage with no samples is never stale."""
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    assert est.rate_is_stale("s", now=10.0, inflight=5, stale_multiple=3.0) is False
