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
    assert est.speed("s", now=1.0) is None
    assert est.num_returns("s") is None


def test_unknown_stage_returns_none() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    _feed(est, stage="s", count=3, returns=8.0)
    assert est.speed("other", now=3.0) is None


def test_speed_is_measured_from_task_durations() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    _feed(est, stage="s", count=3, returns=8.0)
    # 3 tasks of 0.5 s each -> mean duration 0.5 s -> 2.0 tasks/s/worker.
    assert est.speed("s", now=3.0) == pytest.approx(2.0)


def test_num_returns_is_ewma_of_observed_returns() -> None:
    est = estimator.PipelineRateEstimator(_WINDOW_S, 5)
    _feed(est, stage="s", count=4, returns=8.0)
    # Constant returns -> EWMA settles at that value.
    assert est.num_returns("s") == pytest.approx(8.0)
