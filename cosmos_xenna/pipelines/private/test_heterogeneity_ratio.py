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


"""Tests for the cluster heterogeneity ratio helper.

Pin the contract of
:func:`cosmos_xenna.pipelines.private.scheduling_py.bottleneck.compute_heterogeneity_ratio`:

  * Ratio matches ``max(D_k) / min(D_k)`` over stages whose service
    time was finite this cycle. Cold-start stages (NaN, zero, or
    negative service time) are excluded from numerator and
    denominator alike.
  * When fewer than two stages are finite the ratio is undefined;
    the gauge observes ``math.nan`` and no streak update or log
    fires.
  * The streak counter increments each consecutive above-threshold
    cycle and the INFO log fires exactly once when the streak
    reaches ``warn_streak_cycles``.
  * The streak counter resets the moment the ratio drops to or
    below the threshold; a fresh climb-back can re-arm a second
    INFO log only after that drop.
  * The INFO log format matches the regex pinned in the plan row
    (Liu & Ying 2026 ``ksub`` / ``ksuper`` analog wording).
"""

import logging
import math
import re
from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private.scheduling_py import bottleneck
from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import (
    HeterogeneityWarnState,
    compute_heterogeneity_ratio,
)


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in ``test_bottleneck_score.py`` and
    ``test_setup_aware_max_queued.py``: the heterogeneity helper
    logs via ``cosmos_xenna.utils.python_log`` (a thin loguru
    wrapper), so caplog cannot see the records unless loguru is
    bridged into the stdlib handler.
    """
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


@pytest.fixture
def gauge_observations(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, dict[str, str]]]:
    """Capture every ``set()`` call on the cluster heterogeneity gauge.

    Mirrors the gauge-capture fixture in ``test_bottleneck_score.py``
    so the helper can be exercised without a live Ray metrics
    agent. Each call appends a ``(value, tags)`` tuple to the
    returned list.
    """
    captured: list[tuple[float, dict[str, str]]] = []

    def fake_set(value: float | int | None, tags: dict[str, str] | None = None) -> None:
        if value is None:
            return
        captured.append((float(value), dict(tags or {})))

    monkeypatch.setattr(bottleneck._HETEROGENEITY_RATIO_GAUGE, "set", fake_set)
    return captured


class TestComputeHeterogeneityRatio:
    """Pins the cluster heterogeneity ratio helper contract."""

    def test_ratio_correctness_three_stages(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
    ) -> None:
        """Three finite stages -> gauge observes max(D)/min(D) within tolerance."""
        state = HeterogeneityWarnState()

        compute_heterogeneity_ratio(
            service_times_s={"a": 0.1, "b": 0.5, "c": 1.0},
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=20.0,
            warn_streak_cycles=30,
        )

        assert len(gauge_observations) == 1
        value, tags = gauge_observations[0]
        assert value == pytest.approx(10.0)
        assert tags == {"pipeline": "test_pipeline"}

    def test_ratio_excludes_cold_start_stages(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
    ) -> None:
        """A NaN service time contributes to neither numerator nor denominator."""
        state = HeterogeneityWarnState()

        compute_heterogeneity_ratio(
            service_times_s={"a": math.nan, "b": 0.5, "c": 1.0},
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=20.0,
            warn_streak_cycles=30,
        )

        assert len(gauge_observations) == 1
        value, _tags = gauge_observations[0]
        assert value == pytest.approx(2.0)

    def test_ratio_undefined_with_fewer_than_two_finite_stages(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Only one finite stage -> gauge NaN, streak reset, no log fires."""
        state = HeterogeneityWarnState(streak_cycles=5, has_fired=False)

        compute_heterogeneity_ratio(
            service_times_s={"a": math.nan, "b": 0.5},
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=2.0,
            warn_streak_cycles=10,
        )

        assert len(gauge_observations) == 1
        value, _tags = gauge_observations[0]
        assert math.isnan(value)
        assert state.streak_cycles == 0
        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []

    def test_streak_increments_when_above_threshold(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Below-streak above-threshold cycles increment counter without firing log."""
        state = HeterogeneityWarnState()
        warn_streak = 30

        for _ in range(5):
            compute_heterogeneity_ratio(
                service_times_s={"a": 0.1, "b": 1.0},
                pipeline_name="test_pipeline",
                state=state,
                warn_threshold=5.0,
                warn_streak_cycles=warn_streak,
            )

        assert state.streak_cycles == 5
        assert state.has_fired is False
        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []

    def test_log_fires_after_streak_threshold(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Exactly one INFO log fires on the streak-completion cycle, format pinned."""
        state = HeterogeneityWarnState()
        warn_streak = 3

        for _ in range(warn_streak):
            compute_heterogeneity_ratio(
                service_times_s={"fast": 0.1, "slow": 1.0},
                pipeline_name="test_pipeline",
                state=state,
                warn_threshold=5.0,
                warn_streak_cycles=warn_streak,
            )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1, f"expected exactly one INFO log line, got {[r.message for r in info_logs]}"
        pattern = re.compile(
            r"high cluster heterogeneity \(ratio=\d+\.\d+ for \d+ cycles\); "
            r"consider raising over_provisioned_streak_min_cycles for stage \S+ "
            r"\(bottleneck D=\d+\.\d+s\) to give it more recovery margin"
        )
        assert pattern.fullmatch(info_logs[0].message), (
            f"INFO log line did not match the pinned format: {info_logs[0].message!r}"
        )
        assert "stage slow" in info_logs[0].message
        assert state.has_fired is True

    def test_streak_resets_below_threshold(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
    ) -> None:
        """Drop below threshold mid-streak resets counter; next above-threshold restarts at 1."""
        state = HeterogeneityWarnState()
        warn_streak = 30

        for _ in range(warn_streak - 1):
            compute_heterogeneity_ratio(
                service_times_s={"a": 0.1, "b": 1.0},
                pipeline_name="test_pipeline",
                state=state,
                warn_threshold=5.0,
                warn_streak_cycles=warn_streak,
            )
        assert state.streak_cycles == warn_streak - 1

        compute_heterogeneity_ratio(
            service_times_s={"a": 0.5, "b": 1.0},
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=5.0,
            warn_streak_cycles=warn_streak,
        )
        assert state.streak_cycles == 0

        compute_heterogeneity_ratio(
            service_times_s={"a": 0.1, "b": 1.0},
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=5.0,
            warn_streak_cycles=warn_streak,
        )
        assert state.streak_cycles == 1

    def test_log_does_not_fire_twice_without_drop(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Continued above-threshold cycles after first fire stay silent until drop+climb."""
        state = HeterogeneityWarnState()
        warn_streak = 3

        for _ in range(warn_streak + 5):
            compute_heterogeneity_ratio(
                service_times_s={"a": 0.1, "b": 1.0},
                pipeline_name="test_pipeline",
                state=state,
                warn_threshold=5.0,
                warn_streak_cycles=warn_streak,
            )
        first_pass_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(first_pass_logs) == 1, "first streak completion must fire exactly once even with extra cycles"

        compute_heterogeneity_ratio(
            service_times_s={"a": 0.5, "b": 1.0},
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=5.0,
            warn_streak_cycles=warn_streak,
        )
        assert state.has_fired is False, "drop below threshold must re-arm the latch"

        for _ in range(warn_streak):
            compute_heterogeneity_ratio(
                service_times_s={"a": 0.1, "b": 1.0},
                pipeline_name="test_pipeline",
                state=state,
                warn_threshold=5.0,
                warn_streak_cycles=warn_streak,
            )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 2, (
            f"second streak after drop+climb must fire one new INFO log, got {[r.message for r in info_logs]}"
        )
