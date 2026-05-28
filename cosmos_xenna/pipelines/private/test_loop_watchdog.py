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


"""Tests for the saturation-aware scheduler loop watchdog context manager."""

import logging
import time
from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private.scheduling_py.lifecycle import loop_watchdog as loop_watchdog_module
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.loop_watchdog import loop_watchdog


class _FakeHistogram:
    """Records ``(value, tags)`` pairs in place of ``ray.util.metrics.Histogram``."""

    def __init__(self) -> None:
        self.observations: list[tuple[float, dict[str, str]]] = []

    def observe(self, value: float, tags: dict[str, str] | None = None) -> None:
        self.observations.append((value, dict(tags or {})))


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture."""
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


class TestLoopWatchdog:
    """Pin the contract of the loop-watchdog context manager."""

    def test_slow_cycle_emits_warn_and_histogram(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A body exceeding ``threshold * interval_s`` emits one WARN log and one histogram observation."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(loop_watchdog_module, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        # Budget = 0.5 * 0.1s = 50 ms; sleep 200 ms so the body
        # definitively overruns the threshold even on slow CI runners.
        sleep_s = 0.2
        with loop_watchdog(pipeline_name="test-pipeline", threshold_fraction=0.5, interval_s=0.1):
            time.sleep(sleep_s)

        assert len(fake_histogram.observations) == 1, "Histogram must observe exactly one duration per cycle"
        observed_duration, observed_tags = fake_histogram.observations[0]
        assert observed_duration >= sleep_s, (
            f"Observed {observed_duration:.3f}s must be >= injected sleep {sleep_s:.3f}s"
        )
        assert observed_tags == {"pipeline": "test-pipeline"}

        warn_records = [r for r in loguru_caplog.records if r.levelno >= logging.WARNING]
        assert len(warn_records) == 1, f"Expected exactly one WARN record, got {len(warn_records)}"
        assert "loop watchdog" in warn_records[0].message
        assert f"{observed_duration:.2f}s" in warn_records[0].message

    def test_fast_cycle_does_not_warn(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A body well under the threshold observes the duration with no WARN log."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(loop_watchdog_module, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        # Budget = 0.5 * 10s = 5 s; cycle is sub-millisecond.
        with loop_watchdog(pipeline_name="test-pipeline", threshold_fraction=0.5, interval_s=10.0):
            pass

        assert len(fake_histogram.observations) == 1, "Histogram observation is unconditional"
        observed_duration, observed_tags = fake_histogram.observations[0]
        assert observed_duration < 1.0, "Fast cycle should complete in well under one second"
        assert observed_tags == {"pipeline": "test-pipeline"}

        warn_records = [r for r in loguru_caplog.records if r.levelno >= logging.WARNING]
        assert warn_records == [], f"Fast cycle must not emit WARN, got: {[r.message for r in warn_records]}"

    def test_watchdog_records_duration_when_body_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raising body still observes its duration on the histogram before the exception propagates."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(loop_watchdog_module, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        def slow_raising_body() -> None:
            time.sleep(0.05)
            msg = "synthetic body failure"
            raise RuntimeError(msg)

        with (
            pytest.raises(RuntimeError, match="synthetic body failure"),
            loop_watchdog(pipeline_name="test-pipeline", threshold_fraction=0.5, interval_s=10.0),
        ):
            slow_raising_body()

        assert len(fake_histogram.observations) == 1, (
            "Histogram must observe the cycle duration even when the body raises"
        )
        observed_duration, observed_tags = fake_histogram.observations[0]
        assert observed_duration >= 0.05
        assert observed_tags == {"pipeline": "test-pipeline"}

    def test_threshold_boundary_strict_greater_than(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cycle whose duration equals ``threshold * interval_s`` does NOT warn (strict ``>``)."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(loop_watchdog_module, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        # Inject a deterministic perf_counter_ns: first call returns
        # 100_000_000_000 ns (start), second call returns
        # 100_500_000_000 ns (the ``finally`` block sample) so
        # duration_s = 0.5. With threshold_fraction=0.5 and
        # interval_s=1.0, threshold_s = 0.5 -- equal to the duration.
        # Strict ``>`` must reject this and leave the WARN unfired.
        clock_samples = iter([100_000_000_000, 100_500_000_000])
        monkeypatch.setattr(time, "perf_counter_ns", lambda: next(clock_samples))

        with loop_watchdog(pipeline_name="test-pipeline", threshold_fraction=0.5, interval_s=1.0):
            pass

        assert len(fake_histogram.observations) == 1
        observed_duration, _tags = fake_histogram.observations[0]
        assert observed_duration == pytest.approx(0.5, abs=1e-9), (
            "Forced clock should yield exactly the boundary duration"
        )

        warn_records = [r for r in loguru_caplog.records if r.levelno >= logging.WARNING]
        assert warn_records == [], (
            f"A cycle exactly at the threshold must not warn (strict > semantics); got: "
            f"{[r.message for r in warn_records]}"
        )

    def test_observation_carries_pipeline_tag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pipeline tag value passed in is the value reported on the histogram."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(loop_watchdog_module, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        with loop_watchdog(pipeline_name="my-pipeline", threshold_fraction=0.5, interval_s=10.0):
            pass

        assert len(fake_histogram.observations) == 1
        _duration, observed_tags = fake_histogram.observations[0]
        assert observed_tags == {"pipeline": "my-pipeline"}
