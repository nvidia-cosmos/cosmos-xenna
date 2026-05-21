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


"""Tests for the stuck-plan detector and its INFO promotion / metrics."""

import logging
from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private.scheduling_py import stuck_plan
from cosmos_xenna.pipelines.private.scheduling_py.stuck_plan import StuckPlanDetector


class _FakeGauge:
    """Records ``set`` calls in place of ``ray.util.metrics.Gauge``."""

    def __init__(self) -> None:
        self.values: list[tuple[float, dict[str, str]]] = []

    def set(self, value: float, tags: dict[str, str] | None = None) -> None:
        self.values.append((value, dict(tags or {})))


class _FakeCounter:
    """Records ``inc`` calls in place of ``ray.util.metrics.Counter``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def inc(self, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        del value
        self.calls.append(dict(tags or {}))


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


@pytest.fixture
def fake_metrics(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeGauge, _FakeCounter]:
    """Patch the detector module's Gauge and Counter with structural fakes."""
    gauge = _FakeGauge()
    counter = _FakeCounter()
    monkeypatch.setattr(stuck_plan, "_STUCK_PLAN_ACTIVE_GAUGE", gauge)
    monkeypatch.setattr(stuck_plan, "_STUCK_PLAN_CYCLES_COUNTER", counter)
    return gauge, counter


class TestStuckPlanDetector:
    """Pin the contract of the per-stage stuck-plan detector."""

    def test_threshold_breach_emits_info_once(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """Streaming above the threshold for two cycles fires the INFO log only on the first crossing."""
        gauge, counter = fake_metrics
        detector = StuckPlanDetector()

        # Crossing cycle: stuck_cycles == threshold => INFO fires, gauge=1, counter+=1.
        detector.update(stage_name="stage", stuck_cycles=3, threshold_cycles=3, last_intent=2, pipeline_name="p")
        # Continued breach: gauge stays 1, counter increments, INFO does NOT re-fire.
        detector.update(stage_name="stage", stuck_cycles=4, threshold_cycles=3, last_intent=2, pipeline_name="p")

        info_records = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_records) == 1
        assert "stuck plan: stage 'stage'" in info_records[0].message
        assert gauge.values == [(1.0, {"stage": "stage", "pipeline": "p"})] * 2
        assert counter.calls == [{"stage": "stage", "pipeline": "p"}] * 2

    def test_recovery_emits_info_and_clears_gauge(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """Counter reset to 0 after a fired latch emits the recovery INFO and clears the gauge."""
        gauge, _counter = fake_metrics
        detector = StuckPlanDetector()

        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")
        # Recovery: counter back to 0, latch was fired -> recovery INFO + gauge=0.
        detector.update(stage_name="s", stuck_cycles=0, threshold_cycles=3, last_intent=1, pipeline_name="p")

        info_records = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_records) == 2
        assert "recovered" in info_records[1].message
        assert gauge.values[-1] == (0.0, {"stage": "s", "pipeline": "p"})

    def test_sub_threshold_does_not_promote(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """Below-threshold counter values keep the gauge at 0 with no INFO promotion."""
        gauge, counter = fake_metrics
        detector = StuckPlanDetector()

        for cycles in range(1, 3):
            detector.update(stage_name="s", stuck_cycles=cycles, threshold_cycles=3, last_intent=1, pipeline_name="p")

        assert [r for r in loguru_caplog.records if r.levelno == logging.INFO] == []
        assert all(value == 0.0 for value, _ in gauge.values)
        assert counter.calls == []

    def test_re_arming_after_recovery_fires_info_again(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """A second stuck episode after recovery emits a fresh INFO promotion."""
        del fake_metrics
        detector = StuckPlanDetector()

        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")
        detector.update(stage_name="s", stuck_cycles=0, threshold_cycles=3, last_intent=1, pipeline_name="p")
        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")

        info_records = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        # 2 stuck-plan promotion + 1 recovery = 3 lines.
        assert len(info_records) == 3
        assert sum("recovered" in r.message for r in info_records) == 1
        assert sum("stuck for" in r.message for r in info_records) == 2

    def test_reset_clears_latch(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """``reset()`` re-arms the latch so the next breach emits a fresh INFO."""
        del fake_metrics
        detector = StuckPlanDetector()

        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")
        detector.reset()
        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")

        info_records = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert sum("stuck for" in r.message for r in info_records) == 2

    def test_reset_clears_active_gauge_for_every_raised_label(
        self,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """``reset()`` emits gauge=0 for every (stage, pipeline) it ever raised to 1."""
        gauge, _counter = fake_metrics
        detector = StuckPlanDetector()

        detector.update(stage_name="s1", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")
        detector.update(stage_name="s2", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p")
        prefix_len = len(gauge.values)
        detector.reset()
        clearing_calls = gauge.values[prefix_len:]

        # ``_fired`` preserves insertion order, so ``reset()`` clears
        # in the same order the latch entries were created.
        assert clearing_calls == [
            (0.0, {"stage": "s1", "pipeline": "p"}),
            (0.0, {"stage": "s2", "pipeline": "p"}),
        ]

    def test_reset_clears_gauges_across_multiple_pipeline_tags(
        self,
        fake_metrics: tuple[_FakeGauge, _FakeCounter],
    ) -> None:
        """Composite-key storage keeps per-pipeline gauges distinct under reset()."""
        gauge, _counter = fake_metrics
        detector = StuckPlanDetector()

        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p1")
        detector.update(stage_name="s", stuck_cycles=3, threshold_cycles=3, last_intent=1, pipeline_name="p2")
        prefix_len = len(gauge.values)
        detector.reset()
        clearing_calls = gauge.values[prefix_len:]

        assert clearing_calls == [
            (0.0, {"stage": "s", "pipeline": "p1"}),
            (0.0, {"stage": "s", "pipeline": "p2"}),
        ]
