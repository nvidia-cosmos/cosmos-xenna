# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 4-iv loop watchdog in ``Autoscaler.start_autoscale_calculation``.

The watchdog (section 4i.3 of the saturation-aware scheduler plan,
docs/scheduler/saturation-aware/18-loop-watchdog.md) wraps the body of
``Autoscaler.start_autoscale_calculation`` with a ``time.monotonic()``
bracket so every cycle observes its wall-clock duration on the
``xenna_scheduler_cycle_duration_seconds`` Prometheus histogram and a
single WARN log is emitted via ``loguru.logger.bind(pipeline=...)``
when ``duration_s > cycle_time_warn_threshold * interval_s``.

These tests pin the watchdog contract:

  * Slow cycle: WARN fires exactly once and the histogram observed the
    measured duration (within tolerance of the injected sleep).
  * Fast cycle: no WARN fires and the histogram still observed the
    (small) duration -- the histogram is unconditional.
  * Body raises: the histogram is still observed before the exception
    propagates (``try/finally`` ordering).
  * Threshold boundary: a cycle whose duration equals the resolved
    threshold does NOT warn (strict ``>`` semantics on the comparison).
"""

import logging
import time
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import streaming
from cosmos_xenna.pipelines.private.streaming import Autoscaler
from cosmos_xenna.utils.verbosity import VerbosityLevel


class _FakeHistogram:
    """Stand-in for ``ray.util.metrics.Histogram`` that records observations.

    Patched onto ``streaming._CYCLE_DURATION_HISTOGRAM`` via monkeypatch
    so tests can assert on the exact ``(duration, tags)`` pairs the
    watchdog passes to ``observe``. Using a structural fake avoids
    requiring a running Ray session and keeps the test contract
    independent of Ray's internal metric registry.
    """

    def __init__(self) -> None:
        self.observations: list[tuple[float, dict[str, str]]] = []

    def observe(self, value: float, tags: dict[str, str] | None = None) -> None:
        self.observations.append((value, dict(tags or {})))


class _FakeExecutor:
    """No-op stand-in for ``ThreadPoolExecutor``.

    ``start_autoscale_calculation`` only needs an object whose ``submit``
    returns something truthy and assignable to ``_autoscale_future``;
    the watchdog measures wall-clock time on the call itself, not on
    whatever the future eventually completes with, so a no-op submit
    keeps the test single-threaded and deterministic.
    """

    @staticmethod
    def submit(*_args: Any, **_kwargs: Any) -> object:
        return object()


class _FakeAlgorithm:
    """Stand-in for the scheduler algorithm passed into ``executor.submit``.

    Only the ``autoscale`` attribute is read (as the callable argument
    to ``submit``); the fake executor never invokes it, so the body can
    be empty.
    """

    @staticmethod
    def autoscale(_time: float, _problem_state: Any) -> None:
        return None


def _make_autoscaler(
    *,
    pipeline_name: str = "test-pipeline",
    cycle_time_warn_threshold: float = 0.5,
    cycle_interval_s: float = 0.1,
) -> Autoscaler:
    """Build a minimal ``Autoscaler`` with only the watchdog dependencies wired.

    Bypasses ``__init__`` (which requires a real allocator + pipeline
    spec + cluster resources) and sets exactly the instance attributes
    the wrapped body reads or assigns: ``_autoscale_future``,
    ``_verbosity_level``, ``_autoscale_start_time``, ``_executor``,
    ``_cycle_time_warn_threshold``, ``_cycle_interval_s``, and
    ``_pipeline_name``. The default 0.5 threshold against a 0.1 s
    interval yields a 50 ms watchdog budget that tests can straddle
    with short ``time.sleep`` calls without slowing the suite.
    """
    autoscaler = object.__new__(Autoscaler)
    autoscaler._autoscale_future = None  # type: ignore[attr-defined]
    autoscaler._verbosity_level = VerbosityLevel.NONE  # type: ignore[attr-defined]
    autoscaler._autoscale_start_time = 0.0  # type: ignore[attr-defined]
    autoscaler._executor = _FakeExecutor()  # type: ignore[attr-defined]
    autoscaler._algorithm = _FakeAlgorithm()  # type: ignore[attr-defined]
    autoscaler._cycle_time_warn_threshold = cycle_time_warn_threshold  # type: ignore[attr-defined]
    autoscaler._cycle_interval_s = cycle_interval_s  # type: ignore[attr-defined]
    autoscaler._pipeline_name = pipeline_name  # type: ignore[attr-defined]
    return autoscaler


def _install_make_problem_state(autoscaler: Autoscaler, body: Any) -> None:
    """Replace ``_make_problem_state`` with a callable invoking ``body``.

    Direct attribute assignment (rather than ``monkeypatch.setattr``)
    keeps the helper self-contained: each test constructs a fresh
    ``Autoscaler`` shell, so there is no inter-test bleed that
    monkeypatch would need to roll back. ``body`` may sleep, raise, or
    return a placeholder problem-state object; the watchdog must
    behave identically in all three cases.
    """

    def fake_make_problem_state(_pools: Any, _is_dones: Any) -> Any:
        return body()

    autoscaler._make_problem_state = fake_make_problem_state  # type: ignore[method-assign]


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in ``test_donor_warmup_grace.py`` and the
    sibling Phase C / Phase D test files. Without the bridge, the
    watchdog WARN log emitted through ``loguru.logger.bind(...)`` would
    never reach ``caplog.records`` because the project routes logging
    through loguru, which does not propagate to the stdlib ``logging``
    module by default.
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


class TestLoopWatchdog:
    """Pin the four behaviours of the Phase 4-iv loop watchdog."""

    def test_slow_cycle_emits_warn_and_histogram(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A body exceeding ``threshold * interval_s`` emits one WARN log and one histogram observation."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(streaming, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        # Budget = 0.5 * 0.1s = 50 ms; sleep 200 ms so the body
        # definitively overruns the threshold even on slow CI runners.
        sleep_s = 0.2
        autoscaler = _make_autoscaler(cycle_time_warn_threshold=0.5, cycle_interval_s=0.1)
        _install_make_problem_state(autoscaler, lambda: time.sleep(sleep_s))

        autoscaler.start_autoscale_calculation([], [])

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
        monkeypatch.setattr(streaming, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        # Budget = 0.5 * 10s = 5 s; cycle is sub-millisecond.
        autoscaler = _make_autoscaler(cycle_time_warn_threshold=0.5, cycle_interval_s=10.0)
        _install_make_problem_state(autoscaler, lambda: None)

        autoscaler.start_autoscale_calculation([], [])

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
        monkeypatch.setattr(streaming, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        autoscaler = _make_autoscaler(cycle_time_warn_threshold=0.5, cycle_interval_s=10.0)

        def raising_body() -> Any:
            time.sleep(0.05)
            raise RuntimeError("synthetic body failure")

        _install_make_problem_state(autoscaler, raising_body)

        with pytest.raises(RuntimeError, match="synthetic body failure"):
            autoscaler.start_autoscale_calculation([], [])

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
        monkeypatch.setattr(streaming, "_CYCLE_DURATION_HISTOGRAM", fake_histogram)

        # Inject a deterministic monotonic clock: first call returns
        # 100.0 (start_monotonic), second call returns 100.5 (the
        # ``finally`` block sample) so duration_s = 0.5. With
        # threshold=0.5 and interval_s=1.0, threshold_s = 0.5, equal to
        # the duration -- strict ``>`` must reject this and leave the
        # WARN unfired.
        clock_samples = iter([100.0, 100.5])
        # ``time`` is a global singleton module in CPython, so patching
        # ``time.monotonic`` here also patches the reference that
        # ``streaming`` sees via its own ``import time``. monkeypatch
        # restores the original on teardown.
        monkeypatch.setattr(time, "monotonic", lambda: next(clock_samples))

        autoscaler = _make_autoscaler(cycle_time_warn_threshold=0.5, cycle_interval_s=1.0)
        _install_make_problem_state(autoscaler, lambda: None)

        autoscaler.start_autoscale_calculation([], [])

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
