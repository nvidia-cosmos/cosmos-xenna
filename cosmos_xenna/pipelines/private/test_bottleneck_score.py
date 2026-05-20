# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Forced-Flow-Law bottleneck-score emitter.

Pin the contract of
:func:`cosmos_xenna.pipelines.private.scheduling_py.bottleneck.emit_bottleneck_score`
plus the one end-to-end wiring point in
:class:`SaturationAwareScheduler`:

  * Per-stage gauge values match the Forced Flow Law
    ``D_k = V_k * S_k`` (with ``V_k = 1`` for Xenna's linear DAG).
  * The bottleneck stage announced in the INFO log is
    ``argmax_k D_k`` across stages whose service time was observed
    this cycle. Cold-start stages (NaN, zero, or negative service
    time) observe ``math.nan`` on the gauge and are excluded from
    the argmax.
  * Exactly one INFO log line fires per call when at least one
    stage has a finite positive ``D_k``; no INFO log fires when
    every stage is still in cold-start.
  * The INFO log format is regex-stable so operators can
    ``grep`` the line without parsing scientific notation.
  * :meth:`SaturationAwareScheduler.autoscale` invokes the helper
    exactly once per cycle.

The helper is pure observability -- these tests do not exercise
any autoscaler behaviour beyond confirming the wire-up point in
``autoscale()``.
"""

import logging
import math
import re
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py import bottleneck
from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import emit_bottleneck_score
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in ``test_donor_warmup_grace.py``: the
    bottleneck helper logs via ``cosmos_xenna.utils.python_log``
    which is a thin loguru wrapper, so caplog cannot see the records
    unless loguru is bridged into the stdlib handler.
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
    """Capture every ``set()`` call on the module-level bottleneck gauge.

    Replaces ``bottleneck._BOTTLENECK_GAUGE.set`` with a free
    function (not a bound method) so the helper can be exercised
    without a live Ray metrics agent. Each call appends a
    ``(value, tags)`` tuple to the returned list; tests inspect the
    list to verify per-stage observations.
    """
    captured: list[tuple[float, dict[str, str]]] = []

    def fake_set(value: float | int | None, tags: dict[str, str] | None = None) -> None:
        if value is None:
            return
        captured.append((float(value), dict(tags or {})))

    monkeypatch.setattr(bottleneck._BOTTLENECK_GAUGE, "set", fake_set)
    return captured


class TestEmitBottleneckScore:
    """Pins the Forced-Flow-Law bottleneck-score helper contract."""

    def test_three_stage_bottleneck_at_middle_stage(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Three stages with one large D_k: middle stage wins argmax + log.

        Setup: ``{"a": 0.05, "b": 2.0, "c": 0.10}`` -> b is the
        bottleneck with D = 2.0s and throughput bound 0.5 tasks/s.
        Pins (i) gauge values per stage within float tolerance,
        (ii) the INFO log names "b" with D=2.00s and throughput
        bound 0.50 tasks/s, (iii) exactly one INFO log per cycle.
        """
        emit_bottleneck_score(
            service_times_s={"a": 0.05, "b": 2.0, "c": 0.10},
            pipeline_name="test_pipeline",
        )

        assert len(gauge_observations) == 3
        observations_by_stage = {obs[1]["stage"]: obs[0] for obs in gauge_observations}
        assert observations_by_stage["a"] == pytest.approx(0.05)
        assert observations_by_stage["b"] == pytest.approx(2.0)
        assert observations_by_stage["c"] == pytest.approx(0.10)
        for _value, tags in gauge_observations:
            assert tags["pipeline"] == "test_pipeline"

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1, f"expected exactly one INFO log line, got {[r.message for r in info_logs]}"
        message = info_logs[0].message
        assert "bottleneck stage: b" in message
        assert "D = 2.00s" in message
        assert "throughput bound = 0.50 tasks/s" in message

    def test_cold_start_stage_excluded_from_argmax(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A NaN service time observes a NaN gauge but cannot win argmax.

        Setup: ``{"a": NaN, "b": 1.0, "c": 0.5}`` -> b is the
        bottleneck (a is excluded). Pins (i) the cold-start gauge
        is NaN, (ii) the argmax skips it, (iii) the bottleneck
        stage in the INFO log is b, not a.
        """
        emit_bottleneck_score(
            service_times_s={"a": math.nan, "b": 1.0, "c": 0.5},
            pipeline_name="test_pipeline",
        )

        assert len(gauge_observations) == 3
        observations_by_stage = {obs[1]["stage"]: obs[0] for obs in gauge_observations}
        assert math.isnan(observations_by_stage["a"])
        assert observations_by_stage["b"] == pytest.approx(1.0)
        assert observations_by_stage["c"] == pytest.approx(0.5)

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        assert "bottleneck stage: b" in info_logs[0].message
        assert "bottleneck stage: a" not in info_logs[0].message

    def test_all_cold_no_log(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every stage in cold-start: NaN gauges, no INFO log fires.

        Setup: ``{"a": NaN, "b": NaN}`` -> the helper does not
        invent a bottleneck. The gauges still observe NaN so
        Prometheus' cardinality stays stable across cycles.
        """
        emit_bottleneck_score(
            service_times_s={"a": math.nan, "b": math.nan},
            pipeline_name="test_pipeline",
        )

        assert len(gauge_observations) == 2
        for value, _tags in gauge_observations:
            assert math.isnan(value)

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == [], (
            f"no INFO log must fire when every stage is in cold-start; got {[r.message for r in info_logs]}"
        )

    def test_zero_service_time_treated_as_cold(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A 0.0 service time folds into the cold-start sentinel.

        Setup: ``{"a": 0.0, "b": 1.0}`` -> stage "a" comes from
        ``processing_speed_tasks_per_second == 0`` where
        ``S_k = 1 / 0`` is undefined. The gauge for "a" observes
        NaN and "a" is excluded from the argmax; "b" wins.
        """
        emit_bottleneck_score(
            service_times_s={"a": 0.0, "b": 1.0},
            pipeline_name="test_pipeline",
        )

        observations_by_stage = {obs[1]["stage"]: obs[0] for obs in gauge_observations}
        assert math.isnan(observations_by_stage["a"])
        assert observations_by_stage["b"] == pytest.approx(1.0)

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        assert "bottleneck stage: b" in info_logs[0].message

    def test_log_format_pins_throughput_bound(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Pin the INFO log regex so operators can grep without ambiguity.

        The format is critical for log triage:
        ``bottleneck stage: <name> (D = N.NNs, throughput bound =
        N.NN tasks/s)``. The two-decimal precision avoids
        scientific notation for typical service times (10 ms to
        10 s) and keeps the line single-pass grep-friendly.
        """
        emit_bottleneck_score(
            service_times_s={"alpha": 0.04, "beta": 2.5},
            pipeline_name="test_pipeline",
        )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        pattern = re.compile(r"bottleneck stage: \w+ \(D = \d+\.\d{2}s, throughput bound = \d+\.\d{2} tasks/s\)")
        assert pattern.fullmatch(info_logs[0].message), (
            f"INFO log line did not match the pinned format: {info_logs[0].message!r}"
        )

    def test_autoscale_invokes_helper_once_per_cycle(self) -> None:
        """End-to-end smoke: ``autoscale()`` invokes the helper exactly once.

        Sanity-checks the wiring point added in ``saturation_aware.py``:
        the per-cycle bottleneck-score emission must fire exactly
        once per ``autoscale()`` call. The helper itself is
        unit-tested elsewhere in this class; here we only verify
        the call-count contract so a future refactor that moves
        the call out of the cycle-end region (or accidentally
        duplicates it) breaks the test.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(setup_phase_quiescence_enabled=False),
        )
        scheduler = SaturationAwareScheduler(cfg)
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0"),
            },
        )
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        scheduler.setup(
            data_structures.Problem(
                cluster,
                [
                    data_structures.ProblemStage(
                        name="hot",
                        stage_batch_size=1,
                        worker_shape=cpu_shape,
                        requested_num_workers=None,
                        over_provision_factor=None,
                    ),
                ],
            )
        )
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            "hot-w0",
                            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                            num_used_slots=0,
                        )
                    ],
                    slots_per_worker=1,
                    is_finished=False,
                    num_used_slots=0,
                    num_empty_slots=1,
                    input_queue_depth=0,
                ),
            ]
        )

        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.emit_bottleneck_score") as mock_emit:
            scheduler.autoscale(time=0.0, problem_state=ps)
            assert mock_emit.call_count == 1, (
                f"expected exactly one helper invocation per cycle, got {mock_emit.call_count}"
            )
