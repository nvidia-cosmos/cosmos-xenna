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


"""Tests for the Forced-Flow-Law bottleneck-score emitter.

Pin the contract of
:func:`cosmos_xenna.pipelines.private.scheduling_py.bottleneck.emit_bottleneck_score`
plus the one end-to-end wiring point in
:class:`SaturationAwareScheduler`:

  * Per-stage gauge values match the actor-normalized Forced Flow
    Law ``D_k = V_k * S_k / c_k`` (with ``V_k = 1`` for Xenna's
    linear DAG, so ``D_k = S_k / c_k``).
  * The INFO log names the bottleneck stage selected by
    :func:`identify_bottleneck` - the operator-facing log cannot
    disagree with the planner's near-tie verdict.
  * Cold-start stages (NaN or non-positive ``D_k``) observe
    ``math.nan`` on the gauge and are excluded from the argmax.
  * Exactly one INFO log line fires per call when the engagement
    identity has a stage_name; no INFO log fires when the gate is
    disengaged (cold-start or homogeneous cluster).
  * The INFO log format is regex-stable so operators can
    ``grep`` the line without parsing scientific notation.
  * :meth:`SaturationAwareScheduler.autoscale` invokes the helper
    exactly once per cycle.

The helper is pure observability - these tests do not exercise
any autoscaler behaviour beyond confirming the wire-up point in
``autoscale()``.
"""

import logging
import math
import re
from collections.abc import Iterator, Mapping
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py import bottleneck
from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import (
    BottleneckIdentity,
    compute_d_k,
    emit_bottleneck_score,
    identify_bottleneck,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _identity_for(d_k_by_stage: Mapping[str, float]) -> BottleneckIdentity:
    """Build a BottleneckIdentity from a D_k mapping for the gauge tests.

    Uses a near-zero heterogeneity threshold (1.000001) so any finite
    spread engages the gate; tests that need a disengaged identity
    (e.g. cold-start, single-finite-stage) construct one directly.
    """
    return identify_bottleneck(d_k_by_stage, heterogeneity_threshold=1.000001)


def _disengaged_identity() -> BottleneckIdentity:
    """Return a no-bottleneck identity for tests where the gate must not engage."""
    return BottleneckIdentity(
        engaged=False,
        stage_name=None,
        max_d_k=math.nan,
        median_d_k=math.nan,
        heterogeneity_ratio=math.nan,
    )


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


def _build_one_stage_scheduler(
    stage_name: str = "hot",
) -> tuple[SaturationAwareScheduler, data_structures.ProblemState]:
    """Construct a single-stage scheduler + matching ProblemState for wiring tests."""
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
                    name=stage_name,
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
                stage_name=stage_name,
                workers=[
                    data_structures.ProblemWorkerGroupState.make(
                        f"{stage_name}-w0",
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
    return scheduler, ps


class TestEmitBottleneckScore:
    """Pins the actor-normalized bottleneck-score helper contract."""

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
        d_k_by_stage = {"a": 0.05, "b": 2.0, "c": 0.10}
        emit_bottleneck_score(
            d_k_by_stage=d_k_by_stage,
            bottleneck_identity=_identity_for(d_k_by_stage),
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
        assert "bottleneck stage: 'b'" in message
        assert "D = 2.00s" in message
        assert "throughput bound = 0.50 tasks/s" in message

    def test_capacity_field_appears_when_effective_capacities_supplied(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The INFO log includes ``capacity = N`` when capacities are supplied."""
        d_k_by_stage = {"a": 0.05, "b": 2.0, "c": 0.10}
        emit_bottleneck_score(
            d_k_by_stage=d_k_by_stage,
            bottleneck_identity=_identity_for(d_k_by_stage),
            pipeline_name="test_pipeline",
            effective_capacities={"a": 1, "b": 8, "c": 2},
        )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        assert "bottleneck stage: 'b'" in info_logs[0].message
        assert "capacity = 8" in info_logs[0].message

    def test_cold_start_stage_excluded_from_argmax(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A NaN D_k observes a NaN gauge but cannot win argmax.

        Setup: ``{"a": NaN, "b": 1.0, "c": 0.5}`` -> b is the
        bottleneck (a is excluded). Pins (i) the cold-start gauge
        is NaN, (ii) the argmax skips it, (iii) the bottleneck
        stage in the INFO log is b, not a.
        """
        d_k_by_stage = {"a": math.nan, "b": 1.0, "c": 0.5}
        emit_bottleneck_score(
            d_k_by_stage=d_k_by_stage,
            bottleneck_identity=_identity_for(d_k_by_stage),
            pipeline_name="test_pipeline",
        )

        assert len(gauge_observations) == 3
        observations_by_stage = {obs[1]["stage"]: obs[0] for obs in gauge_observations}
        assert math.isnan(observations_by_stage["a"])
        assert observations_by_stage["b"] == pytest.approx(1.0)
        assert observations_by_stage["c"] == pytest.approx(0.5)

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        assert "bottleneck stage: 'b'" in info_logs[0].message
        assert "bottleneck stage: 'a'" not in info_logs[0].message

    def test_all_cold_no_log(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every stage in cold-start: NaN gauges, no INFO log fires."""
        emit_bottleneck_score(
            d_k_by_stage={"a": math.nan, "b": math.nan},
            bottleneck_identity=_disengaged_identity(),
            pipeline_name="test_pipeline",
        )

        assert len(gauge_observations) == 2
        for value, _tags in gauge_observations:
            assert math.isnan(value)

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == [], (
            f"no INFO log must fire when every stage is in cold-start; got {[r.message for r in info_logs]}"
        )

    def test_zero_d_k_treated_as_cold(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-positive D_k observes NaN on the gauge.

        Setup: ``{"a": 0.0, "b": 1.0}`` -> stage "a" is treated as
        cold-start (gauge observes NaN, excluded from finite scores).
        With only one finite stage, the engagement gate stays
        disengaged and no INFO log fires.
        """
        emit_bottleneck_score(
            d_k_by_stage={"a": 0.0, "b": 1.0},
            bottleneck_identity=_disengaged_identity(),
            pipeline_name="test_pipeline",
        )

        observations_by_stage = {obs[1]["stage"]: obs[0] for obs in gauge_observations}
        assert math.isnan(observations_by_stage["a"])
        assert observations_by_stage["b"] == pytest.approx(1.0)

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == [], (
            f"single-finite-stage cycles must not fire an INFO log; got {[r.message for r in info_logs]}"
        )

    def test_disengaged_identity_suppresses_log_even_with_finite_d_k(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Operator log is silenced whenever ``identify_bottleneck`` did not engage.

        Pins the contract that the INFO log uses the planner's
        engagement verdict, not a strict argmax over finite D_k. A
        homogeneous pipeline (ratio close to 1.0) sets
        ``engaged=False`` and the log must stay silent so the
        operator never sees a "bottleneck" claim contradicted by
        the planner decision.
        """
        emit_bottleneck_score(
            d_k_by_stage={"a": 1.0, "b": 1.01, "c": 0.99},
            bottleneck_identity=_disengaged_identity(),
            pipeline_name="test_pipeline",
        )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []

    def test_log_format_pins_throughput_bound(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Pin the INFO log regex so operators can grep without ambiguity.

        The format is critical for log triage:
        ``bottleneck stage: '<name>' (D = N.NNs, throughput bound =
        N.NN tasks/s)``. The two-decimal precision avoids
        scientific notation for typical service times (10 ms to
        10 s) and keeps the line single-pass grep-friendly.
        Stage name is rendered via ``repr()`` so any embedded
        control characters surface escaped rather than corrupting
        the log line.
        """
        d_k_by_stage = {"alpha": 0.04, "beta": 2.5}
        emit_bottleneck_score(
            d_k_by_stage=d_k_by_stage,
            bottleneck_identity=_identity_for(d_k_by_stage),
            pipeline_name="test_pipeline",
        )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        # Stage names produced by ``StageSpec.name(index)`` contain spaces and
        # punctuation (e.g. ``"Stage 00 - ExampleStage"``), so the name token
        # is matched non-greedily up to the first " (" rather than requiring a
        # ``\w+`` identifier. The numeric format ``D = <float>s, throughput
        # bound = <float> tasks/s`` is pinned exactly.
        pattern = re.compile(r"bottleneck stage: .+? \(D = \d+\.\d{2}s, throughput bound = \d+\.\d{2} tasks/s\)")
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
        scheduler, ps = _build_one_stage_scheduler()
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.emit_bottleneck_score") as mock_emit:
            scheduler.autoscale(time=0.0, problem_state=ps)
            assert mock_emit.call_count == 1, (
                f"expected exactly one helper invocation per cycle, got {mock_emit.call_count}"
            )


class TestAutoscaleServiceTimeWiring:
    """``autoscale()`` threads actor-normalized D_k into ``emit_bottleneck_score``."""

    def test_cold_start_passes_nan_per_stage(self) -> None:
        """No measurements observed yet -> helper sees ``{stage: NaN}`` (NOT empty).

        The cold-start contract requires the gauge cardinality stay
        stable across cycles, so the wiring must pass an entry per
        stage even when no completed task has produced a measurement
        yet. ``math.nan`` is the cold-start sentinel folded into the
        cold-start branch downstream.
        """
        scheduler, ps = _build_one_stage_scheduler(stage_name="hot")
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.emit_bottleneck_score") as mock_emit:
            scheduler.autoscale(time=0.0, problem_state=ps)
            assert mock_emit.call_count == 1
            kwargs = mock_emit.call_args.kwargs
            d_k_by_stage = kwargs["d_k_by_stage"]
            assert set(d_k_by_stage.keys()) == {"hot"}
            assert math.isnan(d_k_by_stage["hot"])

    def test_measurements_thread_actor_normalized_d_k_to_helper(self) -> None:
        """Measurement-driven D_k = S_k / c_k reaches the helper after a sample.

        Two ``update_with_measurements`` batches frame a per-cycle
        delta with known total duration / count. The helper must
        observe ``D_k = mean / effective_capacity``: with one CPU
        worker and ``slots_per_worker = 1``, ``c_k = 1`` and
        ``D_k = mean = 0.25``.
        """
        scheduler, ps = _build_one_stage_scheduler(stage_name="hot")

        # Cycle 1: seed the snapshot. 2 tasks at 1.5s each.
        ms1 = data_structures.Measurements(
            time=0.5,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.5, 1) for _ in range(2)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.5, measurements=ms1)
        scheduler.autoscale(time=1.0, problem_state=ps)

        # Cycle 2: 4 more tasks at 0.25s each. Delta: dcount=4, dsum=1.0 -> mean=0.25s.
        ms2 = data_structures.Measurements(
            time=1.5,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.25, 1) for _ in range(4)]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.5, measurements=ms2)

        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.emit_bottleneck_score") as mock_emit:
            scheduler.autoscale(time=2.0, problem_state=ps)
            kwargs = mock_emit.call_args.kwargs
            # Single stage with c_k=1; first finite sample replaces the
            # NaN seed without blending so D_k = 0.25.
            assert kwargs["d_k_by_stage"]["hot"] == pytest.approx(0.25)

    def test_helper_and_heterogeneity_share_d_k_dict(self) -> None:
        """Both observability helpers consume the SAME actor-normalized ``d_k_by_stage``.

        Pins the wiring contract that ``emit_bottleneck_score`` and
        ``compute_heterogeneity_ratio`` see identical per-stage D_k
        in a given cycle, so a regression that threads only one of
        them stays caught.
        """
        scheduler, ps = _build_one_stage_scheduler(stage_name="hot")
        with (
            patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.emit_bottleneck_score") as mock_emit,
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.compute_heterogeneity_ratio"
            ) as mock_hetero,
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)
            emit_arg = mock_emit.call_args.kwargs["d_k_by_stage"]
            hetero_arg = mock_hetero.call_args.kwargs["d_k_by_stage"]
            assert set(emit_arg.keys()) == set(hetero_arg.keys())
            for stage_name, emit_value in emit_arg.items():
                hetero_value = hetero_arg[stage_name]
                if math.isnan(emit_value):
                    assert math.isnan(hetero_value)
                else:
                    assert emit_value == pytest.approx(hetero_value)

    def test_emit_receives_engagement_identity_from_decision(self) -> None:
        """The helper receives the same ``BottleneckIdentity`` the planner uses.

        Pins that the operator-facing log cannot disagree with
        ``identify_bottleneck`` because both consume one identity
        produced once per cycle.
        """
        scheduler, ps = _build_one_stage_scheduler(stage_name="hot")
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.emit_bottleneck_score") as mock_emit:
            scheduler.autoscale(time=0.0, problem_state=ps)
            kwargs = mock_emit.call_args.kwargs
            assert "bottleneck_identity" in kwargs
            assert isinstance(kwargs["bottleneck_identity"], BottleneckIdentity)

    def test_gauge_observes_finite_value_after_measurements(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full path: measurements -> autoscale -> gauge observes finite ``D_k``.

        End-to-end check that exercises every layer (accumulator,
        consumer, EWMA over S_k, ``compute_d_k`` per cycle, gauge)
        without patching the helper itself, so a regression anywhere
        in the chain breaks the test. With one CPU worker, c_k = 1
        and D_k = S_k = 0.4. The single-stage cycle does not engage
        the bottleneck gate, so no INFO log fires; the gauge value
        is the contract under test.
        """
        scheduler, ps = _build_one_stage_scheduler(stage_name="hot")

        ms1 = data_structures.Measurements(
            time=0.5,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.4, 1) for _ in range(3)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.5, measurements=ms1)
        scheduler.autoscale(time=1.0, problem_state=ps)
        loguru_caplog.clear()
        gauge_observations.clear()

        ms2 = data_structures.Measurements(
            time=1.5,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.4, 1) for _ in range(3)]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.5, measurements=ms2)
        scheduler.autoscale(time=2.0, problem_state=ps)

        # Gauge fired with finite D_k = S_k / c_k = 0.4 / 1.
        assert len(gauge_observations) == 1
        value, tags = gauge_observations[0]
        assert value == pytest.approx(0.4)
        assert tags == {"stage": "hot", "pipeline": ""}


class TestIdentifyBottleneck:
    """Pin the engagement gate, argmax selection, and near-tie tolerance."""

    def test_balanced_three_stages_returns_disengaged(self) -> None:
        """A homogeneous pipeline must NOT engage the bottleneck gate."""
        identity = bottleneck.identify_bottleneck(
            {"a": 1.0, "b": 1.0, "c": 1.0},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is False
        assert identity.stage_name is None
        assert identity.heterogeneity_ratio == pytest.approx(1.0)

    def test_heterogeneous_three_stages_engages_with_argmax(self) -> None:
        """A clearly heterogeneous pipeline engages and names argmax_k D_k."""
        identity = bottleneck.identify_bottleneck(
            {"download": 0.05, "caption": 2.0, "embed": 0.10},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is True
        assert identity.stage_name == "caption"
        assert identity.max_d_k == pytest.approx(2.0)

    def test_all_nan_returns_disengaged_with_nan_fields(self) -> None:
        """Cold-start cluster (all NaN) keeps the gate disengaged."""
        identity = bottleneck.identify_bottleneck(
            {"a": math.nan, "b": math.nan, "c": math.nan},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is False
        assert identity.stage_name is None
        assert math.isnan(identity.max_d_k)
        assert math.isnan(identity.heterogeneity_ratio)

    def test_single_finite_stage_returns_disengaged(self) -> None:
        """A pipeline with only one finite D_k cannot engage the gate."""
        identity = bottleneck.identify_bottleneck(
            {"hot": 1.5, "cold1": math.nan, "cold2": math.nan},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is False
        assert identity.max_d_k == pytest.approx(1.5)
        assert math.isnan(identity.median_d_k)

    def test_two_stage_uses_max_over_min_for_ratio(self) -> None:
        """For n=2 the ratio is max/min, not max/median (capped at 2.0)."""
        identity = bottleneck.identify_bottleneck(
            {"fast": 0.5, "slow": 4.0},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is True
        assert identity.stage_name == "slow"
        assert identity.heterogeneity_ratio == pytest.approx(8.0)

    def test_near_tie_uses_lex_smallest_for_stability(self) -> None:
        """Within near_tie_tolerance of max, the lex-smallest name wins."""
        identity = bottleneck.identify_bottleneck(
            {"alpha": 3.95, "beta": 4.0, "gamma": 0.5, "delta": 0.5, "epsilon": 0.5},
            heterogeneity_threshold=2.0,
            near_tie_tolerance=0.05,
        )

        assert identity.engaged is True
        assert identity.stage_name == "alpha"

    def test_strict_argmax_when_tolerance_zero(self) -> None:
        """Tolerance 0.0 means strict max; ties only count when D_k is exactly equal."""
        identity = bottleneck.identify_bottleneck(
            {"alpha": 3.95, "beta": 4.0, "gamma": 0.5, "delta": 0.5, "epsilon": 0.5},
            heterogeneity_threshold=2.0,
            near_tie_tolerance=0.0,
        )

        assert identity.engaged is True
        assert identity.stage_name == "beta"

    def test_zero_or_negative_d_k_treated_as_cold_start(self) -> None:
        """Non-positive D_k inputs are excluded from finite_scores."""
        identity = bottleneck.identify_bottleneck(
            {"hot": 2.0, "broken": 0.0, "negative": -0.5},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is False
        assert identity.max_d_k == pytest.approx(2.0)
        assert math.isnan(identity.median_d_k)

    def test_ratio_just_below_threshold_is_disengaged(self) -> None:
        """``ratio < threshold`` keeps the gate off."""
        identity = bottleneck.identify_bottleneck(
            {"a": 1.0, "b": 1.0, "c": 1.99},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is False

    def test_ratio_at_threshold_is_engaged(self) -> None:
        """``ratio >= threshold`` engages the gate at the boundary."""
        identity = bottleneck.identify_bottleneck(
            {"a": 1.0, "b": 1.0, "c": 2.0},
            heterogeneity_threshold=2.0,
        )

        assert identity.engaged is True
        assert identity.stage_name == "c"

    def test_near_tie_tolerance_negative_is_rejected(self) -> None:
        """A negative tolerance would raise the tie floor above ``max_d`` and empty the tied list."""
        with pytest.raises(ValueError, match=r"near_tie_tolerance must be in \[0.0, 1.0\)"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=1.5,
                near_tie_tolerance=-0.01,
            )

    def test_near_tie_tolerance_one_is_rejected(self) -> None:
        """A tolerance of 1.0 collapses the band so every positive score is tied."""
        with pytest.raises(ValueError, match=r"near_tie_tolerance must be in \[0.0, 1.0\)"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=1.5,
                near_tie_tolerance=1.0,
            )

    def test_near_tie_tolerance_above_one_is_rejected(self) -> None:
        """A tolerance above 1.0 produces a sub-zero floor and the same degenerate verdict."""
        with pytest.raises(ValueError, match=r"near_tie_tolerance must be in \[0.0, 1.0\)"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=1.5,
                near_tie_tolerance=1.01,
            )

    def test_heterogeneity_threshold_at_one_is_rejected(self) -> None:
        """A homogeneous cluster has ratio ``1.0``; the floor must be strictly greater."""
        with pytest.raises(ValueError, match=r"heterogeneity_threshold must be finite and > 1.0"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=1.0,
            )

    def test_heterogeneity_threshold_below_one_is_rejected(self) -> None:
        """A floor below ``1.0`` would engage on near-uniform pipelines."""
        with pytest.raises(ValueError, match=r"heterogeneity_threshold must be finite and > 1.0"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=0.5,
            )

    def test_heterogeneity_threshold_nan_is_rejected(self) -> None:
        """``NaN`` would silently engage every cycle (``ratio < NaN`` is always ``False``)."""
        with pytest.raises(ValueError, match=r"heterogeneity_threshold must be finite and > 1.0"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=math.nan,
            )

    def test_heterogeneity_threshold_inf_is_rejected(self) -> None:
        """``+inf`` passes ``attrs.validators.gt(1.0)`` upstream but silently disables engagement.

        The function-boundary check closes the gap the spec validator
        leaves open: no finite cluster ratio can ever exceed ``+inf``,
        so the gate would be permanently disengaged without a clear
        error.
        """
        with pytest.raises(ValueError, match=r"heterogeneity_threshold must be finite and > 1.0"):
            bottleneck.identify_bottleneck(
                {"a": 1.0, "b": 2.0},
                heterogeneity_threshold=math.inf,
            )


class TestMaybeLogBottleneckEngagement:
    """Pin the persistence-gated engagement INFO log."""

    def test_initial_engaged_state_silenced_after_persistence(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The very first persistent state seeds last_announced without logging."""
        state = bottleneck.BottleneckEngagementState()
        identity = bottleneck.BottleneckIdentity(
            engaged=True,
            stage_name="caption",
            max_d_k=2.0,
            median_d_k=0.5,
            heterogeneity_ratio=4.0,
        )

        for _ in range(2):
            bottleneck.maybe_log_bottleneck_engagement(
                identity=identity,
                state=state,
                persistence_cycles=2,
                pipeline_name="p1",
            )

        assert state.last_announced is True
        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []

    def test_persistent_flip_to_disengaged_logs_once(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After seeding engaged, two consecutive disengaged cycles fire one INFO log."""
        state = bottleneck.BottleneckEngagementState(last_announced=True, candidate_streak=0)
        disengaged = bottleneck.BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=math.nan,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

        for _ in range(2):
            bottleneck.maybe_log_bottleneck_engagement(
                identity=disengaged,
                state=state,
                persistence_cycles=2,
                pipeline_name="p1",
            )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        assert "disengaged" in info_logs[0].message

    def test_short_flip_does_not_log(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A single off-state cycle inside an engaged streak does not fire the log."""
        state = bottleneck.BottleneckEngagementState(last_announced=True, candidate_streak=0)
        engaged = bottleneck.BottleneckIdentity(
            engaged=True,
            stage_name="caption",
            max_d_k=2.0,
            median_d_k=0.5,
            heterogeneity_ratio=4.0,
        )
        disengaged = bottleneck.BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=math.nan,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

        bottleneck.maybe_log_bottleneck_engagement(
            identity=disengaged, state=state, persistence_cycles=2, pipeline_name="p1"
        )
        bottleneck.maybe_log_bottleneck_engagement(
            identity=engaged, state=state, persistence_cycles=2, pipeline_name="p1"
        )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []
        assert state.last_announced is True
        assert state.candidate_streak == 0

    def test_cold_start_mixed_sequence_does_not_seed_last_announced(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Noisy cold-start (True, False, True) must not seed ``last_announced``.

        The persistence gate's contract is "wait until a candidate
        value holds for ``persistence_cycles`` consecutive cycles",
        not "wait ``persistence_cycles`` cycles total". A naive
        implementation that increments the streak on every cycle
        (regardless of candidate value) would seed
        ``last_announced = True`` on the third cycle of the
        ``True, False, True`` sequence because the streak happens
        to reach ``persistence_cycles=3``. The fix tracks the most
        recently observed candidate and resets the streak on a
        value flip so a noisy mixed sequence is correctly debounced
        during the cold-start seeding phase.
        """
        state = bottleneck.BottleneckEngagementState()
        engaged = bottleneck.BottleneckIdentity(
            engaged=True,
            stage_name="caption",
            max_d_k=2.0,
            median_d_k=0.5,
            heterogeneity_ratio=4.0,
        )
        disengaged = bottleneck.BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=math.nan,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

        for identity in (engaged, disengaged, engaged):
            bottleneck.maybe_log_bottleneck_engagement(
                identity=identity,
                state=state,
                persistence_cycles=3,
                pipeline_name="p1",
            )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []
        assert state.last_announced is None, (
            f"mixed cold-start sequence must not seed last_announced; got {state.last_announced}"
        )
        assert state.candidate_streak == 1, (
            f"flip on the final cycle must reset the streak to 1; got {state.candidate_streak}"
        )
        assert state.last_candidate is True, (
            f"last_candidate must track the most recently observed value; got {state.last_candidate}"
        )

    def test_cold_start_consecutive_identical_candidates_seeds_last_announced(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After a flip in cold-start, the streak rebuilds and eventually seeds.

        Demonstrates that the consecutive-identical-candidate rule
        does not block legitimate seeding: once the candidate value
        stabilises for ``persistence_cycles`` cycles in a row, the
        cold-start phase concludes and ``last_announced`` is set.
        Pairs with ``test_cold_start_mixed_sequence_does_not_seed_last_announced``
        to bracket the gate's behaviour at both extremes.
        """
        state = bottleneck.BottleneckEngagementState()
        engaged = bottleneck.BottleneckIdentity(
            engaged=True,
            stage_name="caption",
            max_d_k=2.0,
            median_d_k=0.5,
            heterogeneity_ratio=4.0,
        )
        disengaged = bottleneck.BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=math.nan,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

        # Noisy preamble: streak rebuilds on the False that follows
        # the True, then again on the True that follows the False.
        bottleneck.maybe_log_bottleneck_engagement(
            identity=engaged, state=state, persistence_cycles=3, pipeline_name="p1"
        )
        bottleneck.maybe_log_bottleneck_engagement(
            identity=disengaged, state=state, persistence_cycles=3, pipeline_name="p1"
        )
        # Three consecutive engaged cycles must seed last_announced.
        for _ in range(3):
            bottleneck.maybe_log_bottleneck_engagement(
                identity=engaged, state=state, persistence_cycles=3, pipeline_name="p1"
            )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert info_logs == []
        assert state.last_announced is True
        assert state.candidate_streak == 0


class TestComputeDk:
    """Pin the actor-normalized ``D_k = S_k / c_k`` helper contract.

    Pure helper with no I/O or concurrency, so only adversarial
    axes 1-3 (edge cases, failure handling, boundary conditions)
    apply per ``.cursor/rules/test-creation.mdc``.
    """

    def test_d_k_divides_by_actor_count(self) -> None:
        """``S_k = 10s, c_k = 5`` -> ``D_k = 2s`` - the central contract."""
        assert compute_d_k(10.0, 5) == pytest.approx(2.0)

    def test_zero_actor_count_folds_to_nan(self) -> None:
        """``c_k = 0`` is an undefined capacity signal; result is NaN."""
        assert math.isnan(compute_d_k(2.0, 0))

    def test_negative_actor_count_folds_to_nan(self) -> None:
        """``c_k = -1`` is rejected at the boundary, not silently folded to ``c_k = 1``."""
        assert math.isnan(compute_d_k(2.0, -1))

    def test_negative_service_time_returns_nan(self) -> None:
        """Negative service time cannot represent a physical sample; NaN."""
        assert math.isnan(compute_d_k(-0.1, 4))

    def test_zero_service_time_returns_nan(self) -> None:
        """``S_k = 0`` from ``processing_speed_tasks_per_second == 0`` is undefined."""
        assert math.isnan(compute_d_k(0.0, 4))

    def test_inf_service_time_folds_to_nan(self) -> None:
        """``+inf`` is non-finite and folds to the cold-start sentinel.

        Pins the contract that the helper rejects every non-finite
        sample at the boundary so a pathological measurement cannot
        poison ``identify_bottleneck``'s argmax: a stage whose
        service time was reported as ``+inf`` simply contributes
        NaN to the gauge and is excluded from the bottleneck verdict.
        """
        assert math.isnan(compute_d_k(math.inf, 4))

    def test_nan_service_time_returns_nan(self) -> None:
        """The cold-start sentinel passes through unchanged."""
        assert math.isnan(compute_d_k(math.nan, 4))

    def test_spmd_group_capacity_drives_division(self) -> None:
        """SPMD-style capacity: ``S_k = 16s, c_k = 8`` -> ``D_k = 2s``.

        Pins the production-shape regression for SPMD stages: a
        single worker_group with 8 allocations contributes ``c_k = 8``,
        not ``c_k = 1``. With ``c_k = 1`` the stage would falsely
        win the bottleneck argmax against a stage with ``S_k = 3, c_k = 1``.
        """
        assert compute_d_k(16.0, 8) == pytest.approx(2.0)


class TestBottleneckArgmaxRegression:
    """Pin the production-incident regression: multi-actor stages must not falsely win.

    Reproduces the conditions where the bottleneck identification
    incorrectly named an over-provisioned stage with many idle
    actors: large raw ``S_k`` but high ``c_k`` should not win
    against a smaller stage with low ``c_k``.
    """

    def test_multi_actor_stage_with_idle_capacity_is_not_bottleneck(self) -> None:
        """Production-shape regression: high-S stage with many actors is not the bottleneck.

        Setup: stage A has ``S=34s, c=13`` (D=2.64); stage B has
        ``S=30s, c=2`` (D=15.0). Stage B must win the argmax even
        though A has the larger raw ``S_k``.
        """
        d_a = compute_d_k(34.0, 13)
        d_b = compute_d_k(30.0, 2)
        d_k_by_stage = {"A": d_a, "B": d_b}
        identity = identify_bottleneck(d_k_by_stage, heterogeneity_threshold=2.0)
        assert identity.engaged is True
        assert identity.stage_name == "B"
        assert identity.max_d_k == pytest.approx(d_b)


class TestEmitBottleneckScoreAdversarial:
    """Adversarial axes for the operator-facing log emitter."""

    def test_log_handles_special_characters_in_stage_names(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Stage names with quotes and newlines must not corrupt the log line.

        The emitter renders the name via ``repr()`` so embedded
        control characters surface escaped rather than breaking
        the log-aggregation parser.
        """
        bad_name = 'Stage 09 "Caption"\nINJECTED'
        d_k_by_stage = {bad_name: 2.0, "other": 0.1}
        emit_bottleneck_score(
            d_k_by_stage=d_k_by_stage,
            bottleneck_identity=_identity_for(d_k_by_stage),
            pipeline_name="test_pipeline",
        )

        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        message = info_logs[0].message
        # repr() escapes the embedded newline so the log line stays
        # single-line and the embedded "INJECTED" is visible only as
        # part of the escaped string, not on a separate physical line.
        assert "\n" not in message
        assert "\\n" in message
        assert "INJECTED" in message

    def test_argmax_handles_large_stage_count_without_error(
        self,
        gauge_observations: list[tuple[float, dict[str, str]]],
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """100 stages with a clear bottleneck finish without raising and name the right stage.

        Smoke-tests the linear-in-stages cost. Uses a single
        clearly-dominant stage so the near-tie tolerance does not
        broaden the engaged set: stage-099 has ``D_k = 1000.0``
        while every other stage sits at ``D_k`` < 100, leaving
        stage-099 alone in the near-tie band.
        """
        d_k_by_stage: dict[str, float] = {f"stage-{idx:03d}": float(idx) + 1.0 for idx in range(99)}
        d_k_by_stage["stage-099"] = 1000.0
        emit_bottleneck_score(
            d_k_by_stage=d_k_by_stage,
            bottleneck_identity=_identity_for(d_k_by_stage),
            pipeline_name="test_pipeline",
        )
        assert len(gauge_observations) == 100
        info_logs = [r for r in loguru_caplog.records if r.levelno == logging.INFO]
        assert len(info_logs) == 1
        assert "bottleneck stage: 'stage-099'" in info_logs[0].message


class TestHeterogeneityRatioActorNormalized:
    """The heterogeneity ratio gauge consumes the same actor-normalized D_k as the decision."""

    def test_ratio_uses_actor_normalized_d_k(self) -> None:
        """Two stages with identical ``S_k`` but different ``c_k`` produce a finite ratio.

        Setup: ``A: S=10, c=2 (D=5)``; ``B: S=10, c=10 (D=1)`` ->
        ratio = max/min = 5.0. With raw ``S_k`` the ratio would be
        1.0 (false homogeneity) and the gauge would mislead operators.
        """
        d_k_by_stage = {"A": compute_d_k(10.0, 2), "B": compute_d_k(10.0, 10)}
        state = bottleneck.HeterogeneityWarnState()
        bottleneck.compute_heterogeneity_ratio(
            d_k_by_stage=d_k_by_stage,
            pipeline_name="test_pipeline",
            state=state,
            warn_threshold=2.0,
            warn_streak_cycles=3,
        )
        # Streak counter increments because ratio (5.0) > threshold (2.0).
        assert state.streak_cycles == 1

    def test_ratio_consistent_with_bottleneck_label(self) -> None:
        """The ratio's argmax stage matches identify_bottleneck's stage_name."""
        d_k_by_stage = {
            "fast_with_many_actors": compute_d_k(10.0, 10),
            "slow_with_few_actors": compute_d_k(10.0, 2),
            "average": compute_d_k(2.0, 1),
        }
        identity = identify_bottleneck(d_k_by_stage, heterogeneity_threshold=2.0)
        # max(D_k) is on slow_with_few_actors (D=5.0); the heterogeneity
        # gauge sees the same dict and would name the same stage in its
        # streak-warning log line.
        assert identity.engaged is True
        assert identity.stage_name == "slow_with_few_actors"


class TestComputeBalanceScore:
    """``compute_balance_score`` returns ``1 / max(1, max/min)`` over finite-positive ``D_k``."""

    def test_perfectly_balanced_pipeline_scores_one(self) -> None:
        """Identical ``D_k`` across stages -> ratio=1 -> balance=1.0."""
        d_k_by_stage = {"A": 1.0, "B": 1.0}

        assert bottleneck.compute_balance_score(d_k_by_stage) == 1.0

    def test_severe_bottleneck_drops_score_well_below_one(self) -> None:
        """``max/min = 10`` -> balance = 0.1."""
        d_k_by_stage = {"A": 1.0, "B": 10.0}

        assert math.isclose(bottleneck.compute_balance_score(d_k_by_stage), 0.1, rel_tol=1e-9)

    def test_cold_start_returns_nan(self) -> None:
        """Fewer than two finite-positive stages -> NaN, not 0 / 1."""
        d_k_by_stage = {"A": math.nan, "B": 0.0}

        assert math.isnan(bottleneck.compute_balance_score(d_k_by_stage))


class TestEmitBalanceScore:
    """``emit_balance_score`` updates the gauge AND returns the scalar for the caller."""

    def test_returns_scalar_and_observes_gauge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The helper hands the scalar back AND observes the gauge with the same value."""
        # Monkeypatch ``set`` to capture (value, tags) so the wire format
        # is asserted explicitly, not via reading the gauge's private
        # name / tag-keys attrs.
        captured: list[tuple[float, dict[str, str]]] = []

        def fake_set(value: float, *, tags: dict[str, str] | None = None) -> None:
            captured.append((value, tags or {}))

        monkeypatch.setattr(bottleneck._BALANCE_SCORE_GAUGE, "set", fake_set)

        observed = bottleneck.emit_balance_score({"A": 1.0, "B": 4.0}, pipeline_name="test_pipeline")

        # 1 / max(1, 4/1) = 0.25.
        assert math.isclose(observed, 0.25, rel_tol=1e-9)
        assert len(captured) == 1
        gauge_value, gauge_tags = captured[0]
        assert math.isclose(gauge_value, 0.25, rel_tol=1e-9)
        assert gauge_tags == {"pipeline": "test_pipeline"}

    def test_cold_start_observes_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cold-start cluster observes NaN so the gauge cardinality stays stable."""
        captured: list[tuple[float, dict[str, str]]] = []

        def fake_set(value: float, *, tags: dict[str, str] | None = None) -> None:
            captured.append((value, tags or {}))

        monkeypatch.setattr(bottleneck._BALANCE_SCORE_GAUGE, "set", fake_set)

        observed = bottleneck.emit_balance_score({"A": math.nan, "B": math.nan}, pipeline_name="test_pipeline")

        assert math.isnan(observed)
        assert len(captured) == 1
        gauge_value, _ = captured[0]
        assert math.isnan(gauge_value)
