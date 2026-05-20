# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-phase timing histogram in ``SaturationAwareScheduler.autoscale()``.

The ``_phase_timer`` context manager (see
docs/scheduler/saturation-aware/22-prometheus-metrics.md row
``xenna_scheduler_cycle_phase_duration_seconds``) brackets every phase
block in ``autoscale()`` so each cycle observes exactly one duration
per phase on the ``_PHASE_DURATION_HISTOGRAM`` Histogram, tagged
``{"phase": <label>, "pipeline": <pipeline_name>}``.

These tests pin five behaviours of the wrapper contract:

  * Exactly one observation per cycle for each of the 8 phase labels.
  * Sum of per-phase durations equals the cycle wall-clock duration
    (within histogram-observe overhead tolerance).
  * Memory-pressure gate freezing Phase C still records a ``phase_c``
    observation (always-observe guarantee).
  * Every observation carries the ``pipeline`` tag value threaded into
    ``SaturationAwareScheduler.__init__`` so multi-pipeline clusters
    stay distinguishable in Prometheus.
  * A phase whose body raises still observes its duration via
    ``try/finally`` before the exception propagates.
"""

from typing import Any
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py import saturation_aware
from cosmos_xenna.pipelines.private.scheduling_py.invariants import PhaseBoundary
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig

MEMORY_PRESSURE_MODULE = "cosmos_xenna.pipelines.private.scheduling_py.memory_pressure"
SATURATION_AWARE_MODULE = "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware"

# Pinned set of phase labels emitted by ``autoscale()`` once per cycle.
# Changing this set requires updating producer wrappers AND consumer
# dashboards bound to ``xenna_scheduler_cycle_phase_duration_seconds``.
EXPECTED_PHASE_LABELS = frozenset(
    {
        "pre_phase_setup",
        "phase_a",
        "phase_b",
        "intent",
        "phase_c",
        "phase_d",
        "invariants",
        "into_solution",
    }
)


class _FakeHistogram:
    """Stand-in for ``ray.util.metrics.Histogram`` that records observations.

    Patched onto ``saturation_aware._PHASE_DURATION_HISTOGRAM`` via
    ``monkeypatch.setattr`` so tests can assert on the exact
    ``(duration, tags)`` pairs each wrapper passes to ``observe``.
    Using a structural fake avoids requiring a running Ray session
    and keeps the test contract independent of Ray's internal metric
    registry. Mirrors the pattern from ``test_loop_watchdog.py``.
    """

    def __init__(self) -> None:
        self.observations: list[tuple[float, dict[str, str]]] = []

    def observe(self, value: float, tags: dict[str, str] | None = None) -> None:
        self.observations.append((value, dict(tags or {})))


def _cluster() -> resources.ClusterResources:
    """Single-node CPU cluster sized for the phase-duration test fixtures."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(
                used_cpus=0,
                total_cpus=64,
                gpus=[],
                name="node-0",
            ),
        },
    )


def _scheduler(*, pipeline_name: str = "") -> SaturationAwareScheduler:
    """Build a setup-completed single-stage scheduler.

    Disables warmup graces and floor-stuck grace so a single cycle's
    signal is sufficient and all phases execute without short-circuit
    paths interfering with phase-observation accounting.

    Args:
        pipeline_name: Threaded into ``SaturationAwareScheduler.__init__``
            so tests can verify the Prometheus ``pipeline`` tag travels
            from constructor to observation site. Defaults to the empty
            string for tests that do not exercise the tag value.
    """
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            min_workers=1,
        ),
    )
    cluster = _cluster()
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    problem = data_structures.Problem(
        cluster,
        [
            data_structures.ProblemStage(
                name="stage",
                stage_batch_size=1,
                worker_shape=cpu_shape,
                requested_num_workers=None,
                over_provision_factor=None,
            ),
        ],
    )
    scheduler = SaturationAwareScheduler(cfg, pipeline_name=pipeline_name)
    scheduler.setup(problem)
    return scheduler


def _problem_state_with_workers(*, worker_ids: list[str]) -> data_structures.ProblemState:
    """Single-stage ``ProblemState`` whose workers all sit at 0 used slots."""
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            wid,
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            num_used_slots=0,
        )
        for wid in worker_ids
    ]
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(
                stage_name="stage",
                workers=worker_groups,
                slots_per_worker=1,
                is_finished=False,
                num_used_slots=0,
                num_empty_slots=len(worker_ids),
                input_queue_depth=0,
            ),
        ],
    )


class TestPhaseDurationHistogram:
    """Pin the five behaviours of the per-phase timing histogram."""

    def test_every_phase_label_observed_once_per_cycle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One ``autoscale()`` call observes exactly the 8 canonical phase labels, each once."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(saturation_aware, "_PHASE_DURATION_HISTOGRAM", fake_histogram)

        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}):
            scheduler.autoscale(time=0.0, problem_state=ps)

        observed_labels = [tags["phase"] for _value, tags in fake_histogram.observations]
        assert len(observed_labels) == 8, (
            f"Expected exactly 8 observations (one per phase label), got {len(observed_labels)}: {observed_labels}"
        )
        assert set(observed_labels) == EXPECTED_PHASE_LABELS, (
            f"Observed labels {set(observed_labels)} != expected {EXPECTED_PHASE_LABELS}"
        )
        # Pin the observation ORDER too -- dashboards bound to the
        # canonical phase sequence rely on consistent cycle timelines.
        assert observed_labels == [
            "pre_phase_setup",
            "phase_a",
            "phase_b",
            "intent",
            "phase_c",
            "phase_d",
            "invariants",
            "into_solution",
        ], f"Phase label emission order changed: {observed_labels}"

    def test_sum_of_phases_approximately_equals_cycle_duration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sum of all observed phase durations equals the wall-clock span across the cycle.

        Uses a deterministic fake clock with 16 ascending samples (one
        ``start`` + one ``finally`` per phase, 8 phases) where each
        phase's ``finally`` value equals the next phase's ``start``
        (no gap between phases, matching how the wrappers are wired
        contiguously). The sum then telescopes to ``last - first``
        exactly, pinning a zero-tolerance equality.
        """
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(saturation_aware, "_PHASE_DURATION_HISTOGRAM", fake_histogram)

        # 16 deterministic clock samples: pairs of (start_i, finally_i)
        # where finally_i == start_{i+1}. Each phase delta = 0.001 s.
        delta_per_phase_s = 0.001
        clock_samples: list[float] = []
        t = 100.0
        for _ in range(len(EXPECTED_PHASE_LABELS)):
            clock_samples.append(t)
            t += delta_per_phase_s
            clock_samples.append(t)

        clock_iter = iter(clock_samples)
        monkeypatch.setattr(saturation_aware, "_monotonic", lambda: next(clock_iter))

        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}):
            scheduler.autoscale(time=0.0, problem_state=ps)

        sum_durations = sum(value for value, _tags in fake_histogram.observations)
        total_cycle_s = clock_samples[-1] - clock_samples[0]

        assert sum_durations == pytest.approx(total_cycle_s, abs=1e-9), (
            f"Sum of phase durations {sum_durations:.6f}s must equal total cycle span "
            f"{total_cycle_s:.6f}s (deterministic clock telescopes start_i+1 == finally_i)"
        )

    def test_memory_pressure_gate_still_records_phase_c(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the memory-pressure gate freezes Phase C grow, the wrapper STILL observes ``phase_c``."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(saturation_aware, "_PHASE_DURATION_HISTOGRAM", fake_histogram)

        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        # Force the memory-pressure monitor to report pressure-active
        # so Phase C grow short-circuits to an early ``return``.
        with (
            patch(
                f"{MEMORY_PRESSURE_MODULE}.MemoryPressureMonitor.is_pressure_active",
                return_value=True,
            ),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 3}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        # Confirm the gate fired and froze grow.
        assert solution.stages[0].new_workers == [], "Phase C should be frozen when pressure is active"

        phase_c_observations = [
            (value, tags) for value, tags in fake_histogram.observations if tags["phase"] == "phase_c"
        ]
        assert len(phase_c_observations) == 1, (
            f"phase_c must be observed exactly once even when the gate freezes grow, got "
            f"{len(phase_c_observations)} observations"
        )
        observed_duration, _tags = phase_c_observations[0]
        # Near-zero, NOT NaN, NOT skipped -- the wrapper records a real
        # finite duration on every cycle by design.
        assert observed_duration >= 0.0
        assert observed_duration < 1.0, (
            "Phase C frozen by gate should complete near-instantly; large duration suggests "
            "the gate did not short-circuit"
        )

    def test_pipeline_tag_threaded_from_constructor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every observation carries the ``pipeline_name`` value passed into ``SaturationAwareScheduler.__init__``."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(saturation_aware, "_PHASE_DURATION_HISTOGRAM", fake_histogram)

        scheduler = _scheduler(pipeline_name="my-pipeline")
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}):
            scheduler.autoscale(time=0.0, problem_state=ps)

        for value, tags in fake_histogram.observations:
            assert tags["pipeline"] == "my-pipeline", (
                f"Observation (phase={tags.get('phase')!r}, value={value:.6f}) carried "
                f"pipeline={tags.get('pipeline')!r} but the constructor was called with "
                f"pipeline_name='my-pipeline'"
            )

    def test_phase_a_delete_failure_still_observes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raising ``check_invariants_after_phase(PHASE_A)`` still records ``phase_a`` via ``try/finally``."""
        fake_histogram = _FakeHistogram()
        monkeypatch.setattr(saturation_aware, "_PHASE_DURATION_HISTOGRAM", fake_histogram)

        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        def raising_invariant_check(*, phase_name: PhaseBoundary, problem: Any, ctx: Any) -> None:
            if phase_name is PhaseBoundary.PHASE_A:
                raise AssertionError("synthetic Phase A invariant failure")

        with (
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}),
            patch(
                f"{SATURATION_AWARE_MODULE}.check_invariants_after_phase",
                side_effect=raising_invariant_check,
            ),
            pytest.raises(AssertionError, match="synthetic Phase A invariant failure"),
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)

        observed_labels = [tags["phase"] for _value, tags in fake_histogram.observations]
        # ``pre_phase_setup`` (completed) + ``phase_a`` (raised in
        # ``try``, observed in ``finally``) -- subsequent phases never
        # execute because the exception propagates out of the cycle.
        assert observed_labels == ["pre_phase_setup", "phase_a"], (
            f"Phase A try/finally must record the duration before the exception propagates; "
            f"got observations {observed_labels}"
        )
