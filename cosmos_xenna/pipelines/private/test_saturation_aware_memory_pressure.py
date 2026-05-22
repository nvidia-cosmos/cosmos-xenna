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


"""Tests for the cluster object-store memory-pressure gate on Phase C.

The gate is a cluster-wide kill switch on the saturation-aware
scheduler's Phase C scale-up: when the polled Ray object-store
``used_fraction`` exceeds
``SaturationAwareConfig.memory_pressure_critical_threshold``
(default ``0.85``), every stage's positive intent is frozen for the
cycle. The pinned contract:

  * Phase C scale-up is frozen cluster-wide when pressure is
    active.
  * Phase B floor enforcement and Phase D shrink keep running so
    structural recovery (floor) and pressure relief (shrink) never
    deadlock.
  * Polls are cached for at least
    ``memory_pressure_polling_interval_s`` (default ``5.0`` s) so
    the per-cycle hot path makes at most one Ray API call per
    polling window.
  * A Ray API failure degrades gracefully -- the gate assumes
    pressure inactive, emits one WARN, and Phase C runs as if no
    gate were configured.
  * ``setup()`` clears the cached monitor state so a re-setup
    starts from a clean cache.
"""

import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig

MEMORY_PRESSURE_MODULE = "cosmos_xenna.pipelines.private.scheduling_py.memory_pressure"
SATURATION_AWARE_MODULE = "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware"


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in ``test_donor_warmup_grace.py``.
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


def _patch_ray_resources(
    *,
    total: float,
    available: float,
) -> Any:
    """Build a context manager that mocks the Ray resource APIs the gate consumes.

    Returns a single ``ExitStack``-like context that activates three
    ``unittest.mock.patch`` instances so call-sites can write::

        with _patch_ray_resources(total=100.0, available=15.0):
            scheduler.autoscale(...)

    ``ray.is_initialized`` is forced to ``True`` so the monitor's
    boot-up shortcut does not bypass the mocked resource calls.
    """

    class _StackedPatch:
        def __enter__(self) -> "_StackedPatch":
            self._initialized_patch = patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.is_initialized",
                return_value=True,
            )
            self._cluster_patch = patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.cluster_resources",
                return_value={"object_store_memory": total},
            )
            self._available_patch = patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.available_resources",
                return_value={"object_store_memory": available},
            )
            self.initialized_mock = self._initialized_patch.start()
            self.cluster_mock = self._cluster_patch.start()
            self.available_mock = self._available_patch.start()
            return self

        def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
            self._available_patch.stop()
            self._cluster_patch.stop()
            self._initialized_patch.stop()

    return _StackedPatch()


def _cluster() -> resources.ClusterResources:
    """Single-node CPU cluster sized for the memory-pressure fixtures."""
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


def _scheduler(
    *,
    min_workers: int = 1,
    max_workers: int | None = None,
    max_scale_down_fraction_per_cycle: float = 1.0,
    critical_threshold: float = 0.85,
    polling_interval_s: float = 5.0,
    enable_gate: bool = True,
) -> SaturationAwareScheduler:
    """Build a setup-completed single-stage scheduler for gate fixtures.

    The default config disables both warmup graces (so a single
    cycle's signal is sufficient) and the floor-stuck grace window
    so floor enforcement fires immediately when the gate test needs
    it.
    """
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        memory_pressure_critical_threshold=critical_threshold,
        memory_pressure_polling_interval_s=polling_interval_s,
        enable_memory_pressure_gate=enable_gate,
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            min_workers=min_workers,
            max_workers=max_workers,
            max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
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
    scheduler = SaturationAwareScheduler(cfg)
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


class TestMemoryPressureGate:
    """Cluster object-store memory-pressure gate on Phase C scale-up."""

    def test_high_pressure_freezes_phase_c_scale_up(self) -> None:
        """``used_fraction=0.95 > threshold=0.85`` -> Phase C grow yields zero adds, gauge active."""
        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            _patch_ray_resources(total=100.0, available=5.0),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 3}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert solution.stages[0].new_workers == []
        assert scheduler._memory_pressure_monitor.last_pressure_active is True
        assert scheduler._memory_pressure_monitor.last_used_fraction == pytest.approx(0.95)

    def test_low_pressure_runs_phase_c_normally(self) -> None:
        """``used_fraction=0.20 < threshold=0.85`` -> Phase C grows, gauge inactive."""
        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            _patch_ray_resources(total=100.0, available=80.0),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 3}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(solution.stages[0].new_workers) == 3
        assert scheduler._memory_pressure_monitor.last_pressure_active is False
        assert scheduler._memory_pressure_monitor.last_used_fraction == pytest.approx(0.20)

    def test_floor_unaffected_by_pressure(self) -> None:
        """Below-floor stage gains a worker via Phase B even under high memory pressure."""
        scheduler = _scheduler(min_workers=2)
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            _patch_ray_resources(total=100.0, available=5.0),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(solution.stages[0].new_workers) == 1
        assert solution.stages[0].deleted_workers == []
        assert scheduler._memory_pressure_monitor.last_pressure_active is True

    def test_phase_d_shrink_unaffected_by_pressure(self) -> None:
        """Negative intent under high pressure still removes a worker via Phase D."""
        scheduler = _scheduler(min_workers=1)
        ps = _problem_state_with_workers(worker_ids=["stage-w0", "stage-w1", "stage-w2"])

        with (
            _patch_ray_resources(total=100.0, available=5.0),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": -1}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert solution.stages[0].new_workers == []
        assert len(solution.stages[0].deleted_workers) == 1
        assert scheduler._memory_pressure_monitor.last_pressure_active is True

    def test_polling_failure_degrades_gracefully(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Ray API raising -> no exception propagates, one WARN log, Phase C runs."""
        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            patch(f"{MEMORY_PRESSURE_MODULE}.ray.is_initialized", return_value=True),
            patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.cluster_resources",
                side_effect=RuntimeError("Ray API down"),
            ),
            patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.available_resources",
                return_value={"object_store_memory": 5.0},
            ),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 3}),
        ):
            solution_first = scheduler.autoscale(time=0.0, problem_state=ps)

        warn_records = [r for r in loguru_caplog.records if "memory pressure gate: failed to query" in r.message]
        assert len(warn_records) == 1
        assert warn_records[0].levelno == logging.WARNING
        assert scheduler._memory_pressure_monitor.last_pressure_active is False
        assert len(solution_first.stages[0].new_workers) == 3

    def test_ray_not_initialized_degrades_silently(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Startup before Ray init reports pressure inactive without noisy WARN logs."""
        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            patch(f"{MEMORY_PRESSURE_MODULE}.ray.is_initialized", return_value=False),
            patch(f"{MEMORY_PRESSURE_MODULE}.ray.cluster_resources") as cluster_mock,
            patch(f"{MEMORY_PRESSURE_MODULE}.ray.available_resources") as available_mock,
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 3}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(solution.stages[0].new_workers) == 3
        assert scheduler._memory_pressure_monitor.last_pressure_active is False
        assert scheduler._memory_pressure_monitor.last_used_fraction == 0.0
        cluster_mock.assert_not_called()
        available_mock.assert_not_called()
        assert "memory pressure gate: failed to query" not in loguru_caplog.text

    def test_missing_object_store_capacity_reports_inactive(self) -> None:
        """Clusters reporting no object-store capacity clamp used fraction to zero."""
        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            _patch_ray_resources(total=0.0, available=0.0),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 3}),
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(solution.stages[0].new_workers) == 3
        assert scheduler._memory_pressure_monitor.last_pressure_active is False
        assert scheduler._memory_pressure_monitor.last_used_fraction == 0.0

    def test_pressure_clear_emits_recovery_log(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Active pressure followed by a fresh low-pressure poll logs a single recovery INFO."""
        scheduler = _scheduler(polling_interval_s=5.0)
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            patch(f"{MEMORY_PRESSURE_MODULE}.ray.is_initialized", return_value=True),
            patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.cluster_resources",
                return_value={"object_store_memory": 100.0},
            ),
            patch(
                f"{MEMORY_PRESSURE_MODULE}.ray.available_resources",
                side_effect=[
                    {"object_store_memory": 5.0},
                    {"object_store_memory": 80.0},
                ],
            ),
            patch(f"{SATURATION_AWARE_MODULE}.time.monotonic", side_effect=[0.0, 6.0]),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}),
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)
            scheduler.autoscale(time=6.0, problem_state=ps)

        assert scheduler._memory_pressure_monitor.last_pressure_active is False
        assert scheduler._memory_pressure_monitor.last_used_fraction == pytest.approx(0.20)
        clear_records = [r for r in loguru_caplog.records if "memory pressure gate: CLEARED" in r.message]
        assert len(clear_records) == 1
        assert clear_records[0].levelno == logging.INFO

    def test_polling_interval_caches_fraction(self) -> None:
        """Consecutive cycles inside one polling window make exactly one Ray API call."""
        scheduler = _scheduler(polling_interval_s=5.0)
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            _patch_ray_resources(total=100.0, available=50.0) as ray_patch,
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}),
            patch(
                f"{SATURATION_AWARE_MODULE}.time.monotonic",
                side_effect=[0.0, 2.0, 6.0],
            ),
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)
            scheduler.autoscale(time=2.0, problem_state=ps)
            scheduler.autoscale(time=6.0, problem_state=ps)

        assert ray_patch.cluster_mock.call_count == 2
        assert ray_patch.available_mock.call_count == 2
        assert scheduler._memory_pressure_monitor.poll_count == 2

    def test_setup_resets_monitor_state(self) -> None:
        """A re-``setup()`` clears the cached pressure state."""
        scheduler = _scheduler()
        ps = _problem_state_with_workers(worker_ids=["stage-w0"])

        with (
            _patch_ray_resources(total=100.0, available=5.0),
            patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}),
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)

        monitor = scheduler._memory_pressure_monitor
        before_active = monitor.last_pressure_active
        before_poll_at = monitor.last_poll_at
        before_used_fraction = monitor.last_used_fraction
        assert before_active is True
        assert before_poll_at is not None
        assert before_used_fraction == pytest.approx(0.95)

        cluster = _cluster()
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        scheduler.setup(
            data_structures.Problem(
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
            ),
        )

        after_active = monitor.last_pressure_active
        after_poll_at = monitor.last_poll_at
        after_used_fraction = monitor.last_used_fraction
        after_poll_count = monitor.poll_count
        assert after_active is False
        assert after_poll_at is None
        assert after_used_fraction == 0.0
        assert after_poll_count == 0

    def test_reset_drives_gauges_to_cleared_defaults(self) -> None:
        """``reset()`` writes 0.0 to both gauges so a scrape between reset and the next poll sees the cleared state."""
        from cosmos_xenna.pipelines.private.scheduling_py.memory_pressure import MemoryPressureMonitor

        monitor = MemoryPressureMonitor(
            polling_interval_s=5.0,
            critical_threshold=0.85,
            pipeline_name="test-pipe",
        )

        used_fraction_calls: list[tuple[float, dict[str, str]]] = []
        pressure_active_calls: list[tuple[float, dict[str, str]]] = []
        # Replace the bound .set methods so both writes during reset() are captured.
        monitor._used_fraction_gauge.set = lambda value, tags: used_fraction_calls.append((value, dict(tags)))  # type: ignore[method-assign]
        monitor._pressure_active_gauge.set = lambda value, tags: pressure_active_calls.append((value, dict(tags)))  # type: ignore[method-assign]

        monitor.reset()

        assert used_fraction_calls == [(0.0, {"pipeline": "test-pipe"})]
        assert pressure_active_calls == [(0.0, {"pipeline": "test-pipe"})]
