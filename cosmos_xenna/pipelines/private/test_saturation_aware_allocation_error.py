# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defense-in-depth tests for Phase C ``try_add_worker`` exception absorption."""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py import allocation_failures
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


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


def _cluster() -> resources.ClusterResources:
    """Two-node CPU cluster so the fragmentation snapshot has multiple rows."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0"),
            "node-1": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-1"),
        },
    )


def _scheduler(*, skip_on_error: bool = True, pipeline_name: str = "test-pipe") -> SaturationAwareScheduler:
    """Single-stage scheduler primed for Phase C grow."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        skip_cycle_on_allocation_error=skip_on_error,
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            min_workers=1,
        ),
    )
    cluster = _cluster()
    shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    problem = data_structures.Problem(
        cluster,
        [
            data_structures.ProblemStage(
                name="stage",
                stage_batch_size=1,
                worker_shape=shape,
                requested_num_workers=None,
                over_provision_factor=None,
            ),
        ],
    )
    scheduler = SaturationAwareScheduler(cfg, pipeline_name=pipeline_name)
    scheduler.setup(problem)
    return scheduler


def _problem_state_with_one_worker() -> data_structures.ProblemState:
    """Single-stage ``ProblemState`` carrying one busy worker so Phase C wants more."""
    worker = data_structures.ProblemWorkerGroupState.make(
        "stage-w0",
        [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        num_used_slots=1,
    )
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(
                stage_name="stage",
                workers=[worker],
                slots_per_worker=1,
                is_finished=False,
                num_used_slots=1,
                num_empty_slots=0,
                input_queue_depth=0,
                num_pending_actors=0,
            ),
        ],
    )


class TestAllocationErrorDefense:
    """Pin the AllocationError defense-in-depth contract for Phase C grow."""

    def test_allocation_error_absorbed_logs_snapshot_and_increments_counter(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Injected ``AllocationError`` is absorbed: ERROR log, counter +1, no propagation."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            err = resources.AllocationError("synthetic placement failure")
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker", side_effect=err
            ):
                scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == [{"stage": "stage", "pipeline": "test-pipe"}]
        error_records = [r for r in loguru_caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        assert "saturation-aware allocation failure" in error_records[0].message
        assert "Per-GPU fragmentation snapshot" in error_records[0].message

    def test_unexpected_exception_absorbed_same_treatment(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-AllocationError raise is absorbed by the defensive ``except Exception`` clause."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()

        err = RuntimeError("synthetic non-allocation raise")
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker", side_effect=err
            ):
                scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == [{"stage": "stage", "pipeline": "test-pipe"}]
        error_records = [r for r in loguru_caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        assert "RuntimeError" in error_records[0].message

    def test_kill_switch_off_propagates_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``skip_cycle_on_allocation_error=False`` lets the exception kill the cycle."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler(skip_on_error=False)
        ps = _problem_state_with_one_worker()

        err = resources.AllocationError("synthetic placement failure")
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker", side_effect=err
            ):
                with pytest.raises(resources.AllocationError, match="synthetic placement failure"):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == [{"stage": "stage", "pipeline": "test-pipe"}]

    def test_none_return_absorption_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Planner returning ``None`` (no-fit) still increments the stuck counter, never the failure counter."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                return_value=None,
            ):
                scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == [], "No-fit must not bump the failure counter"
        assert scheduler._stuck_plan_counters["stage"] == 1, "No-fit must increment the stuck counter"
