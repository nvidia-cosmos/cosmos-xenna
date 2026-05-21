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


"""Defense-in-depth tests for Phase C ``try_add_worker`` exception absorption."""

import logging
import uuid
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

    def test_emit_allocation_failure_logs_gpu_fragmentation_rows(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-empty GPU cluster snapshot includes per-GPU used/free fractions."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(
                    used_cpus=0,
                    total_cpus=8,
                    gpus=[
                        resources.GpuResources(index=1, uuid_=uuid.uuid4(), used_fraction=0.25),
                        resources.GpuResources(index=0, uuid_=uuid.uuid4(), used_fraction=0.75),
                    ],
                    name="node-0",
                ),
            },
        )

        allocation_failures.emit_allocation_failure(
            stage_name="stage",
            pipeline_name="test-pipe",
            cluster_resources=cluster,
            exc=resources.AllocationError("synthetic placement failure"),
        )

        assert fake_counter.calls == [{"stage": "stage", "pipeline": "test-pipe"}]
        error_records = [r for r in loguru_caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        assert "'gpu_index': 0" in error_records[0].message
        assert "'used_fraction': 0.75" in error_records[0].message
        assert "'free_fraction': 0.25" in error_records[0].message

    def test_emit_allocation_failure_logs_cpu_fragmentation_rows(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CPU-only stage failure surfaces per-node used/total/free CPU rows.

        Sibling to ``test_emit_allocation_failure_logs_gpu_fragmentation_rows``;
        ensures CPU-only stages whose ``try_add_worker`` fails get an
        actionable snapshot (the GPU snapshot would be empty and thus
        uninformative for those stages).
        """
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(used_cpus=7.5, total_cpus=8, gpus=[], name="node-0"),
                "node-1": resources.NodeResources(used_cpus=2.0, total_cpus=8, gpus=[], name="node-1"),
            },
        )

        allocation_failures.emit_allocation_failure(
            stage_name="cpu_stage",
            pipeline_name="test-pipe",
            cluster_resources=cluster,
            exc=resources.AllocationError("synthetic CPU placement failure"),
        )

        error_records = [r for r in loguru_caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        message = error_records[0].message
        assert "Per-node CPU snapshot" in message
        assert "'node': 'node-0'" in message
        assert "'cpu_used': 7.5" in message
        assert "'cpu_total': 8.0" in message
        # node-0: (8 - 7.5) / 8 = 0.0625; node-1: (8 - 2) / 8 = 0.75
        assert "'cpu_free_fraction': 0.0625" in message
        assert "'cpu_free_fraction': 0.75" in message

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
        # ``_scheduler()`` runs against the CPU-only ``_cluster()`` fixture, so
        # the operator-actionable snapshot is the per-node CPU table. The
        # per-GPU snapshot is still emitted but is empty for this cluster
        # shape, so asserting on it would not validate any cluster signal.
        assert "Per-node CPU snapshot" in error_records[0].message

    def test_unexpected_exception_propagates_and_does_not_increment_counter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-AllocationError raises propagate; the allocation-failure counter is NOT incremented.

        The defense-in-depth scope is narrowed to ``AllocationError``
        only: scheduler bugs surfacing as ``SchedulerInvariantError``,
        ``KeyError``, ``IndexError``, etc. propagate to the autoscaler
        thread instead of being silently re-routed through the absorb
        path. That re-routing previously masked planner defects and
        flagged them as transient placement failures.
        """
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()

        err = RuntimeError("synthetic non-allocation raise")
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker", side_effect=err
            ):
                with pytest.raises(RuntimeError, match="synthetic non-allocation raise"):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == []

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


class TestDonorRetryTransactionality:
    """Pin the contract that donor retry failures roll into a Phase C allocation failure.

    Background:
        ``_attempt_cross_stage_donation`` removes the donor worker from
        the planner via ``ctx.try_remove_worker`` BEFORE the receiver
        retry runs. If the post-donation ``try_add_worker`` retry
        returns ``None`` (the freed placement does not match the
        receiver shape), we cannot roll back the donor removal.

    Contract (post-fix):
        On a donor-retry failure the scheduler must NOT silently log
        and break, because that produces a one-sided shrink (donor
        gone, receiver did not grow). Instead it routes the failure
        through ``_absorb_allocation_failure`` so the allocation
        counter increments, the cycle aborts under the kill switch,
        and the operator-visible log carries the standard
        fragmentation snapshot. Donation bookkeeping is NOT updated
        because the donation did not land end-to-end.
    """

    def test_donor_retry_failure_increments_allocation_counter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retry returning ``None`` after a successful donation increments the failure counter."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()

        # First try_add_worker call (initial Phase C add attempt) returns None
        # to exhaust the cluster; the second call (post-donation retry) also
        # returns None to model the donor's freed placement not matching the
        # receiver shape. The cross-stage donation helper succeeds in between.
        try_add_returns = iter([None, None])
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=lambda *_a, **_k: next(try_add_returns),
            ):
                with patch.object(
                    scheduler,
                    "_attempt_cross_stage_donation",
                    return_value="stage",
                ):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == [{"stage": "stage", "pipeline": "test-pipe"}], (
            "donor-retry failure must increment the standard allocation-failure counter"
        )

    def test_donor_retry_failure_does_not_record_donation_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Donation bookkeeping (cooldown / per-cycle counter) is not updated when retry fails."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()

        try_add_returns = iter([None, None])
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=lambda *_a, **_k: next(try_add_returns),
            ):
                with patch.object(scheduler, "_attempt_cross_stage_donation", return_value="stage"):
                    with patch.object(scheduler, "_record_donation_success") as record_mock:
                        scheduler.autoscale(time=0.0, problem_state=ps)

        assert record_mock.call_count == 0, (
            "_record_donation_success must not be called for a donation that fails the receiver retry"
        )

    def test_donor_retry_failure_propagates_when_kill_switch_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``skip_cycle_on_allocation_error=False`` re-raises the synthetic donor-retry RuntimeError."""
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)

        scheduler = _scheduler(skip_on_error=False)
        ps = _problem_state_with_one_worker()

        try_add_returns = iter([None, None])
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=lambda *_a, **_k: next(try_add_returns),
            ):
                with patch.object(scheduler, "_attempt_cross_stage_donation", return_value="stage"):
                    with pytest.raises(RuntimeError, match="donor-retry-failed"):
                        scheduler.autoscale(time=0.0, problem_state=ps)

        assert fake_counter.calls == [{"stage": "stage", "pipeline": "test-pipe"}], (
            "the standard allocation-failure log/counter must still fire before the re-raise"
        )


class TestAllocationFailureLogIsInjectionSafe:
    """Pin the contract that hostile stage/pipeline names cannot poison the log line.

    Background:
        ``emit_allocation_failure`` interpolates ``stage_name`` and
        ``pipeline_name`` into an f-string log message and into
        Counter ``tags``. If a stage name contained an embedded
        format token (e.g. ``"{0.__class__}"``) or control characters,
        an unsafe formatter could re-evaluate it. loguru's logger
        receives the already-rendered string, so this test pins
        the safety boundary explicitly: the literal stage name
        appears verbatim in the log and the Counter tags, and no
        secondary formatting happens against the message.
    """

    def test_curly_braces_in_stage_name_are_logged_literally(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stage name with ``{token}`` survives unrendered into the ERROR record and the counter tags.

        The log message body interpolates only the stage name via an
        f-string. The pipeline name is carried separately as a counter
        tag (not in the log body), so we only verify the stage name
        round-trip in the log; the pipeline name round-trip is verified
        against ``fake_counter.calls``.
        """
        fake_counter = _FakeCounter()
        monkeypatch.setattr(allocation_failures, "_ALLOCATION_FAILURES_COUNTER", fake_counter)
        hostile_stage = "stage-{0.__class__}-{foo!r}"
        hostile_pipeline = "pipe-{1.__init__}"

        allocation_failures.emit_allocation_failure(
            stage_name=hostile_stage,
            pipeline_name=hostile_pipeline,
            cluster_resources=_cluster(),
            exc=resources.AllocationError("synthetic"),
        )

        assert fake_counter.calls == [{"stage": hostile_stage, "pipeline": hostile_pipeline}]
        error_records = [r for r in loguru_caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        message = error_records[0].message
        assert hostile_stage in message, "hostile stage name must appear verbatim, never re-evaluated"
