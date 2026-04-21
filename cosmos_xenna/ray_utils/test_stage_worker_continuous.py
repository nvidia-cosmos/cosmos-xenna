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

"""Tests continuous-mode dispatch, queueing, and shutdown behavior."""

import asyncio
import collections
import contextlib
import queue
import threading
import time
import typing
from typing import Any
from unittest.mock import MagicMock

import pytest
from ray import ObjectRef

import cosmos_xenna.ray_utils.actor_pool as ap_module
import cosmos_xenna.ray_utils.stage_worker as sw_module
from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.pipelines.private import resources as resources_mod
from cosmos_xenna.ray_utils.continuous_stage import (
    ContinuousInterface,
    ContinuousTaskInput,
    ContinuousTaskOutput,
)


def _mock_object_ref(name: str = "ref") -> ObjectRef[Any]:
    """Return a ``MagicMock`` cast to ``ObjectRef[Any]`` for typed tests."""
    return typing.cast("ObjectRef[Any]", MagicMock(name=name))


class _DoublerContinuousStage(pipelines_v1.Stage, ContinuousInterface):
    """Test stage that doubles each integer input."""

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int] | None:  # pragma: no cover - never called
        raise NotImplementedError("Continuous stage; framework dispatches to run_continuous")

    async def run_continuous(
        self,
        input_queue: asyncio.Queue[ContinuousTaskInput],
        output_queue: asyncio.Queue[ContinuousTaskOutput],
        stop_event: asyncio.Event,
    ) -> None:
        """Drain ``input_queue`` until ``stop_event`` fires; double each int."""
        while not stop_event.is_set():
            try:
                task_in = await asyncio.wait_for(input_queue.get(), timeout=0.1)
            except TimeoutError:
                # Re-check stop_event on a short cadence.
                continue

            doubled = [x * 2 for x in task_in.data]
            await output_queue.put(
                ContinuousTaskOutput(
                    task_id=task_in.task_id,
                    out_data=doubled,
                    timing=task_in.timing,
                    object_sizes=task_in.object_sizes,
                )
            )


@pytest.mark.slow
def test_continuous_stage_streams_through_real_pipeline() -> None:
    """A continuous-mode stage produces correct outputs via run_pipeline()."""
    n_inputs = 16
    spec = pipelines_v1.PipelineSpec(
        input_data=list(range(n_inputs)),
        stages=[_DoublerContinuousStage()],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            logging_interval_s=5,
            return_last_stage_outputs=True,
        ),
    )
    results = typing.cast(list[int], pipelines_v1.run_pipeline(spec))

    assert results is not None
    assert sorted(results) == [x * 2 for x in range(n_inputs)]


def test_continuous_stage_inherits_pipeline_default_slots_per_actor() -> None:
    """A stage with ``slots_per_actor=None`` inherits the pipeline default."""
    stage_spec = pipelines_v1.StageSpec(stage=_DoublerContinuousStage())
    pipeline_config = pipelines_v1.PipelineConfig()

    resolved = stage_spec.override_with_pipeline_params(pipeline_config)

    assert resolved.slots_per_actor == pipeline_config.slots_per_actor


def test_continuous_stage_explicit_slots_per_actor_is_preserved() -> None:
    """An explicit ``slots_per_actor`` on the spec overrides the pipeline default."""
    stage_spec = pipelines_v1.StageSpec(stage=_DoublerContinuousStage(), slots_per_actor=8)

    resolved = stage_spec.override_with_pipeline_params(pipelines_v1.PipelineConfig())

    assert resolved.slots_per_actor == 8


def test_stage_spec_rejects_zero_slots_per_actor() -> None:
    """``slots_per_actor=0`` is rejected at ``StageSpec`` construction time."""
    with pytest.raises(ValueError, match="slots_per_actor"):
        pipelines_v1.StageSpec(stage=_DoublerContinuousStage(), slots_per_actor=0)


def test_pipeline_config_rejects_zero_slots_per_actor() -> None:
    """``slots_per_actor=0`` is rejected at ``PipelineConfig`` construction time."""
    with pytest.raises(ValueError, match="slots_per_actor"):
        pipelines_v1.PipelineConfig(slots_per_actor=0)


# Note: wrapper-selection (``ContinuousWrappedStage`` vs ``WrappedStage``)
# is already covered by the integration test above - if the wrong wrapper
# were selected, the worker would dispatch to ``process_data`` (which raises
# ``NotImplementedError`` in our stage) and the pipeline run would fail.


def _build_ready_actor_with_in_flight_tasks(num_slots: int) -> ap_module._ReadyActor[Any]:
    """Build a real ``_ReadyActor`` with every slot occupied by a real ``Task``.

    The slots use mock ``ObjectRef``s for both ``object_ref`` and inside
    ``Task.task_data`` so the test never touches Ray. Each slot's task is
    distinguishable via ``origin_node_id`` so the assertion can check
    ordering and identity.
    """
    actor: ap_module._ReadyActor[Any] = ap_module._ReadyActor(
        metadata=resources_mod.WorkerMetadata.make_dummy(),
        actor_ref=MagicMock(name="actor_handle"),
        start_time=time.time(),
        slots=collections.deque(),
    )
    for i in range(num_slots):
        task: ap_module.Task[Any] = ap_module.Task(
            task_data=[_mock_object_ref(f"task_data_ref_{i}")],
            origin_node_id=f"node-task-{i}",
        )
        actor.slots.append(
            ap_module._Slot(
                task=ap_module._SlotData(
                    task=task,
                    scheduled_time=time.time(),
                    object_ref=MagicMock(name=f"slot_ref_{i}"),
                ),
            )
        )
    return actor


class TestPoolInitiatedKillRequeue:
    """Verify kill-path re-queue behavior for in-flight tasks."""

    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace ``_kill_actor_and_reap`` with a no-op for every test in the class."""
        monkeypatch.setattr(ap_module, "_kill_actor_and_reap", MagicMock(name="kill_no_op"))

    def _make_pool(self) -> MagicMock:
        """Build a lightweight pool that exercises the real ``_try_delete_ready_actor``."""
        pool = MagicMock(spec=ap_module.ActorPool)
        pool._task_queue = collections.deque()
        pool._ready_actors = {}
        pool._try_delete_ready_actor = ap_module.ActorPool._try_delete_ready_actor.__get__(pool)
        return pool

    def test_inflight_tasks_are_returned_to_task_queue(self) -> None:
        """All ``Task``s from occupied slots end up in ``_task_queue`` (no drops)."""
        actor = _build_ready_actor_with_in_flight_tasks(num_slots=4)
        original_tasks = [s.get_task.task for s in actor.slots]
        pool = self._make_pool()
        pool._ready_actors["actor-A"] = actor

        deleted = pool._try_delete_ready_actor("actor-A")

        assert deleted is True
        queued_ids = {id(t) for t in pool._task_queue}
        original_ids = {id(t) for t in original_tasks}
        assert queued_ids == original_ids, "every in-flight task must reappear in the queue (by identity)"
        assert len(pool._task_queue) == 4, "no duplicates introduced by re-queue"

    def test_actor_is_removed_from_ready_set(self) -> None:
        """The actor entry is popped from ``_ready_actors`` so completion polling skips it."""
        actor = _build_ready_actor_with_in_flight_tasks(num_slots=2)
        pool = self._make_pool()
        pool._ready_actors["actor-B"] = actor

        pool._try_delete_ready_actor("actor-B")

        assert "actor-B" not in pool._ready_actors

    def test_empty_slots_are_not_requeued_as_phantom_tasks(self) -> None:
        """Empty slots must not appear as ``None``/dummy ``Task``s in ``_task_queue``."""
        actor: ap_module._ReadyActor[Any] = ap_module._ReadyActor(
            metadata=resources_mod.WorkerMetadata.make_dummy(),
            actor_ref=MagicMock(name="actor_handle"),
            start_time=time.time(),
            slots=collections.deque([ap_module._Slot(task=None), ap_module._Slot(task=None)]),
        )
        pool = self._make_pool()
        pool._ready_actors["actor-empty"] = actor

        pool._try_delete_ready_actor("actor-empty")

        assert len(pool._task_queue) == 0

    def test_kill_skips_graceful_drain_on_pool_initiated_kill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure re-queue kill path calls actor kill with zero grace period."""
        actor = _build_ready_actor_with_in_flight_tasks(num_slots=2)
        pool = self._make_pool()
        pool._ready_actors["actor-G"] = actor

        graceful_calls: list[float] = []

        def _record_grace(*_args: object, grace_period_s: float = -1.0, **_kwargs: object) -> None:
            graceful_calls.append(grace_period_s)

        monkeypatch.setattr(ap_module, "_kill_actor_and_reap", _record_grace)

        pool._try_delete_ready_actor("actor-G")

        assert graceful_calls == [0.0], (
            "kill on the re-queue path must pass grace_period_s=0.0 so _attempt_graceful_shutdown short-circuits"
        )

    def test_kill_runs_after_requeue_so_dead_objectref_cannot_steal_data(self) -> None:
        """Ensure kill executes after all slot tasks are re-queued."""
        actor = _build_ready_actor_with_in_flight_tasks(num_slots=3)
        pool = self._make_pool()
        pool._ready_actors["actor-K"] = actor

        observed_queue_size_at_kill = -1

        def _record_queue_size_then_noop(*_args: object, **_kwargs: object) -> None:
            nonlocal observed_queue_size_at_kill
            observed_queue_size_at_kill = len(pool._task_queue)

        ap_module._kill_actor_and_reap.side_effect = _record_queue_size_then_noop  # type: ignore[attr-defined]

        pool._try_delete_ready_actor("actor-K")

        assert observed_queue_size_at_kill == 3, "kill must run AFTER all slots are re-queued"
        ap_module._kill_actor_and_reap.assert_called_once()  # type: ignore[attr-defined]

    def test_unknown_actor_id_is_a_noop(self) -> None:
        """Calling on an unknown id returns False and leaves state untouched."""
        pool = self._make_pool()

        deleted = pool._try_delete_ready_actor("does-not-exist")

        assert deleted is False
        assert len(pool._task_queue) == 0
        ap_module._kill_actor_and_reap.assert_not_called()  # type: ignore[attr-defined]


class TestStopWarnsOnAbandonedTasks:
    """``ActorPool.stop`` MUST log a WARNING when it drops queued tasks."""

    def _make_pool(self) -> MagicMock:
        """Build a stop()-ready pool with empty actor / worker-group state."""
        pool = MagicMock(spec=ap_module.ActorPool)
        pool.name = "TestPool"
        pool._task_queue = collections.deque()
        pool._completed_tasks = collections.deque()
        pool._pending_actors = collections.OrderedDict()
        pool._ready_actors = {}
        pool._pending_node_actors = collections.OrderedDict()
        pool._actors_waiting_for_node_setup = collections.defaultdict(list)
        pool._nodes_with_completed_setups = set()
        pool._worker_groups = {}
        pool._worker_groups_to_create = collections.deque()
        pool._worker_groups_to_delete = collections.deque()
        pool.stop = ap_module.ActorPool.stop.__get__(pool)
        return pool

    def test_empty_queue_does_not_emit_warning(self) -> None:
        """When nothing was queued, ``stop()`` must NOT emit the abandoned-task warning."""
        pool = self._make_pool()

        with pytest.MonkeyPatch.context() as mp:
            mock_logger = MagicMock()
            mp.setattr(ap_module, "logger", mock_logger)
            pool.stop()

        warnings_about_abandoned = [
            c for c in mock_logger.warning.call_args_list if "abandoned task(s)" in str(c.args[0] if c.args else "")
        ]
        assert warnings_about_abandoned == []

    def test_non_empty_queue_emits_warning_with_count_and_origins(self) -> None:
        """The WARNING must carry the count and a per-origin Counter for triage."""
        pool = self._make_pool()
        pool._task_queue.extend(
            [
                ap_module.Task(task_data=[_mock_object_ref()], origin_node_id="node-A"),
                ap_module.Task(task_data=[_mock_object_ref()], origin_node_id="node-A"),
                ap_module.Task(task_data=[_mock_object_ref()], origin_node_id="node-B"),
            ]
        )

        with pytest.MonkeyPatch.context() as mp:
            mock_logger = MagicMock()
            mp.setattr(ap_module, "logger", mock_logger)
            pool.stop()

        abandoned_warnings = [c for c in mock_logger.warning.call_args_list if "abandoned task(s)" in c.args[0]]
        assert len(abandoned_warnings) == 1
        msg = abandoned_warnings[0].args[0]
        assert "3 abandoned task(s)" in msg
        assert "'node-A': 2" in msg
        assert "'node-B': 1" in msg

    def test_queue_is_cleared_after_stop(self) -> None:
        """``stop()`` must drain ``_task_queue`` regardless of whether work was queued."""
        pool = self._make_pool()
        pool._task_queue.append(ap_module.Task(task_data=[_mock_object_ref()], origin_node_id="node-X"))

        pool.stop()

        assert len(pool._task_queue) == 0


class TestCollectorDrainsAfterStop:
    """``_collect_continuous_async`` must drain the queue before exiting."""

    class _CollectorHostStub:
        """Minimal stand-in for the attributes the collector touches."""

        def __init__(self) -> None:
            self._results: dict[str, sw_module._Result[Any]] = {}
            self.results_lock = threading.Lock()

    def _drive_collector(
        self,
        publishes_after_drain_done: int,
    ) -> dict[str, sw_module._Result[Any]]:
        """Run the collector against a pre-loaded queue with ``drain_done`` already set."""
        host = self._CollectorHostStub()
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        drain_done = asyncio.Event()
        for i in range(publishes_after_drain_done):
            output_q.put_nowait(
                ContinuousTaskOutput(
                    task_id=f"task-{i}",
                    out_data=[i],
                    timing=sw_module.TimingInfo(),
                    object_sizes=[1],
                )
            )
        drain_done.set()

        # Ray's @ray.remote tracing wrapper adds a positional arg; reach
        # the original coroutine via __wrapped__ to call it directly.
        collector_fn = sw_module.StageWorker._collect_continuous_async.__wrapped__  # type: ignore[attr-defined]
        asyncio.run(collector_fn(host, output_q, drain_done))
        return host._results

    def test_results_published_after_drain_signal_are_delivered(self) -> None:
        """Every result posted before the queue empties must reach ``self._results``."""
        results = self._drive_collector(publishes_after_drain_done=5)

        assert set(results.keys()) == {f"task-{i}" for i in range(5)}, (
            "collector dropped one or more results posted between drain_done.set() and queue drain"
        )

    def test_no_results_means_immediate_clean_exit(self) -> None:
        """When nothing is published, the collector exits cleanly via the drain check."""
        results = self._drive_collector(publishes_after_drain_done=0)

        assert results == {}


class TestUnwrapExceptionGroup:
    """Verify exception-group unwrapping semantics for continuous mode."""

    def test_returns_single_non_cancelled_exception(self) -> None:
        """Return the concrete exception when exactly one non-cancelled error exists."""
        eg = BaseExceptionGroup("outer", [RuntimeError("boom"), asyncio.CancelledError()])

        result = sw_module.StageWorker._unwrap_exception_group(eg)

        assert isinstance(result, RuntimeError)
        assert str(result) == "boom"

    def test_returns_exception_group_for_multiple_non_cancelled_errors(self) -> None:
        """Preserve multi-failure detail by returning an ``ExceptionGroup``."""
        eg = BaseExceptionGroup(
            "outer",
            [
                BaseExceptionGroup("inner", [ValueError("left"), RuntimeError("right")]),
                asyncio.CancelledError(),
            ],
        )

        result = sw_module.StageWorker._unwrap_exception_group(eg)

        assert isinstance(result, ExceptionGroup)
        assert [type(exc) for exc in result.exceptions] == [ValueError, RuntimeError]
        assert [str(exc) for exc in result.exceptions] == ["left", "right"]

    def test_returns_runtime_error_when_only_cancellations_exist(self) -> None:
        """Return a clear fallback error when only cancellation failures exist."""
        eg = BaseExceptionGroup("outer", [asyncio.CancelledError()])

        result = sw_module.StageWorker._unwrap_exception_group(eg)

        assert isinstance(result, RuntimeError)
        assert "cancellations only" in str(result)


class TestFeederTimestampsAtQueueHandoff:
    """Verify feeder timestamps start at successful queue handoff."""

    class _FeederHostStub:
        """Provide only the fields touched by ``_feed_continuous_async``."""

        def __init__(self, task: sw_module._DeserializedTaskDataWithId[Any]) -> None:
            self.deserialized_queue: queue.Queue[sw_module._DeserializedTaskDataWithId[Any]] = queue.Queue()
            self.deserialized_queue.put(task)

    async def _run_feeder_once_with_backpressure(self) -> tuple[float, ContinuousTaskInput]:
        timing = sw_module.TimingInfo()
        task = sw_module._DeserializedTaskDataWithId(
            data_refs=[_mock_object_ref("in-ref")],
            data=[123],
            uuid="task-0",
            timing=timing,
            object_sizes=[11],
        )
        host = self._FeederHostStub(task)
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue(maxsize=1)
        input_q.put_nowait(
            ContinuousTaskInput(
                task_id="blocker",
                data=[0],
                timing=sw_module.TimingInfo(),
                object_sizes=[1],
            )
        )
        stop_event = asyncio.Event()
        feeder_fn = sw_module.StageWorker._feed_continuous_async.__wrapped__  # type: ignore[attr-defined]
        feeder_task = asyncio.create_task(feeder_fn(host, input_q, stop_event))

        await asyncio.sleep(sw_module._CONTINUOUS_POLL_INTERVAL_S * 1.5)
        release_time = time.time()
        _ = input_q.get_nowait()

        enqueued: ContinuousTaskInput | None = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if not input_q.empty():
                enqueued = input_q.get_nowait()
                break
            await asyncio.sleep(sw_module._CONTINUOUS_POLL_INTERVAL_S / 2.0)

        stop_event.set()
        feeder_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feeder_task

        assert enqueued is not None, "feeder did not enqueue test task"
        return release_time, enqueued

    def test_process_start_excludes_input_queue_wait(self) -> None:
        """Timestamp is captured when enqueue succeeds, not while queue is full."""
        release_time, enqueued = asyncio.run(self._run_feeder_once_with_backpressure())

        assert enqueued.task_id == "task-0"
        assert enqueued.timing.process_start_time_s >= release_time


class TestFeederShutdownUnderBackpressure:
    """Verify the feeder exits promptly when stop fires while ``input_q`` is full."""

    class _FeederHostStub:
        def __init__(self, task: sw_module._DeserializedTaskDataWithId[Any]) -> None:
            self.deserialized_queue: queue.Queue[sw_module._DeserializedTaskDataWithId[Any]] = queue.Queue()
            self.deserialized_queue.put(task)

    async def _run(self) -> tuple[bool, asyncio.Queue[ContinuousTaskInput]]:
        task = sw_module._DeserializedTaskDataWithId(
            data_refs=[_mock_object_ref("in-ref")],
            data=[123],
            uuid="held-task",
            timing=sw_module.TimingInfo(),
            object_sizes=[11],
        )
        host = self._FeederHostStub(task)
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue(maxsize=1)
        input_q.put_nowait(
            ContinuousTaskInput(
                task_id="blocker",
                data=[0],
                timing=sw_module.TimingInfo(),
                object_sizes=[1],
            )
        )
        stop_event = asyncio.Event()
        feeder_fn = sw_module.StageWorker._feed_continuous_async.__wrapped__  # type: ignore[attr-defined]
        feeder_task = asyncio.create_task(feeder_fn(host, input_q, stop_event))

        # Let the feeder pick the held task and block on ``input_q.put``.
        await asyncio.sleep(sw_module._CONTINUOUS_POLL_INTERVAL_S * 1.5)
        stop_event.set()
        try:
            await asyncio.wait_for(feeder_task, timeout=1.0)
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True
            feeder_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feeder_task
        return timed_out, input_q

    def test_feeder_returns_when_stop_event_fires_during_backpressure(self) -> None:
        """Stop-wins: the held task is dropped and the feeder returns without deadlock."""
        timed_out, input_q = asyncio.run(self._run())

        assert not timed_out, "feeder did not return after stop_event fired under backpressure"
        # Stop-wins semantics: the held task must not have been enqueued on
        # this worker during shutdown. Only the original blocker remains.
        remaining = [input_q.get_nowait() for _ in range(input_q.qsize())]
        assert [item.task_id for item in remaining] == ["blocker"]
