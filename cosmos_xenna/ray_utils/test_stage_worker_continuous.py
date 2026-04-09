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

"""Tests for StageWorker continuous-mode async methods.

These are pure-async unit tests that exercise the feeding, collecting,
and timing-correction logic from ``StageWorker._feed_continuous_async``,
``_collect_continuous_async``, and ``_watch_stop_flag`` without
instantiating a Ray actor.  The logic under test is replicated from the
StageWorker methods so we can verify behaviour in isolation.
"""

import asyncio
import queue
import threading
import time
from unittest.mock import MagicMock

from cosmos_xenna.ray_utils.continuous_stage import ContinuousInterface, ContinuousTaskInput, ContinuousTaskOutput
from cosmos_xenna.ray_utils.stage_worker import FailureInfo, TaskDataInfo, TaskResultMetadata, TimingInfo


# ---------------------------------------------------------------------------
# Replicated feeder logic (mirrors StageWorker._feed_continuous_async)
# ---------------------------------------------------------------------------

async def _feed_continuous(
    deserialized_queue: queue.Queue,
    input_q: asyncio.Queue[ContinuousTaskInput],
    stop_event: asyncio.Event,
) -> None:
    """Bridge sync deserialized_queue to async input_queue (same as StageWorker)."""
    while not stop_event.is_set():
        try:
            task = await asyncio.to_thread(deserialized_queue.get, timeout=0.1)
        except queue.Empty:
            continue
        await input_q.put(
            ContinuousTaskInput(
                task_id=task.uuid,
                data=task.data,
                timing=task.timing,
                object_sizes=task.object_sizes,
            )
        )


# ---------------------------------------------------------------------------
# Replicated collector logic (mirrors StageWorker._collect_continuous_async)
# ---------------------------------------------------------------------------

async def _collect_continuous(
    output_q: asyncio.Queue[ContinuousTaskOutput],
    stop_event: asyncio.Event,
    results: dict,
    results_lock: threading.Lock,
) -> None:
    """Collect completed tasks with inter-completion-interval timing (same as StageWorker)."""
    last_completion_time: float | None = None
    while not stop_event.is_set():
        try:
            result = await asyncio.wait_for(output_q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

        now = time.time()
        timing = result.timing
        if last_completion_time is not None:
            timing.process_start_time_s = last_completion_time
        timing.process_end_time_s = now
        last_completion_time = now

        with results_lock:
            results[result.task_id] = {
                "task_id": result.task_id,
                "out_data": result.out_data,
                "metadata": TaskResultMetadata(
                    timing,
                    FailureInfo(should_process_further=True, should_restart_worker=False),
                    TaskDataInfo(sum(result.object_sizes)),
                    len(result.out_data),
                ),
            }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeedContinuousAsync:
    """Verify sync-to-async bridging in the feeder."""

    def test_bridges_single_task(self) -> None:
        """A single deserialized task should appear in the async input_queue."""
        deser_q: queue.Queue = queue.Queue()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue(maxsize=4)
        stop = asyncio.Event()

        mock_task = MagicMock()
        mock_task.uuid = "task-1"
        mock_task.data = ["payload"]
        mock_task.timing = TimingInfo()
        mock_task.object_sizes = [42]

        deser_q.put(mock_task)

        async def _drive() -> ContinuousTaskInput:
            feeder = asyncio.create_task(_feed_continuous(deser_q, input_q, stop))
            result = await asyncio.wait_for(input_q.get(), timeout=2.0)
            stop.set()
            feeder.cancel()
            try:
                await feeder
            except asyncio.CancelledError:
                pass
            return result

        runner = asyncio.Runner()
        try:
            got = runner.run(_drive())
        finally:
            runner.close()

        assert got.task_id == "task-1"
        assert got.data == ["payload"]
        assert got.object_sizes == [42]

    def test_bridges_multiple_tasks(self) -> None:
        """Multiple tasks should be delivered in order."""
        deser_q: queue.Queue = queue.Queue()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue(maxsize=4)
        stop = asyncio.Event()

        for i in range(3):
            mock_task = MagicMock()
            mock_task.uuid = f"task-{i}"
            mock_task.data = [f"data-{i}"]
            mock_task.timing = TimingInfo()
            mock_task.object_sizes = [i * 10]
            deser_q.put(mock_task)

        async def _drive() -> list[ContinuousTaskInput]:
            feeder = asyncio.create_task(_feed_continuous(deser_q, input_q, stop))
            results = []
            for _ in range(3):
                r = await asyncio.wait_for(input_q.get(), timeout=2.0)
                results.append(r)
            stop.set()
            feeder.cancel()
            try:
                await feeder
            except asyncio.CancelledError:
                pass
            return results

        runner = asyncio.Runner()
        try:
            got = runner.run(_drive())
        finally:
            runner.close()

        assert [g.task_id for g in got] == ["task-0", "task-1", "task-2"]

    def test_stops_on_event(self) -> None:
        """Feeder should exit when stop_event is set (no tasks to drain)."""
        deser_q: queue.Queue = queue.Queue()
        input_q: asyncio.Queue[ContinuousTaskInput] = asyncio.Queue(maxsize=4)
        stop = asyncio.Event()

        async def _drive() -> None:
            feeder = asyncio.create_task(_feed_continuous(deser_q, input_q, stop))
            await asyncio.sleep(0.2)
            stop.set()
            # Feeder should exit within 0.2s (one timeout cycle).
            await asyncio.wait_for(feeder, timeout=2.0)

        runner = asyncio.Runner()
        try:
            runner.run(_drive())
        finally:
            runner.close()

        assert input_q.empty()


class TestCollectContinuousAsync:
    """Verify collector logic and inter-completion-interval timing."""

    def test_collects_single_result(self) -> None:
        """A single output should be stored in results."""
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()
        results: dict = {}
        lock = threading.Lock()

        timing = TimingInfo()
        output_q.put_nowait(
            ContinuousTaskOutput(
                task_id="out-1",
                out_data=["result"],
                timing=timing,
                object_sizes=[50],
            )
        )

        async def _drive() -> None:
            collector = asyncio.create_task(_collect_continuous(output_q, stop, results, lock))
            # Wait for the result to be processed.
            for _ in range(50):
                await asyncio.sleep(0.05)
                if "out-1" in results:
                    break
            stop.set()
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass

        runner = asyncio.Runner()
        try:
            runner.run(_drive())
        finally:
            runner.close()

        assert "out-1" in results
        assert results["out-1"]["out_data"] == ["result"]
        assert results["out-1"]["metadata"].num_returns == 1
        assert results["out-1"]["metadata"].task_data_info.serialized_input_size == 50

    def test_inter_completion_timing_first_task_unchanged(self) -> None:
        """First completed task should keep its original process_start_time_s."""
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()
        results: dict = {}
        lock = threading.Lock()

        timing = TimingInfo(process_start_time_s=100.0)
        output_q.put_nowait(
            ContinuousTaskOutput(
                task_id="first",
                out_data=["r"],
                timing=timing,
                object_sizes=[10],
            )
        )

        async def _drive() -> None:
            collector = asyncio.create_task(_collect_continuous(output_q, stop, results, lock))
            for _ in range(50):
                await asyncio.sleep(0.05)
                if "first" in results:
                    break
            stop.set()
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass

        runner = asyncio.Runner()
        try:
            runner.run(_drive())
        finally:
            runner.close()

        # First task: process_start_time_s should remain as original (100.0)
        # because there's no previous completion to reference.
        assert results["first"]["metadata"].timing.process_start_time_s == 100.0
        # process_end_time_s should be set to "now" (a recent timestamp).
        assert results["first"]["metadata"].timing.process_end_time_s > 0

    def test_inter_completion_timing_second_task_corrected(self) -> None:
        """Second task should have process_start_time_s set to first task's completion time."""
        output_q: asyncio.Queue[ContinuousTaskOutput] = asyncio.Queue()
        stop = asyncio.Event()
        results: dict = {}
        lock = threading.Lock()

        t1 = TimingInfo(process_start_time_s=100.0)
        t2 = TimingInfo(process_start_time_s=200.0)

        output_q.put_nowait(
            ContinuousTaskOutput(task_id="t1", out_data=["a"], timing=t1, object_sizes=[10])
        )
        output_q.put_nowait(
            ContinuousTaskOutput(task_id="t2", out_data=["b"], timing=t2, object_sizes=[20])
        )

        async def _drive() -> None:
            collector = asyncio.create_task(_collect_continuous(output_q, stop, results, lock))
            for _ in range(100):
                await asyncio.sleep(0.05)
                if "t1" in results and "t2" in results:
                    break
            stop.set()
            collector.cancel()
            try:
                await collector
            except asyncio.CancelledError:
                pass

        runner = asyncio.Runner()
        try:
            runner.run(_drive())
        finally:
            runner.close()

        assert "t1" in results
        assert "t2" in results

        # First task keeps original start time (100.0).
        assert results["t1"]["metadata"].timing.process_start_time_s == 100.0

        # Second task's start time should be set to first task's completion
        # time (inter-completion-interval), NOT 200.0 (its original value).
        t2_start = results["t2"]["metadata"].timing.process_start_time_s
        t1_end = results["t1"]["metadata"].timing.process_end_time_s
        assert t2_start == t1_end, (
            f"Second task's process_start should equal first task's process_end "
            f"(inter-completion interval), got start={t2_start}, expected={t1_end}"
        )

        # The interval should be small (both processed near-instantly).
        interval = results["t2"]["metadata"].timing.process_end_time_s - t2_start
        assert interval < 1.0, f"Inter-completion interval too large: {interval}s"


class TestWatchStopFlag:
    """Verify stop_flag -> stop_event bridge."""

    def test_bridges_threading_event_to_asyncio(self) -> None:
        """Setting stop_flag (threading.Event) should set stop_event (asyncio.Event)."""
        stop_flag = threading.Event()
        stop_event = asyncio.Event()

        async def _watch(sf: threading.Event, se: asyncio.Event) -> None:
            while not sf.is_set():
                await asyncio.sleep(0.05)
            se.set()

        async def _drive() -> None:
            watcher = asyncio.create_task(_watch(stop_flag, stop_event))
            assert not stop_event.is_set()
            stop_flag.set()
            await asyncio.wait_for(watcher, timeout=2.0)
            assert stop_event.is_set()

        runner = asyncio.Runner()
        try:
            runner.run(_drive())
        finally:
            runner.close()


class TestContinuousInterfaceDetection:
    """Verify the isinstance check that StageWorker uses to switch modes."""

    def test_continuous_interface_detected(self) -> None:
        """A ContinuousInterface subclass should pass isinstance check."""

        class MyStage(ContinuousInterface):
            async def run_continuous(self, input_queue, output_queue, stop_event):
                pass

        assert isinstance(MyStage(), ContinuousInterface)

    def test_non_continuous_not_detected(self) -> None:
        """A plain object should not pass isinstance check."""
        assert not isinstance(MagicMock(), ContinuousInterface)

    def test_wrapped_stage_detected(self) -> None:
        """ContinuousWrappedStage should pass isinstance check."""
        from cosmos_xenna.pipelines.private.continuous_wrapped_stage import ContinuousWrappedStage

        wrapped = ContinuousWrappedStage(MagicMock())
        assert isinstance(wrapped, ContinuousInterface)
