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


import multiprocessing
import queue
import threading
import time

import pytest  # noqa: F401
import ray

from cosmos_xenna.pipelines import v1 as pipelines_v1


class Stage(pipelines_v1.Stage):
    def __init__(self, run_time_s: float, batch_size: int = 1):
        self._run_time_s = run_time_s
        self._batch_size = batch_size

    @property
    def stage_batch_size(self) -> int:
        return self._batch_size

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int]:
        time.sleep(self._run_time_s)
        return [x * 2 for x in in_data]


def _start_serving_pipeline(pipeline_spec: pipelines_v1.PipelineSpec) -> None:
    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    except Exception as e:
        raise e
    finally:
        ray.shutdown()


def test_serving() -> None:
    input_queue = multiprocessing.Queue(maxsize=4)
    output_queue = multiprocessing.Queue(maxsize=2)
    serving_queues = pipelines_v1.ServingQueues(source=input_queue, sink=output_queue)

    stages = [
        Stage(0.1),
        Stage(0.4),
        Stage(0.2),
    ]

    spec = pipelines_v1.PipelineSpec(
        [],
        stages,
        pipelines_v1.PipelineConfig(execution_mode=pipelines_v1.ExecutionMode.SERVING),
        serving_queues=serving_queues,
    )

    try:
        # launch the pipeline in a separate thread/process
        pipeline_thread = threading.Thread(target=_start_serving_pipeline, args=(spec,))
        pipeline_thread.start()

        # send one request to the pipeline
        input_queue.put(1)
        output_result = output_queue.get(block=True, timeout=10)
        assert output_result == 8

        # send bursty requests to test back-pressure
        num_requests = 10
        for i in range(num_requests):
            input_queue.put(i)
        time.sleep(1)
        results = []
        while True and len(results) < num_requests:
            try:
                results.append(output_queue.get(block=True, timeout=10))
            except queue.Empty:
                break
        assert len(results) == num_requests
        assert sorted(results) == [x * 8 for x in range(num_requests)]

        # assert that the pipeline is still running
        assert pipeline_thread.is_alive()

    except Exception as e:
        raise e

    finally:
        input_queue.put(None)
        pipeline_thread.join()


class _ServingRun:
    """Runs a SERVING pipeline on a daemon thread and surfaces its exception in the test.

    The thread is a daemon so a pipeline that never returns cannot keep pytest alive after the test has failed.
    """

    def __init__(self, spec: pipelines_v1.PipelineSpec) -> None:
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._run, args=(spec,), daemon=True)

    def _run(self, spec: pipelines_v1.PipelineSpec) -> None:
        try:
            _start_serving_pipeline(spec)
        except BaseException as e:  # noqa: BLE001 - re-raised in the test thread by collect_until_stopped
            self._errors.append(e)

    def start(self) -> "_ServingRun":
        self._thread.start()
        return self

    def collect_until_stopped(self, sink: multiprocessing.Queue, timeout_s: float) -> list[int]:
        """Read the sink until ``run_pipeline`` returns, then return everything that was pushed.

        Reading while waiting matters: with ``serving_drain_on_close`` the pipeline only returns once every result has
        been pushed, and a full sink blocks that push.
        """
        deadline = time.monotonic() + timeout_s
        results: list[int] = []
        while self._thread.is_alive() and time.monotonic() < deadline:
            try:
                results.append(sink.get(timeout=0.5))
            except queue.Empty:
                pass
        if self._errors:
            raise self._errors[0]
        assert not self._thread.is_alive(), f"run_pipeline did not return within {timeout_s}s"
        # Results pushed just before the thread exited may still be in flight through the queue's feeder thread.
        while True:
            try:
                results.append(sink.get(timeout=1))
            except queue.Empty:
                return results


def _serving_spec(
    stages: list[Stage],
    source: multiprocessing.Queue,
    sink: multiprocessing.Queue,
    *,
    drain_on_close: bool,
    drain_timeout_s: float | None = None,
) -> pipelines_v1.PipelineSpec:
    return pipelines_v1.PipelineSpec(
        [],
        stages,
        pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.SERVING,
            serving_drain_on_close=drain_on_close,
            serving_drain_timeout_s=drain_timeout_s,
        ),
        serving_queues=pipelines_v1.ServingQueues(source=source, sink=sink),
    )


def _read_results(sink: multiprocessing.Queue, num_results: int, timeout_s: float) -> list[int]:
    results = []
    while len(results) < num_results:
        results.append(sink.get(block=True, timeout=timeout_s))
    return results


def test_serving_drain_flushes_partial_batches_on_close() -> None:
    """A batched stage in the middle of the pipeline gets its remainder once the source closes."""
    source, sink = multiprocessing.Queue(), multiprocessing.Queue()
    # While the source is open, the batch-4 stage only takes full batches, so 2 of the 6 requests wait in front of it.
    stages = [Stage(0.0), Stage(0.0, batch_size=4), Stage(0.0)]
    run = _ServingRun(_serving_spec(stages, source, sink, drain_on_close=True)).start()

    num_requests = 6
    for i in range(num_requests):
        source.put(i)
    try:
        results = _read_results(sink, 4, timeout_s=60)
    finally:
        source.put(None)
    results += run.collect_until_stopped(sink, timeout_s=60)

    assert sorted(results) == [x * 8 for x in range(num_requests)]


def test_serving_drain_waits_for_in_flight_tasks() -> None:
    """Requests read before None are finished even when None arrives before any worker is ready."""
    source, sink = multiprocessing.Queue(), multiprocessing.Queue()
    num_requests = 3
    for i in range(num_requests):
        source.put(i)
    source.put(None)

    run = _ServingRun(_serving_spec([Stage(1.0), Stage(0.1)], source, sink, drain_on_close=True)).start()
    results = run.collect_until_stopped(sink, timeout_s=120)

    assert sorted(results) == [x * 4 for x in range(num_requests)]


def test_serving_drain_timeout_stops_with_work_in_flight() -> None:
    """When serving_drain_timeout_s expires, the pipeline stops instead of waiting for slow tasks."""
    source, sink = multiprocessing.Queue(), multiprocessing.Queue()
    source.put(0)
    source.put(None)

    # The task would take 10 minutes. Returning within the 120 s collection deadline shows the drain gave up after
    # 2 s and the pools killed the running task.
    spec = _serving_spec([Stage(600.0)], source, sink, drain_on_close=True, drain_timeout_s=2.0)
    results = _ServingRun(spec).start().collect_until_stopped(sink, timeout_s=120)

    assert results == []


def test_serving_close_without_drain_still_stops_immediately() -> None:
    """The default is unchanged: None stops the pipeline and the partial batch in front of a batched stage is dropped.

    This pins the backwards-compatible behaviour that existing SERVING users rely on; draining is opt-in.
    """
    source, sink = multiprocessing.Queue(), multiprocessing.Queue()
    stages = [Stage(0.0), Stage(0.0, batch_size=4)]
    run = _ServingRun(_serving_spec(stages, source, sink, drain_on_close=False)).start()

    num_requests = 6
    for i in range(num_requests):
        source.put(i)
    try:
        results = _read_results(sink, 4, timeout_s=60)
    finally:
        source.put(None)
    results += run.collect_until_stopped(sink, timeout_s=60)

    # Stage 0 may finish requests out of order, so which 4 made the first batch is not fixed.
    assert len(results) == 4
    assert set(results) <= {x * 4 for x in range(num_requests)}


if __name__ == "__main__":
    test_serving()
    test_serving_drain_flushes_partial_batches_on_close()
    test_serving_drain_waits_for_in_flight_tasks()
    test_serving_drain_timeout_stops_with_work_in_flight()
    test_serving_close_without_drain_still_stops_immediately()
