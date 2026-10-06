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


def _get_results(output_queue: multiprocessing.Queue, num_results: int, timeout_s: float) -> list[int]:
    results = []
    while len(results) < num_results:
        try:
            results.append(output_queue.get(block=True, timeout=timeout_s))
        except queue.Empty:
            break
    return results


def test_serving_flushes_partial_batches_on_close() -> None:
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    serving_queues = pipelines_v1.ServingQueues(source=input_queue, sink=output_queue)

    # The second stage only gets full batches of 4 while the source is open.
    stages = [
        Stage(0.0),
        Stage(0.0, batch_size=4),
    ]

    spec = pipelines_v1.PipelineSpec(
        [],
        stages,
        pipelines_v1.PipelineConfig(execution_mode=pipelines_v1.ExecutionMode.SERVING),
        serving_queues=serving_queues,
    )

    num_requests = 6
    pipeline_thread = threading.Thread(target=_start_serving_pipeline, args=(spec,))
    pipeline_thread.start()
    try:
        for i in range(num_requests):
            input_queue.put(i)
        # the first full batch comes out while the source is still open
        results = _get_results(output_queue, 4, timeout_s=60)
        assert len(results) == 4
    finally:
        # closing the source must hand the second stage its remainder
        input_queue.put(None)
        pipeline_thread.join(timeout=60)

    results += _get_results(output_queue, num_requests - len(results), timeout_s=5)
    assert not pipeline_thread.is_alive()
    assert sorted(results) == [x * 4 for x in range(num_requests)]


def test_serving_close_waits_for_in_flight_tasks() -> None:
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    serving_queues = pipelines_v1.ServingQueues(source=input_queue, sink=output_queue)

    stages = [
        Stage(1.0),
        Stage(0.1),
    ]

    spec = pipelines_v1.PipelineSpec(
        [],
        stages,
        pipelines_v1.PipelineConfig(execution_mode=pipelines_v1.ExecutionMode.SERVING),
        serving_queues=serving_queues,
    )

    # close the source while every request is still queued or running
    num_requests = 3
    for i in range(num_requests):
        input_queue.put(i)
    input_queue.put(None)

    pipeline_thread = threading.Thread(target=_start_serving_pipeline, args=(spec,))
    pipeline_thread.start()
    pipeline_thread.join(timeout=120)

    # every request admitted before None must reach the sink before run_pipeline returns
    results = _get_results(output_queue, num_requests, timeout_s=5)
    assert not pipeline_thread.is_alive()
    assert sorted(results) == [x * 4 for x in range(num_requests)]


if __name__ == "__main__":
    test_serving()
    test_serving_flushes_partial_batches_on_close()
    test_serving_close_waits_for_in_flight_tasks()
