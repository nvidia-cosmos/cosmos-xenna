# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    def __init__(self, run_time_s: float):
        self._run_time_s = run_time_s

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


if __name__ == "__main__":
    test_serving()
