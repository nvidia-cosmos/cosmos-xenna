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

import time

import pytest  # noqa: F401

from cosmos_xenna.pipelines import v1 as pipelines_v1


class SimpleStage(pipelines_v1.Stage):
    def __init__(self, should_slow_down: bool):
        self._should_slow_down = bool(should_slow_down)

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=3.0, gpus=0.0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print("setup")
        time.sleep(1)
        self._setup = True
        self._start_time = time.time()

    def process_data(self, in_data: list[int]) -> list[int]:
        """Processes the input data.

        This method must be implemented by subclasses to define specific data processing logic.

        Args:
            in_data (Any): Input data to be processed.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of processing the input data.
        """
        # if not self._setup:
        #     raise ValueError("Stage not set up yet.")
        if self._should_slow_down and time.time() - self._start_time > 60:
            time.sleep(10)
        else:
            time.sleep(1)
        if in_data[0] == 1:
            return []
        return [x * 2 for x in in_data]


def test_empty_return() -> None:
    stages = [
        SimpleStage(True),
        SimpleStage(False),
    ]

    spec = pipelines_v1.PipelineSpec(
        list(range(10)),
        stages,
        pipelines_v1.PipelineConfig(logging_interval_s=5, return_last_stage_outputs=True),
    )

    execution_modes = [pipelines_v1.ExecutionMode.STREAMING, pipelines_v1.ExecutionMode.BATCH]

    for mode in execution_modes:
        spec.config.execution_mode = mode
        results = pipelines_v1.run_pipeline(spec)
        assert results is not None
        print(results)
        assert len(results) == 9


if __name__ == "__main__":
    test_empty_return()
