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

"""Smoke test for ray which runs a CPU heavy process.

This is useful for testing monitoring tools. They need significant load before they are useful.
"""

import pytest  # noqa: F401

from cosmos_xenna.pipelines import v1 as pipelines_v1


class Stage(pipelines_v1.Stage):
    def __init__(self, stage_batch_size: int, num_returns_per_input: float):
        self._batch_size = stage_batch_size
        self._num_returns_per_input = num_returns_per_input

    @property
    def stage_batch_size(self) -> int:
        return self._batch_size

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def process_data(self, in_data: list[int]) -> list[int]:
        num_out = int(self._num_returns_per_input * len(in_data))
        return list(range(num_out))


def test_fanout() -> None:
    stages = [
        Stage(stage_batch_size=1, num_returns_per_input=10),
        Stage(stage_batch_size=7, num_returns_per_input=10),
    ]
    num_inputs = 10
    spec = pipelines_v1.PipelineSpec(
        list(range(num_inputs)),
        stages,
        pipelines_v1.PipelineConfig(logging_interval_s=5, return_last_stage_outputs=True),
    )

    execution_modes = [pipelines_v1.ExecutionMode.STREAMING]

    for mode in execution_modes:
        spec.config.execution_mode = mode
        results = pipelines_v1.run_pipeline(spec)
        assert results is not None
        assert len(results) == 1000


def test_no_fan() -> None:
    stages = [
        Stage(stage_batch_size=10, num_returns_per_input=1),
        Stage(stage_batch_size=10, num_returns_per_input=1),
    ]
    num_inputs = 1000
    spec = pipelines_v1.PipelineSpec(
        list(range(num_inputs)),
        stages,
        pipelines_v1.PipelineConfig(logging_interval_s=5, return_last_stage_outputs=True),
    )

    execution_modes = [pipelines_v1.ExecutionMode.STREAMING]

    for mode in execution_modes:
        spec.config.execution_mode = mode
        results = pipelines_v1.run_pipeline(spec)
        assert results is not None
        print(results)
        assert len(results) == 1000


def test_fanin() -> None:
    stages = [
        Stage(stage_batch_size=10, num_returns_per_input=0.1),
        Stage(stage_batch_size=10, num_returns_per_input=0.1),
    ]
    num_inputs = 1000
    spec = pipelines_v1.PipelineSpec(
        list(range(num_inputs)),
        stages,
        pipelines_v1.PipelineConfig(logging_interval_s=5, return_last_stage_outputs=True),
    )

    execution_modes = [pipelines_v1.ExecutionMode.STREAMING]

    for mode in execution_modes:
        spec.config.execution_mode = mode
        results = pipelines_v1.run_pipeline(spec)
        assert results is not None
        print(results)
        assert len(results) == 10


def main() -> None:
    test_fanout()
    test_no_fan()
    test_fanin()


if __name__ == "__main__":
    main()
