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

import pytest  # noqa: F401

from cosmos_xenna.pipelines import v1 as pipelines_v1


class Stage(pipelines_v1.Stage):
    def __init__(self, batch_size: int):
        self._batch_size = batch_size

    @property
    def stage_batch_size(self) -> int:
        return self._batch_size

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=2.0, gpus=0.0)

    def process_data(self, in_data: list[int]) -> list[int]:
        assert len(in_data) == self._batch_size
        return [x * 2 for x in in_data]


def test_batching() -> None:
    stages = [
        Stage(10),
        Stage(5),
        Stage(10),
    ]

    spec = pipelines_v1.PipelineSpec(
        list(range(20)),
        stages,
        pipelines_v1.PipelineConfig(logging_interval_s=5, return_last_stage_outputs=True),
    )

    execution_modes = [pipelines_v1.ExecutionMode.STREAMING, pipelines_v1.ExecutionMode.BATCH]

    for mode in execution_modes:
        spec.config.execution_mode = mode
        results = pipelines_v1.run_pipeline(spec)
        assert results is not None
        print(results)
        assert len(results) == 20
        assert sorted(results) == [x * 8 for x in range(20)]


if __name__ == "__main__":
    test_batching()
