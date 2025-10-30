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


class Stage(pipelines_v1.Stage):
    def __init__(self):
        self._setup_on_node_result = False

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=2.0, gpus=0.0)

    def setup_on_node(self, node_info: pipelines_v1.NodeInfo, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._setup_on_node_result = True
        time.sleep(0.1)

    def process_data(self, in_data: list[int]) -> list[int]:
        if self._setup_on_node_result:
            self._setup_on_node_result = False
            return [in_data[0] + 1]
        else:
            return [in_data[0]]


def test_batching() -> None:
    stages = [
        Stage(),
        Stage(),
        Stage(),
    ]

    spec = pipelines_v1.PipelineSpec(
        list([0 for _ in range(20)]),
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
        assert sum(results) == 3  # setup_on_node should have been called exactly three times as we only have one node.


if __name__ == "__main__":
    test_batching()
