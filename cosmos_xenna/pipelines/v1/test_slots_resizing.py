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

import pytest

from cosmos_xenna.pipelines import v1 as pipelines_v1


class _ProcessStage(pipelines_v1.Stage):
    def __init__(self, setup_dur: float, process_dur: float) -> None:
        self._setup_dur = float(setup_dur)
        self._process_dur = float(process_dur)

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=0.5, gpus=0.0)

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        time.sleep(self._setup_dur)

    def process_data(self, task: list[float]) -> list[float]:
        time.sleep(self._process_dur)
        return [x * 2 for x in task]


@pytest.mark.slow
def test_slots_resizing() -> None:
    tasks = range(500)
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            # This will be limited to one actor. We should need to create many slots to keep it busy
            pipelines_v1.StageSpec(_ProcessStage(0.0, 0.0), num_workers_per_node=1),
            pipelines_v1.StageSpec(_ProcessStage(0.0, 0.01), num_workers_per_node=10),
        ],
        config=pipelines_v1.PipelineConfig(
            logging_interval_s=5,
            mode_specific=pipelines_v1.StreamingSpecificSpec(autoscale_interval_s=1),
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
        ),
    )
    pipelines_v1.run_pipeline(pipeline_spec)


if __name__ == "__main__":
    test_slots_resizing()
