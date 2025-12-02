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

"""
Simple example which shows the poor allocation when there are limited tasks.

The issue is that when there are limited tasks, and when actors take longer to set up than other actors,
the actor pool will allocate multiple tasks to the already-setup actors, and then will run out of tasks
for new actors.

As a result, actors will be idle even through there is work to do.

Maybe we should do some sort of work-stealing?
"""

import time

import pytest

import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna.pipelines.private import resources


class _ProcessStage(pipelines_v1.Stage):
    def __init__(self, setup_dur: float, process_dur: float) -> None:
        self._setup_dur = float(setup_dur)
        self._process_dur = float(process_dur)

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=0.5, gpus=0.0)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        time.sleep(self._setup_dur)

    def process_data(self, task: list[float]) -> list[float]:
        time.sleep(self._process_dur)
        return [x * 2 for x in task]


@pytest.mark.slow
def test_allocating_with_limited_tasks() -> None:
    tasks = range(12)
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(
                _ProcessStage(setup_dur=0, process_dur=10),
                num_workers_per_node=10,
            ),
        ],
        config=pipelines_v1.PipelineConfig(
            logging_interval_s=5,
            mode_specific=pipelines_v1.StreamingSpecificSpec(
                autoscale_interval_s=30,
                autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.DEBUG,
            ),
            enable_work_stealing=True,
        ),
    )
    pipelines_v1.run_pipeline(pipeline_spec)


if __name__ == "__main__":
    test_allocating_with_limited_tasks()
