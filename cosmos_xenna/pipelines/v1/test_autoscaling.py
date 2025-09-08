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
Simple ray example which uses ray to download, slightly modify and upload a directory of tars.

See the "Running a multinode Ray job" of pipelines/examples/README.md for more info.
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
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        time.sleep(self._setup_dur)

    def process_data(self, task: list[float]) -> list[float]:
        time.sleep(self._process_dur)
        return [x * 2 for x in task]


@pytest.mark.slow
def test_autoscaling() -> None:
    tasks = range(100)
    # We make a "spec" which tells our code how to run our pipeline. This spec is very simple. It is just a list of
    # objects we want to run over and a single stage to run over the objects.
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(_ProcessStage(1, 0), num_workers_per_node=1),
            _ProcessStage(3, 0.1),
            _ProcessStage(1, 1),
            _ProcessStage(5, 0.3),
            pipelines_v1.StageSpec(_ProcessStage(1, 0), num_workers_per_node=1),
        ],
        config=pipelines_v1.PipelineConfig(
            logging_interval_s=5,
            mode_specific=pipelines_v1.StreamingSpecificSpec(
                autoscale_interval_s=1,
                autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.DEBUG,
            ),
        ),
    )
    # Start the pipeline. If we run this locally, it will start a local ray cluster and submit our job. If we run it
    # with "uv run yotta launch --mode=ngc-ray", it will connect to the existing cluster and submit our job.
    pipelines_v1.run_pipeline(pipeline_spec)


if __name__ == "__main__":
    test_autoscaling()
