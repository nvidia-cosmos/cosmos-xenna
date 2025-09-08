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

import pytest  # noqa: F401

import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna.pipelines.private import resources


class _ProcessStage(pipelines_v1.Stage):
    def __init__(self, return_empty: bool) -> None:
        self._return_empty = return_empty

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        pass

    def process_data(self, samples: list[float]) -> list[float]:
        if self._return_empty:
            return []
        return [x * 2 for x in samples]


def test_empty_returns() -> None:
    tasks = range(1000)
    # We make a "spec" which tells our code how to run our pipeline. This spec is very simple. It is just a list of
    # objects we want to run over and a single stage to run over the objects.
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(_ProcessStage(return_empty=False)),
            pipelines_v1.StageSpec(_ProcessStage(return_empty=True)),
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
    test_empty_returns()
