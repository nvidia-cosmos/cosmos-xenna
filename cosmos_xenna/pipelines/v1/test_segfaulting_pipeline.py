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

import ctypes
import random

import pytest  # noqa: F401

from cosmos_xenna.pipelines import v1 as pipelines_v1


class _SegfaultingStage(pipelines_v1.Stage):
    def __init__(self, process_dur_s: float, setup_failure_likelihood: float):
        self._setup_failure_likelihood = float(setup_failure_likelihood)

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        if random.random() < self._setup_failure_likelihood:
            ctypes.string_at(0)  # Segfault

    def process_data(self, in_data: list[int]) -> list[int]:
        return [x * 2 for x in in_data]


# TODO: Enable this in CI
# def test_raises_setup_failures():
#     pipeline_spec = ray_utils.PipelineSpec(
#         input_data=range(1000),
#         stages=[ray_utils.StageSpec(_SegfaultingStage(0.0, 0.1), num_workers=10)],
#         max_setup_failure_percentage=None,
#     )
#     with pytest.raises(ray.exceptions.ActorDiedError):
#         ray_utils.run_pipeline(pipeline_spec)


# TODO: Enable this in CI
# def test_pipeline_ignores_setup_failures_when_asked_to():
#     pipeline_spec = ray_utils.PipelineSpec(
#         input_data=range(1000),
#         stages=[ray_utils.StageSpec(_SegfaultingStage(0.0, 0.1), num_workers=10)],
#         max_setup_failure_percentage=90,
#     )
#     results = ray_utils.run_pipeline(pipeline_spec)
#     assert sorted(results) == [x * 2 for x in range(1000)]


def test_nominal_streaming_pipeline():
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=range(200),
        stages=[_SegfaultingStage(0.1, 0.0), _SegfaultingStage(0.001, 0.0)],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            return_last_stage_outputs=True,
            logging_interval_s=10,
        ),
    )
    results = pipelines_v1.run_pipeline(pipeline_spec)
    assert results is not None
    assert sorted(results) == [x * 4 for x in range(200)]
