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

import os
import subprocess
import typing
from typing import Optional

from cosmos_xenna.pipelines import v1 as pipelines_v1


class _CpuStage(pipelines_v1.Stage):
    @property
    def required_resources(self) -> Optional[pipelines_v1.Resources]:
        return pipelines_v1.Resources(cpus=1.0)

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print(worker_metadata)

    def process_data(self, in_data: list[int]) -> list[int]:
        return [x * 2 for x in in_data]


class _NvdecStage(pipelines_v1.Stage):
    @property
    def required_resources(self) -> Optional[pipelines_v1.Resources]:
        return pipelines_v1.Resources(cpus=1.0, nvdecs=1)

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print(worker_metadata)

    def process_data(self, in_data: list[int]) -> list[int]:
        return [x * 2 for x in in_data]


class _NvencStage(pipelines_v1.Stage):
    @property
    def required_resources(self) -> Optional[pipelines_v1.Resources]:
        return pipelines_v1.Resources(cpus=1.0, nvencs=1)

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print(worker_metadata)

    def process_data(self, in_data: list[int]) -> list[int]:
        return [x * 2 for x in in_data]


class _FractionalGpuStage(pipelines_v1.Stage):
    @property
    def required_resources(self) -> Optional[pipelines_v1.Resources]:
        return pipelines_v1.Resources(cpus=0.1, gpus=0.1, nvdecs=1)

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        print(worker_metadata)

    def process_data(self, in_data: list[int]) -> list[int]:
        return [x * 2 for x in in_data]


def test_pipeline() -> None:
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=range(1000),
        stages=[_CpuStage(), _NvdecStage(), _NvencStage(), _FractionalGpuStage()],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            logging_interval_s=1,
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
        ),
    )
    results = typing.cast(list[int], pipelines_v1.run_pipeline(pipeline_spec))
    assert sorted(results) == [x * 16 for x in range(1000)]


class _FractionGpuCudaPrinter(pipelines_v1.Stage):
    @property
    def required_resources(self) -> Optional[pipelines_v1.Resources]:
        return pipelines_v1.Resources(cpus=0.1, gpus=0.1, nvdecs=1)

    @property
    def stage_batch_size(self) -> int:
        return 1

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._worker_metadata = worker_metadata
        subprocess.check_call("printenv")

    def process_data(self, in_data: list[int]) -> list[tuple[pipelines_v1.WorkerMetadata, Optional[str]]]:
        subprocess.check_call("printenv")
        return [(self._worker_metadata, self._cuda_visible_devices)]


def test_cuda_env_vars_are_set_correctly() -> None:
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=range(1),
        stages=[_FractionGpuCudaPrinter()],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            logging_interval_s=1,
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
        ),
    )
    results = typing.cast(
        list[tuple[pipelines_v1.WorkerMetadata, Optional[str]]], pipelines_v1.run_pipeline(pipeline_spec)
    )
    assert len(results) == 1
    assert results[0][1] == "0"
    assert len(results[0][0].allocation.gpus) == 1
