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

"""Smoke test for ray which checks that the CUDA_VISIBLE_DEVICES environment variable is set correctly.

Needs to be run on a machine with a single GPU.
"""

import os

import torch

from cosmos_xenna.pipelines import v1 as pipelines_v1


class GpuStage(pipelines_v1.Stage):
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=1.0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        pass

    def process_data(self, in_data: list[int]) -> list[tuple[str, int]]:
        return [(os.environ.get("CUDA_VISIBLE_DEVICES", "None"), torch.cuda.current_device())]


def test_actor_pool_heavy() -> None:
    stages = [
        GpuStage(),
    ]

    spec = pipelines_v1.PipelineSpec(
        list(range(1)),
        stages,
        pipelines_v1.PipelineConfig(
            return_last_stage_outputs=True,
        ),
    )

    execution_modes = [pipelines_v1.ExecutionMode.STREAMING, pipelines_v1.ExecutionMode.BATCH]

    gpu_infos = pipelines_v1.get_local_gpu_info()
    assert len(gpu_infos) == 1
    for mode in execution_modes:
        spec.config.execution_mode = mode
        results = pipelines_v1.run_pipeline(spec)
        assert results is not None
        assert len(results) == 1
        assert results[0][0] == f"GPU-{gpu_infos[0].uuid_}"
        assert results[0][1] == torch.cuda.current_device()


if __name__ == "__main__":
    test_actor_pool_heavy()
