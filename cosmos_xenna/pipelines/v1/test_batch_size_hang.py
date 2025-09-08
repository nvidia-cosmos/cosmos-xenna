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

from typing import Optional

import attrs
import pytest  # noqa: F401

from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.utils import python_log as logger


@attrs.define
class _Sample:
    """A single sample processed by the pipeline."""

    text: str


class FirstFanOutStage(pipelines_v1.Stage):
    """Stage for downloading images from URLs."""

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def process_data(self, samples: list[_Sample]) -> list[_Sample]:
        """Download images from URLs, using a cache."""
        return [_Sample(text="hello") for _ in range(2000)]


class BatchStage(pipelines_v1.Stage):
    @property
    def stage_batch_size(self) -> int:
        return 2000

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def process_data(self, samples: list[_Sample]) -> list[_Sample]:
        """Download images from URLs, using a cache."""
        return samples


class SimpleStage(pipelines_v1.Stage):
    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=5.0, gpus=0.0)

    def process_data(self, samples: list[_Sample]) -> list[_Sample]:
        return samples


def main() -> Optional[list[_Sample]]:
    tasks = [_Sample(text="starting") for i in range(1)]
    # Define the simplified pipeline structure
    simple_stages = [pipelines_v1.StageSpec(SimpleStage()) for _ in range(1)]
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(FirstFanOutStage()),
            *simple_stages,
            pipelines_v1.StageSpec(BatchStage()),
        ],
        config=pipelines_v1.PipelineConfig(
            logging_interval_s=5,
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
        ),
    )

    logger.info("Starting Ray pipeline...")
    logger.info(pipeline_spec)
    results = pipelines_v1.run_pipeline(pipeline_spec)
    logger.info("Pipeline finished.")
    return results


# -----------------------------------------------------------------------------
# PyTest entry point
# -----------------------------------------------------------------------------


def test_batch_size_large_batch_does_not_hang() -> None:
    """
    This test is to verify that a large batch size will not cause the pipeline to hang.
    Due to backpressure, it can happen that a downstream stage is not able to get enough tasks to run with batch size.

    The pipeline is:
    - 1st FanOutStage: batch size 1, process 2000 samples
    - 1 SimpleStages: batch size 1, process 1 sample
    - 1 BatchStage: batch size 2000, process 2000 samples

    The first fan out stage will produce 2000 tasks.
    The 1 simple stages will each produce 1 task.
    The batch stage will process 2000 tasks.

    The pipeline will hang if the batch stage is not able to get enough tasks to run with batch size 2000.

    """
    results = main()

    # Basic sanity checks. we mainly care that the call returns.
    assert isinstance(results, list)
    # The first fan-out stage produces 2000 samples and no stage afterwards
    # changes the sample count, so the final output should match.
    assert len(results) == 2000


if __name__ == "__main__":
    main()
