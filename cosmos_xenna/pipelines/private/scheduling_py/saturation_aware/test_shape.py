# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for PipelineShape.from_stage_specs derivation of is_gpu/is_manual."""

import cosmos_xenna.pipelines.v1 as v1
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.shape import PipelineShape


class _Stage(v1.Stage):
    """Minimal stage stub parameterized by GPU fraction and batch size."""

    def __init__(self, *, gpus: float = 0.0, batch_size: int = 1) -> None:
        self._gpus = gpus
        self._batch_size = batch_size

    @property
    def stage_batch_size(self) -> int:
        return self._batch_size

    @property
    def required_resources(self) -> v1.Resources:
        return v1.Resources(cpus=1.0, gpus=self._gpus)

    def setup(self, worker_metadata: object) -> None:
        pass

    def process_data(self, task: list[float]) -> list[float]:
        return task


def test_from_stage_specs_marks_gpu_stage_as_gpu() -> None:
    """A stage requesting a GPU fraction is flagged is_gpu."""
    shape = PipelineShape.from_stage_specs([v1.StageSpec(_Stage(gpus=0.5))])
    assert shape.stages[0].is_gpu is True


def test_from_stage_specs_marks_cpu_stage_as_not_gpu() -> None:
    """A stage with no GPU request is not flagged is_gpu."""
    shape = PipelineShape.from_stage_specs([v1.StageSpec(_Stage(gpus=0.0))])
    assert shape.stages[0].is_gpu is False


def test_from_stage_specs_marks_num_workers_as_manual() -> None:
    """A pinned num_workers count marks the stage operator-managed."""
    shape = PipelineShape.from_stage_specs([v1.StageSpec(_Stage(), num_workers=4)])
    assert shape.stages[0].is_manual is True


def test_from_stage_specs_marks_num_workers_per_node_as_manual() -> None:
    """A pinned per-node worker count also marks the stage operator-managed."""
    shape = PipelineShape.from_stage_specs([v1.StageSpec(_Stage(), num_workers_per_node=2.0)])
    assert shape.stages[0].is_manual is True


def test_from_stage_specs_autoscaled_stage_is_not_manual() -> None:
    """A stage with neither pin set is left for the autoscaler (not manual)."""
    shape = PipelineShape.from_stage_specs([v1.StageSpec(_Stage())])
    assert shape.stages[0].is_manual is False


def test_from_stage_specs_preserves_batch_size_and_order() -> None:
    """Batch size passes through and stages keep their input order."""
    shape = PipelineShape.from_stage_specs([v1.StageSpec(_Stage(batch_size=8)), v1.StageSpec(_Stage(batch_size=3))])
    assert shape.num_stages == 2
    assert (shape.stages[0].batch_size, shape.stages[1].batch_size) == (8, 3)
