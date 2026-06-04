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

"""Tests for the rebuildable solver problem template."""

from typing import Any

import cosmos_xenna.pipelines.v1 as v1
from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.problem_template import SolverProblemTemplate
from cosmos_xenna.pipelines.private.specs import StageSpec


class _CpuStage(v1.Stage):
    """Minimal CPU stage exposing only what the template reads."""

    @property
    def required_resources(self) -> v1.Resources:
        return v1.Resources(cpus=1.0, gpus=0)

    def setup(self, worker_metadata: object) -> None:
        pass

    def process_data(self, task: list[float]) -> list[float]:
        return task


def _cluster(num_nodes: int) -> resources.ClusterResources:
    return resources.ClusterResources(
        nodes={
            f"n{i}": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name=f"n{i}") for i in range(num_nodes)
        }
    )


def test_from_stage_specs_resolves_pinned_counts() -> None:
    """num_workers passes through; num_workers_per_node scales by node count; neither yields None."""
    specs: list[StageSpec[Any, Any]] = [
        v1.StageSpec(_CpuStage(), num_workers=4),
        v1.StageSpec(_CpuStage(), num_workers_per_node=2.0),
        v1.StageSpec(_CpuStage()),
    ]
    template = SolverProblemTemplate.from_stage_specs(specs, _cluster(2))
    assert [stage.requested_num_workers for stage in template.stages] == [4, 4, None]


def test_build_returns_a_problem_with_and_without_overrides() -> None:
    """build() and build(overrides) each produce a fresh solver problem."""
    specs: list[StageSpec[Any, Any]] = [v1.StageSpec(_CpuStage(), num_workers=4)]
    template = SolverProblemTemplate.from_stage_specs(specs, _cluster(1))
    pinned_name = template.stages[0].name
    assert isinstance(template.build(), data_structures.Problem)
    assert isinstance(template.build({pinned_name: 0}), data_structures.Problem)
