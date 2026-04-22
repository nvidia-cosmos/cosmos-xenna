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

"""Non-slow unit tests for ``specs.make_actor_pool_stage_from_stage_spec``.

Guards the wrapper-selection branch in ``make_actor_pool_stage_from_stage_spec``:
``ContinuousInterface`` stages must be wrapped in ``ContinuousWrappedStage``;
all other stages must be wrapped in plain ``WrappedStage``. The end-to-end
integration test that exercises the same path is marked ``@pytest.mark.slow``
and is excluded from the default presubmit suite via
``addopts = "-m 'not slow'"`` in ``pyproject.toml``, so without these tests a
regression in the wrapper-selection branch would only surface in slow-suite
runs.
"""

import asyncio

from cosmos_xenna.pipelines.private import resources, specs
from cosmos_xenna.pipelines.private.continuous_wrapped_stage import ContinuousWrappedStage
from cosmos_xenna.ray_utils.continuous_stage import (
    ContinuousInterface,
    ContinuousTaskInput,
    ContinuousTaskOutput,
)


class _PlainStage(specs.Stage[int, int]):
    """Plain (non-continuous) CPU stage; the wrapper-selection negative case."""

    @property
    def required_resources(self) -> resources.Resources:
        return resources.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int] | None:
        return [x * 2 for x in in_data]


class _ContinuousStage(specs.Stage[int, int], ContinuousInterface):
    """Continuous CPU stage; the wrapper-selection positive case."""

    @property
    def required_resources(self) -> resources.Resources:
        return resources.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int] | None:  # pragma: no cover
        # Never invoked in this unit test (no pipeline run); kept to satisfy
        # the abstract method contract on ``Stage``.
        raise NotImplementedError("Continuous stage; framework dispatches to run_continuous")

    async def run_continuous(
        self,
        input_queue: asyncio.Queue[ContinuousTaskInput],
        output_queue: asyncio.Queue[ContinuousTaskOutput],
        stop_event: asyncio.Event,
    ) -> None:  # pragma: no cover - never invoked in this unit test
        return


def _make_single_cpu_cluster() -> resources.ClusterResources:
    """Build a minimal CPU-only single-node cluster sufficient for ``to_worker_shape``."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(
                used_cpus=0,
                total_cpus=4,
                gpus=[],
                name="node-0",
            ),
        },
    )


def _resolved_spec_for(stage: specs.Stage) -> specs.StageSpec:
    """Build a fully resolved ``StageSpec`` ready for the dispatch factory."""
    spec = specs.StageSpec(stage=stage)
    return spec.override_with_pipeline_params(specs.PipelineConfig())


class TestMakeActorPoolStageFromStageSpecWrapperSelection:
    """Verify the ``ContinuousInterface``-vs-plain wrapper dispatch decision.

    These tests pin the branch at ``specs.make_actor_pool_stage_from_stage_spec``
    that selects ``ContinuousWrappedStage`` vs ``WrappedStage`` based on
    ``isinstance(spec.stage, ContinuousInterface)``.
    """

    def test_continuous_stage_is_wrapped_in_continuous_wrapped_stage(self) -> None:
        """A stage implementing ``ContinuousInterface`` must produce a ``ContinuousWrappedStage``."""
        result = specs.make_actor_pool_stage_from_stage_spec(
            pipeline_config=specs.PipelineConfig(),
            spec=_resolved_spec_for(_ContinuousStage()),
            stage_idx=0,
            cluster_resources=_make_single_cpu_cluster(),
        )

        # Exact-type assertion (not isinstance) so a future common base class
        # cannot mask an inverted-isinstance regression in the dispatcher.
        assert type(result.stage) is ContinuousWrappedStage

    def test_plain_stage_is_wrapped_in_plain_wrapped_stage(self) -> None:
        """A stage NOT implementing ``ContinuousInterface`` must produce a plain ``WrappedStage``."""
        result = specs.make_actor_pool_stage_from_stage_spec(
            pipeline_config=specs.PipelineConfig(),
            spec=_resolved_spec_for(_PlainStage()),
            stage_idx=0,
            cluster_resources=_make_single_cpu_cluster(),
        )

        assert type(result.stage) is specs.WrappedStage
        assert not isinstance(result.stage, ContinuousWrappedStage)
