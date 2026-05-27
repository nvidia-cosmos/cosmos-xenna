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


import ctypes
import random

import pytest

from cosmos_xenna.pipelines import v1 as pipelines_v1


class _SegfaultingStage(pipelines_v1.Stage):
    def __init__(
        self,
        process_dur_s: float,
        setup_failure_likelihood: float,
        *,
        process_failure_inputs: frozenset[int] = frozenset(),
    ):
        self._setup_failure_likelihood = float(setup_failure_likelihood)
        # Inputs that should crash this worker the first time they are seen.
        # Used to exercise unexpected-death recovery in ``process_data``.
        self._process_failure_inputs = frozenset(process_failure_inputs)

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
        for value in in_data:
            if value in self._process_failure_inputs:
                # Hard-crash the worker process. Recovery is the pool's job.
                ctypes.string_at(0)
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


@pytest.mark.skip(reason="Might be flaky in CI")
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


@pytest.mark.CPU
def test_streaming_pipeline_survives_single_process_data_segfault() -> None:
    """Fast end-to-end coverage for the unexpected-death catch + replacement path.

    Small enough (4 inputs, one stage) to stay well under the slow-test budget
    and run in the default fast CI lane, so the actor-death recovery code path
    has real integration coverage every test run - not just the in-process
    bookkeeping covered by ``test_actor_pool_death_recovery.py``.

    The stage segfaults on input ``2``. With recovery in place, the pool catches
    the ``RayActorError`` raised inside ``_process_completed_task``, requeues
    the in-flight task, the next ``_adjust_actors`` tick recreates the worker
    group, the poison input cycles until it hits ``_MAX_ACTOR_DEATH_RETRIES``
    and is dropped, and the non-poison inputs survive. The contract is simply
    that ``run_pipeline`` returns at all instead of raising.

    The ``test_streaming_pipeline_survives_process_data_segfault`` test below is
    a larger, slow-marked variant that exercises more concurrent in-flight work
    and a multi-stage layout; this test is the minimal regression guard.
    """
    inputs = [0, 1, 2, 3]
    poison = 2
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=inputs,
        stages=[_SegfaultingStage(0.0, 0.0, process_failure_inputs=frozenset({poison}))],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            return_last_stage_outputs=True,
            logging_interval_s=5,
        ),
    )

    results = pipelines_v1.run_pipeline(pipeline_spec)

    assert results is not None, "run_pipeline must not raise on transient worker death"
    expected = sorted(x * 2 for x in inputs if x != poison)
    actual = sorted(results)
    assert actual == expected, f"recovery output mismatch:\nexpected={expected}\nactual={actual}"


@pytest.mark.slow
def test_streaming_pipeline_survives_process_data_segfault() -> None:
    """A worker crash in ``process_data`` must not kill the driver.

    The first stage segfaults whenever it sees ``13``. With unexpected-death
    recovery in place, the pool re-queues the in-flight task, the next
    ``_adjust_actors`` tick recreates the worker group, and the pipeline
    completes. The poison input itself is retried up to
    ``_MAX_ACTOR_DEATH_RETRIES`` and then dropped (counted, warned), so the
    final output contains every non-poison value doubled twice and excludes
    the poison value entirely. The important assertion is that
    ``run_pipeline`` returns at all instead of raising.

    Marked ``slow`` because Ray-cluster lifecycle and worker restart latency
    make this expensive; intended for the L1 / on-demand tier rather than the
    default fast CI lane. See ``test_streaming_pipeline_survives_single_process_data_segfault``
    above for the minimal default-lane variant of the same coverage.
    """
    inputs = list(range(20))
    poison = 13
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=inputs,
        stages=[
            _SegfaultingStage(0.0, 0.0, process_failure_inputs=frozenset({poison})),
            _SegfaultingStage(0.0, 0.0),
        ],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            return_last_stage_outputs=True,
            logging_interval_s=5,
        ),
    )

    results = pipelines_v1.run_pipeline(pipeline_spec)

    assert results is not None, "run_pipeline must not raise on transient worker death"
    expected = sorted(x * 4 for x in inputs if x != poison)
    actual = sorted(results)
    # Poison input is dropped after the retry cap; every other input survives
    # the round trip through both stages.
    assert actual == expected, f"recovery output mismatch:\nexpected={expected}\nactual={actual}"
