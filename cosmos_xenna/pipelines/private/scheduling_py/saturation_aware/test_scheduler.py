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

"""Integration tests for SaturationAwareScheduler against the real solver.

These exercise the wiring (sizing -> solve -> protect -> mutate-and-return)
through the native fragmentation solver. The pure control-law math lives in
the chain/floor/grace/estimator unit tests.
"""

from typing import cast

import pytest

import cosmos_xenna.pipelines.v1 as v1
from cosmos_xenna.pipelines.private import allocator, data_structures, resources, streaming
from cosmos_xenna.pipelines.private.autoscaling_algorithms import FragmentationBasedAutoscaler
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.scheduler import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SchedulerKind, StageSpec, StreamingSpecificSpec


class _CpuStage(v1.Stage):
    """Minimal CPU pipeline stage with a fixed per-task duration."""

    def __init__(self, cpus: float, throughput: float) -> None:
        self._cpus = cpus
        self._throughput = throughput

    @property
    def process_duration(self) -> float:
        return 1.0 / self._throughput

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> v1.Resources:
        return v1.Resources(cpus=self._cpus, gpus=0)

    def setup(self, worker_metadata: object) -> None:
        pass

    def process_data(self, task: list[float]) -> list[float]:
        return task


def _build(
    cpus_per_stage: list[float], *, num_cpus: int
) -> tuple[v1.PipelineSpec, resources.ClusterResources, data_structures.Problem]:
    spec = v1.PipelineSpec(
        input_data=range(100),
        stages=[v1.StageSpec(_CpuStage(cpus, 1.0)) for cpus in cpus_per_stage],
    )
    cluster = resources.ClusterResources(
        nodes={
            "n0": resources.NodeResources(used_cpus=0, total_cpus=num_cpus, gpus=[], name="n0"),
        }
    )
    problem = streaming._make_problem_from_pipeline_spec(spec, cluster)
    return spec, cluster, problem


def _scheduler(spec: v1.PipelineSpec, config: SaturationAwareConfig | None = None) -> SaturationAwareScheduler:
    stages = cast(list[StageSpec], spec.stages)
    names = tuple(stage_spec.name(index) for index, stage_spec in enumerate(stages))
    batch_sizes = tuple(stage_spec.stage.stage_batch_size for stage_spec in stages)
    is_gpu = tuple(stage_spec.stage.required_resources.gpus > 0 for stage_spec in stages)
    return SaturationAwareScheduler(
        config=config or SaturationAwareConfig(),
        stage_names=names,
        stage_batch_sizes=batch_sizes,
        stage_is_gpu=is_gpu,
    )


def _state(spec: v1.PipelineSpec, worker_allocator: allocator.WorkerAllocator) -> data_structures.ProblemState:
    stage_states = []
    for index, stage_spec in enumerate(cast(list[StageSpec], spec.stages)):
        name = stage_spec.name(index)
        workers = worker_allocator.get_workers_in_stage(name)
        stage_states.append(
            data_structures.ProblemStageState(
                stage_name=name,
                workers=[streaming.make_problem_worker_state_from_worker_state(w) for w in workers],
                slots_per_worker=2,
                is_finished=False,
            )
        )
    return data_structures.ProblemState(stage_states)


def _measurements(now: float, durations: list[float]) -> data_structures.Measurements:
    return data_structures.Measurements(
        now,
        [
            data_structures.StageMeasurements(
                [data_structures.TaskMeasurement(start_time=now - duration, end_time=now, num_returns=1)]
            )
            for duration in durations
        ],
    )


def test_cold_start_sizes_every_stage_and_deletes_nothing() -> None:
    spec, cluster, problem = _build([0.25, 1.0], num_cpus=16)
    scheduler = _scheduler(spec)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert all(len(stage.new_workers) >= 1 for stage in solution.stages)
    assert all(len(stage.deleted_workers) == 0 for stage in solution.stages)


def test_autoscale_before_setup_raises() -> None:
    spec, cluster, _ = _build([0.25], num_cpus=8)
    scheduler = _scheduler(spec)
    with pytest.raises(RuntimeError):
        scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))


def test_backlog_deflation_grows_the_slower_stage() -> None:
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, SaturationAwareConfig(speed_estimation_min_data_points=1))
    scheduler.setup(problem)
    now = 100.0
    scheduler.update_with_measurements(now, _measurements(now, [0.1, 5.0]))
    solution = scheduler.autoscale(now, _state(spec, allocator.WorkerAllocator.make(cluster)))
    new_counts = [len(stage.new_workers) for stage in solution.stages]
    assert new_counts[1] > new_counts[0]


def test_factory_defaults_to_fragmentation_based() -> None:
    spec, cluster, _ = _build([0.25, 1.0], num_cpus=16)
    algorithm = streaming._make_scheduler_algorithm(spec, cluster, StreamingSpecificSpec())
    assert isinstance(algorithm, FragmentationBasedAutoscaler)


def test_factory_selects_saturation_aware_by_kind() -> None:
    spec, cluster, _ = _build([0.25, 1.0], num_cpus=16)
    mode = StreamingSpecificSpec(scheduler=SchedulerKind.SATURATION_AWARE)
    algorithm = streaming._make_scheduler_algorithm(spec, cluster, mode)
    assert isinstance(algorithm, SaturationAwareScheduler)


def test_interval_defaults_to_fragmentation_cadence() -> None:
    mode = StreamingSpecificSpec()
    assert streaming._effective_autoscale_interval_s(mode) == mode.autoscale_interval_s


def test_saturation_aware_uses_its_own_interval() -> None:
    mode = StreamingSpecificSpec(
        scheduler=SchedulerKind.SATURATION_AWARE,
        saturation_aware=SaturationAwareConfig(interval_s=7.0),
    )
    assert streaming._effective_autoscale_interval_s(mode) == 7.0


def test_all_queued_measurement_batches_are_drained_on_next_autoscale() -> None:
    # min_data_points=2 means a single un-drained batch leaves the estimator below
    # threshold (speed None -> no deflation); deflation here proves both batches applied.
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, SaturationAwareConfig(speed_estimation_min_data_points=2))
    scheduler.setup(problem)
    now = 100.0
    scheduler.update_with_measurements(now, _measurements(now, [0.1, 5.0]))
    scheduler.update_with_measurements(now, _measurements(now, [0.1, 5.0]))
    solution = scheduler.autoscale(now, _state(spec, allocator.WorkerAllocator.make(cluster)))
    new_counts = [len(stage.new_workers) for stage in solution.stages]
    assert new_counts[1] > new_counts[0]
