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
the chain/floor/activity/estimator unit tests.
"""

import concurrent.futures
import uuid
from typing import Any, cast
from unittest.mock import MagicMock

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


class _GpuStage(v1.Stage):
    """Minimal GPU pipeline stage with a configurable per-worker GPU fraction."""

    def __init__(self, gpus: float) -> None:
        self._gpus = gpus

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> v1.Resources:
        return v1.Resources(cpus=1.0, gpus=self._gpus)

    def setup(self, worker_metadata: object) -> None:
        pass

    def process_data(self, task: list[float]) -> list[float]:
        return task


def _gpu_cluster(num_gpus: int) -> resources.ClusterResources:
    return resources.ClusterResources(
        nodes={
            "n0": resources.NodeResources(
                used_cpus=0,
                total_cpus=64,
                gpus=[resources.GpuResources(index=i, uuid_=uuid.uuid4(), used_fraction=0.0) for i in range(num_gpus)],
                name="n0",
            )
        }
    )


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
    gpu_fractions = tuple(float(stage_spec.stage.required_resources.gpus) for stage_spec in stages)
    return SaturationAwareScheduler(
        config=config or SaturationAwareConfig(),
        stage_names=names,
        stage_batch_sizes=batch_sizes,
        stage_gpu_fractions=gpu_fractions,
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


class _NoopExecutor:
    """Executor stub that records submissions without running them."""

    def __init__(self) -> None:
        self.submitted: tuple[object, ...] | None = None

    def submit(self, *args: object) -> concurrent.futures.Future[object]:
        self.submitted = args
        return concurrent.futures.Future()


def _pool_for_activity(name: str, *, queued_tasks: int, used_slots: int, batch_size: int) -> MagicMock:
    pool = MagicMock(name=f"pool-{name}")
    pool.name = name
    pool.slots_per_actor = 1
    pool.num_queued_tasks = queued_tasks
    pool.num_used_slots = used_slots
    pool.stage_batch_size = batch_size
    return pool


def test_cold_start_sizes_every_stage_and_deletes_nothing() -> None:
    spec, cluster, problem = _build([0.25, 1.0], num_cpus=16)
    scheduler = _scheduler(spec)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert all(len(stage.new_workers) >= 1 for stage in solution.stages)
    assert all(len(stage.deleted_workers) == 0 for stage in solution.stages)


def test_cold_start_ramp_caps_fractional_gpu_stage() -> None:
    """A fractional-GPU stage with no measurements is held to a single new worker.

    Without the ramp the solver fills the idle cluster with sub-GPU workers
    (the fragmentation bug). The ramp caps the cold stage at one worker.
    """
    spec = v1.PipelineSpec(input_data=range(100), stages=[v1.StageSpec(_GpuStage(0.25))])
    cluster = _gpu_cluster(4)
    problem = streaming._make_problem_from_pipeline_spec(spec, cluster)
    scheduler = _scheduler(spec)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 1


def test_cold_start_ramp_caps_whole_gpu_stage() -> None:
    """A whole-GPU stage with no measurements is held to a single new worker."""
    spec = v1.PipelineSpec(input_data=range(100), stages=[v1.StageSpec(_GpuStage(1.0))])
    cluster = _gpu_cluster(4)
    problem = streaming._make_problem_from_pipeline_spec(spec, cluster)
    scheduler = _scheduler(spec)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 1


def test_cold_start_ramp_caps_cpu_stage() -> None:
    """A CPU stage with no measurements is held to a single new worker."""
    spec, cluster, problem = _build([1.0], num_cpus=16)
    scheduler = _scheduler(spec)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 1


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


def test_active_stock_holds_floor_when_queue_only_stock_is_empty() -> None:
    scheduler = SaturationAwareScheduler(
        config=SaturationAwareConfig(scale_down_release_cycles=3),
        stage_names=("upstream", "caption"),
        stage_batch_sizes=(1, 1),
        stage_gpu_fractions=(0.0, 1.0),
    )
    scheduler.set_queue_snapshot([0.0, 0.0])
    scheduler.set_activity_snapshot(
        queue_depths=[0.0, 0.0],
        pool_queued_tasks=[15, 0],
        inflight_slots=[0, 0],
        batch_sizes=[1, 1],
    )

    result = scheduler._floors(workers=[10, 15], speeds=[0.1, 0.5], returns=[8.0, 1.0])

    assert result.floors[1] == 15
    assert result.queued_stock == (0.0, 0.0)
    assert result.active_stock == (15.0, 15.0)
    assert result.active_depths == (15.0, 0.0)


def test_missing_activity_snapshot_falls_back_to_queue_depths() -> None:
    scheduler = SaturationAwareScheduler(
        config=SaturationAwareConfig(scale_down_release_cycles=3),
        stage_names=("upstream", "caption"),
        stage_batch_sizes=(1, 1),
        stage_gpu_fractions=(0.0, 1.0),
    )
    scheduler.set_queue_snapshot([15.0, 0.0])

    result = scheduler._floors(workers=[10, 15], speeds=[0.1, 0.5], returns=[8.0, 1.0])

    assert result.floors[1] == 15
    assert result.queued_stock == result.active_stock
    assert result.active_depths == (15.0, 0.0)


def test_queue_snapshot_refresh_clears_stale_activity_snapshot() -> None:
    scheduler = SaturationAwareScheduler(
        config=SaturationAwareConfig(scale_down_release_cycles=3),
        stage_names=("upstream", "caption"),
        stage_batch_sizes=(1, 1),
        stage_gpu_fractions=(0.0, 1.0),
    )
    scheduler.set_queue_snapshot([0.0, 0.0])
    scheduler.set_activity_snapshot(
        queue_depths=[0.0, 0.0],
        pool_queued_tasks=[15, 0],
        inflight_slots=[0, 0],
        batch_sizes=[1, 1],
    )
    scheduler.set_queue_snapshot([2.0, 0.0])

    result = scheduler._floors(workers=[10, 15], speeds=[0.1, 0.5], returns=[8.0, 1.0])

    assert result.active_depths == (2.0, 0.0)
    assert result.active_stock == result.queued_stock


def test_streaming_passes_pool_counters_into_saturation_activity_snapshot() -> None:
    spec, cluster, _ = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec)
    executor = _NoopExecutor()
    autoscaler = cast(Any, object.__new__(streaming.Autoscaler))
    autoscaler._autoscale_future = None
    autoscaler._autoscale_start_time = 0.0
    autoscaler._algorithm = scheduler
    autoscaler._executor = executor
    autoscaler._allocator = MagicMock()
    autoscaler._allocator.get_workers_in_stage.return_value = []
    autoscaler._verbosity_level = 0

    pools = [
        _pool_for_activity("stage-0", queued_tasks=2, used_slots=3, batch_size=4),
        _pool_for_activity("stage-1", queued_tasks=5, used_slots=7, batch_size=8),
    ]

    autoscaler.start_autoscale_calculation(
        pools=pools,
        stages_is_dones=[False, False],
        upstream_queue_lens=[11, 13],
    )

    assert scheduler._activity_snapshot is not None
    assert scheduler._activity_snapshot.active_depths() == (31.0, 109.0)
    assert executor.submitted is not None
