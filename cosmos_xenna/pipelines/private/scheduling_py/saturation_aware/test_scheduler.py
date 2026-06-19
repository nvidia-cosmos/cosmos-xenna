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

These exercise the wiring (capacity -> demand -> solve -> ramp -> floor ->
mutate-and-return) through the native fragmentation solver. The pure control-law
math lives in the capacity/chain/floor/sizing/estimator unit tests.
"""

import logging
import uuid
from collections.abc import Iterator
from typing import Any, cast

import pytest
import ray
from loguru import logger as loguru_logger

import cosmos_xenna.pipelines.v1 as v1
from cosmos_xenna.pipelines.private import allocator, data_structures, resources, streaming
from cosmos_xenna.pipelines.private.autoscaling_algorithms import FragmentationBasedAutoscaler
from cosmos_xenna.pipelines.private.scheduling_py.runtime_signals import RuntimeSignals
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.capacity import CapacityPlan, StageCapacity
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.problem_template import SolverProblemTemplate
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.ramp import RampReason
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.scheduler import SaturationAwareScheduler, _Cycle
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.shape import PipelineShape
from cosmos_xenna.pipelines.private.specs import SchedulerKind, StageSpec, StreamingSpecificSpec

_StageSpecList = list[StageSpec[Any, Any]]


def _mock_object_ref() -> "ray.ObjectRef[Any]":
    """Placeholder ObjectRef for queue length tests (only ``len`` is used).

    The annotation and cast use string forward references because
    ``ray.ObjectRef`` is not subscriptable at runtime on all Ray builds, and
    the project forbids ``from __future__ import annotations``; quoting keeps
    the type for static checkers without evaluating the subscript at import.
    """
    return cast("ray.ObjectRef[Any]", object())


def _stage_specs(spec: v1.PipelineSpec) -> _StageSpecList:
    return cast(_StageSpecList, spec.stages)


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


def _scheduler(
    spec: v1.PipelineSpec,
    cluster: resources.ClusterResources,
    config: SaturationAwareConfig | None = None,
) -> SaturationAwareScheduler:
    stages = _stage_specs(spec)
    return SaturationAwareScheduler(
        config=config or SaturationAwareConfig(),
        shape=PipelineShape.from_stage_specs(stages),
        solver_template=SolverProblemTemplate.from_stage_specs(stages, cluster),
    )


def _state(spec: v1.PipelineSpec, worker_allocator: allocator.WorkerAllocator) -> data_structures.ProblemState:
    stage_states = []
    for index, stage_spec in enumerate(_stage_specs(spec)):
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


def _empty_skip_measurements(now: float) -> data_structures.Measurements:
    """Single-stage measurements with one empty, near-instant skip task.

    Mirrors the production "no clips to transcode" event: a task that produced
    no output (``num_returns=0``) and returned in ~0.1 ms. Such a sample must
    not enter the speed window, or ``1/mean(duration)`` would explode.
    """
    return data_structures.Measurements(
        now,
        [
            data_structures.StageMeasurements(
                [data_structures.TaskMeasurement(start_time=now - 1e-4, end_time=now, num_returns=0)]
            )
        ],
    )


def _upstream_only_measurements(now: float) -> data_structures.Measurements:
    """Two-stage measurements where only stage 0 reports a completed task.

    Holds the downstream stage at zero samples so its cold-ramp warming path can
    be exercised while the upstream stage accrues enough samples to be trusted.
    """
    return data_structures.Measurements(
        now,
        [
            data_structures.StageMeasurements(
                [data_structures.TaskMeasurement(start_time=now - 1.0, end_time=now, num_returns=1)]
            ),
            data_structures.StageMeasurements([]),
        ],
    )


def _stage_names(spec: v1.PipelineSpec) -> list[str]:
    """Return the per-stage names in pipeline order."""
    return [stage_spec.name(index) for index, stage_spec in enumerate(_stage_specs(spec))]


def _backlog(num_stages: int, queue_depth: float) -> RuntimeSignals:
    """Uniform runtime signals: every stage has ``queue_depth`` source items waiting."""
    return RuntimeSignals(
        queue_depths=(queue_depth,) * num_stages,
        pool_queued_tasks=(0,) * num_stages,
        inflight_slots=(0,) * num_stages,
        batch_sizes=(1,) * num_stages,
    )


def _apply_to_allocator(
    spec: v1.PipelineSpec,
    worker_allocator: allocator.WorkerAllocator,
    solution: data_structures.Solution,
) -> None:
    """Commit one autoscale ``Solution`` to the allocator for a multi-cycle simulation.

    The default ``_state`` harness rebuilds from a fresh allocator each call, so
    current workers are always zero and only the cold-start growth path is ever
    exercised. Persisting the solution lets a test drive the real cross-cycle
    control law (capacity EWMA, sticky bottleneck, floor release streaks) and the
    delete path.

    Mirrors the streaming layer's apply order: deleted workers are released
    first so a rebalancing cycle never transiently over-allocates a full
    cluster, then new workers are placed. A post-floor solution can exceed
    cluster capacity for a single tick; additions that do not fit raise
    ``allocator.AllocationError`` and are deferred to the next cycle, exactly as
    ``streaming`` defers unplaceable additions.
    """
    names = _stage_names(spec)
    for stage_solution in solution.stages:
        for worker in stage_solution.deleted_workers:
            worker_allocator.remove_worker(worker.id)
    for index, stage_solution in enumerate(solution.stages):
        name = names[index]
        for worker in stage_solution.new_workers:
            try:
                worker_allocator.add_worker(worker.to_worker_group(name))
            except allocator.AllocationError:
                break  # cluster full this tick; the next autoscale re-proposes


def _worker_counts(spec: v1.PipelineSpec, worker_allocator: allocator.WorkerAllocator) -> list[int]:
    """Return the current per-stage worker counts held by the allocator."""
    return [len(worker_allocator.get_workers_in_stage(name)) for name in _stage_names(spec)]


def test_cold_start_sizes_every_stage_and_deletes_nothing() -> None:
    spec, cluster, problem = _build([0.25, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster)
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
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 1


def test_cold_start_ramp_caps_whole_gpu_stage() -> None:
    """A whole-GPU stage with no measurements is held to a single new worker."""
    spec = v1.PipelineSpec(input_data=range(100), stages=[v1.StageSpec(_GpuStage(1.0))])
    cluster = _gpu_cluster(4)
    problem = streaming._make_problem_from_pipeline_spec(spec, cluster)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 1


def test_cold_start_ramp_caps_cpu_stage() -> None:
    """A CPU stage with no measurements is held to a single new worker."""
    spec, cluster, problem = _build([1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(problem)
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 1


def _runtime_signals(queue_depth: float) -> RuntimeSignals:
    """Single-stage runtime signals with the given upstream queue depth."""
    return RuntimeSignals(queue_depths=(queue_depth,), pool_queued_tasks=(0,), inflight_slots=(0,), batch_sizes=(1,))


def test_no_sample_after_window_with_pending_work_releases_stage_to_solver() -> None:
    """A stage with work waiting but no measurements after a full window spawns past one worker.

    The first decision anchors the warmup clock and caps the cold stage at one
    worker; a later decision past ``speed_estimation_window_s`` with work still
    queued but no measurements treats it as a slow-starter and trusts the solver.
    """
    spec, cluster, problem = _build([1.0], num_cpus=16)
    config = SaturationAwareConfig()
    scheduler = _scheduler(spec, cluster, config)
    scheduler.setup(problem)
    t0 = 100.0
    scheduler.observe_runtime(_runtime_signals(queue_depth=5.0))
    cold = scheduler.autoscale(t0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(cold.stages[0].new_workers) == 1
    later = scheduler.autoscale(
        t0 + config.speed_estimation_window_s + 1.0, _state(spec, allocator.WorkerAllocator.make(cluster))
    )
    assert len(later.stages[0].new_workers) > 1


def test_no_sample_after_window_without_pending_work_stays_capped() -> None:
    """A starved stage (no work waiting) stays at one worker past the window, never over-spawned."""
    spec, cluster, problem = _build([1.0], num_cpus=16)
    config = SaturationAwareConfig()
    scheduler = _scheduler(spec, cluster, config)
    scheduler.setup(problem)
    t0 = 100.0
    scheduler.observe_runtime(_runtime_signals(queue_depth=0.0))
    scheduler.autoscale(t0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    later = scheduler.autoscale(
        t0 + config.speed_estimation_window_s + 1.0, _state(spec, allocator.WorkerAllocator.make(cluster))
    )
    assert len(later.stages[0].new_workers) == 1


def _queue_gradient_starved_downstream_signals() -> RuntimeSignals:
    """Two-stage signals: the upstream has backlog while the downstream input is empty."""
    return RuntimeSignals(
        queue_depths=(200.0, 0.0),
        pool_queued_tasks=(0, 0),
        inflight_slots=(0, 0),
        batch_sizes=(1, 1),
    )


def test_queue_gradient_logs_upstream_bottleneck_and_starved_downstream(
    loguru_caplog: pytest.LogCaptureFixture,
) -> None:
    """A full producer before an empty consumer is logged as the queue bottleneck."""
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=64)
    config = SaturationAwareConfig(speed_estimation_min_data_points=1)
    scheduler = _scheduler(spec, cluster, config)
    scheduler.setup(problem)
    worker_allocator = allocator.WorkerAllocator.make(cluster)
    t0 = 100.0

    loguru_caplog.clear()
    now = t0 + config.speed_estimation_window_s + 1.0
    scheduler.update_with_measurements(now, _upstream_only_measurements(now))
    scheduler.observe_runtime(_queue_gradient_starved_downstream_signals())
    scheduler.autoscale(now, _state(spec, worker_allocator))

    lines = [r.getMessage() for r in loguru_caplog.records if "saturation-aware decision:" in r.getMessage()]
    assert lines, "expected a decision snapshot with qstate fields"
    upstream, downstream = _stage_names(spec)
    assert f"{upstream}[" in lines[-1]
    assert f"{downstream}[" in lines[-1]
    assert "qstate=bottleneck" in lines[-1]
    assert "qstate=starved" in lines[-1]


def test_downstream_zero_sample_stage_grows_one_worker_per_cycle() -> None:
    """A 0-sample downstream stage with a live worker and local backlog gains one worker per cycle.

    Cold start caps the downstream stage at a single worker (it has no live
    worker to accelerate yet). On the next cycle, with its own work still waiting
    and the queue gradient making it the growth owner, pipeline-evidence warming
    lets it add exactly one worker instead of idling at one until its own first
    sample lands - and only one, never the solver's full placeholder-throughput
    demand.
    """
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=64)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=1))
    scheduler.setup(problem)
    worker_allocator = allocator.WorkerAllocator.make(cluster)
    now = 100.0

    # Cold cycle: stage 1 has no live worker yet, so the cold cap holds it at one.
    scheduler.update_with_measurements(now, _upstream_only_measurements(now))
    scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
    cold = scheduler.autoscale(now, _state(spec, worker_allocator))
    assert len(cold.stages[1].new_workers) == 1
    _apply_to_allocator(spec, worker_allocator, cold)
    assert _worker_counts(spec, worker_allocator)[1] == 1

    # Warming cycle: local work waiting + a live worker + growth owner -> +1 only.
    now += 10.0
    scheduler.update_with_measurements(now, _upstream_only_measurements(now))
    scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
    warming = scheduler.autoscale(now, _state(spec, worker_allocator))
    assert len(warming.stages[1].new_workers) == 1
    _apply_to_allocator(spec, worker_allocator, warming)
    assert _worker_counts(spec, worker_allocator)[1] == 2


def test_trusted_upstream_growth_is_bounded_while_downstream_is_cold(
    loguru_caplog: pytest.LogCaptureFixture,
) -> None:
    """A trusted upstream stage is held to the warmup growth step while a downstream stage is still cold.

    This reproduces the run f9fa2dde priority inversion at the wiring level: a
    trusted upstream stage (here stage 0) would otherwise grow toward a w_target
    derived from a provisional bottleneck rate (cold stages are excluded), while
    the downstream stage 1 is cold with pending work. The scheduler must detect
    the still-warming pipeline and tag the upstream stage's growth
    ``growth_cap_warmup``, capping it to ``pipeline_warmup_growth_step`` per
    cycle instead of a one-cycle leap. It also confirms ``pipeline_warming`` is
    surfaced in the growth-control summary for traceability.
    """
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=64)
    config = SaturationAwareConfig(speed_estimation_min_data_points=1)
    scheduler = _scheduler(spec, cluster, config)
    scheduler.setup(problem)
    worker_allocator = allocator.WorkerAllocator.make(cluster)
    upstream = _stage_names(spec)[0]
    now = 100.0

    # Warm-up/fill cycle: stage 0 is trusted but still has zero workers, so its
    # cap_src is 0 and it has no real w_target yet. It is correctly UNCAPPED and
    # the solver fills it (the warming gate guards the w_target growth path, not
    # the very first cold-start fill). This gives stage 0 a real worker count.
    scheduler.update_with_measurements(now, _upstream_only_measurements(now))
    scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
    _apply_to_allocator(spec, worker_allocator, scheduler.autoscale(now, _state(spec, worker_allocator)))
    assert _worker_counts(spec, worker_allocator)[0] > config.pipeline_warmup_growth_step

    # Stage 0 now has a real w_target while stage 1 stays cold with backlog
    # (pipeline still warming). Every subsequent cycle the trusted upstream must
    # be held to the warmup step instead of leaping to its provisional w_target.
    warmup_summaries: list[str] = []
    for _ in range(4):
        now += 10.0
        scheduler.update_with_measurements(now, _upstream_only_measurements(now))
        scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
        loguru_caplog.clear()
        solution = scheduler.autoscale(now, _state(spec, worker_allocator))
        warmup_summaries += [
            record.getMessage()
            for record in loguru_caplog.records
            if "saturation-aware growth control:" in record.getMessage()
        ]
        assert len(solution.stages[0].new_workers) <= config.pipeline_warmup_growth_step
        _apply_to_allocator(spec, worker_allocator, solution)

    # The upstream stage must have been bounded by the pipeline-warming gate
    # (tagged growth_cap_warmup) while the downstream stage was still cold, and
    # the warming flag must be surfaced in the summary for traceability.
    upstream_warmup_lines = [line for line in warmup_summaries if f"{upstream}: {RampReason.CAPPED_WARMUP}" in line]
    assert upstream_warmup_lines, warmup_summaries
    assert "pipeline_warming=True" in upstream_warmup_lines[-1]


def test_autoscale_before_setup_raises() -> None:
    spec, cluster, _ = _build([0.25], num_cpus=8)
    scheduler = _scheduler(spec, cluster)
    with pytest.raises(RuntimeError):
        scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))


def test_bottleneck_stage_grows_more_than_fast_upstream() -> None:
    """The slower (bottleneck) stage receives more workers than its fast upstream.

    At cold start the harness reports zero live workers, so the capacity model
    sees no bottleneck and demand deflates both stages by the same unit
    multiplier. This therefore asserts the *solver's* speed-balancing: handed the
    real per-worker speeds (stage 1 is 50x slower), FRAG allocates more workers
    to the slow stage to equalize throughput.
    """
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=1))
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


def test_upstream_queue_lens_reads_input_then_prior_stage_queues() -> None:
    """Stage 0 reads the pipeline input queue; each later stage reads its upstream output queue."""
    input_queue = streaming.Queue()
    input_queue.by_node_id[None].extend([_mock_object_ref() for _ in range(3)])  # 3 source items
    stage0_out = streaming.Queue()
    stage0_out.by_node_id[None].append(_mock_object_ref())  # 1 item waiting to feed stage 1
    queues = [stage0_out, streaming.Queue()]
    # idx 0 -> len(input_queue) == 3; idx 1 -> len(queues[0]) == 1.
    assert streaming._upstream_queue_lens(input_queue, queues, 2) == [3, 1]


def _streaming_autoscaler(
    spec: v1.PipelineSpec, cluster: resources.ClusterResources, scheduler: SchedulerKind
) -> streaming.Autoscaler:
    spec.config.mode_specific = StreamingSpecificSpec(scheduler=scheduler)
    return streaming.Autoscaler(allocator.WorkerAllocator.make(cluster), spec, cluster)


def test_autoscaler_uses_runtime_signals_for_saturation_aware() -> None:
    """The saturation-aware scheduler is runtime-aware, so its submission is deferred post-transfer."""
    spec, cluster, _ = _build([1.0], num_cpus=16)
    with _streaming_autoscaler(spec, cluster, SchedulerKind.SATURATION_AWARE) as autoscaler:
        assert autoscaler.uses_runtime_signals is True


def test_fragmentation_autoscaler_does_not_use_runtime_signals() -> None:
    """The fragmentation solver ignores runtime signals, so it keeps the early submission point."""
    spec, cluster, _ = _build([1.0], num_cpus=16)
    with _streaming_autoscaler(spec, cluster, SchedulerKind.FRAGMENTATION_BASED) as autoscaler:
        assert autoscaler.uses_runtime_signals is False


def test_all_queued_measurement_batches_are_drained_on_next_autoscale() -> None:
    # min_data_points=2 means a single un-drained batch leaves the estimator below
    # threshold (speed None -> no deflation); deflation here proves both batches applied.
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=2))
    scheduler.setup(problem)
    now = 100.0
    scheduler.update_with_measurements(now, _measurements(now, [0.1, 5.0]))
    scheduler.update_with_measurements(now, _measurements(now, [0.1, 5.0]))
    solution = scheduler.autoscale(now, _state(spec, allocator.WorkerAllocator.make(cluster)))
    new_counts = [len(stage.new_workers) for stage in solution.stages]
    assert new_counts[1] > new_counts[0]


def test_observe_runtime_rejects_stage_count_mismatch() -> None:
    spec, cluster, _ = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster)
    with pytest.raises(ValueError, match="expected 2"):
        scheduler.observe_runtime(
            RuntimeSignals(queue_depths=(0.0,), pool_queued_tasks=(0,), inflight_slots=(0,), batch_sizes=(1,))
        )


def test_autoscale_consumes_runtime_signals_without_error() -> None:
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(problem)
    scheduler.observe_runtime(
        RuntimeSignals(queue_depths=(5.0, 0.0), pool_queued_tasks=(0, 0), inflight_slots=(0, 0), batch_sizes=(1, 1))
    )
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages) == 2


def _cpu_cluster(num_cpus: int) -> resources.ClusterResources:
    return resources.ClusterResources(
        nodes={"n0": resources.NodeResources(used_cpus=0, total_cpus=num_cpus, gpus=[], name="n0")}
    )


def test_pinned_stage_skips_ramp_while_autoscaled_stage_is_capped() -> None:
    """A pinned stage reaches its requested count at cold start; an autoscaled peer is still ramp-capped to one."""
    spec = v1.PipelineSpec(
        input_data=range(100),
        stages=[
            v1.StageSpec(_CpuStage(1.0, 1.0), num_workers=4),
            v1.StageSpec(_CpuStage(1.0, 1.0)),
        ],
    )
    cluster = _cpu_cluster(16)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(streaming._make_problem_from_pipeline_spec(spec, cluster))
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 4
    assert len(solution.stages[1].new_workers) == 1


def test_pinned_stage_over_capacity_holds_at_current_without_raising() -> None:
    """A pinned count that cannot fit holds at the current size instead of aborting the solve."""
    spec = v1.PipelineSpec(input_data=range(100), stages=[v1.StageSpec(_CpuStage(1.0, 1.0), num_workers=4)])
    cluster = _cpu_cluster(2)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(streaming._make_problem_from_pipeline_spec(spec, cluster))
    solution = scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert len(solution.stages[0].new_workers) == 0


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    ``cosmos_xenna.utils.python_log`` routes logging through loguru, which does
    not propagate to the stdlib ``logging`` module, so ``caplog`` is otherwise
    blind to ``logger.error(...)`` calls. The bridge re-emits every loguru record
    through a stdlib logger named ``"loguru"`` and tears the sink down at the end.
    """
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        level=0,
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


def test_pinned_stage_held_at_zero_escalates_to_error(loguru_caplog: pytest.LogCaptureFixture) -> None:
    """A pinned stage that cannot place even one worker is escalated to ERROR.

    The operator pinned a count the saturated cluster cannot host, so the
    infeasible solve retries with the stage held at its current zero workers.
    Holding a pinned stage at zero means it will not run this cycle, so the
    scheduler must surface an ERROR the operator can act on, not a silent retry.
    """
    spec = v1.PipelineSpec(input_data=range(100), stages=[v1.StageSpec(_CpuStage(1.0, 1.0), num_workers=4)])
    cluster = _cpu_cluster(2)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(streaming._make_problem_from_pipeline_spec(spec, cluster))
    scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))
    errors = [record.getMessage() for record in loguru_caplog.records if record.levelno == logging.ERROR]
    assert any("cannot place a worker" in message for message in errors), errors


def test_solve_reraises_when_nothing_can_be_relaxed() -> None:
    """With no pinned stages to hold back, a genuinely infeasible solve propagates."""
    spec, cluster, problem = _build([1.0, 1.0, 1.0], num_cpus=2)
    scheduler = _scheduler(spec, cluster)
    scheduler.setup(problem)
    with pytest.raises(RuntimeError):
        scheduler.autoscale(100.0, _state(spec, allocator.WorkerAllocator.make(cluster)))


def test_release_alpha_is_uniform_and_config_driven() -> None:
    """Every stage shares one release alpha derived from the two release tunables.

    Release is resource-agnostic: a single ``alpha_down = 1 /
    (scale_down_release_cycles * scale_down_release_slowdown)`` smooths the
    sustainable rate for GPU and CPU stages alike, so the slowdown factor is the
    one knob that trades release caution against responsiveness.
    """
    spec, cluster, _ = _build([1.0, 1.0], num_cpus=16)
    config = SaturationAwareConfig(scale_down_release_cycles=6, scale_down_release_slowdown=4.0)
    params = _scheduler(spec, cluster, config)._capacity.params
    assert params.alpha_down == pytest.approx(1.0 / (6 * 4.0))


def test_measured_speed_is_withheld_until_the_trust_threshold_is_reached() -> None:
    """The assembled cycle reports a cold speed until the stage clears the trust threshold.

    Feeds measurements through the public ingest path and reads the resulting
    cycle: below the threshold the demand snapshot speed is ``None``, so a noisy
    early sample cannot set the bottleneck or rate; once enough samples land, the
    measured speed flows into the cycle.
    """
    spec, cluster, problem = _build([1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=3))
    scheduler.setup(problem)
    state = _state(spec, allocator.WorkerAllocator.make(cluster))

    # Two samples (< threshold): autoscale drains them, but the cycle stays cold.
    scheduler.update_with_measurements(100.0, _measurements(100.0, [0.5]))
    scheduler.update_with_measurements(101.0, _measurements(101.0, [0.5]))
    scheduler.autoscale(101.0, state)
    assert scheduler._build_cycle(101.0, state).demand_snapshots[0].speed is None

    # Third sample crosses the threshold: the measured speed now flows through.
    scheduler.update_with_measurements(102.0, _measurements(102.0, [0.5]))
    scheduler.autoscale(102.0, state)
    assert scheduler._build_cycle(102.0, state).demand_snapshots[0].speed is not None


def test_empty_skip_does_not_inflate_measured_speed_in_a_cycle() -> None:
    """An empty + instant skip drained through the ingest path cannot spike the cycle speed.

    Reproduces the incident chain at the scheduler boundary: warm a stage with
    real durations, then feed one ``num_returns=0`` ~0.1 ms skip. The assembled
    cycle's measured speed must stay at the real value, so ``cap_src`` and the
    bottleneck rate it feeds cannot blow up.
    """
    spec, cluster, problem = _build([1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=1))
    scheduler.setup(problem)
    state = _state(spec, allocator.WorkerAllocator.make(cluster))

    scheduler.update_with_measurements(100.0, _measurements(100.0, [0.5]))  # real 0.5 s task -> 2.0/s
    scheduler.autoscale(100.0, state)
    real_speed = scheduler._build_cycle(100.0, state).demand_snapshots[0].speed
    assert real_speed == pytest.approx(2.0)

    scheduler.update_with_measurements(101.0, _empty_skip_measurements(101.0))  # empty + instant skip
    scheduler.autoscale(101.0, state)
    assert scheduler._build_cycle(101.0, state).demand_snapshots[0].speed == pytest.approx(2.0)


def test_in_flight_stall_ages_cycle_speed_and_flags_stale() -> None:
    """A warm stage stuck on an in-flight task ages its cycle speed and flags stale.

    Drives the wiring end to end: warm a stage to the trust threshold, then
    report a worker in flight with no further completions and advance the clock.
    The assembled cycle's measured speed is aged far below the frozen windowed
    rate and the stage is flagged stale for the capacity model.
    """
    spec, cluster, problem = _build([1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=3))
    scheduler.setup(problem)
    state = _state(spec, allocator.WorkerAllocator.make(cluster))
    for t in (100.0, 101.0, 102.0):  # three 1 s tasks -> last completion at t=102
        scheduler.update_with_measurements(t, _measurements(t, [1.0]))
        scheduler.autoscale(t, state)

    scheduler.observe_runtime(
        RuntimeSignals(queue_depths=(5.0,), pool_queued_tasks=(0,), inflight_slots=(1,), batch_sizes=(1,))
    )
    cycle = scheduler._build_cycle(300.0, state)  # ~200 s with no completion while in flight
    speed = cycle.demand_snapshots[0].speed
    assert speed is not None
    assert speed < 0.1  # aged from the frozen windowed 1.0/s toward 1/elapsed
    assert cycle.rate_is_stale[0] is True


def test_warm_pipeline_converges_to_a_stable_split() -> None:
    """A warmed pipeline reaches a steady worker split and then stops churning.

    The slower stage is the bottleneck and must win more workers; once the
    cluster-balanced split is reached, replaying identical cycles must add and
    delete nothing. Any cross-cycle state that double-advances (re-decaying the
    EWMA, double-counting a release streak) would re-open churn here.
    """
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=64)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=1))
    scheduler.setup(problem)
    worker_allocator = allocator.WorkerAllocator.make(cluster)
    now = 100.0
    churn_per_cycle: list[int] = []
    for _ in range(4):
        scheduler.update_with_measurements(now, _measurements(now, [0.1, 0.5]))  # stage 1 is 5x slower
        scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
        solution = scheduler.autoscale(now, _state(spec, worker_allocator))
        churn_per_cycle.append(sum(len(s.new_workers) + len(s.deleted_workers) for s in solution.stages))
        _apply_to_allocator(spec, worker_allocator, solution)
        now += 10.0

    fast, slow = _worker_counts(spec, worker_allocator)
    assert slow > fast  # the bottleneck stage wins more workers
    assert churn_per_cycle[0] > 0  # cold start grows the pipeline
    assert churn_per_cycle[-1] == 0  # converged: no flapping once the split is reached


def test_bottleneck_shift_scales_down_now_fast_stage_gradually() -> None:
    """When the bottleneck moves, the over-provisioned stage drains gradually.

    This is the only path that drives deletes end-to-end. Warm stage 0 as the
    bottleneck, then flip the measured speeds so stage 1 becomes the bottleneck.
    The solver now wants stage 0's workers for stage 1; the scale-down floor
    releases stage 0 a bounded amount per cycle - never to zero and never in a
    single jump - exercising the floor delete-cap and the ramp/floor editor
    composition.
    """
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=64)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=1))
    scheduler.setup(problem)
    worker_allocator = allocator.WorkerAllocator.make(cluster)
    now = 100.0

    # Warm-up: stage 0 is the slow bottleneck and wins the cluster. Run several
    # cycles so the capacity EWMA and the sticky bottleneck identity settle.
    for _ in range(5):
        scheduler.update_with_measurements(now, _measurements(now, [0.5, 0.1]))
        scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
        _apply_to_allocator(spec, worker_allocator, scheduler.autoscale(now, _state(spec, worker_allocator)))
        now += 10.0
    warm = _worker_counts(spec, worker_allocator)
    assert warm[0] > warm[1]  # stage 0 dominates while it is the bottleneck

    # Flip the bottleneck so stage 1 is now slowest; stage 0 must shed workers.
    stage0_over_time: list[int] = []
    for _ in range(8):
        scheduler.update_with_measurements(now, _measurements(now, [0.1, 0.5]))
        scheduler.observe_runtime(_backlog(2, queue_depth=200.0))
        _apply_to_allocator(spec, worker_allocator, scheduler.autoscale(now, _state(spec, worker_allocator)))
        stage0_over_time.append(_worker_counts(spec, worker_allocator)[0])
        now += 10.0

    assert stage0_over_time[-1] < warm[0]  # net scale-down: stage 0 ends below its bottleneck peak
    assert min(stage0_over_time) >= 1  # the floor (min_workers) is never breached
    assert any(
        1 < count < warm[0] for count in stage0_over_time
    )  # gradual: bounded per-cycle release, no one-shot collapse


def _cycle_with_local_pending(local_pending: tuple[float, ...], batch_sizes: tuple[int, ...]) -> _Cycle:
    """Build a _Cycle exercising only has_local_input; other fields are inert."""
    n = len(local_pending)
    zeros = (0.0,) * n
    return _Cycle(
        time=0.0,
        pending_work_ages=zeros,
        workers=(0,) * n,
        demand_snapshots=(),
        batch_sizes=batch_sizes,
        chain_factors=(1.0,) * n,
        is_manual=(False,) * n,
        local_depths=zeros,
        local_pending_depths=local_pending,
        active_depths=zeros,
        ready_workers=(0,) * n,
        rate_is_stale=(False,) * n,
        queued_stock=zeros,
        active_stock=zeros,
        activity_snapshot=None,
    )


def test_has_local_input_true_at_exactly_one_batch() -> None:
    """local_pending == batch_size is one usable batch (>=), so growth is allowed."""
    cycle = _cycle_with_local_pending(local_pending=(4.0,), batch_sizes=(4,))
    assert cycle.has_local_input(0) is True


def test_has_local_input_false_below_one_batch() -> None:
    """local_pending below one batch cannot feed another worker."""
    cycle = _cycle_with_local_pending(local_pending=(3.0,), batch_sizes=(4,))
    assert cycle.has_local_input(0) is False


def test_pending_work_age_resets_when_a_stage_drains() -> None:
    """Per-stage pending-work age starts when work arrives and resets on drain.

    The cold-start ramp's slow-starter release keys off how long work has
    actually been blocked, not how long the scheduler has run, so a stage that
    drains and later refills must start a fresh timer rather than inherit the
    scheduler's elapsed time.
    """
    spec, cluster, _ = _build([1.0, 1.0], num_cpus=8)
    scheduler = _scheduler(spec, cluster)
    assert scheduler._pending_work_ages(0.0, (0.0, 5.0)) == (0.0, 0.0)
    assert scheduler._pending_work_ages(10.0, (0.0, 5.0)) == (0.0, 10.0)
    assert scheduler._pending_work_ages(12.0, (0.0, 0.0)) == (0.0, 0.0)  # drained -> timer reset
    assert scheduler._pending_work_ages(20.0, (0.0, 3.0)) == (0.0, 0.0)  # refilled -> fresh start
    assert scheduler._pending_work_ages(25.0, (0.0, 3.0)) == (0.0, 5.0)


def _capacity_with_bottleneck(
    *,
    bottleneck_stage: int,
    w_targets: tuple[int, ...],
    w_target_is_real: tuple[bool, ...] | None = None,
) -> CapacityPlan:
    """Build a CapacityPlan carrying only the fields _compute_reclaim_beneficial reads.

    The bottleneck index plus each stage's `w_target` / `w_target_is_real` carry
    meaning; the remaining capacity fields are inert placeholders.
    """
    num_stages = len(w_targets)
    real = w_target_is_real if w_target_is_real is not None else (True,) * num_stages
    stages = tuple(
        StageCapacity(
            speed=1.0,
            target_speed=1.0,
            cap_src=1.0,
            a_raw=1.0,
            a_ewma=1.0,
            w_sustain=1,
            w_target=w_targets[i],
            w_target_is_real=real[i],
        )
        for i in range(num_stages)
    )
    return CapacityPlan(
        stages=stages,
        bottleneck_stage=bottleneck_stage,
        bottleneck_rate=1.0,
        next_bottleneck_rate=1.0,
    )


def _cycle_for_reclaim(
    workers: tuple[int, ...],
    is_manual: tuple[bool, ...],
    rate_is_stale: tuple[bool, ...] | None = None,
) -> _Cycle:
    """Build a _Cycle exposing only the workers, is_manual, and rate_is_stale fields the reclaim check reads."""
    n = len(workers)
    zeros = (0.0,) * n
    return _Cycle(
        time=0.0,
        pending_work_ages=zeros,
        workers=workers,
        demand_snapshots=(),
        batch_sizes=(1,) * n,
        chain_factors=(1.0,) * n,
        is_manual=is_manual,
        local_depths=zeros,
        local_pending_depths=zeros,
        active_depths=zeros,
        ready_workers=(0,) * n,
        rate_is_stale=rate_is_stale if rate_is_stale is not None else (False,) * n,
        queued_stock=zeros,
        active_stock=zeros,
        activity_snapshot=None,
    )


def _mixed_pipeline_scheduler() -> SaturationAwareScheduler:
    """A 3-stage pipeline: CPU bottleneck, CPU-only stage, GPU stage (also reserves CPUs)."""
    spec = v1.PipelineSpec(
        input_data=range(10),
        stages=[
            v1.StageSpec(_CpuStage(1.0, 1.0)),
            v1.StageSpec(_CpuStage(4.0, 1.0)),
            v1.StageSpec(_GpuStage(1.0)),
        ],
    )
    return _scheduler(spec, _gpu_cluster(8))


def test_reclaim_beneficial_cpu_bottleneck_excludes_gpu_stage() -> None:
    """A growing CPU bottleneck makes the CPU-only stage reclaimable but never the GPU stage.

    The bottleneck is the only grower; the GPU stage also reserves host CPUs, but
    freeing it would strand a GPU the CPU grower cannot use, so the subset rule
    excludes it.
    """
    scheduler = _mixed_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(10, 1, 1))
    cycle = _cycle_for_reclaim(workers=(2, 5, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, True, False)


def test_reclaim_beneficial_gpu_bottleneck_includes_cpu_and_gpu_stages() -> None:
    """A growing GPU grower (also reserving CPUs) makes both other stages reclaimable; it never marks itself."""
    scheduler = _mixed_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=2, w_targets=(1, 1, 10))
    cycle = _cycle_for_reclaim(workers=(2, 5, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (True, True, False)


def test_reclaim_beneficial_false_when_bottleneck_is_manual() -> None:
    """A manual/pinned stage is excluded as a grower, so freeing any stage helps nothing."""
    scheduler = _mixed_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(10, 1, 1))
    cycle = _cycle_for_reclaim(workers=(2, 5, 7), is_manual=(True, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def test_reclaim_beneficial_false_when_bottleneck_target_not_real() -> None:
    """A cold stage (placeholder w_target) is excluded as a grower, so there is no reclaim signal."""
    scheduler = _mixed_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(10, 1, 1), w_target_is_real=(False, True, True))
    cycle = _cycle_for_reclaim(workers=(2, 5, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def test_reclaim_beneficial_false_when_bottleneck_not_growing() -> None:
    """A stage at or above its target is not a grower; with no grower nothing is reclaimable."""
    scheduler = _mixed_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(2, 1, 1))
    cycle = _cycle_for_reclaim(workers=(2, 5, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def test_reclaim_beneficial_false_when_no_bottleneck() -> None:
    """With no stage wanting to grow there is no reclaim signal (the bottleneck index is unused)."""
    scheduler = _mixed_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=-1, w_targets=(1, 1, 1))
    cycle = _cycle_for_reclaim(workers=(2, 5, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def _two_gpu_pipeline_scheduler() -> SaturationAwareScheduler:
    """A 3-stage pipeline: a CPU-only stage then two GPU stages (each also reserves CPUs)."""
    spec = v1.PipelineSpec(
        input_data=range(10),
        stages=[
            v1.StageSpec(_CpuStage(1.0, 1.0)),
            v1.StageSpec(_GpuStage(1.0)),
            v1.StageSpec(_GpuStage(1.0)),
        ],
    )
    return _scheduler(spec, _gpu_cluster(8))


def test_reclaim_beneficial_idle_gpu_stage_freed_for_nonbottleneck_gpu_grower() -> None:
    """An idle GPU stage is reclaimable for a growing GPU stage that is not the bottleneck.

    The sticky bottleneck is a non-growing CPU stage (stage 0); only GPU stage 1
    wants to grow. The over-provisioned GPU stage 2 is freed for that grower, and
    the grower never marks itself (stage 1).
    """
    scheduler = _two_gpu_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(2, 5, 1))
    cycle = _cycle_for_reclaim(workers=(2, 1, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (True, False, True)


def test_reclaim_beneficial_gpu_stage_excluded_for_nonbottleneck_cpu_grower() -> None:
    """A CPU-only grower never makes a GPU stage reclaimable, even off the bottleneck.

    Only CPU stage 0 wants to grow; freeing either GPU stage would strand a GPU
    the CPU grower cannot use, so the subset rule excludes both.
    """
    scheduler = _two_gpu_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=2, w_targets=(5, 1, 7))
    cycle = _cycle_for_reclaim(workers=(1, 5, 7), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def test_reclaim_beneficial_ignores_manual_grower() -> None:
    """A manual stage that wants to grow is excluded as a grower.

    Same shape as the GPU-grower case but the lone grower (stage 1) is
    operator-pinned, so there is no grower and nothing is reclaimable.
    """
    scheduler = _two_gpu_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(2, 5, 1))
    cycle = _cycle_for_reclaim(workers=(2, 1, 7), is_manual=(False, True, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def test_reclaim_beneficial_unions_over_multiple_growers() -> None:
    """A stage is reclaimable if any grower can use its freed resource.

    Both GPU stages grow, so each is reclaimable for the other and the CPU stage
    is reclaimable for either - the signal is the union over all growers.
    """
    scheduler = _two_gpu_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=1, w_targets=(2, 5, 5))
    cycle = _cycle_for_reclaim(workers=(2, 1, 1), is_manual=(False, False, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (True, True, True)


def test_reclaim_beneficial_false_when_grower_is_stalled() -> None:
    """A stalled stage is not a grower, so it cannot trigger reclaim of a warm peer.

    Same shape as the idle-GPU-freed-for-a-GPU-grower case (the lone grower is GPU
    stage 1, w_target 5 > 1), but the grower is ``rate_is_stale``: capacity has
    clamped its target to workers + 1 so it merely looks like it wants to grow,
    while adding workers cannot raise its collapsing completion rate. With no
    genuine grower nothing is reclaimable - this is the run f9fa2dde regression
    where a stalled InternVideo2EmbeddingStage tore VllmCaptionStage down.
    """
    scheduler = _two_gpu_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=0, w_targets=(2, 5, 1))
    cycle = _cycle_for_reclaim(workers=(2, 1, 7), is_manual=(False, False, False), rate_is_stale=(False, True, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (False, False, False)


def test_reclaim_beneficial_non_stale_grower_still_drives_reclaim_when_a_peer_is_stalled() -> None:
    """Excluding stalled growers is per-stage: a healthy grower still drives reclaim.

    Both GPU stages have a target above their workers, but GPU stage 1 is
    ``rate_is_stale`` and is dropped from the grower set. The non-stale GPU
    grower (stage 2) still makes its peers reclaimable: the CPU stage 0 and the
    stalled GPU stage 1 are freed for it, while stage 2 (the only grower) never
    marks itself. Contrast the all-True union result when neither is stalled.
    """
    scheduler = _two_gpu_pipeline_scheduler()
    capacity = _capacity_with_bottleneck(bottleneck_stage=1, w_targets=(2, 5, 5))
    cycle = _cycle_for_reclaim(workers=(2, 1, 1), is_manual=(False, False, False), rate_is_stale=(False, True, False))
    assert scheduler._compute_reclaim_beneficial(cycle, capacity) == (True, True, False)
