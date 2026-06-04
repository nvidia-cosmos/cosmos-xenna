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
from typing import cast
from unittest import mock

import pytest
from loguru import logger as loguru_logger

import cosmos_xenna.pipelines.v1 as v1
from cosmos_xenna.pipelines.private import allocator, data_structures, resources, streaming
from cosmos_xenna.pipelines.private.autoscaling_algorithms import FragmentationBasedAutoscaler
from cosmos_xenna.pipelines.private.scheduling_py.runtime_signals import RuntimeSignals
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.capacity import CapacityModel
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.floor import ScaleDownFloorPolicy
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.problem_template import SolverProblemTemplate
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.scheduler import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.shape import PipelineShape
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


def _scheduler(
    spec: v1.PipelineSpec,
    cluster: resources.ClusterResources,
    config: SaturationAwareConfig | None = None,
) -> SaturationAwareScheduler:
    stages = cast(list[StageSpec], spec.stages)
    return SaturationAwareScheduler(
        config=config or SaturationAwareConfig(),
        shape=PipelineShape.from_stage_specs(stages),
        solver_template=SolverProblemTemplate.from_stage_specs(stages, cluster),
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
    input_queue.by_node_id[None].extend([object(), object(), object()])  # 3 source items
    stage0_out = streaming.Queue()
    stage0_out.by_node_id[None].append(object())  # 1 item waiting to feed stage 1
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


def test_gpu_stage_uses_slower_release_alpha_than_cpu() -> None:
    """GPU stages decay their sustainable rate four times slower than CPU stages.

    The slower GPU ratchet keeps an expensive warmup stage warm while a
    transient upstream bottleneck clears, instead of tearing it down and
    paying the cold-start cost again when work resumes. The release alphas now
    live on the capacity model, which owns the throughput smoothing.
    """
    spec, cluster, _ = _build([1.0, 1.0], num_cpus=16)
    params = _scheduler(spec, cluster)._capacity.params
    assert params.alpha_down_gpu < params.alpha_down_cpu
    assert params.alpha_down_gpu == pytest.approx(params.alpha_down_cpu / 4.0)


def test_trusted_speed_is_none_below_threshold_then_measured() -> None:
    """A stage's measured speed is withheld until it clears the estimator trust threshold.

    Mirrors the cold-start ramp's trust gate so a single noisy early sample
    cannot set the bottleneck identity or rate before the stage is believed.
    """
    spec, cluster, _ = _build([1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=3))
    name = scheduler.shape.stages[0].name
    now = 100.0
    scheduler._estimator.observe(name, duration_s=0.5, num_returns=1.0, now=now)
    scheduler._estimator.observe(name, duration_s=0.5, num_returns=1.0, now=now)
    assert scheduler._trusted_speed(name, now) is None  # 2 samples < threshold (3)
    scheduler._estimator.observe(name, duration_s=0.5, num_returns=1.0, now=now)
    assert scheduler._trusted_speed(name, now) is not None  # 3 samples >= threshold


def test_capacity_and_floor_advance_once_per_autoscale() -> None:
    """The capacity model and the floor each evaluate exactly once per autoscale.

    Both own cross-cycle state (the capacity EWMA and sticky bottleneck identity,
    the floor release streaks). Evaluating either twice in one cycle would advance
    that state twice -- double-decaying the EWMA or double-counting a release
    streak -- so pin each to a single evaluation per ``autoscale``.
    """
    spec, cluster, problem = _build([1.0, 1.0], num_cpus=16)
    scheduler = _scheduler(spec, cluster, SaturationAwareConfig(speed_estimation_min_data_points=1))
    scheduler.setup(problem)
    now = 100.0
    scheduler.update_with_measurements(now, _measurements(now, [0.1, 5.0]))
    capacity_spy = mock.Mock(wraps=scheduler._capacity)
    floor_spy = mock.Mock(wraps=scheduler._floor)
    scheduler._capacity = cast(CapacityModel, capacity_spy)
    scheduler._floor = cast(ScaleDownFloorPolicy, floor_spy)
    scheduler.autoscale(now, _state(spec, allocator.WorkerAllocator.make(cluster)))
    assert capacity_spy.plan.call_count == 1
    assert floor_spy.plan.call_count == 1
