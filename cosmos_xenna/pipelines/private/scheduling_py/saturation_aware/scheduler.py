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

"""Saturation-aware scheduler: backlog-biased sizing over the fragmentation solver.

Each cycle the scheduler deflates per-stage speed estimates by a backlog
demand multiplier so the shared fragmentation solver grows backed-up
stages, then post-processes the solution it returns to protect
warming-up workers and transiently-starved expensive stages. It edits no
fragmentation-scheduler code: the solver is called read-only and the
returned ``Solution`` is mutated in place via its own setters.

::

    size  -->  solve  -->  grace  -->  floor
    (deflate   (FRAG,      (drop      (clamp deletes:
     speed/m)   read-only)  young)     hold starved stages)

The driver thread only enqueues inputs (``update_with_measurements`` queues a
batch; ``set_queue_snapshot`` publishes the per-stage queue depths);
``autoscale`` runs on a single executor thread and owns all estimator and
cross-cycle state mutation, so estimator state is never shared across threads.
"""

import math
import queue
from collections.abc import Sequence

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.autoscaling_algorithms import (
    Estimate,
    Estimates,
    WorkerIdFactory,
    run_fragmentation_autoscaler,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.estimator import PipelineRateEstimator
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.floor import (
    FloorInputs,
    FloorParams,
    FloorState,
    compute_floors,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.grace import WarmupGrace
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.sizing import BacklogDemandPolicy, StageSnapshot
from cosmos_xenna.utils import python_log as logger

_ALPHA_UP = 1.0
_GPU_RELEASE_SLOWDOWN = 2.0
_RELEASE_CONFIRM_DIVISOR = 3
_MIN_WORKERS = 1
_OVERALLOCATION_TARGET = 1.0


@attrs.define
class SaturationAwareScheduler:
    """Backlog-biased autoscaler built on the read-only fragmentation solver.

    Attributes:
        config: Operator tunables (cadence, catch-up cap, headroom, smoothing).
        stage_names: Canonical per-stage names in pipeline order.
        stage_batch_sizes: Per-stage input items consumed per batch.
        stage_is_gpu: Whether each stage holds GPU workers (slower release).
    """

    config: SaturationAwareConfig
    stage_names: tuple[str, ...]
    stage_batch_sizes: tuple[int, ...]
    stage_is_gpu: tuple[bool, ...]
    _estimator: PipelineRateEstimator = attrs.field(init=False)
    _floor_params: FloorParams = attrs.field(init=False)
    _floor_state: FloorState = attrs.field(init=False)
    _grace: WarmupGrace = attrs.field(init=False, factory=WarmupGrace)
    _worker_id_factory: WorkerIdFactory = attrs.field(init=False, factory=WorkerIdFactory)
    _demand: BacklogDemandPolicy = attrs.field(init=False)
    _problem: data_structures.Problem | None = attrs.field(init=False, default=None)
    _queue_snapshot: tuple[float, ...] = attrs.field(init=False, factory=tuple)
    _pending_measurements: queue.Queue[data_structures.Measurements] = attrs.field(init=False, factory=queue.Queue)

    def __attrs_post_init__(self) -> None:
        """Derive the estimator and floor tuning from the config."""
        config = self.config
        self._estimator = PipelineRateEstimator(
            config.speed_estimation_window_s, config.speed_estimation_min_data_points
        )
        cycles = config.scale_down_release_cycles
        self._floor_params = FloorParams(
            alpha_up=_ALPHA_UP,
            alpha_down_cpu=1.0 / cycles,
            alpha_down_gpu=1.0 / (cycles * _GPU_RELEASE_SLOWDOWN),
            release_confirm_cycles=max(1, math.ceil(cycles / _RELEASE_CONFIRM_DIVISOR)),
            min_workers=_MIN_WORKERS,
        )
        self._floor_state = FloorState.initial(len(self.stage_names))
        self._demand = BacklogDemandPolicy(config)

    def setup(self, problem: data_structures.Problem) -> None:
        """Record the static problem; the solver needs it each cycle."""
        self._problem = problem

    def update_with_measurements(self, time: float, measurements: data_structures.Measurements) -> None:
        """Queue completed-task timings for ingestion at the next autoscale.

        Runs on the driver thread; the batch is folded into the estimator inside
        :meth:`autoscale` so all estimator access stays on the single executor
        thread (the estimator's windowing keys off each task's own timestamp, so
        deferring ingestion by one cycle does not skew the rate estimate).
        """
        self._pending_measurements.put(measurements)

    def set_queue_snapshot(self, queue_depths: Sequence[float]) -> None:
        """Record this cycle's per-stage input queue depths."""
        self._queue_snapshot = tuple(float(depth) for depth in queue_depths)

    def autoscale(self, time: float, problem_state: data_structures.ProblemState) -> data_structures.Solution:
        """Size to backlog, solve placement via FRAG, then protect young/starved stages.

        Args:
            time: Decision timestamp, in seconds.
            problem_state: Current per-stage workers and slots.

        Returns:
            The fragmentation solver's solution with delete sets clamped so
            warming-up workers and transiently-starved stages survive.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._problem is None:
            raise RuntimeError("setup() must be called before autoscale()")
        self._drain_pending_measurements()
        state_stages = list(problem_state.rust.stages)
        workers = [stage.num_workers() for stage in state_stages]
        self._grace.observe((group.id for stage in state_stages for group in stage.worker_groups), time)

        estimates, speeds, returns, multipliers = self._size(time, workers)
        logger.debug(
            "saturation-aware demand: "
            + ", ".join(f"{name}={value:.2f}" for name, value in zip(self.stage_names, multipliers, strict=True))
        )
        solution = run_fragmentation_autoscaler(
            self._problem, problem_state, Estimates(estimates), _OVERALLOCATION_TARGET, self._worker_id_factory
        )
        floors = self._floors(workers, speeds, returns)
        self._protect(solution, workers, floors, time)
        return solution

    def _size(self, now: float, workers: Sequence[int]) -> tuple[list[Estimate], list[float], list[float], list[float]]:
        """Build deflated per-stage estimates plus the floor and logging signals.

        Delegates the demand multiplier to :class:`BacklogDemandPolicy` and
        returns the solver estimates, per-stage measured speed for the floor
        (``0`` when unknown), fan-out for the chain, and the demand multiplier.
        """
        queue_depths = self._queue_for_cycle(len(workers))
        estimates: list[Estimate] = []
        speeds: list[float] = []
        returns: list[float] = []
        multipliers: list[float] = []
        for index, name in enumerate(self.stage_names):
            snapshot = StageSnapshot(
                name=name,
                workers=workers[index],
                queue_depth=queue_depths[index],
                speed=self._estimator.speed(name, now),
                num_returns=self._estimator.num_returns(name),
                batch_size=self.stage_batch_sizes[index],
                sample_count=self._estimator.sample_count(name),
            )
            result = self._demand.size(snapshot)
            estimates.append(Estimate(result.effective_speed, result.num_returns))
            speeds.append(result.measured_speed_for_floor)
            returns.append(result.num_returns)
            multipliers.append(result.multiplier)
        return estimates, speeds, returns, multipliers

    def _floors(self, workers: Sequence[int], speeds: Sequence[float], returns: Sequence[float]) -> tuple[int, ...]:
        """Compute the per-stage scale-down floors."""
        queue_depths = self._queue_for_cycle(len(workers))
        chain_values = chain.chain_factors(returns, self.stage_batch_sizes)
        stock = chain.whole_chain_stock(queue_depths, chain_values)
        inputs = FloorInputs(
            workers=tuple(workers),
            speed=tuple(speeds),
            chain=tuple(chain_values),
            stock_src=tuple(stock),
            batch_sizes=self.stage_batch_sizes,
            is_gpu=self.stage_is_gpu,
        )
        result = compute_floors(inputs, self._floor_state, self._floor_params)
        self._floor_state = result.state
        return result.floors

    def _protect(
        self, solution: data_structures.Solution, workers: Sequence[int], floors: Sequence[int], now: float
    ) -> None:
        """Clamp each stage's delete set for grace and the scale-down floor.

        Drops warming-up workers from the solver's deletes, then caps the
        remaining deletes so the post-delete worker count stays at or above the
        stage floor. Mutates the solution's ``deleted_workers`` in place via the
        native setters, logging each stage where it overrides the solver.
        """
        grace_s = self.config.scale_down_grace_after_ready_s
        rust_solution = solution.rust
        stages = list(rust_solution.stages)
        changed = False
        for index, stage in enumerate(stages):
            deletes = list(stage.deleted_workers)
            if not deletes:
                continue
            deletable_ids = set(self._grace.allowed_deletions([worker.id for worker in deletes], now, grace_s))
            grace_allowed_deletes = [worker for worker in deletes if worker.id in deletable_ids]
            max_deletes = max(0, workers[index] + len(stage.new_workers) - floors[index])
            final = grace_allowed_deletes[:max_deletes]
            if len(final) == len(deletes):
                continue
            stage.deleted_workers = final
            changed = True
            logger.info(
                f"saturation-aware override: stage '{self.stage_names[index]}' kept "
                f"{len(deletes) - len(final)} of {len(deletes)} worker deletion(s) the fragmentation "
                f"solver proposed (grace-protected={len(deletes) - len(grace_allowed_deletes)}, "
                f"floor-protected={len(grace_allowed_deletes) - len(final)}, floor_target={floors[index]})"
            )
        if changed:
            rust_solution.stages = stages

    def _queue_for_cycle(self, num_stages: int) -> tuple[float, ...]:
        """Return the queue snapshot if it matches the stage count, else zeros."""
        if len(self._queue_snapshot) == num_stages:
            return self._queue_snapshot
        return (0.0,) * num_stages

    def _drain_pending_measurements(self) -> None:
        """Fold all queued measurement batches into the estimator (executor thread)."""
        while True:
            try:
                measurements = self._pending_measurements.get_nowait()
            except queue.Empty:
                break
            self._ingest(measurements)

    def _ingest(self, measurements: data_structures.Measurements) -> None:
        """Fold one measurement batch into the per-stage estimator."""
        for index, stage_measurements in enumerate(measurements.rust.stages):
            if index >= len(self.stage_names):
                break
            name = self.stage_names[index]
            for task in stage_measurements.task_measurements:
                self._estimator.observe(
                    name,
                    duration_s=task.end_time - task.start_time,
                    num_returns=float(task.num_returns),
                    now=task.start_time,
                )
