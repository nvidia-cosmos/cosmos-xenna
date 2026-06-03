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
transiently-starved expensive stages. It edits no fragmentation-scheduler
code: the solver is called read-only and the returned ``Solution`` is
mutated in place via its own setters.

::

    size  -->  solve  -->  ramp  -->  floor
    (deflate   (FRAG,      (cap       (clamp deletes:
     speed/m)   read-only)  cold)      hold starved stages)

The driver thread only enqueues inputs (``update_with_measurements`` queues a
batch; ``set_queue_snapshot`` publishes per-stage input queue depths for
growth; ``set_activity_snapshot`` publishes per-stage queued + in-flight work
for the release gate); ``autoscale`` runs on a single executor thread and owns
all estimator and cross-cycle state mutation, so estimator state is never
shared across threads. Warmup-delete protection lives in the shared streaming
apply path (``StreamingSpecificSpec.scale_down_grace_after_ready_s``), not
here, so this scheduler holds no per-worker grace state.
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
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.activity import PipelineActivitySnapshot
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.estimator import PipelineRateEstimator
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.floor import (
    FloorInputs,
    FloorParams,
    FloorState,
    compute_floors,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.ramp import ColdStartRampPolicy, StageRampInput
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.sizing import BacklogDemandPolicy, StageSnapshot
from cosmos_xenna.utils import python_log as logger

_ALPHA_UP = 1.0
_GPU_RELEASE_SLOWDOWN = 2.0
_RELEASE_CONFIRM_DIVISOR = 3
_MIN_WORKERS = 1
_OVERALLOCATION_TARGET = 1.0


@attrs.frozen
class _FloorComputation:
    """Per-cycle floor outputs plus the stock signals behind each override.

    Attributes:
        floors: Per-stage scale-down floor (minimum worker count).
        queued_stock: Whole-chain stock from inter-stage queue depths only.
        active_stock: Whole-chain stock including pool-queued and in-flight work.
        active_depths: Per-stage active depth, in stage-input samples.
    """

    floors: tuple[int, ...]
    queued_stock: tuple[float, ...]
    active_stock: tuple[float, ...]
    active_depths: tuple[float, ...]


@attrs.define
class SaturationAwareScheduler:
    """Backlog-biased autoscaler built on the read-only fragmentation solver.

    Attributes:
        config: Operator tunables (cadence, catch-up cap, headroom, smoothing).
        stage_names: Canonical per-stage names in pipeline order.
        stage_batch_sizes: Per-stage input items consumed per batch.
        stage_gpu_fractions: Per-worker GPU shape per stage (0.0 for CPU stages).
            Drives the floor's slower release for GPU stages and is included in
            ramp logs for context.
    """

    config: SaturationAwareConfig
    stage_names: tuple[str, ...]
    stage_batch_sizes: tuple[int, ...]
    stage_gpu_fractions: tuple[float, ...]
    _estimator: PipelineRateEstimator = attrs.field(init=False)
    _floor_params: FloorParams = attrs.field(init=False)
    _floor_state: FloorState = attrs.field(init=False)
    _worker_id_factory: WorkerIdFactory = attrs.field(init=False, factory=WorkerIdFactory)
    _demand: BacklogDemandPolicy = attrs.field(init=False)
    _ramp: ColdStartRampPolicy = attrs.field(init=False)
    _problem: data_structures.Problem | None = attrs.field(init=False, default=None)
    _queue_snapshot: tuple[float, ...] = attrs.field(init=False, factory=tuple)
    _activity_snapshot: PipelineActivitySnapshot | None = attrs.field(init=False, default=None)
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
        self._ramp = ColdStartRampPolicy(config)

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
        """Record this cycle's per-stage input queue depths (drives growth sizing)."""
        self._queue_snapshot = tuple(float(depth) for depth in queue_depths)
        self._activity_snapshot = None

    def set_activity_snapshot(
        self,
        queue_depths: Sequence[float],
        pool_queued_tasks: Sequence[int],
        inflight_slots: Sequence[int],
        batch_sizes: Sequence[int],
    ) -> None:
        """Record this cycle's active-work snapshot from primitive counters."""
        self._activity_snapshot = PipelineActivitySnapshot.from_counts(
            queue_depths=queue_depths,
            pool_queued_tasks=pool_queued_tasks,
            inflight_slots=inflight_slots,
            batch_sizes=batch_sizes,
        )

    def autoscale(self, time: float, problem_state: data_structures.ProblemState) -> data_structures.Solution:
        """Size to backlog, solve placement via FRAG, then protect starved stages.

        Args:
            time: Decision timestamp, in seconds.
            problem_state: Current per-stage workers and slots.

        Returns:
            The fragmentation solver's solution with delete sets clamped so
            transiently-starved stages survive.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._problem is None:
            raise RuntimeError("setup() must be called before autoscale()")
        self._drain_pending_measurements()
        state_stages = list(problem_state.rust.stages)
        workers = [stage.num_workers() for stage in state_stages]

        estimates, speeds, returns, multipliers = self._size(time, workers)
        logger.debug(
            "saturation-aware demand: "
            + ", ".join(f"{name}={value:.2f}" for name, value in zip(self.stage_names, multipliers, strict=True))
        )
        solution = run_fragmentation_autoscaler(
            self._problem, problem_state, Estimates(estimates), _OVERALLOCATION_TARGET, self._worker_id_factory
        )
        self._apply_cold_start_ramp(solution, workers)
        computation = self._floors(workers, speeds, returns)
        self._protect(solution, workers, computation)
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

    def _apply_cold_start_ramp(self, solution: data_structures.Solution, workers: Sequence[int]) -> None:
        """Trim cold-start over-spawn of not-yet-trusted stages.

        Caps each untrusted stage's new-worker additions so the solver cannot
        make large resource commitments while it is still sizing the stage from
        placeholder throughput. Mutates the solution's ``new_workers`` in place
        via the native setters.

        Logging mirrors the scale-down (delete-override) path so the warm-up is
        debuggable end-to-end:

        - DEBUG ``saturation-aware ramp``: one per-cycle summary listing each
          stage the solver is growing, with its ramp decision (reason, sample
          progress, cap, solver-proposed new count, GPU fraction). This fires
          every cycle so the warm-up trajectory is visible even when nothing is
          trimmed.
        - INFO ``saturation-aware cold-start ramp``: one line per stage where
          the ramp actually overrides the solver, with the solver-vs-ramp
          new/post worker counts and the evidence that drove the cap.
        """
        min_data_points = self.config.speed_estimation_min_data_points
        stages = list(solution.rust.stages)
        changed = False
        summaries: list[str] = []
        for index, stage in enumerate(stages):
            new_workers = list(stage.new_workers)
            if not new_workers:
                continue
            name = self.stage_names[index]
            current = workers[index]
            deleted = len(stage.deleted_workers)
            gpu_fraction = self.stage_gpu_fractions[index]
            samples = self._estimator.sample_count(name)
            frag_new = len(new_workers)
            decision = self._ramp.decide(
                StageRampInput(
                    current_workers=current,
                    deleted_count=deleted,
                    proposed_post=current + frag_new - deleted,
                    sample_count=samples,
                )
            )
            summaries.append(
                f"{name}: {decision.reason} samples={samples}/{min_data_points} "
                f"cap={decision.cap} frag_new={frag_new} gpu_fraction={gpu_fraction}"
            )
            if decision.keep_new is None or decision.keep_new >= frag_new:
                continue
            ramp_new = decision.keep_new
            stage.new_workers = new_workers[:ramp_new]
            changed = True
            frag_post = current + frag_new - deleted
            ramp_post = current + ramp_new - deleted
            confidence = samples / min_data_points
            logger.info(
                f"saturation-aware cold-start ramp: stage='{name}' reason={decision.reason} "
                f"current={current} deleted={deleted} "
                f"frag_new={frag_new} frag_post={frag_post} "
                f"ramp_new={ramp_new} ramp_post={ramp_post} trimmed={frag_new - ramp_new} "
                f"cap={decision.cap} samples={samples}/{min_data_points} confidence={confidence:.2f} "
                f"gpu_fraction={gpu_fraction}"
            )
        if summaries:
            logger.debug("saturation-aware ramp: " + " | ".join(summaries))
        if changed:
            solution.rust.stages = stages

    def _floors(self, workers: Sequence[int], speeds: Sequence[float], returns: Sequence[float]) -> _FloorComputation:
        """Compute the per-stage scale-down floors from active pipeline stock.

        The release gate is driven by ``active_stock`` (queued backlog plus
        upstream pool-queued and in-flight work) rather than queue depth alone,
        so a downstream stage is not released while upstream work is still in
        flight. ``queued_stock`` is computed only for the override logs.
        """
        num_stages = len(workers)
        queue_depths = self._queue_for_cycle(num_stages)
        active_depths = self._active_depths_for_cycle(num_stages, queue_depths)
        chain_values = chain.chain_factors(returns, self.stage_batch_sizes)
        queued_stock = chain.whole_chain_stock(queue_depths, chain_values)
        active_stock = chain.whole_chain_stock(active_depths, chain_values)
        inputs = FloorInputs(
            workers=tuple(workers),
            speed=tuple(speeds),
            chain=tuple(chain_values),
            stock_src=tuple(active_stock),
            batch_sizes=self.stage_batch_sizes,
            is_gpu=tuple(fraction > 0.0 for fraction in self.stage_gpu_fractions),
        )
        result = compute_floors(inputs, self._floor_state, self._floor_params)
        self._floor_state = result.state
        return _FloorComputation(
            floors=result.floors,
            queued_stock=tuple(queued_stock),
            active_stock=tuple(active_stock),
            active_depths=tuple(active_depths),
        )

    def _protect(
        self, solution: data_structures.Solution, workers: Sequence[int], computation: _FloorComputation
    ) -> None:
        """Cap each stage's delete set to the scale-down floor.

        Trims the solver's deletes so the post-delete worker count stays at or
        above the stage floor, holding a transiently-starved expensive stage
        warm. Mutates the solution's ``deleted_workers`` in place via the native
        setters, logging each stage where it overrides the solver alongside the
        queued vs active stock that justified the floor.
        """
        floors = computation.floors
        rust_solution = solution.rust
        stages = list(rust_solution.stages)
        changed = False
        for index, stage in enumerate(stages):
            deletes = list(stage.deleted_workers)
            if not deletes:
                continue
            new_count = len(stage.new_workers)
            max_deletes = max(0, workers[index] + new_count - floors[index])
            final = deletes[:max_deletes]
            if len(final) == len(deletes):
                continue
            stage.deleted_workers = final
            changed = True
            frag_delete = len(deletes)
            sat_delete = len(final)
            frag_post = workers[index] + new_count - frag_delete
            sat_post = workers[index] + new_count - sat_delete
            logger.info(
                f"saturation-aware delete override: stage='{self.stage_names[index]}' "
                f"current={workers[index]} new={new_count} "
                f"frag_delete={frag_delete} frag_post={frag_post} "
                f"sat_delete={sat_delete} sat_post={sat_post} "
                f"floor_protected={frag_delete - sat_delete} floor_target={floors[index]} max_deletes={max_deletes} "
                f"queued_stock={computation.queued_stock[index]:.2f} "
                f"active_stock={computation.active_stock[index]:.2f} "
                f"active_depth={computation.active_depths[index]:.2f}"
            )
        if changed:
            rust_solution.stages = stages

    def _queue_for_cycle(self, num_stages: int) -> tuple[float, ...]:
        """Return the queue snapshot if it matches the stage count, else zeros."""
        if len(self._queue_snapshot) == num_stages:
            return self._queue_snapshot
        return (0.0,) * num_stages

    def _active_depths_for_cycle(self, num_stages: int, queue_depths: tuple[float, ...]) -> tuple[float, ...]:
        """Return per-stage active depths, falling back to ``queue_depths``.

        Falls back to the queue depths (never zeros) when no matching activity
        snapshot was published, so a non-empty queue can never read as drained
        and trigger a premature release.
        """
        snapshot = self._activity_snapshot
        if snapshot is not None and len(snapshot.stages) == num_stages:
            return snapshot.active_depths()
        return queue_depths

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
