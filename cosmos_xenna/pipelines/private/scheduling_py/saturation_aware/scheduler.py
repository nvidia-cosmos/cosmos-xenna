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

"""Saturation-aware scheduler: capacity-target sizing over the fragmentation solver.

Each cycle the scheduler builds one pipeline throughput / capacity model
(``capacity.py``) and uses it to drive both growth and shrink. Demand sizing
(``sizing.py``) deflates a stage's speed estimate only enough to grow it toward
its capacity target ``w_target``, so the shared fragmentation solver is never
handed an inflated ask; the returned solution is then post-processed to ramp
cold stages and to clamp deletes to each stage's capacity hold target
``w_sustain``. It edits no fragmentation-scheduler code: the solver is called
read-only and the returned ``Solution`` is mutated through a
:class:`SolutionEditor`.

::

    snapshot --> capacity --> demand --> solve --> ramp --> floor --> commit
    (trusted    (cap_src,    (mult to   (FRAG,    (cap      (clamp     (write
     speed)      rates,       w_target)  read-     cold)     deletes    edited
                 targets)                only)               to         Solution)
                                                             w_sustain)

The driver thread only enqueues inputs: ``update_with_measurements`` queues a
measurement batch and ``observe_runtime`` publishes the per-stage runtime
signals (queue depth for growth; pool-queued and in-flight work for the
release gate). ``autoscale`` runs on a single executor thread and owns all
estimator, capacity, and floor state mutation, so that state is never shared
across threads.
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
from cosmos_xenna.pipelines.private.scheduling_py.runtime_signals import RuntimeSignals
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.activity import PipelineActivitySnapshot
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.capacity import (
    CapacityInputs,
    CapacityModel,
    CapacityParams,
    CapacityPlan,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.estimator import PipelineRateEstimator
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.floor import (
    FloorInputs,
    FloorParams,
    FloorPlan,
    ScaleDownFloorPolicy,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.problem_template import SolverProblemTemplate
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.ramp import (
    ColdStartRampPolicy,
    RampReason,
    StageRampInput,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.shape import PipelineShape
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.sizing import (
    CapacityDemandPolicy,
    DemandResult,
    StageDemandSnapshot,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.solution_editor import SolutionEditor
from cosmos_xenna.utils import python_log as logger

_ALPHA_UP = 1.0
_GPU_RELEASE_SLOWDOWN = 4.0
_RELEASE_CONFIRM_DIVISOR = 3
_MIN_WORKERS = 1
_OVERALLOCATION_TARGET = 1.0
# Bottleneck hysteresis: a challenger must be >=15% slower than the incumbent
# for 2 consecutive cycles before it takes over, so a one-cycle cap_src dip
# (for example a transient floor cut) cannot flap the pipeline target rate.
_HYSTERESIS_MARGIN = 0.15
_SWITCH_CONFIRM = 2


@attrs.define
class SaturationAwareScheduler:
    """Capacity-target autoscaler built on the read-only fragmentation solver.

    Attributes:
        config: Operator tunables (cadence, capacity headroom, estimator
            window, scale-down release cycles).
        shape: Static per-stage pipeline shape (names, batch sizes, GPU flags).
        solver_template: Rebuilds the solver problem with per-stage request
            overrides, used to hold pinned stages at their current size when
            the cluster cannot grow them to their target.
    """

    config: SaturationAwareConfig
    shape: PipelineShape
    solver_template: SolverProblemTemplate
    _estimator: PipelineRateEstimator = attrs.field(init=False)
    _capacity: CapacityModel = attrs.field(init=False)
    _floor: ScaleDownFloorPolicy = attrs.field(init=False)
    _worker_id_factory: WorkerIdFactory = attrs.field(init=False, factory=WorkerIdFactory)
    _demand: CapacityDemandPolicy = attrs.field(init=False)
    _ramp: ColdStartRampPolicy = attrs.field(init=False)
    _problem: data_structures.Problem | None = attrs.field(init=False, default=None)
    _first_decision_time: float | None = attrs.field(init=False, default=None)
    _queue_snapshot: tuple[float, ...] = attrs.field(init=False, factory=tuple)
    _activity_snapshot: PipelineActivitySnapshot | None = attrs.field(init=False, default=None)
    _pending_measurements: queue.Queue[data_structures.Measurements] = attrs.field(init=False, factory=queue.Queue)

    def __attrs_post_init__(self) -> None:
        """Derive the estimator, capacity, demand, ramp, and floor policies from the config."""
        config = self.config
        self._estimator = PipelineRateEstimator(
            config.speed_estimation_window_s, config.speed_estimation_min_data_points
        )
        cycles = config.scale_down_release_cycles
        self._capacity = CapacityModel.create(
            self.shape.num_stages,
            CapacityParams(
                alpha_up=_ALPHA_UP,
                alpha_down_cpu=1.0 / cycles,
                alpha_down_gpu=1.0 / (cycles * _GPU_RELEASE_SLOWDOWN),
                capacity_headroom=config.capacity_headroom,
                hysteresis_margin=_HYSTERESIS_MARGIN,
                switch_confirm=_SWITCH_CONFIRM,
                min_workers=_MIN_WORKERS,
            ),
        )
        self._floor = ScaleDownFloorPolicy.create(
            self.shape.num_stages,
            FloorParams(
                release_confirm_cycles=max(1, math.ceil(cycles / _RELEASE_CONFIRM_DIVISOR)),
                min_workers=_MIN_WORKERS,
            ),
        )
        self._demand = CapacityDemandPolicy()
        self._ramp = ColdStartRampPolicy(config)

    def setup(self, problem: data_structures.Problem) -> None:
        """Record the static problem; the solver needs it each cycle."""
        self._problem = problem

    def update_with_measurements(self, time: float, measurements: data_structures.Measurements) -> None:
        """Queue completed-task timings for ingestion at the next autoscale.

        Runs on the driver thread; the batch is folded into the estimator inside
        :meth:`autoscale` so all estimator access stays on the single executor
        thread. ``time`` is unused here (the estimator windows by each task's own
        timestamp); it is part of the shared scheduler interface.
        """
        self._pending_measurements.put(measurements)

    def observe_runtime(self, signals: RuntimeSignals) -> None:
        """Record this cycle's runtime work signals.

        Growth sizing reads the input queue depths; the scale-down release gate
        reads the whole active snapshot (queued plus pool-queued plus in-flight).

        Args:
            signals: Per-stage queue, pool-queued, in-flight, and batch counts.

        Raises:
            ValueError: If the signal length does not match the stage count.
        """
        num_stages = self.shape.num_stages
        if len(signals.queue_depths) != num_stages:
            raise ValueError(f"runtime signals cover {len(signals.queue_depths)} stages, expected {num_stages}")
        self._queue_snapshot = signals.queue_depths
        self._activity_snapshot = PipelineActivitySnapshot.from_counts(
            queue_depths=signals.queue_depths,
            pool_queued_tasks=signals.pool_queued_tasks,
            inflight_slots=signals.inflight_slots,
            batch_sizes=signals.batch_sizes,
        )

    def autoscale(self, time: float, problem_state: data_structures.ProblemState) -> data_structures.Solution:
        """Model capacity, size to target, solve via FRAG, ramp, then floor.

        Args:
            time: Decision timestamp, in seconds.
            problem_state: Current per-stage workers and slots.

        Returns:
            The fragmentation solver's solution with new/deleted worker sets
            clamped so transiently-starved stages survive.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._problem is None:
            raise RuntimeError("setup() must be called before autoscale()")
        if self._first_decision_time is None:
            # Anchor every stage's warmup clock to the first decision. The
            # pipeline shape is static, so all stages exist from this cycle;
            # the cold-start ramp uses the elapsed time to release a stage that
            # has produced no sample within a full speed-estimation window.
            self._first_decision_time = time
        self._drain_pending_measurements()
        workers = [stage.num_workers() for stage in problem_state.rust.stages]

        # One throughput model per cycle, shared by growth and the floor.
        snapshots = self._demand_snapshots(time, workers)
        returns = [self._demand.resolve_num_returns(snapshot) for snapshot in snapshots]
        batch_sizes = tuple(stage.batch_size for stage in self.shape.stages)
        chain_values = chain.chain_factors(returns, batch_sizes)
        queue_depths = self._queue_for_cycle(len(workers))
        active_depths = self._active_depths_for_cycle(len(workers), queue_depths)
        queued_stock = chain.whole_chain_stock(queue_depths, chain_values)
        active_stock = chain.whole_chain_stock(active_depths, chain_values)
        capacity = self._capacity.plan(
            CapacityInputs(
                workers=tuple(workers),
                # A cold / untrusted stage reports 0.0 so it is excluded from
                # the bottleneck and the cold-start ramp keeps owning it.
                speed=tuple(max(0.0, snapshot.speed or 0.0) for snapshot in snapshots),
                chain=tuple(chain_values),
                is_gpu=tuple(stage.is_gpu for stage in self.shape.stages),
            )
        )

        # Grow toward w_target only when real whole-chain stock is waiting.
        sizings = [
            self._demand.size(
                snapshots[index],
                capacity.stages[index],
                active_stock[index] > self._stock_threshold(index, chain_values),
            )
            for index in range(len(workers))
        ]
        estimates = [Estimate(sizing.effective_speed, sizing.num_returns) for sizing in sizings]
        solution = self._solve(problem_state, Estimates(estimates), workers)
        editor = SolutionEditor(solution)
        self._apply_cold_start_ramp(editor, workers, time)
        floor_plan = self._apply_scale_down_floor(
            editor, workers, capacity, active_stock, active_depths, chain_values, batch_sizes
        )
        self._log_decision_snapshot(workers, sizings, capacity, floor_plan, queued_stock, active_stock)
        editor.commit()
        return solution

    def _solve(
        self,
        problem_state: data_structures.ProblemState,
        estimates: Estimates,
        workers: Sequence[int],
    ) -> data_structures.Solution:
        """Solve placement, retrying once with pinned stages held at current size.

        A pinned stage's count is a hard solver constraint enforced before any
        donor borrowing, so a saturated cluster makes the full target infeasible
        and the solver raises. Holding every pinned stage at its current count
        satisfies that Phase-1 constraint, letting later cycles grow toward the
        target as resources free. A pinned stage held at zero workers is logged
        at ERROR (it will not run this cycle); a held stage that still has
        workers is merely degraded.

        Raises:
            RuntimeError: If there are no pinned stages to relax, or if the
                retry is still infeasible (for example a non-pinned stage cannot
                place its mandatory first worker).
        """
        assert self._problem is not None
        try:
            return run_fragmentation_autoscaler(
                self._problem, problem_state, estimates, _OVERALLOCATION_TARGET, self._worker_id_factory
            )
        except RuntimeError as exc:
            overrides = {stage.name: workers[index] for index, stage in enumerate(self.shape.stages) if stage.is_manual}
            if not overrides:
                raise
            stalled = sorted(name for name, held in overrides.items() if held < _MIN_WORKERS)
            if stalled:
                logger.error(
                    f"saturation-aware: pinned stage(s) {stalled} cannot place a worker on the current cluster "
                    f"and will not run this cycle; reduce the pinned count or free cluster resources"
                )
            logger.warning(
                f"saturation-aware solve infeasible ({exc}); holding pinned stages "
                f"at current size and retrying: {overrides}"
            )
            return run_fragmentation_autoscaler(
                self.solver_template.build(overrides),
                problem_state,
                estimates,
                _OVERALLOCATION_TARGET,
                self._worker_id_factory,
            )

    def _demand_snapshots(self, now: float, workers: Sequence[int]) -> list[StageDemandSnapshot]:
        """Build one demand snapshot per stage from the estimator.

        The per-worker speed is gated through :meth:`_trusted_speed` so a stage
        with too few samples reports ``None`` (cold) and is excluded from both
        the bottleneck and demand growth until it is trusted.
        """
        return [
            StageDemandSnapshot(
                name=stage.name,
                workers=workers[index],
                speed=self._trusted_speed(stage.name, now),
                num_returns=self._estimator.num_returns(stage.name),
                batch_size=stage.batch_size,
            )
            for index, stage in enumerate(self.shape.stages)
        ]

    def _trusted_speed(self, name: str, now: float) -> float | None:
        """Return the measured speed once trusted, else ``None``.

        Mirrors the cold-start ramp's trust threshold
        (``speed_estimation_min_data_points``) so a single noisy early sample
        cannot set the bottleneck identity or rate before the stage has
        enough samples to be believed.
        """
        if self._estimator.sample_count(name) < self.config.speed_estimation_min_data_points:
            return None
        return self._estimator.speed(name, now)

    def _stock_threshold(self, index: int, chain_values: Sequence[float]) -> float:
        """Return the source-unit stock above which stage ``index`` has real work.

        One batch's worth of source items (``batch_size / chain``); below it the
        stage's whole-chain stock is treated as drained, matching the floor's
        release threshold so growth and release agree on "has work".
        """
        factor = chain_values[index]
        return self.shape.stages[index].batch_size / factor if factor > 0.0 else 0.0

    def _apply_cold_start_ramp(self, editor: SolutionEditor, workers: Sequence[int], now: float) -> None:
        """Trim cold-start over-spawn of not-yet-trusted stages.

        Caps each untrusted stage's new-worker additions so the solver cannot
        make large commitments while it is still sizing the stage from
        placeholder throughput. A stage that has work waiting but produces no
        sample within a full speed-estimation window is released to the solver
        (slow-starter). Logs a per-cycle DEBUG summary of every grown stage and
        an INFO line for each stage it trims or releases as a slow-starter.
        """
        min_data_points = self.config.speed_estimation_min_data_points
        first_decision_time = self._first_decision_time if self._first_decision_time is not None else now
        stage_age_s = now - first_decision_time
        num_stages = len(workers)
        active_depths = self._active_depths_for_cycle(num_stages, self._queue_for_cycle(num_stages))
        summaries: list[str] = []
        for index, stage in enumerate(self.shape.stages):
            if stage.is_manual:
                # Operator pinned this count; the evidence ramp has nothing to
                # ramp toward, so leave the solver's proposal for it intact.
                continue
            frag_new = editor.proposed_new_workers(index)
            if frag_new == 0:
                continue
            current = workers[index]
            deleted = editor.proposed_deletes(index)
            samples = self._estimator.sample_count(stage.name)
            has_pending_work = active_depths[index] > 0.0
            decision = self._ramp.decide(
                StageRampInput(
                    current_workers=current,
                    deleted_count=deleted,
                    proposed_post=current + frag_new - deleted,
                    sample_count=samples,
                    stage_age_s=stage_age_s,
                    has_pending_work=has_pending_work,
                )
            )
            summaries.append(
                f"{stage.name}: {decision.reason} samples={samples}/{min_data_points} "
                f"cap={decision.cap} frag_new={frag_new} is_gpu={stage.is_gpu}"
            )
            if decision.reason is RampReason.SLOW_START:
                # No sample within the warmup window but work is still waiting:
                # the ramp is trusting the solver to spawn all workers rather
                # than trimming. Surface it at INFO so the cause of the
                # (intentional) large spawn is traceable.
                logger.info(
                    f"saturation-aware cold-start ramp: stage='{stage.name}' reason={decision.reason} "
                    f"current={current} frag_new={frag_new} "
                    f"stage_age_s={stage_age_s:.1f} window_s={self.config.speed_estimation_window_s:.1f} "
                    f"samples={samples}/{min_data_points} has_pending_work={has_pending_work} "
                    f"active_depth={active_depths[index]:.2f} is_gpu={stage.is_gpu} "
                    f"(no sample within window and work waiting; trusting solver)"
                )
            if decision.keep_new is None or decision.keep_new >= frag_new:
                continue
            if not editor.trim_new_workers(index, decision.keep_new):
                continue
            ramp_new = decision.keep_new
            frag_post = current + frag_new - deleted
            ramp_post = current + ramp_new - deleted
            confidence = samples / min_data_points
            logger.info(
                f"saturation-aware cold-start ramp: stage='{stage.name}' reason={decision.reason} "
                f"current={current} deleted={deleted} "
                f"frag_new={frag_new} frag_post={frag_post} "
                f"ramp_new={ramp_new} ramp_post={ramp_post} trimmed={frag_new - ramp_new} "
                f"cap={decision.cap} samples={samples}/{min_data_points} confidence={confidence:.2f} "
                f"is_gpu={stage.is_gpu}"
            )
        if summaries:
            logger.debug("saturation-aware ramp: " + " | ".join(summaries))

    def _apply_scale_down_floor(
        self,
        editor: SolutionEditor,
        workers: Sequence[int],
        capacity: CapacityPlan,
        active_stock: Sequence[float],
        active_depths: Sequence[float],
        chain_values: Sequence[float],
        batch_sizes: Sequence[int],
    ) -> FloorPlan:
        """Clamp each stage's delete set to its capacity hold floor.

        Feeds the capacity plan's per-stage ``w_sustain`` to the release gate,
        which is driven by active stock (queued backlog plus upstream
        pool-queued and in-flight work) so a downstream stage is not released
        while upstream work is still in flight. Trims the solver's deletes so the
        post-delete worker count stays at or above the floor, logging each
        overridden stage alongside the capacity signals (speed, cap_src,
        sustainable arrival, the bottleneck ladder) that justified it.

        Returns:
            The cycle's :class:`FloorPlan`, reused by the decision snapshot.
        """
        plan = self._floor.plan(
            FloorInputs(
                workers=tuple(workers),
                chain=tuple(chain_values),
                stock_src=tuple(active_stock),
                active_depths=tuple(active_depths),
                batch_sizes=tuple(batch_sizes),
                w_sustain=tuple(stage.w_sustain for stage in capacity.stages),
            )
        )
        bottleneck_name = (
            self.shape.stages[capacity.bottleneck_stage].name if capacity.bottleneck_stage >= 0 else "none"
        )
        for index in range(editor.stage_count):
            frag_delete = editor.proposed_deletes(index)
            if frag_delete == 0:
                continue
            decision = plan.decisions[index]
            new_count = editor.proposed_new_workers(index)
            max_deletes = max(0, workers[index] + new_count - decision.floor)
            if not editor.cap_deletes(index, max_deletes):
                continue
            sat_delete = editor.proposed_deletes(index)
            cap = capacity.stages[index]
            frag_post = workers[index] + new_count - frag_delete
            sat_post = workers[index] + new_count - sat_delete
            logger.info(
                f"saturation-aware delete override: stage='{self.shape.stages[index].name}' "
                f"current={workers[index]} new={new_count} "
                f"frag_delete={frag_delete} frag_post={frag_post} "
                f"sat_delete={sat_delete} sat_post={sat_post} "
                f"floor_protected={frag_delete - sat_delete} floor_target={decision.floor} max_deletes={max_deletes} "
                f"releasing={decision.releasing} speed={cap.speed:.4f} cap_src={cap.cap_src:.3f} "
                f"a_raw={cap.a_raw:.2f} a_ewma={cap.a_ewma:.2f} w_sustain={cap.w_sustain} w_target={cap.w_target} "
                f"bottleneck_rate={capacity.bottleneck_rate:.3f} "
                f"next_bottleneck_rate={capacity.next_bottleneck_rate:.3f} "
                f"bottleneck='{bottleneck_name}' "
                f"active_stock={active_stock[index]:.2f} active_depth={active_depths[index]:.2f}"
            )
        return plan

    def _log_decision_snapshot(
        self,
        workers: Sequence[int],
        sizings: Sequence[DemandResult],
        capacity: CapacityPlan,
        floor_plan: FloorPlan,
        queued_stock: Sequence[float],
        active_stock: Sequence[float],
    ) -> None:
        """Emit one per-cycle DEBUG line capturing every stage's decision signals.

        Folds the capacity model (cap_src, bottleneck and next-bottleneck
        rates, bottleneck stage, w_sustain / w_target), the demand multiplier,
        the floor decision, the queued and active whole-chain stock, and
        in-flight slot utilization into one record so a scheduler cycle is
        debuggable from logs alone.
        """
        snapshot = self._activity_snapshot
        has_activity = snapshot is not None and len(snapshot.stages) == len(workers)
        bottleneck_name = (
            self.shape.stages[capacity.bottleneck_stage].name if capacity.bottleneck_stage >= 0 else "none"
        )
        groups: list[str] = []
        for index, stage in enumerate(self.shape.stages):
            cap = capacity.stages[index]
            decision = floor_plan.decisions[index]
            inflight = snapshot.stages[index].inflight_slots if has_activity else 0
            pool_queued = snapshot.stages[index].pool_queued_tasks if has_activity else 0
            utilization = inflight / max(workers[index], 1)
            groups.append(
                f"{stage.name}[w={workers[index]} cap_src={cap.cap_src:.2f} "
                f"w_sustain={cap.w_sustain} w_target={cap.w_target} mult={sizings[index].multiplier:.2f} "
                f"floor={decision.floor} releasing={decision.releasing} "
                f"q_stock={queued_stock[index]:.1f} a_stock={active_stock[index]:.1f} "
                f"inflight={inflight} pool_q={pool_queued} util={utilization:.2f}]"
            )
        logger.debug(
            f"saturation-aware decision: bottleneck_rate={capacity.bottleneck_rate:.3f} "
            f"next_bottleneck_rate={capacity.next_bottleneck_rate:.3f} "
            f"bottleneck='{bottleneck_name}' | " + " | ".join(groups)
        )

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
            if index >= self.shape.num_stages:
                break
            name = self.shape.stages[index].name
            for task in stage_measurements.task_measurements:
                self._estimator.observe(
                    name,
                    duration_s=task.end_time - task.start_time,
                    num_returns=float(task.num_returns),
                    now=task.start_time,
                )
