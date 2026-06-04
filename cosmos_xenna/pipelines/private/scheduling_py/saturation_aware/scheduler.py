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
from typing import Any, Self, cast

import attrs

from cosmos_xenna.pipelines.private import data_structures, resources, specs
from cosmos_xenna.pipelines.private.autoscaling_algorithms import (
    Estimate,
    Estimates,
    WorkerIdFactory,
    run_fragmentation_autoscaler,
)
from cosmos_xenna.pipelines.private.scheduling_py.runtime_signals import RuntimeSignals
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain, ramp, sizing
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
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.ramp import RampReason, StageRampInput
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.shape import PipelineShape
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
_FEEDER_BOOST_MAX_MULTIPLIER = 2.0


@attrs.frozen
class _Cycle:
    """Immutable per-cycle derived inputs for one ``autoscale`` pass.

    Built once by :meth:`SaturationAwareScheduler._build_cycle` from scheduler
    state and the live problem state, then threaded through capacity, sizing,
    ramp, floor, and logging so every collaborator reads the same per-stage
    numbers instead of recomputing them. Holds derived inputs only -- no policy
    state advances here.

    Attributes:
        time: Decision timestamp, in seconds.
        stage_age_s: Seconds since the first decision, for the cold-start ramp.
        workers: Current (pre-solve) worker count per stage.
        demand_snapshots: Per-stage estimator snapshot (speed, returns, batch).
        batch_sizes: Per-stage input items consumed per batch.
        chain_factors: Per-stage cumulative fan-out from the source.
        is_gpu: Per-stage GPU flag (drives the capacity release alpha).
        is_manual: Per-stage manual pin flag (blocks feeder-pressure boosts).
        local_depths: Per-stage input queue depth, in stage-input samples.
        local_pending_depths: Per-stage queue plus pool-queued depth, excluding
            in-flight work, in stage-input samples.
        active_depths: Per-stage active work depth (queued, pool, in-flight).
        ready_workers: Per-stage workers not holding an in-flight slot.
        queued_stock: Per-stage queued whole-chain stock, in source units.
        active_stock: Per-stage active whole-chain stock, in source units.
        activity_snapshot: Raw runtime snapshot for this cycle, or ``None`` when
            no runtime signals were observed (used only for decision logging).
    """

    time: float
    stage_age_s: float
    workers: tuple[int, ...]
    demand_snapshots: tuple[sizing.StageDemandSnapshot, ...]
    batch_sizes: tuple[int, ...]
    chain_factors: tuple[float, ...]
    is_gpu: tuple[bool, ...]
    is_manual: tuple[bool, ...]
    local_depths: tuple[float, ...]
    local_pending_depths: tuple[float, ...]
    active_depths: tuple[float, ...]
    ready_workers: tuple[int, ...]
    queued_stock: tuple[float, ...]
    active_stock: tuple[float, ...]
    activity_snapshot: PipelineActivitySnapshot | None

    def has_work(self, index: int) -> bool:
        """Return whether stage ``index`` has real whole-chain stock waiting.

        Uses the same one-batch source-stock boundary as the scale-down floor,
        so growth and release agree on the "has work" line.
        """
        threshold = chain.source_stock_threshold(self.batch_sizes[index], self.chain_factors[index])
        return self.active_stock[index] > threshold

    def has_local_input(self, index: int) -> bool:
        """Return whether local pending input can use another worker."""
        return self.local_pending_depths[index] > float(self.batch_sizes[index])

    def capacity_inputs(self) -> CapacityInputs:
        """Return the capacity model's inputs for this cycle.

        A cold / untrusted stage reports speed ``0.0`` so it is excluded from
        the bottleneck and the cold-start ramp keeps owning it.
        """
        return CapacityInputs(
            workers=self.workers,
            speed=tuple(max(0.0, snapshot.speed or 0.0) for snapshot in self.demand_snapshots),
            chain=self.chain_factors,
            is_gpu=self.is_gpu,
            is_manual=self.is_manual,
            local_qin=self.local_depths,
            local_pending_depth=self.local_pending_depths,
            local_input_threshold=tuple(float(batch_size) for batch_size in self.batch_sizes),
            active_depth=self.active_depths,
            ready_workers=self.ready_workers,
        )

    def floor_inputs(self, capacity: CapacityPlan) -> FloorInputs:
        """Return the scale-down release gate's inputs for this cycle.

        Pairs the cycle's active whole-chain stock and depths with the capacity
        plan's per-stage ``w_sustain`` hold target.
        """
        return FloorInputs(
            workers=self.workers,
            chain=self.chain_factors,
            stock_src=self.active_stock,
            active_depths=self.active_depths,
            batch_sizes=self.batch_sizes,
            w_sustain=tuple(stage.w_sustain for stage in capacity.stages),
            is_gpu=self.is_gpu,
        )


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
    _problem: data_structures.Problem | None = attrs.field(init=False, default=None)
    _first_decision_time: float | None = attrs.field(init=False, default=None)
    _runtime_snapshot: PipelineActivitySnapshot | None = attrs.field(init=False, default=None)
    _pending_measurements: queue.Queue[data_structures.Measurements] = attrs.field(init=False, factory=queue.Queue)

    def __attrs_post_init__(self) -> None:
        """Derive the estimator, capacity, and floor policies from the config.

        Demand sizing and the cold-start ramp are stateless module functions
        (``sizing`` and ``ramp``), so only the stateful policies are built here.
        """
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
                feeder_pressure_confirm=_SWITCH_CONFIRM,
                feeder_arrival_horizon_s=config.interval_s,
                feeder_boost_max_multiplier=_FEEDER_BOOST_MAX_MULTIPLIER,
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

    @classmethod
    def from_pipeline_spec(
        cls,
        pipeline_spec: specs.PipelineSpec,
        cluster_resources: resources.ClusterResources,
        config: SaturationAwareConfig,
    ) -> Self:
        """Build a scheduler from a pipeline spec, hiding the shape/solver wiring.

        Narrows the spec's stages and builds the static :class:`PipelineShape`
        and :class:`SolverProblemTemplate` so the caller depends only on the
        scheduler abstraction. The caller still invokes :meth:`setup`.

        Args:
            pipeline_spec: Pipeline whose stages define the static shape.
            cluster_resources: Cluster the solver template plans against.
            config: Operator tunables for the scheduler.

        Returns:
            A constructed (not yet set up) :class:`SaturationAwareScheduler`.
        """
        stages = [cast(specs.StageSpec[Any, Any], stage) for stage in pipeline_spec.stages]
        return cls(
            config=config,
            shape=PipelineShape.from_stage_specs(stages),
            solver_template=SolverProblemTemplate.from_stage_specs(stages, cluster_resources),
        )

    def setup(self, problem: data_structures.Problem) -> None:
        """Record the static problem; the solver needs it each cycle."""
        self._problem = problem

    def _ensure_setup(self) -> data_structures.Problem:
        """Return the problem recorded by :meth:`setup`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._problem is None:
            raise RuntimeError("setup() must be called before autoscale()")
        return self._problem

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
        # One snapshot serves both signals: the input queue depth drives growth
        # sizing, and the whole active snapshot drives the scale-down release.
        self._runtime_snapshot = PipelineActivitySnapshot.from_counts(
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
        problem = self._ensure_setup()
        self._drain_pending_measurements()

        cycle = self._build_cycle(time, problem_state)
        capacity = self._capacity.plan(cycle.capacity_inputs())
        sizings = sizing.size_pipeline(cycle.demand_snapshots, capacity, cycle.has_local_input)
        estimates = Estimates([Estimate(result.effective_speed, result.num_returns) for result in sizings])

        solution = self._solve(problem, problem_state, estimates, cycle.workers)
        frag_new, frag_delete = self._solution_counts(solution)
        self._apply_cold_start_ramp(solution, cycle)
        sat_new, _ = self._solution_counts(solution)
        floor_plan = self._apply_scale_down_floor(solution, cycle, capacity)
        _, sat_delete = self._solution_counts(solution)

        self._log_decision_snapshot(cycle, sizings, capacity, floor_plan, frag_new, frag_delete, sat_new, sat_delete)
        return solution

    def _build_cycle(self, time: float, problem_state: data_structures.ProblemState) -> _Cycle:
        """Assemble this cycle's immutable derived inputs once.

        Reads scheduler-owned state (shape, estimator, runtime snapshot, warmup
        anchor) and the live problem state, then derives every per-stage value
        the downstream policies share: worker counts, demand snapshots, chain
        factors, queued and active depths, and their whole-chain stock. Anchors
        ``_first_decision_time`` on the first call (lifecycle state) but runs no
        capacity, sizing, placement, ramp, or floor logic and advances no
        cross-cycle policy state.

        Args:
            time: Decision timestamp, in seconds.
            problem_state: Current per-stage workers and slots.

        Returns:
            The immutable :class:`_Cycle` for this decision.
        """
        if self._first_decision_time is None:
            # Anchor every stage's warmup clock to the first decision. The
            # pipeline shape is static, so all stages exist from this cycle; the
            # cold-start ramp uses the elapsed time to release a stage that has
            # produced no sample within a full speed-estimation window.
            self._first_decision_time = time
        workers = tuple(stage.num_workers() for stage in problem_state.rust.stages)
        num_stages = len(workers)
        snapshots = tuple(self._demand_snapshots(time, workers))
        returns = [sizing.resolve_num_returns(snapshot) for snapshot in snapshots]
        batch_sizes = tuple(stage.batch_size for stage in self.shape.stages)
        chain_factors = tuple(chain.chain_factors(returns, batch_sizes))
        runtime = self._runtime_snapshot
        if runtime is not None and len(runtime.stages) == num_stages:
            # Growth reads input queue depth; release reads the whole active
            # snapshot (queued plus pool-queued plus in-flight).
            queue_depths: tuple[float, ...] = tuple(stage.queue_depth_samples for stage in runtime.stages)
            local_pending_depths: tuple[float, ...] = tuple(
                stage.queue_depth_samples + stage.pool_queued_tasks * stage.batch_size for stage in runtime.stages
            )
            active_depths: tuple[float, ...] = runtime.active_depths()
            ready_workers: tuple[int, ...] = tuple(
                max(workers[index] - stage.inflight_slots, 0) for index, stage in enumerate(runtime.stages)
            )
        else:
            # No runtime signal observed yet: treat every depth as drained.
            queue_depths = (0.0,) * num_stages
            local_pending_depths = (0.0,) * num_stages
            active_depths = (0.0,) * num_stages
            ready_workers = workers
        return _Cycle(
            time=time,
            stage_age_s=time - self._first_decision_time,
            workers=workers,
            demand_snapshots=snapshots,
            batch_sizes=batch_sizes,
            chain_factors=chain_factors,
            is_gpu=tuple(stage.is_gpu for stage in self.shape.stages),
            is_manual=tuple(stage.is_manual for stage in self.shape.stages),
            local_depths=queue_depths,
            local_pending_depths=local_pending_depths,
            active_depths=active_depths,
            ready_workers=ready_workers,
            queued_stock=tuple(chain.whole_chain_stock(queue_depths, chain_factors)),
            active_stock=tuple(chain.whole_chain_stock(active_depths, chain_factors)),
            activity_snapshot=runtime,
        )

    def _solution_counts(self, solution: data_structures.Solution) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return proposed new-worker and delete counts from a solution."""
        editor = SolutionEditor(solution)
        return (
            tuple(editor.proposed_new_workers(index) for index in range(editor.stage_count)),
            tuple(editor.proposed_deletes(index) for index in range(editor.stage_count)),
        )

    def _solve(
        self,
        problem: data_structures.Problem,
        problem_state: data_structures.ProblemState,
        estimates: Estimates,
        workers: Sequence[int],
    ) -> data_structures.Solution:
        """Solve placement, retrying once with pinned stages held at current size.

        A pinned stage's count is a hard solver constraint enforced before any
        donor borrowing, so a saturated cluster makes the full target infeasible
        and the solver raises. Holding every pinned stage at its current count
        satisfies that hard constraint, letting later cycles grow toward the
        target as resources free. A pinned stage held at zero workers is logged
        at ERROR (it will not run this cycle); a held stage that still has
        workers is merely degraded.

        Args:
            problem: The static solver problem recorded by :meth:`setup`.
            problem_state: Current per-stage workers and slots.
            estimates: Per-stage solver speed and fan-out for this cycle.
            workers: Current (pre-solve) worker count per stage, used to hold
                pinned stages at their current size on the infeasible retry.

        Returns:
            The fragmentation solver's placement solution.

        Raises:
            RuntimeError: If there are no pinned stages to relax, or if the
                retry is still infeasible (for example a non-pinned stage cannot
                place its mandatory first worker).
        """
        try:
            return run_fragmentation_autoscaler(
                problem, problem_state, estimates, _OVERALLOCATION_TARGET, self._worker_id_factory
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

    def _demand_snapshots(self, now: float, workers: Sequence[int]) -> list[sizing.StageDemandSnapshot]:
        """Build one demand snapshot per stage from the estimator.

        The per-worker speed is gated through :meth:`_trusted_speed` so a stage
        with too few samples reports ``None`` (cold) and is excluded from both
        the bottleneck and demand growth until it is trusted.
        """
        return [
            sizing.StageDemandSnapshot(
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

    def _bottleneck_name(self, capacity: CapacityPlan) -> str:
        """Return the bottleneck stage's name, or ``"none"`` when no stage is trusted yet."""
        if capacity.bottleneck_stage >= 0:
            return self.shape.stages[capacity.bottleneck_stage].name
        return "none"

    def _apply_cold_start_ramp(self, solution: data_structures.Solution, cycle: _Cycle) -> None:
        """Trim cold-start over-spawn of not-yet-trusted stages.

        Caps each untrusted stage's new-worker additions so the solver cannot
        make large commitments while it is still sizing the stage from
        placeholder throughput. A stage that has work waiting but produces no
        sample within a full speed-estimation window is released to the solver
        (slow-starter).

        Runs before :meth:`_apply_scale_down_floor`, committing its own editor
        so the floor reads these trims.

        Args:
            solution: The solver's solution, mutated in place via its own editor.
            cycle: This cycle's immutable derived inputs (workers, depths, age).
        """
        editor = SolutionEditor(solution)
        min_data_points = self.config.speed_estimation_min_data_points
        stage_age_s = cycle.stage_age_s
        summaries: list[str] = []
        for index, stage in enumerate(self.shape.stages):
            if stage.is_manual:
                # Operator pinned this count; the evidence ramp has nothing to
                # ramp toward, so leave the solver's proposal for it intact.
                continue
            frag_new = editor.proposed_new_workers(index)
            if frag_new == 0:
                continue
            current = cycle.workers[index]
            deleted = editor.proposed_deletes(index)
            samples = self._estimator.sample_count(stage.name)
            has_pending_work = cycle.active_depths[index] > 0.0
            # Pipeline evidence: any upstream stage with a trusted (non-None)
            # speed proves the chain is feeding this stage, letting a 0-sample
            # stage grow by one worker per cycle before its own first sample.
            # "Any upstream" (rather than only the immediate feeder) is
            # deliberate: the +1/cycle bound and the has_pending_work gate
            # already contain growth, so the broader signal only front-loads a
            # deep stage's warmup sooner without widening the cycle-1-dark
            # fragmentation window.
            has_upstream_evidence = any(snapshot.speed is not None for snapshot in cycle.demand_snapshots[:index])
            decision = ramp.decide(
                StageRampInput(
                    current_workers=current,
                    deleted_count=deleted,
                    proposed_post=current + frag_new - deleted,
                    sample_count=samples,
                    stage_age_s=stage_age_s,
                    has_pending_work=has_pending_work,
                    has_upstream_evidence=has_upstream_evidence,
                ),
                self.config,
            )
            summaries.append(
                f"{stage.name}: {decision.reason} samples={samples}/{min_data_points} "
                f"cap={decision.cap} frag_new={frag_new} is_gpu={stage.is_gpu} "
                f"has_pending_work={has_pending_work} has_upstream_evidence={has_upstream_evidence}"
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
                    f"active_depth={cycle.active_depths[index]:.2f} is_gpu={stage.is_gpu} "
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
                f"is_gpu={stage.is_gpu} "
                f"has_pending_work={has_pending_work} has_upstream_evidence={has_upstream_evidence}"
            )
        if summaries:
            logger.debug("saturation-aware ramp: " + " | ".join(summaries))
        editor.commit()

    def _apply_scale_down_floor(
        self, solution: data_structures.Solution, cycle: _Cycle, capacity: CapacityPlan
    ) -> FloorPlan:
        """Clamp each stage's delete set to its capacity hold floor.

        Feeds the capacity plan's per-stage ``w_sustain`` to the release gate,
        which is driven by active stock (queued backlog plus upstream
        pool-queued and in-flight work) so a downstream stage is not released
        while upstream work is still in flight. Trims the solver's deletes so the
        post-delete worker count stays at or above the floor.

        Runs after :meth:`_apply_cold_start_ramp` and on its own editor, so the
        post-solve ``new_workers`` it reads already reflect that stage's trims.

        Args:
            solution: The solver's solution, mutated in place via its own editor.
            cycle: This cycle's immutable derived inputs (workers, stock, depths).
            capacity: This cycle's capacity plan (per-stage hold targets).

        Returns:
            The cycle's :class:`FloorPlan`, reused by the decision snapshot.
        """
        editor = SolutionEditor(solution)
        plan = self._floor.plan(cycle.floor_inputs(capacity))
        bottleneck_name = self._bottleneck_name(capacity)
        for index in range(editor.stage_count):
            frag_delete = editor.proposed_deletes(index)
            if frag_delete == 0:
                continue
            decision = plan.decisions[index]
            new_count = editor.proposed_new_workers(index)
            max_deletes = max(0, cycle.workers[index] + new_count - decision.floor)
            if not editor.cap_deletes(index, max_deletes):
                continue
            sat_delete = editor.proposed_deletes(index)
            cap = capacity.stages[index]
            frag_post = cycle.workers[index] + new_count - frag_delete
            sat_post = cycle.workers[index] + new_count - sat_delete
            logger.info(
                f"saturation-aware delete override: stage='{self.shape.stages[index].name}' "
                f"current={cycle.workers[index]} new={new_count} "
                f"frag_delete={frag_delete} frag_post={frag_post} "
                f"sat_delete={sat_delete} sat_post={sat_post} "
                f"floor_protected={frag_delete - sat_delete} floor_target={decision.floor} max_deletes={max_deletes} "
                f"releasing={decision.releasing} churn_guarded={decision.churn_guarded} "
                f"speed={cap.speed:.4f} cap_src={cap.cap_src:.3f} "
                f"a_raw={cap.a_raw:.2f} a_ewma={cap.a_ewma:.2f} w_sustain={cap.w_sustain} w_target={cap.w_target} "
                f"bottleneck_rate={capacity.bottleneck_rate:.3f} "
                f"next_bottleneck_rate={capacity.next_bottleneck_rate:.3f} "
                f"bottleneck_streak={capacity.bottleneck_streak} "
                f"bottleneck='{bottleneck_name}' "
                f"active_stock={cycle.active_stock[index]:.2f} active_depth={cycle.active_depths[index]:.2f}"
            )
        editor.commit()
        return plan

    def _log_decision_snapshot(
        self,
        cycle: _Cycle,
        sizings: Sequence[sizing.StageSizingResult],
        capacity: CapacityPlan,
        floor_plan: FloorPlan,
        frag_new: Sequence[int],
        frag_delete: Sequence[int],
        sat_new: Sequence[int],
        sat_delete: Sequence[int],
    ) -> None:
        """Emit one per-cycle DEBUG line summarizing every stage's decision signals.

        Folds the capacity, demand, floor, stock, and slot-utilization signals
        for each stage into a single record so a scheduler cycle is debuggable
        from logs alone.
        """
        snapshot = cycle.activity_snapshot
        activity_stages = (
            snapshot.stages if snapshot is not None and len(snapshot.stages) == len(cycle.workers) else None
        )
        bottleneck_name = self._bottleneck_name(capacity)
        groups: list[str] = []
        for index, stage in enumerate(self.shape.stages):
            cap = capacity.stages[index]
            decision = floor_plan.decisions[index]
            if activity_stages is not None:
                inflight = activity_stages[index].inflight_slots
                pool_queued = activity_stages[index].pool_queued_tasks
            else:
                inflight = 0
                pool_queued = 0
            utilization = inflight / max(cycle.workers[index], 1)
            if cap.starved_warm or cap.feeder_boost > 0 or cap.feeder_reason:
                feeder_name = self.shape.stages[cap.binding_feeder].name if cap.binding_feeder >= 0 else "none"
                downstreams = tuple(self.shape.stages[item].name for item in cap.feeder_downstreams)
                logger.debug(
                    f"saturation-aware feeder-pressure: stage='{stage.name}' reason='{cap.feeder_reason}' "
                    f"starved_warm={cap.starved_warm} suppress_growth={cap.suppress_growth} "
                    f"local_qin={cycle.local_depths[index]:.2f} "
                    f"local_pending={cycle.local_pending_depths[index]:.2f} "
                    f"local_threshold={float(cycle.batch_sizes[index]):.2f} "
                    f"workers={cycle.workers[index]} ready={cycle.ready_workers[index]} "
                    f"w_sustain={cap.w_sustain} active_depth={cycle.active_depths[index]:.2f} "
                    f"binding_feeder='{feeder_name}' path_delay_s={cap.feeder_path_delay_s:.2f} "
                    f"required_workers={cap.feeder_required_workers} boost_cap={cap.feeder_boost_cap} "
                    f"feeder_boost={cap.feeder_boost} downstreams={downstreams} aggregation='max'"
                )
            groups.append(
                f"{stage.name}[w={cycle.workers[index]} frag_new={frag_new[index]} sat_new={sat_new[index]} "
                f"frag_del={frag_delete[index]} sat_del={sat_delete[index]} cap_src={cap.cap_src:.2f} "
                f"w_sustain={cap.w_sustain} w_target={cap.w_target} mult={sizings[index].multiplier:.2f} "
                f"starved={cap.starved_warm} suppress={cap.suppress_growth} "
                f"feeder_boost={cap.feeder_boost} feeder_reason='{cap.feeder_reason}' "
                f"floor={decision.floor} churn_guarded={decision.churn_guarded} releasing={decision.releasing} "
                f"local_qin={cycle.local_depths[index]:.1f} local_pending={cycle.local_pending_depths[index]:.1f} "
                f"ready={cycle.ready_workers[index]} "
                f"q_stock={cycle.queued_stock[index]:.1f} a_stock={cycle.active_stock[index]:.1f} "
                f"active_depth={cycle.active_depths[index]:.1f} inflight={inflight} pool_q={pool_queued} "
                f"util={utilization:.2f}]"
            )
        logger.debug(
            f"saturation-aware decision: bottleneck_rate={capacity.bottleneck_rate:.3f} "
            f"next_bottleneck_rate={capacity.next_bottleneck_rate:.3f} "
            f"bottleneck='{bottleneck_name}' bottleneck_streak={capacity.bottleneck_streak} "
            f"bottleneck_candidate={capacity.bottleneck_candidate} "
            f"bottleneck_candidate_rate={capacity.bottleneck_candidate_rate:.3f} | " + " | ".join(groups)
        )

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
