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

"""Saturation-aware scheduler - the public class.

Implements the same public API as ``FragmentationBasedAutoscaler``
(``setup``, ``update_with_measurements``, ``autoscale``) so the two
can be swapped at ``Autoscaler.__init__`` via
``StreamingSpecificSpec.scheduler``.

``setup()`` seeds an empty per-stage ``_StageRuntimeState`` map.
Classifier thresholds are resolved lazily on the first
``autoscale()`` cycle: ``ProblemStageState.slots_per_worker``
supplies the M/M/c concurrency the resolver needs, and the
formula's ``K/sqrt(c)`` derivation is unstable until that value is
present. Once resolved, the values live on
``_StageRuntimeState.resolved_thresholds`` across ordinary cycles and
runtime ``slots_per_worker`` changes. A Halfin-Whitt regime transition
is the only re-resolution path: it clears the resolved thresholds and
threshold-relative classifier history before deriving the new band from
the updated effective aggressiveness.

``autoscale()`` constructs an ``AutoscalePlanContext`` per cycle so
per-stage decision logic can stage worker adds and removes against
a working cluster snapshot; the staged plan is frozen into a
``Solution`` via ``ctx.into_solution()``.
"""

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import (
    ResolvedThresholds,
    _resolve_auto_thresholds,
)
from cosmos_xenna.pipelines.private.scheduling_py.donor import DonorCandidate, select_youngest_eligible_donor
from cosmos_xenna.pipelines.private.scheduling_py.invariants import (
    PhaseBoundary,
    check_invariants_after_phase,
    check_solution_shape,
)
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import run_per_stage_pipeline
from cosmos_xenna.pipelines.private.scheduling_py.regime import (
    EXIT_BAND_MULTIPLIER,
    Regime,
    RegimeDetectorState,
    RegimeSignal,
    compute_regime_signal,
    update_regime_state,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


class SaturationAwareScheduler:
    """Pure-Python saturation-aware scheduler.

    Selected via ``StreamingSpecificSpec.scheduler == SchedulerKind.SATURATION_AWARE``.
    Replaces the Rust-backed ``FragmentationBasedAutoscaler`` at the
    algorithm layer; ``Autoscaler``-level concerns (deletion clamps,
    threading, measurement aggregation) remain owned by ``streaming.py``.

    Attributes:
        _config: Cluster-wide configuration. Per-stage configs are
            resolved on demand via
            ``SaturationAwareConfig.get_effective_stage_config``.
        _problem: Pipeline structure captured at ``setup()`` time. Held
            for use by the per-cycle decision pipeline (DAG depth,
            fixed stage indices); ``None`` until ``setup()`` is called.
        _stage_states: Per-stage runtime state, keyed by stage name.
            Populated at ``setup()``; mutated each cycle by the
            per-stage pipeline.
        _stage_names: Stage names in pipeline (DAG) order, captured at
            ``setup()``. Used to iterate stages deterministically.
        _regime_state: Cross-cycle hysteresis state for the
            Halfin-Whitt regime detector. Used to lift the effective
            ``saturation_aggressiveness`` when the cluster is packed
            close to capacity. Defaults to ``SUB_HALFIN_WHITT`` so
            cold-start cycles use the base aggressiveness.
        _worker_ages: Cross-cycle worker age snapshot keyed by worker
            id. Seeded into each planner context so manual shrink and
            floor donor fallback can prefer younger workers.
        _last_intent_deltas: Per-stage signed worker-count intent
            produced by the per-stage decision pipeline on the most
            recent ``autoscale()`` call. Saturation-driven scale-up
            (Phase C) and scale-down (Phase D) consume this map; the
            current iteration computes intent only and does not act
            on the values, so they are exposed for tests and
            observability. Reset on every ``autoscale()`` cycle.
            Finished stages are absent (skipped by
            ``_compute_intent_deltas``).

    """

    def __init__(self, config: SaturationAwareConfig) -> None:
        """Initialize the scheduler.

        Args:
            config: Cluster-wide ``SaturationAwareConfig``. Stored by
                reference; per-stage configs are resolved lazily via
                ``config.get_effective_stage_config``.

        """
        self._config = config
        self._problem: data_structures.Problem | None = None
        self._stage_states: dict[str, _StageRuntimeState] = {}
        self._stage_names: list[str] = []
        self._regime_state: RegimeDetectorState = RegimeDetectorState()
        self._worker_ages: dict[str, int] = {}
        self._last_intent_deltas: dict[str, int] = {}
        # Per-stage counter of consecutive cycles where floor enforcement
        # could not be satisfied because the cluster was full and no eligible
        # donor existed. Reset on any cycle where the receiver makes forward
        # progress; post-donation retry misses raise immediately because the
        # donor removal cannot be rolled back safely.
        self._floor_stuck_counters: dict[str, int] = {}

    def setup(self, problem: data_structures.Problem) -> None:
        """Capture the pipeline shape and seed empty per-stage state.

        Args:
            problem: The frozen pipeline ``Problem``. The Python
                wrapper does not expose stages directly; iterating
                ``.rust.stages`` gives stage names ahead of any
                runtime state arriving in ``autoscale()``.

        """
        self._problem = problem
        self._stage_names = [stage.name for stage in problem.rust.stages]
        self._stage_states = {name: _StageRuntimeState(stage_name=name) for name in self._stage_names}
        self._regime_state = RegimeDetectorState()
        self._worker_ages = {}
        self._floor_stuck_counters = {}
        self._last_intent_deltas = {}

    def _ensure_thresholds_resolved(self, problem_state: data_structures.ProblemState) -> None:
        """Lazily resolve per-stage classifier thresholds on the first cycle.

        Reads ``slots_per_worker`` from each stage's runtime
        ``ProblemStageState`` (the M/M/c concurrency), feeds it to the
        resolver with the regime-aware effective aggressiveness, stores
        the result on ``_StageRuntimeState``, and emits one INFO log
        line per stage. Stages whose ``resolved_thresholds`` is already
        populated short-circuit; mid-run ``slots_per_worker`` changes
        do not re-resolve. A regime transition (handled by
        ``_update_regime_aware_aggressiveness``) drops every stage's
        resolved thresholds and threshold-relative classifier history
        so this method re-derives them with the new effective
        aggressiveness on the same cycle.

        Raises:
            ValueError: If ``problem_state`` carries a stage name not
                present in the ``setup()`` state map (shape mismatch).

        """
        for stage in problem_state.rust.stages:
            runtime = self._stage_states.get(stage.stage_name)
            if runtime is None:
                msg = (
                    f"problem_state stage {stage.stage_name!r} not found in setup() "
                    f"state map (known: {sorted(self._stage_states)}); "
                    "problem and problem_state shapes disagree."
                )
                raise ValueError(msg)
            if runtime.resolved_thresholds is not None:
                continue
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=stage.stage_name,
                spec_override=None,
            )
            effective_aggressiveness = self._effective_aggressiveness(stage_cfg.saturation_aggressiveness)
            resolved = _resolve_auto_thresholds(
                stage_cfg,
                slots_per_actor=stage.slots_per_worker,
                aggressiveness_override=effective_aggressiveness,
            )
            runtime.resolved_thresholds = resolved
            self._log_resolved_thresholds(stage.stage_name, stage_cfg, resolved)

    def _effective_aggressiveness(self, base: float) -> float:
        """Apply the super-Halfin-Whitt lift to the base aggressiveness if active.

        Returns ``base`` unchanged when the cluster is in the
        sub-Halfin-Whitt regime or when the regime-aware lift is
        disabled. Returns ``base + super_halfin_whitt_aggressiveness_lift``
        when the cluster is in super-Halfin-Whitt and the lift is
        enabled.

        """
        if not self._config.enable_regime_aware_aggressiveness:
            return base
        if self._regime_state.current_regime is Regime.SUPER_HALFIN_WHITT:
            return base + self._config.super_halfin_whitt_aggressiveness_lift
        return base

    @staticmethod
    def _log_regime_transition(new_regime: Regime, signal: RegimeSignal, effective_aggressiveness: float) -> None:
        """Emit one INFO line per regime transition."""
        logger.info(
            f"scheduler regime transition: -> {new_regime.value} "
            f"(total_workers={signal.total_workers}, "
            f"cluster_idle_fraction={signal.cluster_idle_fraction:.4f}, "
            f"threshold={signal.threshold:.4f}, "
            f"effective_aggressiveness={effective_aggressiveness:.4f})"
        )

    @staticmethod
    def _log_resolved_thresholds(
        stage_name: str,
        stage_cfg: SaturationAwareStageConfig,
        resolved: ResolvedThresholds,
    ) -> None:
        """Emit one INFO line per stage describing the resolved thresholds."""
        sat_source = "manual override" if resolved.saturation_threshold_was_overridden else "auto"
        act_source = "manual override" if resolved.activation_threshold_was_overridden else "auto"
        logger.info(
            f"scheduler resolved auto thresholds for stage {stage_name!r}: "
            f"slots_per_actor={resolved.slots_per_actor}, "
            f"saturation_aggressiveness={resolved.saturation_aggressiveness}, "
            f"saturation_threshold={resolved.saturation_threshold:.6f} ({sat_source}), "
            f"activation_threshold={resolved.activation_threshold:.6f} ({act_source}), "
            f"over_provisioned_threshold={stage_cfg.over_provisioned_threshold} (config; not auto-derived)"
        )

    def update_with_measurements(
        self,
        time: float,
        measurements: data_structures.Measurements,
    ) -> None:
        """Ingest the latest measurement batch.

        The fragmentation-based scheduler uses this to feed its
        windowed throughput estimator. The saturation-aware scheduler
        is signal-driven from ``ProblemState.actor_pool_state``
        directly, so per-task measurements are not required for its
        own decisions. They are still accepted and discarded so the
        call sites in ``streaming.py`` need no special-case branch.

        Args:
            time: Current wall-clock time in seconds.
            measurements: Per-stage measurement batch since the previous
                cycle.

        """
        del time, measurements

    def autoscale(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Compute the autoscale plan for the current cycle.

        Args:
            time: Current wall-clock time in seconds.
            problem_state: Current per-stage runtime snapshot.

        Returns:
            A ``Solution`` with one ``StageSolution`` per stage in
            ``problem_state``, in the same order, carrying the existing
            ``slots_per_worker`` and any worker adds / removes staged
            on the cycle's ``AutoscalePlanContext``.

        Raises:
            RuntimeError: ``setup()`` was not called, planner context
                construction failed, or the non-manual worker floor
                cannot be satisfied by the current cluster.
            SchedulerInvariantError: A planner-context or
                Solution-shape invariant failed between phases; the
                plan is corrupted and must not be applied. Caller
                should treat as a must-fail signal: log, surface,
                and refuse to apply the plan. See
                ``scheduling_py/invariants.py``.
            ValueError: ``problem`` and ``problem_state`` disagree on
                stage names or count.

        """
        del time
        if self._problem is None:
            msg = "SaturationAwareScheduler.autoscale() called before setup()"
            raise RuntimeError(msg)

        self._update_regime_aware_aggressiveness(problem_state)
        self._ensure_thresholds_resolved(problem_state)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            self._problem,
            problem_state,
            worker_ages=self._next_cycle_worker_ages(),
        )
        self._run_phase_a_delete(ctx, problem_state)
        self._run_phase_a_grow(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_A, problem=self._problem, ctx=ctx)

        self._run_phase_b_floor(ctx, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_B, problem=self._problem, ctx=ctx)

        self._last_intent_deltas = self._compute_intent_deltas(ctx, problem_state)

        solution = ctx.into_solution()
        check_solution_shape(phase_name=PhaseBoundary.INTO_SOLUTION, problem=self._problem, solution=solution)
        self._persist_worker_ages(ctx)
        return solution

    def _next_cycle_worker_ages(self) -> dict[str, int]:
        """Build the planner's age seed for the next autoscale cycle.

        Ages count completed autoscale cycles. Surviving workers age by
        one when a new planning context is created; the Rust planner
        drops ids that are absent from the current ``ProblemState`` and
        assigns age 0 to newly observed workers.
        """
        return {worker_id: age + 1 for worker_id, age in self._worker_ages.items()}

    def _persist_worker_ages(self, ctx: data_structures.AutoscalePlanContext) -> None:
        """Persist live worker ages from a finalized planning context.

        Defensive against the Rust contract that worker ids returned by
        ``worker_ids_by_stage`` are also present in ``worker_ages``: a
        missing entry defaults to age 0 (treated as freshly observed)
        rather than raising ``KeyError`` mid-cycle. The fallback matches
        :func:`select_youngest_eligible_donor`'s missing-age semantics.
        """
        live_worker_ids = {worker_id for stage_ids in ctx.worker_ids_by_stage() for worker_id in stage_ids}
        worker_ages = ctx.worker_ages()
        self._worker_ages = {worker_id: worker_ages.get(worker_id, 0) for worker_id in live_worker_ids}

    def _compute_intent_deltas(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> dict[str, int]:
        """Compute the per-stage signed worker-count intent for this cycle.

        For each non-finished stage, calls
        :func:`run_per_stage_pipeline` with the live slot signals
        sourced from ``ProblemStageState`` and the post-Phase-B worker
        count read from the planner context. The returned delta is the
        algorithm's intent before any cluster-wide feasibility clamps.

        The intent values are exposed via :attr:`_last_intent_deltas`
        for tests and observability. Saturation-driven scale-up
        (Phase C) and scale-down (Phase D) -- not yet implemented --
        will consume these intents and stage worker adds / removes on
        the planner context. Finished stages are absent from the
        returned map: their classifier short-circuits upstream of
        ``run_per_stage_pipeline`` and any drain-state slot signal
        would only mutate per-stage EWMA state without affecting the
        plan.

        Args:
            ctx: The cycle's mutable planner context after Phase B.
                Read-only here (intent only); ``current_workers`` is
                sourced from ``ctx.worker_ids_by_stage()`` so the
                pipeline observes the post-Phase-B worker count.
            problem_state: The cycle's runtime snapshot. Provides the
                three slot signals (``num_used_slots``,
                ``num_empty_slots``, ``input_queue_depth``) populated
                in the streaming layer.

        Returns:
            Mapping of stage name -> signed intent. Positive values
            indicate scale-up intent, negative values scale-down,
            zero a no-op. Finished stages are absent.
        """
        intents: dict[str, int] = {}
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        for stage_index, runtime_stage in enumerate(problem_state.rust.stages):
            if runtime_stage.is_finished:
                continue
            stage_name = runtime_stage.stage_name
            stage_state = self._stage_states[stage_name]
            stage_cfg = self._config.get_effective_stage_config(stage_name=stage_name, spec_override=None)
            current_workers = len(worker_ids_by_stage[stage_index])
            delta = run_per_stage_pipeline(
                stage_state=stage_state,
                num_used_slots=runtime_stage.num_used_slots,
                num_empty_slots=runtime_stage.num_empty_slots,
                input_queue_depth=runtime_stage.input_queue_depth,
                current_workers=current_workers,
                config=stage_cfg,
            )
            intents[stage_name] = delta
        return intents

    def _run_phase_a_delete(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Shrink manual stages whose ``requested_num_workers`` is below current.

        Manual stages are those with ``requested_num_workers`` set.
        For unfinished manual stages whose current worker count exceeds
        the request, surplus workers are staged for deletion through
        ``ctx.try_remove_worker``.

        Raises:
            RuntimeError: ``try_remove_worker`` returned ``False`` for
                a worker that was present in ``problem_state`` -- a
                planner-state inconsistency that must surface
                immediately.

        """
        if self._problem is None:
            msg = "_run_phase_a_delete called before setup()"
            raise RuntimeError(msg)
        worker_ages = ctx.worker_ages()
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            requested = problem_stage.requested_num_workers
            if requested is None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            current = len(runtime_stage.worker_groups)
            if current <= requested:
                continue
            delete_count = current - requested
            worker_ids = [w.id for w in runtime_stage.worker_groups]
            # Delete youngest workers first so long-lived warmed workers survive manual shrink.
            victims = _select_workers_to_delete_youngest_first(
                worker_ids=worker_ids,
                worker_ages=worker_ages,
                delete_count=delete_count,
            )
            for worker_id in victims:
                if not ctx.try_remove_worker(stage_index, worker_id):
                    msg = (
                        f"Manual-shrink: try_remove_worker(stage_index={stage_index}, "
                        f"worker_id={worker_id!r}) returned False on stage "
                        f"{problem_stage.name!r}; the worker was present in problem_state "
                        "but unknown to the planner - snapshot inconsistency."
                    )
                    raise RuntimeError(msg)

    def _run_phase_a_grow(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Grow manual stages whose ``requested_num_workers`` is above current.

        Manual stages are those with ``requested_num_workers`` set.
        For unfinished manual stages whose current worker count is
        below the request, fresh workers are staged through
        ``ctx.try_add_worker`` until the request is met. When the
        working cluster has no remaining placement (returns ``None``),
        growth stops for that stage in this cycle without raising, and
        a single WARNING per affected stage is emitted so operators
        retain visibility into partially-satisfied manual requests.

        Raises:
            RuntimeError: The scheduler has not been set up, or the
                planner reports a corrupted/drained planning context.
            IndexError: The planner rejects the stage index.

        """
        if self._problem is None:
            msg = "_run_phase_a_grow called before setup()"
            raise RuntimeError(msg)
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            requested = problem_stage.requested_num_workers
            if requested is None:
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            current = len(runtime_stage.worker_groups)
            while current < requested:
                if ctx.try_add_worker(stage_index) is None:
                    deficit = requested - current
                    logger.warning(
                        f"manual grow: stage {problem_stage.name!r} requested "
                        f"{requested} workers; cluster placement exhausted at "
                        f"{current} (deficit={deficit}); manual request remains "
                        "partially satisfied this cycle."
                    )
                    break
                current += 1

    def _run_phase_b_floor(
        self,
        ctx: data_structures.AutoscalePlanContext,
        problem_state: data_structures.ProblemState,
    ) -> None:
        """Enforce the per-stage minimum-worker floor on non-manual stages.

        For each non-manual, non-finished stage, brings the current
        worker count up to ``_compute_stage_floors``'s ``target_min``.
        Each shortfall is filled by ``ctx.try_add_worker``; on
        capacity exhaustion the cross-stage donor fallback
        (:func:`select_youngest_eligible_donor`) reallocates one
        worker from another stage and the add is retried. A no-donor
        floor miss accumulates a per-stage stuck-cycle counter;
        ``RuntimeError`` is raised only when the counter exceeds
        ``floor_stuck_grace_cycles``. The counter resets on any cycle
        where the receiver makes forward progress. A post-donation
        retry miss raises immediately because the planner state already
        removed the donor and cannot be rolled back safely.

        Raises:
            RuntimeError: ``setup()`` was not called; OR the floor
                has stayed unsatisfied without progress for more than
                ``floor_stuck_grace_cycles`` consecutive cycles; OR a
                post-donation retry misses; OR the planner rejects a
                staged remove of a worker that is live in the snapshot
                (scheduler defect).
            IndexError: The planner rejects the stage index.

        """
        if self._problem is None:
            msg = "_run_phase_b_floor called before setup()"
            raise RuntimeError(msg)
        num_nodes = self._problem.rust.cluster_resources.num_nodes()
        stage_floors = self._compute_stage_floors(num_nodes)
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            if problem_stage.requested_num_workers is not None:
                # Manual stages have their worker count owned by the manual-shrink/grow path.
                continue
            runtime_stage = problem_state.rust.stages[stage_index]
            if runtime_stage.is_finished:
                continue
            target_min = stage_floors[stage_index]
            # Read the receiver's count from the planner's live snapshot so any
            # mutation by an earlier-iteration donation is observed correctly.
            current = len(ctx.worker_ids_by_stage()[stage_index])
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=problem_stage.name,
                spec_override=None,
            )
            made_progress = False
            stuck = False
            while current < target_min:
                if ctx.try_add_worker(stage_index) is not None:
                    current += 1
                    made_progress = True
                    continue
                # Cluster is full for the receiver's shape; attempt the
                # cross-stage donor fallback before deciding whether to raise.
                donor = select_youngest_eligible_donor(
                    receiver_stage_index=stage_index,
                    stage_floors=stage_floors,
                    worker_ids_by_stage=ctx.worker_ids_by_stage(),
                    worker_ages=ctx.worker_ages(),
                )
                if donor is None:
                    if made_progress and self._config.floor_stuck_grace_cycles > 0:
                        self._warn_floor_partial_progress(
                            stage_name=problem_stage.name,
                            target_min=target_min,
                            current=current,
                        )
                    else:
                        self._on_floor_stuck(
                            stage_name=problem_stage.name,
                            target_min=target_min,
                            current=current,
                            stage_cfg=stage_cfg,
                            num_nodes=num_nodes,
                        )
                        stuck = True
                    break
                if not ctx.try_remove_worker(donor.stage_index, donor.worker_id):
                    msg = (
                        f"Cross-stage floor donor: planner snapshot inconsistency - "
                        f"try_remove_worker(stage_index={donor.stage_index}, "
                        f"worker_id={donor.worker_id!r}) returned False even though the "
                        "worker is present in the live snapshot. This is a scheduler "
                        "defect; report it with the autoscale cycle's problem_state."
                    )
                    raise RuntimeError(msg)
                if ctx.try_add_worker(stage_index) is None:
                    msg = self._format_floor_unmet_message(
                        stage_name=problem_stage.name,
                        target_min=target_min,
                        current=current,
                        stage_cfg=stage_cfg,
                        num_nodes=num_nodes,
                        donor_attempted=True,
                        donor=donor,
                    )
                    raise RuntimeError(msg)
                logger.info(
                    f"[scheduler] {problem_stage.name!r}: cross-stage minimum-floor donor "
                    f"accepted (donor_stage_index={donor.stage_index}, "
                    f"donor_worker_id={donor.worker_id!r}, donor_age={donor.age})"
                )
                current += 1
                made_progress = True
            if made_progress or not stuck:
                # Floor satisfied this cycle (either by direct add, by donation, or
                # because target_min was already met at entry), or the receiver made
                # partial progress; reset the stuck counter.
                self._floor_stuck_counters.pop(problem_stage.name, None)

    @staticmethod
    def _warn_floor_partial_progress(
        *,
        stage_name: str,
        target_min: int,
        current: int,
    ) -> None:
        """Warn when a floor is still short but the receiver grew this cycle."""
        logger.warning(
            f"[scheduler] {stage_name!r}: minimum-worker floor partially satisfied "
            f"(target_min={target_min}, achieved={current}, no eligible cross-stage donor); "
            "stuck counter reset because the receiver made forward progress."
        )

    def _on_floor_stuck(
        self,
        *,
        stage_name: str,
        target_min: int,
        current: int,
        stage_cfg: SaturationAwareStageConfig,
        num_nodes: int,
    ) -> None:
        """Account for a single stuck-cycle and either raise or warn.

        Increments the per-stage stuck counter, raises ``RuntimeError``
        when the counter exceeds ``floor_stuck_grace_cycles``, and
        otherwise emits a single WARNING per stuck cycle so operators
        can correlate transient pressure with eventual escalation.

        Args:
            stage_name: Receiver stage that did not reach its floor.
            target_min: Configured minimum for the stage.
            current: Worker count achieved before the failure.
            stage_cfg: Effective per-stage config (used for the
                operator-actionable error message).
            num_nodes: Cluster node count at this cycle.

        Raises:
            RuntimeError: The per-stage stuck counter has exceeded
                ``floor_stuck_grace_cycles``. Message identifies the
                stage, target floor, both source values, and tells
                the operator to reduce the floor or scale up the
                cluster.

        """
        counter = self._floor_stuck_counters.get(stage_name, 0) + 1
        grace = self._config.floor_stuck_grace_cycles
        if counter > grace:
            msg = self._format_floor_unmet_message(
                stage_name=stage_name,
                target_min=target_min,
                current=current,
                stage_cfg=stage_cfg,
                num_nodes=num_nodes,
                donor_attempted=False,
                donor=None,
            )
            raise RuntimeError(msg)
        self._floor_stuck_counters[stage_name] = counter
        remaining = grace - counter
        logger.warning(
            f"[scheduler] {stage_name!r}: minimum-worker floor stuck "
            f"({counter}/{grace} grace cycles) -- target_min={target_min}, "
            f"achieved={current}, no eligible cross-stage donor; will raise after "
            f"{remaining} more consecutive failed cycles."
        )

    def _compute_stage_floors(self, num_nodes: int) -> dict[int, int]:
        """Compute the donor / receiver floor for every stage in the problem.

        Returns ``target_min`` per stage (problem-stage index → floor)::

            target_min = max(
                stage_cfg.min_workers if stage_cfg.min_workers is not None else 1,
                (stage_cfg.min_workers_per_node * num_nodes) if stage_cfg.min_workers_per_node is not None else 0,
            )

        Applied uniformly to manual and non-manual stages. A manual
        stage may donate down to its configured ``min_workers`` floor;
        ``requested_num_workers`` is a target, not a hard lower bound.
        Operators who need a hard floor must set ``min_workers`` to the
        desired minimum.

        """
        if self._problem is None:
            msg = "_compute_stage_floors called before setup()"
            raise RuntimeError(msg)
        floors: dict[int, int] = {}
        for stage_index, problem_stage in enumerate(self._problem.rust.stages):
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=problem_stage.name,
                spec_override=None,
            )
            floors[stage_index] = max(
                stage_cfg.min_workers if stage_cfg.min_workers is not None else 1,
                stage_cfg.min_workers_per_node * num_nodes if stage_cfg.min_workers_per_node is not None else 0,
            )
        return floors

    @staticmethod
    def _format_floor_unmet_message(
        *,
        stage_name: str,
        target_min: int,
        current: int,
        stage_cfg: SaturationAwareStageConfig,
        num_nodes: int,
        donor_attempted: bool,
        donor: DonorCandidate | None,
    ) -> str:
        """Build the operator-actionable message for an unmet minimum-worker floor.

        Distinguishes the two failure modes (no eligible donor vs.
        donor selected but retry still failed) and -- when a donor was
        selected -- names that donor so operators can correlate the
        failure with the donation log line.
        """
        if donor_attempted:
            donor_label = (
                f" (donor_stage_index={donor.stage_index}, donor_worker_id={donor.worker_id!r})"
                if donor is not None
                else ""
            )
            donor_clause = (
                "donor fallback attempted but post-donation retry returned no placement"
                f"{donor_label} -- the donor's freed slot does not match the receiver's shape"
            )
        else:
            donor_clause = "no eligible cross-stage donor (every other stage at its own floor)"
        return (
            f"Minimum-worker floor for stage {stage_name!r} cannot be satisfied: "
            f"target_min={target_min} (achieved={current}; from "
            f"min_workers={stage_cfg.min_workers}, "
            f"min_workers_per_node={stage_cfg.min_workers_per_node}, "
            f"num_nodes={num_nodes}). Cluster placement exhausted and "
            f"{donor_clause}. Reduce min_workers / min_workers_per_node, "
            "or scale up the cluster."
        )

    def _update_regime_aware_aggressiveness(self, problem_state: data_structures.ProblemState) -> None:
        """Detect the cluster's Halfin-Whitt regime and re-resolve thresholds on transition.

        Computes the per-cycle regime signal, applies hysteresis via
        ``update_regime_state``, and -- on a regime transition -- drops
        every stage's ``resolved_thresholds`` and threshold-relative
        classifier history so the next call to
        ``_ensure_thresholds_resolved`` re-derives thresholds with the
        appropriate effective aggressiveness. Cycles whose signal is
        unavailable (some active stage has not populated
        ``num_used_slots`` / ``num_empty_slots`` yet) leave the regime
        state and resolved thresholds untouched. Respects the
        ``enable_regime_aware_aggressiveness`` flag.

        """
        if not self._config.enable_regime_aware_aggressiveness:
            return

        signal = _aggregate_cluster_regime_signal(problem_state)
        transitioned = update_regime_state(
            self._regime_state,
            signal,
            streak_cycles=self._config.regime_transition_streak_cycles,
            exit_band_multiplier=EXIT_BAND_MULTIPLIER,
        )
        if not transitioned:
            return

        effective = self._effective_aggressiveness(self._config.stage_defaults.saturation_aggressiveness)
        self._log_regime_transition(self._regime_state.current_regime, signal, effective)
        for runtime in self._stage_states.values():
            runtime.resolved_thresholds = None
            runtime.classifier_state = StageState.NORMAL
            runtime.classifier_streak = 0


def _select_workers_to_delete_youngest_first(
    *,
    worker_ids: list[str],
    worker_ages: dict[str, int],
    delete_count: int,
) -> list[str]:
    """Pick ``delete_count`` workers to delete, youngest first.

    Sort key: ``(age ASC, worker_id ASC)``. Workers missing from
    ``worker_ages`` are treated as age 0 (newly observed). The
    ``worker_id`` tiebreaker keeps the choice deterministic when
    every worker has the same age.

    Args:
        worker_ids: Worker ids in the stage's current snapshot.
        worker_ages: Cluster-wide worker ages.
        delete_count: Number of workers to return. Clamped to
            ``len(worker_ids)``.

    Returns:
        The first ``delete_count`` worker ids of the youngest-first
        ordering.

    """
    ranked = sorted(
        ((worker_ages.get(wid, 0), wid) for wid in worker_ids),
        key=lambda pair: (pair[0], pair[1]),
    )
    return [wid for _, wid in ranked[:delete_count]]


def _aggregate_cluster_regime_signal(problem_state: data_structures.ProblemState) -> RegimeSignal:
    """Aggregate the cluster's regime-detection inputs from per-stage state.

    Sums worker counts, used slots, and empty slots across every stage
    in ``problem_state``, then delegates to ``compute_regime_signal``.
    The returned signal is unavailable when any active worker stage
    still carries the ``0/0`` no-signal sentinel, or when no stage
    reports any slot occupancy at all.
    """
    total_workers = 0
    total_used = 0
    total_empty = 0
    for stage in problem_state.rust.stages:
        stage_workers = len(stage.worker_groups)
        stage_slots = stage.num_used_slots + stage.num_empty_slots
        if stage_workers > 0 and not stage.is_finished and stage_slots == 0:
            return compute_regime_signal(
                total_workers=total_workers + stage_workers,
                total_used_slots=0,
                total_empty_slots=0,
            )
        total_workers += stage_workers
        total_used += stage.num_used_slots
        total_empty += stage.num_empty_slots
    return compute_regime_signal(
        total_workers=total_workers,
        total_used_slots=total_used,
        total_empty_slots=total_empty,
    )
