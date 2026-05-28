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

Implements the ``FragmentationBasedAutoscaler`` API
(``setup`` / ``update_with_measurements`` / ``autoscale``) so the
two can be swapped via ``StreamingSpecificSpec.scheduler``.

Per-cycle flow lives in the ``phases/`` subpackage; this module
owns only the public protocol, cross-cycle state, and the
``AutoscaleCycle`` build helper. Per-phase narrow service value
objects live on the ``CycleRunner``; the scheduler exposes
``ledgers`` (cross-cycle mutable state) and ``pipeline`` (immutable
post-setup shape) so callers route reads through the canonical
owners rather than through a forwarder chain. See
``docs/scheduler/saturation-aware/`` for the algorithm
specification.
"""

import types
from collections.abc import Mapping

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.cluster.measurements import MeasurementCollector
from cosmos_xenna.pipelines.private.scheduling_py.cluster.memory_pressure import MemoryPressureMonitor
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.donor.economic_gate import EconomicGate
from cosmos_xenna.pipelines.private.scheduling_py.donor.executor import DonorBackedAddExecutor
from cosmos_xenna.pipelines.private.scheduling_py.donor.policy import FloorPolicy, SaturationPolicy
from cosmos_xenna.pipelines.private.scheduling_py.donor.resource_fit import ResourceFitPlanner
from cosmos_xenna.pipelines.private.scheduling_py.invariants.suite import PhaseInvariantSuite
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.cycle_finalizer import CycleFinalizer
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.cycle_runner import CycleRunner
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.growth_recorder import GrowthModeRecorder
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.loop_watchdog import loop_watchdog
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.post_cycle import PostCycleReporter, StuckPlanInvariant
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.preflight import PreflightBuilder
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.bottleneck_phase import BottleneckPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.services import BottleneckServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.floor_phase import FloorPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.services import FloorServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.grow_phase import SaturationGrowPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.services import GrowServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.capacity import (
    CapacityModel,
    CeilingCalculator,
    FloorCalculator,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase import IntentPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.services import IntentServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stage_decision_pipeline import StageDecisionPipeline
from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.executors import (
    ManualDeleteExecutor,
    ManualGrowExecutor,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.manual_phase import ManualPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.services import ManualServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.services import ShrinkServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.shrink_phase import SaturationShrinkPhase
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime import RegimeDetectorState
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller import RegimeController
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.allocation_failure_gate import AllocationFailureGate
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_protection import BottleneckProtectionLogger
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver import ThresholdResolver
from cosmos_xenna.pipelines.private.scheduling_py.warmup.warmup import WarmupTracker
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


class SaturationAwareScheduler:
    """Pure-Python saturation-aware scheduler.

    Selected via
    ``StreamingSpecificSpec.scheduler == SchedulerKind.SATURATION_AWARE``.
    Owns the public protocol (``setup``,
    ``update_with_measurements``, ``autoscale``), cross-cycle state
    (per-stage runtime state, worker ages, donor cooldown ledgers,
    recommendation histories, regime detector, stuck-plan counters,
    measurement collector), and the ``AutoscaleCycle`` build helper.
    Per-cycle decision logic lives in the ``phases/`` subpackage;
    ``streaming.py`` retains autoscaler-level concerns (deletion
    clamps, threading, measurement aggregation).

    The post-setup composition is intentionally explicit on the
    facade: ``ledgers`` (cross-cycle mutable state),
    ``pipeline`` (immutable post-setup shape), and ``runner``
    (per-phase narrow services + phase pipeline driver) are the
    three public attributes through which observability code and
    tests reach the live scheduler internals. The facade no longer
    re-exports every ledger field via ``@property`` forwarders.
    """

    def __init__(
        self,
        config: SaturationAwareConfig,
        stage_spec_overrides: dict[str, SaturationAwareStageConfig] | None = None,
        *,
        pipeline_name: str = "",
    ) -> None:
        """Initialize the scheduler.

        Construction is split into two phases:

        - ``__init__`` (this method) stores the cluster-wide
          configuration, the validated ``StageSpec`` override map,
          and the cross-cycle ``SchedulerLedgers``. It does NOT
          build per-cycle helpers that need a frozen pipeline
          shape; those depend on the ``PipelineModel`` and are
          assembled inside ``setup()``.
        - ``setup(problem)`` builds the immutable ``PipelineModel``
          from ``problem`` plus the stored config / overrides and
          wires every helper (preflight, runner, finalizer,
          capacity calculators, threshold resolver, post-cycle
          reporter, phase invariant suite) to that model. After
          ``setup()`` returns, the facade no longer reasons about
          ``problem is None``.

        Constructor-time validation surfaces misconfigured
        per-stage overrides before ``setup()`` (and the per-cycle
        hot path). ``stage_spec_overrides`` outrank the named
        overrides on ``config.per_stage_overrides`` and the
        cluster defaults; ``None`` and ``{}`` are equivalent.

        Raises:
            ValueError: An override violates a cluster-wide
                guardrail.

        """
        self._config = config
        self._pipeline_name: str = pipeline_name
        # ``stage_spec_overrides`` is validated eagerly so a misconfigured
        # pipeline cannot reach the per-cycle hot path with a weakened
        # cross-stage donor anti-flap guardrail. The empty-input fast path
        # skips the validator call to keep constructor cost negligible for
        # the common case where no spec-level overrides are present. The
        # caller's map is copied into a fresh ``dict`` and then wrapped in
        # a ``MappingProxyType`` so subsequent caller mutations cannot
        # leak in and post-construction mutation of the stored map fails
        # at runtime (not just by underscore-prefix convention).
        self._stage_spec_overrides: Mapping[str, SaturationAwareStageConfig] = types.MappingProxyType({})
        if stage_spec_overrides:
            self._config.validate_effective_stage_configs(tuple(stage_spec_overrides.values()))
            self._stage_spec_overrides = types.MappingProxyType(dict(stage_spec_overrides))
        # Cross-cycle mutable state aggregated by ``SchedulerLedgers``.
        # The container owns every per-pipeline map / counter / domain
        # detector that must survive across autoscale cycles. The
        # scheduler retains the per-cycle observability hook
        # (``_last_cycle``), the per-pipeline configuration / spec
        # overrides, and the post-setup helper composition; all
        # other cross-cycle state lives on the ledger and is
        # rebuilt by ``setup()``.
        self._ledgers: SchedulerLedgers = SchedulerLedgers(
            warmup=WarmupTracker(),
            measurements=MeasurementCollector(),
            memory_pressure=MemoryPressureMonitor(
                polling_interval_s=config.memory_pressure_polling_interval_s,
                critical_threshold=config.memory_pressure_critical_threshold,
                pipeline_name=pipeline_name,
            ),
        )
        # Cluster regime controller owns regime detection, the
        # SUPER_HALFIN_WHITT aggressiveness lift, and the
        # transition-driven invalidation of per-stage classifier
        # and stabilization state. Built eagerly because the
        # controller is a @attrs.frozen value object bound to
        # the live ledger and config; neither dependency needs
        # the captured pipeline.
        self._regime: RegimeController = RegimeController(config=self._config, ledgers=self._ledgers)
        # Single observability hook for the most-recent autoscale cycle.
        # Phases write their per-cycle outputs onto the ``AutoscaleCycle``
        # that ``_autoscale_body`` constructs; the orchestrator stores
        # the cycle here at the end of every cycle, so tests and post-
        # cycle metrics read ``scheduler.last_cycle.<field>`` and
        # ``last_cycle.bottleneck`` / ``last_cycle.intent`` etc. for
        # the most recent run. ``None`` between ``setup()`` and the
        # first ``autoscale()`` call - callers reading before then
        # must check for ``None`` or use ``scheduler.ledgers.cycle_counter``
        # to detect a fresh setup.
        self._last_cycle: AutoscaleCycle | None = None
        # Post-setup helpers and the ``PipelineModel`` live here as
        # ``None`` placeholders. ``setup()`` builds the model from
        # the captured ``Problem`` and assigns the helpers in one
        # ordered pass. Reading any of these before ``setup()``
        # triggers the explicit ``_require_*`` guards so callers
        # get a single, clear error message.
        self._pipeline: PipelineModel | None = None
        self._threshold_resolver: ThresholdResolver | None = None
        self._floor_calc: FloorCalculator | None = None
        self._ceiling_calc: CeilingCalculator | None = None
        self._phase_invariants: PhaseInvariantSuite | None = None
        self._post_cycle_reporter: PostCycleReporter | None = None
        self._stuck_plan_invariant: StuckPlanInvariant | None = None
        self._finalizer: CycleFinalizer | None = None
        self._preflight: PreflightBuilder | None = None
        self._runner: CycleRunner | None = None

    @property
    def config(self) -> SaturationAwareConfig:
        """Cluster-wide saturation-aware configuration (constructor-bound)."""
        return self._config

    @property
    def pipeline_name(self) -> str:
        """Pipeline tag used in logs and Prometheus labels."""
        return self._pipeline_name

    @property
    def ledgers(self) -> SchedulerLedgers:
        """Cross-cycle mutable state container.

        Public read-only handle to the ledger so observability
        code reaches the canonical owner of every cross-cycle field
        directly (``scheduler.ledgers.s_k_ewma``,
        ``scheduler.ledgers.stuck_plan``, ...) rather than through
        a forwarder chain on this facade. Per-phase
        ``AllocationFailureGate`` instances and the
        bottleneck-protection logger live on their owning
        executors / services and are reached through
        ``scheduler.runner.<phase>_services``.
        """
        return self._ledgers

    @property
    def pipeline(self) -> PipelineModel:
        """Immutable post-setup pipeline shape.

        Raises:
            SchedulerInvariantError: ``setup()`` has not been called.

        """
        return self._require_pipeline()

    @property
    def runner(self) -> CycleRunner:
        """Phase pipeline driver bound to the per-phase narrow services.

        Raises:
            SchedulerInvariantError: ``setup()`` has not been called.

        """
        if self._runner is None:
            msg = "SaturationAwareScheduler.runner read before setup() was called"
            raise SchedulerInvariantError(msg)
        return self._runner

    @property
    def last_cycle(self) -> AutoscaleCycle | None:
        """Most-recent autoscale cycle (``None`` between ``setup()`` and first ``autoscale()``)."""
        return self._last_cycle

    def _require_pipeline(self) -> PipelineModel:
        """Resolve the post-setup ``PipelineModel`` or raise."""
        if self._pipeline is None:
            msg = "SaturationAwareScheduler used before setup() was called"
            raise SchedulerInvariantError(msg)
        return self._pipeline

    def setup(self, problem: data_structures.Problem) -> None:
        """Capture the pipeline shape, seed per-stage state, and wire all helpers.

        Three-step setup with a strict ordering invariant:

        1. Build the immutable ``PipelineModel`` from the captured
           ``Problem`` plus the cluster config and the validated
           spec-level overrides.
        2. Reset every cross-cycle ledger entry the previous run
           may have populated. Mutable maps and typed stores are
           reset in place so direct references already held by
           cross-cycle executors and services from a prior setup
           continue pointing at the live data; the
           :class:`RegimeDetectorState` aggregate is rebuilt by
           re-assignment because nothing keeps a long-lived
           reference to it.
        3. Construct the per-cycle helpers (capacity / floor /
           ceiling calculators, phase invariants, post-cycle
           reporter, finalizer, preflight, runner with its six
           per-phase narrow services and three per-mode
           executors).

        The reset step lands BEFORE construction so executors and
        services capture already-reset state; they hold direct
        references to ledger fields and would otherwise observe
        empty fields populated by step 2 after they were already
        frozen onto the executor / service. ``setup()`` is
        pre-traffic so the streaming executor cannot race the
        measurement / warmup accumulator paths.

        Args:
            problem: The frozen pipeline ``Problem``. The Python
                wrapper does not expose stages directly; iterating
                ``.rust.stages`` gives stage names ahead of any
                runtime state arriving in ``autoscale()``.

        """
        # Step 1: build the post-setup pipeline model.
        pipeline = PipelineModel.from_problem(
            problem=problem,
            config=self._config,
            stage_spec_overrides=self._stage_spec_overrides,
        )
        self._pipeline = pipeline

        # Step 2: reset cross-cycle ledger state IN PLACE so the
        # frozen services / executors built in step 3 capture the
        # already-reset state. Mutable dicts and typed stores are
        # cleared rather than re-assigned to keep object identity
        # stable across setup() re-entries; the
        # :class:`RegimeDetectorState` aggregate is re-assigned
        # because nothing captures it directly.
        ledgers = self._ledgers
        stage_names = pipeline.stage_names
        ledgers.stage_states.clear()
        ledgers.stage_states.update({name: StageRuntimeState(stage_name=name) for name in stage_names})
        ledgers.regime_state = RegimeDetectorState()
        ledgers.worker_ages.clear()
        ledgers.warmup.reset()
        ledgers.floor_stuck_counters.reset()
        ledgers.stuck_plan.reset()
        ledgers.memory_pressure.reset()
        ledgers.heterogeneity_state.reset()
        # NaN-seed the intrinsic ``S_k`` EWMA per stage so the first
        # finite per-stage sample replaces (does not blend with) the
        # seed. Per-cycle actor-normalized D_k, effective capacities,
        # balance score, and bottleneck identity live on the
        # ``AutoscaleCycle`` the bottleneck phase populates and the
        # orchestrator stores in ``_last_cycle`` at end of cycle.
        ledgers.s_k_ewma.reset_seeded(stage_names)
        ledgers.bottleneck_engagement_state.reset()
        ledgers.cycle_counter = 0
        ledgers.last_donation_cycle.clear()
        ledgers.measurements.setup(list(stage_names))
        # Build per-stage stabilization-window buffers from the resolved
        # effective config; both windows are config-time invariants
        # (cross-validated in ``SaturationAwareStageConfig.__attrs_post_init__``)
        # and do not flex during runtime, so the buffers can outlive every
        # ``autoscale()`` cycle without per-cycle re-allocation. Reset in
        # place so downstream services that captured the dict observe the
        # rebuilt buffers without re-receiving a reference.
        ledgers.recommendation_histories.clear()
        ledgers.recommendation_histories.update(
            {
                name: RecommendationHistory(
                    window_up=pipeline.stage_config(name).stabilization_window_cycles_up,
                    window_down=pipeline.stage_config(name).stabilization_window_cycles_down,
                )
                for name in stage_names
            }
        )

        # Step 3: construct the per-cycle helpers bound to the
        # pipeline model and the (now-reset) ledgers. Construction
        # order respects the dependency graph: leaf calculators
        # first, then the donor subsystem, the
        # bottleneck-protection logger, the per-mode allocation
        # gates, and finally the runner + finalizer + preflight.
        self._threshold_resolver = ThresholdResolver(
            ledgers=self._ledgers,
            regime=self._regime,
            pipeline=pipeline,
        )
        self._floor_calc = FloorCalculator(pipeline=pipeline)
        self._ceiling_calc = CeilingCalculator(pipeline=pipeline)
        self._phase_invariants = PhaseInvariantSuite(
            pipeline=pipeline,
            ledgers=self._ledgers,
            floors=self._floor_calc,
        )
        decision_pipeline = StageDecisionPipeline(
            signal_noise_smoothing_level=pipeline.config.classifier_signal_noise_smoothing_level,
        )
        # Donor subsystem: one instance of each value object per
        # scheduler. ``ResourceFitPlanner`` is configured once with
        # the cluster-wide bounds; ``EconomicGate`` is configured
        # once with the cluster config so every Phase B / Phase C
        # receiver reads the same threshold values. The coordinator
        # shares the single ``ResourceFitPlanner`` across floor and
        # saturation policies because the resource-fit search is
        # identical in both modes. Floor and Grow each receive
        # their own ``DonorBackedAddExecutor`` instance pre-wired
        # with the mode-specific policy and an owned
        # :class:`AllocationFailureGate`; the gates are also handed
        # to the post-cycle reporter so the cycle-summary log can
        # attribute an absorbed ``AllocationError`` to its phase.
        resource_fit_planner = ResourceFitPlanner(
            max_plan_size=self._config.cross_stage_donor_max_plan_size,
            max_plan_combinations=self._config.cross_stage_donor_max_plan_combinations,
        )
        donor_coordinator = DonorCoordinator(
            pipeline_name=self._pipeline_name,
            planner=resource_fit_planner,
        )
        floor_allocation_gate = AllocationFailureGate()
        grow_allocation_gate = AllocationFailureGate()
        manual_allocation_gate = AllocationFailureGate()
        floor_donor_executor = DonorBackedAddExecutor(
            coordinator=donor_coordinator,
            policy=FloorPolicy(),
            pipeline=pipeline,
            allocation_gate=floor_allocation_gate,
            stage_states=ledgers.stage_states,
            last_donation_cycle=ledgers.last_donation_cycle,
            s_k_ewma=ledgers.s_k_ewma,
            planning_mode="floor",
        )
        grow_donor_executor = DonorBackedAddExecutor(
            coordinator=donor_coordinator,
            policy=SaturationPolicy(gate=EconomicGate(config=self._config)),
            pipeline=pipeline,
            allocation_gate=grow_allocation_gate,
            stage_states=ledgers.stage_states,
            last_donation_cycle=ledgers.last_donation_cycle,
            s_k_ewma=ledgers.s_k_ewma,
            planning_mode="saturation",
        )
        manual_grow_executor = ManualGrowExecutor(allocation_gate=manual_allocation_gate)
        manual_delete_executor = ManualDeleteExecutor()
        # The bottleneck-protection logger is owned by the shrink
        # services so the once-per-streak INFO log keeps its
        # previous-cycle snapshot across cycles.
        bottleneck_protection_logger = BottleneckProtectionLogger()

        self._post_cycle_reporter = PostCycleReporter(
            pipeline=pipeline,
            ledgers=self._ledgers,
            pipeline_name=self._pipeline_name,
            manual_allocation=manual_allocation_gate,
            floor_allocation=floor_allocation_gate,
            grow_allocation=grow_allocation_gate,
        )
        self._stuck_plan_invariant = StuckPlanInvariant(ledgers=self._ledgers)
        self._finalizer = CycleFinalizer(
            stuck_plan_invariant=self._stuck_plan_invariant,
            post_cycle_reporter=self._post_cycle_reporter,
            ledgers=self._ledgers,
            pipeline=pipeline,
        )
        self._preflight = PreflightBuilder(
            ledgers=self._ledgers,
            regime=self._regime,
            threshold_resolver=self._threshold_resolver,
            pipeline=pipeline,
            pipeline_name=self._pipeline_name,
        )
        self._runner = CycleRunner(
            manual=ManualPhase(),
            floor=FloorPhase(),
            bottleneck=BottleneckPhase(),
            intent=IntentPhase(decision_pipeline=decision_pipeline),
            grow=SaturationGrowPhase(),
            shrink=SaturationShrinkPhase(),
            invariants=self._phase_invariants,
            recorder=GrowthModeRecorder(
                pipeline=pipeline,
                ledgers=self._ledgers,
                decision_pipeline=decision_pipeline,
            ),
            manual_services=ManualServices(
                pipeline=pipeline,
                pipeline_name=self._pipeline_name,
                delete_executor=manual_delete_executor,
                grow_executor=manual_grow_executor,
            ),
            floor_services=FloorServices(
                pipeline=pipeline,
                pipeline_name=self._pipeline_name,
                floors=self._floor_calc,
                donor_executor=floor_donor_executor,
                floor_stuck_counters=ledgers.floor_stuck_counters,
                stage_states=ledgers.stage_states,
            ),
            bottleneck_services=BottleneckServices(
                pipeline=pipeline,
                pipeline_name=self._pipeline_name,
                measurements=ledgers.measurements,
                s_k_ewma=ledgers.s_k_ewma,
                bottleneck_engagement_state=ledgers.bottleneck_engagement_state,
            ),
            intent_services=IntentServices(
                pipeline=pipeline,
                pipeline_name=self._pipeline_name,
                capacity=CapacityModel(s_k_ewma=ledgers.s_k_ewma),
                measurements=ledgers.measurements,
                stage_states=ledgers.stage_states,
                recommendation_histories=ledgers.recommendation_histories,
                warmup=ledgers.warmup,
            ),
            grow_services=GrowServices(
                pipeline=pipeline,
                pipeline_name=self._pipeline_name,
                floors=self._floor_calc,
                ceilings=self._ceiling_calc,
                donor_executor=grow_donor_executor,
                stuck_plan_ledger=ledgers.stuck_plan,
                stage_states=ledgers.stage_states,
            ),
            shrink_services=ShrinkServices(
                pipeline=pipeline,
                pipeline_name=self._pipeline_name,
                floors=self._floor_calc,
                ceilings=self._ceiling_calc,
                bottleneck_protection=bottleneck_protection_logger,
            ),
        )
        # Reset the most-recent-cycle observability hook. ``None``
        # signals "no autoscale() has run yet under this setup";
        # callers must check ``scheduler.ledgers.cycle_counter == 0``
        # (or ``scheduler.last_cycle is None``) before reading per-
        # cycle fields. The next ``autoscale()`` populates a fresh
        # cycle.
        self._last_cycle = None

    def update_with_measurements(
        self,
        time: float,
        measurements: data_structures.Measurements,
    ) -> None:
        """Ingest the latest measurement batch.

        Thin delegate to ``MeasurementCollector``. ``time`` is
        part of the autoscaler protocol but unused; the autoscale
        cycle timestamp is the canonical clock for throughput
        deltas.

        Raises:
            ValueError: ``measurements.rust.stages`` disagrees in
                length with the setup-time stage list.

        """
        del time
        self._ledgers.measurements.update_with_measurements(measurements)

    def autoscale(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Compute the autoscale plan for the current cycle.

        Wraps the per-cycle body with ``loop_watchdog`` so slow
        cycles observe ``xenna_scheduler_cycle_duration_seconds``
        and emit a WARN. Returns one ``StageSolution`` per stage.

        Raises:
            RuntimeError: ``setup()`` not called; planner context
                construction failed; floor cannot be satisfied.
            SchedulerInvariantError: Phase-boundary or solution-
                shape invariant failed - plan is corrupted.
            ValueError: ``problem`` / ``problem_state`` disagree on
                stage names or count.

        """
        with loop_watchdog(
            pipeline_name=self._pipeline_name,
            threshold_fraction=self._config.cycle_time_warn_threshold,
            interval_s=self._config.interval_s,
        ):
            return self._autoscale_body(time, problem_state)

    def _autoscale_body(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Run one autoscale cycle without the loop-watchdog wrap.

        ``autoscale()`` wraps this method with ``loop_watchdog`` so
        the per-cycle duration histogram and WARN log fire on every
        call, including paths that raise.

        Raises:
            SchedulerInvariantError: ``setup()`` was not called.

        """
        # Per-cycle pipeline: preflight builds the cycle, the
        # runner drives the 6 phase classes in declared order
        # (each owning its own post-phase invariant checks and
        # cycle bookkeeping), the finalizer applies the post-
        # runner order (stuck-plan invariant -> post-cycle
        # reporter -> Solution drain -> worker-age persist), and
        # the facade publishes the cycle as the single
        # observability hook. The runner owns the per-phase
        # service value objects that each phase consumes.
        self._require_pipeline()
        assert self._preflight is not None
        assert self._runner is not None
        assert self._finalizer is not None
        preflight = self._preflight.build(time=time, problem_state=problem_state)
        self._runner.run(preflight.cycle)
        solution = self._finalizer.finalize(
            cycle=preflight.cycle,
            problem_state=problem_state,
            prev_stuck_plan_counters=preflight.prev_stuck_plan_counters,
        )
        self._last_cycle = preflight.cycle
        return solution
