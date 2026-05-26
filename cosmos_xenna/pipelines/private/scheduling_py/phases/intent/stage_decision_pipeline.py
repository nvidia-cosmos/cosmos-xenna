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

"""Per-stage decision pipeline - ``StageDecisionPipeline``.

Owns the per-cycle composition of every per-stage primitive:
snapshot -> resolve classifier signal -> classify -> update streak
-> fire-gate -> compute intent delta -> stabilization gate ->
recommendation. The returned delta is the algorithm's
recommendation only; the growth-mode state machine is advanced
separately by ``record_executed_delta`` after Phase C (grow) and
Phase D (shrink) commit. The executed delta passed in is the net
post-commit worker change for the stage. Splitting the
recommendation from the execution recording lets hard worker
caps, fraction clamps, and allocation failures shrink the
planner's output below the recommendation without the
growth-mode timer observing a delta that never landed in the
cluster.

The pipeline is a ``@attrs.frozen`` Strategy: a value object
held by the per-stage call site. It carries no per-cycle state;
state lives entirely on the ``StageRuntimeState`` sub-state
containers. Internal collaborators are themselves frozen
strategies:

    +-----+
    | StageDecisionPipeline                            |
    |  - signal_resolver: ClassifierSignalResolver     |
    |  - pressure_resolver: PressureSignalResolver     |
    |  - composer: RecommendationComposer              |
    |  - transition_logger: ClassifierTransitionLogger |
    |  - signal_noise_smoothing_level: float | None    |
    +-+--+-+--+
      |        |     |        |
      v        v     v        v
    signal -> pressure -> classify+streak -> compose -> log
                                                |
                                                v
                                       stabilization gate
                                                |
                                                v
                                            delta
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.classifier import classify
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.decisions import (
    compute_delta,
    should_fire_action,
    update_streak,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.growth_mode import compute_growth_mode_transition
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure import (
    compute_backlog_time,
    compute_pressure,
    emit_pressure_signals,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stabilization import apply_stabilization_gate
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    GrowthMode,
    StageRuntimeState,
    StageState,
    compute_slots_empty_ratio,
    update_ewma,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_topology import StageTopologyContext
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class ClassifierSignalResolver:
    """Resolve the per-cycle classifier signal (slots-empty-ratio EWMA).

    Three outcomes, mutually exclusive:

    - Fresh sample available -> updates
      ``stage_state.classifier.slots_empty_ratio_ewma`` and
      ``last_valid_slots_empty_ratio_ewma``; returns the new EWMA.
    - Zero ready actors but a prior valid EWMA exists -> returns
      the carry-forward EWMA without mutating any state, so the
      classifier keeps tracking through transient gaps.
    - Cold-start with no prior signal -> returns ``None``; the
      caller short-circuits and emits no recommendation.
    """

    def resolve(
        self,
        *,
        stage_state: StageRuntimeState,
        num_used_slots: int,
        num_empty_slots: int,
        config: SaturationAwareStageConfig,
    ) -> float | None:
        """Resolve the classifier signal for one cycle.

        Returns the EWMA value the classifier reads this cycle.
        See the class docstring for the three outcomes.
        """
        total_slots = num_used_slots + num_empty_slots
        if total_slots > 0:
            raw_ratio = compute_slots_empty_ratio(num_used_slots, num_empty_slots)
            new_ewma = update_ewma(
                stage_state.classifier.slots_empty_ratio_ewma,
                raw_ratio,
                config.slots_empty_ratio_smoothing_level,
            )
            stage_state.classifier.slots_empty_ratio_ewma = new_ewma
            stage_state.classifier.last_valid_slots_empty_ratio_ewma = new_ewma
            return new_ewma

        if stage_state.classifier.last_valid_slots_empty_ratio_ewma is not None:
            return stage_state.classifier.last_valid_slots_empty_ratio_ewma

        return None


@attrs.frozen
class PressureSignalResolver:
    """Resolve the per-cycle pressure signal and maintain noise EWMA.

    On a fresh-sample cycle: computes the raw pressure scalar,
    refreshes ``stage_state.pressure.ewma``, emits the pressure
    gauges, and (when noise tracking is enabled and a prior
    classifier EWMA exists) updates
    ``stage_state.classifier.signal_noise_ewma`` from
    ``|classifier_input - prev_slots_empty_ratio_ewma|``.

    On a carry-forward cycle (no fresh sample): returns the
    stored pressure EWMA (or ``0.0`` when cold-start) without
    mutating any state. Carry-forward must NOT refresh pressure
    because blending stale utilisation with fresh queue depth
    would corrupt the next valid cycle's classification.
    """

    def resolve(
        self,
        *,
        stage_state: StageRuntimeState,
        classifier_input: float,
        prev_slots_empty_ratio_ewma: float | None,
        cycle_has_fresh_sample: bool,
        signal_noise_smoothing_level: float | None,
        input_queue_depth: int,
        observed_throughput: float,
        config: SaturationAwareStageConfig,
        stage_name: str,
        pipeline_name: str,
    ) -> float:
        """Return the post-EWMA pressure scalar for this cycle.

        Bands: ``[0.0, BACKLOG_CAP]``. On the fresh-sample path
        the noise EWMA is updated only when both the smoothing
        level is configured and a prior classifier EWMA was
        observed (cold-start prev=None contributes no usable
        delta; ``update_ewma`` will seed the noise EWMA on the
        first valid delta thereafter).
        """
        if not cycle_has_fresh_sample:
            # Carry-forward path: reuse the prior pressure EWMA.
            # ``None`` collapses to ``0.0`` for the classifier
            # (no-pressure means the slot-only branch decides),
            # matching the cold-start invariant.
            return stage_state.pressure.ewma if stage_state.pressure.ewma is not None else 0.0

        raw_pressure = compute_pressure(
            slots_empty_ratio_ewma=classifier_input,
            input_queue_depth=input_queue_depth,
            observed_throughput=observed_throughput,
            target_backlog_seconds=config.target_backlog_seconds,
        )
        new_pressure_ewma = update_ewma(
            stage_state.pressure.ewma,
            raw_pressure,
            config.pressure_smoothing_level,
        )
        stage_state.pressure.ewma = new_pressure_ewma
        backlog_time = compute_backlog_time(
            input_queue_depth=input_queue_depth,
            observed_throughput=observed_throughput,
            target_backlog_seconds=config.target_backlog_seconds,
        )
        emit_pressure_signals(
            stage_name=stage_name,
            pipeline_name=pipeline_name,
            observed_throughput=observed_throughput,
            backlog_time=backlog_time,
            pressure_ewma=new_pressure_ewma,
        )
        if signal_noise_smoothing_level is not None and prev_slots_empty_ratio_ewma is not None:
            stage_state.classifier.signal_noise_ewma = update_ewma(
                stage_state.classifier.signal_noise_ewma,
                abs(classifier_input - prev_slots_empty_ratio_ewma),
                signal_noise_smoothing_level,
            )
        return new_pressure_ewma


@attrs.frozen
class RecommendationOutcome:
    """The composer's output: signed delta plus the fire-gate verdict.

    ``should_fire`` is exposed separately from ``delta`` because
    ``delta == 0`` is overloaded: it can mean "fire gate did not
    fire" OR "fire gate fired but ``compute_delta`` returned 0
    (target met)". The transition logger surfaces both so
    operators can distinguish a silenced classifier from a
    classifier whose action is already complete.
    """

    delta: int
    should_fire: bool


@attrs.frozen
class RecommendationComposer:
    """Compose the per-stage recommendation magnitude.

    Applies in order: (1) the carry-forward freshness clamp -
    a no-fresh-sample cycle with zero ready actors emits no
    delta regardless of classifier streak; (2) the streak-gated
    firing decision via ``should_fire_action``; (3) the
    ``compute_delta`` magnitude using the effective growth mode
    (TRACKING when the state machine kill switch is off).
    """

    def compose(
        self,
        *,
        new_classifier_state: StageState,
        stage_state: StageRuntimeState,
        current_workers: int,
        cycle_has_fresh_sample: bool,
        config: SaturationAwareStageConfig,
    ) -> RecommendationOutcome:
        """Return the (delta, should_fire) outcome for this cycle.

        Freshness clamp: a zero-actor carry-forward cycle
        cannot meaningfully drive a delta (cannot shrink below
        0; Phase B floor owns cold-start growth). The clamp
        short-circuits before the firing decision so the
        downstream stabilization gate observes a clean 0.
        """
        if not cycle_has_fresh_sample and current_workers == 0:
            return RecommendationOutcome(delta=0, should_fire=False)

        should_fire = should_fire_action(new_classifier_state, stage_state.classifier.streak, config)
        if not should_fire:
            return RecommendationOutcome(delta=0, should_fire=False)

        # Growth-mode kill switch: when the state machine is
        # disabled, force ``compute_delta`` to use ``TRACKING``
        # so HOLD post-shrink suppression does not block
        # SATURATED growth. ACQUIRING vs TRACKING are
        # interchangeable in the capacity-driven sizer; any
        # non-HOLD value is sufficient.
        effective_growth_mode = (
            stage_state.growth.mode if config.enable_growth_mode_state_machine else GrowthMode.TRACKING
        )
        delta = compute_delta(
            new_classifier_state,
            effective_growth_mode,
            current_workers,
            stage_state.growth.capacity_target_workers,
            config,
        )
        return RecommendationOutcome(delta=delta, should_fire=True)


@attrs.frozen
class ClassifierTransitionLogger:
    """Emit the per-stage classifier trace + transition log lines.

    Always emits one DEBUG line summarising this cycle's signal,
    pressure, state, streak, fire-gate verdict, and delta so
    operators can replay the classifier from logs alone. Emits
    one INFO line additionally when the classifier state
    transitioned this cycle, so dashboards and alerts can
    pivot on transition events without scanning every cycle.
    """

    def log(
        self,
        *,
        stage_state: StageRuntimeState,
        classifier_input: float,
        input_queue_depth: int,
        pressure_ewma: float,
        prev_classifier_state: StageState,
        new_classifier_state: StageState,
        should_fire: bool,
        delta: int,
        topology: StageTopologyContext,
    ) -> None:
        """Emit DEBUG trace; emit INFO if the classifier state changed."""
        logger.debug(
            f"classifier trace: stage={stage_state.stage_name!r} "
            f"slots_empty_ratio_ewma={classifier_input:.3f} "
            f"input_queue_depth={input_queue_depth} "
            f"pressure_ewma={pressure_ewma:.3f} "
            f"prev_state={prev_classifier_state.name} "
            f"new_state={new_classifier_state.name} "
            f"streak={stage_state.classifier.streak} "
            f"bottleneck_engaged={topology.engaged} "
            f"upstream_of_bottleneck={topology.is_upstream_of_bottleneck} "
            f"should_fire={should_fire} "
            f"delta={delta}"
        )
        if prev_classifier_state != new_classifier_state:
            logger.info(
                f"classifier transition: stage={stage_state.stage_name!r} "
                f"{prev_classifier_state.name} -> {new_classifier_state.name} "
                f"(pressure_ewma={pressure_ewma:.3f}, slots_empty_ratio_ewma={classifier_input:.3f}, "
                f"queue={input_queue_depth}, streak={stage_state.classifier.streak}, "
                f"bottleneck_engaged={topology.engaged}, "
                f"upstream_of_bottleneck={topology.is_upstream_of_bottleneck}, delta={delta})"
            )


@attrs.frozen
class StageDecisionPipeline:
    """Per-stage decision pipeline - frozen Strategy.

    Holds no mutable per-cycle state. The two entry points are:

    - ``compute_recommendation(...)`` - runs the per-stage
      classifier / pressure / fire-gate / delta / stabilization
      chain and returns the recommended delta. Mutates
      ``stage_state.classifier`` and ``stage_state.pressure``
      sub-states only.
    - ``record_executed_delta(...)`` - advances the growth-mode
      FSM with the post-commit net delta. Mutates
      ``stage_state.growth`` only.

    Args:
        signal_noise_smoothing_level: When set, the EWMA alpha
            for the classifier-signal noise tracker used by the
            donor signal-trust gate. ``None`` disables noise
            tracking (no donor signal trust influence).
        signal_resolver: Strategy that returns the per-cycle
            classifier EWMA (fresh sample / carry-forward /
            cold-start ``None``).
        pressure_resolver: Strategy that returns the per-cycle
            pressure EWMA and (on fresh-sample cycles) maintains
            the classifier noise EWMA.
        composer: Strategy that composes the (delta, should_fire)
            outcome from the new classifier state, current
            workers, and growth mode.
        transition_logger: Strategy that emits the classifier
            trace DEBUG log on every cycle and an INFO log on
            state transition.

    """

    signal_noise_smoothing_level: float | None = None
    signal_resolver: ClassifierSignalResolver = attrs.field(factory=ClassifierSignalResolver)
    pressure_resolver: PressureSignalResolver = attrs.field(factory=PressureSignalResolver)
    composer: RecommendationComposer = attrs.field(factory=RecommendationComposer)
    transition_logger: ClassifierTransitionLogger = attrs.field(factory=ClassifierTransitionLogger)

    def compute_recommendation(
        self,
        *,
        stage_state: StageRuntimeState,
        num_used_slots: int,
        num_empty_slots: int,
        input_queue_depth: int,
        current_workers: int,
        config: SaturationAwareStageConfig,
        recommendation_history: RecommendationHistory | None = None,
        observed_throughput_sample: float = 0.0,
        pipeline_name: str = "",
        topology: StageTopologyContext = StageTopologyContext(),  # noqa: B008 - @attrs.frozen, safe immutable default
    ) -> int:
        """Compute the per-stage scaling recommendation for one cycle.

        Mutates ``stage_state.classifier`` (state, streak, EWMAs,
        signal-noise) and ``stage_state.pressure`` (pressure EWMA).
        Mutates ``stage_state.growth.prev_workers`` as the
        last-cycle worker-count tape used by capacity sizing.
        Returns the stabilization-gated recommended delta. Does
        NOT advance the growth-mode FSM - callers must invoke
        ``record_executed_delta`` after Phase C / D so the HOLD /
        ACQUIRING / TRACKING timers observe the executed (not
        recommended) delta.

        Args:
            topology: Per-stage bottleneck topology projected by the
                caller (the intent phase) from
                ``cycle.bottleneck.identity`` and the consumer's
                DAG index. Defaults to the disengaged
                ``StageTopologyContext()`` so callers exercising
                the classifier path without bottleneck context
                (most direct unit tests) do not have to construct
                one explicitly. The default is safe because
                :class:`StageTopologyContext` is ``@attrs.frozen``.

        Raises:
            SchedulerInvariantError:
                ``stage_state.classifier.resolved_thresholds`` is
                ``None`` (must be resolved before the first call).

        """
        if stage_state.classifier.resolved_thresholds is None:
            msg = (
                f"StageRuntimeState for stage {stage_state.stage_name!r} has no "
                "resolved_thresholds; ThresholdResolver.ensure_resolved "
                "(invoked at the top of autoscale()) is the only legitimate populator. "
                "Tests that construct StageRuntimeState directly must populate "
                "classifier.resolved_thresholds via _resolve_auto_thresholds(...) at fixture time."
            )
            raise SchedulerInvariantError(msg)

        # Snapshot the previous EWMA BEFORE the signal resolver
        # mutates it. The noise tracker compares the prev/next
        # pair for the same stage to measure classifier flicker;
        # sampling after the update would always yield zero
        # delta and falsely report a quiet classifier.
        prev_slots_empty_ratio_ewma = stage_state.classifier.slots_empty_ratio_ewma

        classifier_input = self.signal_resolver.resolve(
            stage_state=stage_state,
            num_used_slots=num_used_slots,
            num_empty_slots=num_empty_slots,
            config=config,
        )
        if classifier_input is None:
            # Zero ready actors, no prior valid EWMA. The
            # structural worker-floor step independently
            # bootstraps; the classifier holds for this cycle.
            # The growth-mode transition is left to the
            # scheduler's post-Phase-D ``record_executed_delta``
            # call which sees delta_executed=0 for a stage that
            # took no action. Pressure is intentionally NOT
            # updated here: without a slot signal we have no
            # utilisation factor, so feeding the EWMA with
            # stale data would corrupt the next valid cycle's
            # demotion decision.
            stage_state.growth.prev_workers = current_workers
            return 0

        # Cycle freshness: distinguish "fresh sample carries the
        # classifier signal" from "carry-forward EWMA from a
        # prior valid cycle". When the current cycle observed
        # zero warmup-filtered ready actors, the signal resolver
        # returned the carry-forward EWMA so the classifier
        # state machine keeps tracking. Pressure, however, MUST
        # NOT refresh on carry-forward: refreshing would blend
        # a stale utilisation factor with a fresh queue depth
        # and corrupt the next valid cycle's classification.
        # The scheduler also clamps the returned delta to ``0``
        # for the same reason via its trust-gate freshness
        # check, so a carry-forward cycle is always a no-action
        # cycle on Phase C / Phase D.
        cycle_has_fresh_sample = (num_used_slots + num_empty_slots) > 0
        pressure_ewma = self.pressure_resolver.resolve(
            stage_state=stage_state,
            classifier_input=classifier_input,
            prev_slots_empty_ratio_ewma=prev_slots_empty_ratio_ewma,
            cycle_has_fresh_sample=cycle_has_fresh_sample,
            signal_noise_smoothing_level=self.signal_noise_smoothing_level,
            input_queue_depth=input_queue_depth,
            observed_throughput=observed_throughput_sample,
            config=config,
            stage_name=stage_state.stage_name,
            pipeline_name=pipeline_name,
        )

        # Classify + streak update inline; the new state and
        # streak feed both the composer and the transition log.
        prev_classifier_state = stage_state.classifier.state
        new_classifier_state = classify(
            slots_empty_ratio_ewma=classifier_input,
            pressure_ewma=pressure_ewma,
            prev_state=stage_state.classifier.state,
            saturation_threshold=stage_state.classifier.resolved_thresholds.saturation_threshold,
            activation_threshold=stage_state.classifier.resolved_thresholds.activation_threshold,
            config=config,
        )
        stage_state.classifier.streak = update_streak(
            stage_state.classifier.state,
            stage_state.classifier.streak,
            new_classifier_state,
        )
        stage_state.classifier.state = new_classifier_state

        outcome = self.composer.compose(
            new_classifier_state=new_classifier_state,
            stage_state=stage_state,
            current_workers=current_workers,
            cycle_has_fresh_sample=cycle_has_fresh_sample,
            config=config,
        )
        delta = outcome.delta

        if recommendation_history is not None:
            # Record-then-gate is encapsulated in
            # ``apply_stabilization_gate`` so callers cannot
            # accidentally gate without recording, which would
            # leave the buffer one cycle behind reality and
            # weaken every future gate decision.
            delta = apply_stabilization_gate(recommendation_history, delta)

        self.transition_logger.log(
            stage_state=stage_state,
            classifier_input=classifier_input,
            input_queue_depth=input_queue_depth,
            pressure_ewma=pressure_ewma,
            prev_classifier_state=prev_classifier_state,
            new_classifier_state=new_classifier_state,
            should_fire=outcome.should_fire,
            delta=delta,
            topology=topology,
        )

        stage_state.growth.prev_workers = current_workers
        return delta

    def record_executed_delta(
        self,
        *,
        stage_state: StageRuntimeState,
        delta_executed: int,
        config: SaturationAwareStageConfig,
    ) -> None:
        """Advance the growth-mode FSM with the post-commit delta.

        Called once per stage per cycle, after Phase C / D commit.
        ``delta_executed`` is the net worker change
        (``post_phase_d_count - pre_phase_c_count``), which may be
        smaller in magnitude than the recommendation if caps or
        allocation failures throttled the planner. Short-circuits
        when ``config.enable_growth_mode_state_machine`` is False
        so re-enabling mid-run resumes from ACQUIRING, not stale
        HOLD.

        """
        if not config.enable_growth_mode_state_machine:
            return
        stage_state.growth.mode, stage_state.growth.streak = compute_growth_mode_transition(
            prev_mode=stage_state.growth.mode,
            prev_streak=stage_state.growth.streak,
            delta_executed=delta_executed,
            config=config,
        )


__all__ = (
    "ClassifierSignalResolver",
    "ClassifierTransitionLogger",
    "PressureSignalResolver",
    "RecommendationComposer",
    "RecommendationOutcome",
    "StageDecisionPipeline",
)
