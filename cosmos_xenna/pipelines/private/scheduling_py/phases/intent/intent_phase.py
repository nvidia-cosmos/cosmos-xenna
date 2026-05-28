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

"""Intent phase: per-stage signed worker-count intent for one cycle.

Runs between the bottleneck phase and Phase C / Phase D. Feeds
each non-finished stage's slot signals into the constructor-injected
``StageDecisionPipeline``; the pipeline records the raw recommendation
into the per-stage asymmetric stabilization window and returns the
post-window-gate delta that Phase C / D execute.

Three frozen Strategies cooperate inside the per-stage loop:

    +----+
    | IntentPhase                                  |
    |  - quiescence: QuiescenceGate                |
    |  - trust: TrustGate                          |
    |  - decision_pipeline: StageDecisionPipeline  |
    +-+--+--+
      |        |        |
      v        v        v
    cold-start   trust-counter    classifier/pressure/
    skip+reset   + zero-clamp     stabilization
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.services import IntentServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stage_decision_pipeline import StageDecisionPipeline
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.outputs import IntentPlan
from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_topology import project_stage_topology
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class QuiescenceGate:
    """Cold-start + hot-pending quiescence guard for the intent phase.

    Two-mode gate. ``is_cold_start`` short-circuits the entire
    per-stage pipeline when the stage has no ready actors yet:
    every slot signal at that point is an artifact of the empty
    pool, not a real measurement. Recording it would corrupt
    the classifier streak and the stabilization-window history,
    so the gate resets the trust counter and stabilization
    buffer (see :meth:`reset_classifier`) and the caller skips
    the rest of the cycle.

    ``should_clamp_positive_delta`` is the hot-pending mode:
    ready actors observe real saturation, but a prior scale-up
    has not finished setup yet. Suppress the additional Phase C
    add to avoid amplifying cold-start noise; the next cycle
    re-evaluates with the new ready count. Phase D shrink
    (``delta < 0``) is left untouched because the still-pending
    actor is independent of any ready-actor removal.
    """

    def is_active(self, *, stage_cfg: SaturationAwareStageConfig, pending_actors: int) -> bool:
        """True when the quiescence policy applies to this stage this cycle.

        Quiescence is configurable per stage; a stage with the
        policy disabled passes neither the cold-start skip nor
        the hot-pending clamp.
        """
        return stage_cfg.setup_phase_quiescence_enabled and pending_actors > 0

    def is_cold_start(self, *, quiescence_active: bool, ready_at_snapshot: int) -> bool:
        """True when an active quiescence policy sees zero ready actors."""
        return quiescence_active and ready_at_snapshot == 0

    def reset_classifier(
        self,
        *,
        stage_state: StageRuntimeState,
        recommendation_history: RecommendationHistory,
    ) -> None:
        """Drop trust counter and stabilization history on a quiescent cycle.

        Reset ``valid_signal_samples`` so any trust accumulated
        before the quiescent gap is invalidated. Without this
        reset, a post-quiescence cycle could fire a non-zero
        recommendation off carry-forward EWMA the moment a
        single ready actor emerges (the trust gate would still
        see ``valid_signal_samples >= min_data_points`` from
        the pre-gap streak). The reset mirrors the no-signal-gap
        path in :class:`TrustGate`, keeping the
        "consecutive valid samples" contract uniform across
        both no-signal-gap paths (cold-start skip + in-pipeline
        warmup gap).

        Drop the per-stage stabilization window too. Without
        this, pre-gap +1/-1 entries can vote alongside post-gap
        entries the moment the first ready actor emerges,
        letting the asymmetric-window gate fire on a mixed
        history that no longer reflects current cluster state.
        """
        stage_state.classifier.valid_signal_samples = 0
        recommendation_history.clear()

    def should_clamp_positive_delta(self, *, quiescence_active: bool, delta: int) -> bool:
        """True when hot-pending and the recommendation would grow the stage."""
        return quiescence_active and delta > 0


@attrs.frozen
class TrustGate:
    """Min-data-points trust gate for non-zero recommendations.

    Tracks how many *strictly consecutive* warmup-excluded valid
    samples the classifier output has accumulated. A sample is
    "valid" when at least one ready actor contributed
    (``num_used_slots + num_empty_slots > 0``). Cycles where
    every ready actor was still within
    ``worker_warmup_measurement_grace_s`` produce zeroed signals;
    those reset the counter to 0 so the gate is gated on *real,
    contiguous* observations, not on actor existence. Phase B
    floor and manual provisioning run outside the intent phase,
    so this gate cannot starve a zero-worker stage.

    The decision pipeline still runs even when the gate would
    clamp the delta so the EWMA cache, classifier state, and
    stabilization history keep tracking reality; only the
    returned delta is clamped to zero while the gate is closed.

    The validator pins ``min_data_points >= 1``, so a counter
    reset always trips this check on the next non-zero
    recommendation.
    """

    def update_freshness_counter(
        self,
        *,
        stage_state: StageRuntimeState,
        stage_cfg: SaturationAwareStageConfig,
        cycle_has_fresh_sample: bool,
        recommendation_history: RecommendationHistory,
    ) -> None:
        """Increment the counter on a fresh sample; reset on a no-sample gap.

        No-signal cycle (every ready actor inside the warmup
        grace, or zero ready actors at all) resets the counter
        AND drops the per-stage stabilization window so any
        pre-gap +1/-1 entries cannot vote alongside post-gap
        entries when the gap closes. Mirrors the cold-start
        quiescent block in :class:`QuiescenceGate` so both no-
        signal-gap paths stay uniform.
        """
        if cycle_has_fresh_sample:
            stage_state.classifier.valid_signal_samples = min(
                stage_state.classifier.valid_signal_samples + 1, stage_cfg.min_data_points
            )
        else:
            stage_state.classifier.valid_signal_samples = 0
            recommendation_history.clear()

    def should_clamp_to_zero(
        self,
        *,
        stage_state: StageRuntimeState,
        stage_cfg: SaturationAwareStageConfig,
        delta: int,
    ) -> bool:
        """True when a non-zero delta would fire before the trust streak builds."""
        return delta != 0 and stage_state.classifier.valid_signal_samples < stage_cfg.min_data_points


@attrs.frozen
class IntentPhase:
    """Per-stage signed worker-count intent for one cycle.

    Stateless ``@attrs.frozen`` Phase implementation. Returns an
    ``IntentPlan`` carrying ``stage_name -> signed_delta`` for
    every non-finished stage that produced a recommendation
    (cold-start quiescent + finished stages are absent). Phase C
    and Phase D consume ``cycle.intent.deltas`` to execute the
    actual grows / shrinks.

    Attributes:
        decision_pipeline: Constructor-injected
            :class:`StageDecisionPipeline` carrying the configured
            noise-smoothing level. Reused across every stage in
            every cycle; the pipeline is stateless beyond its
            smoothing field.
        quiescence: :class:`QuiescenceGate` strategy that owns
            the cold-start skip and the hot-pending positive-
            delta clamp.
        trust: :class:`TrustGate` strategy that owns the
            consecutive-valid-sample counter and the trust-
            below-floor delta clamp.

    """

    decision_pipeline: StageDecisionPipeline = attrs.Factory(StageDecisionPipeline)
    quiescence: QuiescenceGate = attrs.Factory(QuiescenceGate)
    trust: TrustGate = attrs.Factory(TrustGate)

    def run(self, cycle: AutoscaleCycle, services: IntentServices) -> None:
        """Compute per-stage intent deltas and publish them onto ``cycle.intent``.

        Calls :meth:`_compute_intent_deltas` for the inner per-stage
        loop and writes the result onto ``cycle.intent`` so Phase C
        and Phase D observe the same per-cycle plan. Per-stage
        classifier state, stabilization buffers, and the
        capacity-target cache are mutated inside the loop.

        """
        # Pull the per-cycle throughput sample from the measurement
        # collector. ``cycle.time`` is the wall-clock timestamp the
        # autoscaler stamped on the cycle at the top of
        # ``PreflightBuilder.build``; the throughput sampler subtracts
        # it from the previously snapshotted ``(count, timestamp)`` to
        # recover the tasks/sec rate for each stage.
        throughput_samples = services.measurements.consume_throughput_samples(now_ts=cycle.time)
        deltas = self._compute_intent_deltas(
            services,
            cycle,
            throughput_samples=throughput_samples,
        )
        cycle.intent = IntentPlan(deltas=deltas)

    def _compute_intent_deltas(
        self,
        services: IntentServices,
        cycle: AutoscaleCycle,
        *,
        throughput_samples: dict[str, float],
    ) -> dict[str, int]:
        """Compute the per-stage signed worker-count intent for this cycle.

        Orchestrates the three injected strategies for each non-
        finished stage: :class:`QuiescenceGate` (cold-start skip
        + hot-pending clamp), :class:`TrustGate` (consecutive-
        valid-sample accounting + below-threshold clamp), and
        :class:`StageDecisionPipeline` (classifier + pressure +
        stabilization). Returns
        ``{stage_name: signed_intent}`` over the surviving
        stages.

        The per-stage :class:`StageTopologyContext` is projected
        on-the-fly from ``cycle.bottleneck.identity`` and the
        consumer stage's DAG index - no per-stage mirror lives
        on the runtime state. A disengaged bottleneck or a stale
        identity (bottleneck name no longer in the pipeline)
        collapses to the default ``StageTopologyContext()``.

        Raises:
            SchedulerInvariantError: Stage in ``problem_state`` is
                missing from ``services.stage_states``.

        """
        pipeline = services.pipeline
        ctx = cycle.ctx
        problem_state = cycle.problem_state
        now = cycle.time
        intents: dict[str, int] = {}
        worker_ids_by_stage = ctx.worker_ids_by_stage()
        # Snapshot the cycle-wide bottleneck identity once; the
        # per-stage projection runs inside the loop so each
        # consumer sees its own ``StageTopologyContext``.
        bottleneck_identity = cycle.bottleneck.identity

        for stage_index, runtime_stage in enumerate(problem_state.rust.stages):
            if runtime_stage.is_finished:
                continue
            stage_name = runtime_stage.stage_name
            stage_state = services.stage_states.get(stage_name)
            if stage_state is None:
                # Shape drift between problem and problem_state is a
                # scheduler defect (the pre-Manual shape check at
                # ``_check_problem_state_shape_before_manual`` should
                # already have caught this). Raise the project's
                # must-fail signal so the caller routes the failure
                # through the same path as every other phase invariant.
                msg = (
                    f"problem_state stage {stage_name!r} not found in setup() "
                    f"state map (known: {sorted(services.stage_states)}); "
                    "problem and problem_state shapes disagree."
                )
                raise SchedulerInvariantError(msg)
            stage_cfg = pipeline.stage_config(stage_name)
            current_workers = len(worker_ids_by_stage[stage_index])
            pending_actors = runtime_stage.num_pending_actors
            # The quiescence check evaluates the snapshot's ready
            # count (workers visible to the actor pool at
            # observation time), not the post-Phase-B planner
            # count. Phase B's floor enforcement may have staged
            # a fresh add for this same cycle, but that add is
            # not yet visible to the streaming snapshot's slot
            # signals - the per-worker measurements still come
            # from the ready actors that existed at observation
            # time, of which there are zero in the cold-start case.
            ready_at_snapshot = len(runtime_stage.worker_groups)
            # The stabilization-window buffer is allocated alongside
            # the runtime state in ``setup()``; a missing entry
            # would mean the ``problem`` -> ``problem_state`` shape
            # contract above somehow admitted a stage the runtime
            # state map already rejected.
            history = services.recommendation_histories[stage_name]

            quiescence_active = self.quiescence.is_active(stage_cfg=stage_cfg, pending_actors=pending_actors)
            if self.quiescence.is_cold_start(quiescence_active=quiescence_active, ready_at_snapshot=ready_at_snapshot):
                self.quiescence.reset_classifier(stage_state=stage_state, recommendation_history=history)
                logger.debug(
                    f"saturation-aware: stage {stage_name!r} cold-start quiescent "
                    f"(pending={pending_actors}, ready=0); skipping intent pipeline."
                )
                continue

            # Per-worker measurement grace: drop slot-signal
            # contributions from workers younger than
            # ``worker_warmup_measurement_grace_s`` so the EWMA
            # does not absorb cold-start noise from freshly-ready
            # actors. ``input_queue_depth`` is a stage-level
            # signal (not per-worker) and is therefore left
            # unfiltered.
            num_used_slots, num_empty_slots = services.warmup.filter_slot_signals(
                runtime_stage,
                stage_cfg,
                now=now,
            )
            cycle_has_fresh_sample = num_used_slots + num_empty_slots > 0
            self.trust.update_freshness_counter(
                stage_state=stage_state,
                stage_cfg=stage_cfg,
                cycle_has_fresh_sample=cycle_has_fresh_sample,
                recommendation_history=history,
            )

            stage_state.growth.capacity_target_workers = services.capacity.target_workers(
                stage_state=stage_state,
                stage_cfg=stage_cfg,
                input_queue_depth=runtime_stage.input_queue_depth,
                observed_throughput=throughput_samples.get(stage_name, 0.0),
                slots_per_worker=runtime_stage.slots_per_worker,
                stage_name=stage_name,
            )
            topology = project_stage_topology(
                stage_index=stage_index,
                bottleneck_engaged=bottleneck_identity.engaged,
                bottleneck_stage_name=bottleneck_identity.stage_name,
                stage_names=pipeline.stage_names,
            )
            delta = self.decision_pipeline.compute_recommendation(
                stage_state=stage_state,
                topology=topology,
                num_used_slots=num_used_slots,
                num_empty_slots=num_empty_slots,
                input_queue_depth=runtime_stage.input_queue_depth,
                current_workers=current_workers,
                config=stage_cfg,
                recommendation_history=history,
                observed_throughput_sample=throughput_samples.get(stage_name, 0.0),
                pipeline_name=services.pipeline_name,
            )

            if self.trust.should_clamp_to_zero(stage_state=stage_state, stage_cfg=stage_cfg, delta=delta):
                logger.debug(
                    f"saturation-aware: stage {stage_name!r} valid samples "
                    f"{stage_state.classifier.valid_signal_samples}/{stage_cfg.min_data_points} - "
                    f"trust gate clamping recommendation {delta} to 0."
                )
                delta = 0
            if self.quiescence.should_clamp_positive_delta(quiescence_active=quiescence_active, delta=delta):
                logger.debug(
                    f"saturation-aware: stage {stage_name!r} hot-pending quiescent "
                    f"(pending={pending_actors}, ready={ready_at_snapshot}); "
                    f"clamping Phase C intent +{delta} -> 0."
                )
                delta = 0
            intents[stage_name] = delta
        return intents


__all__ = ["IntentPhase", "QuiescenceGate", "TrustGate"]
