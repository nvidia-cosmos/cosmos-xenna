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

"""Per-stage decision pipeline -- composes every primitive each cycle.

Per-cycle flow::

    snapshot -> resolve classifier signal -> classify -> update streak
             -> fire-gate -> compute intent delta -> stabilization gate
             -> return recommendation

The returned delta is the algorithm's recommendation only. The
growth-mode state machine is advanced separately by
``record_executed_delta`` after the planner has committed Phase C
(grow) and Phase D (shrink); the executed delta passed in is the
net post-commit worker change for the stage. Splitting the
recommendation from the execution recording lets hard worker caps,
fraction clamps, and allocation failures shrink the planner's
output below the recommendation without the growth-mode timer
observing a delta that never landed in the cluster.
"""

from cosmos_xenna.pipelines.private.scheduling_py.classifier import classify
from cosmos_xenna.pipelines.private.scheduling_py.decisions import (
    compute_delta,
    should_fire_action,
    update_streak,
)
from cosmos_xenna.pipelines.private.scheduling_py.growth_mode import compute_growth_mode_transition
from cosmos_xenna.pipelines.private.scheduling_py.stabilization import (
    _RecommendationHistory,
    apply_stabilization_gate,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import (
    GrowthMode,
    _StageRuntimeState,
    compute_slots_empty_ratio,
    update_ewma,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def run_per_stage_pipeline(
    *,
    stage_state: _StageRuntimeState,
    num_used_slots: int,
    num_empty_slots: int,
    input_queue_depth: int,
    current_workers: int,
    config: SaturationAwareStageConfig,
    recommendation_history: _RecommendationHistory | None = None,
) -> int:
    """Compute the per-stage scaling recommendation for one cycle.

    Mutates ``stage_state`` in place (classifier state, classifier
    streak, EWMA cache, ``prev_workers``); returns the (possibly
    stabilization-gated) recommended delta. The growth-mode state
    machine is NOT advanced here. The caller is the only component
    that knows what the planner actually executed; it must call
    ``record_executed_delta`` after Phase C / Phase D commit so
    HOLD / ACQUIRING / TRACKING timers observe the post-commit
    delta rather than the pre-commit recommendation.

    Args:
        stage_state: Per-stage runtime state. Mutated in place.
        num_used_slots: Slots currently occupied.
        num_empty_slots: Slots currently free.
        input_queue_depth: Tasks waiting upstream of this stage.
        current_workers: Worker count at cycle start.
        config: Per-stage configuration.
        recommendation_history: Optional asymmetric stabilization-window
            buffer. When provided, the raw delta is recorded into the
            buffer and replaced with ``0`` if the buffer cannot yet
            confirm a sustained recommendation in the same direction;
            this gates Phase C / Phase D against acting on a single
            noisy cycle. ``None`` (used by helper-direct tests) skips
            the stabilization layer entirely.

    Returns:
        Signed recommendation delta: positive to add, negative to
        remove, zero for no action. The scheduler must pass the
        executed post-commit delta to ``record_executed_delta`` for
        every stage that participated in this cycle's intent
        computation, including stages that recommended zero.

    Raises:
        RuntimeError: If ``stage_state.resolved_thresholds`` is
            ``None`` (the scheduler's first ``autoscale()`` cycle
            populates it; tests that build state directly must call
            ``_resolve_auto_thresholds`` first).

    """
    if stage_state.resolved_thresholds is None:
        msg = (
            f"_StageRuntimeState for stage {stage_state.stage_name!r} has no "
            "resolved_thresholds; SaturationAwareScheduler._ensure_thresholds_resolved "
            "(invoked at the top of autoscale()) is the only legitimate populator. "
            "Tests that construct _StageRuntimeState directly must populate "
            "resolved_thresholds via _resolve_auto_thresholds(...) at fixture time."
        )
        raise RuntimeError(msg)

    classifier_input = _resolve_classifier_signal(
        stage_state=stage_state,
        num_used_slots=num_used_slots,
        num_empty_slots=num_empty_slots,
        config=config,
    )
    if classifier_input is None:
        # Zero ready actors, no prior valid EWMA. The structural
        # worker-floor step independently bootstraps; the classifier
        # holds for this cycle. The growth-mode transition is left to
        # the scheduler's post-Phase-D ``record_executed_delta`` call
        # which sees delta_executed=0 for a stage that took no action.
        stage_state.prev_workers = current_workers
        return 0

    new_classifier_state = classify(
        slots_empty_ratio_ewma=classifier_input,
        input_queue_depth=input_queue_depth,
        prev_state=stage_state.classifier_state,
        saturation_threshold=stage_state.resolved_thresholds.saturation_threshold,
        activation_threshold=stage_state.resolved_thresholds.activation_threshold,
        config=config,
    )
    stage_state.classifier_streak = update_streak(
        stage_state.classifier_state,
        stage_state.classifier_streak,
        new_classifier_state,
    )
    stage_state.classifier_state = new_classifier_state

    # Growth-mode kill switch: when the state machine is disabled, force
    # the magnitude calculation to use ``TRACKING`` (additive +1 / +2) so
    # ACQUIRING multiplicative growth and HOLD post-shrink suppression do
    # not influence the per-cycle decision. ``record_executed_delta`` is
    # still called by the scheduler, but with the kill switch off it
    # short-circuits to a no-op (see that function's body).
    effective_growth_mode = stage_state.growth_mode if config.enable_growth_mode_state_machine else GrowthMode.TRACKING

    if should_fire_action(new_classifier_state, stage_state.classifier_streak, config):
        delta = compute_delta(
            new_classifier_state,
            effective_growth_mode,
            current_workers,
            config,
        )
    else:
        delta = 0

    if recommendation_history is not None:
        # Record-then-gate is encapsulated in ``apply_stabilization_gate`` so
        # callers cannot accidentally gate without recording, which would
        # leave the buffer one cycle behind reality and weaken every future
        # gate decision.
        delta = apply_stabilization_gate(recommendation_history, delta)

    stage_state.prev_workers = current_workers
    return delta


def record_executed_delta(
    *,
    stage_state: _StageRuntimeState,
    delta_executed: int,
    config: SaturationAwareStageConfig,
) -> None:
    """Advance the growth-mode state machine with the post-commit delta.

    Called by the scheduler exactly once per stage per cycle, after
    Phase C (grow) and Phase D (shrink) have committed their planner
    mutations. ``delta_executed`` is the net worker change for the
    stage (``post_phase_d_count - pre_phase_c_count``); it can be
    smaller in magnitude than the recommendation if hard caps,
    fractional clamps, or allocation failures throttled the planner.

    When ``config.enable_growth_mode_state_machine`` is ``False`` the
    state machine is short-circuited - neither ``growth_mode`` nor
    ``growth_streak`` is mutated. ``run_per_stage_pipeline`` already
    forces ``TRACKING`` magnitudes in that mode, so leaving the runtime
    state frozen at its construction-time defaults keeps the per-stage
    history clean and re-enabling the flag mid-run resumes from the
    original ACQUIRING entry point rather than from a stale HOLD/etc.

    Args:
        stage_state: Per-stage runtime state. Only
            ``growth_mode`` and ``growth_streak`` are mutated.
        delta_executed: Signed post-commit delta.
        config: Per-stage configuration.
    """
    if not config.enable_growth_mode_state_machine:
        return
    stage_state.growth_mode, stage_state.growth_streak = compute_growth_mode_transition(
        prev_mode=stage_state.growth_mode,
        prev_streak=stage_state.growth_streak,
        delta_executed=delta_executed,
        config=config,
    )


def _resolve_classifier_signal(
    *,
    stage_state: _StageRuntimeState,
    num_used_slots: int,
    num_empty_slots: int,
    config: SaturationAwareStageConfig,
) -> float | None:
    """Determine the EWMA value the classifier reads this cycle.

    Returns:
        Live signal -> updated EWMA. Zero ready actors with a prior
        valid EWMA -> the carry-forward value (no new sample).
        Cold-start with no prior signal -> ``None`` (caller skips
        the classifier this cycle).

    """
    total_slots = num_used_slots + num_empty_slots
    if total_slots > 0:
        raw_ratio = compute_slots_empty_ratio(num_used_slots, num_empty_slots)
        new_ewma = update_ewma(
            stage_state.slots_empty_ratio_ewma,
            raw_ratio,
            config.slots_empty_ratio_smoothing_level,
        )
        stage_state.slots_empty_ratio_ewma = new_ewma
        stage_state.last_valid_slots_empty_ratio_ewma = new_ewma
        return new_ewma

    if stage_state.last_valid_slots_empty_ratio_ewma is not None:
        return stage_state.last_valid_slots_empty_ratio_ewma

    return None
