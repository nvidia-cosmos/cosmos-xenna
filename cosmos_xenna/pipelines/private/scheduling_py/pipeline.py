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
             -> update growth mode -> return delta

The returned delta is the algorithm's intent. Cluster-wide
feasibility (per-stage min/max, per-node caps, capacity, memory
pressure gate) is applied by the scheduler's main loop after this
function returns, so the growth-mode transition observes the
post-stabilization-gate intent delta produced here.
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
    """Run the per-cycle decision pipeline for one stage.

    Mutates ``stage_state`` in place; returns the (possibly
    stabilization-gated) delta.

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
            noisy cycle. The growth-mode transition observes the
            post-gate delta so HOLD timers advance during gated cycles
            instead of resetting on every suppressed shrink. ``None``
            (the legacy contract used by helper-direct tests) skips
            the stabilization layer entirely.

    Returns:
        Signed delta: positive to add, negative to remove, zero for
        no action.

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
        # holds and the growth-mode timer continues to tick.
        stage_state.growth_mode, stage_state.growth_streak = compute_growth_mode_transition(
            prev_mode=stage_state.growth_mode,
            prev_streak=stage_state.growth_streak,
            delta_executed=0,
            config=config,
        )
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

    if should_fire_action(new_classifier_state, stage_state.classifier_streak, config):
        delta = compute_delta(
            new_classifier_state,
            stage_state.growth_mode,
            current_workers,
            config,
        )
    else:
        delta = 0

    if recommendation_history is not None:
        # Record-then-gate is encapsulated in ``apply_stabilization_gate`` so
        # callers cannot accidentally gate without recording, which would
        # leave the buffer one cycle behind reality and weaken every future
        # gate decision. The growth-mode transition below sees the gated
        # delta so HOLD-after-shrink timers advance correctly when the
        # stabilization window suppresses a recommended action.
        delta = apply_stabilization_gate(recommendation_history, delta)

    stage_state.growth_mode, stage_state.growth_streak = compute_growth_mode_transition(
        prev_mode=stage_state.growth_mode,
        prev_streak=stage_state.growth_streak,
        delta_executed=delta,
        config=config,
    )
    stage_state.prev_workers = current_workers
    return delta


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
