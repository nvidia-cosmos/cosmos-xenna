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

``run_per_stage_pipeline`` is the single entry point the scheduler's
main loop invokes for each stage. It consumes raw signals from the
stage's ``ProblemStageState`` snapshot, mutates the stage's
``_StageRuntimeState`` in place to reflect the cycle's outcome, and
returns the unclamped per-cycle delta.

Pipeline order::

    raw stage snapshot
    (used_slots, empty_slots, input_queue_depth, current_workers)
          |
          v
    +------------------------------------------------------------+
    | resolve classifier signal                                  |
    |                                                            |
    |   live slots:              compute ratio -> update EWMA     |
    |   zero actors + old EWMA:  carry forward last valid EWMA    |
    |   zero actors + no EWMA:   no classifier input this cycle   |
    +------------------------------------------------------------+
          |                                      |
          | classifier input                     | no classifier input
          v                                      v
    +-------------------------+          +----------------------------+
    | classify                |          | tick growth mode with       |
    | update classifier_streak|          | delta=0, return 0           |
    | should_fire_action      |          +----------------------------+
    +-------------------------+
          |
          v
    +------------------------------------------------------------+
    | compute intent delta                                       |
    |                                                            |
    |   fire gate closed -> 0                                    |
    |   SATURATED / CRITICAL -> scale-up intent                  |
    |   OVER_PROVISIONED -> scale-down intent                    |
    |   NORMAL / STARVED -> 0                                    |
    +------------------------------------------------------------+
          |
          v
    +------------------------------------------------------------+
    | update growth mode using the returned intent delta          |
    | ACQUIRING / TRACKING / HOLD state is mutated in place       |
    +------------------------------------------------------------+
          |
          v
    return intent delta

The returned delta is the algorithm's intent. Cluster-wide
feasibility (per-stage min/max, per-node caps, capacity, memory
pressure gate) is applied by the scheduler's main loop after this
function returns, so the growth-mode transition observes the intent
delta produced here.
"""

from cosmos_xenna.pipelines.private.scheduling_py.classifier import classify
from cosmos_xenna.pipelines.private.scheduling_py.decisions import (
    compute_delta,
    should_fire_action,
    update_streak,
)
from cosmos_xenna.pipelines.private.scheduling_py.growth_mode import compute_growth_mode_transition
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
) -> int:
    """Run the per-cycle decision pipeline for one stage.

    Composes ratio -> EWMA -> classify -> streak -> fire-gate ->
    delta -> growth-mode-transition. Mutates ``stage_state`` in
    place; returns the unclamped per-cycle delta.

    Args:
        stage_state: Per-stage runtime state. Mutated in place to
            reflect this cycle's outcome.
        num_used_slots: Slots currently occupied by in-flight tasks.
        num_empty_slots: Slots currently free.
        input_queue_depth: Tasks waiting upstream of this stage.
        current_workers: Worker count for the stage at the start of
            this cycle. Used as the base for multiplicative growth
            and for the scale-down fraction.
        config: Per-stage configuration.

    Returns:
        Signed integer: positive to add workers, negative to remove,
        zero for no action (NORMAL/STARVED, fire-gate not reached,
        or zero-actor cold-start with no carry-forward).

    """
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

    Three cases:

      1. Live signal (``num_used_slots + num_empty_slots > 0``):
         compute the raw ratio, update the EWMA, refresh the
         carry-forward field.
      2. Zero ready actors with a prior valid EWMA: hold the EWMA
         steady (no new sample) and reuse the carry-forward value
         for the classifier so a transient zero-actor moment does
         not flip the classification.
      3. Zero ready actors with no prior signal (cold start):
         return ``None`` to signal the caller that the classifier
         must be skipped this cycle.

    Returns:
        The classifier-input EWMA value, or ``None`` for case 3.

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
