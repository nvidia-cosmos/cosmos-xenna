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

"""Slow-start growth-mode state machine.

Three modes orchestrate post-shrink stabilization per stage:
``ACQUIRING`` (no shrink yet), ``TRACKING`` (one or more shrinks
observed), ``HOLD`` (post-shrink window blocks ``SATURATED`` grow;
``SATURATED_CRITICAL`` is never blocked). The pure transition rule
is ``compute_growth_mode_transition``. See
``docs/scheduler/saturation-aware/`` for the algorithm.
"""

from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import GrowthMode
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def compute_growth_mode_transition(
    *,
    prev_mode: GrowthMode,
    prev_streak: int,
    delta_executed: int,
    config: SaturationAwareStageConfig,
) -> tuple[GrowthMode, int]:
    """Compute next cycle's ``(growth_mode, growth_streak)``.

    Pure-function transition rule. ``prev_streak = 0`` is the
    cold-start sentinel. The sign of ``delta_executed`` drives
    the transition.

    Raises:
        ValueError: ``prev_streak`` is negative.

    """
    if prev_streak < 0:
        msg = f"prev_streak must be >= 0, got {prev_streak}"
        raise ValueError(msg)

    if delta_executed < 0:
        # Shrink event:
        #   ACQUIRING -> TRACKING (first ceiling discovery; not HOLD)
        #   TRACKING / HOLD -> HOLD (post-shrink stabilization; restart timer)
        new_mode = GrowthMode.TRACKING if prev_mode == GrowthMode.ACQUIRING else GrowthMode.HOLD
        return new_mode, 1

    if prev_mode == GrowthMode.HOLD and prev_streak >= config.stabilization_window_cycles_down:
        # HOLD timer expired with no further shrink: return to TRACKING.
        return GrowthMode.TRACKING, 1

    return prev_mode, prev_streak + 1
