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

Three modes shape how aggressively the scheduler scales up a stage.
Mode transitions depend on the final delta the caller actually applied,
not on the classifier state alone:

::

    shrink = delta_executed < 0
    grow/no-op = delta_executed >= 0

                           first executed shrink
    +-----------+ ----------------------------------------> +----------+
    | ACQUIRING |                                           | TRACKING |
    +-----------+ <-- grow/no-op: stay, streak += 1         +----------+
                                                                  ^  |
                                                                  |  |
                         HOLD timer expires with no shrink        |  | executed shrink
                         streak >= stabilization window           |  v
                                                             +----------+
                                                             |   HOLD   |
                                                             +----------+
                                                               ^      |
                                                               |      |
                         executed shrink restarts timer        |      | grow/no-op before
                         stay HOLD, streak = 1                 |      | window expires:
                                                               |      | stay, streak += 1
                                                               +------+

Shrink transition table:

    +-----------+-------------------+------------+
    | prev_mode | delta_executed < 0 | next mode  |
    +-----------+-------------------+------------+
    | ACQUIRING | first shrink      | TRACKING   |
    | TRACKING  | later shrink      | HOLD       |
    | HOLD      | re-shrink         | HOLD       |
    +-----------+-------------------+------------+

Non-shrink transition table:

    +-----------+-------------------------------+----------------------+
    | prev_mode | condition                     | next mode            |
    +-----------+-------------------------------+----------------------+
    | ACQUIRING | delta_executed >= 0           | ACQUIRING, streak+1  |
    | TRACKING  | delta_executed >= 0           | TRACKING, streak+1   |
    | HOLD      | streak < stabilization_window | HOLD, streak+1       |
    | HOLD      | streak >= stabilization_window| TRACKING, streak=1   |
    +-----------+-------------------------------+----------------------+

Scale-up intent by growth mode:

    +-----------+-----------------------+------------------------------+
    | mode      | SATURATED             | SATURATED_CRITICAL           |
    +-----------+-----------------------+------------------------------+
    | ACQUIRING | +ceil(0.25 * current) | +ceil(0.50 * current)        |
    | TRACKING  | +1                    | +2                           |
    | HOLD      |  0                    | +1                           |
    +-----------+-----------------------+------------------------------+

``compute_growth_mode_transition`` is the pure-function transition rule.
It returns the (mode, streak) pair for the next cycle.

The streak resets to 1 on any shrink event or mode transition; it
increments otherwise. A re-shrink while already in HOLD restarts the
stabilization timer (streak resets to 1 even though the mode does
not change), so consecutive shrinks always extend the stabilization
window.
"""

from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def compute_growth_mode_transition(
    *,
    prev_mode: GrowthMode,
    prev_streak: int,
    delta_executed: int,
    config: SaturationAwareStageConfig,
) -> tuple[GrowthMode, int]:
    """Compute next cycle's ``(growth_mode, growth_streak)``.

    Args:
        prev_mode: Current growth mode.
        prev_streak: Cycles spent in ``prev_mode`` so far. ``0`` is
            the cold-start sentinel (initial ``_StageRuntimeState``
            value before the first autoscale cycle); ``>= 1`` is the
            steady-state value.
        delta_executed: Final delta applied to this stage this cycle
            (positive = added, negative = removed, zero = no action).
        config: Per-stage config carrying
            ``stabilization_window_cycles_down``.

    Returns:
        Pair of ``(new_mode, new_streak)`` for the next cycle.

    Raises:
        ValueError: If ``prev_streak`` is negative.

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
