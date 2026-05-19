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

"""Phase-boundary invariant checks.

Defensive structural checks invoked between scheduler phases. A
violation indicates a bug in the scheduler itself, not an operator
configuration error: ``SchedulerInvariantError`` propagates out of
``autoscale`` so the caller can surface the corrupted plan rather
than apply it.

Each violation is logged at ERROR level immediately before raising
so operators see the diagnostic even if a higher-level supervisor
catches the exception and converts it to a less-informative outer
error.

The shape check is O(1) (``ctx.num_stages()``); the per-stage
counter check is O(stages) with two FFI hops per stage. Three
boundary calls per autoscale cycle leave the gate well inside the
per-cycle budget for clusters up to several thousand stages.
"""

import enum
from typing import NoReturn

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.errors import SchedulerInvariantError
from cosmos_xenna.utils import python_log as logger


class PhaseBoundary(enum.StrEnum):
    """Stable identifiers for the phase boundaries that invoke the gate.

    The values appear in ``SchedulerInvariantError`` messages and in
    the ``logger.error`` line, so operators can locate the violating
    phase from a log search.
    """

    PHASE_A = "phase_a"
    PHASE_B = "phase_b"
    PHASE_C = "phase_c"
    PHASE_D = "phase_d"
    INTO_SOLUTION = "into_solution"


def _log_and_raise_invariant(msg: str) -> NoReturn:
    """Log an invariant violation at ERROR level and raise ``SchedulerInvariantError``."""
    logger.error(f"scheduler invariant violation: {msg}")
    raise SchedulerInvariantError(msg)


def check_invariants_after_phase(
    *,
    phase_name: PhaseBoundary,
    problem: data_structures.Problem,
    ctx: data_structures.AutoscalePlanContext,
) -> None:
    """Validate planner-context invariants at a phase boundary.

    Invariants:

      1. ``ctx.num_stages()`` matches ``len(problem.stages)``.
      2. Every stage's ``pending_add_count`` and
         ``pending_remove_count`` is non-negative.

    The shape check runs first so a stage-count mismatch cannot
    leak through as ``IndexError`` from the per-stage counter loop.

    Args:
        phase_name: Identifier for the boundary, included in the
            error message so operators can locate the violating phase.
        problem: The frozen pipeline ``Problem``.
        ctx: The planning context whose state is being validated.

    Raises:
        SchedulerInvariantError: An invariant failed; the message
            names the phase, the violated invariant, and the
            offending stage / value.

    """
    num_problem_stages = len(problem.rust.stages)
    num_ctx_stages = ctx.num_stages()
    if num_ctx_stages != num_problem_stages:
        _log_and_raise_invariant(
            f"After {phase_name}: ctx.num_stages() reports "
            f"{num_ctx_stages} stages but problem has {num_problem_stages}. "
            "Planner-state corruption: stages were added or dropped between "
            "construction and the phase boundary. This is a scheduler defect; "
            "report it with the autoscale cycle's problem_state."
        )
    for stage_index in range(num_problem_stages):
        pending_add = ctx.pending_add_count(stage_index)
        pending_remove = ctx.pending_remove_count(stage_index)
        if pending_add < 0 or pending_remove < 0:
            stage_name = problem.rust.stages[stage_index].name
            _log_and_raise_invariant(
                f"After {phase_name}: stage {stage_name!r} (index {stage_index}) "
                f"has pending_add_count={pending_add}, "
                f"pending_remove_count={pending_remove} (must be >= 0). "
                "This is a scheduler defect; report it with the autoscale "
                "cycle's problem_state."
            )


def check_solution_shape(
    *,
    phase_name: PhaseBoundary,
    problem: data_structures.Problem,
    solution: data_structures.Solution,
) -> None:
    """Validate that the emitted ``Solution`` matches the problem's stage shape.

    Solutions are consumed positionally by their callers: a length
    mismatch silently misroutes every staged add and remove, so this
    invariant catches the bug at the source.

    Args:
        phase_name: Identifier for the boundary (used in the error
            message).
        problem: The frozen pipeline ``Problem``.
        solution: The emitted ``Solution``. Stage entries are
            consumed positionally by callers.

    Raises:
        SchedulerInvariantError: ``len(solution.rust.stages)`` does
            not equal ``len(problem.rust.stages)``.

    """
    num_problem_stages = len(problem.rust.stages)
    num_solution_stages = len(solution.rust.stages)
    if num_solution_stages != num_problem_stages:
        _log_and_raise_invariant(
            f"After {phase_name}: Solution has {num_solution_stages} stages "
            f"but problem has {num_problem_stages}. The Solution is consumed "
            "positionally; a shape mismatch silently misroutes worker "
            "mutations. This is a scheduler defect; report it with the "
            "autoscale cycle's problem_state."
        )
