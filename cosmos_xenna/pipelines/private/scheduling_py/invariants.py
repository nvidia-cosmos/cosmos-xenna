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
counter check is O(stages) with two FFI hops per stage. Phase C
adds an O(stages) classifier-state finiteness sweep and Phase D
adds an O(stages) floor sweep plus an O(stages) stuck-plan
monotonicity sweep. All passes are pure-Python and allocate at
most one shallow dict view, so the per-cycle budget remains
negligible for clusters up to several thousand stages.
"""

import enum
import math
from typing import NoReturn

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.state import _StageRuntimeState
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


def check_no_nan_in_classifier_state(
    *,
    phase_name: PhaseBoundary,
    stage_runtime_states: dict[str, _StageRuntimeState],
) -> None:
    """Reject any per-stage EWMA value that is ``NaN`` or ``+/-Inf``.

    Validates ``slots_empty_ratio_ewma``,
    ``last_valid_slots_empty_ratio_ewma``, and ``pressure_ewma`` on
    each entry; ``None`` is treated as a valid cold-start sentinel.

    Args:
        phase_name: Identifier for the boundary, included in the
            error message.
        stage_runtime_states: Per-stage runtime map keyed by stage
            name (typically the scheduler's ``_stage_states``).

    Raises:
        SchedulerInvariantError: Any per-stage EWMA value is not
            finite. The message names the phase, the stage, the
            field, and the offending value.

    """
    for stage_name, runtime in stage_runtime_states.items():
        for field_name, value in (
            ("slots_empty_ratio_ewma", runtime.slots_empty_ratio_ewma),
            ("last_valid_slots_empty_ratio_ewma", runtime.last_valid_slots_empty_ratio_ewma),
            ("pressure_ewma", runtime.pressure_ewma),
        ):
            if value is None:
                continue
            if not math.isfinite(value):
                _log_and_raise_invariant(
                    f"After {phase_name}: stage {stage_name!r} has "
                    f"{field_name}={value!r} (must be finite). NaN or Inf in "
                    "classifier state silently disables saturation-driven "
                    "scale-up. This is a scheduler defect; report it with the "
                    "autoscale cycle's problem_state."
                )


def check_floor_after_phase_d(
    *,
    phase_name: PhaseBoundary,
    problem: data_structures.Problem,
    problem_state: data_structures.ProblemState,
    ctx: data_structures.AutoscalePlanContext,
    stage_floors: dict[int, int],
    pre_phase_d_worker_counts: dict[int, int],
) -> None:
    """Validate Phase D did not reduce any stage below its floor or grow it.

    For every non-manual non-finished stage at index ``i``:
      ``min(pre_phase_d_worker_counts[i], stage_floors[i]) <= current
      <= pre_phase_d_worker_counts[i]``. The lower bound's ``min``
      tolerates Phase B grace-window states. The upper bound pins
      Phase D as remove-only.

    Args:
        phase_name: Identifier for the boundary.
        problem: The frozen pipeline ``Problem``.
        problem_state: The cycle's runtime snapshot.
        ctx: The planning context after Phase D (read-only).
        stage_floors: ``{stage_index: floor}`` from
            ``_compute_stage_floors``.
        pre_phase_d_worker_counts: ``{stage_index: count}``
            captured immediately before ``_run_phase_d_shrink``.

    Raises:
        SchedulerInvariantError: A non-manual non-finished stage's
            post-Phase-D worker count is outside the bounds above.

    """
    worker_ids_by_stage = ctx.worker_ids_by_stage()
    for stage_index, problem_stage in enumerate(problem.rust.stages):
        if problem_stage.requested_num_workers is not None:
            continue
        try:
            problem_state_stage = problem_state.rust.stages[stage_index]
        except IndexError as exc:
            _log_and_raise_invariant(
                f"After {phase_name}: stage {problem_stage.name!r} (index "
                f"{stage_index}) cannot be resolved against "
                f"problem_state.rust.stages: {type(exc).__name__}: {exc!r}. "
                "Planner-state shape mismatch. This is a scheduler defect; "
                "report it with the autoscale cycle's problem_state."
            )
        if problem_state_stage.is_finished:
            continue
        try:
            current = len(worker_ids_by_stage[stage_index])
            floor = stage_floors[stage_index]
            pre_d = pre_phase_d_worker_counts[stage_index]
        except (IndexError, KeyError) as exc:
            _log_and_raise_invariant(
                f"After {phase_name}: stage {problem_stage.name!r} (index "
                f"{stage_index}) cannot be resolved against the Phase-D "
                f"collections (worker_ids_by_stage, stage_floors, "
                f"pre_phase_d_worker_counts): {type(exc).__name__}: {exc!r}. "
                "Planner-state shape mismatch. This is a scheduler defect; "
                "report it with the autoscale cycle's problem_state."
            )
        floor_lower_bound = min(pre_d, floor)
        if current < floor_lower_bound:
            _log_and_raise_invariant(
                f"After {phase_name}: stage {problem_stage.name!r} "
                f"(index {stage_index}) has {current} workers but the "
                f"configured minimum-worker floor is {floor} (pre-Phase-D was "
                f"{pre_d}). Phase D shrank the stage below the floor. This is "
                "a scheduler defect; report it with the autoscale cycle's "
                "problem_state."
            )
        if current > pre_d:
            _log_and_raise_invariant(
                f"After {phase_name}: stage {problem_stage.name!r} "
                f"(index {stage_index}) grew from {pre_d} workers to "
                f"{current}. Phase D must only remove workers. This is a "
                "scheduler defect; report it with the autoscale cycle's "
                "problem_state."
            )


def check_stuck_plan_monotonicity(
    *,
    prev_counters: dict[str, int],
    curr_counters: dict[str, int],
) -> None:
    """Reject stuck-plan counter transitions other than reset-to-0 or strict +1.

    The only legal per-stage transitions are ``curr == 0`` or
    ``curr == prev + 1``. Stages absent from ``prev_counters``
    default to ``prev = 0``; stages absent from ``curr_counters``
    are treated as untouched and skipped.

    Args:
        prev_counters: Counter snapshot taken before Phase C ran.
        curr_counters: Counter snapshot taken after Phase C ran.

    Raises:
        SchedulerInvariantError: A stage's transition is neither a
            reset nor a strict +1 increment.

    """
    for stage_name, curr in curr_counters.items():
        prev = prev_counters.get(stage_name, 0)
        if prev < 0 or curr < 0:
            _log_and_raise_invariant(
                f"Stuck-plan counter for stage {stage_name!r} contains a "
                f"negative value (prev={prev}, curr={curr}). Counters must "
                "be non-negative cycle counts. This is a scheduler defect; "
                "report it with the autoscale cycle's problem_state."
            )
        if curr == 0:
            continue
        if curr == prev + 1:
            continue
        _log_and_raise_invariant(
            f"Stuck-plan counter for stage {stage_name!r} transitioned "
            f"from {prev} to {curr}; only a reset (to 0) or a strict +1 "
            "increment are legal. This is a scheduler defect; report it "
            "with the autoscale cycle's problem_state."
        )
