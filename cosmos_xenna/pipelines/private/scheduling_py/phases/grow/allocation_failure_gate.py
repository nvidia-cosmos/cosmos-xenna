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

"""``try_add_worker`` wrapper that absorbs ``AllocationError``.

The :func:`try_add_worker_with_defense` helper sits between phase
code that calls ``ctx.try_add_worker`` and the planner's allocation
loop, classifying :class:`resources.AllocationError` against the
phase's :class:`AllocationFailureGate` (whose value type lives in
``state/allocation_failure_gate.py``) and the
``skip_cycle_on_allocation_error`` cluster-wide toggle.
"""

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.state.allocation_failure_gate import AllocationFailureGate


def try_add_worker_with_defense(
    *,
    ctx: data_structures.AutoscalePlanContext,
    stage_index: int,
    stage_name: str,
    pipeline_name: str,
    skip_cycle_on_allocation_error: bool,
    gate: AllocationFailureGate,
) -> data_structures.ProblemWorkerGroupState | None:
    """Call ``ctx.try_add_worker`` with the AllocationError tolerance layer.

    Catches only :class:`resources.AllocationError` so scheduler bugs
    surfacing as ``SchedulerInvariantError``, ``KeyError``,
    ``IndexError``, etc. propagate to the autoscaler thread instead
    of being silently re-routed through the absorb path (which would
    mask the real defect).

    On absorb the helper returns ``None`` and routes through
    :meth:`AllocationFailureGate.absorb`, which flips
    ``gate.aborted_cycle`` to ``True`` so the caller can short-circuit
    its per-stage loop. The ``None`` return is intentionally
    indistinguishable from the planner's "no placement" ``None`` so
    callers MUST re-check ``gate.aborted_cycle`` after every wrapped
    call when the absorb-vs-no-placement distinction matters.

    Args:
        ctx: Planner planning context the call is staged against.
        stage_index: Receiver stage index in problem order.
        stage_name: Receiver stage name used in the ERROR log and
            Counter tag.
        pipeline_name: Pipeline tag used in the Counter label and
            the cycle summary log.
        skip_cycle_on_allocation_error: When ``True`` an absorbed
            ``AllocationError`` returns ``None``; when ``False`` the
            original exception propagates after the ERROR log is
            emitted and the Counter is incremented.
        gate: Per-phase :class:`AllocationFailureGate` whose
            ``absorb`` method is invoked on the absorb path.

    Returns:
        The placed (or reused) :class:`ProblemWorkerGroupState`,
        ``None`` when the planner reported no placement, or ``None``
        when an ``AllocationError`` was absorbed. The caller
        disambiguates the last two via ``gate.aborted_cycle``.

    """
    try:
        return ctx.try_add_worker(stage_index)
    except resources.AllocationError as exc:
        gate.absorb(
            ctx=ctx,
            stage_name=stage_name,
            pipeline_name=pipeline_name,
            skip_cycle_on_allocation_error=skip_cycle_on_allocation_error,
            exc=exc,
        )
        return None


__all__ = ["try_add_worker_with_defense"]
