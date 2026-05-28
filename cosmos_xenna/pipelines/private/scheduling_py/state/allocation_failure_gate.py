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

"""Per-phase ``AllocationError`` cycle-skip latch.

Each scheduler phase that adds workers (Manual / Floor / Grow) owns
its own :class:`AllocationFailureGate` instance so the three latches
stay independent. The companion module
``cluster/allocation_failures.py`` owns the ERROR-level formatter
and the Counter; the ``try_add_worker_with_defense`` helper in
``phases/grow/allocation_failure_gate.py`` owns the
``try_add_worker`` wrapper that absorbs the exception and calls
:meth:`AllocationFailureGate.absorb`. This module owns the per-phase
state value object only - no surrounding wrapper logic.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.cluster.allocation_failures import (
    emit_allocation_failure,
)


@attrs.define
class AllocationFailureGate:
    """Per-phase ``AllocationError`` cycle-skip latch.

    Each scheduler phase that adds workers (Manual / Floor / Grow)
    owns its own gate so the three latches stay independent. The
    gate's lifecycle:

    1. The phase calls :meth:`reset` at the top of every run.
    2. :meth:`absorb` is invoked from
       ``try_add_worker_with_defense`` when an
       :class:`AllocationError` is caught; the snapshot is logged,
       the counter is incremented, and ``aborted_cycle`` flips to
       ``True`` (or the exception re-raises when
       ``skip_cycle_on_allocation_error`` is ``False``).
    3. Phase code reads ``aborted_cycle`` after every wrapped call
       so the per-stage loop can short-circuit.
    4. :class:`PostCycleReporter` reads the latch at cycle tail
       to include the per-phase status in the cycle summary.

    Attributes:
        aborted_cycle: ``True`` when the latch was set by an absorbed
            allocation failure during the current phase run. Reset
            to ``False`` on :meth:`reset` (top of every phase run).

    """

    aborted_cycle: bool = False

    def reset(self) -> None:
        """Clear the latch at the top of a phase run."""
        self.aborted_cycle = False

    def absorb(
        self,
        *,
        ctx: data_structures.AutoscalePlanContext,
        stage_name: str,
        pipeline_name: str,
        skip_cycle_on_allocation_error: bool,
        exc: BaseException,
    ) -> None:
        """Log the fragmentation snapshot, raise or set the latch.

        The snapshot uses :meth:`AutoscalePlanContext.cluster_snapshot`
        (a clone of the planner's working cluster) rather than the
        static ``Problem.cluster_resources`` so the emitted log
        reports the resources the planner actually used during the
        cycle. The static snapshot is always cold-start-empty and
        would mislead operators into thinking the cluster had capacity
        available when in fact every node was already drained earlier
        in the same cycle by an upstream phase.

        Args:
            ctx: Planner planning context whose ``cluster_snapshot()``
                is included in the ERROR record.
            stage_name: Receiver stage name used in the ERROR log and
                Counter tag.
            pipeline_name: Pipeline tag used in the Counter label.
            skip_cycle_on_allocation_error: When ``False`` re-raises
                ``exc`` after emitting the ERROR record; when
                ``True`` sets ``aborted_cycle = True`` and returns.
            exc: The absorbed ``AllocationError`` (or other
                ``BaseException`` propagated by a caller that already
                classified the exception); included verbatim in the
                ERROR record.

        Raises:
            BaseException: ``exc`` itself when
                ``skip_cycle_on_allocation_error`` is ``False`` --
                the caller is responsible for matching the configured
                tolerance; this method does not narrow the raise.

        """
        emit_allocation_failure(
            stage_name=stage_name,
            pipeline_name=pipeline_name,
            cluster_resources=ctx.cluster_snapshot(),
            exc=exc,
        )
        if not skip_cycle_on_allocation_error:
            raise exc
        self.aborted_cycle = True


__all__ = ["AllocationFailureGate"]
