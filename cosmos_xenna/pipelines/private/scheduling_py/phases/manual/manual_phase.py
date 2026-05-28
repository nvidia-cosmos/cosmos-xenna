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

"""Phase A: manual stage delete + grow.

Phase A is the operator-driven shrink / grow path. Manual stages
(those with ``requested_num_workers`` set) bring their worker count
to the requested value: surplus workers are deleted youngest-first
so long-lived warmed workers survive the cycle; deficits are grown
through ``ctx.try_add_worker`` with one WARN per stage when cluster
placement is exhausted before the request is met. Finished stages
are skipped on both paths.

The phase delegates to the two ``@attrs.frozen`` executors injected
via :class:`ManualServices` (``delete_executor`` and
``grow_executor``) so the phase entry point is a two-line
orchestration shell over those executors and the executors own
their own state (delete is stateless; grow owns the manual
allocation-failure gate).
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.services import ManualServices
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle


@attrs.frozen
class ManualPhase:
    """Operator-driven delete + grow for manual stages.

    Stateless ``@attrs.frozen`` phase. Delegates to the executors
    on :class:`ManualServices` so the per-cycle entry point owns
    only the ordering (delete before grow). The post-phase
    invariant check guards against planner-state drift introduced
    by either pass.
    """

    def run(self, cycle: AutoscaleCycle, services: ManualServices) -> None:
        """Execute manual delete and grow against the planner context.

        Manual delete runs first so any subsequent grow request
        operates on the post-shrink worker count. The runner
        invokes the post-phase planner invariant after this method
        returns.

        Raises:
            IndexError: The planner rejects a stage index.
            SchedulerInvariantError: Planner-state divergence
                inside the shrink path.

        """
        services.delete_executor.execute(cycle=cycle, services=services)
        services.grow_executor.execute(cycle=cycle, services=services)


__all__ = ["ManualPhase"]
