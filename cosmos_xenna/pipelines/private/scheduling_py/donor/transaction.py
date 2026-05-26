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

"""Probe + atomic-remove transaction for a donor plan.

``DonorTransaction`` wraps the two-step ``ctx.probe_add_after_removals``
+ ``ctx.remove_workers_atomically`` sequence into a single
behaviour bundle. ``commit`` returns a ``DonorCommitOutcome`` so
callers dispatch on the named failure modes rather than juggling
booleans.

The transaction is mode-agnostic: floor mode and saturation mode
both go through this code path. Failure-mode interpretation
(receiver retry, operator-actionable error, scheduler defect)
lives at the caller.
"""

from typing import TYPE_CHECKING

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorCommitOutcome, DonorPlan

if TYPE_CHECKING:
    from cosmos_xenna.pipelines.private.data_structures import AutoscalePlanContext


@attrs.frozen
class DonorTransaction:
    """Probe-and-commit transaction for a donor plan.

    Stateless ``@attrs.frozen`` behaviour bundle. One instance per
    donor coordinator. ``commit`` is the only entry point and
    always returns; ``DonorCommitOutcome.committed`` distinguishes
    success from the two failure modes (``probe_failed``,
    ``atomic_remove_failed``).

    """

    def commit(
        self,
        *,
        plan: DonorPlan,
        ctx: "AutoscalePlanContext",
    ) -> DonorCommitOutcome:
        """Probe placement, then atomically remove the donor workers.

        ``probe_add_after_removals`` re-validates the placement
        without mutating planner state; ``remove_workers_atomically``
        commits the donor removals in a single transaction. The
        outcome routes through ``DonorCommitOutcome``:

        - ``committed`` -> caller owns the receiver retry +
          policy on-commit hook.
        - ``probe_failed`` -> selection invalidated by concurrent
          mutation; caller should record reject and continue.
        - ``atomic_remove_failed`` -> probe-approved plan failed
          atomic removal; this is a scheduler invariant break and
          the caller should raise.

        """
        removals = [(w.stage_index, w.worker_id) for w in plan.removals]

        probe = ctx.probe_add_after_removals(removals, plan.receiver_stage_index)
        if not probe.feasible:
            return DonorCommitOutcome(
                committed=False,
                probe_failed=True,
                atomic_remove_failed=False,
                placement_reject_reason=probe.reject_reason or "",
            )

        if not ctx.remove_workers_atomically(removals):
            return DonorCommitOutcome(
                committed=False,
                probe_failed=False,
                atomic_remove_failed=True,
                placement_reject_reason="",
            )

        return DonorCommitOutcome(
            committed=True,
            probe_failed=False,
            atomic_remove_failed=False,
            placement_reject_reason="",
        )


__all__ = ("DonorTransaction",)
