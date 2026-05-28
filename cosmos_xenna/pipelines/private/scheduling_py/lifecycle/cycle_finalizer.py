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

"""Post-cycle lifecycle orchestration for ``SaturationAwareScheduler``.

Owns the deterministic order that the facade used to inline at
the tail of ``_autoscale_body`` after the phase runner returned::

    runner.run(cycle) -> [Phase A .. Phase D]
            |
            v
    +---------------------------------------------------+
    |             CycleFinalizer.finalize               |
    |                                                   |
    |  1. StuckPlanInvariant.check                      |
    |  2. PostCycleReporter.emit                        |
    |  3. cycle.ctx.into_solution() + shape check       |
    |  4. SchedulerLedgers.worker_ages := live ages     |
    |                                                   |
    +-----------------------+---------------------------+
                            |
                            v
                       Solution

Steps 1 and 2 must run before the drain because the heterogeneity
warn in step 2 reads ``cycle.bottleneck.d_k_now`` while it is
still on the cycle, not on the drained ``Solution``. Step 4 must
run after the drain because the live worker ids come from the
finalised planner context.

Splitting these steps out keeps the scheduler facade focused on
the public protocol (``setup`` / ``update_with_measurements`` /
``autoscale``) plus the cycle-level composition (``preflight ->
runner -> finalizer``). The facade publishes ``_last_cycle`` as a
single observability hook after this returns.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.invariants.checks import PhaseBoundary, check_solution_shape
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.post_cycle import PostCycleReporter, StuckPlanInvariant
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers


@attrs.frozen
class CycleFinalizer:
    """Lifecycle owner for the post-cycle order.

    Receives the finalised ``AutoscaleCycle`` plus the cycle-
    scoped invariant context, runs the stuck-plan monotonicity
    check, emits the post-cycle reporter, drains the planner
    context into a ``Solution``, persists live worker ages, and
    returns the ``Solution``.

    Attributes:
        stuck_plan_invariant: Owner of the stuck-plan counter
            monotonicity invariant.
        post_cycle_reporter: Owner of the four post-cycle balance
            signals and the cycle DEBUG summary.
        ledgers: Cross-cycle ledger the worker-age persist step
            writes to (the sole mutation this class performs).
        pipeline: ``PipelineModel`` captured by ``setup()``;
            provides the frozen pipeline ``Problem`` used by the
            post-drain solution-shape invariant.

    """

    stuck_plan_invariant: StuckPlanInvariant
    post_cycle_reporter: PostCycleReporter
    ledgers: SchedulerLedgers
    pipeline: PipelineModel

    def finalize(
        self,
        *,
        cycle: AutoscaleCycle,
        problem_state: data_structures.ProblemState,
        prev_stuck_plan_counters: dict[str, int],
    ) -> data_structures.Solution:
        """Run the post-cycle order and return the drained ``Solution``.

        Args:
            cycle: Cycle finalised by the phase runner.
            problem_state: Live runtime snapshot used by the
                stuck-plan invariant to classify the cycle.
            prev_stuck_plan_counters: Pre-Phase-C snapshot of
                ``stuck_plan_counters`` the invariant compares
                against to detect non-monotonic steps.

        Returns:
            The drained ``Solution`` ready for the streaming
            executor to apply.

        Raises:
            SchedulerInvariantError: ``setup()`` was not called,
                the stuck-plan counters stepped non-monotonically,
                or the drained ``Solution`` does not match the
                pipeline shape (all three are planner defects).

        """
        self.stuck_plan_invariant.check(
            problem_state=problem_state,
            prev_stuck_plan_counters=prev_stuck_plan_counters,
        )
        # ``PostCycleReporter.emit`` covers the four post-cycle
        # gauges, the regression warn, and the per-cycle DEBUG
        # summary. It must run before the drain because the
        # heterogeneity warn reads ``cycle.bottleneck.d_k_now``
        # while it is still on the cycle, not on the drained
        # ``Solution``.
        self.post_cycle_reporter.emit(cycle)
        solution = self._drain_to_solution(cycle)
        self._persist_worker_ages(cycle)
        return solution

    def _drain_to_solution(self, cycle: AutoscaleCycle) -> data_structures.Solution:
        """Drain ``cycle.ctx`` into a ``Solution`` and validate its shape."""
        solution = cycle.ctx.into_solution()
        check_solution_shape(
            phase_name=PhaseBoundary.SOLUTION_DRAIN,
            problem=self.pipeline.problem,
            solution=solution,
        )
        return solution

    def _persist_worker_ages(self, cycle: AutoscaleCycle) -> None:
        """Persist live worker ages from the finalised planning context.

        Defensive against the Rust contract that worker ids returned
        by ``worker_ids_by_stage`` are also present in ``worker_ages``:
        a missing entry defaults to age 0 (treated as freshly observed)
        rather than raising ``KeyError`` mid-cycle. Matches the
        missing-age semantics that ``FloorPolicy`` /
        ``SaturationPolicy`` apply when building donor candidates
        via ``context.worker_ages.get(worker_id, 0)``.
        """
        ctx = cycle.ctx
        live_worker_ids = {worker_id for stage_ids in ctx.worker_ids_by_stage() for worker_id in stage_ids}
        worker_ages = ctx.worker_ages()
        self.ledgers.worker_ages = {worker_id: worker_ages.get(worker_id, 0) for worker_id in live_worker_ids}


__all__ = ("CycleFinalizer",)
