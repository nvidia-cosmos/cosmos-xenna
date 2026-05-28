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

"""Runner-driven phase-boundary invariant suite.

Centralises the post-phase invariant checks that previously lived inside
each ``Phase.run()`` body. The ``CycleRunner`` invokes the matching
``check_after_<phase>`` method after the producing phase, keeping phase
classes focused on per-stage decision logic and producing a single
audit trail for invariant ownership.

Method names track the semantic phase identity rather than the
historical phase letter: ``check_after_manual`` /
``check_after_floor`` / ``check_after_grow`` / ``check_after_shrink``.

The suite never holds per-cycle state. It binds to the live problem
resolver and a stage-floor computation closure so the same phase-floor
contract used by ``FloorPhase`` / ``SaturationShrinkPhase`` is reused
for the post-Shrink floor invariant.
"""

from collections.abc import Mapping

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.invariants.checks import (
    PhaseBoundary,
    check_floor_after_shrink,
    check_invariants_after_phase,
    check_no_nan_in_classifier_state,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.capacity import FloorCalculator
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState


@attrs.frozen
class PhaseInvariantSuite:
    """Phase-boundary invariant orchestration owned by ``CycleRunner``.

    Each ``check_after_<phase>`` method runs the invariants that
    previously lived at the bottom of the matching ``Phase.run()``
    body. Centralising them here keeps phase classes focused on
    their per-stage decision logic and gives the runner a single
    audit trail for invariant ownership.

    Attributes:
        pipeline: ``PipelineModel`` captured by ``setup()``;
            provides the frozen ``Problem`` consumed by every
            ``check_*`` body.
        ledgers: Cross-cycle ledger; read for the per-stage
            ``stage_states`` view consumed by the post-Grow NaN
            invariant.
        floors: ``FloorCalculator`` bound to ``pipeline``; used
            to compute the per-stage floor map consumed by the
            post-Shrink floor invariant.

    """

    pipeline: PipelineModel
    ledgers: SchedulerLedgers
    floors: FloorCalculator

    def check_after_manual(self, cycle: AutoscaleCycle) -> None:
        """Run the post-Manual planner invariant.

        Raises:
            SchedulerInvariantError: The planner / runtime-state
                invariant failed after manual phase mutations
                landed.

        """
        check_invariants_after_phase(
            phase_name=PhaseBoundary.MANUAL,
            problem=self.pipeline.problem,
            ctx=cycle.ctx,
        )

    def check_after_floor(self, cycle: AutoscaleCycle) -> None:
        """Run the post-Floor planner invariant.

        Raises:
            SchedulerInvariantError: The planner / runtime-state
                invariant failed after the floor phase landed its
                mutations.

        """
        check_invariants_after_phase(
            phase_name=PhaseBoundary.FLOOR,
            problem=self.pipeline.problem,
            ctx=cycle.ctx,
        )

    def check_after_grow(self, cycle: AutoscaleCycle) -> None:
        """Run the post-Grow planner invariant and classifier NaN gate.

        The NaN gate fires loud the moment a defect introduces
        ``NaN`` / ``+/-Inf`` into the per-stage classifier EWMA so
        the corruption does not silently propagate into the Shrink
        decision.

        Raises:
            SchedulerInvariantError: The planner invariant failed,
                or a per-stage EWMA value is not finite.

        """
        check_invariants_after_phase(
            phase_name=PhaseBoundary.GROW,
            problem=self.pipeline.problem,
            ctx=cycle.ctx,
        )
        check_no_nan_in_classifier_state(
            phase_name=PhaseBoundary.GROW,
            stage_runtime_states=self._stage_states(),
        )

    def check_after_shrink(self, cycle: AutoscaleCycle) -> None:
        """Run the post-Shrink planner invariant and floor invariant.

        The floor invariant uses ``cycle.pre_shrink_worker_counts``
        to distinguish a Shrink defect from a Floor grace-window
        miss; the runner captures the snapshot before invoking
        Shrink so the comparison is well-defined.

        Raises:
            SchedulerInvariantError: The planner invariant failed,
                or a stage finished Shrink below its computed
                floor.

        """
        problem = self.pipeline.problem
        check_invariants_after_phase(
            phase_name=PhaseBoundary.SHRINK,
            problem=problem,
            ctx=cycle.ctx,
        )
        num_nodes = problem.rust.cluster_resources.num_nodes()
        stage_floors = self.floors.compute(num_nodes)
        check_floor_after_shrink(
            phase_name=PhaseBoundary.SHRINK,
            problem=problem,
            problem_state=cycle.problem_state,
            ctx=cycle.ctx,
            stage_floors=stage_floors,
            pre_shrink_worker_counts=cycle.pre_shrink_worker_counts,
        )

    def _stage_states(self) -> Mapping[str, StageRuntimeState]:
        """Snapshot of the live per-stage runtime-state map for the NaN gate."""
        return self.ledgers.stage_states


__all__ = ["PhaseInvariantSuite"]
