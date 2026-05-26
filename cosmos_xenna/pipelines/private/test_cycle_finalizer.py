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

"""Contract tests for :class:`CycleFinalizer`.

Verify the documented post-cycle order: stuck-plan invariant
runs first; the post-cycle reporter runs before
``cycle.ctx.into_solution()``; worker ages are persisted only on
the success path. Short-circuit behaviour is asserted through
observable mutations of the cross-cycle ledger and the order log.
"""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.cycle_finalizer import CycleFinalizer
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.post_cycle import PostCycleReporter, StuckPlanInvariant
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers

_SHAPE_PATH = "cosmos_xenna.pipelines.private.scheduling_py.lifecycle.cycle_finalizer.check_solution_shape"


def _stub_cycle(
    *,
    worker_ids_by_stage: list[list[str]] | None = None,
    worker_ages: dict[str, int] | None = None,
    solution: data_structures.Solution | None = None,
) -> AutoscaleCycle:
    """Build a sentinel cycle whose ``ctx`` exposes the post-cycle reads."""
    cycle = MagicMock(spec=AutoscaleCycle)
    ctx = MagicMock(spec=data_structures.AutoscalePlanContext)
    ctx.into_solution.return_value = solution or MagicMock(spec=data_structures.Solution)
    ctx.worker_ids_by_stage.return_value = worker_ids_by_stage or [["w1"]]
    ctx.worker_ages.return_value = worker_ages or {"w1": 3}
    cycle.ctx = ctx
    return cast(AutoscaleCycle, cycle)


def _stub_ledgers() -> SchedulerLedgers:
    """Build a ``SchedulerLedgers`` whose mutable ``worker_ages`` field is observable."""
    ledgers = MagicMock(spec=SchedulerLedgers)
    ledgers.worker_ages = {}
    return cast(SchedulerLedgers, ledgers)


def _build_finalizer(
    *,
    invariant: StuckPlanInvariant,
    reporter: PostCycleReporter,
    ledgers: SchedulerLedgers,
) -> CycleFinalizer:
    """Construct a ``CycleFinalizer`` with mock collaborators and a sentinel pipeline."""
    pipeline = MagicMock(spec=PipelineModel)
    pipeline.problem = MagicMock(spec=data_structures.Problem)
    return CycleFinalizer(
        stuck_plan_invariant=invariant,
        post_cycle_reporter=reporter,
        ledgers=ledgers,
        pipeline=cast(PipelineModel, pipeline),
    )


class TestStuckPlanInvariantRunsFirst:
    """``StuckPlanInvariant.check`` is the first observable side effect of ``finalize``."""

    def test_invariant_failure_short_circuits_before_reporter_and_drain(self) -> None:
        """A stuck-plan invariant raise short-circuits before the reporter or the drain runs."""
        invariant = MagicMock(spec=StuckPlanInvariant)
        invariant.check.side_effect = SchedulerInvariantError("boom")
        reporter = MagicMock(spec=PostCycleReporter)
        ledgers = _stub_ledgers()
        finalizer = _build_finalizer(invariant=invariant, reporter=reporter, ledgers=ledgers)
        cycle = _stub_cycle()
        with pytest.raises(SchedulerInvariantError, match="boom"):
            finalizer.finalize(
                cycle=cycle,
                problem_state=cast(data_structures.ProblemState, MagicMock(spec=data_structures.ProblemState)),
                prev_stuck_plan_counters={},
            )
        reporter.emit.assert_not_called()
        cast(MagicMock, cycle.ctx).into_solution.assert_not_called()
        assert ledgers.worker_ages == {}


class TestReporterBeforeDrain:
    """``PostCycleReporter.emit`` runs before the planner context drains into a ``Solution``."""

    def test_reporter_runs_before_into_solution(self) -> None:
        """``reporter.emit`` is observed before ``ctx.into_solution`` in the call log."""
        log: list[str] = []
        invariant = MagicMock(spec=StuckPlanInvariant)
        reporter = MagicMock(spec=PostCycleReporter)
        reporter.emit.side_effect = lambda _cycle: log.append("reporter")
        ledgers = _stub_ledgers()
        finalizer = _build_finalizer(invariant=invariant, reporter=reporter, ledgers=ledgers)
        cycle = _stub_cycle()

        def _record_drain() -> data_structures.Solution:
            log.append("drain")
            return cast(data_structures.Solution, MagicMock(spec=data_structures.Solution))

        cast(MagicMock, cycle.ctx).into_solution.side_effect = _record_drain
        with patch(_SHAPE_PATH):
            finalizer.finalize(
                cycle=cycle,
                problem_state=cast(data_structures.ProblemState, MagicMock(spec=data_structures.ProblemState)),
                prev_stuck_plan_counters={},
            )
        assert log == ["reporter", "drain"]


class TestWorkerAgesPersistedAfterDrain:
    """Worker ages are persisted only after the drain succeeds."""

    def test_worker_ages_are_persisted_with_defensive_age_zero_default(self) -> None:
        """``ledgers.worker_ages`` is set to ``{worker_id: age}`` for every live worker after drain."""
        invariant = MagicMock(spec=StuckPlanInvariant)
        reporter = MagicMock(spec=PostCycleReporter)
        ledgers = _stub_ledgers()
        cycle = _stub_cycle(
            worker_ids_by_stage=[["w1"], ["w2", "w3"]],
            worker_ages={"w1": 5, "w2": 0},
        )
        finalizer = _build_finalizer(invariant=invariant, reporter=reporter, ledgers=ledgers)
        with patch(_SHAPE_PATH):
            finalizer.finalize(
                cycle=cycle,
                problem_state=cast(data_structures.ProblemState, MagicMock(spec=data_structures.ProblemState)),
                prev_stuck_plan_counters={},
            )
        # Missing ``w3`` defaults to age 0 per the documented defensive contract.
        assert ledgers.worker_ages == {"w1": 5, "w2": 0, "w3": 0}


class TestSolutionShapeFailureRaisesBeforeWorkerAgesPersist:
    """A ``check_solution_shape`` failure raises before worker ages are persisted."""

    def test_shape_check_failure_aborts_worker_age_persist(self) -> None:
        """``ledgers.worker_ages`` stays empty when ``check_solution_shape`` raises post-drain."""
        invariant = MagicMock(spec=StuckPlanInvariant)
        reporter = MagicMock(spec=PostCycleReporter)
        ledgers = _stub_ledgers()
        finalizer = _build_finalizer(invariant=invariant, reporter=reporter, ledgers=ledgers)
        cycle = _stub_cycle()
        with (
            patch(_SHAPE_PATH, side_effect=SchedulerInvariantError("solution shape drift")),
            pytest.raises(SchedulerInvariantError, match="solution shape drift"),
        ):
            finalizer.finalize(
                cycle=cycle,
                problem_state=cast(data_structures.ProblemState, MagicMock(spec=data_structures.ProblemState)),
                prev_stuck_plan_counters={},
            )
        assert ledgers.worker_ages == {}
