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

"""Behaviour tests for the phase-boundary invariant gates.

Pin the contract:

  1. ``check_invariants_after_phase`` raises
     ``SchedulerInvariantError`` when the planner-context shape
     disagrees with the problem (added or dropped stages mid-cycle)
     or reports negative pending counts.
  2. ``check_solution_shape`` raises when the emitted ``Solution``'s
     stage count does not match the problem's stage count.
  3. The scheduler's ``autoscale`` method invokes the gates between
     phases; a valid plan does not raise.
"""

from typing import cast
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.invariants import (
    PhaseBoundary,
    check_invariants_after_phase,
    check_solution_shape,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, total_cpus: int = 16) -> resources.ClusterResources:
    """Single-node cluster sized for the fixtures."""
    return resources.ClusterResources(
        nodes={"node-0": resources.NodeResources(used_cpus=0, total_cpus=total_cpus, gpus=[], name="node-0")},
    )


def _problem(stage_specs: list[tuple[str, int | None]], *, total_cpus: int = 16) -> data_structures.Problem:
    """Build a problem with the given stage specs (``(name, requested_num_workers)``)."""
    cluster = _cluster(total_cpus=total_cpus)
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=requested,
            over_provision_factor=None,
        )
        for name, requested in stage_specs
    ]
    return data_structures.Problem(cluster, stages)


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(name, num_workers, slots_per_worker, is_finished)``."""
    states = [
        data_structures.ProblemStageState(
            stage_name=name,
            workers=[
                data_structures.ProblemWorkerGroupState.make(
                    f"{name}-w{i}",
                    [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                )
                for i in range(num_workers)
            ],
            slots_per_worker=slots,
            is_finished=finished,
        )
        for name, num_workers, slots, finished in stage_specs
    ]
    return data_structures.ProblemState(states)


class _ShapeMismatchContext:
    """Mock planner whose ``num_stages()`` disagrees with the problem."""

    def __init__(self, returned_stages: int) -> None:
        self._returned = returned_stages

    def num_stages(self) -> int:
        return self._returned

    def pending_add_count(self, stage_index: int) -> int:
        del stage_index
        return 0

    def pending_remove_count(self, stage_index: int) -> int:
        del stage_index
        return 0


class _NegativePendingContext:
    """Mock planner that exposes a negative pending count for one or more stages."""

    def __init__(
        self,
        num_stages: int,
        *,
        negative_add_for: int | None = None,
        negative_remove_for: int | None = None,
    ) -> None:
        self._num_stages = num_stages
        self._negative_add_for = negative_add_for
        self._negative_remove_for = negative_remove_for

    def num_stages(self) -> int:
        return self._num_stages

    def pending_add_count(self, stage_index: int) -> int:
        return -1 if stage_index == self._negative_add_for else 0

    def pending_remove_count(self, stage_index: int) -> int:
        return -1 if stage_index == self._negative_remove_for else 0


class _MalformedSolutionRust:
    """Inner Rust-side mock exposing only the shape we need."""

    def __init__(self, num_stages: int) -> None:
        self.stages = [object() for _ in range(num_stages)]


class _MalformedSolution:
    """Mock solution whose Rust-side ``stages`` length does not match the problem."""

    def __init__(self, num_stages: int) -> None:
        self.rust = _MalformedSolutionRust(num_stages)


class TestCheckInvariantsAfterPhase:
    """Pure-helper checks for the per-phase planner-context invariants."""

    def test_valid_state_does_not_raise(self) -> None:
        """A real planner context built from a valid problem-state passes the check."""
        problem = _problem([("A", None), ("B", None)])
        problem_state = _problem_state([("A", 1, 1, False), ("B", 0, 1, False)])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, problem_state)
        check_invariants_after_phase(phase_name=PhaseBoundary.PHASE_A, problem=problem, ctx=ctx)

    def test_zero_stage_problem_does_not_raise(self) -> None:
        """A zero-stage problem with an empty ctx is a valid no-op boundary."""
        problem = _problem([])
        ctx = _ShapeMismatchContext(returned_stages=0)
        check_invariants_after_phase(
            phase_name=PhaseBoundary.PHASE_A,
            problem=problem,
            ctx=cast(data_structures.AutoscalePlanContext, ctx),
        )

    def test_shape_mismatch_raises_with_phase_name(self) -> None:
        """A ctx that reports more stages than the problem raises with operator context."""
        problem = _problem([("A", None), ("B", None)])
        ctx = _ShapeMismatchContext(returned_stages=3)
        with pytest.raises(SchedulerInvariantError, match=r"After phase_a:.*reports 3 stages.*has 2"):
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_A,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )

    def test_shape_mismatch_fewer_stages_raises(self) -> None:
        """A ctx that reports fewer stages than the problem also raises."""
        problem = _problem([("A", None), ("B", None), ("C", None)])
        ctx = _ShapeMismatchContext(returned_stages=1)
        with pytest.raises(SchedulerInvariantError, match=r"reports 1 stages.*has 3"):
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_B,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )

    def test_shape_mismatch_caught_before_pending_count_loop(self) -> None:
        """A stage-count mismatch raises BEFORE the per-stage counter loop runs.

        Pins that the gate cannot leak ``IndexError`` from
        ``ctx.pending_add_count(idx)`` when the planner reports fewer
        stages than the problem -- the shape check must fire first so the
        cheaper ``ctx.num_stages()`` discrepancy surfaces as a clean
        ``SchedulerInvariantError``.
        """

        class _ShortCtxThatRaisesInPendingCounts:
            def num_stages(self) -> int:
                return 1  # Disagrees with the 3-stage problem.

            def pending_add_count(self, stage_index: int) -> int:
                raise IndexError(f"out of range: {stage_index}")

            def pending_remove_count(self, stage_index: int) -> int:
                raise IndexError(f"out of range: {stage_index}")

        problem = _problem([("A", None), ("B", None), ("C", None)])
        ctx = _ShortCtxThatRaisesInPendingCounts()
        with pytest.raises(SchedulerInvariantError, match=r"reports 1 stages.*has 3"):
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_A,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )

    def test_negative_pending_add_count_raises_with_stage_name(self) -> None:
        """A negative pending-add count surfaces the stage name and the violating value."""
        problem = _problem([("A", None), ("B", None), ("C", None)])
        ctx = _NegativePendingContext(num_stages=3, negative_add_for=1)
        with pytest.raises(SchedulerInvariantError, match=r"stage 'B'.*pending_add_count=-1"):
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_A,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )

    def test_negative_pending_remove_count_raises_with_stage_name(self) -> None:
        """A negative pending-remove count surfaces the stage name and the violating value."""
        problem = _problem([("A", None), ("B", None)])
        ctx = _NegativePendingContext(num_stages=2, negative_remove_for=0)
        with pytest.raises(SchedulerInvariantError, match=r"stage 'A'.*pending_remove_count=-1"):
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_B,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )

    def test_both_pending_counts_negative_raises_on_first_stage(self) -> None:
        """When both counters are negative, the first stage to fail surfaces both values."""
        problem = _problem([("A", None), ("B", None)])
        ctx = _NegativePendingContext(num_stages=2, negative_add_for=0, negative_remove_for=0)
        with pytest.raises(
            SchedulerInvariantError,
            match=r"stage 'A'.*pending_add_count=-1.*pending_remove_count=-1",
        ):
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_A,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )

    def test_invariant_violation_logs_at_error_level_before_raising(self) -> None:
        """Each violation is logged at ERROR with the prefix ``scheduler invariant violation:``.

        Pins the observability contract: even if a higher-level supervisor
        catches and converts the exception, the diagnostic still reaches
        the structured log.
        """
        problem = _problem([("A", None), ("B", None)])
        ctx = _NegativePendingContext(num_stages=2, negative_add_for=0)
        with patch("cosmos_xenna.pipelines.private.scheduling_py.invariants.logger.error") as error:
            with pytest.raises(SchedulerInvariantError):
                check_invariants_after_phase(
                    phase_name=PhaseBoundary.PHASE_A,
                    problem=problem,
                    ctx=cast(data_structures.AutoscalePlanContext, ctx),
                )
        assert error.call_count == 1
        log_msg = error.call_args.args[0]
        assert log_msg.startswith("scheduler invariant violation:")
        assert "stage 'A'" in log_msg
        assert "pending_add_count=-1" in log_msg

    def test_stage_name_with_newline_is_repr_escaped_to_prevent_log_injection(self) -> None:
        """Stage names containing newlines are ``!r``-escaped, so the embedded text cannot forge a log line."""
        problem = _problem([("stage\nFAKE_LOG_LINE", None)])
        ctx = _NegativePendingContext(num_stages=1, negative_add_for=0)
        with pytest.raises(SchedulerInvariantError) as exc_info:
            check_invariants_after_phase(
                phase_name=PhaseBoundary.PHASE_A,
                problem=problem,
                ctx=cast(data_structures.AutoscalePlanContext, ctx),
            )
        msg = str(exc_info.value)
        # Critical: the embedded ``\nFAKE_LOG_LINE`` must NOT appear as a bare line
        # in the message; ``!r`` formatting escapes the newline so the injected
        # text appears literally as ``\\n`` instead of producing a fake log entry.
        assert "\nFAKE_LOG_LINE" not in msg
        assert "\\nFAKE_LOG_LINE" in msg


class TestCheckSolutionShape:
    """Pure-helper check for the Solution-shape invariant."""

    def test_valid_solution_does_not_raise(self) -> None:
        """A real solution emitted from a valid context passes the check."""
        problem = _problem([("A", None), ("B", None)])
        problem_state = _problem_state([("A", 1, 1, False), ("B", 0, 1, False)])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, problem_state)
        solution = ctx.into_solution()
        check_solution_shape(phase_name=PhaseBoundary.INTO_SOLUTION, problem=problem, solution=solution)

    def test_solution_with_extra_stage_raises(self) -> None:
        """A solution with more stages than the problem raises with operator context."""
        problem = _problem([("A", None), ("B", None)])
        solution = _MalformedSolution(num_stages=3)
        with pytest.raises(SchedulerInvariantError, match=r"Solution has 3 stages.*problem has 2"):
            check_solution_shape(
                phase_name=PhaseBoundary.INTO_SOLUTION,
                problem=problem,
                solution=cast(data_structures.Solution, solution),
            )

    def test_solution_with_missing_stage_raises(self) -> None:
        """A solution with fewer stages than the problem also raises."""
        problem = _problem([("A", None), ("B", None), ("C", None)])
        solution = _MalformedSolution(num_stages=1)
        with pytest.raises(SchedulerInvariantError, match=r"Solution has 1 stages.*problem has 3"):
            check_solution_shape(
                phase_name=PhaseBoundary.INTO_SOLUTION,
                problem=problem,
                solution=cast(data_structures.Solution, solution),
            )


class TestSchedulerWiringDoesNotRaise:
    """End-to-end: the scheduler's autoscale path passes invariant checks on a valid plan."""

    def test_valid_plan_completes_without_raising(self) -> None:
        """A correctly configured scheduler runs the full autoscale path without invariant violations."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)]))
        # A valid problem state with capacity to satisfy every floor.
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 1, 1, False), ("B", 1, 1, False)]),
        )
        assert len(solution.stages) == 2

    def test_scheduler_invokes_invariants_at_each_phase_boundary(self) -> None:
        """The scheduler's autoscale path calls the invariant gate after each phase.

        Verifies the wiring -- the gate is invoked between Phase A, Phase B,
        and after ``into_solution()`` -- by patching the helpers and asserting
        their call counts. A future refactor that drops one of the calls will
        break this test, surfacing the regression at review time rather than
        in production.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        with (
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_invariants_after_phase"
            ) as phase_check,
            patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_solution_shape") as shape_check,
        ):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 1, 1, False)]),
            )
        # Two phase boundaries (after Phase A, after Phase B).
        assert phase_check.call_count == 2
        # One Solution-shape check (after into_solution).
        assert shape_check.call_count == 1
        # The phase boundaries are tagged so operators can locate the violating phase.
        phase_names = {call.kwargs["phase_name"] for call in phase_check.call_args_list}
        assert phase_names == {PhaseBoundary.PHASE_A, PhaseBoundary.PHASE_B}

    def test_phase_a_invariant_failure_stops_before_phase_b(self) -> None:
        """A Phase A invariant failure propagates and prevents later plan mutation."""
        scheduler = SaturationAwareScheduler(
            SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1)),
        )
        scheduler.setup(_problem([("A", None)]))
        error = SchedulerInvariantError("phase-a corrupted")

        with (
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_invariants_after_phase",
                side_effect=error,
            ) as phase_check,
            patch.object(scheduler, "_run_phase_b_floor") as phase_b_floor,
            patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_solution_shape") as shape_check,
        ):
            with pytest.raises(SchedulerInvariantError, match="phase-a corrupted"):
                scheduler.autoscale(
                    time=0.0,
                    problem_state=_problem_state([("A", 1, 1, False)]),
                )

        assert phase_check.call_count == 1
        assert phase_check.call_args.kwargs["phase_name"] is PhaseBoundary.PHASE_A
        phase_b_floor.assert_not_called()
        shape_check.assert_not_called()

    def test_phase_b_invariant_failure_stops_before_solution_shape_check(self) -> None:
        """A Phase B invariant failure propagates before ``into_solution`` is trusted."""
        scheduler = SaturationAwareScheduler(
            SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1)),
        )
        scheduler.setup(_problem([("A", None)]))

        def _fail_on_phase_b(**kwargs: object) -> None:
            if kwargs["phase_name"] is PhaseBoundary.PHASE_B:
                raise SchedulerInvariantError("phase-b corrupted")

        with (
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_invariants_after_phase",
                side_effect=_fail_on_phase_b,
            ) as phase_check,
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.data_structures."
                "AutoscalePlanContext.into_solution"
            ) as into_solution,
            patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_solution_shape") as shape_check,
        ):
            with pytest.raises(SchedulerInvariantError, match="phase-b corrupted"):
                scheduler.autoscale(
                    time=0.0,
                    problem_state=_problem_state([("A", 1, 1, False)]),
                )

        phase_names = [call.kwargs["phase_name"] for call in phase_check.call_args_list]
        assert phase_names == [PhaseBoundary.PHASE_A, PhaseBoundary.PHASE_B]
        into_solution.assert_not_called()
        shape_check.assert_not_called()

    def test_solution_shape_failure_stops_before_age_persistence(self) -> None:
        """A Solution-shape invariant failure prevents post-cycle state persistence."""
        scheduler = SaturationAwareScheduler(
            SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1)),
        )
        scheduler.setup(_problem([("A", None)]))

        with (
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_solution_shape",
                side_effect=SchedulerInvariantError("solution shape corrupted"),
            ) as shape_check,
            patch.object(scheduler, "_persist_worker_ages") as persist_worker_ages,
        ):
            with pytest.raises(SchedulerInvariantError, match="solution shape corrupted"):
                scheduler.autoscale(
                    time=0.0,
                    problem_state=_problem_state([("A", 1, 1, False)]),
                )

        assert shape_check.call_count == 1
        persist_worker_ages.assert_not_called()
