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

"""Behaviour tests for the manual-stage shrink path.

Pin the contract:

  1. A manual stage (``requested_num_workers is not None``) with
     ``requested < current`` shrinks to exactly ``requested``.
  2. Among the surplus workers, the youngest-first sort selects
     victims; ties are broken by ``worker_id`` ascending so the
     decision is deterministic with uniform ages.
  3. Finished stages, non-manual stages, and stages already at or
     below the request are not touched.

The pure sort helper is tested here directly so regressions surface
on a focused failure rather than through scheduler-integration noise.
"""

from typing import cast

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import (
    SaturationAwareScheduler,
    _select_workers_to_delete_youngest_first,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def _cluster() -> resources.ClusterResources:
    """Single-node cluster with enough headroom for every test fixture."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=16, gpus=[], name="node-0"),
        },
    )


def _problem_with_requested(stage_specs: list[tuple[str, int | None]]) -> data_structures.Problem:
    """Build a ``Problem`` whose stages carry the given ``requested_num_workers``."""
    cluster = _cluster()
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


def _make_workers(stage_name: str, count: int) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers on ``node-0`` named ``<stage>-w<i>``."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(name, num_workers, slots_per_worker, is_finished)``."""
    states = []
    for name, num_workers, slots, finished in stage_specs:
        states.append(
            data_structures.ProblemStageState(
                stage_name=name,
                workers=_make_workers(name, num_workers),
                slots_per_worker=slots,
                is_finished=finished,
            )
        )
    return data_structures.ProblemState(states)


class _RejectingRemoveContext:
    """AutoscalePlanContext-shaped fake that rejects every staged removal."""

    def __init__(self) -> None:
        """Create an empty call recorder."""
        self.calls: list[tuple[int, str]] = []

    def worker_ages(self) -> dict[str, int]:
        """Return no age history so victim selection falls back to worker id."""
        return {}

    def try_remove_worker(self, stage_index: int, worker_id: str) -> bool:
        """Record the attempted removal and reject it."""
        self.calls.append((stage_index, worker_id))
        return False


class TestSelectWorkersToDeleteYoungestFirst:
    """Pure sort helper: ``(age ASC, worker_id ASC)``."""

    def test_uniform_ages_returns_smallest_worker_ids(self) -> None:
        """Default age 0 for every worker -> deterministic worker_id ascending."""
        victims = _select_workers_to_delete_youngest_first(
            worker_ids=["A-w2", "A-w0", "A-w1"],
            worker_ages={},
            delete_count=2,
        )
        assert victims == ["A-w0", "A-w1"]

    def test_youngest_age_wins_over_lex_smaller_worker_id(self) -> None:
        """A-w0 has age 5, A-w1 has age 0 -> A-w1 selected first."""
        victims = _select_workers_to_delete_youngest_first(
            worker_ids=["A-w0", "A-w1", "A-w2"],
            worker_ages={"A-w0": 5, "A-w1": 0, "A-w2": 3},
            delete_count=1,
        )
        assert victims == ["A-w1"]

    def test_age_tie_falls_back_to_worker_id_order(self) -> None:
        """Two workers with age 1 and one with age 5 -> the tied workers selected first."""
        victims = _select_workers_to_delete_youngest_first(
            worker_ids=["A-w0", "A-w1", "A-w2"],
            worker_ages={"A-w0": 1, "A-w1": 5, "A-w2": 1},
            delete_count=2,
        )
        assert victims == ["A-w0", "A-w2"]

    def test_delete_count_zero_returns_empty(self) -> None:
        """A request to delete zero workers returns no victims."""
        victims = _select_workers_to_delete_youngest_first(
            worker_ids=["A-w0"],
            worker_ages={"A-w0": 0},
            delete_count=0,
        )
        assert victims == []

    def test_delete_count_above_population_clamps_to_population(self) -> None:
        """A request to delete more workers than exist returns every worker."""
        victims = _select_workers_to_delete_youngest_first(
            worker_ids=["A-w0", "A-w1"],
            worker_ages={},
            delete_count=10,
        )
        assert sorted(victims) == ["A-w0", "A-w1"]

    def test_missing_worker_age_treated_as_zero(self) -> None:
        """A worker with no entry in the ages map is treated as freshly observed."""
        victims = _select_workers_to_delete_youngest_first(
            worker_ids=["A-w0", "A-w1"],
            worker_ages={"A-w0": 7},  # A-w1 missing -> treated as age 0
            delete_count=1,
        )
        assert victims == ["A-w1"]


class TestManualShrinkScheduler:
    """End-to-end manual shrink via ``SaturationAwareScheduler.autoscale``."""

    def test_manual_shrink_emits_delete_workers_for_excess(self) -> None:
        """4-worker manual stage with ``requested=2`` shrinks to 2."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 2)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False)]),
        )
        deleted = {w.id for w in solution.stages[0].deleted_workers}
        assert len(deleted) == 2
        assert solution.stages[0].new_workers == []

    def test_uniform_ages_delete_lex_smallest_worker_ids_first(self) -> None:
        """With uniform ages the worker_id tiebreaker selects the lex-smallest ids."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 1)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False)]),
        )
        deleted_ids = {w.id for w in solution.stages[0].deleted_workers}
        # Workers are A-w0, A-w1, A-w2, A-w3; uniform age 0; delete 3 youngest
        # by worker_id ASC: A-w0, A-w1, A-w2 -> A-w3 survives.
        assert deleted_ids == {"A-w0", "A-w1", "A-w2"}

    def test_finished_stages_are_skipped(self) -> None:
        """A stage flagged finished is not shrunk regardless of request mismatch."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 1)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, True)]),
        )
        assert solution.stages[0].deleted_workers == []

    def test_non_manual_stages_are_skipped(self) -> None:
        """``requested_num_workers=None`` makes the stage non-manual; the path skips it."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False)]),
        )
        assert solution.stages[0].deleted_workers == []

    def test_no_op_when_current_equals_requested(self) -> None:
        """``current == requested`` produces no delta on a manual stage."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 3)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False)]),
        )
        assert solution.stages[0].deleted_workers == []
        assert solution.stages[0].new_workers == []

    def test_no_op_when_current_below_requested(self) -> None:
        """``current < requested`` is a manual-grow case; the shrink path is a no-op."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 4)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 1, 1, False)]),
        )
        assert solution.stages[0].deleted_workers == []
        assert solution.stages[0].new_workers == []

    def test_requested_zero_removes_every_worker(self) -> None:
        """``requested=0`` shrinks the stage to empty."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 0)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False)]),
        )
        deleted_ids = {w.id for w in solution.stages[0].deleted_workers}
        assert deleted_ids == {"A-w0", "A-w1", "A-w2"}

    def test_only_manual_stage_shrinks_in_mixed_pipeline(self) -> None:
        """A pipeline with one manual + one auto stage shrinks only the manual one."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("manual", 1), ("auto", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state(
                [("manual", 3, 1, False), ("auto", 3, 1, False)],
            ),
        )
        manual_deleted = {w.id for w in solution.stages[0].deleted_workers}
        auto_deleted = {w.id for w in solution.stages[1].deleted_workers}
        assert len(manual_deleted) == 2
        assert auto_deleted == set()

    def test_planner_rejecting_selected_worker_raises_runtime_error(self) -> None:
        """A selected runtime worker missing from the planner snapshot raises."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_requested([("A", 1)]))
        ctx = _RejectingRemoveContext()
        problem_state = _problem_state([("A", 2, 1, False)])

        with pytest.raises(RuntimeError, match="snapshot inconsistency"):
            scheduler._run_phase_a_delete(
                cast(data_structures.AutoscalePlanContext, ctx),
                problem_state,
            )

        assert ctx.calls == [(0, "A-w0")]
