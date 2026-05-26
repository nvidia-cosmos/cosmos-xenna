# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded multi-donor resource-fit search contract.

``_resource_fit_plan`` is the donor-side feasibility oracle: it takes
a gate-filtered candidate pool and an autoscale planner context, then
asks the planner whether each combination of removals would unblock
the receiver's placement. The first feasible combination wins, smallest
plan_size first, with deterministic tiebreak on
``(age ASC, worker_id ASC, stage_index ASC)``.

Tests pin each contract in isolation:

* Empty / non-positive bounds -> ``None``.
* Single-worker plan -> probe called once; deterministic tiebreak.
* Two-worker plan when no single worker fits -> first feasible pair.
* Same-node iteration precedes cross-node iteration when locality
  info is supplied.
* ``max_plan_combinations`` caps probe calls per plan_size.
* Returns ``None`` when every probe is infeasible up to ``max_plan_size``.
"""

from collections.abc import Mapping

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.donor import (
    DonorPlan,
    DonorWorker,
    _resource_fit_plan,
)


@attrs.frozen
class _ProbeResult:
    """Stand-in for ``rust_apc.PlacementProbeResult``."""

    feasible: bool
    reject_reason: str = ""


@attrs.define
class _ScriptedCtx:
    """Fake autoscale planner context with a scripted feasibility predicate.

    The predicate takes the removals list and the receiver index and
    returns a boolean. Each ``probe_add_after_removals`` call is
    appended to ``probe_calls`` in invocation order so tests can assert
    on the search trajectory.
    """

    predicate: object
    probe_calls: list[tuple[list[tuple[int, str]], int]] = attrs.field(factory=list)

    def probe_add_after_removals(
        self,
        removals: list[tuple[int, str]],
        add_stage_index: int,
    ) -> _ProbeResult:
        self.probe_calls.append((list(removals), add_stage_index))
        ok = self.predicate(removals, add_stage_index)  # type: ignore[operator]
        return _ProbeResult(feasible=bool(ok))


def _candidates(*specs: tuple[int, str, int]) -> list[DonorWorker]:
    """Build a candidate list from ``(stage_index, worker_id, age)`` tuples."""
    return [DonorWorker(stage_index=s, worker_id=w, age=a) for s, w, a in specs]


def _all_feasible(_removals: list[tuple[int, str]], _idx: int) -> bool:
    return True


def _all_infeasible(_removals: list[tuple[int, str]], _idx: int) -> bool:
    return False


class TestEmptyOrInvalidInputs:
    """Empty candidate pools or non-positive bounds short-circuit to ``None``."""

    def test_empty_candidate_list_returns_none(self) -> None:
        """Nothing to plan -> ``None`` without a single probe call."""
        ctx = _ScriptedCtx(predicate=_all_feasible)

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=[],
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=4,
            max_plan_combinations=32,
        )

        assert plan is None
        assert ctx.probe_calls == []

    def test_zero_max_plan_size_returns_none(self) -> None:
        """``max_plan_size < 1`` is meaningless and short-circuits before any probe."""
        ctx = _ScriptedCtx(predicate=_all_feasible)

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=_candidates((0, "a-w0", 5)),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=0,
            max_plan_combinations=32,
        )

        assert plan is None
        assert ctx.probe_calls == []

    def test_zero_max_plan_combinations_returns_none(self) -> None:
        """``max_plan_combinations < 1`` blocks every probe -> ``None``."""
        ctx = _ScriptedCtx(predicate=_all_feasible)

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=_candidates((0, "a-w0", 5)),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=4,
            max_plan_combinations=0,
        )

        assert plan is None
        assert ctx.probe_calls == []


class TestSingleWorkerPlan:
    """A single-worker plan wins when ``plan_size=1`` is feasible."""

    def test_first_feasible_single_worker_wins_with_deterministic_tiebreak(self) -> None:
        """``(age ASC, worker_id ASC, stage_index ASC)`` resolves the order."""
        ctx = _ScriptedCtx(predicate=_all_feasible)

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=_candidates(
                (0, "a-w0", 10),
                (0, "a-w1", 7),
                (1, "b-w0", 7),
            ),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=4,
            max_plan_combinations=32,
        )

        # Sorted by (age, worker_id, stage_index): a-w1 (7,a-w1,0) wins
        # over b-w0 (7,b-w0,1) on worker_id; both tie a-w0 (10) on age.
        assert plan == DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w1", age=7),),
            receiver_stage_index=2,
        )
        assert len(ctx.probe_calls) == 1


class TestMultiWorkerPlan:
    """When no single worker fits, the search advances to ``plan_size=2``."""

    def test_two_worker_plan_returned_when_single_worker_infeasible(self) -> None:
        """A predicate that requires both donors models a whole-GPU receiver shape."""

        def predicate(removals: list[tuple[int, str]], _idx: int) -> bool:
            return len(removals) >= 2

        ctx = _ScriptedCtx(predicate=predicate)

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=_candidates((0, "a-w0", 5), (1, "b-w0", 3)),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=4,
            max_plan_combinations=32,
        )

        assert plan is not None
        assert len(plan.removals) == 2
        assert {w.worker_id for w in plan.removals} == {"a-w0", "b-w0"}
        # plan_size=1 ran twice (one per candidate) before plan_size=2.
        assert len(ctx.probe_calls) == 3


class TestLocalityIteration:
    """Same-node combinations are probed before cross-node combinations."""

    def test_same_node_combinations_probed_first(self) -> None:
        """Two donors on ``node-0`` are tried before any cross-node pair."""

        def predicate(removals: list[tuple[int, str]], _idx: int) -> bool:
            return len(removals) >= 2

        ctx = _ScriptedCtx(predicate=predicate)

        worker_nodes: Mapping[str, str] = {
            "a-w0": "node-0",
            "a-w1": "node-1",
            "b-w0": "node-0",
        }

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=_candidates(
                (0, "a-w0", 1),
                (0, "a-w1", 1),
                (1, "b-w0", 1),
            ),
            worker_nodes=worker_nodes,
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=2,
            max_plan_combinations=32,
        )

        assert plan is not None
        same_node_workers = {w.worker_id for w in plan.removals}
        assert same_node_workers == {"a-w0", "b-w0"}, f"Phase 1 must select the same-node pair; got {same_node_workers}"


class TestMaxPlanCombinationsCap:
    """``max_plan_combinations`` bounds probe count per ``plan_size``."""

    def test_combinations_cap_stops_phase_one_iteration(self) -> None:
        """A cap of 1 per plan_size means at most one probe at plan_size=1."""
        ctx = _ScriptedCtx(predicate=_all_infeasible)

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=_candidates(
                (0, "a-w0", 1),
                (0, "a-w1", 2),
                (0, "a-w2", 3),
            ),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=1,
            max_plan_combinations=1,
        )

        assert plan is None
        assert len(ctx.probe_calls) == 1, f"max_plan_combinations=1 must cap at 1 probe; got {len(ctx.probe_calls)}"


class TestExhaustionReturnsNone:
    """When every probe is infeasible up to ``max_plan_size``, return ``None``."""

    def test_all_infeasible_exhausts_search_and_returns_none(self) -> None:
        """``max_plan_size=2`` with 2 candidates probes 3 combos: 2 single + 1 pair."""
        ctx = _ScriptedCtx(predicate=_all_infeasible)

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=_candidates((0, "a-w0", 5), (1, "b-w0", 3)),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=2,
            max_plan_combinations=32,
        )

        assert plan is None
        # 2 single-worker probes + 1 pair probe.
        assert len(ctx.probe_calls) == 3


class TestSmallestSizeWinsAcrossSizes:
    """A feasible single-worker plan beats any feasible larger plan."""

    def test_size_one_winner_short_circuits_size_two(self) -> None:
        """The first plan_size with a feasible probe ends the search."""

        def predicate(removals: list[tuple[int, str]], _idx: int) -> bool:
            return removals == [(0, "a-w0")]

        ctx = _ScriptedCtx(predicate=predicate)

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=_candidates(
                (0, "a-w0", 1),
                (0, "a-w1", 2),
            ),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=4,
            max_plan_combinations=32,
        )

        assert plan == DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=1,
        )
        # Only one probe -- the search exited at plan_size=1.
        assert len(ctx.probe_calls) == 1


@pytest.mark.parametrize("max_size", [1, 2, 3, 4, 5])
class TestParametricBoundedness:
    """For any ``max_plan_size``, the search never exceeds the cap."""

    def test_search_never_runs_combinations_larger_than_cap(self, max_size: int) -> None:
        """Verify no probe call has ``len(removals) > max_plan_size``."""
        ctx = _ScriptedCtx(predicate=_all_infeasible)

        _resource_fit_plan(
            receiver_stage_index=2,
            candidates=_candidates(
                (0, "a-w0", 1),
                (0, "a-w1", 2),
                (1, "b-w0", 3),
            ),
            worker_nodes={},
            ctx=ctx,  # type: ignore[arg-type]
            max_plan_size=max_size,
            max_plan_combinations=64,
        )

        observed_sizes = {len(removals) for removals, _ in ctx.probe_calls}
        if observed_sizes:
            assert max(observed_sizes) <= max_size
