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


"""Bounded multi-donor resource-fit search contract.

``ResourceFitPlanner.find`` is the donor-side feasibility oracle: it
takes a gate-filtered candidate pool and an autoscale planner context,
then asks the planner whether each combination of removals would
unblock the receiver's placement. The first feasible combination wins,
smallest plan_size first, with deterministic tiebreak on
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

from cosmos_xenna.pipelines.private.scheduling_py.donor.resource_fit import ResourceFitPlanner
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorPlan, DonorWorker


def _resource_fit_plan(
    *,
    receiver_stage_index: int,
    candidates: list[DonorWorker],
    worker_nodes: Mapping[str, str],
    ctx: object,
    max_plan_size: int,
    max_plan_combinations: int,
    removable_by_stage: Mapping[int, int],
) -> DonorPlan | None:
    """Construct a ``ResourceFitPlanner`` and dispatch ``find``.

    Tests call this helper once per assertion so each test exercises
    the public class API (``ResourceFitPlanner(...).find(...)``)
    through a single call site; the production module exports
    ``ResourceFitPlanner`` only.

    """
    planner = ResourceFitPlanner(
        max_plan_size=max_plan_size,
        max_plan_combinations=max_plan_combinations,
    )
    return planner.find(
        receiver_stage_index=receiver_stage_index,
        candidates=candidates,
        worker_nodes=worker_nodes,
        ctx=ctx,  # type: ignore[arg-type]
        removable_by_stage=removable_by_stage,
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


def _unbounded_budget(candidates: list[DonorWorker]) -> dict[int, int]:
    """Per-stage budget that allows every candidate to be removed.

    Mirrors a floor=0 configuration so the budget gate is a no-op
    and the test asserts the search behaviour itself rather than the
    budget filter. Tests that exercise the budget filter MUST build
    a tighter mapping by hand.
    """
    counts: dict[int, int] = {}
    for worker in candidates:
        counts[worker.stage_index] = counts.get(worker.stage_index, 0) + 1
    return counts


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
            ctx=ctx,
            max_plan_size=4,
            max_plan_combinations=32,
            removable_by_stage={},
        )

        assert plan is None
        assert ctx.probe_calls == []

    def test_zero_max_plan_size_returns_none(self) -> None:
        """``max_plan_size < 1`` is meaningless and short-circuits before any probe."""
        ctx = _ScriptedCtx(predicate=_all_feasible)
        candidates = _candidates((0, "a-w0", 5))

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=0,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
        )

        assert plan is None
        assert ctx.probe_calls == []

    def test_zero_max_plan_combinations_returns_none(self) -> None:
        """``max_plan_combinations < 1`` blocks every probe -> ``None``."""
        ctx = _ScriptedCtx(predicate=_all_feasible)
        candidates = _candidates((0, "a-w0", 5))

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=4,
            max_plan_combinations=0,
            removable_by_stage=_unbounded_budget(candidates),
        )

        assert plan is None
        assert ctx.probe_calls == []


class TestSingleWorkerPlan:
    """A single-worker plan wins when ``plan_size=1`` is feasible."""

    def test_first_feasible_single_worker_wins_with_deterministic_tiebreak(self) -> None:
        """``(age ASC, worker_id ASC, stage_index ASC)`` resolves the order."""
        ctx = _ScriptedCtx(predicate=_all_feasible)
        candidates = _candidates(
            (0, "a-w0", 10),
            (0, "a-w1", 7),
            (1, "b-w0", 7),
        )

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=4,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
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
        candidates = _candidates((0, "a-w0", 5), (1, "b-w0", 3))

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=4,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
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
        candidates = _candidates(
            (0, "a-w0", 1),
            (0, "a-w1", 1),
            (1, "b-w0", 1),
        )

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes=worker_nodes,
            ctx=ctx,
            max_plan_size=2,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
        )

        assert plan is not None
        same_node_workers = {w.worker_id for w in plan.removals}
        assert same_node_workers == {"a-w0", "b-w0"}, f"Phase 1 must select the same-node pair; got {same_node_workers}"


class TestMaxPlanCombinationsCap:
    """``max_plan_combinations`` bounds probe count per ``plan_size``."""

    def test_combinations_cap_stops_phase_one_iteration(self) -> None:
        """A cap of 1 per plan_size means at most one probe at plan_size=1."""
        ctx = _ScriptedCtx(predicate=_all_infeasible)
        candidates = _candidates(
            (0, "a-w0", 1),
            (0, "a-w1", 2),
            (0, "a-w2", 3),
        )

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=1,
            max_plan_combinations=1,
            removable_by_stage=_unbounded_budget(candidates),
        )

        assert plan is None
        assert len(ctx.probe_calls) == 1, f"max_plan_combinations=1 must cap at 1 probe; got {len(ctx.probe_calls)}"


class TestExhaustionReturnsNone:
    """When every probe is infeasible up to ``max_plan_size``, return ``None``."""

    def test_all_infeasible_exhausts_search_and_returns_none(self) -> None:
        """``max_plan_size=2`` with 2 candidates probes 3 combos: 2 single + 1 pair."""
        ctx = _ScriptedCtx(predicate=_all_infeasible)
        candidates = _candidates((0, "a-w0", 5), (1, "b-w0", 3))

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=2,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
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
        candidates = _candidates(
            (0, "a-w0", 1),
            (0, "a-w1", 2),
        )

        plan = _resource_fit_plan(
            receiver_stage_index=1,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=4,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
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
        candidates = _candidates(
            (0, "a-w0", 1),
            (0, "a-w1", 2),
            (1, "b-w0", 3),
        )

        _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=max_size,
            max_plan_combinations=64,
            removable_by_stage=_unbounded_budget(candidates),
        )

        observed_sizes = {len(removals) for removals, _ in ctx.probe_calls}
        if observed_sizes:
            assert max(observed_sizes) <= max_size


class TestPhaseTwoCrossNodeFallback:
    """Phase 2 (cross-node combinations) takes over when no node alone has enough candidates."""

    def test_cross_node_pair_returned_when_each_node_has_one_candidate(self) -> None:
        """3 candidates on 3 distinct nodes, plan_size=2 -> Phase 1 finds nothing, Phase 2 wins.

        Pins the fallback path: Phase 1 iterates each per-node bucket
        but cannot form a 2-worker plan (only 1 candidate per node), so
        the search advances to Phase 2 and probes cross-node pairs.
        The first feasible cross-node pair is returned.
        """

        def predicate(removals: list[tuple[int, str]], _idx: int) -> bool:
            return len(removals) >= 2

        ctx = _ScriptedCtx(predicate=predicate)
        worker_nodes = {"a-w0": "node-0", "b-w0": "node-1", "c-w0": "node-2"}
        candidates = _candidates((0, "a-w0", 1), (1, "b-w0", 2), (2, "c-w0", 3))

        plan = _resource_fit_plan(
            receiver_stage_index=3,
            candidates=candidates,
            worker_nodes=worker_nodes,
            ctx=ctx,
            max_plan_size=4,
            max_plan_combinations=32,
            removable_by_stage=_unbounded_budget(candidates),
        )

        assert plan is not None
        assert len(plan.removals) == 2
        # Phase 2 enumeration order over the sorted candidate list yields
        # the lexicographically smallest 2-worker pair: (a-w0, b-w0).
        assert {w.worker_id for w in plan.removals} == {"a-w0", "b-w0"}


class TestDonorPlanInvariant:
    """``DonorPlan`` rejects empty ``removals`` at construction."""

    def test_empty_removals_raises_value_error(self) -> None:
        """An empty plan is meaningless -- ``None`` is the canonical 'no donation' sentinel."""
        with pytest.raises(ValueError, match=r"'removals' must be >= 1"):
            DonorPlan(removals=(), receiver_stage_index=0)


class TestPerStageRemovableBudget:
    """The per-stage budget filter blocks combos that violate a donor floor."""

    def test_two_donor_combo_from_one_stage_skipped_when_budget_is_one(self) -> None:
        """Stage 0 has 2 candidates but budget=1 -> the (0,0) pair is never probed.

        The pair-from-one-stage probe is the bug Finding 2 closes:
        without the budget filter the resource-fit search could
        return a 2-worker plan drawing both workers from a single
        donor stage, dropping it below its own floor. With
        ``removable_by_stage[0]=1`` the pair is filtered before
        the probe round-trip, and the predicate is only invoked
        on cross-stage combos that respect every donor's budget.
        """

        def predicate(removals: list[tuple[int, str]], _idx: int) -> bool:
            return len(removals) >= 2

        ctx = _ScriptedCtx(predicate=predicate)
        candidates = _candidates(
            (0, "a-w0", 1),
            (0, "a-w1", 2),
            (1, "b-w0", 3),
        )

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=2,
            max_plan_combinations=32,
            removable_by_stage={0: 1, 1: 1},
        )

        assert plan is not None
        # Only cross-stage pairs survive the budget filter; the (0,0)
        # combo (both workers from stage 0) was skipped because it
        # would consume the entire stage-0 budget at once.
        stage_indices = sorted(w.stage_index for w in plan.removals)
        assert stage_indices == [0, 1], f"plan must respect the per-stage budget; got stage_indices={stage_indices}"
        for removals_arg, _ in ctx.probe_calls:
            stages_in_combo = sorted(s for s, _w in removals_arg)
            assert stages_in_combo != [0, 0], (
                f"a (0,0) combo should never be probed under budget {{0: 1}}; got {removals_arg}"
            )

    def test_missing_stage_in_budget_defaults_to_zero(self) -> None:
        """A stage absent from ``removable_by_stage`` is treated as having no removable workers.

        Pins the documented default: missing entries -> 0. A combo
        containing any worker from that stage is skipped before the
        probe round-trip.
        """
        ctx = _ScriptedCtx(predicate=_all_feasible)
        candidates = _candidates((0, "a-w0", 1), (1, "b-w0", 2))

        plan = _resource_fit_plan(
            receiver_stage_index=2,
            candidates=candidates,
            worker_nodes={},
            ctx=ctx,
            max_plan_size=2,
            max_plan_combinations=32,
            removable_by_stage={0: 1},
        )

        assert plan is not None
        # Stage 1 is not in the budget map -> default 0 -> b-w0 is
        # never selected. The single-worker (a-w0,) plan from stage 0
        # wins because it is the only combo whose stage distribution
        # respects the (implicit) budget.
        assert plan == DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="a-w0", age=1),),
            receiver_stage_index=2,
        )
