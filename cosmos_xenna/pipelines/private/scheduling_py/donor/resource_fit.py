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

"""Bounded multi-donor placement search for the cross-stage donor flow.

``ResourceFitPlanner`` enumerates same-node-first then cross-node
worker combinations against a candidate pool, probing each combo
through ``ctx.probe_add_after_removals`` and returning the first
feasible ``DonorPlan``. The search is bounded on three axes so a
pathological cluster cannot stall the scheduler:

* ``max_plan_size`` - combo width (donors per plan).
* ``max_plan_combinations`` - per-plan-size *probe* budget (the
  expensive ``probe_add_after_removals`` FFI calls).
* ``max_plan_combinations * _ITERATION_BUDGET_MULTIPLIER`` - the
  per-``find`` *enumeration* budget. The probe budget only counts
  combos that clear the cheap stage-budget pre-filter, so it cannot
  bound the raw ``itertools.combinations`` walk when most combos
  fail that filter; this third cap does.

Per-stage donor budgets prevent any single combo from dropping a
donor stage below its own floor.
"""

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorPlan, DonorWorker

if TYPE_CHECKING:
    from cosmos_xenna.pipelines.private.data_structures import AutoscalePlanContext


# Multiplier converting the per-plan-size *probe* budget
# (``max_plan_combinations``) into a per-``find`` *enumeration*
# budget. ``max_plan_combinations`` only counts combos that clear the
# cheap stage-budget pre-filter (``evaluations`` increments *after*
# the filter), so a pool whose combos mostly violate the budget would
# otherwise walk the full ``C(n, plan_size)`` space across both passes
# without the probe cap ever tripping. Bounding total combinations
# examined at ``max_plan_combinations * _ITERATION_BUDGET_MULTIPLIER``
# keeps the enumeration finite regardless of the pre-filter hit rate.
# 64 leaves a wide margin over a healthy search - which examines at
# most a few multiples of ``max_plan_combinations`` across all plan
# sizes - so the cap only fires on pathological pools, never on
# legitimate ones.
_ITERATION_BUDGET_MULTIPLIER = 64


@attrs.frozen
class ResourceFitPlanner:
    """Bounded multi-donor combination search for a feasible donor plan.

    Stateless ``@attrs.frozen`` behaviour bundle. One instance per
    donor coordinator; ``find`` is the only entry point. Performs
    a same-node-first then cross-node combination walk so co-located
    donors are preferred when locality information is available.

    """

    max_plan_size: int
    max_plan_combinations: int

    def find(
        self,
        *,
        receiver_stage_index: int,
        candidates: list[DonorWorker],
        worker_nodes: Mapping[str, str],
        ctx: "AutoscalePlanContext",
        removable_by_stage: Mapping[int, int],
    ) -> DonorPlan | None:
        """Return the smallest feasible donor plan or ``None``.

        Iterates ``plan_size`` from 1 through ``max_plan_size``;
        for each size enumerates same-node-first then cross-node
        combinations, capped at ``max_plan_combinations`` probes.
        Per-stage budgets in ``removable_by_stage`` prevent any
        combo from pulling a stage below its own floor. Tiebreak:
        ``(age ASC, worker_id ASC, stage_index ASC)``.

        Returns ``None`` early if total combinations examined exceed
        ``max_plan_combinations * _ITERATION_BUDGET_MULTIPLIER`` - a
        safety bound on the raw enumeration that the probe budget
        cannot provide (it only counts combos clearing the cheap
        stage-budget pre-filter). A pool needing more than that many
        combinations to surface a feasible plan is itself pathological;
        abandoning the search is preferable to stalling the cycle.

        """
        if not candidates:
            return None
        if self.max_plan_size < 1 or self.max_plan_combinations < 1:
            return None

        # Pre-sort to make ``itertools.combinations`` produce the
        # deterministic tiebreak order in lexicographic ASC.
        sorted_candidates = sorted(
            candidates,
            key=lambda w: (w.age, w.worker_id, w.stage_index),
        )

        # Group candidates by node for same-node-first iteration.
        # Insertion order of nodes follows first-occurrence in the
        # sorted candidate list so deterministic order survives the
        # grouping step. Workers whose node is unknown collapse into
        # a ``""`` bucket which is iterated like any other node.
        candidates_by_node: dict[str, list[DonorWorker]] = {}
        nodes_in_order: list[str] = []
        for worker in sorted_candidates:
            node = worker_nodes.get(worker.worker_id, "")
            bucket = candidates_by_node.get(node)
            if bucket is None:
                candidates_by_node[node] = [worker]
                nodes_in_order.append(node)
            else:
                bucket.append(worker)

        # Per-``find`` total-enumeration guard. ``evaluations`` (reset
        # per plan_size, below) caps *probes* but increments only after
        # a combo clears the cheap stage-budget pre-filter, so it cannot
        # bound the raw ``itertools.combinations`` walk when most combos
        # fail that filter. ``iterations`` counts *every* combo examined
        # - budget-rejected and dedup-skipped included - across all
        # plan sizes and both passes; the search returns ``None`` once
        # it exceeds ``max_iterations`` so a pathological candidate pool
        # cannot monopolise a scheduler cycle while the probe budget
        # still limits FFI cost as before.
        max_iterations = self.max_plan_combinations * _ITERATION_BUDGET_MULTIPLIER
        iterations = 0

        for plan_size in range(1, self.max_plan_size + 1):
            evaluations = 0
            # Dedup key uses the full planner identity
            # ``(stage_index, worker_id)`` so distinct combos that
            # accidentally collide on a global ``worker_id`` are not
            # silently merged.
            seen_combos: set[tuple[tuple[int, str], ...]] = set()

            # Pass 1: same-node combinations first.
            for node in nodes_in_order:
                same_node = candidates_by_node[node]
                if len(same_node) < plan_size:
                    continue
                for combo in itertools.combinations(same_node, plan_size):
                    if evaluations >= self.max_plan_combinations:
                        break
                    iterations += 1
                    if iterations > max_iterations:
                        return None
                    if _combo_violates_stage_budget(combo, removable_by_stage):
                        continue
                    combo_key = tuple((w.stage_index, w.worker_id) for w in combo)
                    seen_combos.add(combo_key)
                    evaluations += 1
                    removals = [(w.stage_index, w.worker_id) for w in combo]
                    probe = ctx.probe_add_after_removals(removals, receiver_stage_index)
                    if probe.feasible:
                        return DonorPlan(
                            removals=tuple(combo),
                            receiver_stage_index=receiver_stage_index,
                        )
                if evaluations >= self.max_plan_combinations:
                    break

            # Pass 2: cross-node combinations. Skip combos Pass 1
            # already evaluated so the probe cap counts unique
            # ``probe_add_after_removals`` calls only.
            if evaluations >= self.max_plan_combinations:
                continue
            for combo in itertools.combinations(sorted_candidates, plan_size):
                if evaluations >= self.max_plan_combinations:
                    break
                iterations += 1
                if iterations > max_iterations:
                    return None
                if _combo_violates_stage_budget(combo, removable_by_stage):
                    continue
                combo_key = tuple((w.stage_index, w.worker_id) for w in combo)
                if combo_key in seen_combos:
                    continue
                seen_combos.add(combo_key)
                evaluations += 1
                removals = [(w.stage_index, w.worker_id) for w in combo]
                probe = ctx.probe_add_after_removals(removals, receiver_stage_index)
                if probe.feasible:
                    return DonorPlan(
                        removals=tuple(combo),
                        receiver_stage_index=receiver_stage_index,
                    )

        return None


def _combo_violates_stage_budget(
    combo: tuple[DonorWorker, ...],
    removable_by_stage: Mapping[int, int],
) -> bool:
    """Return ``True`` when ``combo`` exceeds any donor stage's removable budget.

    A combo's ``stage_index`` distribution must respect every
    donor stage's per-cycle removable count
    (``len(workers) - stage_floors.get(stage_index, 1)``).
    Without this guard the multi-donor resource-fit search could
    drop one stage below its own floor and cascade the rescue.

    """
    stage_counts: dict[int, int] = {}
    for worker in combo:
        stage_counts[worker.stage_index] = stage_counts.get(worker.stage_index, 0) + 1
    return any(count > removable_by_stage.get(stage_index, 0) for stage_index, count in stage_counts.items())


__all__ = ("ResourceFitPlanner",)
