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

"""Per-cycle factory for ``DonorPlanningContext`` snapshots.

``build_donor_planning_context`` consolidates the data the donor
flow needs into a single immutable bundle. It is invoked once per
cycle from the phases that drive donor decisions (Phase B floor,
Phase C saturation).

``build_worker_node_map`` builds the ``worker_id -> node_id``
mapping the resource-fit search uses to prefer same-node
combinations; SPMD worker groups share one node by construction
so the first allocation's node is the canonical answer.
"""

from collections.abc import Mapping

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def build_worker_node_map(problem_state: data_structures.ProblemState) -> dict[str, str]:
    """Return ``worker_id -> node_id`` for every worker_group in the snapshot.

    Consumed by the donor resource-fit search to prefer same-node
    combinations (the FGD/SPMD allocator binds a worker to one
    node at a time). SPMD groups share the same node by
    construction, so the helper reads the first allocation's
    node. Workers without allocations are absent and treated as
    unknown locality (collapse into a ``""`` bucket inside the
    same-node walk).

    """
    worker_nodes: dict[str, str] = {}
    for stage in problem_state.rust.stages:
        for worker_group in stage.worker_groups:
            if not worker_group.resources:
                continue
            worker_nodes[worker_group.id] = worker_group.resources[0].node
    return worker_nodes


def build_donor_planning_context(
    *,
    problem_state: data_structures.ProblemState,
    worker_ids_by_stage: tuple[tuple[str, ...], ...],
    stage_states: Mapping[str, StageRuntimeState],
    stage_configs: Mapping[str, SaturationAwareStageConfig],
    stage_floors: Mapping[int, int],
    worker_ages: Mapping[str, int],
    d_k_now: Mapping[str, float],
    effective_capacities: Mapping[str, int],
    s_k_ewma: Mapping[str, float],
    slots_per_worker_by_stage: Mapping[str, int],
    donor_warmup_exclusions: frozenset[str],
    cycle_counter: int,
    last_donation_cycle: dict[str, int],
    config: SaturationAwareConfig,
) -> DonorPlanningContext:
    """Build a ``DonorPlanningContext`` for one cycle.

    Snapshots every input the donor flow needs in one place so
    policies and the coordinator do not redrive ``problem_state``
    or the per-phase service value object. ``last_donation_cycle``
    is the only field passed in as a live reference;
    ``SaturationPolicy.on_commit`` advances it.

    """
    stage_names: tuple[str, ...] = tuple(stage.stage_name for stage in problem_state.rust.stages)
    worker_node_map = build_worker_node_map(problem_state)
    return DonorPlanningContext(
        stage_names=stage_names,
        stage_configs=dict(stage_configs),
        stage_states=dict(stage_states),
        stage_floors=dict(stage_floors),
        worker_ids_by_stage=worker_ids_by_stage,
        worker_ages=dict(worker_ages),
        worker_node_map=worker_node_map,
        d_k_now=dict(d_k_now),
        effective_capacities=dict(effective_capacities),
        s_k_ewma=dict(s_k_ewma),
        slots_per_worker_by_stage=dict(slots_per_worker_by_stage),
        donor_warmup_exclusions=donor_warmup_exclusions,
        cycle_counter=cycle_counter,
        last_donation_cycle=last_donation_cycle,
        config=config,
    )


__all__ = ("build_donor_planning_context", "build_worker_node_map")
