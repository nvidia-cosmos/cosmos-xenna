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

"""Cluster-wide per-cycle inputs to donor selection.

``DonorPlanningContext`` bundles every cycle-scoped read the donor
flow needs, so policies and the coordinator do not redrive the
data off ``problem_state`` per receiver. Built once per cycle by
the coordinator's factory (lands with ``DonorCoordinator``) from
the cycle snapshot and the per-phase service value object that
owns the donor flow (``FloorServices``).

See ``docs/scheduler/saturation-aware/`` for the algorithm.
"""

from collections.abc import Mapping

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@attrs.frozen
class DonorPlanningContext:
    """Per-cycle planning inputs the donor flow consumes.

    Bundled once per cycle so policies and the coordinator do not
    redrive ``problem_state``, the bottleneck snapshot, the
    measurement collector, and the anti-flap ledger per receiver.
    Cluster-wide (one instance per cycle); the receiver is passed
    separately to ``DonorCoordinator.acquire``.
    ``last_donation_cycle`` is a mutable live reference --
    ``SaturationPolicy.on_commit`` advances it.

    """

    stage_names: tuple[str, ...]
    stage_configs: Mapping[str, SaturationAwareStageConfig]
    stage_states: Mapping[str, StageRuntimeState]
    stage_floors: Mapping[int, int]
    worker_ids_by_stage: tuple[tuple[str, ...], ...]
    worker_ages: Mapping[str, int]
    worker_node_map: Mapping[str, str]
    d_k_now: Mapping[str, float]
    effective_capacities: Mapping[str, int]
    s_k_ewma: Mapping[str, float]
    slots_per_worker_by_stage: Mapping[str, int]
    donor_warmup_exclusions: frozenset[str]
    cycle_counter: int
    last_donation_cycle: dict[str, int]
    config: SaturationAwareConfig


__all__ = ("DonorPlanningContext",)
