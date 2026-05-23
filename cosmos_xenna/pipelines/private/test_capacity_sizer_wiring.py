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

"""Wiring tests for the capacity sizer through the orchestrator.

Pins the contract that
:meth:`SaturationAwareScheduler._refresh_capacity_target_workers`
populates ``_StageRuntimeState.capacity_target_workers`` so the
classifier->compute_delta path observes a non-cold-start target once
``D_k`` becomes finite.
"""

import math

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import _resolve_auto_thresholds
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import (
    SaturationAwareConfig,
    SaturationAwareStageConfig,
)


def _cluster(*, total_cpus_per_node: int = 8) -> resources.ClusterResources:
    """Single-node CPU cluster sufficient for ProblemStage construction."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=total_cpus_per_node, gpus=[], name="node-0"),
        },
    )


def _problem(stage_names: list[str]) -> data_structures.Problem:
    """Single-CPU stage per name for orchestration tests."""
    cluster = _cluster()
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        for name in stage_names
    ]
    return data_structures.Problem(cluster, stages)


@pytest.fixture
def scheduler() -> SaturationAwareScheduler:
    """A scheduler with thresholds resolved for one stage named ``S``."""
    cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig())
    sched = SaturationAwareScheduler(cfg)
    sched.setup(_problem(["S"]))
    state = sched._stage_states["S"]
    state.resolved_thresholds = _resolve_auto_thresholds(stage_cfg=sched._stage_cfg("S"), slots_per_actor=4)
    return sched


class TestRefreshCapacityTargetWorkers:
    """Pin the orchestrator's capacity-target population logic."""

    def test_returns_none_when_s_k_is_unobservable(self, scheduler: SaturationAwareScheduler) -> None:
        """No S_k EWMA sample yet -> cold-start sentinel ``None``."""
        scheduler._s_k_ewma["S"] = math.nan
        result = scheduler._refresh_capacity_target_workers(
            stage_state=scheduler._stage_states["S"],
            stage_cfg=scheduler._stage_cfg("S"),
            input_queue_depth=10,
            observed_throughput=0.5,
            slots_per_worker=1,
            stage_name="S",
        )
        assert result is None

    def test_returns_none_when_thresholds_unresolved(self, scheduler: SaturationAwareScheduler) -> None:
        """Without resolved thresholds the utilisation target cannot be derived."""
        scheduler._stage_states["S"].resolved_thresholds = None
        scheduler._s_k_ewma["S"] = 5.0
        result = scheduler._refresh_capacity_target_workers(
            stage_state=scheduler._stage_states["S"],
            stage_cfg=scheduler._stage_cfg("S"),
            input_queue_depth=10,
            observed_throughput=0.5,
            slots_per_worker=1,
            stage_name="S",
        )
        assert result is None

    def test_returns_target_when_s_k_finite(self, scheduler: SaturationAwareScheduler) -> None:
        """Finite S_k + populated thresholds -> closed-form target."""
        scheduler._s_k_ewma["S"] = 2.0
        result = scheduler._refresh_capacity_target_workers(
            stage_state=scheduler._stage_states["S"],
            stage_cfg=scheduler._stage_cfg("S"),
            input_queue_depth=0,
            observed_throughput=0.5,
            slots_per_worker=1,
            stage_name="S",
        )
        assert result is not None
        assert result >= 1
