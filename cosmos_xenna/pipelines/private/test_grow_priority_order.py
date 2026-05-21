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


"""Tests for the unified Phase C grow-priority ordering helper.

Pin :func:`compute_grow_priority_order` against its three branches:

  * Bottleneck-engaged: stages sort by D_k DESC with depth DESC as
    tiebreak; stages without a finite D_k tail-last among themselves
    in depth-DESC order.
  * Bottleneck-disengaged AND ``enable_dag_priority=True`` -> depth
    DESC (downstream-first).
  * Bottleneck-disengaged AND ``enable_dag_priority=False`` ->
    problem order.

These are pure-function tests; the wiring into autoscale lives in
``test_saturation_aware_dag_growth.py`` (DAG path) and the new
Phase C / Phase D bottleneck-wiring tests in
``test_saturation_aware_scheduler.py``.
"""

import math

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.dag_priority import compute_grow_priority_order


def _problem(stage_names: list[str]) -> data_structures.Problem:
    """Build a CPU-only Problem with one stage per name."""
    cluster = resources.ClusterResources(
        nodes={"node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0")},
    )
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


class TestEmptyAndSingleStageProblems:
    """Pin the trivial cases so they short-circuit before any sort."""

    @pytest.mark.parametrize("bottleneck_engaged", [True, False])
    @pytest.mark.parametrize("enable_dag_priority", [True, False])
    def test_empty_problem_returns_empty_list(
        self,
        bottleneck_engaged: bool,
        enable_dag_priority: bool,
    ) -> None:
        """No stages -> empty order regardless of toggles."""
        order = compute_grow_priority_order(
            _problem([]),
            bottleneck_engaged=bottleneck_engaged,
            d_k_by_stage={},
            enable_dag_priority=enable_dag_priority,
        )
        assert order == []

    @pytest.mark.parametrize("bottleneck_engaged", [True, False])
    @pytest.mark.parametrize("enable_dag_priority", [True, False])
    def test_single_stage_returns_one_index(
        self,
        bottleneck_engaged: bool,
        enable_dag_priority: bool,
    ) -> None:
        """One stage -> [0] regardless of toggles."""
        order = compute_grow_priority_order(
            _problem(["only"]),
            bottleneck_engaged=bottleneck_engaged,
            d_k_by_stage={"only": 1.0},
            enable_dag_priority=enable_dag_priority,
        )
        assert order == [0]


class TestDisengagedFallback:
    """Pin the legacy DAG-depth and problem-order branches."""

    def test_disengaged_with_dag_priority_returns_depth_desc(self) -> None:
        """Bottleneck off + DAG toggle on -> [2, 1, 0]."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=False,
            d_k_by_stage={"A": 1.0, "B": 1.0, "C": 1.0},
            enable_dag_priority=True,
        )
        assert order == [2, 1, 0]

    def test_disengaged_without_dag_priority_returns_problem_order(self) -> None:
        """Bottleneck off + DAG toggle off -> [0, 1, 2]."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=False,
            d_k_by_stage={"A": 5.0, "B": 0.5, "C": 1.0},
            enable_dag_priority=False,
        )
        assert order == [0, 1, 2]


class TestBottleneckEngagedOrdering:
    """Pin the D_k-driven sort and its tiebreak / cold-start handling."""

    def test_engaged_sorts_by_d_k_descending(self) -> None:
        """Engaged + heterogeneous D_k -> argmax wins, others by D_k DESC."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=True,
            d_k_by_stage={"A": 1.0, "B": 4.0, "C": 1.5},
            enable_dag_priority=True,
        )
        assert order == [1, 2, 0]

    def test_engaged_uses_depth_desc_as_d_k_tiebreak(self) -> None:
        """Equal D_k -> deeper stage wins the tiebreak."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=True,
            d_k_by_stage={"A": 1.0, "B": 1.0, "C": 1.0},
            enable_dag_priority=True,
        )
        assert order == [2, 1, 0]

    def test_engaged_places_nan_d_k_at_tail_in_depth_desc(self) -> None:
        """Cold-start stages (NaN D_k) sort last among themselves by depth DESC."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C", "D"]),
            bottleneck_engaged=True,
            d_k_by_stage={"A": math.nan, "B": 4.0, "C": math.nan, "D": 1.0},
            enable_dag_priority=True,
        )
        # B (D=4) > D (D=1) > C (NaN, depth 2) > A (NaN, depth 0)
        assert order == [1, 3, 2, 0]

    def test_engaged_treats_zero_d_k_as_cold_start(self) -> None:
        """``D_k <= 0`` is excluded from the finite sort and tail-placed."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=True,
            d_k_by_stage={"A": 0.0, "B": 4.0, "C": 1.0},
            enable_dag_priority=True,
        )
        # B (4.0) > C (1.0) > A (0.0 -> NaN tail)
        assert order == [1, 2, 0]

    def test_engaged_with_empty_d_k_falls_back_to_depth_desc(self) -> None:
        """Engaged but no D_k values -> all NaN -> depth DESC fallback."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=True,
            d_k_by_stage={},
            enable_dag_priority=True,
        )
        assert order == [2, 1, 0]

    def test_engaged_overrides_disabled_dag_priority(self) -> None:
        """Engaged bottleneck wins over the legacy DAG toggle being False."""
        order = compute_grow_priority_order(
            _problem(["A", "B", "C"]),
            bottleneck_engaged=True,
            d_k_by_stage={"A": 1.0, "B": 4.0, "C": 1.5},
            enable_dag_priority=False,
        )
        assert order == [1, 2, 0]
