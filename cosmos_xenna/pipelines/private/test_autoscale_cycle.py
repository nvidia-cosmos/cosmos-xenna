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

"""Contract tests for ``AutoscaleCycle`` and its companion view types.

The cycle is the cross-phase working state every phase module reads
and mutates. These tests pin:

- construction with the required fields; snapshot fields stay unset
  until their producing phase assigns them;
- reading a snapshot field before its producing phase ran raises
  ``AttributeError`` (the intentional loud failure);
- ``view_for`` returns a view rooted in ``problem_state`` and
  consistent with the live ``stage_states`` map;
- ``cycle_logger(stage=...)`` returns a loguru logger whose
  ``record["extra"]`` carries the documented per-cycle fields.
"""

from typing import Any

import pytest
from loguru import logger as _loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import (
    AutoscaleCycle,
    StageCycleView,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_identity import BottleneckIdentity
from cosmos_xenna.pipelines.private.scheduling_py.state.outputs import BottleneckSnapshot, IntentPlan
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState


def _cluster() -> resources.ClusterResources:
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(
                used_cpus=0,
                total_cpus=16,
                gpus=[],
                name="node-0",
            ),
        },
    )


def _problem(stage_names: list[str]) -> data_structures.Problem:
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


def _problem_state(
    stage_specs: list[tuple[str, int]],
) -> data_structures.ProblemState:
    """Build a ProblemState; each tuple is ``(stage_name, num_workers)``."""
    states = []
    for name, num_workers in stage_specs:
        workers = [
            data_structures.ProblemWorkerGroupState.make(
                f"{name}-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(num_workers)
        ]
        states.append(
            data_structures.ProblemStageState(
                stage_name=name,
                workers=workers,
                slots_per_worker=2,
                is_finished=False,
            ),
        )
    return data_structures.ProblemState(states)


def _ctx(problem: data_structures.Problem, ps: data_structures.ProblemState) -> data_structures.AutoscalePlanContext:
    return data_structures.AutoscalePlanContext.from_problem_state(problem, ps)


def _stage_states(stage_names: list[str]) -> dict[str, StageRuntimeState]:
    return {name: StageRuntimeState(stage_name=name) for name in stage_names}


def _make_cycle(
    *,
    stage_names: list[str] | None = None,
    cycle_counter: int = 1,
    pipeline_name: str = "test-pipeline",
) -> tuple[AutoscaleCycle, dict[str, StageRuntimeState]]:
    """Build a minimal ``AutoscaleCycle`` paired with its stage_states map.

    Cross-cycle state (stage_states, worker_ages, etc.) no longer lives on
    ``AutoscaleCycle``; the test fixtures return the stage_states map
    alongside the cycle so callers can pass it explicitly to
    ``cycle.view_for(stage_index, stage_states)``.
    """
    if stage_names is None:
        stage_names = ["A", "B"]
    problem = _problem(stage_names)
    ps = _problem_state([(name, 1) for name in stage_names])
    cycle = AutoscaleCycle(
        ctx=_ctx(problem, ps),
        problem_state=ps,
        time=1234.0,
        cycle_counter=cycle_counter,
        pipeline_name=pipeline_name,
    )
    return cycle, _stage_states(stage_names)


class TestAutoscaleCycleConstruction:
    """The cycle starts with phase-populated fields unset."""

    def test_required_fields_are_set_on_construction(self) -> None:
        cycle, _ = _make_cycle()
        assert cycle.cycle_counter == 1
        assert cycle.pipeline_name == "test-pipeline"
        assert cycle.time == 1234.0

    def test_snapshot_fields_are_unset_at_construction(self) -> None:
        cycle, _ = _make_cycle()
        assert not hasattr(cycle, "donor_warmup_excluded_ids")
        assert not hasattr(cycle, "bottleneck")
        assert not hasattr(cycle, "intent")
        assert not hasattr(cycle, "pre_grow_worker_counts")
        assert not hasattr(cycle, "pre_shrink_worker_counts")


class TestSnapshotFieldAccess:
    """Reading a snapshot before its producing phase raises ``AttributeError``."""

    def test_reading_bottleneck_before_set_raises(self) -> None:
        cycle, _ = _make_cycle()
        with pytest.raises(AttributeError):
            _ = cycle.bottleneck

    def test_reading_intent_before_set_raises(self) -> None:
        cycle, _ = _make_cycle()
        with pytest.raises(AttributeError):
            _ = cycle.intent

    def test_reading_donor_warmup_excluded_ids_before_set_raises(self) -> None:
        cycle, _ = _make_cycle()
        with pytest.raises(AttributeError):
            _ = cycle.donor_warmup_excluded_ids

    def test_reading_pre_grow_worker_counts_before_set_raises(self) -> None:
        cycle, _ = _make_cycle()
        with pytest.raises(AttributeError):
            _ = cycle.pre_grow_worker_counts

    def test_reading_pre_shrink_worker_counts_before_set_raises(self) -> None:
        cycle, _ = _make_cycle()
        with pytest.raises(AttributeError):
            _ = cycle.pre_shrink_worker_counts

    def test_bottleneck_returns_assigned_snapshot(self) -> None:
        cycle, _ = _make_cycle()
        meta = BottleneckIdentity(
            engaged=True,
            stage_name="A",
            max_d_k=1.5,
            median_d_k=0.7,
            heterogeneity_ratio=2.0,
        )
        snapshot = BottleneckSnapshot(
            identity=meta,
            d_k_now={"A": 1.5},
            effective_capacities={"A": 4},
            channels_per_worker_group={"A": 1},
            balance_score_start=0.5,
        )
        cycle.bottleneck = snapshot
        assert cycle.bottleneck is snapshot
        assert cycle.bottleneck.identity is meta

    def test_intent_returns_assigned_plan(self) -> None:
        cycle, _ = _make_cycle()
        plan = IntentPlan(deltas={"A": 1, "B": -1})
        cycle.intent = plan
        assert cycle.intent is plan
        assert cycle.intent.deltas == {"A": 1, "B": -1}


class TestViewFor:
    """``view_for`` returns a stage view consistent with problem_state."""

    def test_view_for_returns_correct_stage_metadata(self) -> None:
        cycle, stage_states = _make_cycle(stage_names=["A", "B", "C"])
        view = cycle.view_for(1, stage_states)
        assert isinstance(view, StageCycleView)
        assert view.stage_index == 1
        assert view.stage_name == "B"
        assert view.runtime_state is stage_states["B"]

    def test_view_for_current_workers_matches_problem_state(self) -> None:
        problem = _problem(["A"])
        ps = _problem_state([("A", 3)])
        stage_states = _stage_states(["A"])
        cycle = AutoscaleCycle(
            ctx=_ctx(problem, ps),
            problem_state=ps,
            time=0.0,
            cycle_counter=1,
            pipeline_name="p",
        )
        view = cycle.view_for(0, stage_states)
        assert view.current_workers == 3

    def test_view_for_raises_on_out_of_range_index(self) -> None:
        cycle, stage_states = _make_cycle(stage_names=["A", "B"])
        with pytest.raises(IndexError, match="out of range"):
            cycle.view_for(5, stage_states)


class TestCycleLogger:
    """``cycle_logger`` returns a bound logger that emits the documented fields."""

    def test_cycle_logger_binds_cycle_pipeline_stage(self) -> None:
        cycle, _ = _make_cycle(cycle_counter=42, pipeline_name="my-pipe")
        captured: list[dict[str, Any]] = []

        def sink(message: Any) -> None:
            captured.append(dict(message.record["extra"]))

        sink_id = _loguru_logger.add(sink)
        try:
            cycle.cycle_logger(stage="A").info("hello", decision="classify")  # type: ignore[attr-defined]
        finally:
            _loguru_logger.remove(sink_id)

        assert captured, "logger did not emit anything"
        extra = captured[0]
        assert extra["cycle"] == 42
        assert extra["pipeline"] == "my-pipe"
        assert extra["stage"] == "A"
        assert extra["decision"] == "classify"

    def test_cycle_logger_defaults_stage_to_empty_string(self) -> None:
        cycle, _ = _make_cycle(cycle_counter=7, pipeline_name="p")
        captured: list[dict[str, Any]] = []

        def sink(message: Any) -> None:
            captured.append(dict(message.record["extra"]))

        sink_id = _loguru_logger.add(sink)
        try:
            cycle.cycle_logger().info("cluster-level", decision="cycle_summary")  # type: ignore[attr-defined]
        finally:
            _loguru_logger.remove(sink_id)

        assert captured
        assert captured[0]["stage"] == ""
        assert captured[0]["cycle"] == 7
