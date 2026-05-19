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

"""Behaviour tests for the Phase B minimum-worker floor.

Pin the contract:

  1. Every non-manual, non-finished stage reaches at least
     ``target_min = max(min_workers if set else 1,
                        min_workers_per_node * num_nodes if set else 0)``
     by the end of the autoscale cycle.
  2. Manual stages (``requested_num_workers is not None``) are skipped
     by the floor; their worker count is owned by the manual phases.
  3. Finished stages (``is_finished``) are skipped.
  4. When the cluster cannot satisfy the floor directly, the scheduler
     tries a one-worker donor swap before raising ``RuntimeError`` with
     operator-actionable context.
"""

from typing import cast

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 16) -> resources.ClusterResources:
    """Multi-node cluster sized to fit every test fixture."""
    return resources.ClusterResources(
        nodes={
            f"node-{i}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus_per_node,
                gpus=[],
                name=f"node-{i}",
            )
            for i in range(num_nodes)
        },
    )


def _problem(
    stage_specs: list[tuple[str, int | None]],
    *,
    num_nodes: int = 1,
    total_cpus_per_node: int = 16,
) -> data_structures.Problem:
    """Build a ``Problem`` whose stages carry the given ``requested_num_workers``."""
    cluster = _cluster(num_nodes=num_nodes, total_cpus_per_node=total_cpus_per_node)
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


def _make_workers(
    stage_name: str,
    count: int,
    *,
    num_nodes: int = 1,
) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers spread round-robin across nodes."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node=f"node-{i % num_nodes}", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
    *,
    num_nodes: int = 1,
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(name, num_workers, slots_per_worker, is_finished)``."""
    states = [
        data_structures.ProblemStageState(
            stage_name=name,
            workers=_make_workers(name, num_workers, num_nodes=num_nodes),
            slots_per_worker=slots,
            is_finished=finished,
        )
        for name, num_workers, slots, finished in stage_specs
    ]
    return data_structures.ProblemState(states)


class _RaisingAddContext:
    """Fake planner that surfaces a hard floor-enforcement failure."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[int] = []
        self._exc = exc or RuntimeError("planner context is drained")

    def worker_ids_by_stage(self) -> list[list[str]]:
        """Return a single empty stage so the floor loop reads ``current=0``."""
        return [[]]

    def try_add_worker(self, stage_index: int) -> data_structures.ProblemWorkerGroupState | None:
        """Raise the same exception types the planner can raise."""
        self.calls.append(stage_index)
        raise self._exc


class TestPhaseBFloor:
    """End-to-end Phase B floor enforcement via ``SaturationAwareScheduler.autoscale``."""

    def test_implicit_one_worker_floor_on_empty_pipeline(self) -> None:
        """A non-manual stage with zero workers and no explicit floor receives one worker."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 1

    def test_min_workers_floor_grows_to_configured_count(self) -> None:
        """``min_workers=4`` brings a non-manual stage from 0 to 4 workers."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=4))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 4

    def test_min_workers_per_node_scales_floor_with_cluster_size(self) -> None:
        """``min_workers_per_node=1`` on a 4-node cluster yields a floor of 4 workers."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers_per_node=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], num_nodes=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)], num_nodes=4),
        )
        assert len(solution.stages[0].new_workers) == 4

    def test_combined_min_workers_takes_max_of_two_knobs(self) -> None:
        """``min_workers=2`` combined with ``min_workers_per_node=1`` on 3 nodes -> floor=3."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=2, min_workers_per_node=1),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], num_nodes=3))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)], num_nodes=3),
        )
        assert len(solution.stages[0].new_workers) == 3

    def test_no_op_when_current_already_at_floor(self) -> None:
        """A stage already at its floor is not grown further."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=2))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 2, 1, False)]),
        )
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_no_shrink_when_current_above_floor(self) -> None:
        """A stage already above the floor is not shrunk; the floor enforces the lower bound only.

        With a 3-worker stage and an implicit floor of 1, the floor
        enforcement must leave the worker count untouched. Scale-down
        is the responsibility of a different decision phase that does
        not run here.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False)]),
        )
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_manual_stages_are_skipped_by_phase_b(self) -> None:
        """A manual stage's worker count is owned by Phase A, not by the floor.

        With ``requested_num_workers=2`` and ``min_workers=4``, Phase A grows the
        stage to exactly 2 (the operator request); Phase B does NOT push it
        further to 4 because manual stages are excluded from the floor.
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=4))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", 2)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 2

    def test_finished_stages_are_skipped(self) -> None:
        """A finished non-manual stage is not grown by Phase B."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=3))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, True)]),
        )
        assert solution.stages[0].new_workers == []

    def test_per_stage_override_takes_precedence_over_defaults(self) -> None:
        """``per_stage_overrides`` for stage B applies a floor only to B."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"B": SaturationAwareStageConfig(min_workers=3)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False), ("B", 0, 1, False)]),
        )
        # Defaults floor stage A at 1; override floors stage B at 3.
        assert len(solution.stages[0].new_workers) == 1
        assert len(solution.stages[1].new_workers) == 3

    def test_capacity_exhausted_raises_runtime_error(self) -> None:
        """A min_workers floor exceeding cluster capacity raises with operator context."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=10),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=4))
        expected = (
            r"target_min=10 \(achieved=4; from min_workers=10, "
            r"min_workers_per_node=None, num_nodes=1\)\. Cluster placement exhausted"
        )
        with pytest.raises(RuntimeError, match=expected):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 0, 1, False)]),
            )

    def test_capacity_exhausted_error_identifies_per_node_floor_source(self) -> None:
        """A per-node floor failure names the per-node knob and cluster size."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers_per_node=2),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], num_nodes=2, total_cpus_per_node=1))

        expected = (
            r"target_min=4 \(achieved=2; from min_workers=None, "
            r"min_workers_per_node=2, num_nodes=2\)\. Cluster placement exhausted"
        )
        with pytest.raises(RuntimeError, match=expected):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 0, 1, False)], num_nodes=2),
            )

    def test_floor_uses_donor_when_cluster_is_full(self) -> None:
        """A non-manual stage above its floor can donate to bootstrap another stage."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"receiver": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("donor", None), ("receiver", None)], total_cpus_per_node=4))

        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("donor", 3, 1, False), ("receiver", 1, 1, False)]),
        )

        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1

    def test_floor_called_before_setup_raises_runtime_error(self) -> None:
        """Calling the floor phase before ``setup`` fails with a clear scheduler error."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        with pytest.raises(RuntimeError, match="_run_phase_b_floor called before setup"):
            scheduler._run_phase_b_floor(
                cast(data_structures.AutoscalePlanContext, object()),
                _problem_state([("A", 0, 1, False)]),
            )

    def test_planner_runtime_error_propagates(self) -> None:
        """Hard planner failures are not rewritten as floor-capacity exhaustion."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        ctx = _RaisingAddContext()

        with pytest.raises(RuntimeError, match="planner context is drained"):
            scheduler._run_phase_b_floor(
                cast(data_structures.AutoscalePlanContext, ctx),
                _problem_state([("A", 0, 1, False)]),
            )

        assert ctx.calls == [0]

    def test_planner_index_error_propagates(self) -> None:
        """Planner index errors are not rewritten as floor-capacity exhaustion."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        ctx = _RaisingAddContext(IndexError("stage_index 0 out of range"))

        with pytest.raises(IndexError, match="stage_index 0 out of range"):
            scheduler._run_phase_b_floor(
                cast(data_structures.AutoscalePlanContext, ctx),
                _problem_state([("A", 0, 1, False)]),
            )

        assert ctx.calls == [0]

    def test_zero_stage_pipeline_is_no_op(self) -> None:
        """An empty pipeline has no floor entries to enforce."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([]),
        )

        assert solution.stages == []

    def test_manual_shrink_frees_capacity_for_floor_in_same_cycle(self) -> None:
        """Phase A delete frees CPUs that the floor consumes within one autoscale cycle.

        Cluster has room for 4 1-CPU workers. Stage A is manual at
        ``current=3, requested=1`` (frees 2 CPUs). Stage B is non-manual
        with ``min_workers=3`` and ``current=1`` (needs +2 CPUs). The
        delete-before-floor ordering inside ``autoscale`` makes the freed
        CPUs available to the floor loop without round-tripping.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"B": SaturationAwareStageConfig(min_workers=3)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", 1), ("B", None)], total_cpus_per_node=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False), ("B", 1, 1, False)]),
        )
        assert len(solution.stages[0].deleted_workers) == 2
        assert len(solution.stages[1].new_workers) == 2

    def test_manual_grow_and_floor_run_in_one_cycle(self) -> None:
        """Manual grow on A and floor enforcement on B both fire in the same cycle."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=3))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("manual", 2), ("auto", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("manual", 0, 1, False), ("auto", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 2
        assert len(solution.stages[1].new_workers) == 3
