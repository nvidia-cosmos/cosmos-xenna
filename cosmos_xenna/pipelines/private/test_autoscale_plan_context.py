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

"""Tests for the ``AutoscalePlanContext`` Python wrapper.

Pins the construction contract: ``from_problem_state`` accepts a
``Problem`` + ``ProblemState`` and returns a non-None wrapper that:

  1. Reports the right number of stages.
  2. Exposes the underlying Rust object via ``.rust``.
  3. Implements ``try_add_worker``: one fresh placement
     per call via Fragmentation Gradient Descent, ``None`` on no-fit,
     ``IndexError`` on out-of-range stage index.
  4. Implements ``try_remove_worker``: removes by id,
     increments ``pending_remove_count``, returns ``False`` for unknown
     ids, and leaves removed placements available for reuse.
  5. Implements ``into_solution``: drains staged adds/removes into a
     ``Solution`` ordered like the input stages.
"""

import uuid

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources


def _make_cpu_worker(worker_id: str, node: str = "node0") -> data_structures.ProblemWorkerGroupState:
    """Build a one-allocation CPU worker snapshot."""
    worker_resources = resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=[])
    return data_structures.ProblemWorkerGroupState.make(worker_id, [worker_resources])


def _make_two_node_gpu_cluster(num_gpus_per_node: int = 1) -> resources.ClusterResources:
    """Build a 2-node GPU cluster, each node with one CPU and ``num_gpus_per_node`` GPUs.

    Used by SPMD tests that need an allocation spanning multiple nodes.
    """
    return resources.ClusterResources(
        nodes={
            f"node{i}": resources.NodeResources(
                used_cpus=0.0,
                total_cpus=1.0,
                gpus=[
                    resources.GpuResources(index=j, uuid_=uuid.uuid4(), used_fraction=0.0)
                    for j in range(num_gpus_per_node)
                ],
                name=f"node{i}",
            )
            for i in range(2)
        }
    )


def _make_two_node_spmd_problem_and_state(
    *,
    seeded_worker_id: str | None = None,
) -> tuple[data_structures.Problem, data_structures.ProblemState]:
    """Build a one-stage SPMD pipeline that spans two single-GPU nodes.

    The stage is shaped as ``Resources(cpus=1, gpus=2, is_spmd=True)`` so
    the FGD planner must place one actor on each of the two nodes (a
    multi-allocation SPMD worker group). When ``seeded_worker_id`` is
    given, the returned ``ProblemState`` already owns that group.
    """
    cluster = _make_two_node_gpu_cluster(num_gpus_per_node=1)
    shape = resources.Resources(cpus=1.0, gpus=2, is_spmd=True).to_worker_shape(cluster)
    stage = data_structures.ProblemStage(
        name="spmd_stage",
        stage_batch_size=1,
        worker_shape=shape,
        requested_num_workers=None,
        over_provision_factor=None,
    )
    problem = data_structures.Problem(cluster_resources=cluster, stages=[stage])

    workers: list[data_structures.ProblemWorkerGroupState] = []
    if seeded_worker_id is not None:
        seed_allocations = [
            resources.WorkerResourcesInternal(
                node=f"node{i}",
                cpus=1.0,
                gpus=[resources.GpuAllocationInternal(offset=0, used_fraction=1.0)],
            )
            for i in range(2)
        ]
        workers.append(data_structures.ProblemWorkerGroupState.make(seeded_worker_id, seed_allocations))

    state = data_structures.ProblemState(
        stages=[
            data_structures.ProblemStageState(
                stage_name="spmd_stage",
                workers=workers,
                slots_per_worker=1,
                is_finished=False,
            )
        ]
    )
    return problem, state


def _build_one_stage_problem_and_state(
    workers: list[data_structures.ProblemWorkerGroupState] | None = None,
    *,
    state_stage_name: str = "stage_a",
) -> tuple[data_structures.Problem, data_structures.ProblemState]:
    """Build a one-stage CPU pipeline on a 2-node 4-cpu cluster.

    Returns the matched Problem + ProblemState pair so tests can pass
    them straight into ``AutoscalePlanContext.from_problem_state``.
    """
    cluster = resources.ClusterResources(
        nodes={
            f"node{i}": resources.NodeResources(
                used_cpus=0.0,
                total_cpus=4.0,
                gpus=[],
                name=f"node{i}",
            )
            for i in range(2)
        }
    )
    shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
    stage = data_structures.ProblemStage(
        name="stage_a",
        stage_batch_size=1,
        worker_shape=shape,
        requested_num_workers=None,
        over_provision_factor=None,
    )
    problem = data_structures.Problem(cluster_resources=cluster, stages=[stage])
    state = data_structures.ProblemState(
        stages=[
            data_structures.ProblemStageState(
                stage_name=state_stage_name,
                workers=[] if workers is None else workers,
                slots_per_worker=1,
                is_finished=False,
            )
        ]
    )
    return problem, state


class TestAutoscalePlanContextConstruction:
    """``AutoscalePlanContext.from_problem_state`` returns a populated wrapper."""

    def test_returns_a_non_none_wrapper(self) -> None:
        """Build a context from matching problem/state inputs."""
        problem, state = _build_one_stage_problem_and_state()

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx is not None

    def test_exposes_rust_property(self) -> None:
        """The ``.rust`` accessor returns the underlying Rust object.

        Catches refactor regressions where the wrapper hides the Rust
        handle, which downstream callers need for the FGD planning calls.
        """
        problem, state = _build_one_stage_problem_and_state()

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.rust is not None

    def test_num_stages_matches_input(self) -> None:
        """The context tracks one stage per pipeline stage."""
        problem, state = _build_one_stage_problem_and_state()

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.num_stages() == 1

    def test_accepts_existing_worker_snapshot(self) -> None:
        """Seed a context from a state that already owns one worker."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.num_stages() == 1

    def test_zero_stage_pipeline_succeeds(self) -> None:
        """An empty pipeline (zero stages) is a valid degenerate input.

        Catches off-by-one errors in the seeding loop (uniform branch
        of the workload estimate divides by ``len(stages)``; zero
        stages must take the empty-list short-circuit).
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=1.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        problem = data_structures.Problem(cluster_resources=cluster, stages=[])
        state = data_structures.ProblemState(stages=[])

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.num_stages() == 0

    def test_multi_stage_pipeline_reports_correct_stage_count(self) -> None:
        """Three-stage pipeline -> ``num_stages()`` returns 3.

        Catches single-stage-only assumptions in the seeding loop.
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=8.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        stages = [
            data_structures.ProblemStage(
                name=f"stage_{i}",
                stage_batch_size=1,
                worker_shape=shape,
                requested_num_workers=None,
                over_provision_factor=None,
            )
            for i in range(3)
        ]
        problem = data_structures.Problem(cluster_resources=cluster, stages=stages)
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name=f"stage_{i}",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                )
                for i in range(3)
            ]
        )

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.num_stages() == 3

    def test_independent_contexts_have_independent_rust_objects(self) -> None:
        """Two ``from_problem_state`` calls return distinct Rust objects.

        Guards against an accidental singleton or shared-state regression
        where multiple cycles would observe each other's pending plans.
        """
        problem, state = _build_one_stage_problem_and_state()

        ctx_a = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        ctx_b = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx_a is not ctx_b
        assert ctx_a.rust is not ctx_b.rust


class TestIntoSolution:
    """Contract for ``AutoscalePlanContext.into_solution``."""

    def test_idle_context_returns_empty_stage_solution(self) -> None:
        """An idle context still returns one empty solution entry per stage."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        solution = ctx.into_solution()

        assert len(solution.stages) == 1
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_fresh_add_appears_in_new_workers(self) -> None:
        """A staged add is routed to ``new_workers``."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        placed = ctx.try_add_worker(0)

        solution = ctx.into_solution()

        assert placed is not None
        assert [worker.id for worker in solution.stages[0].new_workers] == [placed.id]
        assert solution.stages[0].deleted_workers == []

    def test_remove_appears_in_deleted_workers(self) -> None:
        """A staged remove is routed to ``deleted_workers``."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        assert ctx.try_remove_worker(0, "w0") is True

        solution = ctx.into_solution()

        assert solution.stages[0].new_workers == []
        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["w0"]

    def test_mixed_stage_add_and_remove_route_to_matching_stages(self) -> None:
        """A delete on one stage and an add on another keep their stage index."""
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=3.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        stage_a = data_structures.ProblemStage(
            name="stage_a",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        stage_b = data_structures.ProblemStage(
            name="stage_b",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        problem = data_structures.Problem(cluster_resources=cluster, stages=[stage_a, stage_b])
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[_make_cpu_worker("w0")],
                    slots_per_worker=1,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="stage_b",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                ),
            ]
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.try_remove_worker(0, "w0") is True
        added = ctx.try_add_worker(1)
        assert added is not None

        solution = ctx.into_solution()

        assert solution.stages[0].new_workers == []
        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["w0"]
        assert [worker.id for worker in solution.stages[1].new_workers] == [added.id]
        assert solution.stages[1].deleted_workers == []

    def test_seeded_worker_removed_then_reused_appears_in_neither_list(self) -> None:
        """Cycle-internal reuse must not surface as either an add or a delete.

        Pre-seed a live worker, stage its removal, then call
        ``try_add_worker`` so FGD picks the freed placement back up.
        The Solution should be a no-op: the worker stayed in the live
        cluster the whole time, only un-staged in between.
        """
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        assert ctx.try_remove_worker(0, "w0") is True
        reused = ctx.try_add_worker(0)
        assert reused is not None
        assert reused.id == "w0"

        solution = ctx.into_solution()

        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_freshly_added_then_removed_worker_appears_in_neither_list(self) -> None:
        """Add+remove of the same fresh worker collapses to a no-op.

        Pins the symmetric ``cancel_pending_add`` path on the remove
        side: a worker introduced by ``try_add_worker`` and removed in
        the same cycle must not appear in ``new_workers`` (the planner
        is already retracting the add) and must not appear in
        ``deleted_workers`` (it was never in the live set).
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        added = ctx.try_add_worker(0)
        assert added is not None
        assert ctx.try_remove_worker(0, added.id) is True

        solution = ctx.into_solution()

        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_preserves_per_stage_slots_per_worker(self) -> None:
        """``StageSolution.slots_per_worker`` round-trips the input.

        Build a 2-stage pipeline whose stages declare different
        ``slots_per_worker`` values (4 and 11). The Solution must
        report those exact values per stage; the planner does not
        change them.
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=8.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        stage_a = data_structures.ProblemStage(
            name="stage_a",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        stage_b = data_structures.ProblemStage(
            name="stage_b",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[stage_a, stage_b],
        )
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[],
                    slots_per_worker=4,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="stage_b",
                    workers=[],
                    slots_per_worker=11,
                    is_finished=False,
                ),
            ]
        )

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        solution = ctx.into_solution()

        assert len(solution.stages) == 2
        assert solution.stages[0].slots_per_worker == 4
        assert solution.stages[1].slots_per_worker == 11

    def test_preserves_stage_order(self) -> None:
        """``Solution.stages`` order matches ``Problem.stages`` order.

        The streaming layer applies each ``StageSolution`` to the
        stage at the matching list index, so a reorder would silently
        misroute adds/removes to the wrong stage.
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=8.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        stages = [
            data_structures.ProblemStage(
                name=f"stage_{letter}",
                stage_batch_size=1,
                worker_shape=shape,
                requested_num_workers=None,
                over_provision_factor=None,
            )
            for letter in ("a", "b", "c")
        ]
        problem = data_structures.Problem(cluster_resources=cluster, stages=stages)
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name=f"stage_{letter}",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                )
                for letter in ("a", "b", "c")
            ]
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        # Only the middle stage gets a fresh add.
        added = ctx.try_add_worker(1)
        assert added is not None

        solution = ctx.into_solution()

        assert len(solution.stages) == 3
        assert solution.stages[0].new_workers == []
        assert [worker.id for worker in solution.stages[1].new_workers] == [added.id]
        assert solution.stages[2].new_workers == []

    def test_second_call_returns_empty_solution(self) -> None:
        """Drain semantics: second call yields per-stage empty lists.

        Pinning this prevents a future refactor from cloning the
        pending lists instead of draining them, which would cause
        the streaming layer to apply the same delta twice.
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        ctx.try_add_worker(0)

        first = ctx.into_solution()
        second = ctx.into_solution()

        assert len(first.stages[0].new_workers) == 1
        assert second.stages[0].new_workers == []
        assert second.stages[0].deleted_workers == []

    def test_pending_counts_drop_to_zero_after_into_solution(self) -> None:
        """``pending_*_count`` reports no staged deltas after finalization."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        added = ctx.try_add_worker(0)
        assert added is not None
        assert ctx.try_remove_worker(0, "w0") is True
        assert ctx.pending_add_count(0) == 1
        assert ctx.pending_remove_count(0) == 1

        solution = ctx.into_solution()

        assert len(solution.stages[0].new_workers) == 1
        assert len(solution.stages[0].deleted_workers) == 1
        assert ctx.pending_add_count(0) == 0
        assert ctx.pending_remove_count(0) == 0

    def test_independent_contexts_produce_independent_solutions(self) -> None:
        """Two contexts built from the same input do not share staged plans.

        Catches accidental sharing of the underlying Rust state across
        Python wrappers, which would break the "one context per cycle"
        contract.
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx_a = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        ctx_b = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        ctx_a.try_add_worker(0)
        solution_a = ctx_a.into_solution()
        solution_b = ctx_b.into_solution()

        assert len(solution_a.stages[0].new_workers) == 1
        assert solution_b.stages[0].new_workers == []

    def test_seeded_spmd_worker_remove_routes_to_deleted_workers(self) -> None:
        """A staged remove on an SPMD multi-allocation group reaches the SPMD drain path.

        The SPMD branch keeps live groups in a different internal map
        (``current_worker_groups`` rather than ``current_workers``); a
        regression that mixed up the two would silently drop SPMD
        removes from the Solution. Seeding a multi-node SPMD worker
        and confirming it appears in ``deleted_workers`` pins the
        end-to-end contract.
        """
        problem, state = _make_two_node_spmd_problem_and_state(seeded_worker_id="spmd_w0")
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.try_remove_worker(0, "spmd_w0") is True

        solution = ctx.into_solution()

        assert solution.stages[0].new_workers == []
        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["spmd_w0"]

    def test_freshly_added_spmd_worker_then_removed_collapses_to_no_op(self) -> None:
        """SPMD ``try_remove_worker`` cancels the pending add, leaving no Solution delta.

        Pins the SPMD branch of ``cancel_pending_add``: a multi-
        allocation group introduced via ``try_add_worker`` and removed
        in the same cycle must not surface as either an add or a
        delete (the planner is retracting an addition that was never
        live). Without the SPMD-side cancel logic the Solution would
        contain a ghost add, and the streaming layer would launch an
        actor for a worker that the planner already withdrew.
        """
        problem, state = _make_two_node_spmd_problem_and_state(seeded_worker_id=None)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        added = ctx.try_add_worker(0)
        assert added is not None
        assert ctx.try_remove_worker(0, added.id) is True

        solution = ctx.into_solution()

        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []
        assert ctx.pending_add_count(0) == 0
        assert ctx.pending_remove_count(0) == 0

    def test_try_add_worker_after_into_solution_raises_runtime_error(self) -> None:
        """A drained context refuses further ``try_add_worker`` calls.

        Pins the drained-state guard from the Python side: once
        ``into_solution`` has handed off the plan for this cycle, any
        subsequent staging is a programming bug (the streaming layer
        has already received the per-stage deltas). The guard must
        surface this misuse as ``RuntimeError`` rather than silently
        mutating now-empty pending maps.
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        ctx.into_solution()

        with pytest.raises(RuntimeError):
            ctx.try_add_worker(0)

    def test_try_remove_worker_after_into_solution_raises_runtime_error(self) -> None:
        """A drained context refuses further ``try_remove_worker`` calls.

        Same drained-state contract as the add-side guard, applied to
        the remove entrypoint to prevent post-drain mutation of the
        emitted plan.
        """
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        ctx.into_solution()

        with pytest.raises(RuntimeError):
            ctx.try_remove_worker(0, "w0")

    def test_mixed_cpu_and_spmd_stages_drain_independently(self) -> None:
        """A pipeline mixing CPU and SPMD stages drains each branch correctly.

        Pins that ``into_solution`` does not cross-contaminate
        per-stage lists when one stage uses single-allocation CPU
        workers (``current_workers``) and another uses multi-
        allocation SPMD groups (``current_worker_groups``). A
        regression that used a shared map for both shapes would
        silently merge the lists.

        The cluster sizes each node at 2 CPUs + 1 GPU so both stages
        can place a worker simultaneously: one CPU actor on an
        arbitrary node plus one SPMD group spanning both nodes.
        """
        gpu_cluster = resources.ClusterResources(
            nodes={
                f"node{i}": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=2.0,
                    gpus=[
                        resources.GpuResources(index=0, uuid_=uuid.uuid4(), used_fraction=0.0),
                    ],
                    name=f"node{i}",
                )
                for i in range(2)
            }
        )
        cpu_shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(gpu_cluster)
        spmd_shape = resources.Resources(cpus=1.0, gpus=2, is_spmd=True).to_worker_shape(gpu_cluster)
        cpu_stage = data_structures.ProblemStage(
            name="cpu_stage",
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        spmd_stage = data_structures.ProblemStage(
            name="spmd_stage",
            stage_batch_size=1,
            worker_shape=spmd_shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        problem = data_structures.Problem(cluster_resources=gpu_cluster, stages=[cpu_stage, spmd_stage])
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="cpu_stage",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="spmd_stage",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                ),
            ]
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        cpu_added = ctx.try_add_worker(0)
        spmd_added = ctx.try_add_worker(1)
        assert cpu_added is not None
        assert spmd_added is not None

        solution = ctx.into_solution()

        assert [worker.id for worker in solution.stages[0].new_workers] == [cpu_added.id]
        assert solution.stages[0].deleted_workers == []
        assert [worker.id for worker in solution.stages[1].new_workers] == [spmd_added.id]
        assert solution.stages[1].deleted_workers == []


class TestTryRemoveWorker:
    """Contract for ``AutoscalePlanContext.try_remove_worker``."""

    def test_existing_worker_increments_pending_remove_count(self) -> None:
        """Removing a live worker stages exactly one pending remove."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        removed = ctx.try_remove_worker(0, "w0")

        assert removed is True
        assert ctx.pending_remove_count(0) == 1
        assert ctx.pending_add_count(0) == 0

    def test_unknown_worker_returns_false_without_mutating_counts(self) -> None:
        """Unknown worker ids are no-op failures."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        removed = ctx.try_remove_worker(0, "missing")

        assert removed is False
        assert ctx.pending_remove_count(0) == 0
        assert ctx.pending_add_count(0) == 0

    def test_out_of_range_stage_index_raises_index_error(self) -> None:
        """``stage_index >= num_stages()`` surfaces as ``IndexError``."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(IndexError):
            ctx.try_remove_worker(99, "w0")

    def test_second_remove_of_same_worker_returns_false(self) -> None:
        """The same worker cannot be staged for removal twice."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        first = ctx.try_remove_worker(0, "w0")
        second = ctx.try_remove_worker(0, "w0")

        assert first is True
        assert second is False
        assert ctx.pending_remove_count(0) == 1

    def test_remove_then_add_reuses_removed_worker(self) -> None:
        """A later add revives a removed placement instead of staging a fresh add."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("w0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.try_remove_worker(0, "w0") is True
        reused = ctx.try_add_worker(0)

        assert reused is not None
        assert reused.id == "w0"
        assert ctx.pending_remove_count(0) == 0
        assert ctx.pending_add_count(0) == 0

    def test_freshly_added_worker_removal_cancels_pending_add(self) -> None:
        """Removing a fresh add cancels it instead of staging a delete."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        first = ctx.try_add_worker(0)
        second = ctx.try_add_worker(0)

        assert first is not None
        assert second is not None
        assert ctx.try_remove_worker(0, first.id) is True
        assert ctx.pending_remove_count(0) == 0
        assert ctx.pending_add_count(0) == 1
        replacement = ctx.try_add_worker(0)

        assert replacement is not None
        assert replacement.id != first.id
        assert ctx.pending_remove_count(0) == 0
        assert ctx.pending_add_count(0) == 2


class TestTryAddWorker:
    """Contract for ``AutoscalePlanContext.try_add_worker``.

    Pins the Python wrapper's behaviour: a fresh placement returns a
    ``ProblemWorkerGroupState`` whose resources match a real cluster
    node, sequential calls deplete capacity, and out-of-range stage
    indices surface as ``IndexError``.
    """

    def test_fresh_add_returns_a_placement_with_one_allocation(self) -> None:
        """First add on an empty cluster returns a one-allocation worker.

        Verifies the wrapper materialises the Rust placement back into
        a Python ``ProblemWorkerGroupState`` with the expected shape
        (one resource entry for a CPU-only stage).
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        placed = ctx.try_add_worker(0)

        assert placed is not None
        assert len(placed.resources) == 1, "CPU stages have one allocation per worker"

    def test_fresh_add_increments_pending_add_count(self) -> None:
        """A successful add must be visible via ``pending_add_count``."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.pending_add_count(0) == 0
        ctx.try_add_worker(0)
        assert ctx.pending_add_count(0) == 1

    def test_returns_none_when_cluster_is_full(self) -> None:
        """Once capacity is exhausted, subsequent adds return ``None``.

        Build a 1-node 1-cpu pipeline so the second add cannot fit.
        Verifies the wrapper passes through the Rust ``Ok(None)``
        signal as Python ``None`` rather than raising.
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=1.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_a",
                    stage_batch_size=1,
                    worker_shape=shape,
                    requested_num_workers=None,
                    over_provision_factor=None,
                )
            ],
        )
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                )
            ]
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        first = ctx.try_add_worker(0)
        second = ctx.try_add_worker(0)

        assert first is not None
        assert second is None, "second add must fail with None on a 1-cpu cluster"
        assert ctx.pending_add_count(0) == 1, "only the successful add was committed"

    def test_out_of_range_stage_index_raises_index_error(self) -> None:
        """``stage_index >= num_stages()`` surfaces as ``IndexError``.

        Catches caller bugs (off-by-one) before they can mutate the
        plan. The Rust side returns ``PyIndexError`` which PyO3
        converts to Python's built-in ``IndexError``.
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(IndexError):
            ctx.try_add_worker(99)

    def test_independent_contexts_do_not_share_pending_state(self) -> None:
        """Two contexts built from the same inputs plan independently.

        Adds in ``ctx_a`` must not appear in ``ctx_b``'s pending counts.
        Catches accidental cross-cycle state leakage via shared Rust
        objects or mutable globals.
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx_a = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        ctx_b = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        ctx_a.try_add_worker(0)

        assert ctx_a.pending_add_count(0) == 1
        assert ctx_b.pending_add_count(0) == 0

    def test_fresh_add_skips_seeded_numeric_worker_id(self) -> None:
        """Fresh ids do not collide with live workers from the input state."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("0")])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        placed = ctx.try_add_worker(0)

        assert placed is not None
        assert placed.id != "0"


class TestPendingCounts:
    """Contract for ``pending_add_count`` / ``pending_remove_count``."""

    def test_pending_counts_start_at_zero(self) -> None:
        """A fresh context has nothing staged.

        Pins the construction contract: a context with no planning
        calls must report 0 staged adds and 0 staged removes for every
        stage. Future cluster-wide invariants depend on this.
        """
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.pending_add_count(0) == 0
        assert ctx.pending_remove_count(0) == 0

    def test_pending_add_count_rejects_out_of_range_index(self) -> None:
        """Out-of-range stage index raises ``IndexError``."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(IndexError):
            ctx.pending_add_count(99)

    def test_pending_remove_count_rejects_out_of_range_index(self) -> None:
        """Out-of-range stage index raises ``IndexError``."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(IndexError):
            ctx.pending_remove_count(99)


class TestAutoscalePlanContextValidation:
    """Construction-time validation rejects mismatched inputs."""

    def test_rejects_stage_count_mismatch(self) -> None:
        """Problem with 2 stages + state with 1 stage raises ValueError."""
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=4.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        stage_a = data_structures.ProblemStage(
            name="stage_a",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        stage_b = data_structures.ProblemStage(
            name="stage_b",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        problem = data_structures.Problem(cluster_resources=cluster, stages=[stage_a, stage_b])
        # Only one stage in state -- mismatch.
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                )
            ]
        )

        with pytest.raises(ValueError, match="stage count mismatch"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_stage_name_mismatch(self) -> None:
        """Problem/state stage names must match at each position."""
        problem, state = _build_one_stage_problem_and_state(state_stage_name="wrong_stage")

        with pytest.raises(ValueError, match="stage name mismatch"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_empty_non_spmd_worker_resources(self) -> None:
        """A non-SPMD current worker must carry exactly one allocation."""
        empty_worker = data_structures.ProblemWorkerGroupState.make("empty", [])
        problem, state = _build_one_stage_problem_and_state(workers=[empty_worker])

        with pytest.raises(ValueError, match="exactly 1 resource allocation"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_multi_allocation_non_spmd_worker(self) -> None:
        """A non-SPMD current worker cannot carry multiple allocations."""
        worker = data_structures.ProblemWorkerGroupState.make(
            "multi",
            [
                resources.WorkerResourcesInternal(node="node0", cpus=1.0, gpus=[]),
                resources.WorkerResourcesInternal(node="node1", cpus=1.0, gpus=[]),
            ],
        )
        problem, state = _build_one_stage_problem_and_state(workers=[worker])

        with pytest.raises(ValueError, match="exactly 1 resource allocation"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_unknown_worker_node(self) -> None:
        """Current worker allocations must reference a known cluster node."""
        problem, state = _build_one_stage_problem_and_state(workers=[_make_cpu_worker("missing", node="node9")])

        with pytest.raises(ValueError, match="unknown node node9"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_invalid_gpu_offset(self) -> None:
        """Current worker GPU offsets must exist on the referenced node."""
        worker = data_structures.ProblemWorkerGroupState.make(
            "bad_gpu",
            [
                resources.WorkerResourcesInternal(
                    node="node0",
                    cpus=1.0,
                    gpus=[resources.GpuAllocationInternal(offset=0, used_fraction=1.0)],
                )
            ],
        )
        problem, state = _build_one_stage_problem_and_state(workers=[worker])

        with pytest.raises(ValueError, match="GPU offset 0"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_state_more_stages_than_problem(self) -> None:
        """Mismatch in the opposite direction (state > problem) also fails.

        Without this guard the seeding loop's ``zip`` would silently
        truncate the tail of the state, so the working cluster would
        not reflect every existing worker.
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=4.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_a",
                    stage_batch_size=1,
                    worker_shape=shape,
                    requested_num_workers=None,
                    over_provision_factor=None,
                )
            ],
        )
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="stage_b",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                ),
            ]
        )

        with pytest.raises(ValueError, match="stage count mismatch"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)

    def test_rejects_workers_overflowing_cluster(self) -> None:
        """Existing workers that do not fit the cluster surface as RuntimeError.

        Ensures the seeding loop's allocation failure is converted to a
        Python RuntimeError instead of silently dropping workers (which
        would desync the saturation snapshot from the live placement).
        """
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=1.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_a",
                    stage_batch_size=1,
                    worker_shape=shape,
                    requested_num_workers=None,
                    over_provision_factor=None,
                )
            ],
        )
        # Two 1-cpu workers on a 1-cpu cluster -> the second cannot fit.
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[_make_cpu_worker("w0"), _make_cpu_worker("w1")],
                    slots_per_worker=1,
                    is_finished=False,
                )
            ]
        )

        with pytest.raises(RuntimeError, match="failed to seed cluster"):
            data_structures.AutoscalePlanContext.from_problem_state(problem, state)


class TestAutoscalePlanContextWorkerAges:
    """Worker-age tracking exposed for the saturation-aware donor logic.

    Pins the contract that any caller (cross-stage donor logic, audit
    tooling, age-based eviction policies) relies on: the context
    maintains a per-worker age map that callers seed from the previous
    cycle (already incremented), fresh placements get age 0, and
    cancel-pending-add drops the entry so a re-staging cannot resurrect
    stale data.
    """

    def test_no_seed_means_every_existing_worker_starts_at_age_zero(self) -> None:
        """Cold-start path: missing seed maps every seeded worker to 0."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0"), _make_cpu_worker("w1")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        assert ctx.worker_ages() == {"w0": 0, "w1": 0}

    def test_seed_round_trips_values_through_constructor(self) -> None:
        """Caller-supplied ages survive the FFI hop intact."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0"), _make_cpu_worker("w1")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 3, "w1": 11},
        )
        assert ctx.worker_ages() == {"w0": 3, "w1": 11}

    def test_seed_default_to_zero_for_known_worker_missing_from_seed(self) -> None:
        """Partial seed: unknown ids default to 0 (treated as freshly observed)."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0"), _make_cpu_worker("w1")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 4},
        )
        ages = ctx.worker_ages()
        assert ages["w0"] == 4
        assert ages["w1"] == 0

    def test_seed_filters_stale_ids_not_in_state(self) -> None:
        """Seed entries for dead workers must not bleed into the new cycle."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 2, "ghost": 99},
        )
        ages = ctx.worker_ages()
        assert ages == {"w0": 2}
        assert ctx.worker_age("ghost") is None

    def test_new_worker_placed_via_try_add_has_age_zero(self) -> None:
        """Fresh placements start at age 0 (treated as the youngest possible)."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        placed = ctx.try_add_worker(0)
        assert placed is not None
        assert ctx.worker_age(placed.id) == 0

    def test_new_spmd_worker_placed_via_try_add_has_age_zero(self) -> None:
        """Fresh SPMD placements also start at age 0."""
        problem, state = _make_two_node_spmd_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        placed = ctx.try_add_worker(0)

        assert placed is not None
        assert ctx.worker_age(placed.id) == 0

    def test_oldest_first_sort_returns_correct_order(self) -> None:
        """Descending-age sort returns oldest first (caller-side ordering pin)."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[
                _make_cpu_worker("w_middle"),
                _make_cpu_worker("w_youngest"),
                _make_cpu_worker("w_oldest"),
            ],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w_middle": 5, "w_youngest": 0, "w_oldest": 10},
        )
        ages = ctx.worker_ages()
        ordered = sorted(ages.items(), key=lambda kv: kv[1], reverse=True)
        assert [worker_id for worker_id, _ in ordered] == [
            "w_oldest",
            "w_middle",
            "w_youngest",
        ]

    def test_youngest_first_sort_matches_donor_selection_order(self) -> None:
        """Ascending-age sort returns youngest first (donor selection ordering)."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[
                _make_cpu_worker("w_middle"),
                _make_cpu_worker("w_youngest"),
                _make_cpu_worker("w_oldest"),
            ],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w_middle": 5, "w_youngest": 0, "w_oldest": 10},
        )
        ages = ctx.worker_ages()
        ordered = sorted(ages.items(), key=lambda kv: kv[1])
        assert [worker_id for worker_id, _ in ordered] == [
            "w_youngest",
            "w_middle",
            "w_oldest",
        ]

    def test_age_increments_cycle_to_cycle(self) -> None:
        """Caller-driven cycle-to-cycle increment: ages advance by 1 each cycle."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx_cycle_1 = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        assert ctx_cycle_1.worker_age("w0") == 0

        # Caller advances every surviving worker's age by 1 between cycles.
        next_ages = {worker_id: age + 1 for worker_id, age in ctx_cycle_1.worker_ages().items()}

        # Cycle 2: same physical worker is alive; the seeded age is 1.
        ctx_cycle_2 = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages=next_ages,
        )
        assert ctx_cycle_2.worker_age("w0") == 1

    def test_cancel_pending_add_drops_age_entry(self) -> None:
        """Worker added then immediately removed: the stale age 0 entry must be dropped."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)
        placed = ctx.try_add_worker(0)
        assert placed is not None
        assert ctx.worker_age(placed.id) == 0
        assert ctx.try_remove_worker(0, placed.id) is True
        assert ctx.worker_age(placed.id) is None

    def test_stage_for_removal_keeps_age_for_potential_reuse(self) -> None:
        """A staged-for-removal worker must keep its age so reuse preserves it."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 4},
        )
        assert ctx.try_remove_worker(0, "w0") is True
        assert ctx.worker_age("w0") == 4

    def test_reuse_path_preserves_original_age(self) -> None:
        """Remove + re-add (reuse path) must NOT collapse the age to 0."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 7},
        )
        assert ctx.try_remove_worker(0, "w0") is True
        reused = ctx.try_add_worker(0)
        assert reused is not None
        assert reused.id == "w0"
        assert ctx.worker_age("w0") == 7

    def test_spmd_reuse_path_preserves_original_age(self) -> None:
        """SPMD remove + re-add reuse must NOT collapse age to 0."""
        problem, state = _make_two_node_spmd_problem_and_state(seeded_worker_id="spmd_w0")
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"spmd_w0": 9},
        )

        assert ctx.try_remove_worker(0, "spmd_w0") is True
        reused = ctx.try_add_worker(0)

        assert reused is not None
        assert reused.id == "spmd_w0"
        assert ctx.worker_age("spmd_w0") == 9

    def test_cancel_pending_spmd_add_drops_age_entry(self) -> None:
        """Fresh SPMD add then remove retracts the age entry too."""
        problem, state = _make_two_node_spmd_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        placed = ctx.try_add_worker(0)
        assert placed is not None
        assert ctx.worker_age(placed.id) == 0

        assert ctx.try_remove_worker(0, placed.id) is True

        assert ctx.worker_age(placed.id) is None

    def test_zero_stage_pipeline_yields_empty_age_map(self) -> None:
        """Boundary: no stages -> empty age map regardless of seed."""
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=1.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        problem = data_structures.Problem(cluster_resources=cluster, stages=[])
        state = data_structures.ProblemState(stages=[])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"ghost": 99},
        )
        assert ctx.worker_ages() == {}

    def test_try_add_worker_returning_none_does_not_mutate_age_map(self) -> None:
        """Cluster-full add returns None and must NOT introduce a stale entry.

        Mutation symmetry matters for any age-aware donor logic: a
        failed try_add_worker (cluster full) followed by a donor swap +
        retry must see the SAME pre-add age map -- if a None-returning
        path leaked an entry, the retry would observe a non-existent
        worker as age 0 and incorrectly bias donor selection.
        """
        # Build a 1-node, 1-cpu cluster with a single seeded worker. A
        # second 1-cpu add cannot fit, so try_add_worker returns None.
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=1.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        stage = data_structures.ProblemStage(
            name="stage_a",
            stage_batch_size=1,
            worker_shape=shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        problem = data_structures.Problem(cluster_resources=cluster, stages=[stage])
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[_make_cpu_worker("w0")],
                    slots_per_worker=1,
                    is_finished=False,
                )
            ]
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 4},
        )
        ages_before = ctx.worker_ages()
        assert ctx.try_add_worker(0) is None, "second 1-cpu add must fail on a 1-cpu cluster"
        ages_after = ctx.worker_ages()
        assert ages_after == ages_before
        assert ages_after == {"w0": 4}

    def test_try_remove_worker_unknown_id_does_not_mutate_age_map(self) -> None:
        """Unknown-id remove returns False and must NOT touch worker_ages."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 6},
        )
        ages_before = ctx.worker_ages()
        assert ctx.try_remove_worker(0, "nonexistent") is False
        assert ctx.worker_ages() == ages_before

    def test_cross_stage_age_isolation_across_multi_stage_pipeline(self) -> None:
        """Per-worker ages in different stages remain independent."""
        cluster = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=4.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_a",
                    stage_batch_size=1,
                    worker_shape=shape,
                    requested_num_workers=None,
                    over_provision_factor=None,
                ),
                data_structures.ProblemStage(
                    name="stage_b",
                    stage_batch_size=1,
                    worker_shape=shape,
                    requested_num_workers=None,
                    over_provision_factor=None,
                ),
            ],
        )
        state = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[_make_cpu_worker("a0"), _make_cpu_worker("a1")],
                    slots_per_worker=1,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="stage_b",
                    workers=[_make_cpu_worker("b0")],
                    slots_per_worker=1,
                    is_finished=False,
                ),
            ]
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"a0": 1, "a1": 9, "b0": 5},
        )
        # Mutate stage_a only; stage_b ages must remain untouched.
        assert ctx.try_remove_worker(0, "a0") is True
        new_b = ctx.try_add_worker(1)
        assert new_b is not None
        assert ctx.worker_age("a1") == 9
        assert ctx.worker_age("b0") == 5
        assert ctx.worker_age(new_b.id) == 0

    def test_mixed_seeded_and_fresh_workers_have_independent_ages(self) -> None:
        """Seeded workers keep their seed ages; mid-cycle adds get age 0 alongside them."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("seeded_old", node="node0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"seeded_old": 12},
        )
        fresh = ctx.try_add_worker(0)
        assert fresh is not None
        ages = ctx.worker_ages()
        assert ages["seeded_old"] == 12
        assert ages[fresh.id] == 0
        assert len(ages) == 2

    def test_caller_filter_pattern_persists_ages_correctly_across_cycles(self) -> None:
        """End-to-end documented pattern: get ages, drop deleted, increment, re-seed.

        This is the contract any age-tracking caller follows to
        maintain age across autoscale cycles. Two stages so the FGD
        reuse map (built per-stage from ``pending_removes[stage_name]``)
        cannot cancel the ``stage_a`` removal with the ``stage_b`` add;
        the Solution faithfully reflects one delete and one add and the
        caller-side filter has work to do.
        """
        # Cycle 1: cold start, two stages, two seeded workers.
        cluster_1 = resources.ClusterResources(
            nodes={
                "node0": resources.NodeResources(
                    used_cpus=0.0,
                    total_cpus=4.0,
                    gpus=[],
                    name="node0",
                )
            }
        )
        shape_1 = resources.Resources(cpus=1.0, gpus=0.0).to_worker_shape(cluster_1)
        problem_cycle_1 = data_structures.Problem(
            cluster_resources=cluster_1,
            stages=[
                data_structures.ProblemStage(
                    name="stage_a",
                    stage_batch_size=1,
                    worker_shape=shape_1,
                    requested_num_workers=None,
                    over_provision_factor=None,
                ),
                data_structures.ProblemStage(
                    name="stage_b",
                    stage_batch_size=1,
                    worker_shape=shape_1,
                    requested_num_workers=None,
                    over_provision_factor=None,
                ),
            ],
        )
        state_cycle_1 = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[_make_cpu_worker("doomed_in_a")],
                    slots_per_worker=1,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="stage_b",
                    workers=[_make_cpu_worker("survivor_in_b")],
                    slots_per_worker=1,
                    is_finished=False,
                ),
            ]
        )
        ctx1 = data_structures.AutoscalePlanContext.from_problem_state(
            problem_cycle_1,
            state_cycle_1,
            worker_ages={"doomed_in_a": 7, "survivor_in_b": 3},
        )
        assert ctx1.try_remove_worker(0, "doomed_in_a") is True
        added = ctx1.try_add_worker(1)
        assert added is not None
        # The reuse map is per-stage, so adding to stage_b cannot consume
        # stage_a's pending_removes -- the new worker is genuinely fresh.
        assert added.id != "doomed_in_a"

        solution = ctx1.into_solution()

        # Caller-side persistence pattern: drop deleted, increment the rest.
        deleted_ids: set[str] = {w.id for stage in solution.stages for w in stage.deleted_workers}
        assert deleted_ids == {"doomed_in_a"}
        next_cycle_ages = {
            worker_id: age + 1 for worker_id, age in ctx1.worker_ages().items() if worker_id not in deleted_ids
        }
        # survivor_in_b (age 3 + 1 = 4); freshly-added (age 0 + 1 = 1); doomed dropped.
        assert next_cycle_ages == {"survivor_in_b": 4, added.id: 1}

        # Cycle 2: build context with persisted ages and the post-cycle state.
        state_cycle_2 = data_structures.ProblemState(
            stages=[
                data_structures.ProblemStageState(
                    stage_name="stage_a",
                    workers=[],
                    slots_per_worker=1,
                    is_finished=False,
                ),
                data_structures.ProblemStageState(
                    stage_name="stage_b",
                    workers=[
                        _make_cpu_worker("survivor_in_b"),
                        _make_cpu_worker(added.id),
                    ],
                    slots_per_worker=1,
                    is_finished=False,
                ),
            ]
        )
        ctx2 = data_structures.AutoscalePlanContext.from_problem_state(
            problem_cycle_1,
            state_cycle_2,
            worker_ages=next_cycle_ages,
        )
        assert ctx2.worker_age("survivor_in_b") == 4
        assert ctx2.worker_age(added.id) == 1
        assert ctx2.worker_age("doomed_in_a") is None

    def test_worker_ages_remains_readable_after_into_solution_drained(self) -> None:
        """Read accessors stay valid after drain -- caller MUST inspect ages post-drain."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 5},
        )
        added = ctx.try_add_worker(0)
        assert added is not None
        _ = ctx.into_solution()

        # The drained-state guard blocks try_add / try_remove but read
        # accessors must keep working: the documented caller pattern
        # reads ``worker_ages()`` AFTER ``into_solution()`` to filter
        # against ``solution.deleted_workers`` for the next cycle.
        ages = ctx.worker_ages()
        assert ages["w0"] == 5
        assert ages[added.id] == 0
        assert ctx.worker_age("w0") == 5

    def test_independent_contexts_have_independent_age_maps(self) -> None:
        """Two contexts built from the same input do not share age state."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx_a = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 3},
        )
        ctx_b = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 3},
        )

        # Mutate ctx_a: remove the seeded worker (stage-for-removal keeps
        # the age entry, but pending_removes gets the worker so a later
        # in-stage add reuse path could cancel it). The point is to
        # observe that NONE of these mutations bleed across to ctx_b's
        # independent age map.
        assert ctx_a.try_remove_worker(0, "w0") is True

        # ctx_b sees its original snapshot unchanged regardless of what
        # ctx_a did to its own internal state.
        assert ctx_b.worker_ages() == {"w0": 3}
        assert ctx_b.worker_age("w0") == 3

    def test_worker_ages_returns_a_fresh_dict_on_each_call(self) -> None:
        """Mutating the returned dict does not mutate the context's internal map.

        The Rust implementation clones the underlying ``HashMap`` on every
        call, but the contract is documented at the Python boundary;
        pin it so a future refactor (e.g. exposing a Rust reference)
        cannot silently introduce shared mutable state.
        """
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 4},
        )
        snapshot_a = ctx.worker_ages()
        snapshot_a["w0"] = 999
        snapshot_a["bogus"] = 12345
        snapshot_b = ctx.worker_ages()
        assert snapshot_b == {"w0": 4}
        assert ctx.worker_age("w0") == 4

    def test_reuse_then_drain_carries_age_into_next_cycle(self) -> None:
        """End-to-end: stage-for-removal + reuse must carry the original age into the next cycle.

        The most interesting cross-cycle invariant for any age-aware
        donor selector: a stage-for-removal followed by reuse that
        incorrectly reset to age 0 would only surface in the donor
        selector's behaviour, not in any individual ``worker_age()``
        lookup unless the test specifically chains the operations.
        """
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        ctx1 = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages={"w0": 6},
        )
        assert ctx1.try_remove_worker(0, "w0") is True
        reused = ctx1.try_add_worker(0)
        assert reused is not None
        assert reused.id == "w0", "FGD reuse must restore the same worker id"
        assert ctx1.worker_age("w0") == 6, "reuse path preserves original age"

        solution = ctx1.into_solution()
        deleted_ids: set[str] = {w.id for stage in solution.stages for w in stage.deleted_workers}
        assert deleted_ids == set(), "reuse cancels the staged remove"

        next_cycle_ages = {
            worker_id: age + 1 for worker_id, age in ctx1.worker_ages().items() if worker_id not in deleted_ids
        }
        ctx2 = data_structures.AutoscalePlanContext.from_problem_state(
            problem,
            state,
            worker_ages=next_cycle_ages,
        )
        assert ctx2.worker_age("w0") == 7

    def test_negative_seed_value_raises_overflow_error(self) -> None:
        """Negative ages cannot fit in u64; pyo3 raises OverflowError."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        with pytest.raises(OverflowError):
            data_structures.AutoscalePlanContext.from_problem_state(
                problem,
                state,
                worker_ages={"w0": -1},
            )

    def test_seed_value_above_u64_max_raises_overflow_error(self) -> None:
        """Values exceeding u64::MAX cannot round-trip through the FFI."""
        problem, state = _build_one_stage_problem_and_state(
            workers=[_make_cpu_worker("w0")],
        )
        with pytest.raises(OverflowError):
            data_structures.AutoscalePlanContext.from_problem_state(
                problem,
                state,
                worker_ages={"w0": 2**64},
            )
