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
  3. Keeps the ``try_add_worker`` / ``try_remove_worker`` / ``into_solution``
     methods stubbed (they currently raise ``NotImplementedError``).
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources


def _make_cpu_worker(worker_id: str, node: str = "node0") -> data_structures.ProblemWorkerGroupState:
    """Build a one-allocation CPU worker snapshot."""
    worker_resources = resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=[])
    return data_structures.ProblemWorkerGroupState.make(worker_id, [worker_resources])


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


class TestAutoscalePlanContextStubs:
    """Stubbed methods raise ``NotImplementedError``.

    Pinning the stub behaviour keeps the contract honest: callers cannot
    accidentally rely on a default no-op output before the real
    implementation lands.
    """

    def test_try_add_worker_is_stubbed(self) -> None:
        """``try_add_worker`` is currently unimplemented."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(NotImplementedError):
            ctx.try_add_worker(0)

    def test_try_remove_worker_is_stubbed(self) -> None:
        """``try_remove_worker`` is currently unimplemented."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(NotImplementedError):
            ctx.try_remove_worker(0, "fake_id")

    def test_into_solution_is_stubbed(self) -> None:
        """``into_solution`` is currently unimplemented."""
        problem, state = _build_one_stage_problem_and_state()
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(NotImplementedError):
            ctx.into_solution()


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
