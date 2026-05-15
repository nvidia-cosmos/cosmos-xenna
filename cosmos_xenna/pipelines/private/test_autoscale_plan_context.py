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

"""Tests for the ``AutoscalePlanContext`` Python wrapper (granular sub-iteration 1a-i).

Pins the construction contract: ``from_problem_state`` accepts a
``Problem`` + ``ProblemState`` and returns a non-None wrapper that:

  1. Reports the right number of stages.
  2. Exposes the underlying Rust object via ``.rust``.
  3. Keeps the ``try_add_worker`` / ``try_remove_worker`` / ``into_solution``
     methods stubbed (they raise ``NotImplementedError`` until 1a-ii / 1a-iii /
     1a-iv land); reaching them via the Rust object documents the staging.

Subsequent sub-iterations (1a-ii .. 1a-iv) replace the stubs and extend
this file with behaviour tests.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources


def _make_problem_with_one_cpu_stage() -> data_structures.Problem:
    """Build a minimal Problem with a single CPU stage on a 2-node cluster."""
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
    stage = data_structures.ProblemStage(
        name="stage_a",
        stage_batch_size=1,
        worker_shape=resources.CpuOnly(num_cpus=1.0),
        requested_num_workers=None,
        over_provision_factor=None,
    )
    return data_structures.Problem(cluster_resources=cluster, stages=[stage])


def _make_empty_state(problem: data_structures.Problem) -> data_structures.ProblemState:
    """Build a default ProblemState with one empty stage matching the problem shape."""
    stage_states = [
        data_structures.ProblemStageState(
            stage_name=s.rust.name,
            workers=[],
            slots_per_worker=1,
            is_finished=False,
        )
        for s in problem.rust.stages
    ]
    return data_structures.ProblemState(stages=stage_states)


class TestAutoscalePlanContextConstruction:
    """``AutoscalePlanContext.from_problem_state`` returns a populated wrapper."""

    def test_returns_a_non_none_wrapper(self) -> None:
        """Smoke test from the plan: from_problem_state must not return None."""
        problem = _make_problem_with_one_cpu_stage()
        state = _make_empty_state(problem)

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx is not None

    def test_exposes_rust_property(self) -> None:
        """The ``.rust`` accessor returns the underlying Rust object.

        Catches refactor regressions where the wrapper hides the Rust
        handle (subsequent sub-iterations need it for the FGD calls).
        """
        problem = _make_problem_with_one_cpu_stage()
        state = _make_empty_state(problem)

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.rust is not None

    def test_num_stages_matches_input(self) -> None:
        """The context tracks one stage per pipeline stage."""
        problem = _make_problem_with_one_cpu_stage()
        state = _make_empty_state(problem)

        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.num_stages() == 1


class TestAutoscalePlanContextStubs:
    """Methods that land in 1a-ii / 1a-iii / 1a-iv currently raise NotImplementedError.

    Pinning the stub behaviour keeps the contract honest: callers cannot
    accidentally rely on a default no-op output before the real
    implementation lands.
    """

    def test_try_add_worker_is_stubbed(self) -> None:
        """``try_add_worker`` raises until 1a-ii lands."""
        problem = _make_problem_with_one_cpu_stage()
        state = _make_empty_state(problem)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(NotImplementedError):
            ctx.rust.try_add_worker(0)

    def test_try_remove_worker_is_stubbed(self) -> None:
        """``try_remove_worker`` raises until 1a-iii lands."""
        problem = _make_problem_with_one_cpu_stage()
        state = _make_empty_state(problem)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(NotImplementedError):
            ctx.rust.try_remove_worker(0, "fake_id")

    def test_into_solution_is_stubbed(self) -> None:
        """``into_solution`` raises until 1a-iv lands."""
        problem = _make_problem_with_one_cpu_stage()
        state = _make_empty_state(problem)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        with pytest.raises(NotImplementedError):
            ctx.rust.into_solution()


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
        stage_a = data_structures.ProblemStage(
            name="stage_a",
            stage_batch_size=1,
            worker_shape=resources.CpuOnly(num_cpus=1.0),
            requested_num_workers=None,
            over_provision_factor=None,
        )
        stage_b = data_structures.ProblemStage(
            name="stage_b",
            stage_batch_size=1,
            worker_shape=resources.CpuOnly(num_cpus=1.0),
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
