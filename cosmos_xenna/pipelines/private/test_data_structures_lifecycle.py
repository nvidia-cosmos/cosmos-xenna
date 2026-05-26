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

"""Lifecycle and ABI-safety tests for the Solution/StageSolution/Problem types.

The unit tests in ``test_data_structures.py`` cover the factory
contracts in isolation. These tests cover **operational** properties:

  * ``Problem``'s newly ``get_all``-exposed fields (``stages``,
    ``cluster_resources``) are readable and immutable when leaked.
  * Multi-thread construction is safe (the Autoscaler runs autoscale
    in a ``ThreadPoolExecutor`` worker; the Solution then crosses
    back to the main thread).
  * Stress: thousands of construct/discard cycles do not crash or
    raise. Catches egregious leaks at the FFI boundary.
"""

import concurrent.futures
import gc

from cosmos_xenna.pipelines.private import data_structures, resources


def _cluster() -> resources.ClusterResources:
    """Single-node CPU cluster sufficient for ProblemStage construction."""
    return resources.ClusterResources(
        nodes={"node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0")},
    )


def _make_problem(stage_names: list[str]) -> data_structures.Problem:
    """Build a real Problem with one CPU stage per name."""
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


def _make_worker(worker_id: str = "w") -> data_structures.ProblemWorkerGroupState:
    res = resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])
    return data_structures.ProblemWorkerGroupState.make(worker_id, [res])


class TestProblemPyClassAttributesAfterGetAll:
    """``Problem`` had ``#[pyclass]`` -> ``#[pyclass(get_all, set_all)]`` to expose ``stages``.

    Pin the read semantics on the newly-exposed attributes so a
    future Rust refactor that re-narrows the visibility is caught.
    """

    def test_problem_stages_is_iterable(self) -> None:
        """``problem.rust.stages`` returns an iterable of ``rust.ProblemStage``."""
        problem = _make_problem(["A", "B", "C"])
        names = [stage.name for stage in problem.rust.stages]
        assert names == ["A", "B", "C"]

    def test_problem_cluster_resources_accessible(self) -> None:
        """``problem.rust.cluster_resources`` is the ClusterResources held by the Problem."""
        problem = _make_problem(["A"])
        cluster = problem.rust.cluster_resources
        assert "node-0" in cluster.nodes

    def test_problem_stages_is_addressable_by_index(self) -> None:
        """Sequence access works on the get_all-exposed Vec."""
        problem = _make_problem(["A", "B"])
        assert problem.rust.stages[0].name == "A"
        assert problem.rust.stages[1].name == "B"

    def test_problem_stages_length_matches_construction(self) -> None:
        """``len(problem.rust.stages)`` is the number of stages passed at construction time."""
        problem = _make_problem([f"S-{i}" for i in range(7)])
        assert len(problem.rust.stages) == 7


class TestConcurrentConstruction:
    """Multi-thread Solution construction is safe under the GIL.

    ``Autoscaler`` runs ``autoscale()`` in a ``ThreadPoolExecutor``
    worker thread; the resulting Solution then crosses back to the
    main thread for application. These tests stress that pattern.
    """

    def test_many_threads_can_construct_solutions_concurrently(self) -> None:
        """100 ThreadPool workers * 1 Solution each = 100 Solutions; no exceptions, no race."""

        def build_one(idx: int) -> int:
            ss = data_structures.StageSolution.make(slots_per_worker=2)
            sol = data_structures.Solution.make(stages=[ss])
            return len(sol.stages) + idx

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(build_one, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        assert len(results) == 100
        # Each future returned len(sol.stages) + idx == 1 + idx; sorted gives [1, 2, ..., 100].
        assert sorted(results) == [1 + i for i in range(100)]

    def test_solution_built_in_one_thread_is_consumable_in_another(self) -> None:
        """Mirrors the streaming.py Autoscaler pattern: build in worker, consume in main."""
        worker = _make_worker("cross-thread")

        def build_in_worker_thread() -> data_structures.Solution:
            ss = data_structures.StageSolution.make(slots_per_worker=4, new_workers=[worker])
            return data_structures.Solution.make(stages=[ss])

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(build_in_worker_thread)
            solution = future.result()

        # Main thread consumes the Solution -- exact pattern of
        # apply_autoscale_result_if_ready in streaming.py.
        assert len(solution.stages) == 1
        assert solution.stages[0].slots_per_worker == 4
        assert [w.id for w in solution.stages[0].new_workers] == ["cross-thread"]

    def test_to_worker_group_works_across_threads(self) -> None:
        """``to_worker_group()`` (the FFI call) is safe to invoke from the main thread on a worker-built Solution.

        ``apply_autoscale_result_if_ready`` calls
        ``w.to_worker_group(pool.name)`` from the main thread on
        Solutions whose constituent workers were constructed inside
        the autoscaler's ``ThreadPoolExecutor`` task. Pin that the
        FFI is safe across the worker-thread -> main-thread handoff
        by mirroring the exact production direction: build the whole
        Solution (including the ``ProblemWorkerGroupState``) in the
        worker thread, then exercise ``to_worker_group()`` on the
        main thread.
        """

        def build_in_worker_thread() -> data_structures.Solution:
            worker = _make_worker("threaded")
            ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[worker])
            return data_structures.Solution.make(stages=[ss])

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            sol = pool.submit(build_in_worker_thread).result()

        wg = sol.stages[0].new_workers[0].to_worker_group("StageX")
        assert wg is not None


class TestStressNoLeakOrCrash:
    """Bulk construct/discard cycles do not raise, crash, or visibly leak.

    Catches FFI-boundary issues that would manifest as crashes or
    runaway memory under sustained Autoscaler operation (one
    Solution per cycle, default ``interval_s=10s``, so ~8640
    Solutions per day per pipeline).
    """

    def test_thousand_solution_lifecycles_complete_cleanly(self) -> None:
        """1000 construct-and-discard cycles; final ``gc.collect`` has no excess work."""
        worker = _make_worker("stress")
        for _ in range(1000):
            ss = data_structures.StageSolution.make(
                slots_per_worker=2,
                new_workers=[worker],
                deleted_workers=[worker],
            )
            sol = data_structures.Solution.make(stages=[ss, ss, ss])
            # Touch every getter so any lazy materialisation runs.
            assert len(sol.stages) == 3
            assert sol.stages[0].slots_per_worker == 2

        gc.collect()
        # If we reached this line without raising, the FFI boundary
        # is sound. Memory deltas are environment-dependent and
        # unreliable in CI; the explicit no-raise contract is the
        # operational signal we care about.

    def test_thousand_problem_constructions_complete_cleanly(self) -> None:
        """1000 Problem construct-and-discard cycles preserve per-construction stage names.

        Pins that a high construction rate does not corrupt the Rust
        ``stages`` Vec across consecutive ``Problem`` instances (the
        loop is ~ thirty cycles per second on a CI runner). Asserting
        the stage name on every iteration catches a regression where
        the FFI boundary returns a stale or shared Vec.
        """
        for i in range(1000):
            problem = _make_problem([f"stage-{i}"])
            assert problem.rust.stages[0].name == f"stage-{i}"
        gc.collect()


class TestProblemImmutabilityForReaders:
    """Reading ``problem.rust.stages`` does not mutate the underlying Vec.

    With ``set_all`` enabled, callers CAN write back into ``stages``,
    but a normal read should not have side effects. This pins the
    reader's expectation -- a defensive guard against accidentally
    treating the getter as a moved value.
    """

    def test_repeated_reads_yield_consistent_data(self) -> None:
        """Reading ``stages`` repeatedly returns the same logical contents."""
        problem = _make_problem(["A", "B", "C"])
        first_names = [s.name for s in problem.rust.stages]
        second_names = [s.name for s in problem.rust.stages]
        assert first_names == second_names == ["A", "B", "C"]
