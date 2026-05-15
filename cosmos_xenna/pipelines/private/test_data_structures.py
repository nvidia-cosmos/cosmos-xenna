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

"""Tests for the Python factory methods on Solution and StageSolution.

Pure-Python schedulers cannot produce a Rust ``Solution`` directly
because pyo3 hides the Rust constructors. ``Solution.make`` and
``StageSolution.make`` solve this by delegating to the Rust ``#[new]``
constructors and populating the fields via the ``set_all`` setters,
returning a fully wrapped Python object indistinguishable from one
produced by the legacy Rust autoscaler.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.resources import GpuAllocationInternal, WorkerResourcesInternal


def _make_worker(worker_id: str = "w-1") -> data_structures.ProblemWorkerGroupState:
    """Build a single CPU-only worker for use as a Solution input."""
    resources = WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])
    return data_structures.ProblemWorkerGroupState.make(worker_id, [resources])


def _make_worker_with_gpu(worker_id: str = "w-gpu") -> data_structures.ProblemWorkerGroupState:
    """Build a worker that owns a fractional GPU; verifies the resources field round-trips."""
    resources = WorkerResourcesInternal(
        node="node-0",
        cpus=2.0,
        gpus=[GpuAllocationInternal(offset=0, used_fraction=0.5)],
    )
    return data_structures.ProblemWorkerGroupState.make(worker_id, [resources])


class TestStageSolutionFactory:
    """``StageSolution.make`` produces a fully-formed Python wrapper."""

    def test_empty_stage_solution(self) -> None:
        """No workers in or out -- the most common shape (most stages quiet on most cycles)."""
        ss = data_structures.StageSolution.make(slots_per_worker=2)
        assert ss.slots_per_worker == 2
        assert ss.new_workers == []
        assert ss.deleted_workers == []

    def test_with_new_workers_only(self) -> None:
        """Scale-up case: workers in new_workers, deleted_workers empty."""
        w1 = _make_worker("w-1")
        w2 = _make_worker("w-2")
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[w1, w2])
        ids = [w.id for w in ss.new_workers]
        assert ids == ["w-1", "w-2"]
        assert ss.deleted_workers == []

    def test_with_deleted_workers_only(self) -> None:
        """Scale-down case: workers in deleted_workers, new_workers empty."""
        w = _make_worker("victim")
        ss = data_structures.StageSolution.make(slots_per_worker=4, deleted_workers=[w])
        assert ss.new_workers == []
        assert [w.id for w in ss.deleted_workers] == ["victim"]
        assert ss.slots_per_worker == 4

    def test_with_both_new_and_deleted(self) -> None:
        """Cycle that simultaneously rotates a worker -- delete one, add one."""
        new_w = _make_worker("new")
        old_w = _make_worker("old")
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[new_w], deleted_workers=[old_w])
        assert [w.id for w in ss.new_workers] == ["new"]
        assert [w.id for w in ss.deleted_workers] == ["old"]

    def test_worker_resources_round_trip_through_rust(self) -> None:
        """Resources field on a Python-constructed worker survives the round trip into Rust and back."""
        worker = _make_worker_with_gpu("gpu-w")
        ss = data_structures.StageSolution.make(slots_per_worker=1, new_workers=[worker])
        retrieved = ss.new_workers[0]
        assert retrieved.id == "gpu-w"
        assert retrieved.resources[0].node == "node-0"
        assert retrieved.resources[0].cpus == 2.0
        assert retrieved.resources[0].gpus[0].offset == 0
        assert retrieved.resources[0].gpus[0].used_fraction == 0.5

    def test_to_worker_group_works_on_python_constructed_worker(self) -> None:
        """``to_worker_group`` calls into Rust; the helper's job is to verify the Rust path is live."""
        worker = _make_worker("abc")
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[worker])
        wg = ss.new_workers[0].to_worker_group("MyStage")
        # WorkerGroup is itself a Rust-wrapped type; we only verify it constructs successfully.
        assert wg is not None

    def test_explicit_empty_lists_match_default_omission(self) -> None:
        """``make(slots, new_workers=[], deleted_workers=[])`` is equivalent to omitting them.

        Pins the truthy-check behaviour so a future refactor that
        switches to ``is None`` (which would treat ``[]`` differently)
        is caught immediately.
        """
        explicit = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[], deleted_workers=[])
        default = data_structures.StageSolution.make(slots_per_worker=2)
        assert explicit.new_workers == default.new_workers == []
        assert explicit.deleted_workers == default.deleted_workers == []
        assert explicit.slots_per_worker == default.slots_per_worker == 2

    def test_slots_per_worker_round_trips_for_boundary_values(self) -> None:
        """Verify the ``usize`` round-trip preserves slot counts at minimum and large values."""
        for slots in (1, 8, 128, 65535):
            ss = data_structures.StageSolution.make(slots_per_worker=slots)
            assert ss.slots_per_worker == slots

    def test_negative_slots_per_worker_is_rejected(self) -> None:
        """pyo3 raises ``OverflowError`` when a negative int reaches a Rust ``usize`` field.

        Defensive pin: the Rust binding rejects negative slot counts at
        the FFI boundary, so callers get a precise error rather than
        silent wraparound.
        """
        with pytest.raises(OverflowError):
            data_structures.StageSolution.make(slots_per_worker=-1)

    def test_many_workers_round_trip_in_order(self) -> None:
        """A long ``new_workers`` list round-trips with the right cardinality and order."""
        workers = [_make_worker(f"w-{i}") for i in range(50)]
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=workers)
        ids = [w.id for w in ss.new_workers]
        assert ids == [f"w-{i}" for i in range(50)]

    def test_worker_with_multiple_resource_entries(self) -> None:
        """A single worker holding multiple ``WorkerResourcesInternal`` entries round-trips intact.

        Multi-entry workers exist in real pipelines (e.g., a single
        worker allocated multiple GPU slices on the same node).
        """
        resources_a = WorkerResourcesInternal(
            node="node-0",
            cpus=1.0,
            gpus=[GpuAllocationInternal(offset=0, used_fraction=0.5)],
        )
        resources_b = WorkerResourcesInternal(
            node="node-0",
            cpus=1.0,
            gpus=[GpuAllocationInternal(offset=1, used_fraction=0.5)],
        )
        worker = data_structures.ProblemWorkerGroupState.make("multi-gpu", [resources_a, resources_b])
        ss = data_structures.StageSolution.make(slots_per_worker=1, new_workers=[worker])
        retrieved = ss.new_workers[0]
        assert len(retrieved.resources) == 2
        assert retrieved.resources[0].gpus[0].offset == 0
        assert retrieved.resources[1].gpus[0].offset == 1

    def test_mutating_input_list_after_make_does_not_affect_solution(self) -> None:
        """The factory does not share list references with the caller.

        Pins the contract that ``StageSolution.make(new_workers=lst)``
        snapshots the list at call time; subsequent mutations of
        ``lst`` cannot reach into the Rust-side storage.
        """
        original = [_make_worker("first")]
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=original)

        # Mutate the caller's list AFTER make returns.
        original.append(_make_worker("appended-after"))
        original.clear()

        # The Solution still sees only the original snapshot.
        assert [w.id for w in ss.new_workers] == ["first"]


class TestSolutionFactory:
    """``Solution.make`` produces a fully-formed Python wrapper holding a list of stages."""

    def test_empty_solution(self) -> None:
        """A pipeline with zero stages produces an empty Solution."""
        sol = data_structures.Solution.make()
        assert sol.stages == []

    def test_single_stage_solution(self) -> None:
        """Single-stage pipeline -- the most basic non-trivial case."""
        ss = data_structures.StageSolution.make(slots_per_worker=2)
        sol = data_structures.Solution.make(stages=[ss])
        assert len(sol.stages) == 1
        assert sol.stages[0].slots_per_worker == 2

    def test_multi_stage_solution_preserves_order(self) -> None:
        """Stage order in the input list is preserved in ``Solution.stages``.

        Order matters: streaming.py iterates stages in lockstep with
        the pipeline's pools via ``zip(autoscale_result.stages, pools)``.
        """
        ss_a = data_structures.StageSolution.make(slots_per_worker=1)
        ss_b = data_structures.StageSolution.make(slots_per_worker=2)
        ss_c = data_structures.StageSolution.make(slots_per_worker=3)
        sol = data_structures.Solution.make(stages=[ss_a, ss_b, ss_c])
        slots = [s.slots_per_worker for s in sol.stages]
        assert slots == [1, 2, 3]

    def test_solution_carries_per_stage_workers(self) -> None:
        """End-to-end: workers placed on a stage flow through Solution.make to Solution.stages."""
        worker = _make_worker("w")
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[worker])
        sol = data_structures.Solution.make(stages=[ss])
        retrieved_workers = sol.stages[0].new_workers
        assert [w.id for w in retrieved_workers] == ["w"]

    def test_explicit_empty_stages_list_matches_default_omission(self) -> None:
        """``Solution.make(stages=[])`` is equivalent to ``Solution.make()``.

        Pins the truthy-check on ``stages`` so a refactor switching to
        ``is None`` (which would treat ``[]`` differently) is caught.
        """
        explicit = data_structures.Solution.make(stages=[])
        default = data_structures.Solution.make()
        assert explicit.stages == default.stages == []

    def test_mixed_empty_and_populated_stages(self) -> None:
        """A Solution with [empty, populated, empty] preserves both order and per-stage contents."""
        worker = _make_worker("only")
        ss_empty_a = data_structures.StageSolution.make(slots_per_worker=1)
        ss_populated = data_structures.StageSolution.make(slots_per_worker=4, new_workers=[worker])
        ss_empty_b = data_structures.StageSolution.make(slots_per_worker=2)
        sol = data_structures.Solution.make(stages=[ss_empty_a, ss_populated, ss_empty_b])

        assert [s.slots_per_worker for s in sol.stages] == [1, 4, 2]
        assert sol.stages[0].new_workers == []
        assert [w.id for w in sol.stages[1].new_workers] == ["only"]
        assert sol.stages[2].new_workers == []

    def test_mutating_input_stage_list_after_make_does_not_affect_solution(self) -> None:
        """``Solution.make(stages=lst)`` snapshots the list at call time."""
        ss_a = data_structures.StageSolution.make(slots_per_worker=1)
        original = [ss_a]
        sol = data_structures.Solution.make(stages=original)

        # Mutate the caller's list AFTER make returns.
        ss_b = data_structures.StageSolution.make(slots_per_worker=2)
        original.append(ss_b)
        original.clear()

        # The Solution still sees only the original snapshot.
        assert [s.slots_per_worker for s in sol.stages] == [1]


class TestSolutionInteropWithRustPath:
    """The factory result is interchangeable with what the Rust autoscaler produces."""

    def test_python_constructed_solution_exposes_rust_property(self) -> None:
        """``solution.rust`` returns the underlying Rust object even when built via the factory."""
        ss = data_structures.StageSolution.make(slots_per_worker=2)
        sol = data_structures.Solution.make(stages=[ss])
        # The Rust object must be present so consumers that need it
        # (e.g. ``run_fragmentation_autoscaler``) keep working.
        assert sol.rust is not None
        assert ss.rust is not None

    def test_helper_aggregates_match_python_view(self) -> None:
        """``num_new_workers_per_stage`` (Rust helper) sees the same data as the Python wrapper."""
        w1 = _make_worker("a")
        w2 = _make_worker("b")
        ss = data_structures.StageSolution.make(slots_per_worker=2, new_workers=[w1, w2])
        sol = data_structures.Solution.make(stages=[ss])
        # Calling through .rust exercises the Rust-side aggregator.
        assert sol.rust.num_new_workers_per_stage() == [2]
        assert sol.rust.num_deleted_workers_per_stage() == [0]
