# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-worker ``num_used_slots`` field on ``ProblemWorkerGroupState``.

The field is the per-worker saturation signal Phase D scale-down reads
to prefer idle workers over busy ones. The Python wrapper exposes it
as a keyword-only kwarg defaulting to 0 so existing call sites that
constructed ``ProblemWorkerGroupState.make(id, allocations)`` continue
to compile unchanged.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources


def _allocs() -> list[resources.WorkerResourcesInternal]:
    """Build a single 1-CPU allocation for fixture brevity."""
    return [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])]


def _problem_with_worker(
    worker: data_structures.ProblemWorkerGroupState,
) -> tuple[data_structures.Problem, data_structures.ProblemState]:
    """Build a one-stage problem seeded with ``worker``."""
    cluster = resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0.0, total_cpus=4.0, gpus=[], name="node-0"),
        },
    )
    shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    problem = data_structures.Problem(
        cluster_resources=cluster,
        stages=[
            data_structures.ProblemStage(
                name="stage",
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
                stage_name="stage",
                workers=[worker],
                slots_per_worker=1,
                is_finished=False,
            )
        ],
    )
    return problem, state


class TestNumUsedSlotsField:
    """Pin the new per-worker ``num_used_slots`` field's contract."""

    def test_defaults_to_zero_when_omitted(self) -> None:
        """Backward-compat: the existing two-positional ``make`` call defaults to 0."""
        worker = data_structures.ProblemWorkerGroupState.make("w0", _allocs())

        assert worker.num_used_slots == 0

    def test_round_trips_explicit_value(self) -> None:
        """An explicit kwarg value round-trips through the Rust struct."""
        worker = data_structures.ProblemWorkerGroupState.make("w0", _allocs(), num_used_slots=7)

        assert worker.num_used_slots == 7

    def test_zero_value_round_trips(self) -> None:
        """Boundary: explicit zero is preserved (not silently coerced to a sentinel)."""
        worker = data_structures.ProblemWorkerGroupState.make("w0", _allocs(), num_used_slots=0)

        assert worker.num_used_slots == 0

    def test_large_value_round_trips(self) -> None:
        """A large value (``c=64`` slot count) round-trips intact."""
        worker = data_structures.ProblemWorkerGroupState.make("w0", _allocs(), num_used_slots=64)

        assert worker.num_used_slots == 64

    def test_negative_value_overflows_at_rust_boundary(self) -> None:
        """Rust ``usize`` cast rejects negative values rather than wrapping."""
        with pytest.raises(OverflowError):
            data_structures.ProblemWorkerGroupState.make("w0", _allocs(), num_used_slots=-1)

    def test_field_is_keyword_only(self) -> None:
        """The new field cannot be passed positionally (PEP 3102 enforcement)."""
        with pytest.raises(TypeError):
            data_structures.ProblemWorkerGroupState.make("w0", _allocs(), 5)  # type: ignore[misc]

    def test_other_fields_untouched_when_setting_used_slots(self) -> None:
        """Setting the new field does not disturb the existing ``id`` / ``resources``."""
        worker = data_structures.ProblemWorkerGroupState.make("ingest-w3", _allocs(), num_used_slots=4)

        assert worker.id == "ingest-w3"
        assert len(worker.resources) == 1

    def test_serialize_deserialize_preserves_used_slots(self) -> None:
        """Rust serialization preserves the per-worker idle signal."""
        worker = data_structures.ProblemWorkerGroupState.make("w0", _allocs(), num_used_slots=5)
        serialized = worker.rust.serialize()

        restored = data_structures.ProblemWorkerGroupState(type(worker.rust).deserialize(serialized))

        assert restored.id == "w0"
        assert restored.num_used_slots == 5

    def test_planner_delete_preserves_used_slots_in_solution(self) -> None:
        """A planned delete carries the source worker's used-slot signal."""
        worker = data_structures.ProblemWorkerGroupState.make("busy-w0", _allocs(), num_used_slots=3)
        problem, state = _problem_with_worker(worker)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, state)

        assert ctx.try_remove_worker(0, "busy-w0") is True

        solution = ctx.into_solution()
        deleted = solution.stages[0].deleted_workers
        assert [(worker.id, worker.num_used_slots) for worker in deleted] == [("busy-w0", 3)]
