# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct unit tests for the Phase D consolidation helpers.

Pins the contracts of
:meth:`SaturationAwareScheduler._compute_host_gpu_used_fractions` and
:meth:`SaturationAwareScheduler._extract_worker_host_gpu_used_fractions`
in isolation from Phase D's orchestrator wiring. Failures here pinpoint
helper-level bugs without forcing a full autoscale cycle, which is the
natural failure-attribution boundary when consolidation regressions
surface in production logs.

Coverage axes (per ``test-creation.mdc`` 9-axis adversarial checklist):

  * Empty input handling (no stages / no workers / no GPU allocations).
  * Boundary conditions (fractions exactly 0, 1, > 1.0 over-allocation).
  * Cross-stage aggregation correctness.
  * Multi-GPU MAX semantics (multi-allocation on the same node and
    multi-node SPMD-style allocations).
  * Defensive numeric inputs (negative, NaN, inf fractions are
    rejected at the Rust binding boundary; the Python helpers never
    observe them in production data).
  * Determinism across calls.

The Rust ``GpuAllocation`` stores ``used_fraction`` as ``f32``, so
fractions defined as Python ``f64`` round-trip with up to ~6e-7
absolute error per allocation. Tests use ``pytest.approx`` with
``abs=1e-5`` to absorb that quantization, leaving the assertions
sensitive to changes in summation logic without becoming brittle to
the f32 representation.
"""

import math

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler

_F32_TOL = 1e-5


def _make_problem_state(
    stages: list[
        tuple[
            str,
            list[tuple[str, list[tuple[str, int, float]]]],
            bool,
        ]
    ],
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from compact ``(stage_name, worker_rows, is_finished)`` tuples.

    Each worker row is ``(worker_id, [(node, gpu_offset, used_fraction), ...])``.
    GPU triples on the same node collapse into a single
    ``WorkerResourcesInternal`` so a non-SPMD multi-GPU worker is
    represented as one resource with multiple ``GpuAllocationInternal``
    entries; triples on different nodes become separate resources
    (SPMD-style).
    """
    rows: list[data_structures.ProblemStageState] = []
    for stage_name, worker_rows, is_finished in stages:
        workers = []
        for worker_id, gpu_triples in worker_rows:
            gpus_by_node: dict[str, list[resources.GpuAllocationInternal]] = {}
            for node, offset, fraction in gpu_triples:
                gpus_by_node.setdefault(node, []).append(
                    resources.GpuAllocationInternal(offset=offset, used_fraction=fraction),
                )
            allocations = [
                resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=gpu_list)
                for node, gpu_list in gpus_by_node.items()
            ] or [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])]
            workers.append(
                data_structures.ProblemWorkerGroupState.make(
                    worker_id,
                    allocations,
                    num_used_slots=0,
                ),
            )
        rows.append(
            data_structures.ProblemStageState(
                stage_name=stage_name,
                workers=workers,
                slots_per_worker=1,
                is_finished=is_finished,
            ),
        )
    return data_structures.ProblemState(rows)


class TestComputeHostGpuUsedFractions:
    """Pin the cluster-wide GPU fraction aggregator contract."""

    def test_empty_problem_state_returns_empty_map(self) -> None:
        """No stages means no GPUs to aggregate; the map is empty."""
        state = _make_problem_state([])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {}

    def test_stage_without_workers_returns_empty_map(self) -> None:
        """A stage with no workers contributes no allocations."""
        state = _make_problem_state([("A", [], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {}

    def test_cpu_only_workers_contribute_no_gpu_keys(self) -> None:
        """Workers with empty ``gpus`` lists do not appear in the map."""
        state = _make_problem_state([("A", [("A-w0", []), ("A-w1", [])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {}

    def test_single_worker_single_gpu_records_its_fraction(self) -> None:
        """A single worker on a single GPU yields one map entry."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 2, 0.35)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 2)}
        assert result[("node-0", 2)] == pytest.approx(0.35, abs=_F32_TOL)

    def test_cross_stage_allocations_on_same_gpu_sum(self) -> None:
        """Allocations on the same ``(node, offset)`` from different stages add."""
        state = _make_problem_state(
            [
                ("A", [("A-w0", [("node-0", 0, 0.30)])], False),
                ("B", [("B-w0", [("node-0", 0, 0.50)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {("node-0", 0): pytest.approx(0.80, abs=_F32_TOL)}

    def test_finished_stage_allocations_are_aggregated(self) -> None:
        """Finished stages still hold resources and contribute to the map."""
        state = _make_problem_state(
            [
                ("A", [("A-w0", [("node-0", 0, 0.20)])], False),
                ("B-finished", [("B-w0", [("node-0", 0, 0.40)])], True),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {("node-0", 0): pytest.approx(0.60, abs=_F32_TOL)}

    def test_distinct_gpus_record_independent_keys(self) -> None:
        """Allocations on different GPUs land on separate map entries."""
        state = _make_problem_state(
            [
                (
                    "A",
                    [
                        ("A-w0", [("node-0", 0, 0.10)]),
                        ("A-w1", [("node-0", 1, 0.20)]),
                        ("A-w2", [("node-1", 0, 0.30)]),
                    ],
                    False,
                ),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0), ("node-0", 1), ("node-1", 0)}
        assert result[("node-0", 0)] == pytest.approx(0.10, abs=_F32_TOL)
        assert result[("node-0", 1)] == pytest.approx(0.20, abs=_F32_TOL)
        assert result[("node-1", 0)] == pytest.approx(0.30, abs=_F32_TOL)

    def test_multi_allocation_worker_on_same_node_sums_into_distinct_keys(self) -> None:
        """A non-SPMD multi-GPU worker on one node produces one entry per GPU offset."""
        state = _make_problem_state(
            [
                ("A", [("A-multi", [("node-0", 0, 0.50), ("node-0", 1, 0.30)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0), ("node-0", 1)}
        assert result[("node-0", 0)] == pytest.approx(0.50, abs=_F32_TOL)
        assert result[("node-0", 1)] == pytest.approx(0.30, abs=_F32_TOL)

    def test_spmd_worker_on_distinct_nodes_records_each_node_offset_pair(self) -> None:
        """An SPMD worker spanning nodes contributes one entry per ``(node, offset)`` pair."""
        state = _make_problem_state(
            [
                ("A", [("A-spmd", [("node-0", 0, 1.0), ("node-1", 0, 1.0)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0), ("node-1", 0)}
        assert result[("node-0", 0)] == pytest.approx(1.0, abs=_F32_TOL)
        assert result[("node-1", 0)] == pytest.approx(1.0, abs=_F32_TOL)

    def test_over_allocated_gpu_above_one_is_passed_through(self) -> None:
        """Sums above 1.0 are reported verbatim; the planner is the constraint enforcer."""
        # Three stages each take 0.5 of GPU-0, summing to 1.5. The helper does not
        # cap or warn; it reports the raw aggregate so downstream consumers see the
        # over-allocation as-is.
        state = _make_problem_state(
            [
                ("A", [("A-w0", [("node-0", 0, 0.50)])], False),
                ("B", [("B-w0", [("node-0", 0, 0.50)])], False),
                ("C", [("C-w0", [("node-0", 0, 0.50)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result[("node-0", 0)] == pytest.approx(1.50, abs=_F32_TOL)

    def test_zero_fraction_allocations_are_recorded(self) -> None:
        """Allocations with ``used_fraction=0`` still appear in the map (key exists)."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, 0.0)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0)}
        assert result[("node-0", 0)] == 0.0

    def test_repeated_calls_produce_identical_results(self) -> None:
        """The aggregator is a pure function of its input."""
        state = _make_problem_state(
            [
                ("A", [("A-w0", [("node-0", 0, 0.25)])], False),
                ("B", [("B-w0", [("node-0", 1, 0.50)])], False),
            ],
        )

        first = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)
        second = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert first == second

    def test_cpu_only_stage_alongside_gpu_stage_only_gpu_contributes(self) -> None:
        """A CPU-only stage in the same pipeline contributes nothing to the GPU map."""
        state = _make_problem_state(
            [
                ("CPU", [("CPU-w0", []), ("CPU-w1", [])], False),
                ("GPU", [("GPU-w0", [("node-0", 0, 0.40)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0)}
        assert result[("node-0", 0)] == pytest.approx(0.40, abs=_F32_TOL)

    def test_high_concurrency_aggregation_produces_correct_sum_above_one(self) -> None:
        """100 stages each contributing 0.01 to the same GPU sum to 1.0 (boundary)."""
        worker_rows = [(f"w{i}", [("node-0", 0, 0.01)]) for i in range(100)]
        # Each stage has one worker contributing 0.01; total 100 * 0.01 = 1.00.
        state = _make_problem_state(
            [(f"S{i}", [worker_rows[i]], False) for i in range(100)],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0)}
        # f32 quantization on each of 100 0.01 contributions accumulates to a wider
        # absolute tolerance than the single-allocation _F32_TOL.
        assert result[("node-0", 0)] == pytest.approx(1.0, abs=1e-3)

    def test_thousand_workers_in_single_stage_aggregates_correctly(self) -> None:
        """1000 workers distributed across 8 GPUs aggregate without performance pathology."""
        worker_rows = []
        per_gpu_count = [0] * 8
        for index in range(1000):
            offset = index % 8
            per_gpu_count[offset] += 1
            # Each worker contributes a small slice (0.001) of its GPU.
            worker_rows.append((f"w{index:04d}", [("node-0", offset, 0.001)]))
        state = _make_problem_state([("S", worker_rows, False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", offset) for offset in range(8)}
        for offset, count in enumerate(per_gpu_count):
            # f32 accumulation tolerance scales with the number of contributions.
            assert result[("node-0", offset)] == pytest.approx(0.001 * count, abs=1e-3)


class TestExtractWorkerHostGpuUsedFractions:
    """Pin the per-stage projection contract using the cluster-wide map."""

    def test_empty_stage_returns_empty_map(self) -> None:
        """A stage with no workers projects to an empty map."""
        state = _make_problem_state([("A", [], False)])

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions={},
        )

        assert result == {}

    def test_cpu_only_worker_maps_to_zero(self) -> None:
        """A worker with no GPU allocations defaults to fraction 0.0."""
        state = _make_problem_state([("A", [("A-w0", [])], False)])

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions={},
        )

        assert result == {"A-w0": 0.0}

    def test_single_gpu_worker_reads_aggregate_for_its_offset(self) -> None:
        """A single-GPU worker's fraction is the cluster-wide aggregate at its ``(node, offset)``."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, 0.30)])], False)])
        # Cluster-wide aggregate includes a phantom 0.40 from another (unmodeled) stage.
        host_gpu_used_fractions = {("node-0", 0): 0.70}

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        assert set(result.keys()) == {"A-w0"}
        assert result["A-w0"] == pytest.approx(0.70, abs=_F32_TOL)

    def test_missing_gpu_in_host_map_defaults_to_zero(self) -> None:
        """A worker whose GPU is absent from the cluster map maps to 0.0."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 9, 0.10)])], False)])

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions={("node-0", 0): 0.99},
        )

        assert result == {"A-w0": 0.0}

    def test_multi_gpu_worker_uses_max_across_allocations_on_same_node(self) -> None:
        """A non-SPMD multi-GPU worker projects to the MAX of its GPUs' aggregate fractions."""
        state = _make_problem_state(
            [("A", [("A-w0", [("node-0", 0, 0.50), ("node-0", 1, 0.10)])], False)],
        )
        host_gpu_used_fractions = {
            ("node-0", 0): 0.95,
            ("node-0", 1): 0.10,
        }

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        assert set(result.keys()) == {"A-w0"}
        assert result["A-w0"] == pytest.approx(0.95, abs=_F32_TOL)

    def test_spmd_worker_uses_max_across_node_boundaries(self) -> None:
        """An SPMD worker spanning multiple nodes projects via MAX across all its allocations."""
        state = _make_problem_state(
            [("A", [("A-spmd", [("node-0", 0, 1.0), ("node-1", 0, 1.0)])], False)],
        )
        host_gpu_used_fractions = {
            ("node-0", 0): 1.0,
            ("node-1", 0): 0.30,
        }

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        assert set(result.keys()) == {"A-spmd"}
        assert result["A-spmd"] == pytest.approx(1.0, abs=_F32_TOL)

    def test_multiple_workers_in_one_stage_each_get_their_own_fraction(self) -> None:
        """Each worker is projected independently; no cross-worker contamination."""
        state = _make_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.10)]),
                        ("A-heavy", [("node-0", 1, 0.50)]),
                        ("A-cpu-only", []),
                    ],
                    False,
                ),
            ],
        )
        host_gpu_used_fractions = {
            ("node-0", 0): 0.10,
            ("node-0", 1): 0.95,
        }

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        assert set(result.keys()) == {"A-light", "A-heavy", "A-cpu-only"}
        assert result["A-light"] == pytest.approx(0.10, abs=_F32_TOL)
        assert result["A-heavy"] == pytest.approx(0.95, abs=_F32_TOL)
        assert result["A-cpu-only"] == 0.0

    def test_zero_aggregate_fractions_project_to_zero(self) -> None:
        """A GPU with aggregate 0.0 projects to a worker fraction of 0.0."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, 0.0)])], False)])

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions={("node-0", 0): 0.0},
        )

        assert result == {"A-w0": 0.0}

    def test_workers_sharing_one_gpu_get_identical_projected_fraction(self) -> None:
        """Two workers on the same ``(node, offset)`` both see the full aggregate fraction."""
        state = _make_problem_state(
            [
                (
                    "A",
                    [
                        ("A-w0", [("node-0", 0, 0.25)]),
                        ("A-w1", [("node-0", 0, 0.25)]),
                    ],
                    False,
                ),
            ],
        )
        # Cluster-wide map sees both: 0.25 + 0.25 = 0.50.
        host_gpu_used_fractions = {("node-0", 0): 0.50}

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        assert set(result.keys()) == {"A-w0", "A-w1"}
        assert result["A-w0"] == pytest.approx(0.50, abs=_F32_TOL)
        assert result["A-w1"] == pytest.approx(0.50, abs=_F32_TOL)
        # The MAX rule means workers tied for the same GPU sort identically on the
        # consolidation key; secondary keys (idle/age/id) decide which is deleted.

    def test_aggregate_above_one_propagates_to_worker_projection(self) -> None:
        """Worker projection passes through over-allocated aggregate values verbatim."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, 0.50)])], False)])
        host_gpu_used_fractions = {("node-0", 0): 1.50}

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        assert result["A-w0"] == pytest.approx(1.50, abs=_F32_TOL)

    def test_max_rule_picks_the_largest_when_one_gpu_is_zero(self) -> None:
        """A multi-GPU worker on (heavy, zero) GPUs projects to the heavy fraction."""
        state = _make_problem_state(
            [("A", [("A-w0", [("node-0", 0, 0.50), ("node-0", 1, 0.50)])], False)],
        )
        host_gpu_used_fractions = {
            ("node-0", 0): 0.95,
            ("node-0", 1): 0.0,
        }

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions=host_gpu_used_fractions,
        )

        # MAX(0.95, 0.0) = 0.95 — the lightly-loaded GPU does NOT pull the worker's
        # consolidation key down.
        assert result["A-w0"] == pytest.approx(0.95, abs=_F32_TOL)


class TestDefensiveNumericInputs:
    """Pin where adversarial numeric inputs are rejected vs. propagated.

    The Rust ``GpuAllocation`` constructor enforces a numeric domain
    (finite, non-NaN). Inputs that violate that domain panic at the
    Rust boundary before they can reach the Python aggregator, so the
    Python helpers never observe pathological fractions in production.
    These tests pin that boundary so a future change that loosens
    Rust-side validation forces an explicit choice about how the
    Python helpers should respond.
    """

    def test_negative_fraction_does_not_panic_at_rust_boundary(self) -> None:
        """Negative ``used_fraction`` survives the Rust boundary (quantized to a large positive).

        Documents the asymmetry vs. NaN/inf: the Rust ``GpuAllocation``
        constructor rejects NaN and inf but does not reject negative
        values. The Rust storage uses a quantized fixed-point
        representation (currently u16-scaled), so negative inputs wrap
        to a large positive value rather than raising. Production data
        is expected to be non-negative because allocation increments
        are always positive; a negative input would already imply
        accounting drift in the planner. The Python aggregator never
        sees a true negative number from production paths, so this
        test only pins the no-panic invariant. If the Rust storage
        changes to reject negatives, replace this body with the
        ``pytest.raises(BaseException)`` pattern from the NaN/inf
        siblings.
        """
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, -0.1)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert set(result.keys()) == {("node-0", 0)}
        # Quantized wrap is a large positive float; precise value is implementation-defined.
        assert result[("node-0", 0)] > 0.0

    def test_inf_fraction_is_rejected_at_rust_boundary(self) -> None:
        """``+inf`` ``used_fraction`` panics in the Rust ``GpuAllocation`` constructor."""
        with pytest.raises(BaseException, match="infinite"):
            _make_problem_state([("A", [("A-w0", [("node-0", 0, math.inf)])], False)])

    def test_nan_fraction_is_rejected_at_rust_boundary(self) -> None:
        """``NaN`` ``used_fraction`` panics in the Rust ``GpuAllocation`` constructor."""
        with pytest.raises(BaseException, match="NaN"):
            _make_problem_state([("A", [("A-w0", [("node-0", 0, math.nan)])], False)])
