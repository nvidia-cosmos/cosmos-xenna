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
  * Defensive numeric inputs (negative, NaN, inf fractions).
  * Determinism across calls.
"""

import math

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler


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

        assert result == {("node-0", 2): 0.35}

    def test_cross_stage_allocations_on_same_gpu_sum(self) -> None:
        """Allocations on the same ``(node, offset)`` from different stages add."""
        state = _make_problem_state(
            [
                ("A", [("A-w0", [("node-0", 0, 0.30)])], False),
                ("B", [("B-w0", [("node-0", 0, 0.50)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {("node-0", 0): pytest.approx(0.80)}

    def test_finished_stage_allocations_are_aggregated(self) -> None:
        """Finished stages still hold resources and contribute to the map."""
        state = _make_problem_state(
            [
                ("A", [("A-w0", [("node-0", 0, 0.20)])], False),
                ("B-finished", [("B-w0", [("node-0", 0, 0.40)])], True),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {("node-0", 0): pytest.approx(0.60)}

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

        assert result == {
            ("node-0", 0): pytest.approx(0.10),
            ("node-0", 1): pytest.approx(0.20),
            ("node-1", 0): pytest.approx(0.30),
        }

    def test_multi_allocation_worker_on_same_node_sums_into_distinct_keys(self) -> None:
        """A non-SPMD multi-GPU worker on one node produces one entry per GPU offset."""
        state = _make_problem_state(
            [
                ("A", [("A-multi", [("node-0", 0, 0.50), ("node-0", 1, 0.30)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {
            ("node-0", 0): pytest.approx(0.50),
            ("node-0", 1): pytest.approx(0.30),
        }

    def test_spmd_worker_on_distinct_nodes_records_each_node_offset_pair(self) -> None:
        """An SPMD worker spanning nodes contributes one entry per ``(node, offset)`` pair."""
        state = _make_problem_state(
            [
                ("A", [("A-spmd", [("node-0", 0, 1.0), ("node-1", 0, 1.0)])], False),
            ],
        )

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {
            ("node-0", 0): pytest.approx(1.0),
            ("node-1", 0): pytest.approx(1.0),
        }

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

        assert result[("node-0", 0)] == pytest.approx(1.50)

    def test_zero_fraction_allocations_are_recorded(self) -> None:
        """Allocations with ``used_fraction=0`` still appear in the map (key exists)."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, 0.0)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {("node-0", 0): 0.0}

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

        assert result == {"A-w0": pytest.approx(0.70)}

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

        assert result == {"A-w0": pytest.approx(0.95)}

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

        assert result == {"A-spmd": pytest.approx(1.0)}

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

        assert result == {
            "A-light": pytest.approx(0.10),
            "A-heavy": pytest.approx(0.95),
            "A-cpu-only": 0.0,
        }

    def test_zero_aggregate_fractions_project_to_zero(self) -> None:
        """A GPU with aggregate 0.0 projects to a worker fraction of 0.0."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, 0.0)])], False)])

        result = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
            runtime_stage=state.rust.stages[0],
            host_gpu_used_fractions={("node-0", 0): 0.0},
        )

        assert result == {"A-w0": 0.0}


class TestDefensiveNumericInputs:
    """Pin the helpers' tolerance for adversarial numeric inputs.

    These tests do not assert that NaN/inf/negative values are rejected;
    the data structures accept them at construction. Instead they pin
    the helpers' propagation behavior so a future refactor that adds
    validation explicitly chooses the new contract rather than silently
    breaking on data that previously round-tripped.
    """

    def test_negative_fraction_passes_through_aggregator(self) -> None:
        """The helper does not validate fraction sign; downstream callers may detect it."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, -0.1)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert result == {("node-0", 0): pytest.approx(-0.1)}

    def test_inf_fraction_passes_through_aggregator(self) -> None:
        """``+inf`` fractions are preserved (sort key remains comparable)."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, math.inf)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert math.isinf(result[("node-0", 0)])

    def test_nan_fraction_propagates_through_aggregator(self) -> None:
        """``NaN`` fractions propagate; sort behavior is undefined and explicitly out of scope."""
        state = _make_problem_state([("A", [("A-w0", [("node-0", 0, math.nan)])], False)])

        result = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)

        assert math.isnan(result[("node-0", 0)])
