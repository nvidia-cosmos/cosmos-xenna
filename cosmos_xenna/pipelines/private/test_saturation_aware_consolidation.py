# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the Phase D consolidation tiebreak under fractional-GPU fragmentation.

These tests replay a production failure mode (tracked locally as
``log-b2da3234``) where fractional-GPU workers are spread across
the cluster so no single H100 is fully free for a downstream
whole-GPU consumer. The cluster shape:

  * 4 x H100 cluster (4 nodes; each node carries 1 H100 GPU at
    offset 0, exposed to the planner with capacity 1.0).
  * 3 fractional GPU stages (A, B, C); every worker takes 0.25 of
    a single GPU.
  * 4 workers per stage at cycle start, spread across the four
    physical GPUs rather than consolidated onto the fewest GPUs
    possible. The spread is the failure mode: a downstream
    whole-GPU consumer (for example, ``VllmAsyncCaptionStage``)
    cannot claim a whole H100 because every GPU carries a
    fractional allocation.

The behavioral contract this file pins: after each fractional
stage shrinks from 4 workers to 1 worker (whether in a single
aggressive cycle or across multiple intent=-1 cycles), at least
one whole GPU becomes fully unallocated. The consolidation
tiebreak (``host_gpu_used_fraction`` ASC) drains the least-loaded
GPUs first, so the surviving workers consolidate onto the
heaviest GPU and leave a free GPU for downstream whole-GPU
consumers.

Sibling coverage in ``test_saturation_aware_phase_d_basic.py``,
``test_saturation_aware_scale_down.py``, and
``test_saturation_aware_consolidation_helpers.py`` pins the sort
key in isolation. This file is the multi-stage, multi-node
end-to-end replay that proves the orchestrator wiring actually
frees whole GPUs in the failure shape.
"""

import uuid
from collections.abc import Callable
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig

SchedulerFactory = Callable[..., tuple[SaturationAwareScheduler, data_structures.Problem]]
GpuStateFactory = Callable[..., data_structures.ProblemState]
AutoscaleFactory = Callable[
    [SaturationAwareScheduler, data_structures.ProblemState, dict[str, int]],
    data_structures.Solution,
]


def _h100_cluster(num_nodes: int = 4, total_cpus: int = 16) -> resources.ClusterResources:
    """Build a 4 x H100-shaped cluster: ``num_nodes`` nodes, each with one GPU at offset 0.

    The cluster only carries the GPU slot structure required by the
    planner to validate worker placements; the per-GPU
    ``used_fraction`` is left at 0.0 because the test fixtures
    inject per-worker allocations directly through
    ``ProblemState`` (the cluster slots themselves are not the
    source of truth for the consolidation tiebreak).

    Args:
        num_nodes: Cluster node count. Defaults to 4 (the
            production-incident topology).
        total_cpus: Per-node CPU count.

    Returns:
        A cluster with ``num_nodes`` nodes; each node carries one
        zero-fraction H100 at offset 0.
    """
    return resources.ClusterResources(
        nodes={
            f"node-{index}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus,
                gpus=[
                    resources.GpuResources(
                        index=0,
                        uuid_=uuid.uuid4(),
                        used_fraction=0.0,
                    ),
                ],
                name=f"node-{index}",
            )
            for index in range(num_nodes)
        },
    )


def _make_config(
    *,
    min_workers: int = 1,
    max_scale_down_fraction_per_cycle: float = 1.0,
) -> SaturationAwareConfig:
    """Build a config with the per-cycle fraction cap set explicitly.

    The orchestrator-level ``max_scale_down_fraction_per_cycle`` cap
    defaults to ``0.05`` in production so that an over-provisioned
    stage drains at a measured rate rather than collapsing in a
    single autoscale cycle. The fixtures in this file override the
    cap so each test can pick its own shrink speed:

      * ``1.0`` (default here) lets a single cycle drop a stage
        from 4 workers to 1 worker when the intent is large enough,
        which is the cleanest reproduction of the single-cycle
        contract pinned by test 1.
      * Smaller values such as ``0.25`` slow the shrink to one
        worker per cycle, used by tests 2 and 3 to exercise the
        multi-cycle convergence and per-stage drainage monotonicity.

    Args:
        min_workers: Per-stage floor. Defaults to 1 so a 4 -> 1
            shrink is allowed without bottoming out earlier.
        max_scale_down_fraction_per_cycle: Orchestrator-level
            per-cycle fraction cap on Phase D deletions.

    Returns:
        A ``SaturationAwareConfig`` carrying the requested
        per-stage defaults and zero floor-stuck grace cycles
        (so any floor miss surfaces immediately in tests).
    """
    return SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=min_workers,
            max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
            # Consolidation tests pin Phase D victim selection only and are
            # orthogonal to the donor / measurement warmup graces. Disable
            # both so freshly-warmed workers do not get filtered out of the
            # victim pool (which would mask the consolidation contract).
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
        ),
    )


@pytest.fixture
def make_scheduler() -> SchedulerFactory:
    """Build a fresh ``SaturationAwareScheduler`` over the 4 x H100 cluster shape.

    Returns a closure
    ``factory(stage_specs, *, cfg=None, num_nodes=4, gpu_per_worker=0.25)``
    so each test composes the exact stage configuration it needs.
    Every invocation returns a brand-new scheduler so cross-cycle
    state (``_worker_ages``, ``_cycle_counter``) never leaks
    between tests.

    The stage worker shape carries ``gpu_per_worker`` of one GPU so
    every worker in the spread layout fits as a fractional placement
    on a single H100. The Phase D consolidation tiebreak only reads
    the cluster shape to validate ``(node, offset)`` lookups for the
    per-worker fraction map; the cluster's stored
    ``used_fraction`` is irrelevant because the tests inject the
    runtime allocations directly through ``ProblemState``.
    """

    def _factory(
        stage_specs: list[tuple[str, int | None]],
        *,
        cfg: SaturationAwareConfig | None = None,
        num_nodes: int = 4,
        gpu_per_worker: float = 0.25,
    ) -> tuple[SaturationAwareScheduler, data_structures.Problem]:
        if cfg is None:
            cfg = _make_config()
        cluster = _h100_cluster(num_nodes=num_nodes)
        gpu_shape = resources.Resources(cpus=1.0, gpus=gpu_per_worker).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster,
            [
                data_structures.ProblemStage(
                    name=name,
                    stage_batch_size=1,
                    worker_shape=gpu_shape,
                    requested_num_workers=requested,
                    over_provision_factor=None,
                )
                for name, requested in stage_specs
            ],
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(problem)
        return scheduler, problem

    return _factory


@pytest.fixture
def make_gpu_state() -> GpuStateFactory:
    """Build a ``ProblemState`` whose workers carry explicit GPU allocations.

    The factory accepts a list of per-stage rows shaped as
    ``(stage_name, [(worker_id, [(node, gpu_offset, used_fraction), ...]), ...], is_finished)``.

    Each inner triple list becomes a ``WorkerResourcesInternal``
    plus one ``GpuAllocationInternal`` per ``(node, gpu_offset,
    fraction)`` triple, grouped by node so a single worker may
    appear on multiple GPUs on the same node when the layout
    requires it (none of the tests in this file exercise that
    case, but the helper stays consistent with the sibling
    fixtures in ``test_saturation_aware_phase_d_basic.py``).

    Workers default to ``num_used_slots=0`` so they are idle and
    eligible for shrink; the consolidation primary key dominates
    the idle secondary key when the GPU fractions differ. The
    tests in this file are not idle-key tests, so the default is
    fine.
    """

    def _factory(
        stage_specs: list[
            tuple[
                str,
                list[tuple[str, list[tuple[str, int, float]]]],
                bool,
            ]
        ],
    ) -> data_structures.ProblemState:
        rows: list[data_structures.ProblemStageState] = []
        for stage_name, worker_rows, is_finished in stage_specs:
            workers: list[data_structures.ProblemWorkerGroupState] = []
            for worker_id, gpu_triples in worker_rows:
                gpus_by_node: dict[str, list[resources.GpuAllocationInternal]] = {}
                for node, offset, fraction in gpu_triples:
                    gpus_by_node.setdefault(node, []).append(
                        resources.GpuAllocationInternal(offset=offset, used_fraction=fraction),
                    )
                allocations = [
                    resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=gpu_list)
                    for node, gpu_list in gpus_by_node.items()
                ]
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

    return _factory


@pytest.fixture
def autoscale_with_intents() -> AutoscaleFactory:
    """Run a single ``autoscale`` cycle with the given signed intent deltas.

    Patches ``SaturationAwareScheduler._compute_intent_deltas`` for
    the duration of the call so the per-stage classifier
    machinery is bypassed and the test controls Phase D's
    deletion driver directly. The patch is scoped to one
    ``autoscale`` call so multi-cycle tests can re-invoke the
    factory with different intents on the same scheduler instance.
    """

    def _factory(
        scheduler: SaturationAwareScheduler,
        state: data_structures.ProblemState,
        intents: dict[str, int],
    ) -> data_structures.Solution:
        with patch.object(scheduler, "_compute_intent_deltas", return_value=dict(intents)):
            return scheduler.autoscale(time=0.0, problem_state=state)

    return _factory


def _spread_layout_specs() -> list[tuple[str, list[tuple[str, list[tuple[str, int, float]]]], bool]]:
    """Build the canonical fractional-spread layout for the production failure shape.

    The 12 workers are distributed across 4 H100 GPUs (one per
    node) so that each stage has at least one worker on a low-
    fraction GPU and at least one worker on the heavy GPU::

        node-0 GPU (1.00, heavy):  A-0, B-0, C-0, C-1
        node-1 GPU (0.75, medium): A-1, B-1, C-2
        node-2 GPU (0.50, low):    A-2, B-2
        node-3 GPU (0.75, medium): A-3, B-3, C-3

    Per stage:

      * Stage A: worker on each of node-0, node-1, node-2, node-3.
      * Stage B: worker on each of node-0, node-1, node-2, node-3.
      * Stage C: two workers on node-0 (because the per-node
        capacity is 4 fractional placements and we want one
        whole GPU at the cap), and one each on node-1, node-3.

    The asymmetric assignment of stage C is what makes the
    layout non-symmetric: it gives the consolidation tiebreak
    something to bite on (otherwise all stages would see
    identical fraction maps and a uniform deletion order).

    Returns:
        Stage specs in the shape expected by ``make_gpu_state``.
    """
    return [
        (
            "A",
            [
                ("A-0", [("node-0", 0, 0.25)]),
                ("A-1", [("node-1", 0, 0.25)]),
                ("A-2", [("node-2", 0, 0.25)]),
                ("A-3", [("node-3", 0, 0.25)]),
            ],
            False,
        ),
        (
            "B",
            [
                ("B-0", [("node-0", 0, 0.25)]),
                ("B-1", [("node-1", 0, 0.25)]),
                ("B-2", [("node-2", 0, 0.25)]),
                ("B-3", [("node-3", 0, 0.25)]),
            ],
            False,
        ),
        (
            "C",
            [
                ("C-0", [("node-0", 0, 0.25)]),
                ("C-1", [("node-0", 0, 0.25)]),
                ("C-2", [("node-1", 0, 0.25)]),
                ("C-3", [("node-3", 0, 0.25)]),
            ],
            False,
        ),
    ]


def _survivor_nodes(
    problem_state: data_structures.ProblemState,
    solution: data_structures.Solution,
) -> set[str]:
    """Return the set of node names that still carry at least one worker post-cycle.

    The ``Solution`` carries only the per-cycle delta (added and
    removed workers), not the post-cycle worker set, so survivors
    have to be derived: cycle-start workers from
    ``problem_state`` minus those marked ``deleted_workers`` in the
    matching ``StageSolution``, plus any workers added under
    ``new_workers`` (none in this file's Phase-D-only tests, but
    handled defensively in case Phase B or Phase C ever fires).

    For each survivor, the worker's resource allocations are
    walked to collect every node it occupies. The returned set is
    the set of nodes whose H100 is still fractionally allocated
    once the autoscale cycle commits its deltas.
    """
    survivor_nodes: set[str] = set()
    for stage_state, stage_solution in zip(
        problem_state.rust.stages,
        solution.stages,
        strict=True,
    ):
        deleted_ids = {worker.id for worker in stage_solution.deleted_workers}
        for worker_group in stage_state.worker_groups:
            if worker_group.id in deleted_ids:
                continue
            for resource in worker_group.resources:
                survivor_nodes.add(resource.node)
        for new_worker in stage_solution.new_workers:
            for resource in new_worker.resources:
                survivor_nodes.add(resource.node)
    return survivor_nodes


def _next_cycle_state(
    make_gpu_state: GpuStateFactory,
    base_specs: list[tuple[str, list[tuple[str, list[tuple[str, int, float]]]], bool]],
    deleted_ids: set[str],
) -> data_structures.ProblemState:
    """Build the next-cycle ``ProblemState`` after dropping ``deleted_ids``.

    Filters every stage's worker rows to exclude ``deleted_ids``
    and rebuilds the ``ProblemState`` with the surviving rows.
    The per-worker fractions stay at the original 0.25 because
    each worker only occupies its own 0.25 of the GPU; the
    cluster-wide consolidation key is recomputed at the start of
    every autoscale cycle from this fresh runtime snapshot.
    """
    next_specs: list[tuple[str, list[tuple[str, list[tuple[str, int, float]]]], bool]] = []
    for stage_name, worker_rows, is_finished in base_specs:
        surviving = [(wid, alloc) for wid, alloc in worker_rows if wid not in deleted_ids]
        next_specs.append((stage_name, surviving, is_finished))
    return make_gpu_state(next_specs)


class TestProductionIncidentReplay:
    """Pin the closing-the-loop guarantee for the production-incident shape.

    The contract: when 3 fractional GPU stages run on a 4 x H100
    cluster with workers spread across all four GPUs (no GPU
    free at cycle start), shrinking each stage from 4 workers
    down to 1 worker must free at least one whole H100 so a
    downstream whole-GPU stage can claim it. The Phase D
    consolidation tiebreak primary key
    (``host_gpu_used_fraction`` ASC) drains the least-loaded
    GPUs first, so the surviving workers consolidate onto the
    heaviest GPU.

    Each test pins one behavior:

      * Single-cycle aggressive shrink frees at least one GPU.
      * Multi-cycle gradual shrink converges with at least one
        GPU free.
      * Per-stage drainage is monotonic across cycles
        (the lowest-fraction GPU is targeted first per stage).
    """

    def test_single_cycle_4_to_1_shrink_frees_at_least_one_whole_gpu(
        self,
        make_scheduler: SchedulerFactory,
        make_gpu_state: GpuStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """A single aggressive 4 -> 1 shrink leaves at least one H100 unallocated.

        With ``max_scale_down_fraction_per_cycle=1.0`` and intent
        of -3 per stage, every fractional stage drops to its
        ``min_workers=1`` floor in one autoscale cycle. The
        externally-observable contract verified here is the
        end-state property: at least one node has zero surviving
        workers, so at least one whole H100 is free for a
        downstream whole-GPU consumer.
        """
        scheduler, _ = make_scheduler([("A", None), ("B", None), ("C", None)])
        specs = _spread_layout_specs()
        state = make_gpu_state(specs)

        solution = autoscale_with_intents(scheduler, state, {"A": -3, "B": -3, "C": -3})

        all_nodes = {f"node-{i}" for i in range(4)}
        survivor_nodes = _survivor_nodes(state, solution)
        free_nodes = all_nodes - survivor_nodes
        assert free_nodes, (
            f"single-cycle 4 -> 1 shrink failed to free any whole H100; "
            f"surviving worker nodes were {sorted(survivor_nodes)}"
        )

    def test_multi_cycle_intent_minus_one_converges_with_free_gpu(
        self,
        make_scheduler: SchedulerFactory,
        make_gpu_state: GpuStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Running cycles until every stage hits its 1-worker floor leaves a whole H100 free.

        With ``max_scale_down_fraction_per_cycle=0.25`` and
        intent of -1 per stage per cycle, each cycle deletes
        exactly one worker per stage. The loop runs until the
        scheduler stops emitting deletions (every stage is at
        ``min_workers=1``); the converged state is then inspected
        for at least one fully-unallocated H100. ``max_cycles=10``
        is the watchdog bound; the expected convergence path is
        4 -> 3 -> 2 -> 1 per stage in 3 effective cycles plus a
        final no-op cycle that breaks the loop.
        """
        cfg = _make_config(max_scale_down_fraction_per_cycle=0.25)
        scheduler, _ = make_scheduler([("A", None), ("B", None), ("C", None)], cfg=cfg)
        specs = _spread_layout_specs()
        state = make_gpu_state(specs)
        deleted_total: set[str] = set()
        max_cycles = 10
        cycles_run = 0
        for _ in range(max_cycles):
            solution = autoscale_with_intents(
                scheduler,
                state,
                {"A": -1, "B": -1, "C": -1},
            )
            cycles_run += 1
            cycle_deleted = {worker.id for stage in solution.stages for worker in stage.deleted_workers}
            if not cycle_deleted:
                break
            deleted_total.update(cycle_deleted)
            state = _next_cycle_state(make_gpu_state, specs, deleted_total)
        assert cycles_run < max_cycles, (
            f"multi-cycle shrink did not converge within {max_cycles} cycles "
            f"(deleted ids so far: {sorted(deleted_total)})"
        )

        all_nodes = {f"node-{i}" for i in range(4)}
        survivor_nodes = _survivor_nodes(state, solution)
        free_nodes = all_nodes - survivor_nodes
        assert free_nodes, (
            f"multi-cycle convergence ended with every H100 still allocated; "
            f"surviving worker nodes were {sorted(survivor_nodes)}"
        )

    def test_per_stage_drainage_targets_lowest_fraction_gpu_each_cycle(
        self,
        make_scheduler: SchedulerFactory,
        make_gpu_state: GpuStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Stage A's deleted worker each cycle sits on the lowest-fraction GPU among A's remaining workers.

        For each cycle, the per-worker consolidation key is the
        host-GPU used fraction at cycle start (the MAX across the
        worker's GPU allocations; with single-GPU workers this is
        the GPU's own fraction). The Phase D sort then picks the
        worker with the lowest key.

        The test runs three cycles of intent=-1 for every stage
        (the same ``max_scale_down_fraction_per_cycle=0.25``
        gradual-shrink configuration as the multi-cycle test)
        and asserts that, in every cycle, stage A's deleted
        worker is the one whose cycle-start host-GPU fraction is
        the minimum across A's surviving workers. A regression
        that scattered A's deletions back to a higher-fraction
        GPU before the lower-fraction GPU was drained would
        violate this property.

        Stage A's spread (one worker on each of the four nodes)
        gives the assertion a clean target: every cycle A has at
        least two workers with distinct host-GPU fractions, so
        the minimum is uniquely determined.
        """
        cfg = _make_config(max_scale_down_fraction_per_cycle=0.25)
        scheduler, _ = make_scheduler([("A", None), ("B", None), ("C", None)], cfg=cfg)
        specs = _spread_layout_specs()
        deleted_total: set[str] = set()
        for _cycle in range(3):
            state = _next_cycle_state(make_gpu_state, specs, deleted_total)
            host_gpu_used_fractions = SaturationAwareScheduler._compute_host_gpu_used_fractions(state)
            stage_a = state.rust.stages[0]
            assert stage_a.stage_name == "A"
            per_worker_a = SaturationAwareScheduler._extract_worker_host_gpu_used_fractions(
                runtime_stage=stage_a,
                host_gpu_used_fractions=host_gpu_used_fractions,
            )
            assert per_worker_a, "stage A must have surviving workers before the deletion cycle"
            min_fraction = min(per_worker_a.values())

            solution = autoscale_with_intents(
                scheduler,
                state,
                {"A": -1, "B": -1, "C": -1},
            )

            stage_a_deletes = [worker.id for worker in solution.stages[0].deleted_workers]
            assert len(stage_a_deletes) == 1, f"stage A expected to delete exactly one worker; got {stage_a_deletes}"
            deleted_id = stage_a_deletes[0]
            deleted_fraction = per_worker_a[deleted_id]
            assert deleted_fraction == pytest.approx(min_fraction, abs=1e-5), (
                f"stage A drainage broke monotonicity: cycle deleted {deleted_id!r} "
                f"with host-GPU fraction {deleted_fraction:.4f}, but the minimum "
                f"fraction among A's workers was {min_fraction:.4f} "
                f"(per-worker: {per_worker_a})"
            )
            deleted_total.update(worker.id for stage in solution.stages for worker in stage.deleted_workers)
