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

"""Structural parity harness: saturation-aware vs fragmentation-based scheduler.

Pins three structural-equivalence properties between the legacy
fragmentation-based scheduler and the saturation-aware scheduler:

  - **Manual stages** - both honour ``requested_num_workers``
    exactly (``legacy_final == new_final == requested_num_workers``).
  - **Non-manual stages, lower bound only** - both respect the
    minimum-worker floor (``legacy_final >= floor`` AND
    ``new_final >= floor``). Upper-bound divergence is expected and
    not asserted: legacy Phase 3 (FGD max-min balancing) fills free
    cluster capacity even without measurements, while the
    saturation-aware scheduler's Phase C / Phase D scale-up /
    scale-down are not yet shipped.
  - **Finished stages** - neither scheduler may add or delete a
    worker on a stage with ``is_finished=True``.

Single-cycle structural parity only; multi-cycle convergence parity
is not tested today (will be added when saturation-driven scale-up
and scale-down ship). Rationale for the structural-vs-numerical
scoping decision lives in ``docs/curator/scheduler-tuning.md``.
"""

import uuid

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.autoscaling_algorithms import FragmentationBasedAutoscaler
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 16) -> resources.ClusterResources:
    """CPU-only cluster sized for the parity fixtures.

    Guards reject ``num_nodes < 1`` and ``total_cpus_per_node < 1`` at the
    helper layer so a typo at a fixture call-site fails with an actionable
    Python error instead of an opaque Rust constructor failure.
    """
    if num_nodes < 1:
        msg = f"_cluster: num_nodes must be >= 1, got {num_nodes}"
        raise ValueError(msg)
    if total_cpus_per_node < 1:
        msg = f"_cluster: total_cpus_per_node must be >= 1, got {total_cpus_per_node}"
        raise ValueError(msg)
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


def _gpu_cluster(*, num_nodes: int = 3, total_cpus_per_node: int = 8) -> resources.ClusterResources:
    """Multi-node GPU cluster: each node has one whole GPU.

    Guards reject ``num_nodes < 1`` and ``total_cpus_per_node < 1`` at the
    helper layer so a typo at a fixture call-site fails with an actionable
    Python error instead of an opaque Rust constructor failure.
    """
    if num_nodes < 1:
        msg = f"_gpu_cluster: num_nodes must be >= 1, got {num_nodes}"
        raise ValueError(msg)
    if total_cpus_per_node < 1:
        msg = f"_gpu_cluster: total_cpus_per_node must be >= 1, got {total_cpus_per_node}"
        raise ValueError(msg)
    return resources.ClusterResources(
        nodes={
            f"node-{i}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus_per_node,
                gpus=[
                    resources.GpuResources(index=0, uuid_=uuid.uuid4(), used_fraction=0.0),
                ],
                name=f"node-{i}",
            )
            for i in range(num_nodes)
        },
    )


def _cpu_problem(
    cluster: resources.ClusterResources,
    stage_specs: list[tuple[str, int | None]],
) -> data_structures.Problem:
    """Build a ``Problem`` of CPU-shaped stages."""
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


def _gpu_problem(
    cluster: resources.ClusterResources,
    stage_specs: list[tuple[str, int | None]],
) -> data_structures.Problem:
    """Build a ``Problem`` of single-GPU-shaped stages."""
    gpu_shape = resources.Resources(cpus=1.0, gpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=gpu_shape,
            requested_num_workers=requested,
            over_provision_factor=None,
        )
        for name, requested in stage_specs
    ]
    return data_structures.Problem(cluster, stages)


def _make_cpu_workers(
    stage_name: str,
    count: int,
    *,
    node_index: int = 0,
) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers on ``node-{node_index}`` (default first node)."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node=f"node-{node_index}", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _gpu_worker_on_node(stage_name: str, node_index: int) -> data_structures.ProblemWorkerGroupState:
    """Build one 1-CPU + 1-GPU worker pinned to ``node-{node_index}``."""
    return data_structures.ProblemWorkerGroupState.make(
        f"{stage_name}-w0",
        [
            resources.WorkerResourcesInternal(
                node=f"node-{node_index}",
                cpus=1.0,
                gpus=[resources.GpuAllocationInternal(offset=0, used_fraction=1.0)],
            )
        ],
    )


def _problem_state(
    stage_specs: list[tuple[str, list[data_structures.ProblemWorkerGroupState], int, bool]],
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from per-stage ``(name, workers, slots_per_worker, is_finished)``."""
    states = [
        data_structures.ProblemStageState(
            stage_name=name,
            workers=workers,
            slots_per_worker=slots,
            is_finished=finished,
        )
        for name, workers, slots, finished in stage_specs
    ]
    return data_structures.ProblemState(states)


def _final_worker_count(
    initial: int,
    stage_solution: data_structures.StageSolution,
) -> int:
    """Compute the post-cycle worker count from a per-stage solution."""
    return initial + len(stage_solution.new_workers) - len(stage_solution.deleted_workers)


def _run_both_schedulers(
    *,
    problem: data_structures.Problem,
    state: data_structures.ProblemState,
    cfg: SaturationAwareConfig | None = None,
) -> tuple[data_structures.Solution, data_structures.Solution, SaturationAwareConfig]:
    """Run the legacy and saturation-aware schedulers on the same fixture.

    Neither scheduler receives ``update_with_measurements`` calls, so
    the legacy scheduler's saturation-driven phases are no-ops and
    only its structural decisions (manual + floor) execute - exactly
    the same surface the saturation-aware scheduler covers today.
    Returns the ``cfg`` alongside the two solutions so the caller's
    parity assertion can derive per-stage floors from the same source
    the new scheduler used.
    """
    # ``floor_stuck_grace_cycles=0`` aligns the saturation-aware
    # scheduler with the legacy scheduler's immediate-failure semantics
    # for the parity comparison. The two failure modes differ in their
    # exception class - legacy raises a Rust ``PanicException`` from
    # the autoscaling FFI, the saturation-aware scheduler raises a
    # Python ``RuntimeError`` - but neither error condition is
    # reached on any of the canonical fixtures (each one is feasible
    # by construction).
    if cfg is None:
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
        )
    # Wrap each scheduler call with a broad ``except`` so an unexpected
    # exception is re-raised as an AssertionError that clearly identifies
    # which scheduler raised on which fixture (legacy raises Rust
    # ``PanicException``, saturation-aware raises ``RuntimeError``;
    # bare Python tracebacks would otherwise hide the attribution).
    legacy = FragmentationBasedAutoscaler()
    legacy.setup(problem)
    try:
        legacy_sol = legacy.autoscale(time=0.0, problem_state=state)
    except Exception as e:
        msg = f"legacy FragmentationBasedAutoscaler raised on this fixture: {e!r}"
        raise AssertionError(msg) from e

    new = SaturationAwareScheduler(cfg)
    new.setup(problem)
    try:
        new_sol = new.autoscale(time=0.0, problem_state=state)
    except Exception as e:
        msg = f"SaturationAwareScheduler raised on this fixture: {e!r}"
        raise AssertionError(msg) from e
    return legacy_sol, new_sol, cfg


def _assert_finished_stage_untouched(
    *,
    stage_name: str,
    legacy_stage: data_structures.StageSolution,
    new_stage: data_structures.StageSolution,
) -> None:
    """A finished stage receives zero adds and zero deletes from both schedulers."""
    assert legacy_stage.new_workers == [], f"Finished stage {stage_name!r}: legacy added workers"
    assert legacy_stage.deleted_workers == [], f"Finished stage {stage_name!r}: legacy deleted workers"
    assert new_stage.new_workers == [], f"Finished stage {stage_name!r}: saturation-aware added workers"
    assert new_stage.deleted_workers == [], f"Finished stage {stage_name!r}: saturation-aware deleted workers"


def _assert_new_scheduler_does_not_churn(
    *,
    stage_name: str,
    new_stage: data_structures.StageSolution,
) -> None:
    """Assert no add or delete on a non-manual stage already at or above its floor."""
    assert new_stage.new_workers == [], (
        f"Non-manual stage {stage_name!r}: saturation-aware added workers above floor unexpectedly"
    )
    assert new_stage.deleted_workers == [], (
        f"Non-manual stage {stage_name!r}: saturation-aware deleted workers above floor unexpectedly"
    )


def _assert_manual_stage_exact_match(
    *,
    stage_name: str,
    requested: int,
    legacy_final: int,
    new_final: int,
) -> None:
    """A manual stage's post-cycle count equals ``requested_num_workers`` in both schedulers."""
    assert legacy_final == requested, (
        f"Manual stage {stage_name!r}: legacy_final={legacy_final}, requested_num_workers={requested}"
    )
    assert new_final == requested, (
        f"Manual stage {stage_name!r}: new_final={new_final}, requested_num_workers={requested}"
    )


def _assert_non_manual_satisfies_floor(
    *,
    stage_name: str,
    legacy_final: int,
    new_final: int,
    floor: int,
) -> None:
    """A non-manual non-finished stage's post-cycle count meets the configured minimum."""
    assert legacy_final >= floor, f"Non-manual stage {stage_name!r}: legacy_final={legacy_final} is below floor={floor}"
    assert new_final >= floor, f"Non-manual stage {stage_name!r}: new_final={new_final} is below floor={floor}"


def _stage_floor(*, cfg: SaturationAwareConfig, stage_name: str, num_nodes: int) -> int:
    """Compute the effective floor for a stage from the same config the new scheduler uses."""
    stage_cfg = cfg.get_effective_stage_config(stage_name=stage_name, spec_override=None)
    floor_from_min = stage_cfg.min_workers if stage_cfg.min_workers is not None else 1
    floor_from_per_node = (
        stage_cfg.min_workers_per_node * num_nodes if stage_cfg.min_workers_per_node is not None else 0
    )
    return max(floor_from_min, floor_from_per_node)


def _assert_structural_parity(
    *,
    problem: data_structures.Problem,
    state: data_structures.ProblemState,
    legacy_sol: data_structures.Solution,
    new_sol: data_structures.Solution,
    cfg: SaturationAwareConfig,
) -> None:
    """Assert the schedulers agree on the structural decisions.

    Dispatches to one of four contract helpers per stage based on
    mode: finished stages call :func:`_assert_finished_stage_untouched`;
    manual stages call :func:`_assert_manual_stage_exact_match`;
    non-manual non-finished stages call
    :func:`_assert_non_manual_satisfies_floor` (lower bound) and, when
    seeded at or above the floor, also
    :func:`_assert_new_scheduler_does_not_churn` - saturation-driven
    scale-up and scale-down are not yet shipped, so any add or delete
    on a non-manual stage already at or above its floor is a
    regression. Per-stage floors are computed from ``cfg`` via
    :func:`_stage_floor` so per-stage overrides flow through correctly.
    """
    legacy_stages = legacy_sol.stages
    new_stages = new_sol.stages
    state_stages = state.rust.stages
    problem_stages = problem.rust.stages
    num_nodes = len(problem.rust.cluster_resources.nodes)
    assert len(legacy_stages) == len(new_stages) == len(state_stages) == len(problem_stages), (
        f"Stage-count mismatch: legacy={len(legacy_stages)}, "
        f"new={len(new_stages)}, state={len(state_stages)}, problem={len(problem_stages)}"
    )
    for index, state_stage in enumerate(state_stages):
        problem_stage = problem_stages[index]
        if problem_stage.name != state_stage.stage_name:
            msg = (
                f"Fixture defect at index {index}: problem stage {problem_stage.name!r} "
                f"paired with state stage {state_stage.stage_name!r}"
            )
            raise AssertionError(msg)
        initial = len(state_stage.worker_groups)
        legacy_final = _final_worker_count(initial, legacy_stages[index])
        new_final = _final_worker_count(initial, new_stages[index])
        if state_stage.is_finished:
            _assert_finished_stage_untouched(
                stage_name=state_stage.stage_name,
                legacy_stage=legacy_stages[index],
                new_stage=new_stages[index],
            )
            continue
        if problem_stage.requested_num_workers is not None:
            _assert_manual_stage_exact_match(
                stage_name=problem_stage.name,
                requested=problem_stage.requested_num_workers,
                legacy_final=legacy_final,
                new_final=new_final,
            )
            continue
        floor = _stage_floor(cfg=cfg, stage_name=problem_stage.name, num_nodes=num_nodes)
        _assert_non_manual_satisfies_floor(
            stage_name=problem_stage.name,
            legacy_final=legacy_final,
            new_final=new_final,
            floor=floor,
        )
        if initial >= floor:
            _assert_new_scheduler_does_not_churn(
                stage_name=problem_stage.name,
                new_stage=new_stages[index],
            )


class TestSchedulerStructuralParity:
    """Pin the structural-decision contract across canonical fixtures.

    Manual stages produce identical final counts equal to the
    operator's ``requested_num_workers``; non-manual non-finished
    stages satisfy the implicit one-worker floor; finished stages
    receive zero adds and zero deletes from both schedulers.
    """

    def test_steady_state_meets_implicit_floor(self) -> None:
        """Three non-manual stages: both schedulers respect the implicit 1-worker floor."""
        cluster = _cluster(total_cpus_per_node=8)
        problem = _cpu_problem(cluster, [("A", None), ("B", None), ("C", None)])
        state = _problem_state(
            [
                ("A", _make_cpu_workers("A", 1), 1, False),
                ("B", _make_cpu_workers("B", 1), 1, False),
                ("C", _make_cpu_workers("C", 1), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_drain_tail_finished_stage_is_skipped(self) -> None:
        """A finished stage mid-pipeline is left untouched by both schedulers."""
        cluster = _cluster(total_cpus_per_node=8)
        problem = _cpu_problem(cluster, [("upstream", None), ("draining", None), ("downstream", None)])
        state = _problem_state(
            [
                ("upstream", _make_cpu_workers("upstream", 1), 1, False),
                ("draining", _make_cpu_workers("draining", 2), 1, True),  # finished
                ("downstream", _make_cpu_workers("downstream", 1), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_zero_stage_problem_is_a_no_op_for_both_schedulers(self) -> None:
        """An empty problem must not raise on either scheduler."""
        cluster = _cluster(total_cpus_per_node=4)
        problem = _cpu_problem(cluster, [])
        state = _problem_state([])
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_all_stages_finished_pipeline_is_a_no_op(self) -> None:
        """Every stage finished -> both schedulers are no-ops on every stage."""
        cluster = _cluster(total_cpus_per_node=8)
        problem = _cpu_problem(cluster, [("a", None), ("b", None), ("c", None)])
        state = _problem_state(
            [
                ("a", _make_cpu_workers("a", 1), 1, True),
                ("b", _make_cpu_workers("b", 1), 1, True),
                ("c", _make_cpu_workers("c", 1), 1, True),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_phase_b_floor_enforcement_grows_seeded_below_floor(self) -> None:
        """Phase B floor enforcement actually executes when seed is below the configured floor.

        Cluster has 4 free CPUs (single 8-CPU node, 4 used by other stages).
        Stage ``floor3`` is non-manual with ``min_workers=3`` and ``current=0``;
        the saturation-aware scheduler's Phase B floor loop must add 3 workers,
        the legacy scheduler's Phase 2 must place at least 1 (and likely
        scale further via Phase 3). Both end at ``final >= 3``.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"floor3": SaturationAwareStageConfig(min_workers=3)},
        )
        cluster = _cluster(total_cpus_per_node=8)
        problem = _cpu_problem(cluster, [("filler", None), ("floor3", None)])
        # Filler holds 4 of 8 CPUs to keep the cluster from being trivially empty.
        # The remaining 4 CPUs are enough for floor3 to reach min_workers=3.
        state = _problem_state(
            [
                ("filler", _make_cpu_workers("filler", 4), 1, False),
                ("floor3", _make_cpu_workers("floor3", 0), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg_back = _run_both_schedulers(problem=problem, state=state, cfg=cfg)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg_back,
        )
        # Tighten beyond the lower-bound floor assertion: Phase B MUST have
        # added at least three workers to floor3. Pins that the floor
        # enforcement loop actually executed (would catch a regression that
        # silently no-ops).
        assert len(new_sol.stages[1].new_workers) >= 3, (
            f"Phase B floor enforcement did not grow floor3 to its min_workers=3: "
            f"new_workers={[w.id for w in new_sol.stages[1].new_workers]}"
        )

    def test_balanced_four_stage_chain_meets_floor(self) -> None:
        """Four non-manual CPU stages each above the implicit floor."""
        cluster = _cluster(total_cpus_per_node=16)
        problem = _cpu_problem(cluster, [("A", None), ("B", None), ("C", None), ("D", None)])
        state = _problem_state(
            [
                ("A", _make_cpu_workers("A", 2), 1, False),
                ("B", _make_cpu_workers("B", 2), 1, False),
                ("C", _make_cpu_workers("C", 2), 1, False),
                ("D", _make_cpu_workers("D", 2), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_manual_gpu_stage_grows_to_requested_count_in_both_schedulers(self) -> None:
        """Three single-GPU stages on a 4-node cluster; one stage is manual.

        Both schedulers must honour the manual ``requested_num_workers``
        exactly (ingest grows from 1 to 2 - the spare GPU on node-3 fits
        the second worker). The other two stages must satisfy the
        implicit floor.
        """
        cluster = _gpu_cluster(num_nodes=4, total_cpus_per_node=4)
        problem = _gpu_problem(cluster, [("ingest", 2), ("transform", None), ("export", None)])
        state = _problem_state(
            [
                ("ingest", [_gpu_worker_on_node("ingest", 0)], 1, False),
                ("transform", [_gpu_worker_on_node("transform", 1)], 1, False),
                ("export", [_gpu_worker_on_node("export", 2)], 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_manual_at_requested_is_a_no_change_in_both_schedulers(self) -> None:
        """A manual stage already at ``requested_num_workers`` produces no add or delete."""
        cluster = _cluster(total_cpus_per_node=8)
        problem = _cpu_problem(cluster, [("pinned", 2), ("auto", None)])
        state = _problem_state(
            [
                ("pinned", _make_cpu_workers("pinned", 2), 1, False),
                ("auto", _make_cpu_workers("auto", 1), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )
        # Tighten beyond the manual-equal assertion: the manual stage's
        # cycle delta MUST be zero. Catches a regression where a scheduler
        # adds-and-then-deletes (churn) but lands at the same count.
        for sol_label, sol in (("legacy", legacy_sol), ("saturation-aware", new_sol)):
            assert sol.stages[0].new_workers == [], (
                f"{sol_label}: manual stage at requested produced add: {[w.id for w in sol.stages[0].new_workers]}"
            )
            assert sol.stages[0].deleted_workers == [], (
                f"{sol_label}: manual stage at requested produced delete: "
                f"{[w.id for w in sol.stages[0].deleted_workers]}"
            )

    def test_manual_shrink_lands_in_both_schedulers(self) -> None:
        """A manual stage at ``current > requested`` shrinks to exactly ``requested``.

        Pins the parity surface for the Phase A delete path: legacy Phase 1
        ``remove_best_worker_fn`` and saturation-aware
        ``_run_phase_a_delete`` both bring the stage down to the operator's
        requested count.
        """
        cluster = _cluster(total_cpus_per_node=8)
        problem = _cpu_problem(cluster, [("manual", 1), ("auto", None)])
        state = _problem_state(
            [
                ("manual", _make_cpu_workers("manual", 4), 1, False),
                ("auto", _make_cpu_workers("auto", 1), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )

    def test_manual_grow_at_cluster_capacity_lands_in_both_schedulers(self) -> None:
        """A manual stage's grow saturates the cluster; both schedulers land exactly the request.

        Pins the parity surface where Phase A grow runs against a cluster
        whose remaining headroom equals the deficit - legacy and the
        saturation-aware scheduler must both place the new worker without
        overshooting or short-changing the request.
        """
        # 1 node x 5 CPUs = 5 CPUs total. Seed uses 2 (manual=1 + auto=1),
        # leaving 3 free. Manual requests 4 -> deficit=3 = remaining headroom,
        # so Phase A grow saturates the cluster exactly.
        cluster = _cluster(total_cpus_per_node=5)
        problem = _cpu_problem(cluster, [("manual", 4), ("auto", None)])
        state = _problem_state(
            [
                ("manual", _make_cpu_workers("manual", 1), 1, False),
                ("auto", _make_cpu_workers("auto", 1), 1, False),
            ]
        )
        legacy_sol, new_sol, cfg = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
            cfg=cfg,
        )
