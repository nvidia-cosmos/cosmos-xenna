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

Both schedulers expose the same ``setup`` / ``autoscale`` API.
Their **algorithmic surfaces differ**: the legacy fragmentation
scheduler runs four phases (manual, one-worker floor, max-min
balancing, fragmentation-gradient-descent fill) that opportunistically
consume free cluster capacity even without per-stage throughput
estimates. The saturation-aware scheduler currently only implements
the structural decisions: Phase A (manual delete + grow) and Phase B
(min-worker floor + cross-stage donor fallback). Saturation-driven
scale-up (Phase C) and scale-down (Phase D) ship in later iterations.

Strict per-stage equality is therefore not achievable today and is
not the contract this harness pins. What IS contractually equivalent:

  - **Manual stages**: both schedulers honour ``requested_num_workers``
    exactly. ``legacy_final == new_final == requested_num_workers``.
  - **Non-manual stages, lower bound**: both schedulers respect the
    stage's minimum-worker floor. ``new_final >= floor`` AND
    ``legacy_final >= floor``.
  - **Finished stages**: neither scheduler should add or delete a
    worker on a stage with ``is_finished=True``.

The harness pins these structural-equivalence properties on four
canonical fixtures so any future change that breaks the agreement on
manual stages, the implicit floor, or finished-stage handling is
caught at the parity layer rather than in production.
"""

import uuid

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.autoscaling_algorithms import FragmentationBasedAutoscaler
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 16) -> resources.ClusterResources:
    """CPU-only cluster sized for the parity fixtures."""
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
    """Multi-node GPU cluster: each node has one whole GPU."""
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


def _make_cpu_workers(stage_name: str, count: int) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers on ``node-0``."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
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
) -> tuple[data_structures.Solution, data_structures.Solution]:
    """Run the legacy and saturation-aware schedulers on the same fixture.

    Neither scheduler receives ``update_with_measurements`` calls, so
    the legacy scheduler's saturation-driven phases are no-ops and
    only its structural decisions (manual + floor) execute -- exactly
    the same surface the saturation-aware scheduler covers in Phase 1.
    """
    # ``floor_stuck_grace_cycles=0`` aligns the saturation-aware
    # scheduler with the legacy scheduler's immediate-failure semantics
    # for the parity comparison. The two failure modes differ in their
    # exception class -- legacy raises a Rust ``PanicException`` from
    # the autoscaling FFI, the saturation-aware scheduler raises a
    # Python ``RuntimeError`` -- but neither error condition is
    # reached on any of the canonical fixtures (each one is feasible
    # by construction).
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(min_workers=1),
    )
    legacy = FragmentationBasedAutoscaler()
    legacy.setup(problem)
    legacy_sol = legacy.autoscale(time=0.0, problem_state=state)

    new = SaturationAwareScheduler(cfg)
    new.setup(problem)
    new_sol = new.autoscale(time=0.0, problem_state=state)
    return legacy_sol, new_sol


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


# Floor used by the parity harness's non-manual-stage assertion. The fixtures
# all leave ``min_workers`` at the default (``None`` -> implicit 1) and do
# NOT exercise per-stage overrides; future fixtures that DO override the
# floor will need to feed the per-stage value through.
_PARITY_DEFAULT_FLOOR = 1


def _assert_structural_parity(
    *,
    problem: data_structures.Problem,
    state: data_structures.ProblemState,
    legacy_sol: data_structures.Solution,
    new_sol: data_structures.Solution,
) -> None:
    """Assert the schedulers agree on the structural decisions.

    Dispatches to one of three contract helpers per stage based on
    its mode: finished stages call
    :func:`_assert_finished_stage_untouched`, manual stages call
    :func:`_assert_manual_stage_exact_match`, and non-manual
    non-finished stages call :func:`_assert_non_manual_satisfies_floor`.
    """
    legacy_stages = legacy_sol.stages
    new_stages = new_sol.stages
    state_stages = state.rust.stages
    problem_stages = problem.rust.stages
    assert len(legacy_stages) == len(new_stages) == len(state_stages) == len(problem_stages), (
        f"Stage-count mismatch: legacy={len(legacy_stages)}, "
        f"new={len(new_stages)}, state={len(state_stages)}, problem={len(problem_stages)}"
    )
    for index, state_stage in enumerate(state_stages):
        initial = len(state_stage.worker_groups)
        legacy_final = _final_worker_count(initial, legacy_stages[index])
        new_final = _final_worker_count(initial, new_stages[index])
        problem_stage = problem_stages[index]
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
        _assert_non_manual_satisfies_floor(
            stage_name=problem_stage.name,
            legacy_final=legacy_final,
            new_final=new_final,
            floor=_PARITY_DEFAULT_FLOOR,
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
        legacy_sol, new_sol = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
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
        legacy_sol, new_sol = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
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
        legacy_sol, new_sol = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
        )

    def test_manual_gpu_stage_grows_to_requested_count_in_both_schedulers(self) -> None:
        """Three single-GPU stages on a 4-node cluster; one stage is manual.

        Both schedulers must honour the manual ``requested_num_workers``
        exactly (ingest grows from 1 to 2 -- the spare GPU on node-3 fits
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
        legacy_sol, new_sol = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
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
        legacy_sol, new_sol = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
        )

    def test_manual_grow_at_cluster_capacity_lands_in_both_schedulers(self) -> None:
        """A manual stage's grow saturates the cluster; both schedulers land exactly the request.

        Pins the parity surface where Phase A grow runs against a cluster
        whose remaining headroom equals the deficit -- legacy and the
        saturation-aware scheduler must both place the new worker without
        overshooting or short-changing the request.
        """
        # Cluster has 4 CPUs total; manual stage requests 4 with 1 worker
        # currently. Auto stage at 1. Manual must reach 4 exactly using
        # the 3 free CPUs that remain after the seed.
        cluster = _cluster(total_cpus_per_node=5)
        problem = _cpu_problem(cluster, [("manual", 4), ("auto", None)])
        state = _problem_state(
            [
                ("manual", _make_cpu_workers("manual", 1), 1, False),
                ("auto", _make_cpu_workers("auto", 1), 1, False),
            ]
        )
        legacy_sol, new_sol = _run_both_schedulers(problem=problem, state=state)
        _assert_structural_parity(
            problem=problem,
            state=state,
            legacy_sol=legacy_sol,
            new_sol=new_sol,
        )
