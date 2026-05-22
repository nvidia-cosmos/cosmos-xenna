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


"""Cluster-perturbation integration tests for ``SaturationAwareScheduler``.

The public scheduler API freezes ``cluster_resources`` at
:meth:`SaturationAwareScheduler.setup`. Real cluster scale-in /
scale-out events therefore manifest only as changes to the per-cycle
``ProblemState.worker_groups`` snapshot fed by the streaming layer.

This module focuses on whole-pipeline survival across such events,
where multiple SAT features (Phase B floor, Phase C grow, Phase D
shrink, per-worker bookkeeping, classifier EWMA) must compose
correctly across N cycles. Single-feature node-churn mechanics are
already covered by ``test_node_churn.py``; the assertions here are
integration-level (no crashes, intent decisions still flow, no
phantom worker ids in state, floor + survivor invariants hold).

Scenarios:

  * ``test_node_disappearance_does_not_crash_running_pipeline``
  * ``test_cluster_scale_out_drives_growth_into_new_nodes``
  * ``test_cluster_scale_in_consolidates_remaining_capacity``
"""

from collections.abc import Callable

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _multi_node_cluster(*, num_nodes: int = 2, total_cpus_per_node: int = 4) -> resources.ClusterResources:
    """Build a CPU-only cluster with ``num_nodes`` identical nodes."""
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


def _problem(
    stage_names: list[str],
    cluster: resources.ClusterResources,
) -> data_structures.Problem:
    """Build a ``Problem`` with one CPU stage per name on ``cluster``."""
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


def _worker_group(
    stage_name: str, idx: int, *, node: str, num_used_slots: int = 0
) -> data_structures.ProblemWorkerGroupState:
    """One CPU worker_group keyed by ``{stage_name}-{node}-w{idx}``.

    The id encodes the node so scenario assertions can pin which
    placement a worker landed on without inspecting Rust internals.
    """
    return data_structures.ProblemWorkerGroupState.make(
        f"{stage_name}-{node}-w{idx}",
        [resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=[])],
        num_used_slots=num_used_slots,
    )


def _stage_state(
    *,
    name: str,
    workers_per_node: dict[str, int],
    slots_per_worker: int = 8,
    used_slot_ratio: float = 0.5,
    input_queue_depth: int = 0,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState`` with explicit per-node worker placement.

    Args:
        name: Stage name.
        workers_per_node: Map of node name -> number of workers on
            that node. Iteration order is preserved so worker ids
            are stable across cycles.
        slots_per_worker: Slots advertised per worker_group.
        used_slot_ratio: Fraction of slots reported as occupied
            (``0.0`` = idle, ``1.0`` = saturated).
        input_queue_depth: Stage-level input queue depth.

    """
    used_per_worker = max(0, min(slots_per_worker, round(slots_per_worker * used_slot_ratio)))
    workers: list[data_structures.ProblemWorkerGroupState] = []
    for node, count in workers_per_node.items():
        for i in range(count):
            workers.append(_worker_group(name, i, node=node, num_used_slots=used_per_worker))
    total = sum(workers_per_node.values())
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=workers,
        slots_per_worker=slots_per_worker,
        is_finished=False,
        num_used_slots=total * used_per_worker,
        num_empty_slots=total * (slots_per_worker - used_per_worker),
        input_queue_depth=input_queue_depth,
    )


def _build_scheduler(
    stage_names: list[str],
    cluster: resources.ClusterResources,
    *,
    min_workers: int | None = 1,
    saturated_streak_min_cycles: int = 1,
    over_provisioned_streak_min_cycles: int = 2,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 2,
    max_scale_down_fraction_per_cycle: float = 1.0,
) -> SaturationAwareScheduler:
    """Build a scheduler over the given stages with churn-friendly defaults.

    Disables warmup grace and setup-phase quiescence so signals from
    the per-cycle ``ProblemState`` reach the classifier on every
    cycle without filtering. Lifts ``max_scale_down_fraction_per_cycle``
    so Phase D's per-cycle bound is set entirely by the per-stage
    intent.
    """
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=min_workers,
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
            max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_names, cluster))
    return scheduler


def _run_cycles(
    scheduler: SaturationAwareScheduler,
    state_factory: Callable[[int], data_structures.ProblemState],
    *,
    num_cycles: int,
    cycle_interval_s: float = 10.0,
    start_time_s: float = 0.0,
) -> list[data_structures.Solution]:
    """Drive ``num_cycles`` ``autoscale()`` cycles and return every ``Solution``."""
    solutions: list[data_structures.Solution] = []
    for cycle_idx in range(num_cycles):
        ps = state_factory(cycle_idx)
        sol = scheduler.autoscale(time=start_time_s + cycle_idx * cycle_interval_s, problem_state=ps)
        solutions.append(sol)
    return solutions


class TestNodeDisappearanceDoesNotCrashRunningPipeline:
    """A multi-stage pipeline survives a mid-run node disappearance."""

    def test_node_disappearance_does_not_crash_running_pipeline(self) -> None:
        """3-stage chain, lose all of node-1's workers on cycle 3, run 7 more cycles.

        Across 10 cycles total the scheduler must:

          * Produce a structurally valid :class:`Solution` for every
            cycle (no exceptions raised, all stages present).
          * Evict every node-1 worker id from the per-worker
            bookkeeping maps by the cycle after the loss.
          * Continue to honour each stage's floor on the surviving
            (node-0) workers.
        """
        cluster = _multi_node_cluster(num_nodes=2, total_cpus_per_node=4)
        scheduler = _build_scheduler(["A", "B", "C"], cluster)

        full_layout = {"node-0": 1, "node-1": 1}
        post_loss_layout = {"node-0": 1}

        def state_factory(cycle: int) -> data_structures.ProblemState:
            layout = post_loss_layout if cycle >= 3 else full_layout
            return data_structures.ProblemState(
                [
                    _stage_state(name="A", workers_per_node=layout, used_slot_ratio=0.5),
                    _stage_state(name="B", workers_per_node=layout, used_slot_ratio=0.5),
                    _stage_state(name="C", workers_per_node=layout, used_slot_ratio=0.5),
                ]
            )

        solutions = _run_cycles(scheduler, state_factory, num_cycles=10)

        assert len(solutions) == 10, f"expected 10 solutions, got {len(solutions)}"
        for cycle_idx, sol in enumerate(solutions):
            assert len(sol.stages) == 3, (
                f"cycle {cycle_idx}: solution must keep 3 stages even across loss; got {len(sol.stages)}"
            )

        for stage in ("A", "B", "C"):
            lost_ids = {f"{stage}-node-1-w0"}
            assert lost_ids.isdisjoint(scheduler._worker_ages), (
                f"stale node-1 ids leaked into _worker_ages for {stage}: {lost_ids & set(scheduler._worker_ages)}"
            )
            assert lost_ids.isdisjoint(scheduler._worker_ready_first_seen_at), (
                f"stale node-1 ids leaked into _worker_ready_first_seen_at for {stage}: "
                f"{lost_ids & set(scheduler._worker_ready_first_seen_at)}"
            )

        for stage_idx, stage_name in enumerate(("A", "B", "C")):
            final_solution = solutions[-1].stages[stage_idx]
            pre_cycle_live = 1
            live_after = pre_cycle_live + len(final_solution.new_workers) - len(final_solution.deleted_workers)
            assert live_after >= 1, f"stage {stage_name} dropped below floor on final cycle; got live={live_after}"


class TestClusterScaleOutDrivesGrowthIntoNewNodes:
    """Phase C placements distribute across all cluster nodes when one is idle."""

    def test_cluster_scale_out_drives_growth_into_new_nodes(self) -> None:
        """Saturated stage starts on node-0 only; Phase C eventually lands a worker on node-1.

        The Problem is set up with a 2-node cluster but the cycle 0
        snapshot only populates node-0 (simulating a scale-out: a
        previously cold node becomes available for placement).
        Across a few Phase C cycles the placer must use the idle
        node-1 capacity for at least one new worker.
        """
        cluster = _multi_node_cluster(num_nodes=2, total_cpus_per_node=4)
        scheduler = _build_scheduler(["hot"], cluster)

        def saturated_on_nodes(nodes: dict[str, int]) -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [_stage_state(name="hot", workers_per_node=nodes, used_slot_ratio=1.0, input_queue_depth=8)]
            )

        live: dict[str, int] = {"node-0": 1}
        solutions: list[data_structures.Solution] = []
        for cycle_idx in range(6):
            sol = scheduler.autoscale(time=cycle_idx * 10.0, problem_state=saturated_on_nodes(live))
            solutions.append(sol)
            for new_worker in sol.stages[0].new_workers:
                if not new_worker.resources:
                    continue
                node = new_worker.resources[0].node
                live[node] = live.get(node, 0) + 1

        total_workers = sum(live.values())
        assert total_workers > 1, f"Phase C must grow the saturated stage beyond its initial 1 worker; got live={live}"
        assert live.get("node-1", 0) >= 1, (
            f"Phase C must place at least one new worker on the previously-idle node-1; got live={live}"
        )

    def test_idle_node_does_not_starve_when_initial_workers_only_on_one_node(self) -> None:
        """Two stages saturated; placer uses node-1 once node-0 is full.

        Tracks every Phase C placement across 6 cycles and asserts
        that at least one new worker landed on the initially-idle
        node-1, proving the placer does not starve the alternate
        node when the saturated node already hosts a worker.
        """
        cluster = _multi_node_cluster(num_nodes=2, total_cpus_per_node=2)
        scheduler = _build_scheduler(["A", "B"], cluster)

        live: dict[str, dict[str, int]] = {
            "A": {"node-0": 1},
            "B": {"node-0": 1},
        }

        def layout() -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [
                    _stage_state(name=name, workers_per_node=live[name], used_slot_ratio=1.0, input_queue_depth=8)
                    for name in ("A", "B")
                ]
            )

        all_new_nodes: set[str] = set()
        for cycle_idx in range(6):
            sol = scheduler.autoscale(time=cycle_idx * 10.0, problem_state=layout())
            for stage_idx, stage_name in enumerate(("A", "B")):
                for new_worker in sol.stages[stage_idx].new_workers:
                    if new_worker.resources:
                        node = new_worker.resources[0].node
                        live[stage_name][node] = live[stage_name].get(node, 0) + 1
                        all_new_nodes.add(node)

        assert all_new_nodes, (
            f"expected at least one Phase C placement across 6 cycles with two saturated stages; final live={live}"
        )
        assert "node-1" in all_new_nodes, (
            f"node-1 must absorb growth once node-0 is full; new placements landed on {all_new_nodes}, "
            f"final live={live}"
        )


class TestClusterScaleInConsolidatesRemainingCapacity:
    """Over-provisioned signal under cluster shrinkage emits valid Phase D intent."""

    def test_cluster_scale_in_consolidates_remaining_capacity(self) -> None:
        """Workers spread across 2 nodes, node-1 vanishes, Phase D still respects floor + survivors.

        Drives two cycles with workers on both nodes at idle, then
        a third cycle where node-1's workers vanish. The Phase D
        decision on the third cycle must:

          * Not request more deletes than the surviving count.
          * Never emit a delete for a worker that already vanished.
          * Keep the live count at-or-above ``min_workers``.
        """
        cluster = _multi_node_cluster(num_nodes=2, total_cpus_per_node=4)
        scheduler = _build_scheduler(
            ["hot"],
            cluster,
            min_workers=1,
            over_provisioned_streak_min_cycles=2,
            stabilization_window_cycles_down=2,
        )

        ps_idle_full = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2}, used_slot_ratio=0.0)]
        )
        ps_idle_after_loss = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2}, used_slot_ratio=0.0)]
        )

        scheduler.autoscale(time=0.0, problem_state=ps_idle_full)
        scheduler.autoscale(time=10.0, problem_state=ps_idle_full)
        loss_solution = scheduler.autoscale(time=20.0, problem_state=ps_idle_after_loss)

        hot_solution = loss_solution.stages[0]
        survivors = 2
        floor = 1
        deleted_ids = {worker.id for worker in hot_solution.deleted_workers}

        assert len(hot_solution.deleted_workers) <= survivors - floor, (
            f"Phase D requested {len(hot_solution.deleted_workers)} deletes but only "
            f"{survivors - floor} survivors are eligible above floor={floor}"
        )

        lost_ids = {"hot-node-1-w0", "hot-node-1-w1"}
        assert lost_ids.isdisjoint(deleted_ids), (
            f"Phase D requested deletion of already-gone workers: {lost_ids & deleted_ids}"
        )

        live_after = survivors + len(hot_solution.new_workers) - len(hot_solution.deleted_workers)
        assert live_after >= floor, (
            f"post-loss live count must respect floor={floor}; got {live_after} "
            f"(survivors={survivors}, new={len(hot_solution.new_workers)}, "
            f"deleted={len(hot_solution.deleted_workers)})"
        )

        assert lost_ids.isdisjoint(scheduler._worker_ages), (
            f"node-1 ids leaked into _worker_ages after loss: {lost_ids & set(scheduler._worker_ages)}"
        )
        assert lost_ids.isdisjoint(scheduler._worker_ready_first_seen_at), (
            f"node-1 ids leaked into _worker_ready_first_seen_at: "
            f"{lost_ids & set(scheduler._worker_ready_first_seen_at)}"
        )
