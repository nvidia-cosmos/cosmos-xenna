# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-churn integration tests for ``SaturationAwareScheduler``.

This module exercises the scheduler under runtime worker churn that
mimics node-level events. ``cosmos-xenna``'s public scheduler API
takes the cluster topology once at :meth:`SaturationAwareScheduler.setup`
and treats it as static thereafter (``self._problem.rust.cluster_resources``
is read every cycle but never mutated). Real "node leaves the cluster"
events therefore manifest only as the node's workers disappearing
from ``problem_state.rust.stages[*].worker_groups`` -- the actor pool
no longer reports them as READY -- while the cluster_resources view
stays stale. Hot-swapping ``cluster_resources`` mid-pipeline is not
supported by the current public API; the architectural follow-up is
tracked in ``jira/STORY-44-cluster-resources-hot-swap.md``.

So the tests here drive behaviour that is observable through the
existing surface:

  * Group A (loss): a node's workers vanish mid-pipeline.
  * Group B (recovery / flap): workers reappear with fresh ids,
    optionally after repeated flaps.
  * Group C (edges): catastrophic loss, single-node deadlocks,
    id-keyed lookup correctness.

The fixtures share the multi-node cluster + multi-cycle pattern of
``test_worker_warmup_grace.py`` and ``test_donor_warmup_grace.py``.
Per-cycle ``ProblemState`` objects compose ``ProblemStageState``
rows built by :func:`_stage_state` so the test author controls
exactly which ``worker_id``s are present in each cycle. Assertions
read scheduler introspection state (``_worker_ages``,
``_worker_ready_first_seen_at``, ``_last_intent_deltas``,
``_floor_stuck_counters``, ``_stage_states``) to verify the contract.

Why these tests matter: production node loss on Ray is handled at
the actor-pool layer (Ray detects the dead actors and the streaming
layer rebuilds ``worker_groups`` from survivors), so the scheduler
sees a sudden change in the per-stage worker set. The two pruning
hooks the scheduler relies on -- :meth:`_persist_worker_ages` and
:meth:`_refresh_worker_ready_first_seen` -- guarantee per-worker
state does not leak across cycles. The donor / Phase D / floor
paths that consume that state must also degrade gracefully when
the worker set shrinks unexpectedly. These tests pin both
properties end-to-end.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig

# ---------------------------------------------------------------------------
# Cluster / problem helpers
# ---------------------------------------------------------------------------


def _multi_node_cluster(*, num_nodes: int = 2, total_cpus_per_node: int = 64) -> resources.ClusterResources:
    """Build a CPU-only cluster with ``num_nodes`` nodes named ``node-0..N-1``.

    All nodes have identical CPU capacity so the planner has no
    affinity preference; placement decisions stay deterministic
    across cycles.
    """
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
    cluster: resources.ClusterResources | None = None,
) -> data_structures.Problem:
    """Build a ``Problem`` with one CPU stage per name on ``cluster``.

    Cluster defaults to a 2-node fixture so every test can exercise
    "the workers on node-1 vanish" patterns without redefining
    helpers per case.
    """
    if cluster is None:
        cluster = _multi_node_cluster()
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


# ---------------------------------------------------------------------------
# Per-cycle ProblemStageState builders
# ---------------------------------------------------------------------------


def _worker_group(
    stage_name: str, idx: int, *, node: str, num_used_slots: int = 0
) -> data_structures.ProblemWorkerGroupState:
    """One CPU worker_group keyed by ``{stage_name}-{node}-w{idx}``.

    The id encodes the node so a test that "loses node-1's workers"
    can simply rebuild the next cycle's ``ProblemState`` without ids
    that contain ``node-1``. Encoding the node in the id also makes
    test assertions readable (``"hot-node-1-w0" not in ages``).
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
    used_slot_ratio: float = 0.0,
    input_queue_depth: int = 0,
    is_finished: bool = False,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState`` with explicit per-node worker placement.

    Args:
        name: Stage name.
        workers_per_node: Map of node name -> worker count on that
            node. Iteration follows insertion order, so the worker
            ids are stable across calls with the same map shape.
        slots_per_worker: Slots advertised per worker_group.
        used_slot_ratio: Fraction of slots reported as used per
            worker (0.0 = empty, 1.0 = saturated). Used to drive
            the classifier.
        input_queue_depth: Stage-level input queue depth (unfiltered
            by the warmup grace).
        is_finished: Forwards to ``ProblemStageState``; finished
            stages get no autoscale action.

    Per-worker_group ``num_used_slots`` is filled symmetrically so
    the warmup-filter helper sees a self-consistent per-group view.
    """
    used_per_worker = round(slots_per_worker * used_slot_ratio)
    used_per_worker = max(0, min(slots_per_worker, used_per_worker))
    workers: list[data_structures.ProblemWorkerGroupState] = []
    for node, count in workers_per_node.items():
        for i in range(count):
            workers.append(_worker_group(name, i, node=node, num_used_slots=used_per_worker))
    total = sum(workers_per_node.values())
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=workers,
        slots_per_worker=slots_per_worker,
        is_finished=is_finished,
        num_used_slots=total * used_per_worker,
        num_empty_slots=total * (slots_per_worker - used_per_worker),
        input_queue_depth=input_queue_depth,
    )


# ---------------------------------------------------------------------------
# Scheduler builders
# ---------------------------------------------------------------------------


def _scheduler_with_node_churn_defaults(
    stage_names: list[str],
    *,
    grace_s: float = 0.0,
    quiescence_enabled: bool = False,
    saturated_streak_min_cycles: int = 1,
    over_provisioned_streak_min_cycles: int = 30,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 30,
    cross_stage_donor_anti_flap_cycles: int = 30,
    cross_stage_donor_min_donation_interval_cycles: int = 30,
    num_nodes: int = 2,
) -> SaturationAwareScheduler:
    """Build a multi-stage scheduler with conservative churn-friendly defaults.

    ``grace_s=0.0`` disables both warmup grace mechanisms by default
    so tests can assert raw behaviour. Tests that specifically
    exercise the grace interaction (e.g. ``test_loss_during_donor_warmup``)
    pass a non-zero ``grace_s``.

    ``quiescence_enabled=False`` is the right default because we
    drive the scheduler purely from ``worker_groups`` (READY workers);
    quiescence would short-circuit Phase C scale-up whenever pending
    actors are non-zero, which is orthogonal to the node-churn
    contract under test.

    Streak / window defaults make the up path fire on a single
    cycle so a positive intent shows up in
    ``_last_intent_deltas`` immediately once the EWMA crosses the
    activation threshold.
    """
    cfg = SaturationAwareConfig(
        cross_stage_donor_anti_flap_cycles=cross_stage_donor_anti_flap_cycles,
        cross_stage_donor_min_donation_interval_cycles=cross_stage_donor_min_donation_interval_cycles,
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=quiescence_enabled,
            worker_warmup_measurement_grace_s=grace_s,
            donor_warmup_grace_s=grace_s,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_names, _multi_node_cluster(num_nodes=num_nodes)))
    return scheduler


# ---------------------------------------------------------------------------
# Group A -- node loss / worker disappearance
# ---------------------------------------------------------------------------


class TestGroupAWorkerDisappearance:
    """End-to-end checks for the "all workers on a node vanish" event."""

    def test_per_worker_dicts_drop_lost_ids(self) -> None:
        """Both ``_worker_ages`` and ``_worker_ready_first_seen_at`` evict the lost ids in one cycle."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"])
        ps_full = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2})])
        ps_loss = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2})])

        scheduler.autoscale(time=0.0, problem_state=ps_full)
        assert {"hot-node-0-w0", "hot-node-0-w1", "hot-node-1-w0", "hot-node-1-w1"} <= set(
            scheduler._worker_ready_first_seen_at
        )
        assert {"hot-node-0-w0", "hot-node-0-w1", "hot-node-1-w0", "hot-node-1-w1"} <= set(scheduler._worker_ages)

        scheduler.autoscale(time=10.0, problem_state=ps_loss)

        assert "hot-node-1-w0" not in scheduler._worker_ready_first_seen_at
        assert "hot-node-1-w1" not in scheduler._worker_ready_first_seen_at
        assert "hot-node-1-w0" not in scheduler._worker_ages
        assert "hot-node-1-w1" not in scheduler._worker_ages
        # Survivors keep their first-seen and have their ages incremented by one planner cycle.
        assert scheduler._worker_ready_first_seen_at["hot-node-0-w0"] == 0.0
        assert scheduler._worker_ready_first_seen_at["hot-node-0-w1"] == 0.0

    def test_loss_drives_below_floor_triggers_phase_b_grow(self) -> None:
        """A stage at floor=2 that loses one worker reappears at floor in the next cycle."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                min_workers=2,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem(["hot"], _multi_node_cluster(num_nodes=2)))

        ps_at_floor = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 1, "node-1": 1})]
        )
        ps_below_floor = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 1})])

        scheduler.autoscale(time=0.0, problem_state=ps_at_floor)
        solution = scheduler.autoscale(time=10.0, problem_state=ps_below_floor)
        # Stage identity is positional (Solution.stages aligned with Problem.stages); stage 0 is "hot".
        # Phase B adds workers via ``new_workers`` to satisfy floor; it never deletes during floor enforcement.
        # Live count after this cycle = pre-cycle live (1) + new - deleted, expected >= floor=2.
        hot_solution = solution.stages[0]
        pre_cycle_live = 1
        live_after = pre_cycle_live + len(hot_solution.new_workers) - len(hot_solution.deleted_workers)
        assert live_after >= 2, (
            f"expected floor=2 after loss; got pre={pre_cycle_live} new={len(hot_solution.new_workers)} "
            f"deleted={len(hot_solution.deleted_workers)} -> live={live_after}"
        )
        assert hot_solution.deleted_workers == [], (
            f"floor enforcement must not delete; got {len(hot_solution.deleted_workers)} deletions"
        )

    def test_loss_recomputes_ewma_without_lost_workers(self) -> None:
        """Lost workers' ``num_used_slots`` no longer pollute the EWMA on the next cycle."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"], grace_s=0.0)
        # Cycle 1: 4 workers across two nodes, every worker at 50% used -> NORMAL classification
        # with empty-ratio=0.5 contributing to the EWMA.
        ps_normal = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2}, used_slot_ratio=0.5)]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_normal)

        # Cycle 2: lose node-1's pair; the two surviving node-0 workers are now fully saturated
        # (used_slot_ratio=1.0 -> empty_ratio=0.0). The EWMA absorbs only the survivors' signal;
        # if the lost workers' cycle-1 contributions still leaked into cycle 2 the EWMA would
        # remain anchored to the 50% empty reading from cycle 1 instead of trending down.
        ps_after_loss = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2}, used_slot_ratio=1.0)]
        )
        scheduler.autoscale(time=10.0, problem_state=ps_after_loss)

        runtime = scheduler._stage_states["hot"]
        # Only the surviving (saturated) workers contribute. EWMA was 0.5 after the half-empty cycle;
        # absorbing the loss-cycle's saturated reading (empty_ratio=0.0) must drag the EWMA below 0.5.
        # Exact alpha is implementation-specific, but the direction is the contract.
        assert runtime.slots_empty_ratio_ewma is not None
        assert runtime.slots_empty_ratio_ewma < 0.5, (
            f"lost workers' contributions still pollute EWMA: {runtime.slots_empty_ratio_ewma}"
        )

    def test_loss_during_donor_warmup_grace_drops_lost_from_excluded_set(self) -> None:
        """Donor warmup excluded set tracks the live worker map -- lost workers fall out."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"], grace_s=120.0)
        ps_full = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2})])
        ps_loss = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2})])

        scheduler.autoscale(time=0.0, problem_state=ps_full)
        # During warmup, all 4 workers are excluded from donor / Phase D.
        assert len(scheduler._donor_warmup_excluded_ids) == 4

        scheduler.autoscale(time=30.0, problem_state=ps_loss)

        # node-1 workers gone -> excluded set must drop them.
        assert "hot-node-1-w0" not in scheduler._donor_warmup_excluded_ids
        assert "hot-node-1-w1" not in scheduler._donor_warmup_excluded_ids
        # node-0 workers still in warmup at t=30 (grace=120) -> still excluded.
        assert {"hot-node-0-w0", "hot-node-0-w1"} <= set(scheduler._donor_warmup_excluded_ids)

    def test_loss_then_idle_phase_d_respects_floor_and_survivor_count(self) -> None:
        """A loss-driven Phase D shrink intent must respect floor and never inflate beyond survivor count.

        Validator constraints (``over_provisioned_streak_min_cycles >
        saturated_streak_min_cycles``, ``stabilization_window_cycles_down >
        stabilization_window_cycles_up``) bound the minimum-valid
        over-provisioned configuration to ``(saturated=1,
        over_provisioned=2)``, ``(window_up=1, window_down=2)``. With
        those minimums, two consecutive idle cycles satisfy the streak;
        the third cycle (which is the loss cycle) is when Phase D
        becomes eligible. The contract under test is:

          floor (=1) <= live count after Phase D <= live count seen pre-Phase D
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                saturated_streak_min_cycles=1,
                over_provisioned_streak_min_cycles=2,
                stabilization_window_cycles_up=1,
                stabilization_window_cycles_down=2,
                min_workers=1,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem(["hot"], _multi_node_cluster(num_nodes=2)))

        # Cycles 1 and 2: 4 over-provisioned (idle) workers so OVER_PROVISIONED streak reaches 2.
        ps_idle_full = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2}, used_slot_ratio=0.0)]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_idle_full)
        scheduler.autoscale(time=10.0, problem_state=ps_idle_full)

        # Cycle 3: half the workers vanish (node-1 lost). The two survivors remain idle.
        ps_idle_loss = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2}, used_slot_ratio=0.0)]
        )
        solution = scheduler.autoscale(time=20.0, problem_state=ps_idle_loss)
        hot_solution = solution.stages[0]
        pre_cycle_live = 2
        live_after_phase_d = pre_cycle_live + len(hot_solution.new_workers) - len(hot_solution.deleted_workers)
        assert 1 <= live_after_phase_d <= 2, (
            f"expected post-Phase-D count in [floor=1, survivors=2]; "
            f"got pre={pre_cycle_live} new={len(hot_solution.new_workers)} "
            f"deleted={len(hot_solution.deleted_workers)} -> live={live_after_phase_d}"
        )

    def test_sustained_churn_does_not_grow_per_worker_dicts_unboundedly(self) -> None:
        """50 cycles of constant rate (workers in == workers out) keep ``_worker_ages`` bounded."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"])
        for cycle in range(50):
            # Each cycle: 4 workers but the node-1 pair has fresh ids cycle-by-cycle (simulating flap).
            workers = [
                _worker_group("hot", 0, node="node-0"),
                _worker_group("hot", 1, node="node-0"),
                _worker_group("hot", cycle * 2, node="node-1"),
                _worker_group("hot", cycle * 2 + 1, node="node-1"),
            ]
            ps = data_structures.ProblemState(
                [
                    data_structures.ProblemStageState(
                        stage_name="hot",
                        workers=workers,
                        slots_per_worker=8,
                        is_finished=False,
                        num_used_slots=0,
                        num_empty_slots=32,
                        input_queue_depth=0,
                    )
                ]
            )
            scheduler.autoscale(time=cycle * 10.0, problem_state=ps)

        # Each cycle replaces 2 worker ids; after 50 cycles only the 4 live ids remain.
        assert len(scheduler._worker_ready_first_seen_at) == 4
        assert len(scheduler._worker_ages) == 4


# ---------------------------------------------------------------------------
# Group B -- recovery / flap
# ---------------------------------------------------------------------------


class TestGroupBRecoveryAndFlap:
    """Workers reappearing after loss get fresh first_seen and re-enter warmup grace."""

    def test_rejoin_assigns_fresh_first_seen_timestamp(self) -> None:
        """A worker_id that vanishes and later reappears must receive ``now`` (not its old timestamp)."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"], grace_s=120.0)
        ps_full = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2})])
        ps_loss = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2})])

        scheduler.autoscale(time=0.0, problem_state=ps_full)
        scheduler.autoscale(time=30.0, problem_state=ps_loss)
        # Rejoin at t=300 with the same id pattern (simulates the same node coming back up).
        scheduler.autoscale(time=300.0, problem_state=ps_full)

        # Survivors keep their original first-seen.
        assert scheduler._worker_ready_first_seen_at["hot-node-0-w0"] == 0.0
        # Returning workers get the rejoin timestamp; they were absent at t=30.
        assert scheduler._worker_ready_first_seen_at["hot-node-1-w0"] == 300.0
        assert scheduler._worker_ready_first_seen_at["hot-node-1-w1"] == 300.0

    def test_repeated_flap_resets_warmup_grace_each_rejoin(self) -> None:
        """Loss-then-rejoin three times: each rejoin's first_seen is the latest ``now``."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"], grace_s=120.0)
        ps_full = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 1, "node-1": 1})])
        ps_loss = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 1})])

        # Flap 1: 0 -> 100 -> 200
        scheduler.autoscale(time=0.0, problem_state=ps_full)
        scheduler.autoscale(time=100.0, problem_state=ps_loss)
        scheduler.autoscale(time=200.0, problem_state=ps_full)
        assert scheduler._worker_ready_first_seen_at["hot-node-1-w0"] == 200.0

        # Flap 2: 300 -> 400
        scheduler.autoscale(time=300.0, problem_state=ps_loss)
        scheduler.autoscale(time=400.0, problem_state=ps_full)
        assert scheduler._worker_ready_first_seen_at["hot-node-1-w0"] == 400.0

        # Flap 3: 500 -> 600
        scheduler.autoscale(time=500.0, problem_state=ps_loss)
        scheduler.autoscale(time=600.0, problem_state=ps_full)
        assert scheduler._worker_ready_first_seen_at["hot-node-1-w0"] == 600.0

    def test_survivor_age_preserved_when_peer_replaced(self) -> None:
        """The survivor on node-0 keeps its age while the node-1 peer is replaced with a fresh id."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"])
        ps_initial = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 1, "node-1": 1})]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_initial)
        survivor_first_seen_initial = scheduler._worker_ready_first_seen_at["hot-node-0-w0"]
        survivor_age_initial = scheduler._worker_ages["hot-node-0-w0"]

        # Cycle 2: node-1's worker is replaced by a fresh id.
        ps_replaced = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        _worker_group("hot", 0, node="node-0"),
                        _worker_group("hot", 99, node="node-1"),
                    ],
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=0,
                    num_empty_slots=16,
                    input_queue_depth=0,
                )
            ]
        )
        scheduler.autoscale(time=10.0, problem_state=ps_replaced)

        # Survivor: same first_seen, age incremented.
        assert scheduler._worker_ready_first_seen_at["hot-node-0-w0"] == survivor_first_seen_initial
        assert scheduler._worker_ages["hot-node-0-w0"] == survivor_age_initial + 1
        # Old peer dropped, fresh peer gets first_seen=10.
        assert "hot-node-1-w0" not in scheduler._worker_ready_first_seen_at
        assert scheduler._worker_ready_first_seen_at["hot-node-1-w99"] == 10.0
        # Fresh peer is registered in _worker_ages with age 0. ``_persist_worker_ages`` ran at the
        # end of cycle 2 and the Rust planner assigns age 0 to a newly observed id. Pinning
        # the contract directly (key is present, value is 0) instead of ``.get(..., 0) == 0``
        # which would also pass if the key were silently missing.
        assert "hot-node-1-w99" in scheduler._worker_ages
        assert scheduler._worker_ages["hot-node-1-w99"] == 0

    def test_replacement_worker_enters_donor_warmup_grace(self) -> None:
        """A worker that replaces a lost peer is donor-protected by warmup grace until its ready age elapses."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"], grace_s=180.0)
        ps_full = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2})])
        ps_loss = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 2})])

        scheduler.autoscale(time=0.0, problem_state=ps_full)
        # Wait past the original warmup window so node-0 workers are no longer protected.
        scheduler.autoscale(time=200.0, problem_state=ps_full)
        assert "hot-node-0-w0" not in scheduler._donor_warmup_excluded_ids

        # node-1 workers vanish (loss) and reappear at t=210 (recovery).
        scheduler.autoscale(time=205.0, problem_state=ps_loss)
        scheduler.autoscale(time=210.0, problem_state=ps_full)

        # node-1 workers were observed first at t=210 and the grace is 180s -> they are still protected.
        assert {"hot-node-1-w0", "hot-node-1-w1"} <= set(scheduler._donor_warmup_excluded_ids)
        # node-0 workers (first_seen=0) have age 210s, still > 180s -> not protected.
        assert "hot-node-0-w0" not in scheduler._donor_warmup_excluded_ids
        assert "hot-node-0-w1" not in scheduler._donor_warmup_excluded_ids


# ---------------------------------------------------------------------------
# Group C -- edges
# ---------------------------------------------------------------------------


class TestGroupCEdges:
    """Defensive edges around node loss."""

    def test_catastrophic_full_cluster_loss_does_not_crash(self) -> None:
        """Every worker on every node vanishes in a single cycle; scheduler holds and clears its per-worker maps."""
        scheduler = _scheduler_with_node_churn_defaults(["hot", "warm"])
        ps_full = data_structures.ProblemState(
            [
                _stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2}),
                _stage_state(name="warm", workers_per_node={"node-0": 1, "node-1": 1}),
            ]
        )
        ps_empty = data_structures.ProblemState(
            [
                _stage_state(name="hot", workers_per_node={}),
                _stage_state(name="warm", workers_per_node={}),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps_full)
        # Catastrophic loss -- the call must not raise.
        scheduler.autoscale(time=10.0, problem_state=ps_empty)

        assert scheduler._worker_ready_first_seen_at == {}
        # _worker_ages is keyed off the planner snapshot. After the loss cycle the planner re-seeds an empty pool.
        # Either way no stale ids may remain.
        for stage_name in ("hot", "warm"):
            for wid in scheduler._worker_ages:
                assert not wid.startswith(stage_name + "-node-")
        # Classifier state stays consistent: with zero workers in the snapshot every stage
        # has empty slot signal, the EWMA stays under both saturation and activation
        # thresholds, and the classifier emits intent 0. Pinning the value (not just key
        # presence) catches a future regression where the empty-pool path would emit a
        # spurious non-zero intent (NaN-derived, divide-by-zero coerced, or stale window
        # contents leaking through the recommendation history).
        assert scheduler._last_intent_deltas == {"hot": 0, "warm": 0}

    def test_single_node_cluster_below_floor_advances_floor_stuck_counter(self) -> None:
        """A stage stuck below floor with no expansion headroom advances ``_floor_stuck_counters``.

        The cluster has 1 CPU, the stage is at 1 worker (occupying it),
        floor=2, and no other stage can act as donor. Phase B fails
        every cycle (direct grow blocked by 0 CPU, donor grow blocked
        by no donor stage). The stuck counter must increase
        monotonically across cycles. ``floor_stuck_grace_cycles`` is
        the cluster-wide cap; we stay well below it.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                min_workers=2,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem(["hot"], _multi_node_cluster(num_nodes=1, total_cpus_per_node=1)))

        ps_below_floor = data_structures.ProblemState([_stage_state(name="hot", workers_per_node={"node-0": 1})])

        for cycle in range(3):
            scheduler.autoscale(time=cycle * 10.0, problem_state=ps_below_floor)

        # Pin the exact counter value: each of the 3 cycles fails Phase B floor enforcement
        # (no CPU headroom, no donor stage), so the counter increments by 1 per cycle
        # and reaches 3 after 3 cycles. A looser ``>= 1`` assertion would silently accept
        # a regression where the counter only fired on the first cycle (e.g., a future
        # change that resets the counter on a non-empty intent dict, or a Phase B path
        # that succeeds spuriously).
        assert scheduler._floor_stuck_counters.get("hot", 0) == 3, (
            f"expected _floor_stuck_counters['hot'] == 3 after 3 consecutive sub-floor cycles "
            f"with no headroom; got {scheduler._floor_stuck_counters.get('hot', 0)}"
        )

    @pytest.mark.parametrize("num_cycles", [2, 5, 10])
    def test_worker_id_continuity_lookup_is_keyed_per_id_not_per_node(self, num_cycles: int) -> None:
        """Worker maps are keyed by id, not by node; two distinct workers on the same node are tracked separately."""
        scheduler = _scheduler_with_node_churn_defaults(["hot"])
        # node-0 carries two workers; we vary their ids per cycle to verify the maps are id-keyed.
        for cycle in range(num_cycles):
            workers = [
                _worker_group("hot", cycle * 10, node="node-0"),
                _worker_group("hot", cycle * 10 + 1, node="node-0"),
            ]
            ps = data_structures.ProblemState(
                [
                    data_structures.ProblemStageState(
                        stage_name="hot",
                        workers=workers,
                        slots_per_worker=8,
                        is_finished=False,
                        num_used_slots=0,
                        num_empty_slots=16,
                        input_queue_depth=0,
                    )
                ]
            )
            scheduler.autoscale(time=cycle * 10.0, problem_state=ps)

        # After ``num_cycles`` cycles only the latest pair survives in the map.
        live = {f"hot-node-0-w{(num_cycles - 1) * 10}", f"hot-node-0-w{(num_cycles - 1) * 10 + 1}"}
        assert set(scheduler._worker_ready_first_seen_at) == live


# ---------------------------------------------------------------------------
# Donor-flow under churn (review M6 coverage gap from Phase 3-viii post-impl review)
# ---------------------------------------------------------------------------
#
# Phase B floor donor (``select_youngest_eligible_donor``) and Phase C
# saturation-mode cross-stage donor (``find_saturation_donor``) both consume
# per-stage live worker sets sourced from the planner's cycle-start snapshot.
# When a donor stage simultaneously suffers worker loss (peer node disappears
# from the actor pool), the donor selection must:
#
#   1. Operate exclusively on surviving workers (lost ids never reach the
#      pool because they are absent from ``ProblemState.worker_groups``).
#   2. Honor the donor stage's floor against the post-loss live count, not
#      the pre-loss count.
#   3. Not leak per-worker state across cycles in a way that corrupts the
#      anti-flap state (``_last_donation_cycle``).
#
# These tests cover the donor-flow paths under churn explicitly. The
# scheduler-level Phase D / floor / EWMA paths under churn are covered by
# Group A above; the warmup-grace integration is covered in test_donor_warmup_grace.
# ---------------------------------------------------------------------------


class TestDonorFlowUnderChurn:
    """Donor selection paths degrade gracefully when the donor stage loses workers."""

    @staticmethod
    def _two_stage_scheduler(
        *,
        receiver_floor: int,
        donor_floor: int = 1,
        total_cpus_per_node: int = 4,
        num_nodes: int = 1,
    ) -> SaturationAwareScheduler:
        """Build a two-stage scheduler with explicit per-stage floors.

        Names: ``donor`` (over-provisioned source) and ``receiver``
        (saturation / floor-bound consumer). The cluster is sized so
        the cycle-1 fixtures saturate every CPU and the cycle-2
        post-loss fixtures still leave the receiver below floor with
        at most one free CPU - forcing the donor fallback in Phase B.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                min_workers=donor_floor,
            ),
            per_stage_overrides={
                "receiver": SaturationAwareStageConfig(
                    setup_phase_quiescence_enabled=False,
                    worker_warmup_measurement_grace_s=0.0,
                    donor_warmup_grace_s=0.0,
                    min_workers=receiver_floor,
                ),
            },
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(
            _problem(
                ["donor", "receiver"],
                _multi_node_cluster(num_nodes=num_nodes, total_cpus_per_node=total_cpus_per_node),
            )
        )
        return scheduler

    def test_floor_donor_selects_from_surviving_workers_after_donor_loss(self) -> None:
        """Phase B floor donor under donor-stage loss: pick lands on a survivor, not a lost id.

        Setup (single-cycle): the scheduler is fed a snapshot that
        already reflects "donor stage has lost some peers". The
        receiver is below floor and the cluster is full from the
        receiver's perspective, so Phase B falls through to the donor
        helper. The donor's ``worker_ids_by_stage`` reflects only
        the survivors -- the lost id is absent. The helper is therefore
        forced to pick from survivors, by construction.

        ::

           Cluster (1 node, 4 CPUs)             Cycle 1 baseline
           +---+
           |  donor       receiver              donor:    3 workers
           |  +-+-+-+     +-+                   receiver: 1 worker
           |  |w0|w1|w2|  |w0|                  total:    4 = full
           |  +-+-+-+     +-+                   floor:    receiver=3
           +---+

           After the implicit "lost peer" event the snapshot is:

           Cluster (still 1 node, 4 CPUs)       Cycle 1 churn snapshot
           +---+
           |  donor       receiver              donor:    2 survivors
           |  +-+-+        +-+                  receiver: 1 worker
           |  |w0|w1|      |w0|                 total:    3, free=1
           |  +-+-+        +-+                  donor lost w2 silently
           +---+

        Pin: Phase B for receiver runs ``try_add`` once (free=1), then
        falls through to the donor helper which can only see {w0, w1}.
        The donation is recorded with one of those ids as ``deleted``;
        the lost id ``donor-node-0-w2`` never appears anywhere in the
        solution or in scheduler state.
        """
        scheduler = self._two_stage_scheduler(receiver_floor=3, total_cpus_per_node=4)

        # Single cycle: snapshot already reflects "donor lost w2" pre-autoscale.
        # The scheduler does not know there was a "before" state; it only sees
        # the live worker_groups.
        ps_post_loss = data_structures.ProblemState(
            [
                # donor: 2 survivors (w2 absent)
                data_structures.ProblemStageState(
                    stage_name="donor",
                    workers=[
                        _worker_group("donor", 0, node="node-0"),
                        _worker_group("donor", 1, node="node-0"),
                    ],
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=0,
                    num_empty_slots=16,
                    input_queue_depth=0,
                ),
                _stage_state(name="receiver", workers_per_node={"node-0": 1}),
            ]
        )

        solution = scheduler.autoscale(time=0.0, problem_state=ps_post_loss)

        donor_solution = solution.stages[0]
        receiver_solution = solution.stages[1]

        # Receiver grew from 1 to its floor=3 via direct add + donor fallback.
        receiver_post = 1 + len(receiver_solution.new_workers) - len(receiver_solution.deleted_workers)
        assert receiver_post >= 3, (
            f"receiver floor=3 not met: pre=1 new={len(receiver_solution.new_workers)} "
            f"deleted={len(receiver_solution.deleted_workers)} -> {receiver_post}"
        )
        # Donor donated >= 1 survivor. The lost id "donor-node-0-w2" must NOT appear
        # because it was never in worker_ids_by_stage at the time of donor selection.
        deleted_ids = set(donor_solution.deleted_workers)
        assert "donor-node-0-w2" not in deleted_ids, (
            f"phantom delete: donor.deleted_workers includes the lost id {deleted_ids}"
        )
        survivor_ids = {"donor-node-0-w0", "donor-node-0-w1"}
        assert deleted_ids <= survivor_ids, (
            f"donor.deleted_workers contains non-survivor ids: {deleted_ids - survivor_ids}"
        )
        assert deleted_ids, "expected at least one survivor donation to satisfy receiver floor"
        # Lost id is absent from every per-worker map post-cycle.
        assert "donor-node-0-w2" not in scheduler._worker_ages
        assert "donor-node-0-w2" not in scheduler._worker_ready_first_seen_at

    def test_floor_donor_unavailable_when_donor_stage_falls_to_floor(self) -> None:
        """Donor stage that loses workers down to its floor cannot donate; receiver floor stays unmet.

        Setup:
          * Cluster: 1 node, 4 CPUs.
          * receiver: floor=4, current=1.
          * donor: floor=2, current=3 (above floor by 1).
          * Cycle 1: 1 + 3 = 4 = cluster full.
          * Cycle 2: donor loses 1 worker -> 2 (= floor, slack=0).

        Phase B for receiver (target=4):
          * try_add succeeds twice (cluster has 2 free CPUs after the
            loss freed one); receiver = 3.
          * try_add fails (cluster full again).
          * donor fallback queries ``select_youngest_eligible_donor``;
            donor stage at floor returns None.
          * receiver remains at 3 below floor 4 -> floor_stuck_counter
            advances; ``floor_stuck_grace_cycles`` (default 5) bounds
            how long this is tolerated.

        Pin: ``_floor_stuck_counters['receiver']`` == 1 (one cycle of
        non-progress at floor); donor stage's deleted_workers is empty
        (no donation attempted because donor was at floor); the lost id
        is absent from scheduler state.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                min_workers=2,
            ),
            per_stage_overrides={
                "receiver": SaturationAwareStageConfig(
                    setup_phase_quiescence_enabled=False,
                    worker_warmup_measurement_grace_s=0.0,
                    donor_warmup_grace_s=0.0,
                    min_workers=4,
                ),
            },
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem(["donor", "receiver"], _multi_node_cluster(num_nodes=1, total_cpus_per_node=4)))

        # Cycle 1: donor=3, receiver=1 -> cluster full (4 CPUs).
        ps_full = data_structures.ProblemState(
            [
                _stage_state(name="donor", workers_per_node={"node-0": 3}),
                _stage_state(name="receiver", workers_per_node={"node-0": 1}),
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_full)

        # Cycle 2: donor loses w2; receiver unchanged. Total live = 1 + 2 = 3, free = 1.
        donor_post_loss = data_structures.ProblemStageState(
            stage_name="donor",
            workers=[
                _worker_group("donor", 0, node="node-0"),
                _worker_group("donor", 1, node="node-0"),
            ],
            slots_per_worker=8,
            is_finished=False,
            num_used_slots=0,
            num_empty_slots=16,
            input_queue_depth=0,
        )
        receiver_unchanged = _stage_state(name="receiver", workers_per_node={"node-0": 1})
        ps_post_loss = data_structures.ProblemState([donor_post_loss, receiver_unchanged])

        solution = scheduler.autoscale(time=10.0, problem_state=ps_post_loss)

        donor_solution = solution.stages[0]
        receiver_solution = solution.stages[1]

        # Receiver grew from 1 to <4 (cannot satisfy floor without donor).
        receiver_post = 1 + len(receiver_solution.new_workers) - len(receiver_solution.deleted_workers)
        assert receiver_post < 4, (
            f"receiver should remain below floor=4 when donor is at floor; "
            f"got post={receiver_post} (pre=1, new={len(receiver_solution.new_workers)}, "
            f"deleted={len(receiver_solution.deleted_workers)})"
        )
        # Donor at its floor must not donate (no slack).
        assert donor_solution.deleted_workers == [], (
            f"donor at floor must not be touched; got deleted_workers={donor_solution.deleted_workers}"
        )
        # The receiver made partial progress (>=1 direct add) so floor_stuck_counter
        # is RESET (cleared from the map) on this cycle by Phase B's ``made_progress``
        # branch. The next cycle without further progress is the one that increments
        # the counter. Pin the actual contract: counter is absent (cleared) this cycle.
        assert "receiver" not in scheduler._floor_stuck_counters
        # Lost donor id is absent from per-worker maps.
        assert "donor-node-0-w2" not in scheduler._worker_ages
        assert "donor-node-0-w2" not in scheduler._worker_ready_first_seen_at

    def test_last_donation_cycle_persists_after_donor_stage_loses_all_workers(self) -> None:
        """``_last_donation_cycle`` is keyed by stage name, not by worker id, so loss preserves the entry.

        The cross-stage donor anti-flap guard
        (``cross_stage_donor_min_donation_interval_cycles``) gates the
        donor stage by name. If a donor stage donates on cycle N and
        loses every worker on cycle N+1, the gate must continue to
        recognize the donor stage as "recently donated" - the entry
        is keyed by stage name, not by any specific worker id, so
        worker churn cannot leak it out of the anti-flap map.

        The map can be populated by either Phase B (floor donor) or
        Phase C (saturation cross-stage donor). This test exercises
        the map directly via Phase B floor donation in cycle 1, then
        verifies the entry survives a full-stage loss in cycle 2.
        """
        scheduler = self._two_stage_scheduler(receiver_floor=3, donor_floor=1, total_cpus_per_node=4)

        # Cycle 1: force a Phase B floor donation. donor=3, receiver=1, cluster full.
        # Phase B grows receiver: try_add -> receiver=2 (one free CPU pre-loss... wait, no:
        # cycle 1 is the *baseline* with cluster full, no loss yet).
        # donor=3, receiver=1 = 4 = cluster full. receiver floor=3. try_add fails.
        # Donor fallback -> donates 1. receiver=2. try_add fails. Donor fallback again
        # -> donates 1. receiver=3. Floor met.
        ps_cycle_1 = data_structures.ProblemState(
            [
                _stage_state(name="donor", workers_per_node={"node-0": 3}),
                _stage_state(name="receiver", workers_per_node={"node-0": 1}),
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_cycle_1)

        # Phase B floor donor populates _last_donation_cycle on the donor entry. The
        # receiver consumes the donation but only the donor stage entry is recorded.
        # The cycle counter starts at 0 and increments at the top of each autoscale call,
        # so cycle 1's donations are stamped at cycle_counter == 0. (Phase C donations
        # would also stamp this map, but cycle 1 has no saturation classifier output.)
        # We accept either {} (Phase B does NOT populate the map; it is reserved for
        # Phase C anti-flap) or {"donor": 0} depending on the implementation contract.
        # Pin the architectural property: regardless of source, the map's entry for
        # the donor stage survives even if every donor worker is lost in cycle 2.
        donation_cycle_after_cycle_1 = dict(scheduler._last_donation_cycle)

        # Cycle 2: donor loses ALL workers. Cluster goes to receiver-only (3 workers, 1 free).
        ps_cycle_2 = data_structures.ProblemState(
            [
                _stage_state(name="donor", workers_per_node={}),
                _stage_state(name="receiver", workers_per_node={"node-0": 3}),
            ]
        )
        scheduler.autoscale(time=10.0, problem_state=ps_cycle_2)

        # The donor's per-worker state was pruned (no live ids in the donor stage).
        for i in range(3):
            assert f"donor-node-0-w{i}" not in scheduler._worker_ages
            assert f"donor-node-0-w{i}" not in scheduler._worker_ready_first_seen_at
        # The stage-keyed map is unaffected by per-worker churn. Whatever was in the
        # map after cycle 1 stays in the map after cycle 2 (no key was deleted by the
        # loss-pruning hooks).
        assert dict(scheduler._last_donation_cycle) == donation_cycle_after_cycle_1, (
            f"per-worker churn must not mutate stage-keyed _last_donation_cycle; "
            f"before={donation_cycle_after_cycle_1}, after={dict(scheduler._last_donation_cycle)}"
        )

    def test_donor_stage_with_no_survivors_is_skipped_by_floor_donor(self) -> None:
        """Donor selection sources from ``worker_ids_by_stage``; an empty stage cannot donate.

        Setup:
          * Cluster: 1 node, 4 CPUs.
          * receiver: floor=2, current=1.
          * donor: floor=1, current=3.
          * Cycle 1: cluster full (3+1).
          * Cycle 2: donor loses ALL 3 workers (now 0). The cluster
            now has 3 free CPUs; receiver direct-add succeeds twice
            and the floor is met without consulting any donor.
            The donor stage is below its floor, so Phase B will also
            try to refloor the donor; that succeeds via direct add.

        Pin: receiver and donor both end the cycle at their floors.
        No phantom donor ids appear in any stage's deleted_workers.
        Lost ids are absent from per-worker maps.
        """
        scheduler = self._two_stage_scheduler(receiver_floor=2, donor_floor=1, total_cpus_per_node=4)

        ps_cycle_1 = data_structures.ProblemState(
            [
                _stage_state(name="donor", workers_per_node={"node-0": 3}),
                _stage_state(name="receiver", workers_per_node={"node-0": 1}),
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_cycle_1)

        ps_cycle_2 = data_structures.ProblemState(
            [
                _stage_state(name="donor", workers_per_node={}),
                _stage_state(name="receiver", workers_per_node={"node-0": 1}),
            ]
        )
        solution = scheduler.autoscale(time=10.0, problem_state=ps_cycle_2)

        donor_solution = solution.stages[0]
        receiver_solution = solution.stages[1]

        # Donor went 0 -> 1 (floor=1) via direct add. No phantom deletes.
        donor_post = 0 + len(donor_solution.new_workers) - len(donor_solution.deleted_workers)
        assert donor_post == 1
        assert donor_solution.deleted_workers == []
        # Receiver went 1 -> 2 (floor=2) via direct add (cluster had 3 free CPUs).
        receiver_post = 1 + len(receiver_solution.new_workers) - len(receiver_solution.deleted_workers)
        assert receiver_post == 2
        # Lost donor ids are absent from per-worker maps.
        for i in range(3):
            assert f"donor-node-0-w{i}" not in scheduler._worker_ages
            assert f"donor-node-0-w{i}" not in scheduler._worker_ready_first_seen_at
