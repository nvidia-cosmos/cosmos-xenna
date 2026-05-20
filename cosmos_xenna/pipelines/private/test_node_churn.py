# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-churn integration tests for ``SaturationAwareScheduler``.

This module exercises the scheduler under runtime worker churn that
mimics node-level events. ``cosmos-xenna``'s public scheduler API
takes the cluster topology once at :meth:`SaturationAwareScheduler.setup`
and treats it as static thereafter (``self._problem.cluster_resources``
is read every cycle but never mutated). Real "node leaves the cluster"
events therefore manifest only as the node's workers disappearing
from ``problem_state.rust.stages[*].worker_groups`` -- the actor pool
no longer reports them as READY -- while the cluster_resources view
stays stale. A real "node joins" event is not supported through the
current public API and would require a fresh ``setup()`` call (this
limitation is captured for a future capability in
``jira/STORY-44-cluster-resources-hot-swap.md``).

So the tests here drive behaviour that is observable through the
existing surface:

  * Group A (loss): a node's workers vanish mid-pipeline.
  * Group B (recovery / flap): workers reappear with fresh ids,
    optionally after repeated flaps.
  * Group C (edges): catastrophic loss, single-node deadlocks,
    id-keyed lookup correctness.

The fixtures share the multi-node cluster + multi-cycle pattern of
``test_worker_warmup_grace.py`` and ``test_donor_warmup_grace.py``.
Per-cycle ``ProblemState`` objects are constructed by
:func:`_stage_state` so the test author controls exactly which
``worker_id``s are present in each cycle. Assertions read scheduler
introspection state (``_worker_ages``, ``_worker_ready_first_seen_at``,
``_last_intent_deltas``, ``_floor_stuck_counters``, ``_stage_states``)
to verify the contract.

Why these tests matter: production node loss on Ray is handled at
the actor-pool layer (Ray detects the dead actors and the streaming
layer rebuilds ``worker_groups`` from survivors), so the scheduler
sees a sudden change in the per-stage worker set. The two pruning
hooks introduced in Phase 3 -- :meth:`_persist_worker_ages` and
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
        # Cycle 1: 4 workers, half empty -> NORMAL.
        ps_normal = data_structures.ProblemState(
            [_stage_state(name="hot", workers_per_node={"node-0": 2, "node-1": 2}, used_slot_ratio=0.5)]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_normal)

        # Cycle 2: lose node-1 (the empty pair); survivors are the saturated pair.
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
        # Fresh peer not yet in _worker_ages until the next persistence cycle (Rust planner assigns age 0).
        assert scheduler._worker_ages.get("hot-node-1-w99", 0) == 0

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
        # Classifier state for each stage stays consistent (no exception, intent dict populated).
        assert "hot" in scheduler._last_intent_deltas
        assert "warm" in scheduler._last_intent_deltas

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

        floor_stuck = scheduler._floor_stuck_counters.get("hot", 0)
        assert floor_stuck >= 1, (
            f"expected _floor_stuck_counters['hot'] >= 1 after 3 sub-floor cycles with no headroom; got {floor_stuck}"
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
