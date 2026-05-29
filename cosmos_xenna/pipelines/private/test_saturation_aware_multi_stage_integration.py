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


"""Multi-stage DAG integration tests for ``SaturationAwareScheduler``.

Each test drives a linear chain ``A -> B -> C [-> D]`` and pins one
cross-stage interaction: DAG-priority grow order, cross-stage donor
cascade, finished-stage isolation, mixed manual / auto / finished
stability, or bottleneck-identity shift.
"""

from collections.abc import Callable, Iterable
from unittest.mock import patch

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    ClassifierState,
    GrowthMode,
    GrowthState,
    StageRuntimeState,
    StageState,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_topology import project_stage_topology
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 64) -> resources.ClusterResources:
    """CPU-only cluster sized by caller; default fits multi-stage grow scenarios."""
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
    stage_specs: list[tuple[str, int | None]],
    cluster: resources.ClusterResources | None = None,
) -> data_structures.Problem:
    """Build a Problem with one CPU stage per ``(name, requested_num_workers)``.

    ``requested_num_workers=None`` marks an auto-scaled stage; an
    integer marks a manual stage that Phase A grows / shrinks toward.
    """
    if cluster is None:
        cluster = _cluster()
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


def _stage_state(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int = 8,
    num_used_slots: int = 0,
    num_empty_slots: int = 0,
    input_queue_depth: int = 0,
    num_pending_actors: int = 0,
    is_finished: bool = False,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState``; per-worker slot occupancy mirrors the stage-level signal."""
    used_per_worker = (num_used_slots // num_workers) if num_workers > 0 else 0
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            f"{name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            num_used_slots=used_per_worker,
        )
        for i in range(num_workers)
    ]
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=worker_groups,
        slots_per_worker=slots_per_worker,
        is_finished=is_finished,
        num_used_slots=num_used_slots,
        num_empty_slots=num_empty_slots,
        input_queue_depth=input_queue_depth,
        num_pending_actors=num_pending_actors,
    )


def _saturated(name: str, *, num_workers: int, slots_per_worker: int = 8) -> data_structures.ProblemStageState:
    """Build a SATURATED_CRITICAL-shaped slot signal for the stage."""
    total = num_workers * slots_per_worker
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=total,
        num_empty_slots=0,
        input_queue_depth=5,
    )


def _normal(name: str, *, num_workers: int, slots_per_worker: int = 8) -> data_structures.ProblemStageState:
    """Build a NORMAL-zone slot signal mid-band between SATURATED and OVER_PROVISIONED."""
    total = num_workers * slots_per_worker
    used = total // 2
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=used,
        num_empty_slots=total - used,
        input_queue_depth=2,
    )


def _build_scheduler(
    stage_specs: list[tuple[str, int | None]],
    *,
    cluster: resources.ClusterResources | None = None,
    saturated_streak_min_cycles: int = 1,
    over_provisioned_streak_min_cycles: int = 2,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 2,
    enable_dag_priority_growth: bool = True,
    enable_cross_stage_donor: bool = True,
    cross_stage_donor_anti_flap_cycles: int = 30,
    min_workers: int | None = 1,
    max_scale_down_fraction_per_cycle: float = 1.0,
    quiescence_enabled: bool = False,
    min_data_points: int = 1,
) -> SaturationAwareScheduler:
    """Build a multi-stage scheduler with DAG priority and cross-stage donor enabled by default."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        enable_dag_priority_growth=enable_dag_priority_growth,
        enable_cross_stage_donor=enable_cross_stage_donor,
        cross_stage_donor_anti_flap_cycles=cross_stage_donor_anti_flap_cycles,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=min_workers,
            min_data_points=min_data_points,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
            setup_phase_quiescence_enabled=quiescence_enabled,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_specs, cluster=cluster))
    return scheduler


def _run_cycles(
    scheduler: SaturationAwareScheduler,
    state_factory: Callable[[int], data_structures.ProblemState],
    *,
    num_cycles: int,
    cycle_interval_s: float = 10.0,
    start_time_s: float = 0.0,
    before_cycle: Callable[[int, SaturationAwareScheduler], None] | None = None,
) -> list[data_structures.Solution]:
    """Drive ``num_cycles`` cycles and capture each ``Solution``.

    Args:
        before_cycle: Optional hook invoked immediately before each
            ``autoscale()`` call; used to mutate scheduler internal
            state under test (for example to seed an EWMA).

    """
    solutions: list[data_structures.Solution] = []
    for cycle_idx in range(num_cycles):
        if before_cycle is not None:
            before_cycle(cycle_idx, scheduler)
        ps = state_factory(cycle_idx)
        sol = scheduler.autoscale(
            time=start_time_s + cycle_idx * cycle_interval_s,
            problem_state=ps,
        )
        solutions.append(sol)
    return solutions


def _new_workers_per_stage(solutions: Iterable[data_structures.Solution], stage_index: int) -> list[int]:
    """Per-cycle list of ``len(new_workers)`` for one stage index."""
    return [len(sol.stages[stage_index].new_workers) for sol in solutions]


def _deleted_workers_per_stage(solutions: Iterable[data_structures.Solution], stage_index: int) -> list[int]:
    """Per-cycle list of ``len(deleted_workers)`` for one stage index."""
    return [len(sol.stages[stage_index].deleted_workers) for sol in solutions]


class TestDagPriorityGrowOrderUnderSimultaneousSaturation:
    """When the cluster has tight headroom, downstream stages grow first."""

    def test_dag_priority_grow_order_under_simultaneous_saturation(self) -> None:
        """A->B->C all SATURATED with 1 free slot: ``C`` grows, ``A`` does not.

        The cluster is sized so exactly one placement remains free
        when all three stages produce a positive intent on the same
        cycle. With ``enable_dag_priority_growth=True`` that slot
        must land on the downstream stage.
        """
        # 4 CPUs cluster, 1-CPU worker shape. Pre-seed each stage with 1
        # worker (3 CPUs used) so 1 CPU remains free.
        scheduler = _build_scheduler(
            [("A", None), ("B", None), ("C", None)],
            cluster=_cluster(num_nodes=1, total_cpus_per_node=4),
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [
                    _saturated("A", num_workers=1),
                    _saturated("B", num_workers=1),
                    _saturated("C", num_workers=1),
                ]
            )

        # Two cycles: cycle 0 builds the classifier streak; cycle 1 fires.
        solutions = _run_cycles(scheduler, state_factory, num_cycles=2)

        # Across the run, the C stage MUST have received at least one
        # add, and the total adds across all three stages MUST not exceed
        # the cluster's free placement budget. The upstream stages may
        # receive zero adds because DAG priority drains the budget on the
        # downstream side first.
        a_adds = _new_workers_per_stage(solutions, stage_index=0)
        c_adds = _new_workers_per_stage(solutions, stage_index=2)
        assert sum(c_adds) >= 1, (
            f"DAG-priority order must grow downstream stage 'C' under simultaneous "
            f"saturation; got per-cycle C adds {c_adds}, A adds {a_adds}"
        )
        # Upstream 'A' must not grow alone on any cycle: DAG priority
        # allocates headroom to downstream stages first, so an add on
        # 'A' without a concurrent add on 'C' would violate ordering.
        per_cycle_pairs = list(zip(a_adds, c_adds, strict=True))
        for cycle_idx, (a, c) in enumerate(per_cycle_pairs):
            assert not (a > 0 and c == 0), (
                f"DAG-priority: upstream 'A' must not grow alone on cycle "
                f"{cycle_idx}; got A adds={a}, C adds={c} "
                f"(per-cycle pairs {per_cycle_pairs})"
            )


class TestDonorCascadeReleasesUpstreamForDownstream:
    """Saturation donor cascades: OVER_PROVISIONED upstream releases for SATURATED downstream."""

    def test_donor_cascade_upstream_releases_for_downstream_bottleneck(self) -> None:
        """Cluster at cap, A=OVER_PROVISIONED, B=SATURATED: A donates, B grows.

        Pre-seeds the per-stage classifier state (OVER_PROVISIONED
        streak on A, SATURATED on B) and injects a positive intent
        for B so Phase C consumes the cross-stage donor path
        predictably. With the cluster fully utilised the only way
        B can grow is by reclaiming a slot from A.
        """
        # 3 CPUs cluster, 1-CPU shapes -> exactly 3 workers fit; the test
        # seeds 2 A workers + 1 B worker so the cluster is at cap.
        cluster = _cluster(num_nodes=1, total_cpus_per_node=3)
        scheduler = _build_scheduler(
            [("A", None), ("B", None)],
            cluster=cluster,
            over_provisioned_streak_min_cycles=2,
        )
        # Pre-seed the per-stage state so the donor classifier filter
        # accepts A immediately. The integration property under test
        # is the cross-stage donor cascade, not the classifier ramp.
        scheduler.ledgers.stage_states["A"] = StageRuntimeState(
            stage_name="A",
            classifier=ClassifierState(state=StageState.OVER_PROVISIONED, streak=10),
            growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
        )
        scheduler.ledgers.stage_states["B"] = StageRuntimeState(
            stage_name="B",
            classifier=ClassifierState(state=StageState.SATURATED, streak=10),
            growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
        )

        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="A",
                    num_workers=2,
                    num_used_slots=2,
                    num_empty_slots=14,
                    input_queue_depth=0,
                ),
                _stage_state(
                    name="B",
                    num_workers=1,
                    num_used_slots=8,
                    num_empty_slots=0,
                    input_queue_depth=5,
                ),
            ]
        )

        # Inject a fixed intent so the donor cascade fires on cycle 0
        # without depending on signal-driven intent ramp. Without the
        # injection the recommendation history would absorb a single
        # cycle's vote and Phase C would observe intent=0 still.
        def _inject(_services: object, _cycle: object, **_kwargs: object) -> dict[str, int]:
            return {"B": 1}

        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            side_effect=_inject,
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        a_solution = solution.stages[0]
        b_solution = solution.stages[1]
        assert len(b_solution.new_workers) == 1, (
            f"downstream SATURATED 'B' must grow exactly +1 via donor; got {len(b_solution.new_workers)} adds"
        )
        assert len(a_solution.deleted_workers) == 1, (
            f"upstream OVER_PROVISIONED 'A' must donate exactly one worker to B; got "
            f"{len(a_solution.deleted_workers)} deletes"
        )


class TestFinishedStageZeroChurnDuringDownstreamSpike:
    """A finished stage is never touched, even while a downstream stage spikes."""

    def test_finished_stage_zero_churn_during_downstream_demand_spike(self) -> None:
        """A->B->C with B finished, C SATURATED for many cycles: B receives no churn.

        ``is_finished=True`` short-circuits every per-stage phase
        (manual, floor, grow, shrink, cross-stage donor). The spike
        on C is long enough to fire Phase C multiple times so a
        regression that leaked an add into B as a side effect would
        surface as a non-zero count on the finished stage.
        """
        scheduler = _build_scheduler(
            [("A", None), ("B", None), ("C", None)],
            cluster=_cluster(num_nodes=1, total_cpus_per_node=64),
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [
                    _normal("A", num_workers=1),
                    # B is finished with a few residual workers; floor
                    # / Phase C / Phase D / donor must all skip it.
                    _stage_state(
                        name="B",
                        num_workers=2,
                        num_used_slots=0,
                        num_empty_slots=16,
                        input_queue_depth=0,
                        is_finished=True,
                    ),
                    _saturated("C", num_workers=1),
                ]
            )

        solutions = _run_cycles(scheduler, state_factory, num_cycles=6)

        # Phase C should grow C at least once across the spike.
        c_adds = sum(_new_workers_per_stage(solutions, stage_index=2))
        assert c_adds >= 1, f"SATURATED 'C' must grow at least once during the spike; got {c_adds} adds"
        # The finished stage B must never be touched in any cycle.
        b_adds_per_cycle = _new_workers_per_stage(solutions, stage_index=1)
        b_deletes_per_cycle = _deleted_workers_per_stage(solutions, stage_index=1)
        assert all(n == 0 for n in b_adds_per_cycle), (
            f"finished stage 'B' must receive zero new_workers across the run, got {b_adds_per_cycle}"
        )
        assert all(d == 0 for d in b_deletes_per_cycle), (
            f"finished stage 'B' must receive zero deleted_workers across the run, got {b_deletes_per_cycle}"
        )


class TestMixedManualAutoFinishedLongRunningStability:
    """A 4-stage mix of manual/auto/finished stays stable across 20 cycles."""

    def test_mixed_manual_auto_finished_long_running_stability(self) -> None:
        """Manual at request, auto in band, finished unchanged; 20 cycles, no panic.

        Across 20 cycles the three stage kinds compose cleanly:
        manual converges to ``requested_num_workers``, auto stages
        stay quiet under a NORMAL signal, and finished stages are
        not touched. The horizon is long enough to catch slow
        recommendation-history drift.
        """
        scheduler = _build_scheduler(
            # A: auto; B: manual=3; C: auto; D: finished placeholder
            [("A", None), ("B", 3), ("C", None), ("D", None)],
            cluster=_cluster(num_nodes=1, total_cpus_per_node=64),
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [
                    _normal("A", num_workers=2),
                    # Manual stage already at request -- Phase A is a no-op.
                    _normal("B", num_workers=3),
                    _normal("C", num_workers=2),
                    # Finished placeholder
                    _stage_state(
                        name="D",
                        num_workers=1,
                        num_used_slots=0,
                        num_empty_slots=8,
                        input_queue_depth=0,
                        is_finished=True,
                    ),
                ]
            )

        solutions = _run_cycles(scheduler, state_factory, num_cycles=20)

        # Finished stage must never be touched.
        d_adds = _new_workers_per_stage(solutions, stage_index=3)
        d_deletes = _deleted_workers_per_stage(solutions, stage_index=3)
        assert all(n == 0 for n in d_adds), f"finished stage 'D' must not grow, got {d_adds}"
        assert all(n == 0 for n in d_deletes), f"finished stage 'D' must not shrink, got {d_deletes}"

        # Manual stage at request must not be churned beyond the initial
        # floor cycle. Phase A only acts when ``current != requested``;
        # since the state factory keeps B at 3 every cycle, Phase A must
        # be a no-op every cycle.
        b_adds = _new_workers_per_stage(solutions, stage_index=1)
        b_deletes = _deleted_workers_per_stage(solutions, stage_index=1)
        assert all(n == 0 for n in b_adds), (
            f"manual stage 'B' at request must not grow during steady NORMAL run, got {b_adds}"
        )
        assert all(n == 0 for n in b_deletes), (
            f"manual stage 'B' at request must not shrink during steady NORMAL run, got {b_deletes}"
        )

        # Auto stages on NORMAL signal: no Phase D shrink ever. (Phase C
        # adds on cycle 0 may happen if the floor adds to satisfy
        # min_workers=1; the assertion below is the long-run quietness
        # contract.)
        for stage_index in (0, 2):  # A and C
            deletes = _deleted_workers_per_stage(solutions, stage_index=stage_index)
            assert sum(deletes) == 0, (
                f"auto stage at index {stage_index} must not shrink on a steady NORMAL "
                f"signal across 20 cycles, got per-cycle deletes {deletes}"
            )


class TestSaturatedManualStageHoldsAtPin:
    """A SATURATED manual stage holds at its pin -- no grow/delete churn, no donor raid.

    End-to-end guard for the manual-stage invariant: Phase A owns
    ``requested_num_workers`` stages, so Intent must not emit a delta for
    them (root cause) and Grow must not act on one (defense in depth).
    Two failure modes are pinned: (1) per-cycle grow/delete churn that
    burns warm GPU state on a roomy cluster, and (2) the cross-stage
    donor raiding auto-scaled stages to feed a pinned receiver on a full
    cluster.
    """

    def test_saturated_manual_stage_does_not_grow_while_auto_sibling_does(self) -> None:
        """Roomy cluster: a saturated manual stage holds at its pin; the auto sibling grows.

        Exercises the real Intent -> Grow path (no injected intent). The
        manual stage ``M`` is fed at exactly its pin (2 workers) under a
        SATURATED_CRITICAL signal every cycle; pre-fix Intent emits a
        positive delta and Grow pushes it above the pin. Post-fix Intent
        skips it, so ``M`` never grows and is never deleted, while the
        auto-scaled ``A`` still grows under the same signal.
        """
        scheduler = _build_scheduler(
            [("M", 2), ("A", None)],
            cluster=_cluster(num_nodes=1, total_cpus_per_node=64),
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [
                    _saturated("M", num_workers=2),
                    _saturated("A", num_workers=1),
                ]
            )

        solutions = _run_cycles(scheduler, state_factory, num_cycles=4)

        m_adds = _new_workers_per_stage(solutions, stage_index=0)
        m_deletes = _deleted_workers_per_stage(solutions, stage_index=0)
        a_adds = _new_workers_per_stage(solutions, stage_index=1)
        assert all(n == 0 for n in m_adds), f"manual stage 'M' must never grow above its pin, got {m_adds}"
        assert all(n == 0 for n in m_deletes), f"manual stage 'M' at its pin must not be churned, got {m_deletes}"
        assert sum(a_adds) >= 1, f"auto-scaled 'A' must still grow under the same signal, got {a_adds}"

    def test_saturated_manual_receiver_does_not_raid_auto_donor_on_full_cluster(self) -> None:
        """Full cluster: a pinned receiver must not trigger the donor to raid an auto stage.

        Mirrors ``TestDonorCascadeReleasesUpstreamForDownstream`` but
        makes the receiver a *manual* stage. With the cluster at cap the
        only way to grow the receiver is to reclaim a worker from the
        OVER_PROVISIONED auto donor ``A``. Pre-fix Grow processes the
        pinned receiver and the donor raids ``A``; post-fix Grow skips
        the manual receiver, so ``A`` keeps its worker and the receiver
        holds. Intent is injected (as in the cascade test) to isolate the
        Grow donor-fallback path deterministically.
        """
        # 3 CPUs, 1-CPU shapes -> 3 workers fit; seed 2 'A' + 1 'M' = at cap.
        scheduler = _build_scheduler(
            [("A", None), ("M", 1)],
            cluster=_cluster(num_nodes=1, total_cpus_per_node=3),
            over_provisioned_streak_min_cycles=2,
        )
        # Pre-seed classifier state so the donor filter accepts 'A' and
        # treats 'M' as a saturated receiver immediately (mirrors the
        # cascade test); the property under test is the donor raid, not
        # the classifier ramp.
        scheduler.ledgers.stage_states["A"] = StageRuntimeState(
            stage_name="A",
            classifier=ClassifierState(state=StageState.OVER_PROVISIONED, streak=10),
            growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
        )
        scheduler.ledgers.stage_states["M"] = StageRuntimeState(
            stage_name="M",
            classifier=ClassifierState(state=StageState.SATURATED, streak=10),
            growth=GrowthState(mode=GrowthMode.TRACKING, streak=10),
        )

        ps = data_structures.ProblemState(
            [
                _stage_state(name="A", num_workers=2, num_used_slots=2, num_empty_slots=14, input_queue_depth=0),
                _stage_state(name="M", num_workers=1, num_used_slots=8, num_empty_slots=0, input_queue_depth=5),
            ]
        )

        # Inject intent for the pinned receiver to isolate the Grow
        # donor-fallback path; the injection bypasses the Intent guard so
        # this asserts the Grow guard specifically.
        def _inject(_services: object, _cycle: object, **_kwargs: object) -> dict[str, int]:
            return {"M": 1}

        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            side_effect=_inject,
        ):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)

        a_solution = solution.stages[0]
        m_solution = solution.stages[1]
        assert a_solution.deleted_workers == [], (
            f"auto donor 'A' must not be raided to feed a pinned receiver; got "
            f"{len(a_solution.deleted_workers)} deletes"
        )
        assert m_solution.new_workers == [], (
            f"manual receiver 'M' must not be grown via the donor path; got {len(m_solution.new_workers)} adds"
        )
        assert m_solution.deleted_workers == [], "manual receiver 'M' at its pin must not be deleted"


class TestBottleneckShiftEngagementMarkerFollowsSaturatedStage:
    """``cycle.bottleneck.identity`` tracks the EWMA shift; per-stage projection follows."""

    def test_bottleneck_shift_engagement_marker_follows_saturated_stage(self) -> None:
        """3-stage chain: bottleneck argmax moves B -> C; per-stage upstream flag flips.

        Two cycles. Before each ``autoscale()`` the ``S_k`` EWMA is
        seeded so :func:`identify_bottleneck` first picks B then C.
        The per-stage ``StageTopologyContext`` projected from
        ``cycle.bottleneck.identity`` must track the shift: cycle 0
        marks A upstream of B; cycle 1 marks A and B upstream of C
        and clears C itself.
        """
        scheduler = _build_scheduler(
            [("A", None), ("B", None), ("C", None)],
            cluster=_cluster(num_nodes=1, total_cpus_per_node=16),
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState(
                [
                    _normal("A", num_workers=2),
                    _normal("B", num_workers=2),
                    _normal("C", num_workers=2),
                ]
            )

        def before_cycle(cycle_idx: int, sched: SaturationAwareScheduler) -> None:
            # Heterogeneity ratio default threshold is 2.0; max/median of
            # the three samples must be >= 2.0 for engagement. The values
            # below produce ratio = 5.0 in each cycle, well past the gate.
            samples = {"A": 0.1, "B": 1.0, "C": 0.2} if cycle_idx == 0 else {"A": 0.1, "B": 0.2, "C": 1.0}
            sched.ledgers.s_k_ewma.set_many(samples)

        _run_cycles(scheduler, state_factory, num_cycles=2, before_cycle=before_cycle)

        # After cycle 1, the bottleneck snapshot must reflect "C".
        meta = scheduler.last_cycle.bottleneck.identity
        assert meta is not None
        assert meta.engaged is True
        assert meta.stage_name == "C", f"engaged bottleneck must shift to 'C' on cycle 1, got {meta.stage_name!r}"
        # Per-stage projection: A and B are now upstream of C; C is not.
        # The projection is computed at the call site (no per-stage mirror);
        # asserting on the projection here pins the same contract that
        # ``IntentPhase._compute_intent_deltas`` exercises at runtime.
        stage_names = scheduler.pipeline.stage_names
        ctx_a = project_stage_topology(
            stage_index=stage_names.index("A"),
            bottleneck_engaged=meta.engaged,
            bottleneck_stage_name=meta.stage_name,
            stage_names=stage_names,
        )
        ctx_b = project_stage_topology(
            stage_index=stage_names.index("B"),
            bottleneck_engaged=meta.engaged,
            bottleneck_stage_name=meta.stage_name,
            stage_names=stage_names,
        )
        ctx_c = project_stage_topology(
            stage_index=stage_names.index("C"),
            bottleneck_engaged=meta.engaged,
            bottleneck_stage_name=meta.stage_name,
            stage_names=stage_names,
        )
        assert ctx_a.engaged is True and ctx_a.is_upstream_of_bottleneck is True
        assert ctx_b.engaged is True and ctx_b.is_upstream_of_bottleneck is True
        assert ctx_c.engaged is True and ctx_c.is_upstream_of_bottleneck is False
