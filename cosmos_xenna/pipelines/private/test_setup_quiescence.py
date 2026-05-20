# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup-phase quiescence in ``SaturationAwareScheduler``.

The setup-phase quiescence gate suppresses scale-up decisions for a
stage that still has pending actors (actors past placement but
pre-``stage_setup``). Two distinct half-initialised states are
covered:

  * **Cold-start** (``pending > 0`` and ``ready == 0``) -- skips the
    entire per-stage intent pipeline so the classifier streak and
    recommendation history are not polluted by zero-signal cycles.
    No entry appears in ``_last_intent_deltas`` for that stage.
  * **Hot-pending** (``pending > 0`` and ``ready > 0``) -- runs the
    pipeline against the real signal from ready actors but clamps
    positive intents (Phase C scale-up) to ``0``. Negative intents
    (Phase D scale-down) are preserved.

The gate is keyed off ``ProblemStageState.num_pending_actors``,
which is sourced from ``ActorPool.num_pending_actors`` by the
streaming layer. The gate honours the per-stage
``setup_phase_quiescence_enabled`` config flag.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 64) -> resources.ClusterResources:
    """Single-node CPU cluster with enough headroom for Phase C grow attempts."""
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


def _problem(stage_names: list[str], cluster: resources.ClusterResources | None = None) -> data_structures.Problem:
    """Build a ``Problem`` with one CPU stage per name."""
    if cluster is None:
        cluster = _cluster()
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


def _stage_state(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int,
    num_used_slots: int,
    num_empty_slots: int,
    input_queue_depth: int,
    num_pending_actors: int,
    is_finished: bool = False,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState`` with explicit pending-actor count.

    Each test isolates a single behaviour so the helper exposes
    every signal as a keyword and never silently defaults the
    pending count -- the gate is the unit under test.
    """
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            f"{name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
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


def _saturated_signal(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int = 8,
    num_pending_actors: int = 0,
) -> data_structures.ProblemStageState:
    """Build a SATURATED-shaped slot signal for the stage.

    A ratio of ``empty / total = 1 / 32`` sits well below the default
    activation threshold derived for ``slots_per_worker = 8`` and
    drives the classifier into ``SATURATED_CRITICAL`` on the very
    first cycle. The exact ratio is unimportant to this suite -- only
    the resulting positive intent is.
    """
    total = num_workers * slots_per_worker
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=max(total - 1, 0),
        num_empty_slots=1 if total else 0,
        input_queue_depth=5,
        num_pending_actors=num_pending_actors,
    )


def _over_provisioned_signal(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int = 8,
    num_pending_actors: int = 0,
) -> data_structures.ProblemStageState:
    """Build an OVER_PROVISIONED-shaped slot signal for the stage.

    OVER_PROVISIONED is only distinguishable from STARVED when the
    upstream queue is non-empty -- a stage with empty slots and no
    queue is just idle, not over-provisioned. The classifier reads
    ``input_queue_depth > 0`` as the disambiguator.
    """
    total = num_workers * slots_per_worker
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=1,
        num_empty_slots=total - 1,
        input_queue_depth=8,
        num_pending_actors=num_pending_actors,
    )


def _scheduler_with_stage(
    stage_name: str,
    *,
    quiescence_enabled: bool = True,
    saturated_streak_min_cycles: int = 2,
    over_provisioned_streak_min_cycles: int = 30,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 30,
) -> SaturationAwareScheduler:
    """Build a scheduler whose only stage uses the supplied gate / streak / window config.

    The defaults match ``SaturationAwareStageConfig`` so a test that
    does not care about a knob inherits production behaviour, except
    for the two warmup grace fields: ``worker_warmup_measurement_grace_s``
    and ``donor_warmup_grace_s`` are set to ``0.0`` so signals are
    absorbed immediately and donor / Phase D selection is unfiltered.
    The quiescence suite is orthogonal to the warmup graces; tests
    that exercise warmup behaviour live in
    ``test_worker_warmup_grace.py`` and ``test_donor_warmup_grace.py``
    and configure those fields explicitly.

    Tests that need to fire Phase D within a small number of cycles
    override the streak + window pair so the asymmetric stabilization
    contract is preserved (down window must dominate up window;
    over-provisioned streak must dominate saturated streak).
    """
    cfg = SaturationAwareConfig(
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=quiescence_enabled,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem([stage_name]))
    return scheduler


class TestColdStartSkipsIntentPipeline:
    """``pending > 0`` with ``ready == 0`` skips the whole pipeline for that stage."""

    def test_cold_start_omits_stage_from_intent_dict(self) -> None:
        """A cold-start stage produces no intent entry at all (not even ``0``).

        The gate must skip ``run_per_stage_pipeline`` outright; an
        explicit ``0`` would still mutate the classifier streak and
        recommendation history, defeating the purpose of the gate.
        """
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=2,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert "hot" not in scheduler._last_intent_deltas

    def test_cold_start_does_not_advance_classifier_state(self) -> None:
        """Classifier streak / state is untouched by a cold-start cycle.

        The default classifier state at ``setup()`` is ``NORMAL`` with
        a zero streak; if the cold-start cycle slipped through the
        gate the classifier would observe a zero-signal slot ratio
        and either tick the streak or transition to ``STARVED``.
        """
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=4,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state is StageState.NORMAL
        assert runtime.classifier_streak == 0

    def test_cold_start_does_not_record_into_recommendation_history(self) -> None:
        """The stabilization-window buffer is not advanced during cold-start."""
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=1,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        history = scheduler._recommendation_histories["hot"]
        assert len(history._buffer) == 0

    def test_cold_start_solution_has_no_new_workers_for_stage(self) -> None:
        """Phase C cannot fire because the stage has no positive intent.

        Pins the end-to-end contract: cold-start quiescent stage gets
        zero adds in the cycle's ``Solution``, even when the cluster
        has placement capacity.
        """
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=2,
                ),
            ]
        )

        solution = scheduler.autoscale(time=0.0, problem_state=ps)

        # Phase B floor still adds a worker to satisfy the implicit floor=1.
        # The cold-start gate prevents *Phase C* additional adds, not the floor.
        assert len(solution.stages[0].new_workers) == 1
        assert solution.stages[0].deleted_workers == []


class TestColdStartTransitionsCleanly:
    """Once at least one actor reaches ready, the pipeline re-engages."""

    def test_pending_to_ready_transition_re_engages_classifier(self) -> None:
        """Cycle 1 cold-start, cycle 2 hot signal -> classifier evaluates real ratio."""
        scheduler = _scheduler_with_stage("hot")

        ps_cold = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=4,
                ),
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_cold)
        assert "hot" not in scheduler._last_intent_deltas

        ps_hot = _saturated_signal(name="hot", num_workers=4)
        scheduler.autoscale(time=10.0, problem_state=data_structures.ProblemState([ps_hot]))

        assert "hot" in scheduler._last_intent_deltas
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state in {StageState.SATURATED, StageState.SATURATED_CRITICAL}


class TestHotPendingClampsPhaseC:
    """``pending > 0`` with ``ready > 0`` runs the pipeline but clamps positive intent."""

    def test_hot_pending_clamps_saturated_intent_to_zero(self) -> None:
        """A SATURATED signal still records into the classifier; Phase C is suppressed."""
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=2)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas["hot"] == 0

    def test_hot_pending_does_not_freeze_classifier(self) -> None:
        """The classifier still observes the live signal during hot-pending.

        Hot-pending differs from cold-start in that the ready actors
        produce real signal; suppressing the classifier here would
        delay correct decisions once the pending actor lands.
        """
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=2)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state in {StageState.SATURATED, StageState.SATURATED_CRITICAL}

    def test_hot_pending_solution_has_no_phase_c_adds(self) -> None:
        """End-to-end: SATURATED + pending -> ``Solution`` has zero new workers."""
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=2)])

        solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_hot_pending_phase_c_resumes_after_pending_drains(self) -> None:
        """Once pending count returns to ``0``, the Phase C scale-up fires."""
        scheduler = _scheduler_with_stage("hot")

        ps_blocked = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=1)])
        scheduler.autoscale(time=0.0, problem_state=ps_blocked)
        assert scheduler._last_intent_deltas["hot"] == 0

        ps_free = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=0)])
        solution = scheduler.autoscale(time=10.0, problem_state=ps_free)

        assert scheduler._last_intent_deltas["hot"] > 0
        assert len(solution.stages[0].new_workers) >= 1


class TestHotPendingPreservesPhaseDIntent:
    """Phase D scale-down is allowed even while a stage has pending actors."""

    def test_over_provisioned_classifier_advances_under_hot_pending(self) -> None:
        """A sustained OVER_PROVISIONED signal still records into the classifier.

        Hot-pending only suppresses positive intents; the classifier
        and the stabilization-window history must continue to absorb
        the live signal so Phase D can fire as soon as the streak
        ripens. Pins the contract that the gate does not freeze the
        per-stage decision pipeline -- only Phase C scale-up.
        """
        scheduler = _scheduler_with_stage(
            "cold",
            saturated_streak_min_cycles=1,
            over_provisioned_streak_min_cycles=2,
            stabilization_window_cycles_up=1,
            stabilization_window_cycles_down=2,
        )

        ps = data_structures.ProblemState([_over_provisioned_signal(name="cold", num_workers=4, num_pending_actors=1)])
        # Three cycles: the classifier needs a streak of 2 to fire and the
        # stabilization down-window needs two consecutive shrink votes.
        for cycle in range(3):
            scheduler.autoscale(time=cycle * 10.0, problem_state=ps)

        runtime = scheduler._stage_states["cold"]
        assert runtime.classifier_state is StageState.OVER_PROVISIONED
        assert runtime.classifier_streak >= 2
        # The classifier streak ripened despite the hot-pending state, so
        # the next cycle's negative intent passes through Phase D.
        assert scheduler._last_intent_deltas["cold"] <= 0


class TestQuiescenceDisabledByConfig:
    """``setup_phase_quiescence_enabled=False`` removes both gates."""

    def test_disabled_gate_lets_cold_start_run_pipeline(self) -> None:
        """Cold-start with the gate disabled produces an explicit intent entry.

        With the gate off, the pipeline is no longer guarded against
        zero-signal cycles, so the intent dict carries an entry (the
        value depends on the noise; the contract is just "an entry
        exists rather than being skipped").
        """
        scheduler = _scheduler_with_stage("hot", quiescence_enabled=False)
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=2,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert "hot" in scheduler._last_intent_deltas

    def test_disabled_gate_lets_hot_pending_grow(self) -> None:
        """SATURATED + pending + gate disabled -> positive intent is not clamped."""
        scheduler = _scheduler_with_stage("hot", quiescence_enabled=False)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=2)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas["hot"] > 0


class TestNoQuiescenceWithoutPending:
    """``pending == 0`` is the steady state -- the gate must be a no-op."""

    def test_zero_pending_with_saturated_signal_grows_normally(self) -> None:
        """A SATURATED stage with no pending actors still scales up."""
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4, num_pending_actors=0)])

        solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas["hot"] > 0
        assert len(solution.stages[0].new_workers) >= 1


class TestMixedQuiescenceAcrossStages:
    """Quiescence is per-stage; one cold-start stage does not freeze its peer."""

    def test_one_cold_start_stage_does_not_freeze_peer(self) -> None:
        """In a two-stage pipeline, a quiescent stage does not block a non-quiescent one."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=True,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem(["bootstrap", "running"]))

        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="bootstrap",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=4,
                ),
                _saturated_signal(name="running", num_workers=4, num_pending_actors=0),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert "bootstrap" not in scheduler._last_intent_deltas
        assert scheduler._last_intent_deltas["running"] > 0


class TestFinishedStageBypassesGate:
    """A finished stage is short-circuited regardless of pending count."""

    def test_finished_stage_with_pending_actors_omits_intent(self) -> None:
        """``is_finished=True`` skips the stage upstream of the gate.

        Pins that the gate's quiescence check does not need to
        re-implement the finished-stage filter; the existing filter
        runs first.
        """
        scheduler = _scheduler_with_stage("done")
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="done",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=3,
                    is_finished=True,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas == {}


@pytest.mark.parametrize("pending", [1, 2, 5, 100])
class TestColdStartGateIsBoundaryStable:
    """Any positive ``pending`` count triggers cold-start; the magnitude is irrelevant."""

    def test_any_positive_pending_count_freezes_cold_start(self, pending: int) -> None:
        """The gate fires on ``pending > 0``, not on a magnitude threshold."""
        scheduler = _scheduler_with_stage("hot")
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                    num_pending_actors=pending,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert "hot" not in scheduler._last_intent_deltas
