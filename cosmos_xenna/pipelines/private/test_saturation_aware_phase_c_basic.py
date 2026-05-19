# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SaturationAwareScheduler._run_phase_c_grow_single_target``.

Phase 2-iii applies positive intent deltas as planner adds via
``ctx.try_add_worker``. The contract under test:

    * Positive intent grows the stage by exactly ``intent`` workers
      when the cluster has room.
    * Cluster exhaustion is non-fatal: a single WARNING per affected
      stage; the Solution carries the partial growth.
    * Negative or zero intent is a no-op (Phase D scale-down ships
      separately).
    * Finished stages are skipped.
    * The post-Phase-C invariant gate runs after the grow.

Most tests inject the intent dict directly via
``patch.object(scheduler, "_compute_intent_deltas", ...)`` so each
adversarial case can be exercised without rigging classifier
signals; one integration test exercises the end-to-end signal path.
"""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.invariants import PhaseBoundary
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    ``cosmos_xenna.utils.python_log`` routes logging through loguru,
    which does not propagate to the stdlib ``logging`` module by
    default, so ``caplog`` is empty without this bridge. The sink
    forwards every loguru record through a stdlib logger named
    ``"loguru"``. Mirrors the bridge used in
    ``test_autoscaler_queue_aware_guard.py``.
    """
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 8) -> resources.ClusterResources:
    """CPU-only cluster sized for the Phase C fixtures."""
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
    """Build a Problem with one CPU stage per spec ``(name, requested_num_workers)``."""
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


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
) -> data_structures.ProblemState:
    """Build a ProblemState; signals default to zero so the classifier holds."""
    states = []
    for name, num_workers, slots, finished in stage_specs:
        worker_groups = [
            data_structures.ProblemWorkerGroupState.make(
                f"{name}-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(num_workers)
        ]
        states.append(
            data_structures.ProblemStageState(
                stage_name=name,
                workers=worker_groups,
                slots_per_worker=slots,
                is_finished=finished,
            )
        )
    return data_structures.ProblemState(states)


def _scheduler(stage_specs: list[tuple[str, int | None]]) -> SaturationAwareScheduler:
    """Build a setup-completed scheduler over the given stages."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(min_workers=1),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_specs))
    return scheduler


def _autoscale_with_intents(
    scheduler: SaturationAwareScheduler,
    state: data_structures.ProblemState,
    intents: dict[str, int],
) -> data_structures.Solution:
    """Run autoscale with ``intents`` injected as ``_compute_intent_deltas`` output."""

    def _inject(_ctx: object, _state: object) -> dict[str, int]:
        return dict(intents)

    with patch.object(scheduler, "_compute_intent_deltas", side_effect=_inject):
        return scheduler.autoscale(time=0.0, problem_state=state)


class TestPhaseCBasicGrowth:
    """Positive intent grows the stage by exactly ``intent`` workers when room exists."""

    def test_positive_intent_grows_stage_by_intent(self) -> None:
        """Intent of 3 with cluster headroom of 7 results in exactly 3 new workers."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 3})

        assert len(solution.stages[0].new_workers) == 3
        assert solution.stages[0].deleted_workers == []

    def test_intent_one_grows_by_one(self) -> None:
        """Boundary: intent of 1 produces exactly one add."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 1})

        assert len(solution.stages[0].new_workers) == 1


class TestNonPositiveIntentIsNoOp:
    """Negative and zero intent values do not cause Phase C to act."""

    def test_zero_intent_does_not_grow(self) -> None:
        """NORMAL / STARVED stages produce intent 0 and Phase C is a no-op."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 0})

        assert solution.stages[0].new_workers == []

    def test_negative_intent_does_not_grow(self) -> None:
        """OVER_PROVISIONED produces negative intent; Phase C must not grow.

        Pins the asymmetric responsibility split: scale-up is Phase
        C; scale-down is Phase D. A regression flipping ``intent <= 0``
        to ``intent != 0`` would silently grow over-provisioned
        stages in Phase C and is caught here. Phase D's scale-down
        side effect on the same Solution is asserted in
        ``test_saturation_aware_phase_d_basic.py``; this test is
        scoped to the Phase C contract only.
        """
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 4, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        assert solution.stages[0].new_workers == []


class TestFinishedStageSkipped:
    """Phase C never grows a finished stage even if its intent dict entry is positive."""

    def test_finished_stage_is_not_grown(self) -> None:
        """Defensive: a finished stage with a positive intent entry produces no add."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, True)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 3})

        assert solution.stages[0].new_workers == []


class TestIntentDictAbsenceDefaultsToZero:
    """A stage absent from the intent dict produces no Phase C add (``.get(name, 0)``)."""

    def test_missing_intent_entry_does_not_grow(self) -> None:
        """The defensive ``.get`` default lets Phase C tolerate a stage absent from intent."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {})

        assert solution.stages[0].new_workers == []


class TestClusterExhaustion:
    """Cluster placement exhaustion is non-fatal and emits one WARNING per affected stage."""

    def test_intent_exceeding_cluster_capacity_partially_grows(self) -> None:
        """Intent of 100 on an 8-CPU cluster with 1 seeded worker grows only by remaining capacity."""
        scheduler = _scheduler([("A", None)])
        # 8-CPU cluster, 1 worker uses 1 CPU -> 7 CPUs free.
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 100})

        added = len(solution.stages[0].new_workers)
        assert added <= 7
        assert added > 0

    def test_int_max_intent_terminates_at_cluster_exhaustion(self) -> None:
        """An ``intent`` of ``sys.maxsize`` terminates at cluster exhaustion (no infinite loop)."""
        import sys

        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": sys.maxsize})

        added = len(solution.stages[0].new_workers)
        assert added <= 7

    def test_warning_logged_on_cluster_exhaustion(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Operators see a WARNING with stage name + intent + actual + deficit on partial grows."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        _autoscale_with_intents(scheduler, state, {"A": 100})

        warnings = [record.getMessage() for record in loguru_caplog.records if record.levelname == "WARNING"]
        assert any("'A'" in msg and "deficit" in msg for msg in warnings), (
            f"expected one stage-named deficit warning; got: {warnings}"
        )


class TestMultiStageIndependentGrowth:
    """Without DAG priority each stage with positive intent grows independently.

    NOTE: this test deliberately captures the pre-DAG-priority
    contract. When iteration 2-iv (DAG-priority multi-target growth)
    lands, the iteration order will become DAG-depth DESC and this
    test should be updated or replaced.
    """

    def test_two_saturated_stages_both_grow(self) -> None:
        """Two stages each with positive intent both receive their full intent on a roomy cluster."""
        scheduler = _scheduler([("A", None), ("B", None)])
        state = _problem_state([("A", 1, 1, False), ("B", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 2, "B": 3})

        assert len(solution.stages[0].new_workers) == 2
        assert len(solution.stages[1].new_workers) == 3


class TestPhaseCInvariantBoundary:
    """The post-Phase-C invariant gate runs after the grow."""

    def test_invariants_invoked_at_phase_c_boundary(self) -> None:
        """``check_invariants_after_phase`` is called with ``PhaseBoundary.PHASE_C`` after grow."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_invariants_after_phase"
        ) as phase_check:
            _autoscale_with_intents(scheduler, state, {"A": 1})

        phase_names = [call.kwargs["phase_name"] for call in phase_check.call_args_list]
        assert PhaseBoundary.PHASE_C in phase_names


class TestSaturationDrivenIntegration:
    """End-to-end: a SATURATED_CRITICAL classifier signal grows the stage via Phase C."""

    def test_saturated_critical_signal_grows_via_phase_c(self) -> None:
        """A real classifier signal flowing through ``_compute_intent_deltas`` triggers a Phase C add."""
        scheduler = _scheduler([("hot", None)])
        # 4 workers, 8 slots/worker = 32 slots; 31 used + 1 empty -> ratio ~ 0.03,
        # below activation threshold for c=8 -> SATURATED_CRITICAL on first cycle.
        worker_groups = [
            data_structures.ProblemWorkerGroupState.make(
                f"hot-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(4)
        ]
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=worker_groups,
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=31,
                    num_empty_slots=1,
                    input_queue_depth=5,
                ),
            ]
        )

        solution = scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(solution.stages[0].new_workers) > 0
        assert scheduler._last_intent_deltas["hot"] > 0
        assert len(solution.stages[0].new_workers) == scheduler._last_intent_deltas["hot"]
