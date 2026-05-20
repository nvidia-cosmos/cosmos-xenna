# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SaturationAwareScheduler._run_phase_c_grow`` DAG-priority + stuck-plan counter.

The scheduler extends single-target growth into multi-target
DAG-priority growth and adds the per-stage stuck-plan counter that
feeds the later watchdog. The contract under test:

    * Stages with positive intent are walked downstream-first when
      ``enable_dag_priority_growth=True`` (the default); in problem
      order otherwise.
    * Every stage with positive intent is attempted independently of
      any earlier stage's placement failure. One blocked bottleneck
      must not stop growth attempts for other saturated stages.
    * ``_stuck_plan_counters`` tracks consecutive partial-grow cycles
      per stage: increments by 1 when ``added < intent``; resets to
      ``0`` when ``added == intent`` OR ``intent <= 0``.
    * ``compute_dag_depth_order`` returns indices deepest-first for
      the linear streaming-pipeline model.

Tests inject the intent dict via
``patch.object(scheduler, "_compute_intent_deltas", ...)`` so each
adversarial case exercises placement without rigging classifier
signals.
"""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.dag_priority import compute_dag_depth_order
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog``."""
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
    """CPU-only cluster sized for the contended-slot DAG fixtures."""
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
    """Build a Problem with one CPU stage per ``(name, requested_num_workers)``."""
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
    """Build a ProblemState. Tuples: ``(name, num_workers, slots, is_finished)``."""
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


def _scheduler(
    stage_specs: list[tuple[str, int | None]],
    *,
    cluster: resources.ClusterResources | None = None,
    enable_dag_priority_growth: bool = True,
) -> SaturationAwareScheduler:
    """Build a setup-completed scheduler. ``cluster`` defaults to ``_cluster()``."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        enable_dag_priority_growth=enable_dag_priority_growth,
        stage_defaults=SaturationAwareStageConfig(min_workers=1),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_specs, cluster=cluster))
    return scheduler


def _autoscale_with_intents(
    scheduler: SaturationAwareScheduler,
    state: data_structures.ProblemState,
    intents: dict[str, int],
) -> data_structures.Solution:
    """Run autoscale with ``intents`` injected as ``_compute_intent_deltas`` output."""

    def _inject(_ctx: object, _state: object, **_kwargs: object) -> dict[str, int]:
        return dict(intents)

    with patch.object(scheduler, "_compute_intent_deltas", side_effect=_inject):
        return scheduler.autoscale(time=0.0, problem_state=state)


class TestComputeDagDepthOrder:
    """The pure helper returns stage indices deepest-first for linear pipelines."""

    def test_zero_stage_problem_returns_empty_list(self) -> None:
        """A problem with no stages produces an empty order, no exception."""
        assert compute_dag_depth_order(_problem([])) == []

    def test_single_stage_problem_returns_one_index(self) -> None:
        """Single-stage pipeline: only valid index is 0."""
        assert compute_dag_depth_order(_problem([("only", None)])) == [0]

    def test_three_stage_problem_returns_deepest_first(self) -> None:
        """Three stages [A, B, C] -> DAG order [2, 1, 0]."""
        assert compute_dag_depth_order(_problem([("A", None), ("B", None), ("C", None)])) == [2, 1, 0]


class TestDagPriorityCanonical:
    """Pin a 3-stage pipeline where one free slot is contended."""

    @staticmethod
    def _three_stage_state() -> data_structures.ProblemState:
        """Three stages, 1 worker each. Pair with a 4-CPU cluster -> 1 CPU free."""
        return _problem_state(
            [
                ("s0", 1, 1, False),
                ("s1", 1, 1, False),
                ("s2", 1, 1, False),
            ]
        )

    def test_dag_priority_gives_contended_slot_to_deepest_stage(self) -> None:
        """With ``enable_dag_priority_growth=True``, stage[2] wins the 1 free CPU."""
        scheduler = _scheduler(
            [("s0", None), ("s1", None), ("s2", None)],
            cluster=_cluster(total_cpus_per_node=4),
            enable_dag_priority_growth=True,
        )

        solution = _autoscale_with_intents(scheduler, self._three_stage_state(), {"s0": 1, "s2": 1})

        assert len(solution.stages[2].new_workers) == 1, "deepest stage should win the contended slot"
        assert solution.stages[0].new_workers == [], "shallowest stage receives no add this cycle"

    def test_dag_priority_increments_stuck_counter_for_shallower_stage(self) -> None:
        """Stage that did not get its add increments ``_stuck_plan_counters`` exactly once."""
        scheduler = _scheduler(
            [("s0", None), ("s1", None), ("s2", None)],
            cluster=_cluster(total_cpus_per_node=4),
            enable_dag_priority_growth=True,
        )

        _autoscale_with_intents(scheduler, self._three_stage_state(), {"s0": 1, "s2": 1})

        assert scheduler._stuck_plan_counters["s0"] == 1
        assert scheduler._stuck_plan_counters["s2"] == 0

    def test_problem_order_iteration_when_dag_priority_flag_disabled(self) -> None:
        """With ``enable_dag_priority_growth=False``, stage[0] (problem-order first) wins."""
        scheduler = _scheduler(
            [("s0", None), ("s1", None), ("s2", None)],
            cluster=_cluster(total_cpus_per_node=4),
            enable_dag_priority_growth=False,
        )

        _autoscale_with_intents(scheduler, self._three_stage_state(), {"s0": 1, "s2": 1})

        assert scheduler._stuck_plan_counters["s0"] == 0, "shallowest wins under problem-order"
        assert scheduler._stuck_plan_counters["s2"] == 1, "deepest is starved under problem-order"


class TestIndependentMultiTargetGrowth:
    """Per-stage placement failure does NOT skip other stages with positive intent.

    Multiple failed adds in one cycle each produce independent
    stuck-plan increments rather than the loop short-circuiting on
    first miss.
    """

    def test_capacity_exhaustion_does_not_short_circuit_remaining_stages(self) -> None:
        """3-stage chain, 1 free CPU, all stages with intent=1 -> two stuck increments."""
        scheduler = _scheduler(
            [("s0", None), ("s1", None), ("s2", None)],
            cluster=_cluster(total_cpus_per_node=4),
        )
        state = _problem_state(
            [("s0", 1, 1, False), ("s1", 1, 1, False), ("s2", 1, 1, False)],
        )

        _autoscale_with_intents(scheduler, state, {"s0": 1, "s1": 1, "s2": 1})

        assert scheduler._stuck_plan_counters["s2"] == 0, "deepest grew, counter reset"
        assert scheduler._stuck_plan_counters["s1"] == 1, "stage[1] was attempted (and failed)"
        assert scheduler._stuck_plan_counters["s0"] == 1, "stage[0] was attempted (and failed)"


class TestStuckPlanAccumulation:
    """The counter must be monotonic across consecutive failed cycles."""

    def test_counter_increments_monotonically_across_consecutive_stuck_cycles(self) -> None:
        """Three consecutive same-shape stuck cycles produce counter values 1, 2, 3."""
        scheduler = _scheduler(
            [("A", None)],
            cluster=_cluster(total_cpus_per_node=1),
        )
        state = _problem_state([("A", 1, 1, False)])

        observed: list[int] = []
        for _ in range(3):
            _autoscale_with_intents(scheduler, state, {"A": 5})
            observed.append(scheduler._stuck_plan_counters["A"])

        assert observed == [1, 2, 3]


class TestStuckPlanReset:
    """The counter resets on the OK paths."""

    def test_full_grow_resets_counter(self) -> None:
        """Full intent satisfied -> counter is 0 even if prior cycle was stuck."""
        scheduler = _scheduler(
            [("A", None)],
            cluster=_cluster(total_cpus_per_node=8),
        )
        # Manually seed a prior-stuck state without rigging the cluster.
        scheduler._stuck_plan_counters["A"] = 7
        state = _problem_state([("A", 1, 1, False)])

        _autoscale_with_intents(scheduler, state, {"A": 2})

        assert scheduler._stuck_plan_counters["A"] == 0

    def test_intent_zero_resets_counter_after_prior_stuck(self) -> None:
        """A stage that recovers to ``intent <= 0`` clears its stuck history."""
        scheduler = _scheduler(
            [("A", None)],
            cluster=_cluster(total_cpus_per_node=1),
        )
        state = _problem_state([("A", 1, 1, False)])

        _autoscale_with_intents(scheduler, state, {"A": 5})
        assert scheduler._stuck_plan_counters["A"] == 1

        _autoscale_with_intents(scheduler, state, {"A": 0})

        assert scheduler._stuck_plan_counters["A"] == 0

    def test_negative_intent_resets_counter(self) -> None:
        """``intent < 0`` (Phase D territory) also resets the counter."""
        scheduler = _scheduler(
            [("A", None)],
            cluster=_cluster(total_cpus_per_node=1),
        )
        state = _problem_state([("A", 1, 1, False)])

        _autoscale_with_intents(scheduler, state, {"A": 5})
        _autoscale_with_intents(scheduler, state, {"A": -2})

        assert scheduler._stuck_plan_counters["A"] == 0

    def test_setup_resets_counters(self) -> None:
        """A fresh ``setup()`` clears the stuck-plan dict completely."""
        scheduler = _scheduler(
            [("A", None)],
            cluster=_cluster(total_cpus_per_node=1),
        )
        _autoscale_with_intents(scheduler, _problem_state([("A", 1, 1, False)]), {"A": 5})
        assert scheduler._stuck_plan_counters != {}

        scheduler.setup(_problem([("B", None)]))

        assert scheduler._stuck_plan_counters == {}


class TestStuckPlanFinishedStage:
    """``is_finished`` short-circuits before the counter update; prior values persist."""

    def test_counter_persists_across_is_finished_transition(self) -> None:
        """Cycle 1 leaves counter=1; cycle 2 with the same stage finished does not touch it."""
        scheduler = _scheduler(
            [("A", None)],
            cluster=_cluster(total_cpus_per_node=1),
        )
        _autoscale_with_intents(scheduler, _problem_state([("A", 1, 1, False)]), {"A": 5})
        assert scheduler._stuck_plan_counters["A"] == 1

        _autoscale_with_intents(scheduler, _problem_state([("A", 1, 1, True)]), {"A": 5})

        assert scheduler._stuck_plan_counters["A"] == 1, (
            "current contract: finished short-circuit precedes the counter update; "
            "previously-stuck stages retain their counter value across the transition"
        )


class TestStuckPlanWarning:
    """One WARNING per stuck stage with the stage name and ``deficit`` text."""

    def test_per_stuck_stage_warning_in_dag_order(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """3-stage chain with all positive intent + 1 free slot -> two stuck-stage WARNINGs."""
        scheduler = _scheduler(
            [("s0", None), ("s1", None), ("s2", None)],
            cluster=_cluster(total_cpus_per_node=4),
        )
        state = _problem_state(
            [("s0", 1, 1, False), ("s1", 1, 1, False), ("s2", 1, 1, False)],
        )

        _autoscale_with_intents(scheduler, state, {"s0": 1, "s1": 1, "s2": 1})

        warnings = [r.getMessage() for r in loguru_caplog.records if r.levelname == "WARNING"]
        scaleup = [m for m in warnings if "saturation-aware scale-up" in m]
        assert len(scaleup) == 2
        assert any("'s0'" in m and "deficit" in m for m in scaleup)
        assert any("'s1'" in m and "deficit" in m for m in scaleup)


class TestDagPriorityScale:
    """100-stage pipeline with one contended slot exercises iteration order at scale."""

    def test_one_hundred_stage_pipeline_grows_deepest_first(self) -> None:
        """All 100 stages have positive intent; only deepest gets the contended slot."""
        stage_names = [f"s{i:03d}" for i in range(100)]
        # 100 seeded workers + 1 free CPU slot.
        scheduler = _scheduler(
            [(n, None) for n in stage_names],
            cluster=_cluster(total_cpus_per_node=101),
        )
        state = _problem_state([(n, 1, 1, False) for n in stage_names])

        solution = _autoscale_with_intents(scheduler, state, dict.fromkeys(stage_names, 1))

        assert len(solution.stages[99].new_workers) == 1
        assert sum(len(s.new_workers) for s in solution.stages[:99]) == 0
        assert scheduler._stuck_plan_counters[stage_names[99]] == 0
        assert all(scheduler._stuck_plan_counters[n] == 1 for n in stage_names[:99])
