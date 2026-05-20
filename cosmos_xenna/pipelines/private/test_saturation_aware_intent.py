# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SaturationAwareScheduler._compute_intent_deltas``.

The orchestrator's ``autoscale()`` runs the per-stage decision
pipeline after Phase B floor enforcement and stores the resulting
signed worker-count intents on ``self._last_intent_deltas``.
Saturation-driven scale-up (Phase C) and scale-down (Phase D) will
consume these intents in subsequent iterations; the current
contract is intent-only.

This module pins:

    * Intent dict shape (active stages keyed; finished absent).
    * Shape-mismatch validation surfaces a ``ValueError`` rather
      than a bare ``KeyError``.
    * ``Solution`` is unaffected by intent computation -- the worker
      adds / removes the planner stages are independent of the
      intent dict.
    * Multi-cycle classifier-state convergence is observable.
    * The intent dict is reset on every cycle and on ``setup()``.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 8) -> resources.ClusterResources:
    """Single-node CPU cluster sufficient for ProblemStage construction."""
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


def _problem_with_stages(
    stage_names: list[str],
    cluster: resources.ClusterResources | None = None,
) -> data_structures.Problem:
    """Build a real ``Problem`` with one CPU stage per name. Order preserved."""
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


def _problem_state_with_signals(
    stage_specs: list[tuple[str, int, int, int, int, int, bool]],
) -> data_structures.ProblemState:
    """Build a real ``ProblemState`` populating all three slot signals.

    Args:
        stage_specs: list of ``(stage_name, num_workers, slots_per_worker,
            num_used_slots, num_empty_slots, input_queue_depth, is_finished)``.
    """
    states = []
    for name, num_workers, slots, used, empty, queue, finished in stage_specs:
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
                num_used_slots=used,
                num_empty_slots=empty,
                input_queue_depth=queue,
            )
        )
    return data_structures.ProblemState(states)


class TestIntentDictShape:
    """Pin the shape of ``_last_intent_deltas`` produced by every cycle."""

    def test_zero_stage_pipeline_yields_empty_intent_dict(self) -> None:
        """An empty problem produces an empty intent dict, no error."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages([]))
        ps = data_structures.ProblemState([])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas == {}

    def test_all_finished_pipeline_yields_empty_intent_dict(self) -> None:
        """Finished stages do not produce intent entries."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))
        ps = _problem_state_with_signals(
            [
                ("A", 1, 1, 1, 0, 0, True),
                ("B", 1, 1, 1, 0, 0, True),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas == {}

    def test_mixed_finished_active_pipeline_keys_only_active_stages(self) -> None:
        """Finished stages absent; active stages each get one entry."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["upstream", "draining", "downstream"]))
        ps = _problem_state_with_signals(
            [
                ("upstream", 1, 1, 1, 0, 0, False),
                ("draining", 1, 1, 1, 0, 0, True),
                ("downstream", 1, 1, 0, 1, 0, False),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert sorted(scheduler._last_intent_deltas) == ["downstream", "upstream"]


class TestShapeMismatchSurfacesAsInvariantError:
    """The pre-phase gate must raise a contextual invariant error on a name mismatch."""

    def test_unknown_stage_in_problem_state_raises_scheduler_invariant_error(self) -> None:
        """A stage in ``problem_state`` not present in ``setup()`` fails before Phase A."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_signals([("ghost", 1, 1, 1, 0, 0, False)])

        with pytest.raises(SchedulerInvariantError, match="ghost"):
            scheduler.autoscale(time=0.0, problem_state=ps)


class TestColdStartDoesNotCrash:
    """A stage with zero slots (no actors yet) must not crash the pipeline."""

    def test_zero_slot_stage_produces_intent_zero(self) -> None:
        """Cold-start (no actors) with no prior EWMA returns intent 0; classifier holds."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_signals([("A", 0, 1, 0, 0, 0, False)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas == {"A": 0}


class TestSolutionShapeReflectsIntentAfterPhaseC:
    """Positive intent flows into the Solution as worker adds via Phase C.

    The saturation-aware scheduler applies positive intent deltas via
    ``ctx.try_add_worker`` before emitting the Solution. The Solution
    therefore reflects whichever portion of the intent the cluster
    could absorb. Negative intent (Phase D) and non-fatal capacity
    exhaustion remain decoupled from this contract; both are pinned
    in ``test_saturation_aware_phase_c_basic.py``.
    """

    def test_saturated_critical_signal_grows_on_first_cycle(self) -> None:
        """First-cycle SATURATED_CRITICAL fires immediately and grows the stage.

        ``saturated_critical_streak_min_cycles`` defaults to ``1`` so
        :func:`should_fire_action` returns True on the very first
        cycle; the resulting positive intent flows through Phase C
        ``try_add_worker`` calls and the Solution carries the new
        workers. Pins the integration: classifier output ->
        intent dict -> Phase C planner mutation -> Solution.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        # 8-CPU cluster (default) -> 4 seeded workers + headroom for adds.
        scheduler.setup(_problem_with_stages(["hot"]))
        # 4 workers, 8 slots/worker = 32 slots; 31 used, 1 empty -> ratio ~ 0.03
        # falls below activation threshold (0.035 at c=8 with default K=0.30).
        ps = _problem_state_with_signals([("hot", 4, 8, 31, 1, 5, False)])

        solution = scheduler.autoscale(time=0.0, problem_state=ps)

        intent = scheduler._last_intent_deltas["hot"]
        assert intent > 0, "expected SATURATED_CRITICAL to produce positive intent"
        assert len(solution.stages[0].new_workers) == intent
        assert solution.stages[0].deleted_workers == []


class TestMultiCycleClassifierConvergence:
    """Across cycles the per-stage classifier converges to a stable state."""

    def test_steady_busy_signal_settles_into_saturated_zone(self) -> None:
        """A steady-busy signal classifies the stage in the saturated zone after enough cycles.

        Pins that ``_compute_intent_deltas`` actually runs the
        classifier each cycle and the EWMA tracks the live ratio
        across multiple ``autoscale()`` calls.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["busy"]))
        ps = _problem_state_with_signals([("busy", 4, 8, 30, 2, 10, False)])

        for _ in range(5):
            scheduler.autoscale(time=0.0, problem_state=ps)

        runtime = scheduler._stage_states["busy"]
        assert runtime.classifier_state in {StageState.SATURATED, StageState.SATURATED_CRITICAL}
        assert runtime.classifier_streak >= 1
        assert runtime.slots_empty_ratio_ewma is not None
        assert 0.0 <= runtime.slots_empty_ratio_ewma <= 1.0

    def test_steady_idle_signal_settles_into_over_provisioned_or_starved(self) -> None:
        """A steady-idle signal classifies the stage in the idle zone after enough cycles."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["idle"]))
        # Mostly empty slots and zero queue -> STARVED zone.
        ps = _problem_state_with_signals([("idle", 4, 8, 4, 28, 0, False)])

        for _ in range(5):
            scheduler.autoscale(time=0.0, problem_state=ps)

        runtime = scheduler._stage_states["idle"]
        assert runtime.classifier_state in {StageState.OVER_PROVISIONED, StageState.STARVED}


class TestIntentDictResetsAcrossCycles:
    """The intent dict is reassigned -- never appended -- on every cycle."""

    def test_setup_resets_intent_dict_to_empty(self) -> None:
        """A fresh ``setup()`` clears any prior cycle's intent dict."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem_with_stages(["A"]))
        ps_first = _problem_state_with_signals([("A", 1, 1, 1, 0, 0, False)])
        scheduler.autoscale(time=0.0, problem_state=ps_first)
        assert scheduler._last_intent_deltas != {}

        scheduler.setup(_problem_with_stages(["B"]))

        assert scheduler._last_intent_deltas == {}

    def test_each_cycle_reassigns_the_intent_dict(self) -> None:
        """Successive cycles produce distinct dict instances rather than mutating one in place."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_signals([("A", 1, 1, 1, 0, 0, False)])
        scheduler.autoscale(time=0.0, problem_state=ps)
        first_dict = scheduler._last_intent_deltas

        scheduler.autoscale(time=0.0, problem_state=ps)
        second_dict = scheduler._last_intent_deltas

        assert first_dict is not second_dict


class TestScaleHandlesLargePipeline:
    """A 100-stage pipeline runs the intent loop without surprises."""

    def test_one_hundred_stage_pipeline_produces_one_hundred_intents(self) -> None:
        """Each active stage in a large pipeline gets exactly one intent entry."""
        cluster = _cluster(total_cpus_per_node=200)
        stage_names = [f"s{i}" for i in range(100)]
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(stage_names, cluster=cluster))
        ps = _problem_state_with_signals([(name, 1, 1, 0, 1, 0, False) for name in stage_names])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(scheduler._last_intent_deltas) == 100
        assert set(scheduler._last_intent_deltas) == set(stage_names)
