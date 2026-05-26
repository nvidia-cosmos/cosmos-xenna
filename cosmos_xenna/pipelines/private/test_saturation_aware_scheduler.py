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

"""Public-API tests for ``SaturationAwareScheduler.setup`` and ``autoscale``.

Pins the scheduler's two integration points with the rest of the
streaming executor:

  * ``setup()`` walks ``problem.rust.stages`` and builds a per-stage
    runtime state map keyed by stage name.
  * ``autoscale()`` walks ``problem_state.rust.stages`` and emits one
    ``StageSolution`` per stage in the same order, preserving the
    existing ``slots_per_worker`` and producing no scaling work
    (no-op until the per-stage pipeline is wired to real slot
    signals).

Tests use real ``Problem`` and ``ProblemState`` objects (not mocks)
so the Python -> Rust round-trips are exercised end-to-end.
"""

import copy
import math
import threading

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck import bottleneck_phase
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime import Regime
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState, StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster() -> resources.ClusterResources:
    """Single-node CPU cluster sufficient for ProblemStage construction."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0"),
        },
    )


def _problem_with_stages(stage_names: list[str]) -> data_structures.Problem:
    """Build a real ``Problem`` with one CPU stage per name. Order preserved."""
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


def _problem_state(stage_specs: list[tuple[str, int, int]]) -> data_structures.ProblemState:
    """Build a real ``ProblemState`` with no slot-signal population.

    Each worker carries a single 1-CPU allocation on ``node-0`` so the
    snapshot is consistent with the 8-CPU cluster from
    :func:`_cluster` and ``AutoscalePlanContext.from_problem_state``
    can seed the per-worker allocations without failing the
    consistency check that rejects empty-resource workers.

    Args:
        stage_specs: list of (stage_name, num_workers, slots_per_worker).
    """
    states = []
    for name, num_workers, slots in stage_specs:
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
                is_finished=False,
            )
        )
    return data_structures.ProblemState(states)


def _problem_state_with_slot_signals(
    stage_specs: list[tuple[str, int, int, int, int]],
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` whose stages populate ``num_used_slots`` / ``num_empty_slots``.

    Args:
        stage_specs: list of (stage_name, num_workers, slots_per_worker,
            num_used_slots, num_empty_slots). Used by the regime-aware
            integration tests; the regime detector reads the slot
            signals to compute the cluster-wide idle fraction.
    """
    states = []
    for name, num_workers, slots, used, empty in stage_specs:
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
                is_finished=False,
                num_used_slots=used,
                num_empty_slots=empty,
            )
        )
    return data_structures.ProblemState(states)


def _current_regime(scheduler: SaturationAwareScheduler) -> Regime:
    """Return the scheduler's current regime without narrowing the mutable field in tests."""
    return scheduler.ledgers.regime_state.current_regime


class TestSetup:
    """``setup()`` builds a per-stage runtime state map keyed by stage name."""

    def test_state_map_keyed_by_stage_name(self) -> None:
        """One ``StageRuntimeState`` per stage, keyed by the stage's name."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B", "C"]))
        assert set(scheduler.ledgers.stage_states) == {"A", "B", "C"}
        for name in ("A", "B", "C"):
            assert isinstance(scheduler.ledgers.stage_states[name], StageRuntimeState)
            assert scheduler.ledgers.stage_states[name].stage_name == name

    def test_stage_names_preserve_pipeline_order(self) -> None:
        """``stage_names`` reflects DAG order so deterministic iteration matches Solution order."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["upstream", "middle", "downstream"]))
        assert scheduler.pipeline.stage_names == ("upstream", "middle", "downstream")

    def test_runtime_state_starts_at_default_values(self) -> None:
        """Newly constructed runtime state matches ``StageRuntimeState`` defaults."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        state = scheduler.ledgers.stage_states["A"]
        assert state.classifier.slots_empty_ratio_ewma is None
        assert state.classifier.last_valid_slots_empty_ratio_ewma is None
        assert state.classifier.streak == 0
        assert state.growth.streak == 0
        assert state.growth.prev_workers == 0

    def test_handles_single_stage_pipeline(self) -> None:
        """Smallest non-trivial pipeline -- one stage."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["only"]))
        assert scheduler.pipeline.stage_names == ("only",)
        assert list(scheduler.ledgers.stage_states) == ["only"]

    def test_setup_can_be_called_again_to_rebuild_state(self) -> None:
        """A second ``setup()`` replaces the prior state; useful for test isolation and reset paths."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        scheduler.setup(_problem_with_stages(["X", "Y"]))
        assert scheduler.pipeline.stage_names == ("X", "Y")
        assert "A" not in scheduler.ledgers.stage_states
        assert set(scheduler.ledgers.stage_states) == {"X", "Y"}

    def test_setup_stores_problem_and_config_by_reference(self) -> None:
        """Scheduler holds the same objects passed in (no deep copy).

        Pins the contract so a future refactor that adds an unintended
        ``copy.deepcopy`` is caught -- runtime config patches rely on
        the reference being shared.
        """
        config = SaturationAwareConfig()
        problem = _problem_with_stages(["A"])
        scheduler = SaturationAwareScheduler(config)
        scheduler.setup(problem)
        assert scheduler._config is config
        assert scheduler.pipeline.problem is problem

    def test_handles_many_stages_pipeline(self) -> None:
        """20-stage pipeline: state map size, order, and uniqueness all verified at scale."""
        stage_names = [f"Stage-{i:02d}" for i in range(20)]
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(stage_names))
        assert scheduler.pipeline.stage_names == tuple(stage_names)
        assert len(scheduler.ledgers.stage_states) == 20
        assert all(scheduler.ledgers.stage_states[name].stage_name == name for name in stage_names)


class TestAutoscaleNoOpShape:
    """``autoscale()`` produces a Solution that mirrors the input shape with no scaling work."""

    def test_returns_one_stage_solution_per_problem_state_stage(self) -> None:
        """Solution stage count matches ProblemState stage count -- the streaming.py contract."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B", "C"]))
        solution = scheduler.autoscale(
            time=100.0,
            problem_state=_problem_state([("A", 1, 2), ("B", 2, 2), ("C", 1, 4)]),
        )
        assert len(solution.stages) == 3

    def test_preserves_slots_per_worker_per_stage(self) -> None:
        """Each ``StageSolution.slots_per_worker`` echoes the corresponding ProblemStageState."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B", "C"]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 1, 1), ("B", 2, 2), ("C", 1, 8)]),
        )
        assert [s.slots_per_worker for s in solution.stages] == [1, 2, 8]

    def test_emits_no_workers_added_or_removed(self) -> None:
        """No-op contract: every StageSolution has empty new_workers and deleted_workers lists."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 2), ("B", 1, 1)]),
        )
        for stage in solution.stages:
            assert stage.new_workers == []
            assert stage.deleted_workers == []

    def test_returned_worker_lists_are_empty_lists_not_none(self) -> None:
        """Worker fields are concrete empty lists, not ``None``.

        ``streaming.py:apply_autoscale_result_if_ready`` does
        ``list(result.deleted_workers)`` and iterates ``new_workers``
        with ``for w in ...``. ``None`` would raise ``TypeError``;
        empty list is the silent-no-op contract.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        solution = scheduler.autoscale(time=0.0, problem_state=_problem_state([("A", 1, 2)]))
        stage = solution.stages[0]
        assert isinstance(stage.new_workers, list)
        assert isinstance(stage.deleted_workers, list)


class TestAutoscaleEdgeCases:
    """Edge-case shapes that pin the boundary behaviour of the scheduler."""

    def test_single_stage_pipeline(self) -> None:
        """Single-stage pipeline -- the smallest valid input."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["only"]))
        solution = scheduler.autoscale(time=0.0, problem_state=_problem_state([("only", 1, 1)]))
        assert len(solution.stages) == 1
        assert solution.stages[0].slots_per_worker == 1

    def test_zero_stage_problem_state(self) -> None:
        """Empty pipeline produces an empty Solution."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages([]))
        solution = scheduler.autoscale(time=0.0, problem_state=_problem_state([]))
        assert solution.stages == []

    def test_repeated_calls_are_idempotent_for_no_op_stub(self) -> None:
        """Calling autoscale twice with the same state produces equivalent Solutions.

        The current no-op stub does not maintain state across cycles
        (the per-stage pipeline does not run yet), so this should
        deterministically reproduce the same Solution shape.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state([("A", 2, 4)])
        first = scheduler.autoscale(time=0.0, problem_state=ps)
        second = scheduler.autoscale(time=10.0, problem_state=ps)
        assert [s.slots_per_worker for s in first.stages] == [s.slots_per_worker for s in second.stages]
        assert all(s.new_workers == [] and s.deleted_workers == [] for s in first.stages)
        assert all(s.new_workers == [] and s.deleted_workers == [] for s in second.stages)

    def test_time_parameter_does_not_affect_no_op_output(self) -> None:
        """``autoscale()`` output is independent of the ``time`` argument today.

        Same ``problem_state`` at any time produces the same Solution
        shape because the current decision body stages no decisions.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state([("A", 1, 2)])
        for time_value in (-1e9, 0.0, 1e-6, 1.0, 1e9):
            sol = scheduler.autoscale(time=time_value, problem_state=ps)
            assert [s.slots_per_worker for s in sol.stages] == [2]
            assert sol.stages[0].new_workers == []
            assert sol.stages[0].deleted_workers == []

    def test_finished_stages_still_emit_solution(self) -> None:
        """A stage with ``is_finished=True`` still receives a StageSolution -- no special skip.

        ``streaming.py`` requires ``len(autoscale_result.stages) == len(pools)`` so the
        stage count cannot diverge based on finished state. Pin this contract.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))
        # Build ProblemState manually so we can flip is_finished on the second stage.
        cpu = [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])]
        worker_a = data_structures.ProblemWorkerGroupState.make("a-w0", cpu)
        worker_b = data_structures.ProblemWorkerGroupState.make("b-w0", cpu)
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="A", workers=[worker_a], slots_per_worker=2, is_finished=False
                ),
                data_structures.ProblemStageState(
                    stage_name="B", workers=[worker_b], slots_per_worker=4, is_finished=True
                ),
            ]
        )
        solution = scheduler.autoscale(time=0.0, problem_state=ps)
        assert len(solution.stages) == 2
        assert [s.slots_per_worker for s in solution.stages] == [2, 4]

    def test_stage_state_map_persists_across_autoscale_cycles(self) -> None:
        """``_stage_states`` dict identity is preserved across cycles.

        ``autoscale()`` mutates the existing dict's values in place
        rather than recreating the dict. Identity preservation is the
        contract any per-stage state mutation depends on.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        original_dict = scheduler.ledgers.stage_states
        original_state_a = scheduler.ledgers.stage_states["A"]

        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=_problem_state([("A", 1, 2)]))

        assert scheduler.ledgers.stage_states is original_dict
        assert scheduler.ledgers.stage_states["A"] is original_state_a


class TestUpdateWithMeasurementsAccumulator:
    """``update_with_measurements`` accumulates per-stage completion counts under the lock."""

    def test_empty_measurements_do_not_touch_counts(self) -> None:
        """Empty measurements leave the count AND service-time accumulators untouched."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # Deep-copy so any in-place mutation of a ``StageRuntimeState`` value
        # by ``update_with_measurements`` would surface as inequality below;
        # a shallow ``dict(...)`` would alias the same ``StageRuntimeState``
        # references and mask the regression.
        snapshot_before = copy.deepcopy(scheduler.ledgers.stage_states)
        # ``setup()`` zero-initialises both per-stage accumulators.
        assert scheduler.ledgers.measurements.completed_counts == {"A": 0}
        assert scheduler.ledgers.measurements.completed_service_time_sums == {"A": 0.0}
        empty_measurements = data_structures.Measurements(time=0.0, stage_measurements=[])
        scheduler.update_with_measurements(time=0.0, measurements=empty_measurements)
        assert scheduler.ledgers.stage_states == snapshot_before
        assert scheduler.ledgers.measurements.completed_counts == {"A": 0}
        assert scheduler.ledgers.measurements.completed_service_time_sums == {"A": 0.0}

    def test_task_count_accumulates_across_calls(self) -> None:
        """Each ``TaskMeasurement`` adds 1 to the per-stage count regardless of ``num_returns``."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # First batch: 3 tasks, each producing different num_returns.
        # Counting policy is ``len(task_measurements)`` (matches per-task drain rate).
        ms1 = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[
                        data_structures.TaskMeasurement(0.0, 1.0, num_returns=1),
                        data_structures.TaskMeasurement(0.0, 1.0, num_returns=5),
                        data_structures.TaskMeasurement(0.0, 1.0, num_returns=3),
                    ]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms1)
        assert scheduler.ledgers.measurements.completed_counts["A"] == 3

        # Second batch: 2 more tasks -> running total = 5.
        ms2 = data_structures.Measurements(
            time=1.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[
                        data_structures.TaskMeasurement(0.0, 1.0, num_returns=10),
                        data_structures.TaskMeasurement(0.0, 1.0, num_returns=2),
                    ]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.0, measurements=ms2)
        assert scheduler.ledgers.measurements.completed_counts["A"] == 5

    def test_per_stage_attribution_follows_dag_order(self) -> None:
        """Stage order from the Problem zips with the rust ``stages`` iterator."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["upstream", "downstream"]))

        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                # upstream -> 1 task
                data_structures.StageMeasurements(task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1)]),
                # downstream -> 4 tasks
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(4)]
                ),
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        assert scheduler.ledgers.measurements.completed_counts == {"upstream": 1, "downstream": 4}

    def test_setup_resets_measurement_accumulators(self) -> None:
        """A second ``setup()`` zeroes out the count AND service-time accumulators from the prior pipeline."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.5, 1) for _ in range(7)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        assert scheduler.ledgers.measurements.completed_counts["A"] == 7
        assert scheduler.ledgers.measurements.completed_service_time_sums["A"] == pytest.approx(7 * 0.5)

        # New pipeline -> fresh counters and BOTH per-cycle snapshots
        # cleared so the next sample is cold-start.
        scheduler.setup(_problem_with_stages(["X", "Y"]))
        assert scheduler.ledgers.measurements.completed_counts == {"X": 0, "Y": 0}
        assert scheduler.ledgers.measurements.completed_service_time_sums == {"X": 0.0, "Y": 0.0}
        assert scheduler.ledgers.measurements.last_throughput_sample == {}
        assert scheduler.ledgers.measurements.last_service_time_sample == {}


class TestConsumeThroughputSamples:
    """``_consume_throughput_samples`` derives per-stage tasks/sec from the accumulator."""

    def test_first_cycle_returns_zero_throughput_for_every_stage(self) -> None:
        """The cold-start contract: first cycle has no prior snapshot so dt is undefined."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))

        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(10)]
                ),
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(20)]
                ),
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        samples = scheduler.ledgers.measurements.consume_throughput_samples(now_ts=1.0)
        # Cold-start: every stage seeds the snapshot with current count, returns 0.0.
        assert samples == {"A": 0.0, "B": 0.0}
        # Snapshot now holds the cold-start (count, ts) pair.
        assert scheduler.ledgers.measurements.last_throughput_sample["A"] == (10, 1.0)
        assert scheduler.ledgers.measurements.last_throughput_sample["B"] == (20, 1.0)

    def test_steady_state_returns_dcount_over_dt(self) -> None:
        """Second cycle returns ``(now_count - prev_count) / (now_ts - prev_ts)``."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # Cycle 1: 10 tasks, seed snapshot at t=1.0.
        ms1 = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(10)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms1)
        scheduler.ledgers.measurements.consume_throughput_samples(now_ts=1.0)

        # Cycle 2: 20 more tasks, sampled at t=2.0 -> dcount=20, dt=1.0 -> 20.0 tasks/sec.
        ms2 = data_structures.Measurements(
            time=1.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(20)]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.0, measurements=ms2)
        samples = scheduler.ledgers.measurements.consume_throughput_samples(now_ts=2.0)
        assert samples == {"A": 20.0}

    def test_zero_dt_yields_zero_throughput_no_division_error(self) -> None:
        """A clock that does not advance must return 0.0, not raise ``ZeroDivisionError``."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1)])
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        scheduler.ledgers.measurements.consume_throughput_samples(now_ts=5.0)
        # Same now_ts -> dt = 0; must short-circuit to 0.0 instead of dividing.
        samples = scheduler.ledgers.measurements.consume_throughput_samples(now_ts=5.0)
        assert samples == {"A": 0.0}

    def test_negative_dcount_clamps_to_zero(self) -> None:
        """A reset that lowers the running count must not produce negative throughput.

        In production ``setup()`` would zero the counters AND clear the
        snapshot dict, so this branch is defensive against an unforeseen
        path that resets only the count. ``max(0, ...)`` keeps the
        sample non-negative; pressure thresholds assume tasks/sec is a
        non-negative scalar.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # Manually seed a high snapshot count.
        scheduler.ledgers.measurements.completed_counts["A"] = 100
        scheduler.ledgers.measurements.consume_throughput_samples(now_ts=1.0)
        assert scheduler.ledgers.measurements.last_throughput_sample["A"] == (100, 1.0)

        # Reset the counter (NOT the snapshot) and sample again.
        scheduler.ledgers.measurements.completed_counts["A"] = 50
        samples = scheduler.ledgers.measurements.consume_throughput_samples(now_ts=2.0)
        assert samples == {"A": 0.0}


class TestServiceTimeAccumulator:
    """``update_with_measurements`` accumulates per-stage service-time sums alongside counts."""

    def test_service_time_sum_accumulates_with_count(self) -> None:
        """Each ``TaskMeasurement`` adds its ``end - start`` duration to the per-stage sum."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # 3 tasks with durations 0.5s, 1.0s, 1.5s -> sum = 3.0s.
        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[
                        data_structures.TaskMeasurement(0.0, 0.5, 1),
                        data_structures.TaskMeasurement(0.0, 1.0, 1),
                        data_structures.TaskMeasurement(0.0, 1.5, 1),
                    ]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        assert scheduler.ledgers.measurements.completed_counts["A"] == 3
        assert scheduler.ledgers.measurements.completed_service_time_sums["A"] == pytest.approx(3.0)

    def test_per_stage_service_time_attribution(self) -> None:
        """Stage order in ``StageMeasurements`` zips with ``self._stage_names`` for both accumulators."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["fast", "slow"]))

        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                # fast: 2 tasks at 0.1s each -> sum 0.2s.
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.1, 1) for _ in range(2)]
                ),
                # slow: 4 tasks at 2.0s each -> sum 8.0s.
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 2.0, 1) for _ in range(4)]
                ),
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        assert scheduler.ledgers.measurements.completed_service_time_sums["fast"] == pytest.approx(0.2)
        assert scheduler.ledgers.measurements.completed_service_time_sums["slow"] == pytest.approx(8.0)


class TestConsumeServiceTimeSamples:
    """``_consume_service_time_samples`` derives per-stage mean ``S_k`` from the accumulators."""

    def test_first_cycle_returns_nan_for_every_stage(self) -> None:
        """The cold-start contract: first cycle has no prior snapshot -> NaN per stage."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))

        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.5, 1) for _ in range(10)]
                ),
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 2.0, 1) for _ in range(20)]
                ),
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        samples = scheduler.ledgers.measurements.consume_service_time_samples()
        # Cold-start: every stage returns NaN; bottleneck.py / heterogeneity.py fold NaN into
        # the cold-start branch (gauge observes NaN, stage skipped from argmax).
        assert math.isnan(samples["A"])
        assert math.isnan(samples["B"])
        # Snapshot now holds the cold-start (count, sum) pair so the next cycle has a delta.
        assert scheduler.ledgers.measurements.last_service_time_sample["A"] == (10, pytest.approx(5.0))
        assert scheduler.ledgers.measurements.last_service_time_sample["B"] == (20, pytest.approx(40.0))

    def test_steady_state_returns_dsum_over_dcount(self) -> None:
        """Second cycle returns ``(now_sum - prev_sum) / (now_count - prev_count)``."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # Cycle 1: 10 tasks at 0.5s each -> sum 5.0s; seed snapshot.
        ms1 = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.5, 1) for _ in range(10)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms1)
        scheduler.ledgers.measurements.consume_service_time_samples()

        # Cycle 2: 20 more tasks at 1.0s each -> dsum 20.0s, dcount 20 -> mean 1.0s.
        ms2 = data_structures.Measurements(
            time=1.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(20)]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.0, measurements=ms2)
        samples = scheduler.ledgers.measurements.consume_service_time_samples()
        assert samples["A"] == pytest.approx(1.0)

    def test_no_progress_cycle_returns_nan(self) -> None:
        """A cycle with no new completed tasks has dcount=0 -> NaN (cold-start sentinel)."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # Seed the snapshot with one batch.
        ms = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(task_measurements=[data_structures.TaskMeasurement(0.0, 0.5, 1)])
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms)
        scheduler.ledgers.measurements.consume_service_time_samples()

        # Second cycle with no new measurements -> dcount=0 -> NaN.
        samples = scheduler.ledgers.measurements.consume_service_time_samples()
        assert math.isnan(samples["A"])

    def test_independent_snapshots_from_throughput_consumer(self) -> None:
        """Calling ``_consume_throughput_samples`` first must not corrupt the service-time delta.

        Both consumers carry their own ``(count, ...)`` snapshot; the
        autoscale call site wires both helpers in the same cycle so a
        future re-order of the calls cannot silently zero the
        bottleneck score.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # Cycle 1.
        ms1 = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 0.5, 1) for _ in range(10)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.0, measurements=ms1)
        # Throughput first.
        scheduler.ledgers.measurements.consume_throughput_samples(now_ts=1.0)
        # Then service time.
        scheduler.ledgers.measurements.consume_service_time_samples()

        # Cycle 2.
        ms2 = data_structures.Measurements(
            time=1.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 2.0, 1) for _ in range(5)]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.0, measurements=ms2)
        scheduler.ledgers.measurements.consume_throughput_samples(now_ts=2.0)
        # Even after throughput consumer ran, service-time delta must be (5 tasks * 2.0s) / 5 = 2.0s.
        samples = scheduler.ledgers.measurements.consume_service_time_samples()
        assert samples["A"] == pytest.approx(2.0)


class TestSKEwmaState:
    """Pin the intrinsic ``S_k`` EWMA state and its NaN-safe behaviour.

    The EWMA tracks per-task service time. Per-cycle ``D_k = S_k / c_k``
    is recomputed from this smoothed value plus the live effective
    capacity, so actor-count changes do not need to flow through the
    EWMA's alpha-blend.
    """

    def test_setup_seeds_every_stage_with_nan(self) -> None:
        """Pre-traffic, every stage's S_k EWMA is NaN until a finite sample lands."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))

        assert set(scheduler.ledgers.s_k_ewma.view()) == {"A", "B"}
        assert all(math.isnan(v) for v in scheduler.ledgers.s_k_ewma.view().values())

    def test_first_finite_sample_replaces_nan_seed_without_blending(self) -> None:
        """Cold-start: first finite S_k for a stage is taken as-is (no alpha blend)."""
        scheduler = SaturationAwareScheduler(
            SaturationAwareConfig(bottleneck_d_k_smoothing_level=0.20),
        )
        scheduler.setup(_problem_with_stages(["A", "B"]))

        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 1.5, "B": math.nan})

        assert scheduler.ledgers.s_k_ewma.get("A") == pytest.approx(1.5)
        assert math.isnan(scheduler.ledgers.s_k_ewma.get("B"))

    def test_warm_step_applies_ewma_blend(self) -> None:
        """Subsequent finite samples blend with the prior value via alpha."""
        cfg = SaturationAwareConfig(bottleneck_d_k_smoothing_level=0.25)
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem_with_stages(["A"]))

        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 1.0})
        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 5.0})

        # 1.0 * 0.75 + 5.0 * 0.25 = 0.75 + 1.25 = 2.0
        assert scheduler.ledgers.s_k_ewma.get("A") == pytest.approx(2.0)

    def test_missed_sample_preserves_previous_value(self) -> None:
        """A NaN sample must not corrupt or zero out the prior EWMA."""
        scheduler = SaturationAwareScheduler(
            SaturationAwareConfig(bottleneck_d_k_smoothing_level=0.30),
        )
        scheduler.setup(_problem_with_stages(["A"]))

        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 2.5})
        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": math.nan})

        assert scheduler.ledgers.s_k_ewma.get("A") == pytest.approx(2.5)

    def test_reset_on_setup_drops_prior_state(self) -> None:
        """Re-setup of a recycled scheduler clears the EWMA and bottleneck meta."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 1.5})
        assert math.isfinite(scheduler.ledgers.s_k_ewma.get("A"))

        scheduler.setup(_problem_with_stages(["A", "B"]))

        assert all(math.isnan(v) for v in scheduler.ledgers.s_k_ewma.view().values())
        # ``setup()`` clears the most-recent-cycle observability
        # hook; ``_last_cycle is None`` and the scheduler's own
        # cycle counter resets to zero.
        assert scheduler.last_cycle is None
        assert scheduler.ledgers.cycle_counter == 0
        assert scheduler.ledgers.bottleneck_engagement_state.last_announced is None

    def test_zero_or_negative_sample_treated_as_missed(self) -> None:
        """``S_k <= 0`` is bogus and never replaces a finite EWMA."""
        scheduler = SaturationAwareScheduler(
            SaturationAwareConfig(bottleneck_d_k_smoothing_level=0.30),
        )
        scheduler.setup(_problem_with_stages(["A"]))

        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 1.0})
        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": 0.0})
        bottleneck_phase._update_s_k_ewma(scheduler.runner.bottleneck_services, {"A": -0.5})

        assert scheduler.ledgers.s_k_ewma.get("A") == pytest.approx(1.0)


class TestPressureEndToEnd:
    """End-to-end smoke: measurements -> autoscale updates ``pressure_ewma``."""

    def _problem_state_with_queue(
        self,
        stage_name: str,
        num_workers: int,
        slots_per_worker: int,
        num_used_slots: int,
        num_empty_slots: int,
        input_queue_depth: int,
    ) -> data_structures.ProblemState:
        """Same shape as :func:`_problem_state_with_slot_signals` plus an explicit queue depth."""
        worker_groups = [
            data_structures.ProblemWorkerGroupState.make(
                f"{stage_name}-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(num_workers)
        ]
        return data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name=stage_name,
                    workers=worker_groups,
                    slots_per_worker=slots_per_worker,
                    is_finished=False,
                    num_used_slots=num_used_slots,
                    num_empty_slots=num_empty_slots,
                    input_queue_depth=input_queue_depth,
                )
            ]
        )

    def test_sustained_pressure_threads_through_full_cycle(self) -> None:
        """Two cycles of completed tasks against a non-empty queue populate ``pressure_ewma``.

        The default ``worker_warmup_measurement_grace_s=60`` would zero out the slot
        signals when the test clock is < 60 (so the classifier short-circuits and
        the pressure helper is intentionally skipped). The override pins the grace
        to zero so the integration boundary is exercised at t=1.0 / t=2.0.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
                min_data_points=1,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem_with_stages(["A"]))

        # Cycle 1: seed the throughput snapshot at t=1.0 with no completed work yet,
        # autoscale here cold-starts and observed_throughput stays at 0.0.
        # Slot signals: 31 used / 1 empty -> ratio ~ 0.031 (within SATURATED band).
        # input_queue_depth=200 -> pressure should be high.
        ps = self._problem_state_with_queue(
            stage_name="A",
            num_workers=4,
            slots_per_worker=8,
            num_used_slots=31,
            num_empty_slots=1,
            input_queue_depth=200,
        )

        # 10 tasks completed before cycle 1.
        ms = data_structures.Measurements(
            time=0.5,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(10)]
                )
            ],
        )
        scheduler.update_with_measurements(time=0.5, measurements=ms)
        scheduler.autoscale(time=1.0, problem_state=ps)

        # Cycle 1 is cold-start for throughput (no prior snapshot) so observed_throughput=0.0
        # which triggers the cold-start branch in compute_pressure -> normalized_backlog=cap.
        # pressure should be positive (utilisation * cap).
        first_pressure = scheduler.ledgers.stage_states["A"].pressure.ewma
        assert first_pressure is not None
        assert first_pressure > 0.0

        # Cycle 2: 50 more tasks completed between t=1.0 and t=2.0 -> 50 tasks/sec.
        ms2 = data_structures.Measurements(
            time=1.5,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(50)]
                )
            ],
        )
        scheduler.update_with_measurements(time=1.5, measurements=ms2)
        scheduler.autoscale(time=2.0, problem_state=ps)

        # Cycle 2 has a real throughput sample (50 tasks/sec); the EWMA blends it
        # with the cycle 1 cold-start value so pressure stays elevated.
        second_pressure = scheduler.ledgers.stage_states["A"].pressure.ewma
        assert second_pressure is not None
        # pressure_ewma is bounded by BACKLOG_CAP * 1.0 = 3.0; >= 0 sanity.
        assert 0.0 <= second_pressure <= 3.0


class TestUpdateWithMeasurementsThreadSafety:
    """Concurrent ``update_with_measurements`` callers do not lose or double-count tasks."""

    def test_concurrent_calls_aggregate_correctly(self) -> None:
        """Hammering the accumulator from multiple threads preserves the running total."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))

        # Each thread reports 10 tasks; with 5 threads that is 50 total.
        ms_template = data_structures.Measurements(
            time=0.0,
            stage_measurements=[
                data_structures.StageMeasurements(
                    task_measurements=[data_structures.TaskMeasurement(0.0, 1.0, 1) for _ in range(10)]
                )
            ],
        )

        def call_update() -> None:
            scheduler.update_with_measurements(time=0.0, measurements=ms_template)

        threads = [threading.Thread(target=call_update) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert scheduler.ledgers.measurements.completed_counts["A"] == 50


class TestAutoscalePlanContextLifecycle:
    """``autoscale()`` builds and drains an ``AutoscalePlanContext`` per cycle.

    Pins the contract that the no-op ``Solution`` flows through the
    full Rust planner lifecycle (``from_problem_state`` ->
    ``into_solution``) instead of being constructed directly from
    per-stage ``slots_per_worker`` values. Subsequent decision logic
    will stage worker adds and removes against this context; this
    test class pins the boundary behaviour that logic will build on.
    """

    def test_autoscale_seeds_existing_workers_into_the_context_cluster(self) -> None:
        """Existing workers must be allocated against the context's cluster.

        The 8-CPU cluster from :func:`_cluster` plus 5 1-CPU workers
        across two stages is well below capacity. If the context's
        seed step did not actually consume cluster capacity, a future
        phase that called ``ctx.try_add_worker`` could over-commit
        the cluster. This test exercises seeding for several worker
        counts to pin that the constructor accepts the snapshot AND
        that the seeded cluster reports the correct remaining
        capacity (8 CPUs - 5 used = 3 free) by probing with
        ``try_add_worker``: three 1-CPU adds must succeed before
        the fourth returns ``None``.
        """
        problem = _problem_with_stages(["A", "B"])
        ps = _problem_state([("A", 4, 2), ("B", 1, 1)])

        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(problem)
        solution = scheduler.autoscale(time=0.0, problem_state=ps)
        # Sanity: 5 1-CPU workers fit in the 8-CPU cluster; the
        # constructor would have raised if the seed allocations did
        # not match the cluster shape.
        assert [s.slots_per_worker for s in solution.stages] == [2, 1]

        # Seeded-capacity probe: build a fresh ``AutoscalePlanContext``
        # from the same problem / problem_state and verify the cluster
        # reports exactly 3 free CPU slots. A regression that failed
        # to subtract the seeded workers from cluster capacity would
        # let the probe return non-None on the fourth call (overcommit).
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem, ps)
        for add_index in range(3):
            placement = ctx.try_add_worker(0)
            assert placement is not None, (
                f"add #{add_index + 1} of 3 failed despite 3 free CPU slots; "
                f"seeded cluster did not reflect 8 - 5 = 3 remaining capacity"
            )
        assert ctx.try_add_worker(0) is None, (
            "fourth try_add_worker must return None (cluster exhausted at 8/8 CPUs); "
            "non-None here indicates the seed step double-counted free capacity"
        )

    def test_autoscale_raises_when_called_before_setup(self) -> None:
        """Defensive guard: ``autoscale()`` requires a prior ``setup()`` call."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        with pytest.raises(RuntimeError, match="setup"):
            scheduler.autoscale(time=0.0, problem_state=_problem_state([("A", 1, 1)]))

    def test_autoscale_constructs_a_fresh_context_per_cycle(self) -> None:
        """Each cycle owns its own context; no state bleeds between calls.

        ``AutoscalePlanContext`` is single-shot: ``into_solution`` drains
        it and the read-only accessors stay valid afterwards. Three
        consecutive cycles with the same ``problem_state`` must
        produce three Solutions whose shape matches the input -- if
        a stale context were reused, the second cycle's seeding would
        attempt to re-allocate cluster capacity that the first call
        already consumed and the constructor would raise.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))
        ps = _problem_state([("A", 2, 4), ("B", 1, 1)])
        for _ in range(3):
            solution = scheduler.autoscale(time=0.0, problem_state=ps)
            assert [s.slots_per_worker for s in solution.stages] == [4, 1]
            assert all(s.new_workers == [] and s.deleted_workers == [] for s in solution.stages)


class TestThresholdResolutionTiming:
    """Per-stage classifier thresholds resolve lazily from runtime ``slots_per_worker``.

    The formula's ``c`` is the actor concurrency
    (``ProblemStageState.slots_per_worker``), NOT the per-call batch
    size (``ProblemStage.stage_batch_size``, default 1). The
    distinction matters: using the per-call batch would yield a
    uniform ``saturation_aggressiveness/sqrt(1) = 0.30`` for every
    default stage, defeating the auto-derivation premise. These
    tests pin the wiring contract.
    """

    def test_setup_does_not_resolve_thresholds(self) -> None:
        """``setup()`` cannot resolve -- ``slots_per_worker`` only arrives in ``autoscale()``."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        assert scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds is None

    def test_first_autoscale_resolves_thresholds_from_runtime_slots_per_worker(self) -> None:
        """First cycle reads ``ProblemStageState.slots_per_worker`` and auto-derives.

        Two stages share the same default config (no per-stage override)
        but differ in runtime ``slots_per_worker`` -- 1 vs 64. The
        resolver must produce different thresholds for them
        (``0.30 / sqrt(1) = 0.30`` vs ``0.30 / sqrt(64) = 0.0375``).
        A regression that read ``stage_batch_size`` instead would
        produce identical thresholds for both stages.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["small_c", "large_c"]))
        ps = _problem_state([("small_c", 1, 1), ("large_c", 1, 64)])
        scheduler.autoscale(time=0.0, problem_state=ps)

        small_resolved = scheduler.ledgers.stage_states["small_c"].classifier.resolved_thresholds
        large_resolved = scheduler.ledgers.stage_states["large_c"].classifier.resolved_thresholds
        assert small_resolved is not None
        assert large_resolved is not None
        assert small_resolved.slots_per_actor == 1
        assert large_resolved.slots_per_actor == 64
        assert small_resolved.saturation_threshold == pytest.approx(0.30, rel=1e-3)
        assert large_resolved.saturation_threshold == pytest.approx(0.30 / 8.0, rel=1e-3)

    def test_resolution_is_idempotent_across_cycles(self) -> None:
        """Once resolved, the runtime state is reused across cycles.

        Mid-run changes to a stage's ``slots_per_worker`` (operator
        adjusts via ``Solution.slots_per_worker``) do NOT trigger
        re-resolution by design -- the operator who reshapes a stage
        is responsible for restarting if they also want
        threshold re-derivation.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # First cycle: resolve at slots=8.
        scheduler.autoscale(time=0.0, problem_state=_problem_state([("A", 1, 8)]))
        first_resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert first_resolved is not None
        assert first_resolved.slots_per_actor == 8
        # Second cycle: same stage, different slots_per_worker -- resolution is sticky.
        scheduler.autoscale(time=10.0, problem_state=_problem_state([("A", 1, 64)]))
        second_resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert second_resolved is first_resolved
        assert second_resolved.slots_per_actor == 8


class TestRegimeAwareAggressiveness:
    """Cluster-wide Halfin-Whitt regime detection drives the per-cycle aggressiveness lift.

    Pin the per-cycle regime evaluation, the asymmetric hysteresis on
    transitions, the lifted-aggressiveness re-resolve on a transition,
    the disable flag, and the no-signal defensive guard so a future
    tweak surfaces as a precise failure.
    """

    def test_no_slot_signals_keeps_state_in_sub_hw(self) -> None:
        """Production cycles without slot signals leave the regime in sub-HW."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # `_problem_state` does NOT populate num_used_slots / num_empty_slots.
        for _ in range(5):
            scheduler.autoscale(time=0.0, problem_state=_problem_state([("A", 4, 8)]))
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT
        # Stage thresholds resolved with base aggressiveness 0.30.
        resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.30)

    def test_sustained_busy_signal_transitions_to_super_hw_after_streak(self) -> None:
        """Three consecutive busy cycles commit the regime transition."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # 4 workers, each carrying 8 slots, 31 used / 1 empty -> idle ~ 0.031.
        # threshold = 1/sqrt(4) = 0.50 -> 0.031 < 0.50 -> super-HW entry signal.
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=ps)
        assert _current_regime(scheduler) is Regime.SUPER_HALFIN_WHITT

    def test_super_hw_transition_relifts_aggressiveness(self) -> None:
        """A transition into super-HW re-resolves thresholds with base + lift."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=ps)
        resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert resolved is not None
        # Base 0.30 + lift 0.15 = 0.45.
        assert resolved.saturation_aggressiveness == pytest.approx(0.45)

    def test_non_default_lift_value_is_consumed(self) -> None:
        """The configured lift, not a hardcoded default, drives super-HW resolution."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig(super_halfin_whitt_aggressiveness_lift=0.05))
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=ps)
        resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.35)

    def test_non_default_transition_streak_is_consumed(self) -> None:
        """The configured streak length, not a hardcoded default, gates transitions."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig(regime_transition_streak_cycles=2))
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        scheduler.autoscale(time=0.0, problem_state=ps)
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT
        scheduler.autoscale(time=0.0, problem_state=ps)
        assert _current_regime(scheduler) is Regime.SUPER_HALFIN_WHITT

    def test_oscillation_around_threshold_does_not_flap_regime(self) -> None:
        """Cluster idle oscillating across the boundary holds sub-HW (streak resets)."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # 4 workers, 8 slots/worker -> 32 total slots; threshold = 1/sqrt(4) = 0.50.
        below = _problem_state_with_slot_signals([("A", 4, 8, 19, 13)])  # idle ~ 0.41 < 0.50
        above = _problem_state_with_slot_signals([("A", 4, 8, 14, 18)])  # idle ~ 0.56 >= 0.50
        # Two busy cycles, one idle, repeat: streak never reaches 3.
        for ps in (below, below, above, below, below, above):
            scheduler.autoscale(time=0.0, problem_state=ps)
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT

    def test_disabled_flag_pins_aggressiveness_at_base(self) -> None:
        """``enable_regime_aware_aggressiveness=False`` skips regime tracking entirely."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig(enable_regime_aware_aggressiveness=False))
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(10):
            scheduler.autoscale(time=0.0, problem_state=ps)
        # Regime state stays at default (sub-HW); thresholds resolved at base.
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT
        resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.30)

    def test_super_hw_exit_re_resolves_with_base_aggressiveness(self) -> None:
        """Transitioning back to sub-HW re-resolves with the base aggressiveness."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # Enter super-HW.
        busy = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=busy)
        assert _current_regime(scheduler) is Regime.SUPER_HALFIN_WHITT
        # threshold = 1/sqrt(4) = 0.50; exit band = 0.75. Use idle = 0.84.
        idle = _problem_state_with_slot_signals([("A", 4, 8, 5, 27)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=idle)
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT
        resolved = scheduler.ledgers.stage_states["A"].classifier.resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.30)

    def test_setup_resets_regime_hysteresis(self) -> None:
        """A reused scheduler starts the next setup at base aggressiveness."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        busy = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=busy)
        assert _current_regime(scheduler) is Regime.SUPER_HALFIN_WHITT

        scheduler.setup(_problem_with_stages(["B"]))
        scheduler.autoscale(time=0.0, problem_state=_problem_state([("B", 4, 8)]))
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT
        resolved = scheduler.ledgers.stage_states["B"].classifier.resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.30)

    def test_mixed_slot_signal_snapshot_keeps_regime_state_unchanged(self) -> None:
        """One unreported active worker stage makes the cluster signal unavailable."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["reported", "missing"]))
        reported_workers = [
            data_structures.ProblemWorkerGroupState.make(
                f"reported-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(4)
        ]
        missing_workers = [
            data_structures.ProblemWorkerGroupState.make(
                f"missing-w{i}",
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            )
            for i in range(4)
        ]
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="reported",
                    workers=reported_workers,
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=31,
                    num_empty_slots=1,
                ),
                data_structures.ProblemStageState(
                    stage_name="missing",
                    workers=missing_workers,
                    slots_per_worker=8,
                    is_finished=False,
                ),
            ]
        )
        for _ in range(5):
            scheduler.autoscale(time=0.0, problem_state=ps)
        assert _current_regime(scheduler) is Regime.SUB_HALFIN_WHITT

    def test_regime_transition_resets_threshold_relative_classifier_history(self) -> None:
        """Pre-transition classifier streaks do not carry across the threshold band change.

        The seeded ``classifier_state=SATURATED, classifier_streak=7``
        represent stale history captured under the sub-Halfin-Whitt
        thresholds. When the regime transitions to super-Halfin-Whitt
        the scheduler resets the classifier state to ``NORMAL`` and
        the streak to ``0`` (see
        :meth:`RegimeController.update`); the per-stage
        decision pipeline then re-classifies the live signal under
        the new thresholds and may produce a different state and a
        post-reset streak. The contract tested here is that neither
        the original seed value survives -- the streak counter is no
        longer ``7`` and the classifier state was at least
        re-evaluated rather than carried forward verbatim.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        runtime = scheduler.ledgers.stage_states["A"]
        runtime.classifier.state = StageState.SATURATED
        runtime.classifier.streak = 7

        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=ps)

        assert _current_regime(scheduler) is Regime.SUPER_HALFIN_WHITT
        assert runtime.classifier.streak < 7
        assert runtime.classifier.resolved_thresholds is not None
        assert (
            runtime.classifier.resolved_thresholds.saturation_aggressiveness
            > SaturationAwareConfig().stage_defaults.saturation_aggressiveness
        )


class TestExecutedDeltaRecording:
    """Pin the post-commit executed-delta wiring contract.

    Background:
        ``run_per_stage_pipeline`` produces the per-stage scaling
        recommendation. Phase C / Phase D commit that recommendation
        with hard worker caps, the fractional shrink clamp, and
        allocation failures all able to throttle the executed delta
        below the recommendation magnitude.

    Contract (post-fix):
        ``record_executed_delta`` is invoked exactly once per stage
        that participated in the cycle's intent computation, with the
        post-Phase-D minus pre-Phase-C net worker count. The
        growth-mode state machine therefore observes the committed
        delta rather than the recommendation, so HOLD / ACQUIRING /
        TRACKING timers reflect what actually landed in the cluster.
    """

    def test_record_executed_delta_called_for_each_intent_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scheduler invokes ``record_executed_delta`` once per stage with an intent."""
        from unittest.mock import patch as _patch

        from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stage_decision_pipeline import (
            StageDecisionPipeline,
        )

        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))
        ps = _problem_state([("A", 1, 1), ("B", 1, 1)])

        captured: list[tuple[str, int]] = []

        def _capture(
            self: StageDecisionPipeline,
            *,
            stage_state: StageRuntimeState,
            delta_executed: int,
            config: object,
        ) -> None:
            captured.append((stage_state.stage_name, delta_executed))

        # ``record_executed_delta`` is now a method on
        # ``StageDecisionPipeline``; SaturationShrinkPhase
        # (``phases.phase_d``) constructs a pipeline and invokes
        # the method, so the patch targets the class itself.
        with _patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"A": 0, "B": 0},
        ):
            with _patch.object(
                StageDecisionPipeline,
                "record_executed_delta",
                autospec=True,
                side_effect=_capture,
            ):
                scheduler.autoscale(time=0.0, problem_state=ps)

        recorded_stages = sorted(name for name, _delta in captured)
        assert recorded_stages == ["A", "B"], (
            "record_executed_delta must fire once per stage that participated in the intent computation; "
            f"got {recorded_stages}"
        )
        for _name, delta in captured:
            assert delta == 0, "No Phase C/D mutation occurred; executed delta must be 0 for both stages"

    def test_record_executed_delta_reflects_phase_c_throttling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When Phase C executes fewer adds than the recommendation, the recorded delta matches the commit."""
        from unittest.mock import patch as _patch

        from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stage_decision_pipeline import (
            StageDecisionPipeline,
        )

        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state([("A", 1, 1)])

        captured: list[int] = []

        def _capture(
            self: StageDecisionPipeline,
            *,
            stage_state: StageRuntimeState,
            delta_executed: int,
            config: object,
        ) -> None:
            captured.append(delta_executed)

        # Recommendation is +3 but the first try_add_worker returns None
        # (cluster exhausted) so Phase C executes 0 adds. Recorded delta
        # must be 0, NOT the recommendation. ``record_executed_delta`` is
        # now a method on ``StageDecisionPipeline``; ``SaturationShrinkPhase``
        # constructs a pipeline and invokes the method, so the patch
        # targets the class itself.
        with _patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"A": 3},
        ):
            with _patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                return_value=None,
            ):
                with _patch.object(
                    StageDecisionPipeline,
                    "record_executed_delta",
                    autospec=True,
                    side_effect=_capture,
                ):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert captured == [0], (
            "Phase C executed zero adds despite a +3 recommendation; the recorded delta must mirror "
            f"the commit, not the recommendation; got {captured}"
        )


class TestInvariantPropagation:
    """Non-AllocationError raises propagate; the narrow defense-in-depth scope does not bury them."""

    def test_scheduler_invariant_error_propagates_out_of_autoscale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A planner bug surfacing as ``RuntimeError`` propagates instead of being absorbed."""
        from unittest.mock import patch as _patch

        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state([("A", 1, 1)])

        err = RuntimeError("synthetic scheduler invariant violation")
        with _patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"A": 1},
        ):
            with _patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=err,
            ):
                with pytest.raises(RuntimeError, match="synthetic scheduler invariant violation"):
                    scheduler.autoscale(time=0.0, problem_state=ps)
