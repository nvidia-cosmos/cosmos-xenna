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

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.regime import Regime
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


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


class TestSetup:
    """``setup()`` builds a per-stage runtime state map keyed by stage name."""

    def test_state_map_keyed_by_stage_name(self) -> None:
        """One ``_StageRuntimeState`` per stage, keyed by the stage's name."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B", "C"]))
        assert set(scheduler._stage_states) == {"A", "B", "C"}
        for name in ("A", "B", "C"):
            assert isinstance(scheduler._stage_states[name], _StageRuntimeState)
            assert scheduler._stage_states[name].stage_name == name

    def test_stage_names_preserve_pipeline_order(self) -> None:
        """``_stage_names`` reflects DAG order so deterministic iteration matches Solution order."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["upstream", "middle", "downstream"]))
        assert scheduler._stage_names == ["upstream", "middle", "downstream"]

    def test_runtime_state_starts_at_default_values(self) -> None:
        """Newly constructed runtime state matches ``_StageRuntimeState`` defaults."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        state = scheduler._stage_states["A"]
        assert state.slots_empty_ratio_ewma is None
        assert state.last_valid_slots_empty_ratio_ewma is None
        assert state.classifier_streak == 0
        assert state.growth_streak == 0
        assert state.prev_workers == 0

    def test_handles_single_stage_pipeline(self) -> None:
        """Smallest non-trivial pipeline -- one stage."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["only"]))
        assert scheduler._stage_names == ["only"]
        assert list(scheduler._stage_states) == ["only"]

    def test_setup_can_be_called_again_to_rebuild_state(self) -> None:
        """A second ``setup()`` replaces the prior state; useful for test isolation and reset paths."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        scheduler.setup(_problem_with_stages(["X", "Y"]))
        assert scheduler._stage_names == ["X", "Y"]
        assert "A" not in scheduler._stage_states
        assert set(scheduler._stage_states) == {"X", "Y"}

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
        assert scheduler._problem is problem

    def test_handles_many_stages_pipeline(self) -> None:
        """20-stage pipeline: state map size, order, and uniqueness all verified at scale."""
        stage_names = [f"Stage-{i:02d}" for i in range(20)]
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(stage_names))
        assert scheduler._stage_names == stage_names
        assert len(scheduler._stage_states) == 20
        assert all(scheduler._stage_states[name].stage_name == name for name in stage_names)


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
        original_dict = scheduler._stage_states
        original_state_a = scheduler._stage_states["A"]

        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=_problem_state([("A", 1, 2)]))

        assert scheduler._stage_states is original_dict
        assert scheduler._stage_states["A"] is original_state_a


class TestUpdateWithMeasurementsIsNoOp:
    """``update_with_measurements`` accepts and discards measurements -- signal-driven scheduler ignores them."""

    def test_does_not_raise_or_mutate_state(self) -> None:
        """The method is a documented no-op so call sites in streaming.py can be branchless."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        snapshot_before = dict(scheduler._stage_states)
        empty_measurements = data_structures.Measurements(time=0.0, stage_measurements=[])
        scheduler.update_with_measurements(time=0.0, measurements=empty_measurements)
        # Same dict, same values -- nothing was mutated.
        assert scheduler._stage_states == snapshot_before


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
        counts to pin that the constructor accepts the snapshot.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A", "B"]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 2), ("B", 1, 1)]),
        )
        # Sanity: 5 1-CPU workers fit in the 8-CPU cluster; the
        # constructor would have raised if the seed allocations did
        # not match the cluster shape.
        assert [s.slots_per_worker for s in solution.stages] == [2, 1]

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
        assert scheduler._stage_states["A"].resolved_thresholds is None

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

        small_resolved = scheduler._stage_states["small_c"].resolved_thresholds
        large_resolved = scheduler._stage_states["large_c"].resolved_thresholds
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
        first_resolved = scheduler._stage_states["A"].resolved_thresholds
        assert first_resolved is not None
        assert first_resolved.slots_per_actor == 8
        # Second cycle: same stage, different slots_per_worker -- resolution is sticky.
        scheduler.autoscale(time=10.0, problem_state=_problem_state([("A", 1, 64)]))
        second_resolved = scheduler._stage_states["A"].resolved_thresholds
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
        assert scheduler._regime_state.current_regime is Regime.SUB_HALFIN_WHITT
        # Stage thresholds resolved with base aggressiveness 0.30.
        resolved = scheduler._stage_states["A"].resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.30)

    def test_sustained_busy_signal_transitions_to_super_hw_after_streak(self) -> None:
        """Three consecutive busy cycles commit the regime transition."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        # 4 workers, each carrying 8 slots, 31 used / 1 empty -> idle ~ 0.031.
        # threshold = 1/sqrt(4) = 0.50 -> 0.031 < 0.50 -> super-HW raw verdict.
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=ps)
        assert scheduler._regime_state.current_regime is Regime.SUPER_HALFIN_WHITT

    def test_super_hw_transition_relifts_aggressiveness(self) -> None:
        """A transition into super-HW re-resolves thresholds with base + lift."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=ps)
        resolved = scheduler._stage_states["A"].resolved_thresholds
        assert resolved is not None
        # Base 0.30 + lift 0.15 = 0.45.
        assert resolved.saturation_aggressiveness == pytest.approx(0.45)

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
        assert scheduler._regime_state.current_regime is Regime.SUB_HALFIN_WHITT

    def test_disabled_flag_pins_aggressiveness_at_base(self) -> None:
        """``enable_regime_aware_aggressiveness=False`` skips regime tracking entirely."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig(enable_regime_aware_aggressiveness=False))
        scheduler.setup(_problem_with_stages(["A"]))
        ps = _problem_state_with_slot_signals([("A", 4, 8, 31, 1)])
        for _ in range(10):
            scheduler.autoscale(time=0.0, problem_state=ps)
        # Regime state stays at default (sub-HW); thresholds resolved at base.
        assert scheduler._regime_state.current_regime is Regime.SUB_HALFIN_WHITT
        resolved = scheduler._stage_states["A"].resolved_thresholds
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
        assert scheduler._regime_state.current_regime is Regime.SUPER_HALFIN_WHITT
        # threshold = 1/sqrt(4) = 0.50; exit band = 0.75. Use idle = 0.84.
        idle = _problem_state_with_slot_signals([("A", 4, 8, 5, 27)])
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=idle)
        assert scheduler._regime_state.current_regime is Regime.SUB_HALFIN_WHITT
        resolved = scheduler._stage_states["A"].resolved_thresholds
        assert resolved is not None
        assert resolved.saturation_aggressiveness == pytest.approx(0.30)
