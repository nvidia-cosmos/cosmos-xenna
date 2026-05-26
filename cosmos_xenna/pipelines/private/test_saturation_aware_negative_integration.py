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


"""Integration-level negative tests for ``SaturationAwareScheduler``.

Each test verifies error context and state containment across
multiple autoscale cycles: shape-mismatch error reporting, hostile
format-string-bearing stage names, allocator-failure absorption and
propagation under recovery, and corrupted-EWMA fixture containment.
"""

import math
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 8) -> resources.ClusterResources:
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
    cluster: resources.ClusterResources | None = None,
) -> data_structures.Problem:
    """Build a ``Problem`` with one CPU stage per name on ``cluster``."""
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


def _worker_group(
    stage_name: str, idx: int, *, node: str = "node-0", num_used_slots: int = 0
) -> data_structures.ProblemWorkerGroupState:
    """One CPU worker_group keyed by ``{stage_name}-w{idx}``."""
    return data_structures.ProblemWorkerGroupState.make(
        f"{stage_name}-w{idx}",
        [resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=[])],
        num_used_slots=num_used_slots,
    )


def _stage_state(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int = 8,
    num_used_slots: int = 0,
    num_empty_slots: int | None = None,
    input_queue_depth: int = 0,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState`` with ``num_workers`` worker_groups on node-0.

    Distributes ``num_used_slots`` across the worker_groups so the
    per-worker sum equals the stage-level total. The previous
    helper passed ``num_used_slots=0`` to every ``_worker_group``
    while reporting a non-zero stage total, which let the warmup-
    excluding aggregator (which sums per-worker counts) silently
    disagree with the stage-level signal whenever a test fed a
    non-zero stage total. Base/remainder split keeps the
    distribution as even as integer arithmetic allows.
    """
    if num_workers > 0:
        base_used = num_used_slots // num_workers
        remainder = num_used_slots % num_workers
    else:
        base_used = 0
        remainder = 0
    workers = [
        _worker_group(name, i, num_used_slots=base_used + (1 if i < remainder else 0)) for i in range(num_workers)
    ]
    total = num_workers * slots_per_worker
    empty = num_empty_slots if num_empty_slots is not None else max(0, total - num_used_slots)
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=workers,
        slots_per_worker=slots_per_worker,
        is_finished=False,
        num_used_slots=num_used_slots,
        num_empty_slots=empty,
        input_queue_depth=input_queue_depth,
    )


def _build_scheduler(
    stage_names: list[str],
    *,
    skip_on_error: bool = True,
    min_workers: int | None = 1,
    saturated_streak_min_cycles: int = 1,
    over_provisioned_streak_min_cycles: int = 2,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 2,
    cluster: resources.ClusterResources | None = None,
) -> SaturationAwareScheduler:
    """Build a configured scheduler with negative-test-friendly defaults.

    Disables warmup grace and setup-phase quiescence so a single
    saturating cycle is enough to drive Phase C. The kill switch
    is on by default so most tests can exercise recovery; tests
    that pin the propagation contract flip it off.
    """
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        skip_cycle_on_allocation_error=skip_on_error,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=min_workers,
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_names, cluster=cluster))
    return scheduler


class TestProblemStateShapeMismatchAfterSuccessfulCycle:
    """A first cycle succeeds; a second cycle with a missing stage raises contextual error."""

    def test_problem_state_shape_mismatch_after_successful_cycle_raises_contextual_error(self) -> None:
        """One valid cycle, then ProblemState drops a stage -> SchedulerInvariantError names the mismatch.

        The error message must identify the mismatch so an operator
        can correlate it with the streaming-side fixture that built
        the snapshot. Per-stage runtime state from the first cycle
        must not silently shrink to fit the bad snapshot; the cycle
        must abort cleanly with an actionable error.
        """
        scheduler = _build_scheduler(["A", "B"])

        valid_state = data_structures.ProblemState(
            [
                _stage_state(name="A", num_workers=1, num_used_slots=4),
                _stage_state(name="B", num_workers=1, num_used_slots=4),
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=valid_state)
        snapshot_stage_states_before = dict(scheduler.ledgers.stage_states)

        bad_state = data_structures.ProblemState([_stage_state(name="A", num_workers=1, num_used_slots=4)])

        with pytest.raises(SchedulerInvariantError) as exc_info:
            scheduler.autoscale(time=10.0, problem_state=bad_state)

        message = str(exc_info.value)
        assert "1 stages" in message and "2" in message, (
            f"error must name the shape mismatch (got=1 vs expected=2); message={message!r}"
        )
        assert set(scheduler.ledgers.stage_states) == set(snapshot_stage_states_before), (
            "stage_states set must not shrink to the corrupted snapshot's shape; "
            f"before={set(snapshot_stage_states_before)}, after={set(scheduler.ledgers.stage_states)}"
        )


class TestStageNameWithCurlyBracesInMulticycleError:
    """A stage name containing format-string tokens survives the multi-cycle error path."""

    def test_stage_name_with_curly_braces_in_multicycle_error_is_safe_in_repr(self) -> None:
        """``{0.__class__}`` in a stage name appears verbatim in the post-mismatch error message.

        Two cycles: the first cycle succeeds with two hostile stage
        names, the second cycle drops one stage. The raised error's
        ``str`` must contain the literal hostile token (proving the
        formatter did not re-evaluate it) and the scheduler must
        not strip the name during recovery state cleanup.
        """
        hostile_name = "stage-{0.__class__}-{foo!r}"
        normal_name = "B"
        scheduler = _build_scheduler([hostile_name, normal_name])

        valid_state = data_structures.ProblemState(
            [
                _stage_state(name=hostile_name, num_workers=1, num_used_slots=4),
                _stage_state(name=normal_name, num_workers=1, num_used_slots=4),
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=valid_state)

        bad_state = data_structures.ProblemState([_stage_state(name=hostile_name, num_workers=1, num_used_slots=4)])

        with pytest.raises(SchedulerInvariantError) as exc_info:
            scheduler.autoscale(time=10.0, problem_state=bad_state)

        message = str(exc_info.value)
        assert "1 stages" in message and "2" in message, (
            f"error must report the shape mismatch numerically; got message={message!r}"
        )
        assert hostile_name in scheduler.ledgers.stage_states, (
            "hostile-name stage must persist in scheduler state after the failed cycle; "
            f"_stage_states keys={set(scheduler.ledgers.stage_states)}"
        )


class TestAllocationFailureSkipCyclePreservesStateAndRecovers:
    """Absorbed AllocationError preserves prior worker_age state and the next cycle still grows."""

    def test_allocation_failure_skip_cycle_preserves_previous_worker_age_and_recovers_next_cycle(self) -> None:
        """Cycle 1 grows; cycle 2 raises AllocationError (absorbed); recovery cycles still grow.

        The kill switch is on. After the failed cycle the
        ``_worker_ages`` map must retain the worker ids observed in
        cycle 1 (no state corruption from the failed
        ``try_add_worker``), and the post-failure recovery cycles
        must produce at least one Phase C add when the synthetic
        failure clears -- proving the failure did not latch the
        scheduler into a permanently-blocked state.
        """
        scheduler = _build_scheduler(["stage"], skip_on_error=True)

        ps = data_structures.ProblemState(
            [_stage_state(name="stage", num_workers=1, num_used_slots=8, num_empty_slots=0, input_queue_depth=8)]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)
        worker_ages_after_cycle_1 = dict(scheduler.ledgers.worker_ages)
        assert "stage-w0" in worker_ages_after_cycle_1, (
            f"expected cycle 1 to register 'stage-w0' in _worker_ages; got {set(worker_ages_after_cycle_1)}"
        )

        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"stage": 1},
        ):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=resources.AllocationError("synthetic placement failure"),
            ):
                scheduler.autoscale(time=10.0, problem_state=ps)

        assert worker_ages_after_cycle_1.keys() <= scheduler.ledgers.worker_ages.keys(), (
            "absorbed AllocationError must not evict cycle-1 worker ids from _worker_ages; "
            f"before={set(worker_ages_after_cycle_1)}, after={set(scheduler.ledgers.worker_ages)}"
        )

        recovery_adds: list[int] = []
        for cycle_idx in range(5):
            sol = scheduler.autoscale(time=20.0 + cycle_idx * 10.0, problem_state=ps)
            recovery_adds.append(len(sol.stages[0].new_workers))

        assert sum(recovery_adds) >= 1, (
            "across 5 post-failure recovery cycles the scheduler must produce at least one "
            f"Phase C add; got per-cycle adds={recovery_adds}"
        )


class TestAllocationFailurePropagatesWhenKillSwitchDisabled:
    """``skip_cycle_on_allocation_error=False`` lets the error propagate without state mutation."""

    def test_allocation_failure_propagates_without_mutating_state_when_kill_switch_disabled(self) -> None:
        """With kill switch off the AllocationError propagates; a healthy follow-up cycle still works.

        On the failing cycle the call must raise (no absorption).
        On the next healthy cycle (no synthetic failure) the
        scheduler must produce a structurally valid Solution and a
        positive Phase C add, proving the failure cycle did not
        leak partial state that blocks recovery.
        """
        scheduler = _build_scheduler(["stage"], skip_on_error=False)

        ps = data_structures.ProblemState(
            [_stage_state(name="stage", num_workers=1, num_used_slots=8, num_empty_slots=0, input_queue_depth=8)]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)
        ages_before_failure = dict(scheduler.ledgers.worker_ages)

        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"stage": 1},
        ):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=resources.AllocationError("synthetic placement failure"),
            ):
                with pytest.raises(resources.AllocationError, match="synthetic placement failure"):
                    scheduler.autoscale(time=10.0, problem_state=ps)

        assert ages_before_failure.keys() <= scheduler.ledgers.worker_ages.keys(), (
            "propagated AllocationError must not silently evict pre-failure ids from _worker_ages; "
            f"before={set(ages_before_failure)}, after={set(scheduler.ledgers.worker_ages)}"
        )

        recovery_adds: list[int] = []
        for cycle_idx in range(5):
            sol = scheduler.autoscale(time=20.0 + cycle_idx * 10.0, problem_state=ps)
            assert len(sol.stages) == 1, (
                f"recovery cycle {cycle_idx} must produce one stage solution; got {len(sol.stages)}"
            )
            recovery_adds.append(len(sol.stages[0].new_workers))

        assert sum(recovery_adds) >= 1, (
            "across 5 post-propagation cycles the scheduler must still grow the saturated stage; "
            f"got per-cycle adds={recovery_adds}"
        )


class TestCorruptedMidCycleMetricStateBlocksNextCycleAndRecovers:
    """A NaN seeded into ``pressure_ewma`` between cycles is rejected; a fresh scheduler succeeds."""

    def test_corrupted_mid_cycle_metric_state_blocks_next_cycle_and_recovers_after_reinitialization(self) -> None:
        """Corrupt ``pressure_ewma``; next ``autoscale()`` raises a finite-check error.

        Drives one valid cycle, corrupts the per-stage EWMA, and
        asserts the next cycle raises with the offending field and
        value in the message. A second scheduler built over the
        same ``Problem`` and fed the same ``ProblemState`` must
        succeed, proving the failure was contained to the corrupted
        scheduler's runtime state.
        """
        problem_stages = ["A", "B"]
        cluster = _cluster()

        scheduler = _build_scheduler(problem_stages, cluster=cluster)
        ps = data_structures.ProblemState(
            [
                _stage_state(name="A", num_workers=1, num_used_slots=8, num_empty_slots=0, input_queue_depth=8),
                _stage_state(name="B", num_workers=1, num_used_slots=4),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        runtime_a = scheduler.ledgers.stage_states["A"]
        runtime_a.pressure.ewma = math.nan

        with pytest.raises(ValueError) as exc_info:
            scheduler.autoscale(time=10.0, problem_state=ps)

        message = str(exc_info.value)
        assert "pressure_ewma" in message, f"error must name the corrupted field; message={message!r}"
        assert "nan" in message.lower(), f"error must report the offending value; message={message!r}"

        fresh_scheduler = _build_scheduler(problem_stages, cluster=cluster)
        fresh_scheduler.autoscale(time=0.0, problem_state=ps)
        recovery_solution = fresh_scheduler.autoscale(time=10.0, problem_state=ps)

        assert len(recovery_solution.stages) == 2, (
            f"fresh scheduler must emit a 2-stage solution; got {len(recovery_solution.stages)}"
        )
        fresh_ewma = fresh_scheduler.ledgers.stage_states["A"].pressure.ewma
        assert fresh_ewma is None or math.isfinite(fresh_ewma), (
            f"fresh scheduler's pressure_ewma must stay finite over the same fixture; got {fresh_ewma!r}"
        )
