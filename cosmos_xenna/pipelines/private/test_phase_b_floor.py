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

"""Behaviour tests for the Phase B minimum-worker floor.

Pin the contract:

  1. Every non-manual, non-finished stage reaches at least
     ``target_min = max(min_workers if set else 1,
                        min_workers_per_node * num_nodes if set else 0)``
     by the end of the autoscale cycle.
  2. Manual stages (``requested_num_workers is not None``) are skipped
     by the floor; their worker count is owned by the manual phases.
  3. Finished stages (``is_finished``) are skipped.
  4. When the cluster cannot satisfy the floor directly, the scheduler
     tries a one-worker donor swap before raising ``RuntimeError`` with
     operator-actionable context.
"""

from typing import cast
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.donor.types import DonorAcquireResult, DonorPlan, DonorWorker
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.floor_phase import FloorPhase
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _make_cycle(
    ctx: data_structures.AutoscalePlanContext,
    problem_state: data_structures.ProblemState,
) -> AutoscaleCycle:
    """Build a minimal ``AutoscaleCycle`` for direct phase-method tests."""
    return AutoscaleCycle(
        ctx=ctx,
        problem_state=problem_state,
        time=0.0,
        cycle_counter=0,
        pipeline_name="",
    )


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 16) -> resources.ClusterResources:
    """Multi-node cluster sized to fit every test fixture."""
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
    *,
    num_nodes: int = 1,
    total_cpus_per_node: int = 16,
) -> data_structures.Problem:
    """Build a ``Problem`` whose stages carry the given ``requested_num_workers``."""
    cluster = _cluster(num_nodes=num_nodes, total_cpus_per_node=total_cpus_per_node)
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


def _make_workers(
    stage_name: str,
    count: int,
    *,
    num_nodes: int = 1,
) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers spread round-robin across nodes."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node=f"node-{i % num_nodes}", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
    *,
    num_nodes: int = 1,
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(name, num_workers, slots_per_worker, is_finished)``."""
    states = [
        data_structures.ProblemStageState(
            stage_name=name,
            workers=_make_workers(name, num_workers, num_nodes=num_nodes),
            slots_per_worker=slots,
            is_finished=finished,
        )
        for name, num_workers, slots, finished in stage_specs
    ]
    return data_structures.ProblemState(states)


class _RaisingAddContext:
    """Fake planner that surfaces a hard floor-enforcement failure.

    Exposes the minimal surface ``FloorPhase.run`` reaches BEFORE
    the donor path: ``worker_ids_by_stage`` (read to compute
    ``current``), ``try_add_worker`` (the call site under test),
    and ``cluster_snapshot`` (consumed by the shared
    :mod:`allocation_failures` defense layer when it absorbs an
    :class:`resources.AllocationError`). The snapshot is fixed
    cluster geometry; the absorb-path tests only assert on the
    flag and call-count contract, not on the snapshot content.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[int] = []
        self._exc = exc or RuntimeError("planner context is drained")

    def worker_ids_by_stage(self) -> list[list[str]]:
        """Return a single empty stage so the floor loop reads ``current=0``."""
        return [[]]

    def try_add_worker(self, stage_index: int) -> data_structures.ProblemWorkerGroupState | None:
        """Raise the same exception types the planner can raise."""
        self.calls.append(stage_index)
        raise self._exc

    def cluster_snapshot(self) -> resources.ClusterResources:
        """Return a stable cluster snapshot for the absorb path's diagnostic log."""
        return _cluster()


class _MalformedAcquireResult:
    """Test double simulating a coordinator that violates the DonorAcquireResult invariant.

    The real :class:`DonorAcquireResult.committed` is a ``@property``
    defined as ``self.plan is not None`` (donor/types.py:204), so
    an attrs-constructed instance can never report ``committed=True``
    paired with ``plan=None``. This double duck-types the three
    attributes the FloorPhase reads after a successful commit
    (``committed`` / ``probe_failed_at_commit`` / ``plan``) so the
    phase's defensive coordinator-invariant raise is reachable
    from a test without exercising a real coordinator defect.
    """

    committed = True
    plan = None
    probe_failed_at_commit = False


class TestPhaseBFloor:
    """End-to-end Phase B floor enforcement via ``SaturationAwareScheduler.autoscale``."""

    def test_implicit_one_worker_floor_on_empty_pipeline(self) -> None:
        """A non-manual stage with zero workers and no explicit floor receives one worker."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 1

    def test_min_workers_floor_grows_to_configured_count(self) -> None:
        """``min_workers=4`` brings a non-manual stage from 0 to 4 workers."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=4))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 4

    def test_min_workers_per_node_scales_floor_with_cluster_size(self) -> None:
        """``min_workers_per_node=1`` on a 4-node cluster yields a floor of 4 workers."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers_per_node=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], num_nodes=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)], num_nodes=4),
        )
        assert len(solution.stages[0].new_workers) == 4

    def test_combined_min_workers_takes_max_of_two_knobs(self) -> None:
        """``min_workers=2`` combined with ``min_workers_per_node=1`` on 3 nodes -> floor=3."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=2, min_workers_per_node=1),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], num_nodes=3))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)], num_nodes=3),
        )
        assert len(solution.stages[0].new_workers) == 3

    def test_no_op_when_current_already_at_floor(self) -> None:
        """A stage already at its floor is not grown further."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=2))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 2, 1, False)]),
        )
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_no_shrink_when_current_above_floor(self) -> None:
        """A stage already above the floor is not shrunk; the floor enforces the lower bound only.

        With a 3-worker stage and an implicit floor of 1, the floor
        enforcement must leave the worker count untouched. Scale-down
        is the responsibility of a different decision phase that does
        not run here.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False)]),
        )
        assert solution.stages[0].new_workers == []
        assert solution.stages[0].deleted_workers == []

    def test_manual_stages_are_skipped_by_phase_b(self) -> None:
        """A manual stage's worker count is owned by Phase A, not by the floor.

        With ``requested_num_workers=2`` and ``min_workers=4``, Phase A grows the
        stage to exactly 2 (the operator request); Phase B does NOT push it
        further to 4 because manual stages are excluded from the floor.
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=4))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", 2)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 2

    def test_finished_stages_are_skipped(self) -> None:
        """A finished non-manual stage is not grown by Phase B."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=3))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, True)]),
        )
        assert solution.stages[0].new_workers == []

    def test_per_stage_override_takes_precedence_over_defaults(self) -> None:
        """``per_stage_overrides`` for stage B applies a floor only to B."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"B": SaturationAwareStageConfig(min_workers=3)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False), ("B", 0, 1, False)]),
        )
        # Defaults floor stage A at 1; override floors stage B at 3.
        assert len(solution.stages[0].new_workers) == 1
        assert len(solution.stages[1].new_workers) == 3

    def test_capacity_exhausted_raises_runtime_error(self) -> None:
        """A min_workers floor exceeding cluster capacity raises with operator context.

        Cycle 1 makes partial progress (adds 4 of 10 workers) and is
        graced under the per-cycle progress contract; cycle 2 starts
        with the cluster already full, makes zero progress, and
        raises with the operator-actionable target_min message. Test
        drives both cycles so the assertion targets the cycle that
        legitimately stuck with no further options.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=10),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=4))
        scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)]),
        )
        expected = (
            r"target_min=10 \(achieved=4; from min_workers=10, "
            r"min_workers_per_node=None, num_nodes=1\)\. Cluster placement exhausted"
        )
        with pytest.raises(RuntimeError, match=expected):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 4, 1, False)]),
            )

    def test_capacity_exhausted_error_identifies_per_node_floor_source(self) -> None:
        """A per-node floor failure names the per-node knob and cluster size.

        Cycle 1 reaches 2 of 4 floor workers (one per node) and is
        graced; cycle 2 starts with the cluster full, makes no further
        progress, and raises naming the per-node knob.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers_per_node=2),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], num_nodes=2, total_cpus_per_node=1))
        scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 0, 1, False)], num_nodes=2),
        )

        expected = (
            r"target_min=4 \(achieved=2; from min_workers=None, "
            r"min_workers_per_node=2, num_nodes=2\)\. Cluster placement exhausted"
        )
        with pytest.raises(RuntimeError, match=expected):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 2, 1, False)], num_nodes=2),
            )

    def test_floor_uses_donor_when_cluster_is_full(self) -> None:
        """A non-manual stage above its floor can donate to bootstrap another stage."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"receiver": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("donor", None), ("receiver", None)], total_cpus_per_node=4))

        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("donor", 3, 1, False), ("receiver", 1, 1, False)]),
        )

        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1

    def test_floor_called_before_setup_raises_runtime_error(self) -> None:
        """Reading ``scheduler.runner`` before ``setup`` fails with a clear scheduler error.

        ``FloorPhase().run(cycle, services)`` requires a ``FloorServices``
        value object which is constructed inside ``scheduler.setup()`` and
        owned by the runner. The "called before setup" contract therefore
        manifests as a guard on ``scheduler.runner`` rather than as a
        message-shaped exception inside the phase body.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        with pytest.raises(RuntimeError, match="runner read before setup"):
            _ = scheduler.runner

    def test_planner_runtime_error_propagates(self) -> None:
        """Hard planner failures are not rewritten as floor-capacity exhaustion."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        ctx = _RaisingAddContext()

        with pytest.raises(RuntimeError, match="planner context is drained"):
            FloorPhase().run(
                _make_cycle(
                    cast(data_structures.AutoscalePlanContext, ctx),
                    _problem_state([("A", 0, 1, False)]),
                ),
                scheduler.runner.floor_services,
            )

        assert ctx.calls == [0]

    def test_planner_index_error_propagates(self) -> None:
        """Planner index errors are not rewritten as floor-capacity exhaustion."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        ctx = _RaisingAddContext(IndexError("stage_index 0 out of range"))

        with pytest.raises(IndexError, match="stage_index 0 out of range"):
            FloorPhase().run(
                _make_cycle(
                    cast(data_structures.AutoscalePlanContext, ctx),
                    _problem_state([("A", 0, 1, False)]),
                ),
                scheduler.runner.floor_services,
            )

        assert ctx.calls == [0]

    def test_zero_stage_pipeline_is_no_op(self) -> None:
        """An empty pipeline has no floor entries to enforce."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([]),
        )

        assert solution.stages == []

    def test_manual_shrink_frees_capacity_for_floor_in_same_cycle(self) -> None:
        """Phase A delete frees CPUs that the floor consumes within one autoscale cycle.

        Cluster has room for 4 1-CPU workers. Stage A is manual at
        ``current=3, requested=1`` (frees 2 CPUs). Stage B is non-manual
        with ``min_workers=3`` and ``current=1`` (needs +2 CPUs). The
        delete-before-floor ordering inside ``autoscale`` makes the freed
        CPUs available to the floor loop without round-tripping.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"B": SaturationAwareStageConfig(min_workers=3)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", 1), ("B", None)], total_cpus_per_node=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 3, 1, False), ("B", 1, 1, False)]),
        )
        assert len(solution.stages[0].deleted_workers) == 2
        assert len(solution.stages[1].new_workers) == 2

    def test_manual_grow_and_floor_run_in_one_cycle(self) -> None:
        """Manual grow on A and floor enforcement on B both fire in the same cycle."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=3))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("manual", 2), ("auto", None)]))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("manual", 0, 1, False), ("auto", 0, 1, False)]),
        )
        assert len(solution.stages[0].new_workers) == 2
        assert len(solution.stages[1].new_workers) == 3


class TestPhaseBFloorAllocationErrorDefense:
    """Pin the Phase B side of the shared allocation-failure defense layer.

    The wrapper in
    :func:`cosmos_xenna.pipelines.private.scheduling_py.cluster.allocation_failures.try_add_worker_with_defense`
    catches only :class:`resources.AllocationError`, leaving every
    other exception type to propagate (validated by
    ``test_planner_runtime_error_propagates`` and
    ``test_planner_index_error_propagates`` in the suite above).
    These tests pin the four Phase B branches:

    1. Pre-donor add raising ``AllocationError`` is absorbed.
    2. Pre-donor add raising ``AllocationError`` with the kill
       switch off re-raises after the ERROR log + counter increment.
    3. Post-donor-commit retry raising ``AllocationError`` is
       absorbed (donor removals already committed atomically; the
       cycle accepts the asymmetric outcome and the next cycle
       re-balances).
    4. Post-donor-commit retry returning ``None`` (probe-to-commit
       planner divergence, not a placement exhaustion) still raises
       :class:`SchedulerInvariantError` regardless of the kill
       switch.
    """

    def test_pre_donor_allocation_error_is_absorbed_with_default_skip_switch(self) -> None:
        """A pre-donor ``AllocationError`` is absorbed and only the failure flag flips.

        The wrapper logs the failure, increments the per-stage
        counter, sets
        :attr:`FloorServices.floor_allocation.aborted_cycle`,
        and Phase B returns without re-raising so the rest of the
        autoscale cycle can run.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))
        ctx = _RaisingAddContext(resources.AllocationError("synthetic placement failure"))

        FloorPhase().run(
            _make_cycle(
                cast(data_structures.AutoscalePlanContext, ctx),
                _problem_state([("A", 0, 1, False)]),
            ),
            scheduler.runner.floor_services,
        )

        assert ctx.calls == [0]
        assert scheduler.runner.floor_services.donor_executor.allocation_gate.aborted_cycle is True

    def test_pre_donor_allocation_error_propagates_when_skip_switch_off(self) -> None:
        """``skip_cycle_on_allocation_error=False`` re-raises the pre-donor ``AllocationError``.

        The wrapper still emits the ERROR log + counter increment
        before re-raising, but it does NOT set the cycle-skip flag
        (which is reserved for the absorb path); the autoscaler
        thread sees the original exception.
        """
        cfg = SaturationAwareConfig(skip_cycle_on_allocation_error=False)
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)]))
        ctx = _RaisingAddContext(resources.AllocationError("synthetic placement failure"))

        with pytest.raises(resources.AllocationError, match="synthetic placement failure"):
            FloorPhase().run(
                _make_cycle(
                    cast(data_structures.AutoscalePlanContext, ctx),
                    _problem_state([("A", 0, 1, False)]),
                ),
                scheduler.runner.floor_services,
            )

        assert ctx.calls == [0]
        assert scheduler.runner.floor_services.donor_executor.allocation_gate.aborted_cycle is False

    def test_post_donor_commit_allocation_error_is_absorbed(self) -> None:
        """A post-commit retry raising ``AllocationError`` is absorbed; donor removals persist.

        The donor coordinator already committed the worker removals
        atomically. The post-commit retry is wrapped through the
        same defense layer so an ``AllocationError`` (cluster shape
        no longer fits) sets the cycle-skip flag and Phase B
        returns instead of raising; the asymmetric outcome (donor
        stage -1, receiver +0) is accepted and the next cycle
        re-evaluates against the freed capacity.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"receiver": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg, pipeline_name="test-pipe")
        scheduler.setup(_problem([("donor", None), ("receiver", None)], total_cpus_per_node=4))
        ps = _problem_state([("donor", 3, 1, False), ("receiver", 1, 1, False)])
        err = resources.AllocationError("synthetic post-commit placement failure")
        donor_plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="donor-w0", age=0),),
            receiver_stage_index=1,
        )

        with patch(
            "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
            side_effect=[None, err, None],
        ):
            with patch.object(
                DonorCoordinator,
                "acquire",
                return_value=DonorAcquireResult(
                    plan=donor_plan,
                    attempted_plan=donor_plan,
                    reject_reason=None,
                    placement_reject_reason="",
                    gate_result=None,
                ),
            ):
                with patch(
                    "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
                    return_value={"donor": 0, "receiver": 0},
                ):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler.runner.floor_services.donor_executor.allocation_gate.aborted_cycle is True

    def test_post_donor_commit_none_return_raises_invariant_error(self) -> None:
        """A post-commit retry returning ``None`` (probe-to-commit divergence) raises invariant error.

        ``None`` from ``try_add_worker`` after a probe-approved
        donor plan committed means the planner reported the
        receiver placeable during the probe and unplaceable during
        the commit using the SAME FGD allocator state. That is a
        scheduler defect, not a benign cluster-full event, so the
        defense layer's ``AllocationError`` absorption does NOT
        apply: ``SchedulerInvariantError`` is raised regardless of
        the kill switch.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"receiver": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg, pipeline_name="test-pipe")
        scheduler.setup(_problem([("donor", None), ("receiver", None)], total_cpus_per_node=4))
        ps = _problem_state([("donor", 3, 1, False), ("receiver", 1, 1, False)])
        donor_plan = DonorPlan(
            removals=(DonorWorker(stage_index=0, worker_id="donor-w0", age=0),),
            receiver_stage_index=1,
        )

        with patch(
            "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
            side_effect=[None, None],
        ):
            with patch.object(
                DonorCoordinator,
                "acquire",
                return_value=DonorAcquireResult(
                    plan=donor_plan,
                    attempted_plan=donor_plan,
                    reject_reason=None,
                    placement_reject_reason="",
                    gate_result=None,
                ),
            ):
                with pytest.raises(SchedulerInvariantError, match="planner snapshot diverged"):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler.runner.floor_services.donor_executor.allocation_gate.aborted_cycle is False, (
            "SchedulerInvariantError must not engage the AllocationError absorb path"
        )

    def test_post_donor_commit_malformed_acquire_result_raises_invariant_error(self) -> None:
        """A coordinator returning ``committed=True`` with ``plan=None`` raises the invariant error.

        :attr:`DonorAcquireResult.committed` is defined as ``self.plan
        is not None`` (donor/types.py:204), so an attrs-constructed
        instance cannot violate the pairing. A coordinator defect
        (or a hypothetical subclass that overrides ``committed``)
        returning a mismatched pair is a contract violation that
        must surface as :class:`SchedulerInvariantError` -
        operator-actionable and visible in the structured error
        log - instead of a context-free ``AttributeError`` from the
        downstream ``plan.removals`` dereference. The raise also
        survives ``python -O`` where an ``assert`` would be
        stripped, which was the historical guard at this site.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"receiver": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg, pipeline_name="test-pipe")
        scheduler.setup(_problem([("donor", None), ("receiver", None)], total_cpus_per_node=4))
        ps = _problem_state([("donor", 3, 1, False), ("receiver", 1, 1, False)])

        with patch(
            "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
            return_value=None,
        ):
            with patch.object(DonorCoordinator, "acquire", return_value=_MalformedAcquireResult()):
                with pytest.raises(
                    SchedulerInvariantError,
                    match=r"post-commit retry returned None for receiver 'receiver'",
                ):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler.runner.floor_services.donor_executor.allocation_gate.aborted_cycle is False, (
            "SchedulerInvariantError must not engage the AllocationError absorb path"
        )
