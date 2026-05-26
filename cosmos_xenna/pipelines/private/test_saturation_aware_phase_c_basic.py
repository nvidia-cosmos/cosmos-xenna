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


"""Tests for the saturation-aware Phase C grow entry point (``phases.phase_c.run``).

Phase C applies positive intent deltas as planner adds via
``ctx.try_add_worker``. The contract under test:

    * Positive intent grows the stage by exactly ``intent`` workers
      when the cluster has room.
    * Cluster exhaustion is non-fatal: a single WARNING per affected
      stage; the Solution carries the partial growth.
    * Negative or zero intent is a no-op (Phase D scale-down ships
      separately).
    * Finished stages are skipped.
    * The post-Phase-C invariant gate runs after the grow.

Most tests inject the intent dict directly by patching
``IntentPhase._compute_intent_deltas`` on
``cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase``
so each adversarial case can be exercised without rigging classifier
signals; one integration test exercises the end-to-end signal path.
"""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor.coordinator import DonorCoordinator
from cosmos_xenna.pipelines.private.scheduling_py.invariants.checks import PhaseBoundary
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
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
    """Build a setup-completed scheduler over the given stages.

    The Phase C tests focus on placement orchestration and intent
    consumption, both orthogonal to the warmup grace mechanisms.
    The two grace fields are zeroed so a signal injected on cycle
    one is absorbed by the EWMA immediately, matching the legacy
    pre-grace behaviour the test cases were written against. The
    warmup-grace contract has its own dedicated test module.
    """
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=1,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            min_data_points=1,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_specs))
    return scheduler


def _autoscale_with_intents(
    scheduler: SaturationAwareScheduler,
    state: data_structures.ProblemState,
    intents: dict[str, int],
) -> data_structures.Solution:
    """Run autoscale with ``intents`` injected as the patched ``intent_phase.compute`` output."""

    def _inject(_services: object, _cycle: object, **_kwargs: object) -> dict[str, int]:
        return dict(intents)

    with patch(
        "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
        side_effect=_inject,
    ):
        return scheduler.autoscale(time=0.0, problem_state=state)


class _MalformedAcquireResult:
    """Test double simulating a coordinator that violates the DonorAcquireResult invariant.

    The real :class:`DonorAcquireResult.committed` is a ``@property``
    defined as ``self.plan is not None`` (donor/types.py:204), so
    an attrs-constructed instance can never report ``committed=True``
    paired with ``plan=None``. This double duck-types the two
    attributes ``SaturationGrowPhase`` reads after a successful
    commit (``committed`` and ``plan``) so the phase's defensive
    coordinator-invariant raise is reachable from a test without
    exercising a real coordinator defect.
    """

    committed = True
    plan = None


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
        """NORMAL stages produce intent 0 and Phase C is a no-op."""
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
        """An ``intent`` of ``sys.maxsize`` terminates at cluster exhaustion (no infinite loop).

        The 8-CPU cluster minus the 1 pre-seeded 1-CPU worker leaves
        exactly 7 free CPU slots. An ``intent=sys.maxsize`` must add
        exactly 7 workers before ``ctx.try_add_worker`` returns the
        first ``None`` (cluster full); the loop must terminate at that
        point. Asserting equality (``== 7``) instead of an upper bound
        (``<= 7``) catches a regression that would silently early-exit
        with partial progress (e.g. 5 adds) and otherwise hide the
        cluster-exhaustion-only termination contract.
        """
        import sys

        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": sys.maxsize})

        added = len(solution.stages[0].new_workers)
        assert added == 7, (
            f"Phase C must add exactly 7 workers (8 CPUs total - 1 seeded) before "
            f"hitting cluster exhaustion; got {added} adds. A smaller value indicates "
            "a regression that early-exits before exhausting cluster capacity."
        )

    def test_warning_logged_on_cluster_exhaustion(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Operators see exactly one WARNING with stage name + intent + actual + deficit on partial grows."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        _autoscale_with_intents(scheduler, state, {"A": 100})

        warnings = [record.getMessage() for record in loguru_caplog.records if record.levelname == "WARNING"]
        deficit_count = sum(1 for msg in warnings if "'A'" in msg and "deficit" in msg)
        assert deficit_count == 1, f"expected exactly one stage-named deficit warning; got: {warnings}"


class TestMultiStageIndependentGrowth:
    """Without DAG priority each stage with positive intent grows independently.

    Captures the contract that, with DAG-priority multi-target growth
    disabled, every stage with positive intent is attempted in
    problem order regardless of upstream / downstream relationship.
    The DAG-priority path is exercised in
    ``test_saturation_aware_dag_growth.py``.
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
        """``check_invariants_after_phase`` is called with ``PhaseBoundary.GROW`` after grow."""
        scheduler = _scheduler([("A", None)])
        state = _problem_state([("A", 1, 1, False)])

        # ``check_invariants_after_phase`` is bound on the runner-driven
        # ``PhaseInvariantSuite`` (see ``phase_invariants.py``); a single
        # patch covers every per-phase boundary call. The Phase C boundary
        # is the third invocation in the canonical phase order.
        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.invariants.suite.check_invariants_after_phase"
        ) as boundary_check:
            _autoscale_with_intents(scheduler, state, {"A": 1})

        phase_c_calls = [
            call for call in boundary_check.call_args_list if call.kwargs["phase_name"] is PhaseBoundary.GROW
        ]
        assert len(phase_c_calls) == 1


class TestSaturationDrivenIntegration:
    """End-to-end: a SATURATED_CRITICAL classifier signal grows the stage via Phase C."""

    def test_saturated_critical_signal_grows_via_phase_c(self) -> None:
        """A real classifier signal flowing through ``intent_phase.compute`` triggers a Phase C add."""
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
        assert scheduler.last_cycle.intent.deltas["hot"] > 0
        assert len(solution.stages[0].new_workers) == scheduler.last_cycle.intent.deltas["hot"]


class TestPhaseCDonorCoordinatorInvariantDefense:
    """Phase C raises :class:`SchedulerInvariantError` when the coordinator violates its contract.

    :attr:`DonorAcquireResult.committed` is defined as
    ``self.plan is not None`` (donor/types.py:204), so an
    attrs-constructed instance cannot report ``committed=True``
    paired with ``plan=None``. A coordinator defect (or a
    hypothetical subclass that overrides ``committed``) returning
    a mismatched pair is surfaced by the phase as a hard
    ``SchedulerInvariantError`` rather than a context-free
    ``AttributeError`` from the downstream ``plan.removals``
    dereference. The raise also survives ``python -O`` where the
    historical ``assert`` would have been stripped.
    """

    def test_post_commit_malformed_acquire_result_raises_invariant_error(self) -> None:
        """A malformed acquire result hits the ``DonorBackedAddExecutor`` invariant guard.

        Forces the donor fallback by patching ``try_add_worker``
        to return ``None`` on every call (cluster placement
        exhausted from the receiver's perspective) so the
        ``coordinator.acquire`` patch is reached. The executor
        first runs a post-commit receiver retry; when that retry
        also returns ``None`` the executor raises
        :class:`SchedulerInvariantError` with the planner-divergence
        message. The malformed ``plan=None`` companion check is
        unreachable in this scenario because the retry-None guard
        fires first, but either guard surfaces the same operator-
        actionable invariant break.
        """
        # Two-stage problem so Phase C has another stage to consider
        # as a donor candidate; one CPU per worker on a 4-CPU node
        # so the cluster is already at capacity once both stages have
        # one worker.
        scheduler = _scheduler([("donor", None), ("receiver", None)])
        state = _problem_state([("donor", 3, 1, False), ("receiver", 1, 1, False)])

        with patch(
            "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
            return_value=None,
        ):
            with patch.object(DonorCoordinator, "acquire", return_value=_MalformedAcquireResult()):
                with pytest.raises(
                    SchedulerInvariantError,
                    match=r"post-commit retry returned None for receiver 'receiver'",
                ):
                    _autoscale_with_intents(scheduler, state, {"receiver": 1})

        assert scheduler.runner.grow_services.donor_executor.allocation_gate.aborted_cycle is False, (
            "SchedulerInvariantError must not engage the AllocationError absorb path"
        )
