# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``SaturationAwareScheduler._run_phase_d_shrink``.

Phase D applies negative intent deltas as planner removes via
``ctx.try_remove_worker``. Selection is idle-first, then age-DESC,
then ``worker_id`` ASC, using per-worker ``num_used_slots`` from
``ProblemWorkerGroupState``. The contract under test:

    * Negative intent removes ``min(|intent|, current - floor)``
      workers, idle-first and oldest within each idle/busy bucket.
    * The configured stage floor (``min_workers``,
      ``min_workers_per_node * num_nodes``) is never crossed.
    * Manual stages and finished stages are skipped.
    * The Phase D invariant gate runs after the shrink.
"""

import logging
import sys
import collections
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.streaming import Autoscaler
from cosmos_xenna.pipelines.private.scheduling_py.invariants import PhaseBoundary
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.ray_utils import actor_pool


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in ``test_saturation_aware_phase_c_basic.py``.
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


def _cluster(*, total_cpus: int = 16, num_nodes: int = 1) -> resources.ClusterResources:
    """Build a CPU cluster for Phase D fixtures."""
    return resources.ClusterResources(
        nodes={
            f"node-{index}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus,
                gpus=[],
                name=f"node-{index}",
            )
            for index in range(num_nodes)
        },
    )


def _problem(
    stage_specs: list[tuple[str, int | None]],
    *,
    cfg: SaturationAwareConfig | None = None,
    num_nodes: int = 1,
) -> tuple[SaturationAwareScheduler, data_structures.Problem]:
    """Build a setup-completed scheduler and its matching problem."""
    if cfg is None:
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
        )
    cluster = _cluster(num_nodes=num_nodes)
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    problem = data_structures.Problem(
        cluster,
        [
            data_structures.ProblemStage(
                name=name,
                stage_batch_size=1,
                worker_shape=cpu_shape,
                requested_num_workers=requested,
                over_provision_factor=None,
            )
            for name, requested in stage_specs
        ],
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(problem)
    return scheduler, problem


def _problem_state(
    stage_specs: list[tuple[str, list[str], bool]],
    *,
    worker_used_slots: dict[str, int] | None = None,
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(stage_name, worker_ids, is_finished)`` rows.

    Args:
        stage_specs: Per-stage rows of ``(name, worker_ids, is_finished)``.
        worker_used_slots: Optional ``{worker_id: used_slots}`` mapping.
            Workers absent from the mapping default to 0 used slots
            (idle), matching the production default for Phase D.
    """
    used = worker_used_slots or {}
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(
                stage_name=name,
                workers=[
                    data_structures.ProblemWorkerGroupState.make(
                        worker_id,
                        [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                        num_used_slots=used.get(worker_id, 0),
                    )
                    for worker_id in worker_ids
                ],
                slots_per_worker=1,
                is_finished=finished,
            )
            for name, worker_ids, finished in stage_specs
        ],
    )


def _autoscale_with_intents(
    scheduler: SaturationAwareScheduler,
    state: data_structures.ProblemState,
    intents: dict[str, int],
) -> data_structures.Solution:
    """Run autoscale with injected signed intent deltas."""
    with patch.object(scheduler, "_compute_intent_deltas", return_value=dict(intents)):
        return scheduler.autoscale(time=0.0, problem_state=state)


def _worker_group(worker_id: str) -> resources.WorkerGroup:
    """Build a one-CPU worker group snapshot for streaming fixtures."""
    return resources.WorkerGroup.make(
        worker_id,
        "A",
        [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
    )


def _slot(*, used: bool) -> actor_pool._Slot[object]:
    """Build a ready-actor slot with or without an assigned task."""
    task = cast(actor_pool._SlotData[object], object()) if used else None
    return actor_pool._Slot(task=task)


def _ready_actor(*, used_slots: int, empty_slots: int) -> actor_pool._ReadyActor[object]:
    """Build a ready actor with the requested used/empty slot counts."""
    slots = [_slot(used=True) for _ in range(used_slots)]
    slots.extend(_slot(used=False) for _ in range(empty_slots))
    return actor_pool._ReadyActor(
        metadata=resources.WorkerMetadata.make_dummy(),
        actor_ref=cast(Any, object()),
        start_time=0.0,
        slots=collections.deque(slots),
    )


class _FakeAllocator:
    """Worker allocator shell for end-to-end Phase D tests."""

    def get_workers_in_stage(self, _stage_name: str) -> list[resources.WorkerGroup]:
        """Return the stage's current worker groups."""
        return [_worker_group("busy-A"), _worker_group("idle-B"), _worker_group("idle-C")]


def _real_actor_pool_for_phase_d() -> actor_pool.ActorPool[object, object]:
    """Build an ``ActorPool`` shell whose per-worker signal differentiates workers."""
    pool = actor_pool.ActorPool.__new__(actor_pool.ActorPool)
    pool._name = "A"
    pool._slots_per_actor = 1
    pool._ready_actors = {
        "actor-busy": _ready_actor(used_slots=1, empty_slots=0),
        "actor-idle-b": _ready_actor(used_slots=0, empty_slots=1),
        "actor-idle-c": _ready_actor(used_slots=0, empty_slots=1),
    }
    pool._pending_actors = cast(Any, collections.OrderedDict())
    pool._task_queue = cast(Any, collections.deque())
    pool._worker_groups = {
        "busy-A": actor_pool._WorkerGroup(
            worker_group=_worker_group("busy-A"),
            actors={"actor-busy"},
            state=actor_pool._WorkerGroupState.READY,
            rendevous_params=None,
        ),
        "idle-B": actor_pool._WorkerGroup(
            worker_group=_worker_group("idle-B"),
            actors={"actor-idle-b"},
            state=actor_pool._WorkerGroupState.READY,
            rendevous_params=None,
        ),
        "idle-C": actor_pool._WorkerGroup(
            worker_group=_worker_group("idle-C"),
            actors={"actor-idle-c"},
            state=actor_pool._WorkerGroupState.READY,
            rendevous_params=None,
        ),
    }
    return cast(actor_pool.ActorPool[object, object], pool)


def _make_problem_state_from_actor_pool(pool: actor_pool.ActorPool[object, object]) -> data_structures.ProblemState:
    """Build ``ProblemState`` through the production streaming snapshot method."""
    autoscaler = Autoscaler.__new__(Autoscaler)
    autoscaler._allocator = _FakeAllocator()  # type: ignore[attr-defined, assignment]
    return autoscaler._make_problem_state([pool], [False])


class TestPhaseDScaleDownContract:
    """Pin the Phase D scale-down contract."""

    def test_negative_intent_deletes_requested_workers(self) -> None:
        """A negative intent removes ``abs(intent)`` workers when above floor.

        With age=0 for every test worker (the planner has not yet
        observed prior cycles), the age-DESC sort breaks ties by
        ``worker_id ASC``: ``A-w0`` and ``A-w1`` are removed first.
        """
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        deleted_ids = sorted(worker.id for worker in solution.stages[0].deleted_workers)
        assert deleted_ids == ["A-w0", "A-w1"]

    def test_oldest_workers_are_deleted_before_younger_workers(self) -> None:
        """Worker age decides deletion order before the worker-id tiebreaker."""
        scheduler, _ = _problem([("A", None)])
        scheduler._worker_ages = {
            "young": 0,
            "old": 10,
            "middle": 5,
            "newer": 1,
        }
        state = _problem_state([("A", ["young", "old", "middle", "newer"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["old", "middle"]

    def test_floor_prevents_over_deletion(self) -> None:
        """Scale-down never deletes below the configured stage floor."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=3),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -10})

        assert len(solution.stages[0].deleted_workers) == 2

    def test_per_node_floor_prevents_over_deletion(self) -> None:
        """Scale-down respects ``min_workers_per_node * num_nodes``."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1, min_workers_per_node=2),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg, num_nodes=2)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -10})

        assert len(solution.stages[0].deleted_workers) == 1

    def test_finished_stage_is_not_scaled_down(self) -> None:
        """A finished stage ignores negative intent just as it ignores positive intent."""
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1"], True)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        assert solution.stages[0].deleted_workers == []

    def test_phase_d_boundary_is_checked_before_solution_shape(self) -> None:
        """The Phase D invariant boundary runs after deletes and before finalization."""
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1"], False)])
        call_order: list[PhaseBoundary | str] = []

        def _record_phase(**kwargs: object) -> None:
            phase_name = cast(PhaseBoundary, kwargs["phase_name"])
            if phase_name is PhaseBoundary.PHASE_D:
                ctx = cast(data_structures.AutoscalePlanContext, kwargs["ctx"])
                assert ctx.pending_remove_count(0) == 1
            call_order.append(phase_name)

        def _record_solution_shape(**_kwargs: object) -> None:
            call_order.append("solution_shape")

        with (
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_invariants_after_phase",
                side_effect=_record_phase,
            ),
            patch(
                "cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.check_solution_shape",
                side_effect=_record_solution_shape,
            ),
        ):
            _autoscale_with_intents(scheduler, state, {"A": -1})

        assert PhaseBoundary.PHASE_D in call_order
        assert call_order.index(PhaseBoundary.PHASE_D) < call_order.index("solution_shape")

    def test_idle_worker_is_selected_before_busy_worker(self) -> None:
        """Idle workers are removed before busy workers regardless of age.

        Pins the per-worker idle-first contract: ``busy-A`` carries
        ``num_used_slots > 0`` so it is shielded from removal even
        though it is the lexicographically-first id (the worst case
        for the age-DESC, worker_id-ASC tiebreaker fallback). Only the
        two idle workers are eligible.
        """
        scheduler, _ = _problem([("A", None)])
        state = _problem_state(
            [("A", ["busy-A", "idle-B", "idle-C"], False)],
            worker_used_slots={"busy-A": 1, "idle-B": 0, "idle-C": 0},
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        deleted_ids = [worker.id for worker in solution.stages[0].deleted_workers]
        assert "busy-A" not in deleted_ids, "busy worker must not be removed when idle alternatives exist"
        assert deleted_ids == ["idle-B"]

    def test_actor_pool_idle_signal_survives_to_phase_d_selection(self) -> None:
        """Real actor-pool slot data shields the busy worker through Phase D."""
        scheduler, _ = _problem([("A", None)])
        state = _make_problem_state_from_actor_pool(_real_actor_pool_for_phase_d())

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        deleted_ids = [worker.id for worker in solution.stages[0].deleted_workers]
        assert deleted_ids == ["idle-B"]

    def test_unready_zero_signal_is_eligible_for_idle_first_shrink(self) -> None:
        """Current policy treats a not-yet-ready worker-group signal as idle."""
        scheduler, _ = _problem([("A", None)])
        scheduler._worker_ages = {"busy-old": 100, "unready-new": 0}
        state = _problem_state(
            [("A", ["busy-old", "unready-new"], False)],
            worker_used_slots={"busy-old": 1, "unready-new": 0},
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["unready-new"]

    def test_all_busy_workers_falls_back_to_age_only_selection(self) -> None:
        """When every worker is busy, the helper falls back to age-DESC ordering.

        Pins the degenerate case: the idle key collapses to a single
        bucket of busy workers, so the sort reduces to
        ``(age DESC, worker_id ASC)``. Without this contract, an
        OVER_PROVISIONED stage with no idle workers would refuse to
        shrink even when the floor allows it -- a hang under sustained
        load. Lex-first ``A-w0`` wins on age ties.
        """
        scheduler, _ = _problem([("A", None)])
        state = _problem_state(
            [("A", ["A-w0", "A-w1", "A-w2"], False)],
            worker_used_slots={"A-w0": 1, "A-w1": 1, "A-w2": 1},
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        deleted_ids = [worker.id for worker in solution.stages[0].deleted_workers]
        assert deleted_ids == ["A-w0"]

    def test_two_idle_one_busy_intent_minus_two_takes_only_idle(self) -> None:
        """With intent=-2 and one busy worker, both idle workers are removed.

        Pins that the idle bucket is fully consumed before the busy
        bucket: ``busy-mid`` is shielded even when the deletion count
        would reach into the busy bucket if idle-first did not hold.
        """
        scheduler, _ = _problem([("A", None)])
        state = _problem_state(
            [("A", ["idle-A", "busy-mid", "idle-Z"], False)],
            worker_used_slots={"idle-A": 0, "busy-mid": 5, "idle-Z": 0},
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        deleted_ids = sorted(worker.id for worker in solution.stages[0].deleted_workers)
        assert deleted_ids == ["idle-A", "idle-Z"]

    def test_intent_exceeds_idle_bucket_extends_into_busy_bucket(self) -> None:
        """When intent > idle-bucket size, the helper falls through to busy workers.

        Pins that the idle-first key is a sort priority, not a hard
        gate: scale-down does not stall just because the idle bucket
        is exhausted. With one idle worker and intent=-2, both ``idle``
        and the busiest-age-eligible ``busy`` are removed (subject to
        the floor cap, which is 1 here so 2 deletions are allowed).
        """
        scheduler, _ = _problem([("A", None)])
        state = _problem_state(
            [("A", ["A-w0", "A-w1", "A-w2"], False)],
            worker_used_slots={"A-w0": 1, "A-w1": 0, "A-w2": 1},
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        deleted_ids = sorted(worker.id for worker in solution.stages[0].deleted_workers)
        assert "A-w1" in deleted_ids, "the idle worker must be removed first"
        assert len(deleted_ids) == 2

    def test_planner_refusal_raises_runtime_error(self) -> None:
        """Planner refusing a victim from its own snapshot raises ``RuntimeError``.

        The victim ids come from ``ctx.worker_ids_by_stage()``;
        refusal therefore signals planner-snapshot inconsistency --
        a scheduler defect, not an operator-config issue. The cycle
        must abort loudly rather than silently skip the shrink so
        the corrupted plan does not reach ``into_solution()``.
        """
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2"], False)])

        with (
            patch.object(
                data_structures.AutoscalePlanContext,
                "try_remove_worker",
                return_value=False,
            ),
            pytest.raises(
                RuntimeError,
                match=r"Phase D shrink:.*stage 'A'.*planner refused.*'A-w0'",
            ),
        ):
            _autoscale_with_intents(scheduler, state, {"A": -1})

    def test_intent_exact_to_current_minus_floor_deletes_full_amount(self) -> None:
        """Boundary: ``|intent| == current - floor`` deletes the full intent.

        The off-by-one fault line of Phase D lives in
        ``actual_remove = min(|intent|, current - floor)``. A regression
        to ``current - floor - 1`` would leave one worker over the
        target on every clamped shrink and is caught here.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=2),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -3})

        assert len(solution.stages[0].deleted_workers) == 3

    def test_intent_one_above_floor_clamps_and_logs_deficit_one(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Boundary: one above floor clamps and surfaces ``deficit=1`` in the INFO log."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=2),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -4})

        assert len(solution.stages[0].deleted_workers) == 3
        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        clamp_logs = [m for m in infos if "saturation-aware scale-down" in m]
        assert len(clamp_logs) == 1
        assert "deficit=1" in clamp_logs[0]

    def test_intent_at_floor_exactly_produces_no_deletes(self) -> None:
        """Shrink is a no-op when ``current == floor`` regardless of intent magnitude."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=2),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        assert solution.stages[0].deleted_workers == []

    def test_zero_intent_does_not_shrink(self) -> None:
        """An intent of 0 (NORMAL classifier) is a no-op."""
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 0})

        assert solution.stages[0].deleted_workers == []

    def test_positive_intent_does_not_shrink_phase_d(self) -> None:
        """A positive intent is Phase C's responsibility; Phase D must not delete."""
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": 3})

        assert solution.stages[0].deleted_workers == []

    def test_manual_stage_is_not_scaled_down_by_phase_d(self) -> None:
        """A manual stage is excluded from Phase D even when its intent is negative."""
        scheduler, _ = _problem([("A", 2)])  # manual: requested=2
        state = _problem_state([("A", ["A-w0", "A-w1"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -5})

        assert solution.stages[0].deleted_workers == []

    def test_two_stages_independent_shrink(self) -> None:
        """One stage's floor clamp must not stop the loop from processing the other."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            per_stage_overrides={"A": SaturationAwareStageConfig(min_workers=2)},
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
        )
        scheduler, _ = _problem([("A", None), ("B", None)], cfg=cfg)
        state = _problem_state(
            [
                ("A", ["A-w0", "A-w1"], False),
                ("B", ["B-w0", "B-w1", "B-w2", "B-w3"], False),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -3, "B": -2})

        assert solution.stages[0].deleted_workers == [], "stage A is at its floor; no deletes"
        assert len(solution.stages[1].deleted_workers) == 2, "stage B shrinks independently"

    def test_int_min_intent_terminates_at_floor(self) -> None:
        """A negative intent of ``-sys.maxsize`` floor-clamps without infinite loop."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=2),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -sys.maxsize})

        assert len(solution.stages[0].deleted_workers) == 2

    def test_missing_intent_entry_does_not_shrink(self) -> None:
        """A stage absent from the intent dict defaults to 0 and is a no-op."""
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1"], False)])

        solution = _autoscale_with_intents(scheduler, state, {})

        assert solution.stages[0].deleted_workers == []

    def test_no_info_log_when_request_fully_satisfied(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A clean shrink (no clamp) emits no scale-down INFO log."""
        scheduler, _ = _problem([("A", None)])
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        _autoscale_with_intents(scheduler, state, {"A": -2})

        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        assert not any("saturation-aware scale-down" in m for m in infos)


class TestPhaseDMultiCycleStability:
    """Multi-cycle scale-down stability: floor holds without spurious side effects."""

    def test_shrink_then_floor_subsequent_cycle_no_more_deletes(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After a stage hits its floor, repeated negative intent is a clean no-op."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=2),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        cycle1_state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])
        cycle2_state = _problem_state([("A", ["A-w0", "A-w1"], False)])

        sol1 = _autoscale_with_intents(scheduler, cycle1_state, {"A": -5})
        assert len(sol1.stages[0].deleted_workers) == 3

        loguru_caplog.clear()
        sol2 = _autoscale_with_intents(scheduler, cycle2_state, {"A": -5})
        sol3 = _autoscale_with_intents(scheduler, cycle2_state, {"A": -5})

        assert sol2.stages[0].deleted_workers == []
        assert sol3.stages[0].deleted_workers == []
        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        assert not any("saturation-aware scale-down" in m for m in infos), (
            f"steady-state cycles must not emit clamp INFOs; got: {infos}"
        )
