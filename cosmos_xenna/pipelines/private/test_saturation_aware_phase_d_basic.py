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


"""Tests for ``SaturationAwareScheduler._run_phase_d_shrink``.

Phase D applies negative intent deltas as planner removes via
``ctx.try_remove_worker``. Selection is consolidation-first
(``host_gpu_used_fraction`` ASC), then idle-first, then age-DESC,
then ``worker_id`` ASC, using per-worker ``num_used_slots`` from
``ProblemWorkerGroupState`` and GPU allocation fractions from the
cycle snapshot. The contract under test:

    * Negative intent removes ``min(|intent|, current - floor)``
      workers, idle-first and oldest within each idle/busy bucket.
    * The configured stage floor (``min_workers``,
      ``min_workers_per_node * num_nodes``) is never crossed.
    * Manual stages and finished stages are skipped.
    * The Phase D invariant gate runs after the shrink.
"""

import collections
import logging
import sys
import uuid
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.invariants import PhaseBoundary
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.pipelines.private.streaming import Autoscaler
from cosmos_xenna.ray_utils import actor_pool


def _test_stage_config(**overrides: object) -> SaturationAwareStageConfig:
    """Build a :class:`SaturationAwareStageConfig` with both warmup graces disabled.

    Phase D fixtures focus on shrink victim selection, floor and
    fraction caps, and the consolidation-first sort. None of those
    contracts are about the donor warmup grace, so this helper
    pins both grace fields to ``0.0`` and forwards every other
    setting through. The dedicated warmup-grace tests live in
    ``test_worker_warmup_grace.py`` and
    ``test_donor_warmup_grace.py``.
    """
    return SaturationAwareStageConfig(
        worker_warmup_measurement_grace_s=0.0,
        donor_warmup_grace_s=0.0,
        **overrides,  # type: ignore[arg-type]
    )


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


def _gpu_cluster(
    *,
    total_cpus: int = 16,
    num_nodes: int = 1,
    num_gpus_per_node: int = 4,
) -> resources.ClusterResources:
    """Build a GPU cluster used by the consolidation tiebreak fixtures.

    Each node carries ``num_gpus_per_node`` GPU slots, all initially
    unallocated (``used_fraction=0``). The Rust planner seeds
    per-worker fractions during ``AutoscalePlanContext.from_problem_state``,
    so the cluster only needs the GPU slot structure to satisfy the
    "GPU offset exists on this node" validation.
    """
    return resources.ClusterResources(
        nodes={
            f"node-{index}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus,
                gpus=[
                    resources.GpuResources(
                        index=g,
                        uuid_=uuid.uuid4(),
                        used_fraction=0.0,
                    )
                    for g in range(num_gpus_per_node)
                ],
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
    total_cpus: int = 16,
) -> tuple[SaturationAwareScheduler, data_structures.Problem]:
    """Build a setup-completed scheduler and its matching problem.

    The default fixture sets ``max_scale_down_fraction_per_cycle=1.0``
    so the orchestrator-level fraction cap is effectively a no-op for
    tests that pin floor / intent-magnitude / selection-order
    behaviour. Tests targeting the fraction cap itself construct
    their own ``cfg`` with a smaller fraction.
    """
    if cfg is None:
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=1.0,
            ),
        )
    cluster = _cluster(num_nodes=num_nodes, total_cpus=total_cpus)
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


def _gpu_problem(
    stage_specs: list[tuple[str, int | None]],
    *,
    cfg: SaturationAwareConfig | None = None,
    num_nodes: int = 1,
    num_gpus_per_node: int = 4,
    total_cpus: int = 16,
    gpu_per_worker: float = 0.5,
) -> tuple[SaturationAwareScheduler, data_structures.Problem]:
    """Build a setup-completed scheduler with GPU-shaped worker stages.

    Used by tests that need a cluster with GPU slots so the planner
    accepts pre-seeded worker allocations on specific GPU offsets.

    Args:
        stage_specs: Per-stage rows of ``(name, requested_num_workers)``.
        cfg: Optional override config. When ``None``, uses the default
            ``max_scale_down_fraction_per_cycle=1.0`` so the
            orchestrator-level fraction cap is a no-op.
        num_nodes: Cluster node count.
        num_gpus_per_node: GPU slots per node.
        total_cpus: CPU count per node.
        gpu_per_worker: Fractional GPU per worker for the stage shape.
            Only matters for ``try_add_worker``; pre-seeded workers
            carry their own ``WorkerResources`` allocations directly.

    """
    if cfg is None:
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=1.0,
            ),
        )
    cluster = _gpu_cluster(num_nodes=num_nodes, total_cpus=total_cpus, num_gpus_per_node=num_gpus_per_node)
    gpu_shape = resources.Resources(cpus=1.0, gpus=gpu_per_worker).to_worker_shape(cluster)
    problem = data_structures.Problem(
        cluster,
        [
            data_structures.ProblemStage(
                name=name,
                stage_batch_size=1,
                worker_shape=gpu_shape,
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


def _gpu_problem_state(
    stage_specs: list[
        tuple[
            str,
            list[tuple[str, list[tuple[str, int, float]]]],
            bool,
        ]
    ],
    *,
    worker_used_slots: dict[str, int] | None = None,
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` whose workers carry explicit GPU allocations.

    The ``stage_specs`` schema is one row per stage:

    ``(stage_name, [(worker_id, [(node, gpu_offset, used_fraction), ...]), ...], is_finished)``

    Each inner list of ``(node, gpu_offset, used_fraction)`` triples
    becomes a ``WorkerResourcesInternal`` with one ``GpuAllocationInternal``
    per triple. Use this fixture for tests that need to pin the
    ``host_gpu_used_fraction`` consolidation primary key.

    Args:
        stage_specs: One row per stage; see schema above.
        worker_used_slots: Optional ``{worker_id: used_slots}`` mapping.
            Workers absent from the mapping default to 0 used slots
            (idle), matching the production default for Phase D.

    """
    used = worker_used_slots or {}
    rows: list[data_structures.ProblemStageState] = []
    for stage_name, worker_rows, finished in stage_specs:
        workers = []
        for worker_id, gpu_triples in worker_rows:
            # Group GPU allocations by node so each WorkerResourcesInternal
            # collects every GpuAllocation on the same node. Multiple
            # WorkerResources entries per worker group correspond to SPMD
            # workers spread across nodes; multiple GpuAllocation entries
            # within a single WorkerResources correspond to a non-SPMD
            # multi-GPU worker on one node.
            gpus_by_node: dict[str, list[resources.GpuAllocationInternal]] = {}
            for node, offset, fraction in gpu_triples:
                gpus_by_node.setdefault(node, []).append(
                    resources.GpuAllocationInternal(offset=offset, used_fraction=fraction),
                )
            allocations = [
                resources.WorkerResourcesInternal(node=node, cpus=1.0, gpus=gpu_list)
                for node, gpu_list in gpus_by_node.items()
            ] or [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])]
            workers.append(
                data_structures.ProblemWorkerGroupState.make(
                    worker_id,
                    allocations,
                    num_used_slots=used.get(worker_id, 0),
                )
            )
        rows.append(
            data_structures.ProblemStageState(
                stage_name=stage_name,
                workers=workers,
                slots_per_worker=1,
                is_finished=finished,
            )
        )
    return data_structures.ProblemState(rows)


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
    return cast(
        resources.WorkerGroup,
        type(
            "_FakeWorkerGroup",
            (),
            {
                "id": worker_id,
                "allocations": [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            },
        )(),
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
    """Build an ``ActorPool`` shell whose per-worker signal differentiates workers.

    The shell mirrors the production ``ActorPool`` shape that the streaming
    snapshot reads: every actor-set attribute the production
    ``num_pending_actors`` and ``num_used_slots`` properties touch must
    exist, otherwise ``_make_problem_state`` raises ``AttributeError``
    against ``ActorPool.__new__`` shells. The test only exercises ready
    actors, so every pending-side container is empty.
    """
    pool = actor_pool.ActorPool.__new__(actor_pool.ActorPool)
    pool._name = "A"
    pool._slots_per_actor = 1
    pool._ready_actors = {
        "actor-busy": _ready_actor(used_slots=1, empty_slots=0),
        "actor-idle-b": _ready_actor(used_slots=0, empty_slots=1),
        "actor-idle-c": _ready_actor(used_slots=0, empty_slots=1),
    }
    pool._pending_actors = cast(Any, collections.OrderedDict())
    pool._pending_node_actors = cast(Any, collections.OrderedDict())
    pool._actors_waiting_for_node_setup = cast(Any, {})
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
    """Build ``ProblemState`` through the production streaming snapshot method.

    Passes zeros for ``upstream_queue_lens`` and ones for
    ``stage_batch_sizes`` so this helper exercises only the pool-side
    signal path; the upstream-aggregation path has its own focused
    coverage in ``test_streaming_slot_signals.py``.
    """
    autoscaler = Autoscaler.__new__(Autoscaler)
    autoscaler._allocator = _FakeAllocator()  # type: ignore[attr-defined, assignment]
    return autoscaler._make_problem_state([pool], [False], [0], [1])


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
            stage_defaults=_test_stage_config(
                min_workers=3,
                max_scale_down_fraction_per_cycle=1.0,
            ),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -10})

        assert len(solution.stages[0].deleted_workers) == 2

    def test_per_node_floor_prevents_over_deletion(self) -> None:
        """Scale-down respects ``min_workers_per_node * num_nodes``."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                min_workers_per_node=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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
            stage_defaults=_test_stage_config(
                min_workers=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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
            stage_defaults=_test_stage_config(
                min_workers=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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
            stage_defaults=_test_stage_config(
                min_workers=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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
            per_stage_overrides={
                "A": _test_stage_config(
                    min_workers=2,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            },
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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
            stage_defaults=_test_stage_config(
                min_workers=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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


class TestPhaseDPerCycleFractionClamp:
    """The orchestrator-level ``max_scale_down_fraction_per_cycle`` clamp.

    Defense-in-depth on top of the per-stage / per-node floor cap and
    the existing ``compute_delta._shrink_delta`` magnitude cap. The
    clamp protects against externally-injected intents that bypass
    ``compute_delta`` (e.g. test fixtures, future schedulers that
    compute intent differently), preventing cliff scale-downs on a
    100-actor stage when the floor alone would let the stage drop by
    a large delta in a single cycle.
    """

    def test_fraction_cap_limits_per_cycle_deletions(self) -> None:
        """100-worker stage with ``fraction=0.05`` and ``intent=-50`` deletes only 5."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=0.05,
            ),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg, total_cpus=128)
        worker_ids = [f"A-w{i:03d}" for i in range(100)]
        state = _problem_state([("A", worker_ids, False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -50})

        assert len(solution.stages[0].deleted_workers) == 5

    def test_fraction_cap_floor_one_when_fraction_floors_to_zero(self) -> None:
        """``max(1, floor(current * fraction))`` ensures progress on tiny fractions.

        With ``current=5`` and ``fraction=0.05``, the bare floor would
        be ``floor(0.25) = 0``. The ``max(1, ...)`` guard ensures the
        stage still shrinks by 1 per cycle rather than getting stuck.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=0.05,
            ),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -3})

        assert len(solution.stages[0].deleted_workers) == 1

    def test_floor_cap_dominates_when_smaller_than_fraction_cap(self) -> None:
        """When the floor cap is the tighter constraint, it binds; fraction cap is non-binding."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=4,
                max_scale_down_fraction_per_cycle=1.0,
            ),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -10})

        assert len(solution.stages[0].deleted_workers) == 1, (
            "floor=4 with current=5 allows only 1 deletion regardless of fraction cap"
        )

    def test_fraction_cap_logs_dedicated_info_when_binding(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Operators see a fraction-specific INFO log when the fraction cap binds, not the floor log."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=0.10,
            ),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg, total_cpus=64)
        worker_ids = [f"A-w{i:03d}" for i in range(50)]
        state = _problem_state([("A", worker_ids, False)])

        _autoscale_with_intents(scheduler, state, {"A": -30})

        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        scale_down = [m for m in infos if "saturation-aware scale-down" in m]
        assert len(scale_down) == 1, f"expected one fraction-clamp INFO; got: {scale_down}"
        msg = scale_down[0]
        assert "per-cycle fraction cap" in msg, "fraction-cap log must use distinct text"
        assert "max_scale_down_fraction_per_cycle=0.1" in msg
        assert "deficit=25" in msg

    def test_intent_magnitude_dominates_when_smallest(self) -> None:
        """When ``|intent|`` is smaller than both caps, no clamp binds; intent passes through."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=1.0,
            ),
        )
        scheduler, _ = _problem([("A", None)], cfg=cfg)
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2", "A-w3", "A-w4"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        assert len(solution.stages[0].deleted_workers) == 2


class TestPhaseDMultiCycleStability:
    """Multi-cycle scale-down stability: floor holds without spurious side effects."""

    def test_shrink_then_floor_subsequent_cycle_no_more_deletes(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """After a stage hits its floor, repeated negative intent is a clean no-op."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
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


class TestPhaseDConsolidationTiebreak:
    """Pin the ``host_gpu_used_fraction`` primary sort key contract end-to-end.

    Workers placed on GPUs whose total used fraction is lowest are
    removed first so a fractional shrink can free whole GPUs for
    downstream whole-GPU stages.
    """

    def test_three_to_one_shrink_frees_whole_gpus(self) -> None:
        """A 3->1 shrink keeps the actor on the highest-fraction GPU and frees the others."""
        scheduler, _ = _gpu_problem([("A", None)])
        # Three actors, each on its own GPU, each with a different used_fraction. The
        # 3->1 shrink must keep ``A-heavy`` (the actor on the most-loaded GPU) so the
        # other two GPUs become whole-unallocated.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.20)]),
                        ("A-medium", [("node-0", 1, 0.40)]),
                        ("A-heavy", [("node-0", 2, 0.80)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        deleted_ids = sorted(worker.id for worker in solution.stages[0].deleted_workers)
        assert deleted_ids == ["A-light", "A-medium"], (
            "consolidation tiebreak must delete the workers on the lowest-fraction GPUs"
        )

    def test_cross_stage_fraction_visibility_drives_selection(self) -> None:
        """An upstream stage's allocation on the same GPU lifts the consolidation key."""
        scheduler, _ = _gpu_problem([("A", None), ("B", None)])
        # Stage A has 2 actors. ``A-on-shared`` shares GPU offset 0 on node-0 with stage
        # B's actor (which holds 0.50 of GPU-0). ``A-on-private`` has GPU-1 to itself
        # at 0.30. The consolidation primary key sees:
        #   (node-0, 0) -> 0.30 (A) + 0.50 (B) = 0.80
        #   (node-0, 1) -> 0.30 (A only)
        # Phase D shrinks A by one. ``A-on-private`` must be deleted (lower fraction)
        # so GPU-1 becomes whole-unallocated; ``A-on-shared`` survives because deleting
        # it would only drop GPU-0 from 0.80 to 0.50 (still allocated by B).
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-on-shared", [("node-0", 0, 0.30)]),
                        ("A-on-private", [("node-0", 1, 0.30)]),
                    ],
                    False,
                ),
                (
                    "B",
                    [
                        ("B-on-shared", [("node-0", 0, 0.50)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1, "B": 0})

        a_deleted = [worker.id for worker in solution.stages[0].deleted_workers]
        b_deleted = [worker.id for worker in solution.stages[1].deleted_workers]
        assert a_deleted == ["A-on-private"], "the GPU shared with another stage must be deprioritized for deletion"
        assert b_deleted == [], "stage B was not asked to shrink"

    def test_cpu_only_stage_falls_back_to_idle_age_id_ordering(self) -> None:
        """A CPU-only stage has no GPU footprint; consolidation degrades to the prior 3-key sort."""
        scheduler, _ = _problem([("A", None)])
        scheduler._worker_ages = {"A-w0": 100, "A-w1": 50, "A-w2": 1}
        state = _problem_state([("A", ["A-w0", "A-w1", "A-w2"], False)])

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        # Without GPU fractions every worker has consolidation key 0; the sort
        # degrades to (idle, age DESC, worker_id ASC). All workers idle, oldest-first.
        deleted_ids = sorted(worker.id for worker in solution.stages[0].deleted_workers)
        assert deleted_ids == ["A-w0", "A-w1"]

    def test_equal_gpu_fraction_falls_back_to_idle_age_id(self) -> None:
        """When every actor sees the same GPU fraction, the secondary keys decide."""
        scheduler, _ = _gpu_problem([("A", None)])
        scheduler._worker_ages = {"A-young": 1, "A-mid": 5, "A-old": 50}
        # All three workers on GPUs with the same total fraction. The secondary
        # idle/age/id key picks the oldest of the three for deletion.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-young", [("node-0", 0, 0.50)]),
                        ("A-mid", [("node-0", 1, 0.50)]),
                        ("A-old", [("node-0", 2, 0.50)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        deleted_ids = [worker.id for worker in solution.stages[0].deleted_workers]
        assert deleted_ids == ["A-old"], "equal fractions must defer to age-DESC selection"

    def test_multi_gpu_worker_uses_max_per_worker_fraction(self) -> None:
        """For SPMD-style multi-GPU workers, the most-loaded GPU dominates the sort key."""
        scheduler, _ = _gpu_problem([("A", None)])
        # Two actors. ``A-spread`` straddles GPU-0 (0.95) and GPU-1 (0.10): MAX = 0.95.
        # ``A-single`` sits alone on GPU-2 (0.20). ``A-single`` has the lower per-worker
        # fraction and must be deleted first; ``A-spread`` survives because deleting it
        # would not free GPU-0 (still 0.95 from the worker itself, but post-deletion 0).
        # The MAX-of-allocations rule reflects the worst-case constraint that a worker
        # contributes to: GPU-0 stays loaded when other stages are present.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-spread", [("node-0", 0, 0.95), ("node-0", 1, 0.10)]),
                        ("A-single", [("node-0", 2, 0.20)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        deleted_ids = [worker.id for worker in solution.stages[0].deleted_workers]
        assert deleted_ids == ["A-single"], "the per-worker MAX-fraction must drive the consolidation tiebreak"

    def test_consolidation_outranks_idle_at_orchestrator_layer(self) -> None:
        """A busy worker on a low-fraction GPU is removed before an idle worker on a heavy GPU."""
        scheduler, _ = _gpu_problem([("A", None)])
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-idle-heavy", [("node-0", 0, 0.95)]),
                        ("A-busy-light", [("node-0", 1, 0.10)]),
                    ],
                    False,
                ),
            ],
            worker_used_slots={"A-idle-heavy": 0, "A-busy-light": 5},
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        deleted_ids = [worker.id for worker in solution.stages[0].deleted_workers]
        assert deleted_ids == ["A-busy-light"], (
            "consolidation must outrank idle-status: the heavy GPU is preserved even when its actor is idle"
        )

    def test_consolidation_converges_across_two_cycles(self) -> None:
        """Across two cycles, repeated 3->1 shrinks consistently free the lowest-fraction GPUs.

        Pins the docstring's "converges over multiple cycles" claim: once the planner
        commits to the consolidation order in cycle 1, cycle 2 must NOT undo the
        consolidation (no flap). The test models the steady-state pattern an operator
        would observe in production logs.
        """
        scheduler, _ = _gpu_problem([("A", None)])

        cycle1_state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.20)]),
                        ("A-medium", [("node-0", 1, 0.40)]),
                        ("A-heavy", [("node-0", 2, 0.80)]),
                    ],
                    False,
                ),
            ],
        )
        cycle2_state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-heavy", [("node-0", 2, 0.80)]),
                    ],
                    False,
                ),
            ],
        )

        sol1 = _autoscale_with_intents(scheduler, cycle1_state, {"A": -2})
        cycle1_deleted = sorted(worker.id for worker in sol1.stages[0].deleted_workers)

        sol2 = _autoscale_with_intents(scheduler, cycle2_state, {"A": -1})
        cycle2_deleted = sorted(worker.id for worker in sol2.stages[0].deleted_workers)

        # Cycle 1: light + medium are deleted (lowest fractions), heavy survives.
        assert cycle1_deleted == ["A-light", "A-medium"]
        # Cycle 2: with min_workers=1, the floor blocks any further deletion.
        # Convergence is verified: heavy (the consolidation-preserved actor) remains.
        assert cycle2_deleted == []

    def test_consolidation_does_not_flap_when_intent_repeats(self) -> None:
        """A stage that hits its floor in cycle N stays at floor in cycle N+1 with the same input."""
        scheduler, _ = _gpu_problem([("A", None)])
        steady_state = _gpu_problem_state(
            [
                ("A", [("A-survivor", [("node-0", 0, 0.50)])], False),
            ],
        )

        sol1 = _autoscale_with_intents(scheduler, steady_state, {"A": -5})
        sol2 = _autoscale_with_intents(scheduler, steady_state, {"A": -5})

        # Floor=1 (default) prevents both cycles from deleting; the survivor is stable.
        assert sol1.stages[0].deleted_workers == []
        assert sol2.stages[0].deleted_workers == []

    def test_phase_d_after_phase_b_donor_uses_post_donation_worker_set(self) -> None:
        """When Phase B's donor fallback removes a worker before Phase D runs, Phase D's worker_ids reflect the removal.

        Pins the cross-phase invariant the cycle-start fraction map relies on: even
        though the cluster-wide GPU fraction map is computed once at cycle start,
        Phase D's per-stage ``worker_ids_by_stage`` reflects the live planner state
        after Phase A/B/C mutations. This means donations + shrinks compose
        correctly; a worker already removed by Phase B's donor fallback is not
        re-deleted by Phase D, and the consolidation key still drives the remaining
        deletions.
        """
        scheduler, _ = _gpu_problem([("A", None)])
        # Stage A starts with 4 fractional-GPU workers spread across 4 GPUs.
        # No Phase B donation is actually triggered here (intent of -2 is pure Phase D),
        # but the fixture exercises the same code path: cycle-start fraction map is
        # used to rank remaining workers consistent with the live worker set at the
        # time Phase D runs.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-w0", [("node-0", 0, 0.10)]),
                        ("A-w1", [("node-0", 1, 0.20)]),
                        ("A-w2", [("node-0", 2, 0.30)]),
                        ("A-w3", [("node-0", 3, 0.40)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        # Phase D removes the two workers on the lowest-fraction GPUs.
        deleted_ids = sorted(worker.id for worker in solution.stages[0].deleted_workers)
        assert deleted_ids == ["A-w0", "A-w1"]
        # Survivors: A-w2 (0.30) and A-w3 (0.40) — the highest-fraction GPUs are preserved.
        # No worker is deleted twice; no exception is raised.

    def test_finished_stage_still_aggregates_into_fraction_map(self) -> None:
        """Allocations held by a finished stage still count toward the consolidation key.

        Even though a finished stage cannot be shrunk itself, its workers are still
        physically holding GPU resources, so other stages' Phase D selection must see
        those allocations when computing host-GPU-used-fraction.
        """
        scheduler, _ = _gpu_problem([("A", None), ("B", None)])
        # B is finished but still holds an allocation on GPU-0 (0.5 fraction). A has
        # two actors: one sharing GPU-0 (its own 0.3, total 0.8), one alone on GPU-1
        # (only 0.3). A's shrink must drop A-on-private (lower total) and keep
        # A-on-shared even though it is on a heavier GPU; the finished stage's
        # fraction is part of the planner's input, not a candidate for deletion.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-on-shared", [("node-0", 0, 0.30)]),
                        ("A-on-private", [("node-0", 1, 0.30)]),
                    ],
                    False,
                ),
                (
                    "B",
                    [
                        ("B-finished-on-shared", [("node-0", 0, 0.50)]),
                    ],
                    True,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1})

        a_deleted = [worker.id for worker in solution.stages[0].deleted_workers]
        b_deleted = [worker.id for worker in solution.stages[1].deleted_workers]
        assert a_deleted == ["A-on-private"]
        assert b_deleted == [], "finished stages must never be selected for Phase D shrink"


class TestPhaseDConsolidationEdgeCases:
    """Adversarial edge cases for Phase D consolidation.

    These extend ``TestPhaseDConsolidationTiebreak`` with input shapes
    that surface failure modes outside the canonical 3->1 contract:
    floor-bound stages with strong consolidation signals, manual
    stages with GPU resources, multi-stage reciprocal independence,
    idempotency across repeated cycles, and intent oscillation
    stability. Failures here pinpoint orchestrator-layer regressions
    that the canonical tests would miss.
    """

    def test_stage_at_floor_does_not_delete_despite_strong_consolidation_signal(self) -> None:
        """When ``current == min_workers``, consolidation never deletes; floor wins.

        The consolidation primary key is irrelevant if the floor cap is 0; the
        sort never runs because the deletion count is clamped to zero.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=2,
                max_scale_down_fraction_per_cycle=1.0,
            ),
        )
        scheduler, _ = _gpu_problem([("A", None)], cfg=cfg)
        # current=2, floor=2 -> no shrink possible. A heavy intent (-10) and a clear
        # consolidation signal (one actor on a near-empty GPU) are both ignored.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.05)]),
                        ("A-heavy", [("node-0", 1, 0.95)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -10})

        assert solution.stages[0].deleted_workers == []

    def test_manual_stage_with_gpu_resources_is_immune_to_consolidation_shrink(self) -> None:
        """Manual stages (``requested_num_workers != None``) are skipped by Phase D.

        Even though a manual stage has GPU resources contributing to the cluster-wide
        fraction map, its own workers are never selected by the consolidation sort
        because Phase D filters manual stages out before the sort runs.
        """
        scheduler, _ = _gpu_problem([("A", 3)])  # manual: requested_num_workers=3.
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.05)]),
                        ("A-medium", [("node-0", 1, 0.50)]),
                        ("A-heavy", [("node-0", 2, 0.95)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -2})

        assert solution.stages[0].deleted_workers == []

    def test_two_gpu_stages_independently_consolidate_without_interference(self) -> None:
        """Two GPU stages each shrink to their own consolidation winner; no greedy-break.

        Pins the cross-stage independence invariant for the shrink
        path: an over-provisioned stage's consolidation never blocks
        another stage's consolidation, even if both are shrinking in
        the same cycle.
        """
        scheduler, _ = _gpu_problem([("A", None), ("B", None)])
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.10)]),
                        ("A-heavy", [("node-0", 1, 0.90)]),
                    ],
                    False,
                ),
                (
                    "B",
                    [
                        ("B-light", [("node-0", 2, 0.10)]),
                        ("B-heavy", [("node-0", 3, 0.90)]),
                    ],
                    False,
                ),
            ],
        )

        solution = _autoscale_with_intents(scheduler, state, {"A": -1, "B": -1})

        a_deleted = [worker.id for worker in solution.stages[0].deleted_workers]
        b_deleted = [worker.id for worker in solution.stages[1].deleted_workers]
        # Each stage independently picks the lowest-fraction worker; neither shrink
        # interferes with the other's bucket boundary.
        assert a_deleted == ["A-light"]
        assert b_deleted == ["B-light"]

    def test_idempotent_autoscale_two_calls_same_input_same_deletions(self) -> None:
        """Calling autoscale twice with the same input produces the same deletion set.

        This is a weaker form of multi-cycle convergence: it pins that the helpers
        are deterministic given identical inputs even when scheduler state advances
        (e.g., ``_cycle_counter`` increments). The deletions returned in the
        Solution must match exactly across the two calls.
        """
        scheduler, _ = _gpu_problem([("A", None)])
        state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-w0", [("node-0", 0, 0.15)]),
                        ("A-w1", [("node-0", 1, 0.45)]),
                        ("A-w2", [("node-0", 2, 0.75)]),
                    ],
                    False,
                ),
            ],
        )

        sol1 = _autoscale_with_intents(scheduler, state, {"A": -2})
        sol2 = _autoscale_with_intents(scheduler, state, {"A": -2})

        first_set = sorted(worker.id for worker in sol1.stages[0].deleted_workers)
        second_set = sorted(worker.id for worker in sol2.stages[0].deleted_workers)
        assert first_set == second_set
        assert first_set == ["A-w0", "A-w1"]

    def test_intent_sign_flips_across_cycles_without_thrashing_consolidation(self) -> None:
        """Negative intent (cycle 1) deletes; non-negative intent (cycle 2) is a no-op for Phase D.

        Pins that a sign flip on the intent does not retroactively re-delete the
        consolidation winners; once Phase D commits, the deletions are durable. This
        prevents an oscillating-intent operator from accidentally pumping deletions
        cycle after cycle.
        """
        scheduler, _ = _gpu_problem([("A", None)])
        cycle1_state = _gpu_problem_state(
            [
                (
                    "A",
                    [
                        ("A-light", [("node-0", 0, 0.20)]),
                        ("A-heavy", [("node-0", 1, 0.80)]),
                    ],
                    False,
                ),
            ],
        )
        cycle2_state = _gpu_problem_state(
            [
                ("A", [("A-heavy", [("node-0", 1, 0.80)])], False),
            ],
        )

        sol1 = _autoscale_with_intents(scheduler, cycle1_state, {"A": -1})
        sol2 = _autoscale_with_intents(scheduler, cycle2_state, {"A": +1})
        sol3 = _autoscale_with_intents(scheduler, cycle2_state, {"A": -1})

        cycle1_deleted = [worker.id for worker in sol1.stages[0].deleted_workers]
        cycle2_deleted = [worker.id for worker in sol2.stages[0].deleted_workers]
        cycle3_deleted = [worker.id for worker in sol3.stages[0].deleted_workers]
        # Cycle 1: consolidation wins, A-light deleted.
        assert cycle1_deleted == ["A-light"]
        # Cycle 2: positive intent; Phase D is a no-op (Phase C handles the grow).
        assert cycle2_deleted == []
        # Cycle 3: negative intent again, but floor=1 prevents any further deletion.
        assert cycle3_deleted == []


class TestBottleneckShrinkProtection:
    """Phase D refuses to shrink the engaged bottleneck on transient idle."""

    @staticmethod
    def _heterogeneous_three_stage_state() -> data_structures.ProblemState:
        """3 stages, 4 workers each: caption is the bottleneck (D=4s)."""
        return _problem_state(
            [
                ("download", ["dl-w0", "dl-w1", "dl-w2", "dl-w3"], False),
                ("caption", ["cap-w0", "cap-w1", "cap-w2", "cap-w3"], False),
                ("write", ["wr-w0", "wr-w1", "wr-w2", "wr-w3"], False),
            ]
        )

    @staticmethod
    def _engage_bottleneck(scheduler: SaturationAwareScheduler) -> None:
        """Pre-populate D_k EWMA so identify_bottleneck engages with caption as argmax."""
        scheduler._d_k_ewma = {"download": 0.5, "caption": 4.0, "write": 0.5}

    def test_engaged_bottleneck_with_negative_intent_is_not_shrunk(self) -> None:
        """Toggle ON, engaged, intent<0, no ceiling overflow -> stage NOT shrunk."""
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_shrink_protection=True,
                stage_defaults=_test_stage_config(
                    min_workers=1,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            ),
            total_cpus=16,
        )
        self._engage_bottleneck(scheduler)

        solution = _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": -2, "write": 0},
        )

        caption_stage_index = scheduler._stage_names.index("caption")
        deleted = [w.id for w in solution.stages[caption_stage_index].deleted_workers]
        assert deleted == [], "engaged bottleneck must not shrink on transient idle"

    def test_non_bottleneck_stage_still_shrinks_in_same_cycle(self) -> None:
        """The protection does not block other OVER_PROVISIONED stages from shrinking."""
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_shrink_protection=True,
                stage_defaults=_test_stage_config(
                    min_workers=1,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            ),
            total_cpus=16,
        )
        self._engage_bottleneck(scheduler)

        solution = _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": -2, "caption": -2, "write": 0},
        )

        download_stage_index = scheduler._stage_names.index("download")
        download_deleted = solution.stages[download_stage_index].deleted_workers
        assert len(download_deleted) == 2, "non-bottleneck stage should still shrink"

    def test_disabled_toggle_allows_bottleneck_shrink(self) -> None:
        """``enable_bottleneck_shrink_protection=False`` -> bottleneck stage is shrunk."""
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_shrink_protection=False,
                stage_defaults=_test_stage_config(
                    min_workers=1,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            ),
            total_cpus=16,
        )
        self._engage_bottleneck(scheduler)

        solution = _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": -2, "write": 0},
        )

        caption_stage_index = scheduler._stage_names.index("caption")
        caption_deleted = solution.stages[caption_stage_index].deleted_workers
        assert len(caption_deleted) == 2, "toggle off -> protection skipped, shrink applied"

    def test_ceiling_overflow_bypasses_protection(self) -> None:
        """``ceiling_excess > 0`` (operator just lowered max_workers) bypasses the gate."""
        # max_workers=2 forces ceiling_excess = 4 - 2 = 2 for caption.
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_shrink_protection=True,
                stage_defaults=_test_stage_config(
                    min_workers=1,
                    max_workers=2,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            ),
            total_cpus=16,
        )
        self._engage_bottleneck(scheduler)

        solution = _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": 0, "write": 0},
        )

        caption_stage_index = scheduler._stage_names.index("caption")
        caption_deleted = solution.stages[caption_stage_index].deleted_workers
        assert len(caption_deleted) == 2, "ceiling overflow forces shrink even on bottleneck"

    def test_cold_start_disengaged_allows_shrink(self) -> None:
        """No D_k data (cold start) -> bottleneck identity disengaged -> shrink proceeds."""
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_shrink_protection=True,
                stage_defaults=_test_stage_config(
                    min_workers=1,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            ),
            total_cpus=16,
        )
        # No call to _engage_bottleneck() -> default NaN seed -> not engaged.

        solution = _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": -2, "write": 0},
        )

        caption_stage_index = scheduler._stage_names.index("caption")
        caption_deleted = solution.stages[caption_stage_index].deleted_workers
        assert len(caption_deleted) == 2, "cold-start disengaged -> shrink proceeds"


class TestBottleneckShrinkProtectionLogDebounce:
    """Phase D bottleneck-protection INFO log fires once per protection event.

    Steady-state heterogeneous workloads keep the bottleneck stage in
    sustained protection across many consecutive cycles. Logging the
    skip on every cycle would spam operator logs at the autoscale
    cadence (one line per ``interval_s``); the contract is to log on
    entry into the protection set and stay silent until the stage
    leaves and re-enters the set on a future cycle.
    """

    @staticmethod
    def _heterogeneous_three_stage_state() -> data_structures.ProblemState:
        """3 stages, 4 workers each: caption is the bottleneck (D=4s)."""
        return _problem_state(
            [
                ("download", ["dl-w0", "dl-w1", "dl-w2", "dl-w3"], False),
                ("caption", ["cap-w0", "cap-w1", "cap-w2", "cap-w3"], False),
                ("write", ["wr-w0", "wr-w1", "wr-w2", "wr-w3"], False),
            ]
        )

    @staticmethod
    def _engage_bottleneck(scheduler: SaturationAwareScheduler) -> None:
        """Pre-populate D_k EWMA so identify_bottleneck engages with caption as argmax."""
        scheduler._d_k_ewma = {"download": 0.5, "caption": 4.0, "write": 0.5}

    def _scheduler_with_protection_on(self) -> SaturationAwareScheduler:
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_shrink_protection=True,
                stage_defaults=_test_stage_config(
                    min_workers=1,
                    max_scale_down_fraction_per_cycle=1.0,
                ),
            ),
            total_cpus=16,
        )
        return scheduler

    def test_sustained_protection_logs_only_once_across_many_cycles(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """5 consecutive protected cycles emit exactly 1 INFO log."""
        scheduler = self._scheduler_with_protection_on()
        self._engage_bottleneck(scheduler)

        for _ in range(5):
            self._engage_bottleneck(scheduler)
            _autoscale_with_intents(
                scheduler,
                self._heterogeneous_three_stage_state(),
                {"download": 0, "caption": -2, "write": 0},
            )

        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        protection_logs = [m for m in infos if "phase D bottleneck shrink protected" in m]
        assert len(protection_logs) == 1, (
            f"expected exactly one protection log across 5 sustained cycles; got: {protection_logs}"
        )

    def test_re_entry_after_disengagement_re_arms_log(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Protected -> not protected -> protected fires the INFO log twice (once per entry)."""
        scheduler = self._scheduler_with_protection_on()

        self._engage_bottleneck(scheduler)
        _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": -2, "write": 0},
        )

        # Disengage: bottleneck identity engaged but intent flips to 0 ->
        # protection branch does NOT fire; ledger drops 'caption'.
        self._engage_bottleneck(scheduler)
        _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": 0, "write": 0},
        )

        # Re-engage: intent negative again -> ledger re-arms the log.
        self._engage_bottleneck(scheduler)
        _autoscale_with_intents(
            scheduler,
            self._heterogeneous_three_stage_state(),
            {"download": 0, "caption": -2, "write": 0},
        )

        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        protection_logs = [m for m in infos if "phase D bottleneck shrink protected" in m]
        assert len(protection_logs) == 2, (
            f"expected exactly two protection logs across two protection events; got: {protection_logs}"
        )

    def test_setup_resets_protection_ledger(self) -> None:
        """``setup()`` clears ``_bottleneck_protected_stages_logged``."""
        scheduler = self._scheduler_with_protection_on()
        scheduler._bottleneck_protected_stages_logged = {"caption", "stale_stage"}

        scheduler.setup(scheduler._problem)  # type: ignore[arg-type]

        assert scheduler._bottleneck_protected_stages_logged == set(), (
            "setup() must clear the protection ledger so a recycled scheduler emits a fresh log"
        )


class TestPhaseCDDefensiveAssertion:
    """Phase C / Phase D refuse to run before the bottleneck calc block populates ``_last_bottleneck_meta``."""

    def test_phase_c_raises_on_unpopulated_meta(self) -> None:
        """Calling ``_run_phase_c_grow`` directly with a None meta raises a clear ``RuntimeError``."""
        scheduler, problem_obj = _problem(
            [("A", None)],
            cfg=SaturationAwareConfig(floor_stuck_grace_cycles=0),
            total_cpus=8,
        )
        scheduler._last_bottleneck_meta = None
        ps = _problem_state([("A", ["A-w0"], False)])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem_obj, ps)

        with pytest.raises(RuntimeError, match="bottleneck calc block must run before phase C"):
            scheduler._run_phase_c_grow(ctx, ps)

    def test_phase_d_raises_on_unpopulated_meta(self) -> None:
        """Calling ``_run_phase_d_shrink`` directly with a None meta raises a clear ``RuntimeError``."""
        scheduler, problem_obj = _problem(
            [("A", None)],
            cfg=SaturationAwareConfig(floor_stuck_grace_cycles=0),
            total_cpus=8,
        )
        scheduler._last_bottleneck_meta = None
        ps = _problem_state([("A", ["A-w0"], False)])
        ctx = data_structures.AutoscalePlanContext.from_problem_state(problem_obj, ps)

        with pytest.raises(RuntimeError, match="bottleneck calc block must run before phase D"):
            scheduler._run_phase_d_shrink(ctx, ps)


class TestBottleneckEngagementLogToggleGate:
    """Engagement INFO log is silenced when both decision toggles are disabled.

    The bottleneck calc block keeps running to maintain warm
    ``_d_k_ewma`` state for re-enable; only the operator-facing log
    is gated. Disabling both toggles must not introduce a new INFO
    line.
    """

    @staticmethod
    def _three_stage_state() -> data_structures.ProblemState:
        return _problem_state(
            [
                ("download", ["dl-w0", "dl-w1"], False),
                ("caption", ["cap-w0", "cap-w1"], False),
                ("write", ["wr-w0", "wr-w1"], False),
            ]
        )

    @staticmethod
    def _set_disengaged_homogeneous(scheduler: SaturationAwareScheduler) -> None:
        scheduler._d_k_ewma = {"download": 1.0, "caption": 1.0, "write": 1.0}

    @staticmethod
    def _set_engaged_heterogeneous(scheduler: SaturationAwareScheduler) -> None:
        scheduler._d_k_ewma = {"download": 0.5, "caption": 4.0, "write": 0.5}

    def _drive_engagement_transition(self, scheduler: SaturationAwareScheduler) -> None:
        """Run two cycles: cycle 1 disengaged, cycle 2 engaged.

        With ``persistence_cycles=1`` the helper would log on cycle 2
        if it were called, because that is a genuine ``False -> True``
        flip on the post-seed path.
        """
        # Cycle 1: homogeneous D_k -> identity disengaged. Seeds last_announced=False.
        self._set_disengaged_homogeneous(scheduler)
        _autoscale_with_intents(
            scheduler,
            self._three_stage_state(),
            {"download": 0, "caption": 0, "write": 0},
        )
        # Cycle 2: heterogeneous D_k -> identity engaged. Post-seed flip would log.
        self._set_engaged_heterogeneous(scheduler)
        _autoscale_with_intents(
            scheduler,
            self._three_stage_state(),
            {"download": 0, "caption": 0, "write": 0},
        )

    def test_both_toggles_off_silences_engagement_log(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Disabling both decision toggles must not introduce a new operator log."""
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_priority_growth=False,
                enable_bottleneck_shrink_protection=False,
                bottleneck_engagement_persistence_cycles=1,
                stage_defaults=_test_stage_config(min_workers=1),
            ),
            total_cpus=8,
        )

        self._drive_engagement_transition(scheduler)

        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        engagement_logs = [m for m in infos if "bottleneck-priority decisions" in m]
        assert engagement_logs == [], (
            f"engagement log must be silent when both decision toggles are off; got: {engagement_logs}"
        )

    def test_one_toggle_on_keeps_engagement_log(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Enabling only one decision toggle still surfaces the engagement log."""
        scheduler, _problem_obj = _problem(
            [("download", None), ("caption", None), ("write", None)],
            cfg=SaturationAwareConfig(
                floor_stuck_grace_cycles=0,
                enable_bottleneck_priority_growth=True,
                enable_bottleneck_shrink_protection=False,
                bottleneck_engagement_persistence_cycles=1,
                stage_defaults=_test_stage_config(min_workers=1),
            ),
            total_cpus=8,
        )

        self._drive_engagement_transition(scheduler)

        infos = [r.getMessage() for r in loguru_caplog.records if r.levelname == "INFO"]
        engagement_logs = [m for m in infos if "bottleneck-priority decisions engaged" in m]
        assert len(engagement_logs) == 1, (
            f"engagement log must fire once when at least one decision toggle is on; got: {engagement_logs}"
        )
