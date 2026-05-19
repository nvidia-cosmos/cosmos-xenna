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

"""Python wrappers around the Rust autoscaling data structures."""

from typing import Self

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    autoscale_plan_context as rust_apc,
)
from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    data_structures as rust,
)
from cosmos_xenna.pipelines.private import resources


class ProblemStage:
    """Static autoscaling description of one pipeline stage.

    This is the per-stage entry inside ``Problem``. It describes what a
    worker for this stage costs and how the stage batches work; it does
    not describe the workers currently running.

    """

    def __init__(
        self,
        name: str,
        stage_batch_size: int,
        worker_shape: resources.WorkerShape,
        requested_num_workers: int | None,
        over_provision_factor: float | None,
    ) -> None:
        self._r = rust.ProblemStage(
            name, stage_batch_size, worker_shape.rust, requested_num_workers, over_provision_factor
        )

    @property
    def rust(self) -> rust.ProblemStage:
        """Underlying Rust ``ProblemStage`` object."""
        return self._r


class Problem:
    """Static autoscaling input for one pipeline.

    ``Problem`` answers "what could this pipeline allocate?": total
    cluster capacity plus the ordered ``ProblemStage`` definitions.
    Autoscalers combine this static model with ``ProblemState`` to
    produce an ordered ``Solution``.

    """

    def __init__(self, cluster_resources: resources.ClusterResources, stages: list[ProblemStage]) -> None:
        self._r = rust.Problem(cluster_resources.to_rust(), [s.rust for s in stages])

    @property
    def rust(self) -> rust.Problem:
        return self._r


class ProblemWorkerGroupState:
    """Runtime snapshot of one worker group assigned to a stage.

    Holds the worker group's stable id, resource allocations, and the
    optional ``num_used_slots`` saturation signal. The stage name is
    supplied later when converting this snapshot back into a concrete
    ``WorkerGroup`` for actor-pool operations.
    """

    @classmethod
    def make(
        cls,
        id: str,
        worker_resources: list[resources.WorkerResourcesInternal],
        *,
        num_used_slots: int = 0,
    ) -> Self:
        """Build a worker-group snapshot from Python primitives.

        Args:
            id: Stable identifier for the worker group.
            worker_resources: One entry per resource allocation owned by
                this worker group (typically one for CPU/GPU workers,
                multiple for SPMD groups). The parameter is named
                ``worker_resources`` rather than ``resources`` so it does
                not shadow the module-level ``resources`` import inside
                the method body.
            num_used_slots: Optional task-slot occupancy signal. Defaults
                to 0 (no signal); populated by ``streaming.py`` from the
                live ``ActorPool`` per-actor data.

        Returns:
            A ``ProblemWorkerGroupState`` wrapping a freshly-constructed
            Rust ``rust.ProblemWorkerGroupState``.

        """
        return cls(
            rust.ProblemWorkerGroupState(
                id,
                [x.to_rust() for x in worker_resources],
                num_used_slots,
            )
        )

    def __init__(self, rust_problem_worker_state: rust.ProblemWorkerGroupState) -> None:
        self._r = rust_problem_worker_state

    def to_worker_group(self, stage_name: str) -> resources.WorkerGroup:
        """Convert this snapshot into a concrete ``WorkerGroup`` for ``stage_name``."""
        return resources.WorkerGroup(self._r.to_worker_group(stage_name))

    @property
    def id(self) -> str:
        """Stable identifier for this worker group."""
        return str(self._r.id)

    @property
    def resources(self) -> list[resources.WorkerResourcesInternal]:
        """Resource allocations owned by this worker group."""
        return [resources.WorkerResourcesInternal.from_rust(x) for x in self._r.resources]

    @property
    def num_used_slots(self) -> int:
        """Number of task slots currently occupied on this worker."""
        return int(self._r.num_used_slots)

    @property
    def rust(self) -> rust.ProblemWorkerGroupState:
        """Underlying Rust ``ProblemWorkerGroupState`` object."""
        return self._r


class ProblemStageState:
    """Runtime autoscaling snapshot for one pipeline stage.

    This is the per-stage entry inside ``ProblemState``. It records the
    workers currently assigned to the stage, the active slots-per-worker
    setting, whether the stage has finished, and optional slot/queue
    signals sampled at the same instant as ``worker_groups``.

    The slot-signal kwargs (``num_used_slots``, ``num_empty_slots``,
    ``input_queue_depth``) default to 0, so existing call sites that pass
    only the four positional arguments continue to work unchanged.
    Consumers that do not populate these fields treat the default as
    "no signal".

    Args:
        stage_name: Pipeline stage identifier; positional aligns with
            ``ProblemStage`` in ``Problem``.
        workers: Snapshot of ``ProblemWorkerGroupState`` objects currently
            assigned to this stage.
        slots_per_worker: Active per-worker slot count.
        is_finished: True when the stage has drained and will not receive
            further input.
        num_used_slots: Number of currently-occupied task slots across all
            workers in the stage at sample time.
        num_empty_slots: Number of currently-free task slots across all
            workers at sample time. ``num_used_slots + num_empty_slots``
            is the total in-stage slot capacity at sample time.
        input_queue_depth: Number of pre-batch tasks queued upstream of
            this stage at sample time.

    """

    def __init__(
        self,
        stage_name: str,
        workers: list[ProblemWorkerGroupState],
        slots_per_worker: int,
        is_finished: bool,
        *,
        num_used_slots: int = 0,
        num_empty_slots: int = 0,
        input_queue_depth: int = 0,
    ) -> None:
        self._r = rust.ProblemStageState(
            stage_name,
            [w.rust for w in workers],
            slots_per_worker,
            is_finished,
            num_used_slots,
            num_empty_slots,
            input_queue_depth,
        )

    @property
    def rust(self) -> rust.ProblemStageState:
        return self._r


class ProblemState:
    """Runtime autoscaling snapshot for the whole pipeline.

    ``ProblemState`` answers "what is running right now?": one ordered
    ``ProblemStageState`` per pipeline stage. Autoscalers compare it
    with the static ``Problem`` to decide what a ``Solution`` should
    add, remove, or leave unchanged.

    """

    def __init__(self, stages: list[ProblemStageState]) -> None:
        self._r = rust.ProblemState([s.rust for s in stages])

    @property
    def rust(self) -> rust.ProblemState:
        return self._r


class TaskMeasurement:
    """Runtime timing sample for one completed task.

    Measurements are historical throughput inputs for autoscalers that
    estimate processing rates from task durations and output counts.

    """

    def __init__(self, start_time: float, end_time: float, num_returns: int) -> None:
        self._r = rust.TaskMeasurement(start_time, end_time, num_returns)

    @property
    def rust(self) -> rust.TaskMeasurement:
        return self._r


class StageMeasurements:
    """Runtime timing samples for one pipeline stage.

    Groups the task-level measurements observed for a single stage
    during an autoscaling interval.

    """

    def __init__(self, task_measurements: list[TaskMeasurement]) -> None:
        self._r = rust.StageMeasurements([t.rust for t in task_measurements])

    @property
    def rust(self) -> rust.StageMeasurements:
        return self._r


class Measurements:
    """Runtime timing snapshot for the whole pipeline.

    Carries the measurement timestamp and one ordered ``StageMeasurements``
    entry per pipeline stage. Throughput-based autoscalers use it to
    update their rate estimates between allocation decisions.

    """

    def __init__(self, time: float, stage_measurements: list[StageMeasurements]) -> None:
        self._r = rust.Measurements(time, [s.rust for s in stage_measurements])

    @property
    def rust(self) -> rust.Measurements:
        return self._r


class StageSolution:
    """Autoscale plan for one pipeline stage.

    This is the per-stage entry inside ``Solution``. It describes what
    to apply to one stage in this autoscale cycle: update
    ``slots_per_worker``, create ``new_workers``, and delete
    ``deleted_workers``. Stage identity is positional; the containing
    ``Solution`` aligns each entry with the corresponding pipeline
    stage.

    Attributes:
        deleted_workers: Worker groups to remove from this stage.
        new_workers: Worker groups to add to this stage.
        slots_per_worker: Task slots to configure on each worker.

    """

    def __init__(self, rust_stage_solution: rust.StageSolution) -> None:
        self._r = rust_stage_solution

    @classmethod
    def make(
        cls,
        slots_per_worker: int,
        new_workers: list[ProblemWorkerGroupState] | None = None,
        deleted_workers: list[ProblemWorkerGroupState] | None = None,
    ) -> Self:
        """Build a StageSolution from Python primitives.

        Pure-Python schedulers use this factory to assemble per-stage
        solutions without going through the Rust autoscaler. Mirrors
        ``ProblemWorkerGroupState.make``.

        Args:
            slots_per_worker: Number of task slots to allocate per worker.
            new_workers: Workers to add to the stage. Empty list when omitted.
            deleted_workers: Workers to remove from the stage. Empty list
                when omitted.

        Returns:
            A ``StageSolution`` wrapping a freshly-constructed Rust
            ``rust.StageSolution`` populated with the given workers.

        """
        rust_ss = rust.StageSolution(slots_per_worker)
        if new_workers:
            rust_ss.new_workers = [w.rust for w in new_workers]
        if deleted_workers:
            rust_ss.deleted_workers = [w.rust for w in deleted_workers]
        return cls(rust_ss)

    @property
    def deleted_workers(self) -> list[ProblemWorkerGroupState]:
        return [ProblemWorkerGroupState(w) for w in self._r.deleted_workers]

    @property
    def new_workers(self) -> list[ProblemWorkerGroupState]:
        return [ProblemWorkerGroupState(w) for w in self._r.new_workers]

    @property
    def slots_per_worker(self) -> int:
        """Task slots configured on each worker for this stage."""
        return int(self._r.slots_per_worker)

    @property
    def rust(self) -> rust.StageSolution:
        return self._r


class Solution:
    """Autoscale plan for the whole pipeline.

    ``Solution`` answers "what should change this cycle?": one ordered
    ``StageSolution`` per pipeline stage. The stage name is not stored
    in each entry; callers apply each entry to the stage at the same
    list index.

    Attributes:
        stages: Ordered per-stage allocation plans.

    """

    def __init__(self, rust_solution: rust.Solution) -> None:
        self._r = rust_solution

    @classmethod
    def make(cls, stages: list[StageSolution] | None = None) -> Self:
        """Build a Solution from a list of StageSolutions.

        Pure-Python schedulers use this factory to assemble the final
        autoscale result. Mirrors ``ProblemWorkerGroupState.make`` and
        ``StageSolution.make``.

        Args:
            stages: Per-stage solutions in the same order as the
                pipeline's stages. ``None`` produces an empty solution
                (no per-stage entries).

        Returns:
            A ``Solution`` wrapping a freshly-constructed Rust
            ``rust.Solution`` populated with the given stages.

        """
        rust_sol = rust.Solution()
        if stages:
            rust_sol.stages = [s.rust for s in stages]
        return cls(rust_sol)

    @property
    def stages(self) -> list[StageSolution]:
        return [StageSolution(s) for s in self._r.stages]

    @property
    def rust(self) -> rust.Solution:
        return self._r


class AutoscalePlanContext:
    """Per-cycle autoscale planning context.

    Owns a working copy of the cluster snapshot seeded with the current
    worker placements from ``ProblemState``, plus per-stage maps of
    pending adds and pending removes that the caller stages during one
    autoscale cycle. Construct one instance per cycle via
    ``from_problem_state``; subsequent cycles must construct a fresh
    context.

    Methods:
        ``from_problem_state``: seeded construction.
        ``try_add_worker``: stage one worker add via Fragmentation
            Gradient Descent on the working snapshot.
        ``try_remove_worker``: stage one worker removal by id on the
            working snapshot.
        ``into_solution``: drains staged adds/removes into a ``Solution``.
        ``pending_add_count`` / ``pending_remove_count``: read-only
            counts of staged adds / removes per stage, intended for
            planning-time invariant checks.
        ``num_stages``: number of stages this context tracks.
        ``worker_ages`` / ``worker_age``: per-worker age tracking.
            Workers placed mid-cycle via ``try_add_worker`` start at
            age 0; seeded workers carry the age value supplied to the
            constructor (default 0). Callers that need cross-cycle age
            awareness -- e.g. for picking the youngest eligible donor
            across stages -- read these and persist them, incremented,
            into the next cycle.

    """

    @classmethod
    def from_problem_state(
        cls,
        problem: Problem,
        state: ProblemState,
        *,
        worker_ages: dict[str, int] | None = None,
    ) -> Self:
        """Build a planning context seeded with the current cluster state.

        Clones ``problem.cluster_resources`` and pre-allocates every worker
        currently in ``state``, so subsequent ``try_add_worker`` /
        ``try_remove_worker`` calls respect existing placements.

        Args:
            problem: Static autoscaling input (cluster shape + per-stage
                shape definitions).
            state: Runtime snapshot (per-stage current workers, slots,
                finished flag).
            worker_ages: Optional previous-cycle worker-age snapshot,
                mapping worker id to age in autoscale cycles. Each
                value should already be incremented for the new cycle
                by the caller. Workers present in ``state`` but absent
                from this map default to age 0 (cold-start). Workers
                in this map but not in ``state`` are silently dropped
                (the worker died between cycles). When ``None`` (the
                default; the cold-start cycle, or callers that do not
                maintain cross-cycle age state), every seeded worker
                starts at age 0.

        Returns:
            A fresh ``AutoscalePlanContext`` ready to be mutated by the
            planning methods.

        Raises:
            RuntimeError: If seeding fails because a current worker
                cannot be allocated on the cluster (indicates a
                corrupted snapshot).
            ValueError: If ``problem.stages`` and ``state.stages`` do
                not have the same length.

        """
        return cls(
            rust_apc.AutoscalePlanContext(
                problem.rust,
                state.rust,
                worker_ages=worker_ages,
            )
        )

    def __init__(self, rust_context: rust_apc.AutoscalePlanContext) -> None:
        self._r = rust_context

    @property
    def rust(self) -> rust_apc.AutoscalePlanContext:
        """Underlying Rust planning-context object."""
        return self._r

    def try_add_worker(self, stage_index: int) -> ProblemWorkerGroupState | None:
        """Stage one worker add for a stage via Fragmentation Gradient Descent.

        Mutates the context's working cluster snapshot to reflect the new
        allocation. On a fresh placement, the new worker is appended to
        the stage's ``pending_adds`` list. On a reuse (the FGD search
        chose to revive a pending-remove worker rather than place a
        fresh one), the matching entry is popped from the stage's
        ``pending_removes`` list and ``pending_adds`` is left untouched
        (the worker was never structurally removed; the staged remove
        is simply un-staged).

        Args:
            stage_index: Position of the stage in the problem (and in
                the context's stages list).

        Returns:
            The placed (or reused) worker, or ``None`` when no
            placement exists on the working cluster.

        Raises:
            IndexError: ``stage_index >= num_stages()``.
            RuntimeError: Underlying cluster allocation failed despite
                the planner reporting a feasible placement (would
                indicate a corrupted snapshot or planner bug), OR the
                context has already been drained by ``into_solution``
                (any further staging is a caller bug).

        """
        worker = self._r.try_add_worker(stage_index)
        if worker is None:
            return None
        return ProblemWorkerGroupState(worker)

    def pending_add_count(self, stage_index: int) -> int:
        """Number of workers staged for addition on ``stage_index`` so far.

        Returns 0 for stages that have not been touched in this cycle.
        After ``into_solution`` drains the staged plan, returns 0 even
        for stages that previously had pending adds. Use this as a
        planning-time invariant check, not a post-finalization audit.

        Args:
            stage_index: Position of the stage in the problem.

        Raises:
            IndexError: ``stage_index >= num_stages()``.

        """
        return int(self._r.pending_add_count(stage_index))

    def pending_remove_count(self, stage_index: int) -> int:
        """Number of workers staged for removal on ``stage_index`` so far.

        Returns 0 for stages that have not been touched in this cycle.
        After ``into_solution`` drains the staged plan, returns 0 even
        for stages that previously had pending removes. Use this as a
        planning-time invariant check, not a post-finalization audit.

        Args:
            stage_index: Position of the stage in the problem.

        Raises:
            IndexError: ``stage_index >= num_stages()``.

        """
        return int(self._r.pending_remove_count(stage_index))

    def try_remove_worker(self, stage_index: int, worker_id: str) -> bool:
        """Stage removal of an existing worker.

        Mutates the context's working cluster snapshot to release the
        removed worker's resources and appends the worker to the stage's
        pending-removes list. A later ``try_add_worker`` for the same
        shape can reuse that placement and cancel the pending remove.

        Args:
            stage_index: Position of the stage in the problem.
            worker_id: Stable id of the worker to remove.

        Returns:
            True when the worker was found and staged for removal;
            False when ``worker_id`` is not present in that stage's
            current planning snapshot.

        Raises:
            IndexError: ``stage_index >= num_stages()``.
            RuntimeError: Underlying cluster release failed despite the
                worker being present in the planning snapshot, OR the
                context has already been drained by ``into_solution``
                (any further staging is a caller bug).

        """
        return bool(self._r.try_remove_worker(stage_index, worker_id))

    def into_solution(self) -> Solution:
        """Build a ``Solution`` from the staged plan.

        Produces one ``StageSolution`` per stage, ordered by stage
        position. ``new_workers`` is the list of placements staged via
        ``try_add_worker``; ``deleted_workers`` is the list of removals
        staged via ``try_remove_worker``. Workers reused inside the same
        cycle (a remove cancelled by a later add, or an add cancelled
        by a later remove) appear in NEITHER list, mirroring the
        live-set semantics: they were never structurally swapped.

        ``slots_per_worker`` is preserved from the input
        ``ProblemState`` - the planner does not change it.

        Drains the per-stage pending lists in place and marks the
        context as terminal. A second call on the same context returns
        a ``Solution`` whose per-stage lists are all empty (the drain
        is idempotent), but any further ``try_add_worker`` /
        ``try_remove_worker`` call on a drained context raises
        ``RuntimeError``. Callers must construct a fresh
        ``AutoscalePlanContext`` for each autoscale cycle rather than
        re-using one.

        """
        return Solution(self._r.into_solution())

    def num_stages(self) -> int:
        """Number of stages this context tracks.

        Useful for invariant checks (callers can confirm
        ``len(stage_solutions) == ctx.num_stages()``) and for tests
        that need to verify the seeding round-tripped the input shape.
        """
        return int(self._r.num_stages())

    def worker_ages(self) -> dict[str, int]:
        """Snapshot of the per-worker age map.

        Keys are every worker id present in this cycle's planning
        snapshot (initial seed plus mid-cycle additions); values are
        each worker's age in autoscale cycles (0 for workers placed
        fresh this cycle, positive integers for workers carried over
        from previous cycles).

        Workers staged for removal are still included -- they may be
        reused via the FGD reuse path before ``into_solution`` runs --
        so callers building an age-aware index must intersect this map
        with the per-stage live-worker set to filter out
        scheduled-for-removal entries. After ``into_solution`` drains
        the per-stage pending lists, callers should further filter
        against ``solution.deleted_workers`` before persisting the map
        for the next cycle. Safe to call after ``into_solution()``
        drains the plan; the read accessors deliberately bypass the
        drained-state guard so the caller can persist the post-cycle
        age map for the next cycle.

        Returns:
            A fresh ``dict[str, int]`` (cloned from the underlying Rust
            map). Caller mutations on the returned dict do not affect
            the context's internal state.

        """
        ages: dict[str, int] = self._r.worker_ages()
        return ages

    def worker_age(self, worker_id: str) -> int | None:
        """Age in autoscale cycles for ``worker_id``.

        Cheap O(1) lookup against the same map exposed in bulk by
        :meth:`worker_ages`. Use this when you have a specific worker id
        and want its age without cloning the whole map (for example,
        comparing donor candidates one at a time inside a Python loop).
        Safe to call after ``into_solution()`` for the same reason as
        :meth:`worker_ages`.

        Args:
            worker_id: Stable id of the worker to inspect.

        Returns:
            The worker's age in autoscale cycles, or ``None`` if the
            worker is not present in the planning snapshot (was never
            seeded, was dropped because the seed was stale, or had its
            age entry removed by a cancel-pending-add).

        """
        age = self._r.worker_age(worker_id)
        return None if age is None else int(age)

    def worker_ids_by_stage(self) -> list[list[str]]:
        """Snapshot of the live worker ids per stage.

        Returns one entry per stage in problem order; each entry is the
        list of worker ids currently held by that stage in the planner's
        working snapshot. Reflects every successful ``try_add_worker``
        and ``try_remove_worker`` issued so far in this cycle. Workers
        staged for removal are excluded (they have moved to
        ``pending_removes``); freshly added workers are included.

        Per-stage ids are sorted lexicographically so the output is
        deterministic across calls. Safe to call after
        ``into_solution()`` for the same reason as :meth:`worker_ages`;
        the returned list is freshly allocated and caller mutations do
        not leak into the planner.

        Returns:
            A list of ``len(problem.stages)`` lists; entry ``i`` is the
            ids of the live workers in stage ``i``.

        """
        rows: list[list[str]] = self._r.worker_ids_by_stage()
        return rows
