# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Code for managing a pool of Ray actors. This is used for both Batch and Streaming pipelines."""

from __future__ import annotations

import collections
import copy
import statistics
import time
import typing
from typing import Generic, Optional

import attrs
import ray
import ray.exceptions
import ray.runtime_env
import ray.util.scheduling_strategies
from loguru import logger
from ray import ObjectRef
from ray.actor import ActorHandle
from ray.util.metrics import Counter

from cosmos_xenna.ray_utils import allocator, monitoring, resources, stage, stage_worker
from cosmos_xenna.utils import stats, timing
from cosmos_xenna.utils.verbosity import VerbosityLevel

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_RATE_ESTIMATE_LOOKBACK_S = 60 * 10
_MAX_QUEUED_TASK_METADATAS = 10000
_VERBOSE = False


@attrs.define
class _ActorMetadata:
    worker: resources.Worker


@attrs.define
class _SlotData(Generic[V]):
    task: Task[V]
    # when this task was scheduled to run on the actor.
    scheduled_time: float
    # We use dynamic number of returns, so this is a ref which contains more refs when ray.get() is called.
    # See https://docs.ray.io/en/latest/ray-core/tasks/generators.html#id2
    object_ref: ObjectRef


@attrs.define
class _Slot(Generic[V]):
    task: _SlotData | None

    @property
    def has_task(self) -> bool:
        return self.task is not None

    @property
    def get_task(self) -> _SlotData:
        assert self.task is not None
        return self.task

    def clear_task(self) -> None:
        self.task = None


@attrs.define
class _ReadyActor(Generic[V]):
    """Class used to keep track of and interact with actors in the "ready" state.

    An actor reaches this state after successfully completing both `setup_on_node` (if applicable)
    and `setup`. It is now ready to process tasks via `process_data`.
    """

    metadata: _ActorMetadata
    # A ray reference to the actor
    actor_ref: ActorHandle
    # Tracking the running time of an actor.
    start_time: float
    # A slot which can be filled with tasks.
    # Each actor can have multiple slots. Only one task will ever be processing at a time,
    # but may have additional slots are spots for assigned, but not presently running tasks.
    slots: collections.deque[_Slot] = attrs.field(factory=collections.deque)
    # Class used to estimate the processing rate of this actor
    rate_estimator: timing.RateEstimatorDuration = attrs.field(
        factory=lambda: timing.RateEstimatorDuration(_RATE_ESTIMATE_LOOKBACK_S)
    )

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    @property
    def num_used_slots(self) -> int:
        return len([x for x in self.slots if x.has_task])

    @property
    def num_empty_slots(self) -> int:
        return self.num_slots - self.num_used_slots

    @property
    def is_running(self) -> bool:
        return self.num_used_slots > 0

    @property
    def used_slots(self) -> list[_Slot]:
        return [x for x in self.slots if x.has_task]

    @property
    def idle_slots(self) -> list[_Slot]:
        return [x for x in self.slots if not x.has_task]

    def kill(self) -> None:
        ray.kill(self.actor_ref)

    def maybe_resize_num_slots_per_actor(self, new_slots_per_actor: int) -> None:
        if len(self.slots) == new_slots_per_actor:
            return
        if len(self.slots) < new_slots_per_actor:
            self.slots.append(_Slot(None))
        else:  # len(slots) > new_num_slots:
            # TODO: Implement this. I think it's actually pretty hard? We need to clear running tasks from an
            # actor. Would need to be careful not to leak these.
            raise RuntimeError("Decreasing the number of slots per actor is not supported yet.")

    def to_stats(self) -> monitoring.ReadyActorStats:
        return monitoring.ReadyActorStats(
            self.metadata.worker.id,
            self.metadata.worker.allocation,
            self.rate_estimator.maybe_get_rate(),
            self.num_used_slots,
            self.num_slots,
        )


@attrs.define
class _PendingNodeActor:
    """An actor which is in the process of being set up on a specific node.

    This actor is the designated actor responsible for running the potentially expensive
    `setup_on_node` method for a given Ray node. Only one actor per node will be in this
    state at any time. Other actors scheduled for the same node while this setup is
    running will be placed in the `_actors_waiting_for_node_setup` state.
    """

    metadata: _ActorMetadata
    # A ray reference to the actor
    actor_ref: ActorHandle
    # A reference to the "setup_on_node" call we asked ray to call when we started this worker up.
    node_setup_call_ref: ObjectRef[str]

    def kill(self) -> None:
        ray.kill(self.actor_ref)


@attrs.define
class _ActorWaitingForNodeSetup:
    """An actor which is waiting for the node-level setup (`setup_on_node`) to complete.

    These actors have been scheduled to a node where another actor (`_PendingNodeActor`)
    is already running `setup_on_node`. They wait until that setup finishes before
    proceeding to their own `setup` method (moving to the `_PendingActor` state).
    """

    metadata: _ActorMetadata
    # A ray reference to the actor
    actor_ref: ActorHandle

    def kill(self) -> None:
        ray.kill(self.actor_ref)


@attrs.define
class _PendingActor:
    """An actor which is in the process of running its individual `setup` method.

    An actor enters this state after its node's `setup_on_node` has completed (or if
    `setup_on_node` was skipped because the node was already set up). Once `setup`
    completes successfully, the actor transitions to the `_ReadyActor` state.
    """

    metadata: _ActorMetadata
    # A ray reference to the actor
    actor_ref: ActorHandle
    # A reference to the "setup" call we asked ray to call when we started this worker up
    setup_call_ref: ObjectRef[str]

    def kill(self) -> None:
        ray.kill(self.actor_ref)

    def to_stats(self) -> monitoring.PendingActorStats:
        return monitoring.PendingActorStats(self.metadata.worker.id, self.metadata.worker.allocation)


@attrs.define
class Task(Generic[T]):
    """A task to run on the ActorPool."""

    task_data: list[ObjectRef[T]]
    # An optional string telling us where the data for this task resides. This can be used for location-aware
    # scheduling. e.g. scheduling on a worker on the same node as this data.
    origin_node_id: str | None


@attrs.define
class QueueParams:
    max_queued_tasks_per_actor: int


@attrs.define
class ActorIds:
    """A list of actor IDs for this pool."""

    pending: list[str]
    ready: list[str]


def is_significant_job_failure(
    total_attempts: int, failed_attempts: int, failure_threshold: float, confidence_level: float = 0.95
) -> bool:
    """
    Evaluate whether the observed job failure rate is statistically significant.

    This function uses the binomial test to determine if the observed number of failures
    is significantly higher than the expected number given the failure threshold.

    Args:
        total_attempts (int): The total number of job attempts.
        failed_attempts (int): The number of failed job attempts.
        failure_threshold (float): The threshold failure rate, expressed as a decimal.
            For example, 0.10 represents a 10% failure threshold.
        confidence_level (float, optional): The desired confidence level for the statistical test.
            Default is 0.95, which corresponds to a 95% confidence level.

    Returns:
        - A boolean indicating whether the failure rate is statistically significant
            (True if significant, False otherwise).

    Raises:
        ValueError: If input parameters are invalid (e.g., negative numbers, improper ratios).


    Notes:
        - The function uses a one-sided binomial test, as we're only interested in
          whether the failure rate is significantly higher than the threshold.
        - A small p-value (typically < 0.05) indicates strong evidence against the null hypothesis,
          suggesting that the true failure rate is higher than the threshold.
    """
    # Input validation
    if total_attempts <= 0 or failed_attempts < 0:
        raise ValueError("Attempt counts must be non-negative, and total attempts must be positive.")
    if failed_attempts > total_attempts:
        raise ValueError("Failed attempts cannot exceed total attempts.")
    if not 0 < failure_threshold < 1:
        raise ValueError("Failure threshold must be between 0 and 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    # Perform one-sided binomial test
    p_value = 1 - stats.binom_cdf(failed_attempts - 1, total_attempts, failure_threshold)

    # Determine if the result is statistically significant
    is_significant = float(p_value) < (1 - confidence_level)
    is_above_rate = (failed_attempts / total_attempts) > failure_threshold
    return is_above_rate and is_significant


class ActorPool(Generic[T, V]):
    """Manages a pool of Ray actors for a specific pipeline stage.

    This class orchestrates the lifecycle and task assignment for actors running
    code defined by a `stage.Interface`. It handles actor creation, setup (including
    node-level setup), task scheduling (with locality awareness), result collection,
    and actor scaling/deletion.

    Actor Lifecycle and States:
    1. Creation: An actor is requested via `_add_actor`.
    2. Node Setup Check:
       - If the target node is already in `_nodes_with_completed_setups`, the actor skips to step 4.
       - If another actor is already running `setup_on_node` for the target node (i.e., node in `_pending_node_actors`),
         the new actor enters the `_actors_waiting_for_node_setup` state.
       - Otherwise, this actor becomes the designated node setup actor and enters the `_pending_node_actors` state,
         initiating the `setup_on_node` call.
    3. Node Setup Execution (`_pending_node_actors` state): The actor runs `setup_on_node`.
       - On success (`_check_pending_node_setup_actors`): The node ID is added to `_nodes_with_completed_setups`.
         This actor, and any actors in `_actors_waiting_for_node_setup` for this node, transition to step 4.
       - On failure: The actor dies, potentially failing the pipeline. If other actors were waiting, one is promoted
         to retry node setup.
    4. Individual Setup (`_pending_actors` state): The actor runs its `setup` method.
       - On success (`_move_pending_actors_to_ready`): The actor transitions to step 5.
       - On failure: The actor dies. Failure handling depends on `max_setup_failure_percentage`.
    5. Ready (`_ready_actors` state): The actor is ready to process tasks. `process_data` is called when tasks
       are assigned.
    6. Deletion: Actors can be deleted via `_delete_actor` (called internally by `_adjust_actors` or `stop`).
       This handles actors in any state, returning assigned tasks from ready actors to the queue.

    Key Mechanisms:
    - Slots: Each ready actor has multiple "slots" (`_slots_per_actor`). Assigning a task to a slot triggers a
      `process_data` call. Ray's actor request queueing allows pre-fetching data for tasks in non-active slots.
    - Locality Scheduling: `_pick_actor_for_task` prefers scheduling tasks on actors located on the same node
      as the task's input data (`origin_node_id`).
    - Backpressure: For streaming pipelines (`_queue_params` is set), `should_add_tasks` limits adding new tasks
      if actors are overloaded or the output queue is full.
    """

    def __init__(
        self,
        worker_allocator: allocator.WorkerAllocator,
        stage_interface: stage.Interface,
        params: stage.Params,
        name: str,
        queue_params: QueueParams | None = None,
        verbosity_level: VerbosityLevel = VerbosityLevel.INFO,
    ):
        self._name = str(name)
        self._queue_params = queue_params
        self._stage_interface = stage_interface
        self._params = params
        self._worker_shape = params.shape
        self._allocator = worker_allocator
        self._verbosity_level = verbosity_level

        # State related to node-level setup (`setup_on_node`)
        self._nodes_with_completed_setups: set[str] = set()  # Nodes where setup_on_node succeeded.
        # Actor currently running setup_on_node for a specific node. Indexed by node ID.
        self._pending_node_actors: collections.OrderedDict[str, _PendingNodeActor] = collections.OrderedDict()
        # Actors waiting for the setup_on_node on their target node to complete. Indexed by node ID.
        self._actors_waiting_for_node_setup: collections.defaultdict[str, list[_ActorWaitingForNodeSetup]] = (
            collections.defaultdict(list)
        )

        # State related to individual actor setup (`setup`)
        # Actors running their individual `setup` method. Indexed by actor ID.
        self._pending_actors: collections.OrderedDict[str, _PendingActor] = collections.OrderedDict()

        # State for ready actors
        # Actors ready to process tasks. Indexed by actor ID.
        self._ready_actors: dict[str, _ReadyActor] = {}

        # Task queues
        self._task_queue: collections.deque[Task[T]] = collections.deque()
        self._completed_tasks: collections.deque[Task[V]] = collections.deque()

        # Internal state & stats
        self._num_null_tasks = 0
        self._num_completed_tasks = 0
        self._num_dynamically_spawned_tasks = 0
        self._slots_per_actor = params.slots_per_actor
        self._task_result_metadatas: collections.deque[stage_worker.TaskResultMetadata] = collections.deque(
            maxlen=_MAX_QUEUED_TASK_METADATAS
        )
        self._num_actors_tried_to_start = 0
        self._num_actors_failed_to_start = 0
        self._actors_to_delete: collections.deque[resources.Worker] = collections.deque()
        self._actors_to_create: collections.deque[resources.Worker] = collections.deque()
        # Timestamp of last intentional actor-restart event
        self._last_actor_restart_time = time.time()

        # Metrics
        self._metrics_stage_deserialize_count = Counter(
            "pipeline_stage_deserialize_count_total",
            description="Count of the deserialized object",
            tag_keys=("stage",),
        )
        self._metrics_stage_deserialize_size = Counter(
            "pipeline_stage_deserialize_size_total",
            description="Size of the deserialized objects",
            tag_keys=("stage",),
        )
        self._metrics_stage_deserialize_time = Counter(
            "pipeline_stage_deserialize_time_total",
            description="Time taken to deserialize the pipeline objects",
            tag_keys=("stage",),
        )
        self._metrics_stage_process_time = Counter(
            "pipeline_stage_process_time_total",
            description="Time taken to process the pipeline objects",
            tag_keys=("stage",),
        )
        self._metrics_schedule_task_count = Counter(
            "pipeline_schedule_task_count_total",
            description="Number of tasks scheduled to actors",
            tag_keys=("stage", "affinity"),
        )
        # Initialize scheduling metrics as we may want to calculate ratio
        for affinity in ["local", "remote"]:
            self._metrics_schedule_task_count.inc(
                0.01,
                tags={"stage": self._name, "affinity": affinity},
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def worker_shape(self) -> resources.WorkerShape:
        return self._worker_shape

    @property
    def num_used_slots(self) -> int:
        return sum([x.num_used_slots for x in self._ready_actors.values()], 0)

    @property
    def num_empty_slots(self) -> int:
        return sum([x.num_empty_slots for x in self._ready_actors.values()], 0)

    @property
    def completed_tasks(self) -> collections.deque[Task[V]]:
        return self._completed_tasks

    @property
    def has_work_or_completed(self) -> bool:
        return bool(self._task_queue) or self.num_used_slots > 0 or bool(self._completed_tasks)

    @property
    def num_ready_actors(self) -> int:
        return len(self._ready_actors)

    @property
    def num_pending_actors(self) -> int:
        """Total number of actors currently in setup phases (node or individual)."""
        return (
            len(self._pending_actors)
            + len(self._pending_node_actors)
            + sum(len(v) for v in self._actors_waiting_for_node_setup.values())
        )

    @property
    def num_running_actors(self) -> int:
        return sum([1 for x in self._ready_actors.values() if x.is_running])

    @property
    def num_idle_actors(self) -> int:
        return len(self._ready_actors) - self.num_running_actors

    @property
    def num_actors(self) -> int:
        """Total number of active actors across all states."""
        return self.num_ready_actors + self.num_pending_actors

    @property
    def slots_per_actor(self) -> int:
        return self._slots_per_actor

    @property
    def task_extra_data(self) -> collections.deque[stage_worker.TaskResultMetadata]:
        return self._task_result_metadatas

    def make_stats(self) -> monitoring.ActorPoolStats:
        ids = self.get_actor_ids()
        # TODO: Add counts for node setup states to ActorStats
        return monitoring.ActorPoolStats(
            self._name,
            self._worker_shape,
            monitoring.ActorStats(
                0,  # TODO: Add target_num_actors back here
                self.num_pending_actors,
                self.num_ready_actors,
                self.num_running_actors,
                self.num_idle_actors,
            ),
            monitoring.TaskStats(
                self._num_completed_tasks,
                self._num_null_tasks,
                self._num_dynamically_spawned_tasks,
                len(self._task_queue),
                len(self._completed_tasks),
            ),
            monitoring.SlotStats(self.num_used_slots, self.num_empty_slots),
            self.calc_median_rate_estimate(),
            ids.pending,
            ids.ready,
            [x.to_stats() for x in self._pending_actors.values()],  # TODO: Include node setup actors in stats
            [x.to_stats() for x in self._ready_actors.values()],
        )

    def calc_median_rate_estimate(self) -> float | None:
        rates = [x.rate_estimator.get_rate() for x in self._ready_actors.values()]
        rates = [x for x in rates if x > 0]
        if not rates:
            return None
        return float(statistics.median(rates))

    def add_actor_to_delete(self, worker: resources.Worker) -> None:
        self._actors_to_delete.append(worker)

    def add_actor_to_create(self, worker: resources.Worker) -> None:
        self._actors_to_create.append(worker)

    def set_num_slots_per_actor(self, target: int) -> None:
        if target < self._slots_per_actor:
            raise RuntimeError("Lowering slots per actor is not supported yet.")
        self._slots_per_actor = target

    def maybe_increase_slots_per_actor(self, target: int) -> None:
        if target < self._slots_per_actor:
            return
        self._slots_per_actor = target

    def add_task(self, task: Task[T]) -> None:
        self._task_queue.append(task)

    def has_free_slots(self) -> bool:
        return self.num_empty_slots >= 0

    def should_add_tasks(self) -> bool:
        """Whether this actor pool should be assigned more tasks.

        For streaming pipelines, this provides backpressure based on the number
        of tasks currently being processed or waiting in queues relative to the
        configured limit per actor.
        """
        has_free_slots = self.num_empty_slots >= 0
        if self._queue_params is None:  # No queue limits requested
            return has_free_slots
        else:
            # Calculate total tasks inflight or completed but not yet consumed
            tasks_in_system = self.num_used_slots + len(self.completed_tasks) + len(self._task_queue)
            # Calculate the capacity based on ready actors
            capacity = self.num_ready_actors * self._queue_params.max_queued_tasks_per_actor
            return has_free_slots and tasks_in_system < capacity

    def get_actor_ids(self) -> ActorIds:
        # TODO: Distinguish between pending setup and pending node setup
        pending_ids = [x.actor_ref._ray_actor_id.hex() for x in self._pending_actors.values()]  # type: ignore[attr-defined]
        pending_ids.extend(x.actor_ref._ray_actor_id.hex() for x in self._pending_node_actors.values())  # type: ignore[attr-defined]
        for waiting_list in self._actors_waiting_for_node_setup.values():
            pending_ids.extend(x.actor_ref._ray_actor_id.hex() for x in waiting_list)  # type: ignore[attr-defined]

        ready_ids = [x.actor_ref._ray_actor_id.hex() for x in self._ready_actors.values()]  # type: ignore[attr-defined]
        return ActorIds(pending_ids, ready_ids)

    def _add_actor(self, worker: resources.Worker) -> None:
        """Creates a new actor and initiates its setup process."""
        # Prepare the runtime environment with CUDA_VISIBLE_DEVICES
        gpu_ids = set()
        for gpu in worker.allocation.gpus:
            gpu_ids.add(gpu.gpu_index)
        for alloc in worker.allocation.nvdecs:
            gpu_ids.add(alloc.gpu_index)
        for alloc in worker.allocation.nvencs:  # Corrected typo here
            gpu_ids.add(alloc.gpu_index)
        env_vars = {}
        if self._params.modify_cuda_visible_devices_env_var:
            cuda_visible_devices = ",".join(str(x) for x in sorted(gpu_ids))
            env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        env_vars.update(self._params.runtime_env.extra_env_vars)
        runtime_env = copy.deepcopy(self._params.runtime_env)
        runtime_env.extra_env_vars = env_vars

        if _VERBOSE:
            logger.info(f"Runtime env for stage={self.name}:{runtime_env.format()}")
        self._allocator.add_worker(worker)

        # Ask Ray to start the actor on the specified node
        actor = stage_worker.StageWorker.options(
            max_concurrency=1000,  # High concurrency allows many queued tasks (slots)
            num_cpus=0,  # Resources are managed by the allocator, not directly by Ray options here
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=worker.allocation.node, soft=False
            ),
            runtime_env=runtime_env.to_ray_runtime_env(),
        ).remote(
            self._stage_interface,
            self._params,
            worker,
            self._verbosity_level,
        )
        actor_handle = typing.cast(ActorHandle, actor)

        # Determine the initial state based on node setup status
        node_id = worker.allocation.node
        if node_id in self._nodes_with_completed_setups:
            # Node setup already done, proceed directly to individual setup
            self._add_pending_actor(actor_handle, worker)
        elif node_id in self._pending_node_actors:
            # Node setup in progress, add to waiting list
            self._add_actor_to_waiting_list(actor_handle, worker)
        else:
            # No setup started for this node, designate this actor for node setup
            self._add_node_setup_actor(actor_handle, worker)

    def _add_actor_to_waiting_list(self, actor_handle: ActorHandle, worker: resources.Worker) -> None:
        """Adds an actor to the list waiting for node setup on its target node."""
        self._actors_waiting_for_node_setup[worker.allocation.node].append(
            _ActorWaitingForNodeSetup(
                _ActorMetadata(worker),
                actor_handle,
            )
        )

    def _add_node_setup_actor(self, actor_handle: ActorHandle, worker: resources.Worker) -> None:
        """Designates an actor to perform node setup and adds it to the pending node state."""
        # Ask Ray to call "setup_on_node" on the actor and keep a reference to this call.
        setup_call_ref = typing.cast(ObjectRef, actor_handle.setup_on_node.remote())
        self._pending_node_actors[worker.allocation.node] = _PendingNodeActor(
            _ActorMetadata(worker),
            actor_handle,
            setup_call_ref,
        )

    def _add_pending_actor(self, actor_handle: ActorHandle, worker: resources.Worker) -> None:
        """Adds an actor to the pending state to run its individual setup."""
        # Ask Ray to call "setup" on the actor and keep a reference to this call.
        setup_call_ref = typing.cast(ObjectRef, actor_handle.setup.remote())
        self._pending_actors[worker.id] = _PendingActor(
            _ActorMetadata(worker),
            actor_handle,
            setup_call_ref,
        )
        self._num_actors_tried_to_start += 1

    def _move_pending_actor_to_ready_or_handle_failure(self, actor_id: str) -> None:
        """Attempts to transition a pending actor to ready after setup completes, or handles setup failure."""
        actor = self._pending_actors.pop(actor_id)
        try:
            # Block until the setup call completes or fails
            ray.get(actor.setup_call_ref)
        except ray.exceptions.RayActorError as e:  # Catch more specific exception
            self._num_actors_failed_to_start += 1
            logger.warning(f"Actor {actor_id} failed during setup: {e}")
            # Check if the failure rate exceeds the threshold
            should_fail_pipeline = self._params.max_setup_failure_percentage is not None and is_significant_job_failure(
                self._num_actors_tried_to_start,
                self._num_actors_failed_to_start,
                self._params.max_setup_failure_percentage / 100.0,
            )
            self._allocator.delete_worker(actor.metadata.worker.id)  # Ensure worker is marked as deleted
            if should_fail_pipeline:
                logger.error(
                    f"Significant setup failure rate detected ({self._num_actors_failed_to_start}/"
                    f"{self._num_actors_tried_to_start}). Raising exception."
                )
                raise RuntimeError(f"Actor setup failed significantly for stage {self.name}.") from e
            else:
                logger.info(
                    f"Ignoring actor setup failure for {actor_id}. Current failure rate: "
                    f"{self._num_actors_failed_to_start}/{self._num_actors_tried_to_start}."
                )
                # Optionally, trigger adding a replacement actor here if desired
        except Exception as e:
            # Catch other potential exceptions during ray.get
            self._num_actors_failed_to_start += 1
            logger.error(f"Unexpected error getting setup result for actor {actor_id}: {e}")
            self._allocator.delete_worker(actor.metadata.worker.id)
            # Decide whether to raise based on threshold, similar to above
            # This might indicate a deeper Ray issue rather than just actor setup code failure
            raise RuntimeError(f"Unexpected error during actor setup for stage {self.name}.") from e
        else:
            # Setup successful, move to ready state
            if _VERBOSE:
                logger.info(f"Actor {actor_id} setup complete. Moving to ready.")
            new_actor = _ReadyActor(
                actor.metadata,
                typing.cast(ActorHandle, actor.actor_ref),
                start_time=time.time(),
            )
            # Initialize slots for the newly ready actor
            for _ in range(self._slots_per_actor):
                new_actor.slots.append(_Slot(None))
            self._ready_actors[new_actor.metadata.worker.id] = new_actor

    def _move_pending_actors_to_ready(self) -> None:
        """Checks for completed setup calls and transitions actors from pending to ready."""
        if not self._pending_actors:
            return

        # Get ObjectRefs for all pending setup calls
        setup_refs = [actor.setup_call_ref for actor in self._pending_actors.values()]
        actor_ids = list(self._pending_actors.keys())

        # Use ray.wait to check completion without blocking indefinitely
        ready_refs, _ = ray.wait(setup_refs, num_returns=len(setup_refs), timeout=0, fetch_local=False)

        if not ready_refs:
            return

        # Create a map from ObjectRef back to actor ID for quick lookup
        ref_to_id_map = {ref: actor_id for ref, actor_id in zip(setup_refs, actor_ids)}

        for ready_ref in ready_refs:
            actor_id = ref_to_id_map.get(ready_ref)
            if actor_id and actor_id in self._pending_actors:  # Check if still pending (might have been deleted)
                self._move_pending_actor_to_ready_or_handle_failure(actor_id)

    def _move_pending_node_actor_to_pending(self, node_id: str) -> None:
        """Transitions the node setup actor and waiting actors to the pending state after successful node setup."""
        node_setup_actor = self._pending_node_actors.pop(node_id)
        try:
            # Block until node setup completes or fails
            ray.get(node_setup_actor.node_setup_call_ref)
            if _VERBOSE:
                logger.info(f"Node setup successful for node {node_id} by actor {node_setup_actor.metadata.worker.id}.")
            self._nodes_with_completed_setups.add(node_id)

            # Move the node setup actor itself to pending (individual setup)
            self._add_pending_actor(node_setup_actor.actor_ref, node_setup_actor.metadata.worker)

            # Move all waiting actors for this node to pending
            waiting_actors = self._actors_waiting_for_node_setup.pop(node_id, [])
            if _VERBOSE:
                logger.info(f"Moving {len(waiting_actors)} waiting actors for node {node_id} to pending state.")
            for waiting_actor in waiting_actors:
                self._add_pending_actor(waiting_actor.actor_ref, waiting_actor.metadata.worker)

        except ray.exceptions.RayActorError as e:
            logger.error(
                f"Node setup failed for node {node_id} by actor {node_setup_actor.metadata.worker.id}. Error: {e}"
            )
            self._allocator.delete_worker(node_setup_actor.metadata.worker.id)
            # TODO: Handle cleanup and retries.
            # Promote the next waiting actor (if any) to retry node setup
            # waiting_actors = self._actors_waiting_for_node_setup.pop(node_id, [])
            # if waiting_actors:
            #     next_actor_to_setup = waiting_actors.pop(0)
            #     logger.warning(
            #         f"Promoting actor {next_actor_to_setup.metadata.worker.id} to retry node setup for {node_id}."
            #     )
            #     self._add_node_setup_actor(next_actor_to_setup.actor_ref, next_actor_to_setup.metadata.worker)
            #     # Put remaining actors back into the waiting list for the new setup attempt
            #     if waiting_actors:
            #         self._actors_waiting_for_node_setup[node_id] = waiting_actors
            # # Decide if this failure constitutes a pipeline failure (e.g., based on retries or severity)
            # # For now, we raise directly, but more sophisticated retry logic could be added.
            # # TODO: Add retry logic for node setup failures.
            raise RuntimeError(f"Node setup failed for stage {self.name} on node {node_id}.") from e
        except Exception as e:
            logger.error(
                f"Unexpected error getting node setup result for node {node_id}, "
                f"actor {node_setup_actor.metadata.worker.id}: {e}"
            )
            self._allocator.delete_worker(node_setup_actor.metadata.worker.id)
            # Handle unexpected errors similarly - potentially promote another waiter or fail
            raise RuntimeError(f"Unexpected error during node setup for stage {self.name}.") from e

    def _check_pending_node_setup_actors(self) -> None:
        """Checks for completed node setup calls and transitions actors accordingly."""
        if not self._pending_node_actors:
            return

        node_setup_refs = [actor.node_setup_call_ref for actor in self._pending_node_actors.values()]
        node_ids = list(self._pending_node_actors.keys())

        ready_refs, _ = ray.wait(node_setup_refs, num_returns=len(node_setup_refs), timeout=0, fetch_local=False)

        if not ready_refs:
            return

        ref_to_node_id_map = {ref: node_id for ref, node_id in zip(node_setup_refs, node_ids)}

        for ready_ref in ready_refs:
            node_id = ref_to_node_id_map.get(ready_ref)
            if node_id and node_id in self._pending_node_actors:  # Check if still pending node setup
                self._move_pending_node_actor_to_pending(node_id)

    def _maybe_restart_long_running_actor(self) -> None:
        """Restarts actors that have been running for too long.

        This is a simple heuristic to avoid system OOM in case there is hard-to-debug memory issues.
        """
        # disabled if worker_max_lifetime_m is 0
        if self._params.worker_max_lifetime_m == 0:
            return None
        # avoid restarting too many actors in one pool at once
        time_from_last_restart_m = (time.time() - self._last_actor_restart_time) / 60
        if time_from_last_restart_m < self._params.worker_restart_interval_m:
            return None
        # now find the longest-running actor
        max_running_time_m = 0
        actor_id_to_delete: str | None = None
        worker_allocation_to_restart: resources.Worker | None = None
        for actor_id, actor in self._ready_actors.items():
            # whether the actor has been running too long
            running_time_m = (time.time() - actor.start_time) / 60
            if running_time_m <= self._params.worker_max_lifetime_m:
                continue
            # whether the actor has a task that has been running too long
            if actor.num_used_slots > 0:
                oldest_task_scheduled_time = min(slot.get_task.scheduled_time for slot in actor.used_slots)
                oldest_task_running_time_m = (time.time() - oldest_task_scheduled_time) / 60
                if oldest_task_running_time_m >= self._params.worker_max_lifetime_m * 0.6:
                    # we better let it finish, otherwise this will be a live-lock
                    continue
            if running_time_m > max_running_time_m:
                logger.info(f"Found actor {actor_id} of stage {self._name} running for {running_time_m:.0f} minutes, ")
                max_running_time_m = running_time_m
                actor_id_to_delete = actor_id
                worker_allocation_to_restart = actor.metadata.worker
        # restart the actor if we found one
        if actor_id_to_delete is not None and worker_allocation_to_restart is not None:
            logger.info(f"Restarting {actor_id_to_delete} from {self._name} after {max_running_time_m:.0f} minutes")
            self._delete_actor(actor_id_to_delete)
            self._add_actor(worker_allocation_to_restart)
            self._last_actor_restart_time = time.time()

    def _pick_actor_for_task(self, task: Task[T]) -> str | None:
        """Picks the best ready actor for a task based on locality and busyness.

        Returns the actor ID of the chosen actor, or None if no suitable actor is available.
        Priority:
        1. Actor on the same node as the task data (`task.origin_node_id`).
        2. Actor with the fewest currently running/assigned tasks (least busy).
        Only considers actors with at least one free slot.
        """
        actors_with_free_slots = [actor for actor in self._ready_actors.values() if actor.num_empty_slots > 0]

        if not actors_with_free_slots:
            return None

        def penalty_key(actor: _ReadyActor) -> tuple[bool, int]:
            """Calculates a penalty score for an actor. Lower is better.

            Args:
                actor: The ready actor to evaluate.

            Returns:
                A tuple: (requires_remote_fetch, busyness).
                - `requires_remote_fetch`: True if actor node != task origin node.
                - `busyness`: Number of slots currently used by the actor.
            """
            is_local = actor.metadata.worker.allocation.node == task.origin_node_id
            requires_remote_fetch = not is_local
            busyness = actor.num_used_slots
            return requires_remote_fetch, busyness

        # Find the actor with the minimum penalty
        best_actor = min(actors_with_free_slots, key=penalty_key)

        # metrics
        schedule_affinity = "local" if task.origin_node_id == best_actor.metadata.worker.allocation.node else "remote"
        self._metrics_schedule_task_count.inc(tags={"stage": self._name, "affinity": schedule_affinity})

        return best_actor.metadata.worker.id

    def _maybe_resize_num_slots_per_actor(self) -> None:
        """Ensures all ready actors have the current target number of slots."""
        if not self._ready_actors:  # Optimization: Skip if no ready actors
            return
        for actor in self._ready_actors.values():
            actor.maybe_resize_num_slots_per_actor(self._slots_per_actor)

    def _delete_actor(self, actor_id: str) -> None:
        """Deletes an actor regardless of its current state."""
        deleted = (
            self._try_delete_pending_actor(actor_id)
            or self._try_delete_ready_actor(actor_id)
            or self._try_delete_node_setup_actor(actor_id)
            or self._try_delete_waiting_for_node_setup_actor(actor_id)
        )

        if not deleted:
            raise AssertionError(f"Unknown actor with id: {actor_id} requested for deletion.")

        # Only delete from allocator if we successfully found and killed the actor representation
        if self._verbosity_level >= VerbosityLevel.INFO:
            logger.info(f"Deleting worker {actor_id} from allocator.")
        self._allocator.delete_worker(actor_id)

    def _try_delete_pending_actor(self, actor_id: str) -> bool:
        """Attempts to delete an actor from the pending state."""
        if actor_id in self._pending_actors:
            actor = self._pending_actors.pop(actor_id)
            logger.info(f"Killing pending actor {actor_id}.")
            actor.kill()
            # Cancel the pending setup call if possible? Ray might handle this automatically on kill.
            # ray.cancel(actor.setup_call_ref, force=True) # Might be necessary if kill is not enough
            return True
        return False

    def _try_delete_ready_actor(self, actor_id: str) -> bool:
        """Attempts to delete an actor from the ready state."""
        if actor_id in self._ready_actors:
            actor = self._ready_actors.pop(actor_id)
            if self._verbosity_level >= VerbosityLevel.INFO:
                logger.info(f"Killing ready actor {actor_id}.")
            # Return any tasks assigned to this actor back to the main queue.
            num_returned_tasks = 0
            for task_slot in actor.slots:
                if task_slot.has_task:
                    self._task_queue.appendleft(task_slot.get_task.task)
                    num_returned_tasks += 1
            if num_returned_tasks > 0:
                logger.info(f"Returned {num_returned_tasks} tasks from deleted actor {actor_id} to the queue.")
            actor.kill()
            return True
        return False

    def _try_delete_node_setup_actor(self, actor_id: str) -> bool:
        """Attempts to delete an actor that is currently performing node setup."""
        actor_to_remove_node: Optional[_PendingNodeActor] = None
        node_id_to_remove: Optional[str] = None
        for node_id, pending_node_actor in self._pending_node_actors.items():
            if pending_node_actor.metadata.worker.id == actor_id:
                actor_to_remove_node = pending_node_actor
                node_id_to_remove = node_id
                break

        if actor_to_remove_node is not None and node_id_to_remove is not None:
            logger.info(f"Killing node setup actor {actor_id} for node {node_id_to_remove}.")
            actor_to_remove_node.kill()
            # ray.cancel(actor_to_remove_node.node_setup_call_ref, force=True) # Maybe cancel ref?
            self._pending_node_actors.pop(node_id_to_remove)
            # If other actors were waiting for this node setup, promote the next one.
            waiting_list = self._actors_waiting_for_node_setup.get(node_id_to_remove, [])
            if waiting_list:
                next_actor = waiting_list.pop(0)
                logger.info(
                    f"Promoting waiting actor {next_actor.metadata.worker.id} to perform node setup "
                    f"for {node_id_to_remove}."
                )
                self._add_node_setup_actor(next_actor.actor_ref, next_actor.metadata.worker)
                # If list is now empty, remove the key, otherwise update it
                if not waiting_list:
                    self._actors_waiting_for_node_setup.pop(node_id_to_remove, None)
                else:
                    self._actors_waiting_for_node_setup[node_id_to_remove] = waiting_list

            return True
        return False

    def _try_delete_waiting_for_node_setup_actor(self, actor_id: str) -> bool:
        """Attempts to delete an actor that is waiting for node setup to complete."""
        node_id_parent: Optional[str] = None
        actor_index_to_remove: Optional[int] = None
        for node_id, waiting_list in self._actors_waiting_for_node_setup.items():
            for i, waiting_actor in enumerate(waiting_list):
                if waiting_actor.metadata.worker.id == actor_id:
                    logger.info(f"Killing actor {actor_id} waiting for node setup on {node_id}.")
                    waiting_actor.kill()
                    node_id_parent = node_id
                    actor_index_to_remove = i
                    break
            if node_id_parent is not None:
                break

        if node_id_parent is not None and actor_index_to_remove is not None:
            # Remove the actor from the waiting list
            waiting_list = self._actors_waiting_for_node_setup[node_id_parent]
            waiting_list.pop(actor_index_to_remove)
            # If the list becomes empty, remove the node entry entirely
            if not waiting_list:
                self._actors_waiting_for_node_setup.pop(node_id_parent, None)
            else:
                # Update the list in the defaultdict (though modifying in place might also work)
                self._actors_waiting_for_node_setup[node_id_parent] = waiting_list
            return True
        return False

    def delete_actors(self) -> None:
        while self._actors_to_delete:
            actor = self._actors_to_delete.pop()
            self._delete_actor(actor.id)

    def _adjust_actors(self) -> None:
        """Applies pending actor deletions and creations."""
        while self._actors_to_delete:
            actor = self._actors_to_delete.pop()
            self._delete_actor(actor.id)

        while self._actors_to_create:
            actor = self._actors_to_create.pop()
            self._add_actor(actor)

    def _schedule_task_on_actor(self, actor: _ReadyActor) -> None:
        """Assigns the next task from the queue to an idle slot on the specified actor."""
        idle_slots = actor.idle_slots
        if not idle_slots:
            # This should ideally not happen if called correctly after _pick_actor_for_task
            logger.error(f"Attempted to schedule task on actor {actor.metadata.worker.id} with no idle slots.")
            raise AssertionError(f"No idle slots found on actor {actor.metadata.worker.id} for scheduling.")

        # Pop task from the front of the queue (FIFO)
        task = self._task_queue.popleft()
        # Select the first available idle slot
        idle_slot = idle_slots[0]
        # Submit the task to the actor via the StageWorker interface
        future = typing.cast(
            ObjectRef,
            actor.actor_ref.process_data.remote(stage_worker.TaskData(task.task_data)),  # type: ignore
        )
        # Store task data and future references in the chosen slot
        idle_slot.task = _SlotData(task, time.time(), future)
        if _VERBOSE:
            logger.info(f"Scheduled task {task.task_data} on actor {actor.metadata.worker.id}")

    def _schedule_new_tasks(self) -> None:
        # TODO: This is O(n^2), can clean this up somehow...
        while True:  # Loop through the task queue
            if not self._task_queue:
                break
            first_task = self._task_queue[0]
            actor_id = self._pick_actor_for_task(first_task)
            if actor_id is None:
                if _VERBOSE:
                    logger.info("No suitable actor found for scheduling new tasks.")
                break
            self._schedule_task_on_actor(self._ready_actors[actor_id])

    def _process_completed_tasks(self) -> None:
        actors_to_kill: set[str] = set()
        for actor in self._ready_actors.values():
            for i in range(len(actor.slots)):
                if not actor.slots[i].has_task:
                    continue

                task = actor.slots[i].get_task

                start_time = time.time()
                # Check if the task is complete without blocking
                ready, _ = ray.wait([task.object_ref], timeout=0, fetch_local=False)
                end_time = time.time()
                if end_time - start_time > 0.1:
                    logger.info(f"{self._name} Wait took {end_time - start_time}")
                if not ready:
                    continue
                # If the task is complete
                should_kill_actor = self._process_completed_task(actor, i)
                if should_kill_actor:
                    actors_to_kill.add(actor.metadata.worker.id)

        for actor_id in actors_to_kill:
            logger.info(f"Killing actor={actor_id} because stage.process_data told us to.")
            self._delete_actor(actor_id)

    def _process_completed_task(self, actor: _ReadyActor, slot_num: int) -> bool:
        # Each slot holds a reference to a ray.remote call on StageWorker.process_data.
        # process_data returns a generator, which first produces a TaskResultMetadata,
        # and then produce actual output task data returned from the stage code.
        # TaskResultMetadata includes information related to the execution of the pipeline stage.
        # We don't actually pull down the value of the task data, instead, we just get a reference.
        # For the TaskResultMetadata, we get the value. This is okay as the data is small.
        dynamic_ref = actor.slots[slot_num].get_task.object_ref
        # Get the values for the is_null and time_taken refs.
        refs: list[ObjectRef] = list(ray.get(dynamic_ref))
        # The first ref is the TaskResultMetadata, the rest are the task data.
        metadata_ref = refs[0]
        metadata: stage_worker.TaskResultMetadata = ray.get(metadata_ref)
        out_data_refs = refs[1:]
        if len(out_data_refs) != metadata.num_returns:
            raise ValueError(
                f"Number of output references ({len(out_data_refs)}) does not match the specified number of "
                f"outputs ({metadata.num_returns})"
            )

        self._task_result_metadatas.append(metadata)
        actor.rate_estimator.update(metadata.timing.process_end_time_s - metadata.timing.process_start_time_s)
        # Unless told not to, ignore the data and continue on.
        if not metadata.failure_info.should_process_further:
            self._num_null_tasks += 1
        else:
            self._completed_tasks.append(Task(out_data_refs, actor.metadata.worker.allocation.node))
        self._num_completed_tasks += 1
        # Number of tasks spawned dynamically from this stage
        if metadata.num_returns > self._params.stage_batch_size:
            self._num_dynamically_spawned_tasks += metadata.num_returns - self._params.stage_batch_size
        # Metrics
        self._update_task_metrics(metadata)
        # Mark the slot as empty
        actor.slots[slot_num].clear_task()
        return metadata.failure_info.should_restart_worker

    def _update_task_metrics(self, metadata: stage_worker.TaskResultMetadata) -> None:
        self._metrics_stage_deserialize_count.inc(1, tags={"stage": self._name})
        self._metrics_stage_deserialize_size.inc(
            metadata.task_data_info.serialized_input_size, tags={"stage": self._name}
        )
        if metadata.timing.deserialize_dur is not None and metadata.timing.process_dur is not None:
            self._metrics_stage_deserialize_time.inc(metadata.timing.deserialize_dur, tags={"stage": self._name})
            self._metrics_stage_process_time.inc(metadata.timing.process_dur, tags={"stage": self._name})

    def update(self) -> None:
        """Performs one update cycle of the actor pool state machine."""
        if _VERBOSE:
            logger.debug(f"Starting update cycle for {self.name}")
        # 1. Adjust slot counts on ready actors if needed
        self._maybe_resize_num_slots_per_actor()
        # 2. Check for completed node setups and advance waiting/setup actors
        self._check_pending_node_setup_actors()
        # 3. Check for completed individual setups and move actors to ready
        self._move_pending_actors_to_ready()
        # 4. Apply any requested actor additions or deletions
        self._adjust_actors()
        # 5. Restart long-running actors if configured to do so
        self._maybe_restart_long_running_actor()
        # 6. Check for and process results from completed tasks
        self._process_completed_tasks()
        # 7. Schedule new tasks onto idle slots of ready actors
        self._schedule_new_tasks()
        if _VERBOSE:
            logger.debug(f"Finished update cycle for {self.name}")

    def stop(self) -> None:
        """Terminates all actors managed by this pool and cleans up resources."""
        logger.info(f"Stopping actor pool {self.name}. Terminating all actors.")
        actor_ids_to_kill = set()

        # Collect IDs from all states
        actor_ids_to_kill.update(self._pending_actors.keys())
        actor_ids_to_kill.update(self._ready_actors.keys())
        actor_ids_to_kill.update(pna.metadata.worker.id for pna in self._pending_node_actors.values())
        for waiting_list in self._actors_waiting_for_node_setup.values():
            actor_ids_to_kill.update(wna.metadata.worker.id for wna in waiting_list)

        logger.info(f"Found {len(actor_ids_to_kill)} actor IDs across all states to terminate.")

        # Kill actors (use _delete_actor to handle state and allocator removal)
        # Make a copy as _delete_actor modifies the state dictionaries
        for actor_id in list(actor_ids_to_kill):
            self._delete_actor(actor_id)

        # Clear all state dictionaries explicitly after attempting deletion
        self._pending_actors.clear()
        self._ready_actors.clear()
        self._pending_node_actors.clear()
        self._actors_waiting_for_node_setup.clear()
        self._nodes_with_completed_setups.clear()
        self._actors_to_create.clear()
        self._actors_to_delete.clear()
        self._completed_tasks.clear()
        self._task_queue.clear()

        logger.info(f"Actor pool {self.name} stopped. All states cleared.")
