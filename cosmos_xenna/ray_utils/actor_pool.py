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
import enum
import statistics
import time
import typing
from typing import Generic, Optional

import attrs
import portpicker
import ray
import ray.exceptions
import ray.util.scheduling_strategies
from ray import ObjectRef
from ray.actor import ActorHandle
from ray.util.metrics import Counter

from cosmos_xenna.pipelines.private import allocator, resources
from cosmos_xenna.ray_utils import monitoring, stage, stage_worker
from cosmos_xenna.utils import grouping, stats, timing
from cosmos_xenna.utils import python_log as logger

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_RATE_ESTIMATE_LOOKBACK_S = 60 * 10
_MAX_QUEUED_TASK_METADATAS = 10000


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

    metadata: resources.WorkerMetadata
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
    # Timestamp when this actor last transitioned to the idle state (no tasks in any slot).
    # None indicates the actor is currently busy (has tasks in any slot).
    last_became_idle_time: float | None = None

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
            self.metadata.worker_id,
            self.metadata.allocation,
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

    metadata: resources.WorkerMetadata
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

    metadata: resources.WorkerMetadata
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

    metadata: resources.WorkerMetadata
    # A ray reference to the actor
    actor_ref: ActorHandle
    # A reference to the "setup" call we asked ray to call when we started this worker up
    setup_call_ref: ObjectRef[str]

    def kill(self) -> None:
        ray.kill(self.actor_ref)

    def to_stats(self) -> monitoring.PendingActorStats:
        return monitoring.PendingActorStats(self.metadata.worker_id, self.metadata.allocation)


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
class PoolParams:
    # Enable/disable whole work stealing feature
    enable_work_stealing: bool = False
    # Only allow work stealing if an actor has been idle for this many seconds.
    work_steal_idle_threshold_s: float = 1.0
    # Max number of task object refs to poll per ray.wait call when checking for completed tasks.
    max_tasks_to_poll_per_chunk: int = 8


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


class _WorkerGroupState(enum.Enum):
    WAITING_FOR_SETUP = 0
    READY = 1


@attrs.define
class _WorkerGroup:
    worker_group: resources.WorkerGroup
    actors: set[str]
    state: _WorkerGroupState
    rendevous_params: resources.NcclRendevousParams | None

    @property
    def id_(self) -> str:
        return self.worker_group.id

    @property
    def node_id(self) -> str | None:
        # Worker groups allocations are per-node, so we only have one node if len(allocation) == 1
        if len(self.worker_group.allocations) != 1:
            return None
        return self.worker_group.allocations[0].node

    @property
    def primary_node_id(self) -> str:
        return self.worker_group.allocations[0].node


class _NodePortRegistry:
    def __init__(self) -> None:
        # port -> owned_worker_group_id
        self._port_registry: dict[int, str] = {}
        # owned_worker_group_id -> set of ports (for fast cleanup)
        self._worker_group_to_ports: dict[str, set[int]] = {}

    def register_port(self, port: int, owned_worker_group_id: str) -> None:
        self._port_registry[port] = owned_worker_group_id
        if owned_worker_group_id not in self._worker_group_to_ports:
            self._worker_group_to_ports[owned_worker_group_id] = set()
        self._worker_group_to_ports[owned_worker_group_id].add(port)

    def clear_port(self, owned_worker_group_id: str) -> None:
        # Fast O(1) lookup instead of O(n) iteration
        if owned_worker_group_id in self._worker_group_to_ports:
            ports_to_remove = self._worker_group_to_ports.pop(owned_worker_group_id)
            for port in ports_to_remove:
                self._port_registry.pop(port, None)

    def get_used_ports(self) -> set[int]:
        """Get all currently used ports on this node."""
        return set(self._port_registry.keys())

    def is_empty(self) -> bool:
        """Check if this registry has any ports registered."""
        return len(self._port_registry) == 0


class ClusterPortRegistry:
    def __init__(self) -> None:
        self._port_registry: dict[str, _NodePortRegistry] = collections.defaultdict(lambda: _NodePortRegistry())

    def register_port(self, node_id: str, owned_worker_group_id: str, port: int) -> None:
        self._port_registry[node_id].register_port(port, owned_worker_group_id)

    def clear_port(self, node_id: str, owned_worker_group_id: str) -> None:
        if node_id in self._port_registry:
            self._port_registry[node_id].clear_port(owned_worker_group_id)

    def get_used_ports_for_node(self, node_id: str) -> set[int]:
        """Get all used ports for a specific node."""
        if node_id in self._port_registry:
            return self._port_registry[node_id].get_used_ports()
        return set()


@ray.remote
def _get_rendevous_params_for_node(node_id: str, already_used_ports: set[int]) -> resources.NcclRendevousParams:
    """Determines NCCL rendezvous parameters for SPMD coordination on a specific node.

    This function runs on the target node to get the correct IP address and select
    an available port for NCCL communication. This follows the torchrun model where
    the master address comes from the actual node that will coordinate the distributed
    execution.

    Args:
        node_id: Ray node ID where the rendezvous will be hosted
        already_used_ports: Set of ports already in use on this node to avoid conflicts

    Returns:
        NcclRendevousParams containing the master IP address and port for coordination

    Raises:
        RuntimeError: If no available port can be found after 100 attempts
    """
    ip = ray.util.get_node_ip_address()
    for _ in range(100):
        port = portpicker.pick_unused_port()
        if port not in already_used_ports:
            return resources.NcclRendevousParams(master_addr=ip, master_port=port)
    raise RuntimeError(f"Unable to find available port for NCCL rendezvous on node {node_id}")


class ActorPool(Generic[T, V]):
    """Manages a pool of Ray actors for a specific pipeline stage.

    This class orchestrates the lifecycle and task assignment for actors running
    code defined by a `stage.Interface`. It handles actor creation, setup (including
    node-level setup), task scheduling (with locality awareness), result collection,
    and actor scaling/deletion.

    The ActorPool supports two execution modes:
    1. **Regular Mode**: Independent actors that process tasks individually
    2. **SPMD Mode**: Coordinated worker groups for distributed inference

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

    SPMD-Specific Mechanisms:
    - Worker Groups: SPMD creates coordinated worker groups instead of independent actors
    - NCCL Rendezvous: Automatic setup of master address/port for distributed coordination
    - Environment Variables: Sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
    - CUDA Management: CUDA_VISIBLE_DEVICES is NOT restricted (follows torchrun behavior)
    - Port Registry: Cluster-wide tracking of used ports to prevent conflicts
    """

    def __init__(
        self,
        worker_allocator: allocator.WorkerAllocator,
        cluster_port_registry: ClusterPortRegistry,
        stage_interface: stage.Interface,
        params: stage.Params,
        name: str,
        queue_params: QueueParams | None = None,
        pool_params: PoolParams | None = None,
    ):
        self._name = str(name)
        self._queue_params = queue_params
        self._pool_params = pool_params if pool_params is not None else PoolParams()
        self._stage_interface = stage_interface
        self._params = params
        self._worker_shape = params.shape
        self._allocator = worker_allocator
        self._cluster_port_registry = cluster_port_registry
        self._nodes_with_completed_setups: set[str] = set()  # Nodes where setup_on_node succeeded.
        # Actors waiting for the setup_on_node on their target node to complete. Indexed by node ID.
        self._actors_waiting_for_node_setup: collections.defaultdict[str, list[_ActorWaitingForNodeSetup]] = (
            collections.defaultdict(list)
        )
        # Actor currently running setup_on_node for a specific node. Indexed by node ID.
        self._pending_node_actors: collections.OrderedDict[str, _PendingNodeActor] = collections.OrderedDict()
        # Actors waiting for their individual setup (`setup`) to complete. Indexed by actor ID.
        self._pending_actors: collections.OrderedDict[str, _PendingActor] = collections.OrderedDict()
        # Actors ready to process tasks. Indexed by actor ID.
        self._ready_actors: dict[str, _ReadyActor] = {}

        self._is_spmd = params.shape.is_spmd()
        self._worker_groups_to_delete: collections.deque[resources.WorkerGroup] = collections.deque()
        self._worker_groups_to_create: collections.deque[resources.WorkerGroup] = collections.deque()
        # If the actor pool is SPMD, worker groups (i.e. a group of actors which are acting as one inference unit),
        # may map to multiple ray actors. For consistency, we also track this for non-SPMD actor pools.
        self._worker_groups: dict[str, _WorkerGroup] = {}

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
        self._created_actor_count = 0

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
        # SPMD actor pools need to coordinate multiple actors to act as one unit.
        logger.debug("Initialized ActorPool:{} with shape:{}", self._name, self._worker_shape)

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

    def make_stats(self, ext_output_queue_size: int = 0) -> monitoring.ActorPoolStats:
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
                len(self._completed_tasks) + ext_output_queue_size,
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

    def add_actor_to_delete(self, worker: resources.WorkerGroup) -> None:
        self._worker_groups_to_delete.append(worker)

    def add_actor_to_create(self, worker: resources.WorkerGroup) -> None:
        self._worker_groups_to_create.append(worker)

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

    def get_actor_ids(self) -> ActorIds:
        # TODO: Distinguish between pending setup and pending node setup
        pending_ids = [x.actor_ref._ray_actor_id.hex() for x in self._pending_actors.values()]  # type: ignore[attr-defined]
        pending_ids.extend(x.actor_ref._ray_actor_id.hex() for x in self._pending_node_actors.values())  # type: ignore[attr-defined]
        for waiting_list in self._actors_waiting_for_node_setup.values():
            pending_ids.extend(x.actor_ref._ray_actor_id.hex() for x in waiting_list)  # type: ignore[attr-defined]

        ready_ids = [x.actor_ref._ray_actor_id.hex() for x in self._ready_actors.values()]  # type: ignore[attr-defined]
        return ActorIds(pending_ids, ready_ids)

    def _create_actor_for_worker_group(
        self,
        worker_group: _WorkerGroup,
        allocation: resources.WorkerResourcesInternal,
        index: int | None,
        world_size: int | None,
    ) -> None:
        """Creates a new actor and initiates its setup process."""

        logger.trace(f"Creating actor for worker group: {worker_group.worker_group.id}")
        actor_id = self._make_actor_id()
        # Add the actor ID to the worker group's actor set
        worker_group.actors.add(actor_id)
        # Prepare the runtime environment with CUDA_VISIBLE_DEVICES
        should_set_cuda_visible_devices = self._params.modify_cuda_visible_devices_env_var and not self._is_spmd
        allocation_public = resources.WorkerResources.from_internal(allocation, self._allocator)
        metadata = resources.WorkerMetadata(
            actor_id,
            worker_group.worker_group.id,
            allocation_public,
            should_set_cuda_visible_devices,
            None,
            None,
            None,
            None,
        )
        if self._is_spmd:
            metadata.local_rank = allocation_public.gpus[0].index
            metadata.rank = index
            metadata.world_size = world_size
            assert worker_group.rendevous_params is not None
            metadata.rendevous_params = worker_group.rendevous_params
            if len(allocation.gpus) != 1:
                raise RuntimeError(
                    f"SPMD actor pool should have exactly one GPU allocation per actor, but got {allocation.gpus}"
                )
        env_vars = {}
        # CUDA_VISIBLE_DEVICES handling:
        # - Regular stages: Restrict to only assigned GPUs for isolation
        # - SPMD stages: Don't modify CUDA_VISIBLE_DEVICES (follows torchrun behavior)
        #   This allows distributed frameworks to manage GPU visibility internally
        if should_set_cuda_visible_devices:
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(gpu) for gpu in sorted(set([x.index for x in allocation_public.gpus]))]
            )
        if self._is_spmd:
            # Set up distributed training environment variables for SPMD coordination
            assert metadata.rendevous_params is not None
            env_vars.update(metadata.distributed_execution_params.to_env_var_dict())

        env_vars.update(self._params.runtime_env.extra_env_vars)
        runtime_env = copy.deepcopy(self._params.runtime_env)
        runtime_env.extra_env_vars = env_vars

        logger.trace(f"Runtime env for stage={self.name}:{runtime_env.format()}")

        # Ask Ray to start the actor on the specified node
        actor = stage_worker.StageWorker.options(
            num_cpus=0,  # Resources are managed by the allocator, not directly by Ray options here
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=allocation.node, soft=False
            ),
            runtime_env=runtime_env.to_ray_runtime_env(),
        ).remote(
            self._stage_interface,
            self._params,
            self._name,
            metadata,
        )
        actor_handle = typing.cast(ActorHandle, actor)

        # Determine the initial state based on node setup status
        node_id = allocation.node
        if node_id in self._nodes_with_completed_setups:
            # Node setup already done, proceed directly to individual setup
            self._add_pending_actor(actor_handle, metadata)
        elif node_id in self._pending_node_actors:
            # Node setup in progress, add to waiting list
            self._add_actor_to_waiting_list(actor_handle, metadata)
        else:
            # No setup started for this node, designate this actor for node setup
            self._add_node_setup_actor(actor_handle, metadata)

    def _make_actor_id(self) -> str:
        """Makes a new actor ID."""
        self._created_actor_count += 1
        return f"{self.name}-{self._created_actor_count}"

    def _find_rendevous_params_for_worker_group(self, worker: resources.WorkerGroup) -> resources.NcclRendevousParams:
        """Finds and configures NCCL rendezvous parameters for an SPMD worker group.

        This method sets up the distributed coordination parameters needed for SPMD execution.
        It selects the primary node from the worker group allocations and determines the
        master address and port for NCCL communication.

        The process follows torchrun conventions:
        1. Use the first node in the worker group as the master node
        2. Get that node's IP address via Ray
        3. Select an unused port on that node
        4. Register the port to prevent conflicts with other worker groups

        Args:
            worker: The worker group that needs NCCL coordination

        Returns:
            NcclRendevousParams with master_addr and master_port for distributed coordination
        """
        node_id = worker.allocations[0].node
        already_used_ports_for_this_node = self._cluster_port_registry.get_used_ports_for_node(node_id)

        rendevous_params: resources.NcclRendevousParams = ray.get(
            _get_rendevous_params_for_node.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(node_id, already_used_ports_for_this_node)
        )
        self._cluster_port_registry.register_port(node_id, worker.id, rendevous_params.master_port)
        return rendevous_params

    def _add_worker_group(self, worker: resources.WorkerGroup) -> None:
        if self._is_spmd:
            rendevous_params = self._find_rendevous_params_for_worker_group(worker)
        else:
            rendevous_params = None

        logger.trace(f"Adding worker group: {worker.id}")
        worker_group = _WorkerGroup(worker, set(), _WorkerGroupState.WAITING_FOR_SETUP, rendevous_params)
        self._allocator.add_worker(worker)
        if self._is_spmd:
            logger.trace(f"Creating SPMD actors for worker group: {worker_group.worker_group.id}")
            splits = worker.split_allocation_per_gpu()
            # SPMD requires one actor per GPU across the cluster
            # The autoscaler splits worker groups by node, but we need to create actors for each GPU,
            # so we split the worker group by GPU here.
            old_node = None
            for idx, allocation in enumerate(splits):
                # Every time the node name changes, we need to reset local rank
                if allocation is not None and allocation.node != old_node:
                    old_node = allocation.node
                # Create actor with SPMD coordination (idx=rank, len(splits)=world_size)
                self._create_actor_for_worker_group(worker_group, allocation, idx, len(splits))
        else:
            # Regular (non-SPMD) actors - single allocation per worker group
            assert len(worker.allocations) == 1, "Non-SPMD actors should only have one allocation"
            self._create_actor_for_worker_group(worker_group, worker.allocations[0], None, None)
        self._worker_groups[worker_group.worker_group.id] = worker_group

    def _add_actor_to_waiting_list(self, actor_handle: ActorHandle, metadata: resources.WorkerMetadata) -> None:
        """Adds an actor to the list waiting for node setup on its target node."""
        self._actors_waiting_for_node_setup[metadata.allocation.node].append(
            _ActorWaitingForNodeSetup(
                metadata,
                actor_handle,
            )
        )

    def _add_node_setup_actor(self, actor_handle: ActorHandle, metadata: resources.WorkerMetadata) -> None:
        """Designates an actor to perform node setup and adds it to the pending node state."""
        # Ask Ray to call "setup_on_node" on the actor and keep a reference to this call.
        setup_call_ref = typing.cast(ObjectRef, actor_handle.setup_on_node.remote())
        self._pending_node_actors[metadata.allocation.node] = _PendingNodeActor(
            metadata,
            actor_handle,
            setup_call_ref,
        )

    def _add_pending_actor(self, actor_handle: ActorHandle, metadata: resources.WorkerMetadata) -> None:
        """Adds an actor to the pending state to run its individual setup."""
        # Ask Ray to call "setup" on the actor and keep a reference to this call.
        setup_call_ref = typing.cast(ObjectRef, actor_handle.setup.remote())
        self._pending_actors[metadata.worker_id] = _PendingActor(
            metadata,
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
        except Exception as e:
            self._num_actors_failed_to_start += 1
            self._delete_worker_group(actor.metadata.worker_group_id)
            logger.warning(f"Actor {actor_id} failed during setup: {e}")
            # If max_setup_failure_percentage is None, we fail the pipeline immediately
            if self._params.max_setup_failure_percentage is None:
                raise RuntimeError(f"Actor {actor_id} failed during setup: {e}") from e
            # If max_setup_failure_percentage is set, we check if the failure rate exceeds the threshold
            if is_significant_job_failure(
                self._num_actors_tried_to_start,
                self._num_actors_failed_to_start,
                self._params.max_setup_failure_percentage / 100.0,
            ):
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
        else:
            # Setup successful, move to ready state
            logger.trace(f"Actor {actor_id} setup complete. Moving to ready.")
            new_actor = _ReadyActor(
                actor.metadata,
                typing.cast(ActorHandle, actor.actor_ref),
                start_time=time.time(),
            )
            # Initialize slots for the newly ready actor
            for _ in range(self._slots_per_actor):
                new_actor.slots.append(_Slot(None))
            assert self._worker_groups[new_actor.metadata.worker_group_id].state == _WorkerGroupState.WAITING_FOR_SETUP
            # Newly ready actors start in the idle state.
            new_actor.last_became_idle_time = time.time()
            self._ready_actors[new_actor.metadata.worker_id] = new_actor
            self._move_worker_group_to_ready_if_all_actors_ready(new_actor.metadata.worker_group_id)

    def _move_worker_group_to_ready_if_all_actors_ready(self, worker_group_id: str) -> None:
        """Moves a worker group to the ready state if all actors are ready."""
        assert self._worker_groups[worker_group_id].state == _WorkerGroupState.WAITING_FOR_SETUP, (
            "Worker group is not waiting for setup"
        )
        worker_group = self._worker_groups[worker_group_id]

        # Ensure we have actors to check (prevent race conditions)
        assert worker_group.actors, f"Worker group {worker_group_id} has no actors, cannot transition to ready"

        # Check that all the actors are in the ready state
        if all(actor_id in self._ready_actors for actor_id in worker_group.actors):
            logger.debug(f"Moving worker group {worker_group_id} to ready state.")
            worker_group.state = _WorkerGroupState.READY

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
            logger.trace(f"Node setup successful for node {node_id} by actor {node_setup_actor.metadata.worker_id}.")
            self._nodes_with_completed_setups.add(node_id)

            # Move the node setup actor itself to pending (individual setup)
            self._add_pending_actor(node_setup_actor.actor_ref, node_setup_actor.metadata)

            # Move all waiting actors for this node to pending
            waiting_actors = self._actors_waiting_for_node_setup.pop(node_id, [])
            logger.trace(f"Moving {len(waiting_actors)} waiting actors for node {node_id} to pending state.")
            for waiting_actor in waiting_actors:
                self._add_pending_actor(waiting_actor.actor_ref, waiting_actor.metadata)

        except Exception as e:
            logger.error(
                f"Node setup failed for node {node_id} by actor {node_setup_actor.metadata.worker_id}. Error: {e}"
            )
            self._delete_worker_group(node_setup_actor.metadata.worker_group_id)
            # TODO: Handle cleanup and retries.
            # Promote the next waiting actor (if any) to retry node setup
            # waiting_actors = self._actors_waiting_for_node_setup.pop(node_id, [])
            # if waiting_actors:
            #     next_actor_to_setup = waiting_actors.pop(0)
            #     logger.warning(
            #         f"Promoting actor {next_actor_to_setup.metadata.worker_id} to retry node setup for {node_id}."
            #     )
            #     self._add_node_setup_actor(next_actor_to_setup.actor_ref, next_actor_to_setup.metadata.worker)
            #     # Put remaining actors back into the waiting list for the new setup attempt
            #     if waiting_actors:
            #         self._actors_waiting_for_node_setup[node_id] = waiting_actors
            # # Decide if this failure constitutes a pipeline failure (e.g., based on retries or severity)
            # # For now, we raise directly, but more sophisticated retry logic could be added.
            # # TODO: Add retry logic for node setup failures.
            raise RuntimeError(f"Node setup failed for stage {self.name} on node {node_id}.") from e

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
        # disabled for SPMD actor pools
        # TODO: Enable this for SPMD actor pools
        if self._is_spmd:
            return None
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
        worker_allocation_to_restart: resources.WorkerMetadata | None = None
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
                worker_allocation_to_restart = actor.metadata
        # restart the actor if we found one
        if actor_id_to_delete is not None and worker_allocation_to_restart is not None:
            logger.info(f"Restarting {actor_id_to_delete} from {self._name} after {max_running_time_m:.0f} minutes")
            deleted_worker_group = self._delete_worker_group(worker_allocation_to_restart.worker_group_id)
            self._add_worker_group(deleted_worker_group)
            self._last_actor_restart_time = time.time()

    def _find_worker_groups_with_free_slots(self) -> list[_WorkerGroup]:
        out = []
        for worker_group in self._worker_groups.values():
            if worker_group.state != _WorkerGroupState.READY:
                continue

            # Check if ALL actors in the group have free slots
            has_free_slots = True
            for actor_id in worker_group.actors:
                actor = self._ready_actors[actor_id]
                if actor.num_empty_slots <= 0:
                    has_free_slots = False
                    break

            if has_free_slots:
                out.append(worker_group)

        return out

    def _pick_worker_group_for_task(self, task: Task[T]) -> str | None:
        """Picks the best ready actor for a task based on locality and busyness.

        Returns the actor ID of the chosen actor, or None if no suitable actor is available.
        Priority:
        1. Actor on the same node as the task data (`task.origin_node_id`).
        2. Actor with the fewest currently running/assigned tasks (least busy).
        Only considers actors with at least one free slot.
        """
        worker_groups_with_free_slots = self._find_worker_groups_with_free_slots()

        if not worker_groups_with_free_slots:
            logger.trace(
                f"[{self._name}] No worker groups with free slots available for task "
                f"(origin_node_id: {task.origin_node_id})"
            )
            return None

        def penalty_key(worker_group: _WorkerGroup) -> tuple[bool, int]:
            """Calculates a penalty score for an actor. Lower is better.

            Args:
                actor: The ready actor to evaluate.

            Returns:
                A tuple: (requires_remote_fetch, busyness).
                - `requires_remote_fetch`: True if actor node != task origin node.
                - `busyness`: Number of slots currently used by the actor.
            """
            is_local = worker_group.node_id is not None and worker_group.node_id == task.origin_node_id
            requires_remote_fetch = not is_local
            busyness = sum([self._ready_actors[actor_id].num_used_slots for actor_id in worker_group.actors])
            return requires_remote_fetch, busyness

        # Find the actor with the minimum penalty
        best_wg = min(worker_groups_with_free_slots, key=penalty_key)

        logger.trace(f"[{self._name}] Actor selection for task (origin_node_id: {task.origin_node_id}):")
        logger.trace(f"  Considered {len(worker_groups_with_free_slots)} actors with free slots:")
        for wg in worker_groups_with_free_slots:
            requires_remote_fetch, num_used_slots = penalty_key(wg)
            is_local = not requires_remote_fetch
            logger.trace(f"    Worker group {wg.id_} (node: {wg.node_id}): local={is_local}, busyness={num_used_slots}")
        logger.trace(f"  Selected: Worker group {best_wg.id_} (node: {best_wg.node_id})")

        # metrics
        schedule_affinity = "local" if is_local else "remote"
        self._metrics_schedule_task_count.inc(tags={"stage": self._name, "affinity": schedule_affinity})

        return best_wg.id_

    def _maybe_resize_num_slots_per_actor(self) -> None:
        """Ensures all ready actors have the current target number of slots."""
        if not self._ready_actors:  # Optimization: Skip if no ready actors
            return
        for actor in self._ready_actors.values():
            actor.maybe_resize_num_slots_per_actor(self._slots_per_actor)

    def _delete_worker_group(self, worker_group_id: str) -> resources.WorkerGroup:
        """Deletes a worker group regardless of its current state."""
        worker_group = self._worker_groups.pop(worker_group_id)
        logger.debug(f"Deleting worker group {worker_group_id}.")

        # Remove the entire worker group from allocator (not individual actors)
        self._allocator.remove_worker(worker_group.worker_group.id)
        if self._is_spmd:
            self._cluster_port_registry.clear_port(worker_group.primary_node_id, worker_group_id)

        for actor_id in worker_group.actors:
            self._delete_actor(actor_id)  # Use internal method that skips allocator removal
        return worker_group.worker_group

    def _delete_actor(self, actor_id: str) -> None:
        """Internal method to delete an actor without touching the allocator."""
        deleted = (
            self._try_delete_pending_actor(actor_id)
            or self._try_delete_ready_actor(actor_id)
            or self._try_delete_node_setup_actor(actor_id)
            or self._try_delete_waiting_for_node_setup_actor(actor_id)
        )

        if not deleted:
            # TODO: This should be an error, but it is happening in prod. We need to investigate why.
            logger.warning(f"Unknown actor with id: {actor_id} requested for deletion.")

    def _try_delete_pending_actor(self, actor_id: str) -> bool:
        """Attempts to delete an actor from the pending state."""
        if actor_id in self._pending_actors:
            actor = self._pending_actors.pop(actor_id)
            logger.debug(f"Killing pending actor {actor_id}.")
            actor.kill()
            # Cancel the pending setup call if possible? Ray might handle this automatically on kill.
            # ray.cancel(actor.setup_call_ref, force=True) # Might be necessary if kill is not enough
            return True
        return False

    def _try_delete_ready_actor(self, actor_id: str) -> bool:
        """Attempts to delete an actor from the ready state."""
        if actor_id in self._ready_actors:
            actor = self._ready_actors.pop(actor_id)
            logger.debug(f"Killing ready actor {actor_id}.")
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
            if pending_node_actor.metadata.worker_id == actor_id:
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
                    f"Promoting waiting actor {next_actor.metadata.worker_id} to perform node setup "
                    f"for {node_id_to_remove}."
                )
                self._add_node_setup_actor(next_actor.actor_ref, next_actor.metadata)
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
                if waiting_actor.metadata.worker_id == actor_id:
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

    def delete_worker_groups(self) -> None:
        while self._worker_groups_to_delete:
            worker_group = self._worker_groups_to_delete.pop()
            self._delete_worker_group(worker_group.id)

    def _adjust_actors(self) -> None:
        """Applies pending actor deletions and creations."""
        while self._worker_groups_to_delete:
            worker_group = self._worker_groups_to_delete.pop()
            self._delete_worker_group(worker_group.id)

        while self._worker_groups_to_create:
            worker_group = self._worker_groups_to_create.pop()
            self._add_worker_group(worker_group)

    def _schedule_task_on_worker_group(self, worker_group: _WorkerGroup) -> None:
        # Pop task from the front of the queue (FIFO)
        task = self._task_queue.popleft()

        for actor_id in worker_group.actors:
            actor = self._ready_actors[actor_id]
            self._schedule_task_on_actor(actor, task)

    def _schedule_task_on_actor(self, actor: _ReadyActor, task: Task[T]) -> None:
        """Assigns the next task from the queue to an idle slot on the specified actor."""
        idle_slots = actor.idle_slots
        if not idle_slots:
            # This should ideally not happen if called correctly after _pick_actor_for_task
            logger.error(f"Attempted to schedule task on actor {actor.metadata.worker_id} with no idle slots.")
            raise AssertionError(f"No idle slots found on actor {actor.metadata.worker_id} for scheduling.")

        # Select the first available idle slot
        idle_slot = idle_slots[0]
        # Submit the task to the actor via the StageWorker interface
        future = typing.cast(
            ObjectRef,
            actor.actor_ref.process_data.remote(stage_worker.TaskData(task.task_data)),  # type: ignore
        )
        # Store task data and future references in the chosen slot
        idle_slot.task = _SlotData(task, time.time(), future)
        # Actor is no longer idle after scheduling a task
        actor.last_became_idle_time = None
        logger.trace(f"Scheduled task {task.task_data} on actor {actor.metadata.worker_id}")

    def _schedule_new_tasks(self) -> None:
        # TODO: This is O(n^2), can clean this up somehow...
        while True:  # Loop through the task queue
            if not self._task_queue:
                break
            first_task = self._task_queue[0]
            worker_group_id = self._pick_worker_group_for_task(first_task)
            if worker_group_id is None:
                logger.trace("No suitable worker group found for scheduling new tasks.")
                break
            self._schedule_task_on_worker_group(self._worker_groups[worker_group_id])

    def _process_completed_tasks(self) -> None:
        # get the object refs of all in-flight tasks
        inflight_task_obj_refs: dict[ObjectRef, tuple[_WorkerGroup, _ReadyActor, int, bool]] = {}
        wgs_to_kill: set[str] = set()
        ready_wgs = [wg for wg in self._worker_groups.values() if wg.state == _WorkerGroupState.READY]
        for wg in ready_wgs:
            for actor_id in wg.actors:
                actor = self._ready_actors[actor_id]
                # For non-SPMD actors, rank is None, so treat them as primary
                is_primary = actor.metadata.rank == 0 or actor.metadata.rank is None
                for i in range(len(actor.slots)):
                    if not actor.slots[i].has_task:
                        continue
                    task = actor.slots[i].get_task
                    inflight_task_obj_refs[task.object_ref.completed()] = (wg, actor, i, is_primary)  # pyright: ignore[reportAttributeAccessIssue]

        # find completed tasks using parallel polling
        finished_slots: list[tuple[_WorkerGroup, _ReadyActor, int, bool]] = []
        for chunked_obj_refs in grouping.split_by_chunk_size(
            inflight_task_obj_refs.keys(), self._pool_params.max_tasks_to_poll_per_chunk
        ):
            start_time = time.time()
            # Check if the task is complete without blocking
            ready, _ = ray.wait(chunked_obj_refs, num_returns=len(chunked_obj_refs), timeout=0, fetch_local=False)
            end_time = time.time()
            if end_time - start_time > 0.1:
                logger.info(f"{self._name} Wait took {end_time - start_time} for {len(chunked_obj_refs)} tasks")

            for obj_ref in ready:
                finished_slots.append(inflight_task_obj_refs[obj_ref])

        # Process completed tasks and collect worker groups to kill if needed
        for wg, actor, i, is_primary in finished_slots:
            should_kill_actor = self._process_completed_task(actor, i, is_primary)
            if should_kill_actor:
                wgs_to_kill.add(wg.id_)

        for wg_id in wgs_to_kill:
            logger.info(f"Killing worker group={wg_id} because stage.process_data told us to.")
            self._delete_worker_group(wg_id)

        # Update idle timestamps for actors that became idle after processing completions
        now = time.time()
        for actor in self._ready_actors.values():
            if actor.num_used_slots == 0 and actor.last_became_idle_time is None:
                actor.last_became_idle_time = now

    def _attempt_steal_one_task(self, donor: _ReadyActor, receiver: _ReadyActor) -> bool:
        """Attempts to steal the most recently scheduled not-yet-started task from donor to receiver.

        We first try to cancel the queued task inside the donor actor; only if cancellation succeeds do we
        cancel the remote process_data call, clear the donor slot, and requeue the task in this pool.

        Returns True if a task was successfully stolen and requeued, False otherwise.
        """
        if donor.num_used_slots <= 1:
            logger.trace(f"Donor {donor.metadata.worker_id} has no used slots.")
            return False
        if receiver.num_empty_slots <= 0:
            logger.trace(f"Receiver {receiver.metadata.worker_id} has no empty slots.")
            return False

        # Pick the most recently scheduled slot; that is most likely still queued/not started.
        candidate_slots = [(i, s.get_task) for i, s in enumerate(donor.slots) if s.has_task]
        if not candidate_slots:
            logger.trace(f"Donor {donor.metadata.worker_id} has no candidate slots.")
            return False
        candidate_index, candidate_slotdata = max(candidate_slots, key=lambda x: x[1].scheduled_time)

        task_to_steal = candidate_slotdata.task
        cancel_ref = typing.cast(
            ObjectRef,
            donor.actor_ref.cancel_task.remote(stage_worker.TaskData(task_to_steal.task_data)),
        )  # type: ignore
        # Ask donor worker to remove the task from its internal queues if it hasn't started.
        # Only proceed if cancellation is confirmed.
        try:
            did_cancel = ray.get(cancel_ref)

        except Exception as e:
            logger.trace(f"Failed to request cancel from donor {donor.metadata.worker_id}: {e}")
            raise RuntimeError(f"Failed to request cancel from donor {donor.metadata.worker_id}") from e

        if not did_cancel:
            logger.trace(f"Donor {donor.metadata.worker_id} did not cancel task {task_to_steal.task_data}.")
            return False

        # Now cancel the remote process_data call to unblock it, clear the slot, and requeue the task.
        try:
            ray.cancel(candidate_slotdata.object_ref)
        except (ray.exceptions.RayError, RuntimeError):
            # Even if cancel fails, we already removed from internal queues; continue
            logger.trace(f"Failed to cancel task {task_to_steal.task_data} on donor {donor.metadata.worker_id}.")
            pass

        donor.slots[candidate_index].clear_task()
        # Requeue stolen task at the front so it is scheduled immediately (likely to receiver)
        self._task_queue.appendleft(task_to_steal)
        return True

    def _work_steal(self) -> None:
        """Rebalances tasks across actors by stealing queued tasks from busy actors to idle ones.

        Goal: ensure at most one in-flight/queued task per actor when possible, to avoid initial idling when the
        number of tasks is smaller than the total capacity. This only steals tasks that haven't begun processing.
        """
        if not self._ready_actors:
            return
        ready_list = list(self._ready_actors.values())
        now = time.time()
        idle_receivers = [
            a
            for a in ready_list
            if a.num_used_slots == 0
            and a.num_empty_slots > 0
            and a.last_became_idle_time is not None
            and (now - a.last_became_idle_time) >= self._pool_params.work_steal_idle_threshold_s
        ]
        logger.trace(f"Found {len(idle_receivers)} idle receivers.")
        if not idle_receivers:
            return
        donors = [a for a in ready_list if a.num_used_slots > 1]
        logger.trace(f"Found {len(donors)} donors.")
        if not donors:
            return

        # Try to give one task to each idle receiver if possible
        donor_index = 0
        for receiver in idle_receivers:
            # Round-robin donors
            attempts = 0
            stolen = False
            while attempts < len(donors) and not stolen:
                donor = donors[donor_index % len(donors)]
                donor_index += 1
                attempts += 1
                if donor is receiver:
                    continue
                if donor.num_used_slots <= 1:
                    continue
                logger.trace(
                    f"Attempting to steal task from donor {donor.metadata.worker_id} to receiver "
                    f"{receiver.metadata.worker_id}."
                )
                stolen = self._attempt_steal_one_task(donor, receiver)
                logger.trace(f"Stolen={stolen}")
            # If we couldn't steal for this receiver, continue to next; nothing else to do

    def _process_completed_task(self, actor: _ReadyActor, slot_num: int, is_primary: bool) -> bool:
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

        # Only update metrics for the primary actor. This is rank=0 for spmd.
        if is_primary:
            self._task_result_metadatas.append(metadata)
            actor.rate_estimator.update(metadata.timing.process_end_time_s - metadata.timing.process_start_time_s)
            # Unless told not to, ignore the data and continue on.
            if not metadata.failure_info.should_process_further:
                self._num_null_tasks += 1
            else:
                self._completed_tasks.append(Task(out_data_refs, actor.metadata.allocation.node))
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
        logger.trace(f"Starting update cycle for {self.name}")
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
        if self._pool_params.enable_work_stealing:
            # 8. Rebalance queued tasks via work-stealing. We do this primarily to avoid idling when the number of
            # tasks is smaller than the num_actors * slots_per_actor. In these cases, without this step, we can leave
            # some actors idle because we eagerly schedule stasks to ready actors.
            self._work_steal()
            # 9. Schedule any stolen tasks.
            self._schedule_new_tasks()
        logger.trace(f"Finished update cycle for {self.name}")

    def stop(self) -> None:
        """Terminates all actors managed by this pool and cleans up resources."""
        logger.debug(f"Stopping actor pool {self.name}. Terminating all actors.")
        actor_ids_to_kill = set()

        # Collect IDs from all states
        actor_ids_to_kill.update(self._pending_actors.keys())
        actor_ids_to_kill.update(self._ready_actors.keys())
        actor_ids_to_kill.update(pna.metadata.worker_id for pna in self._pending_node_actors.values())
        for waiting_list in self._actors_waiting_for_node_setup.values():
            actor_ids_to_kill.update(wna.metadata.worker_id for wna in waiting_list)

        logger.debug(f"Found {len(actor_ids_to_kill)} actor IDs across all states to terminate.")

        # Kill worker groups
        for worker_group_id in list(self._worker_groups.keys()):
            self._delete_worker_group(worker_group_id)

        # TODO: Technically, we should clear up the worker groups here as well, but this is not a big deal as we do
        # not expected users to re-use instances of ActorPool.

        # Clear all state dictionaries explicitly after attempting deletion
        self._pending_actors.clear()
        self._ready_actors.clear()
        self._pending_node_actors.clear()
        self._actors_waiting_for_node_setup.clear()
        self._nodes_with_completed_setups.clear()
        self._worker_groups_to_create.clear()
        self._worker_groups_to_delete.clear()
        self._completed_tasks.clear()
        self._task_queue.clear()

        logger.debug(f"Actor pool {self.name} stopped. All states cleared.")
