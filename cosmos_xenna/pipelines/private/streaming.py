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

from __future__ import annotations

import collections
import concurrent.futures
import math
import queue as python_queue
import random
import time
import typing
from typing import Optional

import attrs
import ray
from ray.util.metrics import Gauge

from cosmos_xenna.pipelines.private import (
    allocator,
    autoscaling_algorithms,
    data_structures,
    monitoring,
    resources,
    specs,
)
from cosmos_xenna.ray_utils import actor_pool, stage_worker
from cosmos_xenna.utils import approx, deque, timing, verbosity
from cosmos_xenna.utils import python_log as logger

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_MAX_MAIN_LOOP_RATE_HZ = 100


@attrs.define
class StreamingExecutorTiming:
    iteration_start: float = 0.0
    auto_scaling_apply_end: float = 0.0
    pool_update_end: float = 0.0
    auto_scaling_submit_end: float = 0.0
    monitor_update_end: float = 0.0
    add_tasks_end: float = 0.0
    sleep_end: float = 0.0

    @property
    def auto_scaling_apply_time(self) -> float:
        return self.auto_scaling_apply_end - self.iteration_start

    @property
    def pool_update_time(self) -> float:
        return self.pool_update_end - self.auto_scaling_apply_end

    @property
    def auto_scaling_submit_time(self) -> float:
        return self.auto_scaling_submit_end - self.pool_update_end

    @property
    def monitor_update_time(self) -> float:
        return self.monitor_update_end - self.auto_scaling_submit_end

    @property
    def add_tasks_time(self) -> float:
        return self.add_tasks_end - self.monitor_update_end

    @property
    def sleep_time(self) -> float:
        return self.sleep_end - self.add_tasks_end

    @property
    def total_time(self) -> float:
        return self.sleep_end - self.iteration_start

    def to_log_string(self) -> str:
        stages: list[tuple[str, float]] = [
            ("Auto Scaling Apply", self.auto_scaling_apply_time),
            ("Pool Update", self.pool_update_time),
            ("Auto Scaling Submit", self.auto_scaling_submit_time),
            ("Monitor Update", self.monitor_update_time),
            ("Add Tasks", self.add_tasks_time),
            ("Sleep", self.sleep_time),
            ("Total", self.total_time),
        ]

        log_lines = ["StreamingExecutor Timing Summary:"]
        max_name_length = max(len(name) for name, _ in stages)

        for name, duration in stages:
            log_lines.append(f"  {name:<{max_name_length}}: {duration:.6f} seconds")

        return "\n".join(log_lines)


@attrs.define
class StreamingExecutorStats:
    timing: StreamingExecutorTiming

    def to_log_string(self) -> str:
        return self.timing.to_log_string()


class _StreamingLoopMetrics:
    def __init__(self) -> None:
        self._steps = (
            "auto_scaling_apply",
            "pool_update",
            "auto_scaling_submit",
            "monitor_update",
            "add_tasks",
            "sleep",
            "total",
        )
        # help we get avg & max within a time window
        self._historical_stats: collections.deque[StreamingExecutorTiming] = collections.deque(maxlen=100)
        self._metrics_update_internal_s = 60.0
        self._last_ext_update_timestamp = 0.0
        # define & initialize the metrics
        self._metrics_loop_time = Gauge(
            "pipeline_loop_time_s",
            "Elapsed time for the main loop of streaming executor",
            tag_keys=(
                "step",
                "method",
            ),
        )
        for step in self._steps:
            for method in ("avg", "max"):
                self._metrics_loop_time.set(0.0, tags={"step": step, "method": method})

    def update(self, stats: StreamingExecutorTiming) -> None:
        """Updates the metrics with the provided stats."""

        self._historical_stats.append(stats)
        if time.time() - self._last_ext_update_timestamp > self._metrics_update_internal_s:
            # update to exported metrics
            try:
                for step in self._steps:
                    avg_time = sum(getattr(x, f"{step}_time") for x in self._historical_stats) / len(
                        self._historical_stats
                    )
                    max_time = max(getattr(x, f"{step}_time") for x in self._historical_stats)
                    self._metrics_loop_time.set(avg_time, tags={"step": step, "method": "avg"})
                    self._metrics_loop_time.set(max_time, tags={"step": step, "method": "max"})
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to update external metrics: {e}")
                # suppress the error for some longer period
                self._last_ext_update_timestamp = time.time() + self._metrics_update_internal_s * 10
            else:
                # set the timestamp
                self._last_ext_update_timestamp = time.time()


@attrs.define
class AutoscalerResultForStage:
    slots_per_worker: int
    new_workers: list[resources.WorkerGroup]
    workers_to_delete: list[resources.WorkerGroup]


def _make_problem_from_pipeline_spec(
    pipeline_spec: specs.PipelineSpec, cluster_resources: resources.ClusterResources
) -> data_structures.Problem:
    """Creates a Problem instance from a pipeline specification.

    Args:
        s: Pipeline specification containing stage information.
        cluster_resources: Available cluster resources.

    Returns:
        A new Problem instance configured according to the specification.
    """
    out_stages = []
    num_nodes = len(cluster_resources.nodes)
    for idx, stage in enumerate(pipeline_spec.stages):
        assert isinstance(stage, specs.StageSpec)
        if stage.num_workers is not None:
            num_workers = stage.num_workers
        elif stage.num_workers_per_node is not None:
            num_workers = math.ceil(stage.num_workers_per_node * num_nodes)
        else:
            num_workers = None
        out_stages.append(
            data_structures.ProblemStage(
                stage.name(idx),
                stage.stage.stage_batch_size,
                stage.stage.required_resources.to_worker_shape(cluster_resources),
                requested_num_workers=num_workers,
                over_provision_factor=stage.over_provision_factor,
            )
        )
    return data_structures.Problem(cluster_resources, out_stages)


def make_task_measurement_from_task_metadata(
    task_metadata: stage_worker.TaskResultMetadata,
) -> data_structures.TaskMeasurement:
    return data_structures.TaskMeasurement(
        start_time=task_metadata.timing.process_start_time_s,
        end_time=task_metadata.timing.process_end_time_s,
        num_returns=task_metadata.num_returns,
    )


def make_problem_worker_state_from_worker_state(
    state: resources.WorkerGroup,
) -> data_structures.ProblemWorkerGroupState:
    """Creates a ProblemWorkerState from a Worker instance.

    Args:
        state: Worker instance containing worker state information.

    Returns:
        A new ProblemWorkerState instance.
    """
    return data_structures.ProblemWorkerGroupState.make(state.id, state.allocations)


class Autoscaler:
    """Manages the autoscaling of pipeline stages in a streaming execution mode.

    This class is responsible for calculating and applying changes to the number of
    workers in each stage of the pipeline. To avoid blocking the main execution
    thread, the autoscaling calculation is performed in a background thread.

    The `Autoscaler` is implemented as a context manager to ensure that the
    background thread executor is properly shut down upon exit.

    Attributes:
        _verbosity_level: The level of logging verbosity.
        _allocator: An instance of `WorkerAllocator` to manage worker resources.
        _algorithm: The autoscaling algorithm to use for calculations.
        _executor: A `ThreadPoolExecutor` for running calculations in the background.
        _autoscale_future: A `Future` object representing the pending result of an
            autoscaling calculation. When None, no calculation is in progress and
            any previous results have been applied.
        _autoscale_start_time: The timestamp when the last autoscaling calculation
            was started.
    """

    def __init__(
        self,
        worker_allocator: allocator.WorkerAllocator,
        pipeline_spec: specs.PipelineSpec,
        cluster_resources: resources.ClusterResources,
        verbosity_level: verbosity.VerbosityLevel = verbosity.VerbosityLevel.NONE,
    ) -> None:
        """Initializes the Autoscaler.

        Args:
            worker_allocator: The worker allocator for the cluster.
            pipeline_spec: The specification for the pipeline.
            cluster_resources: The available resources in the cluster.
            verbosity_level: The verbosity level for logging.
        """
        self._verbosity_level = verbosity_level
        self._allocator = worker_allocator
        self._algorithm = autoscaling_algorithms.FragmentationBasedAutoscaler(
            pipeline_spec.config.mode_specific.autoscale_speed_estimation_window_duration_s,
            pipeline_spec.config.mode_specific.autoscale_speed_estimation_min_data_points,
        )
        self._algorithm.setup(_make_problem_from_pipeline_spec(pipeline_spec, cluster_resources))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._autoscale_future: Optional[concurrent.futures.Future] = None
        self._autoscale_start_time: float = 0.0

    def __enter__(self) -> Autoscaler:
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001, ANN204
        """Exits the context manager, shutting down the executor."""
        self._executor.shutdown(wait=True)

    def add_measurements(
        self,
        task_metadata_per_pool: list[list[stage_worker.TaskResultMetadata]],
    ) -> None:
        """Adds new performance measurements to the autoscaling algorithm.

        Args:
            task_metadata_per_pool: A list of task metadata lists, one for each
                stage in the pipeline.
        """
        stage_measurements = []
        for metadatas in task_metadata_per_pool:
            stage_measurements.append(
                data_structures.StageMeasurements([make_task_measurement_from_task_metadata(x) for x in metadatas])
            )
        measurements = data_structures.Measurements(time.time(), stage_measurements)
        self._algorithm.update_with_measurements(time.time(), measurements)

    def _make_problem_state(
        self, actor_pools: list[actor_pool.ActorPool], stages_is_dones: list[bool]
    ) -> data_structures.ProblemState:
        """Creates a `ProblemState` instance from the current pipeline state.

        Args:
            actor_pools: The list of actor pools for each stage.
            stages_is_dones: A list of booleans indicating if each stage is done.

        Returns:
            A `ProblemState` object representing the current state of the pipeline.
        """
        stages = []
        for pool, is_done in zip(actor_pools, stages_is_dones):
            workers = self._allocator.get_workers_in_stage(pool.name)
            stages.append(
                data_structures.ProblemStageState(
                    pool.name,
                    [make_problem_worker_state_from_worker_state(w) for w in workers],
                    pool.slots_per_actor,
                    is_done,
                )
            )
        return data_structures.ProblemState(stages)

    def start_autoscale_calculation(self, pools: list[actor_pool.ActorPool], stages_is_dones: list[bool]) -> None:
        """Starts a new autoscaling calculation in a background thread.

        If a calculation is already in progress, this method does nothing.

        Args:
            pools: The list of actor pools for each stage.
            stages_is_dones: A list of booleans indicating if each stage is done.
        """
        if self._autoscale_future is not None:
            if self._verbosity_level > verbosity.VerbosityLevel.INFO:
                logger.info("Autoscale calculation already in progress or pending application. Skipping.")
            return

        self._autoscale_start_time = time.time()
        problem_state = self._make_problem_state(pools, stages_is_dones)
        self._autoscale_future = self._executor.submit(self._algorithm.autoscale, time.time(), problem_state)

    def apply_autoscale_result_if_ready(self, pools: list[actor_pool.ActorPool]) -> None:
        """Applies the result of a completed autoscaling calculation.

        If the calculation is not yet finished, this method does nothing. Once the
        result is applied, it updates the actor pools with new worker
        configurations.

        Args:
            pools: The list of actor pools to update with the new scaling results.
        """
        if self._autoscale_future is not None and self._autoscale_future.done():
            autoscale_result: data_structures.Solution = self._autoscale_future.result()

            for result, pool in zip(autoscale_result.stages, pools):
                pool.set_num_slots_per_actor(result.slots_per_worker)
                for w in result.new_workers:
                    logger.trace(f"Adding actor to create: {w.to_worker_group(pool.name)}")
                    pool.add_actor_to_create(w.to_worker_group(pool.name))

                for w in result.deleted_workers:
                    logger.trace(f"Adding actor to delete: {w.to_worker_group(pool.name)}")
                    pool.add_actor_to_delete(w.to_worker_group(pool.name))

            self._autoscale_future = None
            autoscale_end_time = time.time()
            if autoscale_end_time - self._autoscale_start_time > 1.0:
                logger.warning(
                    f"Applying autoscale results took {autoscale_end_time - self._autoscale_start_time} seconds"
                )


def _verify_enough_resources(pipeline_spec: specs.PipelineSpec, cluster_resources: resources.ClusterResources) -> None:
    cluster_resource_totals = cluster_resources.total_pool()

    required_resources = resources.PoolOfResources(
        cpus=0.0,
        gpus=0.0,
    )

    for stage in pipeline_spec.stages:
        assert isinstance(stage, specs.StageSpec)
        if stage.num_workers is not None:
            num_required = stage.num_workers
        elif stage.num_workers_per_node is not None:
            num_required = math.ceil(stage.num_workers_per_node * cluster_resources.num_nodes)
        else:
            num_required = 1

        resources_per_worker = stage.stage.required_resources.to_pool()
        required_resources = required_resources.add(resources_per_worker.multiply_by(num_required))

    summary_string = (
        f"If running locally, you can run the pipeline in batch mode by setting the config.execution_mode to BATCH.\n"
        f"Cluster resources: {cluster_resource_totals}\n"
        f"Required resources: {required_resources}"
    )
    if approx.float_lt(cluster_resource_totals.cpus, required_resources.cpus):
        raise ValueError(
            f"Not enough CPU resources to run pipeline in streaming mode. Pipeline requires "
            f"{required_resources.cpus} but only {cluster_resource_totals.cpus} are available.\n{summary_string}"
        )
    if approx.float_lt(cluster_resource_totals.gpus, required_resources.gpus):
        raise ValueError(
            f"Not enough GPU resources to run pipeline in streaming mode. Pipeline requires "
            f"{required_resources.gpus} but only {cluster_resource_totals.gpus} are available.\n{summary_string}"
        )


class Queue:
    """
    A queue that stores Ray ObjectRefs, organized by the originating node ID.
    It allows retrieving batches of items, attempting to pull items evenly
    across nodes.
    """

    def __init__(self, samples_per_task_window: int = 100) -> None:
        """
        Initializes the Queue.

        Args:
            samples_per_task_window: The maximum number of task sample counts
                                     to store for calculating the average.
        """
        # Stores deques of ObjectRefs, keyed by the node ID they originated from.
        # A key of None might be used if the origin node is unknown.
        self.by_node_id: collections.defaultdict[Optional[str], collections.deque[ray.ObjectRef]] = (
            collections.defaultdict(collections.deque)
        )
        # Stores the number of samples added in recent tasks to calculate an average.
        self._samples_per_task: collections.deque[int] = collections.deque(maxlen=samples_per_task_window)

    def __bool__(self) -> bool:
        """Returns True if the queue contains any items, False otherwise."""
        return len(self) > 0

    def __len__(self) -> int:
        """Returns the total number of items across all node queues."""
        return sum(len(q) for q in self.by_node_id.values())

    def avg_samples_per_task(self) -> Optional[float]:
        """
        Calculates the average number of samples added per task, based on the
        recent history defined by samples_per_task_window.

        Returns:
            The average number of samples per task, or None if no tasks have
            been added yet.
        """
        if not self._samples_per_task:
            return None
        return sum(self._samples_per_task) / len(self._samples_per_task)

    def add_task(self, task: actor_pool.Task) -> None:
        """
        Adds all ObjectRefs from a Task to the queue, organizing them by the
        task's origin node ID. Also records the number of samples in the task.

        Args:
            task: The Task object containing ObjectRefs and origin node ID.
        """
        node_queue = self.by_node_id[task.origin_node_id]
        count = 0
        for ref in task.task_data:
            node_queue.append(ref)
            count += 1

        # Only record if samples were actually added
        if count > 0:
            self._samples_per_task.append(count)

    # TODO: Prefer to pull items from nodes that have enough samples to fill the batch.
    def maybe_get_batch(self, stage_batch_size: int) -> Optional[actor_pool.Task]:
        """
        Attempts to retrieve a batch of a specific size from the queue.

        It shuffles the nodes and then iterates through them, pulling items. It will pull fully from a node before
        moving on to the next.

        Args:
            stage_batch_size: The desired number of items in the batch.

        Returns:
            A Task object containing the batch of items and the most frequent
            origin node ID among the batch items, or None if the queue doesn't
            have enough items to form a full batch.

        Raises:
            ValueError: If stage_batch_size is not positive.
        """
        if stage_batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")
        # Check if there are enough total items before starting
        if len(self) < stage_batch_size:
            return None

        vals: list[ray.ObjectRef] = []
        node_ids_in_batch: list[Optional[str]] = []

        # Get node IDs that currently have items
        # Shuffle to avoid consistently prioritizing the same nodes
        active_node_ids = [nid for nid, queue in self.by_node_id.items() if queue]
        random.shuffle(active_node_ids)  # Shuffle for better distribution

        # Keep track of nodes that become empty during this batch retrieval
        emptied_nodes = set()

        # Loop until the batch is full
        while len(vals) < stage_batch_size:
            made_progress = False  # Flag to detect if we are stuck (shouldn't happen if initial len check passes)
            # Iterate through nodes that *might* still have items
            # Use a copy of the list as we might modify the underlying queues
            current_nodes_to_check = [nid for nid in active_node_ids if nid not in emptied_nodes]
            assert current_nodes_to_check, "No nodes to check"

            for node_id in current_nodes_to_check:
                queue = self.by_node_id[node_id]
                assert queue, "Queue is empty"
                while queue and len(vals) < stage_batch_size:
                    # Take one item
                    vals.append(queue.popleft())
                    node_ids_in_batch.append(node_id)
                    made_progress = True

                # If the queue became empty after popping, mark it
                if not queue:
                    emptied_nodes.add(node_id)

                # Check if batch is full *after adding* the item
                if len(vals) == stage_batch_size:
                    break  # Exit the inner for loop immediately

            # If we went through all active nodes and didn't add anything, something is wrong
            if not made_progress:
                # This indicates a potential logic error or race condition if len(self) changed concurrently
                print(
                    f"Warning: No progress made in filling batch. Current size: {len(vals)}, Target: {stage_batch_size}"
                )
                # Depending on requirements, you might return a partial batch or raise an error
                # For now, we break, and the assertion below will likely fail if len(vals) != stage_batch_size
                break

        # If it fails, it implies not enough items were available despite the initial check,
        # or the loop logic has an issue.
        assert len(vals) == stage_batch_size, f"Failed to collect exact batch size. Got {len(vals)}"
        assert len(node_ids_in_batch) > 0, "No node IDs in batch"
        counts = collections.Counter(node_ids_in_batch)
        # Find the node ID that appeared most often in this specific batch
        most_common_node = counts.most_common(1)[0][0]

        # Return the batch as a Task object
        return actor_pool.Task(vals, most_common_node)

    def get_all_samples(self) -> list[ray.ObjectRef]:
        """
        Retrieves all currently held ObjectRefs from all node queues.

        Note: This clears the queue. If you need to keep the items,
              you should copy them first.

        Returns:
            A list containing all ObjectRefs present in the queue.
        """
        all_refs = []
        for node_id in list(self.by_node_id.keys()):  # Iterate over keys copy
            queue = self.by_node_id[node_id]
            all_refs.extend(list(queue))  # Add all items from the deque
            queue.clear()  # Clear the original deque

        # Clean up empty node entries
        empty_nodes = [nid for nid, queue in self.by_node_id.items() if not queue]
        for nid in empty_nodes:
            del self.by_node_id[nid]
        return all_refs


def run_pipeline(
    pipeline_spec: specs.PipelineSpec,
    cluster_resources: resources.ClusterResources,
) -> Optional[list]:
    """Runs a pipeline under STREAMING mode.

    This is the most complex pipeline mode we have. See README.md for more info about what streaming mode does.
    """
    _verify_enough_resources(pipeline_spec, cluster_resources)
    assert isinstance(pipeline_spec.config.mode_specific, specs.StreamingSpecificSpec)
    # Create a worker allocator to keep track of which workers are allocated across the cluster
    worker_allocator = allocator.WorkerAllocator.make(cluster_resources)
    # Keeps track of which ports have been used by which worker groups.
    # This needs to be shared between all actor pools as multiple pools could be utilizing the same node.
    port_registry = actor_pool.ClusterPortRegistry()

    input_queue = Queue()
    input_queue.by_node_id[None].extend(pipeline_spec.input_data)
    del pipeline_spec.input_data
    initital_input_length = len(input_queue)

    queues = [Queue() for _ in pipeline_spec.stages]
    # Create a actor pool for each stage.
    pools: list[actor_pool.ActorPool] = []
    for idx, stage in enumerate(pipeline_spec.stages):
        assert isinstance(stage, specs.StageSpec)
        assert stage.slots_per_actor is not None
        wrapped_stage = specs.make_actor_pool_stage_from_stage_spec(pipeline_spec.config, stage, idx, cluster_resources)
        pool_params = actor_pool.PoolParams(
            enable_work_stealing=pipeline_spec.config.enable_work_stealing,
            max_tasks_to_poll_per_chunk=pipeline_spec.config.max_tasks_to_poll_per_chunk,
        )
        pools.append(
            actor_pool.ActorPool(
                worker_allocator,
                port_registry,
                wrapped_stage.stage,
                wrapped_stage.params,
                stage.name(idx),
                pool_params=pool_params,
            )
        )

    # Create a vector used to track whether a stages are finished or not.
    stage_is_dones = [False for _ in pools]

    autoscale_rate_limiter = timing.RateLimitChecker(1.0 / pipeline_spec.config.mode_specific.autoscale_interval_s)
    rate_limiter = timing.RateLimiter(_MAX_MAIN_LOOP_RATE_HZ)

    logger.debug("Starting main loop")

    streaming_loop_metrics = _StreamingLoopMetrics()
    last_stats: Optional[StreamingExecutorStats] = None
    with (
        Autoscaler(
            worker_allocator,
            pipeline_spec,
            cluster_resources,
            pipeline_spec.config.mode_specific.autoscaler_verbosity_level,
        ) as autoscaler,
        monitoring.PipelineMonitor(
            pipeline_spec.config.logging_interval_s,
            initital_input_length,
            pools,
            pipeline_spec.config.monitoring_verbosity_level,
        ) as monitor,
    ):
        # This is the loop which does all the interesting stuff. It was difficult to find the correct way to iterate
        # through this which managed backpressure the correct way.
        while True:
            new_stats = StreamingExecutorTiming(time.time())

            # Apply autoscale results if they are ready.
            autoscaler.apply_autoscale_result_if_ready(pools)
            new_stats.auto_scaling_apply_end = time.time()

            # Delete all the actors first. We do this as a separate step from "update()" because we may need
            # to clear room for new actors.
            for pool, is_done in zip(pools, stage_is_dones):
                if not is_done:
                    pool.delete_worker_groups()

            # Update all the pools.
            for pool, is_done in zip(pools, stage_is_dones):
                if not is_done:
                    pool.update()
            new_stats.pool_update_end = time.time()

            # Handle scaling the actor pools.
            # This should get called on the first loop through.
            if autoscale_rate_limiter.can_call():
                autoscaler.start_autoscale_calculation(pools, stage_is_dones)
            new_stats.auto_scaling_submit_end = time.time()

            # Grab stats from the pools
            pool_extra_metadatas = [deque.pop_all_deque_elements(x.task_extra_data) for x in pools]
            queue_lengths = [len(x) for x in queues]
            if monitor.update(len(input_queue), queue_lengths, pool_extra_metadatas) and (last_stats is not None):
                if pipeline_spec.config.mode_specific.executor_verbosity_level >= verbosity.VerbosityLevel.INFO:
                    logger.info(last_stats.to_log_string())
                logger.debug(f"Worker allocation:\n{worker_allocator.make_detailed_utilization_table()}")
            # update streaming loop timing metrics
            if last_stats is not None:
                streaming_loop_metrics.update(last_stats.timing)
            new_stats.monitor_update_end = time.time()

            autoscaler.add_measurements(pool_extra_metadatas)

            # In SERVING mode, poll the source queue for new-arriving requests.
            if pipeline_spec.config.execution_mode == specs.ExecutionMode.SERVING:
                assert pipeline_spec.serving_queues is not None
                try:
                    new_task = pipeline_spec.serving_queues.source.get(block=False)
                    if new_task is None:
                        # consider a None task as a termination signal
                        logger.info("Received None task which is the termination signal of serving mode")
                        break
                except python_queue.Empty:
                    pass
                else:
                    input_queue.by_node_id[None].append(new_task)

            # Add tasks if needed and able.
            # Start from the back and
            # NOTE: This is reversed.
            for idx, pool in reversed(list(enumerate(pools))):
                stage_batch_size = pipeline_spec.stages[idx].stage.stage_batch_size  # type: ignore
                assert stage_batch_size is not None
                output_queue = queues[idx]
                completed_tasks = deque.pop_all_deque_elements(pool.completed_tasks)

                is_last_stage = idx == len(pools) - 1
                is_first_stage = idx == 0
                if is_last_stage:
                    if pipeline_spec.serving_queues is not None:
                        # fetch the actual data and push to the sink; block until the sink can accept the results.
                        for task in completed_tasks:
                            for data_ref in task.task_data:
                                pipeline_spec.serving_queues.sink.put(ray.get(data_ref), block=True, timeout=None)
                    elif pipeline_spec.config.return_last_stage_outputs:
                        # Add to the output_queue for final return.
                        [output_queue.add_task(x) for x in completed_tasks]
                    else:
                        # If the last stage and the user did not ask for the results, we don't need to queue them.
                        pass
                else:
                    # pass to the next stage
                    [output_queue.add_task(x) for x in completed_tasks]

                while True:
                    # Deal with backpressure. We want to make sure that we have enough tasks to keep this stage busy,
                    # but not so many that we overwhelm the memory of the cluster.
                    # Previously, when Xenna operated at the task level (e.g. we could not break tasks into different
                    # numbers of samples), this was just num_tasks_completed_or_in_progress <= max_queued.
                    #
                    # We adapt this slightly for the case where track samples directly by taking into account the
                    # average number of samples per task.
                    num_tasks_in_progress = pool.num_used_slots + len(pool._task_queue)
                    avg_samples_per_task = output_queue.avg_samples_per_task()
                    if is_last_stage or avg_samples_per_task is None or avg_samples_per_task == 0:
                        num_tasks_completed = 0
                    else:
                        num_tasks_completed = len(output_queue) / avg_samples_per_task

                    next_stage_batch_size = -1
                    if not is_last_stage:
                        next_stage_batch_size = pipeline_spec.stages[idx + 1].stage.stage_batch_size  # type: ignore

                    max_queued = max(
                        int(
                            pool.num_ready_actors
                            * pool.slots_per_actor
                            * pipeline_spec.config.mode_specific.max_queued_multiplier
                        ),
                        pipeline_spec.config.mode_specific.max_queued_lower_bound,
                        next_stage_batch_size,
                    )

                    if num_tasks_in_progress + num_tasks_completed >= max_queued:
                        break
                    queue_to_pull_from = input_queue if idx == 0 else queues[idx - 1]
                    if not queue_to_pull_from:
                        break
                    # Get the next task to add to the pool.
                    maybe_task = queue_to_pull_from.maybe_get_batch(stage_batch_size)
                    if is_first_stage:  # Special case the first stage because we need to pull from input_queue.
                        if maybe_task is None:  # If there is some "remainder" in the input queue, add it.
                            task = queue_to_pull_from.maybe_get_batch(len(queue_to_pull_from))
                        elif maybe_task is None:  # There are not enough samples to fill the batch.
                            break
                        else:
                            task = maybe_task
                        logger.trace(f"Stage {idx} adding task: {task}")
                        pool.add_task(actor_pool.Task([ray.put(x) for x in task.task_data], None))
                    else:
                        if (
                            maybe_task is None and stage_is_dones[idx - 1]
                        ):  # If the last stage is done, and there is some "remainder" in the input queue, add it.
                            task = queue_to_pull_from.maybe_get_batch(len(queue_to_pull_from))
                        elif maybe_task is None:  # There are not enough samples to fill the batch.
                            break
                        else:
                            task = maybe_task
                        logger.trace(f"Stage {idx} adding task: {task}")
                        pool.add_task(task)  # type: ignore

            new_stats.add_tasks_end = time.time()
            # Determine if any stages are finished. If they are finished, mark them as done and stop the actor pool.
            # NOTE: This is iterating forward
            for idx, pool in enumerate(pools):
                # In SERVING mode, we never mark stages as done.
                if pipeline_spec.config.execution_mode == specs.ExecutionMode.SERVING:
                    break
                # A pool can never become "undone". Skip past anything which is already done.
                if stage_is_dones[idx]:
                    continue
                if idx == 0:  # Special case the first task.
                    is_done = not input_queue and not pool.has_work_or_completed
                else:
                    is_done = stage_is_dones[idx - 1] and not queues[idx - 1] and not pool.has_work_or_completed
                if is_done:
                    stage_is_dones[idx] = True
                    logger.info(f"Stopping stages {idx}")
                    pool.stop()
                else:
                    break

            # Handle finishing the pipeline
            if all(stage_is_dones):
                logger.info("All stages are done. Finishing pipeline.")
                break

            rate_limiter.sleep()
            new_stats.sleep_end = time.time()
            last_stats = StreamingExecutorStats(new_stats)
            logger.debug(f"Input queue: {len(input_queue)}")
            for idx, queue in enumerate(queues):
                logger.debug(f"Queue {idx}: {len(queue)}")
            logger.debug(last_stats.to_log_string())

    if pipeline_spec.config.return_last_stage_outputs:
        return ray.get(queues[-1].get_all_samples())
    else:
        return None
