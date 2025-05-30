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
import random
import time
import typing
from typing import Optional

import attrs
import ray
import ray.util.state
from loguru import logger

from cosmos_xenna.pipelines.private import monitoring, specs
from cosmos_xenna.pipelines.private.scheduling import autoscaling_algorithms, data_structures
from cosmos_xenna.ray_utils import actor_pool, allocator, resources, stage_worker
from cosmos_xenna.utils import approx, deque, timing, verbosity

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_MAX_MAIN_LOOP_RATE_HZ = 100
_VERBOSE = False


@attrs.define
class StreamingExecutorTiming:
    iteration_start: float = 0.0
    pool_update_end: float = 0.0
    monitor_update_end: float = 0.0
    add_tasks_end: float = 0.0
    auto_scaling_end: float = 0.0
    sleep_end: float = 0.0

    def to_log_string(self) -> str:
        stages: list[tuple[str, float]] = [
            ("Auto Scaling", self.auto_scaling_end - self.iteration_start),
            ("Pool Update", self.pool_update_end - self.auto_scaling_end),
            ("Monitor Update", self.monitor_update_end - self.pool_update_end),
            ("Add Tasks", self.add_tasks_end - self.monitor_update_end),
            ("Sleep", self.sleep_end - self.add_tasks_end),
            ("Total", self.sleep_end - self.iteration_start),
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


@attrs.define
class AutoscalerResultForStage:
    slots_per_worker: int
    new_workers: list[resources.Worker]
    workers_to_delete: list[resources.Worker]


class Autoscaler:
    def __init__(
        self,
        worker_allocator: allocator.WorkerAllocator,
        pipeline_spec: specs.PipelineSpec,
        cluster_resources: resources.ClusterResources,
        verbosity_level: verbosity.VerbosityLevel = verbosity.VerbosityLevel.NONE,
    ) -> None:
        self._verbosity_level = verbosity_level
        self._allocator = worker_allocator
        self._algorithm = autoscaling_algorithms.FragmentationBasedAutoscaler(verbosity_level=verbosity_level)
        self._algorithm.setup(data_structures.Problem.make_from_pipeline_spec(pipeline_spec, cluster_resources))

    def add_measurements(
        self,
        task_metadata_per_pool: list[list[stage_worker.TaskResultMetadata]],
    ) -> None:
        stage_measurements = []
        for metadatas in task_metadata_per_pool:
            stage_measurements.append(
                data_structures.StageMeasurements(
                    [data_structures.TaskMeasurement.make_from_task_metadata(x) for x in metadatas]
                )
            )
        measurements = data_structures.Measurements(time.time(), stage_measurements)
        self._algorithm.update_with_measurements(time.time(), measurements)

    def _make_problem_state(
        self, actor_pools: list[actor_pool.ActorPool], stages_is_dones: list[bool]
    ) -> data_structures.ProblemState:
        stages = []
        for pool, is_done in zip(actor_pools, stages_is_dones):
            workers = self._allocator.get_workers_in_stage(pool.name)
            stages.append(
                data_structures.ProblemStageState(
                    pool.name,
                    [data_structures.ProblemWorkerState.make_from_worker_state(w) for w in workers],
                    pool.slots_per_actor,
                    is_done,
                )
            )
        return data_structures.ProblemState(stages)

    def update(self, pools: list[actor_pool.ActorPool], stages_is_dones: list[bool]) -> None:
        autoscale_result = self._algorithm.autoscale(
            time.time(),
            self._make_problem_state(pools, stages_is_dones),
        )
        if self._verbosity_level > verbosity.VerbosityLevel.INFO:
            logger.info(f"Autoscale result:\n{autoscale_result}")

        for result, pool in zip(autoscale_result.stages, pools):
            pool.set_num_slots_per_actor(result.slots_per_worker)
            for w in result.new_workers:
                pool.add_actor_to_create(w.to_worker(pool.name))

            for w in result.deleted_workers:
                pool.add_actor_to_delete(w.to_worker(pool.name))


def _verify_enough_resources(pipeline_spec: specs.PipelineSpec, cluster_resources: resources.ClusterResources) -> None:
    cluster_resource_totals = cluster_resources.totals()

    required_resources = resources.PoolOfResources()

    for stage in pipeline_spec.stages:
        assert isinstance(stage, specs.StageSpec)
        if stage.num_workers is not None:
            num_required = stage.num_workers
        elif stage.num_workers_per_node is not None:
            num_required = stage.num_workers_per_node * cluster_resources.num_nodes
        else:
            num_required = 1

        resources_per_worker = stage.stage.required_resources.to_pool_of_resources(
            cluster_resources.calc_num_nvdecs_per_gpu(), cluster_resources.calc_num_nvencs_per_gpu()
        )
        required_resources += resources_per_worker.mutiply_by(num_required)

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
    if approx.float_lt(cluster_resource_totals.nvdecs, required_resources.nvdecs):
        raise ValueError(
            f"Not enough NVDEC resources to run pipeline in streaming mode. Pipeline requires "
            f"{required_resources.nvdecs} but only {cluster_resource_totals.nvdecs} are available.\n{summary_string}"
        )
    if approx.float_lt(cluster_resource_totals.nvencs, required_resources.nvencs):
        raise ValueError(
            f"Not enough NVENC resources to run pipeline in streaming mode. Pipeline requires "
            f"{required_resources.nvencs} but only {cluster_resource_totals.nvencs} are available.\n{summary_string}"
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
    worker_allocator = allocator.WorkerAllocator(cluster_resources)

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
        wrapped_stage = specs.make_actor_pool_stage_from_stage_spec(pipeline_spec.config, stage, idx)
        pools.append(
            actor_pool.ActorPool(
                worker_allocator,
                wrapped_stage.stage,
                wrapped_stage.params,
                stage.name(idx),
                verbosity_level=pipeline_spec.config.actor_pool_verbosity_level,
            )
        )

    # Create a vector used to track whether a stages are finished or not.
    stage_is_dones = [False for _ in pools]

    autoscale_rate_limiter = timing.RateLimitChecker(1.0 / pipeline_spec.config.mode_specific.autoscale_interval_s)
    rate_limiter = timing.RateLimiter(_MAX_MAIN_LOOP_RATE_HZ)

    autoscaler = Autoscaler(
        worker_allocator,
        pipeline_spec,
        cluster_resources,
        pipeline_spec.config.mode_specific.autoscaler_verbosity_level,
    )

    logger.info("Starting main loop")

    last_stats: Optional[StreamingExecutorStats] = None
    with monitoring.PipelineMonitor(
        pipeline_spec.config.logging_interval_s,
        initital_input_length,
        pools,
        pipeline_spec.config.monitoring_verbosity_level,
    ) as monitor:
        # This is the loop which does all the interesting stuff. It was difficult to find the correct way to iterate
        # through this which managed backpressure the correct way.
        while True:
            new_stats = StreamingExecutorTiming(time.time())

            # Handle scaling the actor pools.
            # This should get called on the first loop through.
            if autoscale_rate_limiter.can_call():
                if pipeline_spec.config.mode_specific.executor_verbosity_level >= verbosity.VerbosityLevel.INFO:
                    logger.info("Autoscaling...")
                autoscaler.update(pools, stage_is_dones)
                if pipeline_spec.config.mode_specific.executor_verbosity_level >= verbosity.VerbosityLevel.INFO:
                    logger.info("Done calculating autoscaling...")
            new_stats.auto_scaling_end = time.time()

            # Delete all the actors first. We do this as a separate step from "update()" because we may need
            # to clear room for new actors.
            for pool, is_done in zip(pools, stage_is_dones):
                if not is_done:
                    pool.delete_actors()

            # Update all the pools.
            for pool, is_done in zip(pools, stage_is_dones):
                if not is_done:
                    pool.update()
            new_stats.pool_update_end = time.time()

            # Grab stats from the pools
            pool_extra_metadatas = [deque.pop_all_deque_elements(x.task_extra_data) for x in pools]

            if monitor.update(len(input_queue), len(queues[-1]), pool_extra_metadatas) and (last_stats is not None):
                if pipeline_spec.config.mode_specific.executor_verbosity_level >= verbosity.VerbosityLevel.INFO:
                    logger.info(last_stats.to_log_string())
                if pipeline_spec.config.log_worker_allocation_layout:
                    logger.info(f"Worker allocation:\n{worker_allocator.make_detailed_utilization_table()}")
            new_stats.monitor_update_end = time.time()

            autoscaler.add_measurements(pool_extra_metadatas)

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
                if is_last_stage and not pipeline_spec.config.return_last_stage_outputs:
                    # If the last stage and the user did not ask for the results, we don't need to queue them.
                    pass
                else:
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
                    max_queued = pool.num_ready_actors * pool.slots_per_actor
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
                        if _VERBOSE:
                            logger.info(f"Stage {idx} adding task: {task}")
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
                        if _VERBOSE:
                            logger.info(f"Stage {idx} adding task: {task}")
                        pool.add_task(task)  # type: ignore

            new_stats.add_tasks_end = time.time()
            # Determine if any stages are finished. If they are finished, mark them as done and stop the actor pool.
            # NOTE: This is iterating forward
            for idx, pool in enumerate(pools):
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
            if _VERBOSE:
                logger.info(f"Input queue: {len(input_queue)}")
                for idx, queue in enumerate(queues):
                    logger.info(f"Queue {idx}: {len(queue)}")
                logger.info(last_stats.to_log_string())

    if pipeline_spec.config.return_last_stage_outputs:
        return ray.get(queues[-1].get_all_samples())
    else:
        return None
