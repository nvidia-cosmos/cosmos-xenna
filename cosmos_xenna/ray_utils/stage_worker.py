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

"""Code used to run the pipeline workers in a streaming Ray pipeline.

This module defines the `StageWorker` class, a Ray actor responsible for processing
tasks within a specific stage of a distributed pipeline. It manages the lifecycle
of a single worker, including setup, task execution, and shutdown.

The `StageWorker` employs a multi-threaded architecture to optimize task processing:
1.  **Downloader Thread**: Fetches task data (`ray.ObjectRef`) from the Ray object store.
    This happens when the data becomes available (signaled by `ray.wait`).
2.  **Deserializer Thread**: Deserializes the fetched task data using `ray.get`.
3.  **Processor Thread**: Executes the actual stage logic (`stage_interface.process_data`)
    on the deserialized data.

This separation allows downloading and deserialization to occur concurrently with
the main processing logic, improving throughput and resource utilization. The worker
also handles retries for setup and task execution based on stage parameters.

Profiling:
    The explicit separation of download and deserialization steps allows for detailed
    profiling of these operations, which is not directly provided by Ray's default mechanisms.

Concurrency:
    The `StageWorker` is designed to handle multiple concurrent calls to `process_data`,
    making it suitable for use within a Ray actor pool (`ActorPool`). Each call to
    `process_data` enqueues a task, which is then processed asynchronously by the internal
    threads.

Typical usage (managed by `ActorPool`):
    ```python
    # StageWorker is typically created and managed by an ActorPool
    # Direct instantiation might look like:
    worker_actor = StageWorker.options(...).remote(stage_interface, params, worker_resource)

    # Setup is called by the pool
    node_location = ray.get(worker_actor.setup.remote())

    # process_data is called by the pool for each task
    task_data_ref = ray.put(my_data)
    result_ref, metadata = worker_actor.process_data.remote(TaskData(task_data_ref))
    result_val = ray.get(result_ref)
    metadata_val = ray.get(metadata)
    ```

Note:
    This module is tightly coupled with the `ActorPool` and `stage.Interface`.
    Understanding Ray concepts (Actors, ObjectRefs, `ray.wait`, `ray.get`) is crucial.
"""

from __future__ import annotations

import abc
import asyncio
import copy
import queue
import threading
import time
import typing
import uuid
from typing import AsyncGenerator, Generic, Optional, Union

import attrs
import ray
import ray.experimental
from ray.util.metrics import Gauge

from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.ray_utils import stage
from cosmos_xenna.utils import gpu, retry
from cosmos_xenna.utils import python_log as logger

T = typing.TypeVar("T")
V = typing.TypeVar("V")


@attrs.define
class _Result(Generic[V]):
    """Internal container holding the final output and metadata for a processed task.

    Used to store results in the `_results` dictionary before they are retrieved
    by the `process_data` method call.
    """

    # Unique identifier for the task.
    uuid: str
    # The processed data returned by `stage_interface.process_data`.
    out_data: list[V]
    # Associated metadata (timing, failure info, etc.).
    extras: TaskResultMetadata


@attrs.define
class _TaskDataWithId(Generic[T]):
    """Internal container for a task that has been submitted but may not yet be downloaded.

    Includes the `ObjectRef` pointing to the data, a unique ID, timing info,
    and optionally the object size (added after download).
    """

    # References to the input data in the Ray object store.
    data_refs: list[ray.ObjectRef[T]]
    # Unique identifier for the task.
    uuid: str
    # Timing information collected so far.
    timing: TimingInfo
    # Size of the serialized data (known after download).
    object_sizes: list[int] = attrs.field(factory=list)


@attrs.define
class _DeserializedTaskDataWithId(Generic[T]):
    """Internal container for a task whose data has been downloaded and deserialized.

    Holds the actual deserialized data object, ready for processing.
    """

    # References to the input data in the Ray object store.
    data_refs: list[ray.ObjectRef[T]]
    # The actual deserialized input data.
    data: list[T]
    # Unique identifier for the task.
    uuid: str
    # Timing information collected so far.
    timing: TimingInfo
    # Size of the original serialized data.
    object_sizes: list[int]


@attrs.define
class _ProcessDataResult(Generic[V]):
    """Internal container for the result of the `_process_data` internal method.

    Bundles the output data with failure information determined during processing.
    """

    # The processed data, or None if processing failed/was skipped.
    out_data: list[V]
    # Information about potential failures and restart signals.
    pool_info: FailureInfo


@attrs.define
class TaskData(Generic[T]):
    """Represents a task submitted to the worker for processing.

    This is the input structure expected by the public `process_data` method.
    """

    # References to the data objects in the Ray object store.
    data_refs: list[ray.ObjectRef[T]]


@attrs.define
class TimingInfo:
    """Stores timestamps for various stages of task processing within the worker.

    Allows calculating durations for different phases (pull, deserialize, process).
    Timestamps are captured using `time.time()`.
    """

    # Time when the task was initially submitted via `process_data`.
    requested_s: float = 0.0
    # Time when `ray.wait` indicated the data ref was available locally.
    pull_s: float = 0.0
    # Time just before calling `ray.get`.
    deserialize_start_s: float = 0.0
    # Time just after `ray.get` returned.
    deserialize_end_s: float = 0.0
    # Time just before calling `stage_interface.process_data`.
    process_start_time_s: float = 0.0
    # Time just after `stage_interface.process_data` returned.
    process_end_time_s: float = 0.0

    @property
    def pull_dur(self) -> float | None:
        """Duration (in seconds) Ray spent making the data available locally."""
        if self.pull_s and self.requested_s:
            return self.pull_s - self.requested_s
        else:
            return None

    @property
    def deserialize_dur(self) -> float | None:
        """Duration (in seconds) spent deserializing the data via `ray.get`."""
        if self.deserialize_end_s and self.deserialize_start_s:
            return self.deserialize_end_s - self.deserialize_start_s
        else:
            return None

    @property
    def process_dur(self) -> float | None:
        """Duration (in seconds) spent executing the stage's process_data method."""
        if self.process_end_time_s and self.process_start_time_s:
            return self.process_end_time_s - self.process_start_time_s
        else:
            return None


@attrs.define
class FailureInfo:
    """Contains information about task processing outcome relevant to the ActorPool.

    Signals whether the task result should be passed to the next stage and whether
    the worker encountered an error that warrants a restart.
    """

    # If False, the ActorPool will not submit this task's result (None) to the next stage.
    should_process_further: bool
    # If True, signals to the ActorPool that this worker should be killed and potentially replaced.
    # Set based on stage_params.restart_workers_on_failure if an exception occurs during processing.
    should_restart_worker: bool
    # Deprecated/Unused? Original comment mentioned ignoring tasks, but logic seems tied to should_process_further.
    failures_return_nones: bool = False


@attrs.define
class TaskDataInfo:
    """Information about the input data size for a processed task."""

    # Size (in bytes) of the task data in the Ray object store.
    serialized_input_size: int


@attrs.define
class TaskResultMetadata:
    """Aggregated metadata returned alongside the processed data reference.

    Contains timing details, failure information, and input data size.
    This is the second return value of the public `process_data` method.
    """

    timing: TimingInfo
    failure_info: FailureInfo
    task_data_info: TaskDataInfo
    num_returns: int


def _get_object_size(ref: ray.ObjectRef) -> int:
    """Gets the size of an object store in Ray's object store."""
    # Get object locations
    locations = ray.experimental.get_object_locations([ref])

    assert ref in locations
    assert "object_size" in locations[ref]
    return int(locations[ref]["object_size"])


@ray.remote
class StageWorker(abc.ABC, Generic[T, V]):
    """A Ray actor that executes a specific stage (`stage.Interface`) of a pipeline.

    Manages the processing of tasks submitted via `process_data`. It uses internal
    queues and threads to handle data fetching (download), deserialization, and
    the actual execution of the stage logic defined in `stage_interface`.

    Attributes:
        task_queue: Queue for tasks submitted but not yet downloaded (_TaskDataWithId).
        downloaded_queue: Queue for tasks downloaded but not yet deserialized (_TaskDataWithId).
        deserialized_queue: Queue for tasks deserialized and ready for processing (_DeserializedTaskDataWithId).
        _results: Dictionary storing results (_Results) of completed tasks, keyed by UUID.
        results_lock: Lock protecting access to `_results` and `_error_map`.
        stop_flag: Event used to signal termination to the worker threads.
        _stage_interface: The user-defined logic for this pipeline stage.
        _params: Configuration parameters for the stage execution (retries, error handling).
        _worker: Information about the worker resources and ID.
        _is_setup: Flag indicating if the `setup` method has completed successfully.
        _node_location: Cached node ID where this actor is running.
        _downloader_thread: Thread executing `_downloader_loop`.
        _deserializer_thread: Thread executing `_deserializer_loop`.
        _process_data_thread: Thread executing `_process_data_loop`.
        _error_map: Stores exceptions encountered during processing for specific tasks.
        _global_error: Stores fatal exceptions that stop the entire worker.

    Why Manual Download/Deserialize Threads?
        Ray's default actor method invocation implicitly handles data fetching and
        deserialization. However, separating these into explicit threads allows:
        1.  **Fine-grained Profiling:** Measure time spent in `ray.wait` (download/pull)
            and `ray.get` (deserialization) distinctly from processing time.
        2.  **Concurrency:** Allows download and deserialization of subsequent tasks
            to overlap with the processing of the current task.
    """

    def __init__(
        self,
        stage_interface: stage.Interface,
        params: stage.Params,
        stage_name: str,
        metadata: resources.WorkerMetadata,
    ) -> None:
        """Initializes the StageWorker actor.

        Args:
            stage_interface: An instance defining the stage's setup and processing logic.
            params: Configuration for the stage's execution (retries, failure handling).
            worker: Resource information associated with this specific worker instance.
        """
        self._stage_interface = stage_interface
        self._params = params
        self._stage_name = stage_name
        self._is_setup = False
        self._node_location: str | None = None
        self._metadata = metadata

        # Tasks waiting to be downloaded
        self.task_queue: queue.Queue[_TaskDataWithId[T]] = queue.Queue()
        # Tasks waiting to be deserialized
        self.downloaded_queue: queue.Queue[_TaskDataWithId[T]] = queue.Queue()
        # Tasks waiting to be processed
        self.deserialized_queue: queue.Queue[_DeserializedTaskDataWithId[T]] = queue.Queue()
        # Completed tasks
        self._results: dict[str, _Result[V]] = {}
        self.results_lock: threading.Lock = threading.Lock()
        self.stop_flag = threading.Event()

        # Synchronization for setup in processing thread
        self._setup_requested = threading.Event()
        self._setup_completed = threading.Event()
        self._setup_error: Exception | None = None

        # Synchronization for setup_on_node in processing thread
        self._setup_on_node_requested = threading.Event()
        self._setup_on_node_completed = threading.Event()
        self._setup_on_node_error: Exception | None = None

        self._downloader_thread = threading.Thread(target=self._downloader_loop)
        self._downloader_thread.start()

        self._deserializer_thread = threading.Thread(target=self._deserializer_loop)
        self._deserializer_thread.start()

        self._process_data_thread = threading.Thread(target=self._process_data_loop)
        self._process_data_thread.start()

        self._error_map: dict[str, Exception] = {}
        self._global_error: Exception | None = None

        # GPU metrics
        self._metrics_gpu_alloc = Gauge(
            "pipeline_stage_gpu_alloc",
            description="Number of GPUs allocated to this stage",
            tag_keys=("stage", "ActorId", "GpuIndex"),
        )

    def _get_node_location(self) -> str:
        if self._node_location is None:
            self._node_location = str(ray.get_runtime_context().get_node_id())
        return self._node_location

    @ray.method(retry_exceptions=False)  # type: ignore
    async def setup_on_node(self) -> None:
        """Setup the actor per node.

        This is guranteed to be called exactly once by the actor pool for each node.

        For example, if we had 8 actors running on a node. This would get called once, on a single actor. NOT 8 times,
        once per actor.

        This is useful if you need to do per node setup. For example, you may need to download weights to the node from
        object storage.
        """
        if (
            self._metadata.should_set_cuda_visible_devices
            and self._metadata.allocation.gpus
            and gpu.get_num_gpus() == 0
        ):
            raise RuntimeError(
                "Worker is a GPU worker, but no GPUs are available. This likely means that the ray cluster was not "
                "started with 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0'. Xenna needs this env variable to be set "
                "before cluster creation as it works around ray's gpu allocation mechanisms."
            )

        # Signal the processing thread to perform setup_on_node
        logger.trace("Requesting setup_on_node in processing thread")
        self._setup_on_node_requested.set()

        # Wait for setup_on_node to complete using asyncio
        logger.trace("Waiting for setup_on_node to complete")
        while not self._setup_on_node_completed.is_set():
            await asyncio.sleep(0.01)
        logger.trace("Setup_on_node completed")

        # Check if setup_on_node failed
        if self._setup_on_node_error:
            raise self._setup_on_node_error

    @ray.method(retry_exceptions=False)  # type: ignore
    async def setup(self) -> str:
        """Sets up the worker by calling the stage's setup method.

        This method delegates the actual setup to the processing thread to ensure
        that setup and process_data run in the same thread context. This is critical
        for PyTorch/CUDA contexts and other thread-local state.

        Returns:
            The Ray node ID (as a string) where this worker is located.

        Raises:
            Exception: If setup fails after all retry attempts.
        """
        if (
            self._metadata.should_set_cuda_visible_devices
            and self._metadata.allocation.gpus
            and gpu.get_num_gpus() == 0
        ):
            raise RuntimeError(
                "Worker is a GPU worker, but no GPUs are available. This likely means that the ray cluster was not "
                "started with 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0'. Xenna needs this env variable to be set "
                "before cluster creation as it works around ray's gpu allocation mechanisms."
            )

        # Signal the processing thread to perform setup
        logger.trace("Requesting setup in processing thread")
        self._setup_requested.set()

        # Wait for setup to complete using asyncio
        logger.trace("Waiting for setup to complete")
        while not self._setup_completed.is_set():
            await asyncio.sleep(0.01)
        logger.trace("Setup completed")

        # Check if setup failed
        if self._setup_error:
            raise self._setup_error

        node_location = self._get_node_location()
        logger.debug(f"Finished setting up actor for stage={self._params.name}")

        # metrics
        for gpu_alloc in self._metadata.allocation.gpus:
            self._metrics_gpu_alloc.set(
                gpu_alloc.used_fraction,
                tags={
                    "stage": self._params.name,
                    "ActorId": self._metadata.worker_id,
                    "GpuIndex": str(gpu_alloc.index),
                },
            )

        return node_location

    @ray.method(retry_exceptions=False)  # type: ignore
    async def process_data(self, task_data: TaskData[T]) -> AsyncGenerator[Union[V, TaskResultMetadata], None]:
        """Submits a task for asynchronous processing by the worker's internal threads.

        This method is the main entry point for tasks. It assigns a unique ID to the
        task, places it in the initial `task_queue`, and then blocks until the task
        has been processed by the downloader, deserializer, and processor threads,
        or until an error occurs.

        Args:
            task_data: A TaskData object containing the `ObjectRef` of the input data.

        Returns:
            A tuple containing:
            1.  The processed data (or None if processing failed/skipped).
            2.  `TaskResultMetadata` object with timing, failure, and size info.

        Raises:
            Exception: If a processing error occurs for this task within the worker threads,
                       or if a global error (`_global_error`) has stopped the worker.
        """
        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Put the task in the queue
        self.task_queue.put(_TaskDataWithId(task_data.data_refs, task_id, TimingInfo(requested_s=time.time())))

        result: Optional[_Result[V]] = None
        # Block until the result is available and propogate any errors using asyncio.
        while True:
            if self._global_error:
                raise self._global_error

            with self.results_lock:
                exception = self._error_map.pop(task_id, None)
                if exception is not None:
                    raise exception
                result = self._results.pop(task_id, None)
            if result is None:
                await asyncio.sleep(0.01)
            else:
                break
        yield result.extras
        for item in result.out_data:
            yield item

    @ray.method(retry_exceptions=False)  # type: ignore
    def cancel_task(self, task_data: TaskData[T]) -> bool:
        """Attempts to cancel a task that has been submitted but not yet started.

        Best-effort removal from internal queues; returns True if the task was found and removed
        from either the submission queue or the downloaded-but-not-deserialized queue.
        If the task has already begun deserialization or processing, cancellation will likely fail
        and this returns False.
        """

        # Helper to remove a matching task from a Queue by draining and reconstructing it.
        def _remove_from_queue(
            q: queue.Queue[_TaskDataWithId[T]] | queue.Queue[_DeserializedTaskDataWithId[T]],
            refs: list[ray.ObjectRef[T]],
        ) -> bool:
            items: list[_TaskDataWithId[T] | _DeserializedTaskDataWithId[T]] = []
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                items.append(item)

            removed_ = False
            # Restore remaining items preserving order as much as possible
            for it in items:
                if it.data_refs == refs:
                    removed_ = True
                    continue
                q.put_nowait(it)  # type: ignore
            return removed_

        refs = task_data.data_refs
        # Try to remove from the submission queue first
        removed = False
        if _remove_from_queue(self.task_queue, refs):
            removed = True
        # If downloader already moved it, try downloaded queue
        if _remove_from_queue(self.downloaded_queue, refs):
            removed = True
        # If deserializer already moved it, try deserialized queue
        if _remove_from_queue(self.deserialized_queue, refs):
            removed = True
        # We cannot reliably cancel from deserialized_queue or once processing has started.
        if not removed:
            logger.debug(f"Failed to remove task {task_data} from queues.")
            logger.debug(f"Task queue: {[item.data_refs for item in self.task_queue.queue]}")
            logger.debug(f"Downloaded queue: {[item.data_refs for item in self.downloaded_queue.queue]}")
            logger.debug(f"Deserialized queue: {[item.data_refs for item in self.deserialized_queue.queue]}")
        return removed

    def _downloader_step(self, local_tasks: dict[str, _TaskDataWithId[T]]) -> None:
        """Performs one iteration of the download loop.

        Checks the `task_queue` for new tasks and adds them to `local_tasks`.
        Uses `ray.wait` to efficiently check which `ObjectRef`s associated with waiting
        tasks in `local_tasks` are now available locally. Once *all* refs for a task
        are ready, moves the task to the `downloaded_queue` after fetching object sizes.

        Args:
            local_tasks: A dictionary holding tasks currently being waited on by this thread.
                         Tasks are added from `task_queue` and removed when fully downloaded.
        """
        # 1. Pull new tasks from the queue and add them to local_tasks.
        while True:
            try:
                # Use timeout=0 for non-blocking check
                task = self.task_queue.get(timeout=0.01)
                if task.uuid not in local_tasks:
                    local_tasks[task.uuid] = task
            except queue.Empty:
                break  # No more tasks in the queue for now

        # 2. Check if there are any tasks waiting locally.
        if not local_tasks:
            # Sleep briefly if no work to do, prevents busy-waiting
            time.sleep(0.01)
            return

        # 3. Create a flattened list of all ObjectRefs from all waiting tasks.
        all_refs_to_wait: list[ray.ObjectRef] = []
        for task in local_tasks.values():
            if not task.data_refs:
                raise ValueError("Task has no data refs")
            all_refs_to_wait.extend(task.data_refs)

        # 4. Call ray.wait on all refs.
        ready_refs_list, _ = ray.wait(
            all_refs_to_wait,
            num_returns=len(all_refs_to_wait),  # Wait for all possible refs
            timeout=0,  # Don't block, return immediately with what's ready
            fetch_local=True,
        )

        if not ready_refs_list:
            # Sleep briefly if nothing became ready, prevents busy-waiting
            time.sleep(0.01)
            return

        pull_time = time.time()
        ready_refs_set = set(ready_refs_list)  # Use a set for faster lookups

        # 5. Identify completed tasks and move them to the downloaded_queue.
        # Iterate over a copy of keys to allow modification of local_tasks dict during iteration
        completed_task_ids = []
        for task_id in list(local_tasks.keys()):
            task = local_tasks[task_id]
            # Check if *all* refs for this task are in the ready set
            if all(ref in ready_refs_set for ref in task.data_refs):
                # Task is fully ready!
                task.timing.pull_s = pull_time

                # Calculate object sizes for all refs in the task
                task.object_sizes = [_get_object_size(ref) for ref in task.data_refs]

                # Put the fully ready task onto the next queue
                self.downloaded_queue.put_nowait(task)

                # Remove the completed task from local tracking
                completed_task_ids.append(task_id)

        # Clean up completed tasks from local_tasks
        for task_id in completed_task_ids:
            del local_tasks[task_id]

    def _downloader_loop(self) -> None:
        """Main loop for the downloader thread.

        Continuously calls `_downloader_step` until the `stop_flag` is set.
        Handles exceptions by calling `_handle_thread_exception`.
        """
        local_tasks: dict[str, _TaskDataWithId[T]] = {}
        while not self.stop_flag.is_set():
            try:
                self._downloader_step(local_tasks)
            except Exception as e:  # noqa: BLE001
                self._handle_thread_exception("downloader", None, e)

    def _deserializer_step(self) -> None:
        """Performs one iteration of the deserialization loop.

        Takes a task from the `downloaded_queue`, calls `ray.get` to deserialize
        the data, records timing information, and places the deserialized task
        into the `deserialized_queue`.
        """
        try:
            task = self.downloaded_queue.get(timeout=0.01)
        except queue.Empty:
            return

        start_time = time.time()
        # Call ray.get on the list of references
        # ray.get handles lists of ObjectRefs, returning a list of results
        result: list[T] = ray.get(task.data_refs)
        out_timing = task.timing
        out_timing.deserialize_start_s = start_time
        out_timing.deserialize_end_s = time.time()
        # Pass the list of object sizes calculated in the downloader
        assert task.object_sizes is not None  # Should have been populated by downloader
        self.deserialized_queue.put_nowait(
            _DeserializedTaskDataWithId(task.data_refs, result, task.uuid, out_timing, task.object_sizes)
        )

    def _deserializer_loop(self) -> None:
        """Main loop for the deserializer thread.

        Continuously calls `_deserializer_step` until the `stop_flag` is set.
        Handles exceptions by calling `_handle_thread_exception`.
        """
        while not self.stop_flag.is_set():
            try:
                self._deserializer_step()
            except Exception as e:  # noqa: BLE001
                self._handle_thread_exception("deserializer", None, e)

    def _process_step(self) -> None:
        """Performs one iteration of the data processing loop.

        Takes a deserialized task from the `deserialized_queue`, calls the internal
        `_process_data` method (which wraps the user's stage logic), records timing,
        and stores the final result or error in the shared `_results` or `_error_map`.
        """
        try:
            task = self.deserialized_queue.get(timeout=0.01)
        except queue.Empty:
            return

        try:
            start_time = time.time()
            result = self._process_data(task.data)
            done_time = time.time()

            out_timing = task.timing
            out_timing.process_start_time_s = start_time
            out_timing.process_end_time_s = done_time
            with self.results_lock:
                self._results[task.uuid] = _Result(
                    task.uuid,
                    result.out_data,
                    TaskResultMetadata(
                        out_timing, result.pool_info, TaskDataInfo(sum(task.object_sizes)), len(result.out_data)
                    ),
                )
        except Exception as e:  # noqa: BLE001
            with self.results_lock:
                self._error_map[task.uuid] = e

    def _process_data_loop(self) -> None:
        """Main loop for the processing thread.

        First waits for either setup_on_node or setup to be requested, performs the first one called,
        then continuously processes tasks. Handles exceptions by calling `_handle_thread_exception`.
        """
        # Wait for either setup_on_node or setup to be requested
        while not self.stop_flag.is_set():
            if self._setup_on_node_requested.is_set():
                logger.trace("Performing setup_on_node in processing thread")
                self._perform_setup_on_node_in_thread()
                logger.trace("Finished performing setup_on_node in processing thread")
                self._setup_on_node_requested.clear()
                # Don't break out of the loop here as we still need to perform setup.
            elif self._setup_requested.is_set():
                logger.trace("Performing setup in processing thread")
                self._perform_setup_in_thread()
                logger.trace("Finished performing setup in processing thread")
                self._setup_requested.clear()
                break  # Break out of the loop as we are now ready to process data.
            time.sleep(0.01)

        # Now process tasks continuously
        while not self.stop_flag.is_set():
            try:
                self._process_step()
            except Exception as e:  # noqa: BLE001
                self._handle_thread_exception("process_data", None, e)

    def _perform_setup_on_node_in_thread(self) -> None:
        """Performs the actual setup_on_node in the processing thread.

        This ensures that setup_on_node and process_data run in the same thread context,
        which is critical for PyTorch/CUDA contexts and other thread-local state.
        """
        try:
            node_location = self._get_node_location()
            with stage.open_context(self._params.logging_context):

                def func_to_call() -> None:
                    return self._stage_interface.setup_on_node(
                        resources.NodeInfo(node_location), copy.deepcopy(self._metadata)
                    )

                logger.debug(
                    f"Setting up actor for stage={self._params.name} on node={node_location} in processing thread"
                )
                retry.do_with_retries(func_to_call, max_attempts=self._params.num_node_setup_retries)
                logger.debug(
                    f"Finished setting up actor for stage={self._params.name} on node={node_location} in processing "
                    "thread"
                )

        except Exception as e:  # noqa: BLE001
            # Store the error for the main thread to pick up
            self._setup_on_node_error = e
            logger.error(f"Setup_on_node failed in processing thread for stage={self._params.name}")
            logger.exception("Setup_on_node failed in processing thread")
        finally:
            # Always signal completion, whether success or failure
            self._setup_on_node_completed.set()

    def _perform_setup_in_thread(self) -> None:
        """Performs the actual setup in the processing thread.

        This ensures that setup and process_data run in the same thread context,
        which is critical for PyTorch/CUDA contexts and other thread-local state.
        """
        try:
            with stage.open_context(self._params.logging_context):

                def func_to_call() -> None:
                    return self._stage_interface.setup(copy.deepcopy(self._metadata))

                logger.debug(f"Setting up actor for stage={self._params.name} in processing thread")
                retry.do_with_retries(func_to_call, max_attempts=self._params.num_setup_retries)
                self._is_setup = True
                logger.debug(f"Finished setting up actor for stage={self._params.name} in processing thread")

        except Exception as e:  # noqa: BLE001
            # Store the error for the main thread to pick up
            self._setup_error = e
            logger.error(f"Setup failed in processing thread for stage={self._params.name}")
            logger.exception("Setup failed in processing thread")
        finally:
            # Always signal completion, whether success or failure
            self._setup_completed.set()

    # TODO: We need to provide each attempt here with a new copy of the input data.
    # As written, the input data can be modified in place, which can cause problems.
    # Maybe it's best to keep the data on the object store and grab it fresh with each attempt.
    # This avoids extra memory usage, but it does mean extra deserialization time.
    def _process_data(self, in_data: list[T]) -> _ProcessDataResult:
        """Wraps the actual call to the user-defined stage processing logic.

        Calls `self._stage_interface.process_data` with the deserialized input data.
        Handles retries based on `self._params.num_run_retries` and manages
        failure reporting based on `self._params.ignore_failures` and
        `self._params.restart_workers_on_failure`.

        Args:
            in_data: The deserialized input data for the task.

        Returns:
            A `_ProcessDataResult` containing the output data and failure information.

        Raises:
            Exception: If processing fails after retries and `ignore_failures` is False.
        """
        if not self._is_setup:
            raise RuntimeError(
                "Error, this worker has not been set up. The setup() method must be called before process_data(). "
                "This indicates a bug in the ActorPool or pipeline orchestration."
            )

        def func_to_call() -> list[V]:
            return self._stage_interface.process_data(in_data)

        try:
            result = retry.do_with_retries(func_to_call, max_attempts=self._params.num_run_retries)
            return _ProcessDataResult(
                result,
                FailureInfo(should_process_further=result is not None, should_restart_worker=False),
            )
        except Exception as e:
            if self._params.ignore_failures:
                logger.error("Ignoring an exception")
                logger.exception("Ignoring an exception")
                return _ProcessDataResult(
                    [],
                    FailureInfo(
                        should_process_further=False,
                        should_restart_worker=self._params.restart_workers_on_failure,
                    ),
                )
            else:
                logger.error("Got an exception in process_data")
                logger.exception("Got an exception in process_data")
                raise e

    def _handle_thread_exception(self, thread_name: str, task_id: str | None, e: Exception) -> None:
        """Handles exceptions occurring within the worker threads.

        Logs the exception, sets the `_global_error` flag to indicate a fatal error,
        and triggers the `stop_flag` to terminate all worker threads.

        Args:
            thread_name: Name of the thread where the exception occurred ("downloader", "deserializer", "process_data").
            task_id: The UUID of the task being processed when the error occurred (if applicable).
            e: The exception object.
        """
        logger.exception(f"Error in {thread_name}. Killing worker.")
        self._global_error = e
        self.stop_flag.set()  # Stop all threads

    def shutdown(self) -> None:
        """Signals the worker threads to stop and waits for them to exit."""
        self.stop_flag.set()
        self._downloader_thread.join()
        self._deserializer_thread.join()
        self._process_data_thread.join()

    def __repr__(self) -> str:
        """Return the name for this stage.

        Ray uses this method to add context to the log, so set it to the stage name.
        """

        # This is apparently used by Ray sometimes when the object's __init__ method hasn't been called.
        # Fallback in this case.
        if hasattr(self, "_stage_name"):
            return self._stage_name
        else:
            return "StageWorker"
