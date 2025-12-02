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

import abc
import copy
import enum
import multiprocessing
import typing
from typing import Any, Generic, Optional, Sequence

import attrs

from cosmos_xenna import file_distribution
from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.ray_utils import runtime_envs, stage
from cosmos_xenna.utils import approx
from cosmos_xenna.utils.verbosity import VerbosityLevel

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_DEFAULT_SLOTS_PER_ACTOR = 2
_DEFAULT_LOG_INTERVAL_S = 60


class ExecutionMode(enum.Enum):
    """
    Enumeration for the mode of execution, defining how data processing should be handled.

    See README.md for more info.
    """

    # All stages are processing data concurrently. This is usually what we want to run when running
    # production workloads on the cloud
    STREAMING = 0
    # Stages are processed sequentially and all the data is materialized between stages. Generally,
    # we cannot run this for production workloads as we generally do not have enough local storage to
    # materialize intermediate data products. Additionally, for many pipelines, streaming mode can be
    # more efficient at processing.
    BATCH = 1
    # Online serving mode where stages run continuously. Similar to STREAMING but designed for online
    # serving scenarios where requests arrive dynamically via an input source queue and results are
    # pushed to output a sink queue. Workers remain active indefinitely waiting for new requests.
    SERVING = 2


class Stage(abc.ABC, Generic[T, V]):
    """Abstract base class representing a processing stage in a Ray data processing pipeline.

    This class serves as a foundation for building stages in yotta pipelines.
    Each stage can perform specific data transformations, with flexible resource allocation for GPU or CPU processing.

    Resource Allocation Rules:
    - Exactly one of num_gpus_per_worker or num_cpus_per_worker must be non-None
    - GPU stages automatically get allocated 1 CPU in addition to their GPU allocation
    - Both GPU and CPU allocations can be fractional (e.g., 0.5 GPU or 0.5 CPU)
    - For fractional GPU allocations, multiple workers can share the same GPU

    Worker Assignment Behavior:
    - CPU workers: Assignment is only relevant at the node level. Workers assigned to different CPUs
      on the same node functionally behave the same way
    - GPU workers: Ray manages CUDA environment variables to ensure each worker only sees its
      assigned GPU(s). For CPU-only stages, CUDA variables will point to no GPUs

    Environment Management:
    - Each stage can run in a separate conda environment
    - The environment can be specified either by:
      1. Implementing the conda_env_name property
      2. Using the environment specified by the model (if a model is set)

    See README.md for more information.
    """

    @property
    def stage_batch_size(self) -> int:
        """The number of samples to process at a time.

        This is used to determine how many samples to process at a time.
        """
        return 1

    @property
    @abc.abstractmethod
    def required_resources(self) -> resources.Resources:
        """The new way to specify resources required for a stage.

        Return a ray_utils.Resources object which represents the size/shape of each worker in this stage.
        If None, inherit from the model's required resources.

        This `Resources` class provides an intuitive interface for specifying resource requirements
        that get translated into more detailed internal worker shapes. Here's how the
        resource specifications map to different worker shapes and their allocation
        behaviors:

        1. CPU-Only Shape:
            - Set cpus > 0
            - Leave gpus = 0
            Example: Resources(cpus=2.0)
            Allocation behavior:
                - Only allocated CPU cores, no GPU resources
                - Multiple workers can share the same CPU cores through fractional allocation
                - Never allocated to GPU resources even if available
                - Ray/Yotta does not actually keep track of what particular cores are assigned to particular workers.
                  Instead, for each node, the cpus are treated as a big pool.

        2. Fractional GPU Shape (sharing GPUs):
            - Set gpus to value between 0 and 1 exclusive
            - Optionally set cpus
            Example: Resources(cpus=1.0, gpus=0.5)
            Allocation behavior:
                - Gets allocated fraction of a single GPU's compute capacity
                - Multiple workers can share same GPU up to 100% total utilization

        3. Whole Numbered GPU Shape:
            - Set gpus to integer ≥ 1
            - Optionally set cpus
            Example: Resources(cpus=1.0, gpus=2)
            Allocation behavior:
                - Gets allocated requested number of whole GPUs
                - Each GPU is allocated exclusively (not shared)
                - System optimizes GPU selection to minimize fragmentation

        4. SPMD Shape (Distributed Inference):
            - Set gpus to integer ≥ 1
            - Set is_spmd=True
            Example: Resources(cpus=1.0, gpus=8, is_spmd=True)
            Allocation behavior:
                - Allocated 8 GPUs (prefers to allocate within a single node)
                - Creates one actor per GPU
                - Emulates torchrun behavior by setting the proper environment variables
                - CUDA_VISIBLE_DEVICES is NOT modified (follows torchrun convention)
                - Required for large models needing tensor parallelism or multi-node inference

        SPMD Use Cases:
            - Large models requiring tensor parallelism (e.g., 241B parameter models)
            - vLLM multi-node inference (use distributed_executor_backend="external_launcher")
            - PyTorch models that only support multi-GPU through distributed coordination
            - Any inference workload requiring coordination across multiple GPUs/nodes

        Resource Allocation Strategy:
        The system uses a fragmentation-aware allocation strategy that:
        - Minimizes resource fragmentation across the cluster
        - Tries to keep related resources (GPU compute) together
        - Prefers allocations that maintain flexibility for future requests
        - Can reuse recently freed allocations to prevent thrashing
        - Balances load across available nodes while respecting constraints
        """
        pass

    @property
    def env_info(self) -> runtime_envs.RuntimeEnv | None:
        """Returns the name of the Conda environment for this stage.

        Can be overwritten by subclasses if needed.

        If this is None, we run this stage in the yotta-core env.

        If a model is present, we use that model's conda env name.
        """
        return None

    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        """Returns a list of download requests for this stage.

        This property allows stages to specify artifacts (model weights, conda environments, etc.) that need to be
        downloaded to all cluster nodes before pipeline execution begins. The distributed P2P download
        system will efficiently download these files once and share them between nodes.

        Before starting the pipeline, Xenna will use xenna.file_distribution.download_distributed to download
        all requests from all stages to all nodes. This ensures that required files are available locally
        on each node, eliminating the need for repeated downloads during pipeline execution.

        Returns:
            A list of DownloadRequest objects specifying files to download. Each request can be:
            - ObjectDownloadRequest: Download a single file (with optional unpacking)
            - PrefixDownloadRequest: Download all files under a prefix/directory

        Example:
            ```python
            @property
            def download_requests(self) -> list[file_distribution.DownloadRequest]:
                return [
                    # Download a model file
                    file_distribution.DownloadRequest(
                        value=file_distribution.ObjectDownloadRequest(
                            uri="models/pytorch_model.bin",
                            destination=pathlib.Path("/tmp/model.bin"),
                        )
                    ),
                    # Download and extract an archive
                    file_distribution.DownloadRequest(
                        value=file_distribution.ObjectDownloadRequest(
                            uri="datasets/data.tar.gz",
                            destination=pathlib.Path("/tmp/data.tar.gz"),
                            unpack_options=file_distribution.UnpackOptions(
                                destination=pathlib.Path("/tmp/extracted/"),
                            ),
                        )
                    ),
                    # Download all files under a prefix
                    file_distribution.DownloadRequest(
                        value=file_distribution.PrefixDownloadRequest(
                            uri="tokenizer/", destination=pathlib.Path("/tmp/tokenizer/")
                        )
                    ),
                ]
            ```

        Note:
            - Files are guaranteed to be available locally when setup_on_node(), setup() and process_data() are called
            - The system uses the Rust object_store crate for high-performance storage access
            - URIs in download requests must be RELATIVE to the ObjectStoreConfig base URI
            - The system uses intelligent caching to avoid re-downloading unchanged files
            - Large files are automatically chunked and distributed via P2P for efficiency
            - Authentication is handled via DistributedDownloadConfig passed to run_pipeline()

        See:
            - cosmos_xenna/file_distribution/README.md for detailed documentation
            - DistributedDownloadConfig for authentication and performance tuning
        """
        return []

    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        """Sets up a worker in this stage.

        Can be overwritten by subclasses if needed.

        This is called on every newly created worker in this stage. Typically, this would be used to load a model into
        gpu or create an S3 client.
        """
        pass

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        """Sets up a worker in this stage.

        Can be overwritten by subclasses if needed.

        This is called on every newly created worker in this stage. Typically, this would be used to load a model into
        gpu or create an S3 client.

        By default, if a model is present, this will call the model's setup function.
        """
        pass

    @abc.abstractmethod
    def process_data(self, in_data: list[T]) -> list[V] | None:
        """Processes the input data.

        This method must be implemented by subclasses to define specific data processing logic.

        Args:
            in_data: Input data to be processed.

        Returns:
            The result of processing the input data. This can be any pickleble type. This can also be None. If this is
            None, the data will not be passed to the next stage. This is useful if you want to ignore this piece of
            data for the rest of the pipeline. For example, if the input data was invalid.
        """
        pass


def validate_stage(stage: Stage[Any, Any], cluster_resources: resources.ClusterResources) -> None:
    stage.required_resources.to_worker_shape(cluster_resources)


@attrs.define
class StageSpec(typing.Generic[T, V]):
    """Specification for a pipeline stage.

    This class defines the configuration for a pipeline stage, including the worker to be used and various optional
    parameters.
    """

    stage: Stage[T, V]
    # Hard-coded number of workers to use for this stage. If this and num_workers are both None, we let the scheduling
    # algorithm decide.
    num_workers: int | None = None
    # Hard-coded number of workers per node to use for this stage. If this and num_workers are both None, we let the
    # scheduling algorithm decide.
    num_workers_per_node: float | None = None

    # The following parameters correspond to parameters in PipelineSpec.
    # For this stage, if these values are None, we use the values set in PipelineSpec. Otherwise, these parameters
    # take precedent. See PipelineSpec for documentation on the parameters.
    num_setup_attempts_python: int | None = None
    num_run_attempts_python: int | None = None
    ignore_failures: bool | None = None
    reset_workers_on_failure: bool | None = None
    slots_per_actor: int | None = None
    worker_max_lifetime_m: int | None = None
    worker_restart_interval_m: int | None = None
    max_setup_failure_percentage: float | None = None

    # Over-provision factor for this stage. It is applied to the measured processing
    # speed of the stage to influence the worker allocation.
    over_provision_factor: float | None = None

    def name(self, index: int | None = None) -> str:
        if index is None:
            return str(type(self.stage).__name__)
        else:
            return f"Stage {index:02d} - {type(self.stage).__name__}"

    def validate(self, cluster_resources: resources.ClusterResources) -> None:
        if self.num_workers is not None and self.num_workers_per_node is not None:
            raise ValueError(
                "Expected only one of self.num_workers and self.num_workers_per_node to be non-None. "
                f"However, got {self.num_workers=} and {self.num_workers_per_node=}"
            )
        validate_stage(self.stage, cluster_resources)

    def override_with_pipeline_params(self, p: PipelineConfig) -> StageSpec:
        """Maybe override some fields using the global params.

        The StageSpec and PipelineSpec share some params we want to override the stage with the global params if the
        stage params are None.
        """
        c = copy.deepcopy(self)

        def _override_if_none(attr_name: str):  # noqa: ANN202
            if getattr(c, attr_name) is None:
                setattr(c, attr_name, getattr(p, attr_name))

        _override_if_none("num_setup_attempts_python")
        _override_if_none("num_run_attempts_python")
        _override_if_none("ignore_failures")
        _override_if_none("reset_workers_on_failure")
        _override_if_none("slots_per_actor")
        _override_if_none("worker_max_lifetime_m")
        _override_if_none("worker_restart_interval_m")
        _override_if_none("max_setup_failure_percentage")
        return c


@attrs.define
class StreamingSpecificSpec:
    # How often to run the stage auto-scaler.
    autoscale_interval_s: float = 60 * 3.0
    # Window size with which the auto-scaler estimates the processing speed of each stage.
    # Making it larger makes the estimate more stable, but also less responsive to changes.
    autoscale_speed_estimation_window_duration_s: float = 60 * 3.0
    # Minimum number of data points to keep even if they are outside the window.
    autoscale_speed_estimation_min_data_points: int = 5
    # In streaming mode, when the numeber of max queued tasks exceeds `num_actors * num_slots_per_actor`,
    # i.e. when there is no empty slot, Xenna applies a back-pressure to upstream stages to
    # prevent memory and storage from blowout. The 2 parameters below can help tune that behavior.
    # - This multiplier is applied as `num_actors * num_slots_per_actor * max_queued_multiplier`,
    #   i.e. when you have enough system memory, increase this value to (e.g.) 1.5 is typically beneficial.
    max_queued_multiplier: float = 1.0
    # - When certain stage is super fast and hence scaled down to e.g. just 1 actor, then the
    #   max_queued will be very small (e.g. 2 if slots_per_actor=2). This can make the pipeline
    #   unstable that performance fluctuation can cause downstream stages to get starved.
    #   So this parameter sets a lower bound on max_queued to prevent that.
    max_queued_lower_bound: int = 8
    # Add verbosity level for the autoscaler
    autoscaler_verbosity_level: VerbosityLevel = VerbosityLevel.NONE
    executor_verbosity_level: VerbosityLevel = VerbosityLevel.INFO


@attrs.define
class DashboardSpec:
    port: int = 8080

    def get_ip(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@attrs.define
class PipelineConfig:
    # Execution mode to run the pipeline under. See ExecutionMode and README.md
    execution_mode: ExecutionMode = ExecutionMode.STREAMING
    # Number of attempts to try to call Stage.setup(). If this is > 1, we will log any exceptions and try the specified
    # number of times.
    num_setup_attempts_python: int = 1
    # Number of attempts to try to call Stage.process_data() per task. If this is > 1, we will log any exceptions and
    # try the specified number of times.
    num_run_attempts_python: int = 1
    # Sometimes, setup() can fail sporatically. This is often due to Lustre jankiness. It can be helpful to ignore these
    # failures and retry. If this is non-None, we will retry and then fail the pipeline if a stage fails to setup more
    # than this percentage. For example, if this value is 50 and we try to start 10 actors and 6 of them fail, we will
    # fail the pipeline. If 4 of them fail, we will continue running the pipeline.
    max_setup_failure_percentage: float | None = None
    # If true, any failures in "process_data" will be ignored. If this is false, failures will crash the pipeline.
    # Be careful with this, it can be helpful for catching rare errors, but can also cause a pipeline to run continually
    # in a very broken state. If you choose to set this to True, make sure to examine the logs of the pipeline to
    # check the health.
    ignore_failures: bool = False
    # If true, reset workers when a failure occurs. This can be helpful if you have some class of errors which break the
    # the GPU and only a reset worker can clear it.
    # NOTE: For now, this is only enbled if ignore_failures is set to True.
    reset_workers_on_failure: bool = False
    # Number of tasks to request concurrently per actor. This is an internal detail for streaming pipelines. We request
    # ray to process multiple tasks per worker (default 2). This forces Ray to pre-fetch data and should make it so we
    # are very unlikely to be blocked on IO.
    slots_per_actor: int = _DEFAULT_SLOTS_PER_ACTOR
    # When work stealing is enabled, Xenna will steal queued tasks from busy actors and give them to idle actors.
    # Without this, Xenna can leave some nodes idle when the number of tasks supplied to the pipeline are less than
    # num_actors * num_slots_per_actor (typically == 2); i.e. when there are very few tasks.
    # Ideally, this would always be turned on. However, right now, work stealing can be slow for large jobs.
    enable_work_stealing: bool = False
    # When polling for completed tasks, Xenna groups inflight tasks into chunks and call ray.wait once per chunk.
    max_tasks_to_poll_per_chunk: int = 8
    # Maxmum lifetime in minutes for stage workers before getting terminated and restarted.
    worker_max_lifetime_m: int = 0
    # Interval in minutes between two over-lifetime restart within a stage's actor pool.
    worker_restart_interval_m: int = 1
    # How long to wait between loging pipeline status. Default is every 60 seconds.
    logging_interval_s: float = _DEFAULT_LOG_INTERVAL_S
    # If true, failed tasks will return Nones. This means that the task will not be retried.
    # Be careful with this, this may be the incorrect thing to do for your pipeline.
    failures_return_nones: bool = False
    # If true, the outputs of the last stage will be retained and returned from by `run_pipeline`, otherwise they will
    # be discarded and `run_pipeline` will return None. Retaining this data can be useful if you want to further process
    # it. However, users can also very easily forget that they are doing this and run our of memory.
    return_last_stage_outputs: bool = False
    # Logging verbosity control
    actor_pool_verbosity_level: VerbosityLevel = VerbosityLevel.INFO
    monitoring_verbosity_level: VerbosityLevel = VerbosityLevel.INFO
    # Mode specific parameters
    mode_specific: StreamingSpecificSpec | None = None
    # Whether to log the layout of the ray workers.
    # This can be useful for debugging scheduling/allocation, but is very verbose.
    log_worker_allocation_layout: bool = False
    # The percentage of CPU resources to allocate to the pipeline. This is used to leave some CPU resources for the
    # node manager and other internal ray processes.
    cpu_allocation_percentage: float = 0.95
    # If true, clear the CUDA_VISIBLE_DEVICES environment variable on CPU actors.
    # Otherwise, CUDA_VISIBLE_DEVICES will be set as they are on the node.
    # This is needed to turn off sometimes for libraries which require GPU access on import.
    clear_cuda_visible_devices_on_cpu_actors: bool = True


@attrs.define
class JobInfo:
    """Info about the pipeline job.

    This info can be used to tag reported pipeline metrics.
    """

    pipeline_type: str
    pipeline_version: str
    pipeline_mode: str


@attrs.define
class ServingQueues:
    source: multiprocessing.Queue
    sink: multiprocessing.Queue

    def __deepcopy__(self, memo: dict) -> "ServingQueues":
        # do a shallow copy
        return ServingQueues(self.source, self.sink)


@attrs.define
class PipelineSpec:
    """Specification for a simplified ray pipeline.

    This class encapsulates the configuration for the entire pipeline, including
    the input data and the sequence of stages.

    See ray_utils/README.md for more info.
    """

    # offline processing with pre-populated input data
    # TODO: Can we support a generator here?
    input_data: Sequence[Any]
    stages: Sequence[StageSpec | Stage]
    config: PipelineConfig = attrs.field(factory=PipelineConfig)
    job_info: Optional[JobInfo] = None

    # online serving with input queue to poll for new requests and output queue to push results
    serving_queues: ServingQueues | None = None

    def _format_stage_spec(self, stage_spec: StageSpec) -> str:
        stage = stage_spec.stage
        stage_info = f"   class_name: {type(stage).__name__}\n"
        stage_info += f"   required_resources: {stage.required_resources}\n"

        for field in attrs.fields(StageSpec):
            if field.name != "stage":
                stage_info += f"      {field.name}: {getattr(stage_spec, field.name)}\n"

        return stage_info

    def __str__(self) -> str:
        info = "PipelineSpec:\n"

        for field in attrs.fields(PipelineSpec):
            if field.name not in ["input_data", "stages"]:
                info += f"  {field.name}: {getattr(self, field.name)}\n"

        for i, stage_spec in enumerate(self.stages):
            assert isinstance(stage_spec, StageSpec)
            info += f"  Stage {i}:\n"
            info += self._format_stage_spec(stage_spec)

        return info


class WrappedStage(stage.Interface):
    def __init__(self, stage: Stage):
        self._stage = stage

    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        self._stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        self._stage.setup(worker_metadata)

    def process_data(self, data: Any) -> Any:
        return self._stage.process_data(data)


@attrs.define
class StageAndParams:
    stage: WrappedStage
    params: stage.Params


def make_actor_pool_stage_from_stage_spec(
    pipeline_config: PipelineConfig, spec: StageSpec, stage_idx: int, cluster_resources: resources.ClusterResources
) -> StageAndParams:
    assert spec.slots_per_actor is not None
    assert spec.worker_max_lifetime_m is not None
    assert spec.worker_restart_interval_m is not None
    assert spec.num_setup_attempts_python is not None
    assert spec.num_run_attempts_python is not None
    assert spec.ignore_failures is not None
    assert spec.reset_workers_on_failure is not None

    if approx.float_gt(spec.stage.required_resources.gpus, 0.0):
        modify_cuda_visible_devices_env_var = True
    else:
        # This is a little confusing. If the stage requires no GPUs, we don't want to modify the CUDA_VISIBLE_DEVICES.
        # This means that (assuming the node has gpus) the stage will have the same CUDA_VISIBLE_DEVICES as the rest of
        # the node.
        modify_cuda_visible_devices_env_var = pipeline_config.clear_cuda_visible_devices_on_cpu_actors
    return StageAndParams(
        WrappedStage(spec.stage),
        stage.Params(
            shape=spec.stage.required_resources.to_worker_shape(cluster_resources),
            stage_batch_size=spec.stage.stage_batch_size,
            slots_per_actor=spec.slots_per_actor,
            worker_max_lifetime_m=spec.worker_max_lifetime_m,
            worker_restart_interval_m=spec.worker_restart_interval_m,
            name=spec.name(stage_idx),
            num_node_setup_retries=1,  # TODO: Make this configurable
            num_setup_retries=spec.num_setup_attempts_python,
            num_run_retries=spec.num_run_attempts_python,
            ignore_failures=spec.ignore_failures,
            restart_workers_on_failure=spec.reset_workers_on_failure,
            runtime_env=spec.stage.env_info if spec.stage.env_info is not None else runtime_envs.RuntimeEnv(),
            logging_context=None,  # TODO: Make this configurable
            max_setup_failure_percentage=spec.max_setup_failure_percentage,
            modify_cuda_visible_devices_env_var=modify_cuda_visible_devices_env_var,
        ),
    )
