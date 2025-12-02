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

import concurrent.futures
import copy
import inspect
from types import TracebackType
from typing import Optional

import attrs
import ray

from cosmos_xenna import file_distribution
from cosmos_xenna._cosmos_xenna import setup_logging
from cosmos_xenna.pipelines.private import batch, specs, streaming
from cosmos_xenna.ray_utils import cluster
from cosmos_xenna.utils import python_log as logger
from cosmos_xenna.utils.verbosity import VerbosityLevel


def _validate_method_signature(
    instance_class: type,
    instance_method_name: str,
    base_class: type,
    base_method_name: str,
) -> None:
    """
    Validates that if a method is overridden in instance_class, its signature matches
    the signature of the corresponding method in base_class.
    """
    instance_method_attr = getattr(instance_class, instance_method_name, None)
    base_method_attr = getattr(base_class, base_method_name, None)

    if instance_method_attr is None:
        logger.warning(
            f"Method {instance_method_name} not found in {instance_class.__name__}. Cannot validate signature."
        )
        return

    if base_method_attr is None:
        # This should not happen if base_class and base_method_name are correct
        logger.error(f"Base method {base_method_name} not found in {base_class.__name__}. Cannot validate.")
        return

    # Check if the method is overridden.
    # This means the function object in the subclass is different from the base class's.
    if instance_method_attr is base_method_attr:
        return  # Not overridden, signature is implicitly correct

    # If overridden, inspect signatures
    try:
        # Inspect the method as defined on the class
        instance_sig = inspect.signature(instance_method_attr)
        base_sig = inspect.signature(base_method_attr)
    except (ValueError, TypeError):
        logger.warning(
            f"Could not inspect signature for method {instance_method_name} on {instance_class.__name__} "
            f"or {base_method_name} on {base_class.__name__} (it might not be a Python function)."
        )
        return

    instance_params = list(instance_sig.parameters.values())
    base_params = list(base_sig.parameters.values())

    if len(instance_params) != len(base_params):
        raise TypeError(
            f"Method '{instance_method_name}' in stage '{instance_class.__name__}' "
            f"has an incorrect number of parameters. "
            f"Expected {len(base_params)} (from {base_class.__name__}.{base_method_name}), got {len(instance_params)}. "
            f"Expected signature: {base_sig}, Actual signature: {instance_sig}."
        )


@attrs.define
class DistributedDownloadConfig:
    """Configuration for distributed artifact downloading in pipelines.

    This class configures the distributed P2P file download system used to pre-download
    artifacts (model weights, datasets, etc.) to all cluster nodes before pipeline execution.
    The system eliminates redundant downloads by downloading each chunk only once and sharing
    it between nodes via peer-to-peer transfers.

    Key Features:
        - **Intelligent Chunking**: Large files split into chunks for parallel downloading
        - **P2P Distribution**: Each chunk downloaded from S3 only once, then shared between nodes
        - **Smart Caching**: Downloaded files cached with metadata validation
        - **Load Balancing**: Downloads distributed across nodes to prevent bottlenecks
        - **Cost Optimization**: Minimizes S3 egress costs through P2P sharing

    Attributes:
        object_store_config: Object store configuration for authentication and connection
            using the Rust object_store crate. Can be a single ObjectStoreConfig or
            ObjectStoreConfigByProfile for multiple storage backends. The URI in each
            config serves as the base URI, and download request URIs are interpreted
            as relative keys/paths within the configured object store base URI.
        chunk_size_bytes: Size of chunks for large file downloads in bytes. If None,
            uses system defaults (100MB). Smaller chunks enable better parallelism but
            increase overhead. Recommended: 50-200MB depending on file sizes.
        node_parallelism: Maximum number of concurrent downloads per node. If None,
            uses system defaults (10x CPU count). Higher values increase speed but
            may overwhelm individual nodes.
        object_store_parallelism: Maximum number of concurrent connections to the
            object store across all nodes. Higher values increase download speed
            but may hit rate limits.
        verbose: Whether to enable verbose logging during downloads. Useful for
            debugging performance issues or monitoring large downloads.


    Example:
        >>> # Basic configuration
        >>> config = DistributedDownloadConfig(
        ...     object_store_config=ObjectStoreConfig.make_for_s3(
        ...         bucket="my-models", access_key_id="AKIA...", secret_access_key="...", region="us-west-2"
        ...     )
        ... )

        >>> # Multi-profile configuration
        >>> config = DistributedDownloadConfig(
        ...     object_store_config=ObjectStoreConfigByProfile(
        ...         profiles={
        ...             "models": ObjectStoreConfig.make_for_s3(...),
        ...             "datasets": ObjectStoreConfig.make_for_s3(...),
        ...         }
        ...     )
        ... )

    See Also:
        - cosmos_xenna/file_distribution/README.md for detailed documentation
        - ObjectStoreConfig for authentication setup
        - Stage.download_requests for specifying what to download
    """

    object_store_config: file_distribution.ObjectStoreConfig | file_distribution.ObjectStoreConfigByProfile
    chunk_size_bytes: Optional[int] = None
    node_parallelism: Optional[int] = None
    object_store_parallelism: int = 100
    verbose: bool = False


def _deduplicate_download_requests(
    requests: list[file_distribution.DownloadRequest],
) -> list[file_distribution.DownloadRequest]:
    """Deduplicate download requests based on uri and destination, ignoring profile_name, cache, and unpack_method.

    For uniqueness, considers:
    - uri
    - destination
    - unpack_destination (if unpack_options is provided)

    Ignores when determining uniqueness:
    - profile_name
    - cache
    - unpack_method

    Args:
        requests: List of download requests to deduplicate

    Returns:
        Deduplicated list of download requests
    """
    if not requests:
        return requests

    seen_keys: set[tuple] = set()
    deduplicated: list[file_distribution.DownloadRequest] = []
    duplicate_count = 0

    for request in requests:
        # Build a key for uniqueness check based on the request type
        if isinstance(request.value, file_distribution.ObjectDownloadRequest):
            obj_req = request.value
            # Create key: (uri, destination, unpack_destination if present)
            if obj_req.unpack_options:
                key = (obj_req.uri, str(obj_req.destination), str(obj_req.unpack_options.unpack_destination))
            else:
                key = (obj_req.uri, str(obj_req.destination), None)

            if key in seen_keys:
                duplicate_count += 1
                unpack_destination = obj_req.unpack_options.unpack_destination if obj_req.unpack_options else None
                logger.info(
                    f"Removing duplicate ObjectDownloadRequest: uri={obj_req.uri}, "
                    f"destination={obj_req.destination}, "
                    f"unpack_destination={unpack_destination}"
                )
            else:
                seen_keys.add(key)
                deduplicated.append(request)

        elif isinstance(request.value, file_distribution.PrefixDownloadRequest):
            prefix_req = request.value
            # Create key: (uri, destination)
            key = (prefix_req.uri, str(prefix_req.destination), None)

            if key in seen_keys:
                duplicate_count += 1
                logger.info(
                    f"Removing duplicate PrefixDownloadRequest: uri={prefix_req.uri}, "
                    f"destination={prefix_req.destination}"
                )
            else:
                seen_keys.add(key)
                deduplicated.append(request)
        else:
            raise ValueError(f"Unknown download request type: {type(request.value)}")

    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate download request(s)")

    return deduplicated


def download_artifacts(
    stages: list[specs.StageSpec | specs.Stage],
    config: DistributedDownloadConfig,
) -> None:
    """Download artifacts from pipeline stages using the distributed P2P download system.

    This function collects download requests from all stages and downloads the specified
    artifacts to all cluster nodes using Xenna's distributed P2P file distribution system.
    Each chunk is downloaded from the object store only once and then shared between nodes
    via fast peer-to-peer transfers.

    Args:
        stages: List of StageSpec objects from a pipeline. Download requests are collected
            from each stage's download_requests property.
        config: Configuration for the distributed download system including
            object store credentials, chunk sizes, and parallelism settings

    Example:
        >>> # With pipeline stages
        >>> pipeline_spec = PipelineSpec(stages=[MyStage1(), MyStage2()])
        >>> config = DistributedDownloadConfig(object_store_config=ObjectStoreConfig.make_for_s3(...))
        >>> download_artifacts(pipeline_spec.stages, config)
    """
    if not ray.is_initialized():
        cluster.init_or_connect_to_cluster()

    # Collect download requests from all stages
    requests: list[file_distribution.DownloadRequest] = []
    stages_with_downloads: list[str] = []

    for stage_spec_item in stages:
        if isinstance(stage_spec_item, specs.Stage):
            stage = stage_spec_item
        elif isinstance(stage_spec_item, specs.StageSpec):
            stage = stage_spec_item.stage
        else:
            raise ValueError(f"Unknown stage type: {type(stage_spec_item)}")
        stage_requests = stage.download_requests
        if stage_requests:
            requests.extend(stage_requests)
            stages_with_downloads.append(stage.__class__.__name__)

    logger.info(f"Collected {len(requests)} download request(s) from stages: {stages_with_downloads}")

    # Deduplicate requests
    requests = _deduplicate_download_requests(requests)

    if not requests:
        logger.info("No download requests found. Skipping distributed download.")
        return

    logger.info(f"Downloading {len(requests)} artifact(s) (after deduplication)")
    logger.info("Using distributed P2P download system to pre-fetch artifacts to all cluster nodes...")

    file_distribution.download_distributed(
        requests,
        config.object_store_config,
        chunk_size_bytes=config.chunk_size_bytes,
        node_parallelism=config.node_parallelism,
        object_store_parallelism=config.object_store_parallelism,
        verbose=config.verbose,
    )

    logger.info(f"Successfully downloaded all {len(requests)} requested artifacts to all cluster nodes.")


class BackgroundArtifactDownloader:
    """Context manager for downloading artifacts from pipeline stages in a background thread.

    This class allows you to start artifact downloads in the background and continue
    doing other work while the downloads proceed. It automatically collects download
    requests from all stages and handles them asynchronously. The downloads will be
    waited on automatically when exiting the context manager, or you can explicitly
    wait earlier by calling the wait() method.

    Uses concurrent.futures for clean thread management and exception handling.

    Attributes:
        stages: List of pipeline stages to collect download requests from
        config: Configuration for the distributed download system
        is_complete: Property indicating whether the download has completed

    Example:
        >>> # Start download in background, do other work, then wait
        >>> pipeline_spec = PipelineSpec(stages=[MyStage1(), MyStage2()])
        >>> with BackgroundArtifactDownloader(pipeline_spec.stages, config) as downloader:
        ...     # Downloads are happening in the background
        ...     setup_cluster()
        ...     validate_pipeline()
        ...     # Optionally check if done early
        ...     if downloader.is_complete:
        ...         print("Downloads finished early!")
        ... # Downloads guaranteed to be complete here

        >>> # Explicit wait before context exit
        >>> with BackgroundArtifactDownloader(pipeline_spec.stages, config) as downloader:
        ...     do_preparatory_work()
        ...     downloader.wait()  # Block until downloads complete
        ...     use_downloaded_artifacts()
    """

    def __init__(
        self,
        stages: list[specs.StageSpec | specs.Stage],
        config: DistributedDownloadConfig,
    ):
        """Initialize the background downloader.

        Args:
            stages: List of StageSpec or Stage objects from a pipeline
            config: Configuration for distributed downloads
        """
        self._stages = stages
        self._config = config
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._future: Optional[concurrent.futures.Future] = None

    def __enter__(self) -> BackgroundArtifactDownloader:
        """Start downloading in a background thread."""
        logger.info("Starting background download of artifacts from pipeline stages...")
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future = self._executor.submit(download_artifacts, self._stages, self._config)
        return self

    def wait(self) -> None:
        """Wait for downloads to complete.

        This method blocks until all downloads finish. If the download thread
        raised an exception, it will be re-raised here.

        Raises:
            Exception: Any exception that occurred during downloading
        """
        if self._future:
            # This will block until complete and re-raise any exceptions
            self._future.result()

    @property
    def is_complete(self) -> bool:
        """Check if downloads have completed (successfully or with error).

        Returns:
            True if downloads are done, False if still in progress
        """
        if self._future is None:
            # No downloads were needed
            return True
        return self._future.done()

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> bool:
        """Wait for downloads to complete and cleanup executor when exiting the context.

        Returns:
            False to propagate any exceptions from the with-block
        """
        try:
            self.wait()
        finally:
            # Always cleanup the executor
            if self._executor:
                self._executor.shutdown(wait=True)
        return False


def run_pipeline(
    pipeline_spec: specs.PipelineSpec,
    distibuted_download_config: DistributedDownloadConfig | None = None,
) -> Optional[list]:
    """Entry point for calling a pipeline.

    Depending on the environment and the pipeline spec, this can call a STREAMING, BATCH or BATCH_DEBUG pipeline.

    Before we start a pipeline, we do the following:

    1. **Artifact Download**: Collect download requests from all stages and download required artifacts
       (model weights, datasets, etc.) to all cluster nodes using distributed P2P downloading
    2. **Cluster Setup**: Connect to a ray cluster if running on the cloud, otherwise start a local ray cluster
    3. **Pipeline Execution**: Run the pipeline with the specified execution mode
    4. **Result Collection**: (if pipeline_spec.return_last_stage_outputs is True) Return the results
       from the last stage

    ## Distributed Downloading

    Each stage can specify artifacts to download by implementing the `download_requests` property
    (inherited from specs.Stage). This property should return a list of DownloadRequest objects
    specifying the files to download. These requests are collected and downloaded to all cluster
    nodes before pipeline execution begins. This ensures that all required files (model weights,
    datasets, etc.) are available locally on each node, eliminating the need for repeated downloads
    during pipeline execution.

    The download system uses:
    - **P2P Distribution**: Files are downloaded once and shared between nodes for efficiency
    - **Chunked Downloads**: Large files are split into chunks and downloaded in parallel
    - **Caching**: Previously downloaded files are validated and reused when possible
    - **Profile-based Authentication**: Different object store credentials can be used per download

    See yotta/ray_utils/README.md for more info on running pipelines.

    Args:
        pipeline_spec: The pipeline specification containing stages and configuration
        distibuted_download_config: Configuration for distributed artifact downloading.
            Required if any stage has download_requests. Contains object store credentials,
            chunk sizes, and parallelism settings.

    Returns:
        (If pipeline_spec.config.return_last_stage_outputs is true) The list of items from the last stage in the
        pipeline. NOTE: These are pulled down to the host machine. You probably
        do not want to return anything
        heavy-weight here.

    Raises:
        ValueError: If stages specify download_requests but no distibuted_download_config is provided
    """

    # Setup logging in rust.
    setup_logging()
    # Convert the stages field into StageSpecs if needed.
    pipeline_spec = copy.deepcopy(pipeline_spec)
    pipeline_spec.stages = [x if isinstance(x, specs.StageSpec) else specs.StageSpec(x) for x in pipeline_spec.stages]

    for stage_spec_item in pipeline_spec.stages:
        assert isinstance(stage_spec_item, specs.StageSpec)
        actual_stage_instance = stage_spec_item.stage
        stage_class = type(actual_stage_instance)

        # Validate methods from specs.Stage
        _validate_method_signature(stage_class, "setup", specs.Stage, "setup")
        _validate_method_signature(stage_class, "setup_on_node", specs.Stage, "setup_on_node")
        _validate_method_signature(stage_class, "process_data", specs.Stage, "process_data")

    # Override stage level params with global params if needed
    for idx in range(len(pipeline_spec.stages)):
        pipeline_spec.stages[idx] = pipeline_spec.stages[idx].override_with_pipeline_params(pipeline_spec.config)
    if not pipeline_spec.input_data and pipeline_spec.config.execution_mode != specs.ExecutionMode.SERVING:
        logger.warning(
            "No input data specified for the pipeline. Skipping running the pipeline and return an empty list."
        )
        return []
    stage_names = [x.name(i) for i, x in enumerate(pipeline_spec.stages)]
    assert len(stage_names) == len(set(stage_names)), f"Expected stage names to be unique, but got: {stage_names}"

    if pipeline_spec.config.monitoring_verbosity_level >= VerbosityLevel.INFO:
        logger.info(pipeline_spec)

    logger.info("Initialized Ray cluster.")
    cluster.init_or_connect_to_cluster()

    cluster_resources = cluster.make_cluster_resources_from_ray_nodes(
        cpu_allocation_percentage=pipeline_spec.config.cpu_allocation_percentage
    )
    # Validate the stages:
    for stage_spec_item in pipeline_spec.stages:
        assert isinstance(stage_spec_item, specs.StageSpec)
        stage_spec_item.validate(cluster_resources)
    logger.info(f"Cluster resources: {cluster_resources}")
    logger.info(f"Created/connected to cluster with resources: {cluster_resources.total_pool()}")

    # Download all requested artifacts using distributed P2P system
    if distibuted_download_config is not None:
        download_artifacts(pipeline_spec.stages, distibuted_download_config)  # pyright: ignore[reportArgumentType]

    if (
        pipeline_spec.config.execution_mode == specs.ExecutionMode.STREAMING
        or pipeline_spec.config.execution_mode == specs.ExecutionMode.SERVING
    ):
        if pipeline_spec.config.mode_specific is None:
            pipeline_spec.config.mode_specific = specs.StreamingSpecificSpec()
        return streaming.run_pipeline(pipeline_spec, cluster_resources)
    elif pipeline_spec.config.execution_mode == specs.ExecutionMode.BATCH:
        return batch.run_pipeline(pipeline_spec, cluster_resources)
    else:
        raise ValueError(f"unknown execution mode={pipeline_spec.config.execution_mode}")
