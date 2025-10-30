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

"""Provides a distributed file download system for S3, optimized for performance
and cost-efficiency in a cluster environment.

This module uses Ray to parallelize downloads across multiple nodes. Key features include:
- **Intelligent Chunking**: Large files are split into chunks for parallel downloading.
- **Peer-to-Peer (P2P) Transfers**: Chunks are shared between nodes to minimize
  redundant S3 downloads. Each chunk is downloaded from S3 only once.
- **Persistent Caching**: Downloaded files are cached locally on nodes with metadata
  validation to avoid re-downloads.
- **Automatic Archive Unpacking**: Supports automatic extraction of TAR, GZ, and ZIP files.

The main entry point is the `download_distributed` function.
"""

from __future__ import annotations

import collections
import math
import os
import random
import time
import typing
import uuid
from typing import Any, List, Optional, Union

import attrs
import obstore
import psutil
import ray
import ray.util.scheduling_strategies
import tqdm
from tabulate import tabulate

from cosmos_xenna._cosmos_xenna.file_distribution import data_plane
from cosmos_xenna._cosmos_xenna.file_distribution import models as rust_models
from cosmos_xenna.file_distribution import _models as models
from cosmos_xenna.utils import python_log as logger
from cosmos_xenna.utils import timing

_MAIN_LOOP_FREQ_HZ = 10
_DEFAULT_DISPLAY_RATE_S = 5.0


@attrs.define
class _DownloadRequestWithId:
    """Internal wrapper for download requests that includes a unique identifier.

    Attributes:
        request_id: Unique identifier for this download request
        request: The actual download request
    """

    request_id: uuid.UUID
    request: Union[models.ObjectDownloadRequest, models.PrefixDownloadRequest]


def _calculate_chunk_ranges(file_size: int, chunk_size_bytes: int) -> Optional[List[models.ByteRange]]:
    """Calculates byte ranges for chunking a file.

    Args:
        file_size: Total size of the file in bytes
        chunk_size_bytes: Desired size of each chunk in bytes

    Returns:
        List of byte ranges for each chunk, or None if chunking is not needed

    Raises:
        ValueError: If chunk_size_bytes is not positive or file_size is invalid
    """
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    if file_size < 0:
        raise ValueError("file_size cannot be negative")

    if file_size <= chunk_size_bytes:
        return None

    # Handle empty file case
    if file_size == 0:
        raise ValueError("file_size cannot be 0")

    chunks = []
    # range(start, stop, step) generates start indices
    for start in range(0, file_size, chunk_size_bytes):
        # Calculate the end index (exclusive) for the chunk
        # It's the minimum of (start + chunk_size) and (file_size)
        end = min(start + chunk_size_bytes, file_size)
        chunks.append(models.ByteRange(start, end))

    return chunks


def _create_download_catalog(
    download_requests: list[_DownloadRequestWithId],
    configs_by_profile: models.ObjectStoreConfigByProfile,
    chunk_size: int,
) -> models._DownloadCatalog:
    """Creates a catalog of all items to be downloaded.

    This function:
    1. Separates requests into objects and prefixes
    2. Expands prefixes into individual objects
    3. Gets sizes and last modified timestamps for all objects
    4. Creates chunks for large objects

    Args:
        download_requests: List of download requests with IDs
        client_cache: Cache of S3 clients
        chunk_size: Size of each chunk in bytes

    Returns:
        A catalog containing all items to be downloaded
    """
    object_stores_by_profile = models.ObjectStoreByProfile.make_from_config_by_profile(configs_by_profile)
    prefixes_by_profile: dict[str | None, list[_DownloadRequestWithId]] = collections.defaultdict(list)
    objects_by_profile: dict[str | None, list[_DownloadRequestWithId]] = collections.defaultdict(list)
    for request in download_requests:
        if isinstance(request.request, models.PrefixDownloadRequest):
            prefixes_by_profile[request.request.profile_name].append(request)
        elif isinstance(request.request, models.ObjectDownloadRequest):
            objects_by_profile[request.request.profile_name].append(request)
        else:
            raise ValueError(f"Unknown request type: {type(request.request.value)}")

    objects: list[models._S3ObjectDownload] = []

    # Add all objects to the object list directly
    for profile_name, requests_ in objects_by_profile.items():
        client = object_stores_by_profile.profiles[profile_name]
        for request in requests_:
            assert isinstance(request.request, models.ObjectDownloadRequest)
            metadata = client.head(request.request.uri)
            unix_micros = math.ceil(metadata["last_modified"].timestamp() * 1_000_000)
            objects.append(
                models._S3ObjectDownload(
                    uuid.uuid4(),
                    request.request_id,
                    profile_name,
                    request.request.uri,
                    request.request.destination,
                    models.CacheInfo(
                        uri=request.request.uri,
                        size=metadata["size"],
                        last_modified_unix_micros=unix_micros,
                    ),
                    request.request.unpack_options,
                )
            )

    # Search all prefixes and add the objects to the object list
    if prefixes_by_profile:
        logger.info(f"Listing prefixes for {len(prefixes_by_profile)} profiles.")
    for profile_name, requests_ in prefixes_by_profile.items():
        client = object_stores_by_profile.profiles[profile_name]
        for request in requests_:
            result: list[obstore.ObjectMeta] = list(client.list(request.request.uri))

            prefix_destination = request.request.destination
            for list_of_objects in result:
                for obj in list_of_objects:
                    uri = str(obj["path"])  # type: ignore
                    assert uri.startswith(request.request.uri)
                    object_relative_to_prefix = uri[len(request.request.uri) :].strip()
                    object_destination = prefix_destination / object_relative_to_prefix
                    unix_micros = math.ceil(obj["last_modified"].timestamp() * 1_000_000)  # type: ignore
                    objects.append(
                        models._S3ObjectDownload(
                            uuid.uuid4(),
                            request.request_id,
                            profile_name,
                            uri,
                            object_destination,
                            models.CacheInfo(uri=uri, size=obj["size"], last_modified_unix_micros=unix_micros),  # type: ignore
                        )
                    )

    # Create chunks for all objects
    logger.info(f"Calculating chunks for {len(objects)} objects.")
    chunks_by_object: dict[uuid.UUID, list[uuid.UUID]] = collections.defaultdict(list)
    chunks = []
    for obj in objects:
        assert obj.cache_info is not None
        chunk_ranges = _calculate_chunk_ranges(obj.cache_info.size, chunk_size)
        if chunk_ranges is None:  # Chunking is not needed
            chunk_id = uuid.uuid4()
            chunks.append(
                models._DownloadChunk(
                    chunk_id,
                    obj.object_id,
                    obj.profile_name,
                    models.ObjectAndRange(obj.uri, None),
                    obj.destination,
                    obj.cache_info.size,  # Use the full object size
                )
            )
            chunks_by_object[obj.object_id].append(chunk_id)
        else:
            for chunk_range in chunk_ranges:
                # Calculate the size of this specific byte range chunk
                chunk_length = chunk_range.end - chunk_range.start + 1
                chunk_id = uuid.uuid4()
                chunks.append(
                    models._DownloadChunk(
                        chunk_id,
                        obj.object_id,
                        obj.profile_name,
                        models.ObjectAndRange(obj.uri, models.ByteRange(chunk_range.start, chunk_range.end)),
                        obj.destination,
                        chunk_length,  # Use the calculated chunk size
                    )
                )
                chunks_by_object[obj.object_id].append(chunk_id)
    logger.info(f"Calculated {len(chunks)} chunks.")
    return models._DownloadCatalog(objects, chunks, dict(chunks_by_object))


def _get_network_capacity_gbps() -> float | None:
    """Get the network capacity of the primary network interface in Gbps.

    This function is designed for Linux systems and may not work on other
    operating systems. It determines the speed of the primary network
    interface by reading from `/sys/class/net`.

    Returns:
        The network capacity in Gbps, or None if it cannot be determined.
    """
    try:
        # Get all network interfaces
        interfaces = os.listdir("/sys/class/net/")

        # Find the primary interface (usually starts with 'e' or 'w')
        primary_interface = None
        for interface in interfaces:
            if interface.startswith(("e", "w")):
                primary_interface = interface
                break

        if not primary_interface:
            return None

        # Read the speed from the interface's sysfs entry
        speed_path = f"/sys/class/net/{primary_interface}/speed"
        if os.path.exists(speed_path):
            with open(speed_path) as f:
                speed_mbps = int(f.read().strip())
                return speed_mbps / 1000  # Convert Mbps to Gbps
        return None
    except (FileNotFoundError, ValueError, OSError):
        return None


class NodeWorker:
    """Ray actor that handles downloading and assembling chunks on a node.

    This actor is a stateful service that runs in its own process on a specific node
    in the Ray cluster. By making it an actor, we can:
    1. Manage state (like the S3 client cache and download plan) across multiple
       remote calls without having to pass it back and forth.
    2. Pin the actor to a specific node, ensuring that downloads for that node
       happen locally and write to that node's disk.
    3. Control concurrency at the node level using `max_concurrency`.

    This actor:
    1. Downloads chunks assigned to its node from S3.
    2. Assembles chunks into complete files by fetching chunk data from other nodes.
    3. Manages node-local storage for the downloaded files.
    4. Runs an HTTP server for serving chunks to peer nodes.

    Ray-specific behavior:
    - Runs as a Ray actor on a specific node
    - Maintains state between method calls (client cache, download plan)
    - Can handle multiple concurrent downloads (controlled by max_concurrency)
    - Uses node-local storage for better performance
    - Serves chunks via HTTP for peer-to-peer transfers
    """

    def __init__(
        self,
        node_id: str,
        object_store_config_by_profile: models.ObjectStoreConfigByProfile,
        download_catalog: models._DownloadCatalog,
        parallelism: int,
        p2p_retry_config: RetryConfig,
        object_store_retry_config: RetryConfig,
        is_test: bool = False,
    ):
        object_stores_by_profile = rust_models.ObjectStoreByProfile(
            object_store_config_by_profile.to_rust(),
        )
        # TODO: Make clients
        self._rust_data_plane = data_plane.DataPlane(
            node_id=node_id,
            is_test=is_test,
            node_parallelism=parallelism,
            download_catalog=download_catalog.to_rust(),
            object_store_by_profile=object_stores_by_profile,
            p2p_retry_config=p2p_retry_config.to_rust(),
            object_store_retry_config=object_store_retry_config.to_rust(),
        )
        self._p2p_retry_config = p2p_retry_config
        self._object_store_retry_config = object_store_retry_config
        self._is_test = is_test
        self._node_id = node_id
        self._last_network_io_counters: typing.Any = None
        self._last_status_check_time: float | None = None
        self._network_capacity_gbps = _get_network_capacity_gbps()

    def get_metadata(self) -> models.NodeMetadata:
        """Get the metadata for this node."""
        port = self._rust_data_plane.start_p2p_server(port=None)
        return models.NodeMetadata(
            node_id=self._node_id,
            ip_address=ray.util.get_node_ip_address() if not self._is_test else "127.0.0.1",
            uploader_port=port,
        )

    def start(self) -> None:
        """Lightweight setup of the download worker.

        This just initializes the client cache and marks the worker as ready.
        The actual download plan will be provided later via receive_plan().
        """
        self._rust_data_plane.start()

    def update(self, orders: models.Orders) -> models.NodeStatus:
        """Get current status of this node worker.

        Returns:
            NodeStatus with current state including cached items and slot availability
        """
        rstatus = self._rust_data_plane.update(orders.to_rust())

        status = models.NodeStatus(node_id=self._node_id)
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Calculate network utilization
        current_time = time.time()
        current_io_counters = psutil.net_io_counters()
        if self._last_network_io_counters is None:
            bytes_sent_per_second = 0
            bytes_recv_per_second = 0
        else:
            time_delta = current_time - self._last_status_check_time  # type: ignore
            bytes_sent_delta = current_io_counters.bytes_sent - self._last_network_io_counters.bytes_sent
            bytes_recv_delta = current_io_counters.bytes_recv - self._last_network_io_counters.bytes_recv
            bytes_sent_per_second = bytes_sent_delta / time_delta
            bytes_recv_per_second = bytes_recv_delta / time_delta

        status.merge_with_rust(rstatus)
        status.cpu_utilization = cpu_percent
        status.memory_utilization = memory_percent
        status.network_bytes_sent = int(bytes_sent_per_second)
        status.network_bytes_recv = int(bytes_recv_per_second)
        status.network_capacity_gbps = self._network_capacity_gbps

        self._last_network_io_counters = current_io_counters
        self._last_status_check_time = current_time
        return status

    def teardown(self) -> None:
        del self._rust_data_plane


@attrs.define
class _NodeSlotUsage:
    num_download_slots: int
    num_upload_slots: int


@attrs.define
class _S3OrdersResults:
    s3_orders_by_node: dict[str, list[models._DownloadChunk]]
    node_slot_usage: dict[str, _NodeSlotUsage]


def _ray_wait_with_progress(object_refs: List[ray.ObjectRef], description: str) -> None:
    """Waits for Ray object references to complete with a progress bar.

    This function uses Ray's wait() mechanism to implement progress tracking. This
    pattern is useful for monitoring long-running parallel tasks.

    How it works:
    1. It calls `ray.wait()` with a short timeout. This call returns immediately
       with any tasks that have completed in that interval.
    2. It updates a progress bar with the number of newly completed tasks.
    3. It repeats this process on the remaining (not-yet-completed) tasks until
       the list of remaining tasks is empty.

    Args:
        object_refs: List of Ray object references to wait for
        description: Description to show in the progress bar
    """
    remaining_refs = object_refs[:]
    total_tasks = len(remaining_refs)

    with tqdm.tqdm(total=total_tasks, desc=description) as pbar:
        while remaining_refs:
            ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=len(remaining_refs), timeout=0.1)
            pbar.update(len(ready_refs))


def _ray_get_with_progress(object_refs: List[ray.ObjectRef], description: str) -> list:
    """Gets results from Ray object references with a progress bar.

    This function uses Ray's wait() mechanism to implement progress tracking while
    collecting results. This pattern is useful for monitoring long-running parallel
    tasks and collecting their results in order.

    How it works:
    1. It calls `ray.wait()` with a short timeout. This call returns immediately
       with any tasks that have completed in that interval.
    2. It gets the results from completed tasks using `ray.get()`.
    3. It updates a progress bar with the number of newly completed tasks.
    4. It repeats this process on the remaining (not-yet-completed) tasks until
       the list of remaining tasks is empty.
    5. It returns the results in the same order as the input object_refs.

    Args:
        object_refs: List of Ray object references to get results from
        description: Description to show in the progress bar

    Returns:
        List of results from the object references, in the same order as input
    """
    if not object_refs:
        return []

    remaining_refs = object_refs[:]
    total_tasks = len(remaining_refs)
    results = {}  # Map from object_ref to result

    with tqdm.tqdm(total=total_tasks, desc=description) as pbar:
        while remaining_refs:
            ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=len(remaining_refs), timeout=0.1)

            # Get results from completed tasks
            if ready_refs:
                ready_results = ray.get(ready_refs)
                for ref, result in zip(ready_refs, ready_results):
                    results[ref] = result

                pbar.update(len(ready_refs))

    # Return results in the same order as the original object_refs
    return [results[ref] for ref in object_refs]


class _Scheduler:
    """Intelligent scheduler for optimizing distributed downloads across the cluster.

    The scheduler is the brain of the distributed download system. It analyzes
    the current state of all nodes and makes optimal decisions about:
    - Which chunks to download from S3 vs peers
    - Which nodes should download which chunks
    - How to balance load across the cluster
    - How to minimize total download time and S3 bandwidth

    Scheduling Algorithm:
    The scheduler implements a sophisticated multi-criteria optimization:

    1. **Rarity Prioritization**: Files cached on fewer nodes get higher priority
       to ensure better distribution and reduce future S3 downloads

    2. **Load Balancing**: Work is distributed to least-loaded nodes to maximize
       cluster utilization and prevent bottlenecks

    3. **Cache Affinity**: Nodes that already have some chunks of a file are
       preferred for downloading remaining chunks of the same file

    4. **P2P Optimization**: When possible, downloads from peer nodes instead
       of S3 to reduce bandwidth costs and improve speed

    5. **Slot Management**: Respects per-node upload/download capacity limits
       to prevent resource exhaustion

    Efficiency Features:
    - Each chunk is downloaded from S3 at most once across the entire cluster
    - Subsequent transfers happen via fast peer-to-peer within the cluster
    - Smart peer selection chooses least-loaded nodes as sources
    - Random tie-breaking prevents deterministic bottlenecks

    Performance Characteristics:
    - Scales to hundreds of nodes with minimal coordination overhead
    - Adapts to changing cluster conditions (node failures, varying loads)
    - Minimizes S3 egress costs through intelligent P2P sharing
    - Optimizes for both throughput and cost efficiency
    """

    def __init__(
        self,
        nodes: list[str],
        download_catalog: models._DownloadCatalog,
        metadata_by_node: dict[str, models.NodeMetadata],
        num_s3_download_slots: int = 100,
        num_node_download_slots: int = 100,
    ):
        """Initialize the scheduler with cluster topology and resource limits.

        Args:
            nodes: List of node IDs in the cluster
            download_catalog: Complete catalog of all objects and chunks to download
            num_s3_download_slots: Global limit on concurrent S3 downloads across
                the entire cluster (prevents overwhelming S3)
            num_node_download_slots: Per-node limit on concurrent downloads
                (prevents overwhelming individual nodes)
        """
        self._nodes = nodes
        self._download_catalog = download_catalog
        self._total_s3_download_slots = num_s3_download_slots
        self._total_node_download_slots_per_node = num_node_download_slots
        self._chunks_downloaded_from_s3: set[uuid.UUID] = set()
        self._metadata_by_node = metadata_by_node
        logger.info(f"Total S3 download slots: {self._total_s3_download_slots}")
        logger.info(f"Total node download slots per node: {self._total_node_download_slots_per_node}")

    def _make_s3_download_orders(self, status_by_node: dict[str, models.NodeStatus]) -> _S3OrdersResults:
        orders_by_node: dict[str, list[models._DownloadChunk]] = {}
        node_slot_usage: dict[str, _NodeSlotUsage] = {}
        s3_slots_used = 0

        # Initialize empty orders for each node
        for node_id in self._nodes:
            orders_by_node[node_id] = []
            node_slot_usage[node_id] = _NodeSlotUsage(
                num_download_slots=len(status_by_node[node_id].downloading_s3_chunks)
                + len(status_by_node[node_id].downloading_p2p_chunks),
                num_upload_slots=status_by_node[node_id].num_active_uploads,
            )
            s3_slots_used += len(status_by_node[node_id].downloading_s3_chunks)

        # Get all chunks that still need to be downloaded
        # Use a dict keyed by chunk_id to automatically deduplicate
        chunks_needing_download: dict[uuid.UUID, models._DownloadChunk] = {}
        all_available_or_downloading_chunks = set()
        for status in status_by_node.values():
            all_available_or_downloading_chunks.update(status.available_chunks)
            all_available_or_downloading_chunks.update(status.downloading_s3_chunks)
            all_available_or_downloading_chunks.update(status.writing_chunks)

        for chunk in self._download_catalog.chunks:
            # Skip if any node already has this chunk available or is downloading/writing it
            if chunk.chunk_id not in all_available_or_downloading_chunks:
                chunks_needing_download[chunk.chunk_id] = chunk

        # Assign chunks to nodes
        for chunk in chunks_needing_download.values():
            if s3_slots_used >= self._total_s3_download_slots:
                break

            # Find nodes that have download capacity
            available_nodes = []
            for node_id in self._nodes:
                if node_slot_usage[node_id].num_download_slots < self._total_node_download_slots_per_node:
                    available_nodes.append(node_id)

            # Skip if no nodes have capacity
            if not available_nodes:
                continue

            # Pick the least loaded node that has download capacity
            best_node = min(available_nodes, key=lambda node_id: node_slot_usage[node_id].num_download_slots)

            # Add the chunk to the node's orders
            orders_by_node[best_node].append(chunk)

            # Update the node's slot usage
            node_slot_usage[best_node].num_download_slots += 1

            s3_slots_used += 1

        return _S3OrdersResults(s3_orders_by_node=orders_by_node, node_slot_usage=node_slot_usage)

    def _make_p2p_download_orders(
        self, status_by_node: dict[str, models.NodeStatus], node_slot_usage: dict[str, _NodeSlotUsage]
    ) -> dict[str, list[models.DownloadFromNodeOrder]]:
        """Generate P2P transfer orders using a BitTorrent-inspired rarest-first algorithm.

        This implements a sophisticated peer-to-peer scheduling algorithm that:
        1. Prioritizes rare chunks (available on fewer nodes) to improve swarm health
        2. Balances load across nodes by choosing least-loaded sources and destinations
        3. Respects upload and download capacity limits to prevent overload
        4. Minimizes redundant transfers by checking current download state

        The algorithm is inspired by BitTorrent's rarest-first strategy, which ensures
        that rare pieces get distributed quickly to improve overall download efficiency.
        """
        orders_by_node: dict[str, list[models.DownloadFromNodeOrder]] = {node_id: [] for node_id in self._nodes}

        # Use same limit for uploads as downloads (could be configurable in future)
        max_upload_slots_per_node = self._total_node_download_slots_per_node

        # Build chunk availability map: chunk_id -> set of nodes that have it
        chunk_availability: dict[uuid.UUID, set[str]] = {}
        for node_id, status in status_by_node.items():
            for chunk_id in status.available_chunks:
                if chunk_id not in chunk_availability:
                    chunk_availability[chunk_id] = set()
                chunk_availability[chunk_id].add(node_id)

        # Find chunks that need P2P transfers, paired with their rarity score
        chunks_needing_transfer: list[tuple[models._DownloadChunk, int]] = []

        for chunk in self._download_catalog.chunks:
            if chunk.chunk_id in chunk_availability:
                # Calculate rarity: lower number = rarer chunk (available on fewer nodes)
                available_node_count = len(chunk_availability[chunk.chunk_id])

                # Check if any nodes need this chunk
                for node_id in self._nodes:
                    status = status_by_node[node_id]

                    # A node needs this chunk if:
                    # 1. It doesn't already have it available
                    # 2. It's not already downloading it (S3 or P2P)
                    # 3. It's not currently being written to disk
                    # 4. It doesn't have the complete parent object
                    # 5. The object is actually needed (not marked as unneeded)
                    needs_chunk = (
                        chunk.chunk_id not in status.available_chunks
                        and chunk.chunk_id not in status.downloading_p2p_chunks
                        and chunk.chunk_id not in status.downloading_s3_chunks
                        and chunk.chunk_id not in status.writing_chunks
                        and chunk.parent_object_id not in status.completed_or_cached_objects
                        and chunk.parent_object_id not in status.unneeded_objects
                    )

                    if needs_chunk:
                        chunks_needing_transfer.append((chunk, available_node_count))
                        break  # No need to check other nodes for this chunk

        # Sort by rarity (rarest first) with random tie-breaking
        # BitTorrent's rarest-first strategy: prioritize chunks available on fewer nodes
        chunks_needing_transfer.sort(key=lambda x: (x[1], random.randint(0, 1000)))

        # Assign P2P transfers using load balancing
        for chunk, _ in chunks_needing_transfer:
            available_sources = chunk_availability[chunk.chunk_id]

            # Find destination nodes that need this chunk and have download capacity
            candidate_destinations = []
            for node_id in self._nodes:
                status = status_by_node[node_id]

                # Check if this node needs the chunk (same logic as above)
                needs_chunk = (
                    chunk.chunk_id not in status.available_chunks
                    and chunk.chunk_id not in status.downloading_p2p_chunks
                    and chunk.chunk_id not in status.downloading_s3_chunks
                    and chunk.chunk_id not in status.writing_chunks
                    and chunk.parent_object_id not in status.completed_or_cached_objects
                    and chunk.parent_object_id not in status.unneeded_objects
                )

                # Check if node has download capacity
                has_download_capacity = (
                    node_slot_usage[node_id].num_download_slots < self._total_node_download_slots_per_node
                )

                if needs_chunk and has_download_capacity:
                    candidate_destinations.append(node_id)

            if not candidate_destinations:
                continue  # No nodes need this chunk or all are at capacity

            # Find source nodes with upload capacity
            candidate_sources = []
            for source_node_id in available_sources:
                has_upload_capacity = node_slot_usage[source_node_id].num_upload_slots < max_upload_slots_per_node

                if has_upload_capacity:
                    candidate_sources.append(source_node_id)

            if not candidate_sources:
                continue  # No sources have upload capacity

            # Load balancing: choose least loaded source (fewest active uploads)
            best_source = min(candidate_sources, key=lambda node_id: node_slot_usage[node_id].num_upload_slots)

            # Load balancing: choose least loaded destination (fewest active downloads)
            best_destination = min(
                candidate_destinations, key=lambda node_id: node_slot_usage[node_id].num_download_slots
            )

            # Create the P2P transfer order
            order = models.DownloadFromNodeOrder(
                download_chunk=chunk,
                source_node_id=best_source,
                source_node_ip=self._metadata_by_node[best_source].ip_address,
                source_node_port=self._metadata_by_node[best_source].uploader_port,
            )
            orders_by_node[best_destination].append(order)

            # Update slot usage to account for this transfer
            # This prevents double-booking of capacity in this scheduling round
            node_slot_usage[best_destination].num_download_slots += 1
            node_slot_usage[best_source].num_upload_slots += 1

        return orders_by_node

    def make_orders(self, status_by_node: dict[str, models.NodeStatus]) -> dict[str, models.Orders]:
        """Generate optimal download orders for each node based on current cluster state.

        This is the core scheduling method that analyzes the cluster state and
        creates an optimal work distribution plan. It's called every 100ms by
        the main coordination loop.

        The algorithm considers:
        - Current cache state of all nodes
        - Load balancing across nodes (active downloads/uploads)
        - File rarity for prioritization
        - Available peer sources for chunks
        - Resource constraints (download/upload slots)

        Optimization Goals:
        1. **Minimize Total Time**: Spread work to maximize parallelism
        2. **Minimize S3 Bandwidth**: Use P2P transfers when possible
        3. **Maximize Cache Efficiency**: Prioritize rare files for better distribution
        4. **Balance Load**: Prevent any single node from becoming a bottleneck
        5. **Respect Constraints**: Honor per-node and global resource limits

        """
        orders_by_node: dict[str, models.Orders] = {}

        s3_order_results = self._make_s3_download_orders(status_by_node)
        for node_id, s3_orders in s3_order_results.s3_orders_by_node.items():
            orders_by_node[node_id] = models.Orders(download_from_s3=s3_orders, download_from_node=[])
        node_slot_usage = s3_order_results.node_slot_usage

        p2p_order_results = self._make_p2p_download_orders(status_by_node, node_slot_usage)
        for node_id, p2p_orders in p2p_order_results.items():
            orders_by_node[node_id].download_from_node = p2p_orders

        return orders_by_node


def _display_progress(
    statuses: list[models.NodeStatus],
    total_objects_needed: int,
    total_chunks_needed: int,
    loop_rate: float,
    is_dones: list[bool],
    runtime_s: float,
) -> None:
    """Display the progress of the download in a structured table.

    Args:
        statuses: A list of NodeStatus objects, one for each node in the cluster.
        total_objects_needed: The total number of objects to be downloaded.
        total_chunks_needed: The total number of chunks to be downloaded.
        loop_rate: The rate of the main coordination loop in Hz.
        is_dones: A list of booleans indicating if each node is done.
    """
    headers = [
        "Node ID",
        "Status",
        "Objects Cached",
        "Chunks Cached",
        "S3 DL",
        "P2P DL",
        "Writing",
        "Uploads",
        "Unpacking",
        "Assembling",
        "CPU %",
        "Mem %",
        "Net TX (Gbps)",
        "Net RX (Gbps)",
    ]
    data = []
    for i, status in enumerate(statuses):
        node_cached_objects = len(status.completed_or_cached_objects)
        if total_objects_needed > 0:
            objects_percentage = (node_cached_objects / total_objects_needed) * 100
            node_objects_progress = f"{node_cached_objects}/{total_objects_needed} ({objects_percentage:.1f}%)"
        else:
            node_objects_progress = f"{node_cached_objects}/{total_objects_needed}"

        node_cached_chunks = len(status.available_chunks)
        if total_chunks_needed > 0:
            chunks_percentage = (node_cached_chunks / total_chunks_needed) * 100
            node_chunks_progress = f"{node_cached_chunks}/{total_chunks_needed} ({chunks_percentage:.1f}%)"
        else:
            node_chunks_progress = f"{node_cached_chunks}/{total_chunks_needed}"
        node_status = "In progress" if not is_dones[i] else "Done"
        s3_downloads = len(status.downloading_s3_chunks)
        p2p_downloads = len(status.downloading_p2p_chunks)
        writing = status.num_active_file_writing_tasks
        uploads = status.num_active_uploads
        unpacking = status.num_active_unpacking_tasks
        assembling = status.num_active_assembling_tasks
        cpu = status.cpu_utilization
        mem = status.memory_utilization
        net_sent_gbps = (status.network_bytes_sent * 8) / (1000**3)
        net_recv_gbps = (status.network_bytes_recv * 8) / (1000**3)
        if status.network_capacity_gbps:
            net_tx_str = f"{net_sent_gbps:.2f}/{status.network_capacity_gbps:.2f}"
            net_rx_str = f"{net_recv_gbps:.2f}/{status.network_capacity_gbps:.2f}"
        else:
            net_tx_str = f"{net_sent_gbps:.2f}"
            net_rx_str = f"{net_recv_gbps:.2f}"

        data.append(
            [
                str(status.node_id)[:10],
                node_status,
                node_objects_progress,
                node_chunks_progress,
                s3_downloads,
                p2p_downloads,
                writing,
                uploads,
                unpacking,
                assembling,
                f"{cpu:.1f}",
                f"{mem:.1f}",
                net_tx_str,
                net_rx_str,
            ]
        )
    table = tabulate(data, headers=headers, tablefmt="fancy_grid")
    extra_display_msg = f"\nRuntime: {runtime_s:.2f}s"
    logger.info(f"Xenna P2P File Distribution ({loop_rate:.2f} Hz){extra_display_msg}\n{table}")


@attrs.define
class RetryConfig:
    num_retries: int = 5
    base_delay_millis: int = 200

    def to_rust(self) -> rust_models.RetryConfig:
        return rust_models.RetryConfig(self.num_retries, self.base_delay_millis)


def download_distributed(
    download_requests: list[models.DownloadRequest],
    object_store_config: models.ObjectStoreConfig | models.ObjectStoreConfigByProfile,
    chunk_size_bytes: Optional[int] = None,
    node_parallelism: Optional[int] = None,
    object_store_parallelism: int = 100,
    p2p_retry_config: Optional[RetryConfig] = None,
    object_store_retry_config: Optional[RetryConfig] = None,
    verbose: bool = False,
    testing_info: Optional[models.SingleNodeTestingInfo] = None,
    display_rate_s: float = _DEFAULT_DISPLAY_RATE_S,
) -> None:
    """Orchestrates a distributed download across a Ray cluster using P2P file sharing.

    This is the main entry point for the distributed file download system. It coordinates
    the download of object store files across all nodes in a Ray cluster,
    using intelligent chunking and peer-to-peer transfers to minimize bandwidth usage
    and maximize download speed.

    The system uses the Rust object_store crate for high-performance access to various
    storage backends (S3, GCS, Azure, local filesystem) with a consistent API.

    **How it works:**
    1. **Catalog Creation**: Analyzes all download requests and creates a catalog of
       objects and chunks to download
    2. **Worker Setup**: Deploys NodeWorker actors on each cluster node to handle
       local downloads and P2P serving
    3. **Intelligent Scheduling**: Uses a BitTorrent-inspired scheduler to optimize
       which nodes download which chunks and from where (S3 vs peers)
    4. **P2P Distribution**: Each chunk is downloaded from S3 only once, then shared
       between nodes via HTTP for maximum efficiency
    5. **File Assembly**: Chunks are assembled into complete files with validation
    6. **Optional Unpacking**: Archives are automatically extracted if requested

    **Key Benefits:**
    - **Cost Efficient**: Each chunk downloaded from S3 only once across entire cluster
    - **Fast**: Parallel downloads with intelligent load balancing
    - **Reliable**: Automatic retries, validation, and error handling
    - **Smart Caching**: Avoids re-downloading unchanged files

    Args:
        download_requests: A list of `DownloadRequest` objects, each specifying
            an object or prefix to download. Can include single files, archives
            to extract, or entire directory prefixes.
        object_store_config: Authentication and connection configuration for the
            object store using the object_store crate. Can be a single config or
            profile-based config for multiple object stores. The URI in the config
            serves as the base URI, and download request URIs are relative paths
            within this base.
        chunk_size_bytes: The size (in bytes) for file chunks. Defaults to 100MB.
            Smaller chunks provide better parallelism but more overhead.
            Recommended: 50-200MB depending on file sizes.
        node_parallelism: The maximum number of concurrent downloads per node.
            Defaults to 10x CPU count. Higher values increase speed but may
            overwhelm individual nodes.
        object_store_parallelism: The maximum number of concurrent connections
            to the object store across all nodes. Defaults to 1000. Higher values
            increase speed but may hit rate limits.
        verbose: If True, enables detailed progress logging and performance metrics.
            Useful for monitoring large downloads or debugging performance issues.
        testing_info: If provided, runs in a single-node test mode that simulates
            a multi-node environment. Used for testing and development.
        display_rate_s: The rate at which to display progress. Defaults to once every 5.0 seconds.

    Raises:
        RuntimeError: If Ray is not initialized or no nodes are available
        ValueError: If download requests are invalid or object store config is incorrect
        FileNotFoundError: If requested objects don't exist in the object store
        PermissionError: If authentication fails or access is denied

    Example:
        >>> import ray
        >>> from cosmos_xenna.file_distribution import *
        >>> # Initialize Ray cluster
        >>> ray.init(address="ray://head-node:10001")
        >>> # Configure S3 authentication
        >>> s3_config = ObjectStoreConfig.make_for_s3(
        ...     bucket="my-models", access_key_id="AKIA...", secret_access_key="...", region="us-west-2"
        ... )
        >>> # Define what to download
        >>> requests = [
        ...     DownloadRequest(
        ...         value=ObjectDownloadRequest(
        ...             uri="models/bert-base/pytorch_model.bin", destination=pathlib.Path("/tmp/model.bin")
        ...         )
        ...     ),
        ...     DownloadRequest(
        ...         value=ObjectDownloadRequest(
        ...             uri="datasets/train.tar.gz",
        ...             destination=pathlib.Path("/tmp/train.tar.gz"),
        ...             unpack_options=UnpackOptions(
        ...                 destination=pathlib.Path("/tmp/train/"),
        ...             ),
        ...         )
        ...     ),
        ... ]
        >>> # Download to all cluster nodes
        >>> download_distributed(
        ...     download_requests=requests,
        ...     object_store_config=s3_config,
        ...     chunk_size_bytes=50 * 1024 * 1024,  # 50MB chunks
        ...     verbose=True,
        ... )

    See Also:
        - cosmos_xenna/file_distribution/README.md for detailed documentation
        - DownloadRequest for specifying what to download
        - ObjectStoreConfig for authentication setup
    """
    if node_parallelism is None:
        node_parallelism = os.cpu_count() * 2

    if chunk_size_bytes is None:
        chunk_size_bytes = 100 * 1024 * 1024  # 100MB

    if testing_info is None:
        nodes = [str(node["NodeID"]) for node in ray.nodes()]
    else:
        nodes = [str(x) for x in range(testing_info.num_fake_nodes)]

    if p2p_retry_config is None:
        p2p_retry_config = RetryConfig()

    if object_store_retry_config is None:
        object_store_retry_config = RetryConfig()

    if not ray.is_initialized():
        raise RuntimeError(
            "Ray is not initialized. Please call ray.init() before calling this function. "
            "For distributed downloads, ensure Ray is configured with multiple nodes."
        )

    if isinstance(object_store_config, models.ObjectStoreConfig):
        object_store_config_by_profile = models.ObjectStoreConfigByProfile(profiles={None: object_store_config})
    else:
        object_store_config_by_profile = object_store_config

    logger.info(f"Starting Xenna p2p file distribution with {len(nodes)} nodes")
    node_worker_cls = ray.remote(NodeWorker)

    download_catalog = _create_download_catalog(
        [_DownloadRequestWithId(uuid.uuid4(), x.value) for x in download_requests],
        object_store_config_by_profile,
        chunk_size_bytes,
    )

    # ---------------------------Phase 1: Setup Workers---------------------------
    logger.info("Setting up workers...")
    workers_by_node: dict[str, Any] = {}
    for node in nodes:
        if testing_info is None:
            # Create a lightweight worker on each node
            worker = node_worker_cls.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node, soft=False
                ),
            ).remote(
                node,
                object_store_config_by_profile,
                download_catalog,
                node_parallelism,
                p2p_retry_config,
                object_store_retry_config,
            )
        else:
            worker = node_worker_cls.options().remote(
                node,
                object_store_config_by_profile,
                download_catalog,
                node_parallelism,
                p2p_retry_config,
                object_store_retry_config,
                is_test=True,  # type: ignore
            )
        workers_by_node[node] = worker

    worker_metadatas = _ray_get_with_progress(
        [worker.get_metadata.remote() for worker in workers_by_node.values()], "Getting worker metadata"
    )
    worker_registry = {worker.node_id: worker for worker in worker_metadatas}

    try:
        _ray_wait_with_progress([worker.start.remote() for worker in workers_by_node.values()], "Starting workers")

        rate_limiter = timing.RateLimiter(_MAIN_LOOP_FREQ_HZ)
        display_rate_checker = timing.RateLimitChecker(1.0 / display_rate_s)
        scheduler = _Scheduler(
            list(workers_by_node.keys()),
            download_catalog,
            worker_registry,
            num_s3_download_slots=object_store_parallelism,
            num_node_download_slots=node_parallelism,
        )
        # Main coordination loop
        total_objects_needed = len(download_catalog.objects)
        total_chunks_needed = len(download_catalog.chunks)

        last_loop_time = time.time()
        last_orders = {node_id: models.Orders(download_from_s3=[], download_from_node=[]) for node_id in nodes}
        start_runtime = time.time()
        while True:
            # Get the status of the workers.
            t0 = time.time()
            statuses: list[models.NodeStatus] = ray.get(
                [worker.update.remote(last_orders[node_id]) for node_id, worker in workers_by_node.items()]
            )
            update_duration_s = time.time() - t0
            if update_duration_s > 1.0:
                logger.warning(f"Updating workers took: {update_duration_s:.3f}s (> 1.0s)")
            elif verbose:
                logger.debug(f"Updating workers took: {update_duration_s:.4f}s")
            statuses_by_node = {status.node_id: status for status in statuses}

            # Check if we're done - all objects are cached everywhere
            is_dones = [status.is_done(download_catalog) for status in statuses]
            if all(is_dones):
                logger.info(f"All {total_objects_needed} objects are now cached across the cluster. Download complete!")
                break

            # Make the orders.
            t0 = time.time()
            orders_by_node = scheduler.make_orders(statuses_by_node)
            if verbose:
                logger.debug(f"Making orders took: {time.time() - t0:.4f}s")
            last_orders = orders_by_node

            # Log progress per-node
            new_time = time.time()
            loop_rate = 1 / (new_time - last_loop_time)
            last_loop_time = new_time
            # Rate limit the display to 1 Hz
            if display_rate_checker.can_call():
                runtime_s = time.time() - start_runtime
                _display_progress(
                    statuses,
                    total_objects_needed,
                    total_chunks_needed,
                    loop_rate,
                    is_dones,
                    runtime_s=runtime_s,
                )

            rate_limiter.sleep()
    except Exception as e:
        logger.exception(f"Error in main coordination loop: {e}")
        raise e
    finally:
        # Final runtime report
        final_runtime_s = None
        if "start_runtime" in locals():
            final_runtime_s = time.time() - start_runtime
            logger.info(f"Total runtime: {final_runtime_s:.2f}s")
        _ray_wait_with_progress(
            [worker.teardown.remote() for worker in workers_by_node.values()], "Cleaning up workers"
        )
