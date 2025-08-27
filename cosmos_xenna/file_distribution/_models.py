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

import enum
import pathlib
import uuid
from typing import Optional, Union

import attrs
import obstore as obs

from cosmos_xenna._cosmos_xenna.file_distribution import models as rust_models


@attrs.define
class NodeMetadata:
    node_id: str
    ip_address: str
    uploader_port: int


@attrs.define
class ByteRange:
    """Represents a range of bytes within a file.

    Attributes:
        start: Starting byte position (inclusive)
        end: Ending byte position (exclusive)
    """

    start: int
    end: int

    def to_rust(self) -> rust_models.ByteRange:
        return rust_models.ByteRange(start=self.start, end=self.end)


@attrs.define
class ObjectAndRange:
    """Represents an S3 object with an optional byte range for partial downloads.

    This class is fundamental to the chunking system, allowing the same S3 object
    to be referenced with different byte ranges for parallel chunk downloads.
    The crc32_checksum field enables data integrity validation.

    Attributes:
        object_uri: Full S3 URI (e.g., "s3://bucket/key")
        range: Optional byte range for partial object downloads (None = full object)
        crc32_checksum: Optional CRC32 checksum for data integrity verification

    Examples:
        Full object: ObjectAndRange("s3://bucket/file.dat", None)
        Chunk: ObjectAndRange("s3://bucket/file.dat", ByteRange(0, 1048575))  # First 1MB
    """

    object_uri: str
    range: Optional[ByteRange] = None
    crc32_checksum: Optional[int] = None

    def to_rust(self) -> rust_models.ObjectAndRange:
        return rust_models.ObjectAndRange(
            object_uri=self.object_uri,
            range=self.range.to_rust() if self.range else None,
            crc32_checksum=self.crc32_checksum,
        )


@attrs.define
class SingleNodeTestingInfo:
    """Configuration for single-node testing mode.

    This class enables testing the distributed download system on a single machine
    by simulating multiple nodes. Each "fake node" uses separate storage directories
    to simulate the isolation that would exist on separate physical machines.

    This is essential for:
    - Unit testing P2P scheduling logic
    - Development without requiring a full Ray cluster
    - CI/CD pipeline testing
    - Debugging distributed coordination issues

    Each fake node will:
    - Have its own storage root under /tmp/p2p_download_test/{node_id}/
    - Run its own NodeWorker actor (but all on the same machine)
    - Participate in the full P2P scheduling algorithm
    - Share chunks with other fake nodes as if they were remote

    Attributes:
        num_fake_nodes: Number of simulated nodes to create (typically 2-5 for testing)
    """

    num_fake_nodes: int


class UnpackMethod(enum.Enum):
    """Specifies the archive format for unpacking a downloaded file."""

    AUTO = "auto"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    ZIP = "zip"

    def to_rust(self) -> rust_models.UnpackMethod:
        if self == UnpackMethod.AUTO:
            return rust_models.UnpackMethod.Auto
        elif self == UnpackMethod.TAR:
            return rust_models.UnpackMethod.Tar
        elif self == UnpackMethod.TAR_GZ:
            return rust_models.UnpackMethod.TarGz
        elif self == UnpackMethod.ZIP:
            return rust_models.UnpackMethod.Zip
        else:
            raise ValueError(f"Unknown unpack method: {self}")


@attrs.define
class UnpackOptions:
    """Options for unpacking a downloaded file after it is assembled.

    Attributes:
        unpack_destination: The directory where the archive contents should be
            unpacked.
        unpack_method: The archive format (e.g., TAR, ZIP) to use for
            unpacking.
    """

    unpack_destination: pathlib.Path
    unpack_method: UnpackMethod = UnpackMethod.AUTO

    def to_rust(self) -> rust_models.UnpackOptions:
        return rust_models.UnpackOptions(
            destination=self.unpack_destination,
            unpack_method=self.unpack_method.to_rust(),
        )


@attrs.define
class CacheInfo:
    """Metadata for a cached S3 object, stored in a sidecar JSON file.

    This class represents the data stored in a sidecar file (e.g.,
    `my_file.txt.s3_cache`) to validate a locally cached file against its
    S3 source. A local file is considered a valid cache if its metadata
    matches the current metadata of the object in S3.

    Attributes:
        uri: The full S3 URI of the source object (e.g., "s3://bucket/key").
        size: The size of the S3 object in bytes (ContentLength).
        last_modified: The last modified timestamp of the S3 object, as a
            string.
    """

    uri: str
    size: int
    last_modified_unix_micros: int

    def to_rust(self) -> rust_models.CacheInfo:
        return rust_models.CacheInfo(
            uri=self.uri,
            size=self.size,
            last_modified_unix_micros=self.last_modified_unix_micros,
        )


@attrs.define
class ObjectDownloadRequest:
    """Request to download a single S3 object with optional post-processing.

    This class represents a complete download specification for a single S3 object,
    including authentication, destination, caching behavior, and post-download actions
    like archive extraction and symlink creation.

    Attributes:
        profile_name: AWS profile name for S3 authentication and authorization
        uri: Complete S3 URI (e.g., "s3://bucket/path/file.tar.gz")
        destination: Local filesystem path where the object should be saved
        cache: Whether to enable intelligent caching with metadata validation
            (default: True). When enabled, creates .s3_cache_info sidecar files
        unpack_options: Optional archive extraction configuration. If provided,
            the downloaded file will be automatically extracted after download
        symlink_path: Optional path to create a symbolic link pointing to the
            downloaded file. Useful for creating stable reference paths

    Cache Behavior:
        When cache=True, the system will:
        - Check for existing cached files before downloading
        - Validate cache using S3 metadata (size, last_modified, URI)
        - Skip download if valid cache exists
        - Create .s3_cache_info sidecar files for future validation

    Archive Extraction:
        When unpack_options is provided:
        - Downloads the archive file first
        - Extracts to the specified destination atomically
        - Supports TAR, TAR.GZ, and ZIP formats with auto-detection
        - Creates cache info for both the archive and extracted directory
    """

    uri: str
    destination: pathlib.Path
    cache: bool = True
    unpack_options: Optional[UnpackOptions] = None
    symlink_path: Optional[pathlib.Path] = None
    profile_name: str | None = None


@attrs.define
class PrefixDownloadRequest:
    """Request to download all objects under an S3 prefix (directory-like structure).

    This class enables bulk downloading of entire S3 "directories" by specifying
    a prefix. The system will recursively list all objects under the prefix and
    download them while preserving the directory structure.

    Attributes:
        profile_name: AWS profile name for S3 authentication and authorization
        uri: S3 prefix URI (e.g., "s3://bucket/dataset/" or "s3://bucket/path/to/data/")
        destination: Local directory where all objects should be saved,
            preserving the relative path structure from the prefix
        cache: Whether to enable intelligent caching for all downloaded objects
            (default: True). Each object gets its own cache validation

    Behavior:
        1. Lists all objects recursively under the specified prefix
        2. Creates individual ObjectDownloadRequest for each discovered object
        3. Preserves directory structure relative to the prefix
        4. Downloads objects in parallel across the cluster

    Example:
        For prefix "s3://bucket/dataset/" containing:
        - s3://bucket/dataset/train/images/img1.jpg
        - s3://bucket/dataset/train/labels/label1.txt
        - s3://bucket/dataset/test/images/img2.jpg

        With destination "/local/data/", creates:
        - /local/data/train/images/img1.jpg
        - /local/data/train/labels/label1.txt
        - /local/data/test/images/img2.jpg
    """

    uri: str
    destination: pathlib.Path
    cache: bool = True
    profile_name: str | None = None


@attrs.define
class DownloadRequest:
    value: Union[ObjectDownloadRequest, PrefixDownloadRequest]


@attrs.define
class _S3ObjectDownload:
    """Represents a single S3 object to be downloaded.

    Attributes:
        object_id: Unique identifier for this object download
        parent_request_id: ID of the parent download request
        profile_name: The S3 profile to use for authentication
        value: The S3 object to download
        destination: Local path where the content should be saved
        size: Size of the object in bytes, if known
        last_modified: Last modified timestamp from S3
    """

    object_id: uuid.UUID
    parent_request_id: uuid.UUID
    profile_name: str | None
    uri: str
    destination: pathlib.Path
    cache_info: CacheInfo
    unpack_options: Optional[UnpackOptions] = None

    def to_rust(self) -> rust_models.ObjectToDownload:
        return rust_models.ObjectToDownload(
            object_id=self.object_id,
            parent_request_id=self.parent_request_id,
            profile_name=self.profile_name,
            uri=self.uri,
            destination=self.destination,
            cache_info=self.cache_info.to_rust(),
            unpack_options=self.unpack_options.to_rust() if self.unpack_options else None,
        )


@attrs.define
class _DownloadChunk:
    """Represents a chunk of data to be downloaded from an S3 object.

    Attributes:
        chunk_id: Unique identifier for this chunk
        parent_object_id: ID of the parent object this chunk belongs to
        profile_name: The S3 profile to use for authentication
        value: The S3 object and byte range to download
        destination: Local path where the chunk should be saved
        size: Size of the chunk in bytes
    """

    chunk_id: uuid.UUID
    parent_object_id: uuid.UUID
    profile_name: str | None
    value: ObjectAndRange
    destination: pathlib.Path
    size: int

    def to_rust(self) -> rust_models.ChunkToDownload:
        return rust_models.ChunkToDownload(
            chunk_id=self.chunk_id,
            parent_object_id=self.parent_object_id,
            profile_name=self.profile_name,
            value=self.value.to_rust(),
            destination=self.destination,
            size=self.size,
        )


@attrs.define
class _DownloadCatalog:
    """Catalog of all items to be downloaded.

    Attributes:
        objects: List of individual S3 objects to download
        chunks: List of chunks to download
    """

    objects: list[_S3ObjectDownload]
    chunks: list[_DownloadChunk]
    chunks_by_object: dict[uuid.UUID, list[uuid.UUID]]

    def to_rust(self) -> rust_models.DownloadCatalog:
        return rust_models.DownloadCatalog(
            [obj.to_rust() for obj in self.objects],
            [chunk.to_rust() for chunk in self.chunks],
            self.chunks_by_object,
        )


@attrs.define
class DownloadFromNodeOrder:
    download_chunk: _DownloadChunk
    source_node_id: str
    source_node_ip: str
    source_node_port: int

    def to_rust(self) -> rust_models.DownloadFromNodeOrder:
        return rust_models.DownloadFromNodeOrder(
            self.download_chunk.to_rust(),
            self.source_node_id,
            self.source_node_ip,
            self.source_node_port,
        )


@attrs.define
class Orders:
    """Download orders issued by the scheduler to a node worker.

    This class represents the scheduler's decisions about what a specific node
    should download in the current scheduling cycle. Orders are sent to nodes
    every 100ms and include both S3 downloads and peer-to-peer transfers.

    The scheduler creates these orders based on:
    - Current cluster state and load balancing
    - File rarity and cache affinity
    - Available peer sources for chunks
    - Node capacity and current utilization

    Attributes:
        download_from_s3: List of chunks this node should download directly from S3.
            These are typically new chunks or chunks not available from peers
        download_from_node: List of peer-to-peer transfer orders. Each specifies
            a chunk to download from another node that already has it cached
    """

    download_from_s3: list[_DownloadChunk]
    download_from_node: list[DownloadFromNodeOrder]

    def to_rust(self) -> rust_models.Orders:
        return rust_models.Orders(
            download_from_s3=[chunk.to_rust() for chunk in self.download_from_s3],
            download_from_node=[order.to_rust() for order in self.download_from_node],
        )


@attrs.define
class NodeStatus:
    """Real-time status information for a cluster node."""

    node_id: str
    downloading_p2p_chunks: set[uuid.UUID] = attrs.field(factory=set)
    downloading_s3_chunks: set[uuid.UUID] = attrs.field(factory=set)
    available_chunks: set[uuid.UUID] = attrs.field(factory=set)
    completed_or_cached_objects: set[uuid.UUID] = attrs.field(factory=set)
    unneeded_objects: set[uuid.UUID] = attrs.field(factory=set)
    num_active_uploads: int = 0
    num_active_assembling_tasks: int = 0
    num_active_unpacking_tasks: int = 0
    cpu_utilization: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    memory_utilization: float = 0.0
    network_capacity_gbps: float | None = None

    def is_done(self, catalog: _DownloadCatalog) -> bool:
        return (
            all(
                obj.object_id in self.completed_or_cached_objects or obj.object_id in self.unneeded_objects
                for obj in catalog.objects
            )
            and self.num_active_uploads == 0
            and self.num_active_assembling_tasks == 0
            and self.num_active_unpacking_tasks == 0
            and self.downloading_p2p_chunks == set()
            and self.downloading_s3_chunks == set()
        )

    def merge_with_rust(self, rust_status: rust_models.NodeStatus) -> None:
        self.downloading_p2p_chunks = set(rust_status.downloading_p2p_chunks)
        self.downloading_s3_chunks = set(rust_status.downloading_s3_chunks)
        self.available_chunks = set(rust_status.available_chunks)
        self.completed_or_cached_objects = set(rust_status.completed_or_cached_objects)
        self.unneeded_objects = set(rust_status.unneeded_objects)
        self.num_active_uploads = rust_status.num_active_uploads
        self.num_active_assembling_tasks = rust_status.num_active_assembling_tasks
        self.num_active_unpacking_tasks = rust_status.num_active_unpacking_tasks


@attrs.define
class ObjectStoreConfig:
    """Configuration for an S3 object store."""

    uri: str
    config_args: dict[str, str] = attrs.field(factory=dict)

    def to_rust(self) -> rust_models.ObjectStoreConfig:
        return rust_models.ObjectStoreConfig(uri=self.uri, config_args=self.config_args)


@attrs.define
class ObjectStoreConfigByProfile:
    """Configuration for an S3 object store by profile."""

    profiles: dict[str | None, ObjectStoreConfig]

    def to_rust(self) -> rust_models.ObjectStoreConfigByProfile:
        return rust_models.ObjectStoreConfigByProfile(
            profiles={profile: config.to_rust() for profile, config in self.profiles.items()}
        )


@attrs.define
class ObjectStoreByProfile:
    """Configuration for an S3 object store by profile."""

    profiles: dict[str | None, obs.store._ObjectStoreMixin]

    @classmethod
    def make_from_config_by_profile(cls, config_by_profile: ObjectStoreConfigByProfile) -> ObjectStoreByProfile:
        return cls(
            profiles={
                profile: obs.store.from_url(config.uri, config=config.config_args)  # type: ignore
                for profile, config in config_by_profile.profiles.items()
            }
        )
