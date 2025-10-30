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
    """Request to download a single object with optional post-processing.

    This class represents a complete download specification for a single object,
    including authentication, destination, caching behavior, and post-download actions
    like archive extraction and symlink creation.

    Attributes:
        profile_name: Profile name for object store authentication and authorization.
            This maps to an ObjectStoreConfig in ObjectStoreConfigByProfile.
            If None, uses the default profile.
        uri: Object key/path relative to the object store base URI (e.g., "dataset/file.tar.gz").
            CRITICAL: This is NOT a full URI but a relative key within the object store
            configured for the specified profile_name. The object_store crate combines
            the base URI from ObjectStoreConfig with this relative key.

            Example: If ObjectStoreConfig has uri="s3://my-bucket/", then this uri should
            be "path/to/file.txt" (NOT "s3://my-bucket/path/to/file.txt"). The full path
            becomes base_uri + relative_key = "s3://my-bucket/path/to/file.txt".
        destination: Local filesystem path where the object should be saved
        unpack_options: Optional archive extraction configuration. If provided,
            the downloaded file will be automatically extracted after download

    Cache Behavior:
        - Check for existing cached files before downloading
        - Validate cache using object store metadata (size, last_modified, URI)
        - Skip download if valid cache exists
        - Create .s3_cache_info sidecar files for future validation

    Archive Extraction:
        When unpack_options is provided:
        - Downloads the archive file first
        - Extracts to the specified destination atomically
        - Supports TAR, TAR.GZ, and ZIP formats with auto-detection
        - Creates cache info for both the archive and extracted directory

    Examples:
        >>> # Download a single model file
        >>> request = ObjectDownloadRequest(
        ...     uri="models/bert-base/pytorch_model.bin",
        ...     destination=pathlib.Path("/tmp/model.bin"),
        ... )

        >>> # Download and extract an archive
        >>> request = ObjectDownloadRequest(
        ...     uri="datasets/imagenet.tar.gz",
        ...     destination=pathlib.Path("/tmp/imagenet.tar.gz"),
        ...     unpack_options=UnpackOptions(
        ...         destination=pathlib.Path("/tmp/imagenet/"),
        ...     ),
        ... )

        >>> # Download with specific authentication profile
        >>> request = ObjectDownloadRequest(
        ...     profile_name="internal-models",
        ...     uri="proprietary/my-model.bin",
        ...     destination=pathlib.Path("/tmp/my-model.bin"),
        ... )
    """

    uri: str
    destination: pathlib.Path
    unpack_options: Optional[UnpackOptions] = None
    profile_name: str | None = None


@attrs.define
class PrefixDownloadRequest:
    """Request to download all objects under a prefix (directory-like structure).

    This class enables bulk downloading of entire object store "directories" by specifying
    a prefix. The system will recursively list all objects under the prefix and
    download them while preserving the directory structure. This is ideal for downloading
    complete model directories, datasets, or any collection of related files.

    Attributes:
        profile_name: Profile name for object store authentication and authorization.
            This maps to an ObjectStoreConfig in ObjectStoreConfigByProfile.
            If None, uses the default profile.
        uri: Prefix key/path relative to the object store base URI (e.g., "dataset/" or "path/to/data/").
            CRITICAL: This is NOT a full URI but a relative prefix within the object store
            configured for the specified profile_name. The object_store crate combines
            the base URI from ObjectStoreConfig with this relative prefix.

            Example: If ObjectStoreConfig has uri="s3://my-bucket/", then this uri should
            be "dataset/" (NOT "s3://my-bucket/dataset/"). Objects under the full prefix
            "s3://my-bucket/dataset/" will be downloaded.
        destination: Local directory where all objects should be saved,
            preserving the relative path structure from the prefix
        cache: Whether to enable intelligent caching for all downloaded objects
            (default: True). Each object gets its own cache validation

    Behavior:
        1. Lists all objects recursively under the specified prefix
        2. Creates individual ObjectDownloadRequest for each discovered object
        3. Preserves directory structure relative to the prefix
        4. Downloads objects in parallel across the cluster

    Examples:
        >>> # Download entire model directory
        >>> request = PrefixDownloadRequest(uri="models/bert-base/", destination=pathlib.Path("/tmp/models/"))

        >>> # Download dataset with specific profile
        >>> request = PrefixDownloadRequest(
        ...     profile_name="datasets", uri="imagenet/train/", destination=pathlib.Path("/data/imagenet/train/")
        ... )

        >>> # Download all tokenizer files
        >>> request = PrefixDownloadRequest(
        ...     uri="tokenizers/gpt2/", destination=pathlib.Path("/tmp/tokenizer/"), cache=True
        ... )

    Directory Structure Example:
        For prefix "s3://bucket/dataset/" containing:
        - s3://bucket/dataset/train/images/img1.jpg
        - s3://bucket/dataset/train/labels/label1.txt
        - s3://bucket/dataset/test/images/img2.jpg

        With destination "/local/data/", creates:
        - /local/data/train/images/img1.jpg
        - /local/data/train/labels/label1.txt
        - /local/data/test/images/img2.jpg

    Performance Notes:
        - Each file in the prefix is treated as a separate object for P2P distribution
        - Large directories benefit from the distributed chunking and sharing system
        - Consider the total number of files when tuning parallelism settings
        - For single large archives, ObjectDownloadRequest may be more efficient
    """

    uri: str
    destination: pathlib.Path
    cache: bool = True
    profile_name: str | None = None


@attrs.define
class DownloadRequest:
    """A request to download either a single object or a prefix of objects.

    This is a wrapper class that can contain either an ObjectDownloadRequest
    for downloading a single object, or a PrefixDownloadRequest for downloading
    multiple objects under a prefix.

    The profile_name in the contained request determines which ObjectStoreConfig
    to use for authentication and base URI configuration. See ObjectDownloadRequest
    and PrefixDownloadRequest for details on how URIs are interpreted as relative
    keys within the configured object store.
    """

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
    writing_chunks: set[uuid.UUID] = attrs.field(factory=set)
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
    num_active_file_writing_tasks: int = 0

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
        self.writing_chunks = set(rust_status.writing_chunks)
        self.available_chunks = set(rust_status.available_chunks)
        self.completed_or_cached_objects = set(rust_status.completed_or_cached_objects)
        self.unneeded_objects = set(rust_status.unneeded_objects)
        self.num_active_uploads = rust_status.num_active_uploads
        self.num_active_assembling_tasks = rust_status.num_active_assembling_tasks
        self.num_active_unpacking_tasks = rust_status.num_active_unpacking_tasks
        self.num_active_file_writing_tasks = rust_status.num_active_file_writing_tasks


@attrs.define
class ObjectStoreConfig:
    """Configuration for an object store using the object_store crate.

    This class provides configuration for connecting to various object storage backends
    including S3, GCS, Azure Blob Storage, and local filesystems. The object_store crate
    uses URIs to identify different storage backends and accepts additional configuration
    parameters via config_args.

    The object_store crate supports these URI schemes:
    - s3://bucket-name/ - Amazon S3 or S3-compatible storage
    - gs://bucket-name/ - Google Cloud Storage
    - azure://container-name/ - Azure Blob Storage
    - file:///path/to/directory/ - Local filesystem

    Common configuration parameters include:
    - access_key_id: AWS access key ID for S3
    - secret_access_key: AWS secret access key for S3
    - region: AWS region for S3 (e.g., "us-west-2")
    - endpoint: Custom endpoint URL for S3-compatible services
    - token: Authentication token for some services

    URI Relationship with Download Requests:
        The URI in this config serves as the "base URI" for object store operations.
        Download request URIs are interpreted as **relative keys/paths** within this base:

        - If ObjectStoreConfig has uri="s3://my-bucket/", then download requests
          specify relative keys like "path/to/file.txt" (not full URIs)
        - The full object path becomes base_uri + key = "s3://my-bucket/path/to/file.txt"
        - The profile_name in download requests (ObjectDownloadRequest,
          PrefixDownloadRequest) determines which ObjectStoreConfig to use
        - This mapping is handled by ObjectStoreConfigByProfile

    Examples:
        Basic S3 configuration:
        >>> config = ObjectStoreConfig.make_for_s3(
        ...     bucket="my-bucket",
        ...     endpoint="https://s3.amazonaws.com",
        ...     access_key_id="AKIAIOSFODNN7EXAMPLE",
        ...     secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ... )

        S3-compatible service (like MinIO):
        >>> config = ObjectStoreConfig.make_for_s3(
        ...     bucket="my-bucket",
        ...     endpoint="http://localhost:9000",
        ...     access_key_id="minioadmin",
        ...     secret_access_key="minioadmin",
        ... )

        Local filesystem:
        >>> config = ObjectStoreConfig.make_for_local("/tmp/storage")

    Attributes:
        uri: The object store URI identifying the backend and bucket/container
        config_args: Dictionary of additional configuration parameters passed to the object_store crate
    """

    uri: str
    config_args: dict[str, str] = attrs.field(factory=dict)

    # This is taken from obstore.store.ClientConfig
    # Valid HTTP client configuration fields
    _VALID_HTTP_CLIENT_FIELDS = {  # noqa: RUF012
        "allow_http",
        "allow_invalid_certificates",
        "connect_timeout",
        "default_content_type",
        "default_headers",
        "http1_only",
        "http2_keep_alive_interval",
        "http2_keep_alive_timeout",
        "http2_keep_alive_while_idle",
        "http2_only",
        "pool_idle_timeout",
        "pool_max_idle_per_host",
        "proxy_url",
        "timeout",
        "user_agent",
    }

    @staticmethod
    def is_client_option(config_key: str) -> bool:
        return config_key in ObjectStoreConfig._VALID_HTTP_CLIENT_FIELDS

    @classmethod
    def make_for_s3(
        cls,
        bucket: str,
        endpoint: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str | None = None,
        timeout_s: int = 300,
        connect_timeout_s: int = 60,
    ) -> ObjectStoreConfig:
        """Create an ObjectStoreConfig for S3 or S3-compatible storage.

        Args:
            bucket: S3 bucket name
            endpoint: S3 endpoint URL (e.g., "https://s3.amazonaws.com" for AWS S3) (optional)
            access_key_id: AWS access key ID (optional)
            secret_access_key: AWS secret access key (optional)
            region: AWS region (optional, e.g., "us-west-2") (optional)
            timeout_s: Timeout for the S3 operation in seconds (default: 300)
            connect_timeout_s: Connect timeout for the S3 operation in seconds (default: 60)

        Returns:
            ObjectStoreConfig configured for S3 access

        Example:
            >>> config = ObjectStoreConfig.make_for_s3(
            ...     bucket="my-data-bucket",
            ...     endpoint="https://s3.us-west-2.amazonaws.com",
            ...     access_key_id="some_access_key_id",
            ...     secret_access_key="some_secret_access_key",
            ...     region="us-west-2",
            ... )
        """
        config_args: dict[str, str] = {}
        if access_key_id is not None:
            config_args["access_key_id"] = access_key_id
        if secret_access_key is not None:
            config_args["secret_access_key"] = secret_access_key
        if endpoint is not None:
            config_args["endpoint"] = endpoint
        if region is not None:
            config_args["region"] = region

        config_args["timeout"] = f"{timeout_s}s"
        config_args["connect_timeout"] = f"{connect_timeout_s}s"

        return cls(uri=f"s3://{bucket}/", config_args=config_args)

    @classmethod
    def make_for_local(cls, directory_path: str) -> ObjectStoreConfig:
        """Create an ObjectStoreConfig for local filesystem storage.

        Args:
            directory_path: Local directory path to use as storage root

        Returns:
            ObjectStoreConfig configured for local filesystem access

        Example:
            >>> config = ObjectStoreConfig.make_for_local("/tmp/object_storage")
        """
        return cls(uri=f"file://{directory_path}", config_args={})

    @classmethod
    def make_for_gcs(cls, bucket: str, service_account_key: str | None = None) -> ObjectStoreConfig:
        """Create an ObjectStoreConfig for Google Cloud Storage.

        Args:
            bucket: GCS bucket name
            service_account_key: Optional path to service account key JSON file

        Returns:
            ObjectStoreConfig configured for GCS access

        Example:
            >>> config = ObjectStoreConfig.make_for_gcs(
            ...     bucket="my-gcs-bucket", service_account_key="/path/to/service-account.json"
            ... )
        """
        config_args = {}
        if service_account_key is not None:
            config_args["service_account_key"] = service_account_key

        return cls(uri=f"gs://{bucket}/", config_args=config_args)

    def to_rust(self) -> rust_models.ObjectStoreConfig:
        return rust_models.ObjectStoreConfig(uri=self.uri, config_args=self.config_args)


# @attrs.define
# class RetryConfig:
#     num_retries: int = 5
#     base_delay_s: float = 1.0
#     delay_factor: float = 2.0
#     max_delay_s: float = 16

#     def to_rust(self) -> rust_models.RetryConfig:
#         return rust_models.RetryConfig(
#             num_retries=self.num_retries,
#             base_delay_s=self.base_delay_s,
#             delay_factor=self.delay_factor,
#             max_delay_s=self.max_delay_s,
#         )


@attrs.define
class ObjectStoreConfigByProfile:
    """A mapping of ObjectStoreConfig by profile name.

    This class enables using multiple object store configurations within a single
    download session. Each profile maps to a different ObjectStoreConfig, allowing
    downloads from different buckets, storage services, or authentication contexts.

    When processing download requests, the profile_name in ObjectDownloadRequest or
    PrefixDownloadRequest determines which ObjectStoreConfig to use:
    - If profile_name is None, uses the config mapped to None (default profile)
    - If profile_name is "prod", uses the config mapped to "prod"
    - The download request URI is interpreted as a relative key within the selected config

    Example:
        configs = ObjectStoreConfigByProfile(profiles={
            None: ObjectStoreConfig.make_for_s3(bucket="default-bucket", ...),
            "prod": ObjectStoreConfig.make_for_s3(bucket="prod-bucket", ...),
            "dev": ObjectStoreConfig.make_for_s3(bucket="dev-bucket", ...),
        })

        # This request would use the "prod" profile's ObjectStoreConfig
        request = ObjectDownloadRequest(
            profile_name="prod",
            uri="data/file.txt",  # Relative key, becomes s3://prod-bucket/data/file.txt
            destination=Path("/local/file.txt")
        )
    """

    profiles: dict[str | None, ObjectStoreConfig]
    # retry_config: RetryConfig

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
                profile: obs.store.from_url(
                    config.uri,
                    config={k: v for k, v in config.config_args.items() if not ObjectStoreConfig.is_client_option(k)},  # pyright: ignore[reportArgumentType]
                    client_options={  # pyright: ignore[reportArgumentType]
                        k: v for k, v in config.config_args.items() if ObjectStoreConfig.is_client_option(k)
                    },
                )  # type: ignore
                for profile, config in config_by_profile.profiles.items()
            }
        )
