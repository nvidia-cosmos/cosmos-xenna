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

import pathlib
import uuid
from enum import Enum
from typing import Dict, List, Optional, Set

class UnpackMethod(Enum):
    Auto: UnpackMethod
    Tar: UnpackMethod
    TarGz: UnpackMethod
    Zip: UnpackMethod

class UnpackOptions:
    destination: pathlib.Path
    unpack_method: UnpackMethod
    def __init__(self, destination: pathlib.Path, unpack_method: UnpackMethod) -> None: ...

class ObjectToDownload:
    object_id: uuid.UUID
    parent_request_id: uuid.UUID
    profile_name: Optional[str]
    uri: str
    destination: pathlib.Path
    cache_info: CacheInfo
    unpack_options: Optional[UnpackOptions]
    def __init__(
        self,
        object_id: uuid.UUID,
        parent_request_id: uuid.UUID,
        profile_name: Optional[str],
        uri: str,
        destination: pathlib.Path,
        cache_info: CacheInfo,
        unpack_options: Optional[UnpackOptions],
    ) -> None: ...

class DownloadCatalog:
    objects: Dict[uuid.UUID, ObjectToDownload]
    chunks: Dict[uuid.UUID, ChunkToDownload]
    chunks_by_object: Dict[uuid.UUID, List[uuid.UUID]]
    def __init__(
        self,
        objects: List[ObjectToDownload],
        chunks: List[ChunkToDownload],
        chunks_by_object: Dict[uuid.UUID, List[uuid.UUID]],
    ) -> None: ...

class ByteRange:
    start: int
    end: int
    def __init__(self, start: int, end: int) -> None: ...

class ObjectAndRange:
    object_uri: str
    range: Optional[ByteRange]
    crc32_checksum: Optional[int]
    def __init__(self, object_uri: str, range: Optional[ByteRange], crc32_checksum: Optional[int]) -> None: ...

class ChunkToDownload:
    chunk_id: uuid.UUID
    parent_object_id: uuid.UUID
    profile_name: Optional[str]
    value: ObjectAndRange
    destination: pathlib.Path
    size: int
    def __init__(
        self,
        chunk_id: uuid.UUID,
        parent_object_id: uuid.UUID,
        profile_name: Optional[str],
        value: ObjectAndRange,
        destination: pathlib.Path,
        size: int,
    ) -> None: ...

class DownloadFromNodeOrder:
    download_chunk: ChunkToDownload
    source_node_id: str
    source_node_ip: str
    source_node_port: int
    def __init__(
        self, download_chunk: ChunkToDownload, source_node_id: str, source_node_ip: str, source_node_port: int
    ) -> None: ...

class Orders:
    download_from_s3: List[ChunkToDownload]
    download_from_node: List[DownloadFromNodeOrder]
    def __init__(
        self, download_from_s3: List[ChunkToDownload], download_from_node: List[DownloadFromNodeOrder]
    ) -> None: ...

class ObjectMetadata:
    size: int
    last_modified: str
    def __init__(self, size: int, last_modified: str) -> None: ...

class ObjectNameAndMetadata:
    uri: str
    metadata: ObjectMetadata
    def __init__(self, uri: str, metadata: ObjectMetadata) -> None: ...

class NodeStatus:
    node_id: str
    downloading_p2p_chunks: Set[uuid.UUID]
    downloading_s3_chunks: Set[uuid.UUID]
    writing_chunks: Set[uuid.UUID]
    available_chunks: Set[uuid.UUID]
    completed_or_cached_objects: Set[uuid.UUID]
    unneeded_objects: Set[uuid.UUID]
    num_active_uploads: int
    num_active_assembling_tasks: int
    num_active_unpacking_tasks: int
    num_active_file_writing_tasks: int

class CacheInfo:
    uri: str
    size: int
    last_modified_unix_micros: int
    def __init__(self, uri: str, size: int, last_modified_unix_micros: int) -> None: ...

class NodeMetadata:
    @property
    def node_id(self) -> str: ...
    @property
    def ip(self) -> str: ...
    @property
    def port(self) -> int: ...

class ObjectStoreConfig:
    uri: str
    config_args: Dict[str, str]
    def __init__(self, uri: str, config_args: Dict[str, str]) -> None: ...

class ObjectStoreConfigByProfile:
    def __init__(self, profiles: Dict[Optional[str], ObjectStoreConfig]) -> None: ...

class ObjectStoreByProfile:
    def __init__(self, config_by_profile: ObjectStoreConfigByProfile) -> None: ...

class RetryConfig:
    num_retries: int
    base_delay_millis: int

    def __init__(self, num_retries: int, base_delay_millis: int) -> None: ...
