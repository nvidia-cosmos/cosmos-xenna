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

import uuid
from typing import Dict, List, Optional

class WorkerShape:
    def get_num_cpus(self) -> float: ...
    def get_num_gpus(self) -> float: ...

class WorkerMetadata:
    worker_id: str
    allocation: WorkerResources

    def __init__(self, worker_id: str, allocation: WorkerResources) -> None: ...

class Resources:
    cpus: float
    gpus: float
    nvdecs: int
    nvencs: int
    entire_gpu: bool

    def __init__(self, cpus: float, gpus: float, nvdecs: int, nvencs: int, entire_gpu: bool) -> None: ...
    def to_pool(self, num_nvdecs_per_gpu: int, num_nvencs_per_gpu: int) -> PoolOfResources: ...
    def to_shape(self) -> WorkerShape: ...

class GPUAllocation:
    gpu_index: int
    fraction: float

    def __init__(self, gpu_index: int, fraction: float) -> None: ...

class CodecAllocation:
    gpu_index: int
    codec_index: int

    def __init__(self, gpu_index: int, codec_index: int) -> None: ...

class PoolOfResources:
    cpus: float
    gpus: float
    nvdecs: float
    nvencs: float

    def __init__(self, cpus: float, gpus: float, nvdecs: float, nvencs: float) -> None: ...
    def total_num(self) -> float: ...
    def multiply_by(self, factor: float) -> "PoolOfResources": ...
    def add(self, other: "PoolOfResources") -> "PoolOfResources": ...
    def sub(self, other: "PoolOfResources") -> "PoolOfResources": ...
    def div(self, other: "PoolOfResources") -> "PoolOfResources": ...
    def contains(self, other: "PoolOfResources") -> bool: ...
    def to_dict(self) -> Dict[str, float]: ...

class GpuResources:
    index: int
    uuid_: uuid.UUID
    gpu_fraction: float

    def __init__(
        self, index: int, uuid_: uuid.UUID, gpu_fraction: float, nvdecs: List[int], nvencs: List[int]
    ) -> None: ...
    @staticmethod
    def make_from_num_codecs(gpu_fraction_available: float, num_nvdecs: int, num_nvencs: int) -> "GpuResources": ...
    def num_nvdecs(self) -> int: ...
    def num_nvencs(self) -> int: ...
    def totals(self) -> PoolOfResources: ...

class WorkerResources:
    node: str
    cpus: float
    gpus: List[GPUAllocation]
    nvdecs: List[CodecAllocation]
    nvencs: List[CodecAllocation]

    def __init__(
        self,
        node: str,
        cpus: float,
        gpus: Optional[List[GPUAllocation]] = ...,
        nvdecs: Optional[List[CodecAllocation]] = ...,
        nvencs: Optional[List[CodecAllocation]] = ...,
    ) -> None: ...
    def validate(self) -> None: ...
    def to_pool(self) -> PoolOfResources: ...

class NodeResources:
    cpus: float
    gpus: List[GpuResources]
    name: Optional[str]

    def __init__(self, cpus: float, gpus: Optional[List[GpuResources]] = ..., name: Optional[str] = ...) -> None: ...
    @staticmethod
    def make_uniform(
        num_cpus: int, num_gpus: int, num_nvdecs_per_gpu: int, num_nvencs_per_gpu: int
    ) -> "NodeResources": ...
    def totals(self) -> PoolOfResources: ...
    def copy_and_allocate(self, resources: WorkerResources) -> "NodeResources": ...
    def allocate(self, resources: WorkerResources) -> None: ...
    def release_allocation(self, resources: WorkerResources) -> None: ...

class ClusterResources:
    nodes: Dict[str, NodeResources]

    def __init__(self, nodes: Optional[Dict[str, NodeResources]] = ...) -> None: ...
    @staticmethod
    def make_uniform(node_resources: NodeResources, node_ids: List[str]) -> "ClusterResources": ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_gpus(self) -> int: ...
    def calc_num_nvdecs_per_gpu(self) -> int: ...
    def calc_num_nvencs_per_gpu(self) -> int: ...
    def totals(self) -> PoolOfResources: ...
    def is_overallocated(self) -> bool: ...
    def copy_and_clear_allocation(self, resources: WorkerResources) -> "ClusterResources": ...
    def clear_allocation(self, resources: WorkerResources) -> None: ...

class Worker:
    id: str
    stage_name: str
    allocation: WorkerResources

    def __init__(self, id: str, stage_name: str, allocation: WorkerResources) -> None: ...

class NodeInfo:
    node_id: str

    def __init__(self, node_id: str) -> None: ...
