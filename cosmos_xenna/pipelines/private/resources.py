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

"""Data structures used to represent allocated/available resources on a cluster/node/gpu.

Many of the classes in this module are "shapes". A shape is a fully specified resource requirement for something.
Shapes are meant to specified by users on a per-stage basis.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional, Union

import attrs
import ray
import ray.util.scheduling_strategies

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import resources as rust  # type: ignore
from cosmos_xenna.utils import python_log as logger

try:
    import pynvml

    HAS_NVML = True
except ImportError:
    pynvml = None
    HAS_NVML = False


class AllocationError(Exception):
    pass


@attrs.define
class PoolOfResources:
    cpus: float
    gpus: float

    def add(self, other: PoolOfResources) -> PoolOfResources:
        return PoolOfResources(cpus=self.cpus + other.cpus, gpus=self.gpus + other.gpus)

    def multiply_by(self, other: int) -> PoolOfResources:
        return PoolOfResources(cpus=self.cpus * other, gpus=self.gpus * other)


class WorkerShape:
    def __init__(self, rust_worker_shape: rust.WorkerShape):
        self._r = rust_worker_shape

    @property
    def rust(self) -> rust.WorkerShape:
        return self._r

    def get_num_cpus(self) -> float:
        return self._r.get_num_cpus()

    def get_num_gpus(self) -> float:
        return self._r.get_num_gpus()

    def __reduce__(self) -> Any:
        """Make the class pickleable by serializing the Rust object to a string."""
        # Serialize the Rust object to a string
        serialized = self._r.serialize()
        # Return a tuple: (callable, args) where callable reconstructs the object
        return (self._reconstruct, (serialized,))

    @classmethod
    def _reconstruct(cls, serialized: str) -> WorkerShape:
        """Reconstruct a WorkerShape from a serialized string."""
        # Deserialize the string back to a Rust WorkerShape
        rust_worker_shape = rust.WorkerShape.deserialize(serialized)
        # Create a new Python WorkerShape instance
        return cls(rust_worker_shape)


class Worker:
    @classmethod
    def make(cls, id: str, stage_name: str, allocation: WorkerResources) -> Worker:
        return cls(rust.Worker(id, stage_name, allocation.to_rust()))

    def __init__(self, rust_worker: rust.Worker):
        self._r = rust_worker

    @property
    def id(self) -> str:
        return self._r.id

    @property
    def stage_name(self) -> str:
        return self._r.stage_name

    @property
    def allocation(self) -> WorkerResources:
        return WorkerResources.from_rust(self._r.allocation)

    @property
    def rust(self) -> rust.Worker:
        return self._r

    def __reduce__(self) -> Any:
        """Make the class pickleable by serializing the Rust object to a string."""
        # Serialize the Rust object to a string
        serialized = self._r.serialize()
        # Return a tuple: (callable, args) where callable reconstructs the object
        return (self._reconstruct, (serialized,))

    @classmethod
    def _reconstruct(cls, serialized: str) -> Worker:
        """Reconstruct a Worker from a serialized string."""
        # Deserialize the string back to a Rust Worker
        rust_worker = rust.Worker.deserialize(serialized)
        # Create a new Python Worker instance
        return cls(rust_worker)


@attrs.define
class GpuResources:
    index: int
    uuid_: uuid.UUID
    used_fraction: float

    @classmethod
    def from_rust(cls, rust_gpu_resources: rust.GpuResources) -> GpuResources:
        return GpuResources(
            index=rust_gpu_resources.index,
            uuid_=rust_gpu_resources.uuid_,
            used_fraction=rust_gpu_resources.used_fraction,
        )

    def to_rust(self) -> rust.GpuResources:
        return rust.GpuResources(
            index=self.index,
            uuid_=self.uuid_,
            used_fraction=self.used_fraction,
        )


@attrs.define
class GpuAllocation:
    index: int
    used_fraction: float

    @classmethod
    def from_rust(cls, rust_gpu_allocation: rust.GPUAllocation) -> GpuAllocation:
        return GpuAllocation(
            index=rust_gpu_allocation.index,
            used_fraction=rust_gpu_allocation.used_fraction,
        )

    def to_rust(self) -> rust.GPUAllocation:
        return rust.GPUAllocation(
            index=self.index,
            used_fraction=self.used_fraction,
        )


@attrs.define
class WorkerResources:
    node: str
    cpus: float
    gpus: list[GpuAllocation]

    @staticmethod
    def from_rust(r: rust.WorkerMetadata) -> WorkerResources:
        return WorkerResources(
            r.node,
            r.cpus,
            [GpuAllocation.from_rust(x) for x in r.gpus],
        )

    def to_rust(self) -> rust.WorkerResources:
        return rust.WorkerResources(
            node=self.node,
            cpus=self.cpus,
            gpus=[x.to_rust() for x in self.gpus],
        )


@attrs.define
class WorkerMetadata:
    worker_id: str
    allocation: WorkerResources

    @staticmethod
    def make_dummy() -> WorkerMetadata:
        return WorkerMetadata(
            worker_id="debug_worker",
            allocation=rust.WorkerResources(node="debug_node", cpus=1.0, gpus=[]),
        )

    @staticmethod
    def from_rust(rust_worker_metadata: rust.WorkerMetadata) -> WorkerMetadata:
        return WorkerMetadata(
            worker_id=rust_worker_metadata.worker_id,
            allocation=rust_worker_metadata.allocation,
        )


@attrs.define
class NodeInfo:
    node_id: str

    @staticmethod
    def from_rust(rust_node_info: rust.NodeInfo) -> NodeInfo:
        return NodeInfo(node_id=rust_node_info.node_id)


@attrs.define
class Resources:
    """A user friendly way to specify the resources required for something.

    This class provides an intuitive interface for specifying resource requirements
    that get translated into more detailed internal worker shapes.

    See `yotta.ray_utils._specs.Stage.required_resources` for much more info.
    """

    cpus: float = 0.0
    gpus: Union[float, int] = 0

    def to_dict(self) -> dict[str, float]:
        return {"cpu": self.cpus, "gpu": self.gpus}

    def to_rust(self) -> rust.Resources:
        return rust.Resources(
            cpus=self.cpus,
            gpus=self.gpus,
        )

    def to_worker_shape(self) -> WorkerShape:
        return WorkerShape(self.to_rust().to_shape())

    def to_pool(self) -> PoolOfResources:
        return PoolOfResources(cpus=self.cpus, gpus=self.gpus)

    def __repr__(self) -> str:
        return repr(self.to_rust())

    def __str__(self) -> str:
        return repr(self)


@attrs.define
class NodeResources:
    used_cpus: float
    total_cpus: float
    gpus: list[GpuResources]
    name: Optional[str]

    @staticmethod
    def from_rust(rust_node_resources: rust.NodeResources) -> NodeResources:
        return NodeResources(
            used_cpus=rust_node_resources.used_cpus,
            total_cpus=rust_node_resources.total_cpus,
            gpus=[GpuResources.from_rust(x) for x in rust_node_resources.gpus],
            name=rust_node_resources.name,
        )

    def to_rust(self) -> rust.NodeResources:
        return rust.NodeResources(
            used_cpus=self.used_cpus,
            total_cpus=self.total_cpus,
            gpus=[x.to_rust() for x in self.gpus],
            name=self.name,
        )


@attrs.define
class ClusterResources:
    nodes: dict[str, NodeResources]

    @staticmethod
    def from_rust(rust_cluster_resources: rust.ClusterResources) -> ClusterResources:
        return ClusterResources(
            nodes={k: NodeResources.from_rust(v) for k, v in rust_cluster_resources.nodes.items()},
        )

    def total_pool(self) -> PoolOfResources:
        return PoolOfResources(
            cpus=sum(node.total_cpus for node in self.nodes.values()),
            gpus=sum(len(node.gpus) for node in self.nodes.values()),
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def to_rust(self) -> rust.ClusterResources:
        return rust.ClusterResources(nodes={k: v.to_rust() for k, v in self.nodes.items()})


@attrs.define
class GpuInfo:
    index: int
    name: str
    uuid_: uuid.UUID


@attrs.define
class ResourceInfoFromNode:
    node_id: str
    cpus: int
    gpus: list[GpuInfo]


def parse_visible_cuda_devices(cuda_visible_devices: Optional[str]) -> list[int | uuid.UUID | str] | None:
    """Parse a CUDA_VISIBLE_DEVICES string into typed tokens.

    Returns a list where each element is one of:
    - int: a GPU index
    - uuid.UUID: a full GPU UUID (regardless of whether "GPU-" prefix was given)
    - str: a normalized short UUID prefix (no "GPU-" prefix)

    If the input is None, returns None.
    Raises ValueError for malformed tokens (e.g., "GPU-" with no content).
    """
    if cuda_visible_devices is None:
        return None

    tokens = [tok.strip() for tok in cuda_visible_devices.split(",") if tok.strip()]
    out: list[int | uuid.UUID | str] = []
    for tok in tokens:
        # Try index
        try:
            out.append(int(tok))
            continue
        except ValueError:
            pass

        tok_norm = tok.strip()
        if tok_norm.lower().startswith("gpu-"):
            tok_norm = tok_norm[4:]

        # Try full UUID
        try:
            out.append(uuid.UUID(tok_norm))
            continue
        except ValueError:
            pass

        # Otherwise, treat as short UUID prefix. Normalize by removing hyphens.
        if not tok_norm:
            raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES token: {tok}") from None
        out.append(tok_norm)

    return out


def filter_gpus_by_cuda_visible_devices(gpus: list[GpuInfo], cuda_visible_devices: Optional[str]) -> list[GpuInfo]:
    """Return GPUs filtered according to a CUDA_VISIBLE_DEVICES string.

    Supports:
    - index-based lists (e.g. "0,2")
    - full UUIDs with or without the "GPU-" prefix (e.g. "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    - short UUID prefixes with or without the "GPU-" prefix (e.g. "GPU-3b7c", "3b7c", "3b7c8a10")

    If the string is is None, returns the input list unchanged.
    """
    parsed = parse_visible_cuda_devices(cuda_visible_devices)
    if parsed is None:
        return gpus

    allowed_indices: set[int] = {p for p in parsed if isinstance(p, int)}
    allowed_full_uuids: set[uuid.UUID] = {p for p in parsed if isinstance(p, uuid.UUID)}
    # Strings are normalized compact prefixes (no "GPU-" prefix)
    allowed_uuid_prefixes: set[str] = {p for p in parsed if isinstance(p, str)}

    filtered: list[GpuInfo] = []
    for gpu in gpus:
        if gpu.index in allowed_indices:
            filtered.append(gpu)
            continue
        if isinstance(gpu.uuid_, uuid.UUID):
            if gpu.uuid_ in allowed_full_uuids:
                filtered.append(gpu)
                continue
            uuid_str = str(gpu.uuid_)
            if any(uuid_str.startswith(p) for p in allowed_uuid_prefixes):
                filtered.append(gpu)

    return filtered


def get_local_gpu_info() -> list[GpuInfo]:
    """Uses pynvml to get information about GPUs on the local node."""
    gpus = []
    if not HAS_NVML:
        logger.warning("pynvml is not installed. Assuming no GPUs.")
        return []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
            # nvml returns bytes of the form b"GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
            if isinstance(raw_uuid, bytes):
                uuid_str = raw_uuid.decode("utf-8", errors="ignore")
            else:
                uuid_str = str(raw_uuid)
            if uuid_str.lower().startswith("gpu-"):
                uuid_str = uuid_str[4:]
            parsed_uuid = uuid.UUID(uuid_str)
            gpus.append(GpuInfo(index=i, name=str(name), uuid_=parsed_uuid))
    except pynvml.NVMLError as e:
        logger.warning(f"Could not initialize NVML or get GPU info: {e}. Assuming no GPUs.")
        # Return empty list if NVML fails (e.g., no NVIDIA driver)
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            # Ignore shutdown errors if initialization failed
            pass
    return gpus


def _respect_cuda_visible_devices(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """Filter GPUs to those listed in CUDA_VISIBLE_DEVICES, if set.

    Supports:
    - index-based lists (e.g. "0,2")
    - full UUIDs with or without the "GPU-" prefix (e.g. "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    - short UUID prefixes with or without the "GPU-" prefix (e.g. "GPU-3b7c", "3b7c")

    If the env var is not set, returns the input list unchanged.
    """
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    return filter_gpus_by_cuda_visible_devices(gpus, cuda_visible_devices)


@ray.remote
def _get_node_info_from_current_node() -> ResourceInfoFromNode:
    """Get the resources for a node."""
    node_id = ray.get_runtime_context().get_node_id()
    num_cpus = os.cpu_count()
    if num_cpus is None:
        raise ValueError("Could not determine number of CPUs on this node.")
    gpus = _respect_cuda_visible_devices(get_local_gpu_info())
    if not gpus:
        return ResourceInfoFromNode(node_id=node_id, cpus=num_cpus, gpus=[])
    return ResourceInfoFromNode(
        node_id=node_id,
        cpus=num_cpus,
        gpus=[GpuInfo(index=x.index, name=x.name, uuid_=x.uuid_) for x in gpus],
    )


def make_cluster_resources_for_ray_cluster(
    cpu_allocation_percentage: float = 1.0,
    nodes: Optional[list] = None,
) -> ClusterResources:
    """
    Make a ClusterResources object for a ray cluster.

    If nodes is None, calls ray.nodes() to get a list of connected nodes.

    ray.nodes() returns something which looks like this:
    [
        {
            "NodeID": "xx",
            "Alive": true,
            "NodeManagerAddress": "xx",
            "NodeManagerHostname": "xx",
            "NodeManagerPort": 11,
            "ObjectManagerPort": 11,
            "ObjectStoreSocketName": "/tmp/ray/session_2024-08-23_09-07-26_009842_799459/sockets/plasma_store",
            "RayletSocketName": "/tmp/ray/session_2024-08-23_09-07-26_009842_799459/sockets/raylet",
            "MetricsExportPort": 11,
            "NodeName": "xx",
            "RuntimeEnvAgentPort": 11,
            "alive": true,
            "Resources": {
                "GPU": 1.0,
                "accelerator_type:RTX": 1.0,
                "memory": 11,
                "node:__internal_head__": 1.0,
                "object_store_memory": 11,
                "node:xx": 1.0,
                "CPU":11
            },
            "Labels": {
                "ray.io/node_id": "xx"
            }
        },
        ...
    ]

    We will use this node info to collect the number of CPUS and GPUs for each node. We also rely on a
    user-provided "resources_per_gpu" parameter. This parameter tells use how many NVDECs/NVENCs are on each
    GPU. Ideally, which is something Ray does not give us.
    """
    if nodes is None:
        nodes = ray.nodes()

    out_dict = {}
    alive_nodes: list[str] = []
    for node in nodes:
        node_id = node["NodeID"]
        node_name = node.get("NodeManagerHostname", "unknown")
        alive = node.get("Alive", True)
        if not alive:
            logger.warning(f"Node {node_id} on {node_name} is not alive?? Skipping it.")
            continue
        alive_nodes.append(node_id)

    futures = [
        _get_node_info_from_current_node.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=x,
                soft=False,  # 'soft=False' means the task will fail if the node is not available
            )
        ).remote()
        for x in alive_nodes
    ]
    logger.debug(f"Waiting for {len(futures)} node info futures to complete...")
    infos: list[ResourceInfoFromNode] = ray.get(futures)
    logger.debug(f"Node info futures completed. Results: {infos}")

    for node_id, info in zip(alive_nodes, infos):
        out_dict[str(node_id)] = NodeResources(
            used_cpus=0.0,
            total_cpus=int(info.cpus * cpu_allocation_percentage),
            gpus=[
                GpuResources(
                    index=x.index,
                    uuid_=x.uuid_,
                    used_fraction=0.0,
                )
                for x in info.gpus
            ],
            name=str(node_id),
        )

    out = ClusterResources(out_dict)
    return out
