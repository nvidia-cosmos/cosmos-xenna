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
from cosmos_xenna.pipelines.private import allocator
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

    def is_spmd(self) -> bool:
        return self._r.is_spmd()

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


class WorkerGroup:
    @classmethod
    def make(cls, id: str, stage_name: str, allocations: list[WorkerResourcesInternal]) -> WorkerGroup:
        return cls(rust.WorkerGroup(id, stage_name, [x.to_rust() for x in allocations]))

    def __init__(self, rust_worker: rust.WorkerGroup):
        self._r = rust_worker

    @property
    def id(self) -> str:
        return self._r.id

    @property
    def stage_name(self) -> str:
        return self._r.stage_name

    @property
    def allocations(self) -> list[WorkerResourcesInternal]:
        return [WorkerResourcesInternal.from_rust(x) for x in self._r.allocations]

    @property
    def rust(self) -> rust.WorkerGroup:
        return self._r

    def split_allocation_per_gpu(self) -> list[WorkerResourcesInternal]:
        """Splits the worker group's allocations into separate WorkerResources for each GPU.

        This method is useful for distributed training/inference scenarios where you need to treat
        each GPU as a separate worker with its own resource allocation. The CPUs are
        divided evenly among all GPUs in each allocation.

        Returns:
            list[WorkerResources]: A list of WorkerResources, one for each GPU in the
                worker group. Each entry contains:
                - The same node as the original allocation
                - A fraction of the CPUs (total CPUs / number of GPUs in that allocation)
                - A single GPU allocation

        Example:
            If a worker group has an allocation with 8 CPUs and 4 GPUs, this method will
            return 4 WorkerResources entries, each with 2 CPUs and 1 GPU.
        """
        return [WorkerResourcesInternal.from_rust(x) for x in self._r.split_allocation_per_gpu()]

    def __reduce__(self) -> Any:
        """Make the class pickleable by serializing the Rust object to a string."""
        # Serialize the Rust object to a string
        serialized = self._r.serialize()
        # Return a tuple: (callable, args) where callable reconstructs the object
        return (self._reconstruct, (serialized,))

    @classmethod
    def _reconstruct(cls, serialized: str) -> WorkerGroup:
        """Reconstruct a Worker from a serialized string."""
        # Deserialize the string back to a Rust Worker
        rust_worker = rust.WorkerGroup.deserialize(serialized)
        # Create a new Python WorkerGroup instance
        return cls(rust_worker)

    def __hash__(self) -> int:
        return hash(self._r.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorkerGroup):
            return False
        return self._r.id == other._r.id

    def __repr__(self) -> str:
        return self._r.__repr__()

    def __str__(self) -> str:
        return self._r.__str__()


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
class GpuAllocationInternal:
    """Represents the allocation a worker is taking up for a given GPU.

    This class describes how much of a GPU's resources are allocated to a worker.
    It's a lightweight reference that points to a GPU in a node's GPU list rather
    than storing full GPU details.

    Attributes:
        offset: **Index into the node's NodeResources.gpus list**, not the hardware GPU index.
                This indirection allows the same allocation to be used with different nodes,
                and keeps the allocation struct small. To get the actual hardware GPU index
                or UUID, you must look up node_resources.gpus[offset].

        used_fraction: Fraction of the GPU's compute capacity allocated (0.0 to 1.0).
                      For whole-GPU allocations, this is 1.0. For fractional allocations,
                      this can be any value like 0.25, 0.5, etc.

    Important: Offset vs. GPU Index
        The `offset` field is **not** the hardware GPU index! It's the position in the
        NodeResources.gpus list. For example:
        - If a node has 4 GPUs and you want GPU at hardware index 2, you need to find
          which position in the gpus list corresponds to that GPU.
        - The actual hardware index is stored in GpuResources.index
        - The GPU UUID is stored in GpuResources.uuid_

    Example:
        >>> # Create an allocation for the first GPU in a node's list (offset=0)
        >>> # using 50% of its capacity
        >>> alloc = GpuAllocation(offset=0, used_fraction=0.5)
        >>>
        >>> # To get the actual hardware GPU index:
        >>> # hardware_index = node_resources.gpus[alloc.offset].index
    """

    offset: int
    used_fraction: float

    @classmethod
    def from_rust(cls, rust_gpu_allocation: rust.GpuAllocation) -> GpuAllocationInternal:
        return GpuAllocationInternal(
            offset=rust_gpu_allocation.offset,
            used_fraction=rust_gpu_allocation.used_fraction,
        )

    def to_rust(self) -> rust.GpuAllocation:
        return rust.GpuAllocation(
            offset=self.offset,
            used_fraction=self.used_fraction,
        )


@attrs.define
class WorkerResourcesInternal:
    """Like WorkerResources, but track gpu offsets rather than indices.

    This is used internally, becuase it is easier to track gpu offsets (the offset into the node's visible gpus list)
    rather than indices, but it is confusing to use show it to users, which is what we use WorkerResources for.
    """

    node: str
    cpus: float
    gpus: list[GpuAllocationInternal]

    @staticmethod
    def from_rust(r: rust.WorkerMetadata) -> WorkerResourcesInternal:
        return WorkerResourcesInternal(
            r.node,
            r.cpus,
            [GpuAllocationInternal.from_rust(x) for x in r.gpus],
        )

    def to_rust(self) -> rust.WorkerResources:
        return rust.WorkerResources(
            node=self.node,
            cpus=self.cpus,
            gpus=[x.to_rust() for x in self.gpus],
        )


@attrs.define
class GpuAllocation:
    """Tracks the allocation of a GPU to a worker."""

    # The hardward index of the GPU (e.g. 0, 1, 2, 3)
    index: int
    # Fraction of the GPU's compute capacity the worker has been allocated.
    used_fraction: float


@attrs.define
class WorkerResources:
    """Tracks the resources a worker is taking up."""

    # Node the worker is running on.
    node: str
    # Number of CPUs the worker has been allocated.
    cpus: float
    # List of GPUs the worker has been allocated.
    gpus: list[GpuAllocation]

    @classmethod
    def from_internal(cls, internal: WorkerResourcesInternal, allocator: allocator.WorkerAllocator) -> WorkerResources:
        return WorkerResources(
            node=internal.node,
            cpus=internal.cpus,
            gpus=[
                GpuAllocation(index=allocator.get_gpu_index(internal.node, gpu.offset), used_fraction=gpu.used_fraction)
                for gpu in internal.gpus
            ],
        )


@attrs.define
class NcclRendevousParams:
    master_addr: str
    master_port: int


@attrs.define
class DistributedExecutionParams:
    """
    Parameters required for distributed training and inference, specifically SPMD/torchrun style training/inference.

    Attributes:
        rank: Global rank of this process.
        world_size: Total number of processes.
        local_rank: Rank within the current node.
        master_addr: Address of the master node.
        master_port: Port used for communication.
    """

    # Global rank of this gpu.
    rank: int
    # Total number of gpus
    world_size: int
    # Rank within the current node
    # Usually interpreted as the local gpu id.
    local_rank: int
    # Address of the master node.
    master_addr: str
    # Port used for communication on the master node.
    master_port: int

    @classmethod
    def from_env_vars(cls, env_vars: dict[str, str] | None = None) -> DistributedExecutionParams:
        if env_vars is None:
            env_vars = dict(os.environ)  # Use current environment variables.
        return cls(
            rank=int(env_vars["RANK"]),
            world_size=int(env_vars["WORLD_SIZE"]),
            local_rank=int(env_vars["LOCAL_RANK"]),
            master_addr=env_vars["MASTER_ADDR"],
            master_port=int(env_vars["MASTER_PORT"]),
        )

    def to_env_var_dict(self) -> dict[str, str]:
        """Create a dict which can be used to set the env vars.

        This follows what torch does.
        """
        return {
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }


@attrs.define
class WorkerMetadata:
    worker_id: str
    worker_group_id: str
    allocation: WorkerResources
    should_set_cuda_visible_devices: bool
    world_size: int | None
    local_rank: int | None
    rank: int | None
    rendevous_params: NcclRendevousParams | None

    @staticmethod
    def make_dummy() -> WorkerMetadata:
        return WorkerMetadata(
            worker_id="debug_worker",
            worker_group_id="debug_worker_group",
            allocation=WorkerResources(node="debug_node", cpus=1.0, gpus=[]),
            should_set_cuda_visible_devices=False,
            world_size=None,
            local_rank=None,
            rank=None,
            rendevous_params=None,
        )

    @property
    def distributed_execution_params(self) -> DistributedExecutionParams | None:
        if self.rank is None or self.world_size is None or self.rendevous_params is None or self.local_rank is None:
            return None
        return DistributedExecutionParams(
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            master_addr=self.rendevous_params.master_addr,
            master_port=self.rendevous_params.master_port,
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

    Attributes:
        cpus: Number of CPU cores required (can be fractional)
        gpus: Number of GPUs required (can be fractional for single-GPU stages,
              must be integer for SPMD stages)
        is_spmd: Whether this stage requires SPMD (Single Program, Multiple Data)
                 execution for distributed inference across multiple GPUs/nodes

    SPMD Mode:
        When is_spmd=True, the stage runs in distributed inference mode, emulating
        torchrun behavior. This is required for:
        - Models/frameworks that only support multi-GPU inference via SPMD coordination (e.g. diffusion models)
        - Multi-node inference with frameworks like vLLM

        SPMD stages get special treatment:
        - CUDA_VISIBLE_DEVICES is not modified (follows torchrun behavior)
        - Environment variables set for distributed coordination (MASTER_ADDR, MASTER_PORT, RANK, etc.)
        - One actor per GPU, all coordinated as a single worker group
        - NCCL rendezvous parameters automatically configured

    Examples:
        >>> # Single GPU stage
        >>> Resources(cpus=2.0, gpus=1.0)

        >>> # Multi-GPU SPMD stage (for large model inference)
        >>> Resources(cpus=1.0, gpus=8, is_spmd=True)

        >>> # CPU-only stage
        >>> Resources(cpus=4.0, gpus=0.0)

    See `yotta.ray_utils._specs.Stage.required_resources` for much more info.
    """

    cpus: float = 0.0
    gpus: Union[float, int] = 0
    is_spmd: bool = False

    def to_dict(self) -> dict[str, float]:
        return {"cpu": self.cpus, "gpu": self.gpus}

    def to_rust(self) -> rust.Resources:
        return rust.Resources(
            cpus=self.cpus,
            gpus=self.gpus,
            is_spmd=self.is_spmd,
        )

    def to_worker_shape(self, cluster_resources: ClusterResources) -> WorkerShape:
        return WorkerShape(self.to_rust().to_shape(cluster_resources.to_rust()))

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


@attrs.define
class ResourceInfoFromRay:
    node_id: str
    cpus: float | None
    gpus: float | None


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
    reported_resources: dict[str, ResourceInfoFromRay] = {}
    for node in nodes:
        node_id = node["NodeID"]
        node_name = node.get("NodeManagerHostname", "unknown")
        alive = node.get("Alive", True)
        if not alive:
            logger.warning(f"Node {node_id} on {node_name} is not alive?? Skipping it.")
            continue
        alive_nodes.append(node_id)
        reported_resources[node_id] = ResourceInfoFromRay(
            node_id=node_id,
            cpus=node.get("Resources", {}).get("CPU", None),
            gpus=node.get("Resources", {}).get("GPU", None),
        )

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
            total_cpus=int(_get_cpu_count(reported_resources[node_id], info) * cpu_allocation_percentage),
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


def _get_cpu_count(info_from_ray: ResourceInfoFromRay, info_from_node: ResourceInfoFromNode) -> int:
    # use the Ray-reported CPU count if available, otherwise use the number of CPUs from the node info
    if info_from_ray.cpus is not None and isinstance(info_from_ray.cpus, float) and info_from_ray.cpus > 0:
        return int(info_from_ray.cpus)
    return info_from_node.cpus
