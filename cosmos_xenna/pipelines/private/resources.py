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
from typing import Optional, Union

import attrs
import ray
import ray.util.scheduling_strategies

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import resources as rust_resources
from cosmos_xenna.utils import python_log as logger

try:
    import pynvml

    HAS_NVML = True
except ImportError:
    pynvml = None
    HAS_NVML = False


CICD_ENV_VAR = "IS_RUNNING_IN_CICD"


class AllocationError(Exception):
    pass


@attrs.define
class WorkerMetadata:
    worker_id: str
    allocation: rust_resources.WorkerResources

    @staticmethod
    def make_dummy() -> WorkerMetadata:
        return WorkerMetadata(
            worker_id="debug_worker",
            allocation=rust_resources.WorkerResources(node="debug_node", cpus=1.0, gpus=[]),
        )

    @staticmethod
    def from_rust(rust_worker_metadata: rust_resources.WorkerMetadata) -> WorkerMetadata:
        return WorkerMetadata(
            worker_id=rust_worker_metadata.worker_id,
            allocation=rust_worker_metadata.allocation,
        )


@attrs.define
class NodeInfo:
    node_id: str

    @staticmethod
    def from_rust(rust_node_info: rust_resources.NodeInfo) -> NodeInfo:
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
    nvdecs: int = 0
    nvencs: int = 0
    entire_gpu: bool = False

    def to_dict(self) -> dict[str, float]:
        return {"cpu": self.cpus, "gpu": self.gpus, "nvdecs": self.nvdecs, "nvencs": self.nvencs}

    def to_rust(self) -> rust_resources.Resources:
        return rust_resources.Resources(
            cpus=self.cpus,
            gpus=self.gpus,
            nvdecs=self.nvdecs,
            nvencs=self.nvencs,
            entire_gpu=self.entire_gpu,
        )


@attrs.define
class GpuResources:
    index: int
    uuid_: uuid.UUID
    num_nvdecs: int = 0
    num_nvencs: int = 0

    def to_rust(self) -> rust_resources.GpuResources:
        return rust_resources.GpuResources.make_from_num_codecs(
            gpu_fraction_available=1.0,
            num_nvdecs=self.num_nvdecs,
            num_nvencs=self.num_nvencs,
        )


def _make_nvdecs_and_nvencs_from_gpu_name(index: int, uuid_: uuid.UUID, gpu_name: str) -> tuple[int, int]:
    """This is a hack which determines the number of nvdec/nvencs per gpu based on the GPU name.

    Ideally, we'd have a better source for this data, but we couldn't find a good one.
    """
    if "H100" in gpu_name:
        return 7, 0
    elif "A100" in gpu_name:
        return 7, 0
    elif "L40" in gpu_name:
        return 3, 3
    elif "L4" in gpu_name:
        return 4, 2
    elif "RTX 6000" in gpu_name:
        return 3, 3
    elif "RTX A6000" in gpu_name:
        return 3, 3
    elif "NVIDIA" in gpu_name:
        return 0, 0
    else:
        raise ValueError(
            f"Unknown gpu type: {gpu_name}. Likely it needs to be added to "
            "cosmos_xenna.ray_utils.cluster._make_gpu_resources_from_gpu_name"
        )


@attrs.define
class GpuInfo:
    index: int
    name: str
    uuid_: uuid.UUID
    num_nvdecs: int = 0
    num_nvencs: int = 0


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


def _get_nvdecs_and_nvencs_from_gpu_infos(gpu_infos: list[GpuInfo]) -> tuple[int, int]:
    # HACK: when running in CI/CD, we ignore 'NVIDIA DGX Display' gpus
    # This hack is incomplete. We also need to make sure the cuda env vars are set correctly.
    if CICD_ENV_VAR in os.environ:
        logger.info("Running in CI/CD. Ignoring 'NVIDIA DGX Display' gpus")
        gpu_infos = [x for x in gpu_infos if "NVIDIA DGX Display" not in x.name]

    unique_names = set([str(x.name) for x in gpu_infos])
    if len(unique_names) != 1:
        raise ValueError(f"Running on a node with multiple gpu types: {unique_names}. This is not supported as of now.")
    name = next(iter(unique_names))
    logger.debug(f"Gpu with name {name} found. Looking up nvdecs and nvencs...")
    return _make_nvdecs_and_nvencs_from_gpu_name(gpu_infos[0].index, gpu_infos[0].uuid_, name)


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
    nvdecs, nvencs = _get_nvdecs_and_nvencs_from_gpu_infos(gpus)
    return ResourceInfoFromNode(
        node_id=node_id,
        cpus=num_cpus,
        gpus=[GpuInfo(index=x.index, name=x.name, uuid_=x.uuid_, num_nvdecs=nvdecs, num_nvencs=nvencs) for x in gpus],
    )


def make_cluster_resources_for_ray_cluster(
    cpu_allocation_percentage: float = 1.0,
    nodes: Optional[list] = None,
) -> rust_resources.ClusterResources:
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
        out_dict[str(node_id)] = rust_resources.NodeResources(
            cpus=int(info.cpus * cpu_allocation_percentage),
            gpus=[
                rust_resources.GpuResources(
                    index=x.index,
                    uuid_=x.uuid_,
                    gpu_fraction=1.0,
                    nvdecs=list(range(x.num_nvdecs)),
                    nvencs=list(range(x.num_nvencs)),
                )
                for x in info.gpus
            ],
            name=str(node_id),
        )

    out = rust_resources.ClusterResources(out_dict)
    return out
