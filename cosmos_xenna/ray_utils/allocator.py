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

"""Resource allocation manager for distributed pipeline workers.

This module provides resource allocation and tracking capabilities for a distributed
pipeline system. It ensures safe and efficient distribution of compute resources
(CPU, GPU, NVDEC, NVENC) across multiple nodes while maintaining pipeline stage
organization.

The WorkerAllocator tracks both the physical allocation of resources across nodes
and the logical organization of workers into pipeline stages. It prevents resource
oversubscription and provides utilities for monitoring resource utilization.

Typical usage:
    ```python
    # Create allocator with cluster resources
    allocator = WorkerAllocator(cluster_resources)

    # Add workers for different pipeline stages
    allocator.add_worker(Worker("worker1", "stage1", resources))
    allocator.add_worker(Worker("worker2", "stage1", resources))

    # Monitor resource usage
    print(allocator.make_detailed_utilization_table())
    ```
"""

from __future__ import annotations

import collections
import copy
from collections.abc import Iterable
from typing import Optional

import attrs
import cattrs
import tabulate

from cosmos_xenna.ray_utils import resources
from cosmos_xenna.ray_utils.resources import Worker


@attrs.define
class NodeWorkers:
    """Container for workers allocated to a specific node.

    Attributes:
        by_id: Dictionary mapping worker IDs to Worker instances for this node.
    """

    by_id: dict[str, Worker] = attrs.field(factory=dict)


@attrs.define
class StageWorkers:
    """Container for workers assigned to a specific pipeline stage.

    Attributes:
        by_id: Dictionary mapping worker IDs to Worker instances for this stage.
    """

    by_id: dict[str, Worker] = attrs.field(factory=dict)


def _create_bar_chart(used: float, total: float, width: int = 20) -> str:
    """Creates an ASCII bar chart showing resource utilization.

    Args:
        used: Amount of resource currently in use.
        total: Total amount of resource available.
        width: Width of the bar chart in characters.

    Returns:
        String representation of a bar chart showing utilization.
    """
    if total == 0:
        return f"[{'-' * width}] 0.00/0.00"
    filled = int(min(used / total, 1.0) * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {used:.2f}/{total:.2f}"


class OverAllocatedError(Exception):
    """Raised when attempted resource allocation exceeds available resources."""

    pass


class WorkerAllocator:
    """Manages resource allocation for distributed pipeline workers across nodes.

    This class is responsible for:
    1. Tracking available compute resources (CPU, GPU, NVDEC, NVENC) across nodes
    2. Managing worker allocation to both nodes and pipeline stages
    3. Preventing resource oversubscription
    4. Providing utilization monitoring and reporting

    The allocator maintains both physical (node-based) and logical (stage-based)
    views of worker allocation to support pipeline execution while ensuring
    safe resource usage.

    Attributes:
        num_nodes: Number of nodes in the cluster.
        totals: Total available resources across all nodes.
        remaining_resources: Currently unallocated resources across all nodes.
    """

    def __init__(self, cluster_resources: resources.ClusterResources, workers: Optional[list[Worker]] = None) -> None:
        """Initialize the WorkerAllocator.

        Args:
            cluster_resources: Available resources across all nodes.
            workers: Optional list of pre-existing workers to track.
        """
        if workers is None:
            workers = []
        self._cluster_resources = copy.deepcopy(cluster_resources)
        self._nodes_state = {node_id: NodeWorkers({}) for node_id in self._cluster_resources.nodes}
        self._stages_state = collections.defaultdict(StageWorkers)
        self.add_workers(workers)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes_state)

    @property
    def totals(self) -> resources.ClusterResources:
        return copy.deepcopy(self._cluster_resources)

    @property
    def available_resources(self) -> resources.ClusterResources:
        return self._available_resources

    def _check_worker_id_does_not_exist(self, id_: str) -> None:
        maybe_worker = self.get_worker_if_exists(id_)
        if maybe_worker is not None:
            raise ValueError(f"Expected worker with ID={id_} to not exist, but it does exist.")

    def add_worker(self, worker: Worker) -> None:
        """Adds a single worker to the allocation tracking.

        The worker will be tracked both by its assigned node and pipeline stage.
        Validates resource allocation and prevents oversubscription.

        Args:
            worker: Worker instance to add.

        Raises:
            ValueError: If worker ID already exists.
            OverAllocatedError: If adding worker would exceed available resources.
        """
        worker.allocation.validate()
        self._check_worker_id_does_not_exist(worker.id)
        self._nodes_state[worker.allocation.node].by_id[worker.id] = worker
        self._stages_state[worker.stage_name].by_id[worker.id] = worker
        self._calculate_available_resources()

    def add_workers(self, workers: Iterable[Worker]) -> None:
        """Adds multiple workers to allocation tracking.

        Args:
            workers: Iterable of Worker instances to add.

        Raises:
            ValueError: If any worker ID already exists.
            OverAllocatedError: If adding workers would exceed available resources.
        """
        for worker in workers:
            worker.allocation.validate()
            self._check_worker_id_does_not_exist(worker.id)
            self._nodes_state[worker.allocation.node].by_id[worker.id] = worker
            self._stages_state[worker.stage_name].by_id[worker.id] = worker
        self._calculate_available_resources()

    def get_worker(self, worker_id: str) -> Worker:
        """Retrieves a worker by ID.

        Args:
            worker_id: ID of the worker to retrieve.

        Returns:
            The requested Worker instance.

        Raises:
            ValueError: If no worker exists with the given ID.
        """
        maybe_worker = self.get_worker_if_exists(worker_id)
        if maybe_worker is None:
            raise ValueError(f"No worker with id {worker_id} found.")
        else:
            return maybe_worker

    def get_worker_if_exists(self, worker_id: str) -> Optional[Worker]:
        """Return the worker or None, if it does not exist."""
        for node in self._nodes_state.values():
            maybe = node.by_id.get(worker_id, None)
            if maybe is not None:
                return maybe
        return None

    def delete_worker(self, worker_id: str) -> Worker:
        worker = self.get_worker(worker_id)
        self._nodes_state[worker.allocation.node].by_id.pop(worker_id)
        self._stages_state[worker.stage_name].by_id.pop(worker_id)
        self._calculate_available_resources()
        return worker

    def delete_workers(self, worker_ids: list[str]) -> None:
        workers: list[Worker] = []
        assert len(worker_ids) == len(set(worker_ids))
        for worker_id in worker_ids:
            workers.append(self.get_worker(worker_id))

        for worker in workers:
            self._nodes_state[worker.allocation.node].by_id.pop(worker.id)
            self._stages_state[worker.stage_name].by_id.pop(worker.id)
        self._calculate_available_resources()

    def get_workers_in_stage(self, stage_name: str) -> list[Worker]:
        """Retrieves all workers assigned to a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            List of Worker instances assigned to the stage.
        """
        return list(self._stages_state[stage_name].by_id.values())

    def get_workers(self) -> list[Worker]:
        out = []
        for x in self._stages_state.values():
            out.extend(x.by_id.values())
        return out

    def get_num_workers_per_stage(self) -> dict[str, int]:
        return {stage: len(workers.by_id) for stage, workers in self._stages_state.items()}

    def _calculate_available_resources(self) -> None:
        """Updates tracking of remaining available resources across nodes.

        This method recalculates available resources by subtracting all allocated
        resources from the total available resources. It tracks CPU, GPU, NVDEC,
        and NVENC allocations.

        Raises:
            OverAllocatedError: If current allocation exceeds available resources.
        """
        remaining_resources = copy.deepcopy(self._cluster_resources)
        for node_ in self._nodes_state.values():
            for worker in node_.by_id.values():
                node = remaining_resources.nodes[worker.allocation.node]
                node.cpus -= worker.allocation.cpus

                for gpu_alloc in worker.allocation.gpus:
                    gpu = node.gpus[gpu_alloc.gpu_index]
                    gpu.gpu_fraction -= gpu_alloc.fraction

                for x in worker.allocation.nvdecs:
                    if x.codec_index in node.gpus[x.gpu_index].nvdecs:
                        node.gpus[x.gpu_index].nvdecs.remove(x.codec_index)
                    else:
                        raise OverAllocatedError("Tried to allocate already allocated nvdec")

                for x in worker.allocation.nvencs:
                    if x.codec_index in node.gpus[x.gpu_index].nvencs:
                        node.gpus[x.gpu_index].nvencs.remove(x.codec_index)
                    else:
                        raise OverAllocatedError("Tried to allocate already allocated nvenc")

        if remaining_resources.is_overallocated():
            raise OverAllocatedError(f"Cluster is over-allocated. Current allocation:\n{remaining_resources}")
        self._available_resources = remaining_resources

    def worker_ids_and_node_cpu_utilizations(
        self,
        workers_ids_to_consider: Optional[set[str]] = None,
    ) -> list[tuple[float, str]]:
        """Returns worker IDs sorted by their node's CPU utilization.

        Useful for load balancing and resource optimization decisions.

        Args:
            workers_ids_to_consider: Optional set of worker IDs to limit consideration to.

        Returns:
            List of tuples (cpu_utilization, worker_id) sorted by utilization.
        """
        node_utilizations = self._calculate_node_cpu_utilizations()
        out: list[tuple[float, str]] = []

        for node_id, node_workers in self._nodes_state.items():
            for worker_id in node_workers.by_id:
                if workers_ids_to_consider is None or worker_id in workers_ids_to_consider:
                    out.append((node_utilizations[node_id], worker_id))

        return out

    def calculate_lowest_allocated_node_by_cpu(self) -> str:
        utils = self._calculate_node_cpu_utilizations()
        # Get the key with the lowest value
        return min(utils, key=lambda x: utils[x])

    def _calculate_node_cpu_utilizations(self) -> dict[str, float]:
        """
        Calculate the current CPU utilization for each node.

        Returns:
            List[float]: A list of CPU utilization ratios for each node.
        """
        utilizations = {}
        node_ids = set(self._cluster_resources.nodes).union(set(self._available_resources.nodes))
        for node_id in node_ids:
            total = self._cluster_resources.nodes[node_id]
            remaining = self._available_resources.nodes[node_id]
            used_cpus = total.cpus - remaining.cpus
            utilization = used_cpus / total.cpus if total.cpus > 0 else 0.0
            utilizations[node_id] = utilization
        return utilizations

    def make_detailed_utilization_table(self) -> str:
        """Generates a human-readable table showing resource utilization.

        Creates an ASCII table showing CPU, GPU, NVDEC, and NVENC utilization
        for each node in the cluster. Uses bar charts to visualize usage levels.

        Returns:
            Formatted string containing the utilization table.
        """
        table_data = []

        node_ids = set(self._cluster_resources.nodes).union(set(self._nodes_state))
        for node_index, node_id in enumerate(node_ids):
            total = self._cluster_resources.nodes[node_id]
            node_util = self._nodes_state[node_id]
            cpu_usage = sum(worker.allocation.cpus for worker in node_util.by_id.values())
            gpu_usage = [0.0] * len(total.gpus)
            nvdec_usage = [0.0] * len(total.gpus)
            nvenc_usage = [0.0] * len(total.gpus)

            for worker in node_util.by_id.values():
                for gpu in worker.allocation.gpus:
                    gpu_usage[gpu.gpu_index] += gpu.fraction
                for nvdec in worker.allocation.nvdecs:
                    nvdec_usage[nvdec.gpu_index] += 1
                for nvenc in worker.allocation.nvencs:
                    nvenc_usage[nvenc.gpu_index] += 1

            cpu_bar = _create_bar_chart(cpu_usage, total.cpus)
            table_data.append([f"Node {node_index}", f"CPUs: {cpu_bar}", "", ""])

            for i, gpu in enumerate(total.gpus):
                gpu_bar = _create_bar_chart(gpu_usage[i], 1.0)
                nvdec_bar = _create_bar_chart(nvdec_usage[i], gpu.num_nvdecs)
                nvenc_bar = _create_bar_chart(nvenc_usage[i], gpu.num_nvencs)
                table_data.append([f"  GPU {i}", f"GPU: {gpu_bar}", f"NVDEC: {nvdec_bar}", f"NVENC: {nvenc_bar}"])

            if node_index < len(self._cluster_resources.nodes) - 1:
                table_data.append(["", "", "", ""])

        headers = ["Component", "Utilization", "NVDEC", "NVENC"]
        table = tabulate.tabulate(table_data, headers=headers, tablefmt="plain")
        return table

    def __repr__(self) -> str:
        return f"WorkerAllocator(\ncluster_resources={self._cluster_resources!r},\nworkers={self.get_workers()!r},\n)"

    def to_dict(self) -> dict:
        return {
            "cluster_resources": cattrs.unstructure(self._cluster_resources),
            "workers": cattrs.unstructure(self.get_workers()),
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkerAllocator:
        return WorkerAllocator(
            cattrs.structure(data["cluster_resources"], resources.ClusterResources),
            cattrs.structure(data["workers"], list[resources.Worker]),
        )
