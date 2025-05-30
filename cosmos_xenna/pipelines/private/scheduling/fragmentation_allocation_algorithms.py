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

"""Allocation algorithms which rely on an expected distribution of jobs and the concept of "fragmentation".

This is just one component of our pipeline scheduling algorithm. It's basically just solving the bin packing problem.
Esentially, we have a certain set of resources distrubted across the cluster. We need functions which tell us which node
gpus, nvdecs/nvencs to allocate to a particular worker. This is essentially the multi-dimensional bin-packing problem,
but with some twists. To solve this, we created a new algorithm heavily inspired by the algorithm in this paper:
Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent

We extend the ideas in this paper by considering NVDEC and NVENC allocation, which results in a more
complex algorithm. We also consider the removal of workers, which is a simple extension.
"""

from __future__ import annotations

import copy
import itertools
import pprint

import attrs
from loguru import logger

from cosmos_xenna.ray_utils import allocator, resources
from cosmos_xenna.utils import grouping
from cosmos_xenna.utils.approx import float_eq, float_gte, float_lt, float_lte

_VERBOSE = False


@attrs.define
class Stage:
    """A stage in the workload with associated frequency and resource shape requirements.

    As described in the paper, each stage represents a recurring task type in the workload
    with its resource requirements and relative frequency/popularity.

    Attributes:
        frequency: A float between 0 and 1 representing how often this stage occurs in workload.
            The sum of all stage frequencies in a workload should equal 1.
        shape: A WorkerShape object defining the resource requirements (CPU, GPU, etc.)
            for this stage of the workload.
    """

    frequency: float
    shape: resources.WorkerShape


@attrs.define
class Workload:
    """Represents a complete workload consisting of multiple stages.

    A workload models the expected distribution of tasks in the cluster, used to
    calculate fragmentation metrics. As per the paper, production ML workloads
    consist of recurring tasks that follow certain resource requirement patterns.

    Attributes:
        stages: A list of Stage objects representing the different task types
            and their frequencies in this workload.
    """

    stages: list[Stage]


@attrs.define
class FragmentationResult:
    """Results from calculating fragmentation for a particular allocation scenario.

    Captures the fragmentation state before and after a potential allocation to help
    evaluate scheduling decisions.

    Attributes:
        fragmentation_before: Float indicating fragmentation level before allocation
        fragmentation_after: Float indicating fragmentation level after allocation
        node_remaining_resources: Float representing resources left on node after allocation
        worker_allocation: WorkerResources object describing the actual allocation
        maybe_reused_worker: If this was the result of re-allocating a previous worker, record the worker here.
    """

    fragmentation_before: float
    fragmentation_after: float
    node_remaining_resources: float
    worker_allocation: resources.WorkerResources
    maybe_reused_worker: resources.Worker | None = None

    def fragmentation_change(self) -> float:
        """Calculates the change in fragmentation caused by this allocation.

        Returns:
            Float representing increase (positive) or decrease (negative) in fragmentation.
            Used by scheduler to evaluate allocation decisions.
        """
        return self.fragmentation_after - self.fragmentation_before

    def is_reused_worker(self) -> bool:
        return self.maybe_reused_worker is not None


class GpuResourceHelpers:
    """Helper class for managing GPU-specific resource calculations and checks.

    This class encapsulates logic for determining if GPUs have sufficient resources
    for different types of workloads. It handles the complexity of different GPU
    allocation types (fractional, whole, codecs, etc.) as described in Section 2.1
    of the paper.

    Attributes:
        resources: Current available GPU resources
        totals: Total GPU resources on the node
    """

    def __init__(self, available: resources.GpuResources, totals: resources.GpuResources) -> None:
        self.available = available
        self.totals = totals

    def is_fully_unallocated(self) -> bool:
        """Checks if GPU is completely free with all resources available.

        A GPU is considered fully unallocated if it has:
        - 100% of compute capacity available (gpu_fraction_available == 1.0)
        - All hardware video decoders (nvdecs) available
        - All hardware video encoders (nvencs) available

        Returns:
            bool: True if GPU is completely unused, False otherwise
        """
        return (
            float_eq(self.available.gpu_fraction, 1.0)
            and (self.available.num_nvdecs == self.totals.num_nvdecs)
            and (self.available.num_nvencs == self.totals.num_nvencs)
        )

    def can_be_used_to_allocate(self, shape: resources.WorkerShape, available_cpus: float) -> bool:
        """Determines if this GPU can accommodate the given worker shape requirements.

        It doesn't have to be able to fully allocate the shape, but it does need to be able to contribute to the
        allocation. So, if the shape requires 2 gpus and this is a fully unallocated gpu, this will return True.

        This method implements the allocation feasibility check described in Section 2.1
        of the paper. It handles different GPU allocation types:
        - CPU-only workloads
        - Codec (video encoder/decoder) workloads
        - Fractional GPU workloads
        - Whole-numbered GPU workloads
        - Entire GPU workloads

        Args:
            shape: WorkerShape describing resource requirements
            available_cpus: Number of CPU cores available on the node

        Returns:
            bool: True if the GPU can accommodate this shape, False otherwise
        """
        if float_lt(available_cpus, shape.get_num_cpus()):
            return False

        if shape.type == resources.WorkerShapeType.CPU_ONLY:
            return False
        elif shape.type == resources.WorkerShapeType.CODEC:
            concrete_shape = shape.codec()
            return (self.available.num_nvdecs >= concrete_shape.num_nvdecs) and (
                self.available.num_nvencs >= concrete_shape.num_nvencs
            )
        elif shape.type == resources.WorkerShapeType.FRACTIONAL_GPU:
            concrete_shape = shape.fractional_gpu()
            return (
                float_gte(self.available.gpu_fraction, concrete_shape.num_gpus)
                and (self.available.num_nvdecs >= concrete_shape.num_nvdecs)
                and (self.available.num_nvencs >= concrete_shape.num_nvencs)
            )
        elif shape.type == resources.WorkerShapeType.WHOLE_NUMBERED_GPU:
            concrete_shape = shape.whole_numbered_gpu()
            return (
                float_eq(self.available.gpu_fraction, 1.0)
                and (self.available.num_nvdecs >= concrete_shape.num_nvdecs_per_gpu)
                and (self.available.num_nvencs >= concrete_shape.num_nvencs_per_gpu)
            )
        elif shape.type == resources.WorkerShapeType.ENTIRE_GPU:
            return self.is_fully_unallocated()
        else:
            raise AssertionError()


class NodeResourceHelpers:
    """Helper class for node-level resource management and fragmentation calculation.

    This class implements the core fragmentation analysis described in Section 3 of the paper.
    It helps determine how fragmented a node's resources are from the perspective of different
    workload types.

    Attributes:
        resources: Current available resources on the node
        totals: Total resources on the node
        gpus: List of GpuResourceHelpers for each GPU on the node
    """

    def __init__(self, available: resources.NodeResources, totals: resources.NodeResources) -> None:
        self.available = available
        self.totals = totals
        self.gpus = [GpuResourceHelpers(x, y) for x, y in zip(available.gpus, totals.gpus)]

    def _number_of_gpus_which_can_be_used_for_shape(self, shape: resources.WorkerShape) -> int:
        """Counts number of GPUs on node that can accommodate given shape.

        Args:
            shape: WorkerShape describing resource requirements

        Returns:
            int: Number of GPUs that can be used for this shape
        """
        return len([x for x in self.gpus if x.can_be_used_to_allocate(shape, self.available.cpus)])

    def _find_codec_allocations(self, num_codecs: int, codec_type: str) -> list[list[resources.CodecAllocation]]:
        """Finds all possible ways to allocate video encoders/decoders across GPUs.

        Uses backtracking to find all valid combinations of codec allocations.

        Args:
            num_codecs: Number of encoders/decoders needed
            codec_type: Type of codec ("nvdecs" or "nvencs")

        Returns:
            List of possible codec allocation combinations
        """
        all_allocations = []
        current_allocation = []

        def backtrack(gpu_index: int, remaining: int) -> None:
            """Recursive backtracking helper to find codec allocations.

            Args:
                gpu_index: Current GPU being considered
                remaining: Number of codecs still needing allocation
            """
            if remaining == 0:
                all_allocations.append(current_allocation.copy())
                return

            for i in range(gpu_index, len(self.available.gpus)):
                gpu = self.available.gpus[i]
                available_codecs = getattr(gpu, codec_type)

                for count in range(1, min(len(available_codecs), remaining) + 1):
                    codec_indices = list(itertools.islice(available_codecs, count))
                    current_allocation.extend([resources.CodecAllocation(i, j) for j in codec_indices])
                    backtrack(i + 1, remaining - count)
                    for _ in range(count):
                        current_allocation.pop()

        backtrack(0, num_codecs)
        return all_allocations

    def _num_fully_unallocated_gpus(self) -> int:
        """Counts number of completely unused GPUs on node.

        Returns:
            int: Number of fully unallocated GPUs
        """
        return len([1 for x in self.gpus if x.is_fully_unallocated()])

    def can_allocate(self, shape: resources.WorkerShape) -> bool:
        """Determines if node has sufficient resources for given shape.

        This implements the node-level allocation feasibility check described in
        Section 3.2 of the paper.

        Args:
            shape: WorkerShape describing resource requirements

        Returns:
            bool: True if node can accommodate shape, False otherwise
        """
        if float_lt(self.available.cpus, shape.get_num_cpus()):
            return False

        total_available = self.available.totals()
        if shape.type == resources.WorkerShapeType.CPU_ONLY:
            return True
        elif shape.type == resources.WorkerShapeType.CODEC:
            concrete = shape.codec()
            return (concrete.num_nvdecs <= total_available.nvdecs) and (concrete.num_nvencs <= total_available.nvencs)
        elif shape.type == resources.WorkerShapeType.FRACTIONAL_GPU:
            return 0 < self._number_of_gpus_which_can_be_used_for_shape(shape)
        elif shape.type == resources.WorkerShapeType.WHOLE_NUMBERED_GPU:
            concrete = shape.whole_numbered_gpu()
            return float_lte(concrete.num_gpus, self._number_of_gpus_which_can_be_used_for_shape(shape))
        elif shape.type == resources.WorkerShapeType.ENTIRE_GPU:
            concrete = shape.entire_gpu()
            return float_lte(concrete.num_gpus, self._num_fully_unallocated_gpus())
        else:
            raise AssertionError()

    def can_allocate_resources(self, resources: resources.WorkerResources) -> bool:
        """Determines if these node resources can accommodate the requested allocation.

        This method is similar to the NodeResources.allocate() method but only checks feasibility
        without modifying state. It uses the floating point comparison functions for numerical stability.

        Args:
            resources: WorkerResources object describing the requested allocation

        Returns:
            bool: True if allocation is possible, False otherwise
        """
        # Check if we have enough CPUs
        if float_lt(self.available.cpus, resources.cpus):
            return False

        # Track GPU allocations to validate overall capacity
        gpu_allocations = {}  # gpu_index -> total fraction requested

        # Check GPU resource availability
        for gpu_alloc in resources.gpus:
            # Track total GPU fraction requested per GPU
            if gpu_alloc.gpu_index not in gpu_allocations:
                gpu_allocations[gpu_alloc.gpu_index] = 0.0
            gpu_allocations[gpu_alloc.gpu_index] += gpu_alloc.fraction

            # Check if this would exceed GPU capacity
            if float_lt(self.gpus[gpu_alloc.gpu_index].available.gpu_fraction, gpu_allocations[gpu_alloc.gpu_index]):
                return False

        # Check NVDEC availability
        for nvdec in resources.nvdecs:
            if nvdec.codec_index not in self.gpus[nvdec.gpu_index].available.nvdecs:
                return False

        # Check NVENC availability
        for nvenc in resources.nvencs:
            if nvenc.codec_index not in self.gpus[nvenc.gpu_index].available.nvencs:
                return False

        # If we got here, all resource checks passed
        return True

    def find_possible_allocations(self, shape: resources.WorkerShape, node_id: str) -> list[resources.WorkerResources]:
        """Finds all valid ways to allocate resources for given shape on this node.

        This is a key method implementing the allocation possibilities analysis
        described in Section 3.2 of the paper. It handles different resource
        requirement types and finds all valid allocation combinations.

        Args:
            shape: WorkerShape describing resource requirements
            node_id: ID of this node

        Returns:
            List of possible WorkerResources allocations. Empty if none are possible.
        """
        if not self.can_allocate(shape):
            return []

        out: list[resources.WorkerResources] = []
        if shape.type == resources.WorkerShapeType.CPU_ONLY:
            # CPU-only tasks just need the requested CPU cores
            concrete = shape.cpu_only()
            return [resources.WorkerResources(node_id, concrete.num_cpus)]
        elif shape.type == resources.WorkerShapeType.CODEC:
            # Find all ways to allocate video encoders/decoders across GPUs
            concrete = shape.codec()
            nvdec_allocations = self._find_codec_allocations(concrete.num_nvdecs, "nvdecs")
            nvenc_allocations = self._find_codec_allocations(concrete.num_nvencs, "nvencs")

            for nvdec_alloc in nvdec_allocations:
                for nvenc_alloc in nvenc_allocations:
                    out.append(
                        resources.WorkerResources(node_id, concrete.num_cpus, nvdecs=nvdec_alloc, nvencs=nvenc_alloc)
                    )
        elif shape.type == resources.WorkerShapeType.FRACTIONAL_GPU:
            # For fractional GPU, try allocating on each GPU that has enough capacity
            for gpu_index, gpu in enumerate(self.gpus):
                concrete = shape.fractional_gpu()
                if gpu.can_be_used_to_allocate(shape, self.available.cpus):
                    nvdecs = [
                        resources.CodecAllocation(gpu_index, idx)
                        for idx in list(gpu.available.nvdecs)[: concrete.num_nvdecs]
                    ]
                    nvencs = [
                        resources.CodecAllocation(gpu_index, idx)
                        for idx in list(gpu.available.nvencs)[: concrete.num_nvencs]
                    ]
                    out.append(
                        resources.WorkerResources(
                            node_id,
                            concrete.num_cpus,
                            [resources.GPUAllocation(gpu_index, concrete.num_gpus)],
                            nvdecs=nvdecs,
                            nvencs=nvencs,
                        )
                    )
        elif shape.type == resources.WorkerShapeType.WHOLE_NUMBERED_GPU:
            # For whole numbered GPU allocation, find all valid combinations of GPUs
            # that can satisfy the request for multiple whole GPUs
            concrete = shape.whole_numbered_gpu()
            available_gpus = [
                gpu_index
                for gpu_index, gpu in enumerate(self.gpus)
                if gpu.can_be_used_to_allocate(shape, self.available.cpus)
            ]
            # Try all possible combinations of the required number of GPUs
            for combination in itertools.combinations(available_gpus, concrete.num_gpus):
                gpus = [resources.GPUAllocation(gpu_index, 1.0) for gpu_index in combination]
                # Allocate required encoders/decoders for each GPU
                nvdecs = [
                    resources.CodecAllocation(gpu_index, nvdec_index)
                    for gpu_index in combination
                    for nvdec_index in list(self.gpus[gpu_index].totals.nvdecs)[: concrete.num_nvdecs_per_gpu]
                ]
                nvencs = [
                    resources.CodecAllocation(gpu_index, nvdec_index)
                    for gpu_index in combination
                    for nvdec_index in list(self.gpus[gpu_index].totals.nvencs)[: concrete.num_nvencs_per_gpu]
                ]
                out.append(resources.WorkerResources(node_id, concrete.num_cpus, gpus, nvdecs, nvencs))
        elif shape.type == resources.WorkerShapeType.ENTIRE_GPU:
            # For entire GPU allocation, find combinations of completely unallocated GPUs
            concrete = shape.entire_gpu()
            fully_unallocated_gpus = [i for i, gpu in enumerate(self.gpus) if gpu.is_fully_unallocated()]
            for combination in itertools.combinations(fully_unallocated_gpus, concrete.num_gpus):
                # Allocate entire GPUs with all their encoders/decoders
                gpus = [resources.GPUAllocation(gpu_index, 1.0) for gpu_index in combination]
                nvdecs = [
                    resources.CodecAllocation(gpu_index, nvdec_index)
                    for gpu_index in combination
                    for nvdec_index in self.gpus[gpu_index].totals.nvdecs
                ]
                nvencs = [
                    resources.CodecAllocation(gpu_index, nvdec_index)
                    for gpu_index in combination
                    for nvdec_index in self.gpus[gpu_index].totals.nvencs
                ]
                out.append(resources.WorkerResources(node_id, concrete.num_cpus, gpus, nvdecs, nvencs))
        else:
            raise AssertionError()
        return out

    def _calculate_unallocatable_gpus_fragment_for_shape(
        self,
        shape: resources.WorkerShape,
    ) -> float:
        """Calculates amount of GPU resources that cannot be allocated to a specific shape.

        This implements the task-level fragmentation measure F_n(m) described in Section 3.2
        of the paper. It measures how many GPU resources cannot be allocated to a given
        task shape due to various constraints.

        Args:
            shape: WorkerShape describing resource requirements

        Returns:
            float: Amount of GPU resources that cannot be allocated to this shape.
            A higher value indicates more fragmentation from this shape's perspective.
        """
        total_available_gpus = sum([x.gpu_fraction for x in self.available.gpus], 0.0)

        # Case 1: Task requests no GPU
        # All available GPU resources are considered fragmented since they can't be used
        if shape.get_num_gpus() == 0:
            return total_available_gpus

        # Case 2: Shape cannot be allocated to the node
        # All available GPU resources are considered fragmented
        if not self.can_allocate(shape):
            return total_available_gpus

        # Case 3: Shape requires GPUs and can be allocated on the node
        # Count GPUs that have insufficient capacity for this shape
        out = 0.0
        for gpu in self.gpus:
            if not gpu.can_be_used_to_allocate(shape, self.available.cpus):
                out += gpu.available.gpu_fraction
        return out

    def estimate_fragmentation(
        self,
        workload: Workload,
    ) -> float:
        """Estimates overall fragmentation from perspective of entire workload.

        This implements the node-level fragmentation measure F_n(M) described in
        Section 3.2 of the paper. It calculates the expected fragmentation by
        weighting each shape's fragmentation by its frequency in the workload.

        Args:
            workload: Workload object containing stages with shapes and frequencies

        Returns:
            float: Estimated fragmentation level for this node given the workload
        """
        out = 0.0
        for stage in workload.stages:
            unallocatable_gpus = self._calculate_unallocatable_gpus_fragment_for_shape(stage.shape)
            # if _VERBOSE:
            #     logger.info(f"{stage.frequency=},{unallocatable_gpus=}")
            out += stage.frequency * unallocatable_gpus
        return out


class ClusterResourceHelpers:
    """Helper class for cluster-level resource management and fragmentation calculation.

    This class implements the cluster-level fragmentation analysis described in
    Section 3.2 of the paper. It helps evaluate fragmentation across all nodes
    to guide scheduling decisions.

    Attributes:
        nodes: Dict mapping node IDs to NodeResourceHelpers objects
    """

    @classmethod
    def make_from_allocator(cls, allocator: allocator.WorkerAllocator) -> ClusterResourceHelpers:
        return ClusterResourceHelpers(allocator.available_resources, allocator.totals)

    def __init__(self, available: resources.ClusterResources, totals: resources.ClusterResources):
        """Initialize with current and total cluster resources.

        Args:
            resources: Current available resources in cluster
            totals: Total resources in cluster
        """
        self.nodes = {
            name: NodeResourceHelpers(x, y) for name, x, y in grouping.dict_zip(available.nodes, totals.nodes)
        }

    def copy_and_allocate(self, resources: resources.WorkerResources) -> ClusterResourceHelpers:
        new = copy.deepcopy(self)
        new.nodes[resources.node].available.allocate(resources)
        return new

    def copy_and_release_allocation(self, resources: resources.WorkerResources) -> ClusterResourceHelpers:
        new = copy.deepcopy(self)
        new.nodes[resources.node].available.release_allocation(resources)
        return new

    def estimate_fragmentation(
        self,
        workload: Workload,
    ) -> float:
        """Estimates total fragmentation across entire cluster.

        This implements the cluster-level fragmentation measure F_N(M) described
        in Section 3.2 of the paper.

        Args:
            workload: Workload object describing expected task distribution

        Returns:
            float: Estimated cluster-wide fragmentation level
        """
        return sum([x.estimate_fragmentation(workload) for _, x in self.nodes.items()], 0.0)


@attrs.define
class AllocationResult:
    did_allocate: bool
    resources: resources.WorkerResources | None = None
    reused_worker: resources.Worker | None = None


def find_best_allocation_using_fragmentation_gradient_descent(
    cluster: ClusterResourceHelpers,
    workload: Workload,
    shape: resources.WorkerShape,
    reusable_workers: set[resources.Worker] | None = None,
    worker_reuse_fragmentation_equivalent: float = 10.0,
) -> AllocationResult:
    """Finds the best allocation for a shape that minimizes fragmentation increase.

    This implements the Fragmentation Gradient Descent (FGD) algorithm described
    in Section 4.2 of the paper. It tries all possible allocations and chooses
    the one that causes the minimum increase in fragmentation.

    Args:
        cluster: Cluster resource helper
        workload: Workload object describing expected task distribution
        shape: WorkerShape to be allocated
        reusable_workers: Workers we could potentially re-use. This is helpful to avoid thrashing in our auto-scaling
            algorithm. We assume these are the same shape as "shape", but do not check this.
        worker_reuse_fragementation_equivalent: A reward for re-using workers.

    Returns:
        WorkerResources describing best allocation, or None if no allocation possible
    """
    results: list[FragmentationResult] = []

    # Try reusing recently removed workers
    if reusable_workers is not None and reusable_workers:
        for worker in reusable_workers:
            node = cluster.nodes[worker.allocation.node]
            if not node.can_allocate_resources(worker.allocation):
                continue
            # TODO: This only needs to be done once.
            current_frag = node.estimate_fragmentation(workload)
            # Calculate fragmentation impact of reallocating this worker
            new_remaining_node_resources = node.available.copy_and_allocate(worker.allocation)
            new_node = NodeResourceHelpers(new_remaining_node_resources, node.totals)
            new_frag = new_node.estimate_fragmentation(workload)

            results.append(
                FragmentationResult(
                    current_frag,
                    new_frag,
                    new_remaining_node_resources.totals().total_num(),
                    worker.allocation,
                    worker,
                )
            )

    for node_id, node in cluster.nodes.items():
        # Skip nodes that can't accommodate this shape
        if not node.can_allocate(shape):
            continue

        # Calculate current fragmentation level
        current_frag = node.estimate_fragmentation(workload)
        # Try each possible allocation and calculate resulting fragmentation
        possible_allocations = node.find_possible_allocations(shape, node_id)
        for allocation in possible_allocations:
            new_remaining_node_resources = node.available.copy_and_allocate(allocation)
            new_node = NodeResourceHelpers(new_remaining_node_resources, node.totals)
            new_frag = new_node.estimate_fragmentation(workload)
            results.append(
                FragmentationResult(
                    current_frag, new_frag, new_remaining_node_resources.totals().total_num(), allocation
                )
            )

    if _VERBOSE:
        logger.info(f"Changes:\n{pprint.pformat(results)}")

    if not results:
        return AllocationResult(False, None, None)

    # Choose allocation that minimizes fragmentation increase (with some buffer to prefer re-allocation)
    # If multiple options have same fragmentation change, prefer one that leaves more resources
    def cost(x: FragmentationResult) -> tuple:
        fragementation_change = x.fragmentation_change()
        if x.is_reused_worker:
            fragementation_change -= worker_reuse_fragmentation_equivalent
        return (fragementation_change, -x.node_remaining_resources)

    best_allocation = min(
        results,
        key=cost,
    )
    return AllocationResult(True, best_allocation.worker_allocation, best_allocation.maybe_reused_worker)


@attrs.define
class _FragmentationDeleteResult:
    """Results from calculating fragmentation impact of removing a worker.

    Used by the scheduler to evaluate which worker to terminate when resources
    need to be freed.

    Attributes:
        fragmentation_before: Float indicating fragmentation before worker removal
        fragmentation_after: Float indicating fragmentation after worker removal
        node_remaining_resources: Float indicating resources that would be freed
        worker_id: ID of the worker being considered for removal
    """

    fragmentation_before: float
    fragmentation_after: float
    node_remaining_resources: float
    worker: resources.Worker

    def fragmentation_change(self) -> float:
        """Calculates change in fragmentation from removing this worker.

        Returns:
            float: Change in fragmentation (positive means increased fragmentation)
        """
        return self.fragmentation_after - self.fragmentation_before


def find_worker_to_delete_using_fragmentation_gradient_descent(
    cluster: ClusterResourceHelpers,
    workload: Workload,
    potential_workers: list[resources.Worker],
) -> resources.Worker:
    """Identifies best worker to remove to minimize resulting fragmentation.

    This implements the worker removal strategy using FGD principles. It evaluates
    removing each candidate worker and chooses the one that results in minimum
    fragmentation increase.

    Args:
        cluster: WorkerAllocator managing cluster resources
        workload: Workload object describing expected task distribution
        potential_worker_ids: List of worker IDs that could be removed

    Returns:
        Worker: Worker that should be removed
    """
    assert potential_workers
    current_fragmentation = cluster.estimate_fragmentation(workload)
    changes: list[_FragmentationDeleteResult] = []

    # Evaluate impact of removing each potential worker
    for worker in potential_workers:
        new_cluster = cluster.copy_and_release_allocation(worker.allocation)
        new_frag = new_cluster.estimate_fragmentation(workload)
        changes.append(
            _FragmentationDeleteResult(
                current_fragmentation,
                new_frag,
                cluster.nodes[worker.allocation.node].available.totals().total_num(),
                worker,
            )
        )

    if _VERBOSE:
        logger.info(f"Results\n:{pprint.pformat(changes)}")
    # Choose worker that minimizes fragmentation and preserves most resources
    best_clear = min(changes, key=lambda x: (x.fragmentation_after, -x.node_remaining_resources))
    return best_clear.worker
