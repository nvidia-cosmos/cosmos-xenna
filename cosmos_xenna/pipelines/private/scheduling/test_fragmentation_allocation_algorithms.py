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

import pytest

from cosmos_xenna.pipelines.private.scheduling import fragmentation_allocation_algorithms as frag_alloc
from cosmos_xenna.ray_utils import allocator, resources


def make_cluster(nodes: list[resources.NodeResources]) -> resources.ClusterResources:
    node_dict = {}
    for i, node in enumerate(nodes):
        node_dict[str(i)] = node
    return resources.ClusterResources(node_dict)


def test_calculate_unallocatable_gpus_fragment_for_shape() -> None:
    available = resources.NodeResources(
        cpus=4.0,
        gpus=[
            resources.GpuResources(gpu_fraction=0.5),
            resources.GpuResources(gpu_fraction=1.0),
        ],
    )
    totals = resources.NodeResources(
        cpus=8.0,
        gpus=[
            resources.GpuResources(gpu_fraction=1.0),
            resources.GpuResources(gpu_fraction=1.0),
        ],
    )
    shape = resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7))
    node = frag_alloc.NodeResourceHelpers(available, totals)
    result = node._calculate_unallocatable_gpus_fragment_for_shape(shape)
    assert result == 0.5, "Expected 0.5 GPU to be unallocatable"


def test_estimate_node_fragmentation() -> None:
    available = resources.NodeResources(
        cpus=4.0,
        gpus=[
            resources.GpuResources(gpu_fraction=0.5),
            resources.GpuResources(gpu_fraction=1.0),
        ],
    )
    totals = resources.NodeResources(
        cpus=8.0,
        gpus=[
            resources.GpuResources(gpu_fraction=1.0),
            resources.GpuResources(gpu_fraction=1.0),
        ],
    )
    node = frag_alloc.NodeResourceHelpers(available, totals)
    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.6,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            ),
            frag_alloc.Stage(
                frequency=0.4,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3)),
            ),
        ]
    )

    result = node.estimate_fragmentation(workload)
    assert 0.3 <= result <= 0.5, f"Expected fragmentation between 0.3 and 0.5, got {result}"


def test_calculate_cluster_fragmentation() -> None:
    available = make_cluster(
        nodes=[
            resources.NodeResources(
                cpus=4.0,
                gpus=[
                    resources.GpuResources(gpu_fraction=0.5),
                    resources.GpuResources(gpu_fraction=1.0),
                ],
            ),
            resources.NodeResources(
                cpus=8.0,
                gpus=[
                    resources.GpuResources(gpu_fraction=0.7),
                    resources.GpuResources(gpu_fraction=0.3),
                ],
            ),
        ]
    )
    totals = make_cluster(
        nodes=[
            resources.NodeResources(
                cpus=8.0,
                gpus=[
                    resources.GpuResources(gpu_fraction=1.0),
                    resources.GpuResources(gpu_fraction=1.0),
                ],
            ),
            resources.NodeResources(
                cpus=8.0,
                gpus=[
                    resources.GpuResources(gpu_fraction=1.0),
                    resources.GpuResources(gpu_fraction=1.0),
                ],
            ),
        ]
    )
    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.6,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            ),
            frag_alloc.Stage(
                frequency=0.4,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3)),
            ),
        ]
    )

    cluster_helpers = frag_alloc.ClusterResourceHelpers(available, totals)
    result = cluster_helpers.estimate_fragmentation(workload)
    assert 0.2 <= result <= 1.0, f"Expected cluster fragmentation between 0.6 and 1.0, got {result}"


def test_maybe_allocate_worker_using_fragmentation_gradient_descent() -> None:
    cluster_resources = make_cluster(
        nodes=[
            resources.NodeResources(
                cpus=8.0,
                gpus=[
                    resources.GpuResources(gpu_fraction=1.0),
                    resources.GpuResources(gpu_fraction=1.0),
                ],
            )
        ]
    )
    workers = [
        resources.Worker(
            id="worker1",
            stage_name="stage1",
            allocation=resources.WorkerResources(
                node="0", cpus=4.0, gpus=[resources.GPUAllocation(gpu_index=0, fraction=0.5)]
            ),
        )
    ]
    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)

    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.6,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            ),
            frag_alloc.Stage(
                frequency=0.4,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3)),
            ),
        ]
    )
    shape = resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7))
    cluster_helpers = frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals)
    result = frag_alloc.find_best_allocation_using_fragmentation_gradient_descent(
        cluster_helpers, workload, shape
    ).resources
    assert result is not None, "Expected a valid allocation"
    assert result.node == "0", "Expected allocation on node 0"
    assert any(gpu.fraction == 0.7 for gpu in result.gpus), "Expected 0.7 GPU allocation"


def test_find_worker_to_delete_using_fragmentation_gradient_descent() -> None:
    cluster_resources = make_cluster(
        nodes=[
            resources.NodeResources(
                cpus=8.0,
                gpus=[
                    resources.GpuResources(gpu_fraction=1.0),
                    resources.GpuResources(gpu_fraction=1.0),
                ],
            )
        ]
    )
    workers = [
        resources.Worker(
            id="worker1",
            stage_name="stage1",
            allocation=resources.WorkerResources(
                node="0", cpus=2.0, gpus=[resources.GPUAllocation(gpu_index=0, fraction=0.5)]
            ),
        ),
        resources.Worker(
            id="worker2",
            stage_name="stage1",
            allocation=resources.WorkerResources(
                node="0", cpus=2.0, gpus=[resources.GPUAllocation(gpu_index=1, fraction=0.7)]
            ),
        ),
    ]
    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)

    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.6,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            ),
            frag_alloc.Stage(
                frequency=0.4,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3)),
            ),
        ]
    )
    potential_worker_ids: list[str] = ["worker1", "worker2"]

    result = frag_alloc.find_worker_to_delete_using_fragmentation_gradient_descent(
        frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals),
        workload,
        _make_workers_from_ids(allocator_, potential_worker_ids),
    )
    assert result.id in potential_worker_ids, f"Expected result to be one of {potential_worker_ids}, got {result}"


# Helper function updates
def make_cluster_resources(
    num_nodes: int,
    cpus_per_node: float,
    gpus_per_node: int,
    gpu_memory: float = 1.0,
    num_nvdecs_per_gpu: int = 0,
    num_nvencs_per_gpu: int = 0,
) -> resources.ClusterResources:
    return make_cluster(
        nodes=[
            resources.NodeResources(
                cpus=cpus_per_node,
                gpus=[
                    resources.GpuResources(
                        gpu_fraction=gpu_memory,
                        nvdecs=set(range(num_nvdecs_per_gpu)),
                        nvencs=set(range(num_nvencs_per_gpu)),
                    )
                    for _ in range(gpus_per_node)
                ],
            )
            for _ in range(num_nodes)
        ]
    )


def make_worker(
    id: str, stage_name: str, node: str, cpus: float, gpu_allocations: list[tuple[int, float]]
) -> resources.Worker:
    return resources.Worker(
        id=id,
        stage_name=stage_name,
        allocation=resources.WorkerResources(
            node=node,
            cpus=cpus,
            gpus=[resources.GPUAllocation(gpu_index=idx, fraction=frac) for idx, frac in gpu_allocations],
        ),
    )


@pytest.mark.parametrize(
    ("shape", "expected_allocation"),
    [
        (
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.5)),
            True,
        ),
        (
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=4.0, num_gpus=0.8)),
            True,
        ),
        (
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=6.0, num_gpus=1.0)),
            False,
        ),
        (
            resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=2.0, num_gpus=1)),
            True,
        ),
        (
            resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=4.0, num_gpus=2)),
            False,
        ),
        (
            resources.WorkerShape.make(resources.EntireGpu(num_cpus=2.0, num_gpus=1)),
            True,
        ),
        (resources.WorkerShape.make(resources.CpuOnly(num_cpus=2.0)), True),
        (resources.WorkerShape.make(resources.CpuOnly(num_cpus=8.0)), False),
        (
            resources.WorkerShape.make(resources.Codec(num_cpus=1.0, num_nvdecs=1, num_nvencs=0)),
            True,
        ),
        (
            resources.WorkerShape.make(resources.Codec(num_cpus=1.0, num_nvdecs=10, num_nvencs=2)),
            False,
        ),
    ],
)
def test_maybe_allocate_worker_various_shapes(shape: resources.WorkerShape, expected_allocation: bool) -> None:
    cluster_resources = make_cluster_resources(
        num_nodes=2, cpus_per_node=8.0, gpus_per_node=2, num_nvencs_per_gpu=1, num_nvdecs_per_gpu=1
    )
    workers = [
        make_worker("worker1", "stage1", "0", 4.0, [(0, 0.5)]),
        make_worker("worker2", "stage1", "1", 2.0, [(0, 0.3), (1, 0.2)]),
    ]
    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)

    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.6,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            ),
            frag_alloc.Stage(
                frequency=0.4,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3)),
            ),
        ]
    )
    cluster_helpers = frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals)
    result = frag_alloc.find_best_allocation_using_fragmentation_gradient_descent(
        cluster_helpers, workload, shape
    ).resources

    if expected_allocation:
        assert result is not None, f"Expected a valid allocation for shape {shape}"
        assert result.node in ["0", "1"], f"Expected allocation on node 0 or 1, got {result.node}"
        assert result.cpus == shape.get_num_cpus(), f"Expected {shape.get_num_cpus()} CPUs, got {result.cpus}"
        assert sum(gpu.fraction for gpu in result.gpus) == shape.get_num_gpus(), (
            f"Expected {shape.get_num_gpus()} GPUs, got {sum(gpu.fraction for gpu in result.gpus)}"
        )
    else:
        assert result is None, f"Expected no allocation for shape {shape}, but got {result}"


@pytest.mark.parametrize(
    ("cluster_config", "workers", "shape", "expected_allocation"),
    [
        (
            {"num_nodes": 1, "cpus_per_node": 8.0, "gpus_per_node": 2},
            [make_worker("worker1", "stage1", "0", 4.0, [(0, 0.5)])],
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            True,
        ),
        (
            {"num_nodes": 2, "cpus_per_node": 4.0, "gpus_per_node": 1},
            [
                make_worker("worker1", "stage1", "0", 2.0, [(0, 0.5)]),
                make_worker("worker2", "stage1", "1", 2.0, [(0, 0.5)]),
            ],
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=3.0, num_gpus=0.6)),
            False,
        ),
        (
            {"num_nodes": 1, "cpus_per_node": 8.0, "gpus_per_node": 2},
            [make_worker("worker1", "stage1", "0", 6.0, [(0, 0.8), (1, 0.8)])],
            resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=3.0, num_gpus=1)),
            False,
        ),
    ],
)
def test_maybe_allocate_worker_various_cluster_states(
    cluster_config: dict,
    workers: list[resources.Worker],
    shape: resources.WorkerShape,
    expected_allocation: bool,
) -> None:
    cluster_resources = make_cluster_resources(**cluster_config)
    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)

    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.5,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.5)),
            ),
            frag_alloc.Stage(
                frequency=0.5,
                shape=resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=4.0, num_gpus=1)),
            ),
        ]
    )

    cluster_helpers = frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals)
    result = frag_alloc.find_best_allocation_using_fragmentation_gradient_descent(
        cluster_helpers, workload, shape
    ).resources

    if expected_allocation:
        assert result is not None, f"Expected a valid allocation for shape {shape} in cluster state {cluster_config}"
        assert 0 <= int(result.node) < cluster_config["num_nodes"], f"Invalid node allocation: {result.node}"
        assert result.cpus == shape.get_num_cpus(), f"Expected {shape.get_num_cpus()} CPUs, got {result.cpus}"
        assert sum(gpu.fraction for gpu in result.gpus) == shape.get_num_gpus(), (
            f"Expected {shape.get_num_gpus()} GPUs, got {sum(gpu.fraction for gpu in result.gpus)}"
        )
    else:
        assert result is None, (
            f"Expected no allocation for shape {shape} in cluster state {cluster_config}, but got {result}"
        )


def _make_workers_from_ids(allocator_: allocator.WorkerAllocator, worker_ids: list[str]) -> list[resources.Worker]:
    return [allocator_.get_worker(x) for x in worker_ids]


@pytest.mark.parametrize(
    ("cluster_config", "workers", "potential_delete_ids", "expected_deletion"),
    [
        (
            {"num_nodes": 2, "cpus_per_node": 8.0, "gpus_per_node": 2},
            [
                make_worker("worker1", "stage1", "0", 4.0, [(0, 0.5)]),
                make_worker("worker2", "stage1", "0", 2.0, [(1, 0.3)]),
                make_worker("worker3", "stage2", "1", 6.0, [(0, 0.8), (1, 0.7)]),
            ],
            ["worker1", "worker2"],
            "worker1",
        ),
        (
            {"num_nodes": 1, "cpus_per_node": 16.0, "gpus_per_node": 4},
            [
                make_worker("worker1", "stage1", "0", 4.0, [(0, 1.0)]),
                make_worker("worker2", "stage1", "0", 4.0, [(1, 1.0)]),
                make_worker("worker3", "stage2", "0", 4.0, [(2, 0.5), (3, 0.5)]),
            ],
            ["worker1", "worker2", "worker3"],
            "worker3",
        ),
        (
            {"num_nodes": 3, "cpus_per_node": 8.0, "gpus_per_node": 2},
            [make_worker(f"worker{i}", "stage1", str(i % 3), 2.0, [(i % 2, 0.3)]) for i in range(6)],
            [f"worker{i}" for i in range(6)],
            "worker0",  # Assuming the first worker is chosen when all are equivalent
        ),
    ],
)
def test_find_worker_to_delete_various_cluster_states(
    cluster_config: dict, workers: list[resources.Worker], potential_delete_ids: list[str], expected_deletion: str
) -> None:
    cluster_resources = make_cluster_resources(**cluster_config)
    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)

    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.6,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7)),
            ),
            frag_alloc.Stage(
                frequency=0.4,
                shape=resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3)),
            ),
        ]
    )

    result = frag_alloc.find_worker_to_delete_using_fragmentation_gradient_descent(
        frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals),
        workload,
        _make_workers_from_ids(allocator_, potential_delete_ids),
    )
    assert result.id == expected_deletion, f"Expected {expected_deletion} to be deleted, but got {result}"


def test_prefer_lower_allocated_nodes():
    # Create a cluster with two nodes, each having 8 GPUs and 240 CPUs
    cluster_resources = make_cluster(
        nodes=[
            resources.NodeResources(
                cpus=240.0,
                gpus=[resources.GpuResources(gpu_fraction=1.0) for _ in range(8)],
            ),
            resources.NodeResources(
                cpus=240.0,
                gpus=[resources.GpuResources(gpu_fraction=1.0) for _ in range(8)],
            ),
        ]
    )

    # Create workers to simulate the state you described
    workers = [
        # Workers on Node 0 (highly allocated)
        resources.Worker("worker1", "stage1", resources.WorkerResources("0", 100)),
        # Workers on Node 1 (lightly allocated)
        resources.Worker("worker2", "stage2", resources.WorkerResources("1", 10)),
    ]

    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)

    workload = frag_alloc.Workload(
        stages=[
            frag_alloc.Stage(
                frequency=0.5,
                shape=resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=1.0, num_gpus=1)),
            ),
            frag_alloc.Stage(
                frequency=0.5,
                shape=resources.WorkerShape.make(resources.CpuOnly(num_cpus=1.0)),
            ),
        ]
    )

    # The shape we're trying to allocate
    shape = resources.WorkerShape.make(resources.CpuOnly(num_cpus=1.0))

    # Attempt to allocate a new worker
    cluster_helpers = frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals)
    result = frag_alloc.find_best_allocation_using_fragmentation_gradient_descent(
        cluster_helpers, workload, shape
    ).resources

    assert result is not None, "Expected a valid allocation"
    assert result.node == "1", f"Expected allocation on node 1 (less allocated), but got node {result.node}"
    assert result.cpus == 1.0, f"Expected 1.0 CPUs, got {result.cpus}"


def test_node_resources_totals() -> None:
    # Create cluster with 1 node having 2 GPUs
    cluster_resources = make_cluster_resources(
        num_nodes=1, cpus_per_node=16.0, gpus_per_node=2, num_nvdecs_per_gpu=2, num_nvencs_per_gpu=2
    )

    cluster_helpers = frag_alloc.ClusterResourceHelpers(cluster_resources, cluster_resources)
    node = cluster_helpers.nodes["0"]

    assert len(node.gpus) == 2
    assert node.available.cpus == 16.0
    assert sum(gpu.available.gpu_fraction for gpu in node.gpus) == 2.0
    assert sum(len(gpu.available.nvdecs) for gpu in node.gpus) == 4
    assert sum(len(gpu.available.nvencs) for gpu in node.gpus) == 4


@pytest.mark.parametrize(
    ("gpu_resource", "shape", "available_cpus", "expected"),
    [
        (
            resources.GpuResources(gpu_fraction=1.0, nvdecs={0, 1}, nvencs={0, 1}),
            resources.WorkerShape.make(resources.EntireGpu(num_cpus=4, num_gpus=1)),
            16.0,
            True,
        ),
        (
            resources.GpuResources(gpu_fraction=0.5, nvdecs={0, 1}, nvencs={0, 1}),
            resources.WorkerShape.make(resources.EntireGpu(num_cpus=4, num_gpus=1)),
            16.0,
            False,
        ),
        (
            resources.GpuResources(gpu_fraction=1.0, nvdecs={0, 1}, nvencs={0, 1}),
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=4, num_gpus=0.5, num_nvdecs=1, num_nvencs=1)),
            16.0,
            True,
        ),
        (
            resources.GpuResources(gpu_fraction=0.4, nvdecs={0, 1}, nvencs={0, 1}),
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=4, num_gpus=0.5, num_nvdecs=1, num_nvencs=1)),
            16.0,
            False,
        ),
        (
            resources.GpuResources(gpu_fraction=1.0, nvdecs={0}, nvencs={0}),
            resources.WorkerShape.make(resources.Codec(num_cpus=4, num_nvdecs=2, num_nvencs=2)),
            16.0,
            False,
        ),
    ],
)
def test_gpu_resource_helpers_can_be_used_to_allocate(
    gpu_resource: resources.GpuResources,
    shape: resources.WorkerShape,
    available_cpus: float,
    expected: bool,
) -> None:
    helper = frag_alloc.GpuResourceHelpers(gpu_resource, gpu_resource)
    result = helper.can_be_used_to_allocate(shape, available_cpus)
    assert result == expected


@pytest.mark.parametrize(
    ("cluster_config", "shape", "expected_allocations"),
    [
        (
            {
                "num_nodes": 1,
                "cpus_per_node": 16.0,
                "gpus_per_node": 2,
                "num_nvdecs_per_gpu": 2,
                "num_nvencs_per_gpu": 2,
            },
            resources.WorkerShape.make(resources.CpuOnly(num_cpus=4)),
            1,  # Only one way to allocate CPU-only
        ),
        (
            {
                "num_nodes": 1,
                "cpus_per_node": 16.0,
                "gpus_per_node": 2,
                "num_nvdecs_per_gpu": 2,
                "num_nvencs_per_gpu": 2,
            },
            resources.WorkerShape.make(resources.Codec(num_cpus=4, num_nvdecs=2, num_nvencs=2)),
            9,  # Multiple ways to allocate codecs across GPUs
        ),
        (
            {
                "num_nodes": 1,
                "cpus_per_node": 16.0,
                "gpus_per_node": 2,
                "num_nvdecs_per_gpu": 2,
                "num_nvencs_per_gpu": 2,
            },
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=4, num_gpus=0.5, num_nvdecs=1, num_nvencs=1)),
            2,  # One allocation per GPU
        ),
    ],
)
def test_node_resource_helpers_find_possible_allocations(
    cluster_config: dict,
    shape: resources.WorkerShape,
    expected_allocations: int,
) -> None:
    cluster_resources = make_cluster_resources(**cluster_config)
    node = frag_alloc.NodeResourceHelpers(cluster_resources.nodes["0"], cluster_resources.nodes["0"])
    allocations = node.find_possible_allocations(shape, "0")
    assert len(allocations) == expected_allocations

    for allocation in allocations:
        assert isinstance(allocation, resources.WorkerResources)
        assert allocation.node == "0"
        assert allocation.cpus == shape.get_num_cpus()

        if shape.type == resources.WorkerShapeType.CODEC:
            codec = shape.codec()
            assert len(allocation.nvdecs) == codec.num_nvdecs
            assert len(allocation.nvencs) == codec.num_nvencs

        elif shape.type == resources.WorkerShapeType.FRACTIONAL_GPU:
            frac = shape.fractional_gpu()
            assert len(allocation.gpus) == 1
            assert allocation.gpus[0].fraction == frac.num_gpus
            assert len(allocation.nvdecs) == frac.num_nvdecs
            assert len(allocation.nvencs) == frac.num_nvencs


@pytest.mark.parametrize(
    ("cluster_config", "workers", "workload", "shape", "expected_allocations"),
    [
        (
            {
                "num_nodes": 1,
                "cpus_per_node": 16.0,
                "gpus_per_node": 3,
                "num_nvdecs_per_gpu": 2,
                "num_nvencs_per_gpu": 2,
            },
            [
                make_worker("worker1", "stage1", "0", 4.0, [(0, 1.0)]),
                make_worker("worker2", "stage1", "0", 4.0, [(1, 0.5)]),
            ],
            frag_alloc.Workload(
                [
                    frag_alloc.Stage(
                        0.6, resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.7))
                    ),
                    frag_alloc.Stage(
                        0.4, resources.WorkerShape.make(resources.FractionalGpu(num_cpus=1.0, num_gpus=0.3))
                    ),
                ]
            ),
            resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.5, num_nvdecs=1, num_nvencs=1)),
            True,  # Should be able to allocate on remaining GPU
        ),
        (
            {
                "num_nodes": 1,
                "cpus_per_node": 8.0,
                "gpus_per_node": 2,
                "num_nvdecs_per_gpu": 1,
                "num_nvencs_per_gpu": 1,
            },
            [
                make_worker("worker1", "stage1", "0", 6.0, [(0, 0.8), (1, 0.8)]),
            ],
            frag_alloc.Workload(
                [
                    frag_alloc.Stage(
                        0.5, resources.WorkerShape.make(resources.FractionalGpu(num_cpus=2.0, num_gpus=0.5))
                    ),
                    frag_alloc.Stage(
                        0.5, resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=4.0, num_gpus=1))
                    ),
                ]
            ),
            resources.WorkerShape.make(resources.WholeNumberedGpu(num_cpus=3.0, num_gpus=1)),
            False,  # Not enough resources available
        ),
    ],
)
def test_mixed_resource_scenarios(
    cluster_config: dict,
    workers: list[resources.Worker],
    workload: frag_alloc.Workload,
    shape: resources.WorkerShape,
    expected_allocations: bool,
) -> None:
    cluster_resources = make_cluster_resources(**cluster_config)
    allocator_ = allocator.WorkerAllocator(cluster_resources, workers)
    cluster_helpers = frag_alloc.ClusterResourceHelpers(allocator_.available_resources, allocator_.totals)

    result = frag_alloc.find_best_allocation_using_fragmentation_gradient_descent(
        cluster_helpers, workload, shape
    ).resources
    assert (result is not None) == expected_allocations
