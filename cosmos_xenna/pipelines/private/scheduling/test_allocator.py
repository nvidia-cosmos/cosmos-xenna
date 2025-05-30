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

from cosmos_xenna.ray_utils import resources as rds
from cosmos_xenna.ray_utils.allocator import OverAllocatedError, WorkerAllocator
from cosmos_xenna.ray_utils.resources import GPUAllocation, Worker, WorkerResources


def make_simple_cluster() -> rds.ClusterResources:
    return rds.ClusterResources(
        {
            "0": rds.NodeResources(
                cpus=8,
                gpus=[
                    rds.GpuResources(gpu_fraction=1, nvdecs={0, 1}, nvencs={0, 1}),
                    rds.GpuResources(gpu_fraction=1, nvdecs={0, 1}, nvencs={0, 1}),
                ],
            ),
            "1": rds.NodeResources(
                cpus=4,
                gpus=[rds.GpuResources(gpu_fraction=1, nvdecs={0}, nvencs={1})],
            ),
        }
    )


def make_allocator() -> WorkerAllocator:
    simple_cluster = make_simple_cluster()
    return WorkerAllocator(simple_cluster, [])


def test_init():
    allocator = make_allocator()
    assert allocator.num_nodes == 2


def test_add_worker():
    allocator = make_allocator()
    worker = Worker(
        "w1",
        "stage1",
        rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
    )
    allocator.add_worker(worker)
    assert "w1" in allocator._nodes_state["0"].by_id


def test_add_workers():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
        ),
        Worker("w2", "stage2", rds.WorkerResources(node="1", cpus=1)),
    ]
    allocator.add_workers(workers)
    assert "w1" in allocator._nodes_state["0"].by_id
    assert "w2" in allocator._nodes_state["1"].by_id


def test_delete_workers():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
        ),
        Worker("w2", "stage2", rds.WorkerResources(node="1", cpus=1)),
    ]
    allocator.add_workers(workers)
    allocator.delete_workers(["w1"])
    assert "w1" not in allocator._nodes_state["0"].by_id
    assert "w2" in allocator._nodes_state["1"].by_id


def test_delete_non_existent_worker():
    allocator = make_allocator()
    with pytest.raises(ValueError):
        allocator.delete_workers(["non_existent"])


def test_calculate_remaining_resources():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
        ),
        Worker("w2", "stage2", rds.WorkerResources(node="1", cpus=1)),
    ]
    allocator.add_workers(workers)

    remaining = allocator._available_resources
    assert remaining.nodes["0"].cpus == 6  # 8 - 2
    assert remaining.nodes["0"].gpus[0].gpu_fraction == 0.5  # 1 - 0.5
    assert remaining.nodes["1"].cpus == 3  # 4 - 1


def test_make_detailed_utilization_table():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
        ),
        Worker("w2", "stage2", rds.WorkerResources(node="1", cpus=1)),
    ]
    allocator.add_workers(workers)

    table = allocator.make_detailed_utilization_table()
    assert isinstance(table, str)
    assert "Node 0" in table
    assert "Node 1" in table


def test_overallocation():
    allocator = make_allocator()
    worker = Worker("w1", "stage1", rds.WorkerResources(node="0", cpus=10))  # More CPUs than available
    with pytest.raises(OverAllocatedError):
        allocator.add_worker(worker)


def test_overallocation_single_gpu():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(node="0", cpus=1, gpus=[rds.GPUAllocation(0, 0.5)]),
        ),
        Worker(
            "w2",
            "stage1",
            rds.WorkerResources(node="0", cpus=1, gpus=[rds.GPUAllocation(0, 0.7)]),
        ),
    ]  # GPU allocation > 1
    with pytest.raises(OverAllocatedError):
        allocator.add_workers(workers)


def test_overallocation_single_gpu_seperate_calls():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(node="0", cpus=1, gpus=[rds.GPUAllocation(0, 0.5)]),
        ),
        Worker(
            "w2",
            "stage1",
            rds.WorkerResources(node="0", cpus=1, gpus=[rds.GPUAllocation(0, 0.7)]),
        ),
    ]  # GPU allocation > 1
    allocator.add_worker(workers[0])
    with pytest.raises(OverAllocatedError):
        allocator.add_worker(workers[1])


def test_adding_workers_with_existing_ids_raises():
    allocator = make_allocator()
    workers = [
        Worker(
            id="1",
            stage_name="1",
            allocation=WorkerResources(
                node="0", cpus=0.0, gpus=[GPUAllocation(gpu_index=0, fraction=1.0)], nvdecs=[], nvencs=[]
            ),
        ),
        Worker(
            id="2",
            stage_name="1",
            allocation=WorkerResources(
                node="1", cpus=0.0, gpus=[GPUAllocation(gpu_index=0, fraction=0.7)], nvdecs=[], nvencs=[]
            ),
        ),
        Worker(
            id="2",
            stage_name="1",
            allocation=WorkerResources(
                node="1", cpus=0.0, gpus=[GPUAllocation(gpu_index=0, fraction=0.31)], nvdecs=[], nvencs=[]
            ),
        ),
    ]
    with pytest.raises(ValueError):
        allocator.add_workers(workers)


def test_overallocation_with_fractional_resources():
    allocator = make_allocator()
    workers = [
        Worker(
            id="1",
            stage_name="1",
            allocation=WorkerResources(
                node="0", cpus=0.0, gpus=[GPUAllocation(gpu_index=0, fraction=1.0)], nvdecs=[], nvencs=[]
            ),
        ),
        Worker(
            id="2",
            stage_name="1",
            allocation=WorkerResources(
                node="1", cpus=0.0, gpus=[GPUAllocation(gpu_index=0, fraction=0.7)], nvdecs=[], nvencs=[]
            ),
        ),
        Worker(
            id="3",
            stage_name="1",
            allocation=WorkerResources(
                node="1", cpus=0.0, gpus=[GPUAllocation(gpu_index=0, fraction=0.31)], nvdecs=[], nvencs=[]
            ),
        ),
    ]
    with pytest.raises(OverAllocatedError):
        allocator.add_workers(workers)


def test_gpu_allocation_limit():
    allocator = make_allocator()
    worker = Worker(
        "w1",
        "stage1",
        rds.WorkerResources(node="0", cpus=1, gpus=[rds.GPUAllocation(0, 1.5)]),
    )  # GPU allocation > 1
    with pytest.raises(OverAllocatedError):
        allocator.add_worker(worker)


def test_get_worker():
    allocator = make_allocator()
    worker = Worker(
        "w1",
        "stage1",
        rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
    )
    allocator.add_worker(worker)
    retrieved_worker = allocator.get_worker("w1")
    assert retrieved_worker.id == "w1"
    assert retrieved_worker.stage_name == "stage1"


def test_get_nonexistent_worker():
    allocator = make_allocator()
    with pytest.raises(ValueError):
        allocator.get_worker("nonexistent")


def test_delete_worker():
    allocator = make_allocator()
    worker = Worker(
        "w1",
        "stage1",
        rds.WorkerResources(node="0", cpus=2, gpus=[rds.GPUAllocation(0, 0.5)]),
    )
    allocator.add_worker(worker)
    allocator.delete_worker("w1")
    assert "w1" not in allocator._nodes_state["0"].by_id


def test_delete_nonexistent_worker():
    allocator = make_allocator()
    with pytest.raises(ValueError):
        allocator.delete_worker("nonexistent")


def test_worker_ids_and_node_cpu_utilizations():
    allocator = make_allocator()
    workers = [
        Worker("w1", "stage1", rds.WorkerResources(node="0", cpus=4)),
        Worker("w2", "stage1", rds.WorkerResources(node="0", cpus=2)),
        Worker("w3", "stage2", rds.WorkerResources(node="1", cpus=2)),
    ]
    allocator.add_workers(workers)
    utilizations = allocator.worker_ids_and_node_cpu_utilizations()
    assert len(utilizations) == 3
    assert utilizations[0][1] in {"w1", "w2"}  # Higher utilization node first
    assert utilizations[2][1] == "w3"  # Lower utilization node last


def test_worker_ids_and_node_cpu_utilizations_with_subset():
    allocator = make_allocator()
    workers = [
        Worker("w1", "stage1", rds.WorkerResources(node="0", cpus=4)),
        Worker("w2", "stage1", rds.WorkerResources(node="0", cpus=2)),
        Worker("w3", "stage2", rds.WorkerResources(node="1", cpus=2)),
    ]
    allocator.add_workers(workers)
    utilizations = allocator.worker_ids_and_node_cpu_utilizations({"w1", "w3"})
    assert len(utilizations) == 2
    assert utilizations[0][1] == "w1"
    assert utilizations[1][1] == "w3"


def test_overallocation_with_nvdec_nvenc():
    allocator = make_allocator()
    workers = [
        Worker(
            "w1",
            "stage1",
            rds.WorkerResources(
                node="0",
                cpus=1,
                gpus=[rds.GPUAllocation(0, 0.5)],
                nvdecs=[
                    rds.CodecAllocation(0, 0),
                    rds.CodecAllocation(0, 1),
                ],
                nvencs=[
                    rds.CodecAllocation(0, 0),
                    rds.CodecAllocation(0, 1),
                ],
            ),
        ),
        Worker(
            "w2",
            "stage2",
            rds.WorkerResources(
                node="0",
                cpus=1,
                gpus=[rds.GPUAllocation(0, 0.5)],
                nvdecs=[
                    rds.CodecAllocation(0, 0),
                    rds.CodecAllocation(0, 1),
                ],
                nvencs=[
                    rds.CodecAllocation(0, 0),
                    rds.CodecAllocation(0, 1),
                ],
            ),
        ),
    ]
    with pytest.raises(OverAllocatedError):
        allocator.add_workers(workers)


def test_calculate_node_cpu_utilizations():
    allocator = make_allocator()
    workers = [
        Worker("w1", "stage1", rds.WorkerResources(node="0", cpus=4)),
        Worker("w2", "stage2", rds.WorkerResources(node="1", cpus=2)),
    ]
    allocator.add_workers(workers)
    utilizations = allocator._calculate_node_cpu_utilizations()
    assert len(utilizations) == 2
    assert utilizations["0"] == 0.5  # 4 / 8
    assert utilizations["1"] == 0.5  # 2 / 4
