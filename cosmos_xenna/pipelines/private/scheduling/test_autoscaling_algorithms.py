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

from cosmos_xenna.pipelines.private.scheduling import autoscaling_algorithms, data_structures, naiive_worker_allocation
from cosmos_xenna.ray_utils import allocator, resources
from cosmos_xenna.ray_utils.resources import (
    ClusterResources,
    Codec,
    CpuOnly,
    EntireGpu,
    FractionalGpu,
    NodeResources,
    WholeNumberedGpu,
    Worker,
    WorkerResources,
    WorkerShape,
)


def test_speed_estimator_basic():
    num_stages = 3
    window_duration = 10.0
    estimator = autoscaling_algorithms.SpeedAndNumberOfReturnsEstimator(num_stages, window_duration)

    current_time = 100.0
    measurements = data_structures.Measurements(
        time=current_time,
        stages=[
            data_structures.StageMeasurements(
                [data_structures.TaskMeasurement(start_time=99.0, end_time=100.0, num_returns=1)]
            ),
            data_structures.StageMeasurements(
                [data_structures.TaskMeasurement(start_time=98.0, end_time=100.0, num_returns=1)]
            ),
            data_structures.StageMeasurements([]),  # Empty measurements
        ],
    )

    # Update and check speeds
    estimator.update_with_measurements(measurements)
    estimates = estimator.get_estimates(current_time)
    speeds = [x.batches_per_second_per_worker for x in estimates.stages]

    # First stage: 1 task per second (1.0 duration)
    assert abs(speeds[0] - 1.0) < 0.001

    # Second stage: 0.5 tasks per second (2.0 duration)
    assert abs(speeds[1] - 0.5) < 0.001

    # Third stage: No measurements
    assert speeds[2] is None


def test_speed_estimator_window():
    """Test windowing behavior of SpeedEstimator."""
    estimator = autoscaling_algorithms.SpeedAndNumberOfReturnsEstimator(1, window_duration=10.0)

    # Add measurements at t=0
    measurements = data_structures.Measurements(
        time=0.0,
        stages=[
            data_structures.StageMeasurements(
                [data_structures.TaskMeasurement(start_time=0.0, end_time=1.0, num_returns=1)]
            )
        ],
    )
    estimator.update_with_measurements(measurements)

    # Check speed at t=5 (within window)
    estimates = estimator.get_estimates(5.0)
    assert estimates.stages[0].batches_per_second_per_worker is not None
    assert abs(estimates.stages[0].batches_per_second_per_worker - 1.0) < 0.001

    # Check speed at t=15 (outside window). This will be kept around because min_num_events>1
    estimates = estimator.get_estimates(15.0)
    assert estimates.stages[0].batches_per_second_per_worker is not None
    assert abs(estimates.stages[0].batches_per_second_per_worker - 1.0) < 0.001

    # Last valid speed should still be available
    last_estimates = estimator.get_last_valid_estimates(15.0)
    assert last_estimates.stages[0].batches_per_second_per_worker is not None
    assert abs(last_estimates.stages[0].batches_per_second_per_worker - 1.0) < 0.001


def test_calculate_slots_no_speeds():
    """Test slot calculation when no speed measurements are available."""
    num_workers = [2, 3, 1]
    current_slots = [4, 5, 6]
    last_valid_speeds = [None, 1.0, None]

    # Should return current slots when any speeds are missing
    slots = autoscaling_algorithms._calculate_num_slots_per_worker_for_all_stages(
        num_workers, current_slots, last_valid_speeds
    )
    assert slots == current_slots


def test_calculate_slots_with_speeds():
    """Test slot calculation with valid speed measurements."""
    num_workers = [1, 1, 1]  # Equal workers per stage
    current_slots = [1, 1, 1]
    last_valid_speeds = [1.0, 2.0, 0.5]  # Different processing speeds
    min_rate = 0.1  # 0.1 Hz scheduling rate

    slots = autoscaling_algorithms._calculate_num_slots_per_worker_for_all_stages(
        num_workers,
        current_slots,
        last_valid_speeds,
        min_scheduling_algorithm_rate_hz=min_rate,
        min_slots=2,
    )

    assert slots == [10, 10, 10]

    # Test with faster scheduling rate
    slots_fast = autoscaling_algorithms._calculate_num_slots_per_worker_for_all_stages(
        num_workers,
        current_slots,
        last_valid_speeds,
        min_scheduling_algorithm_rate_hz=10.0,
        min_slots=2,
    )

    assert slots_fast == [2, 2, 2]


def test_calculate_slots_respects_minimum():
    """Test that slot calculation doesn't reduce existing slot counts."""
    num_workers = [1, 1, 1]
    current_slots = [10, 20, 30]  # High existing slot counts
    last_valid_speeds = [1.0, 1.0, 1.0]  # Equal speeds

    slots = autoscaling_algorithms._calculate_num_slots_per_worker_for_all_stages(
        num_workers, current_slots, last_valid_speeds
    )

    # Should not reduce existing slot counts
    assert slots == current_slots


def create_cpu_only_test_problem(num_stages: int = 3) -> data_structures.Problem:
    """Helper to create a test problem with CPU-only stages."""
    cluster = ClusterResources.make_uniform(NodeResources.make_uniform(num_cpus=9), {"node1", "node2"})

    stages = []
    for i in range(num_stages):
        stages.append(
            data_structures.ProblemStage(
                name=f"stage_{i}", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
            )
        )

    return data_structures.Problem(cluster, stages)


def create_test_state(problem: data_structures.Problem) -> data_structures.ProblemState:
    """Helper to create an empty initial state for a problem."""
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(stage_name=stage.name, workers=[], slots_per_worker=1, is_finished=False)
            for stage in problem.stages
        ]
    )


def test_naiive_allocation_cpu_only():
    """Test naive allocation with CPU-only stages."""
    problem = create_cpu_only_test_problem(num_stages=3)
    state = create_test_state(problem)
    estimates = autoscaling_algorithms.Estimates(
        [
            autoscaling_algorithms.Estimate(batches_per_second_per_worker=1.0, num_returns_per_batch=1.0)
            for _ in problem.stages
        ]
    )

    result = autoscaling_algorithms._run_naiive_allocation(problem, state, estimates)

    num_workers = [x.num_workers for x in result.allocation_result.stages]
    assert num_workers == [6, 6, 6]


def test_naiive_allocation_with_varying_speeds():
    """Test naive allocation with varying processing speeds."""
    problem = create_cpu_only_test_problem(num_stages=3)
    # Second stage twice as fast, third stage half as fast
    state = create_test_state(problem)
    estimates = autoscaling_algorithms.Estimates(
        [
            autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
            for speed in [1.0, 2.0, 0.5]
        ]
    )

    result = autoscaling_algorithms._run_naiive_allocation(problem, state, estimates)

    num_workers = [x.num_workers for x in result.allocation_result.stages]
    assert num_workers == [5, 3, 10]


def create_cluster(
    num_nodes: int,
    cpus_per_node: int,
    gpus_per_node: int = 0,
    nvdecs_per_gpu: int = 0,
    nvencs_per_gpu: int = 0,
    heterogeneous: bool = False,
) -> resources.ClusterResources:
    """Helper to create a test cluster with GPU resources."""
    nodes = {}
    for i in range(num_nodes):
        if heterogeneous and i % 2 == 0:
            cpus = cpus_per_node // 2
            gpus = gpus_per_node // 2
        else:
            cpus = cpus_per_node
            gpus = gpus_per_node
        nodes[f"node{i}"] = NodeResources.make_uniform(
            num_cpus=cpus,
            num_gpus=gpus,
            num_nvdecs_per_gpu=nvdecs_per_gpu,
            num_nvencs_per_gpu=nvencs_per_gpu,
        )
    return resources.ClusterResources(nodes)


def make_default_state_for_stages(problem: data_structures.Problem) -> data_structures.ProblemState:
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(stage_name=x.name, workers=[], slots_per_worker=2, is_finished=False)
            for x in problem.stages
        ]
    )


def apply_solution_to_state(
    state: data_structures.ProblemState,
    solution: data_structures.Solution,
    allocator: allocator.WorkerAllocator,
) -> None:
    """Apply autoscaling solution to problem state.

    Args:
        state: Current state to update
        solution: Solution to apply
        allocator: Worker allocator to keep in sync
    """
    # Process each stage's changes
    for stage_state, stage_solution in zip(state.stages, solution.stages):
        # First handle deletions
        deleted_worker_ids = {w.id for w in stage_solution.deleted_workers}
        stage_state.workers = [w for w in stage_state.workers if w.id not in deleted_worker_ids]

        # Update allocator
        for worker in stage_solution.deleted_workers:
            allocator.delete_worker(worker.id)

        # Then handle additions
        stage_state.workers.extend(stage_solution.new_workers)

        # Update allocator
        for worker in stage_solution.new_workers:
            allocator.add_worker(Worker(worker.id, stage_state.stage_name, worker.resources))

        # Update slots
        stage_state.slots_per_worker = stage_solution.slots_per_worker


class TestFragmentationAutoscaler:
    """Test suite for run_fragmentation_autoscaler function."""

    def test_simple_autoscaling(self):
        """Test basic autoscaling with a single CPU-only stage."""
        # Create a simple cluster with just CPU resources
        cluster = ClusterResources.make_uniform(NodeResources.make_uniform(num_cpus=4), {"node1"})

        # Create a simple problem with one CPU-only stage
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                )
            ],
        )
        state = make_default_state_for_stages(problem)
        estimates = autoscaling_algorithms.Estimates(
            [autoscaling_algorithms.Estimate(batches_per_second_per_worker=1.0, num_returns_per_batch=1.0)]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=1.5, estimates=estimates
        )

        assert solution.num_new_workers_per_stage == [4]
        assert solution.num_deleted_workers_per_stage == [0]

    def test_manual_worker_count_respected(self):
        """Test that manually specified worker counts are exactly respected."""
        # Create a cluster with plenty of resources
        cluster = create_cluster(num_nodes=2, cpus_per_node=24, gpus_per_node=4)

        # Create a problem with one stage requesting exact worker count
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    requested_num_workers=10,  # Explicitly request 3 workers
                    stage_batch_size=1,
                ),
                data_structures.ProblemStage(
                    name="stage_1",
                    worker_shape=WorkerShape.make(WholeNumberedGpu(1)),
                    stage_batch_size=1,
                ),
                data_structures.ProblemStage(
                    name="stage_2",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    requested_num_workers=5,  # Explicitly request 3 workers
                    stage_batch_size=1,
                ),
            ],
        )

        # Create initial state with no workers
        state = make_default_state_for_stages(problem)

        # Run autoscaler
        speeds = [1.0, 0.5, 10.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=1.5, estimates=estimates
        )

        assert solution.num_new_workers_per_stage == [10, 8, 5]
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]

    def test_minimum_one_worker_per_stage(self):
        """Test that each stage gets at least one worker."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=8)

        # Create problem with multiple stages, no manual worker counts
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name=f"stage_{i}", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                )
                for i in range(3)
            ],
        )

        state = make_default_state_for_stages(problem)

        # Some stages have unknown speeds
        speeds = [1.0, None, 0.5]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=1.5, estimates=estimates
        )

        assert solution.num_new_workers_per_stage == [2, 2, 4]
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]

    def test_throughput_optimization(self):
        """Test that the autoscaler optimizes throughput by balancing workers."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=16)
        factory = autoscaling_algorithms.WorkerIdFactory()

        # Create problem with three stages having different speeds
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name=f"stage_{i}", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                )
                for i in range(3)
            ],
        )

        # Initial state with one worker per stage
        initial_workers = []
        for _ in range(3):
            initial_workers.append(
                data_structures.ProblemWorkerState(
                    id=factory.make_new_id(), resources=WorkerResources(node="node0", cpus=1.0)
                )
            )

        state = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name=f"stage_{i}", workers=[initial_workers[i]], slots_per_worker=1, is_finished=False
                )
                for i in range(3)
            ]
        )

        # Stage speeds: first stage is slowest, second is fastest
        speeds = [0.5, 2.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=1.5, estimates=estimates, factory=factory
        )

        assert solution.num_new_workers_per_stage == [8, 2, 3]  # This means the totals are [9, 3, 4] == 16 total
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]

    def test_gpu_fragmentation_awareness(self):
        """Test that the autoscaler handles GPU fragmentation properly."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=8, gpus_per_node=2, nvdecs_per_gpu=2, nvencs_per_gpu=2)

        # Create a problem with mixed GPU resource requirements
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                # Stage 0: Needs fractional GPU
                data_structures.ProblemStage(
                    name="stage_0",
                    worker_shape=WorkerShape.make(FractionalGpu(0.5, num_cpus=1.0)),
                    stage_batch_size=1,
                ),
                # Stage 1: Needs entire GPU
                data_structures.ProblemStage(
                    name="stage_1", worker_shape=WorkerShape.make(EntireGpu(1, num_cpus=1.0)), stage_batch_size=1
                ),
                # Stage 2: Needs GPU decoders
                data_structures.ProblemStage(
                    name="stage_2", worker_shape=WorkerShape.make(Codec(num_cpus=1.0, num_nvdecs=1)), stage_batch_size=1
                ),
            ],
        )

        # Start with empty state
        state = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name=stage.name, workers=[], slots_per_worker=1, is_finished=False
                )
                for stage in problem.stages
            ]
        )

        speeds = [1.0, 1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=1.5, estimates=estimates
        )
        assert solution.num_new_workers_per_stage == [2, 1, 1]
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]

    def test_overallocation_target(self):
        """Test that the autoscaler respects the overallocation target."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=1000, gpus_per_node=8)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_1", worker_shape=WorkerShape.make(EntireGpu(1)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_2", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [0.1, 1.0, 0.05]
        overallocation_target = 1.5  # 50% overallocation
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=overallocation_target, estimates=estimates
        )

        # Stage 2 is limited by the number of gpus, so will have throughput of 8
        # Stage 1 needs 1.5 * 8 / 0.1 == 120 workers
        # Stage 3 needs 1.5 * 8 / 0.05 == 240 worers
        assert solution.num_new_workers_per_stage == [120, 8, 240]
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]

    def test_overallocate_by_manually_allocated(self):
        """Test behavior when cluster resources are exhausted."""
        # Create a cluster with limited resources
        cluster = create_cluster(num_nodes=1, cpus_per_node=4, gpus_per_node=1)

        # Create a problem that would need more resources than available
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0",
                    worker_shape=WorkerShape.make(EntireGpu(1, num_cpus=2.0)),
                    requested_num_workers=2,  # Requires 2 entire GPUs
                    stage_batch_size=1,
                )
            ],
        )

        # Start with no workers
        state = data_structures.ProblemState(
            [data_structures.ProblemStageState(stage_name="stage_0", workers=[], slots_per_worker=1, is_finished=False)]
        )

        speeds = [1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )

        # Should raise RuntimeError due to insufficient resources
        with pytest.raises(naiive_worker_allocation.AllocationError):
            autoscaling_algorithms.run_fragmentation_autoscaler(
                problem=problem, state=state, overallocation_target=1.5, estimates=estimates
            )

    def test_mixed_resource_optimization(self):
        """Test optimization with mixed resource types (CPU, GPU, codecs)."""
        cluster = create_cluster(num_nodes=2, cpus_per_node=16, gpus_per_node=2, nvdecs_per_gpu=2, nvencs_per_gpu=2)

        # Create problem with different resource requirements
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                # CPU-intensive stage
                data_structures.ProblemStage(
                    name="cpu_stage", worker_shape=WorkerShape.make(CpuOnly(2.0)), stage_batch_size=1
                ),
                # GPU-intensive stage
                data_structures.ProblemStage(
                    name="gpu_stage",
                    worker_shape=WorkerShape.make(FractionalGpu(0.5, num_cpus=1.0)),
                    stage_batch_size=1,
                ),
                # Codec-intensive stage
                data_structures.ProblemStage(
                    name="codec_stage",
                    worker_shape=WorkerShape.make(Codec(num_cpus=1.0, num_nvdecs=1, num_nvencs=1)),
                    stage_batch_size=1,
                ),
            ],
        )

        # Start with no workers
        state = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name=stage.name, workers=[], slots_per_worker=1, is_finished=False
                )
                for stage in problem.stages
            ]
        )

        speeds = [1.0, 1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                for speed in speeds
            ]
        )
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, overallocation_target=1.5, estimates=estimates
        )
        assert solution.num_new_workers_per_stage == [9, 8, 6]
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]

    # ------------------------------------

    def test_no_allocation_possible_due_to_resource_constraints(self):
        """Test behavior when resource constraints prevent any allocations."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=2)

        # Create a problem that requires more CPUs than available
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(3.0)), stage_batch_size=1
                )
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        with pytest.raises(naiive_worker_allocation.AllocationError):
            autoscaling_algorithms.run_fragmentation_autoscaler(problem=problem, state=state, estimates=estimates)

    def test_stage_with_no_speed_measurement(self):
        """Test that stages with unknown speeds are still allocated minimum workers."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=4)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_1", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [None, 1.0]  # First stage has unknown speed
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates
        )

        # Both stages should have at least one worker
        assert solution.num_new_workers_per_stage == [2, 2]

    def test_mixed_manual_and_automatic_allocation(self):
        """Test handling of mixed manual and automatic worker allocations."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=10)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                # Manual allocation
                data_structures.ProblemStage(
                    name="stage_manual",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    requested_num_workers=5,
                    stage_batch_size=1,
                ),
                # Automatic allocation
                data_structures.ProblemStage(
                    name="stage_auto",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    stage_batch_size=1,
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates
        )

        assert solution.num_new_workers_per_stage == [5, 5]

    def test_overallocation_with_insufficient_resources(self):
        """Test that over-allocation doesn't exceed cluster capacity."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=8)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_1", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [0.5, 0.5]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, overallocation_target=2.0
        )

        # Total CPUs are 8, cannot allocate more than that
        total_workers = sum(solution.num_new_workers_per_stage)
        assert total_workers == 8
        assert solution.num_new_workers_per_stage == [4, 4]

    def test_error_when_manual_allocation_exceeds_resources(self):
        """Test that an error is raised when manual allocations exceed available resources."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=4)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_manual",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    requested_num_workers=5,  # Exceeds available CPUs
                    stage_batch_size=1,
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        with pytest.raises(naiive_worker_allocation.AllocationError):
            autoscaling_algorithms.run_fragmentation_autoscaler(problem=problem, state=state, estimates=estimates)

    def test_allocation_with_heterogeneous_cluster(self):
        """Test allocation on a cluster with heterogeneous nodes."""
        cluster = create_cluster(num_nodes=2, cpus_per_node=8, heterogeneous=True)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(2.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_1", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates
        )

        # Verify that workers are allocated respecting the node capacities
        total_workers_stage_0 = solution.num_new_workers_per_stage[0]
        total_workers_stage_1 = solution.num_new_workers_per_stage[1]

        assert total_workers_stage_0 * 2.0 + total_workers_stage_1 <= 12  # Total CPUs available

    def test_fragmentation_with_complex_resource_shapes(self):
        """Test handling of complex resource shapes including fractional GPUs and codecs."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=16, gpus_per_node=2, nvdecs_per_gpu=2)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                # Fractional GPU stage
                data_structures.ProblemStage(
                    name="stage_frac_gpu",
                    worker_shape=WorkerShape.make(FractionalGpu(0.25, num_cpus=1.0)),
                    stage_batch_size=1,
                ),
                # Codec stage
                data_structures.ProblemStage(
                    name="stage_codec",
                    worker_shape=WorkerShape.make(Codec(num_cpus=1.0, num_nvdecs=1)),
                    stage_batch_size=1,
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates
        )

        # Verify that resources are allocated without exceeding capacities
        total_frac_gpu_workers = solution.num_new_workers_per_stage[0]
        total_codec_workers = solution.num_new_workers_per_stage[1]

        assert total_frac_gpu_workers * 0.25 <= 2  # Total GPUs available
        assert total_codec_workers <= 4  # Total NVDECs available (2 GPUs * 2 NVDECs)

    # def test_handling_of_finished_stages(self):
    #     """Test that stages marked as finished do not receive new allocations."""
    #     cluster = create_cluster(num_nodes=1, cpus_per_node=8)

    #     problem = data_structures.Problem(
    #         cluster_resources=cluster,
    #         stages=[
    #             data_structures.ProblemStage(name="stage_finished", worker_shape=WorkerShape.make(CpuOnly(1.0))),
    #             data_structures.ProblemStage(name="stage_active", worker_shape=WorkerShape.make(CpuOnly(1.0))),
    #         ],
    #     )
    #     state = data_structures.ProblemState(
    #         [
    #             data_structures.ProblemStageState(
    #                 stage_name="stage_finished", workers=[], slots_per_worker=1, is_finished=True
    #             ),
    #             data_structures.ProblemStageState(
    #                 stage_name="stage_active", workers=[], slots_per_worker=1, is_finished=False
    #             ),
    #         ]
    #     )
    #     speeds = [1.0, 1.0]

    #     solution = autoscaling_algorithms.run_fragmentation_autoscaler(problem=problem, state=state, speeds=speeds)

    #     # Finished stage should not receive any new workers
    #     assert solution.num_new_workers_per_stage == [0, 8]

    def test_slot_calculation_with_changing_speeds(self):
        """Test that slot counts per worker adjust correctly with changing speeds."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=8)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )
        state = data_structures.ProblemState(
            [data_structures.ProblemStageState(stage_name="stage_0", workers=[], slots_per_worker=2, is_finished=False)]
        )
        speeds = [2.0]  # Speed increased
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates
        )

        # Slots per worker should adjust accordingly
        slots_per_worker = solution.stages[0].slots_per_worker
        assert slots_per_worker >= 2  # Should not decrease below current

    def test_allocation_with_fractional_cpu_requirements(self):
        """Test allocation when workers require fractional CPUs."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=8)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_fractional", worker_shape=WorkerShape.make(CpuOnly(0.5)), stage_batch_size=1
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates
        )

        # Can allocate up to 16 workers with 0.5 CPU each
        assert solution.num_new_workers_per_stage[0] == 16

    def test_error_on_invalid_worker_shape(self):
        """Test that invalid worker shapes raise an error."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=4)

        # Invalid shape with negative CPU
        invalid_shape = WorkerShape.make(CpuOnly(-1.0))

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(name="stage_invalid", worker_shape=invalid_shape, stage_batch_size=1),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        with pytest.raises(naiive_worker_allocation.AllocationError):
            autoscaling_algorithms.run_fragmentation_autoscaler(problem=problem, state=state, estimates=estimates)

    def test_allocation_with_large_overallocation_target(self):
        """Test behavior with a very high overallocation target."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=8)

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )
        state = make_default_state_for_stages(problem)
        speeds = [1.0]
        overallocation_target = 10.0  # Very high overallocation
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, overallocation_target=overallocation_target
        )

        # Should allocate as many workers as possible without exceeding resources
        assert solution.num_new_workers_per_stage[0] == 8

    def test_change_speed_with_one_stage_does_not_change_workers(self):
        """Test that increasing speed results in deallocating workers."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=10)
        factory = autoscaling_algorithms.WorkerIdFactory()

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_0", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )

        state = make_default_state_for_stages(problem)
        allocator_ = allocator.WorkerAllocator(
            problem.cluster_resources, autoscaling_algorithms._make_workers_from_problem_state(state)
        )

        # Initial speed is 1.0 tasks/sec
        speeds = [1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with initial speed
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem,
            state=state,
            estimates=estimates,
            factory=factory,
            overallocation_target=1.0,  # Disable over-allocation for clearer testing
        )

        # Should allocate all available CPUs
        assert solution.num_new_workers_per_stage == [10]
        assert solution.num_deleted_workers_per_stage == [0]

        # Apply solution
        apply_solution_to_state(state, solution, allocator_)
        assert len(state.stages[0].workers) == 10

        # Simulate speed increase to 2.0 tasks/sec
        speeds = [2.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with increased speed
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, factory=factory, overallocation_target=1.0
        )

        # Should deallocate half the workers since speed doubled
        assert solution.num_new_workers_per_stage == [0]
        assert solution.num_deleted_workers_per_stage == [0]

    def test_reallocate_workers_between_stages_on_speed_change(self):
        """Test that workers are reallocated between stages when speeds change."""
        # Create cluster with limited resources
        cluster = create_cluster(num_nodes=1, cpus_per_node=12)
        factory = autoscaling_algorithms.WorkerIdFactory()

        # Create problem with two stages
        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_slow", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_fast", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
            ],
        )

        # Create initial empty state
        state = make_default_state_for_stages(problem)

        # Create allocator for tracking resource usage
        allocator_ = allocator.WorkerAllocator(
            problem.cluster_resources, autoscaling_algorithms._make_workers_from_problem_state(state)
        )

        # Initial speeds: slow stage is slower
        speeds = [0.5, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with initial speeds
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, factory=factory
        )

        # Verify initial allocation
        assert solution.num_new_workers_per_stage == [8, 4]
        assert solution.num_deleted_workers_per_stage == [0, 0]

        # Apply solution to state
        apply_solution_to_state(state, solution, allocator_)

        # Simulate speed change: slow stage becomes faster
        speeds = [1.0, 0.5]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with new speeds
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, factory=factory
        )

        # Verify reallocation
        assert solution.num_new_workers_per_stage == [0, 4]
        assert solution.num_deleted_workers_per_stage == [4, 0]

    def test_changing_speeds_with_mixed_manual_and_automatic_allocation(self):
        """Test handling of mixed manual and automatic worker allocations."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=10)
        factory = autoscaling_algorithms.WorkerIdFactory()

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                # Manual allocation
                data_structures.ProblemStage(
                    name="stage_manual",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    requested_num_workers=5,
                    stage_batch_size=1,
                ),
                # Automatic allocation
                data_structures.ProblemStage(
                    name="stage_auto",
                    worker_shape=WorkerShape.make(CpuOnly(1.0)),
                    stage_batch_size=1,
                ),
            ],
        )

        state = make_default_state_for_stages(problem)
        allocator_ = allocator.WorkerAllocator(
            problem.cluster_resources, autoscaling_algorithms._make_workers_from_problem_state(state)
        )

        # Initial speeds
        speeds = [1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with initial speeds
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, factory=factory
        )

        # Manual stage should get exactly 5 workers, auto stage gets remaining
        assert solution.num_new_workers_per_stage == [5, 5]
        assert solution.num_deleted_workers_per_stage == [0, 0]

        # Apply solution
        apply_solution_to_state(state, solution, allocator_)

        # Simulate speed decrease in automatic stage
        speeds = [1.0, 0.5]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with new speeds
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem, state=state, estimates=estimates, factory=factory
        )

        # Manual stage should maintain 5 workers
        # Auto stage is limited by remaining CPUs
        assert solution.num_new_workers_per_stage == [0, 0]
        assert solution.num_deleted_workers_per_stage == [0, 0]

    def test_overallocation_target_maintained_on_speed_change(self):
        """Test that overallocation target is maintained when speeds change."""
        cluster = create_cluster(num_nodes=1, cpus_per_node=10, gpus_per_node=1)
        factory = autoscaling_algorithms.WorkerIdFactory()

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_cpu", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_gpu", worker_shape=WorkerShape.make(WholeNumberedGpu(1)), stage_batch_size=1
                ),
            ],
        )

        state = make_default_state_for_stages(problem)
        allocator_ = allocator.WorkerAllocator(
            problem.cluster_resources, autoscaling_algorithms._make_workers_from_problem_state(state)
        )

        overallocation_target = 2.0

        # Initial speeds
        speeds = [1.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with initial speeds
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem,
            state=state,
            estimates=estimates,
            factory=factory,
            overallocation_target=overallocation_target,
        )

        # Should allocate all CPUs evenly with overallocation
        assert solution.num_new_workers_per_stage == [2, 1]
        assert solution.num_deleted_workers_per_stage == [0, 0]

        # Apply solution
        apply_solution_to_state(state, solution, allocator_)

        speeds = [2.0, 1.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        # Run autoscaler with new speeds
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem,
            state=state,
            estimates=estimates,
            factory=factory,
            overallocation_target=overallocation_target,
        )

        assert solution.num_new_workers_per_stage == [0, 0]
        assert solution.num_deleted_workers_per_stage == [1, 0]

    def test_does_not_thrash_with_heterogenuous_resources(self):
        cluster = create_cluster(num_nodes=2, cpus_per_node=120, gpus_per_node=8)
        factory = autoscaling_algorithms.WorkerIdFactory()

        problem = data_structures.Problem(
            cluster_resources=cluster,
            stages=[
                data_structures.ProblemStage(
                    name="stage_cpu", worker_shape=WorkerShape.make(CpuOnly(1.0)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_slow_gpu", worker_shape=WorkerShape.make(WholeNumberedGpu(1)), stage_batch_size=1
                ),
                data_structures.ProblemStage(
                    name="stage_fast_gpu", worker_shape=WorkerShape.make(FractionalGpu(0.25)), stage_batch_size=1
                ),
            ],
        )

        state = make_default_state_for_stages(problem)
        allocator_ = allocator.WorkerAllocator(
            problem.cluster_resources, autoscaling_algorithms._make_workers_from_problem_state(state)
        )

        overallocation_target = 2.0
        speeds = [0.2, 1.0, 2.0]
        estimates = autoscaling_algorithms.Estimates(
            [
                autoscaling_algorithms.Estimate(batches_per_second_per_worker=speed, num_returns_per_batch=1.0)
                if speed is not None
                else autoscaling_algorithms.Estimate(batches_per_second_per_worker=None, num_returns_per_batch=None)
                for speed in speeds
            ]
        )

        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem,
            state=state,
            estimates=estimates,
            factory=factory,
            overallocation_target=overallocation_target,
        )
        apply_solution_to_state(state, solution, allocator_)
        solution = autoscaling_algorithms.run_fragmentation_autoscaler(
            problem=problem,
            state=state,
            estimates=estimates,
            factory=factory,
            overallocation_target=overallocation_target,
        )
        # The speeds did not change, so we should not be deleting and allocating new workers.
        assert solution.num_new_workers_per_stage == [0, 0, 0]
        assert solution.num_deleted_workers_per_stage == [0, 0, 0]
