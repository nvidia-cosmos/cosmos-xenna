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

"""Resource allocation optimizer for multi-stage streaming pipelines using linear programming.

This module implements an optimization-based resource allocation system that determines
the optimal number of workers for each stage in a streaming pipeline while respecting
resource constraints and throughput objectives. It handles both automatically scaled
stages and manually configured stages.

This module is "naiive" in the sense that it does not consider bin-packing. As a result, it is not a true
optimal solution to Yotta's auto-scaling problem. It may end up allocating miss-matched stages. It is a useful
estimate, however.

Key Features:
- Linear programming-based optimization
- Support for multiple resource types (CPU, GPU, NVDEC, NVENC)
- Balanced throughput across stages
- Mixed manual and automatic worker allocation
- Resource constraint satisfaction
- Throughput maximization

The optimizer uses PuLP to solve a mixed-integer linear programming problem that:
1. Maximizes minimum throughput across stages
2. Balances stage throughputs to prevent bottlenecks
3. Respects cluster resource limits
4. Handles manually specified worker counts
"""

from __future__ import annotations

from typing import Optional

import attrs
import pulp
import tabulate

from cosmos_xenna.ray_utils import resources


class AllocationError(Exception):
    pass


@attrs.define
class AllocationProblemStage:
    """Represents a single stage in the resource allocation problem.

    Each stage has specific resource requirements, processing speed characteristics,
    and optionally a fixed number of workers.

    Attributes:
        name: Unique identifier for the stage
        batches_per_second_per_worker: Processing speed (batches/second) for each worker
        average_num_returns_per_batch: Average number of returns per batch
        stage_batch_size: Number of inputs per batch
        resources_per_worker: Resource requirements (CPU, GPU, etc.) per worker
        requested_num_workers: Optional fixed number of workers for this stage
    """

    name: str
    batches_per_second_per_worker: float
    num_returns_per_batch: float
    stage_batch_size: int
    resources_per_worker: resources.PoolOfResources
    requested_num_workers: Optional[int] = None

    @property
    def was_manually_specified(self) -> bool:
        """Check if this stage has a manually specified worker count.

        Returns:
            bool: True if requested_num_workers is set, False otherwise
        """
        return self.requested_num_workers is not None


@attrs.define
class AllocationProblem:
    """Defines a complete resource allocation problem for optimization.

    Contains all information needed to solve the worker allocation problem:
    - Stage definitions with resource requirements
    - Available cluster resources
    - Processing speed characteristics
    - Manual worker count specifications

    Attributes:
        stages: List of all pipeline stages to allocate resources for
        cluster_resources: Total available resources in the cluster
    """

    stages: list[AllocationProblemStage]
    cluster_resources: resources.PoolOfResources


@attrs.define
class AllocationResultStage:
    """Results of resource allocation for a single pipeline stage.

    Tracks both the original problem definition and the solved allocation
    for a stage, providing methods to calculate resource usage and throughput.

    Attributes:
        problem: Original stage definition from the allocation problem
        num_workers: Number of workers allocated to this stage
    """

    problem: AllocationProblemStage
    num_workers: int
    input_samples_per_sample: float

    @property
    def name(self) -> str:
        """Get stage identifier.

        Returns:
            str: Unique name of the stage
        """
        return self.problem.name

    @property
    def throughput_per_worker(self) -> float:
        """Calculate per-worker throughput for this stage.

        Returns:
            float: Tasks processed per second per worker
        """
        return self.problem.batches_per_second_per_worker * self.problem.num_returns_per_batch

    @property
    def total_throughput(self) -> float:
        """Calculate total stage throughput across all workers.

        Returns:
            float: Total tasks processed per second for the stage
        """
        return self.problem.batches_per_second_per_worker * self.problem.num_returns_per_batch * self.num_workers

    def total_resource_usage(self) -> resources.PoolOfResources:
        """Calculate total resources used by all workers in this stage.

        Returns:
            PoolOfResources: Combined resource usage across all workers
        """
        return self.problem.resources_per_worker.mutiply_by(self.num_workers)


@attrs.define
class AllocationResult:
    """Complete results of the resource allocation optimization.

    Contains the solved allocation for all stages and provides methods
    to analyze and display the results.

    Attributes:
        stages: Allocation results for each pipeline stage
        cluster_resources: Total available cluster resources
        throughput: Achieved pipeline throughput (minimum across stages)
    """

    stages: list[AllocationResultStage]
    cluster_resources: resources.PoolOfResources
    throughput: float

    def to_debug_str(self) -> str:
        """Generate a human-readable representation of the allocation results.

        Creates a formatted table showing key metrics for each stage:
        - Stage name
        - Per-worker throughput
        - Manual specification status
        - Allocated workers
        - Total throughput
        - Resource usage (CPU, GPU, NVDEC, NVENC)

        Returns:
            str: Formatted table string with allocation details
        """
        table_data = []
        headers = [
            "Stage",
            "Throughput per worker",
            "Input samples per sample",
            "Input samples throughput per worker",
            "Average num returns per batch",
            "Batch size",
            "Manually specified",
            "Workers",
            "Throughput",
            "Input samples throughput",
            "CPUs",
            "GPUs",
            "NVDECs",
            "NVENCs",
        ]

        for stage in self.stages:
            resources = stage.total_resource_usage()
            table_data.append(
                [
                    stage.name,
                    stage.problem.batches_per_second_per_worker,
                    stage.input_samples_per_sample,
                    stage.problem.batches_per_second_per_worker * stage.input_samples_per_sample,
                    stage.problem.num_returns_per_batch,
                    stage.problem.stage_batch_size,
                    stage.problem.was_manually_specified,
                    stage.num_workers,
                    stage.total_throughput,
                    stage.total_throughput * stage.input_samples_per_sample,
                    resources.cpus,
                    resources.gpus,
                    resources.nvdecs,
                    resources.nvencs,
                ]
            )

        table = tabulate.tabulate(table_data, headers=headers, tablefmt="grid")
        return f"Allocation Result (Throughput: {self.throughput}):\n{table}"


def solve_allocation_with_no_manual_stages(
    problem: AllocationProblem, input_samples_per_sample: list[float]
) -> AllocationResult:
    """Solve the worker allocation problem for automatically scaled stages using linear programming.

    This function implements the core optimization algorithm that determines optimal
    worker counts for pipeline stages. It formulates and solves a mixed-integer linear
    programming problem that:

    1. Maximizes the minimum *adjusted* throughput across all stages
    2. Respects all resource constraints
    3. Minimizes excess capacity using slack variables

    Args:
        problem: Complete problem definition including stages and resources

    Returns:
        AllocationResult: Complete allocation solution with worker counts and throughput
    """
    assert len(problem.stages) == len(input_samples_per_sample)
    # Initialize the LP problem
    prob = pulp.LpProblem("Worker_Allocation", pulp.LpMaximize)

    # Define decision variables
    z = pulp.LpVariable("z", lowBound=0)  # Minimum throughput across all stages
    x = {}  # Number of workers for each stage
    t = {}  # Throughput of each stage
    s = {}  # Slack variables for excess capacity

    for i, _ in enumerate(problem.stages):
        x[i] = pulp.LpVariable(f"x_{i}", lowBound=1, cat="Integer")
        t[i] = pulp.LpVariable(f"t_{i}", lowBound=0)
        s[i] = pulp.LpVariable(f"s_{i}", lowBound=0)  # Slack variable for excess capacity

    # Define the objective function:
    # Maximize minimum throughput while minimizing slack
    # The coefficient 0.001 ensures throughput remains the primary objective
    slack_penalty = 0.001 * pulp.lpSum(s[i] for i in range(len(problem.stages)))
    prob += z - slack_penalty

    # Add constraints
    for i, stage in enumerate(problem.stages):
        # Throughput must be at least z
        prob += t[i] >= z
        # Throughput here is "number of stage input samples per second", which is:
        # x[i] (number of workers) * batches_per_second * stage_batch_size. However, we normalize it by the expected
        # number of first stage input samples per stage input sample, so it becomes something like
        # "number of first stage input samples per second". This is needed because some stages are
        # expected to see many more samples than other stages, so we want to account for that.
        prob += t[i] == (
            x[i] * input_samples_per_sample[i] * stage.batches_per_second_per_worker * stage.stage_batch_size
        )
        # The slack variable s_i measures excess capacity above the target throughput z
        prob += t[i] == z + s[i]  # Any throughput above z is captured in the slack

    # Resource constraints
    prob += (
        pulp.lpSum(stage.resources_per_worker.cpus * x[i] for i, stage in enumerate(problem.stages))
        <= problem.cluster_resources.cpus
    )
    prob += (
        pulp.lpSum(stage.resources_per_worker.gpus * x[i] for i, stage in enumerate(problem.stages))
        <= problem.cluster_resources.gpus
    )
    prob += (
        pulp.lpSum(stage.resources_per_worker.nvdecs * x[i] for i, stage in enumerate(problem.stages))
        <= problem.cluster_resources.nvdecs
    )
    prob += (
        pulp.lpSum(stage.resources_per_worker.nvencs * x[i] for i, stage in enumerate(problem.stages))
        <= problem.cluster_resources.nvencs
    )

    # Solve the optimization problem
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Check solution feasibility
    if status != pulp.LpStatusOptimal:
        raise AllocationError(f"No feasible solution found. Problem:\n{problem}")

    # Extract and format the results
    stages = []
    for i, stage in enumerate(problem.stages):
        num_workers = round(pulp.value(x[i]))
        stages.append(
            AllocationResultStage(
                problem=stage,
                num_workers=num_workers,
                input_samples_per_sample=input_samples_per_sample[i],
            )
        )

    # Get the actual throughput
    throughput = float(pulp.value(z))  # type: ignore

    return AllocationResult(stages, problem.cluster_resources, throughput)


def calculate_input_samples_per_sample(
    batch_sizes: list[int],
    num_returns_per_batch: list[float],
) -> list[float]:
    assert len(batch_sizes) == len(num_returns_per_batch)
    input_samples_per_sample = [0.0] * len(batch_sizes)
    input_samples_per_sample[0] = 1.0
    for i in range(1, len(batch_sizes)):
        input_samples_per_sample[i] = input_samples_per_sample[i - 1] * (
            batch_sizes[i - 1] / num_returns_per_batch[i - 1]
        )
    return input_samples_per_sample


def solve_allocation(
    problem: AllocationProblem,
) -> AllocationResult:
    """Solve the complete resource allocation problem, handling both manual and auto-scaling stages.

    This function serves as the main entry point for resource allocation. It:
    1. Validates input parameters
    2. Separates manually specified stages from auto-scaling stages
    3. Reserves resources for manual stages
    4. Optimizes remaining resources for auto-scaling stages
    5. Combines results into a complete allocation

    The process ensures that:
    - Manual stage specifications are honored exactly
    - Resource constraints are respected
    - Remaining resources are optimally allocated to auto-scaling stages
    - Pipeline throughput is maximized within constraints

    Args:
        problem: Complete allocation problem definition
        balance_multiplier: Factor controlling allowed throughput variation (default: 1.5)
        balance_weight: Weight given to throughput balancing (default: 0.1)

    Returns:
        AllocationResult: Complete allocation solution for all stages

    Raises:
        ValueError: If stage speeds are invalid or manual allocations exceed resources
    """

    # Validate stage speeds
    for stage in problem.stages:
        if stage.batches_per_second_per_worker <= 0.0:
            raise ValueError(
                f"Expected stage.batches_per_second_per_worker to be positive, but got "
                f"{stage.batches_per_second_per_worker} "
                f"for stage {stage.name}"
            )
        if stage.num_returns_per_batch <= 0.0:
            raise ValueError(
                f"Expected stage.average_num_returns_per_batch to be positive, but got "
                f"{stage.num_returns_per_batch} "
                f"for stage {stage.name}"
            )
        if stage.stage_batch_size <= 0:
            raise ValueError(
                f"Expected stage.stage_batch_size to be positive, but got {stage.stage_batch_size} "
                f"for stage {stage.name}"
            )

    input_samples_per_sample = calculate_input_samples_per_sample(
        [stage.stage_batch_size for stage in problem.stages],
        [stage.num_returns_per_batch for stage in problem.stages],
    )

    # Separate manual and auto-scaling stages
    manual_stages = [stage for stage in problem.stages if stage.requested_num_workers is not None]
    auto_stages = [stage for stage in problem.stages if stage.requested_num_workers is None]
    auto_samples_per_sample = [
        input_samples_per_sample[idx] for idx, stage in enumerate(problem.stages) if stage.requested_num_workers is None
    ]

    # Calculate resources needed for manual stages
    manual_resources_used = resources.PoolOfResources()
    for stage in manual_stages:
        assert stage.requested_num_workers is not None
        manual_resources_used += stage.resources_per_worker.mutiply_by(stage.requested_num_workers)

    # Verify manual allocations are feasible
    if not problem.cluster_resources.contains(manual_resources_used):
        raise AllocationError("Manually specified stages exceed available resources")

    # Calculate remaining resources for auto-scaling
    remaining_resources = problem.cluster_resources - manual_resources_used

    # Create and solve problem for auto-scaling stages
    auto_problem = AllocationProblem(stages=auto_stages, cluster_resources=remaining_resources)
    if auto_stages:
        auto_result = solve_allocation_with_no_manual_stages(auto_problem, auto_samples_per_sample)
    else:
        auto_result = AllocationResult([], remaining_resources, 0)

    # Combine manual and auto allocations into final result
    stages: list[AllocationResultStage] = []
    auto_index = 0
    for i, stage in enumerate(problem.stages):
        if stage.requested_num_workers is not None:
            num_workers = stage.requested_num_workers
        else:
            num_workers = auto_result.stages[auto_index].num_workers
            auto_index += 1

        stages.append(
            AllocationResultStage(
                problem=stage, num_workers=num_workers, input_samples_per_sample=input_samples_per_sample[i]
            )
        )

    # Calculate actual throughput as minimum across all stages
    actual_throughput = min(stage.total_throughput for stage in stages)

    return AllocationResult(stages, problem.cluster_resources, actual_throughput)
