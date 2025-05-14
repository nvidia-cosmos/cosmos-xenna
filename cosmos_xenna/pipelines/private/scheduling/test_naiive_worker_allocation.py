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

from typing import Optional

import pytest

from cosmos_xenna.pipelines.private.scheduling import naiive_worker_allocation
from cosmos_xenna.pipelines.private.scheduling.naiive_worker_allocation import (
    AllocationProblem,
    AllocationProblemStage,
    solve_allocation,
)
from cosmos_xenna.ray_utils import resources


def assert_allocation_result(result, expected_workers, expected_throughput: Optional[float] = None, tolerance=1e-6):
    print(result.to_debug_str())
    assert len(result.stages) == len(expected_workers)
    for stage, expected in zip(result.stages, expected_workers):  # py310 strict=False
        assert stage.num_workers == expected
    if expected_throughput is not None:
        assert abs(result.throughput - expected_throughput) < tolerance


def test_single_stage_allocation():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            )
        ],
        resources.PoolOfResources(cpus=5.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [5], 5.0)


def test_two_stage_equal_speed():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=6.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [3, 3], 3.0)


def test_two_stage_different_speed():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=6.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [4, 2], 4.0)


def test_three_stage_multiple_resources():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=0.5, gpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=2.0),
            ),
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=3.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.5, gpus=3.0),
            ),
        ],
        resources.PoolOfResources(cpus=10.0, gpus=20.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [6, 3, 2], 6.0)


def test_minimum_one_worker_per_stage():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=10.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=3.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [2, 1], 2.0)


def test_resource_limited_allocation():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=2.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0, gpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=10.0, gpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [3, 3], 3, tolerance=0.1)


def test_infeasible_problem():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0),
            ),
        ],
        resources.PoolOfResources(cpus=1.0),
    )
    with pytest.raises(naiive_worker_allocation.AllocationError):
        solve_allocation(problem)


def test_manual_and_auto_stages():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
                requested_num_workers=3,
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [3, 7], 3.0, tolerance=0.1)
    assert result.stages[0].problem.was_manually_specified
    assert not result.stages[1].problem.was_manually_specified


def test_extreme_speed_differences():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=0.1,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=100.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=20.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [19, 1], 1.9, tolerance=0.1)


def test_one_stage_dominates_resources():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=0.1, gpus=0.1),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=10.0, gpus=10.0),
            ),
        ],
        resources.PoolOfResources(cpus=100.0, gpus=100.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [9, 9], 9.0, tolerance=0.1)


def test_zero_speed_stage():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=0.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=100.0),
    )
    with pytest.raises(ValueError):
        solve_allocation(problem)


def test_very_small_resource_requirement():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=0.001),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=0.001),
            ),
        ],
        resources.PoolOfResources(cpus=1.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [500, 500], 500.0, tolerance=0.1)


def test_very_large_resource_requirement():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1e6),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1e6),
            ),
        ],
        resources.PoolOfResources(cpus=3e6),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [1, 1], 1.0, tolerance=0.1)


def test_exact_resource_match():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=2.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0, gpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=9.0, gpus=9.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [3, 3], 3.0, tolerance=0.1)


def test_fractional_resource_requirements():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=0.3, gpus=0.7),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=0.7, gpus=0.3),
            ),
        ],
        resources.PoolOfResources(cpus=10.0, gpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [10, 10], 10.0, tolerance=0.1)


def test_overallocation_limited_by_single_resource():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=0.1),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=0.1),
            ),
        ],
        resources.PoolOfResources(cpus=300.0, gpus=15.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [75, 75], 75.0, tolerance=0.1)


def test_single_manual_stage():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
                requested_num_workers=5,
            ),
        ],
        resources.PoolOfResources(cpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [5], 5.0)
    assert result.stages[0].problem.was_manually_specified


def test_multiple_manual_stages():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
                requested_num_workers=2,
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
                requested_num_workers=3,
            ),
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=3.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [2, 3, 5], 2.0, tolerance=0.1)
    assert result.stages[0].problem.was_manually_specified
    assert result.stages[1].problem.was_manually_specified
    assert not result.stages[2].problem.was_manually_specified


def test_manual_stages_exceed_resources():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0),
                requested_num_workers=3,
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
                requested_num_workers=5,
            ),
        ],
        resources.PoolOfResources(cpus=10.0),
    )
    with pytest.raises(naiive_worker_allocation.AllocationError):
        solve_allocation(problem)


def test_manual_stages_use_all_resources():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0),
                requested_num_workers=3,
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
                requested_num_workers=4,
            ),
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=3.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=10.0),
    )
    with pytest.raises(naiive_worker_allocation.AllocationError):
        solve_allocation(problem)


def test_manual_stages_with_multiple_resources():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=2.0),
                requested_num_workers=2,
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0, gpus=1.0),
            ),
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=3.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=10.0, gpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [2, 3, 2], 2.0, tolerance=0.1)
    assert result.stages[0].problem.was_manually_specified
    assert not result.stages[1].problem.was_manually_specified
    assert not result.stages[2].problem.was_manually_specified


def test_dont_overallocate():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=10.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=10000.0, gpus=8.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [80, 8], 80.0, tolerance=0.1)


def test_dont_overallocate_multiple_stages():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=3.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0, gpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=1000.0, gpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [30, 15, 10], 30.0, tolerance=0.1)


def test_dont_overallocate_limited_by_resources():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=2.0, gpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=60.0, gpus=10.0),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [20, 10], 20.0, tolerance=0.1)


def test_unbalanced_batching():
    problem = AllocationProblem(
        [
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1000,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=1000),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [1, 1, 998], tolerance=0.1)


def test_simple_batching():
    problem = AllocationProblem(
        [
            # Stage A processes 10 samples per second
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=10,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            # Stage B processes 1 sample per second, but sees 1/10 as many samples
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=1.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=1000),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [500, 500], tolerance=0.1)


def test_simple_batching_2():
    problem = AllocationProblem(
        [
            # Stage A processes 20 samples per second and produces 2000 samples per second
            AllocationProblemStage(
                name="A",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1000,
                stage_batch_size=10,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            # Stage B processes 2 sample per second, sees 2000 samples per second and produces 2 samples per second
            AllocationProblemStage(
                name="B",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            # Stage C processes 2000 samples per second
            AllocationProblemStage(
                name="C",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=10,
                stage_batch_size=1000,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
            AllocationProblemStage(
                name="D",
                batches_per_second_per_worker=2.0,
                num_returns_per_batch=1,
                stage_batch_size=1,
                resources_per_worker=resources.PoolOfResources(cpus=1.0),
            ),
        ],
        resources.PoolOfResources(cpus=1000),
    )
    result = solve_allocation(problem)
    assert_allocation_result(result, [1, 988, 1, 10], tolerance=0.1)
