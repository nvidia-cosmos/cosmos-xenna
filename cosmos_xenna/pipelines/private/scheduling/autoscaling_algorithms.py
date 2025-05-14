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

"""Algorithms for auto-scaling streaming pipeline workers with smart resource packing.

This module implements an adaptive worker allocation system for streaming pipelines that optimizes
resource utilization while maintaining performance. The system handles multi-stage pipelines where
each stage has specific resource requirements and throughput characteristics.

Key Features:
- Adaptive worker allocation based on real-time performance measurements
- Smart resource packing to minimize fragmentation
- Multi-stage pipeline support with different resource requirements per stage
- Automatic slot calculation to prevent work starvation
- Balanced worker distribution across stages based on throughput

The main algorithm combines:
1. Naive allocation to determine target worker counts
2. Fragmentation-aware worker placement
3. Priority-based stage scaling
4. Performance-based slot adjustment

The system continuously monitors stage performance and adjusts worker counts and slot allocations
to maintain optimal throughput while respecting resource constraints.
"""

import collections
import copy
import math
import statistics
import time
import typing
from typing import Optional, Union

import attrs
from loguru import logger

from cosmos_xenna.pipelines.private.scheduling import (
    data_structures,
    fragmentation_allocation_algorithms,
    naiive_worker_allocation,
)
from cosmos_xenna.ray_utils import allocator, resources
from cosmos_xenna.utils import attrs_utils, timing, verbosity


class AllocationError(Exception):
    pass


class WorkerIdFactory:
    """Generates unique worker IDs for new worker allocations.

    This simple counter-based ID generator ensures each worker gets a unique string identifier.
    The IDs are generated sequentially starting from 0.
    """

    def __init__(self) -> None:
        self._count = 0

    def make_new_id(self) -> str:
        """Generate a new unique worker ID.

        Returns:
            str: A unique string ID representing the worker
        """
        out = str(self._count)
        self._count += 1
        return out


# TODO: This contains a lot of copy-paste from the RateEstimatorDuration class. We should refactor this to use the same
# class for both.
class NumberOfReturnsEstimator:
    """Maintains rolling window estimates of the number of returns for pipeline stages."""

    def __init__(self, window_duration: float = 60 * 3, min_num_events: int = 5) -> None:
        self._window_duration = float(window_duration)
        self._events: collections.deque[tuple[float, int]] = collections.deque()
        self._min_num_events = min_num_events

    def _remove_old(self, current_time: float) -> None:
        """Removes old events based on time window and optionally min_num_events."""
        while self._events:  # Check if deque is non-empty first
            # Check if the oldest event is outside the time window
            is_too_old = current_time - self._events[0][0] > self._window_duration

            # If min_num_events is set, check if we are above the minimum count
            can_remove_based_on_count = self._min_num_events is None or len(self._events) > self._min_num_events

            # Remove if it's too old AND we are allowed to remove based on count (or count isn't enforced)
            if is_too_old and can_remove_based_on_count:
                self._events.popleft()
            else:
                # If the oldest isn't removable (either too new or protected by min_count), stop checking
                break

    def update(self, num_returns: int, current_time: Optional[float] = None) -> None:
        """
        Updates the estimator with a new duration and timestamp.

        Removes old entries based on the configured time window and `min_num_events` policy.

        Args:
            duration: Duration of the event in seconds.
            current_time: Optional timestamp for the event; defaults to `time.time()`.
        """
        if current_time is None:
            current_time = time.time()
        # Add the new duration
        self._events.append((current_time, num_returns))
        self._remove_old(current_time)

    def get_average_num_returns(self, current_time: Optional[float] = None) -> float:
        """
        Calculates and returns the average number of returns.

        The average is based on the average number of returns of the events currently stored,
        respecting the time window and optional `min_num_events` criteria.

        Args:
            current_time: Optional timestamp to use as 'now'; defaults to `time.time()`.

        Returns:
            The average number of returns. Returns 0 if no events are
            stored.
        """
        if current_time is None:
            current_time = time.time()
        self._remove_old(current_time)
        # Need at least 1 event to calculate a rate based on average duration.
        if not self._events:
            return 0.0  # No data

        average_num_returns = statistics.mean([x[1] for x in self._events])
        return average_num_returns

    def maybe_get_average_num_returns(self, current_time: Optional[float] = None) -> Optional[float]:
        """
        Calculates and returns the average number of returns, or None if insufficient data.

        Similar to `get_average_num_returns`, but returns None if no events are stored.

        Args:
            current_time: Optional timestamp to use as 'now'; defaults to `time.time()`.

        Returns:
            The average number of returns, or None if no events are stored.
        """
        if current_time is None:
            current_time = time.time()
        self._remove_old(current_time)
        # Need at least 1 event to calculate a rate.
        if not self._events:
            return None  # No data

        # Calculate the average number of returns of events within the window/min_events
        average_num_returns = statistics.mean([x[1] for x in self._events])
        return average_num_returns


@attrs.define
class Estimate:
    batches_per_second_per_worker: Optional[float]
    num_returns_per_batch: Optional[float]


@attrs.define
class Estimates:
    stages: list[Estimate]


class SpeedAndNumberOfReturnsEstimator:
    """Maintains rolling window estimates of processing speeds for pipeline stages.

    This class tracks the processing rate (tasks/second) for each stage in the pipeline
    using a sliding time window. It maintains both current and historical speed estimates
    to ensure stable scaling decisions.

    Args:
        num_stages (int): Number of pipeline stages to track
        window_duration (float): Duration of the sliding window in seconds (default: 60 * 3)
    """

    def __init__(
        self,
        num_stages: int,
        window_duration: float = 60 * 3,
        min_num_events: int = 5,
        verbosity_level: verbosity.VerbosityLevel = verbosity.VerbosityLevel.NONE,
    ) -> None:
        self._window_duration = float(window_duration)
        self._speed_estimators: list[timing.RateEstimatorDuration] = [
            timing.RateEstimatorDuration(window_duration, min_num_events=min_num_events) for _ in range(num_stages)
        ]
        self._num_returns_estimators: list[NumberOfReturnsEstimator] = [
            NumberOfReturnsEstimator(window_duration, min_num_events=min_num_events) for _ in range(num_stages)
        ]
        self._last_valid_speeds: list[Optional[float]] = [None for _ in range(num_stages)]
        self._last_valid_num_returns: list[Optional[float]] = [None for _ in range(num_stages)]
        self._verbosity_level = verbosity_level

    def update_with_measurements(self, measurements: data_structures.Measurements) -> None:
        """Update speed estimates with new task timing measurements.

        Args:
            measurements: Collection of task timing measurements for each stage
        """
        for stage_measurements, speed_estimator, num_returns_estimator in zip(
            measurements.stages, self._speed_estimators, self._num_returns_estimators
        ):
            for measurement in stage_measurements.task_measurements:
                speed_estimator.update(measurement.end_time - measurement.start_time, measurement.start_time)
                num_returns_estimator.update(measurement.num_returns, measurement.start_time)

    def get_estimates(self, current_time: float) -> Estimates:
        """Get current speed estimates for all stages.

        Args:
            current_time (float): Current timestamp for calculating rates

        Returns:
            list[Optional[float]]: List of speed estimates (tasks/second) for each stage,
                               None if no measurements available in the window
        """

        maybe_speeds = [x.maybe_get_rate(current_time) for x in self._speed_estimators]
        maybe_num_returns = [x.maybe_get_average_num_returns(current_time) for x in self._num_returns_estimators]
        for i, (maybe_speed, maybe_num_return) in enumerate(zip(maybe_speeds, maybe_num_returns)):
            if maybe_speed is not None:
                self._last_valid_speeds[i] = maybe_speed
            if maybe_num_return is not None:
                self._last_valid_num_returns[i] = maybe_num_return
        if self._verbosity_level >= verbosity.VerbosityLevel.DEBUG:
            logger.debug(
                f"Got the following speeds:\n{maybe_speeds}\nand the following last valid speeds:\n"
                f"{self._last_valid_speeds}"
            )
        return Estimates(
            [
                Estimate(batches_per_second_per_worker=x, num_returns_per_batch=y)
                for x, y in zip(self._last_valid_speeds, self._last_valid_num_returns)
            ]
        )

    def get_last_valid_estimates(self, current_time: float) -> Estimates:
        """Get most recent valid speed estimates for all stages.

        This method returns the last known valid speed for each stage, even if the
        current window has no measurements. This provides stability for scaling decisions.

        Args:
            current_time (float): Current timestamp

        Returns:
            list[Optional[float]]: List of most recent valid speed estimates for each stage
        """
        self.get_estimates(current_time)
        return Estimates(
            [
                Estimate(batches_per_second_per_worker=x, num_returns_per_batch=y)
                for x, y in zip(self._last_valid_speeds, self._last_valid_num_returns)
            ]
        )


def _calculate_num_slots_per_worker_for_all_stages(
    num_workers: list[int],
    current_slots: list[int],
    last_valid_speeds: Union[list[float], list[Optional[float]]],
    min_scheduling_algorithm_rate_hz: float = 1.0,
    min_slots: int = 2,
) -> list[int]:
    """Calculate optimal number of task slots for workers in each stage.

    This function ensures each stage has enough task slots to prevent work starvation
    between scheduler iterations. It accounts for:
    - Pipeline throughput
    - Number of workers per stage
    - Scheduler update frequency
    - Current slot allocations

    The calculation ensures that even fast stages with few workers have enough slots
    to maintain throughput between scheduler updates.

    Formula per stage:
    - num_slots_per_worker >= (pipeline_throughput * scheduler_period * 2) / num_workers

    Args:
        num_workers: Number of workers per stage
        current_slots: Current number of slots per worker for each stage
        last_valid_speeds: Most recent speed measurements for each stage
        min_scheduling_algorithm_rate_hz: Minimum frequency of scheduler updates (default: 1.0)

    Returns:
        list[int]: Updated number of slots per worker for each stage
    """
    assert len(num_workers) == len(current_slots) == len(last_valid_speeds)
    # We will only adjust slots if we know how speed estimates for all the stages
    if any([x is None for x in last_valid_speeds]):
        return list(current_slots)
    if not num_workers:
        return []
    stage_speeds: list[float] = [x * y for x, y in zip(num_workers, last_valid_speeds)]  # type: ignore
    pipeline_speed = min(stage_speeds, default=0.0)
    num_tasks_processed_per_slowest_scheduler_loop_call = pipeline_speed * 1.0 / min_scheduling_algorithm_rate_hz

    out: list[int] = []
    for num_workers_for_stage, current_slots_for_stage in zip(num_workers, current_slots):
        total_slots_needed = num_tasks_processed_per_slowest_scheduler_loop_call
        slots_needed_per_worker = math.ceil(2.0 * total_slots_needed / num_workers_for_stage)
        num_slots = max([slots_needed_per_worker, current_slots_for_stage, min_slots])
        out.append(num_slots)
    return out


@attrs.define
class _NaiiveAllocationResult:
    """Results from the initial naive worker allocation phase.

    Attributes:
        allocation_result: Worker counts and resources allocated per stage
        num_slots_per_state: Number of task slots assigned per worker for each stage
    """

    allocation_result: naiive_worker_allocation.AllocationResult
    num_slots_per_state: list[int]


def _run_naiive_allocation(
    problem: data_structures.Problem,
    state: data_structures.ProblemState,
    estimates: Estimates,
    verbosity_level: verbosity.VerbosityLevel = verbosity.VerbosityLevel.NONE,
) -> _NaiiveAllocationResult:
    """Perform initial naive allocation of workers to stages.

    This function creates a first-pass allocation that:
    1. Respects basic resource constraints
    2. Attempts to balance pipeline throughput
    3. Maintains minimum worker counts per stage

    The naive allocation serves as a starting point for the more sophisticated
    fragmentation-aware allocation process.

    Args:
        problem: Pipeline configuration and constraints
        state: Current pipeline state
        avg_speeds: Average processing speed for each stage

    Returns:
        NaiiveAllocationResult: Initial worker and slot allocations
    """

    stages = []
    for i in range(len(estimates.stages)):
        batches_per_second_per_worker = 1.0
        num_returns_per_batch = problem.stages[i].stage_batch_size
        if estimates.stages[i].batches_per_second_per_worker is not None:
            batches_per_second_per_worker = estimates.stages[i].batches_per_second_per_worker
        if estimates.stages[i].num_returns_per_batch is not None and estimates.stages[i].num_returns_per_batch > 0:
            num_returns_per_batch = estimates.stages[i].num_returns_per_batch
        stages.append(
            naiive_worker_allocation.AllocationProblemStage(
                name=problem.stages[i].name,
                stage_batch_size=problem.stages[i].stage_batch_size,
                batches_per_second_per_worker=typing.cast(float, batches_per_second_per_worker),
                num_returns_per_batch=typing.cast(float, num_returns_per_batch),
                resources_per_worker=problem.stages[i].worker_shape.to_pool_of_resources(
                    problem.cluster_resources.calc_num_nvdecs_per_gpu(),
                    problem.cluster_resources.calc_num_nvencs_per_gpu(),
                ),
                requested_num_workers=problem.stages[i].requested_num_workers,
            )
        )

    allocation_problem = naiive_worker_allocation.AllocationProblem(
        stages=stages,
        cluster_resources=problem.cluster_resources.totals(),
    )
    if verbosity_level >= verbosity.VerbosityLevel.INFO:
        logger.info(
            f"Solving the following naiive allocation problem:\n{attrs_utils.format_attrs_object(allocation_problem)}"
        )
    allocation_result = naiive_worker_allocation.solve_allocation(allocation_problem)
    num_slots_per_stage = _calculate_num_slots_per_worker_for_all_stages(
        [x.num_workers for x in allocation_result.stages],
        [x.slots_per_worker for x in state.stages],
        [x.batches_per_second_per_worker for x in estimates.stages],
    )
    out = _NaiiveAllocationResult(allocation_result, num_slots_per_stage)
    if verbosity_level >= verbosity.VerbosityLevel.INFO:
        logger.info(f"Got the following naiive allocation result:\n{attrs_utils.format_attrs_object(out)}")
    return out


def _calculate_workload_based_on_naiive_allocation(
    naiive_allocation: _NaiiveAllocationResult, problem: data_structures.Problem
) -> fragmentation_allocation_algorithms.Workload:
    """Convert naive allocation results into a workload representation for fragmentation analysis.

    This function transforms the initial naive allocation into a normalized workload
    representation that can be used by the fragmentation-aware allocation algorithms.
    Each stage's workload is weighted based on its proportion of total requested workers.

    Args:
        naiive_allocation: Results from the initial naive allocation
        problem: Original pipeline configuration and constraints

    Returns:
        Workload: Normalized workload representation for fragmentation analysis
    """
    total_num_requested_workers = sum([x.num_workers for x in naiive_allocation.allocation_result.stages], 0)

    return fragmentation_allocation_algorithms.Workload(
        [
            fragmentation_allocation_algorithms.Stage(
                result_stage.num_workers / total_num_requested_workers,
                problem_stage.worker_shape,
            )
            for result_stage, problem_stage in zip(naiive_allocation.allocation_result.stages, problem.stages)
        ]
    )


def _make_workers_from_problem_state(state: data_structures.ProblemState) -> list[resources.Worker]:
    """Convert pipeline state into a list of worker instances.

    This helper function transforms the current pipeline state into a list of
    Worker objects that can be used by the allocation algorithms.

    Args:
        state: Current state of the pipeline including stage definitions

    Returns:
        list[Worker]: List of worker instances with their current resource allocations
    """
    out = []
    for x in state.stages:
        for w in x.workers:
            out.append(resources.Worker(w.id, x.stage_name, w.resources))
    return out


@attrs.define
class _Stage:
    """Represents a pipeline stage with its current state and performance metrics."""

    name: str
    current_workers: int
    speed_per_worker: Optional[float]
    stage_batch_size: int
    num_returns_per_batch: Optional[float]
    # The expected number of first stage input samples per input sample to this stage.
    # For stage 1, this is 1.0.
    # For stage 2, this is 1.0 * stage_1_num_returns_per_batch / stage_1_batch_size.
    # For stage 3, this is 1.0 * stage_1_num_returns_per_batch / stage_1_batch_size * stage_2_num_returns_per_batch / stage_2_batch_size.  # noqa: E501
    # And so on.
    #
    # This is needed because balancing the pipeline is done by balancing the stage throughput * num_input_samples_per_sample.  # noqa: E501
    num_input_samples_per_sample: Optional[float]
    shape: resources.WorkerShape
    requested_num_workers: Optional[int] = None

    @property
    def throughput(self) -> float:
        if self.speed_per_worker is None:
            raise ValueError("self.speed_per_worker is None")
        if self.num_input_samples_per_sample is None:
            raise ValueError("self.num_input_samples_per_sample is None")
        return self.current_workers * self.speed_per_worker * self.num_input_samples_per_sample * self.stage_batch_size

    @property
    def throughput_if_one_worker_removed(self) -> float:
        if self.speed_per_worker is None:
            raise ValueError("self.speed_per_worker is None")
        if self.num_input_samples_per_sample is None:
            raise ValueError("self.num_input_samples_per_sample is None")
        return (
            (self.current_workers - 1)
            * self.speed_per_worker
            * self.num_input_samples_per_sample
            * self.stage_batch_size
        )

    @property
    def throughput_if_one_worker_added(self) -> float:
        if self.speed_per_worker is None:
            raise ValueError("self.speed_per_worker is None")
        if self.num_input_samples_per_sample is None:
            raise ValueError("self.num_input_samples_per_sample is None")
        return (
            (self.current_workers + 1)
            * self.speed_per_worker
            * self.num_input_samples_per_sample
            * self.stage_batch_size
        )

    def __hash__(self) -> int:
        return hash(self.name)


def run_fragmentation_autoscaler(
    problem: data_structures.Problem,
    state: data_structures.ProblemState,
    estimates: Estimates,
    verbosity_level: verbosity.VerbosityLevel = verbosity.VerbosityLevel.NONE,
    overallocation_target: float = 1.5,
    factory: Optional[WorkerIdFactory] = None,
) -> data_structures.Solution:
    """Autoscaling algorithm for streaming pipeline workers that optimizes resource allocation.

    This algorithm handles worker allocation for multi-stage streaming pipelines where each stage
    processes data and passes it to the next stage. The goal is to maximize the minimum throughput
    across all stages while respecting resource constraints and minimizing fragmentation.

    The algorithm operates in four distinct phases:
    1. Manual Allocation: Ensures stages with manually specified worker counts get exactly what they need
    2. Minimum Workers: Guarantees every stage has at least one worker to maintain pipeline flow
    3. Throughput Optimization: Systematically improves the slowest stage's throughput by:
       - Attempting direct allocation of new workers
       - Removing workers from faster stages if it would help
       - Continues until no further improvements are possible
    4. Over-allocation: Adds additional workers to provide throughput headroom, up to the specified
       overallocation target

    Key Features:
    - Respects manually specified worker counts exactly
    - Maintains minimum one worker per stage
    - Handles stages with unknown speeds (no measurements yet)
    - Uses fragmentation-aware allocation to optimize resource usage
    - Provides controlled over-allocation for throughput stability

    Resource Allocation Strategy:
    - Uses gradient descent for fragmentation-aware worker placement
    - Considers multi-dimensional resources (CPU, GPU, encoders, decoders)
    - Only removes workers if it enables improving the slowest stage
    - Balances throughput across stages while respecting resource constraints

    Args:
        problem: Pipeline configuration including stage definitions and resource constraints
        state: Current pipeline state with existing worker allocations
        speeds: List of speed measurements for each stage (tasks/sec), None for stages without measurements
        overallocation_target: Target multiplier for over-allocation (e.g., 1.5 means 50% extra capacity)
        factory: Optional worker ID generator, creates sequential IDs if not provided
        verbosity_level: Controls the level of logging detail.

    Returns:
        Solution containing worker additions and removals for each stage

    Raises:
        RuntimeError: If unable to satisfy manual worker requirements or minimum one worker per stage
    """
    if factory is None:
        factory = WorkerIdFactory()
    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info(
            "Running fragmentation based autoscaler on the following problem:\n"
            f"{attrs_utils.format_attrs_object(problem)}\n"
            f"And the following estimates:\n"
            f"{estimates}\n"
        )
    # Initialize stage tracking with current state and requirements
    stages: list[_Stage] = []
    for stage_problem, stage_stage, stage_estimate in zip(problem.stages, state.stages, estimates.stages):
        stages.append(
            _Stage(
                stage_problem.name,
                stage_stage.num_workers,
                stage_estimate.batches_per_second_per_worker,
                stage_problem.stage_batch_size,
                stage_estimate.num_returns_per_batch,
                None,
                stage_problem.worker_shape,
                stage_problem.requested_num_workers,
            )
        )

    stage_names = [x.name for x in stages]
    assert len(stage_names) == len(set(stage_names)), f"Expected stage names to be unique, but got: {stage_names}"
    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info("Running naiive allocation.")

    for stage in stages:
        if stage.speed_per_worker is None:
            if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
                logger.warning(f"Stage {stage.name} has no speed measurements. Using 1.0 as speed.")
            stage.speed_per_worker = 1.0
        if stage.num_returns_per_batch is None or stage.num_returns_per_batch == 0:
            if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
                logger.warning(
                    f"Stage {stage.name} has no num_returns_per_batch measurements. "
                    "Using stage_batch_size as num_returns_per_batch."
                )
            stage.num_returns_per_batch = stage.stage_batch_size

    input_samples_per_sample = naiive_worker_allocation.calculate_input_samples_per_sample(
        [stage.stage_batch_size for stage in problem.stages],
        [typing.cast(float, stage.num_returns_per_batch) for stage in stages],
    )
    for input_samples_per_sample_, stage in zip(input_samples_per_sample, stages):
        stage.num_input_samples_per_sample = input_samples_per_sample_

    # Setup allocation tracking structures
    # - Use naive allocation to get initial slot counts and workload estimates
    # - Track all worker additions and removals for generating final solution
    naiive_result = _run_naiive_allocation(problem, state, estimates, verbosity_level)
    workload_estimate = _calculate_workload_based_on_naiive_allocation(naiive_result, problem)
    allocator_ = allocator.WorkerAllocator(problem.cluster_resources, _make_workers_from_problem_state(state))
    workers_to_add: dict[str, set[resources.Worker]] = collections.defaultdict(set)
    workers_to_remove: dict[str, set[resources.Worker]] = collections.defaultdict(set)

    def _try_allocate_worker(
        stage: _Stage,
    ) -> bool:
        """Attempt to allocate a new worker for a stage with fragmentation-aware placement."""
        allocation = fragmentation_allocation_algorithms.find_best_allocation_using_fragmentation_gradient_descent(
            fragmentation_allocation_algorithms.ClusterResourceHelpers.make_from_allocator(allocator_),
            workload_estimate,
            stage.shape,
            workers_to_remove[stage.name],
        )
        if not allocation.did_allocate:
            return False

        # Resuse a worker.
        if allocation.reused_worker is not None:
            allocator_.add_worker(allocation.reused_worker)
            workers_to_remove[stage.name].remove(allocation.reused_worker)
            stage.current_workers += 1
        # Allocate a new worker
        else:
            assert allocation.resources is not None
            worker = resources.Worker(factory.make_new_id(), stage.name, allocation.resources)
            allocator_.add_worker(worker)
            workers_to_add[stage.name].add(worker)
            stage.current_workers += 1
        return True

    def _remove_best_worker(
        stage: _Stage,
    ) -> None:
        """Remove the worker that minimizes fragmentation impact from a stage."""
        worker = fragmentation_allocation_algorithms.find_worker_to_delete_using_fragmentation_gradient_descent(
            fragmentation_allocation_algorithms.ClusterResourceHelpers.make_from_allocator(allocator_),
            workload_estimate,
            allocator_.get_workers_in_stage(stage.name),
        )
        allocator_.delete_worker(worker.id)
        workers_to_remove[stage.name].add(worker)
        stage.current_workers -= 1

    def _make_output() -> data_structures.Solution:
        # Create final solution with all our allocation decisions
        out = data_structures.Solution(
            [data_structures.StageSolution(slots_per_worker=x) for x in naiive_result.num_slots_per_state]
        )

        for idx, stage in enumerate(problem.stages):
            out.stages[idx].new_workers = [
                data_structures.ProblemWorkerState.make_from_worker_state(x) for x in workers_to_add[stage.name]
            ]
            out.stages[idx].deleted_workers = [
                data_structures.ProblemWorkerState.make_from_worker_state(x) for x in workers_to_remove[stage.name]
            ]
        return out

    # Phase 1: Handle manually specified worker counts
    #
    # Process all stages with explicit worker count requirements first. These take
    # priority and must be satisfied exactly. Failing to meet these requirements
    # is a fatal error.
    #
    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info("Running phase 1...")
    for stage in stages:
        if stage.requested_num_workers is None:
            continue

        # Add workers until we reach the requested count
        while stage.current_workers < stage.requested_num_workers:
            if not _try_allocate_worker(stage):
                raise AllocationError(
                    f"Unable to allocate requested workers for stage={stage.name}. "
                    f"Requested={stage.requested_num_workers}, Current={stage.current_workers}"
                )

        # Remove excess workers if we have too many
        while stage.current_workers > stage.requested_num_workers:
            _remove_best_worker(stage)

    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info(f"Phase 1 complete. Current state:\n{attrs_utils.format_attrs_list(stages)}")

    # Phase 2: Ensure minimum workers
    #
    # Every stage must have at least one worker to maintain pipeline flow.
    # Skip stages with manual worker counts (already handled in Phase 1).
    #
    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info("Running phase 2...")
    for stage in stages:
        if stage.requested_num_workers is not None:
            continue

        if stage.current_workers < 1:
            if not _try_allocate_worker(stage):
                raise RuntimeError(f"Unable to allocate minimum worker for stage={stage.name}")

    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info(f"Phase 2 complete. Current state:\n{attrs_utils.format_attrs_list(stages)}")

    # Phase 3: Optimize minimum throughput
    #
    # This is the core balancing phase of the algorithm. We repeatedly try to improve
    # the slowest stage's throughput by either:
    # a) Directly allocating more workers if resources are available
    # b) Moving resources from faster stages if beneficial
    #
    # Only consider "active" stages:
    # - Exclude manually specified stages (already handled)
    # - Exclude stages without speed measurements (can't optimize what we can't measure)
    #
    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info("Running phase 3...")
    active_stages = [
        s
        for s in stages
        if s.requested_num_workers is None  # Skip manually specified
        and s.speed_per_worker is not None  # Skip unknown speed
    ]

    if not active_stages:
        return _make_output()

    while True:
        # Find current slowest stage - this is what we're trying to improve
        min_throughput_stage = min(active_stages, key=lambda s: s.throughput)
        min_throughput = min_throughput_stage.throughput

        # First attempt: Try direct allocation to slowest stage
        if _try_allocate_worker(min_throughput_stage):
            # Direct allocation succeeded. Move on to the next iteration to try to allocate again
            continue

        # Direct allocation failed - look for resources in other stages
        # Sort stages by their throughput after theoretical worker removal
        removable_stages = [
            stage
            for stage in active_stages
            if (
                stage.throughput_if_one_worker_removed > min_throughput  # Won't become bottleneck
                and stage.current_workers > 1  # Keep minimum workers
                and stage != min_throughput_stage  # Don't remove from stage we're trying to improve
            )
        ]

        if not removable_stages:
            # No stages can safely give up workers, so we're done
            break

        # Take resources from stage with highest remaining throughput after removal
        stage_to_remove_from = max(removable_stages, key=lambda s: s.throughput_if_one_worker_removed)
        _remove_best_worker(stage_to_remove_from)

    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info(f"Phase 3 complete. Current state:\n{attrs_utils.format_attrs_list(active_stages)}")
    #
    # Phase 4: Over-allocation
    #
    # After optimizing minimum throughput, we try to add extra capacity up to
    # our over-allocation target. This provides headroom for throughput spikes
    # and helps stability.
    #
    # We prioritize slower stages but will add to any stage that:
    # a) Won't exceed the over-allocation target
    # b) Has available resources for allocation
    #
    if verbosity_level >= verbosity.VerbosityLevel.INFO:
        logger.info("Running phase 4...")

    # Find current slowest stage - this is what we're trying to improve
    min_throughput_stage = min(active_stages, key=lambda s: s.throughput)
    min_throughput = min_throughput_stage.throughput
    while active_stages:
        # Filter stages that won't exceed over-allocation target with one more worker
        stages_to_consider = [
            s for s in active_stages if s.throughput_if_one_worker_added <= (min_throughput * overallocation_target)
        ]
        if not stages_to_consider:
            break

        # Prefer slower stages (may need headroom more)
        sorted_stages_to_consider = sorted(stages_to_consider, key=lambda s: s.throughput)

        # Try to allocate to any stage
        allocated_to_any = False
        stages_to_remove = set()

        for stage in sorted_stages_to_consider:
            if _try_allocate_worker(stage):
                allocated_to_any = True
                break
            else:
                # If we can't allocate to this stage, remove it from future consideration
                stages_to_remove.add(stage)

        # If we couldn't allocate to any stage we tried, we're done
        if not allocated_to_any:
            break

        # Remove stages that we know we can't allocate to anymore
        active_stages = [s for s in active_stages if s not in stages_to_remove]

    if verbosity_level >= verbosity.VerbosityLevel.DEBUG:
        logger.info(f"Phase 4 complete. Current state:\n{attrs_utils.format_attrs_list(active_stages)}")
    return _make_output()


class FragmentationBasedAutoscaler(data_structures.AutoScalingAlgorithmInterface):
    """Fragmentation based autoscaling algorith.

    See run_fragmentation_autoscaler for details on the algorith. This class is just a wrapper which does speed
    estimation and fulfills the `AutoScalingAlgorithmInterface`.
    """

    def __init__(
        self,
        speed_estimation_window_duration_s: float = 60.0 * 3,
        verbosity_level: verbosity.VerbosityLevel = verbosity.VerbosityLevel.NONE,
    ):
        self._speed_estimation_window_duration_s = speed_estimation_window_duration_s
        self._id_factory = WorkerIdFactory()
        self._verbosity_level = verbosity_level

    @property
    def name(self) -> str:
        return "fragmentation_based_autoscaler"

    def setup(self, problem: data_structures.Problem) -> None:
        if self._verbosity_level >= verbosity.VerbosityLevel.INFO:
            logger.info(f"Setting up fragmentation based autoscaler with problem: {problem}")
        self._problem = problem
        stage_names = [x.name for x in problem.stages]
        assert len(stage_names) == len(set(stage_names)), f"Expected stage names to be unique, but got: {stage_names}"
        self._speed_calculator = SpeedAndNumberOfReturnsEstimator(
            len(problem.stages), self._speed_estimation_window_duration_s, verbosity_level=self._verbosity_level
        )

    def update_with_measurements(self, current_time: float, measurements: data_structures.Measurements) -> None:
        self._speed_calculator.update_with_measurements(measurements)

    def autoscale(
        self,
        current_time: float,
        state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        assert len(state.stages) == len(self._problem.stages)
        state = copy.deepcopy(state)
        avg_speeds = self._speed_calculator.get_last_valid_estimates(current_time)

        if self._verbosity_level >= verbosity.VerbosityLevel.DEBUG:
            logger.info("Running algorithm...")
        # Call a function which isolates the actual algorithm. We do this to make it slightly easier to test.
        out = run_fragmentation_autoscaler(
            self._problem,
            state,
            avg_speeds,
            verbosity_level=self._verbosity_level,
            overallocation_target=1.5,
            factory=self._id_factory,
        )
        if self._verbosity_level >= verbosity.VerbosityLevel.DEBUG:
            logger.info("Finished running algorithm.")
        return out
