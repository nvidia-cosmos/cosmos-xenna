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

"""Simulation system for evaluating auto-scaling strategies in streaming pipelines.

This module implements a discrete event simulation for testing and evaluating different
auto-scaling algorithms on multi-stage streaming pipelines. The simulation models:

- Multi-stage pipelines with different resource requirements per stage
- Complex resource types (CPU, GPU, NVDEC, NVENC)
- Worker setup time and processing time
- Queue sizes and backpressure between stages
- Resource allocation and fragmentation

The simulation runs in fixed time steps and measures key metrics like:
- Per-stage throughput and utilization
- Queue sizes and bottlenecks
- Resource usage efficiency
- End-to-end pipeline performance

See `run_simulator.py` for a usage example.

TODO: Xenna now operates on a sample-by-sample basis. We should update the simulator to match this.
"""

from __future__ import annotations

import copy
import datetime
import os
import time

import attrs
import plotly.graph_objs as go
import tabulate
from loguru import logger
from plotly.subplots import make_subplots

from cosmos_xenna.pipelines.private.scheduling import data_structures
from cosmos_xenna.ray_utils import allocator, resources

max_runtime = 60 * 10  # 5 minutes

_VERBOSE = False


def _is_multiple(number: float, base: float, threshold: float = 1e-9) -> bool:
    """Checks if a number is approximately a multiple of another number."""
    if base == 0:
        return abs(number) <= threshold

    ratio = number / base
    return abs(ratio - round(ratio)) <= threshold


@attrs.define
class SimulationStage:
    """Represents a single stage in the simulated pipeline.

    Each stage models a processing unit that can have multiple workers assigned to it.
    The stage has specific timing characteristics for processing tasks and setting up
    new workers.

    Attributes:
        problem: The underlying stage configuration and resource requirements
        process_time_per_worker: Time in seconds for a worker to process one task
        setup_time: Time in seconds required to initialize a new worker before it can
            begin processing tasks
    """

    problem_stage: data_structures.ProblemStage
    process_time_per_worker: float  # Processing speed per worker
    setup_time: float  # Time taken to setup a new worker of this stage
    stage_batch_size: int = 1
    num_returns: int = 1

    def setup_ticks(self, sim_timestep: float) -> int:
        """Converts setup time into simulation time ticks."""
        assert _is_multiple(self.setup_time, sim_timestep)
        return int(self.setup_time / sim_timestep)

    def process_ticks(self, sim_timestep: float) -> int:
        """Converts processing time into simulation time ticks."""
        assert _is_multiple(self.process_time_per_worker, sim_timestep)
        return int(self.process_time_per_worker / sim_timestep)


@attrs.define
class SimulationProblem:
    """Defines the complete pipeline configuration for simulation.

    Contains the available cluster resources and all pipeline stages with their
    configurations.

    Attributes:
        cluster_resources: Available compute resources including CPUs, GPUs, etc.
        stages: List of stage configurations making up the pipeline
    """

    cluster_resources: resources.ClusterResources
    stages: list[SimulationStage]

    def to_autoscaling_problem(self) -> data_structures.Problem:
        """Converts simulation problem to format expected by scheduling algorithms."""
        return data_structures.Problem(self.cluster_resources, [x.problem_stage for x in self.stages])

    def validate(self) -> None:
        for stage in self.stages:
            stage.problem_stage.worker_shape.validate()


@attrs.define
class SimulationParams:
    """Parameters controlling simulation execution.

    Attributes:
        num_tasks: Total number of tasks to process through the pipeline
        timestep: Size of each simulation timestep in seconds
        rescale_interval_s: How often to run auto-scaling algorithm in seconds
        generate_plots: Whether to generate visualization plots of results
    """

    num_tasks: int = 1000
    timestep: float = 0.1
    rescale_interval_s: float = 60 * 3
    generate_plots: bool = False

    @property
    def rescale_interval_ticks(self) -> int:
        assert _is_multiple(self.rescale_interval_s, self.timestep)
        return int(self.rescale_interval_s / self.timestep)


@attrs.define
class StatsPerStage:
    """Statistics tracked for each pipeline stage."""

    stage_name: str
    total_tasks_completed: int = 0


@attrs.define
class SimulationStats:
    """Aggregate statistics for the entire simulation run."""

    stages: list[StatsPerStage]
    total_tasks_completed: int = 0
    total_processing_time: float = 0
    max_queue_size: int = 0


@attrs.define
class SimulationStateAtTime:
    """Complete pipeline state captured at a specific simulation time.

    Attributes:
        tick_num: Current simulation tick number
        time: Current simulation time in seconds
        state: Complete pipeline state
        active_allocation_fractions: Resource utilization of active workers
        assigned_allocation_fractions: Resource utilization including idle workers
    """

    tick_num: int
    time: float
    state: State
    active_allocation_fractions: resources.PoolOfResources
    assigned_allocation_fractions: resources.PoolOfResources


@attrs.define
class SimulationResult:
    """Result of a simulation.

    Returns a timeseries of the sim state as well as some stats.

    Can be used to make various plots.
    """

    states: list[SimulationStateAtTime]
    stats: SimulationStats

    def generate_plots(self) -> None:
        # Prepare data
        times = [state.time for state in self.states]
        num_stages = len(self.states[0].state.stages)

        # Calculate the number of rows needed
        num_rows = 10 + num_stages  # 10 original plots + separate backpressure plots for each stage

        # Create subplots
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(
                "Tasks Completed",
                "Queue Sizes",
                "Total Workers per Stage",
                "Idle Workers per Stage",
                "GPUs Used per Stage",
                "Idle GPUs per Stage",
                "Worker Counts",
                "Active Resource Utilization",
                "Assigned Resource Utilization",
                *[f"Backpressure Status: {stage.stage_name}" for stage in self.states[0].state.stages],
            ),
        )

        # Add traces for each plot
        self._add_tasks_completed_plot(fig, times)
        self._add_queue_sizes_plot(fig, times)
        self._add_total_workers_plot(fig, times)
        self._add_idle_workers_plot(fig, times)
        self._add_gpus_used_plot(fig, times)
        self._add_idle_gpus_plot(fig, times)
        self._add_worker_counts_plot(fig, times)
        self._add_active_resource_utilization_plot(fig, times)
        self._add_assigned_resource_utilization_plot(fig, times)
        self._add_backpressure_status_plot(fig, times)

        # Update layout
        fig.update_layout(
            height=300 * num_rows,  # Adjust the height based on the number of subplots
            title_text="Simulation Results Overview",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Save to file
        self._save_plot(fig)

    def _add_tasks_completed_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.stats.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[state.state.stages[i].total_tasks_completed for state in self.states],
                    name=f"{stage.stage_name} Tasks Completed",
                    mode="lines",
                ),
                row=1,
                col=1,
            )

    def _add_queue_sizes_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[state.state.stages[i].output_queue_size for state in self.states],
                    name=f"{stage.stage_name} Queue Size",
                    mode="lines",
                ),
                row=2,
                col=1,
            )

    def _add_total_workers_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[len(state.state.stages[i].workers) for state in self.states],
                    name=f"{stage.stage_name} Total Workers",
                    mode="lines",
                ),
                row=3,
                col=1,
            )

    def _add_idle_workers_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[
                        sum(
                            1
                            for worker in state.state.stages[i].workers.values()
                            if worker.current_task_start_ticks is None
                        )
                        for state in self.states
                    ],
                    name=f"{stage.stage_name} Idle Workers",
                    mode="lines",
                ),
                row=4,
                col=1,
            )

    def _add_gpus_used_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[
                        sum(
                            sum(gpu.fraction for gpu in worker.resources.gpus)
                            for worker in state.state.stages[i].workers.values()
                        )
                        for state in self.states
                    ],
                    name=f"{stage.stage_name} GPUs Used",
                    mode="lines",
                ),
                row=5,
                col=1,
            )

    def _add_idle_gpus_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[
                        sum(
                            sum(gpu.fraction for gpu in worker.resources.gpus)
                            for worker in state.state.stages[i].workers.values()
                            if worker.current_task_start_ticks is None
                        )
                        for state in self.states
                    ],
                    name=f"{stage.stage_name} Idle GPUs",
                    mode="lines",
                ),
                row=6,
                col=1,
            )

    def _add_worker_counts_plot(self, fig: go.Figure, times: list[float]) -> None:
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[len(state.state.stages[i].workers) for state in self.states],
                    name=f"{stage.stage_name} Worker Count",
                    mode="lines",
                ),
                row=7,
                col=1,
            )

    def _add_active_resource_utilization_plot(self, fig: go.Figure, times: list[float]) -> None:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.active_allocation_fractions.cpus * 100 for state in self.states],
                name="CPU allocation percentage.",
                mode="lines",
            ),
            row=8,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.active_allocation_fractions.gpus * 100 for state in self.states],
                name="GPU allocation percentage.",
                mode="lines",
            ),
            row=8,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.active_allocation_fractions.nvdecs * 100 for state in self.states],
                name="NVDEC allocation percentage.",
                mode="lines",
            ),
            row=8,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.active_allocation_fractions.nvencs * 100 for state in self.states],
                name="NVENC allocation percentage.",
                mode="lines",
            ),
            row=8,
            col=1,
        )
        fig.update_yaxes(range=[0, 100], row=10, col=1)

    def _add_assigned_resource_utilization_plot(self, fig: go.Figure, times: list[float]) -> None:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.assigned_allocation_fractions.cpus * 100 for state in self.states],
                name="CPU allocation percentage.",
                mode="lines",
            ),
            row=9,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.assigned_allocation_fractions.gpus * 100 for state in self.states],
                name="GPU allocation percentage.",
                mode="lines",
            ),
            row=9,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.assigned_allocation_fractions.nvdecs * 100 for state in self.states],
                name="NVDEC allocation percentage.",
                mode="lines",
            ),
            row=9,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[state.assigned_allocation_fractions.nvencs * 100 for state in self.states],
                name="NVENC allocation percentage.",
                mode="lines",
            ),
            row=9,
            col=1,
        )
        fig.update_yaxes(range=[0, 100], row=11, col=1)

    def _add_backpressure_status_plot(self, fig: go.Figure, times: list[float]) -> None:
        backpressure_numbers: list[list[float]] = [x.state.make_backpressure_vals() for x in self.states]
        for i, stage in enumerate(self.states[0].state.stages):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=[x[i] for x in backpressure_numbers],
                    name=f"{stage.stage_name} Backpressure",
                    mode="lines",
                ),
                row=10 + i,  # Start from row 9 and add a new row for each stage
                col=1,
            )
            # Update y-axis range for each backpressure subplot
            fig.update_yaxes(range=[-1.1, 1.1], title_text="Backpressure", row=9 + i, col=1)

        # Add a horizontal line at y=0 for each backpressure subplot
        for i in range(len(self.states[0].state.stages)):
            fig.add_shape(
                type="line",
                x0=min(times),
                y0=0,
                x1=max(times),
                y1=0,
                line=dict(color="red", width=1, dash="dash"),
                row=10 + i,
                col=1,
            )

    def _save_plot(self, fig: go.Figure) -> None:
        # Create directory if it doesn't exist
        output_dir = os.path.expanduser("~/yotta_sim_results/")
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{timestamp}.html"
        filepath = os.path.join(output_dir, filename)

        # Save figure to file
        fig.write_html(filepath)
        print(f"Simulation results saved to: {filepath}")


@attrs.define
class WorkerState:
    """State of an individual worker in the simulation.

    Attributes:
        id: Unique worker identifier
        resources: Allocated compute resources
        current_task_start_ticks: When current task started, None if idle
        setup_start_ticks: When setup started, None if ready
    """

    id: str
    resources: resources.WorkerResources
    current_task_start_ticks: int | None = None
    setup_start_ticks: int | None = None

    def to_problem_state(self) -> data_structures.ProblemWorkerState:
        return data_structures.ProblemWorkerState(self.id, self.resources)

    @property
    def has_a_task(self) -> bool:
        return self.current_task_start_ticks is not None

    @property
    def is_being_set_up(self) -> bool:
        return self.setup_start_ticks is not None


@attrs.define
class StageState:
    """Complete state of a pipeline stage.

    Tracks all workers, queues, and throughput metrics for a stage.

    Attributes:
        stage_name: Name identifying the stage
        workers: Map of worker IDs to worker states
        output_queue_size: Number of completed tasks waiting to be processed by next stage
        total_tasks_completed: Running count of completed tasks
        slots_per_worker: Maximum concurrent tasks per worker
    """

    stage_name: str
    workers: dict[str, WorkerState] = attrs.field(factory=dict)
    output_queue_size: int = 0
    total_tasks_completed: int = 0
    slots_per_worker: int = 2

    def calc_active_resources(self) -> resources.PoolOfResources:
        out = resources.PoolOfResources(0.0, 0.0, 0.0, 0.0)
        for worker in self.workers.values():
            if worker.current_task_start_ticks is not None:
                out += worker.resources.to_pool()
        return out

    def calc_assigned_but_idle_resources(self) -> resources.PoolOfResources:
        out = resources.PoolOfResources(0.0, 0.0, 0.0, 0.0)
        for worker in self.workers.values():
            if worker.current_task_start_ticks is None:
                out += worker.resources.to_pool()
        return out

    def make_problem_state(self) -> data_structures.ProblemStageState:
        return data_structures.ProblemStageState(
            self.stage_name,
            [x.to_problem_state() for x in self.workers.values()],
            slots_per_worker=self.slots_per_worker,
            is_finished=False,
        )


@attrs.define
class State:
    """Complete state of the pipeline simulation.

    Tracks remaining input tasks and state of all pipeline stages.

    Attributes:
        num_input_tasks_remaining: Number of tasks not yet started
        stages: State of each pipeline stage
    """

    num_input_tasks_remaining: int
    stages: list[StageState]

    def make_problem_state(self) -> data_structures.ProblemState:
        return data_structures.ProblemState([x.make_problem_state() for x in self.stages])

    def calc_active_utilization(self) -> resources.PoolOfResources:
        out = resources.PoolOfResources(0.0, 0.0, 0.0, 0.0)
        for stage in self.stages:
            out += stage.calc_active_resources()
        return out

    def calc_assigned_utilization(self) -> resources.PoolOfResources:
        out = resources.PoolOfResources(0.0, 0.0, 0.0, 0.0)
        for stage in self.stages:
            out += stage.calc_active_resources()
            out += stage.calc_assigned_but_idle_resources()
        return out

    def calc_active_utilization_fractions(self, cluster: resources.ClusterResources) -> resources.PoolOfResources:
        totals = cluster.totals()
        return self.calc_active_utilization() / totals

    def calc_assigned_utilization_fractions(self, cluster: resources.ClusterResources) -> resources.PoolOfResources:
        totals = cluster.totals()
        return self.calc_assigned_utilization() / totals

    def make_backpressure_vals(self) -> list[float]:
        """Calculates backpressure status for each stage.

        Returns normalized values between -1 and 1 where:
            1.0 = Stage is backpressured (output queue full)
            0.0 = Stage is balanced
           -1.0 = Stage is starved (no input tasks)
        """
        out = []
        for idx, stage in enumerate(self.stages):
            if idx == 0:
                input_num = self.num_input_tasks_remaining
            else:
                input_num = self.stages[idx - 1].output_queue_size

            num_active_workers = len([x for x in stage.workers.values() if x.has_a_task])
            max_in_progress_or_complete = len(stage.workers) * stage.slots_per_worker
            tasks_in_process_or_queued = stage.output_queue_size + num_active_workers
            if tasks_in_process_or_queued >= max_in_progress_or_complete:
                # Backpressured
                out.append(1.0)
            elif input_num <= 0:
                # Starved
                out.append(-1.0)
            else:
                # TODO: Should this be number of active workers instead??
                # Somewhere in between
                zero_to_1_val = tasks_in_process_or_queued / max_in_progress_or_complete
                # translate it to -1 to 1
                out.append(zero_to_1_val * 2 - 1)
        return out


class Simulator:
    """Main simulation engine for evaluating pipeline auto-scaling.

    The simulator processes tasks through the pipeline stages while periodically
    running the auto-scaling algorithm to adjust worker counts. It tracks detailed
    metrics and can generate visualizations of the results.

    Args:
        algorithm: Auto-scaling algorithm to evaluate
        problem: Pipeline configuration to simulate
        params: Parameters controlling the simulation

    Example:
        >>> simulator = Simulator(algorithm=my_scaling_algorithm, problem=pipeline_config, params=simulation_params)
        >>> results = simulator.run()
    """

    def __init__(
        self,
        algorithm: data_structures.AutoScalingAlgorithmInterface,
        problem: SimulationProblem,
        params: SimulationParams,
    ):
        problem.validate()
        self._algorithm = algorithm
        self._params = params
        self._problem = problem
        self._current_ticks = 0
        self._worker_allocator = allocator.WorkerAllocator(problem.cluster_resources, [])
        self._state = State(
            num_input_tasks_remaining=self._params.num_tasks,
            stages=[StageState(x.problem_stage.name) for x in self._problem.stages],
        )
        self._results = SimulationResult(
            [], SimulationStats([StatsPerStage(x.problem_stage.name) for x in self._problem.stages])
        )

    @property
    def current_time_s(self) -> float:
        return self._current_ticks * self._params.timestep

    def _record_state(self) -> None:
        self._results.states.append(
            SimulationStateAtTime(
                self._current_ticks,
                self.current_time_s,
                copy.deepcopy(self._state),
                self._state.calc_active_utilization_fractions(self._problem.cluster_resources),
                self._state.calc_assigned_utilization_fractions(self._problem.cluster_resources),
            )
        )

    def run(self) -> SimulationResult:
        """Executes the simulation until completion.

        Runs until all tasks are processed or timeout occurs. Each timestep:
        1. Records current state
        2. Processes tasks through pipeline stages
        3. Updates measurements for auto-scaling algorithm
        4. Periodically runs auto-scaling algorithm

        Returns:
            SimulationResult: Complete simulation results and metrics

        Raises:
            ValueError: If simulation exceeds maximum runtime
        """
        start_time = time.time()
        self._algorithm.setup(self._problem.to_autoscaling_problem())
        while self._results.stats.total_tasks_completed < self._params.num_tasks:
            self._record_state()
            measurements = self._process_tasks()
            self._algorithm.update_with_measurements(self.current_time_s, measurements)
            # Run auto-scaling on configured interval
            if self._current_ticks % self._params.rescale_interval_ticks == 0:
                state = self._state.make_problem_state()
                if _VERBOSE:
                    logger.info("Running autoscaling...")
                result = self._algorithm.autoscale(self.current_time_s, state)
                state_and_result = data_structures.ProblemStateAndSolution(state, result)
                if _VERBOSE:
                    logger.info(f"Got the following results:\n{state_and_result}")
                if len(result.stages) != len(self._state.stages):
                    raise ValueError(
                        f"Expected the returned number of stages to equal {len(self._state.stages)}, "
                        f"but got {len(result.stages)} instead."
                    )
                self._apply_rescale_result(result)
                self._print_summary()

            if time.time() - start_time > max_runtime:
                raise ValueError("Simulation timed out")

            self._current_ticks += 1
        self._print_summary()
        if self._params.generate_plots:
            self._results.generate_plots()
        return self._results

    def _process_tasks(self) -> data_structures.Measurements:
        """Processes tasks through pipeline stages for one timestep.

        For each stage:
        1. Completes setup of any pending workers
        2. Completes any finished tasks and updates queues
        3. Assigns new tasks to idle workers if available and not backpressured

        Returns:
            Measurements: Processing timing measurements for this timestep
        """
        measurements = data_structures.Measurements(
            self.current_time_s, [data_structures.StageMeasurements() for _ in self._problem.stages]
        )
        num_stages = len(self._problem.stages)
        for stage_index, (stage_state, stage_problem, stage_stats) in enumerate(
            zip(self._state.stages, self._problem.stages, self._results.stats.stages)
        ):
            if stage_index == 0:
                maybe_previous_stage = None
            else:
                maybe_previous_stage = self._state.stages[stage_index - 1]

            # Complete worker setup if ready
            for worker in stage_state.workers.values():
                # Worker is in setup stage and has been setting up for long enough.
                if worker.setup_start_ticks is not None and (
                    self._current_ticks >= worker.setup_start_ticks + stage_problem.setup_ticks(self._params.timestep)
                ):
                    # Move the worker out of setup mode
                    worker.setup_start_ticks = None

            # Complete finished tasks and update queues
            for worker in stage_state.workers.values():
                # Skip workers still in setup
                if worker.setup_start_ticks is not None:
                    continue
                # Check if current task is complete
                if (
                    worker.current_task_start_ticks is not None
                    and self._current_ticks
                    >= worker.current_task_start_ticks + stage_problem.process_ticks(self._params.timestep)
                ):
                    # Update task completion stats
                    if stage_index == num_stages - 1:
                        self._results.stats.total_tasks_completed += 1
                    else:
                        stage_state.output_queue_size += 1
                    # Record timing measurement
                    measurements.stages[stage_index].task_measurements.append(
                        data_structures.TaskMeasurement(
                            worker.current_task_start_ticks * self._params.timestep,
                            self.current_time_s,
                            num_returns=stage_problem.num_returns,
                        )
                    )
                    worker.current_task_start_ticks = None
                    stage_stats.total_tasks_completed += 1
                    stage_state.total_tasks_completed += 1

            # Calculate backpressure status
            num_active_workers = len(
                [worker for worker in stage_state.workers.values() if worker.current_task_start_ticks is not None]
            )
            num_queued_or_in_progress = num_active_workers + stage_state.output_queue_size
            max_tasks_to_assign = max(
                0, len(stage_state.workers) * stage_state.slots_per_worker - num_queued_or_in_progress
            )

            assigned_tasks = 0
            # Assign tasks if worker is idle and not backpressured.
            for worker in stage_state.workers.values():
                # We have assigned too many tasks. Abort.
                if assigned_tasks >= max_tasks_to_assign:
                    break
                # The worker is in setup mode, so cannot accept tasks
                if worker.setup_start_ticks is not None:
                    continue
                # The worker already has a task. Cannot assign.
                if worker.current_task_start_ticks is not None:
                    continue
                # Assign a task if there is one available.
                if stage_index == 0 and self._state.num_input_tasks_remaining > 0:
                    self._state.num_input_tasks_remaining -= 1
                    worker.current_task_start_ticks = self._current_ticks
                    assigned_tasks += 1
                elif stage_index > 0 and maybe_previous_stage.output_queue_size:
                    maybe_previous_stage.output_queue_size -= 1
                    worker.current_task_start_ticks = self._current_ticks
                    assigned_tasks += 1
        # for backpressure_status, out in zip(self._state.make_backpressure_vals(), measurements.stages, strict=False):
        #     out.backpressure_status = backpressure_status
        return measurements

    def _apply_rescale_result(self, result: data_structures.Solution) -> None:
        """Applies auto-scaling changes to the pipeline state.

        Handles both adding new workers and removing existing ones while ensuring:
        1. In-progress tasks are properly reassigned
        2. Resource allocations are updated correctly
        3. Workers are moved through setup phase as needed

        Args:
            result: Auto-scaling algorithm's worker adjustment decisions
        """
        # First handle worker removals for each stage
        for stage_index, (stage_result, stage_state) in enumerate(zip(result.stages, self._state.stages)):
            # Update slots per worker
            assert stage_result.slots_per_worker >= stage_state.slots_per_worker
            stage_state.slots_per_worker = stage_result.slots_per_worker

            # Process worker deletions
            worker_ids_to_delete = [x.id for x in stage_result.deleted_workers]
            self._worker_allocator.delete_workers(worker_ids_to_delete)

            for id_ in worker_ids_to_delete:
                worker = stage_state.workers[id_]
                # Return in-progress task to queue
                if worker.current_task_start_ticks is not None:
                    if stage_index == 0:
                        self._state.num_input_tasks_remaining += 1
                    else:
                        self._state.stages[stage_index - 1].output_queue_size += 1
                stage_state.workers.pop(id_)

        # Then handle worker additions for each stage
        for _, (stage_result, stage_state) in enumerate(zip(result.stages, self._state.stages)):
            # Add new workers to allocator
            self._worker_allocator.add_workers(
                [resources.Worker(x.id, stage_state.stage_name, x.resources) for x in stage_result.new_workers]
            )

            # Add workers to stage and start setup phase
            for worker in stage_result.new_workers:
                stage_state.workers[worker.id] = WorkerState(
                    worker.id, worker.resources, setup_start_ticks=self._current_ticks
                )

    def _print_summary(self) -> None:
        """Prints detailed current state of the pipeline simulation.

        Displays tables showing:
        1. Per-stage metrics (workers, queues, throughput)
        2. Resource utilization
        3. Overall pipeline performance
        """
        headers = [
            "Stage",
            "Total\nWorkers",
            "Pending\nWorkers",
            "Ready\nWorkers",
            "Active\nWorkers",
            "Idle\nWorkers",
            "Queue\nSize",
            "Completed\nTasks",
            "Per-Worker\nThroughput",
            "Stage\nThroughput",
            "CPUs\nUsed",
            "GPUs\nUsed",
            "NVDECs\nUsed",
            "NVENCs\nUsed",
            "Backpressure\nStatus",
        ]
        table_data = []

        backpressure_status = self._state.make_backpressure_vals()
        for i, (stage, stage_stats, stage_problem) in enumerate(
            zip(self._state.stages, self._results.stats.stages, self._problem.stages)  # py310: strict=False
        ):
            total_workers = len(stage.workers)
            pending_workers = sum(1 for worker in stage.workers.values() if worker.setup_start_ticks is not None)
            ready_workers = total_workers - pending_workers
            active_workers = sum(1 for worker in stage.workers.values() if worker.current_task_start_ticks is not None)
            idle_workers = ready_workers - active_workers
            queue_size = stage.output_queue_size
            per_worker_throughput = 1.0 / (stage_problem.process_time_per_worker)
            if ready_workers:
                stage_throughput = per_worker_throughput * ready_workers
            else:
                stage_throughput = 0.0

            # Calculate resource usage
            cpus_used = sum(worker.resources.cpus for worker in stage.workers.values())
            gpus_used = sum(sum(gpu.fraction for gpu in worker.resources.gpus) for worker in stage.workers.values())
            nvdecs_used = sum(sum(1.0 for nvdec in worker.resources.nvdecs) for worker in stage.workers.values())
            nvencs_used = sum(sum(1.0 for nvenc in worker.resources.nvencs) for worker in stage.workers.values())

            table_data.append(
                [
                    stage_problem.problem_stage.name,
                    total_workers,
                    pending_workers,
                    ready_workers,
                    active_workers,
                    idle_workers,
                    queue_size,
                    stage_stats.total_tasks_completed,
                    f"{per_worker_throughput:.2f} task/s",
                    f"{stage_throughput:.2f} task/s",
                    f"{cpus_used:.2f}",
                    f"{gpus_used:.2f}",
                    f"{nvdecs_used:.2f}",
                    f"{nvencs_used:.2f}",
                    f"{backpressure_status[i]:.2f}",
                ]
            )

        table = tabulate.tabulate(table_data, headers=headers, tablefmt="grid")
        logger.info("-------------------------------------------------------------")
        logger.info(f"Simulation Summary at time {self.current_time_s:.2f}:\n{table}")
        logger.info(f"Total tasks completed: {self._results.stats.total_tasks_completed}")
        logger.info(f"Total processing time: {self._results.stats.total_processing_time:.2f}")
        logger.info(f"Max queue size: {self._results.stats.max_queue_size}")
        if self.current_time_s > 0:
            throughput = self._results.stats.total_tasks_completed / self.current_time_s
        else:
            throughput = 0.0
        logger.info(f"Average throughput: {throughput} tasks/s")
        logger.info(f"Detailed allocations:\n{self._worker_allocator.make_detailed_utilization_table()}")
