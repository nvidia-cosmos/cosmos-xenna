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

"""Data structures used by autoscaling algorithms and simulations.

This module presents an interface for autoscaling algorithms. This interface formulates the autoscaling information as
a "Problem" and "Solution". It provides data structures for representing resource allocation problems and their
solutions in a distributed computing environment.
"""

from __future__ import annotations

import abc

import attrs
import tabulate

from cosmos_xenna.pipelines.private import specs
from cosmos_xenna.ray_utils import resources, stage_worker


@attrs.define
class ProblemStage:
    """Represents a single stage in the allocation problem.

    A stage represents a discrete step in the processing pipeline that requires
    specific resource allocations.

    Attributes:
        name: A unique identifier for the stage.
        worker_shape: Resource requirements for each worker in this stage.
        requested_num_workers: Optional explicitly requested number of workers.
            If specified, this is the exact number of workers requested for the stage.
            If None, the number of workers will be determined by the autoscaling algorithm.
    """

    name: str
    stage_batch_size: int
    worker_shape: resources.WorkerShape
    requested_num_workers: int | None = None


@attrs.define
class ProblemWorkerState:
    """Represents the state of a worker in the system.

    Attributes:
        id: Unique identifier for the worker.
        resources: Current resource allocation for this worker.
    """

    id: str
    resources: resources.WorkerResources

    @classmethod
    def make_from_worker_state(cls, state: resources.Worker) -> ProblemWorkerState:
        """Creates a ProblemWorkerState from a Worker instance.

        Args:
            state: Worker instance containing worker state information.

        Returns:
            A new ProblemWorkerState instance.
        """
        return ProblemWorkerState(state.id, state.allocation)

    def to_worker(self, stage_name: str) -> resources.Worker:
        """Converts this state to a Worker instance.

        Args:
            stage_name: Name of the stage this worker belongs to.

        Returns:
            A Worker instance representing this state.
        """
        return resources.Worker(self.id, stage_name, self.resources)


@attrs.define
class ProblemStageState:
    """Represents the current state of a stage including its workers.

    Attributes:
        stage_name: Name identifier for this stage.
        workers: List of workers currently assigned to this stage.
        slots_per_worker: Number of task slots available per worker.
        is_finished: Boolean indicating if this stage has completed processing.
    """

    stage_name: str
    workers: list[ProblemWorkerState]
    slots_per_worker: int
    is_finished: bool

    @property
    def num_workers(self) -> int:
        """Returns the current number of workers in this stage."""
        return len(self.workers)


@attrs.define
class ProblemState:
    """Represents the complete current state of the allocation problem.

    Provides a snapshot of all stages and their current resource allocations.

    Attributes:
        stages: List of all stage states in the system.
    """

    stages: list[ProblemStageState]

    def __str__(self) -> str:
        """Returns a formatted string representation of the problem state.

        Returns:
            A string containing a tabulated view of all stages and their resource allocations.
        """
        table_data = []
        headers = ["Stage", "Worker ID", "Node", "CPUs", "GPUs", "NVDECs", "NVENCs"]

        for stage_idx, stage in enumerate(self.stages):
            for worker in stage.workers:
                gpu_alloc = ", ".join([f"{g.gpu_index}:{g.fraction:.2f}" for g in worker.resources.gpus])
                nvdec_alloc = ", ".join([f"{n.gpu_index}.{n.codec_index}:{1.0:.2f}" for n in worker.resources.nvdecs])
                nvenc_alloc = ", ".join([f"{n.gpu_index}.{n.codec_index}:{1.0:.2f}" for n in worker.resources.nvencs])

                table_data.append(
                    [
                        f"Stage {stage_idx}",
                        worker.id,
                        worker.resources.node,
                        f"{worker.resources.cpus:.2f}",
                        gpu_alloc,
                        nvdec_alloc,
                        nvenc_alloc,
                    ]
                )

            if stage_idx < len(self.stages) - 1:
                table_data.append([""] * len(headers))

        return tabulate.tabulate(table_data, headers=headers, tablefmt="grid")


@attrs.define
class Problem:
    """Represents the complete allocation problem to be solved.

    This class encapsulates all information needed to solve the resource
    allocation problem, including cluster resources and stage definitions.

    Attributes:
        cluster_resources: Total available resources in the cluster.
        stages: List of all stages that need resource allocation.
    """

    cluster_resources: resources.ClusterResources
    stages: list[ProblemStage]

    @classmethod
    def make_from_pipeline_spec(
        cls,
        s: specs.PipelineSpec,
        cluster_resources: resources.ClusterResources,
    ) -> Problem:
        """Creates a Problem instance from a pipeline specification.

        Args:
            s: Pipeline specification containing stage information.
            cluster_resources: Available cluster resources.

        Returns:
            A new Problem instance configured according to the specification.
        """
        out = Problem(cluster_resources, [])
        num_nodes = len(cluster_resources.nodes)
        for idx, stage in enumerate(s.stages):
            assert isinstance(stage, specs.StageSpec)
            if stage.num_workers is not None:
                num_workers = stage.num_workers
            elif stage.num_workers_per_node is not None:
                num_workers = stage.num_workers_per_node * num_nodes
            else:
                num_workers = None
            out.stages.append(
                ProblemStage(
                    stage.name(idx),
                    stage.stage.stage_batch_size,
                    stage.stage.required_resources.to_shape(),
                    requested_num_workers=num_workers,
                )
            )
        return out


@attrs.define
class StageSolution:
    """Represents the allocation result for a single stage.

    Contains information about resource allocation changes for a specific stage.

    Attributes:
        slots_per_worker: Number of task slots to allocate per worker.
        new_workers: List of workers to be added to the stage.
        deleted_workers: List of workers to be removed from the stage.
    """

    slots_per_worker: int
    new_workers: list[ProblemWorkerState] = attrs.field(factory=list)
    deleted_workers: list[ProblemWorkerState] = attrs.field(factory=list)


@attrs.define
class Solution:
    """Represents the complete result of the allocation problem.

    Contains the complete set of changes to be applied to the system.

    Attributes:
        stages: List of solutions for each stage in the system.
    """

    stages: list[StageSolution] = attrs.field(factory=list)

    @property
    def num_new_workers_per_stage(self) -> list[int]:
        return [len(x.new_workers) for x in self.stages]

    @property
    def num_deleted_workers_per_stage(self) -> list[int]:
        return [len(x.deleted_workers) for x in self.stages]

    def __str__(self) -> str:
        """Returns a formatted string representation of the solution.

        Returns:
            A string containing a tabulated view of all resource allocation changes.
        """
        table_data = []
        headers = ["Stage", "Action", "Worker ID", "Node", "CPUs", "GPUs", "NVDECs", "NVENCs"]

        for stage_idx, stage_result in enumerate(self.stages):
            new_rows = []
            for worker in stage_result.new_workers:
                gpu_alloc = ", ".join([f"{g.gpu_index}:{g.fraction:.2f}" for g in worker.resources.gpus])
                nvdec_alloc = ", ".join([f"{n.gpu_index}.{n.codec_index}:{1.0:.2f}" for n in worker.resources.nvdecs])
                nvenc_alloc = ", ".join([f"{n.gpu_index}.{n.codec_index}:{1.0:.2f}" for n in worker.resources.nvencs])

                new_rows.append(
                    [
                        f"Stage {stage_idx}",
                        "New",
                        worker.id,
                        worker.resources.node,
                        f"{worker.resources.cpus:.2f}",
                        gpu_alloc,
                        nvdec_alloc,
                        nvenc_alloc,
                    ]
                )

            for worker in stage_result.deleted_workers:
                gpu_alloc = ", ".join([f"{g.gpu_index}:{g.fraction:.2f}" for g in worker.resources.gpus])
                nvdec_alloc = ", ".join([f"{n.gpu_index}.{n.codec_index}:{1.0:.2f}" for n in worker.resources.nvdecs])
                nvenc_alloc = ", ".join([f"{n.gpu_index}.{n.codec_index}:{1.0:.2f}" for n in worker.resources.nvencs])

                new_rows.append(
                    [
                        f"Stage {stage_idx}",
                        "Deleted",
                        worker.id,
                        worker.resources.node,
                        f"{worker.resources.cpus:.2f}",
                        gpu_alloc,
                        nvdec_alloc,
                        nvenc_alloc,
                    ]
                )

            if new_rows:
                table_data.extend(new_rows)
                if stage_idx < len(self.stages) - 1:
                    table_data.append([""] * len(headers))

        if not table_data:
            return "No changes in allocation"

        return tabulate.tabulate(table_data, headers=headers, tablefmt="grid")


@attrs.define
class ProblemStateAndSolution:
    """Represents both the current state and solution of the allocation problem.

    This class combines both the current state of the system and the proposed
    changes, allowing for complete context when reviewing allocation decisions.

    Attributes:
        state: Current state of the system.
        result: Proposed changes to the system.
    """

    state: ProblemState
    result: Solution

    def __str__(self) -> str:
        """Returns a formatted string representation of both state and solution.

        Returns:
            A string containing both the current state and proposed changes.
        """
        out_lines = []
        out_lines.append("Problem State and Result:")
        out_lines.append("State:")
        out_lines.append(str(self.state))
        out_lines.append("Result:")
        out_lines.append(str(self.result))
        return "\n".join(out_lines)


@attrs.define
class TaskMeasurement:
    """Contains timing measurements for a single task.

    Attributes:
        start_time: Time when the task started processing.
        end_time: Time when the task completed processing.
    """

    start_time: float
    end_time: float
    num_returns: int

    @classmethod
    def make_from_task_metadata(cls, c: stage_worker.TaskResultMetadata) -> TaskMeasurement:
        """Creates a TaskMeasurement from task metadata.

        Args:
            c: Task result metadata containing timing information.

        Returns:
            A new TaskMeasurement instance.
        """
        return TaskMeasurement(c.timing.process_start_time_s, c.timing.process_end_time_s, c.num_returns)

    @property
    def duration(self) -> float:
        """Calculates the duration of the task.

        Returns:
            The duration of the task in seconds.
        """
        return self.end_time - self.start_time


@attrs.define
class StageMeasurements:
    """Contains measurements for a single stage.

    Attributes:
        task_measurements: List of measurements for individual tasks in this stage.
    """

    task_measurements: list[TaskMeasurement] = attrs.field(factory=list)


@attrs.define
class Measurements:
    """Contains measurements across multiple stages.

    These measurements can be used by the auto-scaling algorithm to estimate
    the processing rate of the stages.

    Attributes:
        time: Timestamp when these measurements were taken.
        stages: List of measurements for each stage.
    """

    time: float
    stages: list[StageMeasurements]


class AutoScalingAlgorithmInterface(abc.ABC):
    """Abstract base class for autoscaling algorithms.

    This interface defines the required methods that any autoscaling
    algorithm must implement.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the autoscaling algorithm.

        Returns:
            String identifier for the algorithm.
        """
        pass

    @abc.abstractmethod
    def setup(self, problem: Problem) -> None:
        """Initializes the algorithm with the problem definition.

        Args:
            problem: The resource allocation problem to be solved.
        """
        pass

    @abc.abstractmethod
    def update_with_measurements(self, time: float, measurements: Measurements) -> None:
        """Updates the algorithm's state with new measurements.

        Args:
            time: Current timestamp.
            measurements: New measurements to consider.
        """
        pass

    @abc.abstractmethod
    def autoscale(self, current_time: float, state: ProblemState) -> Solution:
        """Computes a new resource allocation solution.

        Args:
            current_time: Current timestamp.
            state: Current state of the system.

        Returns:
            A Solution instance containing the proposed changes.
        """
        pass
