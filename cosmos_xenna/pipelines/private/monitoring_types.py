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

from __future__ import annotations

import enum

import attrs
from tabulate import tabulate

from cosmos_xenna.pipelines.private import allocator
from cosmos_xenna.ray_utils import monitoring, stage_worker


@attrs.define
class PipelineStats:
    """Class for storing and displaying various stats for a pipeline.

    This is used to report the status of the pipeline.
    """

    start_time: float
    time: float
    num_initial_input_tasks: int
    num_input_tasks_remaining: int
    num_outputs: int
    actor_pools: list[monitoring.ActorPoolStats]
    pipeline_duration_s: float
    main_loop_rate_hz: float
    cluster: RayClusterInfo
    # TODO: This should probably be merged with "actors_pools".
    resource_usage_per_stage: dict[str, RayStageResourceUsage]
    # TODO: This should probably be merged with "actors_pools".
    extra_data_per_stage: dict[str, list[stage_worker.TaskResultMetadata]]

    @property
    def inputs_processed_per_second(self) -> float:
        denominator = self.time - self.start_time
        numerator = self.num_initial_input_tasks - self.num_input_tasks_remaining
        if denominator <= 0.0:
            return 0.0
        else:
            return numerator / denominator

    @property
    def outputs_per_second(self) -> float:
        denominator = self.time - self.start_time
        numerator = self.num_outputs
        if denominator <= 0.0:
            return 0.0
        else:
            return numerator / denominator

    def make_backpressure_statuses(self) -> list[BackpressureStatus]:
        out = []
        for idx, pool in enumerate(self.actor_pools):
            if idx == 0:
                input_num = self.num_input_tasks_remaining
            else:
                input_num = self.actor_pools[idx - 1].task_stats.output_queue_size

            max_in_progress_or_complete = pool.slot_stats.num_empty + pool.slot_stats.num_used
            in_process_or_complete = (
                pool.slot_stats.num_used + pool.task_stats.input_queue_size + pool.task_stats.output_queue_size
            )
            if in_process_or_complete >= max_in_progress_or_complete:
                # TODO: This is bad. I should be looking at how much is in progress vs completed instead.
                # High in progress is healthy. High sitting completed means backpressured
                out.append(BackpressureStatus.TOO_MANY_OUTPUTS)
            elif input_num <= 0:
                out.append(BackpressureStatus.NOT_ENOUGH_INPUTS)
            else:
                out.append(BackpressureStatus.NORMAL)
        return out

    def display(self) -> str:
        stage_names = [x.name for x in self.actor_pools]
        out_lines = []
        out_lines.append("Pipeline Stats:")
        out_lines.append(f"Pipeline duration: {self.pipeline_duration_s / 60} minutes")
        out_lines.append(f"Number of initial input samples: {self.num_initial_input_tasks}")
        out_lines.append(f"Number of input samples remaining: {self.num_input_tasks_remaining}")
        out_lines.append(f"Streaming pipeline main loop rate: {self.main_loop_rate_hz}")

        # Add resource information table
        resource_headers = ["Resource", "Total", "Available"]
        resource_data = [
            ["CPUs", self.cluster.total.num_cpus, self.cluster.available.num_cpus],
            ["GPUs", self.cluster.total.num_gpus, self.cluster.available.num_gpus],
            ["Memory (GB)", self.cluster.total.memory / 1e9, self.cluster.available.memory / 1e9],
            [
                "Object Store Memory (GB)",
                self.cluster.total.object_store_memory / 1e9,
                self.cluster.available.object_store_memory / 1e9,
            ],
        ]
        out_lines.append("\nCluster Resources:")
        out_lines.append(tabulate(resource_data, headers=resource_headers, tablefmt="fancy_grid"))

        # Add resource usage by stage table
        usage_headers = ["Stage", "CPU %", "Memory (GB)", "Actor Count", "CPU % per worker", "Memory (GB) per worker"]
        usage_data = []
        for stage_name in stage_names:
            if stage_name in self.resource_usage_per_stage:
                usage = self.resource_usage_per_stage[stage_name]
                usage_data.append(
                    [
                        stage_name,
                        f"{usage.cpu_utilization:.2f}",
                        f"{usage.memory_usage / 1e9:.2f}",
                        usage.actor_count,
                        f"{usage.cpu_utilization / usage.actor_count:.2f}",
                        f"{usage.memory_usage / usage.actor_count / 1e9:.2f}",
                    ]
                )
            else:
                usage_data.append(
                    [
                        stage_name,
                        f"{0.0:.2f}",
                        f"{0.0 / 1e9:.2f}",
                        0,
                        f"{0.0:.2f}",
                        f"{0.0:.2f}",
                    ]
                )
        out_lines.append("\nResource Usage by Stage:")
        out_lines.append(tabulate(usage_data, headers=usage_headers, tablefmt="fancy_grid"))
        out_lines.append("\nStage state:")
        headers = [
            "Stage",
            "Actors:\nTarget",
            "Actors:\nPending",
            "Actors:\nReady",
            "Actors:\nRunning",
            "Actors:\nIdle",
            "Tasks:\nCompleted",
            "Tasks:\nReturned None",
            "Queue:\nInput Size",
            "Queue:\nOutput Size",
            "Slots:\nNum Used",
            "Slots:\nNum Empty",
            "Speed:\nTasks/actor/s",
        ]
        data = []
        for pool in self.actor_pools:
            row: list = [pool.name]
            row.extend(
                [
                    pool.actor_stats.target,
                    pool.actor_stats.pending,
                    pool.actor_stats.ready,
                    pool.actor_stats.running,
                    pool.actor_stats.idle,
                ]
            )
            row.extend(
                [
                    pool.task_stats.total_completed,
                    pool.task_stats.total_returned_none,
                    pool.task_stats.input_queue_size,
                    pool.task_stats.output_queue_size,
                ]
            )
            row.extend([pool.slot_stats.num_used, pool.slot_stats.num_empty, pool.processing_speed_tasks_per_second])
            data.append(row)

        out_lines.append(tabulate(data, headers=headers, tablefmt="fancy_grid"))
        return "\n".join(out_lines)


@attrs.define
class StatsToSend:
    worker_allocator: allocator.WorkerAllocator
    pipeline_stats: PipelineStats


@attrs.define
class ActorInfo:
    """Actor info derived from ray.util.state.list_actors."""

    actor_id: str = ""
    name: str | None = None
    namespace: str = ""
    runtime_env: dict | None = None
    job_id: str = ""
    pid: int | None = -1
    ip_address: str | None = ""
    port: int | None = -1
    state: str = ""
    class_name: str | None = None
    function_descriptor: dict | None = attrs.field(factory=dict)
    resources: dict = attrs.field(factory=dict)
    actor_type: str = ""
    current_task_desc: str | None = None
    num_restarts: int = 0
    timestamp: float = 0.0
    node_id: str = ""
    repr_name: str = ""


@attrs.define
class ActorResourceUsage:
    cpu_utilization: float
    memory_usage: float


@attrs.define
class ActorResourceUsageAndMetadata:
    metadata: ActorInfo
    resource_usage: ActorResourceUsage


@attrs.define
class RayStageResourceUsage:
    """Resource usage for a single stage."""

    pool_name: str
    cpu_utilization: float
    memory_usage: float
    actor_count: int


class BackpressureStatus(enum.Enum):
    """Simple enum used to figure out which stage is throttling a pipeline."""

    # The stage does not have any inputs, so it can't run
    NOT_ENOUGH_INPUTS = 0
    # The stage has enough inputs to run.
    NORMAL = 1
    # The stage cannot run because it has too many outputs queued up
    TOO_MANY_OUTPUTS = 2


@attrs.define
class Resources:
    num_cpus: float
    num_gpus: float
    memory: float
    object_store_memory: float


@attrs.define
class RayClusterInfo:
    total: Resources
    available: Resources
    actors: list[ActorInfo]
