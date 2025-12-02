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

"""Utitilities for monitoring our Ray jobs.

Things which still need to be tracked:

# Node level
- Disk
- Thread count
- Thread limit

# Stage level
- Average GPU utilization
- Total GPU utilization

# Task level
- Average serialize duration
- Average push duration
- Average output size
- per-task pipeline specific stuff
"""

# TODO: This module is a bit disjoint. Lots of stuff happening in a lot of different classes. This
# should be consididated into fewer classes and properly unit-tested.
from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any, List

import attrs
import ray
import ray.util.scheduling_strategies
import ray.util.state
from ray.util.metrics import Gauge

from cosmos_xenna.pipelines.private.monitoring_types import (
    ActorInfo,
    ActorResourceUsage,
    ActorResourceUsageAndMetadata,
    PipelineStats,
    RayClusterInfo,
    RayStageResourceUsage,
    Resources,
)
from cosmos_xenna.ray_utils import actor_pool, resource_monitor, stage_worker
from cosmos_xenna.utils import python_log as logger
from cosmos_xenna.utils import timing
from cosmos_xenna.utils.verbosity import VerbosityLevel


@attrs.define
class SystemDataAndProcessTree:
    system_data: resource_monitor.SystemData
    process_tree: resource_monitor.ProcessTree


@ray.remote(num_cpus=1.0)
class NodeResourceMonitor:
    """This is a Ray Actor which our pipelines will start on each node.

    Its purpose is to collect process info. We need to know per-process utilization stats so that we can
    roll them into per-actor and per-stage stats.

    # TODO: This needs to be fixed. It's probably better to not have a background thread and just make the calculation
    # happen at the callsite. This way, we cover everything happening between calls. However, this is a bit tricky as
    # it may impact the speed at which "get_latest_metrics" runs.
    """

    def __init__(self) -> None:
        self._node_id = ray.get_runtime_context().get_node_id()
        self._latest_metrics: SystemDataAndProcessTree | None = None
        self._exception = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._metrics_loop)
        self._thread.start()

    def _metrics_loop(self) -> None:
        monitor = resource_monitor.ResourceMonitor()
        sleeper = timing.RateLimiter(1.0)
        while not self._stop_event.is_set():
            try:
                self._latest_metrics = SystemDataAndProcessTree(monitor.update(), resource_monitor.ProcessTree.make())
                self._exception = None
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Error in metrics collection: {e}")
                self._exception = str(e)
                break
            sleeper.sleep()

    def get_latest_metrics(self) -> SystemDataAndProcessTree | None:
        if self._exception:
            raise Exception(f"Error in metrics collection: {self._exception}")
        return self._latest_metrics


class RayResourceMonitor:
    """A class which starts up starts up a NodeResourceMonitor on each node and collect stats from them."""

    def __init__(self) -> None:
        self._node_ids = [x["NodeID"] for x in ray.nodes() if x.get("Alive", True)]
        self._monitors = [
            NodeResourceMonitor.options(  # type: ignore
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote()
            for node_id in self._node_ids
        ]

    def update(self) -> dict[str, SystemDataAndProcessTree | None]:
        """Collect stats from each node."""
        results = ray.get([monitor.get_latest_metrics.remote() for monitor in self._monitors])  # type: ignore
        out = {}
        for node_id, result in zip(self._node_ids, results):
            out[node_id] = result
        return out


def get_ray_actors(state: str = "ALIVE") -> List[ActorInfo]:
    # By default, ray will only list up to 10,000 actors, but sometimes we set the limit higher via an env var
    limit = int(os.environ.get("RAY_MAX_LIMIT_FROM_API_SERVER", "10000"))
    actors_data: Any = ray.util.state.list_actors(filters=[("state", "=", "ALIVE")], limit=limit)
    actor_infos = []

    for actor_data in actors_data:
        actor_info = ActorInfo(
            actor_id=actor_data.get("actor_id", ""),
            name=actor_data.get("name"),
            namespace=actor_data.get("namespace", ""),
            runtime_env=actor_data.get("runtime_env"),
            job_id=actor_data.get("job_id", ""),
            pid=actor_data.get("pid"),
            ip_address=actor_data.get("address", {}).get("ip_address"),
            port=actor_data.get("address", {}).get("port"),
            state=actor_data.get("state", ""),
            class_name=actor_data.get("class_name"),
            function_descriptor=actor_data.get("function_descriptor"),
            resources=actor_data.get("resources", {}),
            actor_type=actor_data.get("type", ""),
            current_task_desc=actor_data.get("current_task_desc"),
            num_restarts=actor_data.get("num_restarts", 0),
            timestamp=actor_data.get("timestamp", 0.0),
            node_id=actor_data.get("node_id", ""),
            repr_name=actor_data.get("repr_name", ""),
        )
        if actor_info.state == state:
            actor_infos.append(actor_info)

    return actor_infos


def calculate_actor_resource_usage(
    actors: list[ActorInfo],
    process_info_per_node: dict[str, resource_monitor.ProcessTree],
) -> list[ActorResourceUsageAndMetadata]:
    """Merge Ray actor info with process-info per node to determine per-actor resource usage."""
    result = []

    node_and_pid_to_process: dict[str, dict[int, resource_monitor.ProcessInfo]] = collections.defaultdict(dict)

    def _build_pid_map(node_id: str, graph_node: resource_monitor.ProcessInfo) -> None:
        node_and_pid_to_process[node_id][graph_node.pid] = graph_node
        for child in graph_node.children:
            _build_pid_map(node_id, child)

    for node_id, tree in process_info_per_node.items():
        _build_pid_map(node_id, tree.root)

    for actor in actors:
        pid = actor.pid
        node = actor.node_id
        assert pid is not None
        assert node is not None
        process_map = node_and_pid_to_process[node]
        if actor.pid is not None and actor.pid in process_map:
            process = process_map[actor.pid]
            cpu_percent = process.total_cpu_utilization()
            memory_usage = process.ray_memory_utilization()

            resource_usage = ActorResourceUsage(cpu_utilization=cpu_percent, memory_usage=memory_usage)

            result.append(ActorResourceUsageAndMetadata(metadata=actor, resource_usage=resource_usage))

    return result


def sum_actor_resource_usage_by_stage_name(
    actor_resource_usage_list: List[ActorResourceUsageAndMetadata],
    actor_id_to_stage_name_mapping: dict[str, str],
) -> dict[str, RayStageResourceUsage]:
    """Merge resource usage per actor and a map of actor ids to stage name to create resource usage per stage."""
    group_usage: dict[str, RayStageResourceUsage] = {}

    for item in actor_resource_usage_list:
        stage_name = actor_id_to_stage_name_mapping.get(item.metadata.actor_id, "")

        if stage_name not in group_usage:
            group_usage[stage_name] = RayStageResourceUsage(
                pool_name=stage_name, cpu_utilization=0.0, memory_usage=0, actor_count=0
            )

        group = group_usage[stage_name]
        group.cpu_utilization += item.resource_usage.cpu_utilization
        group.memory_usage += item.resource_usage.memory_usage
        group.actor_count += 1

    return group_usage


def make_ray_cluster_info() -> RayClusterInfo:
    cluster_resources = ray.cluster_resources()
    available_cluster_resources = ray.available_resources()
    total = Resources(
        cluster_resources.get("CPU", 0.0),
        cluster_resources.get("GPU", 0.0),
        cluster_resources.get("memory", 0.0),
        cluster_resources.get("object_store_memory", 0.0),
    )
    available = Resources(
        available_cluster_resources.get("CPU", 0.0),
        available_cluster_resources.get("GPU", 0.0),
        available_cluster_resources.get("memory", 0.0),
        available_cluster_resources.get("object_store_memory", 0.0),
    )

    return RayClusterInfo(
        total=total,
        available=available,
        actors=get_ray_actors(),
    )


class PipelineMonitor:
    """A class which monitors the state of a Ray pipeline.

    There is a bunch of random and somewhat hacky things we need to do to gather monitoring info for our pipelines.
    Ray doesn't make this particularly easy.

    This class will do two things with this info:
    1. Log it
    2. (If parseable is configured) Dump it to parseable

    To gather this info, this class will
    1. Start a NodeResourceMonitor actor per-node and gather process info from it
    2. Use ray's internal state API to get a mapping of actor_id to process id
    3. Get stage -> actor_ids info from the actor pools
    4. Collect a bunch of other stats from the actor pools.

    """

    def __init__(
        self,
        log_interval_s: float,
        initial_input_len: int,
        actor_pools: list[actor_pool.ActorPool],
        verbosity_level: VerbosityLevel = VerbosityLevel.INFO,
    ) -> None:
        self._actor_pools = list(actor_pools)
        self._log_interval_s = float(log_interval_s)
        self._initital_input_length = int(initial_input_len)
        self._verbosity_level = verbosity_level
        self._opened = False

    def __enter__(self) -> PipelineMonitor:
        assert not self._opened
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        # final update for metrics
        if os.environ.get("XENNA_RAY_METRICS_PORT", None) is not None:
            stats = PipelinestatsWithTime(
                time.time(), self._make_stats(0, [0 for _ in self._actor_pools], [[] for _ in self._actor_pools])
            )
            self._update_ray_metrics(stats.pipeline)
            # wait a few seconds to ensure metrisc are updated
            time.sleep(10)
        # exit
        assert self._opened
        self.close()

    def open(self) -> None:
        """Start up all the stateful stuff in the class."""
        assert not self._opened
        self._rate_estimator = timing.RateEstimator(120.0)
        self._pipeline_start_time = time.time()
        self._log_rate_limiter = timing.RateLimitChecker(1.0 / self._log_interval_s)
        self._nodes_resource_monitor = RayResourceMonitor()
        self._create_ray_metrics()
        self._opened = True

    def update(
        self,
        input_len: int,
        ext_output_lens: list[int],
        task_metadata_per_pool: list[list[stage_worker.TaskResultMetadata]],
    ) -> bool:
        # TODO: This update method is pretty slow. This should be refactored to hide it behind a background thread or
        # something. For now, we just limit it to run every 30 seconds.
        assert self._opened
        self._rate_estimator.update()
        should_log = self._log_rate_limiter.can_call(check_only=True)
        if not should_log:
            return False

        start = time.time()
        stats = PipelinestatsWithTime(time.time(), self._make_stats(input_len, ext_output_lens, task_metadata_per_pool))

        logger.debug(f"took {time.time() - start} to get stats.")

        # Update metrics
        self._update_ray_metrics(stats.pipeline)

        # Clear actors. We don't really need to keep track of this info and it's pretty heavy.
        stats.pipeline.cluster.actors = []
        # Maybe log the current state.
        if self._log_rate_limiter.can_call():
            if self._verbosity_level >= VerbosityLevel.INFO:
                self._print_state(stats.pipeline)
            return True
        else:
            return False

    def _make_stats(
        self,
        input_len: int,
        ext_output_lens: list[int],
        task_metadata_per_pool: list[list[stage_worker.TaskResultMetadata]],
    ) -> PipelineStats:
        start = time.time()
        node_resource_data = self._nodes_resource_monitor.update()
        logger.debug(f"Took {time.time() - start} seconds to get node resource info.")

        start = time.time()
        cluster_info = make_ray_cluster_info()
        logger.debug(f"Took {time.time() - start} seconds to get cluster info.")
        stats = [pool.make_stats(ext_output_lens[idx]) for idx, pool in enumerate(self._actor_pools)]
        actor_id_to_pool_mapping = {}
        for pool_stats in stats:
            for x in pool_stats.pending_actor_pool_ids + pool_stats.ready_actor_pool_ids:
                actor_id_to_pool_mapping[x] = pool_stats.name
        start = time.time()
        actors = get_ray_actors()
        logger.debug(f"Took {time.time() - start} seconds to get actor info.")
        actor_resource_usage = calculate_actor_resource_usage(
            actors,
            {k: v.process_tree for k, v in node_resource_data.items() if v is not None},
        )
        t = time.time()

        extra_outputs = collections.defaultdict(list)
        for pool, metadatas in zip(self._actor_pools, task_metadata_per_pool):
            extra_outputs[pool.name] = metadatas

        contents = PipelineStats(
            self._pipeline_start_time,
            time.time(),
            num_initial_input_tasks=self._initital_input_length,
            num_input_tasks_remaining=input_len,
            num_outputs=ext_output_lens[-1] if len(ext_output_lens) > 0 else 0,
            actor_pools=stats,
            pipeline_duration_s=t - self._pipeline_start_time,
            main_loop_rate_hz=self._rate_estimator.get_rate(),
            cluster=cluster_info,
            resource_usage_per_stage=sum_actor_resource_usage_by_stage_name(
                actor_resource_usage, actor_id_to_pool_mapping
            ),
            extra_data_per_stage=extra_outputs,
        )
        return contents

    def _print_state(self, stats: PipelineStats) -> None:
        display = stats.display()
        print(display)

    def close(self) -> None:
        assert self._opened

    def _create_ray_metrics(self) -> None:
        self._metrics_input_tasks = Gauge(
            "pipeline_input_tasks",
            "Number of total input tasks",
            tag_keys=("xenna_user", "xenna_job_name", "xenna_job_id"),
        )
        self._metrics_finished_tasks = Gauge(
            "pipeline_finished_tasks",
            "Number of finished tasks",
            tag_keys=("stage",),
        )
        self._metrics_finished_tasks_norm = Gauge(
            "pipeline_finished_tasks_normalized",
            "Number of finished tasks normalized to original input to the pipeline",
            tag_keys=("stage",),
        )
        self._metrics_pipeline_progress = Gauge(
            "pipeline_progress",
            "Progress of the pipeline",
            tag_keys=None,
        )
        self._metrics_slots_used = Gauge(
            "pipeline_slots_used",
            "Number of slots used per stage",
            tag_keys=("stage",),
        )
        self._metrics_slots_empty = Gauge(
            "pipeline_slots_empty",
            "Number of slots empty per stage",
            tag_keys=("stage",),
        )
        self._metrics_input_queue_size = Gauge(
            "pipeline_input_queue_size",
            "Input task queue size per stage",
            tag_keys=("stage",),
        )
        self._metrics_output_queue_size = Gauge(
            "pipeline_output_queue_size",
            "Output task queue size per stage",
            tag_keys=("stage",),
        )
        self._metrics_actor_count = Gauge(
            "pipeline_actor_count",
            "Number of actors per stage per state",
            tag_keys=("stage", "state"),
        )
        self._metrics_actor_process_time = Gauge(
            "pipeline_actor_process_time",
            "Time taken to process one task by one actor",
            tag_keys=("stage",),
        )
        self._metrics_actor_resource_request = Gauge(
            "pipeline_actor_resource_request",
            "Resource request for actors",
            tag_keys=("stage", "resource"),
        )
        self._metrics_actor_resource_usage = Gauge(
            "pipeline_actor_resource_usage",
            "Resource usage for actors",
            tag_keys=("stage", "resource"),
        )

        # set initial values
        self._metrics_input_tasks.set(
            self._initital_input_length,
            tags={
                "xenna_user": os.getenv("SLURM_JOB_USER", "unknown"),
                "xenna_job_name": os.getenv("SLURM_JOB_NAME", "unknown"),
                "xenna_job_id": os.getenv("SLURM_JOB_ID", "unknown"),
            },
        )
        for pool in self._actor_pools:
            self._metrics_actor_resource_request.set(
                pool.worker_shape.get_num_cpus(),
                tags={"stage": pool.name, "resource": "cpu"},
            )
            self._metrics_actor_resource_request.set(
                pool.worker_shape.get_num_gpus(),
                tags={"stage": pool.name, "resource": "gpu"},
            )

    def _update_ray_metrics(self, stats: PipelineStats) -> None:
        # calculate a normalization factor for the next stage when calculating progress
        normalization_factor = 1.0
        # total completed task stages
        total_completed_task_stages = 0

        # loop through all stages
        for pool_stats in stats.actor_pools:
            self._metrics_finished_tasks.set(pool_stats.task_stats.total_completed, tags={"stage": pool_stats.name})
            # for progress tracking
            num_completed_tasks = pool_stats.task_stats.total_completed
            num_spawned_tasks = pool_stats.task_stats.total_dynamically_spawned
            # apply normalization factor from last stage
            total_completed_norm = pool_stats.task_stats.total_completed * normalization_factor
            self._metrics_finished_tasks_norm.set(
                total_completed_norm,
                tags={"stage": pool_stats.name},
            )
            total_completed_task_stages += total_completed_norm
            # calculate the normalization factor for the next stage
            if pool_stats.task_stats.total_completed > 0:
                new_normalization_factor = num_completed_tasks / (num_completed_tasks + num_spawned_tasks)
                normalization_factor *= new_normalization_factor
            # state of current stage's actor pool
            self._metrics_slots_used.set(pool_stats.slot_stats.num_used, tags={"stage": pool_stats.name})
            self._metrics_slots_empty.set(pool_stats.slot_stats.num_empty, tags={"stage": pool_stats.name})
            self._metrics_input_queue_size.set(pool_stats.task_stats.input_queue_size, tags={"stage": pool_stats.name})
            self._metrics_output_queue_size.set(
                pool_stats.task_stats.output_queue_size, tags={"stage": pool_stats.name}
            )
            # count of actors in different states, e.g. pending or ready, busy or idle
            for state, count in attrs.asdict(pool_stats.actor_stats).items():
                self._metrics_actor_count.set(count, tags={"stage": pool_stats.name, "state": state})
            # speed measurement
            if pool_stats.processing_speed_tasks_per_second is not None:
                self._metrics_actor_process_time.set(
                    1.0 / pool_stats.processing_speed_tasks_per_second,
                    tags={"stage": pool_stats.name},
                )

        # overall pipeline progress
        if self._initital_input_length > 0:
            pipeline_progress = total_completed_task_stages / (self._initital_input_length * len(stats.actor_pools))
            self._metrics_pipeline_progress.set(pipeline_progress)
        else:
            self._metrics_pipeline_progress.set(0.0)
        # resource usage per stage
        for stage_name, usage_stats in stats.resource_usage_per_stage.items():
            self._metrics_actor_resource_usage.set(
                usage_stats.cpu_utilization,
                tags={"stage": stage_name, "resource": "cpu"},
            )
            self._metrics_actor_resource_usage.set(
                usage_stats.memory_usage,
                tags={"stage": stage_name, "resource": "memory"},
            )


@attrs.define
class PipelinestatsWithTime:
    unix_time_s: float
    pipeline: PipelineStats
