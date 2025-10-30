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

import math
import typing

import ray

from cosmos_xenna.pipelines.private import (
    allocator,
    autoscaling_algorithms,
    data_structures,
    monitoring,
    resources,
    specs,
)
from cosmos_xenna.ray_utils import actor_pool
from cosmos_xenna.utils import deque, grouping, timing
from cosmos_xenna.utils import python_log as logger

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_MAX_MAIN_LOOP_RATE_HZ = 100


def _determine_number_of_workers_and_scale_pool(
    stage_spec: specs.StageSpec,
    pool: actor_pool.ActorPool,
    cluster_resources: resources.ClusterResources,
    worker_id_factory: autoscaling_algorithms.WorkerIdFactory,
) -> None:
    """Determines how many workers to assign to the active pool/stage and assigns them to the actor pool.

    We re-use the autoscaling algorithm we use for streaming. This is kind of a hack as it is much more complex
    than we need, but a batch pipeline is basically just a single-stage streaming pipeline, so it should work fine.
    """
    maybe_requested_num_workers = stage_spec.num_workers
    if stage_spec.num_workers_per_node is not None:
        maybe_requested_num_workers = math.ceil(cluster_resources.num_nodes * stage_spec.num_workers_per_node)

    problem = data_structures.Problem(
        cluster_resources=cluster_resources,
        stages=[
            data_structures.ProblemStage(
                name=pool.name,
                stage_batch_size=1,
                worker_shape=pool.worker_shape,
                requested_num_workers=maybe_requested_num_workers,
                over_provision_factor=None,
            )
        ],
    )
    state = data_structures.ProblemState(
        [data_structures.ProblemStageState(stage_name=pool.name, workers=[], slots_per_worker=2, is_finished=False)]
    )
    solution = autoscaling_algorithms.run_fragmentation_autoscaler(
        problem=problem,
        state=state,
        estimates=autoscaling_algorithms.Estimates(
            stages=[autoscaling_algorithms.Estimate(batches_per_second_per_worker=1, num_returns_per_batch=1)]
        ),
        overallocation_target=1.0,
        worker_id_factory=worker_id_factory,
    )
    assert solution.stages[0].deleted_workers == []
    for worker_to_add in solution.stages[0].new_workers:
        pool.add_actor_to_create(worker_to_add.to_worker_group(pool.name))


def run_pipeline(
    pipeline_spec: specs.PipelineSpec,
    cluster_resources: resources.ClusterResources,
) -> list | None:
    """Runs a pipeline under BATCH mode."""
    # Create a worker allocator to keep track of which workers are allocated across the cluster
    # We will not use this directly, but it is used by the actor pools
    worker_allocator = allocator.WorkerAllocator.make(cluster_resources)
    port_registry = actor_pool.ClusterPortRegistry()

    worker_id_factory = autoscaling_algorithms.WorkerIdFactory()
    logger.info("Putting all inputs into ray memory store.")
    assert isinstance(pipeline_spec.stages[0], specs.StageSpec)
    groups = grouping.split_by_chunk_size(pipeline_spec.input_data, pipeline_spec.stages[0].stage.stage_batch_size)
    inputs: list[actor_pool.Task] = []
    for group in groups:
        obj_refs = [ray.put(x) for x in group]
        inputs.append(actor_pool.Task(obj_refs, None))
    logger.info("Done putting all inputs into ray memory store.")
    input_copy = inputs
    initial_input_len = len(inputs)
    outputs: list[ray.ObjectRef] = []
    rate_limiter = timing.RateLimiter(_MAX_MAIN_LOOP_RATE_HZ)

    pools: list[actor_pool.ActorPool] = []
    for idx, spec in enumerate(pipeline_spec.stages):
        assert isinstance(spec, specs.StageSpec)
        wrapped_stage = specs.make_actor_pool_stage_from_stage_spec(pipeline_spec.config, spec, idx, cluster_resources)
        pool_params = actor_pool.PoolParams(
            enable_work_stealing=pipeline_spec.config.enable_work_stealing,
            max_tasks_to_poll_per_chunk=pipeline_spec.config.max_tasks_to_poll_per_chunk,
        )
        pool = actor_pool.ActorPool(
            worker_allocator,
            port_registry,
            wrapped_stage.stage,
            wrapped_stage.params,
            spec.name(idx),
            pool_params=pool_params,
        )
        pools.append(pool)

    with monitoring.PipelineMonitor(
        pipeline_spec.config.logging_interval_s,
        initial_input_len,
        pools,
        pipeline_spec.config.monitoring_verbosity_level,
    ) as monitor:
        for idx, (spec, pool) in enumerate(zip(pipeline_spec.stages, pools)):
            assert isinstance(spec, specs.StageSpec)
            logger.info(f"Starting stage={pool.name}")
            if idx != 0:
                groups = grouping.split_by_chunk_size(outputs, spec.stage.stage_batch_size)
                inputs = [actor_pool.Task(group, None) for group in groups]
                outputs = []

            for input in inputs:
                pool.add_task(input)

            _determine_number_of_workers_and_scale_pool(spec, pool, cluster_resources, worker_id_factory)
            while pool.has_work_or_completed:
                pool.update()
                # TODO: task_metadata_per_pool should be filled in for reporting purposes.
                # Skipping for now as it is non-trivial
                monitor.update(
                    len(input_copy), ext_output_lens=[0 for _ in pools], task_metadata_per_pool=[[] for _ in pools]
                )
                latest_outputs = deque.pop_all_deque_elements(pool.completed_tasks)
                # If this is not the last stage OR the user asked us to return the outputs, record them.
                # Otherwise, let them get garbage collected.
                if idx + 1 != len(pools) or pipeline_spec.config.return_last_stage_outputs:
                    for out_task in latest_outputs:
                        outputs.extend(out_task.task_data)
                rate_limiter.sleep()
            logger.info(f"stage={pool.name} finished. Stopping workers.")
            pool.stop()
        if pipeline_spec.config.return_last_stage_outputs:
            return ray.get(outputs)
        else:
            return None
