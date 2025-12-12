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

"""
Specifically test the autoscaling algorithm.
"""

import time
import uuid

import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna._cosmos_xenna import setup_logging
from cosmos_xenna.pipelines.private import allocator, autoscaling_algorithms, data_structures, resources, streaming
from cosmos_xenna.utils import python_log as logger


class _ProcessStage(pipelines_v1.Stage):
    def __init__(self, cpus: float, gpus: float, throughput: float) -> None:
        self._cpus = cpus
        self._gpus = gpus
        self._throughput = throughput

    @property
    def process_duration(self) -> float:
        return 1.0 / self._throughput

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cpus, gpus=self._gpus)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        pass

    def process_data(self, task: list[float]) -> list[float]:
        time.sleep(self.process_duration)
        return [x * 2 for x in task]


_TEST_SETS = {
    "why_not_scaling_down": {
        "stage_params": [
            [0.25, 0, 0.67, 4],  # download
            [5, 0, 0.77, None],  # decode
            [1, 0.25, 7.1, None],  # transnet
            [3, 0, 0.28, None],  # encoding
            [16, 0, 0.034, None],  # prepare
            [1, 4, 0.033, None],  # caption
            [1, 1, 0.98, None],  # t5
            [0.25, 0, 0.016, 8],  # writer
        ],
        "cluster_resources": {
            "num_nodes": 10,
            "total_cpus": 192,
            "num_gpus_per_node": 8,
        },
        "sequence_of_stage_ready": [5, 0],
    },
}


def _make_pipeline_spec(stage_params: list[list[float]]) -> pipelines_v1.PipelineSpec:
    stages: list[pipelines_v1.StageSpec] = []
    for stage_param in stage_params:
        stages.append(
            pipelines_v1.StageSpec(
                _ProcessStage(cpus=stage_param[0], gpus=stage_param[1], throughput=stage_param[2]),
                num_workers_per_node=stage_param[3],
            )
        )
    return pipelines_v1.PipelineSpec(
        input_data=range(100),
        stages=stages,
    )


def _make_cluster_resources(cluster_resources: dict[str, int]) -> resources.ClusterResources:
    return resources.ClusterResources(
        nodes={
            f"node-{i}": resources.NodeResources(
                used_cpus=0,
                total_cpus=cluster_resources["total_cpus"],
                gpus=[
                    resources.GpuResources(index=j, uuid_=uuid.uuid4(), used_fraction=0.0)
                    for j in range(cluster_resources["num_gpus_per_node"])
                ],
                name=f"node-{i}",
            )
            for i in range(cluster_resources["num_nodes"])
        }
    )


def _make_measurements(
    pipeline_spec: pipelines_v1.PipelineSpec,
    num_data_points: int,
    num_stages_with_measurements: int = 0,
) -> data_structures.Measurements:
    stage_measurements = []
    for _ in range(num_data_points):
        end_time = time.time()
        for i, stage_spec in enumerate(pipeline_spec.stages):
            start_time = end_time - stage_spec.stage.process_duration  # pyright: ignore[reportAttributeAccessIssue]
            is_measurement_ready = num_stages_with_measurements == 0 or i < num_stages_with_measurements
            stage_measurements.append(
                data_structures.StageMeasurements(
                    [data_structures.TaskMeasurement(start_time=start_time, end_time=end_time, num_returns=1)]
                    if is_measurement_ready
                    else []
                )
            )
    return data_structures.Measurements(time.time(), stage_measurements)


def _dump_worker_allocation(
    worker_allocator: allocator.WorkerAllocator, pipeline_spec: pipelines_v1.PipelineSpec
) -> None:
    for idx, stage_spec in enumerate(pipeline_spec.stages):
        stage_name = stage_spec.name(idx)  # pyright: ignore[reportAttributeAccessIssue]
        worker_groups = worker_allocator.get_workers_in_stage(stage_name)
        logger.info(f"Curnent {stage_name} has {len(worker_groups)} worker groups")
        if stage_spec.stage.required_resources.gpus > 0:  # pyright: ignore[reportAttributeAccessIssue]
            gpus = []
            for worker_group in worker_groups:
                for worker in worker_group.allocations:
                    worker_gpu_offsets = ",".join([str(gpu.offset) for gpu in worker.gpus])
                    gpus.append(f"{worker.node}:{worker_gpu_offsets}")
            gpu_display = " | ".join(sorted(gpus))
            logger.info(f"\tGPUs: {gpu_display}")


def _make_problem_state(
    worker_allocator: allocator.WorkerAllocator, pipeline_spec: pipelines_v1.PipelineSpec
) -> data_structures.ProblemState:
    stage_states = []
    for idx, stage_spec in enumerate(pipeline_spec.stages):
        stage_name = stage_spec.name(idx)  # pyright: ignore[reportAttributeAccessIssue]
        workers = worker_allocator.get_workers_in_stage(stage_name)
        stage_states.append(
            data_structures.ProblemStageState(
                stage_name=stage_name,
                workers=[streaming.make_problem_worker_state_from_worker_state(w) for w in workers],
                slots_per_worker=2,
                is_finished=False,
            )
        )
    return data_structures.ProblemState(stage_states)


def run_autoscaling_test(test_name: str) -> None:
    # make a pipeline and a cluster
    pipeline_spec = _make_pipeline_spec(_TEST_SETS[test_name]["stage_params"])
    cluster_resources = _make_cluster_resources(_TEST_SETS[test_name]["cluster_resources"])

    # setup autoscaler
    SPEED_ESTIMATION_WINDOW_DURATION_S = 180
    SPEED_ESTIMATION_MIN_DATA_POINTS = 1
    problem = streaming._make_problem_from_pipeline_spec(pipeline_spec, cluster_resources)
    algorithm = autoscaling_algorithms.FragmentationBasedAutoscaler(
        speed_estimation_window_duration_s=SPEED_ESTIMATION_WINDOW_DURATION_S,
        speed_estimation_min_data_points=SPEED_ESTIMATION_MIN_DATA_POINTS,
    )
    algorithm.setup(problem)

    # create a worker allocator
    worker_allocator = allocator.WorkerAllocator.make(cluster_resources)
    # measure and scale
    for i in range(len(_TEST_SETS[test_name]["sequence_of_stage_ready"]) + 1):
        # autoscale
        problem_state = _make_problem_state(worker_allocator, pipeline_spec)
        solution = algorithm.autoscale(time.time(), problem_state)

        # dump solution
        for idx, (result, stage_spec) in enumerate(zip(solution.stages, pipeline_spec.stages)):
            stage_name = stage_spec.name(idx)  # pyright: ignore[reportAttributeAccessIssue]
            logger.info(
                f"Solution for {stage_name}: adding {len(result.new_workers)} workers "
                f"and deleting {len(result.deleted_workers)} workers. "
            )

        # remove to-be-deleted workers
        for idx, (result, stage_spec) in enumerate(zip(solution.stages, pipeline_spec.stages)):
            stage_name = stage_spec.name(idx)  # pyright: ignore[reportAttributeAccessIssue]
            for w in result.deleted_workers:
                worker_allocator.remove_worker(w.id)

        # add to-be-added workers
        for idx, (result, stage_spec) in enumerate(zip(solution.stages, pipeline_spec.stages)):
            stage_name = stage_spec.name(idx)  # pyright: ignore[reportAttributeAccessIssue]
            for w in result.new_workers:
                worker_allocator.add_worker(w.to_worker_group(stage_name))

        # dump worker allocation
        _dump_worker_allocation(worker_allocator, pipeline_spec)

        # add measurements
        num_stages_with_measurements = _TEST_SETS[test_name]["sequence_of_stage_ready"][i - 1]
        measurements = _make_measurements(pipeline_spec, SPEED_ESTIMATION_MIN_DATA_POINTS, num_stages_with_measurements)
        algorithm.update_with_measurements(time.time(), measurements)
        display_str = f"first {num_stages_with_measurements}" if num_stages_with_measurements > 0 else "all"
        logger.info(f"Added measurements for {display_str} stages")


if __name__ == "__main__":
    setup_logging()
    run_autoscaling_test("why_not_scaling_down")
