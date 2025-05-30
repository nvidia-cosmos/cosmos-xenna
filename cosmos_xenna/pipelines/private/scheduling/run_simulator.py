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

"""Runs simulations for known difficult pipelines.

For now, this is run and checked manually, but eventually these should be turned into test cases for the auto-scaling
algorithm.

For now, the code is changed manually to examine different test cases.
"""

from cosmos_xenna.pipelines.private.scheduling import autoscaling_algorithms, data_structures, simulator
from cosmos_xenna.ray_utils import resources


def make_h100_cluster_resources(num_nodes: int) -> resources.ClusterResources:
    return resources.ClusterResources(
        {
            str(idx): resources.NodeResources.make_uniform(
                num_cpus=240, num_gpus=8, num_nvdecs_per_gpu=7, num_nvencs_per_gpu=0
            )
            for idx in range(num_nodes)
        }
    )


def make_stage(
    name: str,
    shape: resources.WorkerShapeTypes,
    process_time_per_worker: float,
    setup_time: float = 0.0,
    stage_batch_size: int = 1,
    num_returns: int = 1,
) -> simulator.SimulationStage:
    return simulator.SimulationStage(
        problem_stage=data_structures.ProblemStage(
            name=name,
            worker_shape=resources.WorkerShape.make(shape),
            stage_batch_size=stage_batch_size,
        ),
        process_time_per_worker=process_time_per_worker,
        setup_time=setup_time,
        num_returns=num_returns,
        stage_batch_size=stage_batch_size,
    )


def make_filtering_problem(num_nodes: int = 10) -> simulator.SimulationProblem:
    cpu_setup_time = 5.0
    model_setup_time = 60.0
    other_gpu_setup_time = 20.0

    return simulator.SimulationProblem(
        stages=[
            make_stage(
                "EncodedClipDownloadStage",
                process_time_per_worker=3.1,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
            make_stage(
                "MotionVectorDecodeStage",
                process_time_per_worker=38.7,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
            make_stage(
                "MotionVectorStage",
                process_time_per_worker=1.5,
                setup_time=other_gpu_setup_time,
                shape=resources.FractionalGpu(num_gpus=0.2, num_cpus=1.0),
            ),
            make_stage(
                "FrameExtractionStage",
                process_time_per_worker=20.2,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(4.0),
            ),
            make_stage(
                "CLIPImageEmbeddingsStage",
                process_time_per_worker=3.9,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "AestheticFilteringStage",
                process_time_per_worker=0.1,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "WaterMarkDetectionStage",
                process_time_per_worker=1.6,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "InternVideo2FrameCreationStage",
                process_time_per_worker=0.9,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(4.0),
            ),
            make_stage(
                "InternVideo2EmbeddingStage",
                process_time_per_worker=0.9,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "WindowingStage",
                process_time_per_worker=31.8,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
            make_stage(
                "VilaInputBuildStage",
                process_time_per_worker=9.6,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(4.0),
            ),
            make_stage(
                "VilaCaptionStage",
                process_time_per_worker=7.7,
                setup_time=model_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "T5Stage",
                process_time_per_worker=0.6,
                setup_time=model_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "Upload",
                process_time_per_worker=3.1,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
        ],
        cluster_resources=make_h100_cluster_resources(num_nodes),
    )


def make_filtering_problem_with_nvdec(num_nodes: int = 10) -> simulator.SimulationProblem:
    cpu_setup_time = 5.0
    model_setup_time = 60.0
    other_gpu_setup_time = 20.0

    return simulator.SimulationProblem(
        stages=[
            make_stage(
                "EncodedClipDownloadStage",
                process_time_per_worker=3.1,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
            make_stage(
                "MotionVectorDecodeStage",
                process_time_per_worker=38.7,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
            make_stage(
                "MotionVectorStage",
                process_time_per_worker=1.5,
                setup_time=other_gpu_setup_time,
                shape=resources.FractionalGpu(num_gpus=0.2, num_cpus=1.0),
            ),
            make_stage(
                "FrameExtractionStage",
                process_time_per_worker=20.2,
                setup_time=cpu_setup_time,
                shape=resources.Codec(num_cpus=1, num_nvdecs=1),
            ),
            make_stage(
                "CLIPImageEmbeddingsStage",
                process_time_per_worker=3.9,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "AestheticFilteringStage",
                process_time_per_worker=0.1,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "WaterMarkDetectionStage",
                process_time_per_worker=1.6,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "InternVideo2FrameCreationStage",
                process_time_per_worker=0.9,
                setup_time=cpu_setup_time,
                shape=resources.Codec(num_nvdecs=1),
            ),
            make_stage(
                "InternVideo2EmbeddingStage",
                process_time_per_worker=0.9,
                setup_time=other_gpu_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "WindowingStage",
                process_time_per_worker=3,
                setup_time=cpu_setup_time,
                shape=resources.Codec(num_nvdecs=1),
            ),
            make_stage(
                "VilaInputBuildStage",
                process_time_per_worker=9.6,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(4.0),
            ),
            make_stage(
                "VilaCaptionStage",
                process_time_per_worker=7.7,
                setup_time=model_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "T5Stage",
                process_time_per_worker=0.6,
                setup_time=model_setup_time,
                shape=resources.WholeNumberedGpu(num_gpus=1, num_cpus=1.0),
            ),
            make_stage(
                "Upload",
                process_time_per_worker=3.1,
                setup_time=cpu_setup_time,
                shape=resources.CpuOnly(1.0),
            ),
        ],
        cluster_resources=make_h100_cluster_resources(num_nodes),
    )


def main() -> None:
    problem = make_filtering_problem_with_nvdec(num_nodes=2)
    algorithm = autoscaling_algorithms.FragmentationBasedAutoscaler()
    params = simulator.SimulationParams(num_tasks=10000, rescale_interval_s=60 * 3, timestep=0.1, generate_plots=True)
    sim = simulator.Simulator(algorithm, problem, params)
    sim.run()


if __name__ == "__main__":
    main()
