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
Smoke test for multi-node SPMD (Single Program, Multiple Data) execution with torch distributed.

This test verifies that:
1. SPMD pipeline stages can properly initialize torch distributed process groups
2. GPU allocation works correctly across multiple nodes/processes
3. Basic distributed operations (all_reduce, broadcast, all_gather) function properly
4. Distributed inference workloads can be executed
5. CUDA_VISIBLE_DEVICES is set correctly for each worker
"""

import os
import time
import typing
from typing import Optional

import pytest
import ray
import torch
import torch.distributed as dist

from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.utils import python_log as logger


class _MultiNodeSpmdStage(pipelines_v1.Stage):
    def __init__(self, num_gpus_per_node: int, num_nodes: int):
        self._distributed_initialized = False
        self._num_gpus_per_node = num_gpus_per_node
        self._num_nodes = num_nodes

    @property
    def required_resources(self) -> Optional[pipelines_v1.Resources]:
        return pipelines_v1.Resources(cpus=1.0, gpus=self._num_gpus_per_node * self._num_nodes, is_spmd=True)

    @property
    def stage_batch_size(self) -> int:
        return 16

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        logger.info(f"Worker metadata: {worker_metadata}")
        assert worker_metadata.world_size is not None
        assert worker_metadata.rank is not None
        assert worker_metadata.rendevous_params is not None

        env = os.environ
        time.sleep(10)

        # Get physical GPU information
        physical_gpu_info = self._get_physical_gpu_info()

        logger.info(
            f"{env['MASTER_ADDR']}:{env['MASTER_PORT']} rank={env['RANK']} world_size={env['WORLD_SIZE']} "
            f"local_rank={env['LOCAL_RANK']}, node={ray.get_runtime_context().get_node_id()}, "
            f"logical_device={torch.cuda.current_device()}, {physical_gpu_info}"
        )
        time.sleep(10)
        # DEBUG: Print all environment variables before init
        # self._debug_distributed_env_vars()

        # Initialize torch distributed if we have the necessary metadata
        logger.info(
            f"Initializing torch distributed: rank={worker_metadata.rank}, world_size={worker_metadata.world_size}"
        )

        # Initialize the process group using NCCL backend for GPU communication
        dist.init_process_group()

        logger.info("Torch distributed initialized successfully!")
        print(
            f"Local rank: {worker_metadata.distributed_execution_params.local_rank}, "
            f"Device: {torch.cuda.current_device()}"
        )
        assert len(worker_metadata.allocation.gpus) == 1

        # Test basic distributed communication
        self._test_distributed_communication()
        self._test_distributed_inference()

    def _get_physical_gpu_info(self) -> str:
        """Get information about the physical GPU this process is using."""
        try:
            current_logical = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_logical)

            # Parse CUDA_VISIBLE_DEVICES to get physical GPU ID
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                physical_devices = cuda_visible.split(",")
                if current_logical < len(physical_devices):
                    physical_id = physical_devices[current_logical]
                    return f"physical_gpu={physical_id}, gpu_name='{gpu_name}'"
                else:
                    return f"physical_gpu=unknown, gpu_name='{gpu_name}'"
            else:
                return f"physical_gpu={current_logical}, gpu_name='{gpu_name}'"
        except Exception as e:  # noqa: BLE001
            return f"gpu_info_error={e!s}"

    def _debug_distributed_env_vars(self) -> None:
        """Debug method to print all relevant environment variables for torch distributed."""

        logger.info("=== TORCH DISTRIBUTED ENVIRONMENT VARIABLES ===")

        required_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
        optional_vars = ["CUDA_VISIBLE_DEVICES", "TORCH_NCCL_BLOCKING_WAIT", "TORCH_NCCL_ASYNC_ERROR_HANDLING"]

        logger.info("--- Required Variables ---")
        missing_vars = []
        for var in required_vars:
            value = os.environ.get(var)
            if value is not None:
                logger.info(f"{var}: {value}")
            else:
                logger.info(f"{var}: NOT SET")
                missing_vars.append(var)

        logger.info("--- Optional Variables ---")
        for var in optional_vars:
            value = os.environ.get(var)
            logger.info(f"{var}: {value if value is not None else 'NOT SET'}")

        logger.info("--- Analysis ---")
        if missing_vars:
            logger.error(f"Missing required variables: {missing_vars}")
            logger.error("This will cause dist.init_process_group() to hang!")
        else:
            logger.info("All required variables are set")

            # Additional validation
            try:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                local_rank = int(os.environ["LOCAL_RANK"])

                logger.info("--- Validation ---")
                logger.info(
                    f"RANK ({rank}) should be between 0 and {world_size - 1}: "
                    f"{'OK' if 0 <= rank < world_size else 'NOT OK'}"
                )
                logger.info(f"LOCAL_RANK ({local_rank}) should be between 0 and num_gpus_per_node-1")

                # Check if RANK and LOCAL_RANK make sense
                if rank == local_rank and world_size > 1:
                    logger.warning("WARNING: RANK == LOCAL_RANK, this might be incorrect for multi-node setups")

            except (ValueError, KeyError) as e:
                logger.error(f"Error validating variables: {e}")

        logger.info("=== END DEBUG INFO ===")

    def _test_distributed_communication(self) -> None:
        logger.info("calling barrier...")
        dist.barrier()
        logger.info("barrier done")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"Testing distributed communication on rank {rank}/{world_size}")

        # Test 1: All-reduce operation
        tensor = torch.ones(4, device="cuda") * (rank + 1)
        print(f"Rank {rank} before all_reduce: {tensor}")

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(1, world_size + 1))  # Sum of ranks + 1
        print(f"Rank {rank} after all_reduce: {tensor} (expected: {expected_sum})")

        # Test 2: Broadcast operation
        if rank == 0:
            broadcast_tensor = torch.tensor([42.0, 84.0], device="cuda")
        else:
            broadcast_tensor = torch.zeros(2, device="cuda")

        print(f"Rank {rank} before broadcast: {broadcast_tensor}")
        dist.broadcast(broadcast_tensor, src=0)
        print(f"Rank {rank} after broadcast: {broadcast_tensor}")

        # Test 3: Simple distributed inference simulation
        self._test_distributed_inference()

    def _test_distributed_inference(self) -> None:
        """Simulate a simple distributed inference workload"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create a simple model shard (different weights per rank)
        model_shard = torch.nn.Linear(10, 5).to("cuda")

        # Create input data
        batch_size = 8
        input_data = torch.randn(batch_size, 10, device="cuda")

        # Forward pass
        with torch.no_grad():
            output = model_shard(input_data)

        print(f"Rank {rank} inference output shape: {output.shape}")

        # Gather outputs from all ranks
        gathered_outputs = [torch.zeros_like(output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, output)

        if rank == 0:
            print(f"Gathered outputs from all {world_size} ranks")
            for i, out in enumerate(gathered_outputs):
                print(f"  Rank {i} output mean: {out.mean().item():.4f}")

    def process_data(self, in_data: list[int]) -> list[int]:
        # Convert to tensor, do some GPU operations, convert back
        # Test basic distributed communication
        self._test_distributed_communication()
        self._test_distributed_inference()
        return [int(x) for x in in_data]


def test_pipeline() -> None:
    # Skip test if no GPUs are available
    gpu_infos = pipelines_v1.get_local_gpu_info()
    if not gpu_infos:
        pytest.skip("No GPUs available on this machine")

    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=range(1000),
        stages=[_MultiNodeSpmdStage(num_gpus_per_node=len(gpu_infos), num_nodes=len(gpu_infos))],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            logging_interval_s=10,
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
        ),
    )
    results = typing.cast(list[int], pipelines_v1.run_pipeline(pipeline_spec))

    # Verify we got results
    assert len(results) > 0, "Pipeline should return results"
    print(f"Pipeline completed successfully with {len(results)} results")


def test_torch_spmd_functionality() -> None:
    """Test specifically focused on torch distributed SPMD functionality"""
    # Skip test if no GPUs are available
    gpu_infos = pipelines_v1.get_local_gpu_info()
    if not gpu_infos:
        pytest.skip("No GPUs available on this machine")

    # Create a smaller test to focus on torch functionality
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=range(32),  # Smaller dataset for focused testing
        stages=[_MultiNodeSpmdStage(num_gpus_per_node=len(gpu_infos), num_nodes=len(gpu_infos))],
        config=pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.STREAMING,
            logging_interval_s=5,
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
        ),
    )

    print("Starting torch SPMD functionality test...")
    results = typing.cast(list[int], pipelines_v1.run_pipeline(pipeline_spec))

    # Verify results are processed correctly
    assert len(results) == 32, f"Expected 32 results, got {len(results)}"
