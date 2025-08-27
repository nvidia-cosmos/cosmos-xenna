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

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling.resources import (
    ClusterResources,
    Worker,
)

class WorkerAllocator:
    """Manages resource allocation for distributed pipeline workers across nodes.

    This class is responsible for:
    1. Tracking available compute resources (CPU, GPU, NVDEC, NVENC) across nodes
    2. Managing worker allocation to both nodes and pipeline stages
    3. Preventing resource oversubscription
    4. Providing utilization monitoring and reporting

    The allocator maintains both physical (node-based) and logical (stage-based)
    views of worker allocation to support pipeline execution while ensuring
    safe resource usage.
    """

    def __init__(self, cluster_resources: ClusterResources) -> None: ...
    def totals(self) -> ClusterResources: ...
    def available_resources(self) -> ClusterResources: ...
    def num_nodes(self) -> int: ...
    def add_worker(self, worker: Worker) -> None: ...
    def add_workers(self, workers: list[Worker]) -> None: ...
    def delete_worker(self, worker_id: str) -> Worker: ...
    def delete_workers(self, worker_ids: list[str]) -> None: ...
    def get_worker(self, worker_id: str) -> Worker | None: ...
    def get_workers_in_stage(self, stage_name: str) -> list[Worker]: ...
    def get_workers(self) -> list[Worker]: ...
    def get_num_workers_per_stage(self) -> dict[str, int]: ...
    def calculate_lowest_allocated_node_by_cpu(self) -> str | None: ...
    def worker_ids_and_node_cpu_utilizations(
        self, workers_ids_to_consider: list[str] | None
    ) -> list[tuple[float, str]]: ...
    def make_detailed_utilization_table(self) -> str: ...
