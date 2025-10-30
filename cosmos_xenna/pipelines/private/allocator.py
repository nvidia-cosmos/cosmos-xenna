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

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import allocator as rust  # type: ignore
from cosmos_xenna.pipelines.private import resources


class WorkerAllocator:
    @classmethod
    def make(cls, cluster_resources: resources.ClusterResources) -> WorkerAllocator:
        return cls(rust.WorkerAllocator(cluster_resources.to_rust()))

    def __init__(self, rust_allocator: rust.WorkerAllocator):
        self._rust_allocator = rust_allocator

    def add_worker(self, worker: resources.WorkerGroup) -> None:
        self._rust_allocator.add_worker(worker.rust)

    def remove_worker(self, worker_id: str) -> None:
        self._rust_allocator.remove_worker(worker_id)

    def get_gpu_index(self, node_id: str, gpu_offset: int) -> int:
        return self._rust_allocator.get_gpu_index(node_id, gpu_offset)

    # def totals(self) -> resources.ClusterResources:
    #     return resources.ClusterResources.from_rust(self._rust_allocator.totals())

    def get_workers_in_stage(self, stage_name: str) -> list[resources.WorkerGroup]:
        return [resources.WorkerGroup(x) for x in self._rust_allocator.get_workers_in_stage(stage_name)]

    def make_detailed_utilization_table(self) -> str:
        return self._rust_allocator.make_detailed_utilization_table()
