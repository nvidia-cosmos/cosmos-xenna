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

from typing import List, Optional

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import resources

class ProblemStage:
    name: str
    stage_batch_size: int
    worker_shape: resources.WorkerShape
    requested_num_workers: Optional[int]
    over_provision_factor: Optional[float]

    def __init__(
        self,
        name: str,
        stage_batch_size: int,
        worker_shape: resources.WorkerShape,
        requested_num_workers: Optional[int],
        over_provision_factor: Optional[float],
    ) -> None: ...

class ProblemWorkerState:
    id: str
    resources: resources.WorkerResources

    @staticmethod
    def from_worker(worker: resources.Worker) -> ProblemWorkerState: ...
    def __init__(self, id: str, resources: resources.WorkerResources) -> None: ...
    def to_worker(self, stage_name: str) -> resources.Worker: ...

class ProblemStageState:
    stage_name: str
    workers: List[ProblemWorkerState]
    slots_per_worker: int
    is_finished: bool

    def __init__(
        self,
        stage_name: str,
        workers: List[ProblemWorkerState],
        slots_per_worker: int,
        is_finished: bool,
    ) -> None: ...

class ProblemState:
    stages: List[ProblemStageState]

    def __init__(self, stages: List[ProblemStageState]) -> None: ...
    def __str__(self) -> str: ...

class Problem:
    cluster_resources: resources.ClusterResources
    stages: List[ProblemStage]

    def __init__(
        self,
        cluster_resources: resources.ClusterResources,
        stages: List[ProblemStage],
    ) -> None: ...

class StageSolution:
    slots_per_worker: int
    new_workers: List[ProblemWorkerState]
    deleted_workers: List[ProblemWorkerState]

    def __init__(
        self,
        slots_per_worker: int,
        new_workers: List[ProblemWorkerState],
        deleted_workers: List[ProblemWorkerState],
    ) -> None: ...

class Solution:
    stages: List[StageSolution]

    def __init__(self, stages: List[StageSolution]) -> None: ...
    def num_new_workers_per_stage(self) -> List[int]: ...
    def num_deleted_workers_per_stage(self) -> List[int]: ...
    def __str__(self) -> str: ...

class ProblemStateAndSolution:
    state: ProblemState
    result: Solution

    def __init__(self, state: ProblemState, result: Solution) -> None: ...
    def __str__(self) -> str: ...

class TaskMeasurement:
    start_time: float
    end_time: float
    num_returns: int

    def __init__(self, start_time: float, end_time: float, num_returns: int) -> None: ...

class StageMeasurements:
    task_measurements: List[TaskMeasurement]

    def __init__(self, task_measurements: List[TaskMeasurement]) -> None: ...

class Measurements:
    time: float
    stages: List[StageMeasurements]

    def __init__(self, time: float, stages: List[StageMeasurements]) -> None: ...
