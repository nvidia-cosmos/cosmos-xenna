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

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import data_structures as rust  # type: ignore
from cosmos_xenna.pipelines.private import resources


class ProblemStage:
    def __init__(
        self,
        name: str,
        stage_batch_size: int,
        worker_shape: resources.WorkerShape,
        requested_num_workers: int | None,
        over_provision_factor: float | None,
    ) -> None:
        self._r = rust.ProblemStage(
            name, stage_batch_size, worker_shape.rust, requested_num_workers, over_provision_factor
        )

    @property
    def rust(self) -> rust.Problem:
        return self._r


class Problem:
    def __init__(self, cluster_resources: resources.ClusterResources, stages: list[ProblemStage]) -> None:
        self._r = rust.Problem(cluster_resources.to_rust(), [s.rust for s in stages])

    @property
    def rust(self) -> rust.Problem:
        return self._r


class ProblemWorkerGroupState:
    @classmethod
    def make(cls, id: str, resources: list[resources.WorkerResourcesInternal]) -> ProblemWorkerGroupState:
        return cls(rust.ProblemWorkerGroupState(id, [x.to_rust() for x in resources]))

    def __init__(self, rust_problem_worker_state: rust.ProblemWorkerGroupState) -> None:
        self._r = rust_problem_worker_state

    @property
    def id(self) -> str:
        return self._r.id

    @property
    def resources(self) -> list[resources.WorkerResourcesInternal]:
        return [resources.WorkerResourcesInternal.from_rust(x) for x in self._r.resources]

    def to_worker_group(self, stage_name: str) -> resources.WorkerGroup:
        return resources.WorkerGroup(self._r.to_worker_group(stage_name))

    @property
    def rust(self) -> rust.ProblemWorkerGroupState:
        return self._r


class ProblemStageState:
    def __init__(
        self, stage_name: str, workers: list[ProblemWorkerGroupState], slots_per_worker: int, is_finished: bool
    ) -> None:
        self._r = rust.ProblemStageState(stage_name, [w.rust for w in workers], slots_per_worker, is_finished)

    @property
    def rust(self) -> rust.ProblemStageState:
        return self._r


class ProblemState:
    def __init__(self, stages: list[ProblemStageState]) -> None:
        self._r = rust.ProblemState([s.rust for s in stages])

    @property
    def rust(self) -> rust.ProblemState:
        return self._r


class TaskMeasurement:
    def __init__(self, start_time: float, end_time: float, num_returns: int) -> None:
        self._r = rust.TaskMeasurement(start_time, end_time, num_returns)

    @property
    def rust(self) -> rust.TaskMeasurement:
        return self._r


class StageMeasurements:
    def __init__(self, task_measurements: list[TaskMeasurement]) -> None:
        self._r = rust.StageMeasurements([t.rust for t in task_measurements])

    @property
    def rust(self) -> rust.StageMeasurements:
        return self._r


class Measurements:
    def __init__(self, time: float, stage_measurements: list[StageMeasurements]) -> None:
        self._r = rust.Measurements(time, [s.rust for s in stage_measurements])

    @property
    def rust(self) -> rust.Measurements:
        return self._r


class StageSolution:
    def __init__(self, rust_stage_solution: rust.StageSolution) -> None:
        self._r = rust_stage_solution

    @property
    def deleted_workers(self) -> list[ProblemWorkerGroupState]:
        return [ProblemWorkerGroupState(w) for w in self._r.deleted_workers]

    @property
    def new_workers(self) -> list[ProblemWorkerGroupState]:
        return [ProblemWorkerGroupState(w) for w in self._r.new_workers]

    @property
    def slots_per_worker(self) -> int:
        return self._r.slots_per_worker

    @property
    def rust(self) -> rust.StageSolution:
        return self._r


class Solution:
    def __init__(self, rust_solution: rust.Solution) -> None:
        self._r = rust_solution

    @property
    def stages(self) -> list[StageSolution]:
        return [StageSolution(s) for s in self._r.stages]

    @property
    def rust(self) -> rust.Solution:
        return self._r
