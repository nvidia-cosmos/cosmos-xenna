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

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import autoscaling_algorithms as rust  # type: ignore
from cosmos_xenna.pipelines.private import data_structures


class WorkerIdFactory:
    def __init__(self) -> None:
        self._rust_worker_id_factory = rust.WorkerIdFactory()

    def get_worker_id(self) -> str:
        return self._rust_worker_id_factory.get_worker_id()

    @property
    def rust(self) -> rust.WorkerIdFactory:
        return self._rust_worker_id_factory


class FragmentationBasedAutoscaler:
    def __init__(
        self,
        speed_estimation_window_duration_s: float = 60 * 3.0,
        speed_estimation_min_data_points: int = 5,
    ) -> None:
        self._rust_autoscaler = rust.FragmentationBasedAutoscaler(
            speed_estimation_window_duration_s,
            speed_estimation_min_data_points,
        )

    def setup(self, problem: data_structures.Problem) -> None:
        self._rust_autoscaler.setup(problem.rust)

    def update_with_measurements(self, time: float, measurements: data_structures.Measurements) -> None:
        self._rust_autoscaler.update_with_measurements(time, measurements.rust)

    def autoscale(self, time: float, problem_state: data_structures.ProblemState) -> data_structures.Solution:
        return data_structures.Solution(self._rust_autoscaler.autoscale(time, problem_state.rust))


class Estimate:
    def __init__(self, batches_per_second_per_worker: float, num_returns_per_batch: float) -> None:
        self._r = rust.Estimate(batches_per_second_per_worker, num_returns_per_batch)

    @property
    def rust(self) -> rust.Estimate:
        return self._r


class Estimates:
    def __init__(self, stages: list[Estimate]) -> None:
        self._r = rust.Estimates([e.rust for e in stages])

    @property
    def rust(self) -> rust.Estimates:
        return self._r


def run_fragmentation_autoscaler(
    problem: data_structures.Problem,
    state: data_structures.ProblemState,
    estimates: Estimates,
    overallocation_target: float,
    worker_id_factory: WorkerIdFactory,
) -> data_structures.Solution:
    return data_structures.Solution(
        rust.run_fragmentation_autoscaler(
            problem.rust, state.rust, estimates.rust, overallocation_target, worker_id_factory.rust
        )
    )
