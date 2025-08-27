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

from . import data_structures as ds

class WorkerIdFactory:
    count: int

    def __init__(self) -> None: ...
    def make_new_id(self) -> str: ...

class Estimate:
    batches_per_second_per_worker: float | None
    num_returns_per_batch: float | None

    def __init__(
        self,
        batches_per_second_per_worker: float | None,
        num_returns_per_batch: float | None,
    ) -> None: ...

class Estimates:
    stages: list[Estimate]

    def __init__(self, stages: list[Estimate]) -> None: ...

def run_fragmentation_autoscaler(
    problem: ds.Problem,
    state: ds.ProblemState,
    estimates: Estimates,
    overallocation_target: float,
    worker_id_factory: WorkerIdFactory,
) -> ds.Solution: ...

class FragmentationBasedAutoscaler:
    def __init__(self) -> None: ...
    def name(self) -> str: ...
    def setup(self, problem: ds.Problem) -> None: ...
    def update_with_measurements(self, time: float, measurements: ds.Measurements) -> None: ...
    def autoscale(self, current_time: float, state: ds.ProblemState) -> ds.Solution: ...
