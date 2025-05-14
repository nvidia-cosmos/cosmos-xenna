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

import os
import time
from typing import Optional

import attrs
import pytest
import ray

from cosmos_xenna.ray_utils import cluster, resources, stage, stage_worker
from cosmos_xenna.utils.verbosity import VerbosityLevel


@pytest.fixture
def ray_cluster():
    cluster.init_or_connect_to_cluster()
    yield
    ray.shutdown()


@attrs.define
class ComplexData:
    some_bytes: Optional[bytes]

    @classmethod
    def make_random(cls, size: int) -> ComplexData:
        return ComplexData(os.urandom(size))


class _ComplexStage(stage.Interface):
    def __init__(self, setup_dur: float) -> None:
        self._setup_dur = float(setup_dur)

    def setup_on_node(self, node_info: resources.NodeInfo) -> None:
        pass

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        time.sleep(self._setup_dur)

    def process_data(self, tasks: list[ComplexData]) -> list[ComplexData]:
        # Simply return the list of tasks unchanged
        return tasks


class _SimpleStage(stage.Interface):
    def __init__(self, setup_dur: float, process_dur: float) -> None:
        self._setup_dur = float(setup_dur)
        self._process_dur = float(process_dur)

    def setup_on_node(self, node_info: resources.NodeInfo) -> None:
        pass

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        time.sleep(self._setup_dur)

    def process_data(self, tasks: list[float]) -> list[float]:
        time.sleep(self._process_dur)
        # Process each item in the list
        return [task * 2 for task in tasks]


class _ExceptionStage(stage.Interface):
    def setup_on_node(self, node_info: resources.NodeInfo) -> None:
        pass

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        pass

    def process_data(self, should_throw_list: list[bool]) -> list[int]:
        print(f"should_throw_list: {should_throw_list}")
        processed_results = []
        for should_throw in should_throw_list:
            if should_throw:
                raise ValueError("Throwing as requested")
            processed_results.append(1)
        return processed_results


def test_process_data_basic(ray_cluster) -> None:
    # TaskData now takes a list of refs
    task = stage_worker.TaskData([ray.put(10)])
    actor = stage_worker.StageWorker.remote(
        _SimpleStage(0.0, 0.0),
        stage.Params(),
        resources.Worker(
            "my_worker",
            "my_stage",
            resources.WorkerResources("my_node", 1.0, [], [], []),
        ),
        VerbosityLevel.INFO,
    )
    node_id = ray.get(actor.setup.remote())  # type: ignore
    print(node_id)
    dynamic_results = actor.process_data.remote(task)  # type: ignore
    _, *results = ray.get(dynamic_results)
    assert len(results) == 1
    assert ray.get(results[0]) == 20


def test_process_data_in_parallel(ray_cluster) -> None:
    # Each task holds a list of refs (here, just one ref per task)
    task1 = stage_worker.TaskData([ray.put(10)])
    task2 = stage_worker.TaskData([ray.put(20)])
    # Example with multiple refs in one task
    task3 = stage_worker.TaskData([ray.put(5), ray.put(15)])
    actor = stage_worker.StageWorker.options(max_concurrency=1000).remote(
        _SimpleStage(0.0, 1.0),
        stage.Params(),
        resources.Worker(
            "my_worker",
            "my_stage",
            resources.WorkerResources("my_node", 1.0, [], [], []),
        ),
        VerbosityLevel.INFO,
    )
    ray.get(actor.setup.remote())  # type: ignore

    ref1 = actor.process_data.remote(task1)  # type: ignore
    ref2 = actor.process_data.remote(task2)  # type: ignore
    ref3 = actor.process_data.remote(task3)  # type: ignore

    # Get the dynamic results list first
    metadata1, *result_refs1 = ray.get(ref1)
    metadata2, *result_refs2 = ray.get(ref2)
    metadata3, *result_refs3 = ray.get(ref3)

    # Now get the actual results from their refs
    results1 = ray.get(result_refs1)
    results2 = ray.get(result_refs2)
    results3 = ray.get(result_refs3)

    assert len(results1) == 1
    assert results1[0] == 20
    assert len(results2) == 1
    assert results2[0] == 40
    assert len(results3) == 2
    assert results3 == [10, 30]

    print(results1)
    print(metadata1)
    print(results2)
    print(metadata2)
    print(results3)
    print(metadata3)


def test_complex_data_in_parallel(ray_cluster) -> None:
    # Use lists for data_refs
    task1 = stage_worker.TaskData([ray.put(ComplexData.make_random(1 * 1024 * 1024 * 1024))])
    task2 = stage_worker.TaskData([ray.put(ComplexData.make_random(1 * 1024 * 1024 * 1024))])
    actor = stage_worker.StageWorker.options(max_concurrency=1000).remote(
        _ComplexStage(0.0),
        stage.Params(),
        resources.Worker(
            "my_worker",
            "my_stage",
            resources.WorkerResources("my_node", 1.0, [], [], []),
        ),
        VerbosityLevel.INFO,
    )
    ray.get(actor.setup.remote())  # type: ignore

    ref1 = actor.process_data.remote(task1)  # type: ignore
    ref2 = actor.process_data.remote(task2)  # type: ignore

    # Get dynamic results, ignore actual result refs for this test
    metadata1, *_ = ray.get(ref1)
    metadata2, *_ = ray.get(ref2)

    print(metadata1)
    print(metadata2)


def test_multiple_tasks_with_exceptions(ray_cluster) -> None:
    # Use lists for data_refs
    task1 = stage_worker.TaskData([ray.put(False), ray.put(False)])  # Task that should succeed
    task2 = stage_worker.TaskData([ray.put(False), ray.put(True)])  # Task that should fail
    actor = stage_worker.StageWorker.options(max_concurrency=1000).remote(
        _ExceptionStage(),
        stage.Params(),
        resources.Worker(
            "my_worker",
            "my_stage",
            resources.WorkerResources("my_node", 1.0, [], [], []),
        ),
        VerbosityLevel.INFO,
    )
    ray.get(actor.setup.remote())  # type: ignore

    ref1 = actor.process_data.remote(task1)  # type: ignore
    ref2 = actor.process_data.remote(task2)  # type: ignore

    # Get dynamic results for the successful task
    metadata1, *result_refs1 = ray.get(ref1)
    # Get actual results from refs
    results1 = ray.get(result_refs1)
    # Since task1 had [False, False], it should return [1, 1]
    assert results1 == [1, 1]

    # Get dynamic results for the successful task
    metadata2, *_ = ray.get(ref2)

    # Check that the second task raises an exception upon ray.get()
    with pytest.raises(Exception):  # noqa: B017
        _ = ray.get(metadata2)

    print(results1)
    print(ray.get(metadata1))
