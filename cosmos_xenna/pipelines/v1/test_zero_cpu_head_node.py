# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Run a pipeline on a cluster whose head node has no CPUs, like Anyscale.

On Anyscale the head node has no CPUs in Ray so that no work is scheduled on it. Xenna used to probe every node with a
one-CPU task and start a one-CPU monitor actor on every node, so `run_pipeline` failed with `TaskUnschedulableError`
before any stage started. It also counted a node's CPUs with `os.cpu_count()` when Ray reported none.
"""

import socket
import threading

import pytest
import ray
from ray.cluster_utils import Cluster

from cosmos_xenna.pipelines import v1 as pipelines_v1

_TIMEOUT_S = 120.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NodeIdStage(pipelines_v1.Stage):
    """Returns the ID of the node each task ran on."""

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def process_data(self, in_data: list[int]) -> list[str]:
        return [ray.get_runtime_context().get_node_id() for _ in in_data]


@pytest.mark.slow
def test_pipeline_does_not_use_head_node_without_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set before the nodes start, as `init_or_connect_to_cluster` would for a local cluster.
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("RAY_MAX_LIMIT_FROM_API_SERVER", "40000")
    monkeypatch.setenv("RAY_MAX_LIMIT_FROM_DATA_SOURCE", "40000")
    # Earlier pipeline tests in the same process leave their local Ray cluster running.
    ray.shutdown()
    head_node_args = {"num_cpus": 0, "num_gpus": 0, "include_dashboard": True, "dashboard_port": _free_port()}
    cluster = Cluster(initialize_head=True, head_node_args=head_node_args)
    try:
        head_node_id = cluster.head_node.node_id
        worker_node_id = cluster.add_node(num_cpus=2, num_gpus=0).node_id
        # A node with a GPU but no CPUs: no stage can run here, but the per-node probe still runs on it.
        cluster.add_node(num_cpus=0, num_gpus=1)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)

        spec = pipelines_v1.PipelineSpec(
            input_data=list(range(20)),
            stages=[_NodeIdStage()],
            config=pipelines_v1.PipelineConfig(logging_interval_s=1, return_last_stage_outputs=True),
        )
        outputs: list[str] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                outputs.extend(pipelines_v1.run_pipeline(spec) or [])
            except BaseException as e:  # noqa: BLE001 - re-raised in the test thread below
                errors.append(e)

        # A daemon thread so that a pipeline stuck waiting for a node fails the test instead of hanging it.
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(_TIMEOUT_S)
        if errors:
            raise errors[0]
        assert not thread.is_alive(), f"run_pipeline did not return within {_TIMEOUT_S}s"
        assert len(outputs) == 20
        assert head_node_id not in outputs
        assert set(outputs) == {worker_node_id}
    finally:
        ray.shutdown()
        cluster.shutdown()
