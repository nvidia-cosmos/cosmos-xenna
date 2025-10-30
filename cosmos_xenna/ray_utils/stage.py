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

import abc
import contextlib
from collections.abc import Iterator
from typing import Any, Optional

import attrs

from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.ray_utils import runtime_envs


@attrs.define
class Openable(abc.ABC):
    @abc.abstractmethod
    def open(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


@contextlib.contextmanager
def open_context(openable: Optional[Openable]) -> Iterator[None]:
    if openable is None:
        yield
    else:
        openable.open()
        try:
            yield
        finally:
            openable.close()


@attrs.define
class Params:
    shape: resources.WorkerShape
    stage_batch_size: int = 1
    slots_per_actor: int = 2
    # Maxmum lifetime in minutes before we internally terminate and restart a worker. 0 means disabled.
    # Not enabled for SPMD actor pools.
    worker_max_lifetime_m: int = 0
    # Restart interval in minutes between two consecutive over-lifetime restart within each actor pool.
    # Not enabled for SPMD actor pools.
    worker_restart_interval_m: int = 1
    # The name of the worker stage.
    name: str = "default"
    # Number of times to retry worker setup on node. Defaults to 1.
    num_node_setup_retries: int = 1
    # Number of times to retry worker setup. Defaults to 1.
    num_setup_retries: int = 1
    # Number of times to retry task execution. Defaults to 1.
    num_run_retries: int = 1
    # Whether to ignore failures during processing. Defaults to False.
    ignore_failures: bool = False
    # Whether to restart workers on failure. Defaults to False.
    restart_workers_on_failure: bool = False
    runtime_env: runtime_envs.RuntimeEnv = attrs.field(factory=runtime_envs.RuntimeEnv)
    logging_context: Optional[Openable] = None
    # Sometimes, setup() can fail sporatically. For example, this can happen due to distributed filesystem flakiness.
    # It can be helpful to ignore these failures and retry. If this is non-None, we will retry and then fail the
    # pipeline if a stage fails to setup more than this percentage. For example, if this value is 50 and we try to start
    # 10 actors and 6 of them fail, we will fail the pipeline. If 4 of them fail, we will continue running the pipeline.
    max_setup_failure_percentage: Optional[float] = None
    # If true, modify the CUDA_VISIBLE_DEVICES environment variable.
    modify_cuda_visible_devices_env_var: bool = True


class Interface(abc.ABC):
    @abc.abstractmethod
    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        pass

    @abc.abstractmethod
    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        pass

    @abc.abstractmethod
    def process_data(self, data: list[Any]) -> list[Any]:
        pass
