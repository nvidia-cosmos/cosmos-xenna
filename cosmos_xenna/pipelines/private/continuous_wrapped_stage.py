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

"""Adapter that bridges ``specs.Stage + ContinuousInterface`` to ``stage.Interface``.

::

    User-facing class hierarchy         Internal Xenna hierarchy
    ===========================         ========================

    CuratorStage (specs.Stage)          stage.Interface
         |                                   |
    ContinuousInterface (mixin)         ContinuousInterface (mixin)
         |                                   |
    MyGpuStage                          ContinuousWrappedStage
    (user writes this)                  (this module -- adapter)

``ContinuousWrappedStage`` inherits both ``stage.Interface`` (so
``StageWorker`` can manage it) and ``ContinuousInterface`` (so
``StageWorker`` can detect continuous mode via ``isinstance``).
"""

import asyncio
from typing import Any

from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.ray_utils import stage
from cosmos_xenna.ray_utils.continuous_stage import ContinuousInterface, ContinuousTaskInput, ContinuousTaskOutput


class ContinuousWrappedStage(stage.Interface, ContinuousInterface):
    """Adapt a ``Stage + ContinuousInterface`` for ``StageWorker``.

    Bridges ``specs.Stage`` (user-facing) to ``stage.Interface`` (internal),
    preserving continuous-mode support through dual inheritance.  The factory
    function ``make_actor_pool_stage_from_stage_spec`` in ``specs.py`` creates
    this wrapper when the user's stage implements ``ContinuousInterface``.
    """

    def __init__(self, stage_obj: Any) -> None:
        """Wrap a user-facing ``Stage`` that also implements ``ContinuousInterface``."""
        self._stage = stage_obj

    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        """Delegate to wrapped stage's ``setup_on_node``."""
        self._stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        """Delegate to wrapped stage's ``setup``."""
        self._stage.setup(worker_metadata)

    def process_data(self, data: list[Any]) -> list[Any]:
        """Not used -- continuous mode bypasses ``process_data``."""
        raise NotImplementedError("ContinuousWrappedStage uses run_continuous(), not process_data()")

    @property
    def continuous_input_queue_size(self) -> int:
        """Delegate to the wrapped stage's ``continuous_input_queue_size``."""
        return self._stage.continuous_input_queue_size

    async def run_continuous(
        self,
        input_queue: asyncio.Queue[ContinuousTaskInput],
        output_queue: asyncio.Queue[ContinuousTaskOutput],
        stop_event: asyncio.Event,
    ) -> None:
        """Delegate to the wrapped stage's ``run_continuous``."""
        await self._stage.run_continuous(input_queue, output_queue, stop_event)
