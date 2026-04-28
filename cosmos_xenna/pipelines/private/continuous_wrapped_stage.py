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

"""Adapter bridging a user ``Stage + ContinuousInterface`` to ``stage.Interface``.

Two parallel class hierarchies meet here:

::

    User-facing                         Internal Xenna
    ===========                         ==============

    CuratorStage (specs.Stage)          stage.Interface
         |                                   |
    ContinuousInterface (mixin) -------+     |
         |                             |     |
    MyGpuStage                         |     |
    (user writes this)                 v     v
                                  ContinuousWrappedStage
                                  (this module)

``ContinuousWrappedStage`` inherits **both**:

* ``stage.Interface``  - so ``StageWorker`` can manage it (call
  ``setup_on_node``, ``setup``); ``process_data`` is implemented as a
  hard error to catch any caller that bypasses continuous mode.
* ``ContinuousInterface``  - so ``StageWorker`` can detect continuous
  mode through the same structural ``isinstance`` check it uses on the
  raw user stage, and so the ``run_continuous`` coroutine remains the
  single dispatch surface.

The factory ``make_actor_pool_stage_from_stage_spec`` in ``specs.py``
constructs this wrapper exactly when the user stage already mixes in
``ContinuousInterface``; otherwise the legacy ``WrappedStage`` is used.
"""

import asyncio
import typing

from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.ray_utils import stage
from cosmos_xenna.ray_utils.continuous_stage import (
    ContinuousInterface,
    ContinuousTaskInput,
    ContinuousTaskOutput,
)


class ContinuousWrappedStage(stage.Interface, ContinuousInterface):
    """Adapter that lets ``StageWorker`` drive a continuous-mode user stage.

    Forwards lifecycle hooks (``setup_on_node``, ``setup``) and the
    continuous run-loop (``run_continuous``) to the wrapped user stage,
    and refuses ``process_data`` so that an accidental batch-mode caller
    fails loudly instead of silently bypassing continuous mode.

    The wrapped stage is held by reference (no copy); its identity is
    preserved so that downstream consumers that introspect
    ``CuratorStageSpec.stage`` keep seeing the user's class - the
    wrapper is purely an internal detail of the worker layer.
    """

    def __init__(self, stage_obj: typing.Any) -> None:
        """Wrap a user stage that mixes in ``ContinuousInterface``.

        Args:
            stage_obj: The user-facing stage. The constructor does NOT
                isinstance-check it because the caller (the spec factory)
                has already verified the marker; checking here too would
                duplicate the contract without adding safety.

        """
        self._stage = stage_obj

    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        """Forward the once-per-node setup hook to the wrapped stage."""
        self._stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        """Forward the once-per-worker setup hook to the wrapped stage."""
        self._stage.setup(worker_metadata)

    def process_data(self, data: list[typing.Any]) -> list[typing.Any]:
        """Refuse batch-mode dispatch - continuous stages own their own loop.

        The framework's ``StageWorker`` selects the run loop by
        ``isinstance(stage, ContinuousInterface)`` and never reaches this
        method when it dispatches correctly. Raising preserves a loud
        failure mode if some future code path forgets the dispatch
        check.
        """
        msg = (
            "ContinuousWrappedStage.process_data() must not be called - the worker should dispatch into "
            "run_continuous() based on the ContinuousInterface marker. This indicates a bug in the StageWorker "
            "dispatch logic or in a custom ActorPool integration."
        )
        raise NotImplementedError(msg)

    async def run_continuous(
        self,
        input_queue: asyncio.Queue[ContinuousTaskInput],
        output_queue: asyncio.Queue[ContinuousTaskOutput],
        stop_event: asyncio.Event,
    ) -> None:
        """Forward the continuous run-loop to the wrapped stage."""
        await self._stage.run_continuous(input_queue, output_queue, stop_event)
