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

"""Continuous processing mode interface for GPU-bound pipeline stages.

Provides an opt-in mixin that replaces the blocking ``process_data(batch)``
contract with a persistent async loop.  This eliminates the "drain tail"
problem where GPU utilization drops while the slowest sequence in a batch
finishes.

::

    Batch mode (process_data):          Continuous mode (run_continuous):

    GPU  |████████                      GPU  |████████████████████████████
    util |    ████                       util |  task A    task B    task C
         |      ██   GAP   next              |   finishing  entering  ...
         +----------+-----+---->             +------------------------------>
         drain tail wastes GPU               near-constant GPU utilization

Usage::

    class MyCaptionStage(CuratorStage, ContinuousInterface):
        async def run_continuous(self, input_queue, output_queue, stop_event):
            while not stop_event.is_set():
                task_input = await input_queue.get()
                result = await self._process(task_input.data)
                await output_queue.put(ContinuousTaskOutput(...))

        def process_data(self, tasks):
            # Fallback for batch mode (optional, can raise NotImplementedError)
            ...
"""

import abc
import asyncio
from typing import Any

import attrs

from cosmos_xenna.ray_utils.stage_worker import TimingInfo


@attrs.define
class ContinuousTaskInput:
    """Task delivered to a continuous-mode stage via ``input_queue``.

    Attributes:
        task_id: Unique identifier matching the StageWorker's internal task UUID.
        data: Deserialized task payload (same content as ``process_data`` receives).
        timing: Timing info accumulated during download and deserialization phases.
        object_sizes: Serialized sizes of each Ray ObjectRef for metrics reporting.

    """

    task_id: str
    data: list[Any]
    timing: TimingInfo
    object_sizes: list[int]


@attrs.define
class ContinuousTaskOutput:
    """Completed task returned by a continuous-mode stage via ``output_queue``.

    Attributes:
        task_id: Must match the ``ContinuousTaskInput.task_id`` it originated from.
        out_data: Processed results (same shape as ``process_data`` return value).
        timing: Timing info -- ``process_start/end`` are set by the collector, not the stage.
        object_sizes: Pass-through from input for serialized-size metrics.

    """

    task_id: str
    out_data: list[Any]
    timing: TimingInfo
    object_sizes: list[int]


class ContinuousInterface(abc.ABC):
    """Mixin for stages that support continuous processing mode.

    Add this alongside ``CuratorStage`` (or ``specs.Stage``) to opt a stage
    into continuous mode.  Instead of repeated ``process_data(batch)`` calls,
    Xenna calls ``run_continuous()`` once on an asyncio event loop running on
    the processor thread.

    The stage receives ``asyncio.Queue`` objects for input/output and an
    ``asyncio.Event`` for stop signaling.  Xenna owns the event loop -- the
    stage just writes async code.

    The processor thread is dedicated to the stage for CUDA/NCCL thread
    affinity -- same guarantee as ``process_data()``.

    ::

        class MyCaptionStage(CuratorStage, ContinuousInterface):
            def process_data(self, task): ...  # batch fallback (optional)
            async def run_continuous(self, input_q, output_q, stop): ...
    """

    @property
    def continuous_input_queue_size(self) -> int:
        """Maximum number of deserialized tasks buffered for the stage.

        The StageWorker's feeder bridges ``deserialized_queue`` to an async
        ``input_queue`` of this size.  Backpressure propagates when the queue
        is full: feeder blocks, deserialized_queue backs up, downloads stall.

        Override to increase the buffer for GPU-bound stages where the
        download/deserialize pipeline is slower than inference.  Larger values
        reduce engine starvation gaps at the cost of holding more deserialized
        task data in memory.

        The framework auto-adjusts ``slots_per_actor`` to ``max(configured,
        continuous_input_queue_size + 2)`` so the download/deserialize
        pipeline stays ahead of the input queue.
        """
        return 4

    @abc.abstractmethod
    async def run_continuous(
        self,
        input_queue: asyncio.Queue[ContinuousTaskInput],
        output_queue: asyncio.Queue[ContinuousTaskOutput],
        stop_event: asyncio.Event,
    ) -> None:
        """Persistent async processing loop.

        Called once inside an asyncio event loop that Xenna creates on the
        processor thread (CUDA/NCCL affinity preserved).

        Must:
        - Pull tasks via ``await input_queue.get()``
        - Process them using asyncio concurrency (``create_task``, etc.)
        - Push completed results via ``await output_queue.put()``
        - Return when ``stop_event`` is set and all in-flight work is drained

        """
        ...
