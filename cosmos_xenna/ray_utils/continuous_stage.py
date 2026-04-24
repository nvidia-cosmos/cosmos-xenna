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

"""Continuous-mode task contracts for Xenna stages.

Stages opt in by implementing ``ContinuousInterface.run_continuous``.
The runtime feeds tasks through ``input_queue`` and collects results from
``output_queue`` until ``stop_event`` is set.
"""

import abc
import asyncio
import typing

import attrs

from cosmos_xenna.ray_utils.task_metadata import TimingInfo


@attrs.define
class ContinuousTaskInput:
    """Single task payload delivered to ``run_continuous``.

    Attributes:
        task_id: Task identifier that must be echoed on output.
        data: Deserialized task payload.
        timing: Per-task timing container.
        object_sizes: Serialized input sizes in bytes.

    """

    task_id: str
    data: list[typing.Any]
    timing: TimingInfo
    object_sizes: list[int]


@attrs.define
class ContinuousTaskOutput:
    """Completed task payload emitted by ``run_continuous``.

    Attributes:
        task_id: Identifier matching the originating input item.
        out_data: Stage output payload.
        timing: Per-task timing container.
        object_sizes: Serialized input sizes in bytes.

    """

    task_id: str
    out_data: list[typing.Any]
    timing: TimingInfo
    object_sizes: list[int]


class ContinuousInterface(abc.ABC):
    """Mark a stage as continuous-mode capable."""

    @abc.abstractmethod
    async def run_continuous(
        self,
        input_queue: asyncio.Queue[ContinuousTaskInput],
        output_queue: asyncio.Queue[ContinuousTaskOutput],
        stop_event: asyncio.Event,
    ) -> None:
        """Process tasks until ``stop_event`` is set and return cleanly."""
