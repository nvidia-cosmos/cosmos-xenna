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

"""Task lifecycle data classes used across the Xenna runtime."""

import typing

import attrs
import ray

T = typing.TypeVar("T")


@attrs.define
class TaskData(typing.Generic[T]):
    """Input wrapper for one scheduled task.

    Attributes:
        data_refs: Ray ``ObjectRef`` handles for serialized payload items.
            The runtime resolves and deserializes these refs before invoking
            stage logic.

    """

    data_refs: list["ray.ObjectRef[T]"]


@attrs.define
class TimingInfo:
    """Wall-clock phase timestamps for a single task lifecycle."""

    requested_s: float = 0.0
    pull_s: float = 0.0
    deserialize_start_s: float = 0.0
    deserialize_end_s: float = 0.0
    process_start_time_s: float = 0.0
    process_end_time_s: float = 0.0

    @property
    def pull_dur(self) -> float | None:
        """Seconds Ray spent making the data available locally, or ``None``."""
        if self.pull_s and self.requested_s:
            return self.pull_s - self.requested_s
        return None

    @property
    def deserialize_dur(self) -> float | None:
        """Seconds spent in ``ray.get`` deserializing the data, or ``None``."""
        if self.deserialize_end_s and self.deserialize_start_s:
            return self.deserialize_end_s - self.deserialize_start_s
        return None

    @property
    def process_dur(self) -> float | None:
        """Seconds spent inside the stage's processing logic, or ``None``."""
        if self.process_end_time_s and self.process_start_time_s:
            return self.process_end_time_s - self.process_start_time_s
        return None


@attrs.define
class FailureInfo:
    """Per-task outcome flags.

    Attributes:
        should_process_further: When ``False``, the task result is dropped
            instead of being forwarded downstream.
        should_restart_worker: When ``True``, the worker handling this task
            should be replaced.
        failures_return_nones: Reserved for future compatibility.

    """

    should_process_further: bool
    should_restart_worker: bool
    failures_return_nones: bool = False


@attrs.define
class TaskDataInfo:
    """Per-task input-size accounting (bytes in the Ray object store)."""

    serialized_input_size: int


@attrs.define
class TaskResultMetadata:
    """Aggregate metadata produced per task.

    Attributes:
        timing: Wall-clock phase timestamps for the task.
        failure_info: Outcome flags for downstream task handling.
        task_data_info: Input size accounting in bytes.
        num_returns: Number of output items produced for this task.
        rate_duration_s: Optional service-rate duration override used by
            throughput estimation; ``None`` means use ``timing.process_dur``.

    """

    timing: TimingInfo
    failure_info: FailureInfo
    task_data_info: TaskDataInfo
    num_returns: int
    rate_duration_s: float | None = None
