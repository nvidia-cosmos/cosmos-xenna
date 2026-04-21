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

"""Helpers for unwrapping ``asyncio.TaskGroup`` exception groups."""

import asyncio
from collections.abc import Iterator

MAX_TASKGROUP_FAILURES = 10
"""Cap on failure leaves kept in the returned ``ExceptionGroup``."""


class TruncatedFailuresError(RuntimeError):
    """Synthetic leaf appended when a TaskGroup ``ExceptionGroup`` is capped."""


def iter_exception_group_leaves(eg: BaseExceptionGroup) -> Iterator[BaseException]:
    """Yield non-group leaves from a possibly-nested ``BaseExceptionGroup``."""
    for exc in eg.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            yield from iter_exception_group_leaves(exc)
        else:
            yield exc


def unwrap_taskgroup_exception_group(eg: BaseExceptionGroup) -> Exception:
    """Collapse a TaskGroup ``BaseExceptionGroup`` into a single ``Exception``.

    Drops ``CancelledError`` leaves; wraps non-``Exception`` leaves in
    ``RuntimeError``; returns the bare exception when one remains, else a
    flat ``ExceptionGroup`` capped at ``MAX_TASKGROUP_FAILURES`` (with a
    trailing ``TruncatedFailuresError`` when truncated). Returns a
    ``RuntimeError`` when only cancellations are present.
    """
    leaves: list[Exception] = []
    for exc in iter_exception_group_leaves(eg):
        if isinstance(exc, asyncio.CancelledError):
            continue
        leaves.append(
            exc if isinstance(exc, Exception) else RuntimeError(f"Non-Exception failure inside TaskGroup: {exc!r}")
        )

    if not leaves:
        return RuntimeError(f"TaskGroup exited with cancellations only: {eg!r}")
    if len(leaves) == 1:
        return leaves[0]

    if len(leaves) > MAX_TASKGROUP_FAILURES:
        dropped = len(leaves) - MAX_TASKGROUP_FAILURES
        leaves = leaves[:MAX_TASKGROUP_FAILURES]
        leaves.append(TruncatedFailuresError(f"... and {dropped} more failures truncated"))

    return ExceptionGroup("Multiple failures inside TaskGroup", leaves)
