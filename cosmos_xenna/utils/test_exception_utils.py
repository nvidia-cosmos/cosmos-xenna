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

"""Tests for ``cosmos_xenna.utils.exception_utils``."""

import asyncio

from cosmos_xenna.utils import exception_utils as eu


class TestUnwrapTaskgroupExceptionGroup:
    """Verify TaskGroup ``ExceptionGroup`` unwrap semantics."""

    def test_returns_single_non_cancelled_exception(self) -> None:
        """Return the concrete exception when exactly one non-cancelled error exists."""
        eg = BaseExceptionGroup("outer", [RuntimeError("boom"), asyncio.CancelledError()])

        result = eu.unwrap_taskgroup_exception_group(eg)

        assert isinstance(result, RuntimeError)
        assert str(result) == "boom"

    def test_returns_exception_group_for_multiple_non_cancelled_errors(self) -> None:
        """Preserve multi-failure detail by returning a flat ``ExceptionGroup``."""
        eg = BaseExceptionGroup(
            "outer",
            [
                BaseExceptionGroup("inner", [ValueError("left"), RuntimeError("right")]),
                asyncio.CancelledError(),
            ],
        )

        result = eu.unwrap_taskgroup_exception_group(eg)

        assert isinstance(result, ExceptionGroup)
        assert [type(exc) for exc in result.exceptions] == [ValueError, RuntimeError]
        assert [str(exc) for exc in result.exceptions] == ["left", "right"]

    def test_returns_runtime_error_when_only_cancellations_exist(self) -> None:
        """Return a clear fallback error when only cancellation failures exist."""
        eg = BaseExceptionGroup("outer", [asyncio.CancelledError()])

        result = eu.unwrap_taskgroup_exception_group(eg)

        assert isinstance(result, RuntimeError)
        assert "cancellations only" in str(result)

    def test_truncates_excess_failures_with_marker_leaf(self) -> None:
        """Cap leaves at ``MAX_TASKGROUP_FAILURES`` and append a truncation marker."""
        cap = eu.MAX_TASKGROUP_FAILURES
        excess = 3
        eg = BaseExceptionGroup("outer", [RuntimeError(f"f{i}") for i in range(cap + excess)])

        result = eu.unwrap_taskgroup_exception_group(eg)

        assert isinstance(result, ExceptionGroup)
        assert len(result.exceptions) == cap + 1
        assert isinstance(result.exceptions[-1], eu.TruncatedFailuresError)
        assert f"{excess} more" in str(result.exceptions[-1])
