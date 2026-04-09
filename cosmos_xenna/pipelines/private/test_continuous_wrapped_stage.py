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

"""Unit tests for ContinuousWrappedStage adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cosmos_xenna.pipelines.private.continuous_wrapped_stage import ContinuousWrappedStage
from cosmos_xenna.ray_utils.continuous_stage import ContinuousInterface
from cosmos_xenna.ray_utils import stage


class TestContinuousWrappedStage:
    """Verify delegation, isinstance detection, and process_data guard."""

    def _make_mock_stage(self) -> MagicMock:
        """Create a mock user stage with ContinuousInterface methods."""
        mock = MagicMock()
        mock.run_continuous = AsyncMock()
        return mock

    def test_isinstance_stage_interface(self) -> None:
        """Wrapper should be an instance of stage.Interface."""
        wrapped = ContinuousWrappedStage(self._make_mock_stage())
        assert isinstance(wrapped, stage.Interface)

    def test_isinstance_continuous_interface(self) -> None:
        """Wrapper should be an instance of ContinuousInterface for detection."""
        wrapped = ContinuousWrappedStage(self._make_mock_stage())
        assert isinstance(wrapped, ContinuousInterface)

    def test_setup_on_node_delegates(self) -> None:
        """setup_on_node should delegate to wrapped stage."""
        inner = self._make_mock_stage()
        wrapped = ContinuousWrappedStage(inner)

        node_info = MagicMock()
        worker_meta = MagicMock()
        wrapped.setup_on_node(node_info, worker_meta)

        inner.setup_on_node.assert_called_once_with(node_info, worker_meta)

    def test_setup_delegates(self) -> None:
        """setup should delegate to wrapped stage."""
        inner = self._make_mock_stage()
        wrapped = ContinuousWrappedStage(inner)

        worker_meta = MagicMock()
        wrapped.setup(worker_meta)

        inner.setup.assert_called_once_with(worker_meta)

    def test_process_data_raises_not_implemented(self) -> None:
        """process_data should raise NotImplementedError."""
        wrapped = ContinuousWrappedStage(self._make_mock_stage())
        with pytest.raises(NotImplementedError, match="run_continuous"):
            wrapped.process_data([])

    def test_run_continuous_delegates(self) -> None:
        """run_continuous should delegate to the wrapped stage's run_continuous."""
        inner = self._make_mock_stage()
        wrapped = ContinuousWrappedStage(inner)

        input_q = asyncio.Queue()
        output_q = asyncio.Queue()
        stop = asyncio.Event()

        runner = asyncio.Runner()
        try:
            runner.run(wrapped.run_continuous(input_q, output_q, stop))
        finally:
            runner.close()

        inner.run_continuous.assert_awaited_once_with(input_q, output_q, stop)
