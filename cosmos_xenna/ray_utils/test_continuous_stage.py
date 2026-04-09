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

"""Unit tests for continuous_stage.py -- ContinuousInterface, ContinuousTaskInput, ContinuousTaskOutput."""

import asyncio
from unittest.mock import MagicMock

import pytest

from cosmos_xenna.ray_utils.continuous_stage import ContinuousInterface, ContinuousTaskInput, ContinuousTaskOutput
from cosmos_xenna.ray_utils.stage_worker import TimingInfo


class TestContinuousTaskInput:
    """Verify ContinuousTaskInput attrs fields and construction."""

    def test_fields_assigned(self) -> None:
        """All fields should be stored as given."""
        timing = TimingInfo()
        inp = ContinuousTaskInput(
            task_id="abc-123",
            data=["payload"],
            timing=timing,
            object_sizes=[42, 99],
        )
        assert inp.task_id == "abc-123"
        assert inp.data == ["payload"]
        assert inp.timing is timing
        assert inp.object_sizes == [42, 99]

    def test_empty_data(self) -> None:
        """Empty data list should be allowed."""
        inp = ContinuousTaskInput(
            task_id="empty",
            data=[],
            timing=TimingInfo(),
            object_sizes=[],
        )
        assert inp.data == []
        assert inp.object_sizes == []


class TestContinuousTaskOutput:
    """Verify ContinuousTaskOutput attrs fields and construction."""

    def test_fields_assigned(self) -> None:
        """All fields should be stored as given."""
        timing = TimingInfo()
        out = ContinuousTaskOutput(
            task_id="out-1",
            out_data=[{"result": True}],
            timing=timing,
            object_sizes=[100],
        )
        assert out.task_id == "out-1"
        assert out.out_data == [{"result": True}]
        assert out.timing is timing
        assert out.object_sizes == [100]


class TestContinuousInterface:
    """Verify ContinuousInterface is abstract and enforces run_continuous."""

    def test_cannot_instantiate_directly(self) -> None:
        """Instantiating ContinuousInterface without implementing run_continuous should fail."""
        with pytest.raises(TypeError, match="run_continuous"):
            ContinuousInterface()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        """A concrete subclass implementing run_continuous should instantiate."""

        class ConcreteStage(ContinuousInterface):
            async def run_continuous(self, input_queue, output_queue, stop_event):
                pass

        stage = ConcreteStage()
        assert isinstance(stage, ContinuousInterface)

    def test_run_continuous_is_async(self) -> None:
        """run_continuous on a concrete subclass should be a coroutine function."""

        class ConcreteStage(ContinuousInterface):
            async def run_continuous(self, input_queue, output_queue, stop_event):
                pass

        stage = ConcreteStage()
        assert asyncio.iscoroutinefunction(stage.run_continuous)

    def test_isinstance_detection(self) -> None:
        """isinstance check should work for ContinuousInterface mixins."""

        class MyStage(ContinuousInterface):
            async def run_continuous(self, input_queue, output_queue, stop_event):
                pass

        stage = MyStage()
        assert isinstance(stage, ContinuousInterface)

        # A plain object should not match.
        assert not isinstance(MagicMock(), ContinuousInterface)
