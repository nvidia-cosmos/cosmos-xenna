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

"""Service value object passed to :class:`ManualPhase`."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.executors import (
    ManualDeleteExecutor,
    ManualGrowExecutor,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel


@attrs.frozen
class ManualServices:
    """Service view consumed by :class:`ManualPhase` (operator-driven shrink + grow).

    Manual stages are owned end-to-end by ``ManualPhase``; the
    phase delegates the per-stage delete and grow loops to the
    two executors injected here.

    Attributes:
        pipeline: Immutable post-setup pipeline shape (problem,
            stage names, effective per-stage configs).
        pipeline_name: Pipeline tag used in log lines and
            Prometheus labels.
        delete_executor: Per-stage delete strategy; deletes
            surplus manual workers youngest-first.
        grow_executor: Per-stage grow strategy; owns the manual
            allocation-failure gate and the per-stage add loop.

    """

    pipeline: PipelineModel
    pipeline_name: str
    delete_executor: ManualDeleteExecutor
    grow_executor: ManualGrowExecutor


__all__ = ("ManualServices",)
