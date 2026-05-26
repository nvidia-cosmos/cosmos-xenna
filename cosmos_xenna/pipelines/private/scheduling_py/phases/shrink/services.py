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

"""Service value object passed to :class:`SaturationShrinkPhase`."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.capacity import (
    CeilingCalculator,
    FloorCalculator,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_protection import BottleneckProtectionLogger


@attrs.frozen
class ShrinkServices:
    """Service view consumed by :class:`SaturationShrinkPhase`.

    The shrink phase removes workers driven by negative intent and
    hard-cap overflow, clamping by floor / fraction cap / donor
    warmup grace. It reads the per-stage floor and ceiling, the
    cycle's bottleneck snapshot and intent plan (via ``cycle``),
    and the bottleneck-protection latch directly.

    Attributes:
        pipeline: Immutable post-setup pipeline shape.
        pipeline_name: Pipeline tag for logs / labels.
        floors: Per-stage floor calculator (lower bound on shrink).
        ceilings: Per-stage hard ceiling calculator (drives forced
            shrink via ``ceiling_excess = current - ceiling``).
        bottleneck_protection: Streak-debounced INFO-once protection
            logger; the Shrink phase calls
            :meth:`BottleneckProtectionLogger.maybe_log` on entry
            and :meth:`BottleneckProtectionLogger.replace_snapshot`
            at the tail.

    """

    pipeline: PipelineModel
    pipeline_name: str
    floors: FloorCalculator
    ceilings: CeilingCalculator
    bottleneck_protection: BottleneckProtectionLogger


__all__ = ("ShrinkServices",)
