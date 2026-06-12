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

"""Static per-stage pipeline shape for the saturation-aware scheduler.

Captured once at setup and read every cycle, so the scheduler iterates one
ordered collection of typed stages instead of several parallel per-stage
tuples (names, batch sizes, GPU flags) indexed in lockstep.
"""

from collections.abc import Sequence
from typing import Any, Self

import attrs

from cosmos_xenna.pipelines.private import specs
from cosmos_xenna.utils import attrs_utils


@attrs.frozen
class StageShape:
    """Static shape of one pipeline stage.

    Attributes:
        name: Canonical stage name (estimator and floor-state key).
        batch_size: Input items consumed per processed batch.
        is_gpu: Whether the stage's workers hold GPU resources.
        is_manual: Whether the operator pinned the worker count
            (``num_workers`` / ``num_workers_per_node``); pinned stages are
            exempt from the cold-start ramp.
    """

    name: str
    batch_size: int = attrs.field(validator=attrs_utils.validate_positive_int)
    is_gpu: bool
    is_manual: bool


@attrs.frozen
class PipelineShape:
    """Per-stage static shape in pipeline order.

    Attributes:
        stages: One :class:`StageShape` per pipeline stage, in order.
    """

    stages: tuple[StageShape, ...]

    @classmethod
    def from_stage_specs(cls, stage_specs: Sequence[specs.StageSpec[Any, Any]]) -> Self:
        """Build the shape from ordered pipeline stage specs.

        Args:
            stage_specs: The pipeline's stage specs, in execution order.

        Returns:
            The immutable per-stage shape.
        """
        return cls(
            stages=tuple(
                StageShape(
                    name=spec.name(index),
                    batch_size=spec.stage.stage_batch_size,
                    is_gpu=float(spec.stage.required_resources.gpus) > 0.0,
                    is_manual=spec.num_workers is not None or spec.num_workers_per_node is not None,
                )
                for index, spec in enumerate(stage_specs)
            )
        )

    @property
    def num_stages(self) -> int:
        """Return the number of stages."""
        return len(self.stages)
