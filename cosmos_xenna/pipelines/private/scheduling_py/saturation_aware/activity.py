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

"""Active pipeline-work snapshot for the saturation-aware scale-down floor.

The scheduler grows on one signal and releases on another:

- queued backlog (inter-stage input queue depth) drives demand sizing;
- active work (queued backlog + pool-queued + in-flight) drives the
  scale-down release gate, so a downstream stage is not released while
  upstream work is still in flight even though its own input queue is
  momentarily empty.

This module owns only the second signal as native-free value objects. All
counts are normalized to stage-input sample units so they compose with
:func:`chain.whole_chain_stock`: a queue depth is already in input samples,
while pool-queued tasks and in-flight slots count batches and so multiply by
the stage batch size.
"""

from collections.abc import Sequence
from typing import Self

import attrs

from cosmos_xenna.utils import attrs_utils


@attrs.frozen
class StageActivity:
    """One stage's pending and in-flight work, in stage-input sample units.

    Attributes:
        queue_depth_samples: Inter-stage input queue depth, already in
            stage-input samples.
        pool_queued_tasks: Batches queued in the pool awaiting a free slot.
        inflight_slots: Logical in-flight tasks (one per running batch; one per
            SPMD worker group), not raw actor-rank slots.
        batch_size: Stage input items consumed per batch (each > 0); converts
            the batch-counted terms to stage-input samples.
    """

    queue_depth_samples: float = attrs.field(validator=attrs.validators.ge(0.0))
    pool_queued_tasks: int = attrs.field(validator=attrs.validators.ge(0))
    inflight_slots: int = attrs.field(validator=attrs.validators.ge(0))
    batch_size: int = attrs.field(validator=attrs_utils.validate_positive_int)

    def active_depth(self) -> float:
        """Return total at-this-stage work, in stage-input samples.

        The queue depth is already in samples; pool-queued and in-flight
        batches are scaled by the batch size to the same unit.
        """
        return self.queue_depth_samples + (self.pool_queued_tasks + self.inflight_slots) * self.batch_size


@attrs.frozen
class PipelineActivitySnapshot:
    """Per-stage active-work snapshot in pipeline order.

    Attributes:
        stages: One :class:`StageActivity` per pipeline stage, in order.
    """

    stages: tuple[StageActivity, ...]

    @classmethod
    def from_counts(
        cls,
        queue_depths: Sequence[float],
        pool_queued_tasks: Sequence[int],
        inflight_slots: Sequence[int],
        batch_sizes: Sequence[int],
    ) -> Self:
        """Build a snapshot from per-stage queue, pool, and batch counters."""
        return cls(
            stages=tuple(
                StageActivity(
                    queue_depth_samples=float(queue_depth),
                    pool_queued_tasks=pool_queued,
                    inflight_slots=inflight,
                    batch_size=batch_size,
                )
                for queue_depth, pool_queued, inflight, batch_size in zip(
                    queue_depths, pool_queued_tasks, inflight_slots, batch_sizes, strict=True
                )
            )
        )

    def active_depths(self) -> tuple[float, ...]:
        """Return each stage's active depth, in stage-input samples."""
        return tuple(stage.active_depth() for stage in self.stages)
