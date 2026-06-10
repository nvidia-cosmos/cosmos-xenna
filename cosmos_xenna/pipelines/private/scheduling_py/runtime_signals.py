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

"""Scheduler-agnostic per-cycle runtime work signals.

The streaming autoscaler gathers per-stage queue, pool, and in-flight counters
each cycle and delivers them to any scheduler that opts into :class:`RuntimeAware`
via :func:`deliver`. Both the signals and the capability are generic pipeline
telemetry: the autoscaler never names a concrete scheduler implementation, and a
scheduler that ignores runtime signals (such as the fragmentation solver) is
simply skipped.
"""

import typing

import attrs


@attrs.frozen
class RuntimeSignals:
    """Per-cycle runtime work signals for one pipeline, in stage order.

    Attributes:
        queue_depths: Per-stage inter-stage input queue depth, in input samples.
        pool_queued_tasks: Per-stage batches queued in the pool awaiting a slot.
        inflight_slots: Per-stage count of logical in-flight batches/tasks (one
            per running task), not raw actor-rank slots.
        batch_sizes: Per-stage input items consumed per batch.
    """

    queue_depths: tuple[float, ...]
    pool_queued_tasks: tuple[int, ...]
    inflight_slots: tuple[int, ...]
    batch_sizes: tuple[int, ...]

    def __attrs_post_init__(self) -> None:
        """Reject ragged signal arrays so a malformed snapshot fails fast."""
        lengths = {
            len(self.queue_depths),
            len(self.pool_queued_tasks),
            len(self.inflight_slots),
            len(self.batch_sizes),
        }
        if len(lengths) != 1:
            raise ValueError(f"runtime signal arrays must share one length, got {sorted(lengths)}")


@typing.runtime_checkable
class RuntimeAware(typing.Protocol):
    """A scheduler that consumes runtime signals before each autoscale cycle."""

    def observe_runtime(self, signals: RuntimeSignals) -> None: ...


def deliver(algorithm: object, signals: RuntimeSignals) -> None:
    """Deliver signals to ``algorithm`` only if it implements :class:`RuntimeAware`.

    A scheduler without the capability is skipped, so callers stay free of
    concrete scheduler types.
    """
    if isinstance(algorithm, RuntimeAware):
        algorithm.observe_runtime(signals)
