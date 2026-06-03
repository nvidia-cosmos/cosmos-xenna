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

"""Unit tests for the scheduler-agnostic runtime-signal seam."""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py import runtime_signals


class _RecordingScheduler:
    """RuntimeAware stub that records the last signals it received."""

    def __init__(self) -> None:
        self.received: runtime_signals.RuntimeSignals | None = None

    def observe_runtime(self, signals: runtime_signals.RuntimeSignals) -> None:
        self.received = signals


class _PlainScheduler:
    """Scheduler without the runtime-signal capability."""


def _signals() -> runtime_signals.RuntimeSignals:
    return runtime_signals.RuntimeSignals(
        queue_depths=(2.0, 0.0), pool_queued_tasks=(1, 3), inflight_slots=(2, 4), batch_sizes=(4, 8)
    )


def test_runtime_signals_rejects_ragged_arrays() -> None:
    with pytest.raises(ValueError, match="one length"):
        runtime_signals.RuntimeSignals(
            queue_depths=(0.0, 0.0), pool_queued_tasks=(0,), inflight_slots=(0, 0), batch_sizes=(1, 1)
        )


def test_deliver_routes_to_runtime_aware_scheduler() -> None:
    scheduler = _RecordingScheduler()
    signals = _signals()
    runtime_signals.deliver(scheduler, signals)
    assert scheduler.received is signals


def test_deliver_is_a_noop_for_plain_scheduler() -> None:
    """A scheduler without the capability is skipped, not errored."""
    runtime_signals.deliver(_PlainScheduler(), _signals())


def test_runtime_aware_is_structural() -> None:
    assert isinstance(_RecordingScheduler(), runtime_signals.RuntimeAware)
    assert not isinstance(_PlainScheduler(), runtime_signals.RuntimeAware)
