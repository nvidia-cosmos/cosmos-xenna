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

"""Unit tests for :func:`_kill_actors_and_reap_parallel`.

Ray is not imported: the helper's orchestration is a thin
``ThreadPoolExecutor`` fan-out and is unit-testable with mocks. The
underlying ``_kill_actor_and_reap`` / ``_reap_pids`` behavior is covered
by ``reap_pids_test.py``.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

import cosmos_xenna.ray_utils.actor_pool as ap_module
from cosmos_xenna.ray_utils.actor_pool import _kill_actors_and_reap_parallel


@pytest.mark.L1
@pytest.mark.CPU
def test_empty_kills_is_noop() -> None:
    """No tuples -> no thread pool, no calls."""
    with (
        patch.object(ap_module, "_kill_actor_and_reap") as kill_mock,
        patch.object(ap_module.concurrent.futures, "ThreadPoolExecutor") as pool_mock,
    ):
        _kill_actors_and_reap_parallel([])

    kill_mock.assert_not_called()
    pool_mock.assert_not_called()


@pytest.mark.L1
@pytest.mark.CPU
def test_forwards_each_tuple_verbatim() -> None:
    """Each tuple is passed as positional args to ``_kill_actor_and_reap``."""
    kills: list[tuple[object, str, str, float]] = [
        (object(), "node-a", "ready actor 0", 0.0),
        (object(), "node-b", "pending actor 1", 30.0),
        (object(), "node-c", "waiting actor 2", 15.0),
    ]

    with patch.object(ap_module, "_kill_actor_and_reap") as kill_mock:
        _kill_actors_and_reap_parallel(kills)  # type: ignore[arg-type]

    assert kill_mock.call_count == len(kills)
    # ThreadPoolExecutor.map preserves input order in results but not
    # necessarily in call ordering; compare as a set of positional-arg tuples.
    observed = {tuple(call.args) for call in kill_mock.call_args_list}
    expected = {tuple(k) for k in kills}
    assert observed == expected


@pytest.mark.L1
@pytest.mark.CPU
def test_kills_run_in_parallel() -> None:
    """Wall time reflects concurrent execution."""
    per_kill_s = 0.3
    n = 4

    def slow_kill(*_args: object, **_kwargs: object) -> None:
        time.sleep(per_kill_s)

    kills: list[tuple[object, str, str, float]] = [(object(), f"node-{i}", f"actor {i}", 0.0) for i in range(n)]

    with patch.object(ap_module, "_kill_actor_and_reap", side_effect=slow_kill):
        start = time.monotonic()
        _kill_actors_and_reap_parallel(kills)  # type: ignore[arg-type]
        elapsed = time.monotonic() - start

    # Parallel wall time should be ~``per_kill_s`` regardless of ``n``; use a
    # generous headroom to absorb CI scheduling noise.
    assert elapsed < 0.6, f"expected parallel wall time < 0.6s but got {elapsed:.3f}s"


@pytest.mark.L1
@pytest.mark.CPU
def test_exception_in_one_worker_propagates() -> None:
    """A worker exception surfaces to the caller (via ``list(ex.map(...))``)."""
    kills: list[tuple[object, str, str, float]] = [
        (object(), "node-0", "actor 0", 0.0),
        (object(), "node-1", "actor 1", 0.0),
        (object(), "node-2", "actor 2", 0.0),
    ]
    poisoned_label = kills[1][2]

    def maybe_raise(_actor: object, _node: str, label: str, _grace: float) -> None:
        if label == poisoned_label:
            raise RuntimeError("boom")

    with (
        patch.object(ap_module, "_kill_actor_and_reap", side_effect=maybe_raise),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _kill_actors_and_reap_parallel(kills)  # type: ignore[arg-type]


@pytest.mark.L1
@pytest.mark.CPU
def test_max_parallel_bounds_peak_concurrency() -> None:
    """``max_parallel`` is a hard cap on concurrent worker invocations."""
    max_parallel = 2
    kills: list[tuple[object, str, str, float]] = [(object(), f"node-{i}", f"actor {i}", 0.0) for i in range(10)]

    lock = threading.Lock()
    active = 0
    peak = 0

    def observe_concurrency(*_args: object, **_kwargs: object) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Hold the slot long enough for additional workers to overlap.
        time.sleep(0.05)
        with lock:
            active -= 1

    with patch.object(ap_module, "_kill_actor_and_reap", side_effect=observe_concurrency):
        _kill_actors_and_reap_parallel(kills, max_parallel=max_parallel)  # type: ignore[arg-type]

    assert peak <= max_parallel, f"peak concurrency {peak} exceeded cap {max_parallel}"
    assert peak >= 1
