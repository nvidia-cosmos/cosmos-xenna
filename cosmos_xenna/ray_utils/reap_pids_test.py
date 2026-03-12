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

"""Tests for the _reap_pids PID-reuse safety guard and related actor kill helpers.

These tests focus on the edge cases introduced by the create_time guard:
- PID-reuse protection: don't kill a process if its create_time doesn't match the snapshot
- Fallback (None create_time): kill without protection when actor was unresponsive at snapshot time
- Already-dead PIDs: handled gracefully
- get_pid_tree() returns (pid, create_time) pairs that survive a round-trip through _reap_pids
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

import psutil
import pytest
import ray

from cosmos_xenna.ray_utils.actor_pool import _reap_pids


@pytest.fixture(scope="module")
def ray_session() -> Iterator[None]:
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


def _start_sleeper() -> subprocess.Popen:  # type: ignore[type-arg]
    """Start a long-lived subprocess to use as a kill target."""
    return subprocess.Popen(["sleep", "300"])


@pytest.mark.L1
@pytest.mark.CPU
def test_pid_reuse_guard_skips_mismatched_create_time(ray_session: None) -> None:
    """If create_time doesn't match the snapshot, the process must NOT be killed.

    This is the core PID-reuse scenario: the original actor died, the OS recycled
    its PID to a new innocent process, and _reap_pids runs with the old create_time.
    The mismatch should cause _reap_pids to skip the kill entirely.
    """
    proc = _start_sleeper()
    try:
        actual_create_time = psutil.Process(proc.pid).create_time()
        wrong_create_time = actual_create_time + 1000.0  # clearly wrong

        ray.get(_reap_pids.remote([(proc.pid, wrong_create_time)]))  # type: ignore[attr-defined]

        assert psutil.pid_exists(proc.pid), (
            f"pid={proc.pid} was killed despite create_time mismatch "
            f"(expected={wrong_create_time:.3f}, actual={actual_create_time:.3f})"
        )
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.L1
@pytest.mark.CPU
def test_pid_reuse_guard_kills_on_matching_create_time(ray_session: None) -> None:
    """If create_time matches the snapshot, the process IS killed.

    Normal operation: actor survived ray.kill(), reaper has the correct snapshot,
    kill proceeds.
    """
    proc = _start_sleeper()
    pid = proc.pid
    create_time = psutil.Process(pid).create_time()

    ray.get(_reap_pids.remote([(pid, create_time)]))  # type: ignore[attr-defined]

    proc.wait(timeout=5)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.L1
@pytest.mark.CPU
def test_already_dead_pid_handled_gracefully(ray_session: None) -> None:
    """If the PID is already gone (ray.kill() worked), _reap_pids should not raise.

    This is the common happy path: the safety net runs but finds nothing to do.
    """
    proc = _start_sleeper()
    pid = proc.pid
    create_time = psutil.Process(pid).create_time()

    proc.kill()
    proc.wait()

    # Should complete without error even though pid is already gone
    ray.get(_reap_pids.remote([(pid, create_time)]))  # type: ignore[attr-defined]


@pytest.mark.L1
@pytest.mark.CPU
def test_fallback_none_create_time_kills_without_protection(ray_session: None) -> None:
    """None create_time (fallback path) kills the process without PID-reuse protection.

    This matches the case where get_pid_tree() timed out and we fell back to the
    Ray state API PID. We don't have a create_time to compare, so we kill anyway.
    """
    proc = _start_sleeper()
    pid = proc.pid

    ray.get(_reap_pids.remote([(pid, None)]))  # type: ignore[attr-defined]

    proc.wait(timeout=5)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.L1
@pytest.mark.CPU
def test_empty_pid_list_is_noop(ray_session: None) -> None:
    """An empty pid_entries list completes without error."""
    ray.get(_reap_pids.remote([]))  # type: ignore[attr-defined]


@ray.remote
class _PidTreeActor:
    """Minimal actor that exposes get_pid_tree() for testing the interface contract."""

    def get_pid_tree(self) -> list[tuple[int, float]]:
        import os

        import psutil

        try:
            p = psutil.Process(os.getpid())
            procs = [p, *p.children(recursive=True)]
            result = []
            for proc in procs:
                try:
                    result.append((proc.pid, proc.create_time()))
                except psutil.NoSuchProcess:
                    pass
            return result if result else [(p.pid, p.create_time())]
        except psutil.NoSuchProcess:
            return [(os.getpid(), 0.0)]


@pytest.mark.L1
@pytest.mark.CPU
def test_get_pid_tree_returns_pid_create_time_pairs(ray_session: None) -> None:
    """get_pid_tree() returns (pid, create_time) tuples, not bare ints.

    Ensures the stage_worker side of the interface matches what actor_pool expects.
    The create_time for each PID should match what psutil reports locally.
    """
    actor: _PidTreeActor = _PidTreeActor.remote()  # type: ignore[assignment]
    try:
        result: list[tuple[int, float]] = ray.get(actor.get_pid_tree.remote())

        assert isinstance(result, list)
        assert len(result) >= 1

        for pid, create_time in result:
            assert isinstance(pid, int) and pid > 0
            assert isinstance(create_time, float) and create_time > 0.0
            # create_time should match psutil on the same node
            assert create_time == pytest.approx(psutil.Process(pid).create_time(), abs=0.1)
    finally:
        ray.kill(actor)


@pytest.mark.L1
@pytest.mark.CPU
def test_get_pid_tree_entries_survive_reap_gracefully(ray_session: None) -> None:
    """get_pid_tree() entries fed to _reap_pids after actor is already dead don't raise.

    Snapshot an actor's PID tree, kill it, then hand the entries to _reap_pids.
    The actor is already gone so all PIDs will hit NoSuchProcess — verifies the
    happy path where ray.kill() already did the job before the reaper runs.
    """
    actor: _PidTreeActor = _PidTreeActor.remote()  # type: ignore[assignment]
    pid_entries: list[tuple[int, float | None]] = ray.get(actor.get_pid_tree.remote())
    ray.kill(actor)

    # Reaper should handle all-already-dead entries gracefully
    ray.get(_reap_pids.remote(pid_entries))  # type: ignore[attr-defined]
