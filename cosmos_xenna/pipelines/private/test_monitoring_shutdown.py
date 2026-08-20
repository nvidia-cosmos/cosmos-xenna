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

"""Tests for the monitoring-actor teardown path.

``NodeResourceMonitor`` had no shutdown path at all: its ``_stop_event`` was checked by
both background loops but never set anywhere, and ``RayResourceMonitor`` never tore down
the actors it created. A leaked monitor scans the whole process table once a second
forever, and under Ray 2.57's default ``process_group_cleanup_enabled`` the raylet waits
for a gracefully-disconnecting worker to exit before sweeping its process group, so such a
monitor is never reaped.

Ray is not initialised: the underlying method functions are exercised against a
``MagicMock`` self, following the ``__wrapped__``/``__get__`` pattern used in
``cosmos_xenna/ray_utils/test_stage_worker_shutdown.py``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable
from unittest import mock
from unittest.mock import MagicMock

import pytest

import cosmos_xenna.pipelines.private.monitoring as monitoring

_JOIN_TIMEOUT_S = 5.0


def _get_impl(owner: Any, name: str) -> Callable[..., Any]:
    """Return the raw function behind a method on a ``@ray.remote``-wrapped class.

    ``@ray.remote`` replaces the class with an actor-class descriptor whose own
    ``__init__`` constructs an actor, so the original methods are reached via
    ``__ray_metadata__.modified_class``. Ray's tracing decorator then exposes each
    underlying function as ``__wrapped__``.
    """
    cls = getattr(owner, "__ray_metadata__", None)
    method = getattr(cls.modified_class if cls is not None else owner, name)
    return getattr(method, "__wrapped__", method)


def _spawn_stoppable_loop(stop_event: threading.Event) -> tuple[threading.Thread, threading.Event]:
    """Start a daemon thread that runs until ``stop_event`` fires, plus an exit witness."""
    exited = threading.Event()

    def loop() -> None:
        while not stop_event.wait(0.01):
            pass
        exited.set()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread, exited


@pytest.mark.L1
@pytest.mark.CPU
def test_node_monitor_stop_sets_event_and_joins_threads() -> None:
    """``stop()`` ends both background loops and waits for them."""
    mock_self = MagicMock(name="node_resource_monitor")
    mock_self._node_ip = "127.0.0.1"
    mock_self._stop_event = threading.Event()
    mock_self._thread, metrics_exited = _spawn_stoppable_loop(mock_self._stop_event)
    mock_self._orphan_thread, orphan_exited = _spawn_stoppable_loop(mock_self._stop_event)

    _get_impl(monitoring.NodeResourceMonitor, "stop")(mock_self)

    assert mock_self._stop_event.is_set()
    assert metrics_exited.is_set()
    assert orphan_exited.is_set()
    assert not mock_self._thread.is_alive()
    assert not mock_self._orphan_thread.is_alive()


@pytest.mark.L1
@pytest.mark.CPU
def test_node_monitor_stop_tolerates_absent_orphan_thread() -> None:
    """The orphan thread only exists when NVML is available, so ``None`` must be fine."""
    mock_self = MagicMock(name="node_resource_monitor")
    mock_self._node_ip = "127.0.0.1"
    mock_self._stop_event = threading.Event()
    mock_self._thread, metrics_exited = _spawn_stoppable_loop(mock_self._stop_event)
    mock_self._orphan_thread = None

    _get_impl(monitoring.NodeResourceMonitor, "stop")(mock_self)

    assert metrics_exited.is_set()


@pytest.mark.L1
@pytest.mark.CPU
def test_metrics_thread_is_daemon() -> None:
    """A non-daemon metrics thread blocks interpreter finalization forever."""
    mock_self = MagicMock(name="node_resource_monitor")
    with (
        mock.patch.object(monitoring.ray, "get_runtime_context"),
        mock.patch.object(monitoring.ray.util, "get_node_ip_address", return_value="127.0.0.1"),
    ):
        _get_impl(monitoring.NodeResourceMonitor, "__init__")(mock_self)

    thread = mock_self._thread
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True
    mock_self._stop_event.set()
    thread.join(timeout=_JOIN_TIMEOUT_S)


@pytest.mark.L1
@pytest.mark.CPU
def test_orphan_scan_loop_exits_without_waiting_out_its_interval() -> None:
    """The scan interval is an interruptible wait, so ``stop()`` is not blocked ~30 s."""
    mock_self = MagicMock(name="node_resource_monitor")
    mock_self._stop_event = threading.Event()

    thread = threading.Thread(
        target=_get_impl(monitoring.NodeResourceMonitor, "_orphan_scan_loop"),
        args=(mock_self,),
        daemon=True,
    )
    thread.start()
    mock_self._stop_event.set()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert not thread.is_alive()
    mock_self._scan_gpu_orphans.assert_not_called()


def _make_ray_monitor_mock(num_nodes: int) -> MagicMock:
    mock_self = MagicMock(name="ray_resource_monitor")
    mock_self._node_ids = [f"node-{idx}" for idx in range(num_nodes)]
    mock_self._monitors = [MagicMock(name=f"monitor-{idx}") for idx in range(num_nodes)]
    return mock_self


@pytest.mark.L1
@pytest.mark.CPU
def test_ray_monitor_stop_stops_then_kills_every_actor() -> None:
    """Every per-node actor gets a graceful ``stop()`` followed by a ``ray.kill()``."""
    mock_self = _make_ray_monitor_mock(3)
    monitors = list(mock_self._monitors)

    with (
        mock.patch.object(monitoring.ray, "get") as ray_get,
        mock.patch.object(monitoring.ray, "kill") as ray_kill,
    ):
        _get_impl(monitoring.RayResourceMonitor, "stop")(mock_self)

    assert ray_get.call_count == 3
    assert [call.args[0] for call in ray_kill.call_args_list] == monitors
    for monitor in monitors:
        monitor.stop.remote.assert_called_once()
    # Cleared so a second stop() (e.g. close() then __exit__) is a no-op.
    assert not mock_self._monitors
    assert not mock_self._node_ids


@pytest.mark.L1
@pytest.mark.CPU
def test_ray_monitor_stop_swallows_unreachable_actors() -> None:
    """Teardown runs while Ray may already be going away, so it must not raise."""
    mock_self = _make_ray_monitor_mock(2)
    monitors = list(mock_self._monitors)

    with (
        mock.patch.object(monitoring.ray, "get", side_effect=RuntimeError("gcs is gone")),
        mock.patch.object(monitoring.ray, "kill", side_effect=RuntimeError("gcs is gone")) as ray_kill,
    ):
        _get_impl(monitoring.RayResourceMonitor, "stop")(mock_self)

    # A failure on the first actor must not skip the rest.
    assert ray_kill.call_count == len(monitors)


@pytest.mark.L1
@pytest.mark.CPU
def test_pipeline_monitor_close_stops_the_node_monitors() -> None:
    """``close()`` was a bare assert, so the actors it opened were never torn down."""
    mock_self = MagicMock(name="pipeline_monitor")
    mock_self._opened = True

    monitoring.PipelineMonitor.close(mock_self)

    mock_self._nodes_resource_monitor.stop.assert_called_once_with()
    assert mock_self._opened is False
