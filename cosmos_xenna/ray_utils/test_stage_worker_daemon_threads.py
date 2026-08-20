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

"""Regression tests for :class:`StageWorker`'s ability to always reach process exit.

Ray 2.57 enables ``process_group_cleanup_enabled`` by default: each worker becomes its
own process-group leader and, on a graceful disconnect, the raylet polls for the worker
process to exit before sweeping the group. A worker that never finishes interpreter
shutdown therefore survives forever. ``StageWorker``'s three loop threads only exit on
``stop_flag``, which used to be set exclusively by ``shutdown()``, so every other exit
route (SIGTERM, handle going out of scope, ``ray.actor.exit_actor()``) left CPython's
``threading._shutdown()`` blocking on the joins.

These tests pin the two properties that fix it: the loop threads are daemons, and an
``atexit`` hook sets ``stop_flag`` so the loops also stop cooperatively.

Ray is not initialised: the underlying method functions are exercised against a
``MagicMock`` self, following the ``__wrapped__``/``__get__`` pattern already used in
``test_stage_worker_shutdown.py``.
"""

from __future__ import annotations

import threading
from typing import Any, Callable
from unittest import mock
from unittest.mock import MagicMock

import pytest

import cosmos_xenna.ray_utils.stage_worker as sw_module

_THREAD_NAMES = ("_downloader_thread", "_deserializer_thread", "_process_data_thread")
_JOIN_TIMEOUT_S = 5.0


def _get_impl(name: str) -> Callable[..., Any]:
    """Return the raw function behind a ``StageWorker`` method.

    ``@ray.remote`` replaces the class with an actor-class descriptor whose own
    ``__init__`` constructs an actor, so the original methods are reached via
    ``__ray_metadata__.modified_class``. Ray's tracing decorator then exposes each
    underlying function as ``__wrapped__``.
    """
    cls: Any = sw_module.StageWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    method = getattr(cls, name)
    return getattr(method, "__wrapped__", method)


@pytest.fixture
def worker() -> MagicMock:
    """A mock self whose ``__init__`` has run for real, so its threads are real threads.

    The loop targets resolve to ``MagicMock`` attributes, so each thread returns
    immediately; only the thread objects themselves are under test. ``atexit.register`` is
    patched out so the hook is observable and does not outlive the test.
    """
    mock_self = MagicMock(name="stage_worker")
    with mock.patch.object(sw_module.atexit, "register") as register:
        _get_impl("__init__")(mock_self, MagicMock(), MagicMock(), "test_stage", MagicMock())
    mock_self.atexit_register = register
    for name in _THREAD_NAMES:
        getattr(mock_self, name).join(timeout=_JOIN_TIMEOUT_S)
    return mock_self


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize("thread_name", _THREAD_NAMES)
def test_worker_loop_threads_are_daemon(worker: MagicMock, thread_name: str) -> None:
    """Daemon threads are the backstop that lets interpreter finalization complete."""
    thread = getattr(worker, thread_name)
    assert isinstance(thread, threading.Thread)
    assert thread.daemon is True


@pytest.mark.L1
@pytest.mark.CPU
def test_init_registers_atexit_stop_hook(worker: MagicMock) -> None:
    """``__init__`` registers the cooperative stop hook against the stop flag itself.

    Registering the ``Event`` rather than the worker keeps the hook from pinning the actor
    (and the user stage it holds) alive for the life of the process.
    """
    worker.atexit_register.assert_called_once_with(sw_module._signal_stop_at_exit, worker.stop_flag)


@pytest.mark.L1
@pytest.mark.CPU
def test_atexit_hook_sets_stop_flag() -> None:
    """The hook breaks the loops rather than relying on the daemon flag alone."""
    stop_flag = threading.Event()

    sw_module._signal_stop_at_exit(stop_flag)

    assert stop_flag.is_set()


@pytest.mark.L1
@pytest.mark.CPU
def test_atexit_hook_never_raises() -> None:
    """Interpreter teardown must not be perturbed by a failing hook."""
    exploding = MagicMock()
    exploding.set.side_effect = RuntimeError("interpreter is shutting down")

    sw_module._signal_stop_at_exit(exploding)

    exploding.set.assert_called_once()


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize("loop_name", ["_downloader_loop", "_deserializer_loop"])
def test_loops_exit_when_stop_flag_is_set(loop_name: str) -> None:
    """Setting ``stop_flag`` is sufficient to end a running loop."""
    mock_self = MagicMock(name="stage_worker")
    mock_self.stop_flag = threading.Event()

    thread = threading.Thread(target=_get_impl(loop_name), args=(mock_self,), daemon=True)
    thread.start()
    mock_self.stop_flag.set()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert not thread.is_alive()
