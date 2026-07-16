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

"""Unit tests for the shutdown path in :class:`StageWorker`.

Focused on the ``_call_user_stage_destroy`` guard that skips the user stage's
``destroy()`` when setup has not signalled completion. Running ``destroy()``
concurrently with an in-flight ``setup()`` races against the resources setup
is still acquiring (e.g. subprocesses being spawned, sockets being
handshaked); the guard prevents that race and defers cleanup to ``ray.kill()``
plus the node-local pid reap.

Ray is not initialised: we exercise the underlying method function against a
``MagicMock`` self, following the same ``__wrapped__``/``__get__`` pattern used
in ``test_stage_worker_continuous.py`` for other ``@ray.remote``-wrapped
methods.
"""

from __future__ import annotations

import threading
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

import cosmos_xenna.ray_utils.stage_worker as sw_module


def _get_destroy_impl() -> Callable[..., Any]:
    """Return the underlying ``_call_user_stage_destroy`` function.

    ``@ray.remote`` wraps ``StageWorker`` in an actor-class descriptor, so
    static type checkers (pyright) do not see the raw class methods as
    attributes. ``getattr`` with a string bypasses the static check while
    still resolving the attribute at runtime. If ``@ray.method`` (or a
    similar wrapper) has been applied, the raw function is exposed via
    ``__wrapped__``; otherwise the attribute itself is the function.
    """
    # ``getattr`` (rather than attribute access) is intentional: it hides the
    # attribute from pyright, which does not model the ``@ray.remote`` actor-
    # class descriptor as exposing the raw class methods.
    method = getattr(sw_module.StageWorker, "_call_user_stage_destroy")  # noqa: B009
    return getattr(method, "__wrapped__", method)


def _make_worker_mock(setup_completed: bool) -> MagicMock:
    """Build a minimal ``StageWorker``-shaped mock for ``_call_user_stage_destroy``."""
    mock = MagicMock(name="stage_worker")
    mock._setup_completed = threading.Event()
    if setup_completed:
        mock._setup_completed.set()
    mock._stage_interface = MagicMock(name="stage_interface")
    mock._params = MagicMock(name="params")
    mock._params.name = "test_stage"
    return mock


@pytest.mark.L1
@pytest.mark.CPU
def test_destroy_skipped_when_setup_incomplete() -> None:
    """``destroy()`` is not invoked when ``_setup_completed`` is not set."""
    mock_self = _make_worker_mock(setup_completed=False)
    call = _get_destroy_impl()

    call(mock_self, 10.0)

    mock_self._stage_interface.destroy.assert_not_called()


@pytest.mark.L1
@pytest.mark.CPU
def test_destroy_runs_when_setup_complete() -> None:
    """``destroy()`` runs normally once ``_setup_completed`` is set."""
    mock_self = _make_worker_mock(setup_completed=True)
    call = _get_destroy_impl()

    call(mock_self, 10.0)

    mock_self._stage_interface.destroy.assert_called_once()


@pytest.mark.L1
@pytest.mark.CPU
def test_destroy_skipped_when_timeout_nonpositive_even_if_setup_complete() -> None:
    """The pre-existing ``timeout_s <= 0`` short-circuit is preserved."""
    mock_self = _make_worker_mock(setup_completed=True)
    call = _get_destroy_impl()

    call(mock_self, 0.0)

    mock_self._stage_interface.destroy.assert_not_called()


@pytest.mark.L1
@pytest.mark.CPU
def test_setup_guard_takes_priority_over_timeout() -> None:
    """The setup guard evaluates before the timeout check.

    A caller passing ``timeout_s > 0`` for a mid-setup actor still gets the
    no-op behavior; the guard is the source of truth for "is there anything
    to destroy right now?".
    """
    mock_self = _make_worker_mock(setup_completed=False)
    call = _get_destroy_impl()

    call(mock_self, 45.0)

    mock_self._stage_interface.destroy.assert_not_called()
