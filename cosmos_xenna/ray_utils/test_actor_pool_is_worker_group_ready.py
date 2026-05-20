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

"""Unit tests for ``ActorPool.is_worker_group_ready``.

The predicate is consumed by ``Autoscaler.apply_autoscale_result_if_ready``
to defer scale-down of worker groups whose actors are still in setup or
have only just finished setup. The contract under test:

  * Returns ``True`` only when the worker group exists, its state has
    transitioned to ``READY`` (all actors finished setup), *and* the
    optional ``min_age_s`` post-Ready grace has elapsed.
  * Returns ``False`` for unknown ids, so callers can collapse "not
    present" and "not ready" into a single "not safe to scale down" gate.
"""

import pytest

from cosmos_xenna.ray_utils.actor_pool import (
    ActorPool,
    _WorkerGroup,
    _WorkerGroupState,
)


def _make_pool_with_groups(groups: dict[str, _WorkerGroup]) -> ActorPool:
    """Build a bare ``ActorPool`` exposing only the field under test."""
    pool = object.__new__(ActorPool)
    pool._worker_groups = groups
    return pool


def _make_group(state: _WorkerGroupState, ready_at: float | None = None) -> _WorkerGroup:
    """Construct a minimal ``_WorkerGroup`` for state-only assertions.

    ``worker_group``/``rendevous_params`` are not exercised by
    ``is_worker_group_ready``; ``None`` keeps the fixture lightweight.
    """
    return _WorkerGroup(
        worker_group=None,  # type: ignore[arg-type]
        actors=set(),
        state=state,
        rendevous_params=None,
        ready_at=ready_at,
    )


def test_returns_true_for_ready_group() -> None:
    """A group that has reached READY is safe to scale down."""
    pool = _make_pool_with_groups({"wg-ready": _make_group(_WorkerGroupState.READY)})
    assert pool.is_worker_group_ready("wg-ready") is True


def test_returns_false_for_waiting_for_setup_group() -> None:
    """A group whose actors are still in setup must not be scaled down."""
    pool = _make_pool_with_groups({"wg-pending": _make_group(_WorkerGroupState.WAITING_FOR_SETUP)})
    assert pool.is_worker_group_ready("wg-pending") is False


def test_returns_false_for_unknown_worker_group_id() -> None:
    """Unknown ids are treated as not-ready so the deferral path stays safe."""
    pool = _make_pool_with_groups({})
    assert pool.is_worker_group_ready("does-not-exist") is False


def test_distinguishes_groups_by_id() -> None:
    """Lookup is per-id and does not aggregate across groups."""
    pool = _make_pool_with_groups(
        {
            "wg-ready": _make_group(_WorkerGroupState.READY),
            "wg-pending": _make_group(_WorkerGroupState.WAITING_FOR_SETUP),
        }
    )
    assert pool.is_worker_group_ready("wg-ready") is True
    assert pool.is_worker_group_ready("wg-pending") is False


def test_grace_period_blocks_freshly_ready_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """A group that became Ready less than ``min_age_s`` ago is still protected."""
    fake_now = 1_000.0
    monkeypatch.setattr("cosmos_xenna.ray_utils.actor_pool.time.time", lambda: fake_now)
    pool = _make_pool_with_groups({"wg-fresh": _make_group(_WorkerGroupState.READY, ready_at=fake_now - 10.0)})
    assert pool.is_worker_group_ready("wg-fresh", min_age_s=60.0) is False


def test_grace_period_releases_after_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once ``min_age_s`` has elapsed the group is eligible for scale-down."""
    fake_now = 1_000.0
    monkeypatch.setattr("cosmos_xenna.ray_utils.actor_pool.time.time", lambda: fake_now)
    pool = _make_pool_with_groups({"wg-mature": _make_group(_WorkerGroupState.READY, ready_at=fake_now - 75.0)})
    assert pool.is_worker_group_ready("wg-mature", min_age_s=60.0) is True


def test_grace_period_disabled_with_zero_min_age() -> None:
    """``min_age_s=0`` short-circuits to the simple Ready check (no timestamp needed)."""
    pool = _make_pool_with_groups({"wg-ready-no-ts": _make_group(_WorkerGroupState.READY, ready_at=None)})
    assert pool.is_worker_group_ready("wg-ready-no-ts", min_age_s=0.0) is True


def test_grace_period_skipped_when_ready_at_missing() -> None:
    """Defensive fallthrough: missing ``ready_at`` does not strand a Ready group."""
    pool = _make_pool_with_groups({"wg-no-ts": _make_group(_WorkerGroupState.READY, ready_at=None)})
    # Even with a grace requested, an absent timestamp must not flip Ready -> not-ready;
    # otherwise a code-path that forgot to stamp ``ready_at`` would silently strand
    # a healthy worker indefinitely.
    assert pool.is_worker_group_ready("wg-no-ts", min_age_s=60.0) is True


def test_grace_period_does_not_promote_pending_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """A group still in setup remains not-ready regardless of how ``min_age_s`` is set."""
    fake_now = 1_000.0
    monkeypatch.setattr("cosmos_xenna.ray_utils.actor_pool.time.time", lambda: fake_now)
    pool = _make_pool_with_groups({"wg-pending": _make_group(_WorkerGroupState.WAITING_FOR_SETUP, ready_at=None)})
    assert pool.is_worker_group_ready("wg-pending", min_age_s=0.0) is False
    assert pool.is_worker_group_ready("wg-pending", min_age_s=60.0) is False
