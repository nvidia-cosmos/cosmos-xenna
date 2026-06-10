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

"""Unit tests for the ``ActorPool.num_inflight_tasks`` public property."""

from types import SimpleNamespace
from typing import Any

from cosmos_xenna.ray_utils.actor_pool import ActorPool


def _ready_actor(used_slots: int, rank: int | None) -> Any:
    """Minimal ready-actor stand-in; only used slots and rank are read."""
    return SimpleNamespace(num_used_slots=used_slots, metadata=SimpleNamespace(rank=rank))


def test_num_inflight_tasks_zero_on_empty_pool() -> None:
    """A pool with no ready actors reports zero in-flight logical tasks."""
    pool = object.__new__(ActorPool)
    pool._ready_actors = {}

    assert pool.num_inflight_tasks == 0


def test_num_inflight_tasks_counts_each_used_slot_for_non_spmd() -> None:
    """Non-SPMD actors (rank None) each contribute their own used slots."""
    pool = object.__new__(ActorPool)
    pool._ready_actors = {
        "a": _ready_actor(used_slots=2, rank=None),
        "b": _ready_actor(used_slots=1, rank=None),
    }

    assert pool.num_inflight_tasks == 3
    assert pool.num_inflight_tasks == pool.num_used_slots


def test_num_inflight_tasks_counts_each_spmd_group_once() -> None:
    """One SPMD logical task spread across rank actors is counted once, not per rank.

    A 4-rank SPMD group runs a single logical batch on one slot of every rank,
    so ``num_used_slots`` over-counts it as four; only the primary (rank 0) is
    counted toward in-flight logical work.
    """
    pool = object.__new__(ActorPool)
    pool._ready_actors = {
        "r0": _ready_actor(used_slots=1, rank=0),
        "r1": _ready_actor(used_slots=1, rank=1),
        "r2": _ready_actor(used_slots=1, rank=2),
        "r3": _ready_actor(used_slots=1, rank=3),
    }

    assert pool.num_inflight_tasks == 1
    assert pool.num_used_slots == 4
