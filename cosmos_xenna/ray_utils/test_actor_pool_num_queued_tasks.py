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

"""Unit tests for the ``ActorPool.num_queued_tasks`` public property."""

import collections

from cosmos_xenna.ray_utils.actor_pool import ActorPool, Task


def _make_task(node_id: str | None = None) -> Task:
    """Build a minimal ``Task`` instance with empty data for queue-length tests."""
    return Task(task_data=[], origin_node_id=node_id)


def test_num_queued_tasks_zero_on_empty_pool() -> None:
    """Fresh pool with an empty task queue reports zero queued tasks."""
    pool = object.__new__(ActorPool)
    pool._task_queue = collections.deque()

    assert pool.num_queued_tasks == 0


def test_num_queued_tasks_reflects_appended_tasks() -> None:
    """Property mirrors the length of the internal task queue after appends."""
    pool = object.__new__(ActorPool)
    pool._task_queue = collections.deque()
    pool._task_queue.append(_make_task())
    pool._task_queue.append(_make_task())
    pool._task_queue.append(_make_task())

    assert pool.num_queued_tasks == 3
