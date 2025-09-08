# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import attrs

from cosmos_xenna.pipelines.private import resources


@attrs.define
class PendingActorStats:
    id: str
    resources: resources.WorkerResources


@attrs.define
class ReadyActorStats:
    id: str
    resources: resources.WorkerResources
    speed_tasks_per_second: Optional[float]
    num_used_slots: int
    max_num_slots: int


@attrs.define
class ActorStats:
    target: int
    pending: int
    ready: int
    running: int
    idle: int


@attrs.define
class TaskStats:
    total_completed: int
    total_returned_none: int
    total_dynamically_spawned: int
    input_queue_size: int
    output_queue_size: int


@attrs.define
class SlotStats:
    num_used: int
    num_empty: int


@attrs.define
class ActorPoolStats:
    name: str
    shape: resources.WorkerShape
    actor_stats: ActorStats
    task_stats: TaskStats
    slot_stats: SlotStats
    processing_speed_tasks_per_second: Optional[float]
    pending_actor_pool_ids: list[str]
    ready_actor_pool_ids: list[str]
    pending_actors: list[PendingActorStats]
    ready_actors: list[ReadyActorStats]
