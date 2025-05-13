from typing import Optional

import attrs

from cosmos_xenna.ray_utils import resources


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
