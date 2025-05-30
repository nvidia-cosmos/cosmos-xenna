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

import collections
import random
from typing import Any, List, Optional

import pytest

from cosmos_xenna.pipelines.private.streaming import Queue
from cosmos_xenna.ray_utils.actor_pool import Task as ActualTask


# Helper function to create task instances (either real or mock)
def _create_task(data: List[Any], node_id: Optional[str] = None):
    # The items in 'data' will represent ray.ObjectRef for the purpose of these tests.
    # No need to actually call ray.put or create real ObjectRefs,
    # as Queue just stores and retrieves them.
    return ActualTask(task_data=data, origin_node_id=node_id)


class TestQueue:
    def test_initialization(self):
        q_custom = Queue(samples_per_task_window=50)
        assert q_custom._samples_per_task.maxlen == 50
        q_default = Queue()
        assert q_default._samples_per_task.maxlen == 100

    def test_len_and_bool_empty(self):
        q = Queue()
        assert len(q) == 0
        assert not q
        assert bool(q) is False

    def test_add_task_single_node(self):
        q = Queue()
        task1_data = ["obj1", "obj2", "obj3"]
        task1 = _create_task(task1_data, "nodeA")
        q.add_task(task1)  # type: ignore[arg-type]

        assert len(q) == 3
        assert q.by_node_id["nodeA"] == collections.deque(task1_data)
        assert q._samples_per_task == collections.deque([3])

        task2_data = ["obj4", "obj5"]
        task2 = _create_task(task2_data, "nodeA")
        q.add_task(task2)  # type: ignore[arg-type]
        assert len(q) == 5
        assert q.by_node_id["nodeA"] == collections.deque(task1_data + task2_data)
        assert q._samples_per_task == collections.deque([3, 2])

    def test_add_task_multiple_nodes(self):
        q = Queue()
        task_n1_1_data = [1, 2]
        q.add_task(_create_task(task_n1_1_data, "node1"))  # type: ignore[arg-type]

        task_n2_1_data = [10, 20]
        q.add_task(_create_task(task_n2_1_data, "node2"))  # type: ignore[arg-type]

        task_n1_2_data = [3]
        q.add_task(_create_task(task_n1_2_data, "node1"))  # type: ignore[arg-type]

        assert len(q) == 5  # 2 + 2 + 1
        assert q.by_node_id["node1"] == collections.deque([1, 2, 3])
        assert q.by_node_id["node2"] == collections.deque([10, 20])
        assert list(q._samples_per_task) == [2, 2, 1]  # Order of addition

    def test_add_task_no_data(self):
        q = Queue()
        task_empty = _create_task([], "node1")
        q.add_task(task_empty)  # type: ignore[arg-type]

        assert len(q) == 0
        # Node might exist with an empty deque if tasks were added and removed,
        # but add_task with no data doesn't add to _samples_per_task
        assert not q.by_node_id.get("node1") or not q.by_node_id["node1"]
        assert not q._samples_per_task  # "Only record if samples were actually added"

        task_with_data = _create_task([1], "node1")
        q.add_task(task_with_data)  # type: ignore[arg-type]
        assert len(q) == 1
        assert q._samples_per_task == collections.deque([1])

    def test_len_and_bool_with_items(self):
        q = Queue()
        q.add_task(_create_task([1, 2], "node1"))  # type: ignore[arg-type]
        assert len(q) == 2
        assert q
        assert bool(q) is True

    def test_avg_samples_per_task_empty(self):
        q = Queue()
        assert q.avg_samples_per_task() is None

    def test_avg_samples_per_task_single_task(self):
        q = Queue()
        q.add_task(_create_task([1, 2, 3], "node1"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == 3.0

    def test_avg_samples_per_task_multiple_tasks(self):
        q = Queue()
        q.add_task(_create_task([1, 2], "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task([10, 20, 30, 40], "node2"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == (2 + 4) / 2.0  # 3.0

    def test_avg_samples_per_task_with_empty_tasks(self):
        q = Queue()
        q.add_task(_create_task([1, 2], "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task([], "node2"))  # type: ignore[arg-type]
        q.add_task(_create_task([1, 2, 3, 4], "node1"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == (2 + 4) / 2.0  # Still 3.0

    def test_avg_samples_per_task_exceeds_window(self):
        q = Queue(samples_per_task_window=2)
        q.add_task(_create_task([1] * 1, "n1"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == 1.0
        q.add_task(_create_task([1] * 2, "n1"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == 1.5
        q.add_task(_create_task([1] * 3, "n1"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == 2.5
        q.add_task(_create_task([1] * 4, "n1"))  # type: ignore[arg-type]
        assert q.avg_samples_per_task() == 3.5

    def test_maybe_get_batch_empty_queue(self):
        q = Queue()
        assert q.maybe_get_batch(5) is None

    def test_maybe_get_batch_not_enough_items(self):
        q = Queue()
        q.add_task(_create_task([1, 2], "node1"))  # type: ignore[arg-type]
        assert q.maybe_get_batch(3) is None
        assert len(q) == 2  # Items should remain

    def test_maybe_get_batch_exact_size_single_node(self):
        q = Queue()
        items = [1, 2, 3]
        q.add_task(_create_task(items, "node1"))  # type: ignore[arg-type]
        batch = q.maybe_get_batch(3)
        assert batch is not None
        assert batch.task_data == items
        assert batch.origin_node_id == "node1"
        assert len(q) == 0

    def test_maybe_get_batch_partial_size_single_node(self):
        q = Queue()
        items = [1, 2, 3, 4, 5]
        q.add_task(_create_task(items, "node1"))  # type: ignore[arg-type]
        batch = q.maybe_get_batch(3)
        assert batch is not None
        assert batch.task_data == [1, 2, 3]  # FIFO from deque
        assert batch.origin_node_id == "node1"
        assert len(q) == 2
        assert list(q.by_node_id["node1"]) == [4, 5]

    def test_maybe_get_batch_exact_size_multiple_nodes_and_distribution(self):
        q = Queue()
        items_n1 = [1, 2, 3]  # 3 items
        items_n2 = [11, 12]  # 2 items
        items_n3 = [21]  # 1 item
        q.add_task(_create_task(items_n1, "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task(items_n2, "node2"))  # type: ignore[arg-type]
        q.add_task(_create_task(items_n3, "node3"))  # type: ignore[arg-type]
        # Total 3+2+1 = 6 items

        random.seed(42)  # Control shuffle for test repeatability

        batch = q.maybe_get_batch(4)  # Request 4 items
        assert batch is not None
        assert len(batch.task_data) == 4
        assert len(q) == 2  # 6 - 4 = 2 items left

        # Check that items are from the original set
        original_items = set(items_n1 + items_n2 + items_n3)
        for item in batch.task_data:
            assert item in original_items

    def test_maybe_get_batch_invalid_batch_size(self):
        q = Queue()
        q.add_task(_create_task([1, 2, 3], "node1"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Batch size must be greater than 0"):
            q.maybe_get_batch(0)
        with pytest.raises(ValueError, match="Batch size must be greater than 0"):
            q.maybe_get_batch(-1)
        assert len(q) == 3  # No change to queue

    def test_maybe_get_batch_multiple_calls_drain_queue(self):
        q = Queue()
        q.add_task(_create_task([1, 2], "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task([3, 4], "node2"))  # type: ignore[arg-type]

        batch1 = q.maybe_get_batch(2)
        # Test that the queue has fully drained from one node
        assert q.by_node_id[batch1.origin_node_id] == collections.deque()
        batch2 = q.maybe_get_batch(2)
        assert q.maybe_get_batch(1) is None  # Queue is empty

        all_pulled_items = batch1.task_data + batch2.task_data
        assert set(all_pulled_items) == {1, 2, 3, 4}

    def test_maybe_get_batch_node_empties_during_batch(self):
        q = Queue()
        q.add_task(_create_task([1], "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task([11, 12, 13], "node2"))  # type: ignore[arg-type]
        batch = q.maybe_get_batch(3)  # Request 3 items
        assert len(batch.task_data) == 3
        assert len(q) == 1

    def test_get_all_samples_empty_queue(self):
        q = Queue()
        assert q.get_all_samples() == []
        assert not q.by_node_id  # Should be empty

    def test_get_all_samples_single_node(self):
        q = Queue()
        items = [1, 2, 3]
        q.add_task(_create_task(items, "node1"))  # type: ignore[arg-type]
        all_samples = q.get_all_samples()
        # Order from list(queue) then extend
        assert all_samples == items
        assert len(q) == 0
        assert not q.by_node_id.get("node1")  # Node entry should be removed

    def test_get_all_samples_multiple_nodes(self):
        q = Queue()
        items_n1 = [1, 2]
        items_n2 = [10, 20]
        # Order of adding tasks matters for _samples_per_task, but not for final by_node_id content before get_all
        q.add_task(_create_task(items_n1, "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task(items_n2, "node2"))  # type: ignore[arg-type]

        all_samples = q.get_all_samples()
        # Order depends on dict iteration order of by_node_id.keys() and then list(queue)
        # So, check for content and size.
        assert len(all_samples) == 4
        assert set(all_samples) == {1, 2, 10, 20}
        assert len(q) == 0
        assert not q.by_node_id  # Check that nodes are cleaned up

    def test_get_all_samples_clears_queue_and_empties_nodes(self):
        q = Queue()
        q.add_task(_create_task([1], "node1"))  # type: ignore[arg-type]
        q.add_task(_create_task([2], "node2"))  # type: ignore[arg-type]
        q.get_all_samples()
        assert len(q) == 0
        assert not q  # bool is False
        assert not q.by_node_id
        # _samples_per_task is not cleared by get_all_samples
        assert q._samples_per_task == collections.deque([1, 1])

    def test_integration_add_batch_get_all(self):
        q = Queue()
        # Add tasks
        q.add_task(_create_task([1, 2, 3], "N1"))  # type: ignore[arg-type]
        q.add_task(_create_task([4, 5], "N2"))  # type: ignore[arg-type]
        q.add_task(_create_task([6], "N1"))  # type: ignore[arg-type]
        # State: N1: [1,2,3,6], N2: [4,5]. Total = 6.
        # _samples_per_task: [3,2,1] -> avg (3+2+1)/3 = 2
        assert len(q) == 6
        assert q.avg_samples_per_task() == 2.0

        random.seed(10)  # N2 then N1
        # Batch 1 (size 3):
        # Pull N2 (4), N1 (1), N2 (5) -> Batch [4,1,5], Nodes [N2,N1,N2]. Most common: N2
        # Remaining: N1: [2,3,6], N2: []
        batch1 = q.maybe_get_batch(3)
        assert batch1 is not None
        assert len(batch1.task_data) == 3
        assert set(batch1.task_data) == {1, 4, 5}
        assert batch1.origin_node_id == "N2"
        assert len(q) == 3
        assert q.avg_samples_per_task() == 2.0  # Unchanged by get

        # Remaining: N1: [2,3,6] (node N2 is empty and removed from active consideration)
        # Batch 2 (size 2):
        # Pull N1 (2), N1 (3) -> Batch [2,3]. Nodes [N1,N1]. Most common: N1
        batch2 = q.maybe_get_batch(2)
        assert batch2 is not None
        assert len(batch2.task_data) == 2
        assert set(batch2.task_data) == {2, 3}
        assert batch2.origin_node_id == "N1"
        assert len(q) == 1

        # Get all remaining (should be [6] from N1)
        remaining_samples = q.get_all_samples()
        assert remaining_samples == [6]
        assert len(q) == 0
        assert not q

        all_retrieved = batch1.task_data + batch2.task_data + remaining_samples
        assert set(all_retrieved) == {1, 2, 3, 4, 5, 6}
        assert q.avg_samples_per_task() == 2.0  # Still unchanged
