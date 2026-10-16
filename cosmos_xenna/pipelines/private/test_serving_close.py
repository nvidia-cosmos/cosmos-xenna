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

"""Ray-free unit tests for the SERVING close options (``serving_drain_on_close`` / ``serving_drain_timeout_s``).

The end-to-end drain behaviour is covered by ``pipelines/v1/test_serving.py``, which needs a Ray cluster. These tests
pin down the config validation and the pure helpers the streaming loop uses to decide when to stop and what to report.
"""

import collections
import math
import multiprocessing
from typing import Any
from unittest.mock import MagicMock

import pytest

from cosmos_xenna.pipelines.private import pipelines, resources, specs
from cosmos_xenna.pipelines.private.streaming import Queue, _pending_work, _serving_drain_timed_out
from cosmos_xenna.ray_utils import cluster
from cosmos_xenna.ray_utils.actor_pool import Task


class _DoubleStage(specs.Stage[int, int]):
    @property
    def required_resources(self) -> resources.Resources:
        return resources.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int]:  # pragma: no cover
        # Never invoked: these tests stop before a pipeline runs.
        return [x * 2 for x in in_data]


class TestServingCloseConfig:
    def test_defaults_keep_the_immediate_stop_behaviour(self):
        config = specs.PipelineConfig()
        assert config.serving_drain_on_close is False
        assert config.serving_drain_timeout_s is None
        config.validate_serving_close_options()

    @pytest.mark.parametrize("timeout_s", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_rejects_a_timeout_that_is_not_a_finite_positive_number(self, timeout_s: float):
        with pytest.raises(ValueError, match="serving_drain_timeout_s must be a finite number > 0"):
            specs.PipelineConfig(serving_drain_on_close=True, serving_drain_timeout_s=timeout_s)

    def test_rejects_an_invalid_timeout_assigned_after_construction(self):
        config = specs.PipelineConfig(serving_drain_on_close=True)
        with pytest.raises(ValueError, match="serving_drain_timeout_s must be a finite number > 0"):
            config.serving_drain_timeout_s = 0.0

    def test_accepts_drain_with_and_without_a_timeout(self):
        specs.PipelineConfig(serving_drain_on_close=True).validate_serving_close_options()
        specs.PipelineConfig(serving_drain_on_close=True, serving_drain_timeout_s=0.5).validate_serving_close_options()

    def test_rejects_a_timeout_without_drain(self):
        config = specs.PipelineConfig(serving_drain_timeout_s=30.0)
        with pytest.raises(ValueError, match="requires serving_drain_on_close=True"):
            config.validate_serving_close_options()

    def test_run_pipeline_rejects_a_timeout_without_drain_before_starting_ray(self, monkeypatch: pytest.MonkeyPatch):
        def _fail_if_called() -> None:
            raise AssertionError("the cluster must not be started for an invalid config")

        monkeypatch.setattr(cluster, "init_or_connect_to_cluster", _fail_if_called)
        spec = specs.PipelineSpec(
            [],
            [_DoubleStage()],
            specs.PipelineConfig(execution_mode=specs.ExecutionMode.SERVING, serving_drain_timeout_s=30.0),
            serving_queues=specs.ServingQueues(source=multiprocessing.Queue(), sink=multiprocessing.Queue()),
        )

        with pytest.raises(ValueError, match="requires serving_drain_on_close=True"):
            pipelines.run_pipeline(spec)


class TestServingDrainTimedOut:
    def test_never_times_out_while_the_source_is_open(self):
        assert not _serving_drain_timed_out(closed_at=None, timeout_s=1.0, now=1_000.0)

    def test_never_times_out_without_a_timeout(self):
        assert not _serving_drain_timed_out(closed_at=0.0, timeout_s=None, now=1_000_000.0)

    def test_times_out_only_after_the_timeout_has_fully_elapsed(self):
        assert not _serving_drain_timed_out(closed_at=100.0, timeout_s=5.0, now=104.9)
        assert not _serving_drain_timed_out(closed_at=100.0, timeout_s=5.0, now=105.0)
        assert _serving_drain_timed_out(closed_at=100.0, timeout_s=5.0, now=105.1)


class TestPendingWork:
    @staticmethod
    def _pool(*, queued: int = 0, inflight: int = 0, completed: int = 0) -> Any:
        pool = MagicMock()
        pool.num_queued_tasks = queued
        pool.num_inflight_tasks = inflight
        pool.completed_tasks = collections.deque(object() for _ in range(completed))
        return pool

    @staticmethod
    def _queue(num_samples: int, node_id: str | None = None) -> Queue:
        q = Queue()
        if num_samples:
            # Queue only stores and counts the refs, so plain objects stand in for ray.ObjectRef.
            samples: list[Any] = [object() for _ in range(num_samples)]
            q.add_task(Task(task_data=samples, origin_node_id=node_id))
        return q

    def test_idle_pipeline_has_no_pending_work(self):
        pending = _pending_work(self._queue(0), [self._queue(0), self._queue(0)], [self._pool(), self._pool()])
        assert pending.is_empty
        assert pending.describe() == "0 queued sample(s) and 0 task(s) queued or running in stage pools"

    def test_counts_samples_waiting_in_every_queue(self):
        stage_queues = [self._queue(2, node_id="node-a"), self._queue(1)]
        pending = _pending_work(self._queue(3), stage_queues, [self._pool(), self._pool()])
        assert pending.queued_samples == 6
        assert pending.pool_tasks == 0
        assert not pending.is_empty

    def test_counts_queued_running_and_undelivered_completed_pool_tasks(self):
        pools = [self._pool(queued=1, inflight=2), self._pool(completed=3)]
        pending = _pending_work(self._queue(0), [self._queue(0), self._queue(0)], pools)
        assert pending.queued_samples == 0
        assert pending.pool_tasks == 6
        assert not pending.is_empty
