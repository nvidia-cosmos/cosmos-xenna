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

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.activity import (
    PipelineActivitySnapshot,
    StageActivity,
)


def test_stage_activity_converts_batch_counts_to_input_samples() -> None:
    activity = StageActivity(
        queue_depth_samples=7.0,
        pool_queued_tasks=3,
        inflight_slots=2,
        batch_size=4,
    )

    assert activity.active_depth() == 27.0


def test_pipeline_snapshot_builds_active_depths_from_primitive_counts() -> None:
    snapshot = PipelineActivitySnapshot.from_counts(
        queue_depths=[5, 0],
        pool_queued_tasks=[1, 3],
        inflight_slots=[2, 4],
        batch_sizes=[4, 8],
    )

    assert snapshot.active_depths() == (17.0, 56.0)


def test_activity_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="pool_queued_tasks"):
        StageActivity(
            queue_depth_samples=0.0,
            pool_queued_tasks=-1,
            inflight_slots=0,
            batch_size=1,
        )


def test_activity_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        StageActivity(
            queue_depth_samples=0.0,
            pool_queued_tasks=0,
            inflight_slots=0,
            batch_size=0,
        )
