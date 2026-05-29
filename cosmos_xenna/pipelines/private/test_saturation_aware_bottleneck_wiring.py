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

"""Contract tests for ``project_stage_topology``.

The intent phase projects the cycle-wide
:class:`BottleneckIdentity` into a per-stage
:class:`StageTopologyContext` (``engaged`` +
``is_upstream_of_bottleneck`` booleans) at the call site of
:meth:`StageDecisionPipeline.compute_recommendation`. No per-stage
mirror is stored on the runtime state.

These tests pin the projection contract:

  * Disengaged identity collapses every stage to the default
    ``StageTopologyContext()``.
  * Stale identity (bottleneck name no longer in the stage list)
    also collapses to the default.
  * Engaged identity makes every strictly-upstream stage observe
    ``is_upstream_of_bottleneck=True``; the bottleneck itself and
    every downstream stage observe ``False``.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.state.stage_topology import (
    StageTopologyContext,
    project_stage_topology,
)


class TestProjectStageTopologyDisengaged:
    """A disengaged identity collapses every stage to the default."""

    def test_disengaged_returns_default(self) -> None:
        """``bottleneck_engaged=False`` -> default ``StageTopologyContext()``."""
        topology = project_stage_topology(
            stage_index=2,
            bottleneck_engaged=False,
            bottleneck_stage_name=None,
            stage_names=("A", "B", "C", "D"),
        )

        assert topology == StageTopologyContext()

    def test_disengaged_with_lingering_stage_name_still_default(self) -> None:
        """A disengaged identity carrying a stage name still collapses."""
        topology = project_stage_topology(
            stage_index=2,
            bottleneck_engaged=False,
            bottleneck_stage_name="B",
            stage_names=("A", "B", "C", "D"),
        )

        assert topology == StageTopologyContext()


class TestProjectStageTopologyStaleIdentity:
    """A bottleneck name absent from ``stage_names`` collapses to the default."""

    def test_unknown_stage_name_returns_default(self) -> None:
        """``bottleneck_stage_name`` outside ``stage_names`` -> default."""
        topology = project_stage_topology(
            stage_index=1,
            bottleneck_engaged=True,
            bottleneck_stage_name="X",
            stage_names=("A", "B", "C"),
        )

        assert topology == StageTopologyContext()


class TestProjectStageTopologyEngaged:
    """An engaged identity partitions the pipeline into upstream / not-upstream."""

    def test_upstream_stage_observes_is_upstream_true(self) -> None:
        """A stage with strictly smaller DAG index than the bottleneck is upstream."""
        topology = project_stage_topology(
            stage_index=0,
            bottleneck_engaged=True,
            bottleneck_stage_name="C",
            stage_names=("A", "B", "C", "D"),
        )

        assert topology == StageTopologyContext(
            engaged=True,
            is_upstream_of_bottleneck=True,
        )

    def test_bottleneck_stage_itself_is_not_upstream(self) -> None:
        """The bottleneck stage observes ``is_upstream_of_bottleneck=False``."""
        topology = project_stage_topology(
            stage_index=2,
            bottleneck_engaged=True,
            bottleneck_stage_name="C",
            stage_names=("A", "B", "C", "D"),
        )

        assert topology == StageTopologyContext(
            engaged=True,
            is_upstream_of_bottleneck=False,
        )

    def test_downstream_stage_is_not_upstream(self) -> None:
        """A stage with strictly larger DAG index than the bottleneck is not upstream."""
        topology = project_stage_topology(
            stage_index=3,
            bottleneck_engaged=True,
            bottleneck_stage_name="C",
            stage_names=("A", "B", "C", "D"),
        )

        assert topology == StageTopologyContext(
            engaged=True,
            is_upstream_of_bottleneck=False,
        )

    def test_first_stage_is_upstream_when_second_stage_is_bottleneck(self) -> None:
        """Boundary: first stage upstream of second-stage bottleneck."""
        topology = project_stage_topology(
            stage_index=0,
            bottleneck_engaged=True,
            bottleneck_stage_name="B",
            stage_names=("A", "B"),
        )

        assert topology == StageTopologyContext(
            engaged=True,
            is_upstream_of_bottleneck=True,
        )

    def test_last_stage_is_not_upstream_when_first_stage_is_bottleneck(self) -> None:
        """Boundary: last stage downstream of first-stage bottleneck."""
        topology = project_stage_topology(
            stage_index=1,
            bottleneck_engaged=True,
            bottleneck_stage_name="A",
            stage_names=("A", "B"),
        )

        assert topology == StageTopologyContext(
            engaged=True,
            is_upstream_of_bottleneck=False,
        )


class TestProjectStageTopologyStageIndexBounds:
    """An engaged projection fails fast on an out-of-range ``stage_index``."""

    def test_out_of_range_stage_index_raises_index_error(self) -> None:
        """A ``stage_index`` past the stage list raises rather than returning a silent default."""
        with pytest.raises(IndexError, match=r"stage_index=5 out of range for 3 stage names"):
            project_stage_topology(
                stage_index=5,
                bottleneck_engaged=True,
                bottleneck_stage_name="B",
                stage_names=("A", "B", "C"),
            )
