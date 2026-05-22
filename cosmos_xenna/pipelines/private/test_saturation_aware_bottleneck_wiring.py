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

"""Orchestrator wiring tests for ``_refresh_cycle_bottleneck_context``.

The scheduler publishes a per-cycle ``BottleneckCycleContext`` onto
every stage's runtime state right after :func:`identify_bottleneck`
returns. The per-stage decision pipeline reads it for diagnostic
log fields. These tests pin the wiring contract: which stages get
``is_upstream_of_bottleneck=True``, what happens when no bottleneck
is engaged, and how the publish step behaves when the bottleneck moves between
cycles.
"""

import math
from collections.abc import Callable

import pytest

from cosmos_xenna.pipelines.private import specs as cc_specs
from cosmos_xenna.pipelines.private.scheduling_py.bottleneck import (
    BottleneckCycleContext,
    BottleneckIdentity,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import _StageRuntimeState
from cosmos_xenna.pipelines.private.test_saturation_aware_scheduler import _problem_with_stages

SchedulerFactory = Callable[[list[str]], SaturationAwareScheduler]


@pytest.fixture
def make_scheduler_with_stages() -> SchedulerFactory:
    """Build a fresh ``SaturationAwareScheduler`` with the given stage names."""

    def _factory(stage_names: list[str]) -> SaturationAwareScheduler:
        scheduler = SaturationAwareScheduler(cc_specs.SaturationAwareConfig())
        scheduler.setup(_problem_with_stages(stage_names))
        return scheduler

    return _factory


def _engaged_meta(stage_name: str) -> BottleneckIdentity:
    """Build an engaged ``BottleneckIdentity`` for the named stage."""
    return BottleneckIdentity(
        engaged=True,
        stage_name=stage_name,
        max_d_k=2.0,
        median_d_k=0.5,
        heterogeneity_ratio=4.0,
    )


def _disengaged_meta() -> BottleneckIdentity:
    """Build a disengaged ``BottleneckIdentity`` (cluster homogeneous or cold-start)."""
    return BottleneckIdentity(
        engaged=False,
        stage_name=None,
        max_d_k=math.nan,
        median_d_k=math.nan,
        heterogeneity_ratio=math.nan,
    )


def _ctx(stage_state: _StageRuntimeState) -> BottleneckCycleContext:
    """Read the published context off a stage state."""
    return stage_state.cycle_bottleneck_context


class TestEngagedBottleneckMarksStrictUpstream:
    """``is_upstream_of_bottleneck=True`` is set only for stages strictly upstream of the bottleneck."""

    def test_strict_upstream_stages_marked_upstream(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """In ``[A, B, C]`` with bottleneck C: A and B get is_upstream_of_bottleneck=True."""
        scheduler = make_scheduler_with_stages(["A", "B", "C"])
        scheduler._last_bottleneck_meta = _engaged_meta("C")

        scheduler._refresh_cycle_bottleneck_context()

        assert _ctx(scheduler._stage_states["A"]) == BottleneckCycleContext(
            engaged=True, is_upstream_of_bottleneck=True
        )
        assert _ctx(scheduler._stage_states["B"]) == BottleneckCycleContext(
            engaged=True, is_upstream_of_bottleneck=True
        )


class TestEngagedBottleneckClearsDownstreamUpstream:
    """``is_upstream_of_bottleneck=False`` for stages downstream of the bottleneck and for the bottleneck itself."""

    def test_bottleneck_stage_itself_has_is_upstream_false(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """The bottleneck stage's own context has ``engaged=True`` but ``is_upstream_of_bottleneck=False``."""
        scheduler = make_scheduler_with_stages(["A", "B", "C"])
        scheduler._last_bottleneck_meta = _engaged_meta("B")

        scheduler._refresh_cycle_bottleneck_context()

        assert _ctx(scheduler._stage_states["B"]) == BottleneckCycleContext(
            engaged=True, is_upstream_of_bottleneck=False
        )

    def test_strict_downstream_stages_have_is_upstream_false(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """In ``[A, B, C, D]`` with bottleneck B: C and D have is_upstream_of_bottleneck=False."""
        scheduler = make_scheduler_with_stages(["A", "B", "C", "D"])
        scheduler._last_bottleneck_meta = _engaged_meta("B")

        scheduler._refresh_cycle_bottleneck_context()

        assert _ctx(scheduler._stage_states["C"]) == BottleneckCycleContext(
            engaged=True, is_upstream_of_bottleneck=False
        )
        assert _ctx(scheduler._stage_states["D"]) == BottleneckCycleContext(
            engaged=True, is_upstream_of_bottleneck=False
        )


class TestDisengagedBottleneckClearsAllStages:
    """A disengaged meta clears every stage's context to the no-bottleneck default."""

    def test_disengaged_clears_is_upstream_everywhere(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """Disengaged meta: every stage carries the default ``BottleneckCycleContext()``."""
        scheduler = make_scheduler_with_stages(["A", "B", "C"])
        # Pre-seed an engaged context on the first stage to verify the refresh wipes it.
        scheduler._stage_states["A"].cycle_bottleneck_context = BottleneckCycleContext(
            engaged=True, is_upstream_of_bottleneck=True
        )
        scheduler._last_bottleneck_meta = _disengaged_meta()

        scheduler._refresh_cycle_bottleneck_context()

        for state in scheduler._stage_states.values():
            assert _ctx(state) == BottleneckCycleContext()


class TestDefensiveStaleBottleneckName:
    """A bottleneck name not present in the stage list is treated as disengaged."""

    def test_unknown_bottleneck_name_is_treated_as_disengaged(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """A stale ``stage_name`` (e.g. after a stage list change) clears every stage's context."""
        scheduler = make_scheduler_with_stages(["A", "B", "C"])
        scheduler._last_bottleneck_meta = _engaged_meta("UnknownStage")

        scheduler._refresh_cycle_bottleneck_context()

        for state in scheduler._stage_states.values():
            assert _ctx(state) == BottleneckCycleContext()

    def test_empty_string_bottleneck_name_is_treated_as_disengaged(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """An empty-string ``stage_name`` falls through the same defensive branch."""
        scheduler = make_scheduler_with_stages(["A", "B", "C"])
        scheduler._last_bottleneck_meta = _engaged_meta("")

        scheduler._refresh_cycle_bottleneck_context()

        for state in scheduler._stage_states.values():
            assert _ctx(state) == BottleneckCycleContext()


class TestEmptyStageStates:
    """An empty ``_stage_states`` is a no-op on every branch."""

    def test_empty_stage_states_with_engaged_meta_does_not_raise(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """Engaged meta with no stages registered returns cleanly."""
        scheduler = make_scheduler_with_stages(["A"])
        scheduler._stage_states = {}
        scheduler._stage_names = []
        scheduler._last_bottleneck_meta = _engaged_meta("anything")

        scheduler._refresh_cycle_bottleneck_context()

        assert scheduler._stage_states == {}

    def test_empty_stage_states_with_disengaged_meta_does_not_raise(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """Disengaged meta with no stages registered returns cleanly."""
        scheduler = make_scheduler_with_stages(["A"])
        scheduler._stage_states = {}
        scheduler._stage_names = []
        scheduler._last_bottleneck_meta = _disengaged_meta()

        scheduler._refresh_cycle_bottleneck_context()

        assert scheduler._stage_states == {}


class TestBottleneckMovesBetweenCycles:
    """When the bottleneck moves cycle-to-cycle the per-stage context flips correctly."""

    def test_bottleneck_moves_from_b_to_c_flips_is_upstream(
        self,
        make_scheduler_with_stages: SchedulerFactory,
    ) -> None:
        """Bottleneck C in cycle N -> bottleneck B in cycle N+1: B becomes downstream of itself."""
        scheduler = make_scheduler_with_stages(["A", "B", "C"])

        # Cycle N: bottleneck is C.
        scheduler._last_bottleneck_meta = _engaged_meta("C")
        scheduler._refresh_cycle_bottleneck_context()
        assert _ctx(scheduler._stage_states["B"]).is_upstream_of_bottleneck is True

        # Cycle N+1: bottleneck moves to B; B itself is no longer upstream of the bottleneck.
        scheduler._last_bottleneck_meta = _engaged_meta("B")
        scheduler._refresh_cycle_bottleneck_context()
        assert _ctx(scheduler._stage_states["A"]).is_upstream_of_bottleneck is True
        assert _ctx(scheduler._stage_states["B"]).is_upstream_of_bottleneck is False
        assert _ctx(scheduler._stage_states["C"]).is_upstream_of_bottleneck is False
