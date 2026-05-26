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

"""Contract tests for :class:`PreflightBuilder`.

Verify the documented pre-phase order: shape check, cycle-counter
advance, warmup refresh, stuck-plan snapshot, regime update,
threshold resolution, donor warmup exclusion build. The tests
record the boundary order by side effect on shared lists rather
than asserting through private wiring per ``test-creation.mdc``.
"""

from typing import cast
from unittest.mock import MagicMock, patch

import attrs
import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.preflight import PreflightBuilder, PreflightResult
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller import RegimeController
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver import ThresholdResolver
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def _stub_stage(name: str = "s0") -> MagicMock:
    """Build a sentinel stage descriptor with a configurable name."""
    stage = MagicMock()
    stage.name = name
    stage.stage_name = name
    return stage


def _stub_problem(stage_names: tuple[str, ...] = ("s0",)) -> data_structures.Problem:
    """Build a sentinel pipeline ``Problem`` with the given stage order."""
    problem = MagicMock(spec=data_structures.Problem)
    problem.rust = MagicMock()
    problem.rust.stages = [_stub_stage(name) for name in stage_names]
    return cast(data_structures.Problem, problem)


def _stub_problem_state(stage_names: tuple[str, ...] = ("s0",)) -> data_structures.ProblemState:
    """Build a sentinel runtime ``ProblemState`` matching ``stage_names``."""
    state = MagicMock(spec=data_structures.ProblemState)
    state.rust = MagicMock()
    state.rust.stages = [_stub_stage(name) for name in stage_names]
    return cast(data_structures.ProblemState, state)


def _stub_ctx(*, worker_ids_by_stage: list[list[str]] | None = None) -> data_structures.AutoscalePlanContext:
    """Build a sentinel planner context for ``donor_warmup_excluded_ids``."""
    ctx = MagicMock(spec=data_structures.AutoscalePlanContext)
    ctx.worker_ids_by_stage.return_value = worker_ids_by_stage or [[]]
    return cast(data_structures.AutoscalePlanContext, ctx)


def _stub_pipeline(stage_names: tuple[str, ...] = ("s0",)) -> PipelineModel:
    """Build a ``PipelineModel`` whose attribute surface matches the builder's reads."""
    pipeline = MagicMock(spec=PipelineModel)
    pipeline.problem = _stub_problem(stage_names)
    pipeline.stage_names = stage_names
    pipeline.config = MagicMock(spec=SaturationAwareConfig)
    pipeline.config.enable_memory_pressure_gate = False
    return cast(PipelineModel, pipeline)


def _stub_ledgers() -> SchedulerLedgers:
    """Build a ``SchedulerLedgers`` whose mutable fields are observable.

    ``MagicMock`` instances are returned for the nested ``warmup`` and
    ``stuck_plan`` attributes (not ``spec=<concrete>``) so the test can
    stub their methods directly without ``spec``-enforced signature
    checks getting in the way of mock-only attributes like
    ``return_value`` / ``side_effect``.
    """
    ledgers = MagicMock(spec=SchedulerLedgers)
    ledgers.cycle_counter = 0
    ledgers.worker_ages = {}
    warmup = MagicMock()
    warmup.excluded_ids.return_value = frozenset()
    ledgers.warmup = warmup
    stuck_plan = MagicMock()
    stuck_plan.snapshot.return_value = {"s0": 7}
    ledgers.stuck_plan = stuck_plan
    ledgers.memory_pressure = MagicMock()
    return cast(SchedulerLedgers, ledgers)


def _build_builder(
    *,
    ledgers: SchedulerLedgers,
    regime: RegimeController | None = None,
    resolver: ThresholdResolver | None = None,
    pipeline: PipelineModel | None = None,
) -> PreflightBuilder:
    """Construct a ``PreflightBuilder`` with mock collaborators."""
    return PreflightBuilder(
        ledgers=ledgers,
        regime=regime or cast(RegimeController, MagicMock(spec=RegimeController)),
        threshold_resolver=resolver or cast(ThresholdResolver, MagicMock(spec=ThresholdResolver)),
        pipeline=pipeline or _stub_pipeline(),
        pipeline_name="p",
    )


def _build_with_ctx(
    builder: PreflightBuilder,
    *,
    problem_state: data_structures.ProblemState,
    ctx: data_structures.AutoscalePlanContext | None = None,
    time: float = 0.0,
) -> PreflightResult:
    """Invoke ``builder.build`` with the planner-context factory patched out."""
    target = "cosmos_xenna.pipelines.private.scheduling_py.lifecycle.preflight.data_structures.AutoscalePlanContext"
    with patch(f"{target}.from_problem_state", return_value=ctx or _stub_ctx()):
        return builder.build(time=time, problem_state=problem_state)


class TestShapeInvariant:
    """``problem_state`` shape drift raises before any other step runs."""

    def test_length_mismatch_raises_scheduler_invariant_error(self) -> None:
        """Stage count drift raises ``SchedulerInvariantError`` attributed to the manual phase boundary."""
        ledgers = _stub_ledgers()
        builder = _build_builder(ledgers=ledgers, pipeline=_stub_pipeline(("s0", "s1")))
        problem_state = _stub_problem_state(("s0",))
        with pytest.raises(SchedulerInvariantError, match="Before manual"):
            builder.build(time=0.0, problem_state=problem_state)
        assert ledgers.cycle_counter == 0

    def test_name_mismatch_raises_scheduler_invariant_error(self) -> None:
        """Per-index stage-name drift raises ``SchedulerInvariantError`` attributed to the manual phase boundary."""
        ledgers = _stub_ledgers()
        builder = _build_builder(ledgers=ledgers, pipeline=_stub_pipeline(("a",)))
        problem_state = _stub_problem_state(("b",))
        with pytest.raises(SchedulerInvariantError, match="Before manual"):
            builder.build(time=0.0, problem_state=problem_state)
        assert ledgers.cycle_counter == 0


class TestCycleCounterAdvance:
    """The cycle counter advances by exactly one per call."""

    def test_cycle_counter_increments_by_one(self) -> None:
        """``ledgers.cycle_counter`` advances from ``N`` to ``N+1`` on a successful call."""
        ledgers = _stub_ledgers()
        ledgers.cycle_counter = 11
        builder = _build_builder(ledgers=ledgers)
        result = _build_with_ctx(builder, problem_state=_stub_problem_state())
        assert ledgers.cycle_counter == 12
        assert result.cycle.cycle_counter == 12


class TestWarmupRefresh:
    """The warmup tracker refresh runs once with the cycle snapshot."""

    def test_warmup_refresh_is_called_with_problem_state_and_time(self) -> None:
        """``warmup.refresh`` is invoked with the cycle's ``problem_state`` and the cycle clock."""
        ledgers = _stub_ledgers()
        builder = _build_builder(ledgers=ledgers)
        problem_state = _stub_problem_state()
        _build_with_ctx(builder, problem_state=problem_state, time=123.5)
        cast(MagicMock, ledgers.warmup.refresh).assert_called_once_with(problem_state, now=123.5)


class TestRegimeBeforeThresholdResolution:
    """Regime update fires before threshold resolution in the documented order."""

    def test_regime_update_runs_before_threshold_resolution(self) -> None:
        """``regime.update`` is recorded before ``threshold_resolver.ensure_resolved`` in the call log."""
        log: list[str] = []
        regime = MagicMock(spec=RegimeController)
        regime.update.side_effect = lambda _state: log.append("regime")
        resolver = MagicMock(spec=ThresholdResolver)
        resolver.ensure_resolved.side_effect = lambda _state: log.append("thresholds")
        builder = _build_builder(
            ledgers=_stub_ledgers(),
            regime=cast(RegimeController, regime),
            resolver=cast(ThresholdResolver, resolver),
        )
        _build_with_ctx(builder, problem_state=_stub_problem_state())
        assert log == ["regime", "thresholds"]


class TestStuckPlanSnapshotBeforeRegime:
    """Stuck-plan counter snapshot is captured before the regime update mutates state."""

    def test_stuck_plan_snapshot_is_captured_before_regime_update(self) -> None:
        """``stuck_plan.snapshot`` runs before ``regime.update`` so the snapshot reflects pre-regime state."""
        log: list[str] = []
        ledgers = _stub_ledgers()

        def _snapshot() -> dict[str, int]:
            log.append("snapshot")
            return {"s0": 7}

        cast(MagicMock, ledgers.stuck_plan.snapshot).side_effect = _snapshot
        regime = MagicMock(spec=RegimeController)
        regime.update.side_effect = lambda _state: log.append("regime")
        builder = _build_builder(ledgers=ledgers, regime=cast(RegimeController, regime))
        result = _build_with_ctx(builder, problem_state=_stub_problem_state())
        assert log == ["snapshot", "regime"]
        assert result.prev_stuck_plan_counters == {"s0": 7}


class TestDonorWarmupExclusionBuild:
    """``donor_warmup_excluded_ids`` is built from ``warmup.excluded_ids`` with the per-cycle inputs."""

    def test_donor_warmup_excluded_ids_is_set_on_returned_cycle(self) -> None:
        """The cycle exposes the frozenset returned by ``warmup.excluded_ids`` verbatim."""
        ledgers = _stub_ledgers()
        excluded = frozenset({"w1"})
        cast(MagicMock, ledgers.warmup.excluded_ids).return_value = excluded
        builder = _build_builder(ledgers=ledgers)
        result = _build_with_ctx(builder, problem_state=_stub_problem_state())
        assert isinstance(result.cycle, AutoscaleCycle)
        assert result.cycle.donor_warmup_excluded_ids is excluded


class TestPreflightResultIsImmutable:
    """``PreflightResult`` is the contracted ``@attrs.frozen`` value object."""

    def test_preflight_result_is_frozen(self) -> None:
        """Attempting to mutate ``PreflightResult.cycle`` raises ``attrs.exceptions.FrozenInstanceError``."""
        ledgers = _stub_ledgers()
        builder = _build_builder(ledgers=ledgers)
        result = _build_with_ctx(builder, problem_state=_stub_problem_state())
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            result.cycle = result.cycle  # type: ignore[misc]
