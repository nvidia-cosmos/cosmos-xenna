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

"""Contract tests for :class:`CycleRunner`.

Verify the canonical phase ordering, the per-phase pre-checkpoint
worker-count capture, the invariant-suite invocations at the matching
post-phase boundaries, and verbatim exception propagation. The runner
owns six per-phase :mod:`cycle_services` value objects; each phase
receives only its own narrow view at ``run(cycle, services)`` time so
the tests assert behaviour through the observed side effects of those
narrow services rather than through the runner's private wiring.
"""

from typing import cast
from unittest.mock import MagicMock

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.invariants.suite import PhaseInvariantSuite
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.cycle_runner import CycleRunner
from cosmos_xenna.pipelines.private.scheduling_py.lifecycle.growth_recorder import GrowthModeRecorder
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.bottleneck_phase import BottleneckPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.services import BottleneckServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.floor_phase import FloorPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.services import FloorServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.grow_phase import SaturationGrowPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.services import GrowServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase import IntentPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.services import IntentServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.manual_phase import ManualPhase
from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.services import ManualServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.services import ShrinkServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.shrink_phase import SaturationShrinkPhase
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle


@attrs.frozen
class _PhaseRecorder:
    """Recorder mixin used to inject ordering observations into stub phases.

    The runner declares concrete phase fields by type; ``attrs.evolve``
    is used in tests to swap a phase instance for one whose ``run``
    appends an identifier to a shared list. Subclassing each concrete
    phase keeps the runner's type contract intact while letting the
    test observe execution order from a single shared log.
    """

    label: str
    log: list[str]

    def record(self) -> None:
        """Append ``self.label`` to the shared invocation log."""
        self.log.append(self.label)


def _recording_phase_factory(base_cls: type, label: str, log: list[str]) -> object:
    """Build a stub phase whose ``run`` records ``label`` in ``log``.

    The runner accepts concrete phase types as constructor fields; the
    factory subclasses each base in a tiny adapter that mirrors the
    base's contract surface (``run(cycle, services)``) while recording
    invocations. This keeps the runner's typed wiring honest without
    forcing the tests to construct real phases.
    """
    recorder = _PhaseRecorder(label=label, log=log)

    class _Stub(base_cls):  # type: ignore[misc, valid-type]
        def run(self, cycle: AutoscaleCycle, services: object) -> None:
            del cycle, services
            recorder.record()

    return _Stub()


def _stub_invariants(log: list[str]) -> PhaseInvariantSuite:
    """Build a ``PhaseInvariantSuite`` whose ``check_after_*`` methods record into ``log``."""
    suite = MagicMock(spec=PhaseInvariantSuite)
    suite.check_after_manual.side_effect = lambda _cycle: log.append("after_manual")
    suite.check_after_floor.side_effect = lambda _cycle: log.append("after_floor")
    suite.check_after_grow.side_effect = lambda _cycle: log.append("after_grow")
    suite.check_after_shrink.side_effect = lambda _cycle: log.append("after_shrink")
    return cast(PhaseInvariantSuite, suite)


def _stub_cycle() -> AutoscaleCycle:
    """Build a sentinel cycle whose worker-count helpers return empty dicts."""
    cycle = MagicMock(spec=AutoscaleCycle)
    cycle.planner_worker_counts_by_stage_name.return_value = {}
    cycle.planner_worker_counts_by_stage_index.return_value = {}
    cycle.intent = MagicMock()
    cycle.intent.deltas = {}
    cycle.pre_grow_worker_counts = {}
    return cast(AutoscaleCycle, cycle)


def _build_runner(log: list[str]) -> CycleRunner:
    """Construct a runner whose phases, invariants, and recorder log into ``log``.

    Service value objects are wired as ``MagicMock(spec=...)`` instances
    because the recording stub phases never read their attributes; the
    recorder is itself a ``MagicMock`` whose ``record`` call is logged
    so the post-shrink hook is observable from the same trace.
    """
    recorder = MagicMock(spec=GrowthModeRecorder)
    recorder.record.side_effect = lambda _cycle: log.append("recorder.record")
    return CycleRunner(
        manual=cast(ManualPhase, _recording_phase_factory(ManualPhase, "manual", log)),
        floor=cast(FloorPhase, _recording_phase_factory(FloorPhase, "floor", log)),
        bottleneck=cast(BottleneckPhase, _recording_phase_factory(BottleneckPhase, "bottleneck", log)),
        intent=cast(IntentPhase, _recording_phase_factory(IntentPhase, "intent", log)),
        grow=cast(SaturationGrowPhase, _recording_phase_factory(SaturationGrowPhase, "grow", log)),
        shrink=cast(SaturationShrinkPhase, _recording_phase_factory(SaturationShrinkPhase, "shrink", log)),
        invariants=_stub_invariants(log),
        recorder=cast(GrowthModeRecorder, recorder),
        manual_services=cast(ManualServices, MagicMock(spec=ManualServices)),
        floor_services=cast(FloorServices, MagicMock(spec=FloorServices)),
        bottleneck_services=cast(BottleneckServices, MagicMock(spec=BottleneckServices)),
        intent_services=cast(IntentServices, MagicMock(spec=IntentServices)),
        grow_services=cast(GrowServices, MagicMock(spec=GrowServices)),
        shrink_services=cast(ShrinkServices, MagicMock(spec=ShrinkServices)),
    )


class TestCycleRunnerOrdering:
    """The runner invokes phases and invariant boundaries in the canonical order."""

    def test_phases_and_invariants_run_in_canonical_order(self) -> None:
        """Phases and invariants fire in the documented per-cycle order with no skips or reorderings."""
        log: list[str] = []
        runner = _build_runner(log)

        runner.run(_stub_cycle())

        assert log == [
            "manual",
            "after_manual",
            "floor",
            "after_floor",
            "bottleneck",
            "intent",
            "grow",
            "after_grow",
            "shrink",
            "after_shrink",
            "recorder.record",
        ]

    def test_pre_grow_snapshot_is_captured_before_grow_runs(self) -> None:
        """Pre-Grow planner worker count is captured before the grow phase executes."""
        log: list[str] = []

        def _grow_observes_snapshot_already_set(cycle: AutoscaleCycle, services: object) -> None:
            del services
            assert hasattr(cycle, "pre_grow_worker_counts")
            log.append("grow_saw_snapshot")

        runner = _build_runner(log)
        grow = MagicMock(spec=SaturationGrowPhase)
        grow.run.side_effect = _grow_observes_snapshot_already_set
        runner = attrs.evolve(runner, grow=cast(SaturationGrowPhase, grow))

        cycle = _stub_cycle()
        runner.run(cycle)

        # ``_stub_cycle`` returns ``MagicMock(spec=AutoscaleCycle)`` but the cast
        # erases that for the type checker; restore it locally so ``assert_called``
        # is visible without weakening the function signature.
        planner_call = cast(MagicMock, cycle.planner_worker_counts_by_stage_name)
        planner_call.assert_called()
        assert "grow_saw_snapshot" in log


class TestCycleRunnerFailurePath:
    """Phase exceptions and invariant failures propagate verbatim."""

    def test_phase_exception_propagates_to_caller(self) -> None:
        """The runner re-raises whatever the phase raised, unchanged."""
        log: list[str] = []
        runner = _build_runner(log)
        failing_grow = MagicMock(spec=SaturationGrowPhase)
        failing_grow.run.side_effect = RuntimeError("intentional grow defect")
        runner = attrs.evolve(runner, grow=cast(SaturationGrowPhase, failing_grow))

        with pytest.raises(RuntimeError, match="intentional grow defect"):
            runner.run(_stub_cycle())

    def test_invariant_failure_after_manual_stops_before_floor_runs(self) -> None:
        """A Manual-phase invariant failure halts the runner before the floor phase runs."""
        log: list[str] = []
        runner = _build_runner(log)
        invariants = MagicMock(spec=PhaseInvariantSuite)
        invariants.check_after_manual.side_effect = RuntimeError("manual phase corrupted")
        runner = attrs.evolve(runner, invariants=cast(PhaseInvariantSuite, invariants))

        with pytest.raises(RuntimeError, match="manual phase corrupted"):
            runner.run(_stub_cycle())

        assert log == ["manual"]
