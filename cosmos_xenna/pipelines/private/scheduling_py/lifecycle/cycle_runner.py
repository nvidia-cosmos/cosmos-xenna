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

"""Per-cycle phase pipeline driver.

The runner owns:

* the canonical phase order (Manual -> Floor -> Bottleneck ->
  Intent -> Grow -> Shrink),
* the per-phase narrow service value object passed to each phase
  (one ``@attrs.frozen`` bundle per phase, constructed once in
  ``scheduler.setup()`` and stored on the runner),
* the named pre-phase checkpoints (Grow and Shrink capture the
  planner worker-count snapshots that the floor invariant and
  the executed-delta recorder consume),
* the runner-driven invariant suite (``PhaseInvariantSuite``),
  invoked at the matching post-phase boundary,
* delegation to the constructor-injected ``GrowthModeRecorder``
  after the post-Shrink invariant gate.

Phase classes carry only their decision logic; the runner is the
single audit trail for ordering, snapshots, and invariants. The
growth-mode state machine lives on the recorder, not on the runner.
The runner is the only object holding all six service value objects
simultaneously; phases see only their own narrow view.

::

    +----------+   +---------+   +-------+   +---------+   +---------+   +---------+
    | Manual   |-->| Floor   |-->| Bnck  |-->| Intent  |-->| Grow    |-->| Shrink  |
    +----------+   +---------+   +-------+   +---------+   +---------+   +---------+
         |              |                                        ^             ^
         v              v                                        |             |
    inv.after_     inv.after_                                    |             v
    manual         floor                                         |        inv.after_
                                                                 |        shrink
                                                       pre-grow snapshot       |
                                                                               v
                                                                       recorder.record
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.invariants.suite import PhaseInvariantSuite
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
class CycleRunner:
    """Ordered phase pipeline driver with explicit checkpoint hooks.

    Each phase is a constructor-injected ``@attrs.frozen`` Strategy
    instance reused across cycles, paired with its own narrow
    ``@attrs.frozen`` service value object. The runner is the sole
    owner of the phase order, pre-phase snapshot capture, and the
    post-phase invariant invocation. The post-Phase-D executed-delta
    recording is delegated to the ``GrowthModeRecorder``; the runner
    owns only orchestration, not domain mutation logic.
    ``scheduler.setup()`` is the only writer of the runner's
    services; subsequent ``run(cycle)`` calls reuse the same
    instances.

    Attributes:
        manual: Operator-driven manual delete/grow phase.
        floor: Per-stage floor enforcement with donor fallback.
        bottleneck: Bottleneck identification (no invariant gate).
        intent: Per-stage signed worker-count intent (no invariant
            gate).
        grow: Saturation-driven grow phase (applies positive intent
            as planner adds).
        shrink: Saturation-driven shrink phase (applies negative
            intent / ceiling overflow as planner removes).
        invariants: Suite invoked at the Manual, Floor, Grow and
            Shrink boundaries.
        recorder: Post-cycle growth-mode state-machine recorder
            invoked after the post-Shrink invariant gate.
        manual_services: Service view consumed by ``manual``.
        floor_services: Service view consumed by ``floor``.
        bottleneck_services: Service view consumed by ``bottleneck``.
        intent_services: Service view consumed by ``intent``.
        grow_services: Service view consumed by ``grow``.
        shrink_services: Service view consumed by ``shrink``.

    """

    manual: ManualPhase
    floor: FloorPhase
    bottleneck: BottleneckPhase
    intent: IntentPhase
    grow: SaturationGrowPhase
    shrink: SaturationShrinkPhase
    invariants: PhaseInvariantSuite
    recorder: GrowthModeRecorder

    manual_services: ManualServices
    floor_services: FloorServices
    bottleneck_services: BottleneckServices
    intent_services: IntentServices
    grow_services: GrowServices
    shrink_services: ShrinkServices

    def run(self, cycle: AutoscaleCycle) -> None:
        """Drive the per-cycle phase pipeline against ``cycle``.

        Phase ordering, pre-phase worker-count capture, and invariant
        invocation are all owned by this method so reading the runner
        body is the canonical narrative of one autoscale cycle. Each
        phase receives the narrow service value object the runner
        owns for it; phases never see other phases' views. The
        post-Shrink growth-mode recording is delegated to
        ``self.recorder``.

        Raises:
            SchedulerInvariantError: Propagated from the phase or
                the post-phase invariant suite; phases never swallow
                their own failures.

        """
        # Manual phase: operator-driven manual delete then grow.
        # Invariants verify planner state after both mutations.
        self.manual.run(cycle, self.manual_services)
        self.invariants.check_after_manual(cycle)

        # Floor phase: per-stage floor enforcement (with donor fallback).
        self.floor.run(cycle, self.floor_services)
        self.invariants.check_after_floor(cycle)

        # Bottleneck identification + intent planning. Neither phase
        # touches the planner directly, so no invariant gate fires
        # between them and Grow.
        self.bottleneck.run(cycle, self.bottleneck_services)
        self.intent.run(cycle, self.intent_services)

        # Grow phase: capture the pre-grow planner worker count by
        # stage name so the post-Shrink executed-delta recorder can
        # compute ``post_shrink_count - pre_grow_count`` for each
        # stage's growth-mode state machine. The post-Grow invariant
        # adds a NaN gate on top of the planner check so a corrupted
        # classifier EWMA cannot leak into the Shrink decision.
        cycle.pre_grow_worker_counts = cycle.planner_worker_counts_by_stage_name()
        self.grow.run(cycle, self.grow_services)
        self.invariants.check_after_grow(cycle)

        # Shrink phase: capture the pre-shrink planner worker count
        # by stage index so ``check_floor_after_shrink`` can
        # distinguish a Shrink defect (floor crossed during shrink)
        # from a Floor grace-window miss (floor never reached before
        # Shrink ran). The recorder runs after the invariant gate so
        # a Shrink defect cannot poison the recommendation histories.
        cycle.pre_shrink_worker_counts = cycle.planner_worker_counts_by_stage_index()
        self.shrink.run(cycle, self.shrink_services)
        self.invariants.check_after_shrink(cycle)
        self.recorder.record(cycle)


__all__ = ["CycleRunner"]
