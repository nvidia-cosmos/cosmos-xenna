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

"""Cross-cycle mutable state aggregated by ``SchedulerLedgers``.

Concentrates every mutable map / counter / domain detector that
survives across autoscale cycles into a single container. The
container is owned by ``SaturationAwareScheduler``; phases and
extracted services consume it through narrower interfaces.

Ownership rules:

- ``SaturationAwareScheduler`` owns exactly one ``SchedulerLedgers``
  instance built once in ``__init__`` and reset in-place during
  ``setup()``.
- Per-cycle outputs (bottleneck signals, intent deltas, planner
  context) live on ``AutoscaleCycle``, not on the ledger. The
  ledger is for state that must outlive a single
  ``autoscale()`` call.
- Mutations are local to the phase or per-stage decision
  pipeline that owns the corresponding state. The ledger
  exposes the state as plain mutable mappings; setters are
  intentionally omitted because every writer lives in-process
  inside the same package boundary.

The ledger is a single ``@attrs.define`` so future component
extractions (``ThresholdResolver``, ``RegimeController``, etc.)
can take a ``SchedulerLedgers`` reference instead of the whole
scheduler.
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.cluster.measurements import MeasurementCollector
from cosmos_xenna.pipelines.private.scheduling_py.cluster.memory_pressure import MemoryPressureMonitor
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime import RegimeDetectorState
from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_engagement_state import BottleneckEngagementState
from cosmos_xenna.pipelines.private.scheduling_py.state.floor_stuck_counters import FloorStuckCounters
from cosmos_xenna.pipelines.private.scheduling_py.state.heterogeneity_state import HeterogeneityWarnState
from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state.sk_ewma_store import SkEwmaStore
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageStateMap
from cosmos_xenna.pipelines.private.scheduling_py.state.stuck_plan_ledger import StuckPlanLedger
from cosmos_xenna.pipelines.private.scheduling_py.warmup.warmup import WarmupTracker


@attrs.define
class SchedulerLedgers:
    """Cross-cycle mutable state for the saturation-aware scheduler.

    Holds every map / counter / domain detector that must persist
    across ``autoscale()`` calls. Fields are grouped by purpose;
    field-level semantics live next to each attribute.

    Attributes:
        warmup: Wall-clock first-seen-READY tracker; clocks the
            warmup grace independently of planner ages.
        measurements: Thread-safe measurement collector
            decoupling the multi-tick ingest path from the
            per-cycle consume path.
        memory_pressure: Cluster object-store pressure gate.
        cycle_counter: Monotonic per-pipeline cycle counter
            incremented at the top of every ``autoscale()`` call.
        stage_states: Per-stage runtime state map keyed by stage
            name; ``setup()`` instantiates an entry for every
            stage in the problem.
        recommendation_histories: Per-stage stabilization-window
            buffers; the per-stage decision pipeline writes
            recommendations and gates against the buffer's
            history.
        worker_ages: Per-worker cross-cycle age counters.
        s_k_ewma: Per-stage intrinsic service-time EWMA store used
            by the capacity sizer; the Bottleneck phase writes via
            :meth:`SkEwmaStore.update`.
        last_donation_cycle: Per-stage donor anti-flap ledger;
            stages map to the cycle at which they last donated.
        floor_stuck_counters: Per-stage floor-stuck grace counter
            store; the Floor phase increments / resets via
            :meth:`FloorStuckCounters.increment_stuck` /
            :meth:`FloorStuckCounters.reset_for`.
        stuck_plan: Composite owner for the Grow-phase stuck-plan
            counter dict + WARN-to-INFO detector. The Grow phase
            advances it through :meth:`StuckPlanLedger.record` so
            the dict and detector stay in lockstep.
        regime_state: Halfin-Whitt regime detector state for
            regime-aware aggressiveness.
        bottleneck_engagement_state: Bottleneck-engagement
            debounce ledger.
        heterogeneity_state: Cluster heterogeneity-ratio warn
            streak ledger.

    Per-phase ``AllocationFailureGate`` instances are owned by
    :class:`DonorBackedAddExecutor` and :class:`ManualGrowExecutor`;
    the bottleneck-protection logger is owned by
    :class:`ShrinkServices`. The scheduler holds those instances
    directly and forwards them to the post-cycle reporter so the
    cycle-summary observability stays intact.

    """

    warmup: WarmupTracker
    measurements: MeasurementCollector
    memory_pressure: MemoryPressureMonitor

    cycle_counter: int = 0
    stage_states: StageStateMap = attrs.Factory(dict)
    recommendation_histories: dict[str, RecommendationHistory] = attrs.Factory(dict)

    worker_ages: dict[str, int] = attrs.Factory(dict)

    s_k_ewma: SkEwmaStore = attrs.Factory(SkEwmaStore)
    last_donation_cycle: dict[str, int] = attrs.Factory(dict)
    floor_stuck_counters: FloorStuckCounters = attrs.Factory(FloorStuckCounters)

    stuck_plan: StuckPlanLedger = attrs.Factory(StuckPlanLedger)

    regime_state: RegimeDetectorState = attrs.Factory(RegimeDetectorState)
    bottleneck_engagement_state: BottleneckEngagementState = attrs.Factory(BottleneckEngagementState)
    heterogeneity_state: HeterogeneityWarnState = attrs.Factory(HeterogeneityWarnState)


__all__ = ("SchedulerLedgers",)
