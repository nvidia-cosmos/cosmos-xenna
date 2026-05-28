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

"""Regression test for the bottlenecked-stage starvation pattern.

The captioning stage observed during a production video pipeline run
held a long input queue (~60 tasks) and a high per-task service time
(~21 s) while another stage held idle slots. Under the previous
discrete grow magnitudes (+1 / +2 per cycle), the captioning stage
took many cycles to catch up. The capacity sizer's closed-form target
captures the demand in one cycle so the planner can grow the stage
to capacity in a single decision (still capped by
``aggressive_growth_max_per_cycle`` on a single cycle).
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.decisions import compute_delta
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure import compute_capacity_target_workers
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import GrowthMode, StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Default per-stage config; no per-stage overrides for the regression scenario."""
    return SaturationAwareStageConfig()


class TestStarvedCaptioningStageGrowsToCapacity:
    """The Stage-09-style scenario: large queue, slow service, very few workers.

    Under the previous coarse-magnitude system the stage recovered in
    O(queue / +2) cycles. The capacity-driven sizer collapses the
    multi-cycle climb into one decision sized to the observed demand,
    bounded by ``aggressive_growth_max_per_cycle``.
    """

    def test_capacity_target_reflects_full_demand(self) -> None:
        """target_rate=2.05 t/s, D_k=21 s, util=0.85 -> 51 slots = 51 workers."""
        target = compute_capacity_target_workers(
            queue_depth=60,
            observed_throughput=0.05,
            d_k_seconds=21.0,
            slots_per_worker=1,
            target_backlog_seconds=30.0,
            utilization_target=0.85,
        )
        assert target == 51

    def test_compute_delta_grows_at_per_cycle_cap_when_target_far_above_current(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """current=1, target=51 -> shortfall 50, capped by aggressive_growth_max_per_cycle (4)."""
        delta = compute_delta(
            StageState.SATURATED_CRITICAL,
            GrowthMode.ACQUIRING,
            current_workers=1,
            capacity_target_workers=51,
            config=cfg,
        )
        assert delta == cfg.aggressive_growth_max_per_cycle

    def test_compute_delta_stops_growing_once_at_target(self, cfg: SaturationAwareStageConfig) -> None:
        """Closed-loop stability: once current == target, no further grow even on SATURATED."""
        delta = compute_delta(
            StageState.SATURATED,
            GrowthMode.ACQUIRING,
            current_workers=51,
            capacity_target_workers=51,
            config=cfg,
        )
        assert delta == 0


class TestOverProvisionedDonorShrinksFreeingSlots:
    """The dual scenario: a stage holding idle slots while the bottleneck starves.

    The prior 4-state classifier already eliminated the ``STARVED`` short-circuit
    that previously locked donor stages in place. The capacity sizer composes
    on top: an OVER_PROVISIONED stage with a lower capacity target shrinks
    proportionally rather than emitting a fixed -1 per cycle.
    """

    def test_over_provisioned_shrinks_to_target(self, cfg: SaturationAwareStageConfig) -> None:
        """current=20, target=4 -> excess 16; fraction cap floor(20*0.05)=1; -1 per cycle."""
        delta = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=20,
            capacity_target_workers=4,
            config=cfg,
        )
        assert delta == -1

    def test_over_provisioned_at_target_does_not_shrink(self, cfg: SaturationAwareStageConfig) -> None:
        """Donor cannot starve itself once it has reached the capacity target."""
        delta = compute_delta(
            StageState.OVER_PROVISIONED,
            GrowthMode.TRACKING,
            current_workers=4,
            capacity_target_workers=4,
            config=cfg,
        )
        assert delta == 0
