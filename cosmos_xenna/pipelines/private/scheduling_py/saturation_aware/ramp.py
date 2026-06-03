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

"""Cold-start worker ramp for the saturation-aware scheduler.

Bounds how fast any not-yet-trusted stage may grow while the fragmentation
solver is still sizing it from placeholder throughput. With no completed sample
a stage is capped at one worker; while warming, allowed growth scales with
sample confidence; once the speed estimate is trusted, the stage is uncapped
and the solver decides. Pure and native-extension-free, so it is unit-testable
without the solver.
"""

import math

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig


@attrs.frozen
class StageRampInput:
    """One stage's inputs to the cold-start ramp.

    Attributes:
        current_workers: Live pre-solve worker count.
        deleted_count: Workers the solver proposes to delete this cycle.
        proposed_post: Post-solve worker count the solver proposes
            (``current_workers + new - deleted``).
        sample_count: Measured throughput samples observed for this stage.
    """

    current_workers: int
    deleted_count: int
    proposed_post: int
    sample_count: int


@attrs.frozen
class RampDecision:
    """Cold-start cap outcome for one stage.

    Attributes:
        cap: Maximum post-solve worker count this cycle, or ``None`` when the
            stage is trusted (uncapped).
        keep_new: How many of the solver's proposed new workers to keep, or
            ``None`` when none are trimmed (keep all).
        reason: Short tag for logs (``cold``, ``warming``, or ``uncapped``).
    """

    cap: int | None
    keep_new: int | None
    reason: str


@attrs.frozen
class ColdStartRampPolicy:
    """Evidence-scaled cold-start cap for not-yet-trusted stages.

    Allowed growth interpolates from one worker (no evidence) to the solver's
    full proposal (trusted), scaled by sample confidence.

    Attributes:
        config: Operator tunables (provides the trust threshold).
    """

    config: SaturationAwareConfig

    def decide(self, stage: StageRampInput) -> RampDecision:
        """Return the cold-start cap and trim count for one stage.

        No completed sample caps the stage at one worker; while warming, the
        allowed growth is ``ceil(confidence * solver_growth)`` (at least one);
        trusted stages are uncapped.
        """
        min_data_points = self.config.speed_estimation_min_data_points
        if stage.sample_count >= min_data_points:
            return RampDecision(cap=None, keep_new=None, reason="uncapped")
        if stage.sample_count == 0:
            # No completed task means no evidence: hold at a single worker so a
            # 0-sample stage cannot creep upward cycle after cycle.
            cap = 1
            reason = "cold"
        else:
            confidence = stage.sample_count / min_data_points
            solver_growth = max(0, stage.proposed_post - stage.current_workers)
            cap = stage.current_workers + max(1, math.ceil(confidence * solver_growth))
            reason = "warming"
        if stage.proposed_post <= cap:
            return RampDecision(cap=cap, keep_new=None, reason=reason)
        keep_new = max(0, cap - (stage.current_workers - stage.deleted_count))
        return RampDecision(cap=cap, keep_new=keep_new, reason=reason)
