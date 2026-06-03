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
and the solver decides. A stage that produces no sample within a full
speed-estimation window is treated as a confirmed slow-starter and is uncapped
so all its workers spawn and warm up in parallel rather than one at a time.
"""

import enum
import math

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig


class RampReason(enum.StrEnum):
    """Why the cold-start ramp reached its decision (log/diagnostic tag).

    Attributes:
        COLD: No completed sample yet, still within the warmup window; held at
            one worker.
        WARMING: Some samples but below the trust threshold; growth scaled by
            confidence.
        SLOW_START: No sample after a full speed-estimation window; released to
            the solver as a confirmed slow-starter.
        UNCAPPED: Enough samples to trust the speed estimate; the solver owns
            growth.
    """

    COLD = "cold"
    WARMING = "warming"
    SLOW_START = "slow_start"
    UNCAPPED = "uncapped"


@attrs.frozen
class StageRampInput:
    """One stage's inputs to the cold-start ramp.

    Attributes:
        current_workers: Live pre-solve worker count.
        deleted_count: Workers the solver proposes to delete this cycle.
        proposed_post: Post-solve worker count the solver proposes
            (``current_workers + new - deleted``).
        sample_count: Measured throughput samples observed for this stage.
        stage_age_s: Seconds since the scheduler first saw this stage. Lets a
            stage that has produced no sample within a full speed-estimation
            window be released to the solver as a confirmed slow-starter.
    """

    current_workers: int
    deleted_count: int
    proposed_post: int
    sample_count: int
    stage_age_s: float


@attrs.frozen
class RampDecision:
    """Cold-start cap outcome for one stage.

    Attributes:
        cap: Maximum post-solve worker count this cycle, or ``None`` when the
            stage is trusted (uncapped).
        keep_new: How many of the solver's proposed new workers to keep, or
            ``None`` when none are trimmed (keep all).
        reason: Which ramp branch produced this decision (log/diagnostic tag).
    """

    cap: int | None
    keep_new: int | None
    reason: RampReason


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

        No completed sample caps the stage at one worker until a full
        speed-estimation window has elapsed with no sample, after which the
        stage is treated as a slow-starter and uncapped; while warming, the
        allowed growth is ``ceil(confidence * solver_growth)`` (at least one);
        trusted stages are uncapped.
        """
        min_data_points = self.config.speed_estimation_min_data_points
        if stage.sample_count >= min_data_points:
            return RampDecision(cap=None, keep_new=None, reason=RampReason.UNCAPPED)
        if stage.sample_count == 0:
            if stage.stage_age_s >= self.config.speed_estimation_window_s:
                # A full estimation window has passed with no completed task:
                # this is a slow-warmup stage (its first result lands long
                # after the window). Trust the solver so every worker spawns
                # now.
                return RampDecision(cap=None, keep_new=None, reason=RampReason.SLOW_START)
            # No completed task yet, but still within the warmup window: hold at
            # a single worker so a 0-sample stage cannot creep upward cycle
            # after cycle while it could still produce its first sample.
            cap = 1
            reason = RampReason.COLD
        else:
            confidence = stage.sample_count / min_data_points
            solver_growth = max(0, stage.proposed_post - stage.current_workers)
            cap = stage.current_workers + max(1, math.ceil(confidence * solver_growth))
            reason = RampReason.WARMING
        if stage.proposed_post <= cap:
            return RampDecision(cap=cap, keep_new=None, reason=reason)
        keep_new = max(0, cap - (stage.current_workers - stage.deleted_count))
        return RampDecision(cap=cap, keep_new=keep_new, reason=reason)
