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
and the solver decides. A stage that still has work waiting but produces no
sample within a full speed-estimation window is treated as a confirmed
slow-starter and is uncapped so all its workers spawn and warm up in parallel
rather than one at a time. Pure and native-extension-free, so it is
unit-testable without the solver.
"""

import enum
import math

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig


class RampReason(enum.StrEnum):
    """Why the cold-start ramp reached its decision (log/diagnostic tag).

    Attributes:
        COLD: No completed sample yet, still within the warmup window, and no
            upstream evidence the pipeline is feeding this stage; held at one
            worker.
        PIPELINE_WARMING: No completed sample yet, but work is waiting and an
            upstream stage is already trusted (the pipeline is provably feeding
            this stage); allowed one extra worker this cycle.
        WARMING: Some samples but below the trust threshold; growth scaled by
            confidence.
        SLOW_START: No sample after a full speed-estimation window; released to
            the solver as a confirmed slow-starter.
        UNCAPPED: Enough samples to trust the speed estimate; the solver owns
            growth.
    """

    COLD = "cold"
    PIPELINE_WARMING = "pipeline_warming"
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
        has_pending_work: Whether the stage has work waiting (queued, pool-queued,
            or in-flight). Gates the slow-starter release so a stage merely
            starved of input is not over-spawned from placeholder throughput.
        has_upstream_evidence: Whether any upstream stage already has a trusted
            speed. Proof the pipeline is feeding work down the chain, used to let
            a 0-sample stage grow by one worker per cycle before its own first
            sample lands. Resource-shape-agnostic by design (a boolean, not a
            GPU fraction), so the pure ramp stays measurement-driven.
    """

    current_workers: int
    deleted_count: int
    proposed_post: int
    sample_count: int
    stage_age_s: float
    has_pending_work: bool
    has_upstream_evidence: bool


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


def decide(stage: StageRampInput, config: SaturationAwareConfig) -> RampDecision:
    """Return the cold-start cap and trim count for one stage.

    Evidence-scaled cold-start cap for not-yet-trusted stages: allowed growth
    interpolates from one worker (no evidence) to the solver's full proposal
    (trusted), scaled by sample confidence. No completed sample caps the stage
    at one worker until a full speed-estimation window has elapsed with work
    still waiting, after which the stage is treated as a slow-starter and
    uncapped; while warming, the allowed growth is
    ``ceil(confidence * solver_growth)`` (at least one); trusted stages are
    uncapped.

    Pipeline-evidence warming bridges the gap before a stage's own first sample:
    a 0-sample stage that has work waiting and at least one already-trusted
    upstream stage is allowed one extra worker per cycle, so an expensive
    downstream stage can begin loading a
    second model in parallel instead of idling at a single worker. Growth stays
    at ``+1`` per cycle, so a 0-sample stage can never perform the large
    first-cycle over-spawn the cold cap was built to prevent, and the
    upstream-trust gate keeps this branch dark on cycle one (before any stage is
    trusted, no upstream evidence exists).

    Args:
        stage: One stage's pre-solve counts, sample count, age, and work flag.
        config: Operator tunables (provides the trust threshold and window).

    Returns:
        The cold-start :class:`RampDecision` (cap, trim count, and reason tag).
    """
    min_data_points = config.speed_estimation_min_data_points
    if stage.sample_count >= min_data_points:
        return RampDecision(cap=None, keep_new=None, reason=RampReason.UNCAPPED)
    if stage.sample_count == 0:
        window_elapsed = stage.stage_age_s >= config.speed_estimation_window_s
        if window_elapsed and stage.has_pending_work:
            # A full estimation window has passed with no completed task yet
            # work is still waiting: this is a slow-warmup stage (its first
            # result lands long after the window). Trust the solver so every
            # worker spawns now and their models load in parallel. The
            # pending-work gate keeps a stage merely starved of input capped,
            # so the solver cannot over-spawn it from placeholder throughput.
            return RampDecision(cap=None, keep_new=None, reason=RampReason.SLOW_START)
        if stage.current_workers >= 1 and stage.has_pending_work and stage.has_upstream_evidence:
            # No completed task yet, but work is waiting and an upstream stage is
            # already trusted: the pipeline is probably feeding this stage. Grow
            # by exactly one worker so an expensive downstream stage can begin a
            # second model load in parallel before its own first sample lands.
            # The +1 bound preserves the original anti-fragmentation guarantee
            # (no large first-cycle over-spawn from placeholder throughput), and
            # the current_workers >= 1 guard means this only accelerates a stage
            # that has already cleared the very first cold cap.
            cap = stage.current_workers + 1
            reason = RampReason.PIPELINE_WARMING
        else:
            # No completed task yet and no pipeline evidence (still within the
            # warmup window with no trusted upstream stage, or no work waiting):
            # hold at a single worker so a 0-sample stage cannot creep upward
            # cycle after cycle while it could still produce its first sample.
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
