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
solver is still sizing it from placeholder throughput. One generic rule governs
every stage, regardless of resource shape (CPU-only, whole-GPU, or
fractional-GPU): a not-yet-trusted stage may grow by at most one worker per
cycle, and only when the stage has its own pending work to feed a new worker.
With no completed sample and no pending work the stage is held at one worker;
once the speed estimate is trusted the stage is capped at its capacity growth
target ``w_target`` (the per-cycle growth ceiling), so the solver may place and
degrade within that ceiling but never grow the stage past the size the capacity
model computed. A stage that still has work waiting but produces no sample within
a full speed-estimation window is treated as a confirmed slow-starter and is
uncapped so all its workers spawn and warm up in parallel rather than one at a
time. Pure and native-extension-free, so it is unit-testable without the solver.
"""

import enum

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig


class RampReason(enum.StrEnum):
    """Why the cold-start ramp reached its decision (log/diagnostic tag).

    Attributes:
        COLD: No completed sample yet and the slow-starter release did not fire
            (no pending work, or no live worker to accelerate yet); held at one
            worker. Reached within the warmup window and also past it whenever
            the slow-starter release is gated by missing pending work.
        PIPELINE_WARMING: No completed sample yet, but the stage has a live worker
            and its own pending work; allowed one extra worker this cycle.
        WARMING: Some samples but below the trust threshold; allowed one extra
            worker per cycle under the same pending-work gate, never scaled by
            the solver's proposal.
        SLOW_START: No sample after a full speed-estimation window with work
            waiting; released to the solver as a confirmed slow-starter.
        CAPPED: Enough samples to trust the speed estimate; held at the capacity
            growth target ``w_target`` (the per-cycle growth ceiling).
        UNCAPPED: Enough samples to trust the speed estimate but no capacity
            target this cycle (no measured bottleneck); the solver owns growth.
    """

    COLD = "cold"
    PIPELINE_WARMING = "pipeline_warming"
    WARMING = "warming"
    SLOW_START = "slow_start"
    CAPPED = "capped"
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
        pending_work_age_s: Seconds this stage has had pending work waiting
            (reset when its work drains). Lets a stage that produced no sample
            while work stayed blocked for a full speed-estimation window be
            released to the solver as a confirmed slow-starter.
        has_pending_work: Whether the stage has work waiting (queued, pool-queued,
            or in-flight). Gates the slow-starter release so a stage merely
            starved of input is not over-spawned from placeholder throughput, and
            authorizes one warming worker when the stage has its own backlog.
        w_target: Capacity growth target for this stage this cycle, or ``None``
            when no capacity target is available (no measured bottleneck yet).
            Consulted only once the stage is trusted, where it is the per-cycle
            growth ceiling; ``None`` then falls back to uncapped.
    """

    current_workers: int
    deleted_count: int
    proposed_post: int
    sample_count: int
    pending_work_age_s: float
    has_pending_work: bool
    w_target: int | None


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


def _apply_cap(stage: StageRampInput, cap: int, reason: RampReason) -> RampDecision:
    """Trim the solver's new workers so the post-solve count stays within ``cap``.

    Computes how many of the solver's proposed new workers to keep so the
    post-solve count does not exceed ``cap``, flooring the kept count at zero so
    a cap below the surviving worker count never converts the solver's proposal
    into a forced delete.

    Args:
        stage: One stage's pre-solve counts and solver proposal.
        cap: Maximum post-solve worker count this cycle.
        reason: The branch that produced this cap (log/diagnostic tag).

    Returns:
        The :class:`RampDecision`; ``keep_new`` is ``None`` when the proposal is
        already within ``cap`` (nothing trimmed).
    """
    if stage.proposed_post <= cap:
        return RampDecision(cap=cap, keep_new=None, reason=reason)
    keep_new = max(0, cap - (stage.current_workers - stage.deleted_count))
    return RampDecision(cap=cap, keep_new=keep_new, reason=reason)


def decide(stage: StageRampInput, config: SaturationAwareConfig) -> RampDecision:
    """Return the per-cycle growth cap and trim count for one stage.

    One generic rule for every stage, independent of resource shape: a
    not-yet-trusted stage grows by at most one worker per cycle, and only when
    the stage has its own pending work to feed the new worker. A trusted stage
    (sample count at or above the threshold) is capped at its capacity growth
    target ``w_target``; with no capacity target it is uncapped and the solver
    owns growth. A 0-sample stage with work still waiting after a full
    speed-estimation window is released to the solver as a confirmed
    slow-starter.

    Neither the fixed one-per-cycle warming step nor the trusted ``w_target``
    cap scales with the solver's proposal, so a stage can never convert a large
    solver proposal into a first-cycle burst.

    Args:
        stage: One stage's pre-solve counts, sample count, age, and SAT signals.
        config: Operator tunables (provides the trust threshold and window).

    Returns:
        The :class:`RampDecision` (cap, trim count, and reason tag).
    """
    min_data_points = config.speed_estimation_min_data_points
    if stage.sample_count >= min_data_points:
        if stage.w_target is None:
            # Trusted, but capacity has no useful target this cycle (no measured
            # bottleneck): let the solver own growth.
            return RampDecision(cap=None, keep_new=None, reason=RampReason.UNCAPPED)
        # Trusted: SAT's capacity target is the per-cycle growth ceiling. The
        # shared cap-application trims growth without ever forcing a shrink
        # (keep_new floors at zero); the scale-down floor still owns shrink.
        return _apply_cap(stage, stage.w_target, RampReason.CAPPED)

    is_cold = stage.sample_count == 0
    if is_cold and stage.pending_work_age_s >= config.speed_estimation_window_s and stage.has_pending_work:
        # A full estimation window has passed with no completed task while work
        # is still waiting: a slow-warmup stage whose first result lands long
        # after the window. Trust the solver so every worker spawns now and their
        # models load in parallel. The pending-work gate keeps a stage merely
        # starved of input capped, so the solver cannot over-spawn it from
        # placeholder throughput.
        return RampDecision(cap=None, keep_new=None, reason=RampReason.SLOW_START)

    # Authorize one warming worker only when the stage has its own pending work
    # to feed it; a locally dry stage has no pending work and stays capped.
    if stage.current_workers >= 1 and stage.has_pending_work:
        # The current_workers >= 1 guard means this only accelerates a stage that
        # has already cleared the first cold cap; cold stages still start at one.
        cap = stage.current_workers + 1
        reason = RampReason.PIPELINE_WARMING if is_cold else RampReason.WARMING
    elif is_cold:
        # No completed task and no usable pending work: hold at a single worker
        # so a 0-sample stage cannot creep upward while it could still produce
        # its first sample.
        cap = 1
        reason = RampReason.COLD
    else:
        # Warming with no pending work: hold at the current size rather than
        # scaling growth off the solver proposal.
        cap = stage.current_workers
        reason = RampReason.WARMING

    return _apply_cap(stage, cap, reason)
