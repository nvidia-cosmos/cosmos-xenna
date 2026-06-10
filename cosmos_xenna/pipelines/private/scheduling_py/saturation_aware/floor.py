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

"""Scale-down release gate for the saturation-aware scheduler.

A thin lower-bound gate over the capacity model. The throughput math (per-stage
``cap_src``, the global bottleneck, the smoothed sustainable arrival, and the
hold target ``w_sustain``) lives in ``capacity.py``; this module only decides,
per stage, how far the solver may shrink it this cycle:

    hold     -> floor = stabilized min(w_sustain, workers)
    release  -> floor = min(min_workers, workers)

The floor never demands growth (``floor <= workers``), so growth stays owned by
the demand multiplier. A lower hold target must persist before deletes are
allowed to follow it, while rises are accepted immediately. Release is driven by
whole-chain at-or-upstream stock so a downstream stage is not released while
upstream work is still in flight.
"""

from typing import Self

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import chain


@attrs.frozen
class FloorParams:
    """Release tuning for the scale-down floor.

    Attributes:
        release_confirm_cycles: Consecutive low-stock cycles before a stage
            releases to ``min_workers``.
        min_workers: Lower bound a stage is never shrunk below while active.
    """

    release_confirm_cycles: int
    min_workers: int = 1


@attrs.frozen
class FloorState:
    """Cross-cycle floor state, indexed by stage.

    Attributes:
        release_streak: Consecutive low-stock cycles per stage.
        held_floor: Last stabilized hold floor per stage; ``0`` means unset.
        shrink_streak: Consecutive cycles with a lower desired hold floor.
        pending_shrink_floor: Conservative lower hold floor being confirmed.
    """

    release_streak: tuple[int, ...]
    held_floor: tuple[int, ...]
    shrink_streak: tuple[int, ...]
    pending_shrink_floor: tuple[int, ...]

    @classmethod
    def initial(cls, num_stages: int) -> Self:
        """Return empty state for a pipeline with ``num_stages`` stages."""
        return cls(
            release_streak=(0,) * num_stages,
            held_floor=(0,) * num_stages,
            shrink_streak=(0,) * num_stages,
            pending_shrink_floor=(0,) * num_stages,
        )


@attrs.frozen
class FloorInputs:
    """Per-cycle observed inputs for the floor, indexed by stage.

    Attributes:
        workers: Current (observed pre-solve) worker count per stage.
        chain: Chain factors from :func:`chain.chain_factors`.
        stock_src: Whole-chain at-or-upstream stock per stage, source units.
        active_depths: Stage-local active work per stage, in stage-input
            sample units (this stage's own queued + pool-queued + in-flight
            depth, NOT source-normalized). Used only to keep a zero-fanout
            stage (``chain <= 0``) protected while it still owns admitted work
            that ``stock_src`` cannot express in source units; positive-chain
            stages are unaffected by this field.
        batch_sizes: Per-stage input batch sizes.
        w_sustain: Per-stage capacity hold target from the capacity plan; the
            floor clamps deletes to ``min(w_sustain, workers)`` while holding.
        protect_downstream_of: Rate-source stage whose downstream stages should
            not shrink while source-normalized stock is still in flight.
    """

    workers: tuple[int, ...]
    chain: tuple[float, ...]
    stock_src: tuple[float, ...]
    active_depths: tuple[float, ...]
    batch_sizes: tuple[int, ...]
    w_sustain: tuple[int, ...]
    protect_downstream_of: int = -1


@attrs.frozen
class FloorDecision:
    """One stage's floor outcome plus the signals that produced it.

    Attributes:
        floor: Minimum worker count (lower bound) applied this cycle.
        releasing: Whether the stage is releasing to ``min_workers`` (its
            whole-chain stock drained for ``release_confirm_cycles``).
        release_streak: Consecutive low-stock cycles observed so far.
        stock_threshold: Source-unit stock threshold below which the stage is
            considered drained (one batch's worth, ``batch_size / chain``).
        shrink_streak: Consecutive lower-hold-target cycles observed so far.
        pending_shrink_floor: Lower hold floor being confirmed.
        shrink_deferred: Whether a lower hold target is awaiting confirmation.
    """

    floor: int
    releasing: bool
    release_streak: int
    stock_threshold: float
    shrink_streak: int = 0
    pending_shrink_floor: int = 0
    shrink_deferred: bool = False


@attrs.frozen
class FloorPlan:
    """Per-cycle floor output: one decision per stage.

    Attributes:
        decisions: One :class:`FloorDecision` per stage, in pipeline order.
    """

    decisions: tuple[FloorDecision, ...]

    @property
    def floors(self) -> tuple[int, ...]:
        """Return the per-stage floor (lower bound), derived from decisions."""
        return tuple(decision.floor for decision in self.decisions)


@attrs.frozen
class FloorResult:
    """A cycle's :class:`FloorPlan` plus the state to carry into the next cycle.

    Attributes:
        plan: This cycle's per-stage release-gate decisions.
        state: Updated :class:`FloorState` to carry into the next cycle.
    """

    plan: FloorPlan
    state: FloorState


def compute_floors(inputs: FloorInputs, prev: FloorState, params: FloorParams) -> FloorResult:
    """Compute per-stage scale-down floors for one cycle.

    Each stage holds at a stabilized ``min(w_sustain, workers)`` until its
    whole-chain at-or-upstream stock stays drained for
    ``release_confirm_cycles`` consecutive cycles, at which point it releases
    as far as ``min_workers`` without exceeding current workers. Lower hold
    targets must persist for the same confirm window; higher targets take
    effect immediately.

    Args:
        inputs: Per-cycle observed per-stage inputs, including the capacity
            plan's ``w_sustain``.
        prev: State carried from the previous cycle.
        params: Release tuning.

    Returns:
        The cycle's floor plan and the :class:`FloorState` to carry into the
        next cycle.
    """
    num_stages = len(inputs.workers)
    if not (
        len(inputs.chain)
        == len(inputs.stock_src)
        == len(inputs.active_depths)
        == len(inputs.batch_sizes)
        == len(inputs.w_sustain)
        == len(prev.release_streak)
        == len(prev.held_floor)
        == len(prev.shrink_streak)
        == len(prev.pending_shrink_floor)
        == num_stages
    ):
        raise ValueError(
            "floor inputs length mismatch: "
            f"workers={num_stages} chain={len(inputs.chain)} stock_src={len(inputs.stock_src)} "
            f"active_depths={len(inputs.active_depths)} batch_sizes={len(inputs.batch_sizes)} "
            f"w_sustain={len(inputs.w_sustain)} prev_release_streak={len(prev.release_streak)} "
            f"prev_held_floor={len(prev.held_floor)} prev_shrink_streak={len(prev.shrink_streak)} "
            f"prev_pending_shrink_floor={len(prev.pending_shrink_floor)}"
        )

    decisions: list[FloorDecision] = []
    next_release_streak: list[int] = []
    next_held_floor: list[int] = []
    next_shrink_streak: list[int] = []
    next_pending_shrink_floor: list[int] = []

    for k in range(num_stages):
        if inputs.chain[k] <= 0.0 and inputs.active_depths[k] > 0.0:
            # A zero-fanout / drop stage (chain <= 0) cannot express its own
            # admitted work in source units, so whole_chain_stock() omits it
            # from stock_src and the source-normalized threshold collapses to
            # 0. Hold it while local active work remains so the floor never
            # releases a stage that still owns in-flight or queued work.
            threshold = 0.0
            has_stock = inputs.stock_src[k] > 0.0
            streak = 0
            releasing = False
        else:
            threshold = chain.source_stock_threshold(inputs.batch_sizes[k], inputs.chain[k])
            # One-batch boundary. A positive threshold counts stock at or above
            # it as work; a collapsed (0.0) threshold from zero / sub-MIN
            # fan-out counts only strictly positive stock, so a fully drained
            # stage can still release instead of looking permanently stocked.
            has_stock = inputs.stock_src[k] >= threshold if threshold > 0.0 else inputs.stock_src[k] > 0.0
            streak = 0 if has_stock else prev.release_streak[k] + 1
            releasing = not has_stock and streak >= params.release_confirm_cycles

        min_floor = min(params.min_workers, inputs.workers[k])
        desired = max(min_floor, min(inputs.w_sustain[k], inputs.workers[k]))
        if inputs.protect_downstream_of >= 0 and k > inputs.protect_downstream_of and has_stock:
            desired = inputs.workers[k]
        base = min(prev.held_floor[k], inputs.workers[k])
        if base <= 0 or desired >= base:
            held = desired
            shrink_streak = 0
            pending_shrink_floor = 0
            shrink_deferred = False
        else:
            shrink_streak = prev.shrink_streak[k] + 1
            prior_pending = prev.pending_shrink_floor[k]
            # pending_shrink_floor is the lower hold target being confirmed.
            # With no prior target (prior_pending <= 0) start fresh at desired;
            # otherwise combine the carried target with the new one:
            #   min(prior_pending, base) clamps the prior target down to the
            #     current base (base only ever decreases), so a stale-high prior
            #     cannot hold the floor above what we hold now; and
            #   max(..., desired) keeps the most conservative (highest) shrink
            #     target between that clamped prior and the new desired.
            # held stays at base until shrink_streak reaches
            # params.release_confirm_cycles, the confirming cycle that drops held
            # to pending_shrink_floor and resets shrink_streak and
            # pending_shrink_floor to 0 for the next window.
            pending_shrink_floor = desired if prior_pending <= 0 else max(min(prior_pending, base), desired)
            shrink_confirmed = shrink_streak >= params.release_confirm_cycles
            held = pending_shrink_floor if shrink_confirmed else base
            shrink_deferred = not shrink_confirmed
            if shrink_confirmed:
                shrink_streak = 0
                pending_shrink_floor = 0
        floor = min_floor if releasing else held
        if releasing:
            held = min_floor
            shrink_streak = 0
            pending_shrink_floor = 0
            shrink_deferred = False
        decisions.append(
            FloorDecision(
                floor=floor,
                releasing=releasing,
                release_streak=streak,
                stock_threshold=threshold,
                shrink_streak=shrink_streak,
                pending_shrink_floor=pending_shrink_floor,
                shrink_deferred=shrink_deferred,
            )
        )
        next_release_streak.append(streak)
        next_held_floor.append(held)
        next_shrink_streak.append(shrink_streak)
        next_pending_shrink_floor.append(pending_shrink_floor)

    return FloorResult(
        plan=FloorPlan(decisions=tuple(decisions)),
        state=FloorState(
            release_streak=tuple(next_release_streak),
            held_floor=tuple(next_held_floor),
            shrink_streak=tuple(next_shrink_streak),
            pending_shrink_floor=tuple(next_pending_shrink_floor),
        ),
    )


@attrs.define
class ScaleDownFloorPolicy:
    """Owns the scale-down floor's cross-cycle state and release tuning.

    Wraps the pure :func:`compute_floors` so the release streaks persist inside
    the policy across cycles, rather than being threaded through the scheduler.

    Attributes:
        params: Release tuning.
    """

    params: FloorParams
    _state: FloorState

    @classmethod
    def create(cls, num_stages: int, params: FloorParams) -> Self:
        """Build a policy with empty state for ``num_stages`` stages."""
        return cls(params, FloorState.initial(num_stages))

    def plan(self, inputs: FloorInputs) -> FloorPlan:
        """Return this cycle's floor plan and advance the internal state."""
        result = compute_floors(inputs, self._state, self.params)
        self._state = result.state
        return result.plan
