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

    hold     -> floor = min(w_sustain, workers)   (clamp deletes to the
                                                    capacity hold target)
    release  -> floor = min_workers               (whole-chain work has
                                                    drained for release_confirm_cycles)

The floor never demands growth (``floor <= workers``), so growth stays owned by
the demand multiplier. The release decision is driven by whole-chain
at-or-upstream stock so a downstream stage is not released while upstream work
is still in flight; a zero-fanout / drop stage (whose admitted work cannot be
expressed in source units) is instead held while it still has local active
work.
"""

from typing import Self

import attrs


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
    """

    release_streak: tuple[int, ...]

    @classmethod
    def initial(cls, num_stages: int) -> Self:
        """Return empty state for a pipeline with ``num_stages`` stages."""
        return cls(release_streak=(0,) * num_stages)


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
    """

    workers: tuple[int, ...]
    chain: tuple[float, ...]
    stock_src: tuple[float, ...]
    active_depths: tuple[float, ...]
    batch_sizes: tuple[int, ...]
    w_sustain: tuple[int, ...]


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
    """

    floor: int
    releasing: bool
    release_streak: int
    stock_threshold: float


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

    Each stage holds at ``min(w_sustain, workers)`` (so the solver may shrink
    an over-fed stage down to its capacity hold target but no further) until
    its whole-chain at-or-upstream stock stays drained for
    ``release_confirm_cycles`` consecutive cycles, at which point it releases
    to ``min_workers``. A zero-fanout / drop stage cannot express its admitted
    work in source units, so it is gated on local active depth instead and is
    never released while it still owns in-flight or queued work.

    Args:
        inputs: Per-cycle observed per-stage inputs, including the capacity
            plan's ``w_sustain``.
        prev: State carried from the previous cycle.
        params: Release tuning.

    Returns:
        The cycle's floor plan and the :class:`FloorState` to carry into the
        next cycle.
    """
    decisions: list[FloorDecision] = []
    next_streak: list[int] = []

    for k in range(len(inputs.workers)):
        if inputs.chain[k] <= 0.0 and inputs.active_depths[k] > 0.0:
            # A zero-fanout / drop stage (chain <= 0) cannot express its own
            # admitted work in source units, so whole_chain_stock() omits it
            # from stock_src and the source-normalized threshold collapses to
            # 0. Hold it while local active work remains so the floor never
            # releases a stage that still owns in-flight or queued work.
            threshold = 0.0
            streak = 0
            releasing = False
        else:
            threshold = inputs.batch_sizes[k] / inputs.chain[k] if inputs.chain[k] > 0.0 else 0.0
            streak = 0 if inputs.stock_src[k] > threshold else prev.release_streak[k] + 1
            releasing = inputs.stock_src[k] <= threshold and streak >= params.release_confirm_cycles

        held = max(params.min_workers, min(inputs.w_sustain[k], inputs.workers[k]))
        floor = params.min_workers if releasing else held
        decisions.append(
            FloorDecision(floor=floor, releasing=releasing, release_streak=streak, stock_threshold=threshold)
        )
        next_streak.append(streak)

    return FloorResult(
        plan=FloorPlan(decisions=tuple(decisions)),
        state=FloorState(release_streak=tuple(next_streak)),
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
