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


"""Asymmetric stabilization-window gate for per-stage recommendations.

A second-layer stabilization mechanism that sits between the per-stage
decision pipeline (which produces a signed worker-count intent each
cycle) and the planner phases (Phase C scale-up, Phase D scale-down).

The gate keeps a fixed-size ring buffer of the most recent recommendation
directions per stage. A scale action is allowed only when every cycle
in the relevant window agrees on the same direction. Asymmetric windows
let the operator be aggressive on growth (default ``window_up = 1``) and
conservative on shrink (default ``window_down = 30`` -- 5 minutes at the
10 s autoscale cycle, matching Kubernetes HPA's
``scaleDown.stabilizationWindowSeconds = 300``).

The gate is independent of the classifier streak machine
(``decisions.update_streak`` / ``should_fire_action``) and the
growth-mode HOLD timer (``growth_mode.compute_growth_mode_transition``):

::

    classifier  -->  streak gate  -->  growth-mode  -->  raw delta
    (transient)      (per state)      (per event)        |
                                                         v
                                                +-------------------+
                                                | stabilization     |
                                                | recommendation    |
                                                | history (this     |
                                                | module)           |
                                                +-------------------+
                                                         |
                                                         v
                                              gated delta -> Phase C / D

Each layer has a different role:

  * The classifier streak guards against acting on a single noisy
    cycle of one ``StageState``.
  * The growth-mode HOLD timer guards against ping-ponging back to
    growth immediately after a shrink event.
  * This module guards against acting on a recommendation that has
    not been stable across the whole window. It prevents a single
    OVER_PROVISIONED cycle (after the streak fires) from triggering
    a shrink when the next cycle would have flipped back to NORMAL.

Concrete cycle flow with the gate enabled:

  1. ``run_per_stage_pipeline`` produces a raw signed delta.
  2. The pipeline records the direction (``+1`` / ``0`` / ``-1``)
     into the per-stage ``_RecommendationHistory`` instance.
  3. The pipeline asks the gate whether the recommendation may fire
     this cycle. If the gate refuses, the delta is zeroed before the
     growth-mode transition runs, so HOLD timers advance correctly.
  4. The (possibly zeroed) delta is returned to the scheduler and
     consumed by Phase C / Phase D as before.

The buffer holds ``max(window_up, window_down)`` cycles so a single
instance can answer both gate queries without separate buffers.
"""

import collections

_DIRECTION_UP = 1
_DIRECTION_NOOP = 0
_DIRECTION_DOWN = -1


class _RecommendationHistory:
    """Per-stage ring buffer of recent recommendation directions.

    One instance per stage, kept on the scheduler across cycles. The
    buffer stores only the SIGN of each cycle's recommendation, not its
    magnitude -- the gate reasons about consensus, not size. The
    asymmetric ``window_up`` / ``window_down`` parameters mirror
    HPA-style stabilization windows and accept the same semantic
    contract (``window_up = 1`` -> immediate growth on a single cycle;
    larger ``window_down`` -> shrink only after a sustained streak).

    Attributes:
        window_up: Required consecutive cycles of ``+1`` recommendation
            before :meth:`gate_up_allowed` returns ``True``.
        window_down: Required consecutive cycles of ``-1`` recommendation
            before :meth:`gate_down_allowed` returns ``True``.
        capacity: The buffer's effective capacity, equal to
            ``max(window_up, window_down)``. Older entries are dropped
            silently when a new direction is recorded.

    """

    __slots__ = ("_buffer", "_window_down", "_window_up")

    def __init__(self, *, window_up: int, window_down: int) -> None:
        """Initialize an empty buffer with the given asymmetric windows.

        Args:
            window_up: Cycles required to fire a scale-up.
            window_down: Cycles required to fire a scale-down.

        Raises:
            ValueError: If either window is non-positive. Cross-window
                ordering (``window_down > window_up``) is enforced at
                config time on ``SaturationAwareStageConfig``; this
                primitive accepts any positive pair so it remains
                usable in unit tests with degenerate windows.

        """
        if window_up < 1:
            msg = f"window_up must be >= 1, got {window_up}"
            raise ValueError(msg)
        if window_down < 1:
            msg = f"window_down must be >= 1, got {window_down}"
            raise ValueError(msg)
        self._window_up = window_up
        self._window_down = window_down
        self._buffer: collections.deque[int] = collections.deque(maxlen=max(window_up, window_down))

    @property
    def window_up(self) -> int:
        """Required consecutive ``+1`` cycles for scale-up to fire."""
        return self._window_up

    @property
    def window_down(self) -> int:
        """Required consecutive ``-1`` cycles for scale-down to fire."""
        return self._window_down

    @property
    def capacity(self) -> int:
        """Buffer capacity, equal to ``max(window_up, window_down)``."""
        # ``maxlen`` is set in ``__init__`` so it cannot be ``None`` here.
        # Treat the impossible ``None`` case as a buffer-construction bug
        # rather than silently lying about capacity.
        if self._buffer.maxlen is None:
            msg = "_RecommendationHistory buffer was constructed without maxlen"
            raise RuntimeError(msg)
        return self._buffer.maxlen

    def __len__(self) -> int:
        """Return the number of cycles currently retained in the buffer."""
        return len(self._buffer)

    def record(self, raw_delta: int) -> None:
        """Append the SIGN of ``raw_delta`` to the buffer.

        Args:
            raw_delta: This cycle's pre-gate intent. Positive maps to
                ``+1`` (scale-up recommendation), negative to ``-1``
                (scale-down), zero to ``0`` (no action).

        """
        if raw_delta > 0:
            self._buffer.append(_DIRECTION_UP)
        elif raw_delta < 0:
            self._buffer.append(_DIRECTION_DOWN)
        else:
            self._buffer.append(_DIRECTION_NOOP)

    def gate_up_allowed(self) -> bool:
        """Return ``True`` when scale-up may fire this cycle.

        Cold-start protection: with fewer than ``window_up`` records the
        gate refuses regardless of direction so a fresh stage cannot
        scale based on a single un-replicated sample.

        Returns:
            ``True`` if the most recent ``window_up`` recorded
            directions are all ``+1``. ``False`` for an under-filled
            buffer or any non-up cycle in the window.

        """
        if len(self._buffer) < self._window_up:
            return False
        return self._all_recent(self._window_up, _DIRECTION_UP)

    def gate_down_allowed(self) -> bool:
        """Return ``True`` when scale-down may fire this cycle.

        Cold-start protection: with fewer than ``window_down`` records
        the gate refuses, ensuring a fresh pipeline cannot shrink
        before its slot signals have stabilized.

        Returns:
            ``True`` if the most recent ``window_down`` recorded
            directions are all ``-1``. ``False`` for an under-filled
            buffer or any non-down cycle in the window.

        """
        if len(self._buffer) < self._window_down:
            return False
        return self._all_recent(self._window_down, _DIRECTION_DOWN)

    def clear(self) -> None:
        """Drop every recorded direction.

        Intended for tests and for the rare cycle in which the
        scheduler explicitly resets a stage's stabilization state
        (for example, after a structural worker-count change made
        outside the autoscaler).
        """
        self._buffer.clear()

    def _all_recent(self, count: int, expected: int) -> bool:
        """Whether the last ``count`` recorded directions all equal ``expected``.

        Walks the buffer from the right (newest) end backwards rather
        than building a list slice; this keeps the per-cycle cost
        bounded by ``count`` and avoids allocations on the hot path.
        """
        return all(self._buffer[len(self._buffer) - 1 - i] == expected for i in range(count))


def apply_stabilization_gate(history: _RecommendationHistory, raw_delta: int) -> int:
    """Record ``raw_delta`` into ``history`` and return the gated delta.

    The two-step contract (record then gate) is encapsulated here so
    callers cannot accidentally gate without recording, which would
    leave the buffer one cycle behind reality and silently weaken
    every future gate decision.

    Args:
        history: Per-stage history. Mutated by the record step.
        raw_delta: This cycle's unclamped intent.

    Returns:
        ``raw_delta`` when the gate allows the recommended direction
        (or when the recommendation is zero); ``0`` when the gate
        refuses.

    """
    history.record(raw_delta)
    if raw_delta > 0:
        return raw_delta if history.gate_up_allowed() else 0
    if raw_delta < 0:
        return raw_delta if history.gate_down_allowed() else 0
    return 0
