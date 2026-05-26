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

"""Per-stage ring buffer of recent scale-recommendation directions.

A scale action fires only when every cycle in the relevant
direction-window agrees on the same direction. Asymmetric defaults
(``window_up = 1``, ``window_down = 30``) match Kubernetes HPA's
``scaleDown.stabilizationWindowSeconds = 300`` while keeping grow
responsive. This module owns the per-stage state value object
(:class:`RecommendationHistory`); the surrounding record-then-gate
function (``apply_stabilization_gate``) lives in
``phases/intent/stabilization.py``.
"""

import collections

from cosmos_xenna.pipelines.private.scheduling_py.scheduler.errors import SchedulerInvariantError

_DIRECTION_UP = 1
_DIRECTION_NOOP = 0
_DIRECTION_DOWN = -1


class RecommendationHistory:
    """Per-stage ring buffer of recent recommendation directions.

    Stores only the SIGN of each cycle's recommendation; the gate
    reasons about consensus, not magnitude. Asymmetric windows
    mirror HPA-style stabilization (``window_up = 1`` is immediate
    grow; larger ``window_down`` blocks shrink until streak fills).

    Attributes:
        window_up: Cycles of ``+1`` to allow grow.
        window_down: Cycles of ``-1`` to allow shrink.
        capacity: Effective ring-buffer capacity.

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
            msg = "RecommendationHistory buffer was constructed without maxlen"
            raise SchedulerInvariantError(msg)
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


__all__ = ["RecommendationHistory"]
