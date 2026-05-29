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

"""Stuck-plan ledger: per-stage counter and detector under one entry point.

The Grow phase advances a per-stage stuck-plan counter every cycle a
positive-intent grow request goes partly unsatisfied. The counter
and the structured-log detector that consumes it must always advance
together; :class:`StuckPlanLedger` composes them so callers cannot
write one without notifying the other.

Architecture::

    +---------------------------------------+
    | StuckPlanLedger                       |
    |                                       |
    |   per-stage stuck-cycle counter       |
    |   StuckPlanDetector (WARN-to-INFO +   |
    |                      metrics)         |
    |                                       |
    |   .record(stage, value, ...)          |  single entry point
    +---------------------------------------+
"""

from collections.abc import Mapping
from types import MappingProxyType

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.state.stuck_plan_detector import StuckPlanDetector


@attrs.define
class StuckPlanLedger:
    """Composite owner for the Grow-phase stuck-plan counter and detector.

    Aggregates the per-stage stuck cycles counter and the
    :class:`StuckPlanDetector` under a single :meth:`record` write
    path so the dict and the detector always advance together. The
    Grow phase calls :meth:`record` for every stage every cycle: zero
    on no-intent cycles, increment on stuck cycles, reset on
    successful cycles. The post-cycle invariant uses
    :meth:`snapshot` to capture the pre-Grow state and the
    :meth:`view` / :meth:`get_counter` accessors to compare against
    the post-Grow state.
    """

    _counters: dict[str, int] = attrs.Factory(dict)
    detector: StuckPlanDetector = attrs.Factory(StuckPlanDetector)

    def record(
        self,
        stage_name: str,
        value: int,
        *,
        last_intent: int,
        threshold_cycles: int,
        pipeline_name: str,
    ) -> None:
        """Set the per-stage stuck counter and notify the detector.

        Single update point so the counter dict and the detector
        always advance together; the post-cycle invariant compares
        the post-Grow snapshot against the pre-Grow snapshot and
        would flag any out-of-band write that bypasses this entry.

        Args:
            stage_name: The stage whose counter is being updated.
            value: The new counter value to record.
            last_intent: The per-stage signed intent that produced
                this update (forwarded to the detector so the
                structured log can distinguish a true stuck plan
                from an intent-driven reset).
            threshold_cycles: Stuck-plan detection threshold from
                ``SaturationAwareConfig.stuck_plan_detection_cycles``.
            pipeline_name: Pipeline tag for the detector's structured
                log and Prometheus labels.

        """
        self._counters[stage_name] = value
        self.detector.update(
            stage_name=stage_name,
            stuck_cycles=value,
            threshold_cycles=threshold_cycles,
            last_intent=last_intent,
            pipeline_name=pipeline_name,
        )

    def get_counter(self, stage_name: str) -> int:
        """Return the per-stage stuck-cycle counter (``0`` if absent)."""
        return self._counters.get(stage_name, 0)

    def view(self) -> Mapping[str, int]:
        """Return a read-only mapping view over the counter dict."""
        return MappingProxyType(self._counters)

    def snapshot(self) -> dict[str, int]:
        """Return a defensive copy of the counter dict.

        The post-cycle invariant uses the snapshot taken before the
        Grow phase to assert that every per-stage counter advanced by
        exactly ``+1`` or reset to ``0`` between Grow's start and end.
        """
        return dict(self._counters)

    def reset(self) -> None:
        """Reset the counter dict and the detector state in place.

        Object identity of the counter mapping and the detector is
        preserved so cross-cycle references captured before re-setup
        remain valid.
        """
        self._counters.clear()
        self.detector.reset()


__all__ = ["StuckPlanLedger"]
