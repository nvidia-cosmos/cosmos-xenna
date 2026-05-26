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

"""Bottleneck engagement persistence streak ledger.

:class:`BottleneckEngagementState` debounces the bottleneck-
engagement INFO log emitted by
``phases/bottleneck/identity.py::maybe_log_bottleneck_engagement``.
Counts consecutive cycles that agree with the current candidate
state (engaged or disengaged) before announcing the transition;
this matches the same debounced model the heterogeneity gauge uses.
"""

import attrs


@attrs.define
class BottleneckEngagementState:
    """Streak ledger for debouncing the bottleneck-engagement INFO log.

    Cold-start (``last_announced is None``): ``candidate_streak``
    counts consecutive cycles that match ``last_candidate``;
    flipping the value resets the counter. Post-seeded
    (``last_announced is not None``): ``candidate_streak`` counts
    consecutive disagreement cycles; agreement resets. In both
    phases the counter must reach ``persistence_cycles`` before
    the gate fires. ``last_candidate`` is the cold-start value
    detector and is unused once ``last_announced`` is set.

    """

    last_announced: bool | None = None
    candidate_streak: int = 0
    last_candidate: bool | None = None

    def reset(self) -> None:
        """Reset to a fresh ledger."""
        self.last_announced = None
        self.candidate_streak = 0
        self.last_candidate = None


__all__ = ["BottleneckEngagementState"]
