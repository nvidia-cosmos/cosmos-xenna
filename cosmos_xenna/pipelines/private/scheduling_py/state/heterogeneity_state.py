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

"""Cluster-wide heterogeneity warn-log streak ledger.

:class:`HeterogeneityWarnState` debounces the operator-facing INFO
log emitted by :func:`compute_heterogeneity_ratio`
(``phases/bottleneck/heterogeneity.py``) so a brief heterogeneity
spike does not produce log spam, and a sustained heterogeneity above
the configured threshold produces exactly one INFO line until the
cluster recovers.
"""

import attrs


@attrs.define
class HeterogeneityWarnState:
    """Per-instance streak + once-per-spike latch for the heterogeneity warn log.

    Lives as a per-instance attribute on
    ``SaturationAwareScheduler`` so re-instantiation (e.g. tests)
    starts from a clean ledger. ``streak_cycles`` counts
    consecutive above-threshold cycles; ``has_fired`` latches
    after the INFO log to suppress re-emission until the ratio
    drops back to or below the threshold.

    """

    streak_cycles: int = 0
    has_fired: bool = False

    def reset(self) -> None:
        """Reset both fields to their construction-time defaults.

        Called from ``SaturationAwareScheduler.setup()`` so a
        re-setup of the scheduler (e.g. a new pipeline run on the
        same scheduler instance) starts from a clean streak ledger
        without any inherited has-fired latch.
        """
        self.streak_cycles = 0
        self.has_fired = False


__all__ = ["HeterogeneityWarnState"]
