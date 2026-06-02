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

"""Per-stage speed and fan-out estimator for the saturation-aware scheduler.

Each stage gets a windowed per-worker speed estimate (from completed-task
durations, via ``RateEstimatorDuration``) and an EWMA of returns-per-batch
(fan-out). Both read ``None`` until the first sample for that stage arrives.
"""

import attrs

from cosmos_xenna.utils.timing import RateEstimatorDuration

_DEFAULT_RETURNS_ALPHA = 0.3


@attrs.define
class _StageEstimator:
    """Windowed per-worker speed + fan-out (returns-per-batch) for one stage."""

    _speed: RateEstimatorDuration
    _returns_alpha: float
    _returns_ewma: float | None = None
    _sample_count: int = 0

    def observe(self, duration_s: float, num_returns: float, now: float) -> None:
        """Record one completed task's service duration and return count."""
        self._speed.update(duration_s, now)
        if self._returns_ewma is None:
            self._returns_ewma = num_returns
        else:
            self._returns_ewma = self._returns_alpha * num_returns + (1.0 - self._returns_alpha) * self._returns_ewma
        self._sample_count += 1

    def speed(self, now: float) -> float | None:
        """Return the measured per-worker speed, or ``None`` before any sample."""
        if self._sample_count == 0:
            return None
        return self._speed.get_rate(now)

    def num_returns(self) -> float | None:
        """Return the measured fan-out, or ``None`` before any sample."""
        return self._returns_ewma

    @property
    def sample_count(self) -> int:
        """Return the number of measured samples observed so far."""
        return self._sample_count


@attrs.define
class PipelineRateEstimator:
    """Owns one :class:`_StageEstimator` per stage, keyed by stage name.

    The scheduler feeds completed-task measurements via :meth:`observe` and
    reads :meth:`speed` / :meth:`num_returns` per stage each cycle.
    """

    _window_s: float
    _min_data_points: int
    _returns_alpha: float = _DEFAULT_RETURNS_ALPHA
    _stages: dict[str, _StageEstimator] = attrs.field(factory=dict)

    def observe(self, stage_name: str, duration_s: float, num_returns: float, now: float) -> None:
        """Record one completed task for ``stage_name``."""
        self._ensure(stage_name).observe(duration_s, num_returns, now)

    def speed(self, stage_name: str, now: float) -> float | None:
        """Return the stage's measured per-worker speed, or ``None`` when unknown."""
        estimator = self._stages.get(stage_name)
        return estimator.speed(now) if estimator is not None else None

    def num_returns(self, stage_name: str) -> float | None:
        """Return the stage's measured fan-out, or ``None`` when unknown."""
        estimator = self._stages.get(stage_name)
        return estimator.num_returns() if estimator is not None else None

    def sample_count(self, stage_name: str) -> int:
        """Return the measured-sample count for ``stage_name`` (0 if unseen)."""
        estimator = self._stages.get(stage_name)
        return estimator.sample_count if estimator is not None else 0

    def _ensure(self, stage_name: str) -> _StageEstimator:
        estimator = self._stages.get(stage_name)
        if estimator is None:
            estimator = _StageEstimator(
                RateEstimatorDuration(self._window_s, self._min_data_points),
                self._returns_alpha,
            )
            self._stages[stage_name] = estimator
        return estimator
