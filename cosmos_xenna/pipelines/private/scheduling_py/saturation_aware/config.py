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

"""Configuration types for the saturation-aware scheduler.

Pure value objects with no dependency on the native scheduling
extension, so ``specs`` and ``cosmos_curator`` can import them without
building the Rust solver.
"""

from typing import Self

import attrs

from cosmos_xenna.utils import attrs_utils

_UNIT_INTERVAL = attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0))
_OPEN_UNIT_INTERVAL = attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0))


@attrs.frozen
class SaturationAwareConfig:
    """Tunables for the saturation-aware scheduler.

    Every field has a working default; a normal run touches none of them.

    Attributes:
        interval_s: Measure-and-decide cadence and backlog catch-up
            horizon, in seconds.
        max_backlog_boost: Primary knob; the most a stage may request
            relative to its feed-matched size.
        burst_headroom: Spare-capacity fraction kept on every active
            stage; the demand-multiplier floor is ``1 + burst_headroom``.
        backlog_smoothing: EWMA alpha for the demand multiplier; ``None``
            (or ``1.0``) disables smoothing.
        speed_estimation_window_s: Throughput-estimator window.
        speed_estimation_min_data_points: Samples retained even when
            older than the window.
        scale_down_release_cycles: Scale-down floor release speed;
            ``alpha_down = 1 / scale_down_release_cycles`` (larger means
            the floor decays slower, so an expensive stage is held
            longer through a transient upstream lull).
        scale_down_grace_after_ready_s: Window after a worker is first
            observed before it may be deleted, in seconds.
    """

    interval_s: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0))
    max_backlog_boost: float = attrs.field(default=8.0, validator=attrs.validators.ge(1.0))
    burst_headroom: float = attrs.field(default=0.10, validator=_UNIT_INTERVAL)
    backlog_smoothing: float | None = attrs.field(
        default=0.4,
        validator=attrs.validators.optional(_OPEN_UNIT_INTERVAL),
    )
    speed_estimation_window_s: float = attrs.field(default=60.0, validator=attrs.validators.gt(0.0))
    speed_estimation_min_data_points: int = attrs.field(default=5, validator=attrs_utils.validate_positive_int)
    scale_down_release_cycles: int = attrs.field(default=6, validator=attrs_utils.validate_positive_int)
    scale_down_grace_after_ready_s: float = attrs.field(default=60.0, validator=attrs.validators.ge(0.0))

    @classmethod
    def resolve(cls, config: Self | None) -> Self:
        """Return ``config`` when set, else default-constructed tunables."""
        return config if config is not None else cls()
