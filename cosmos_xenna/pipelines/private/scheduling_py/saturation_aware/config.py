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


@attrs.frozen
class SaturationAwareConfig:
    """Tunables for the saturation-aware scheduler.

    Every field has a working default; a normal run touches none of them.

    Attributes:
        interval_s: Measure-and-decide cadence, in seconds. It sets how
            often the scheduler re-measures throughput and re-plans
            capacity targets; it is not a backlog catch-up horizon.
        capacity_headroom: Spare-capacity fraction added on top of the
            bottleneck rate when computing a stage's useful growth target
            ``w_target`` (the bounded source-rate read-ahead). It is a
            target headroom, not a demand-multiplier floor.
        speed_estimation_window_s: Throughput-estimator window.
        speed_estimation_min_data_points: Samples retained even when
            older than the window; also the trust threshold below which a
            stage is treated as cold/unmeasured for capacity and demand.
        speed_estimation_min_task_duration_s: Lower service-time bound that
            distinguishes a degenerate empty skip from real work. A sample
            that produced no output AND completed faster than this is not a
            service-rate observation (it would drive ``1/mean(duration)``
            toward infinity), so it is excluded from the speed window and the
            trusted-sample count. Real zero-output filter stages, which take
            real time, are unaffected.
        scale_down_release_cycles: Base scale-down release speed, combined with
            ``scale_down_release_slowdown`` as ``alpha_down = 1 /
            (scale_down_release_cycles * scale_down_release_slowdown)`` (larger
            means the smoothed sustainable rate decays slower, so a stage is held
            longer through a transient upstream lull).
        scale_down_release_slowdown: Extra release-time multiplier applied
            uniformly to every stage. The default holds a stage longer through a
            transient dip; ``1.0`` restores the fast base release for all stages.
    """

    interval_s: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0))
    capacity_headroom: float = attrs.field(default=0.10, validator=_UNIT_INTERVAL)
    speed_estimation_window_s: float = attrs.field(default=60.0, validator=attrs.validators.gt(0.0))
    speed_estimation_min_data_points: int = attrs.field(default=5, validator=attrs_utils.validate_positive_int)
    speed_estimation_min_task_duration_s: float = attrs.field(default=1e-3, validator=attrs.validators.gt(0.0))
    scale_down_release_cycles: int = attrs.field(default=6, validator=attrs_utils.validate_positive_int)
    scale_down_release_slowdown: float = attrs.field(default=4.0, validator=attrs.validators.ge(1.0))

    @classmethod
    def resolve(cls, config: Self | None) -> Self:
        """Return ``config`` when set, else default-constructed tunables."""
        return config if config is not None else cls()
