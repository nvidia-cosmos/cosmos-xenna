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
# Smoothing weights must be strictly positive (0.0 would freeze the EWMA at its
# previous value) and at most 1.0 (1.0 = no smoothing, take the raw sample).
_SMOOTHING_ALPHA = attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0))


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
        speed_estimation_min_data_points: Trust threshold below which a stage
            is treated as cold/unmeasured for capacity and demand. It does not
            set the averaging depth (see ``speed_estimation_averaging_samples``).
        speed_estimation_averaging_samples: Minimum completed-task samples the
            speed estimator retains for its ``1/mean(duration)`` average, even
            when older than the window. Decoupled from the trust threshold so
            the per-worker speed can be averaged over enough samples to be
            stable for heterogeneous task durations without delaying cold-start
            trust. Must be >= ``speed_estimation_min_data_points``.
        speed_alpha_up: EWMA weight when a stage's measured per-worker speed
            rises. Kept modest so a single transient fast task cannot collapse
            the worker target; distinct from the arrival-rate ``alpha_up``.
        speed_alpha_down: EWMA weight when measured per-worker speed falls during
            normal completion variance. Kept small (protective) so a one-cycle
            dip does not mislabel a capable stage as the bottleneck. A genuine
            stall instead snaps the sizing rate down (see
            ``speed_stale_multiple``), so this only governs noise, not stalls.
        speed_stale_multiple: Overdue factor, relative to a stage's mean service
            time, beyond which a busy stage with no completion is treated as
            stalled. A stalled stage's sizing rate snaps down to the aged rate
            and its growth is bounded (see ``speed_stale_growth_step``).
        speed_stale_growth_step: Maximum workers a stalled stage may grow per
            cycle, so its collapsing per-worker rate cannot explode the divisive
            growth target into a node-filling request before completions resume.
        speed_estimation_min_task_duration_s: Lower service-time bound that
            distinguishes a degenerate empty skip from real work. A sample
            that produced no output AND completed faster than this is not a
            service-rate observation (it would drive ``1/mean(duration)``
            toward infinity), so it is excluded from the speed window and the
            trusted-sample count. Real zero-output filter stages, which take
            real time, are unaffected.
        pipeline_warmup_growth_step: Maximum workers a trusted stage may grow per
            cycle while the pipeline as a whole is still warming (some
            work-bearing stage is untrusted, so ``bottleneck_rate`` excludes it
            and the derived ``w_target`` is provisional). It bounds growth so a
            fast upstream stage cannot leap to a node-filling ``w_target`` and
            over-claim a shared resource (e.g. CPU) before slower downstream
            stages have warmed. Lower is safer (leaves more headroom for
            slow-starting stages to acquire resources during warmup); higher
            ramps a genuinely source-bound pipeline faster once trusted. Has no
            effect once every work-bearing stage is trusted.
        scale_down_release_cycles: Base scale-down release speed, combined with
            ``scale_down_release_slowdown`` as ``alpha_down = 1 /
            (scale_down_release_cycles * scale_down_release_slowdown)`` (larger
            means the smoothed sustainable rate decays slower, so a stage is held
            longer through a transient upstream lull).
        scale_down_release_slowdown: Extra release-time multiplier applied
            uniformly to every stage. The default holds a stage longer through a
            transient dip; ``1.0`` restores the fast base release for all stages.
        reclaim_confirm_cycles: Consecutive cycles a downstream stage must look
            idle, over-provisioned, and beneficial-to-reclaim (its freed resource
            would help an under-capacity stage grow) before the scale-down floor
            releases its warm pin. A longer window makes a transient demand spike
            less able to trigger a coupled scale-down / scale-up.
    """

    interval_s: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0))
    capacity_headroom: float = attrs.field(default=0.10, validator=_UNIT_INTERVAL)
    speed_estimation_window_s: float = attrs.field(default=60.0, validator=attrs.validators.gt(0.0))
    speed_estimation_min_data_points: int = attrs.field(default=5, validator=attrs_utils.validate_positive_int)
    speed_estimation_averaging_samples: int = attrs.field(
        default=10,
        validator=attrs_utils.validate_positive_int,
    )
    speed_alpha_up: float = attrs.field(default=0.3, validator=_SMOOTHING_ALPHA)
    speed_alpha_down: float = attrs.field(default=0.1, validator=_SMOOTHING_ALPHA)
    speed_stale_multiple: float = attrs.field(default=3.0, validator=attrs.validators.gt(1.0))
    speed_stale_growth_step: int = attrs.field(default=1, validator=attrs_utils.validate_positive_int)
    speed_estimation_min_task_duration_s: float = attrs.field(default=1e-3, validator=attrs.validators.gt(0.0))
    pipeline_warmup_growth_step: int = attrs.field(default=4, validator=attrs_utils.validate_positive_int)
    scale_down_release_cycles: int = attrs.field(default=6, validator=attrs_utils.validate_positive_int)
    scale_down_release_slowdown: float = attrs.field(default=4.0, validator=attrs.validators.ge(1.0))
    reclaim_confirm_cycles: int = attrs.field(default=6, validator=attrs_utils.validate_positive_int)

    def __attrs_post_init__(self) -> None:
        """Validate cross-field invariants."""
        if self.speed_estimation_averaging_samples >= self.speed_estimation_min_data_points:
            return

        msg = (
            "speed_estimation_averaging_samples "
            f"({self.speed_estimation_averaging_samples}) must be >= "
            "speed_estimation_min_data_points "
            f"({self.speed_estimation_min_data_points})"
        )
        raise ValueError(msg)

    @classmethod
    def resolve(cls, config: Self | None) -> Self:
        """Return ``config`` when set, else default-constructed tunables."""
        return config if config is not None else cls()
