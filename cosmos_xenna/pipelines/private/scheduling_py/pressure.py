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


"""Backlog-time pressure signal and capacity sizer for the saturation-aware scheduler.

Pressure is ``utilisation * normalized_backlog`` (Little's Law
``W_q = queue / throughput`` divided by ``target_backlog_seconds``);
either factor near zero collapses the product. The classifier reads it
as a demotion gate inside the existing slot-ratio branches:

::

    utilisation       low                 high
                  +-----------+---------------------+
    normalized
    backlog  low |    ~0      |  ~0  (transient     |
                 |  (idle)    |       burst:        |
                 |            |       queue drains) |
                 +-----------+---------------------+
              high|    ~0      |  HIGH (genuine     |
                 | (stuck     |       saturation)   |
                 | downstream)|                     |
                  +-----------+---------------------+

The Prometheus gauges below expose the raw inputs and the smoothed
output for operator audit.

The :func:`compute_capacity_target_workers` helper sits alongside the
pressure pipeline and answers a different question: given the same
queueing inputs (queue depth, observed throughput, per-task service
time, target drain time, operating utilisation), how many workers
does the stage need? Its output drives the magnitude of
``compute_delta``.
"""

import math

from ray.util.metrics import Gauge

from cosmos_xenna.utils import python_log as logger

PRESSURE_OBSERVED_THROUGHPUT_METRIC = "xenna_stage_observed_throughput"
PRESSURE_BACKLOG_TIME_METRIC = "xenna_stage_backlog_time"
PRESSURE_EWMA_METRIC = "xenna_stage_pressure_ewma"

# Upper clamp on the normalised backlog factor; keeps the cold-start
# branch finite when throughput is zero with a non-empty queue.
BACKLOG_CAP: float = 3.0

_OBSERVED_THROUGHPUT_GAUGE = Gauge(
    PRESSURE_OBSERVED_THROUGHPUT_METRIC,
    description=(
        "Per-stage observed throughput sample in completed tasks/sec, "
        "computed inline by SaturationAwareScheduler.autoscale() from "
        "the per-cycle delta in update_with_measurements counters. "
        "Raw, not smoothed - the EWMA lives on the composite pressure."
    ),
    tag_keys=("stage", "pipeline"),
)
_BACKLOG_TIME_GAUGE = Gauge(
    PRESSURE_BACKLOG_TIME_METRIC,
    description=(
        "Per-stage raw backlog-drain time in seconds "
        "(input_queue_depth / observed_throughput, Little's Law W_q). "
        "Bounded to target_backlog_seconds * BACKLOG_CAP when "
        "throughput collapses to zero with queue > 0 (cold-start). "
        "Raw, not smoothed."
    ),
    tag_keys=("stage", "pipeline"),
)
_PRESSURE_EWMA_GAUGE = Gauge(
    PRESSURE_EWMA_METRIC,
    description=(
        "Per-stage smoothed pressure signal "
        "(utilisation * normalized_backlog). The classifier reads this "
        "as the demotion gate inside the existing slot-ratio branches."
    ),
    tag_keys=("stage", "pipeline"),
)


def compute_pressure(
    *,
    slots_empty_ratio_ewma: float,
    input_queue_depth: int,
    observed_throughput: float,
    target_backlog_seconds: float,
    backlog_cap: float = BACKLOG_CAP,
) -> float:
    """Compute the backlog-time pressure for one cycle.

    ``utilisation * normalized_backlog`` clamped to ``backlog_cap``;
    either factor near zero collapses the product. ``observed_throughput
    <= 0`` with a non-empty queue is the cold-start branch and clamps
    to ``backlog_cap`` rather than dividing by zero.

    Args:
        slots_empty_ratio_ewma: Smoothed empty-slot fraction. Finite
            values are clamped to ``[0, 1]`` before computing
            ``utilisation`` so the EWMA's tiny floating-point drift
            cannot produce negative pressure.
        input_queue_depth: Tasks waiting upstream; ``0`` collapses the
            output to ``0.0``.
        observed_throughput: Completed tasks/sec since the last cycle.
        target_backlog_seconds: Drain time at which
            ``normalized_backlog == 1.0``. Must be strictly positive.
        backlog_cap: Upper clamp on ``normalized_backlog``. Must be
            finite and strictly positive.

    Returns:
        Pressure scalar in ``[0.0, backlog_cap]``.

    Raises:
        ValueError: ``target_backlog_seconds`` is not strictly
            positive, ``backlog_cap`` is not finite and ``> 0``, or
            ``slots_empty_ratio_ewma`` / ``observed_throughput`` is
            non-finite. NaN inputs would silently propagate to the
            classifier's pressure gate (the ``observed_throughput
            <= 0.0`` cold-start branch is False for NaN, so the
            divide-and-cap branch would emit NaN through the
            classifier); non-positive ``backlog_cap`` would invert
            the sign of the cold-start branch.

    """
    if target_backlog_seconds <= 0.0:
        msg = f"target_backlog_seconds must be > 0, got {target_backlog_seconds}"
        raise ValueError(msg)
    if not math.isfinite(backlog_cap) or backlog_cap <= 0.0:
        msg = f"backlog_cap must be finite and > 0, got {backlog_cap!r}"
        raise ValueError(msg)
    if not math.isfinite(slots_empty_ratio_ewma):
        msg = f"slots_empty_ratio_ewma must be finite, got {slots_empty_ratio_ewma!r}"
        raise ValueError(msg)
    if not math.isfinite(observed_throughput):
        msg = f"observed_throughput must be finite, got {observed_throughput!r}"
        raise ValueError(msg)

    clamped_slots_empty = max(0.0, min(1.0, slots_empty_ratio_ewma))
    utilisation = 1.0 - clamped_slots_empty
    if input_queue_depth <= 0:
        return 0.0
    if observed_throughput <= 0.0:
        normalized_backlog = backlog_cap
    else:
        backlog_time = input_queue_depth / observed_throughput
        normalized_backlog = min(backlog_time / target_backlog_seconds, backlog_cap)
    return utilisation * normalized_backlog


def emit_pressure_signals(
    *,
    stage_name: str,
    pipeline_name: str,
    observed_throughput: float,
    backlog_time: float,
    pressure_ewma: float,
) -> None:
    """Record the three pressure gauges for one stage in one cycle.

    Args:
        stage_name: Stage label.
        pipeline_name: Pipeline label.
        observed_throughput: Raw throughput sample (tasks/sec).
        backlog_time: Raw backlog-drain time (seconds).
        pressure_ewma: Smoothed composite pressure consumed by the classifier.
    """
    tags = {"stage": stage_name, "pipeline": pipeline_name}
    _OBSERVED_THROUGHPUT_GAUGE.set(float(observed_throughput), tags=tags)
    _BACKLOG_TIME_GAUGE.set(float(backlog_time), tags=tags)
    _PRESSURE_EWMA_GAUGE.set(float(pressure_ewma), tags=tags)
    logger.trace(
        f"saturation-aware pressure: stage {stage_name!r} "
        f"observed_throughput={observed_throughput:.4f}, "
        f"backlog_time={backlog_time:.4f}, "
        f"pressure_ewma={pressure_ewma:.4f}"
    )


def compute_backlog_time(
    *,
    input_queue_depth: int,
    observed_throughput: float,
    target_backlog_seconds: float,
    backlog_cap: float = BACKLOG_CAP,
) -> float:
    """Compute the raw backlog-drain time for the Prometheus gauge.

    Returns ``0.0`` for an empty queue and
    ``target_backlog_seconds * backlog_cap`` for the cold-start branch
    (zero throughput with non-empty queue).

    Args:
        input_queue_depth: Tasks waiting upstream.
        observed_throughput: Tasks/sec since the last cycle.
        target_backlog_seconds: Operator-facing target. Must be strictly
            positive.
        backlog_cap: Upper clamp on the displayed value. Must be
            finite and strictly positive.

    Returns:
        Backlog-drain time in seconds.

    Raises:
        ValueError: ``target_backlog_seconds`` is not strictly
            positive, ``backlog_cap`` is not finite and ``> 0``, or
            ``observed_throughput`` is non-finite. A non-positive
            cap would invert the sign of the cold-start gauge value
            the dashboards display; a non-finite throughput would
            slip past the ``<= 0.0`` cold-start branch (False for
            NaN) and emit NaN through the ``_BACKLOG_TIME_GAUGE``.

    """
    if target_backlog_seconds <= 0.0:
        msg = f"target_backlog_seconds must be > 0, got {target_backlog_seconds}"
        raise ValueError(msg)
    if not math.isfinite(backlog_cap) or backlog_cap <= 0.0:
        msg = f"backlog_cap must be finite and > 0, got {backlog_cap!r}"
        raise ValueError(msg)
    if not math.isfinite(observed_throughput):
        msg = f"observed_throughput must be finite, got {observed_throughput!r}"
        raise ValueError(msg)
    if input_queue_depth <= 0:
        return 0.0
    if observed_throughput <= 0.0:
        return target_backlog_seconds * backlog_cap
    return input_queue_depth / observed_throughput


def compute_capacity_target_workers(
    *,
    queue_depth: int,
    observed_throughput: float,
    d_k_seconds: float,
    slots_per_worker: int,
    target_backlog_seconds: float,
    utilization_target: float,
) -> int | None:
    """Return the worker count required to clear the backlog at the target rate.

    Forced Flow Law applied to a queue with a drain target:

    ::

        target_rate    = observed_throughput + (queue_depth / target_backlog_seconds)
                         (tasks/sec)            (tasks/sec)
        target_slots   = ceil((target_rate * d_k_seconds) / utilization_target)
        target_workers = ceil(target_slots / slots_per_worker)

    The drain term ``queue_depth / target_backlog_seconds`` is the
    extra arrival-rate equivalent needed to flush the existing backlog
    within ``target_backlog_seconds``; adding it to
    ``observed_throughput`` keeps both contributions in tasks/sec.

    Args:
        queue_depth: Tasks waiting upstream of the stage. ``>= 0``.
        observed_throughput: Per-cycle completed tasks/sec sample. ``>= 0``.
        d_k_seconds: Forced-Flow service time. ``<= 0`` or non-finite is
            the cold-start sentinel and returns ``None``.
        slots_per_worker: Concurrent task slots per worker. ``>= 1``.
        target_backlog_seconds: Drain target for the queue component.
            Must be strictly positive.
        utilization_target: Operating utilisation in ``(0, 1]``.

    Returns:
        Required worker count (``>= 0``), or ``None`` when ``d_k_seconds``
        is unobservable so the caller can fall back to discrete sizing.

    Raises:
        ValueError: For negative ``queue_depth``, non-finite or negative
            ``observed_throughput``, ``slots_per_worker < 1``, non-positive
            ``target_backlog_seconds``, or ``utilization_target`` outside
            ``(0, 1]``. A non-finite throughput would slip past the
            ``< 0.0`` check (False for NaN) and propagate into
            ``target_rate``, surfacing only as an indirect
            ``ValueError``/``OverflowError`` from ``math.ceil``.

    """
    if queue_depth < 0:
        msg = f"queue_depth must be >= 0, got {queue_depth}"
        raise ValueError(msg)
    if not math.isfinite(observed_throughput):
        msg = f"observed_throughput must be finite, got {observed_throughput!r}"
        raise ValueError(msg)
    if observed_throughput < 0.0:
        msg = f"observed_throughput must be >= 0, got {observed_throughput}"
        raise ValueError(msg)
    if slots_per_worker < 1:
        msg = f"slots_per_worker must be >= 1, got {slots_per_worker}"
        raise ValueError(msg)
    if target_backlog_seconds <= 0.0:
        msg = f"target_backlog_seconds must be > 0, got {target_backlog_seconds}"
        raise ValueError(msg)
    if not (0.0 < utilization_target <= 1.0):
        msg = f"utilization_target must be in (0, 1], got {utilization_target}"
        raise ValueError(msg)
    if not math.isfinite(d_k_seconds) or d_k_seconds <= 0.0:
        return None

    target_rate = observed_throughput + (queue_depth / target_backlog_seconds)
    target_slots = math.ceil((target_rate * d_k_seconds) / utilization_target)
    return math.ceil(target_slots / slots_per_worker)


__all__ = [
    "BACKLOG_CAP",
    "PRESSURE_BACKLOG_TIME_METRIC",
    "PRESSURE_EWMA_METRIC",
    "PRESSURE_OBSERVED_THROUGHPUT_METRIC",
    "compute_backlog_time",
    "compute_capacity_target_workers",
    "compute_pressure",
    "emit_pressure_signals",
]
