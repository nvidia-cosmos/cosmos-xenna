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


"""Backlog-time pressure signal for the saturation-aware classifier.

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
"""

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
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in ``[0, 1]``.
        input_queue_depth: Tasks waiting upstream; ``0`` collapses the
            output to ``0.0``.
        observed_throughput: Completed tasks/sec since the last cycle.
        target_backlog_seconds: Drain time at which
            ``normalized_backlog == 1.0``.
        backlog_cap: Upper clamp on ``normalized_backlog``.

    Returns:
        Pressure scalar in ``[0.0, backlog_cap]``.

    """
    utilisation = 1.0 - slots_empty_ratio_ewma
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
        target_backlog_seconds: Operator-facing target.
        backlog_cap: Upper clamp on the displayed value.

    Returns:
        Backlog-drain time in seconds.

    """
    if input_queue_depth <= 0:
        return 0.0
    if observed_throughput <= 0.0:
        return target_backlog_seconds * backlog_cap
    return input_queue_depth / observed_throughput


__all__ = [
    "BACKLOG_CAP",
    "PRESSURE_BACKLOG_TIME_METRIC",
    "PRESSURE_EWMA_METRIC",
    "PRESSURE_OBSERVED_THROUGHPUT_METRIC",
    "compute_backlog_time",
    "compute_pressure",
    "emit_pressure_signals",
]
