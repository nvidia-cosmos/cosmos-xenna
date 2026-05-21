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


"""MFI-style compound pressure signal for the saturation-aware classifier.

Combines the existing slots-empty utilisation EWMA with a normalised
backlog time (Little's Law ``W_q = queue / throughput`` divided by an
operator-facing ``target_backlog_seconds``) into a single product. Both
factors must be elevated for pressure to fire; either factor near zero
collapses the product. The classifier consumes the smoothed pressure as
a demotion gate inside the existing slot-ratio branches:

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

The Prometheus gauges in this module surface the raw inputs and the
smoothed output so operators can audit classifier decisions without
re-running the math from logs. The metrics are SA-only; importing this
module from a fragmentation-based pipeline pays the import-time
registration cost but no observations are ever recorded.
"""

from ray.util.metrics import Gauge

from cosmos_xenna.utils import python_log as logger

PRESSURE_OBSERVED_THROUGHPUT_METRIC = "xenna_stage_observed_throughput"
PRESSURE_BACKLOG_TIME_METRIC = "xenna_stage_backlog_time"
PRESSURE_EWMA_METRIC = "xenna_stage_pressure_ewma"

# Upper clamp on the normalised backlog factor. Bounds the composite
# pressure signal even when ``observed_throughput`` collapses to zero
# while the queue is non-empty (cold-start). Matches the cap-based
# cross-field validator on ``SaturationAwareStageConfig`` thresholds.
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
    """Compute the MFI-style compound pressure for one cycle.

    Pressure is the product ``utilisation * normalized_backlog`` where
    utilisation is ``1 - slots_empty_ratio_ewma`` and normalized_backlog
    is ``W_q / target_backlog_seconds`` clamped to ``[0, backlog_cap]``.
    The product captures the AND-criterion as a single multiplication:
    either factor near zero kills the output.

    Args:
        slots_empty_ratio_ewma: Smoothed empty-slot fraction in
            ``[0.0, 1.0]``. Sourced from the existing utilisation EWMA
            consumed by the slot-ratio classifier.
        input_queue_depth: Tasks waiting upstream of the stage. Zero
            means the queue is empty and pressure collapses to ``0.0``
            regardless of throughput.
        observed_throughput: Completed stage tasks per second since
            the previous autoscale cycle. ``0.0`` (or negative) is
            treated as "no progress this cycle"; combined with a
            non-empty queue this produces a ``backlog_cap``-bounded
            normalised backlog so cold-start cycles still fire the
            CRITICAL/SATURATED branches.
        target_backlog_seconds: Operator-facing primary knob. The
            queue drain time at which ``normalized_backlog == 1.0``.
        backlog_cap: Upper clamp on ``normalized_backlog``. Defaults to
            ``BACKLOG_CAP`` (``3.0``); kept as a parameter so unit
            tests can pin the clamp without mutating the module
            constant.

    Returns:
        Pressure scalar in ``[0.0, backlog_cap]``. The classifier
        applies threshold gates against this value; values close to
        zero are demoted to ``NORMAL``, values above
        ``pressure_critical_threshold`` fire ``SATURATED_CRITICAL``.

    Cold-start contract::

       queue == 0                              -> 0.0
       throughput <= 0 AND queue > 0           -> utilisation * backlog_cap
       throughput > 0  AND queue > 0           -> utilisation *
                                                  min(W_q / target, cap)

    The cold-start branch deliberately uses ``backlog_cap`` instead of
    ``+inf`` arithmetic so the pressure stays a finite scalar.

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

    Called from the per-stage decision pipeline after pressure has been
    computed and the EWMA refreshed. Tags are normalised to
    ``{"stage": stage_name, "pipeline": pipeline_name}`` so the Grafana
    dashboard can join all three series on the same key set.

    Args:
        stage_name: Stage being reported on.
        pipeline_name: Value of the ``pipeline`` Prometheus tag.
        observed_throughput: Raw throughput sample (tasks/sec).
        backlog_time: Raw backlog-drain time (seconds), bounded to
            ``target_backlog_seconds * BACKLOG_CAP`` when throughput
            is zero with queue > 0.
        pressure_ewma: Smoothed composite pressure consumed by the
            classifier this cycle.
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

    The classifier consumes ``compute_pressure``'s product directly;
    this helper exists so the gauge series surfaces the underlying
    Little's Law value (rather than the product) without divide-by-zero
    arithmetic in the caller. Returns ``0.0`` for an empty queue and
    ``target_backlog_seconds * backlog_cap`` for the cold-start branch
    (zero throughput with non-empty queue) so the displayed value never
    overflows or NaN-poisons downstream aggregation.

    Args:
        input_queue_depth: Same as for ``compute_pressure``.
        observed_throughput: Same as for ``compute_pressure``.
        target_backlog_seconds: Same as for ``compute_pressure``.
        backlog_cap: Same as for ``compute_pressure``.

    Returns:
        Backlog-drain time in seconds, bounded to
        ``target_backlog_seconds * backlog_cap``.

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
