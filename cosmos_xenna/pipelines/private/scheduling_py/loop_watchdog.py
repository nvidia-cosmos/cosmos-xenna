# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-cycle wall-clock watchdog for the saturation-aware scheduler.

See ``docs/scheduler/saturation-aware/18-loop-watchdog.md`` for the
design intent and ``22-prometheus-metrics.md`` for the metric catalogue.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger as _loguru_logger
from ray.util.metrics import Histogram

CYCLE_DURATION_METRIC = "xenna_scheduler_cycle_duration_seconds"

# Bucket boundaries straddle the default budget (cycle_time_warn_threshold *
# interval_s = 0.5 * 10.0 = 5.0 s) so the same histogram captures both the
# healthy sub-second cycles and the pathological multi-second overruns
# without operator tuning.
_CYCLE_DURATION_HISTOGRAM = Histogram(
    name=CYCLE_DURATION_METRIC,
    description=(
        "Wall-clock duration of one SaturationAwareScheduler.autoscale "
        "cycle in seconds. Operators read the histogram tail to alert "
        "on slow cycles; the loop watchdog emits a WARN log when a "
        "single observation exceeds cycle_time_warn_threshold * "
        "interval_s."
    ),
    boundaries=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0],
    tag_keys=("pipeline",),
)


@contextmanager
def loop_watchdog(
    *,
    pipeline_name: str,
    threshold_fraction: float,
    interval_s: float,
) -> Iterator[None]:
    """Bracket one autoscale cycle with the loop watchdog.

    Args:
        pipeline_name: Value for the ``pipeline`` Prometheus tag.
        threshold_fraction: Fraction of ``interval_s`` above which the
            cycle duration triggers a WARN log.
        interval_s: Cluster-wide cycle interval in seconds.

    Yields:
        ``None``. The histogram observation lives in ``finally`` so a
        body that raises still records the duration.
    """
    t_start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        duration_s = (time.perf_counter_ns() - t_start_ns) / 1e9
        _CYCLE_DURATION_HISTOGRAM.observe(duration_s, tags={"pipeline": pipeline_name})
        threshold_s = threshold_fraction * interval_s
        # Strict ``>`` so a cycle exactly at the budget does not warn.
        if duration_s > threshold_s:
            _loguru_logger.bind(pipeline=pipeline_name).warning(
                f"saturation-aware loop watchdog: autoscale cycle took "
                f"{duration_s:.2f}s (threshold={threshold_s:.2f}s = "
                f"{threshold_fraction} * interval_s={interval_s})"
            )


__all__ = [
    "CYCLE_DURATION_METRIC",
    "loop_watchdog",
]
