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

"""Cluster object-store memory-pressure gate for the saturation-aware scheduler.

The cluster-wide memory-pressure gate is the only cluster-level
signal the scheduler consults; every other classifier signal is
per-stage. Two stages can each respect their own slot contract while
the sum of their per-task footprints saturates Ray's object store -
no per-stage observable can see that. The gate watches Ray's
reported object-store usage and freezes Phase C scale-up when the
``used_fraction`` rises above a configured critical threshold.

::

    poll cache (refreshed every memory_pressure_polling_interval_s)
                   |
                   v
    used_fraction = (cluster_object_store - available_object_store)
                    / cluster_object_store
                   |
                   v
    pressure_active = used_fraction > critical_threshold
                   |
                   v
    Phase B floor enforcement  --- RUNS (structural; recovery path)
    Phase C saturation grow    --- FROZEN when pressure_active
    Phase D saturation shrink  --- RUNS (shrinking relieves pressure)

Polling is deliberately cheap: one Ray API call per
``polling_interval_s``-second window, the cached value is reused for
every cycle inside the window. Polling failure is treated as
``pressure inactive`` (graceful degradation - never block scheduling
because Ray's resource reporter hiccuped) and emits a single WARN
log; the log stops repeating until polling succeeds again so a
sustained Ray-reporter outage does not flood the operator log.

The two ``ray.util.metrics.Gauge`` instances
(``xenna_scheduler_cluster_object_store_used_fraction`` and
``xenna_scheduler_memory_pressure_active``) are owned by the
monitor and refreshed every call to :meth:`is_pressure_active` so
the Prometheus scrape always observes the latest value the
scheduler acted on - even on cycles where the cached fraction was
reused.
"""

from typing import Any

import ray
from ray.util.metrics import Gauge

from cosmos_xenna.utils import python_log as logger

CLUSTER_OBJECT_STORE_USED_FRACTION_METRIC = "xenna_scheduler_cluster_object_store_used_fraction"
MEMORY_PRESSURE_ACTIVE_METRIC = "xenna_scheduler_memory_pressure_active"


class MemoryPressureMonitor:
    """Polled cluster object-store pressure with sticky cache and metric emission.

    Owns the polling cache (last poll wall-clock, last polled
    ``used_fraction``, last active flag), and the two
    ``ray.util.metrics.Gauge`` instances. Designed for the
    saturation-aware scheduler's per-cycle hot path - one Ray API
    call per polling interval, O(1) reads inside the interval.

    Attributes:
        polling_interval_s: Minimum seconds between Ray cluster
            resource queries. Cached values are reused inside this
            window.
        critical_threshold: ``used_fraction`` above which the
            monitor reports pressure as active.
        pipeline_name: Tag value attached to the two emitted gauges
            so multi-pipeline deployments can be distinguished in
            Prometheus.

    """

    def __init__(
        self,
        *,
        polling_interval_s: float,
        critical_threshold: float,
        pipeline_name: str = "",
    ) -> None:
        """Construct the monitor.

        Args:
            polling_interval_s: Minimum seconds between Ray cluster
                resource polls. Must be > 0; the field validator on
                ``SaturationAwareConfig.memory_pressure_polling_interval_s``
                enforces this at the config layer.
            critical_threshold: ``used_fraction`` strictly above
                which the monitor reports pressure as active. Must
                be in ``(0.0, 1.0]``; the field validator on
                ``SaturationAwareConfig.memory_pressure_critical_threshold``
                enforces this at the config layer.
            pipeline_name: Optional tag attached to emitted gauges
                so multi-pipeline operators can distinguish each
                pipeline's signal in Prometheus.

        """
        self.polling_interval_s = polling_interval_s
        self.critical_threshold = critical_threshold
        self.pipeline_name = pipeline_name

        self._last_poll_at: float | None = None
        self._cached_used_fraction: float = 0.0
        self._cached_pressure_active: bool = False
        # Suppress repeated WARN spam from a sustained Ray-reporter
        # outage. Set to True after the first WARN, cleared back to
        # False after the next successful poll so a recovery is
        # immediately re-loggable.
        self._poll_failure_logged: bool = False
        # Counter exposed for observability tests that need to assert
        # the underlying Ray API was called the expected number of
        # times across multiple scheduler cycles.
        self._poll_count: int = 0

        self._used_fraction_gauge: Gauge = Gauge(
            CLUSTER_OBJECT_STORE_USED_FRACTION_METRIC,
            "Cluster Ray object-store used fraction (0.0-1.0).",
            tag_keys=("pipeline",),
        )
        self._pressure_active_gauge: Gauge = Gauge(
            MEMORY_PRESSURE_ACTIVE_METRIC,
            "1 when cluster memory pressure is active; 0 otherwise.",
            tag_keys=("pipeline",),
        )

    def is_pressure_active(self, now: float) -> bool:
        """Return True when the cluster is under memory pressure.

        Polls Ray's cluster resources at most once per
        ``polling_interval_s`` window; consecutive calls inside the
        window reuse the cached ``used_fraction``. Every call
        refreshes the two gauges so Prometheus observes the
        currently-acted-on value, including on cached-read cycles.

        Args:
            now: Current wall-clock time in seconds. Caller is
                responsible for passing a monotonically increasing
                clock; the monitor uses ``now`` as both the
                cache-invalidation reference and the timestamp
                stored alongside the cached fraction.

        Returns:
            True when the cached ``used_fraction`` is strictly
            greater than ``critical_threshold``. A polling failure
            (Ray API raises) degrades to ``False`` to avoid
            blocking scheduling on a transient Ray-reporter
            outage.

        """
        if self._last_poll_at is None or (now - self._last_poll_at) >= self.polling_interval_s:
            self._poll(now)

        tags = {"pipeline": self.pipeline_name}
        self._used_fraction_gauge.set(self._cached_used_fraction, tags=tags)
        self._pressure_active_gauge.set(1.0 if self._cached_pressure_active else 0.0, tags=tags)
        return self._cached_pressure_active

    def reset(self) -> None:
        """Drop the cached pressure state.

        Called from ``SaturationAwareScheduler.setup()`` so a
        re-setup (test re-run, mid-flight pipeline replacement)
        starts from a clean polling cache rather than carrying the
        prior run's last observation forward.
        """
        self._last_poll_at = None
        self._cached_used_fraction = 0.0
        self._cached_pressure_active = False
        self._poll_failure_logged = False
        self._poll_count = 0

    @property
    def last_used_fraction(self) -> float:
        """Most-recently polled ``used_fraction`` (or 0.0 if never polled / poll failed)."""
        return self._cached_used_fraction

    @property
    def last_pressure_active(self) -> bool:
        """Most-recently determined pressure flag."""
        return self._cached_pressure_active

    @property
    def last_poll_at(self) -> float | None:
        """Wall-clock time of the most recent poll, ``None`` if never polled."""
        return self._last_poll_at

    @property
    def poll_count(self) -> int:
        """Total number of Ray API polls since construction or last :meth:`reset`."""
        return self._poll_count

    def _poll(self, now: float) -> None:
        """Query Ray for object-store usage and refresh the cache.

        Graceful degradation has two distinct failure paths:

          * ``ray.is_initialized()`` is ``False`` - common during
            process startup and unit-test environments where the
            scheduler is exercised without a live Ray cluster.
            The cache is set to ``inactive`` silently; no WARN is
            emitted because this is the expected boot-up state,
            not a runtime failure.
          * Ray raises during the resource query - the cache is
            set to ``inactive`` and a single WARN is emitted
            (suppressed on repeats until the next successful poll
            so a sustained Ray-reporter outage does not flood the
            operator log).
        """
        self._poll_count += 1
        if not ray.is_initialized():
            self._cached_used_fraction = 0.0
            self._cached_pressure_active = False
            self._last_poll_at = now
            return
        try:
            cluster_resources: dict[str, Any] = ray.cluster_resources()  # type: ignore[no-untyped-call]
            available_resources: dict[str, Any] = ray.available_resources()  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001 - graceful degradation per design doc 20.
            if not self._poll_failure_logged:
                logger.warning(
                    "memory pressure gate: failed to query Ray cluster resources; "
                    f"assuming pressure inactive and continuing. error={exc!r}"
                )
                self._poll_failure_logged = True
            self._cached_used_fraction = 0.0
            self._cached_pressure_active = False
            self._last_poll_at = now
            return

        self._poll_failure_logged = False
        used_fraction = _compute_used_fraction(cluster_resources, available_resources)
        previous_active = self._cached_pressure_active
        self._cached_used_fraction = used_fraction
        self._cached_pressure_active = used_fraction > self.critical_threshold
        self._last_poll_at = now

        if self._cached_pressure_active and not previous_active:
            logger.warning(
                "memory pressure gate: ACTIVE - cluster object-store "
                f"used_fraction={used_fraction:.4f} exceeds "
                f"critical_threshold={self.critical_threshold:.4f}; "
                "Phase C scale-up will be frozen until pressure clears."
            )
        elif previous_active and not self._cached_pressure_active:
            logger.info(
                "memory pressure gate: CLEARED - cluster object-store "
                f"used_fraction={used_fraction:.4f} now within "
                f"critical_threshold={self.critical_threshold:.4f}; "
                "Phase C scale-up resumes."
            )


def _compute_used_fraction(
    cluster_resources: dict[str, Any],
    available_resources: dict[str, Any],
) -> float:
    """Derive ``used_fraction`` from Ray's two resource maps.

    Handles the edge cases that ``ray.cluster_resources()`` documents
    but the scheduler must absorb: the ``object_store_memory`` key
    can be absent (cluster has no object store configured) or zero
    (cluster size reported but no allocation yet); the available
    figure can briefly exceed total during cluster scale events.

    Args:
        cluster_resources: ``ray.cluster_resources()`` result.
        available_resources: ``ray.available_resources()`` result.

    Returns:
        ``used_fraction`` clamped to ``[0.0, 1.0]``. Returns ``0.0``
        when the cluster reports no object-store capacity (missing
        or non-positive total).

    """
    total = float(cluster_resources.get("object_store_memory", 0.0) or 0.0)
    if total <= 0.0:
        return 0.0
    available = float(available_resources.get("object_store_memory", 0.0) or 0.0)
    used = total - available
    return _clamp(used / total, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed interval ``[low, high]``."""
    return max(low, min(value, high))


__all__ = [
    "CLUSTER_OBJECT_STORE_USED_FRACTION_METRIC",
    "MEMORY_PRESSURE_ACTIVE_METRIC",
    "MemoryPressureMonitor",
]
