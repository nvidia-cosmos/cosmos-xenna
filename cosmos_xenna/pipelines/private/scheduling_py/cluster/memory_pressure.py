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

The only cluster-level signal the scheduler consults (others are
per-stage). Polls Ray's object-store usage every
``polling_interval_s`` (cached between polls); pressure latches
when ``used_fraction >= critical_threshold``. Freezes Phase C
grow (Phase B floor and Phase D shrink still run; shrinking
relieves pressure). Polling failure degrades to "inactive" with
a rate-limited WARN log. Gauges
``xenna_scheduler_cluster_object_store_used_fraction`` and
``xenna_scheduler_memory_pressure_active`` are refreshed on every
``is_pressure_active`` call. See
``docs/scheduler/saturation-aware/`` for the algorithm.
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
    ``used_fraction``, last active flag) and the two
    ``ray.util.metrics.Gauge`` instances. One Ray API call per
    polling interval, O(1) reads inside the interval. The
    ``critical_threshold`` comparison uses ``>=`` so configured
    ``1.0`` trips when the object store reports fully saturated.

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
            polling_interval_s: Minimum seconds between polls
                (> 0; enforced by the config layer).
            critical_threshold: ``used_fraction`` at or above which
                pressure is active (``>=``; in ``(0.0, 1.0]``).
            pipeline_name: Optional Prometheus tag for the gauges.

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
        ``polling_interval_s`` (calls inside the window reuse the
        cached fraction). Every call refreshes both gauges so
        Prometheus observes the acted-on value. Polling failure
        degrades to ``False`` to avoid blocking scheduling on
        transient Ray-reporter outages. ``now`` must be
        monotonically increasing.

        """
        if self._last_poll_at is None or (now - self._last_poll_at) >= self.polling_interval_s:
            self._poll(now)

        tags = {"pipeline": self.pipeline_name}
        self._used_fraction_gauge.set(self._cached_used_fraction, tags=tags)
        self._pressure_active_gauge.set(1.0 if self._cached_pressure_active else 0.0, tags=tags)
        return self._cached_pressure_active

    def reset(self) -> None:
        """Drop the cached pressure state and clear the exported gauges.

        Called from ``SaturationAwareScheduler.setup()`` so a
        re-setup (test re-run, mid-flight pipeline replacement)
        starts from a clean polling cache. The two Prometheus
        gauges are also driven to their cleared defaults so a
        scrape that lands between ``reset()`` and the first
        ``is_pressure_active()`` call observes the same state the
        scheduler internally tracks.
        """
        self._last_poll_at = None
        self._cached_used_fraction = 0.0
        self._cached_pressure_active = False
        self._poll_failure_logged = False
        self._poll_count = 0

        tags = {"pipeline": self.pipeline_name}
        self._used_fraction_gauge.set(0.0, tags=tags)
        self._pressure_active_gauge.set(0.0, tags=tags)

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

        Graceful degradation: ``ray.is_initialized() is False`` is
        the boot-up state - cache set to inactive silently.
        Ray-resource-query failure also degrades to inactive but
        emits one WARN per outage (suppressed on repeats until the
        next successful poll so a sustained outage cannot flood
        the operator log).
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
        self._cached_pressure_active = used_fraction >= self.critical_threshold
        self._last_poll_at = now

        if self._cached_pressure_active and not previous_active:
            logger.warning(
                "memory pressure gate: ACTIVE - cluster object-store "
                f"used_fraction={used_fraction:.4f} is at or above "
                f"critical_threshold={self.critical_threshold:.4f}; "
                "Phase C scale-up will be frozen until pressure clears."
            )
        elif previous_active and not self._cached_pressure_active:
            logger.info(
                "memory pressure gate: CLEARED - cluster object-store "
                f"used_fraction={used_fraction:.4f} now below "
                f"critical_threshold={self.critical_threshold:.4f}; "
                "Phase C scale-up resumes."
            )


def _compute_used_fraction(
    cluster_resources: dict[str, Any],
    available_resources: dict[str, Any],
) -> float:
    """Derive ``used_fraction`` from Ray's two resource maps.

    Absorbs edge cases the scheduler must tolerate: missing or
    zero ``object_store_memory`` (no object store configured /
    not yet allocated), and a brief ``available > total`` window
    during cluster scale events. Returns ``used_fraction`` clamped
    to ``[0.0, 1.0]``; ``0.0`` when no object-store capacity is
    reported.

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
