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

"""Per-stage measurement accumulator for the saturation-aware scheduler.

Two cadences: ``update_with_measurements`` is called by
``streaming.Autoscaler`` on every monitor tick (faster than
``interval_s``); ``consume_throughput_samples`` and
``consume_service_time_samples`` are called once per autoscale
cycle. Lock decouples ingest from consume. Two independent
``(count, sum)`` snapshots so consume order does not matter
within a cycle.
"""

import math
import threading

from cosmos_xenna.pipelines.private import data_structures


class MeasurementCollector:
    """Thread-safe per-stage completed-task and service-time accumulator.

    Mutated by ``update_with_measurements`` on every monitor tick;
    snapshotted once per autoscale cycle by
    ``consume_throughput_samples`` and
    ``consume_service_time_samples``. The internal lock decouples
    the cadences. Consumer is single-threaded; the lock guards the
    multi-monitor-tick ingest path.

    """

    def __init__(self) -> None:
        """Build an empty collector. ``setup`` seeds per-stage zeros."""
        self._lock: threading.Lock = threading.Lock()
        self._stage_names: list[str] = []
        self.completed_counts: dict[str, int] = {}
        self.completed_service_time_sums: dict[str, float] = {}
        self.last_throughput_sample: dict[str, tuple[int, float]] = {}
        self.last_service_time_sample: dict[str, tuple[int, float]] = {}

    def setup(self, stage_names: list[str]) -> None:
        """Reset accumulator state for a fresh pipeline.

        ``stage_names`` defines the iteration order for the
        ``consume_*`` samplers and the shape-validation key set
        for :meth:`update_with_measurements`. The reset is locked
        because a recycled scheduler may re-call ``setup`` while
        a previous monitor tick's ingest path holds the lock.

        Args:
            stage_names: Stage names in DAG order; copied into the
                collector so the caller may reuse the list.

        """
        with self._lock:
            self._stage_names = list(stage_names)
            self.completed_counts = {name: 0 for name in stage_names}
            self.completed_service_time_sums = {name: 0.0 for name in stage_names}
            self.last_throughput_sample = {}
            self.last_service_time_sample = {}

    def update_with_measurements(
        self,
        measurements: data_structures.Measurements,
    ) -> None:
        """Ingest the latest measurement batch.

        Accumulates per-stage completed-task counts and per-task
        service-time sums for the per-cycle samplers. One tick per
        ``TaskMeasurement`` (``flat_map`` stages still count one
        drain per task - matches ``input_queue_depth``'s unit).
        Iterates by position over ``measurements.rust.stages``
        zipped with the captured stage-name list, so attribution
        is DAG-ordered (no name hashing).

        Raises:
            ValueError: ``measurements.rust.stages`` is non-empty
                and disagrees in length with the stage-name list.

        """
        rust_stages = measurements.rust.stages
        # Shape validation runs BEFORE the lock so a corrupted
        # measurement batch cannot leave ``completed_counts`` or
        # ``completed_service_time_sums`` half-updated. An empty
        # ``rust_stages`` is the legitimate "no measurements this
        # tick" signal and is a no-op (the inner zip yields nothing).
        # Any non-empty list whose length disagrees with the
        # setup-time stage count is a Rust <-> Python boundary
        # violation: silent truncation here would corrupt the
        # bottleneck and throughput aggregates for every subsequent
        # cycle, so we fail loud with the same shape-mismatch
        # convention used by ``_resolve_thresholds``.
        if rust_stages and len(rust_stages) != len(self._stage_names):
            msg = (
                f"update_with_measurements shape mismatch: "
                f"measurements.rust.stages has {len(rust_stages)} entries but "
                f"setup() captured {len(self._stage_names)} stage names "
                f"(known: {sorted(self._stage_names)})"
            )
            raise ValueError(msg)
        with self._lock:
            for stage_name, stage_measurements in zip(self._stage_names, rust_stages, strict=False):
                task_measurements = stage_measurements.task_measurements
                count = len(task_measurements)
                if count == 0:
                    continue
                self.completed_counts[stage_name] = self.completed_counts.get(stage_name, 0) + count
                # Accumulate cumulative per-stage service-time sums in
                # the same loop so the bottleneck score / heterogeneity
                # ratio see the same per-cycle window the backlog-time
                # pressure throughput sample sees. ``duration()`` is the
                # Rust-side ``end - start`` accessor; each ``count``
                # contributes one sample, so ``mean = sum / count`` is a
                # straightforward Forced-Flow ``S_k`` estimate (V_k = 1
                # for Xenna's linear DAG, so D_k = S_k).
                sum_service = sum(tm.duration() for tm in task_measurements)
                self.completed_service_time_sums[stage_name] = (
                    self.completed_service_time_sums.get(stage_name, 0.0) + sum_service
                )

    def consume_throughput_samples(self, now_ts: float) -> dict[str, float]:
        """Snapshot the per-stage completed-count delta and emit throughput samples.

        Called once per autoscale cycle. Computes ``dcount / dt``
        against the previously snapshotted ``(count, ts)`` under
        the collector's lock and refreshes the snapshot. Returns
        ``0.0`` for cold-start stages. The returned mapping always
        contains every name in the collector's stage list so
        callers can index without a missing-key fallback.

        """
        samples: dict[str, float] = {}
        with self._lock:
            for stage_name in self._stage_names:
                now_count = self.completed_counts.get(stage_name, 0)
                prev = self.last_throughput_sample.get(stage_name)
                if prev is None:
                    samples[stage_name] = 0.0
                else:
                    prev_count, prev_ts = prev
                    dt = now_ts - prev_ts
                    dcount = max(0, now_count - prev_count)
                    samples[stage_name] = dcount / dt if dt > 0.0 else 0.0
                self.last_throughput_sample[stage_name] = (now_count, now_ts)
        return samples

    def consume_service_time_samples(self) -> dict[str, float]:
        """Snapshot per-stage cumulative service-time deltas and emit mean ``S_k`` samples.

        Computes per-stage mean per-task service time as
        ``dsum / dcount`` over the in-cycle window. Cold-start
        stages yield ``math.nan``. Keeps an independent
        ``(count, sum)`` snapshot so the call order with
        ``consume_throughput_samples`` is irrelevant. Returns a
        dict containing every name in the collector's stage list.

        """
        samples: dict[str, float] = {}
        with self._lock:
            for stage_name in self._stage_names:
                now_count = self.completed_counts.get(stage_name, 0)
                now_sum = self.completed_service_time_sums.get(stage_name, 0.0)
                prev = self.last_service_time_sample.get(stage_name)
                if prev is None:
                    samples[stage_name] = math.nan
                else:
                    prev_count, prev_sum = prev
                    dcount = max(0, now_count - prev_count)
                    dsum = now_sum - prev_sum
                    if dcount > 0 and dsum > 0.0:
                        samples[stage_name] = dsum / dcount
                    else:
                        samples[stage_name] = math.nan
                self.last_service_time_sample[stage_name] = (now_count, now_sum)
        return samples


__all__ = ["MeasurementCollector"]
