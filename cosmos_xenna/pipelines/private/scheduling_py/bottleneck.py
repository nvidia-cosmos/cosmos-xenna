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

"""Per-cycle Forced-Flow-Law bottleneck-score emission.

The Forced Flow Law (Lazowska, Zahorjan, Graham, Sevcik 1984,
*Quantitative System Performance*, Chapter 3) gives the per-stage
service demand of a separable queueing network as::

    D_k = V_k * S_k

where:

  * ``V_k`` is the per-task visit ratio at stage ``k`` -- how many
    times one end-to-end task visits that stage.
  * ``S_k`` is the mean per-task service time at stage ``k`` in
    seconds (the time one actor spends processing one task).

Xenna's streaming pipeline is a linear DAG: every input task visits
every stage exactly once, so ``V_k = 1`` for all stages and the
service demand reduces to ``D_k = S_k``. The bottleneck is the
stage with the largest ``D_k``; pipeline throughput is bounded by
``1 / max_k D_k`` regardless of how many workers the non-bottleneck
stages carry (Little's Law plus the Forced Flow Law -- see
``cosmos-xenna/docs/scheduler/saturation-aware/23-bottleneck-score-metric.md``).

Pure observability
==================

This module emits the metric and the per-cycle INFO log line only.
The autoscaler's behaviour is unchanged whether the bottleneck
score is observed or not; the metric is a diagnostic substrate for
operators and (later) for a Dominant Resource Fairness donor pass.
The single value-add is letting operators rank stages by
bottleneck candidacy from one panel -- or one ``grep`` of the
scheduler log -- instead of cross-correlating throughput, queue
depth, and per-task service time across many per-stage panels.

Cold-start contract
===================

A stage whose mean per-task service time has not yet been observed
this cycle has an undefined ``S_k``. The helper folds every shape
of "no measurement" into a single sentinel -- ``math.nan`` -- and
pins the following contract for the cycle:

  * The ``xenna_stage_bottleneck_score{stage, pipeline}`` gauge is
    observed with ``math.nan`` so the metric's label cardinality
    stays stable across cycles. Prometheus' text exposition format
    represents NaN as the literal ``NaN``; consumers must already
    tolerate NaN samples from any per-stage gauge that depends on
    a first task completion.
  * The stage is excluded from the ``argmax_k D_k`` computation.
    A phantom NaN winner would mis-attribute the bottleneck.
  * The per-cycle INFO log line skips the stage. If every stage in
    the input mapping is cold, no INFO log fires that cycle.

Three input shapes map to the cold-start sentinel:

  * ``math.nan`` (no measurement yet -- the canonical cold-start).
  * ``0.0`` -- comes from
    ``processing_speed_tasks_per_second == 0`` where
    ``S_k = 1 / 0`` is undefined; treated identically to NaN so
    the call site does not have to special-case division by zero.
  * Any non-positive value -- a physical service time is
    strictly positive; treating negative samples as cold-start
    prevents a pathological measurement from poisoning the argmax.

ASCII flow
==========

::

      service_times_s = {                                    +-> per-stage gauge.set(D_k)
        "download": 0.05s,   <-- S_k                         |
        "caption":  2.00s,   <-- S_k (bottleneck)            |   {stage, pipeline}
        "embed":    0.10s,   <-- S_k                         |
      }                                                      |
                |                                            |
                v                                            |
        +--------------------+                               |
        | for each stage k:  |  D_k = V_k * S_k    V_k = 1  >+
        |   set gauge(D_k)   |                              \\
        |   if finite, keep  |                               \\
        +--------------------+                                \\
                |                                              v
                v                                          NaN samples
        argmax_k D_k = "caption"                           skipped from
                |                                          argmax / log
                v
        logger.info("bottleneck stage: caption "
                    "(D = 2.00s, throughput bound = 0.50 tasks/s)")

Cluster heterogeneity ratio extension
=====================================

The Forced-Flow-Law per-stage view above is augmented by a
**cluster-wide** scalar -- the ratio ``max_k D_k / min_k D_k``
across stages whose service time was observed this cycle. The
ratio is the operational analog of Liu & Ying 2026's heavy-traffic
``ksub`` / ``ksuper`` bounds: a homogeneous pipeline (ratio = 1)
behaves like a single-class M/M/c queue, whereas a heterogeneous
pipeline with one slow stage and many fast ones converges more
slowly and benefits from a longer shrink streak on the bottleneck
to absorb measurement noise.

When the ratio sits above ``cluster_heterogeneity_warn_threshold``
for a streak of ``cluster_heterogeneity_warn_streak`` consecutive
cycles, :func:`compute_heterogeneity_ratio` emits one INFO log
line naming the bottleneck stage and recommending the
``over_provisioned_streak_min_cycles`` knob as the primary tuning
lever. The streak gate prevents alert fatigue on transient spikes
(a brief NaN cycle or one above-threshold cycle does not fire)
while the once-per-streak semantics avoid flooding the log when
the heterogeneous regime is sustained. The ratio gauge is tagged
``{pipeline}`` only -- not per-stage -- because the ratio is a
single-cluster scalar and adding a stage tag would multiply the
metric cardinality without any new operator-relevant signal.
"""

import math
from collections.abc import Mapping

import attrs
from ray.util.metrics import Gauge

from cosmos_xenna.utils import python_log as logger

# Module-level Gauge instance. Ray's Gauge holds a Cython metric
# handle that registers with the Ray metrics agent on construction;
# constructing one per call would re-register on every cycle and is
# 2-3 orders of magnitude slower than reusing a single handle. The
# tag keys ``{stage, pipeline}`` are the cardinality envelope: one
# observation per stage per pipeline per cycle.
_BOTTLENECK_GAUGE = Gauge(
    "xenna_stage_bottleneck_score",
    description=(
        "Forced Flow Law per-stage service demand D_k = V_k * S_k in "
        "seconds. For Xenna's linear DAG V_k = 1 so the metric reduces "
        "to the mean per-task service time. The bottleneck stage is "
        "argmax_k D_k; pipeline throughput is bounded by 1 / max_k D_k. "
        "NaN samples indicate cold-start stages with no completed "
        "task in the current cycle."
    ),
    tag_keys=("stage", "pipeline"),
)


def _service_time_to_score(service_time_s: float) -> float:
    """Map a per-stage service-time sample to its Forced-Flow ``D_k``.

    Xenna's pipeline is a linear DAG, so the visit ratio ``V_k``
    equals 1 for every stage and ``D_k = V_k * S_k`` reduces to
    ``D_k = S_k``. Any non-finite or non-positive input is folded
    into the cold-start sentinel ``math.nan``:

      * ``math.nan`` (no measurement yet) passes through unchanged.
      * ``0.0`` comes from ``processing_speed_tasks_per_second == 0``
        where ``S_k = 1 / 0`` is undefined.
      * Strictly negative values cannot represent physical service
        times and are also folded into the cold-start case so a
        pathological sample cannot win the argmax.

    Args:
        service_time_s: Mean per-task service time in seconds for
            one stage. Typically computed at the call site as
            ``1.0 / pool_stats.processing_speed_tasks_per_second``
            from ``ActorPoolStats``.

    Returns:
        ``D_k`` in seconds, or ``math.nan`` when the input is not
        a finite positive number.
    """
    if not math.isfinite(service_time_s) or service_time_s <= 0.0:
        return math.nan
    return service_time_s


def emit_bottleneck_score(
    *,
    service_times_s: Mapping[str, float],
    pipeline_name: str,
) -> None:
    """Emit per-stage bottleneck-score gauges and one INFO log line.

    Computes ``D_k = V_k * S_k`` for every stage in ``service_times_s``
    (``V_k = 1`` for Xenna's linear DAG, so ``D_k = S_k``), observes
    the ``xenna_stage_bottleneck_score{stage, pipeline}`` gauge for
    each, and emits one INFO log line naming the bottleneck.

    Side effects per call:

      * Every entry in ``service_times_s`` triggers exactly one
        gauge observation. Cold-start stages (NaN, zero, or
        negative service time) observe ``math.nan`` so the gauge's
        cardinality stays stable across cycles even when a stage
        has not produced its first completed-task sample.
      * If at least one stage has a finite positive ``D_k``, a
        single INFO log line is emitted naming the bottleneck
        stage and the throughput bound
        ``1 / max_k D_k`` in tasks per second. The format is
        pinned by ``test_log_format_pins_throughput_bound``.
      * If every stage is in cold-start, no INFO log fires that
        cycle. Operators see the gauges go NaN; the bottleneck is
        unknown and the helper does not pretend otherwise.

    The helper is pure observability -- it does NOT mutate any
    scheduler state, does NOT influence the per-cycle ``Solution``,
    and is safe to call once per autoscale cycle from
    ``SaturationAwareScheduler.autoscale``.

    Args:
        service_times_s: Mapping from stage name to mean per-task
            service time in seconds (``S_k``). The mapping is
            consumed read-only. An empty mapping is a valid input
            and produces no observations and no log line --
            matching the cold-start contract for the very first
            cycle when the scheduler has not yet wired up
            ``ActorPoolStats``.
        pipeline_name: Pipeline identifier used as the ``pipeline``
            Prometheus tag so multiple pipelines running inside
            the same Ray cluster remain distinguishable on the
            metric.
    """
    if not service_times_s:
        return

    finite_scores: dict[str, float] = {}
    for stage_name, service_time_s in service_times_s.items():
        score = _service_time_to_score(service_time_s)
        _BOTTLENECK_GAUGE.set(
            score,
            tags={"stage": stage_name, "pipeline": pipeline_name},
        )
        if math.isfinite(score):
            finite_scores[stage_name] = score

    if not finite_scores:
        return

    bottleneck_name, bottleneck_score = max(
        finite_scores.items(),
        key=lambda item: item[1],
    )
    throughput_bound = 1.0 / bottleneck_score
    logger.info(
        f"bottleneck stage: {bottleneck_name} "
        f"(D = {bottleneck_score:.2f}s, "
        f"throughput bound = {throughput_bound:.2f} tasks/s)"
    )


# Cluster heterogeneity ratio.
#
# Module-level Gauge for the cluster-wide ratio. ``tag_keys`` is
# ``("pipeline",)`` only -- the ratio is a single scalar per cluster,
# so adding a ``stage`` tag would falsely multiply the cardinality
# without any new operator-relevant signal. Reusing one Cython
# handle across cycles keeps the per-cycle observation cost
# negligible (constructing a Gauge re-registers with the Ray
# metrics agent, which is 2-3 orders of magnitude slower).
_HETEROGENEITY_RATIO_GAUGE = Gauge(
    "xenna_scheduler_cluster_heterogeneity_ratio",
    description=(
        "Cluster heterogeneity ratio max(D_k) / min(D_k) across stages "
        "whose Forced-Flow service demand was observed this cycle. "
        "Ratio = 1.0 means a perfectly homogeneous pipeline; ratios "
        "much above 1.0 indicate a single bottleneck stage that "
        "dominates pipeline throughput. NaN samples mean fewer than "
        "two stages had a finite D_k this cycle (cold-start or "
        "all-but-one cold-start)."
    ),
    tag_keys=("pipeline",),
)


@attrs.define
class HeterogeneityWarnState:
    """Per-instance streak + once-per-spike latch for the heterogeneity warn log.

    Lives as a per-instance attribute on
    :class:`~cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.SaturationAwareScheduler`
    rather than at module level because the scheduler may be
    re-instantiated across tests; sharing a streak counter across
    instances would leak streak state into a fresh scheduler and
    fire spurious logs on the very first cycles of a new run.

    Attributes:
        streak_cycles: Number of consecutive ``autoscale()`` cycles
            in which the cluster heterogeneity ratio has stayed
            strictly above
            ``SaturationAwareConfig.cluster_heterogeneity_warn_threshold``.
            Resets to ``0`` the moment the ratio falls back to or
            below the threshold, or becomes ``math.nan`` (fewer
            than two stages with a finite ``D_k`` -- there is no
            meaningful heterogeneity verdict on a cold-start
            cluster).
        has_fired: ``True`` once the streak has reached the
            ``cluster_heterogeneity_warn_streak`` threshold and the
            INFO log has been emitted. Pinned to suppress the
            once-per-streak log from re-firing on every subsequent
            above-threshold cycle. Cleared back to ``False`` when
            the ratio drops to or below the threshold so a future
            climb-back can re-arm a fresh INFO log.
    """

    streak_cycles: int = 0
    has_fired: bool = False

    def reset(self) -> None:
        """Reset both fields to their construction-time defaults.

        Called from ``SaturationAwareScheduler.setup()`` next to
        the existing ``MemoryPressureMonitor.reset()`` call so a
        re-setup of the scheduler (e.g. a new pipeline run on the
        same scheduler instance) starts from a clean streak ledger
        without any inherited has-fired latch.
        """
        self.streak_cycles = 0
        self.has_fired = False


def compute_heterogeneity_ratio(
    *,
    service_times_s: Mapping[str, float],
    pipeline_name: str,
    state: HeterogeneityWarnState,
    warn_threshold: float,
    warn_streak_cycles: int,
) -> None:
    """Emit the cluster heterogeneity ratio gauge and an optional INFO log.

    Computes ``ratio = max_k D_k / min_k D_k`` across the stages in
    ``service_times_s`` whose service time is finite and strictly
    positive (cold-start stages -- ``math.nan``, ``0.0``, or
    negative values -- are excluded via :func:`_service_time_to_score`).
    When fewer than two stages have a finite ``D_k`` the ratio is
    undefined; the gauge then observes ``math.nan`` so its label
    cardinality stays stable across cycles, and the streak counter
    resets to 0 (a cold-start cluster has no heterogeneity verdict).

    The state argument is mutated in place:

      * On a finite ratio at or below ``warn_threshold``, the
        streak counter resets and the ``has_fired`` latch clears
        so a future climb-back can re-arm a fresh INFO log.
      * On a finite ratio strictly above ``warn_threshold``, the
        streak counter increments. When it reaches
        ``warn_streak_cycles`` (inclusive) and ``has_fired`` is
        still ``False``, exactly one INFO log line fires naming
        the bottleneck stage (the same ``argmax_k D_k`` that
        :func:`emit_bottleneck_score` identifies, per Liu & Ying
        2026's ``mu_max`` definition for the heavy-traffic limit)
        and ``has_fired`` is set to ``True`` so subsequent
        above-threshold cycles do not re-emit the log.
      * On an undefined ratio (``math.nan``), the streak counter
        resets but ``has_fired`` is intentionally NOT cleared --
        a transient cold-start cycle does not re-arm the log;
        only an actual drop to or below the threshold does.

    The helper is pure observability -- it does NOT mutate any
    scheduler state outside ``state``, does NOT influence the
    per-cycle ``Solution``, and is safe to call once per autoscale
    cycle from ``SaturationAwareScheduler.autoscale``.

    Args:
        service_times_s: Mapping from stage name to mean per-task
            service time in seconds (``S_k``). The mapping is
            consumed read-only and shared with
            :func:`emit_bottleneck_score` -- both helpers see the
            same per-stage ``D_k`` view in a given cycle.
        pipeline_name: Pipeline identifier used as the
            ``pipeline`` Prometheus tag. The gauge is tagged
            ``{pipeline}`` only because the ratio is a
            single-cluster scalar; per-stage tagging would
            multiply cardinality without any new signal.
        state: Mutable streak ledger owned by the scheduler. The
            helper updates ``streak_cycles`` and ``has_fired`` in
            place; callers must keep the same instance across
            cycles for the streak gate to work.
        warn_threshold: Ratio threshold above which the streak
            counter accumulates. Operators tune via
            ``SaturationAwareConfig.cluster_heterogeneity_warn_threshold``.
        warn_streak_cycles: Number of consecutive
            above-threshold cycles required before the INFO log
            fires. Operators tune via
            ``SaturationAwareConfig.cluster_heterogeneity_warn_streak``.
    """
    finite_scores: dict[str, float] = {}
    for stage_name, service_time_s in service_times_s.items():
        score = _service_time_to_score(service_time_s)
        if math.isfinite(score):
            finite_scores[stage_name] = score

    if len(finite_scores) < 2:
        # Undefined ratio: emit NaN to keep cardinality stable, reset
        # streak (a cold-start cluster has no heterogeneity verdict).
        # ``has_fired`` is intentionally NOT cleared -- only a real
        # drop to or below the threshold re-arms a fresh INFO log.
        _HETEROGENEITY_RATIO_GAUGE.set(
            math.nan,
            tags={"pipeline": pipeline_name},
        )
        state.streak_cycles = 0
        return

    max_d = max(finite_scores.values())
    min_d = min(finite_scores.values())
    ratio = max_d / min_d
    _HETEROGENEITY_RATIO_GAUGE.set(
        ratio,
        tags={"pipeline": pipeline_name},
    )

    if ratio <= warn_threshold:
        state.streak_cycles = 0
        state.has_fired = False
        return

    state.streak_cycles += 1
    if state.streak_cycles < warn_streak_cycles or state.has_fired:
        return

    bottleneck_name, bottleneck_score = max(
        finite_scores.items(),
        key=lambda item: item[1],
    )
    logger.info(
        f"high cluster heterogeneity (ratio={ratio:.1f} for "
        f"{state.streak_cycles} cycles); consider raising "
        f"over_provisioned_streak_min_cycles for stage "
        f"{bottleneck_name} (bottleneck D={bottleneck_score:.1f}s) "
        f"to give it more recovery margin"
    )
    state.has_fired = True
