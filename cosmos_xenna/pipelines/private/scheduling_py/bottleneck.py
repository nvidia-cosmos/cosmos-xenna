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

The per-stage service demand is defined as::

    D_k = V_k * S_k / c_k

where:

  * ``V_k`` is the per-task visit ratio at stage ``k`` - how many
    times one end-to-end task visits that stage.
  * ``S_k`` is the mean per-task service time at stage ``k`` in
    seconds (the time one actor spends processing one task).
  * ``c_k`` is the effective ready service capacity at stage ``k``
    (count of concurrent service channels: each non-SPMD worker
    contributes ``slots_per_worker`` channels, and an SPMD worker
    group contributes ``slots_per_worker * num_allocations``
    channels because all allocations process tasks in parallel).

Xenna's streaming pipeline is a linear DAG: every input task visits
every stage exactly once, so ``V_k = 1`` for all stages and the
service demand reduces to ``D_k = S_k / c_k``. The bottleneck is
the stage with the largest ``D_k``; pipeline throughput is bounded
by ``min_k c_k / S_k = 1 / max_k D_k`` regardless of how many
workers the non-bottleneck stages carry (Little's Law plus the
Forced Flow Law - see
``cosmos-xenna/docs/scheduler/saturation-aware/23-bottleneck-score-metric.md``).

Pure observability
==================

This module emits the metric and the per-cycle INFO log line only.
The autoscaler's behaviour is unchanged whether the bottleneck
score is observed or not; the metric is a diagnostic substrate for
operators and (later) for a Dominant Resource Fairness donor pass.
The single value-add is letting operators rank stages by
bottleneck candidacy from one panel - or one ``grep`` of the
scheduler log - instead of cross-correlating throughput, queue
depth, and per-task service time across many per-stage panels.

Cold-start contract
===================

A stage whose mean per-task service time has not yet been observed
this cycle, or whose effective ready capacity is zero, has an
undefined ``D_k``. The helper folds every shape of "no
measurement" into a single sentinel - ``math.nan`` - and pins
the following contract for the cycle:

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

Inputs that map to the cold-start sentinel:

  * ``service_time_s`` is ``math.nan`` (no completed-task sample yet).
  * ``service_time_s`` is ``<= 0`` (defensive: a physical service
    time is strictly positive, and ``0.0`` arrives from
    ``processing_speed_tasks_per_second == 0`` where ``S_k = 1 / 0``
    is undefined).
  * ``effective_capacity`` is ``<= 0`` (no ready service channels;
    division would be undefined).

Flow chart
==========

::

      d_k_by_stage = {                                       +-> per-stage gauge.set(D_k)
        "download": 0.025s,  <-- S_k=0.05s, c_k=2            |
        "caption":  0.250s,  <-- S_k=2.0s,  c_k=8 (bottleneck)|  {stage, pipeline}
        "embed":    0.050s,  <-- S_k=0.10s, c_k=2            |
      }                                                      |
                |                                            |
                v                                            |
        +--------------------+                               |
        | for each stage k:  |  D_k already actor-normalized>+
        |   set gauge(D_k)   |                              \\
        |   if finite, keep  |                               \\
        +--------------------+                                \\
                |                                              v
                v                                          NaN samples
        argmax_k D_k = "caption"                           skipped from
                |                                          argmax / log
                v
        logger.info("bottleneck stage: caption "
                    "(D = 0.25s, throughput bound = 4.00 tasks/s, "
                    "capacity = 8)")

Cluster heterogeneity ratio extension
=====================================

The Forced-Flow-Law per-stage view above is augmented by a
**cluster-wide** scalar - the ratio ``max_k D_k / min_k D_k``
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
``{pipeline}`` only - not per-stage - because the ratio is a
single-cluster scalar and adding a stage tag would multiply the
metric cardinality without any new operator-relevant signal.
"""

import math
import statistics
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
        "Forced Flow Law per-stage actor-normalized service demand "
        "D_k = V_k * S_k / c_k in seconds-per-channel. For Xenna's "
        "linear DAG V_k = 1 so D_k = S_k / c_k where c_k is the "
        "effective ready capacity (concurrent service channels). The "
        "bottleneck stage is argmax_k D_k; pipeline throughput is "
        "bounded by 1 / max_k D_k. NaN samples indicate cold-start "
        "stages with no completed task in the current cycle, or "
        "stages with zero effective capacity."
    ),
    tag_keys=("stage", "pipeline"),
)


def compute_d_k(service_time_s: float, effective_capacity: int) -> float:
    """Compute the actor-normalized Forced-Flow service demand ``D_k``.

    Xenna's pipeline is a linear DAG, so the visit ratio ``V_k``
    equals 1 for every stage and ``D_k = V_k * S_k / c_k`` reduces
    to ``D_k = service_time_s / effective_capacity``. The result
    is the per-actor-channel service demand in seconds.

    Any cold-start input is folded into the sentinel ``math.nan``:

      * ``service_time_s`` is non-finite or ``<= 0`` (no sample,
        or ``processing_speed_tasks_per_second == 0`` where
        ``S_k = 1 / 0`` is undefined).
      * ``effective_capacity`` is ``<= 0`` (no ready channels;
        division undefined).

    Args:
        service_time_s: Mean per-task service time in seconds
            (intrinsic ``S_k``). Typically the EWMA-smoothed mean
            of ``dsum / dcount`` from ``ActorPoolStats``.
        effective_capacity: Concurrent service channels at this
            stage; sum over ready worker groups of
            ``slots_per_worker * max(1, num_allocations)``.

    Returns:
        ``D_k`` in seconds-per-channel, or ``math.nan`` when either
        input fails the finite-positive contract.
    """
    if not math.isfinite(service_time_s) or service_time_s <= 0.0:
        return math.nan
    if effective_capacity <= 0:
        return math.nan
    return service_time_s / effective_capacity


def emit_bottleneck_score(
    *,
    d_k_by_stage: Mapping[str, float],
    bottleneck_identity: "BottleneckIdentity",
    pipeline_name: str,
    effective_capacities: Mapping[str, int] | None = None,
) -> None:
    """Emit per-stage bottleneck-score gauges and one INFO log line.

    Observes the actor-normalized ``D_k`` already computed by
    :func:`compute_d_k` for every stage in ``d_k_by_stage`` on the
    ``xenna_stage_bottleneck_score{stage, pipeline}`` gauge, then
    emits one INFO log line naming the bottleneck stage selected
    by :func:`identify_bottleneck`.

    Side effects per call:

      * Every entry in ``d_k_by_stage`` triggers exactly one gauge
        observation. Cold-start stages (NaN or non-positive ``D_k``)
        observe ``math.nan`` so the gauge's cardinality stays stable
        across cycles even when a stage has not produced its first
        completed-task sample.
      * If ``bottleneck_identity.engaged`` and at least one stage
        has a finite positive ``D_k``, a single INFO log line is
        emitted naming the bottleneck stage from
        ``bottleneck_identity.stage_name`` and the throughput bound
        ``1 / max_k D_k`` in tasks per second. Sharing identity with
        :func:`identify_bottleneck` guarantees the operator-facing
        log cannot disagree with the planner's near-tie selection.
      * If no stage has a finite ``D_k``, no INFO log fires.
        Operators see the gauges go NaN; the bottleneck is
        unknown and the helper does not pretend otherwise.

    The helper is pure observability - it does NOT mutate any
    scheduler state, does NOT influence the per-cycle ``Solution``,
    and is safe to call once per autoscale cycle from
    ``SaturationAwareScheduler.autoscale``.

    Args:
        d_k_by_stage: Mapping from stage name to actor-normalized
            ``D_k`` in seconds-per-channel (the same mapping fed to
            :func:`identify_bottleneck` and
            :func:`compute_heterogeneity_ratio`). Consumed read-only.
            An empty mapping produces no observations and no log
            line - matching the cold-start contract for the very
            first cycle.
        bottleneck_identity: Per-cycle identity from
            :func:`identify_bottleneck`. The INFO log uses
            ``identity.stage_name`` so the line cannot disagree with
            the near-tie selection that drives Phase C / Phase D.
        pipeline_name: Pipeline identifier used as the ``pipeline``
            Prometheus tag so multiple pipelines running inside
            the same Ray cluster remain distinguishable.
        effective_capacities: Optional mapping from stage name to
            effective ready capacity (``c_k``). When supplied the
            bottleneck stage's capacity is included in the INFO log
            for operator sanity-checking; absent or missing keys
            produce a log line without the ``capacity`` field.
    """
    if not d_k_by_stage:
        return

    finite_scores: dict[str, float] = {}
    for stage_name, d_k in d_k_by_stage.items():
        is_finite_positive = math.isfinite(d_k) and d_k > 0.0
        observed = d_k if is_finite_positive else math.nan
        _BOTTLENECK_GAUGE.set(
            observed,
            tags={"stage": stage_name, "pipeline": pipeline_name},
        )
        if is_finite_positive:
            finite_scores[stage_name] = d_k

    if not finite_scores:
        return

    if bottleneck_identity.stage_name is None or bottleneck_identity.stage_name not in finite_scores:
        return

    bottleneck_name = bottleneck_identity.stage_name
    bottleneck_score = finite_scores[bottleneck_name]
    throughput_bound = 1.0 / bottleneck_score
    if effective_capacities is not None and bottleneck_name in effective_capacities:
        capacity_suffix = f", capacity = {effective_capacities[bottleneck_name]}"
    else:
        capacity_suffix = ""
    logger.info(
        f"bottleneck stage: {bottleneck_name!r} "
        f"(D = {bottleneck_score:.2f}s, "
        f"throughput bound = {throughput_bound:.2f} tasks/s"
        f"{capacity_suffix})"
    )


# Cluster heterogeneity ratio.
#
# Module-level Gauge for the cluster-wide ratio. ``tag_keys`` is
# ``("pipeline",)`` only - the ratio is a single scalar per cluster,
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


# Cluster pipeline balance score (1.0 / max(1.0, heterogeneity_ratio)).
#
# Same per-cluster cardinality as the heterogeneity ratio; the
# balance score is the inverse-ratio formulation operators reach
# for when reasoning about "how close is this pipeline to
# perfectly balanced". Score = 1.0 means perfectly balanced (or a
# single-stage cluster); scores approaching 0 mean a single stage
# is the dominant bottleneck. NaN samples mean fewer than two
# stages had a finite D_k this cycle.
_BALANCE_SCORE_GAUGE = Gauge(
    "xenna_scheduler_pipeline_balance_score",
    description=(
        "Cluster pipeline balance score = 1.0 / max(1.0, heterogeneity_ratio). "
        "Score = 1.0 means a perfectly balanced pipeline; scores approaching "
        "0 mean a single bottleneck stage dominates throughput. NaN means "
        "fewer than two stages had a finite D_k this cycle."
    ),
    tag_keys=("pipeline",),
)


def compute_balance_score(d_k_by_stage: Mapping[str, float]) -> float:
    """Compute ``1.0 / max(1.0, max(D_k) / min(D_k))`` across stages with finite ``D_k``.

    Filters ``d_k_by_stage`` to entries whose value is finite and
    strictly positive (cold-start stages -- ``math.nan``, ``0.0``,
    or negative values -- are excluded). When fewer than two stages
    qualify the ratio is undefined and the helper returns
    ``math.nan`` so callers comparing balance across cycles can
    treat NaN as "no verdict" without distinguishing it from an
    actual zero.

    Args:
        d_k_by_stage: Mapping from stage name to actor-normalized
            ``D_k`` in seconds-per-channel. The same view that
            :func:`compute_heterogeneity_ratio` consumes.

    Returns:
        The balance score in ``(0.0, 1.0]`` when at least two
        stages have a finite positive ``D_k``; ``math.nan``
        otherwise.

    """
    finite = [v for v in d_k_by_stage.values() if math.isfinite(v) and v > 0.0]
    if len(finite) < 2:
        return math.nan
    ratio = max(finite) / min(finite)
    return 1.0 / max(1.0, ratio)


def emit_balance_score(d_k_by_stage: Mapping[str, float], *, pipeline_name: str) -> float:
    """Emit ``xenna_scheduler_pipeline_balance_score`` and return the scalar.

    Computes the score via :func:`compute_balance_score`, observes
    it on ``_BALANCE_SCORE_GAUGE``, and hands the value back so the
    caller can use it for the per-cycle balance regression check
    without a second computation.

    Args:
        d_k_by_stage: Same view that :func:`emit_bottleneck_score`
            and :func:`compute_heterogeneity_ratio` consume.
        pipeline_name: ``pipeline`` Prometheus tag value.

    Returns:
        The observed balance score (or ``math.nan`` for cold-start
        cycles).

    """
    score = compute_balance_score(d_k_by_stage)
    _BALANCE_SCORE_GAUGE.set(score, tags={"pipeline": pipeline_name})
    return score


_BOTTLENECK_NEAR_TIE_TOLERANCE: float = 0.05
"""Default ``near_tie_tolerance`` for bottleneck stage selection.

Shared by :func:`identify_bottleneck` (the Phase C/D engagement
selector) and :func:`compute_heterogeneity_ratio` (the operator-
facing warn log) so both name the same stage when two or more
``D_k`` values are within the tolerance band of the leader.
Drift between the two would surface as a warn log that points at
a different stage than the scheduler is acting on.
"""


def _pick_lex_stable_argmax(
    finite_scores: Mapping[str, float],
    *,
    near_tie_tolerance: float,
) -> str:
    """Pick the bottleneck stage with lex-stable near-tie resolution.

    Returns the lexicographically-smallest stage whose ``D_k`` is
    within ``near_tie_tolerance`` of the leader. Sharing this
    primitive between :func:`identify_bottleneck` and
    :func:`compute_heterogeneity_ratio` keeps Phase C/D engagement
    and the operator-facing heterogeneity warn log naming the same
    stage on near-ties; using raw ``max(...)`` in either place
    would resolve ties by dict insertion order and drift away from
    the lex pick.

    Args:
        finite_scores: Per-stage finite, positive ``D_k`` values
            (caller is responsible for filtering NaN / non-positive
            entries up front).
        near_tie_tolerance: Fractional tie band in ``[0.0, 1.0)``;
            ``0.0`` collapses to a strict-argmax with lex tie-break
            still applied.

    Returns:
        Stage name selected as the bottleneck.

    Raises:
        ValueError: ``finite_scores`` is empty or
            ``near_tie_tolerance`` is outside ``[0.0, 1.0)``.
    """
    if not finite_scores:
        msg = "finite_scores must contain at least one entry"
        raise ValueError(msg)
    if not 0.0 <= near_tie_tolerance < 1.0:
        msg = f"near_tie_tolerance must be in [0.0, 1.0), got {near_tie_tolerance}"
        raise ValueError(msg)

    max_d = max(finite_scores.values())
    near_tie_floor = max_d * (1.0 - near_tie_tolerance)
    return min(name for name, d in finite_scores.items() if d >= near_tie_floor)


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
            than two stages with a finite ``D_k`` - there is no
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
    d_k_by_stage: Mapping[str, float],
    pipeline_name: str,
    state: HeterogeneityWarnState,
    warn_threshold: float,
    warn_streak_cycles: int,
) -> None:
    """Emit the cluster heterogeneity ratio gauge and an optional INFO log.

    Computes ``ratio = max_k D_k / min_k D_k`` across the stages in
    ``d_k_by_stage`` whose actor-normalized ``D_k`` is finite and
    strictly positive (cold-start stages - ``math.nan``, ``0.0``,
    or negative values - are excluded). When fewer than two stages
    have a finite ``D_k`` the ratio is undefined; the gauge then
    observes ``math.nan`` so its label cardinality stays stable
    across cycles, and the streak counter resets to 0 (a cold-start
    cluster has no heterogeneity verdict).

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

    The helper is pure observability - it does NOT mutate any
    scheduler state outside ``state``, does NOT influence the
    per-cycle ``Solution``, and is safe to call once per autoscale
    cycle from ``SaturationAwareScheduler.autoscale``.

    Args:
        d_k_by_stage: Mapping from stage name to actor-normalized
            ``D_k`` in seconds-per-channel. Consumed read-only and
            shared with :func:`emit_bottleneck_score` and
            :func:`identify_bottleneck` - all three helpers see the
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
    for stage_name, d_k in d_k_by_stage.items():
        if math.isfinite(d_k) and d_k > 0.0:
            finite_scores[stage_name] = d_k

    if len(finite_scores) < 2:
        # Undefined ratio: emit NaN to keep cardinality stable, reset
        # streak (a cold-start cluster has no heterogeneity verdict).
        # ``has_fired`` is intentionally NOT cleared - only a real
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

    bottleneck_name = _pick_lex_stable_argmax(
        finite_scores,
        near_tie_tolerance=_BOTTLENECK_NEAR_TIE_TOLERANCE,
    )
    bottleneck_score = finite_scores[bottleneck_name]
    logger.info(
        f"high cluster heterogeneity (ratio={ratio:.1f} for "
        f"{state.streak_cycles} cycles); consider raising "
        f"over_provisioned_streak_min_cycles for stage "
        f"{bottleneck_name!r} (bottleneck D={bottleneck_score:.1f}s) "
        f"to give it more recovery margin"
    )
    state.has_fired = True


@attrs.define(frozen=True)
class BottleneckIdentity:
    """Per-cycle bottleneck identification.

    Produced by :func:`identify_bottleneck`. Consumers gate on
    ``engaged``; ``stage_name`` is the argmax stage when engaged.

    Attributes:
        engaged: True if at least two finite ``D_k`` samples exist
            and the heterogeneity ratio reaches the threshold.
        stage_name: Bottleneck stage when ``engaged``; otherwise None.
        max_d_k: Maximum finite ``D_k`` this cycle; NaN if none.
        median_d_k: Median (n>=3) or mean (n=2) of finite samples;
            NaN if fewer than two stages contribute.
        heterogeneity_ratio: ``max / median`` for n>=3, ``max / min``
            for n=2; NaN otherwise.
    """

    engaged: bool
    stage_name: str | None
    max_d_k: float
    median_d_k: float
    heterogeneity_ratio: float


@attrs.frozen
class BottleneckCycleContext:
    """Per-cycle bottleneck signal scoped to one stage.

    Attributes:
        engaged: True when the cluster-wide bottleneck gate is engaged
            this cycle. Mirrors ``BottleneckIdentity.engaged``.
        is_upstream_of_bottleneck: True when the owning stage sits
            strictly upstream (smaller DAG index) of the engaged
            bottleneck. Always False when ``engaged`` is False, and
            always False for the bottleneck stage itself.
    """

    engaged: bool = False
    is_upstream_of_bottleneck: bool = False


@attrs.define
class BottleneckEngagementState:
    """Streak ledger for debouncing the bottleneck-engagement INFO log.

    Attributes:
        last_announced: Last engagement state announced via INFO log;
            ``None`` until the first announcement.
        candidate_streak: Cycle counter whose semantic depends on
            ``last_announced``:

              * ``last_announced is None`` (cold-start seeding):
                consecutive cycles whose engagement matches
                ``last_candidate``. A value flip resets the counter
                to ``1``.
              * ``last_announced is not None`` (post-seeded): consecutive
                cycles whose engagement differs from ``last_announced``;
                resets on any agreement cycle.

            In both phases the counter advances to
            ``persistence_cycles`` before the gate fires.
        last_candidate: Most recently observed candidate value during
            the cold-start seeding phase; ``None`` until the first
            cycle. Used to detect value flips so a noisy mixed
            sequence (e.g. ``True, False, True`` with
            ``persistence_cycles=3``) cannot get its
            ``persistence_cycles``'th cycle's arbitrary value
            seeded as the announced state. Not consulted once
            ``last_announced`` is set (the post-seeded branch
            already gets consecutive-identical semantics for free
            because ``engaged: bool`` has only one "other" value).
    """

    last_announced: bool | None = None
    candidate_streak: int = 0
    last_candidate: bool | None = None

    def reset(self) -> None:
        """Reset to a fresh ledger."""
        self.last_announced = None
        self.candidate_streak = 0
        self.last_candidate = None


def identify_bottleneck(
    d_k_by_stage: Mapping[str, float],
    *,
    heterogeneity_threshold: float,
    near_tie_tolerance: float = _BOTTLENECK_NEAR_TIE_TOLERANCE,
) -> BottleneckIdentity:
    """Identify the engaged bottleneck stage from actor-normalized ``D_k``.

    The ratio is ``max / median`` for n>=3 and ``max / min`` for n=2.
    Engagement requires at least two finite, positive samples and a
    ratio that reaches ``heterogeneity_threshold``. Near-ties (within
    ``near_tie_tolerance`` of the leader) are resolved by
    lexicographic ``stage_name`` for cycle-to-cycle stability.

    Args:
        d_k_by_stage: Per-stage actor-normalized ``D_k`` in
            seconds-per-channel as produced by :func:`compute_d_k`;
            cold-start stages contribute ``math.nan``.
        heterogeneity_threshold: Engagement floor. Must be > 1.0.
        near_tie_tolerance: Fractional tie band in ``[0.0, 1.0)``;
            0.0 = strict argmax. Values >= 1.0 collapse the band so
            every positive score is "tied" and the verdict reduces
            to a lex-smallest pick, defeating the function; negative
            values would raise the floor above ``max_d`` and produce
            an empty ``tied``.

    Returns:
        ``BottleneckIdentity`` describing the cycle.

    Raises:
        ValueError: If ``heterogeneity_threshold`` is non-finite or
            ``<= 1.0`` (a homogeneous cluster has ratio ``1.0``;
            the floor must be strictly greater), or if
            ``near_tie_tolerance`` is outside ``[0.0, 1.0)``.
    """
    if not math.isfinite(heterogeneity_threshold) or heterogeneity_threshold <= 1.0:
        msg = f"heterogeneity_threshold must be finite and > 1.0, got {heterogeneity_threshold!r}"
        raise ValueError(msg)
    if not 0.0 <= near_tie_tolerance < 1.0:
        msg = f"near_tie_tolerance must be in [0.0, 1.0), got {near_tie_tolerance}"
        raise ValueError(msg)

    finite_scores: dict[str, float] = {}
    for name, d_k in d_k_by_stage.items():
        if math.isfinite(d_k) and d_k > 0.0:
            finite_scores[name] = d_k

    if not finite_scores:
        return BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=math.nan,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

    max_d = max(finite_scores.values())
    if len(finite_scores) < 2:
        return BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=max_d,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

    if len(finite_scores) == 2:
        min_d = min(finite_scores.values())
        median_d = (max_d + min_d) / 2.0
        ratio = max_d / min_d
    else:
        median_d = statistics.median(finite_scores.values())
        ratio = max_d / median_d if median_d > 0.0 else math.nan

    if not math.isfinite(ratio) or ratio < heterogeneity_threshold:
        return BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=max_d,
            median_d_k=median_d,
            heterogeneity_ratio=ratio,
        )

    return BottleneckIdentity(
        engaged=True,
        stage_name=_pick_lex_stable_argmax(
            finite_scores,
            near_tie_tolerance=near_tie_tolerance,
        ),
        max_d_k=max_d,
        median_d_k=median_d,
        heterogeneity_ratio=ratio,
    )


def maybe_log_bottleneck_engagement(
    *,
    identity: BottleneckIdentity,
    state: BottleneckEngagementState,
    persistence_cycles: int,
    pipeline_name: str,
) -> None:
    """Emit one INFO log when engagement has flipped persistently.

    Mutates ``state`` to debounce log spam at the gate boundary.

    Args:
        identity: This cycle's bottleneck identification.
        state: Mutable streak ledger owned by the scheduler.
        persistence_cycles: Cycles a candidate state must hold
            before announcement. Must be >= 1.
        pipeline_name: Pipeline identifier for the log line.
    """
    if state.last_announced is None:
        # Cold-start seeding: count only consecutive identical
        # candidate values. A noisy mixed sequence (True, False,
        # True, ...) must not seed ``last_announced`` based on
        # whichever value happens to land on the
        # ``persistence_cycles``'th cycle - the gate's contract is
        # persistence-based debouncing, not just "wait N cycles".
        if state.last_candidate is None or identity.engaged == state.last_candidate:
            state.candidate_streak += 1
        else:
            state.candidate_streak = 1
        state.last_candidate = identity.engaged
        if state.candidate_streak >= persistence_cycles:
            state.last_announced = identity.engaged
            state.candidate_streak = 0
        return

    if identity.engaged == state.last_announced:
        state.candidate_streak = 0
        return

    state.candidate_streak += 1
    if state.candidate_streak < persistence_cycles:
        return

    if identity.engaged:
        logger.info(
            f"bottleneck-priority decisions engaged for pipeline {pipeline_name!r}: "
            f"argmax stage {identity.stage_name!r} (D={identity.max_d_k:.2f}s, "
            f"heterogeneity_ratio={identity.heterogeneity_ratio:.2f})"
        )
    else:
        logger.info(
            f"bottleneck-priority decisions disengaged for pipeline {pipeline_name!r} "
            "(cluster homogeneous or cold-start)"
        )
    state.last_announced = identity.engaged
    state.candidate_streak = 0
