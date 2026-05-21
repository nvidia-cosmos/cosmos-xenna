# 06 — Backlog-Time Signal (Compound AND-Criterion)

> **Status: PLANNED, not yet wired.**
>
> The classifier today consumes only the smoothed empty-slot ratio
> and the integer `input_queue_depth`. There is no throughput
> estimate threaded through `autoscale(...)`, so the
> `backlog_time = input_queue_depth / observed_throughput` term
> described below is not yet computed. Wiring this signal requires
> extending the scheduler protocol; that work is tracked in the
> saturation-aware roadmap and is intentionally out of scope for
> the current MR. This document remains as the design target.

## TL;DR

Saturation is fired only when **both** observables agree: the smoothed
empty-slot ratio is below `saturation_threshold` AND the smoothed
backlog drain time is above `target_backlog_seconds`. A momentary
slot fill is no longer enough — the queue must also be genuinely
backlogged.

## Problem

The smoothed empty-slot ratio that
[`classify`](../../../cosmos_xenna/pipelines/private/scheduling_py/classifier.py)
already consumes is a single-axis observable. It answers "are the
worker slots busy right now?" but says nothing about whether more
workers would help. Two operational scenarios show why utilisation
alone is not enough:

- **Transient input burst.** A short spike momentarily fills every
  slot. The empty-slot EWMA dips below `saturation_threshold` for a
  few cycles, but the input queue is still draining (no sustained
  work has built up). A utilisation-only classifier would scale up;
  by the time the new workers warm up, the burst is gone and the
  next cycle observes `OVER_PROVISIONED`. The autoscaler oscillates
  and pays warmup cost for no throughput gain.
- **Queue stuck despite idle slots.** A downstream stage cannot grow
  because of placement constraints (cluster full, donor cooldown).
  The slot signal looks idle (empty slots ≥ threshold), yet the input
  queue keeps growing. A utilisation-only classifier would treat
  this as `NORMAL` or even `OVER_PROVISIONED` and miss the structural
  bottleneck entirely.

The signal needed is the **queue-time** observable — "how long would
the current queue take to drain at the current throughput?" That
number says where the stage is heading; the empty-slot ratio says
where it is right now. The two are the principal observables that
[Little's Law](https://en.wikipedia.org/wiki/Little%27s_law) connects:
`L = lambda * W`, where `L` is queue length, `lambda` is arrival
rate, and `W` is the per-task waiting time. They tell complementary
stories and either one alone misclassifies real workloads.

## Decision

Use a compound AND-criterion on (utilisation, backlog-time) for the
saturation zones. A stage is classified `SATURATED` only when both:

```
    slots_empty_ratio_ewma  <  saturation_threshold      (utilisation low)
    AND
    backlog_time            >  target_backlog_seconds   (queue not draining)
```

with `backlog_time = input_queue_depth / observed_throughput` (the
W_q observable of Little's Law, in seconds). `SATURATED_CRITICAL`
mirrors the criterion at tighter thresholds; `OVER_PROVISIONED`
mirrors it on the opposite side. `STARVED` keeps a separate
empty-queue guard and is not part of the AND-criterion.

```
                              slots_empty_ratio_ewma
                      < saturation_threshold      ≥ saturation_threshold
                  ┌─────────────────────────────┬─────────────────────────────┐
  backlog_time    │         SATURATED           │           NORMAL            │
  >  target_      │   (scale up: both           │   (backpressured            │
  backlog_seconds │    observables agree on     │    elsewhere — queue stuck  │
                  │    pressure)                │    despite idle slots; this │
                  │                             │    stage is NOT the         │
                  │                             │    bottleneck and growing   │
                  │                             │    it will not help)        │
                  ├─────────────────────────────┼─────────────────────────────┤
  backlog_time    │           NORMAL            │      NORMAL / STARVED       │
  ≤  target_      │   (transient burst —        │   (genuine idle; STARVED    │
  backlog_seconds │    queue is draining;       │    on a sustained empty-    │
                  │    do not react)            │    queue streak)            │
                  └─────────────────────────────┴─────────────────────────────┘

         backlog_time = input_queue_depth / observed_throughput
                       (Little's Law W_q observable, in seconds)
```

The four quadrants decompose cleanly:

- **Both saturated** — utilisation low AND backlog growing →
  `SATURATED` (genuine scale-up signal).
- **Only utilisation** — slots momentarily full, queue still
  draining → `NORMAL` (transient burst; warmup cost would dominate).
- **Only backlog** — queue stuck despite idle slots → `NORMAL`
  (placement / dispatch issue; the fix is at the bottlenecked
  downstream stage — either growing it directly or routing a
  worker to it through the cross-stage donor protocol; growing
  *this* stage is the wrong response).
- **Neither** — `NORMAL` (or `STARVED` once the empty-queue streak
  reaches its threshold).

**Trade-off.** The AND-criterion is strictly more conservative than
utilisation alone: every cycle where the utilisation-only rule
would have fired `SATURATED` but the queue is draining (low
backlog), the compound rule emits `NORMAL` instead. False-positive
scale-ups drop; true-positive scale-ups on sustained pressure are
unchanged because sustained pressure builds queue depth by
construction. The cost is one extra EWMA-smoothed signal
(`observed_throughput`) and one new operator-facing knob
(`target_backlog_seconds`).

## How it works

Per-cycle inside
[`run_per_stage_pipeline`](../../../cosmos_xenna/pipelines/private/scheduling_py/pipeline.py):

- The empty-slot EWMA is refreshed by `_resolve_classifier_signal`
  using the existing `slots_empty_ratio_smoothing_level` weight from
  [`SaturationAwareStageConfig`](../../../cosmos_xenna/pipelines/private/specs.py).
- A second EWMA tracks per-stage observed throughput. The weight is
  `observed_throughput_smoothing_level`. The cold-start edge case
  (no completed batches yet, `observed_throughput == 0`) maps
  `backlog_time` to `+inf` when `input_queue_depth > 0`, which forces
  the AND-criterion to fire `SATURATED_CRITICAL` on cycle 1 — the
  "queue exists but no progress" pathology must not wait for a
  throughput sample.
- A final EWMA smooths the derived `backlog_time` itself with
  `backlog_time_smoothing_level`. This dampens the noisy
  (queue / throughput) ratio without losing structural pressure
  signals.
- The compound AND-criterion is evaluated inside
  [`classify`](../../../cosmos_xenna/pipelines/private/scheduling_py/classifier.py)
  using the smoothed slot ratio, `input_queue_depth`, and the
  smoothed `backlog_time`. The output feeds the existing streak
  counter and the
  [`compute_delta`](../../../cosmos_xenna/pipelines/private/scheduling_py/decisions.py)
  gate untouched — adding the backlog-time axis does not change the
  action-firing steps that follow inside the per-stage decision
  pipeline.

The `SATURATED_CRITICAL` and `OVER_PROVISIONED` thresholds for
`backlog_time` are derived from `target_backlog_seconds`
(`> 3 * target_backlog_seconds` for critical pressure;
`< 0.5 * target_backlog_seconds` for over-provisioned), so a single
operator-facing knob captures the intent across all three zones.

## Knobs

All on
[`SaturationAwareStageConfig`](../../../cosmos_xenna/pipelines/private/specs.py):

| Knob | Effect |
|---|---|
| `target_backlog_seconds` | AND-threshold for `SATURATED`. Higher = more conservative (longer queue accepted before scale-up); lower = more aggressive. `SATURATED_CRITICAL` and `OVER_PROVISIONED` ratios scale from this value. |
| `backlog_time_smoothing_level` | EWMA weight on the new `backlog_time` sample. Lower = smoother (filters noise); higher = more reactive to bursts. |
| `observed_throughput_smoothing_level` | EWMA weight on the per-cycle throughput sample. Lower = smoother (queue-time noise dampens); higher = reacts faster to step changes in stage speed. |

## See also

- [00 — Per-cycle overview](00-overview.md) — where the compound
  AND-criterion sits inside the intent compute stage.
- [05 — State classifier](05-state-classifier.md) — the five-zone
  decomposition that consumes the compound criterion.
- [07 — Streak stabilization](07-streak-stabilization.md) — the
  asymmetric streak counters that run after the AND-criterion fires.
- [Little's Law](https://en.wikipedia.org/wiki/Little%27s_law) —
  external reference for the queue-time observable used as the
  second axis.
- [Cloud Dataflow Streaming Engine autoscaling](https://cloud.google.com/dataflow/docs/streaming-engine)
  — external reference for the original `backlog_time` autoscaler
  signal.
