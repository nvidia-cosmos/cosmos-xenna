# 23 — Bottleneck Score Metric

## TL;DR

Each autoscale cycle the scheduler exports a per-stage Prometheus
gauge `xenna_stage_bottleneck_score{stage,pipeline}` equal to the
Forced Flow Law per-stage *service demand* `D_k = V_k × S_k`
(Lazowska, Zahorjan, Graham, Sevcik 1984, Chapter 3). For Xenna's
linear streaming DAG `V_k = 1`, so the metric reduces to per-stage
mean service time `S_k`. The stage with the largest `D_k` is the
bottleneck; pipeline throughput is bounded by `1 / max_k D_k` no
matter how much capacity the other stages have. A single INFO log
line per cycle names the current bottleneck so operators do not
need a dashboard open to diagnose one cycle.

## Problem

A streaming pipeline exposes many per-stage signals — actor count,
queue depth, slot occupancy, GPU utilisation, per-task wall clock —
and the operator instinct is to "scale the busy-looking stage".
That instinct is the most common autoscaler tuning mistake:

- The CPU-busiest stage is often **not** the throughput bottleneck.
  A stage at 90 % CPU but 50 ms per task is not the bottleneck if a
  downstream stage takes 2 s per task; the upstream stage is
  busy because it can produce faster than the downstream stage
  consumes.
- Adding workers to a non-bottleneck stage **cannot** raise total
  pipeline throughput — the new workers just push more inventory
  onto the same downstream queue. Pipeline throughput is bounded
  by `1 / max_k D_k`; non-max stages have headroom by definition.
- Identifying the bottleneck from the existing per-stage panels
  requires cross-correlating throughput, queue depth, and per-task
  service time across every stage simultaneously. That is exactly
  the kind of multi-panel correlation operators get wrong under
  pressure.

## Decision

Expose a per-stage Prometheus gauge whose value is the Forced Flow
Law service demand `D_k = V_k × S_k`:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   xenna_stage_bottleneck_score{stage, pipeline}              │
│   ─ value:  D_k = V_k * S_k                                  │
│   ─ where   V_k = 1   (linear Xenna DAG: every task visits   │
│                        every stage exactly once)             │
│             S_k = mean per-task service time                 │
│                  = 1 / processing_speed_tasks_per_second     │
│                                                              │
│   bottleneck stage           = argmax over k of D_k          │
│   pipeline throughput bound  = 1 / max_k D_k                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Example three-stage pipeline (`download` → `caption` → `embed`):

```
   ┌──────────┐    ┌───────────┐    ┌─────────┐
   │ download │ ─► │  caption  │ ─► │  embed  │
   │ S = 0.05s│    │ S = 2.00s │    │ S=0.10s │
   │ D = 0.05s│    │ D = 2.00s │    │ D=0.10s │
   └──────────┘    └───────────┘    └─────────┘
                         ▲▲▲
                  max_k D_k = 2.00 s
                  bottleneck stage = "caption"
                  pipeline throughput ≤ 1 / 2.00 = 0.5 tasks/s
```

The metric is numerically equal to the existing per-stage
`pipeline_actor_process_time` gauge published from
[`monitoring.py`](../../../cosmos_xenna/pipelines/private/monitoring.py),
so it adds no new measurement and no new sampling overhead. The
value of the new gauge is the **labelling**: the
`xenna_stage_bottleneck_score` name plus a per-cycle INFO log line
naming the argmax stage make the Forced-Flow framing explicit, so
an operator does not need to know the queueing-theory derivation
to read the dashboard.

**Trade-off.** The cost is one extra Prometheus gauge whose
cardinality is bounded by the (static) stage list per pipeline. The
benefit is a single number per stage that ranks stages by
bottleneck candidacy and prevents the "scale the busy-looking
stage" mistake. The same gauge is the diagnostic substrate for two
scheduler decisions consumed in
[25 — Bottleneck decision integration](25-bottleneck-decision-integration.md):
Phase C grow-priority ordering and Phase D shrink protection. The
metric remains the operator's diagnostic of record; the autoscaler
just feeds the same EWMA-smoothed value back into its own decisions
so the dashboard and the scheduler share one truth. The cardinality
envelope is paid once and reused.

## How it works

```
                       autoscale() cycle for pipeline P
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  update_with_measurements (per monitor tick) │
              │  for each stage k:                           │
              │    completed_count[k]      += len(tasks)     │
              │    service_time_sum[k]     += sum(tm.duration())
              │    (cumulative; thread-safe under self._lock)│
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  autoscale (per cycle)                       │
              │  _consume_service_time_samples():            │
              │    dcount = now_count - prev_count           │
              │    dsum   = now_sum   - prev_sum             │
              │    if dcount > 0 and dsum > 0:               │
              │       S_k = dsum / dcount                    │
              │    else:                                     │
              │       S_k = math.nan  (cold-start)           │
              │    snapshot (now_count, now_sum)             │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  emit_bottleneck_score(service_times_s={k:S_k│
              │                                             │
              │    D_k = V_k * S_k         (V_k = 1)         │
              │    gauge.set(D_k, {stage: k, pipeline: P})   │
              │    NaN samples skipped from argmax           │
              │  identify bottleneck = argmax_k D_k          │
              │  logger.info(                                │
              │     f"bottleneck stage: {name} "             │
              │     f"(D = {D:.2f}s, throughput bound = "    │
              │     f"{1/D:.2f} tasks/s)")                   │
              └──────────────────────────────────────────────┘
```

The mean per-task service time `S_k` is computed directly from the
`TaskMeasurement.duration()` (`end - start`) values that
`update_with_measurements` already receives on every monitor tick.
Two cumulative accumulators (`_completed_counts` and
`_completed_service_time_sums`) feed two independent per-cycle
samplers (`_consume_throughput_samples` and
`_consume_service_time_samples`) so the backlog-time pressure rate
and the Forced-Flow service demand stay decoupled. Stages whose
delta-count or delta-sum is non-positive in the current cycle
observe `math.nan` on the gauge and are excluded from the argmax
and the INFO log; they re-enter once their first non-empty
measurement batch lands.

The single per-cycle log line means an operator triaging a slow
pipeline does not need a Grafana dashboard to identify the
bottleneck for that cycle — `grep "bottleneck stage" scheduler.log`
is enough. The full historical series for trend ranking remains in
Prometheus.

## Knobs

None. The metric is auto-emitted with the rest of the per-cycle
Prometheus set; there is no toggle, threshold, or sampling knob.
This is intentional: a pure-observability gauge with bounded
cardinality has no operationally interesting tuning surface, and
forcing operators to opt-in would defeat the "ranked at a glance"
promise.

## See also

- [22 — Prometheus metrics catalogue](22-prometheus-metrics.md) —
  the full set of `xenna_scheduler_*` and `xenna_stage_*` gauges,
  their label cardinality, and the sampling cycle they share.
- [25 — Bottleneck decision integration](25-bottleneck-decision-integration.md) —
  how the EWMA-smoothed `D_k` value drives Phase C grow-priority
  ordering and Phase D shrink protection.
- [12 — Multi-target DAG growth](12-multi-target-dag-growth.md) —
  Phase C falls back to downstream-first DAG-depth ordering when
  the bottleneck gate is disengaged; the Forced-Flow framing in
  this doc is the formal justification for the bottleneck-first
  override on top of that ordering.
- Lazowska, Zahorjan, Graham, Sevcik (1984), *Quantitative System
  Performance: Computer System Analysis Using Queueing Network
  Models*, Chapter 3 — the original Forced Flow Law and the
  bottleneck identification it implies for separable queueing
  networks.
- Ghodsi, Zaharia, Hindman, Konwinski, Shenker, Stoica (2011),
  *Dominant Resource Fairness: Fair Allocation of Multiple
  Resource Types*, NSDI 2011 — the DRF allocation principle that
  this metric is the diagnostic substrate for, should the
  autoscaler later consume `D_k` for donor selection.
