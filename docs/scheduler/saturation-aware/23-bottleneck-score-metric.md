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

**Trade-off.** Pure observability — the autoscaler does not
consume `D_k` for any decision today. The cost is one extra
Prometheus gauge whose cardinality is bounded by the (static)
stage list per pipeline. The benefit is a single number per stage
that ranks stages by bottleneck candidacy and prevents the "scale
the busy-looking stage" mistake. The same gauge is the diagnostic
substrate for a future Dominant Resource Fairness donor-selection
pass (Ghodsi, Zaharia, Hindman, Konwinski, Shenker, Stoica 2011,
NSDI) — the cardinality envelope is paid once and reused.

## How it works

```
                       autoscale() cycle for pipeline P
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  read pool_stats per stage                   │
              │  (xenna ActorPoolStats; same sampler that    │
              │   drives pipeline_*_metrics)                 │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  for each stage k:                           │
              │    if processing_speed_tasks_per_second is   │
              │       None:  skip (no service-time sample    │
              │              yet — e.g. first cycle)        │
              │    else:                                     │
              │       S_k = 1 / processing_speed_tasks_...   │
              │       D_k = V_k * S_k         (V_k = 1)      │
              │       gauge.set(D_k, {stage: k.name,         │
              │                       pipeline: P.name})     │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  identify bottleneck = argmax_k D_k          │
              │  ─ logger.info(                              │
              │       "[scheduler] bottleneck stage = "      │
              │       f"{name!r} D={D:.3f}s "                │
              │       f"throughput_bound={1/D:.3f} tasks/s") │
              │  ─ emitted once per cycle, INFO level        │
              └──────────────────────────────────────────────┘
```

The mean per-task service time `S_k` is already computed by the
per-stage sampler that publishes `pipeline_actor_process_time` —
the bottleneck-score gauge reads the same
`pool_stats.processing_speed_tasks_per_second` field on
[`ActorPoolStats`](../../../cosmos_xenna/ray_utils/monitoring.py)
and applies the Forced-Flow framing on top. Stages whose service
rate is still `None` (no completed task yet, e.g. cold start) are
skipped for the cycle and do not appear in the per-cycle log line;
they re-enter once their first sample lands.

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
  the full set of `xenna_*` and `pipeline_*` gauges, their label
  cardinality, and the sampling cycle they share.
- [12 — Multi-target DAG growth](12-multi-target-dag-growth.md) —
  Phase C grows stages **downstream-first** so a scarce placement
  slot lands at the bottleneck end first; the Forced-Flow framing
  in this doc is the formal justification for that ordering.
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
