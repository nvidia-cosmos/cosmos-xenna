# 22 — Prometheus Metrics Catalogue

## TL;DR

Scheduler observability is organised as a stable **Prometheus
catalogue with five families** — **cycle**, **per-stage state**,
**decision counters**, **safety**, and **configuration**. Every
metric is keyed by a small, bounded label set (`pipeline`, `stage`,
`kind`) so a 100-stage pipeline produces a few hundred series, not
tens of thousands. The **metric names are the contract**: dashboards
and alerts bind to the names and labels, the scheduler is free to
re-implement the decision logic behind them.

## Problem

When scheduler observability grows organically, three failure modes
emerge before a dashboard is ever built:

- **Ad-hoc gauges.** Each new decision point (regime detector,
  stuck-plan counter, donor transfer) ships its own `Gauge` under
  whatever name fits the file it lives in. There is no single answer
  to "is the autoscaler healthy right now?" — the operator has to
  know every gauge name to assemble one.
- **Label cardinality blow-up.** A naive "one series per per-stage
  decision per cycle" emission policy multiplies stages × decision
  types × free-form reason strings. A 100-stage pipeline with five
  decision types and a free-form `reason` label produces tens of
  thousands of series, blowing past Prometheus retention budgets.
- **Refactor churn breaks dashboards.** When a scheduler refactor
  renames an internal counter or splits one decision into two, every
  dashboard and alert that bound to the old name breaks silently —
  the panel goes empty, the alert never fires, and no one notices
  until production has already drifted.

## Decision

Adopt a **stable five-family Prometheus catalogue** with bounded
label cardinality and a name-stability contract.

- **Five families, each scoped to one operator decision.** Cycle
  metrics answer "is throughput healthy?", per-stage-state "where
  should I tune?", decision counters "is the autoscaler making
  decisions?", safety "did a kill-switch fire?", configuration
  "what flags are active?". An operator builds one dashboard per
  family without cross-referencing.
- **Bounded label set.** Every metric uses at most three labels
  from a closed vocabulary: `pipeline` (one per running job),
  `stage` (one per pipeline stage), `kind` (a small enum, e.g.
  `acq | trk | hold` for growth mode, `sat | act | over` for
  thresholds). No free-form `reason`, no per-worker label, no
  per-cycle label. Per-stage gauges add `O(num_stages)` series per
  pipeline — typical pipelines have under 20 stages, well inside
  Prometheus limits; pipelines exceeding 100 stages aggregate via
  [Prometheus recording rules](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)
  rather than adding more labels.
- **Names are the contract; logic is the implementation.** The
  scheduler may rewrite a phase or replace the classifier
  wholesale, but it must keep emitting metrics like
  `xenna_scheduler_cycle_duration_seconds` with the same labels and
  semantics. Dashboards survive scheduler refactors because they
  bind to names, not call sites.
- **Two prefixes.** `xenna_scheduler_*` for cluster-wide series
  (one per pipeline), `xenna_stage_*` for per-stage series. The
  split lets PromQL aggregate cleanly without joining on
  `stage=""`.

```
                          xenna_*  catalogue
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌──────────┐     ┌──────────────┐     ┌──────────┐
        │  Cycle   │     │ Per-stage    │     │  Safety  │
        │          │     │ state        │     │          │
        ├──────────┤     ├──────────────┤     ├──────────┤
        │ duration │     │ bottleneck_  │     │ mem_     │
        │ phase_   │     │  score       │     │  pressure│
        │  duration│     │ heterogen.   │     │ object_  │
        │          │     │  ratio       │     │  store_  │
        │          │     │ stuck_plan_  │     │  used_   │
        │          │     │  active      │     │  fraction│
        │          │     │ stuck_plan_  │     │ alloc_   │
        │          │     │  cycles_     │     │  failures│
        │          │     │  total       │     │  _total  │
        └──────────┘     └──────────────┘     └──────────┘
         O(1) per         O(num_stages)        O(1) cluster
         pipeline         per pipeline         per pipeline
                          + cluster-wide
                          ratio
```

**Trade-off.** A fixed five-family catalogue is less flexible than
ad-hoc emission — a new decision type cannot just "drop a gauge". It
must claim a family, pick a name that matches the family's prefix
convention, and prove its label set is bounded. The cost is one
extra design pass per metric; the benefit is that operators can
read the dashboard without learning the scheduler's internal names.

## How it works

Each family carries the same metric type vocabulary — gauge for
current state, counter for monotonically increasing decisions,
histogram for latency distributions — with one metric per family-
local question.

| Family | Metric | Type | Labels | What it tells the operator |
|---|---|---|---|---|
| Cycle | `xenna_scheduler_cycle_duration_seconds` | histogram | `pipeline` | Cycle wall-clock distribution; tail tracks loop watchdog |
| Cycle | `xenna_scheduler_cycle_phase_duration_seconds` | histogram | `pipeline`, `phase` | Which of `pre_phase_setup` / `phase_a` / `phase_b` / `intent` / `phase_c` / `phase_d` / `invariants` / `into_solution` dominated cycle time |
| Per-stage state | `xenna_stage_bottleneck_score` | gauge | `pipeline`, `stage` | Forced-Flow-Law bottleneck — see [23](23-bottleneck-score-metric.md) |
| Per-stage state | `xenna_scheduler_cluster_heterogeneity_ratio` | gauge | `pipeline` | `max_k D_k / min_k D_k` across stages with finite service demand |
| Per-stage state | `xenna_scheduler_stuck_plan_active` | gauge (0/1) | `pipeline`, `stage` | 1 when a stage's Phase C grow has been stuck above the detection threshold |
| Per-stage state | `xenna_scheduler_stuck_plan_cycles_total` | counter | `pipeline`, `stage` | Total cycles a stage has been stuck above the detection threshold |
| Safety | `xenna_scheduler_memory_pressure_active` | gauge (0/1) | `pipeline` | Memory-pressure gate engaged this cycle |
| Safety | `xenna_scheduler_cluster_object_store_used_fraction` | gauge | `pipeline` | Last polled Ray object-store used fraction (`0.0`-`1.0`) |
| Safety | `xenna_scheduler_allocation_failures_total` | counter | `pipeline`, `stage` | Phase C `try_add_worker` exceptions absorbed by the defense layer |

Cycle metrics are observed once per `autoscale()` cycle: the
`cycle_duration_seconds` histogram in the
[loop watchdog](18-loop-watchdog.md) ctxmgr, the
`cycle_phase_duration_seconds` histogram in the per-phase timer
inside `SaturationAwareScheduler.autoscale`. Per-stage state metrics
update at their own cadence: `bottleneck_score` and the
heterogeneity ratio fire at the end of every cycle from the
[bottleneck score helper](23-bottleneck-score-metric.md);
`stuck_plan_active` and `stuck_plan_cycles_total` update each time
the per-stage `_stuck_plan_counters` mutates (driven by
`_set_stuck_plan_counter` in `saturation_aware.py`). Safety metrics
update opportunistically: the memory-pressure gauges refresh on
every `is_pressure_active()` call (whether a cache hit or a fresh
Ray poll), and the allocation-failures counter increments only on
the absorbed-exception path of `_try_add_worker_with_defense`.

## Knobs

Metric **names** are not knobs — they are part of the catalogue
contract and operators must be able to depend on them. The
producing modules pin every metric name as a module-level constant
(`CYCLE_DURATION_METRIC`, `STUCK_PLAN_ACTIVE_METRIC`,
`MEMORY_PRESSURE_ACTIVE_METRIC`, etc.) so a refactor that renames a
metric breaks `from ... import` rather than silently breaking
dashboards.

Pipelines exceeding ~100 stages should not raise the label budget;
instead, pre-aggregate per-stage gauges via Prometheus recording
rules (`sum`, `max`, `topk(5, ...)` by `pipeline`) and dashboard the
recording-rule output. Raw per-stage series remain available for
drill-down without being the default panel source.

## See also

- [18 — Loop watchdog](18-loop-watchdog.md) — produces the
  `xenna_scheduler_cycle_duration_seconds` histogram tail that the
  cycle family alerts on, and the WARN log that complements it.
- [23 — Bottleneck score metric](23-bottleneck-score-metric.md) —
  the Forced-Flow-Law gauge in the per-stage-state family.
- [24 — Structured logging](24-structured-logging.md) — the
  per-decision INFO log contract that complements the decision
  counters with human-readable context.
- [`cosmos-xenna/cosmos_xenna/pipelines/private/scheduling_py/`](../../../cosmos_xenna/pipelines/private/scheduling_py/)
  — the modules that own the metric handles
  (`loop_watchdog.py`, `bottleneck.py`, `stuck_plan.py`,
  `memory_pressure.py`, `allocation_failures.py`, plus the
  per-phase histogram in `saturation_aware.py`).
