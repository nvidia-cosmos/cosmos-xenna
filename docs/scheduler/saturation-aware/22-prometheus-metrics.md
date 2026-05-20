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
  scheduler may rewrite `_handle_grow` or replace the classifier
  wholesale, but it must keep emitting
  `xenna_scheduler_grow_decisions_total` with the same labels and
  monotonic semantics. Dashboards survive scheduler refactors
  because they bind to names, not call sites.
- **Two prefixes.** `xenna_scheduler_*` for cluster-wide series
  (one per pipeline), `xenna_stage_*` for per-stage series. The
  split lets PromQL aggregate cleanly without joining on
  `stage=""`.

```
                          xenna_*  catalogue
                                 │
        ┌───────────────┬────────┴────────┬───────────────┬───────────────┐
        ▼               ▼                 ▼               ▼               ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────────┐  ┌──────────┐  ┌──────────┐
  │  Cycle   │   │ Per-stage    │   │ Decision     │  │  Safety  │  │  Config  │
  │          │   │ state        │   │ counters     │  │          │  │          │
  ├──────────┤   ├──────────────┤   ├──────────────┤  ├──────────┤  ├──────────┤
  │ duration │   │ workers      │   │ grow_total   │  │ mem_     │  │ info     │
  │ phase_   │   │ zone         │   │ shrink_total │  │  pressure│  │ thresh_  │
  │  duration│   │ slots_empty_ │   │ donor_       │  │ cycle_   │  │  values  │
  │ overrun_ │   │  ratio_ewma  │   │  transfers_  │  │  overrun │  │ aggr_k   │
  │  total   │   │ streak{kind} │   │  total       │  │ stuck_   │  │ regime   │
  │ cycles_  │   │ bottleneck_  │   │ cluster_full_│  │  plan    │  │          │
  │  total   │   │  score       │   │  total       │  │ alloc_   │  │          │
  │          │   │ growth_mode  │   │              │  │  errors  │  │          │
  └──────────┘   └──────────────┘   └──────────────┘  └──────────┘  └──────────┘
   one series    O(num_stages)      O(num_stages) +    cluster-wide  small fixed
   per pipeline  per pipeline       O(donor pairs)     per pipeline  per pipeline
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

| Family | Example metric | Type | Labels | What it tells the operator |
|---|---|---|---|---|
| Cycle | `xenna_scheduler_cycle_duration_seconds` | histogram | `pipeline` | Cycle wall-clock distribution; tail tracks loop watchdog |
| Cycle | `xenna_scheduler_cycle_phase_duration_seconds` | histogram | `pipeline`, `phase` | Which of A/B/C/D dominates cycle time |
| Per-stage state | `xenna_stage_zone` | gauge (enum) | `pipeline`, `stage` | Five-zone classifier output as an int |
| Per-stage state | `xenna_stage_slots_empty_ratio_ewma` | gauge | `pipeline`, `stage` | Smoothed saturation signal feeding the classifier |
| Per-stage state | `xenna_stage_streak_cycles` | gauge | `pipeline`, `stage`, `kind=sat\|over` | How long the stage has held its current intent |
| Per-stage state | `xenna_stage_bottleneck_score` | gauge | `pipeline`, `stage` | Forced-Flow-Law bottleneck — see [23](23-bottleneck-score-metric.md) |
| Decision counters | `xenna_scheduler_grow_decisions_total` | counter | `pipeline`, `stage` | Did Phase C actually grow this stage? |
| Decision counters | `xenna_scheduler_shrink_decisions_total` | counter | `pipeline`, `stage` | Did Phase D actually shrink this stage? |
| Decision counters | `xenna_scheduler_donor_transfers_total` | counter | `pipeline`, `donor`, `receiver` | Cross-stage donor protocol activity |
| Safety | `xenna_scheduler_memory_pressure_active` | gauge (0/1) | `pipeline` | Memory-pressure gate engaged this cycle |
| Safety | `xenna_scheduler_cycle_overrun_total` | counter | `pipeline` | Loop watchdog fired — see [18](18-loop-watchdog.md) |
| Safety | `xenna_scheduler_stuck_plan_cycles` | gauge | `pipeline` | Maximum of per-stage `_stuck_plan_counters` |
| Configuration | `xenna_scheduler_config_info` | gauge (=1) | `pipeline`, `key` | One series per active flag — survives label joins |
| Configuration | `xenna_stage_thresholds` | gauge | `pipeline`, `stage`, `kind=sat\|act\|over` | Resolved thresholds after auto-derivation |

Enum-encoded gauges (`zone`, `growth_mode`, `regime`) map enum
members to small non-negative integers; the mapping is documented
in the producing module's docstring and never renumbered once
emitted. Cycle and per-stage-state metrics are sampled at the end
of each `autoscale()` cycle (after Phase D and the
`Solution`-monotonicity invariant pass). Decision counters
increment inside the phase that took the decision. Safety metrics
update opportunistically — the loop watchdog increments
`xenna_scheduler_cycle_overrun_total` when a cycle exceeds
`cycle_time_warn_threshold * interval_s`; the memory-pressure gate
flips `xenna_scheduler_memory_pressure_active` on entry and exit.
Configuration metrics are set once at scheduler construction and
republished on regime transitions so a relabel never silently
drops a flag.

## Knobs

Metric **names** are not knobs — they are part of the catalogue
contract and operators must be able to depend on them.
Configuration **values** that surface as `xenna_scheduler_config_info`
and `xenna_stage_thresholds` series are the scheduler config knobs
that operators most often want to dashboard alongside behaviour. The
canonical surfaced fields live on `SaturationAwareConfig` and
`SaturationAwareStageConfig` in
[`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Family touchpoint | Config field surfaced as a metric series |
|---|---|
| `xenna_scheduler_config_info{key=...}` | each cluster boolean flag on `SaturationAwareConfig` (`enable_regime_aware_aggressiveness`, `enable_dag_priority_growth`, `enable_cross_stage_donor`, `enable_memory_pressure_gate`) is one series with `value = 1` |
| `xenna_scheduler_aggressiveness_k` | `SaturationAwareStageConfig.saturation_aggressiveness` (per-stage effective `K`, after regime lift) |
| `xenna_stage_thresholds{kind=sat\|act\|over}` | resolved `saturation_threshold`, `activation_threshold`, `over_provisioned_threshold` |

Pipelines exceeding ~100 stages should not raise the label budget;
instead, pre-aggregate per-stage gauges via Prometheus recording
rules (`sum`, `max`, `topk(5, ...)` by `pipeline`) and dashboard the
recording-rule output. Raw per-stage series remain available for
drill-down without being the default panel source.

## See also

- [18 — Loop watchdog](18-loop-watchdog.md) — produces
  `xenna_scheduler_cycle_overrun_total` and the cycle-duration
  histogram tail that the safety family alerts on.
- [23 — Bottleneck score metric](23-bottleneck-score-metric.md) —
  the Forced-Flow-Law gauge in the per-stage-state family.
- [24 — Structured logging](24-structured-logging.md) — the
  per-decision INFO log contract that complements the decision
  counters with human-readable context.
- [`cosmos-xenna/cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
  and
  [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py)
  — emission sites for the cycle, per-stage-state, and decision
  counter families, and the config fields surfaced through
  `xenna_scheduler_config_info` and `xenna_stage_thresholds`.
