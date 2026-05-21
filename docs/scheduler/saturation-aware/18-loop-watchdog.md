# 18 — Loop Watchdog

## TL;DR

Each autoscale cycle measures its own wall-clock duration at the
top and bottom of `SaturationAwareScheduler.autoscale()`. When the
duration exceeds `cycle_time_warn_threshold * interval_s` (default
`0.5 * interval_s` = 5 s with the stock 10 s cycle), the scheduler
emits a `WARN` log and observes a Prometheus duration histogram so
a slow cycle becomes operationally visible **before** the next
cycle backs up.

The `cycle_time_warn_threshold` knob ships on
`SaturationAwareConfig` with a validated default of `0.5`. The
wall-clock measurement, histogram observation, and WARN log are
implemented in
[`scheduling_py/loop_watchdog.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/loop_watchdog.py)
as a `@contextmanager` that wraps every
`SaturationAwareScheduler.autoscale()` cycle.

## Problem

Without a watchdog, a slow cycle accumulates silently:

- The next `interval_s` tick still expires on time, but the
  previous `autoscale()` call has not returned yet — the run-loop
  rate limiter in
  [`streaming.py`](../../../cosmos_xenna/pipelines/private/streaming.py)
  defers `apply_autoscale_result_if_ready` until the in-flight
  future completes.
- The next snapshot is collected against an autoscaler that is
  still mid-decision; classifier zone history, EWMA values, streak
  counters, and worker-age timestamps are read out from a
  half-applied state.
- The second cycle sees stale signals and emits a plan that
  contradicts the first, magnifying flap.

The fault mode is invisible without instrumentation because every
output — `Solution`, log line, plan diff — is still well-formed.
Only the wall clock can detect that the cycle ran long.

A one-line debug print at the bottom of `autoscale()` would help
during a single debug session, but is not enough for production:

- The operator does not know what duration is "too slow" — the
  budget is a fraction of the cycle interval, not a fixed number
  of milliseconds, and rescales with `interval_s`.
- A print is invisible in a log stream of thousands of structured
  INFO lines per pipeline.
- A persistent metric is required so an alerting rule (p95, p99)
  can fire without modifying scheduler code.

## Decision

Measure cycle wall-clock duration at the top and bottom of
`autoscale()`, compare against `cycle_time_warn_threshold *
interval_s`, and on every cycle:

- **Always observe** a Prometheus duration histogram (every cycle,
  not only slow ones) so operators can build their own p95 / p99
  panels and alerting rules in Grafana without modifying scheduler
  code.
- **Emit a WARN log only when the threshold is exceeded**, carrying
  both the measured duration and the resolved threshold. An
  operator can grep autoscaler logs to find slow cycles without
  scraping metrics.

```
   one autoscale cycle, interval_s = 10 s
   threshold = cycle_time_warn_threshold * interval_s
             = 0.5 * 10 s = 5 s

   t = 0 s             5 s                10 s
     │  cycle budget   │  WARN line       │  next cycle tick
     │  (threshold)    │  crossed here    │  due here
     ▼                 ▼                  ▼
     ├─────────────────┼──────────────────┤
     │                                    │
     ├───┐                                │
     │ #1│ 1.8 s                          │
     ├───┘                                │
     │ │                                  │
     │ └─► histogram.observe(1.8)         │
     │     (no WARN; well under budget)   │
     │                                    │
     │                                    ├──────────────────┐
     │                                    │ #2  6.4 s        │
     │                                    ├──────────*───────┤
     │                                               │
     │                                               ├─► histogram.observe(6.4)
     │                                               └─► WARN:
     │                                                   "autoscale cycle
     │                                                    exceeded watchdog
     │                                                    threshold"
     │                                                   (duration=6.4 s,
     │                                                    threshold=5.0 s)
     │
     │  next tick fires while cycle #2 is still running;
     │  apply_autoscale_result_if_ready is deferred until #2 returns
```

**Trade-off.** A fixed-budget alerting rule embedded in the
scheduler would be tighter but inflexible: operators with different
SLOs would have to re-deploy. The histogram + threshold split keeps
the scheduler's responsibility narrow (observe and warn) and pushes
policy (alert thresholds, dashboards, paging) into the operator's
monitoring stack. The cost is one extra metric per cycle; the
histogram is bucketed and cheap.

**Default.** `cycle_time_warn_threshold = 0.5` is intentionally
conservative. A healthy cycle on a saturation-aware-sized pipeline
finishes well under half the cycle interval, leaving the other half
for the planner submit, the result-apply pass in
`Autoscaler.apply_autoscale_result_if_ready`, and any incidental
thread scheduling. Operators tune the threshold up only when real
cycles consistently approach the budget on a larger cluster, and
tune it down only after profiling shows the planner is the actual
bottleneck.

## How it works

The wall-clock samples bracket the entire cycle — the
shape-check, the per-cycle pre-flight (cycle counter, worker
ready-timestamp refresh, stuck-plan snapshot, regime-aware
aggressiveness update, threshold resolution, planner-context
build), all four phases (A / B / C / D), the invariant checks at
each phase boundary, and the final `ctx.into_solution()` freeze.
The duration therefore reflects every action the autoscaler takes
inside one cycle envelope.

```
   SaturationAwareScheduler.autoscale(time, problem_state)
                   │
                   ▼
   ┌──────────────────────────────────────────────┐
   │  t_start = time.perf_counter_ns()            │
   └──────────────────────────────────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────────────────┐
   │  Pre-flight + Phase A/B/C/D + invariants     │
   │  (see 00-overview.md and 04-per-cycle-       │
   │   pipeline.md)                               │
   └──────────────────────────────────────────────┘
                   │
                   ▼
   ┌──────────────────────────────────────────────┐
   │  duration_s = (perf_counter_ns()             │
   │                - t_start) / 1e9              │
   │  budget_s   = cfg.cycle_time_warn_threshold  │
   │               * cfg.interval_s               │
   │                                              │
   │  cycle_duration_histogram.observe(duration_s)│
   │                                              │
   │  if duration_s > budget_s:                   │
   │      logger.warning(                         │
   │          "autoscale cycle exceeded watchdog  │
   │           threshold ...")                    │
   └──────────────────────────────────────────────┘
```

The histogram observation is unconditional. Every cycle contributes
a sample regardless of whether the threshold was crossed; the
threshold gates only the WARN log. Histogram bucket boundaries are
chosen at scheduler init from `interval_s` so the buckets straddle
the default threshold without operator tuning.

`time.perf_counter_ns()` is the chosen clock source rather than
`time.time()`: it is monotonic, immune to wall-clock jumps, and
high-resolution on every supported platform, which matters because
a single cycle is sub-second on small pipelines and the comparison
is against a sub-`interval_s` threshold.

## Knobs

Both knobs live on `SaturationAwareConfig` in
[`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Field | Default | Effect |
|---|---|---|
| `cycle_time_warn_threshold` | `0.5` | Fraction of `interval_s` above which a cycle's wall-clock duration triggers the WARN log. Range `(0.0, 1.0]` enforced by an `attrs` validator. Higher = quieter logs; lower = earlier warnings. |
| `interval_s` | `10.0` | Cluster-wide cycle interval (seconds). Sets the absolute watchdog budget via `cycle_time_warn_threshold * interval_s` (default budget = 5 s). Changing `interval_s` rescales the watchdog budget automatically. |

The watchdog has no separate `enable_watchdog: bool` flag. To make
WARN logging less sensitive without disabling the histogram, set
`cycle_time_warn_threshold = 1.0`: WARN then fires only when a cycle
exceeds the full interval (`duration_s > interval_s`).

## See also

- [00 — Per-cycle overview](00-overview.md) — the four-phase
  cycle that the watchdog measures end-to-end.
- [04 — Per-cycle pipeline](04-per-cycle-pipeline.md) — the
  per-phase breakdown of where wall-clock time is actually spent
  inside one autoscale cycle.
- [19 — Phase invariants](19-phase-invariants.md) — the
  structural checks that run between phases; the watchdog
  complements them by guarding the cycle envelope rather than its
  internal state.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — the
  catalogue under which the cycle-duration histogram is
  registered.
