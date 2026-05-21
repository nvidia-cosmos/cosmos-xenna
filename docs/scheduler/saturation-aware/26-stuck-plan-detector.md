# 26 — Stuck-Plan Detector

## TL;DR

`_stuck_plan_counters` is a raw per-stage integer that ticks on
every Phase C cycle where the grow intent went partly unsatisfied
(see [21 — Allocation error tolerance](21-allocation-error-tolerance.md)).
On its own it is just a counter — operators have to scrape the
log stream and reconstruct each stage's history to know whether a
stage is genuinely stuck. The
[`StuckPlanDetector`](../../../cosmos_xenna/pipelines/private/scheduling_py/stuck_plan.py)
turns the counter into two **operator-facing** signals:

1. **One INFO log per stuck episode** — fires once when the counter
   crosses `stuck_plan_detection_cycles` and once again when the
   stage recovers (counter resets to `0`). Per-cycle Phase C
   WARNs are kept for diagnosis; the INFO promotion is the line
   that page-dashboards key on.
2. **Two Prometheus series**, tagged by `(stage, pipeline)`:
   `xenna_scheduler_stuck_plan_active` (0/1 gauge — a single
   alert rule fires when any stage is `1`) and
   `xenna_scheduler_stuck_plan_cycles_total` (monotonic counter
   — Grafana plots show the exact stuck-time history per stage).

## Problem

Phase C grow can fail to place a worker for two very different
reasons that look identical from the autoscaler's perspective:

- **Transient cluster fragmentation.** A peer stage absorbed the
  free placement slots earlier in the same cycle's DAG-priority
  walk. Next cycle a Phase D shrink frees the slot and the grow
  succeeds. Counter ticks once, resets next cycle. **No operator
  action needed.**
- **Genuinely infeasible plan.** No worker shape on this cluster
  will ever fit the receiver: the GPU budget is too tight,
  the per-node cap binds, every donor is in cooldown, or the
  pipeline is structurally over-subscribed. Counter ticks every
  cycle, never resets. **Operator action needed.**

The two cases are indistinguishable from a single Phase C log
line — the WARN at cycle `N` for "transient fragmentation" looks
exactly like the WARN at cycle `N` for "structurally infeasible".
The distinguishing signal is **persistence over many cycles**:
keep counting and a real stuck plan eventually crosses any
sensible threshold, while a transient one repeatedly resets to
zero.

The naive approach — emit the WARN every cycle and let the
operator sort it out — produces a wall of identical log lines.
At `interval_s=10s` and a 6-hour pipeline run, a single stuck
stage emits ~2160 WARNs that all say the same thing. Operators
either suppress the warning class (and miss the next real one)
or write ad-hoc dashboards that re-derive "this stage has been
stuck for N cycles" from the counter — work the scheduler should
do once.

## Decision

Promote the per-stage Phase C signal at **two well-defined
transitions**, not on every cycle:

```
   stuck_cycles      counter  detector  log              gauge
   ────────────      ───────  ────────  ───              ─────
                  0    init      idle    (silent)             0
   crosses N (= threshold)  →   FIRED   INFO "stuck"          1
   continues stuck (N+1, ...)   FIRED   (silent)              1
                            ↓
   resets to 0 (recovery)   →   ARMED   INFO "recovered"      0
   crosses N again          →   FIRED   INFO "stuck" (again)  1
```

The latch lives on a per-instance
[`StuckPlanDetector`](../../../cosmos_xenna/pipelines/private/scheduling_py/stuck_plan.py).
It is keyed by `(stage_name, pipeline_name)` so each label this
detector ever raised on the gauge can be re-zeroed when the
scheduler is re-set up. Without the pipeline tag in the key,
`reset()` could not restore the previously-raised gauge to `0.0`
and a new run would inherit a stuck-at-1 alert from a prior run
on the same process.

```
                       ┌────────────────────────────────┐
                       │  Phase C grow loop             │
                       │  ──────────────────────────    │
                       │  added < intent on stage k:    │
                       │     _stuck_plan_counters[k]++  │
                       │  added == intent or no intent: │
                       │     _stuck_plan_counters[k]=0  │
                       └────────────────────────────────┘
                                       │
                                       ▼  every mutation
                       ┌────────────────────────────────┐
                       │  _set_stuck_plan_counter(k, v) │
                       │     - writes the counter dict  │
                       │     - calls detector.update()  │
                       └────────────────────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │  StuckPlanDetector.update()    │
                       │  is_stuck = v >= threshold     │
                       │     gauge.set(1, tags)         │
                       │     counter.inc(tags)          │
                       │     if not was_fired:          │
                       │         INFO "stuck"; latch=1  │
                       │  not stuck:                    │
                       │     gauge.set(0, tags)         │
                       │     if was_fired and v == 0:   │
                       │         INFO "recovered";      │
                       │         latch=0                │
                       └────────────────────────────────┘
```

`reset()` runs from `SaturationAwareScheduler.setup()`. It walks
every `(stage, pipeline)` key the detector ever recorded and
emits one `gauge.set(0.0, tags=...)` per key before clearing the
latch dict, so a re-`setup()` cannot leave the gauge stuck at `1`
for any tag this detector raised. The Prometheus counter is
intentionally **not** touched on reset — counters are monotonic;
clearing them would violate Prometheus semantics and would also
delete the aggregate stuck-time history operators rely on for
post-incident analysis.

**Trade-off.** A stage that flaps in and out of the threshold
(e.g. counter goes `17 → 18 → 17 → 18 → ...`) will fire the INFO
log once and then stay silent until it actually recovers
(`counter == 0`). The flap is still visible on the gauge and on
the per-cycle WARN stream; the INFO is intentionally "edge
triggered, recovery only" so a single stuck episode never produces
more than two log lines.

## How it works

The detector is a 90-line module with one class and two methods.
The `update()` method is the only place the metrics fire; the
`reset()` method is the only place stale gauge labels are cleared.

```
   class StuckPlanDetector:
       _fired: dict[(stage_name, pipeline_name), bool]

       def update(stage_name, stuck_cycles, threshold_cycles,
                  last_intent, pipeline_name):
           # gauge / counter on every cycle:
           if is_stuck:  gauge.set(1); counter.inc()
           else:         gauge.set(0)
           # INFO log only on TRANSITIONS:
           if is_stuck and not was_fired: INFO "stuck"; latch=1
           if not is_stuck and was_fired and stuck_cycles==0:
                                          INFO "recovered"; latch=0

       def reset():
           for (stage, pipeline) in _fired:
               gauge.set(0, tags={stage, pipeline})
           _fired.clear()
```

`update()` is called from `_set_stuck_plan_counter` in
[`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
on every counter mutation — both increments and resets. Because
`_set_stuck_plan_counter` is the single source of truth for
counter writes, the detector cannot drift from the underlying
counter: every counter step is observed by the detector in the
same call.

## Knobs

The detector itself has no knobs. The single threshold lives on
the cluster config:

| Field | Class | Default | Effect |
|---|---|---|---|
| `stuck_plan_detection_cycles` | `SaturationAwareConfig` | `18` | Cycles of `_stuck_plan_counters[stage] >= threshold` before the detector promotes the per-cycle Phase C WARN to a one-shot INFO. At the default `interval_s=10s`, `18` cycles ≈ **3 minutes** of continuous stuck — long enough to ride out cluster fragmentation, short enough that a structural infeasibility is visible inside one shift. |

Tuning rules of thumb:

- **Short pipelines / aggressive growth** — lower toward `~6` so
  a 1-minute stuck episode promotes; the per-cycle WARN stream
  then carries the diagnosis.
- **Heterogeneous clusters with frequent fragmentation flaps** —
  raise toward `~30` so a 5-minute stuck episode is required;
  this trades alerting latency for fewer false-positive INFO logs
  on benign transients.
- **Never set below `floor_stuck_grace_cycles`** — Phase B's
  hard `RuntimeError` would fire before the operator-visible
  INFO, so the page line would be the crash, not the warning.

## See also

- [12 — Multi-target DAG growth](12-multi-target-dag-growth.md) —
  the Phase C grow loop that owns the `_stuck_plan_counters`
  increments / resets the detector consumes.
- [19 — Phase invariants](19-phase-invariants.md) —
  `check_stuck_plan_monotonicity` enforces the only legal
  counter transitions (`curr == 0` or `curr == prev + 1`); a
  Phase C bug that double-increments is caught before the plan
  reaches the actor pool.
- [21 — Allocation error tolerance](21-allocation-error-tolerance.md)
  — the policy that absorbs Phase C `None` returns, ticks the
  counter, and decides when to escalate to `RuntimeError`. The
  counter mechanics live there; the operator-facing detector
  lives here.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — the
  catalogue entries for `xenna_scheduler_stuck_plan_active` and
  `xenna_scheduler_stuck_plan_cycles_total`.
- [24 — Structured logging](24-structured-logging.md) — the
  shape of the "stuck plan" and "recovered" INFO lines produced
  by `update()`.
