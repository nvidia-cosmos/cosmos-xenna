# 06 — Backlog-Time Pressure Signal

## TL;DR

Saturation is fired only when the slot pin AND the smoothed compound
**pressure** scalar agree. The pressure factor combines the existing
empty-slot ratio with a normalised backlog-drain time (Little's Law
`W_q = queue / throughput` divided by `target_backlog_seconds`). The
pressure scalar is consumed by the classifier inside the existing
slot-ratio branches as a **demotion gate** — slot-pin SATURATED stays
SATURATED only when pressure is high; slot-pin OVER_PROVISIONED is
demoted to NORMAL when pressure is high (the queue is stuck elsewhere
and shrinking would amplify the bottleneck).

## Problem

The smoothed empty-slot ratio that
[`classify`](../../../cosmos_xenna/pipelines/private/scheduling_py/classifier.py)
 consumed alone is a single-axis observable. It answers
"are the worker slots busy right now?" but says nothing about whether
more workers would help. Two operational scenarios show why utilisation
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
Little's Law connects: `L = lambda * W`, where `L` is queue length,
`lambda` is arrival rate, and `W` is the per-task waiting time. They
tell complementary stories and either one alone misclassifies real
workloads.

## Decision

Use the **backlog-time pressure** approach: the existing slot-ratio
gate continues to select the classifier branch, and a single smoothed
**pressure** scalar acts as a **demotion gate** inside each branch.
Pressure is the multiplicative composition of the two observables:

```
    pressure = utilisation * normalized_backlog

    where:
      utilisation        = 1 - slots_empty_ratio_ewma
      normalized_backlog = min(W_q / target_backlog_seconds, BACKLOG_CAP)
      W_q                = input_queue_depth / observed_throughput
```

`BACKLOG_CAP = 3.0` is an upper clamp that guarantees pressure stays
finite (especially during cold-start when `observed_throughput == 0`
and `queue > 0`). Pressure is smoothed by an EWMA before reaching the
classifier so cycle-to-cycle noise on the throughput sample dampens.

The signal fires only when **both** factors are elevated: the
multiplication collapses the product to ≈0 whenever either utilisation
is low (idle slots) or normalised backlog is low (queue draining). The
single scalar surfaces this AND-criterion as one value the classifier
and dashboards can read directly.

```
                              utilisation
                  low (idle slots)            high (busy slots)
                ┌─────────────────────────┬──────────────────────────┐
  normalized   │      pressure ≈ 0       │  pressure HIGH (real     │
  backlog HIGH │  (queue stuck else-     │   saturation: both       │
  (queue stuck)│   where; growing this   │   factors agree)         │
                │   stage would not help) │                          │
                ├─────────────────────────┼──────────────────────────┤
  normalized   │      pressure ≈ 0       │  pressure ≈ 0           │
  backlog LOW  │  (genuine idle —         │  (transient burst —     │
  (queue       │   OVER_PROVISIONED       │   queue is draining;    │
  draining)    │   eligible for shrink)   │   warmup cost would      │
                │                         │   dominate)              │
                └─────────────────────────┴──────────────────────────┘
```

## Two-layer classifier behaviour

The slot-pin gate decides which branch a cycle enters; the pressure
gate then demotes that branch when the AND-criterion is not met.
Each slot-pin branch has its own pressure threshold:

| Slot-pin branch | Pressure threshold | If pressure exceeds | Otherwise |
|---|---|---|---|
| `SATURATED_CRITICAL` (`slots_empty < activation`) | `pressure_critical_threshold` (default `2.0`) | stays `SATURATED_CRITICAL` | falls through to `SATURATED` gate |
| `SATURATED` (`slots_empty < saturation`) | `pressure_saturation_threshold` (default `1.0`) | stays `SATURATED` | demoted to `NORMAL` |
| `OVER_PROVISIONED` (`slots_empty ≥ over_provisioned`) | `pressure_normal_threshold` (default `0.3`) | demoted to `NORMAL` (queue stuck downstream) | stays `OVER_PROVISIONED` |

The slot-pin gate continues to honour the existing **asymmetric
deadband** for hysteresis (saturation deadband, over-provisioned
deadband). The pressure gate carries no deadband because the EWMA
already smooths the raw composite signal.

**Trade-off.** The compound criterion is strictly more conservative
than utilisation alone: every cycle where the utilisation-only rule
would have fired `SATURATED` but the queue is draining (low
pressure), the compound rule emits `NORMAL` instead. False-positive
scale-ups drop; true-positive scale-ups on sustained pressure are
unchanged because sustained pressure builds queue depth by
construction.

## Throughput sample plumbing

The backlog-time pressure signal needs a per-cycle `observed_throughput`
sample. The streaming layer calls `update_with_measurements()` on
every monitor tick (cadence governed by `streaming.Autoscaler`,
typically much faster than `interval_s`); the SA scheduler accumulates
**`len(task_measurements)` per stage** under a `threading.Lock` and
the autoscale cycle consumes the per-cycle delta:

```
streaming.Autoscaler             SaturationAwareScheduler
─────────────────────             ────────────────────────
update_with_measurements ───►    self._completed_counts[stage] += len(...)

autoscale (per interval_s)  ───► _consume_throughput_samples(now_ts)
                                   for each stage:
                                     dcount = max(0, now - prev_count)
                                     dt     = now_ts - prev_ts
                                     sample = dcount / dt if dt > 0 else 0.0
                                 ───► run_per_stage_pipeline(observed_throughput_sample=sample)
```

Counting policy is **`len(task_measurements)`** rather than summing
`num_returns`. The matching drain-rate unit for `input_queue_depth`
is "completed stage tasks per second" (one queue drain per task), not
"produced output items per second" — a `flat_map`-style stage that
returns `N` items per task should still count one queue drain per
task to match the drain rate the queue depth is reported in.

## Cold-start handling

`compute_pressure` is a pure function with three branches:

```
queue == 0                              -> 0.0
throughput <= 0 AND queue > 0           -> utilisation * BACKLOG_CAP
throughput > 0  AND queue > 0           -> utilisation *
                                           min(W_q / target, BACKLOG_CAP)
```

Cold-start (no measurements yet but the queue already has work) maps
to the bounded `BACKLOG_CAP` rather than `+inf` arithmetic so the
pressure stays a finite scalar. This ensures the first autoscale cycle
after `setup()` still emits a SATURATED-eligible pressure when the
queue is non-empty.

When the slot signal itself is missing (zero ready actors, no prior
valid EWMA), `_resolve_classifier_signal` returns `None` and
`run_per_stage_pipeline` short-circuits **before** updating the
pressure EWMA. This is intentional: feeding `compute_pressure` with
a zero utilisation while the actual pool is in setup-phase quiescence
would corrupt the next valid cycle's demotion decision.

## Knobs

All on
[`SaturationAwareStageConfig`](../../../cosmos_xenna/pipelines/private/specs.py):

| Knob | Default | Effect |
|---|---|---|
| `target_backlog_seconds` | `30.0` | Operator-facing primary knob. Drain-time at which `normalized_backlog == 1.0`; the same knob also sets the capacity sizer's drain term (see [28 — Capacity sizer](28-capacity-sizer.md)). Higher = more conservative; lower = more aggressive. |
| `pressure_smoothing_level` | `0.20` | EWMA alpha on the composite pressure scalar. Lower = smoother (filters throughput noise); higher = more reactive. Bounded `(0.0, 1.0]`. |
| `pressure_critical_threshold` | `2.0` | Pressure above which a slot-pin `SATURATED_CRITICAL` actually fires. Strictly larger than `pressure_saturation_threshold`. |
| `pressure_saturation_threshold` | `1.0` | Pressure above which a slot-pin `SATURATED` actually fires. |
| `pressure_normal_threshold` | `0.3` | Pressure above which a slot-pin `OVER_PROVISIONED` is demoted to `NORMAL` (queue is stuck elsewhere). |

Cross-field invariants enforced at construction time:

```
pressure_critical_threshold > pressure_saturation_threshold > pressure_normal_threshold
pressure_critical_threshold ≤ BACKLOG_CAP   (= 3.0)
```

## See also

- [00 — Per-cycle overview](00-overview.md) — where the pressure
  gate sits inside the intent compute stage.
- [02 — Configuration model](02-configuration-model.md) — every
  per-stage knob including the six new pressure-gate fields.
- [05 — State classifier](05-state-classifier.md) — the four-zone
  decomposition with the slot-pin-then-pressure-demotion ordering.
- [07 — Streak stabilization](07-streak-stabilization.md) — the
  asymmetric streak counters that run after the classifier fires.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — the three
  pressure gauges (`xenna_stage_observed_throughput`,
  `xenna_stage_backlog_time`, `xenna_stage_pressure_ewma`) operators
  read to audit classifier decisions.
- [tuning.md](tuning.md) — workload-class tuning advice including
  `target_backlog_seconds` calibration.
