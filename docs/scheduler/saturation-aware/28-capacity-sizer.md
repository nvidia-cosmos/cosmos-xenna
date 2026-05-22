# 28 — Capacity Sizer

The capacity sizer turns the per-stage signals (queue depth,
throughput, per-task service time, slot capacity) into a single
target worker count for the cycle. The autoscaler then decides
how to close the gap between the current worker count and that
target, bounded by the per-cycle blast-radius caps.

## Why it exists

Every cycle the classifier tells the scheduler *which way* a
stage should move (grow, hold, shrink). The sizer answers the
companion question: *by how much*. A bottlenecked stage with a
large input queue and a slow per-task service time needs many
new workers at once; a stage that is over-provisioned by a wide
margin should release several workers, not one. Picking the
right magnitude in a single decision keeps recovery time bounded
by the per-cycle blast radius, not by the number of cycles it
takes to drift toward capacity.

The capacity sizer answers the magnitude question with one
closed-form decision: *how many workers does this stage need
right now*, computed directly from queueing-theory inputs the
scheduler already smooths (queue depth, observed throughput,
EWMA-smoothed service demand, slot capacity, and the operator's
drain-time SLO).

## What the formula computes

```
                 queue_depth
target_rate = throughput + ------------------------     [tasks / s]
                            target_backlog_seconds

target_slots = ceil( target_rate × D_k / utilization_target )

                          target_slots
target_workers = ceil( ----------------- )
                       slots_per_worker
```

Each input has a single source:

| Input                       | Source                                                           | Purpose                                                                |
|---|---|---|
| `throughput`                | EWMA-smoothed `committed_throughput` per stage                   | Steady-state arrival rate the stage already serves                     |
| `queue_depth`               | Stage `input_queue_depth` from the planning context              | Backlog the sizer wants drained inside `target_backlog_seconds`        |
| `target_backlog_seconds`    | `SaturationAwareStageConfig.target_backlog_seconds` (default `30.0`) | Operator's drain-time SLO; same knob the pressure classifier uses  |
| `D_k`                       | EWMA-smoothed Forced-Flow-Law service demand                     | Per-task GPU/CPU work time, including queueing-theory busy time        |
| `utilization_target`        | `1 - saturation_threshold` (resolved per stage)                   | Headroom target; same threshold the slot-pin classifier already uses   |
| `slots_per_worker`          | Stage spec                                                        | Worker boundary to convert the slot target into a worker target        |

Putting the pieces together:

```
┌──────────────────┐      ┌───────────────────────┐      ┌──────────────────┐
│ EWMA throughput  │      │ EWMA service demand   │      │ queue depth      │
│ (Phase A signal) │      │ D_k (Forced Flow Law) │      │ (planning ctx)   │
└────────┬─────────┘      └──────────┬────────────┘      └────────┬─────────┘
         │                           │                            │
         │   ┌────────────────────┐  │                            │
         │   │ target_backlog_sec │◄─┼────────────────────────────┘
         │   │  (config knob)     │  │
         │   └─────────┬──────────┘  │
         │             │             │
         ▼             ▼             ▼
   ┌──────────────────────────────────────────────────────┐
   │ target_rate    = throughput + queue / target_backlog │
   └──────────────────────────┬───────────────────────────┘
                              │
                              │  utilization_target = 1 − saturation_threshold
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │ target_slots   = ⌈ target_rate · D_k / util ⌉        │
   └──────────────────────────┬───────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │ target_workers = ⌈ target_slots / slots_per_worker ⌉ │
   └──────────────────────────────────────────────────────┘
```

A single `ceil` runs at the slot boundary so fractional slots
round up exactly once; a second `ceil` then converts the integer
slot target into a worker count when stages pack multiple slots
into one worker.

## Cold-start fallback

When the EWMA `D_k` is not yet finite (cold start, fewer than
`min_data_points` valid samples, or a stage that never measured
service time), `compute_capacity_target_workers` returns `None`.
The caller then falls back to discrete sizing inside
`compute_delta`:

| Classifier state       | Cold-start delta |
|---|---|
| `SATURATED_CRITICAL`   | `+1`             |
| `SATURATED`            | `+1`             |
| `OVER_PROVISIONED`     | `-1`             |
| `NORMAL`               | `0`              |

The fallback keeps the very first cycle from multiplying the
worker count while still letting a decision happen. Once `D_k`
becomes finite, every following cycle uses the closed-form
target.

## Per-cycle blast-radius caps

The sizer reports the *unbounded ideal* target. Two caps then
bound how much of the gap a single cycle is allowed to close,
so a volatile signal cannot cause a runaway move:

```
                     ┌───────────────────┐
                     │  capacity target  │
                     └─────────┬─────────┘
                               │
                               ▼
                shortfall = max(0, target − current)
                excess    = max(0, current − target)
                               │
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
   ┌─────────────────────┐           ┌──────────────────────────┐
   │ grow magnitude:     │           │ shrink magnitude:        │
   │   min(shortfall,    │           │   min(excess,            │
   │     max_per_cycle)  │           │     ⌊cur · max_frac⌋)    │
   └─────────────────────┘           └──────────────────────────┘
```

| Knob                                | Default | Bounds                                                 |
|---|---|---|
| `aggressive_growth_max_per_cycle`   | `4`     | Hard cap on per-cycle additions.                       |
| `max_scale_down_fraction_per_cycle` | `0.05`  | Per-cycle removal fraction (5% of current by default). |

## How the growth-mode state machine fits

`compute_growth_mode_transition` still owns the per-stage
`ACQUIRING → TRACKING → HOLD` lifecycle. After the sizer change,
the mode is consumed by `compute_delta` as a binary gate rather
than a magnitude shaper:

| Mode        | `SATURATED` grow gate | `SATURATED_CRITICAL` grow gate |
|---|---|---|
| `ACQUIRING` | allowed               | allowed                        |
| `TRACKING`  | allowed               | allowed                        |
| `HOLD`      | blocked (`delta = 0`) | allowed                        |

`HOLD` is therefore the only mode that affects sizing today, and
it acts only on the `SATURATED` zone — true bursts
(`SATURATED_CRITICAL`) still grow. See
[11 — Growth-mode state machine](11-growth-mode-state-machine.md).

## Operator-facing knobs

Sizing is driven by one drain-time SLO and two blast-radius
caps; HOLD's grow gate has its own kill switch:

| Knob                                | Default | Role                                                                                  |
|---|---|---|
| `target_backlog_seconds`            | `30.0`  | Drain-time SLO. Lower = larger capacity target for the same queue (more aggressive).  |
| `aggressive_growth_max_per_cycle`   | `4`     | Per-cycle grow cap. Bounds blast radius when the capacity gap is large.               |
| `max_scale_down_fraction_per_cycle` | `0.05`  | Per-cycle shrink fraction. Bounds blast radius when the capacity excess is large.     |
| `enable_growth_mode_state_machine`  | `True`  | Kill switch for HOLD's binary block on `SATURATED` grow during stabilization windows. |

A workload-specific tuning playbook lives in
[`tuning.md`](tuning.md). The most common adjustment is
`target_backlog_seconds`: lowering it shortens the drain-time
SLO and produces a larger capacity target for the same queue,
raising it does the opposite.

## Why the formula

Three queueing-theory laws (Conservation, Forced-Flow,
Utilisation) compose to produce the formula:

- **Drain term** comes from job-flow balance: completions over
  a window must equal arrivals plus the queue we want drained.
  Solving `Λ' · T = λ · T + queue_depth` with
  `T = target_backlog_seconds` gives
  `target_rate = throughput + queue_depth / target_backlog_seconds`.
- **Slot count** comes from the Utilisation Law applied to `m`
  parallel slots: `U = X · D_k / m`. Solving for `m` at a
  target utilisation `u_target` gives
  `m = target_rate · D_k / u_target`.
- **Per-task service demand `D_k`** is the per-task busy time
  at this stage. Each task visits the stage once, so the
  Forced-Flow visit count is `1` and `D_k = S_k`.

The classifier's `saturation_threshold` defines
`u_target = 1 − saturation_threshold` so the sizer's headroom
matches the slot-pin classifier's saturation boundary, and the
operator only tunes one knob.

## See also

- [05 — State classifier](05-state-classifier.md) — the four-zone
  classifier whose `saturation_threshold` defines the sizer's
  `utilization_target`.
- [06 — Backlog-time pressure signal](06-backlog-time-signal.md)
  — the same `target_backlog_seconds` knob calibrated for the
  classifier.
- [11 — Growth-mode state machine](11-growth-mode-state-machine.md)
  — the binary gate that composes with the sizer.
- [25 — Bottleneck decision integration](25-bottleneck-decision-integration.md)
  — the EWMA-smoothed `D_k` the sizer reads.
