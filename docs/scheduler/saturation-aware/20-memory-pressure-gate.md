# 20 — Memory Pressure Gate

## TL;DR

A **single cluster-level kill switch** on Phase C scale-up: when
Ray's object-store used fraction crosses
`memory_pressure_critical_threshold` (default `0.85`), every
stage's positive intent is frozen for the cycle. Phase A
(manual), Phase B (floor), and Phase D (shrink) keep running.
The gate is the only cluster-wide signal the autoscaler
consults — every other signal is per-stage — because
OOM-by-summation is an emergent property no per-stage observable
can see.

## Problem

The per-stage slot signal that drives the classifier is
**already size-aware**. A stage that ships large per-task
payloads holds its slots longer, the empty-slot ratio falls
accordingly, and the autoscaler reads it as `SATURATED`. The
per-task footprint never has to enter the model explicitly — the
slot signal absorbs it naturally through residence time.

What the slot signal **cannot** observe is the sum across
stages. Two stages may each respect their own slot contract
while their combined per-task footprints exceed the cluster's
object-store memory:

```
   ┌──────────────────────────┐    ┌──────────────────────────┐
   │ Stage A   SATURATED      │    │ Stage B   SATURATED      │
   │ task size  ≈ 500 MB      │    │ task size  ≈ 300 MB      │
   │ 20 workers × 2 slots     │    │ 30 workers × 2 slots     │
   │ ≈ 20 GB resident         │    │ ≈ 18 GB resident         │
   └──────────────────────────┘    └──────────────────────────┘
                  Σ ≈ 38 GB  ───▶  object store ≈ 40 GB
                  next Phase C cycle pushes both above the line ─▶  OOM
```

Each stage's classifier dutifully votes `SATURATED → grow`. Each
individual decision is locally correct. Their **sum** is not.
The autoscaler needs a cluster-level circuit breaker that does
not require any single stage to detect a condition the per-stage
view is structurally unable to see.

## Decision

Adopt a **single cluster-level kill switch** on Phase C, gated
on Ray's reported object-store usage. The gate produces one
boolean per cycle (`phase_c_frozen = used_fraction >= threshold`)
that Phase C consults before emitting any positive intent. The
comparison uses `>=` (not strict `>`) so a configured
`threshold = 1.0` - the closed-right end of the validator range
`(0.0, 1.0]` - still fires when the object store reports fully
saturated;

```
   object-store used fraction (over autoscale cycles)

        1.00 ┤
             │              ▓▓▓▓▓▓▓▓▓
        0.90 ┤            ▓▓▓▓▓▓▓▓▓▓▓▓
             │           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
        0.85 ┤━━━━━━━━━━▓▓ threshold ▓▓━━━━━━━━━━━━━━  memory_pressure_
             │         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 critical_threshold
        0.70 ┤        ░▓▓▓▓▓▓▓▓▓▓▓▓▓▓░
             │      ░░░               ░░░
        0.50 ┤    ░░                     ░░░
             │  ░░                          ░░░░
        0.00 ┴──┴───────────────────────────────────►  time (cycles)

                 ◄── below ──►◄── above ──►◄── below ──►
                              gate fires

       Phase A   manual                ┃  RUNS  ┃  RUNS  ┃  RUNS  ┃
       Phase B   floor enforcement     ┃  RUNS  ┃  RUNS  ┃  RUNS  ┃
       Phase C   saturation-driven up  ┃  RUNS  ┃▓FROZEN▓┃  RUNS  ┃
       Phase D   saturation-driven dn  ┃  RUNS  ┃  RUNS  ┃  RUNS  ┃
```

**Phase B still runs.** The floor is **structural**, not
operational. A stage at zero workers produces no slot signal at
all and never recovers on its own; the floor enforcer is the
only path back. Freezing the floor on memory pressure would
convert a transient pressure spike into a permanent stuck stage.
Floors are intentionally small (often one worker), so the
marginal pressure they add is negligible compared with the
contribution of already-provisioned stages.

**Phase D still runs.** Shrinking **relieves** pressure: every
worker the scheduler removes drops its slot-resident payloads
and returns object-store bytes to the cluster. Freezing Phase D
would deadlock the recovery path — the gate clears only when
some workload completes or the autoscaler trims an
over-provisioned stage. The asymmetry is deliberate: the gate
suppresses adding pressure, never relieving it.

**Trade-off.** A pressure spike caused by a transient upstream
burst freezes legitimate Phase C growth on innocent stages for
one or more cycles. The cost is a brief throughput delay; the
alternative is an OOM-driven Ray worker death and the
multi-minute placement-group rebuild that follows. The default
`0.85` is calibrated so healthy clusters never trigger and only
clusters genuinely approaching OOM pay the freeze cost.

**Why a single threshold, not a band.** Phase C already
debounces — the asymmetric streak counters require multiple
cycles of sustained `SATURATED` before any positive intent
fires (see [07 — Streak stabilization](07-streak-stabilization.md)).
Layering a hysteresis band on top of an already-debounced signal
would delay both the freeze and the unfreeze. A single threshold
lets the gate react on the cycle it observes the crossing in
either direction.

## How it works

```
                       autoscale(time, problem_state)
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Pre-flight                                  │
              │  ─ refresh cluster-memory gauge if cache is  │
              │    older than memory_pressure_polling_       │
              │    interval_s                                │
              │  ─ used_fraction = used / total              │
              │  ─ phase_c_frozen = enable_memory_pressure_  │
              │     gate  AND  used_fraction >= threshold    │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase A — Manual                   (RUNS)   │
              │  Phase B — Floor enforcement        (RUNS)   │
              │           reason: structural; a stage at 0   │
              │           workers can never recover          │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase C — Saturation-driven grow            │
              │  ┌───────────────────────────────────────┐   │
              │  │ if phase_c_frozen:                    │   │
              │  │     skip every positive intent        │   │
              │  │     emit one INFO log naming the gate │   │
              │  │ else:                                 │   │
              │  │     normal DAG-priority growth loop   │   │
              │  └───────────────────────────────────────┘   │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase D — Saturation-driven shrink (RUNS)   │
              │           reason: shrinking relieves the     │
              │           pressure; freezing it would        │
              │           deadlock recovery                  │
              └──────────────────────────────────────────────┘
```

The poll-and-cache layer is deliberately cheap. A direct Ray
resource query inside `autoscale()` would add RPC latency to
every cycle; the cached gauge is refreshed at most every
`memory_pressure_polling_interval_s` seconds and read O(1) by
the planner. The cluster object-store usage is read from Ray's
own resource reporting (see
[Ray object store](https://docs.ray.io/en/latest/ray-core/objects.html)).

The gate is best understood as a layered defense rather than a
replacement for the inter-stage backpressure that the streaming
executor already runs outside the autoscaler. Per-stage
`max_queued` caps on the executor prevent any single fast stage
from filling the cluster on its own; the memory-pressure gate
sits one layer out, at the autoscaler, and catches the case where
every stage individually respects its cap but their **sum** does
not.

## Knobs

All fields live on `SaturationAwareConfig` in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Field                                  | Default | Effect |
|---|---|---|
| `enable_memory_pressure_gate`          | `True`  | Master switch. When `False`, Phase C never freezes regardless of memory pressure. |
| `memory_pressure_critical_threshold`   | `0.85`  | Fraction in `(0.0, 1.0]`. When `used_fraction >= threshold`, Phase C is frozen for the cycle (the closed-right `1.0` is meaningful). |
| `memory_pressure_polling_interval_s`   | `5.0`   | Minimum interval (seconds) between Ray cluster-memory queries; the cached gauge is reused inside this window. |

Push `memory_pressure_critical_threshold` toward `0.95` on
clusters with abundant object-store headroom and a tolerance for
the occasional OOM-driven restart. Pull it toward `0.70` on
clusters running close to their object-store budget where a
restart is more expensive than a throughput pause. The
`(0.0, 1.0]` validator on the field rejects values outside the
half-open interval at construction time.

## See also

- [00 — Per-cycle overview](00-overview.md) — where Phase C
  sits in the four-phase cycle.
- [04 — Per-cycle pipeline](04-per-cycle-pipeline.md) — how
  Phase C and Phase D are dispatched and what they consume.
- [16 — Hard caps and floors](16-hard-caps-and-floors.md) — the
  per-stage / per-node floors that Phase B enforces and that the
  memory gate deliberately does not override.
- [21 — Allocation-error tolerance](21-allocation-error-tolerance.md)
  — the complementary recovery path when a placement actually
  fails despite the gate.
- [Ray object store](https://docs.ray.io/en/latest/ray-core/objects.html)
  — the underlying memory pool the gate watches.
