# 00 — Per-Cycle Overview

## TL;DR

Every autoscale cycle the scheduler runs four phases —
**A** (manual), **B** (floor), **C** (saturation-driven grow),
**D** (saturation-driven shrink) — against a per-cycle planning
context, then freezes the staged plan into a `Solution` after
crossing structural invariant checks between phases.

## Problem

A streaming autoscaler that simply emits "current best worker
counts" each cycle has three failure modes seen in practice:

- Manual operator intent (per-stage `requested_num_workers`)
  collides with classifier-driven decisions; whichever runs last
  wins.
- Floor enforcement (`min_workers`, `min_workers_per_node`)
  collides with cluster capacity; a stage gets stuck at zero
  workers because the planner has nothing to give it.
- A bug in any single decision step (NaN ratio, negative count,
  off-by-one in floor math) silently emits a corrupted plan; the
  pipeline drifts before anyone notices.

## Decision

Treat one autoscale call as a **fixed four-phase pipeline** with
**per-phase invariants** instead of a free-form decision blob.

- **Phase A** runs manual operator intent first. Manual deletes
  free placement slots that Phase B / Phase C can reuse this
  cycle, so manual changes converge in one cycle, not two.
- **Phase B** enforces per-stage and per-node floors next, before
  any classifier reasoning, because zero-worker stages have no
  slot signal to classify against. Phase B is the only place that
  may invoke cross-stage donor logic in floor mode.
- **Phase C** (grow) and **Phase D** (shrink) run last, separated
  so a single cycle never both grows and shrinks the same stage.
  Phase C consumes positive intents from the classifier;
  Phase D consumes negative intents and applies the
  consolidation-aware victim ordering.
- **Invariants** run at every phase boundary. Any structural
  violation raises `SchedulerInvariantError` and refuses to emit
  a plan. The trade-off: a few microseconds per cycle for
  fail-loud behaviour on internal corruption.

## How it works

```
                       autoscale(time, problem_state)
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Pre-flight                                  │
              │  ─ shape check (problem ↔ problem_state)     │
              │  ─ cycle counter += 1                        │
              │  ─ refresh per-worker READY timestamps       │
              │  ─ snapshot stuck-plan counters              │
              │  ─ regime detector + lift effective K        │
              │  ─ resolve auto-thresholds (lazy, per stage) │
              │  ─ build AutoscalePlanContext (FGD bridge)   │
              │  ─ build donor-warmup-excluded set           │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase A — Manual operator intent            │
              │  ─ delete excess workers (manual stages)     │
              │  ─ add up to requested_num_workers           │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓
              ┌──────────────────────────────────────────────┐
              │  Phase B — Floor enforcement                 │
              │  ─ ensure current_workers ≥ floor for every  │
              │    non-finished stage                        │
              │  ─ on cluster-full failure: floor-mode       │
              │    cross-stage donor (youngest-first)        │
              │  ─ raises RuntimeError after grace exhausted │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓
              ┌──────────────────────────────────────────────┐
              │  Compute per-stage intent deltas             │
              │  ─ aggregate slot signals (warmup-excluded)  │
              │  ─ EWMA smoothing                            │
              │  ─ classify into 5 zones (with hysteresis)   │
              │  ─ asymmetric streak counters                │
              │  ─ trust gate (min_data_points)              │
              │  ─ stabilization-window consensus            │
              │  ─ growth-mode state machine                 │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase C — Saturation-driven grow            │
              │  ─ DAG-priority loop over positive intents   │
              │  ─ saturation-mode cross-stage donor when    │
              │    cluster is full but a downstream stage    │
              │    needs to grow                             │
              │  ─ stuck-plan counters tick                  │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓ + NaN check
              ┌──────────────────────────────────────────────┐
              │  Phase D — Saturation-driven shrink          │
              │  ─ apply negative intents                    │
              │  ─ idle-first + GPU-consolidation ordering   │
              │  ─ skip workers inside donor-warmup grace    │
              │  ─ never drop below per-stage / per-node     │
              │    floor                                     │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓ + floor check + monotonicity
              ┌──────────────────────────────────────────────┐
              │  Freeze plan → Solution                      │
              │  ─ ctx.into_solution()                       │
              │  ─ persist worker ages (cycle-counted)       │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
                                 Solution
```

The whole flow is implemented in
[`SaturationAwareScheduler.autoscale`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py).
Each phase is a private method on the scheduler; helper modules
under
[`scheduling_py/`](../../../cosmos_xenna/pipelines/private/scheduling_py/)
own the pure-function decision primitives (classifier, EWMA,
streak, growth mode, donor selection, scale-down ordering).

## Knobs

The cycle structure itself is not configurable. Knobs that affect
*how each phase behaves* live on `SaturationAwareConfig` (cluster)
and `SaturationAwareStageConfig` (per-stage); see the linked
feature docs.

| Phase | Key knobs (linked to feature docs) |
|---|---|
| Pre-flight | [auto-derived thresholds](08-auto-derived-thresholds.md), [regime-aware lift](09-regime-aware-aggressiveness.md) |
| Phase A | none (operator intent only) |
| Phase B | [hard caps and floors](16-hard-caps-and-floors.md), [cross-stage donor](13-cross-stage-donor.md) |
| Intent compute | [state classifier](05-state-classifier.md), [backlog-time (PLANNED)](06-backlog-time-signal.md), [streak stabilization](07-streak-stabilization.md), [slow-start](10-slow-start-mechanisms.md), [growth mode](11-growth-mode-state-machine.md) |
| Phase C | [multi-target DAG growth](12-multi-target-dag-growth.md), [cross-stage donor](13-cross-stage-donor.md), [worker age tracking](14-worker-age-tracking.md) |
| Phase D | [idle-first scale-down](15-idle-first-scale-down.md) |
| Invariants | [phase invariants](19-phase-invariants.md) |
| Cycle | [loop watchdog](18-loop-watchdog.md), [memory-pressure gate](20-memory-pressure-gate.md), [allocation-error tolerance](21-allocation-error-tolerance.md) |

## See also

- [01 — Scheduler selection](01-scheduler-selection.md) —
  how a pipeline opts into this scheduler.
- [02 — Configuration model](02-configuration-model.md) — the
  config classes and resolver order.
- [03 — Planning context](03-planning-context.md) — the Rust
  planner bridge that Phases A–D mutate.
- [04 — Per-cycle pipeline](04-per-cycle-pipeline.md) —
  drill-down on what each phase actually does.
- [19 — Phase invariants](19-phase-invariants.md) — what each
  invariant check enforces.
