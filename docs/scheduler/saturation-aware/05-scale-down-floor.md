# 05: Scale-down floor (anti-shrink)

## The problem

A transiently-starved downstream stage reads near-zero throughput. A throughput-only signal would shrink it, but if that stage runs an expensive model on the GPU, re-acquiring the worker later costs a full model reload (tens of seconds of GPU time). For a lull that lasted a few seconds, the shrink is a net loss. We need to **hold expensive capacity warm through transient lulls** while still **releasing it on a genuine, sustained upstream-bound phase**.

## What we do

The floor is a thin **release gate** over the capacity model, and it is a
**shrink-veto only**: `floor ≤ current workers`, so it never delays growth. It
consumes each stage's `w_sustain` (the bottleneck-matched hold target) and
decides, per stage, how far the solver may shrink this cycle.

```
  per stage, each cycle:

    stock still arriving  ─────────────▶  HOLD at stabilized min(w_sustain, workers)
                                          (trim an over-fed stage to its hold target,
                                           no further)

    no ready worker AND ≥ 1 queued batch ─▶  SATURATION HOLD at current workers
                                          (demonstrably under-provisioned; ignore a
                                           transiently-decayed w_sustain)

    lower hold target appears ───────────▶  SHRINK CONFIRM
                                          (must persist release_confirm_cycles before
                                           the floor follows it down)

    whole-chain stock drained for ───────▶  RELEASE to min_workers
    release_confirm_cycles                 (only once upstream work is truly gone)
```

![Timeline: the worker count holds at the floor through a transient lull while the solver's delete proposals are vetoed; after upstream truly drains, the floor steps down and the stage releases to min workers.](assets/05-shrink-veto.png)

*The floor is a lower bound on shrink: it vetoes the solver's deletes during a
transient lull (holding the stage warm) and steps down only after upstream work
has truly drained for a confirmation window.*

![Animated worker-count timeline where the solver repeatedly asks to delete workers during a lull, the floor holds the stage warm, and then releases workers only after drain confirmation.](assets/05-floor-release.gif)

*The gray line is the solver's delete ask; the teal line is the floor's answer.
Short lulls stay warm, while a confirmed drain lets the floor step down and
release workers.*

The key to "warm through a lull, release on a real bottleneck" is how
`w_sustain` moves: it is smoothed by an **asymmetric EWMA**, fast up (re-protect
quickly) and slow down (release reluctantly, at a uniform rate across stages).

- A **transient lull** does not last long enough for `w_sustain` to decay, so
  the floor holds the expensive stage warm.
- A **persistent upstream bottleneck** lets `w_sustain` decay toward the
  genuinely sustainable size, so the stage shrinks and frees resources for the
  real bottleneck.

Two extra guards handle cases the basic gate cannot:

- **Downstream guard.** While the current rate-source candidate is *upstream* of
  a stage and source-normalized stock is still in flight, that downstream stage
  is held at current workers: capacity is not donated while upstream work is
  still on the way, without reserving anything for a named stage
  (`protect_downstream_of` in `floor.py`).
- **Local saturation veto.** **Any** stage with **no ready worker** and **at
  least one queued batch** is demonstrably under-provisioned this cycle, so a
  `w_sustain` decayed by a transient `bottleneck_rate` dip cannot shrink it
  against its own backlog. It is held at current workers and self-clears the
  moment it drains (a worker frees, or the queue falls below one batch). This is a purely local check; it is not gated on the stage being the bottleneck.

> **What "ready worker" means.** A worker is **ready** when it is *not* holding
> an in-flight task slot - it is idle and free to pick up the next batch. The
> scheduler counts readiness per stage as
> `ready_workers = max(workers - inflight_slots, 0)`, where `inflight_slots` is
> the number of tasks the stage currently has dispatched (each occupies one
> worker). So `ready_workers == 0` means **every** worker is busy this cycle
> (`inflight_slots >= workers`), and the `max(..., 0)` clamp keeps a transient
> over-count from going negative (`scheduler.py`, `_cycle`). Readiness is a
> *this-cycle* occupancy signal, not a warmup or health check.

![A stage with all workers busy and at least one queued batch: the shrink is vetoed because the stage is demonstrably under-provisioned, and the veto self-clears when it drains.](assets/05-saturation-veto.png)

*Local saturation veto: any stage with no ready worker and a queued batch is
demonstrably under-provisioned, so a decayed `w_sustain` cannot shrink it against
its own backlog. It self-clears the moment it drains.*

The floor never classifies a stage by GPU fraction or warmup cost; it only
time-confirms lower hold targets so one-cycle integer-boundary dips do not
trigger delete/re-add churn. A confirmed shrink still happens; a transient one is
deferred (logged as `shrink_deferred`).

## Trade-offs

| Cost | Benefit |
|---|---|
| A stage stays warm for a few cycles after work genuinely stops. | A short lull never pays a model-reload cold-start cost. |
| Release waits for whole-chain stock to drain for a confirmation window. | A downstream stage is not torn down while upstream work is still in flight. |
| Asymmetric decay holds an over-fed stage above `w_sustain` briefly after a bottleneck shift. | No flapping: a one-cycle lower target cannot delete a worker. |

## Implementation pointer

- `floor.py`: `compute_floors`, the per-stage release gate, the asymmetric
  `w_sustain` EWMA, the downstream guard, and the local saturation veto.
- `chain.py`: whole-chain active-stock math (queue depth + pool-queued +
  in-flight, normalized to source units) used for the release decision.
- Decision snapshot fields: `floor`, `shrink_deferred`, `shrink_streak`,
  `pending_shrink_floor`, `releasing`.
- Config: `scale_down_release_cycles`, `scale_down_release_slowdown`
  (see `tuning.md`).
