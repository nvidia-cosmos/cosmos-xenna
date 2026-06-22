# 05: Scale-down floor (anti-shrink)

## The problem

A transiently-starved downstream stage reads near-zero throughput. A throughput-only signal would shrink it, but if that stage runs an expensive model on the GPU, re-acquiring the worker later costs a full model reload (tens of seconds of GPU time). For a lull that lasted a few seconds, the shrink is a net loss. We need to **hold expensive capacity warm through transient lulls** while still **releasing it on a genuine, sustained upstream-bound phase**.

## How a stage shrinks

The floor only ever **shrinks** a stage (`floor <= workers`); growth is bounded
separately by the capacity target `w_target` ([01](01-capacity-model.md)) and the
cold-start ramp ([04](04-cold-start-ramp.md)), so a stage can never outgrow the
rate its slowest upstream stage can feed. Given that, a stage gets smaller in
exactly three ways:

1. **Shrink to `w_sustain`** (the normal path). As the bottleneck-matched hold
   target `w_sustain` decays under a sustained low feed rate, the floor follows
   it down to `min(w_sustain, workers)`. No demand from another stage is needed -
   this is how any stage returns to its sustainable size.
2. **Release to `min_workers`** (the drain path). Once the whole-chain
   at-or-upstream stock stays below one batch for `release_confirm_cycles`, the
   stage's work is genuinely done and it releases all the way to `min_workers`.
3. **Reclaim early** (the benefit-gated exception). A stage *downstream of the
   bottleneck* is normally held warm - pinned above `w_sustain` while upstream
   work is still in flight - so it is ready when the bottleneck drains. It is
   allowed to shrink to `w_sustain` *early* only when a blocked, growth-wanting
   *stage* (any grower, not just the bottleneck) needs the resource it would
   free, confirmed over `reclaim_confirm_cycles` cycles.

Paths 1 and 2 always apply; path 3 only decides whether a downstream stage's warm
pin is released *before* its work drains. The rest of this note details the gate
that implements all three.

Paths 1 and 3 size a stage against `w_sustain`, so they only apply once the stage
has a **trusted** target. A **cold** stage (no measured per-worker speed yet)
carries only a placeholder `w_sustain = min_workers` from the capacity model, so
the floor holds it at its current workers and leaves its growth to the cold-start
ramp ([04](04-cold-start-ramp.md)). Only path 2 (a genuine whole-chain drain)
shrinks a cold stage, so a still-warming stage is never torn down toward a target
the model has not measured.

![Infographic "how a stage shrinks": a STAGE box fans out through three teal arrows to three outcomes - "w_sustain decays" to SHRINK TO w_sustain (normal path), "upstream work drains" to RELEASE TO min workers (work is done), and "a blocked stage needs it" (with an "A GROWER needs CPU" tag and a "confirmed N cycles" pill) to RECLAIM EARLY to w_sustain (downstream of the bottleneck). Subtitle: the floor only shrinks; growth is capped by w_target.](assets/05-how-shrinks.png)

*Three ways a stage gets smaller: its hold target `w_sustain` decays (normal), its
upstream work drains so it releases to `min_workers`, or - only if it is
downstream of the bottleneck and a blocked stage needs the resource it holds - it
reclaims early to `w_sustain`. The floor never grows a stage; `w_target` caps
growth.*

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

Three extra guards handle cases the basic gate cannot:

- **Cold-stage hold.** A stage with no trusted per-worker speed yet
  (`w_target_is_real == False`) gets a placeholder `w_sustain = min_workers` from
  the capacity model, so the basic gate would clamp it to `min_workers` and invite
  a teardown the streaming post-ready grace must veto every cold cycle (a wasted
  proposal and a layer inversion). The floor instead holds a cold stage at its
  current workers: the cold-start ramp owns cold *growth*, and the floor declines
  to *shrink* a stage toward a target it cannot trust. The whole-chain release path
  still drops a cold stage to `min_workers` on a genuine drain, so the hold never
  pins a finished stage; the moment the stage gathers enough samples to become
  trusted it sizes down normally (`w_target_is_real` in `floor.py`).

- **Downstream guard (benefit-gated).** While the current rate-source candidate
  is *upstream* of a stage and source-normalized stock is still in flight, that
  downstream stage is held warm at current workers - capacity is not donated
  while upstream work is on the way. The exception is a *genuinely reclaimable*
  stage: idle (a ready worker), over-provisioned (`workers > w_sustain`),
  eligible (a real growth target, not cold; not operator-pinned), and
  **beneficial** - freeing it would help an under-capacity stage grow: the stage
  reserves only resource types some growth-wanting, non-manual stage also uses,
  so nothing it frees is wasted (a whole-GPU stage is not reclaimed for its
  incidental host CPUs to feed a CPU-only grower). The grower need not be the
  sticky bottleneck - any non-stalled stage whose real `w_target` exceeds its
  workers counts - so an idle GPU stage can be freed for a GPU-starved sibling
  even while the bottleneck is a different-resource stage. A *stalled* stage
  (`rate_is_stale`) is **not** a grower even though capacity clamps its target to
  `workers + speed_stale_growth_step` (making it look perpetually growing):
  adding workers to a stall
  only drains its queued backlog and cannot raise its collapsing completion rate,
  so reclaiming an expensive warm peer for it would strand the freed resources.
  The exclusion self-clears once completions resume. Once a genuine reclaim holds
  for `reclaim_confirm_cycles` cycles the hold target falls to `min(w_sustain,
  workers)`, so an over-provisioned downstream stage returns resources a grower
  needs instead of stranding them (`protect_downstream_of`, `reclaim_beneficial`,
  `benefit_streak` in `floor.py`).

![Two-row infographic titled "downstream guard: hold warm vs reclaim early". Behind the bottleneck, an idle, over-provisioned downstream stage is held warm. Top row: when nobody needs its CPU it HOLDS WARM at its current size. Bottom row: when a blocked grower needs its CPU (confirmed for N cycles) it RECLAIMS EARLY, shrinking to w_sustain. A banner notes that either way it still releases to min workers once upstream work drains.](assets/05-downstream-guard.png)

*Releasing the downstream warm pin: a stage behind the bottleneck is held warm
(pinned above its hold target) while upstream work is still in flight, and
released early toward `min(w_sustain, workers)` only when a blocked,
growth-wanting stage needs the resource it would free - confirmed for
`reclaim_confirm_cycles` cycles. This pin is the only thing the benefit gate
governs: the stage still shrinks to its sustainable size by the normal capacity
path and drains to `min_workers` once upstream work is gone, so warm-hold is
temporary, not permanent.*

> **Why demand-coupled, not bottleneck-coupled.** Keying this gate to the single
> sticky bottleneck once let an idle, over-provisioned GPU stage stay pinned at
> full size while a GPU-starved sibling that feeds it could not grow - because at
> that moment the bottleneck was a different-resource (CPU) stage, so freeing the
> GPU stage "helped the bottleneck" by nothing. The solver kept proposing to
> delete the idle GPU workers and the floor kept vetoing every one: a circular
> hold (the stage hoards the resource its own feeder needs) that the chain cannot
> escape. The gate now asks "does freeing this help *any* stage that wants to
> grow?", so the idle stage is released to its sustainable size and the starved
> sibling grows. The subset rule still applies, so a GPU stage is never torn down
> to feed a CPU-only grower.

![Infographic "a stalled stage is not a grower": a STALLED STAGE box with three BUSY workers (each on a long task) and an amber tag "w_target = workers + 1 (looks like it wants to grow)" feeds a red gate "extra workers can't raise its rate - one long task is not parallelizable", which leads to "NOT counted as a grower" and a teal box "downstream caption stage stays WARM (not reclaimed for a stall)". Banner: the exclusion self-clears the moment completions resume.](assets/05-stalled-grower.png)

*A stalled stage (`rate_is_stale`) is the one grower the demand-coupled gate must
ignore. Capacity clamps a stall's target to `workers + speed_stale_growth_step`,
so it looks perpetually growing, but the stall is stuck on a single long
in-flight task that more workers cannot parallelize - they would only drain its
backlog, not raise its collapsing completion rate. Counting it as a grower would
reclaim an expensive warm peer (the caption stage) for resources the stall cannot
turn into throughput. The exclusion self-clears the instant completions resume
and `rate_is_stale` drops.*

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
deferred (logged as `shrink_deferred`). The benefit-gated reclaim is likewise
resource-type generic, never GPU-specific: it is **demand-coupled**, scanning
*every* non-manual, non-stalled grower (not the single sticky bottleneck) and
requiring the benefit to hold for `reclaim_confirm_cycles` cycles. A *stalled*
grower is excluded because its inflated `workers + speed_stale_growth_step`
target is not real demand, so neither a transient demand spike nor a stage that
is merely stuck on one long task can churn an expensive downstream stage.

## Trade-offs

| Cost | Benefit |
|---|---|
| A stage stays warm for a few cycles after work genuinely stops. | A short lull never pays a model-reload cold-start cost. |
| Release waits for whole-chain stock to drain for a confirmation window. | A downstream stage is not torn down while upstream work is still in flight. |
| Asymmetric decay holds an over-fed stage above `w_sustain` briefly after a bottleneck shift. | No flapping: a one-cycle lower target cannot delete a worker. |

## Implementation pointer

- `floor.py`: `compute_floors`, the per-stage release gate, the asymmetric
  `w_sustain` EWMA, the cold-stage hold (`w_target_is_real`), the benefit-gated
  downstream guard, and the local saturation veto. The reclaim signal
  (`reclaim_beneficial`) is computed in
  `scheduler.py::_compute_reclaim_beneficial` and confirmed via `benefit_streak`.
- `chain.py`: whole-chain active-stock math (queue depth + pool-queued +
  in-flight, normalized to source units) used for the release decision.
- Decision snapshot fields: `floor`, `shrink_deferred`, `shrink_streak`,
  `pending_shrink_floor`, `releasing`, `reclaim_beneficial`, `benefit_streak`.
- Config: `scale_down_release_cycles`, `scale_down_release_slowdown`,
  `reclaim_confirm_cycles` (see `tuning.md`).
