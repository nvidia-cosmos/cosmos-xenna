# 14 — Worker Age Tracking

## TL;DR

The saturation-aware planner keeps a per-worker age (in autoscale
cycles) on the **Rust planning context** so donor selection
(floor mode, saturation mode) and Phase D shrink can prefer the
**youngest eligible worker** instead of picking randomly. Ages
mutate atomically with every `try_add_worker` / `try_remove_worker`
call; the Python scheduler only owns the cross-cycle persistence
map. There are no operator knobs — this feature is the substrate
that other features tune.

## Problem

Long-running workers are expensive to evict. A worker that has
been on a stage for tens of cycles carries warm in-process state
(model caches, compiled execution graphs), allocated GPU memory
that a replacement would have to re-allocate from scratch, and
primed setup objects (tokenisers, processors, parsers) that
amortise their cost across many tasks. Donating one of these to
another stage costs **minutes** of pipeline idle while a
replacement boots and re-warms. Donating a worker placed in the
current or previous cycle costs only its startup time — the
warmup grace clock has barely started.

The planner therefore needs a stable, per-worker age signal it can
use to bias victim selection toward young workers. Two design
constraints make this non-trivial:

- **Atomicity with the planner snapshot.** Age mutations have to
  happen on the exact same state transitions as `try_add_worker`
  and `try_remove_worker` — including the FGD reuse path
  (un-staging a pending remove) and the cancel-pending-add path
  (un-staging a fresh add). Splitting age tracking into a Python
  mirror would duplicate every branch of those state machines,
  with a guaranteed drift bug the first time someone edits one
  side and forgets the other.
- **Cross-cycle persistence with stale-id drop.** Workers that die
  between cycles must disappear from the age map; workers the
  scheduler has never seen must not inherit some other worker's
  age just because the constructor was called with leftover state.

## Decision

Maintain a per-cycle `worker_ages: HashMap<String, u64>` on
[`AutoscalePlanContext`](../../../src/pipelines/private/scheduling/autoscale_plan_context.rs)
and mutate it from the same Rust methods that already mutate the
working cluster snapshot. Python consumers read it via
[`AutoscalePlanContext.worker_ages()`](../../../cosmos_xenna/pipelines/private/data_structures.py)
on the wrapper. The scheduler in
[`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
owns the cross-cycle persistence map only.

```
┌─ Cycle 1 (cold start) ─────────────────────────────────┐
│  ProblemState reports stage S has 0 workers            │
│                                                        │
│  Phase C places W1, W2, W3, W4 via try_add_worker      │
│                                                        │
│  ctx.worker_ages()        {W1:0, W2:0, W3:0, W4:0}     │
│  scheduler._worker_ages   {W1:0, W2:0, W3:0, W4:0}     │
└────────────────────────────────────────────────────────┘
                              │
                              ▼ next cycle: each age + 1
┌─ Cycle 2 (one stale drop, one new placement) ──────────┐
│  next-cycle seed          {W1:1, W2:1, W3:1, W4:1}     │
│  ProblemState now reports W1, W2, W3 only              │
│  (W4 died); Rust drops the stale W4 seed entry         │
│                                                        │
│  ctx.worker_ages() after seeding                       │
│                           {W1:1, W2:1, W3:1}           │
│                                                        │
│  Phase C places W5 via try_add_worker                  │
│                                                        │
│  ctx.worker_ages()        {W1:1, W2:1, W3:1, W5:0}     │
│  scheduler._worker_ages   {W1:1, W2:1, W3:1, W5:0}     │
└────────────────────────────────────────────────────────┘
                              │
                              ▼ next cycle: each age + 1
┌─ Cycle 3 (donor selection) ────────────────────────────┐
│  next-cycle seed          {W1:2, W2:2, W3:2, W5:1}     │
│                                                        │
│  Phase C: a downstream receiver needs one extra        │
│  worker; the cluster is full; the cross-stage donor    │
│  selector iterates candidates on stage S and picks     │
│  min by (age, worker_id):                              │
│                                                        │
│      {(2,W1), (2,W2), (2,W3), (1,W5)}                  │
│           └───── youngest eligible ─────┘              │
│                                                        │
│  → W5 selected as donor (cheap; ~1 cycle of warmup)    │
└────────────────────────────────────────────────────────┘
```

**Trade-off.** The per-worker age map adds an O(W) hash map
alongside the cluster snapshot (a handful of bytes per worker per
cycle) plus one increment-and-drop pass at the cycle boundary. In
return the planner never destroys a worker with significantly more
warm state than its alternatives — measured in minutes of idle
GPU time when the alternative is to boot a replacement on a
multi-minute model load.

## How it works

1. **Cross-cycle seed (Python).** At the top of `autoscale()`,
   `_next_cycle_worker_ages` builds
   `{wid: age + 1 for wid, age in self._worker_ages.items()}` and
   passes the map into
   `AutoscalePlanContext.from_problem_state(..., worker_ages=...)`.

2. **Seed projection (Rust).** The constructor iterates the
   seeded `reserved_worker_ids` (every worker present in the new
   `ProblemState`) and looks each id up in the supplied seed map,
   defaulting missing entries to 0. Ids that were in the seed but
   are no longer in `ProblemState` are silently dropped — a dead
   worker leaves no trace, and a worker the scheduler has never
   seen starts at age 0.

3. **Fresh placement.** `try_add_worker` writes
   `worker_ages.insert(placement.id, 0)` after the cluster
   allocation commits. A fresh placement always starts at age 0
   regardless of any seeded value (it cannot have been seen before
   this cycle).

4. **Reuse path.** When the FGD search chooses to revive a worker
   already staged for removal, the reuse branch deliberately does
   **not** touch `worker_ages`: the worker never structurally left
   the cluster, so its existing age is preserved. This is what
   makes "did we already donate this worker once this cycle?"
   checks meaningful — the same id keeps the same age whether
   the planner reused the placement or not.

5. **Cancel-pending-add vs. stage-for-removal.** `try_remove_worker`
   has two branches:
   - **Cancel-pending-add** (the add was staged earlier in this
     cycle): drop the age entry. The worker never structurally
     existed before this cycle and must not appear in any
     next-cycle seed.
   - **Stage-for-removal** (the worker pre-existed this cycle):
     keep the age entry. A later `try_add_worker` may revive the
     placement via the FGD reuse path; preserving the age keeps
     subsequent youngest-first selection meaningful.

6. **Consumers.** Three call sites read the map this cycle, each
   via `ctx.worker_ages()`:
   - Floor-mode cross-stage donor (Phase B):
     [`select_youngest_eligible_donor`](../../../cosmos_xenna/pipelines/private/scheduling_py/donor.py)
     picks `min(candidates, key=(age, worker_id))`.
   - Saturation-mode cross-stage donor (Phase C):
     `find_saturation_donor` applies its anti-flap filters and
     then breaks ties with the same `(age, worker_id)` key.
   - Phase D shrink:
     `_select_workers_to_delete_youngest_first` in
     `saturation_aware.py` orders victims by the same key.

7. **End-of-cycle persistence (Python).** After
   `ctx.into_solution()` drains the staged plan,
   `_persist_worker_ages` reads `ctx.worker_ages()` and intersects
   with `ctx.worker_ids_by_stage()`, so workers that ended up in
   `solution.deleted_workers` (drained out of the live set) are
   not carried into the next cycle's seed. The result is stored
   on `self._worker_ages` for the next call to
   `_next_cycle_worker_ages`.

## Knobs

This feature has **no operator-facing configuration**. There is no
`enable_worker_age_tracking` flag, no age-unit setting, and no
"old enough to be donated" threshold. Worker-age tracking is the
substrate on which the operator knobs of other features run:

- Aggressiveness, anti-flap windows, and donation limits live on
  [13 — Cross-stage donor](13-cross-stage-donor.md).
- Phase D victim ordering and consolidation logic live on
  [15 — Idle-first scale-down](15-idle-first-scale-down.md).

If a deployment wants "never donate workers older than N cycles",
the threshold goes on the donor, not on age tracking.

## See also

- [03 — Planning context](03-planning-context.md) — owns the Rust
  data structure that holds `worker_ages`.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the primary
  consumer of youngest-first selection.
- [15 — Idle-first scale-down](15-idle-first-scale-down.md) —
  combines age with idle-first ordering when picking shrink
  victims.
