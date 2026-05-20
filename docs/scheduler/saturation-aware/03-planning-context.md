# 03 — Planning Context

## TL;DR

A Rust `pyclass`, `AutoscalePlanContext`, bridges Fragmentation
Gradient Descent (FGD) placement into Python. The
saturation-aware scheduler builds one context per autoscale
cycle, stages worker adds and removes against the context's
working cluster snapshot as Phases A–D run, then drains the
staged plan into a `Solution` via `into_solution()`. FGD stays
the single source of truth for placement; Python orchestrates
per-stage decisions on top of it.

## Problem

The scheduler must plan worker adds and removes per stage,
interleaving classifier signals, floor enforcement, and
cross-stage donor logic between every decision. The pre-existing
Rust autoscaler is all-or-nothing: it plans an entire cycle in
one opaque call. Two viable paths exist to drive the same
algorithm one decision at a time:

- Re-implement FGD in Python. Two placement algorithms then have
  to stay in sync forever; any drift silently degrades placement
  quality on whichever path goes stale first.
- Expose narrow per-stage primitives on a shared context that
  keeps FGD's working state. Python orchestrates; Rust still
  owns the algorithm.

A second-order requirement is **reuse**: a worker removed by an
earlier phase (donor / shrink) must be eligible to satisfy a
later phase's add request in the same cycle without surfacing as
a delete-then-create pair that actuates as worker churn.

## Decision

Treat each autoscale cycle as a planning transaction against a
Rust `pyclass`, `AutoscalePlanContext`, that exposes three small
primitives to Python: `try_add_worker(stage_index)`,
`try_remove_worker(stage_index, worker_id)`, and
`into_solution()`. One context is built at the start of each
`autoscale()` call via `AutoscalePlanContext.from_problem_state`,
Phases A–D mutate it, and `into_solution()` flips an
`is_drained` flag that converts any further `try_add_worker` /
`try_remove_worker` call into `RuntimeError`. The next cycle
constructs a fresh context from the new `ProblemState`.

Same-cycle reuse is built into the context's internal
accounting: a worker removed by `try_remove_worker` and
re-allocated by a later `try_add_worker` in the same cycle
appears in NEITHER `new_workers` NOR `deleted_workers` on the
resulting `StageSolution`, matching the live-set view that the
worker was never structurally swapped. The per-worker age map
seeded into the context drives youngest-first donor selection
across stages; FGD's reuse path preserves the age entry so a
donor's "age since the planner first observed it" survives an
in-cycle remove/add round-trip.

The trade-off is paying an FFI hop per per-stage call instead of
the pre-existing one-shot full-cycle Rust call. The trade is
deliberate: it buys a single placement algorithm (no Python copy
of FGD to drift) plus per-stage planning Python can interleave
with classifier, donor, and floor logic.

### Single-cycle lifecycle

```
              ┌────────────────────────────────────────┐
              │ from_problem_state(problem, state,     │
              │                    worker_ages=...)    │
              └────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────────────────────────────────────────────────┐
   │  AutoscalePlanContext                                │
   │  ┌────────────────────────────────────────────────┐  │
   │  │  working cluster snapshot                      │  │
   │  │    cluster        : ClusterResources           │  │
   │  │    current_workers / current_worker_groups     │  │
   │  │    pending_adds   / pending_removes (by stage) │  │
   │  │    worker_ages    : HashMap<id, cycles>        │  │
   │  └────────────────────────────────────────────────┘  │
   │     ▲        ▲        ▲        ▲                     │
   │     │        │        │        │                     │
   │     │  try_add_worker(stage_index)                   │
   │     │  try_remove_worker(stage_index, worker_id)     │
   │     │        │        │        │                     │
   │  ┌──┴───┐ ┌──┴───┐ ┌──┴───┐ ┌──┴───┐                 │
   │  │Phase │ │Phase │ │Phase │ │Phase │                 │
   │  │  A   │ │  B   │ │  C   │ │  D   │                 │
   │  │manual│ │floor │ │ grow │ │shrink│                 │
   │  └──────┘ └──────┘ └──────┘ └──────┘                 │
   └──────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ ctx.into_solution()    │
                    │ drains pending maps    │
                    │ sets is_drained = True │
                    └────────────────────────┘
                                │
                                ▼
                            Solution
                per-stage new_workers + deleted_workers;
                same-cycle reuse appears in NEITHER list
```

## How it works

**Construction (`from_problem_state`).** Clones
`problem.cluster_resources`, then allocates every worker
currently in `state.stages[*].worker_groups` against the clone
so FGD sees live placement and fragmentation before any
planning runs. Initialises per-stage `pending_adds` and
`pending_removes` maps, snapshots each stage's
`slots_per_worker` so `into_solution` can round-trip it
unchanged, and seeds the per-worker age map. Seed ages come
from the caller's previous-cycle map (each entry already
incremented for the new cycle); workers present in `state` but
absent from the seed default to age 0, and ids in the seed but
absent from `state` are silently dropped.

**Staging adds (`try_add_worker(stage_index)`).** Runs FGD
against the working cluster snapshot. Three outcomes:

- *Fresh allocation* — mints a worker id (the context's id
  factory consults `reserved_worker_ids` to avoid aliasing
  seeded live workers or already-pending entries), allocates
  on the working cluster, and pushes the worker to
  `pending_adds[stage_name]`. Age starts at 0.
- *Reuse* — FGD chose an already-staged-for-removal worker
  over a fresh placement. The reused worker is popped from
  `pending_removes[stage_name]`, the cluster is re-allocated
  for that placement, and the original age is preserved.
  `pending_adds` is intentionally NOT touched: the worker was
  never structurally removed from the live set, only un-staged.
- *No placement* — returns `None` with no mutation. Callers
  fall back to donor logic or grace counters.

**Staging removes (`try_remove_worker(stage_index, id)`).**
Pops the worker from `current_workers` /
`current_worker_groups`, releases its allocation against the
working cluster, and appends the original
`ProblemWorkerGroupState` to `pending_removes[stage_name]`. If
the worker was a fresh add staged earlier in the same cycle,
the cancel-pending-add branch drops the age entry; otherwise
the age entry is preserved so the FGD reuse path can still
pick the worker up later in the cycle.

**Draining (`into_solution()`).** Walks the stage list in
problem order, drains `pending_adds` into
`StageSolution.new_workers` and `pending_removes` into
`StageSolution.deleted_workers`, and copies the snapshotted
`slots_per_worker` into each per-stage solution unchanged. Sets
`is_drained = True`. The drain is idempotent (a second call
returns a `Solution` with empty per-stage lists), but
`try_add_worker` and `try_remove_worker` raise `RuntimeError`
on a drained context. Read accessors (`worker_ages()`,
`worker_age(id)`, `worker_ids_by_stage()`) deliberately bypass
the drained-state guard so the scheduler can persist the
post-cycle worker-age map for the next cycle after the plan has
been emitted.

## Knobs

This feature has no operator knobs. The contract is fixed in
code: one context per cycle, mutate then drain, never re-use.
Internal FGD parameters (e.g. the reuse-equivalent
fragmentation reward) live in the Rust module and are pinned to
match the standalone Rust autoscaler's values so the two
placement paths agree.

## See also

- [00 — Per-cycle overview](00-overview.md) — where
  `AutoscalePlanContext` sits inside the `autoscale()` call.
- [04 — Per-cycle pipeline](04-per-cycle-pipeline.md) — what
  Phases A–D actually do with the context.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — consumer
  of the same-cycle reuse semantics.
- [14 — Worker age tracking](14-worker-age-tracking.md) —
  consumer of the seeded age map.
- [19 — Phase invariants](19-phase-invariants.md) — invariant
  checks that run against the context between phases.
- Python wrapper:
  [`cosmos_xenna/pipelines/private/data_structures.py`](../../../cosmos_xenna/pipelines/private/data_structures.py).
- Rust `pyclass`:
  [`src/pipelines/private/scheduling/autoscale_plan_context.rs`](../../../src/pipelines/private/scheduling/autoscale_plan_context.rs).
