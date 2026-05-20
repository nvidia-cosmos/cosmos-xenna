# 15 — Idle-First, GPU-Consolidating Scale-Down

## TL;DR

Phase D removes workers using a four-key composite sort:
`(host_gpu_used_fraction ASC, idle DESC, age DESC, worker_id ASC)`.
The primary key prefers victims on the most-empty GPUs so deletion
has the best chance of freeing a whole GPU; the idle key prevents
mid-task kills; the inverted age key rotates older actors out
under sustained over-provisioning; the `worker_id` key keeps the
plan reproducible across cycles.

## Problem

When the classifier emits a negative intent for a stage, or an
operator-lowered `max_workers` cap overflows, Phase D must delete
`N` workers from that stage's pool. The naive choices each fail:

- Pick by `worker_id` alone: deterministic but blind. A worker
  holding a tiny fraction of a busy GPU dies; the GPU stays
  pinned by larger allocations and never becomes recoverable.
- Pick the youngest first (the manual-shrink heuristic): inverts
  the saturation case. Manual delete reverses an operator's most
  recent intent; saturation-driven shrink corrects sustained
  over-provisioning, where the long-tenured actor is the one most
  likely to be carrying stale state.
- Pick busy workers: kills work in flight and forces re-dispatch,
  which is the opposite of "relieve pressure".

Three forces have to be ordered. GPU consolidation
(fragmentation recovery) is the goal of the shrink; idle is the
safety condition; worker rotation is the steady-state hygiene
benefit.

## Decision

Use a four-key composite sort, with consolidation as the primary
key and `worker_id` as the deterministic tail.

```
                 candidate workers in one stage
                                │
                                ▼
   ┌──────────────────────────────────────────────────┐
   │  1.  host_gpu_used_fraction  ASC   (consolidate) │
   │  2.  idle                    DESC  (safety)      │
   │  3.  age                     DESC  (rotation)    │
   │  4.  worker_id               ASC   (determinism) │
   └──────────────────────────────────────────────────┘
                                │
                                ▼
       first  delete_count  ids  ─▶  Phase D victims
```

Worked example: four candidates, `delete_count = 1`.

```
   candidates                            after 4-key sort
   ─────────────────────────────         ─────────────────────────
     wid    gpu_frac  idle  age            1.  w-001  ◀── victim
     w-001    0.05    yes    12            2.  w-002
     w-002    0.05     no    20            3.  w-003
     w-003    1.00    yes    30            4.  w-004
     w-004    1.00    yes     5
```

w-001 wins the tied (0.05) GPU bucket because idle dominates
busy. w-002 still beats both 1.00-GPU candidates — the
consolidation key dominates age. w-003 beats w-004 on the
remaining tie because older actors rotate first.

**Trade-off.** GPU fraction is recomputed once per cycle from the
runtime snapshot. Phase A / B / C mutations during the cycle may
shift live cluster fractions before Phase D runs; the
consolidation key is a per-cycle approximation that converges
over multiple cycles rather than a live-cluster reading. The
cheaper approximation was preferred over an extra FFI hop to the
Rust planner state.

### Why the keys are in this order

- **GPU fraction ASC — consolidation.** A worker on a (1.00, 0.05)
  GPU is *the* lever for whole-GPU recovery: deleting the 0.05
  leftover leaves the GPU at 1.00 (still pinned); deleting the
  1.00 worker leaves it at 0.05 (close to recoverable). The
  MAX-across-worker-GPUs aggregation captures the binding
  constraint for multi-GPU workers — the most-loaded GPU is what
  prevents the worker's deletion from freeing anything.

- **Idle DESC — safety.** Within a consolidation bucket, never
  pick a busy worker if an idle one exists. The signal is
  `num_used_slots == 0`; idle is the only condition under which
  deletion is loss-free.

- **Age DESC — rotation.** Phase D is a sustained-pressure
  response — the classifier emits a negative intent only after
  the streak / EWMA filters agree that the stage has been
  over-provisioned for many cycles. In that regime the older
  actor is the higher-value victim: it has been alive longest and
  is most likely to carry model-cache drift, allocator
  fragmentation history, or leaked references. This is the
  opposite of the manual-delete path, which removes the youngest
  worker so the operator's most recent add is the first thing
  reversed.

- **Worker id ASC — determinism.** With the other three keys
  tied, the helper still has to pick. Lexicographic ids give the
  same plan on the same inputs across cycles, which keeps replay
  traces and unit tests stable.

The donor-warmup grace is applied **before** the sort: any worker
whose READY timestamp is younger than the grace window is dropped
from the candidate pool regardless of its sort key. Freshly-warmed
actors have not had time to absorb work yet, so their
`num_used_slots` reading is unreliable and their age is too low
to warrant rotation. Excluding them in one place keeps Phase D
and the cross-stage donor path consistent.

## How it works

```
Phase D for one stage
  │
  ├─ requested_remove = max(-intent, ceiling_excess)
  │
  ├─ clamp by  floor   /  fraction-per-cycle cap
  │
  ├─ build  worker_used_slots               ◀─ runtime snapshot
  ├─ build  worker_host_gpu_used_fractions  ◀─ MAX across GPUs
  │
  ▼
  select_workers_to_remove_oldest_first(
      worker_ids, worker_ages, delete_count,
      worker_used_slots, worker_host_gpu_used_fractions,
      excluded_worker_ids,                  ◀─ donor warmup grace
  )
  │
  ▼
  ctx.try_remove_worker(stage, victim)  for each returned id
```

`_compute_host_gpu_used_fractions` walks every worker group in
every stage and aggregates each GPU allocation's `used_fraction`
by `(node_name, gpu_offset)`. The map is built once per cycle.

`_extract_worker_host_gpu_used_fractions` projects the cluster-
wide map onto a single stage's workers, taking `max(...)` across
each worker's GPU allocations. CPU-only workers (no GPU
allocations) and GPUs with no recorded usage default to `0.0`,
which makes them most-consolidatable — that is, CPU-only stages
naturally fall back to the older `(idle, age, worker_id)`
ordering because the consolidation key collapses to a constant.

`select_workers_to_remove_oldest_first` filters out
`excluded_worker_ids` (the donor-warmup grace set), sorts the
remaining candidates by the four-key tuple, and returns the first
`delete_count` worker ids. Missing entries default to `0` (age),
`0` (used slots, treated as idle), and `0.0` (GPU fraction,
treated as most-consolidatable), so under-populated inputs
degrade gracefully rather than raise.

## Knobs

| Knob | Lives on | Effect on Phase D victim choice |
|---|---|---|
| `donor_warmup_grace_s` | `SaturationAwareStageConfig` | Window after a worker reaches READY in which it is excluded from victim selection. Larger values let freshly-warmed actors keep filling their queues before they can be reclaimed. |
| `max_scale_down_fraction_per_cycle` | `SaturationAwareStageConfig` | Caps `delete_count` to a fraction of `current` per cycle; orthogonal to the sort key but bounds how many of the sorted candidates Phase D consumes. |
| `min_workers` / `min_workers_per_node` | `SaturationAwareStageConfig` | Hard floor; Phase D never drives a stage below it regardless of how many candidates the sort returns. |
| `worker_max_lifetime_m`, `worker_restart_interval_m` | scheduled-rotation knobs | Independent worker-lifetime rotation; not part of Phase D. Phase D's age key only orders victims **among workers already eligible for removal this cycle**. |

## See also

- [10 — Slow-start mechanisms](10-slow-start-mechanisms.md) —
  the donor-warmup grace that supplies `excluded_worker_ids` for
  this selection.
- [14 — Worker age tracking](14-worker-age-tracking.md) — the
  cycle-counted age substrate that feeds the age key, and the
  contrasting youngest-first ordering used on operator-driven
  shrink paths.
- [`scale_down.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/scale_down.py)
  — the `select_workers_to_remove_oldest_first` helper.
- [`SaturationAwareScheduler._run_phase_d_shrink`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
  — the cycle-level orchestration that builds the inputs and
  applies the floor / fraction-cap clamps before calling the
  helper.
