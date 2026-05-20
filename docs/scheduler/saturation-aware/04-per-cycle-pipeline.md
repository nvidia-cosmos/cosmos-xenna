# 04 — Per-Cycle Pipeline (Phases A → B → C → D)

## TL;DR

Each `autoscale()` call dispatches four phase methods on
`SaturationAwareScheduler` — `_run_phase_a_delete` /
`_run_phase_a_grow`, `_run_phase_b_floor`, `_run_phase_c_grow`,
`_run_phase_d_shrink` — against a per-cycle `AutoscalePlanContext`.
Each phase has a closed input footprint, mutates only `ctx` via
`try_add_worker` / `try_remove_worker`, and is gated by a typed
invariant check before the next phase runs. The per-cycle
`_last_intent_deltas` map computed between Phase B and Phase C is
the only Python state that crosses a phase boundary.

## Problem

[00 — Per-cycle overview](00-overview.md) shows the *order* of the
four phases. It does not show, per phase:

- which slice of `problem_state` and which precomputed maps the
  phase reads;
- which planner methods the phase calls (and on which stages);
- which invariant runs after the phase and which exception class
  is raised on violation;
- whether a failure aborts the cycle (raise) or degrades it
  (warn + continue).

Without those details, a regression in Phase D or a `RuntimeError`
from Phase B floor enforcement is hard to localise. This doc fills
that gap with a drill-down on the shipped code.

## Decision

Codify each phase as an independent private method with a closed
input/mutation contract. `ctx` is the only mutation surface; the
inter-phase Python state is restricted to a single per-cycle map
(`_last_intent_deltas`).

```
                         ┌─────────────────────────────┐
                         │   AutoscalePlanContext      │
                         │   • worker_ids_by_stage()   │
                         │   • worker_ages()           │
                         │   • try_add_worker(...)     │
                         │   • try_remove_worker(...)  │
                         └─────────────────────────────┘
                                      ▲
                                      │  read + mutate (every phase)
                                      │
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Phase A  │───▶│ Phase B  │───▶│ Phase C  │───▶│ Phase D  │
   │ manual   │ ✓  │ floor    │ ✓  │ grow     │ ✓  │ shrink   │ ✓
   └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │ shape         │ shape         │ shape         │ shape
       │ + counters    │ + counters    │ + counters    │ + counters
                                       │ + NaN check   │ + floor
                                                       │ + stuck-plan
                                                       │   monotonicity

   Inter-phase Python state crossing a boundary:
      _last_intent_deltas (computed once between Phase B and Phase C,
                           read by Phase C and Phase D)
```

Key consequences:

- **Closed input/output contract per phase.** Each phase reads its
  slice of `problem_state` plus a small set of precomputed maps
  (floors, ceilings, DAG order, host-GPU used fractions, donor-
  warmup excluded ids); writes go only through `ctx.try_add_worker`
  / `ctx.try_remove_worker`. The frozen `Solution` is produced
  once by `ctx.into_solution()` after the last phase.
- **Manual vs non-manual partitioning.** Phase A owns manual
  stages (`requested_num_workers is not None`). Phases B / C / D
  skip manual stages so the operator-driven path stays the single
  source of truth for those.
- **Invariants are the structural firewall.** Every boundary calls
  [`check_invariants_after_phase`](../../../cosmos_xenna/pipelines/private/scheduling_py/invariants.py)
  with the corresponding `PhaseBoundary` value; Phase C adds a
  classifier-state NaN sweep, Phase D adds a floor sweep and a
  stuck-plan monotonicity sweep, and `INTO_SOLUTION` adds a
  `Solution` shape check. A failure raises
  `SchedulerInvariantError` — the corrupted plan is never returned.
- **Trade-off.** The cycle pays a few microseconds of pure-Python
  invariant overhead per stage per boundary in exchange for
  fail-loud behaviour on internal corruption. A single
  free-form decision blob would be cheaper but would silently
  drift on partial corruption and would be harder to test
  phase-by-phase.

## How it works

The per-phase contract cards below summarise what each phase
reads, what it stages on `ctx`, what runs after it, and how it
fails. The methods live in
[`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py).

```
┌────────────────────────────────────────────────────────────────────┐
│ Phase A — manual operator intent                                   │
│   reads:    problem_stage.requested_num_workers,                   │
│             problem_state.rust.stages[*].worker_groups,            │
│             ctx.worker_ages()                                      │
│   mutates:  ctx.try_remove_worker (youngest first) and             │
│             ctx.try_add_worker, manual stages only                 │
│   gate:     check_invariants_after_phase(PHASE_A)                  │
│             (shape + non-negative pending counters)                │
│   raises:   RuntimeError on planner snapshot inconsistency         │
│             (try_remove_worker returns False on a live worker)     │
│   degrades: WARNING on partial grow (cluster placement exhausted)  │
├────────────────────────────────────────────────────────────────────┤
│ Phase B — floor enforcement                                        │
│   reads:    _compute_stage_floors() (min_workers,                  │
│             min_workers_per_node, num_nodes),                      │
│             ctx.worker_ids_by_stage(), ctx.worker_ages(),          │
│             _floor_stuck_counters                                  │
│   mutates:  ctx.try_add_worker (receiver) and ctx.try_remove_worker│
│             (floor-mode donor via select_youngest_eligible_donor), │
│             non-manual stages only                                 │
│   gate:     check_invariants_after_phase(PHASE_B)                  │
│   raises:   RuntimeError after _floor_stuck_counters exceeds       │
│             floor_stuck_grace_cycles; RuntimeError on              │
│             post-donation retry miss (planner accepted the         │
│             donor remove but cannot place the receiver)            │
│   degrades: WARNING on partial progress (some adds succeeded;      │
│             stuck counter reset because forward progress was made) │
├────────────────────────────────────────────────────────────────────┤
│ Phase C — saturation-driven grow                                   │
│   reads:    _last_intent_deltas (positive entries),                │
│             compute_dag_depth_order() (when                        │
│             enable_dag_priority_growth=True),                      │
│             _compute_stage_ceilings()                              │
│   mutates:  ctx.try_add_worker (receiver) and ctx.try_remove_worker│
│             (saturation-mode cross-stage donor via                 │
│             find_saturation_donor), non-manual stages only         │
│   gate:     check_invariants_after_phase(PHASE_C) +                │
│             check_no_nan_in_classifier_state                       │
│   raises:   SchedulerInvariantError when EWMA state is NaN         │
│             (defensive: classifier corrupted)                      │
│   degrades: WARNING on placement exhaustion + no donor;            │
│             _stuck_plan_counters ticks for the affected stage      │
├────────────────────────────────────────────────────────────────────┤
│ Phase D — saturation-driven shrink                                 │
│   reads:    _last_intent_deltas (negative entries),                │
│             _compute_stage_floors(), _compute_stage_ceilings(),    │
│             stage_cfg.max_scale_down_fraction_per_cycle,           │
│             _compute_host_gpu_used_fractions(),                    │
│             _donor_warmup_excluded_ids                             │
│   mutates:  ctx.try_remove_worker via                              │
│             select_workers_to_remove_oldest_first                  │
│             (idle-first + GPU-consolidation), non-manual stages    │
│   gate:     check_invariants_after_phase(PHASE_D) +                │
│             check_floor_after_phase_d +                            │
│             check_stuck_plan_monotonicity                          │
│   raises:   RuntimeError when planner refuses removal of a worker  │
│             selected from its own snapshot (scheduler defect);     │
│             SchedulerInvariantError if Phase D reduced a stage     │
│             below its floor                                        │
│   degrades: never silently — the three clamps (floor,              │
│             fraction cap, classifier magnitude) compose so the     │
│             requested shrink is always bounded by design           │
└────────────────────────────────────────────────────────────────────┘
```

### Per-phase notes

- **Phase A** consumes operator intent (`requested_num_workers`) as
  a target, not a hard lower bound: when capacity is short, the
  manual grow degrades gracefully so the cycle still produces a
  valid plan. The manual-delete branch selects youngest workers
  first so long-lived warmed actors survive a manual shrink. Manual
  deletes free placement slots that Phase B / Phase C may reuse
  later in the same cycle.

- **Phase B** runs *before* the classifier so a zero-actor stage
  never sees an intent computation against an empty slot signal.
  Each shortfall is filled by `try_add_worker`; on placement
  exhaustion, the floor-mode donor protocol (which intentionally
  bypasses `donor_warmup_excluded_ids` — see
  [13 — cross-stage donor](13-cross-stage-donor.md)) reallocates
  one worker. A no-donor floor miss accumulates a per-stage stuck
  counter; the cycle raises `RuntimeError` only after
  `floor_stuck_grace_cycles` consecutive failures. A post-donation
  retry miss raises immediately because the donor remove has
  already been staged and cannot be safely rolled back.

- **Phase C** consumes `_last_intent_deltas` set by
  `_compute_intent_deltas` (see
  [11 — growth mode](11-growth-mode-state-machine.md) for how the
  intent is derived). It walks stages in DAG-depth order and clamps
  each grow request to the per-stage ceiling. Every stage with
  positive intent is attempted independently, so one blocked
  bottleneck cannot stop growth attempts for other saturated
  stages. The saturation-mode cross-stage donor fallback fires when
  the cluster is full but a downstream stage still needs to grow;
  the donor cooldown is advanced only on a complete donate+retry
  cycle so a failed retry leaves the donor eligible next cycle.

- **Phase D** combines two shrink drivers — negative classifier
  intent and hard-cap overflow — and applies three independent
  clamps (per-stage floor, per-cycle fraction cap, classifier
  magnitude cap) before selecting victims. The victim ordering is
  GPU-consolidation-aware (see
  [15 — idle-first scale-down](15-idle-first-scale-down.md))
  and skips workers inside the donor warmup grace
  (`_donor_warmup_excluded_ids`). The post-phase
  `check_floor_after_phase_d` distinguishes "Phase D reduced below
  floor" (a defect — raise) from "Phase B left the stage below
  floor" (a grace-window scenario — accepted) using the
  pre-Phase-D worker-count snapshot.

After Phase D the scheduler calls `ctx.into_solution()`, runs
`check_solution_shape` at the `INTO_SOLUTION` boundary, and
persists per-worker ages. The frozen `Solution` is returned to
the streaming dispatcher.

## Knobs

The four-phase order, the per-phase input footprints, and the
invariant boundaries are not configurable — they are structural.
Knobs that affect *how each phase behaves* live on
`SaturationAwareConfig` and `SaturationAwareStageConfig`; see the
per-feature docs:

- Phase A: none (operator intent only).
- Phase B: [16 — hard caps and floors](16-hard-caps-and-floors.md),
  [13 — cross-stage donor](13-cross-stage-donor.md).
- Phase C: [12 — multi-target DAG growth](12-multi-target-dag-growth.md),
  [13 — cross-stage donor](13-cross-stage-donor.md),
  [11 — growth mode](11-growth-mode-state-machine.md).
- Phase D: [15 — idle-first scale-down](15-idle-first-scale-down.md),
  [16 — hard caps and floors](16-hard-caps-and-floors.md).

## See also

- [00 — Per-cycle overview](00-overview.md) — the full cycle
  including pre-flight setup and intent computation.
- [03 — Planning context](03-planning-context.md) — the
  `AutoscalePlanContext` API that every phase mutates.
- [19 — Phase invariants](19-phase-invariants.md) — the exact
  predicate each `check_*` helper enforces and how violations are
  surfaced.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the
  floor-mode and saturation-mode donor protocols Phase B / Phase C
  use on cluster exhaustion.
- [15 — Idle-first scale-down](15-idle-first-scale-down.md) — the
  victim-selection key Phase D applies before removing workers.
