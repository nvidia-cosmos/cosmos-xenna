# 16 — Hard Caps and Floors

## TL;DR

Four per-stage operator knobs — `min_workers`, `min_workers_per_node`,
`max_workers`, `max_workers_per_node` — bound each stage between a
structural FLOOR that Phase B enforces every cycle (ignoring the
classifier) and a hard CEILING that gates Phase C grow and forces a
Phase D shrink when the cap sits below the current count. They are
the autoscaler's HPA-style `minReplicas` / `maxReplicas` plus
K8s-style topology-spread bounds, projected onto a single per-stage
worker-count number line.

## Problem

A streaming autoscaler driven purely by classifier signals
(utilisation, queue time, streak counters) is brittle in three
operator scenarios:

- A pre-warm stage whose model takes minutes to load needs a minimum
  live worker count even while the classifier marks it
  `OVER_PROVISIONED`; otherwise the next surge pays the cold-start
  latency.
- A stage with a strict resource budget (GPU memory ceiling,
  license-bound count) must never exceed K workers regardless of how
  starved a downstream stage looks; otherwise saturation-driven
  growth eventually overruns the budget.
- A stage with topology constraints — at most M workers per node —
  needs that bound enforced before placement; otherwise the planner
  may pack the cap's worth of workers onto one node and trip a
  downstream OOM.

The classifier reads slot occupancy, not budget; burying these
policies in stage code couples operator intent to implementation.

## Decision

Expose four `SaturationAwareStageConfig` fields and apply them at
two distinct lifecycle points:

- `min_workers` / `min_workers_per_node` define a **structural
  floor** enforced by **Phase B** every cycle, before the classifier
  runs. The effective floor is `max(1, min_workers if set,
  min_workers_per_node * num_nodes if set)`; the implicit `1` keeps
  every non-finished stage above zero. Cluster-full escalates to the
  floor-mode cross-stage donor; sustained failure raises
  `RuntimeError` after `floor_stuck_grace_cycles` consecutive failed
  cycles, and a post-donation retry miss raises immediately (the
  donor remove cannot be safely rolled back).
- `max_workers` / `max_workers_per_node` define a **hard ceiling**
  checked in **Phase C** (clamping the grow intent before
  `try_add_worker`) and in **Phase D** (folding `current > ceiling`
  overflow into the shrink driver). The effective ceiling is
  `min(max_workers if set, max_workers_per_node * num_nodes if
  set)`, or `None` when neither knob is set. Phase C clamps positive
  classifier intent to `headroom = max(0, ceiling - current)` so the
  autoscaler never proposes a count the planner would not place;
  Phase D pulls a stage back to a freshly lowered cap on the next
  cycle.
- **Phase D floor clamp.** Shrink drivers (negative classifier
  intent OR ceiling overflow) are bounded by `actual_remove =
  min(requested_remove, allowed_by_floor, fraction_cap)` where
  `allowed_by_floor = max(0, current - floor)`. The per-stage worker
  count is snapshotted by `autoscale()` **just before** Phase D
  runs, so `check_floor_after_phase_d` can distinguish *Phase D
  dropped below floor* (a defect — raise `SchedulerInvariantError`)
  from *Phase B left the stage below floor* (a grace-window
  scenario — accepted; Phase B retries next cycle).

```
       Number line of "current workers" for a single stage:

        0           floor                  current               ceiling           ∞
        ●─────────────●──────────────────────●──────────────────────●──────────────▶
                      ▲                                              ▼
                      │                                              │
                      │  Phase B: raise to floor every cycle         │
                      │  (ignores classifier; escalates to           │
                      │  floor-mode donor on cluster-full;           │
                      │  RuntimeError after grace)                   │
                      │                                              │
                      │           Phase A grow ─────────────────▶    │
                      │             (manual stages; target, not      │
                      │              a hard lower bound)             │
                      │                                              │
                      │           Phase C grow ─────────────────▶    │
                      │             (classifier intent, clamped at   │
                      │              ceiling via headroom)           │
                      │                                              │
                      │  ◀───────── Phase D shrink ─────────────     │
                      │             (negative intent OR ceiling      │
                      │              overflow, clamped at floor)     │

        Composition of the per-stage / per-node knobs (side inset):

            floor   = max( 1, min_workers, min_workers_per_node * N )
            ceiling = min(    max_workers, max_workers_per_node * N )

            N = num_nodes. floor uses max() because operators want the
            STRONGEST lower bound; ceiling uses min() because operators
            want the TIGHTEST upper bound. Cross-field validation
            rejects min_workers > max_workers (and the per-node
            analogue) at construction time.
```

Trade-off: caps clamp **post-decision** so the operator never sees
the autoscaler propose a count it ultimately would not place; floors
enforce **structurally** so a misconfigured pipeline fails LOUD on
the first viable cycle, not silently mid-run.

## How it works

The knobs live on `SaturationAwareStageConfig` in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).
Two helpers on `SaturationAwareScheduler` in
[`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
materialise the per-cycle bounds: `_compute_stage_floors(num_nodes)`
returns `{stage_index: target_min}` and
`_compute_stage_ceilings(num_nodes)` returns
`{stage_index: ceiling_or_None}`. Both apply uniformly to manual and
non-manual stages — on a manual stage `requested_num_workers` is a
*target*, not a hard lower bound, so the floor still wins on a
capacity squeeze.

The phases consume the helpers as follows:

- `_run_phase_b_floor` walks every non-manual, non-finished stage
  and calls `ctx.try_add_worker` until `current >= target_min`;
  cluster-full triggers the floor-mode cross-stage donor (see
  [13 — Cross-stage donor](13-cross-stage-donor.md)). A no-donor
  miss accumulates `_floor_stuck_counters[stage_name]` and raises
  `RuntimeError` once the counter exceeds `floor_stuck_grace_cycles`.
- `_run_phase_c_grow` reads `_compute_stage_ceilings` once at the
  top of the phase, computes `headroom = max(0, ceiling - current)`,
  and clamps `intent = min(intent, headroom)`. A single INFO log
  per stage reports any suppressed amount so the cap surfaces
  whenever it actively binds.
- `_run_phase_d_shrink` combines both shrink drivers into
  `requested_remove = max(-intent if negative else 0,
  ceiling_excess)` where `ceiling_excess = max(0, current -
  ceiling)`. The three clamps — floor,
  `max_scale_down_fraction_per_cycle`, classifier magnitude —
  compose so the per-cycle shrink stays bounded; Phase D skips
  manual stages.
- `autoscale()` snapshots the per-stage worker count between Phase
  C and Phase D and passes it to `check_floor_after_phase_d`, so the
  invariant attributes any post-shrink floor violation correctly.

Cross-field consistency (`min_workers <= max_workers` and
`min_workers_per_node <= max_workers_per_node`) is rejected at
construction time by
`SaturationAwareStageConfig.__attrs_post_init__`; see
[17 — Config validation](17-config-validation.md).

## Knobs

All four knobs are per-stage on `SaturationAwareStageConfig` in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).
`None` means *not configured*: floor falls back to `1`, ceiling to
unbounded.

| Field | Default | Effect |
|---|---|---|
| `min_workers` | `None` -> 1 | Cluster-wide minimum. Phase B raises the stage to this count every cycle, ignoring the classifier. Failure raises `RuntimeError` after the donor-fallback grace. |
| `min_workers_per_node` | `None` -> 0 | Per-node minimum. Composes with `min_workers` via `max(min_workers, min_workers_per_node * num_nodes)`. K8s-style topology-spread on the lower bound. |
| `max_workers` | `None` -> unbounded | Cluster-wide cap. Phase C clamps grow intent to the remaining headroom; Phase D forces a shrink when `current > max_workers` (e.g. operator lowered the cap mid-run). |
| `max_workers_per_node` | `None` -> unbounded | Per-node cap. Composes with `max_workers` via `min(max_workers, max_workers_per_node * num_nodes)`. K8s-style topology-spread on the upper bound. |

External analogues:
[Kubernetes HPA `minReplicas` / `maxReplicas`](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
bound the replica count between two operator knobs regardless of
metric pressure; [Kubernetes topology spread constraints](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/)
bound replica count per topology domain (here: per node).

## See also

- [00 — Per-cycle overview](00-overview.md) — where the floor sweep
  (Phase B) and the cap clamps (Phase C grow / Phase D shrink) fit
  in the four-phase pipeline.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the floor-mode
  donor protocol Phase B escalates to on cluster-full, and the
  donor-floor preservation rule both donor modes must respect.
- [17 — Config validation](17-config-validation.md) — the cross-field
  validators that reject `min_workers > max_workers` (and the
  per-node analogue) at construction time.
