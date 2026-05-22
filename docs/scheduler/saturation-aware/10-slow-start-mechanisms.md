# 10 — Slow-Start Mechanisms

## TL;DR

A freshly-added worker becomes READY with an empty dispatcher queue.
Its slots read "empty" for several cycles while tasks from the
upstream stage flow into the input queue and the dispatcher routes
them onto its slots. Three independent layers stop those transient
empty-slot readings from oscillating the cluster: **per-worker
measurement grace** hides warmup samples from the EWMA, **donor
warmup grace** hides warmup workers from victim selection, and the
**growth-mode HOLD gate** blocks `SATURATED` grow during a
post-shrink stabilization window so a re-saturation blip cannot
immediately re-grow the stage.

## Problem

After Phase B / Phase C stages an add, the actor publishes a READY
observation, then several more cycles pass before the dispatcher
fills its queue. During this window:

- `worker_groups[*].num_used_slots == 0` is a measurement artefact,
  not a real "over-provisioned" signal.
- A raw EWMA that absorbs those samples drags the running average
  toward the over-provisioned band; the classifier flips
  `OVER_PROVISIONED`, the streak fires, and Phase D shrinks the
  worker we just added on the very next cycle.
- If the stage was classified `SATURATED` immediately before the
  blip, the capacity sizer would re-add the workers we just removed
  on the very next cycle, oscillating the cluster.

A single global "ignore warmup workers" knob over-fits one of these
mechanisms. The scheduler protects each consumer of the warmup
signal — the EWMA, donor selection, and post-shrink stabilization —
with its own grace policy.

## Decision

Three independent layers, each scoped to one mechanism and each
configurable per stage. Both grace knobs are wall-clock seconds
because cycles can be uneven (catch-up loops, planner stalls,
operator-driven interval changes); a "ready age in cycles" would
drift across those.

```
                   cycle   1     2     3     4     5   ...   N
                           │     │     │     │     │         │
   EWMA               ▓▓▓▓▓ ▓▓▓▓▓ ░░░░░ ░░░░░ ░░░░░ ░░░░░ ░░░░░
   contribution       └─── excluded ──┘└────── steady-state ────┘
                       (age < worker_warmup_measurement_grace_s)

   Donor / victim     ▓▓▓▓▓ ▓▓▓▓▓ ▓▓▓▓▓ ▓▓▓▓▓ ░░░░░ ░░░░░ ░░░░░
   eligibility        └─────── excluded ─────┘└── eligible ────┘
                       (age < donor_warmup_grace_s; saturation-mode
                        donor + Phase D shrink only)

   Growth mode         A     A     A     A     A     A   ...  T
   (per stage)         └──── ACQUIRING (no shrink yet) ────┘└─ TRACKING ─┘
                                                            ^
                                                            first executed
                                                            shrink

   Legend: ▓ blocked / excluded   ░ unblocked / eligible
           A = GrowthMode.ACQUIRING   T = GrowthMode.TRACKING
```

Trade-off: a brief EWMA blind-spot at startup is preferred to a
false Phase D shrink of the worker just added. The cross-field
validator in
[`SaturationAwareStageConfig`](../../../cosmos_xenna/pipelines/private/specs.py)
enforces `donor_warmup_grace_s >= worker_warmup_measurement_grace_s`
so a worker is never donatable before its own stage's EWMA has had
a chance to absorb its mature contribution.

## How it works

### Layer 1 — per-worker measurement grace

`SaturationAwareScheduler._aggregate_slot_signals_excluding_warmup`
runs from `_compute_intent_deltas` before the EWMA absorbs the
cycle's slot signals. For each `worker_group` in the stage it
reads the worker's first-seen-READY timestamp from
`_worker_ready_first_seen_at` (refreshed once per cycle by
`_refresh_worker_ready_first_seen`); if
`now - first_seen < worker_warmup_measurement_grace_s`, the worker's
used/empty slots are excluded from the aggregate. The filtered
`(num_used, num_empty)` pair is what the EWMA absorbs.

If every worker is still in warmup, the helper returns `(0, 0)` and
`_resolve_classifier_signal` carries the last valid EWMA forward,
holding classifier state constant until at least one worker matures.
`input_queue_depth` is a stage-level signal and is not filtered.

### Layer 2 — donor warmup grace

`SaturationAwareScheduler._build_donor_warmup_excluded_ids` runs
once per cycle after the planning context is built, producing a
frozen set of every worker whose ready-age is below its stage's
`donor_warmup_grace_s`. Two consumers read the cached set:

- `find_saturation_donor` (the saturation-mode cross-stage donor):
  excludes warmup workers from the donor pool so a worker that has
  not yet contributed a mature sample cannot be sacrificed.
- `select_workers_to_remove_oldest_first` (Phase D shrink): excludes
  warmup workers from the victim pool before the consolidation-aware
  ordering runs.

Floor-mode donor selection (`select_youngest_eligible_donor`, called
from `_run_phase_b_floor`) **bypasses** this layer. A floor miss is
a hard structural failure — deadlocking the cluster on
warmup-protected donors is worse than killing a young donor.

### Layer 3 — growth-mode HOLD gate

`compute_growth_mode_transition` advances the per-stage lifecycle
seeded with `GrowthMode.ACQUIRING`. The mode stays put on any
`delta_executed >= 0` cycle. On the first cycle whose
`delta_executed < 0` — i.e. Phase D actually shrunk the stage,
which only happens after an `OVER_PROVISIONED` streak fires — the
mode transitions to `TRACKING`. Subsequent shrinks transition
`TRACKING -> HOLD`, which blocks `SATURATED` grow for
`stabilization_window_cycles_down` cycles before returning to
`TRACKING`. `SATURATED_CRITICAL` is never blocked, so a true burst
can still grow during the cool-down window.

The grow magnitude itself is the closed-form capacity gap from
[`28-capacity-sizer.md`](28-capacity-sizer.md), bounded by
`aggressive_growth_max_per_cycle`. The growth mode is consumed by
`compute_delta` as a binary gate on top of the sizer; only `HOLD`
has any effect, and only on `SATURATED` grow. See
[11-growth-mode-state-machine.md](11-growth-mode-state-machine.md)
for the full transition diagram.

## Knobs

| Knob | Default | Purpose |
|---|---|---|
| `worker_warmup_measurement_grace_s` | `60.0` | Layer 1: seconds a freshly-READY worker is excluded from EWMA contribution. Setting to `0` disables Layer 1. |
| `donor_warmup_grace_s` | `180.0` | Layer 2: seconds a freshly-READY worker is excluded from saturation-mode donor and Phase D victim selection. Cross-field: must be `>= worker_warmup_measurement_grace_s`. Setting to `0` disables Layer 2. |
| `aggressive_growth_max_per_cycle` | `4` | Layer 3: hard cap on any single-cycle scale-up; bounds the blast radius when capacity demand jumps. |
| `stabilization_window_cycles_down` | (see config) | Layer 3 follow-on: cycles in `HOLD` before returning to `TRACKING`; tunes post-shrink stabilization. |

Floor-mode donor selection always ignores Layer 2 regardless of the
configured `donor_warmup_grace_s`.

## See also

- [11 — Growth-mode state machine](11-growth-mode-state-machine.md)
  — the full `ACQUIRING -> TRACKING -> HOLD` transitions and the
  grow-gate effect table.
- [14 — Worker age tracking](14-worker-age-tracking.md) — planner
  ages (cycles since add) vs. ready first-seen timestamps (seconds
  since READY), and why the warmup graces use the latter.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the
  saturation-mode donor protocol that consumes Layer 2.
- [15 — Idle-first scale-down](15-idle-first-scale-down.md) — the
  Phase D victim ordering that consumes Layer 2.
- [05 — State classifier](05-state-classifier.md) — the EWMA-based
  classifier that consumes Layer 1.
