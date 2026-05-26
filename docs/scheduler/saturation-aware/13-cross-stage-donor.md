# 13 — Cross-Stage Donor Protocol

## TL;DR

When the cluster is full but a stage still needs to grow, the
scheduler picks a worker (or a small set of workers) off one or more
over-provisioned **donor** stages, removes them from the planner, and
reuses the freed placements for the **receiver**. Two modes share the
same primitive: **floor mode** in Phase B (hard structural
requirement, runs under a grace window) and **saturation mode** in
Phase C (four anti-flap layers + a throughput-first economic gate).
Donor-floor preservation is non-negotiable in both modes.

The bounded multi-donor search and the throughput-first commit gate
are documented in
[29-cross-stage-donor-resource-fit.md](29-cross-stage-donor-resource-fit.md).
This doc covers the eligibility filters that build the candidate
pool; the resource-fit search and economic gate then pick which
combination of those candidates actually commits.

## Problem

A streaming autoscaler that can grow on fresh placement but never
rebalance has two failure modes that show up in production:

- **Floor stuck**: a stage with `min_workers >= 1` has zero live
  workers because the cluster is fully booked by earlier-started
  stages. The classifier has no slot signal to drive, the pipeline
  cannot make forward progress, and no amount of waiting helps.
- **Saturation stuck**: a downstream bottleneck classifies SATURATED
  and wants `+N` workers, but an upstream stage that classified
  SATURATED earlier in the run is holding the cluster. The pipeline
  is throughput-bound on the wrong stage.

A naive "take a worker, give a worker" loop solves both — and then
introduces three new failures:

- **Flicker**: two stages whose load oscillates around the classifier
  boundary will rotate the same worker between them on consecutive
  cycles, paying every worker startup cost for no net allocation
  change.
- **Shape mismatch**: freeing a CPU-only worker does not actually
  unblock a whole-GPU receiver; the donor disappears but the receiver
  still cannot fit. (Production regression.)
- **Donor flip**: the freed donor stage now has too few workers and
  flips into the new bottleneck — throughput regresses while balance
  numerically improves.

## Decision

Share a single donor primitive across both modes, but stack three
mechanisms in saturation mode that floor mode skips:

1. **Eligibility filters** (this doc): cheap classifier-level checks
   that prune stages out of the candidate pool. Both modes apply
   donor-floor preservation; saturation mode also applies four
   anti-flap layers.
2. **Resource-fit search** (see
   [29](29-cross-stage-donor-resource-fit.md)): the bounded
   multi-donor planner that asks the Rust planner whether each
   candidate combination would actually unblock the receiver shape.
   Shared by both modes.
3. **Economic gate** (see
   [29](29-cross-stage-donor-resource-fit.md)): the throughput-first
   commit gate. Saturation mode runs it; floor mode skips it because
   floor enforcement is non-negotiable.

Both modes select the **youngest** eligible worker (age ascending,
worker id ascending as deterministic tie-break), so warm/expensive
workers are preserved (see
[14-worker-age-tracking.md](14-worker-age-tracking.md)).

The cooldown bookkeeping (`_last_donation_cycle`) is advanced **only
after** the post-donation `try_add_worker` retry succeeds, and is
advanced for **every distinct donor stage** in the committed plan.
A donation that removes the donor but cannot place the receiver
(probe-vs-actual race) does not consume cooldown budget.

## How it works

The donor-selection helpers live in
[`scheduling_py/donor.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/donor.py)
(`select_youngest_eligible_donor` for floor mode,
`find_saturation_donor` for saturation mode). The receiver-side
orchestration lives in
[`scheduling_py/saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
(`_run_phase_b_floor`, `_run_phase_c_grow`,
`_attempt_cross_stage_donation`, `_record_donation_success`).

The eligibility funnel for "is donor X eligible to feed receiver Y
this cycle?":

```
        receiver_needs +1 worker AND try_add_worker returned None
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │  Layer 0 (both modes): donor-floor preservation     │
        │  donor.stage_index ≠ receiver                       │
        │  len(donor_workers) − 1 ≥ stage_floors[donor]       │
        │  (sat mode only) strictly upstream when             │
        │     donor_must_be_strictly_upstream is True         │
        └─────────────────────────────────────────────────────┘
                                  │ pass
                                  ▼
                  saturation mode only ────────────┐
                                  │                │ floor mode
                                  ▼                │ skips layers
        ┌─────────────────────────────────────────────────────┐
        │  Layer 1: donor classifier                          │
        │  state = OVER_PROVISIONED AND                       │
        │  streak ≥ over_provisioned_streak_min_cycles        │
        │  (gated by cross_stage_donor_require_over_*)        │
        └─────────────────────────────────────────────────────┘
                                  │ pass             │
                                  ▼                  │
        ┌─────────────────────────────────────────────────────┐
        │  Layer 2: donor growth mode                         │
        │  growth_mode ≠ HOLD                                 │
        │  (gated by cross_stage_donor_exclude_hold_state)    │
        └─────────────────────────────────────────────────────┘
                                  │ pass             │
                                  ▼                  │
        ┌─────────────────────────────────────────────────────┐
        │  Layer 3: receiver post-donate cooldown             │
        │  cycle − last_donation_cycle[receiver]              │
        │      ≥ cross_stage_donor_anti_flap_cycles           │
        └─────────────────────────────────────────────────────┘
                                  │ pass             │
                                  ▼                  │
        ┌─────────────────────────────────────────────────────┐
        │  Layer 4: donor signal trust                        │
        │  min(streak, trust_streak_cap)                      │
        │      / (1 + classifier_signal_noise_ewma)           │
        │      ≥ cross_stage_donor_min_trust                  │
        └─────────────────────────────────────────────────────┘
                                  │ pass             │
                                  ▼                  ▼
        ┌─────────────────────────────────────────────────────┐
        │  candidate pool emitted to                          │
        │  _resource_fit_plan + (sat mode only) economic gate │
        │  ─ floor mode prefers upstream donors when any      │
        │    upstream candidate exists                        │
        │  ─ saturation mode skips workers in the donor       │
        │    warmup-excluded set                              │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        see 29-cross-stage-donor-resource-fit.md for the
        bounded multi-donor search, atomicity contract, and
        throughput-first commit gate.
```

Floor mode also tracks a per-stage `_floor_stuck_counters` map. A
cycle where no donor plan is feasible *and* the receiver made no
progress increments the counter; any progress (direct add or
donation) resets it. The counter exceeding `floor_stuck_grace_cycles`
raises `RuntimeError` — the operator must lower the floor or scale
the cluster.

The cross-field validator on `SaturationAwareConfig` enforces
`cross_stage_donor_anti_flap_cycles >= max(over_provisioned_streak_min_cycles)`
across every effective stage config — without that invariant a
freshly-OVER_PROVISIONED stage could donate, receive on the next
cycle, and donate again before its classifier streak resets.

The signal-trust gate (Layer 4) replaces the older fixed-cooldown
mechanism that used to live in this doc as Layers 4 and 5
(`cross_stage_donor_max_per_cycle` and
`cross_stage_donor_min_donation_interval_cycles`). Both knobs are
gone; per-cycle absorption is naturally bounded by the receiver's
Phase C intent (capped by `aggressive_growth_max_per_cycle`), and
the donor-side cooldown is subsumed by Layer 1 (require
OVER_PROVISIONED + full streak) plus Layer 4 (signal-trust on the
EWMA-smoothed classifier flicker).

## Knobs

All cross-stage donor knobs live on
[`SaturationAwareConfig`](../../../cosmos_xenna/pipelines/private/specs.py)
(cluster-wide; no per-stage override).

| Field                                              | Default | Effect                                                                                  |
| -------------------------------------------------- | ------- | --------------------------------------------------------------------------------------- |
| `enable_cross_stage_donor`                         | `True`  | Master toggle for both modes. `False` disables donor logic; floor enforcement may raise. |
| `donor_must_be_strictly_upstream`                  | `True`  | Saturation mode: reject donors at or downstream of receiver in DAG order.                |
| `cross_stage_donor_require_over_provisioned`       | `True`  | Layer 1: require donor classifier OVER_PROVISIONED + full streak.                        |
| `cross_stage_donor_exclude_hold_state`             | `True`  | Layer 2: reject donors whose growth mode is HOLD.                                        |
| `cross_stage_donor_anti_flap_cycles`               | `30`    | Layer 3: receiver-side cooldown after donating.                                          |
| `cross_stage_donor_min_trust`                      | `1.0`   | Layer 4: minimum `min(streak, cap) / (1 + noise_ewma)` a donor must clear.               |
| `cross_stage_donor_trust_streak_cap`               | `60`    | Layer 4: clamps the streak input to `signal_trust` so a long streak cannot dominate.     |
| `classifier_signal_noise_smoothing_level`          | `0.2`   | Layer 4: EWMA smoothing for the per-stage classifier-noise tracker.                      |
| `floor_stuck_grace_cycles`                         | `6`     | Floor mode only: consecutive no-progress cycles tolerated before `RuntimeError`.         |

The resource-fit search and economic-gate knobs
(`cross_stage_donor_max_plan_size`,
`cross_stage_donor_max_plan_combinations`,
`cross_stage_donor_streak_bonus`,
`cross_stage_donor_bottleneck_weight`,
`cross_stage_donor_intent_weight`,
`cross_stage_donor_streak_cap`,
`cross_stage_donor_spread_threshold`,
`cross_stage_donor_throughput_tolerance`,
`cross_stage_donor_donor_flip_tolerance`,
`cross_stage_donor_balance_tolerance`,
`cross_stage_donor_balance_regression_tolerance`)
are documented in
[29-cross-stage-donor-resource-fit.md](29-cross-stage-donor-resource-fit.md).

## See also

- [00-overview.md](00-overview.md) — where the donor protocol sits in
  the per-cycle phase pipeline (Phase B, Phase C).
- [11-growth-mode-state-machine.md](11-growth-mode-state-machine.md) —
  the `ACQUIRING / TRACKING / HOLD` machine that Layer 2 reads from.
- [14-worker-age-tracking.md](14-worker-age-tracking.md) — how worker
  ages are maintained so "youngest" is well-defined and the donor
  warmup grace can exclude freshly-warmed workers.
- [16-hard-caps-and-floors.md](16-hard-caps-and-floors.md) — the
  per-stage / per-node floor that Layer 0 must preserve.
- [19-phase-invariants.md](19-phase-invariants.md) — the structural
  invariants that run between Phases A → B → C → D and would catch
  a corrupted donation before the plan is emitted.
- [29-cross-stage-donor-resource-fit.md](29-cross-stage-donor-resource-fit.md)
  — bounded multi-donor resource-fit search, atomicity contract,
  throughput-first commit gate, marginal-value scoring, and the
  decision-log INFO/DEBUG schema.
