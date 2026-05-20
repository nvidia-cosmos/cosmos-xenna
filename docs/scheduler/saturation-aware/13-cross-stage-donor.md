# 13 — Cross-Stage Donor Protocol

## TL;DR

When the cluster is full but a stage still needs to grow, the scheduler
picks a worker off an over-provisioned **donor** stage, removes it from
the planner, and reuses the freed placement for the **receiver**. Two
modes share the same primitive: **floor mode** in Phase B (hard
structural requirement, runs under a grace window) and **saturation
mode** in Phase C (five independent anti-flap layers, runs only when
the classifier is confident). Donor-floor preservation is
non-negotiable in both modes.

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
introduces a third failure: two stages whose load oscillates around
the classifier boundary will rotate the same worker between them on
consecutive cycles, paying every worker startup cost for no net
allocation change.

## Decision

Share a single donor primitive across both modes, but gate it with
mode-appropriate eligibility filters:

- **Floor mode** (Phase B) gates lightly: any non-receiver stage that
  can spare one worker without breaching its own floor is eligible,
  with upstream donors preferred. The floor is a hard requirement,
  so we accept the rebalance cost; `floor_stuck_grace_cycles` absorbs
  the (rare) transient capacity miss.
- **Saturation mode** (Phase C) gates heavily: five independent
  anti-flap layers stack on top of the donor-floor rule. The
  receiver's growth signal is statistical, so the bar to evict
  another stage's worker is correspondingly high.
- Both modes select the **youngest** eligible worker (age ascending,
  worker id ascending as deterministic tie-break), so warm/expensive
  workers are preserved (see [14-worker-age-tracking.md](14-worker-age-tracking.md)).
- The cooldown bookkeeping (`_last_donation_cycle`,
  `_donations_received_this_cycle`) is advanced **only after** the
  post-donation `try_add_worker` retry succeeds. A donation that
  removes the donor but cannot place the receiver does not consume
  cooldown budget.

Trade-off: the saturation-mode gates make a steady-state rebalance
slow (default values cap each stage at one cross-stage donation per
~30 cycles, see Knobs). Floor mode trades a different cost — a
post-donation retry miss in floor mode raises `RuntimeError`
immediately because the donor removal cannot be safely rolled back.

## How it works

The donor-selection helpers live in
[`scheduling_py/donor.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/donor.py)
(`select_youngest_eligible_donor` for floor mode,
`find_saturation_donor` for saturation mode). The receiver-side
orchestration lives in
[`scheduling_py/saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
(`_run_phase_b_floor`, `_run_phase_c_grow`,
`_attempt_cross_stage_donation`, `_record_donation_success`).

The decision flow for "is donor X eligible to feed receiver Y this
cycle?" is the same five-layer funnel in saturation mode every time:

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
        │  Layer 4: receiver per-cycle absorption cap         │
        │  donations_received_this_cycle[receiver]            │
        │      < cross_stage_donor_max_per_cycle              │
        └─────────────────────────────────────────────────────┘
                                  │ pass             │
                                  ▼                  │
        ┌─────────────────────────────────────────────────────┐
        │  Layer 5: donor cooldown                            │
        │  cycle − last_donation_cycle[donor]                 │
        │      ≥ cross_stage_donor_min_donation_interval_*    │
        └─────────────────────────────────────────────────────┘
                                  │ pass             │
                                  ▼                  ▼
        ┌─────────────────────────────────────────────────────┐
        │  pick min(candidates, key=(age, worker_id))         │
        │  ─ floor mode prefers upstream donors when any      │
        │    upstream candidate exists                        │
        │  ─ saturation mode skips workers in the donor       │
        │    warmup-excluded set                              │
        └─────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │  ctx.try_remove_worker(donor)                       │
        │  ctx.try_add_worker(receiver)                       │
        │                                                     │
        │  on retry success only:                             │
        │    _last_donation_cycle[donor] = cycle              │
        │    _donations_received_this_cycle[receiver] += 1    │
        │                                                     │
        │  on retry FAIL:                                     │
        │    saturation mode → warn, move on (no rollback)    │
        │    floor mode      → raise RuntimeError immediately │
        └─────────────────────────────────────────────────────┘
```

Floor mode also tracks a per-stage `_floor_stuck_counters` map. A
cycle where no donor is eligible *and* the receiver made no progress
increments the counter; any progress (direct add or donation) resets
it. The counter exceeding `floor_stuck_grace_cycles` raises
`RuntimeError` — the operator must lower the floor or scale the
cluster.

At default values (`cross_stage_donor_anti_flap_cycles = 30`,
`cross_stage_donor_min_donation_interval_cycles = 30`,
`cross_stage_donor_max_per_cycle = 1`), each stage can donate at
most once per 30 cycles and receive at most once per 30 cycles. The
cluster as a whole can still see multiple donations per cycle, but
no single stage pair can oscillate.

The cross-field validator on `SaturationAwareConfig` enforces
`cross_stage_donor_anti_flap_cycles >= max(over_provisioned_streak_min_cycles)`
across every effective stage config — without that invariant a
freshly-OVER_PROVISIONED stage could donate, receive on the next
cycle, and donate again before its classifier streak resets.

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
| `cross_stage_donor_max_per_cycle`                  | `1`     | Layer 4: max donations a single receiver absorbs in one cycle.                           |
| `cross_stage_donor_min_donation_interval_cycles`   | `30`    | Layer 5: donor-side cooldown between consecutive donations.                              |
| `floor_stuck_grace_cycles`                         | `6`     | Floor mode only: consecutive no-progress cycles tolerated before `RuntimeError`.         |

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
