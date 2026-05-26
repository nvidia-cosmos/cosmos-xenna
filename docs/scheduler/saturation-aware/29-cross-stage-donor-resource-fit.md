# 29 — Cross-Stage Donor Resource-Fit and Commit Gate

## TL;DR

Once the eligibility filters in
[13-cross-stage-donor.md](13-cross-stage-donor.md) build a candidate
pool, the **resource-fit search** asks the Rust planner which subset
of those candidates would actually unblock the receiver's worker
shape. The first feasible plan wins (smallest plan_size first,
deterministic tiebreak). In saturation mode the chosen plan is then
run through a **throughput-first economic gate**: if the plan
regresses cluster throughput, flips a donor into the new bottleneck,
fails the spread / signal-trust thresholds, or (when throughput is
tied) regresses balance beyond tolerance, the plan is rejected and
no commit happens. Floor mode skips the economic gate — floor
enforcement is non-negotiable. Every accept and every reject emits
exactly one structured log line.

## Vocabulary

This doc references the per-stage Forced Flow Law metrics that
[23 — Bottleneck score metric](23-bottleneck-score-metric.md)
defines and
[25 — Bottleneck decision integration](25-bottleneck-decision-integration.md)
plumbs through Phase C / Phase D. Short form:

| Symbol         | Meaning                                                                                                                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `S_k`          | Per-stage mean per-task service time in seconds. EWMA-smoothed across cycles; held fixed across the gate's post-plan simulation because a single cycle's plan cannot retroactively change it.      |
| `c_k`          | Per-stage effective ready capacity — count of non-warmup ready actors. Every donor commit subtracts from one stage's `c_k` and adds to the receiver's.                                             |
| `D_k`          | Per-stage actor-normalized service demand: `D_k = S_k / c_k`. The cluster bottleneck is `argmax_k D_k`. Pipeline throughput bound = `1 / max_k D_k`.                                                |
| `max(D_k)`     | Maximum `D_k` across stages with finite positive `D_k` this cycle. Drives the throughput estimate and the donor-flip guard.                                                                        |
| `median_D_k`   | Median `D_k` across stages with finite positive `D_k`. The bottleneck-severity term in receiver value is `D_k − median_D_k`.                                                                       |
| `balance_score` | `1.0 / max(1.0, max_k D_k / min_k D_k)`. Score = 1 means perfectly balanced; scores approaching 0 mean a single stage dominates. Observed on the `xenna_scheduler_pipeline_balance_score` gauge.   |

The gate's post-plan simulation (`_compute_post_plan_d_k`) holds
`S_k` fixed, recomputes `c_k` per affected stage from the planned
removals, and divides to produce the post-plan `D_k` mapping.

## Problem

The eligibility filters tell the planner *which donor stages are
allowed*; they do not tell the planner *whether the donation
actually helps*. Three concrete failures show up when the gate is
absent:

- **Shape mismatch**: a CPU-only donor's freed slot does not satisfy
  a whole-GPU receiver shape. The donor disappears, the receiver
  still cannot fit, and the planner is left with a one-sided shrink.
- **Throughput regression**: a small over-provisioned donor stage
  with two channels donates one — its `D_k` doubles (because `D_k =
  S_k / c_k` and `c_k` halves; see
  [23 — Bottleneck score metric](23-bottleneck-score-metric.md)).
  If the donor's post-plan `D_k` exceeds the previous cluster
  bottleneck `max(D_k)`, the donor flips into the new bottleneck
  and pipeline throughput (`1 / max_k D_k`) drops even though the
  balance score numerically improves.
- **Noisy classifier**: a donor stage whose classifier oscillates
  in and out of OVER_PROVISIONED clears the streak gate by accident
  on a noisy cycle. Without trust gating, a flickering classifier
  drives flicker donations.

A single resource-fit search + a throughput-first economic gate
addresses all three: the search consults the Rust planner for shape
feasibility, the gate runs marginal-value math against the
post-plan `D_k` simulation, and the signal-trust gate clamps noisy
donors out before they can drive a probe.

## Decision

Two helpers in
[`scheduling_py/donor.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/donor.py)
own the resource-fit + gate path:

- `_resource_fit_plan(...)`: bounded multi-donor search using
  `ctx.probe_add_after_removals` as the feasibility oracle. Shared
  by floor mode and saturation mode. Returns the smallest feasible
  `DonorPlan`, or `None`.
- `_evaluate_economic_gate(...)`: throughput-first commit gate that
  composes four pure helpers (`_donor_cost`, `_receiver_value`,
  `_signal_trust`, `_compute_post_plan_d_k`) and returns a
  `_GateResult` carrying every metric the decision log surfaces.
  Saturation mode only — floor mode skips it.

The donor commit path is atomic across multi-worker plans:

```
            select_youngest_eligible_donor / find_saturation_donor
                              │
                              ▼
                    _resource_fit_plan
                              │
                  ┌───────────┴───────────┐
                  │ floor mode            │ saturation mode
                  ▼                       ▼
        DonorPlan (skip gate)   _evaluate_economic_gate
                  │                       │
                  │                       ├── throughput non-regression
                  │                       ├── donor-flip guard
                  │                       ├── spread ≥ threshold
                  │                       ├── signal-trust ≥ min
                  │                       └── balance (only on tie)
                  │                       │
                  │           ┌───────────┴───────────┐
                  │           │ accept                │ reject
                  │           ▼                       ▼
                  ▼     DonorDecision           one DEBUG line
        ctx.probe_add_after_removals (defence-in-depth re-probe)
                              │
                              ▼
        ctx.remove_workers_atomically(plan.removals)
                              │
                              ▼
        ctx.try_add_worker(receiver)
                              │
                              ▼
        on success: _record_donation_success(plan)
                    + one INFO commit log line
                    + advance _last_donation_cycle for every
                      distinct donor stage in plan.removals
```

## How it works

### Resource-fit search

`_resource_fit_plan` iterates `plan_size` from 1 to
`cross_stage_donor_max_plan_size`. At each size:

1. Group candidates by node. Same-node combinations are probed
   first (whole-GPU/SPMD receivers are more likely to fit when
   freed donors share a node); cross-node combinations follow.
2. Cap evaluations at `cross_stage_donor_max_plan_combinations` per
   `plan_size` so a pathological cluster cannot blow up cycle time.
3. For each combination, call
   `ctx.probe_add_after_removals(removals, receiver_stage_index)`.
   The Rust planner clones the working cluster, simulates the
   removals, and runs the same FGD / SPMD allocator that
   `try_add_worker` would consult on the live cluster.
4. Return the first feasible plan. Within a fixed `plan_size`,
   `itertools.combinations` over a list pre-sorted on
   `(age ASC, worker_id ASC, stage_index ASC)` produces the
   lexicographic deterministic tiebreak for free.

The Rust planner is the single source of truth for placement
feasibility — the helper does no shape arithmetic of its own. This
eliminates the duplication risk of re-implementing FGD / SPMD reuse
rules in Python.

### Throughput-first economic gate (saturation mode only)

`_evaluate_economic_gate` runs five checks in order, short-
circuiting at the first failure:

1. **Signal trust per donor**: `_signal_trust = min(streak, cap) /
   (1 + classifier_signal_noise_ewma)` must be at or above
   `cross_stage_donor_min_trust` for every donor stage in the plan.
2. **Spread threshold**: `spread = receiver_value - donor_cost`
   must be at or above `cross_stage_donor_spread_threshold`.
3. **Throughput non-regression**: `throughput_after =
   1 / max(post_plan_D_k)` must not regress beyond
   `cross_stage_donor_throughput_tolerance`.
4. **Donor-flip guard**: no donor stage's post-plan `D_k` may
   exceed the pre-plan `max(D_k) +
   cross_stage_donor_donor_flip_tolerance`. A donor that flips
   above the cluster's previous bottleneck creates a worse
   imbalance than the original.
5. **Balance regression** (only when throughput is tied within
   tolerance): `balance_after = 1 / max(1, max_k D_k / min_k D_k)`
   must not drop beyond `cross_stage_donor_balance_tolerance` (see
   the Vocabulary above for the `balance_score` definition).

Cold-start cycles (fewer than two stages with finite `D_k`) make
the throughput / donor-flip / balance comparisons NaN-tolerant and
short-circuit to "no regression". The signal-trust and spread
gates still apply.

### Marginal-value scoring

```
donor_cost = slots_empty_ratio_ewma * num_workers
             - cross_stage_donor_streak_bonus
                 * min(streak, cross_stage_donor_streak_cap)

receiver_value = pressure_ewma * num_workers
                 + cross_stage_donor_bottleneck_weight
                     * (D_k - median_D_k)
                 + cross_stage_donor_intent_weight * intent

spread         = receiver_value - donor_cost
signal_trust   = min(classifier_streak, trust_streak_cap)
                 / (1 + classifier_signal_noise_ewma)
```

The cost formula rewards stages with sustained idle slots (long
OVER_PROVISIONED streak) so stable donors are preferred. The value
formula pulls demand toward stages whose `D_k` exceeds the cluster
median (real bottlenecks), and breaks ties between similarly
saturated stages by their declared Phase C intent.

### Atomicity contract

Both modes share the same commit sequence:

1. `_resource_fit_plan` selects a plan via the planner's probe.
2. `_attempt_cross_stage_donation` runs a defence-in-depth re-probe
   right before commit; a concurrent cluster mutation between
   selection and commit could invalidate the result, and the repeat
   probe is cheap (cloned cluster, no mutation). A failure here
   surfaces as a logged DEBUG miss with
   `reject_reason="resource_fit"` and the Rust planner's
   `placement_reject_reason`.
3. `ctx.remove_workers_atomically(plan.removals)` commits every
   removal in a batch with rollback. A `False` return after
   pre-validation reported every donor as present is a
   `SchedulerInvariantError` — planner-snapshot divergence is a
   scheduler defect, not a benign cluster-full event.
4. `ctx.try_add_worker(receiver)` retries placement.
5. `_record_donation_success(plan)` advances
   `_last_donation_cycle` for every distinct donor stage in
   `plan.removals` exactly once.

The `_record_donation_success` deduplication matters for
multi-worker plans that draw multiple workers from the same donor
stage: a single successful donation must not bypass the receiver's
anti-flap window on subsequent cycles.

### Balance gauge + regression invariant

`xenna_scheduler_pipeline_balance_score = 1 / max(1, max(D_k) / min(D_k))`
is observed once per cycle alongside the heterogeneity ratio. The
gauge value is the cycle's measured `D_k` snapshot — operators see
"current cluster balance" graphed alongside heterogeneity.

A separate end-of-cycle invariant compares `balance_score_start`
(captured right after `identify_bottleneck`) to a simulated
`balance_score_end` derived from the planner's post-Phase-D worker
counts (intrinsic `S_k` held fixed). When the drop exceeds
`cross_stage_donor_balance_regression_tolerance`, one WARN log
fires. The WARN does NOT raise — balance is a secondary objective
behind the throughput gate the donor planner already enforces.

## Knobs

All resource-fit and economic-gate knobs live on
[`SaturationAwareConfig`](../../../cosmos_xenna/pipelines/private/specs.py)
(cluster-wide; no per-stage override).

| Field                                              | Default | Effect                                                                                          |
| -------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `cross_stage_donor_max_plan_size`                  | `4`     | Bounded multi-donor search width. Floor mode also honours this cap.                             |
| `cross_stage_donor_max_plan_combinations`          | `32`    | Per-`plan_size` cap on probes evaluated. Bounds search depth on pathological clusters.          |
| `cross_stage_donor_streak_bonus`                   | `0.05`  | Discount applied to donor cost per cycle of OVER_PROVISIONED streak (clamped at `streak_cap`).  |
| `cross_stage_donor_streak_cap`                     | `60`    | Caps the streak input to donor cost so a long streak cannot dominate the score.                 |
| `cross_stage_donor_bottleneck_weight`              | `1.0`   | Weight on `(D_k − median_D_k)` in receiver value; pulls demand toward severe bottlenecks.       |
| `cross_stage_donor_intent_weight`                  | `0.5`   | Weight on `receiver_intent` in receiver value; breaks ties between similarly saturated peers.   |
| `cross_stage_donor_spread_threshold`               | `0.5`   | Minimum `receiver_value − donor_cost` required to commit a donation.                            |
| `cross_stage_donor_throughput_tolerance`           | `0.01`  | Maximum allowed regression in pipeline throughput estimate before the gate rejects.             |
| `cross_stage_donor_donor_flip_tolerance`           | `0.10`  | Maximum amount a donor stage's post-plan `D_k` may exceed the pre-plan `max(D_k)`.              |
| `cross_stage_donor_balance_tolerance`              | `0.05`  | Maximum balance regression allowed when throughput is tied within tolerance.                    |
| `cross_stage_donor_balance_regression_tolerance`   | `0.05`  | End-of-cycle balance regression threshold for the soft WARN. Does NOT raise.                    |

The signal-trust gate's knobs (`cross_stage_donor_min_trust`,
`cross_stage_donor_trust_streak_cap`,
`classifier_signal_noise_smoothing_level`) live with the
eligibility-filter knobs in
[13-cross-stage-donor.md](13-cross-stage-donor.md).

## See also

- [13-cross-stage-donor.md](13-cross-stage-donor.md) — the four-layer
  eligibility-filter funnel that builds the candidate pool this doc
  consumes.
- [03-planning-context.md](03-planning-context.md) — the
  `AutoscalePlanContext` Rust API the resource-fit search probes
  through (`probe_add_after_removals`,
  `remove_workers_atomically`).
- [22-prometheus-metrics.md](22-prometheus-metrics.md) — the
  observability gauges, including
  `xenna_scheduler_pipeline_balance_score`.
- [23-bottleneck-score-metric.md](23-bottleneck-score-metric.md) —
  the `D_k` definition and the heterogeneity ratio the gate's
  median calculation reads.
- [24-structured-logging.md](24-structured-logging.md) — the
  per-decision INFO logging contract this gate extends.
- [25-bottleneck-decision-integration.md](25-bottleneck-decision-integration.md)
  — Phase C grow priority and Phase D shrink protection driven by
  the same `D_k` snapshot the gate uses.
