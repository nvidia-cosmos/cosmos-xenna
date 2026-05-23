# 09 — Regime-Aware Aggressiveness

## TL;DR

Detect which heavy-traffic regime the cluster is in each cycle —
**sub-Halfin-Whitt** (slack headroom) or **super-Halfin-Whitt**
(packed close to capacity) — and lift the effective
`saturation_aggressiveness` by
`super_halfin_whitt_aggressiveness_lift` while the cluster sits in
the packed regime. An asymmetric hysteresis
(`regime_transition_streak_cycles` cycles on entry, the same streak
above a wider exit band on exit) keeps boundary noise from flapping
the decision.

## Problem

The autoscaler pays a different price for the same wrong move
depending on cluster pressure:

- **Sub-Halfin-Whitt** (comfortable headroom). A wrong scale-down
  recovers next cycle — the freed slot is still available when
  the classifier wakes up.
- **Super-Halfin-Whitt** (packed close to capacity). A freed slot
  is likely claimed by another stage's growth before its original
  owner notices the loss, and reclaiming it can cost up to
  `log(N)` cycles of throughput while the donor cascade plays out.

Using one aggressiveness for both regimes is wrong in both
directions. Too low under-fits the packed regime (sluggish
scale-up while throughput is on the floor); too high over-fits the
slack regime (extra donor churn for no gain). Heavy-traffic theory
(Halfin & Whitt 1981) names the dividing threshold on the cluster's
empty-slot fraction; applying it lets the scheduler use the cheap
response when slack is available and the stronger one when it is
not.

## Decision

Run a two-state machine on top of a single cluster-wide signal,
re-derive the effective aggressiveness from its state each cycle,
and feed the result through to the auto-derived thresholds step so
the classifier fires SATURATED earlier whenever the packed regime
holds:

```
                       entry: cluster_idle < threshold
                              for streak_cycles cycles
                       ─────────────────────────────────►
   ┌────────────────────┐                                 ┌──────────────────────┐
   │ SUB_HALFIN_WHITT   │                                 │ SUPER_HALFIN_WHITT   │
   │ (base agg)         │                                 │ (base + lift)        │
   └────────────────────┘                                 └──────────────────────┘
                       ◄─────────────────────────────────
                       exit: cluster_idle ≥ threshold * 1.5
                              for streak_cycles cycles

   ┌─────────────────────────────────────────────────────────────────┐
   │ cluster_idle fraction                                           │
   │                                                                 │
   │  0  ─────────●──────────●────────────────────────────────►   1 │
   │           threshold     threshold * 1.5                         │
   │           (enter edge)  (exit edge)                             │
   │              ◄── flap-zone, state unchanged ──►                 │
   └─────────────────────────────────────────────────────────────────┘
```

**Trade-off.** The lift trades a small amount of stability for
noticeably faster recovery when the cluster is packed — a stage
may briefly overshoot by one worker on the transition cycle, which
Phase D consolidates the following cycle. The exit band is
asymmetric on purpose: dropping the lift while the cluster is still
packed is the more expensive mistake (the cluster re-enters
super-Halfin-Whitt within a few cycles and pays the `log(N)`
cascade again), so the detector requires the same streak length
above a wider band before it lets the lift go.

## How it works

```
problem_state ─► _aggregate_cluster_regime_signal  ─► RegimeSignal
                                                       │
                                                       ▼
                                              update_regime_state
                                              (hysteresis + streak)
                                                       │
                      transitioned?  yes  ◄────────────┤
                          │                  no  ─► return
                          ▼
              drop resolved_thresholds for every stage
              reset classifier_state + classifier_streak
              reset valid_signal_samples (trust-gate counter)
              clear stabilization recommendation buffer
                          │
                          ▼
              _ensure_thresholds_resolved re-derives this cycle
              with _effective_aggressiveness(base) reflecting the
              new regime
```

1. **Signal.** `_aggregate_cluster_regime_signal` sums
   `num_used_slots` and `num_empty_slots` across every stage and
   delegates to `compute_regime_signal`. The cycle's
   `threshold = 1 / sqrt(total_workers)` (clamped to `1.0` when
   `total_workers <= 1`), and
   `cluster_idle_fraction = total_empty / (total_used +
   total_empty)`.

2. **Signal-availability guard.** When `total_used + total_empty
   == 0` — or any active worker stage still carries the `0/0`
   no-signal sentinel — the signal is flagged unavailable and
   `update_regime_state` leaves the hysteresis state untouched.
   This prevents an empty-signal cycle from reading as "comfortable
   headroom" and silently dropping the lift while slot telemetry is
   briefly unpopulated.

3. **Asymmetric hysteresis.** From `SUB_HALFIN_WHITT`, enter
   `SUPER_HALFIN_WHITT` only after
   `regime_transition_streak_cycles` consecutive cycles whose
   signal is below `threshold`. From `SUPER_HALFIN_WHITT`, exit
   only after the same streak at or above
   `threshold * EXIT_BAND_MULTIPLIER` (1.5). The streak counter
   resets on any disagreeing cycle and never crosses an
   unavailable signal.

4. **On transition.** `_update_regime_aware_aggressiveness` drops
   every stage's `resolved_thresholds`, resets each stage's
   threshold-relative classifier state and streak, resets each
   stage's `valid_signal_samples` trust-gate counter to zero, and
   clears the stabilization-window recommendation buffer. The
   trust-gate reset forces the post-transition cycles to re-accrue
   `min_data_points` consecutive warmup-excluded valid samples
   before a non-zero recommendation may pass; without it the gate
   would carry forward freshness consensus built against the old
   threshold band. `_ensure_thresholds_resolved` re-derives
   thresholds with the new `_effective_aggressiveness` value on
   the same cycle, so the new bias takes effect immediately rather
   than next cycle.

5. **Lift application.** `_effective_aggressiveness(base)` returns
   `base + super_halfin_whitt_aggressiveness_lift` while
   `current_regime is SUPER_HALFIN_WHITT`, otherwise returns
   `base`. The auto-derived thresholds step consumes the result.

6. **Observability.** `_log_regime_transition` emits one INFO line
   per transition reporting the new regime, `total_workers`,
   `cluster_idle_fraction`, `threshold`, and the resulting
   `effective_aggressiveness`.

## Knobs

All on `SaturationAwareConfig` in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Field | Default | Notes |
|---|---|---|
| `enable_regime_aware_aggressiveness` | `True` | Master switch. `False` pins the effective aggressiveness at base and skips regime tracking entirely (useful for A/B comparisons against the queueing-theory-pure baseline). |
| `super_halfin_whitt_aggressiveness_lift` | `0.15` | Additive lift applied to base aggressiveness in super-Halfin-Whitt. Range `[0.0, 0.5]`. `0.0` keeps the regime tracker live but neutralises the numeric lift. |
| `regime_transition_streak_cycles` | `3` | Consecutive cycles required to commit a transition in either direction. Combined with `interval_s`, sets the minimum reaction time to a regime change. |

`EXIT_BAND_MULTIPLIER` (`1.5`) is fixed in
[`regime.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/regime.py)
and not surfaced as a knob — the wider exit band is part of the
contract, not a tuning parameter.

## See also

- [00 — Per-cycle overview](00-overview.md) — where the regime
  detector runs inside the pre-flight stage.
- [08 — Auto-derived thresholds](08-auto-derived-thresholds.md) —
  consumes the effective aggressiveness to shape the SATURATED and
  ACTIVATION thresholds.
- [07 — Streak stabilization](07-streak-stabilization.md) — owns
  the recommendation buffer that is cleared on every regime
  transition.
- [05 — State classifier](05-state-classifier.md) — the consumer
  that reads the re-derived thresholds each cycle.
- [`regime.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/regime.py)
  — pure-function detector (`Regime`, `RegimeSignal`,
  `RegimeDetectorState`, `compute_regime_signal`,
  `update_regime_state`).
- [`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
  — `_aggregate_cluster_regime_signal`,
  `_update_regime_aware_aggressiveness`,
  `_effective_aggressiveness`, `_log_regime_transition`.
