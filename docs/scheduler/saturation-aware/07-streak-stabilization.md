# 07 — Streak Stabilization

## TL;DR

The slots-empty signal is single-sample noisy; acting on every cycle
flaps the worker count. The scheduler stabilises that signal in
three asymmetric layers — EWMA smoothing on the raw ratio, streak
counters that require a sustained classifier output before firing,
and a recommendation-history consensus window mirroring Kubernetes
HPA's `stabilizationWindowSeconds`. Scale-up fires after one or two
cycles; scale-down requires roughly thirty cycles of agreement. A
wrong scale-up costs a few extra workers briefly; a wrong scale-down
kills warm GPU state and pays minutes of model warmup again.

## Problem

The slots-empty ratio (`empty_slots / total_slots`) is the primary
saturation signal, but a single cycle can swing it dramatically: a
brief co-arrival of dispatches reads zero free slots, a single slow
task on one actor pins its slot for one cycle, a cluster-wide GC
pause looks like a saturation spike. If every cycle's raw ratio fed
straight into a classifier and into a scale action, the worker count
would oscillate. Each oscillation costs warm GPU state — new workers
must reload weights, rebuild VRAM caches, and replay an inference
warmup pass that can dominate cycle time. The scheduler must
therefore separate *signal* (a sustained shift in true saturation)
from *noise* (a single-cycle measurement excursion) before changing
the worker count.

## Decision

Stabilise the slots-empty signal in three complementary layers. Each
layer guards a different property; a recommendation must survive all
three before the scheduler grows or shrinks workers.

- **EWMA smoothing.** The raw ratio is exponentially smoothed by
  `slots_empty_ratio_smoothing_level` (default `0.20`, ~3-cycle
  half-life) before the classifier ever sees it. A single timescale
  reshapes amplitude without shifting the trend mean, so a sustained
  high or low ratio still classifies correctly while one-cycle
  excursions are damped.
- **Asymmetric streak counters.** `update_streak` counts consecutive
  cycles in the same classifier state; `should_fire_action` releases
  the action only when the streak meets a per-state threshold —
  `saturated_critical` after **1** cycle (burst), `saturated` after
  **2**, `over_provisioned` after **30**. The 1:15 up-to-down ratio
  encodes the operational cost asymmetry: a wrong scale-up briefly
  pays for a few unused workers, while a wrong scale-down pays full
  model warmup minutes later — and the scheduler may shrink again
  before the new workers are warm. The split is in the same spirit
  as TCP slow-start: ramp up freely, ramp down only after evidence.
- **Stabilization-window consensus.** A separate per-stage ring
  buffer records the *direction* (`+1 / 0 / -1`) of every cycle's
  recommended delta. `apply_stabilization_gate` releases a delta
  only when every cycle in the relevant window agrees on the same
  direction — `stabilization_window_cycles_up` (default `1`) for
  growth, `stabilization_window_cycles_down` (default `30`) for
  shrink. This mirrors Kubernetes HPA's
  `scaleDown.stabilizationWindowSeconds` and is independent of the
  streak machine: a single OVER_PROVISIONED cycle that satisfies
  the streak gate but flips back to NORMAL on the next cycle is
  still suppressed by the consensus window.

The classifier transitions are already discrete events, so an extra
MACD-style fast-vs-slow EMA crossover is not used: it would only
restate the streak property in a noisier form for no gain.

```
Scenario A — short saturation burst (2-cycle streak fires scale-up):

   ┌─────┬─────┬─────┬─────┬─────┐
   │  1  │  2  │  3  │  4  │  5  │   cycle
   ├─────┼─────┼─────┼─────┼─────┤
   │ SAT │ SAT │NORM │NORM │NORM │   classifier state
   ├─────┼─────┼─────┼─────┼─────┤
   │0.18 │0.16 │0.24 │0.31 │0.38 │   EWMA (slots-empty ratio)
   ├─────┼─────┼─────┼─────┼─────┤
   │  1  │ [2] │  1  │  2  │  3  │   streak
   └─────┴─────┴─────┴─────┴─────┘
                ▲
                fires (saturated_streak_min_cycles = 2)


Scenario B — sustained idle (30-cycle streak fires scale-down):

   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
   │  1  │  2  │ ... │ 28  │ 29  │ 30  │ 31  │   cycle
   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
   │ OP  │ OP  │ ... │ OP  │ OP  │ OP  │ OP  │   classifier state
   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
   │0.62 │0.65 │ ... │0.84 │0.85 │0.85 │0.85 │   EWMA
   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
   │  1  │  2  │ ... │ 28  │ 29  │[30] │ 31  │   streak
   └─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                                        ▲
                                        fires (over_provisioned_streak_min_cycles = 30)
```

## How it works

The per-stage decision pipeline (`run_per_stage_pipeline` in
[`scheduling_py/pipeline.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/pipeline.py))
composes the three layers in this order:

1. **Raw ratio + EWMA.** `compute_slots_empty_ratio` and
   `update_ewma` in
   [`scheduling_py/state.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/state.py)
   produce the smoothed signal
   `new = level * sample + (1 - level) * prev`. Cold-start seeds the
   EWMA with the first sample (no warmup tax); a stage with zero
   ready actors carries the previous valid EWMA forward, and the
   first cycle with no signal at all skips the per-stage pipeline
   entirely without disturbing the streak counters.
2. **Classify.** The classifier reads the EWMA and input queue
   depth and emits one of five `StageState` zones with hysteresis
   — see [05 — State classifier](05-state-classifier.md).
3. **Streak gate.** `update_streak` increments `classifier_streak`
   when the state holds, or resets to `1` on transition.
   `should_fire_action` then gates on the per-state minimum
   (`saturated_critical_streak_min_cycles`,
   `saturated_streak_min_cycles`,
   `over_provisioned_streak_min_cycles`,
   `starved_streak_min_cycles`). NORMAL never fires.
4. **Compute delta.** `compute_delta` produces the signed
   worker-count intent for the firing state; magnitude is shaped
   by the growth-mode state machine.
5. **Stabilization gate.** `apply_stabilization_gate` records the
   sign of the raw delta into the per-stage `_RecommendationHistory`
   ring buffer, then asks `gate_up_allowed` (last `window_up`
   entries all `+1`) or `gate_down_allowed` (last `window_down`
   entries all `-1`). When the gate refuses, the delta is replaced
   by `0` before the growth-mode transition runs, so HOLD timers
   advance correctly across suppressed cycles instead of resetting
   on every gated shrink.

The (possibly zeroed) delta then feeds Phase C (positive intents)
or Phase D (negative intents) of the per-cycle plan — see
[00 — Per-cycle overview](00-overview.md). Cluster-wide feasibility
is enforced by the planner phases after this function returns; the
three stabilization layers shape only the *intent*.

## Knobs

All four streak thresholds, both stabilization windows, and the EWMA
smoothing level live on `SaturationAwareStageConfig` (per stage,
via the three-tier resolver — see
[02 — Configuration model](02-configuration-model.md)).

| Field | Default | Purpose |
|---|---|---|
| `slots_empty_ratio_smoothing_level` | `0.20` | EWMA weight on the new sample (`alpha`); ~3-cycle half-life. `1.0` disables smoothing; `0.0` is rejected (would freeze the value). |
| `saturated_critical_streak_min_cycles` | `1` | Cycles in SATURATED_CRITICAL before the burst delta fires. |
| `saturated_streak_min_cycles` | `2` | Cycles in SATURATED before the ordinary scale-up delta fires. |
| `over_provisioned_streak_min_cycles` | `30` | Cycles in OVER_PROVISIONED before the scale-down delta fires. Cross-field invariant: must be strictly greater than `saturated_streak_min_cycles`. |
| `starved_streak_min_cycles` | `6` | Cycles in STARVED before the upstream-bottleneck warning is logged. |
| `stabilization_window_cycles_up` | `1` | Recommendation-history depth for scale-up. `1` means a single up cycle suffices. |
| `stabilization_window_cycles_down` | `30` | Recommendation-history depth for scale-down. At the default `interval_s = 10.0` autoscale period this is 5 minutes, matching Kubernetes HPA's `scaleDown.stabilizationWindowSeconds = 300`. Cross-field invariant: must be strictly greater than `stabilization_window_cycles_up`. |

The cross-field validators in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py)
enforce both asymmetry invariants at config-construction time so
an operator cannot accidentally configure a faster shrink than grow
signal.

## See also

- [00 — Per-cycle overview](00-overview.md) — where streak
  stabilization sits inside one autoscale call.
- [05 — State classifier](05-state-classifier.md) — the
  five-zone classifier whose output the streak counters track.
- [06 — Backlog-time signal](06-backlog-time-signal.md) — the
  compound AND criterion that supplements the slots-empty signal
  with queue-time evidence.
- [11 — Growth-mode state machine](11-growth-mode-state-machine.md)
  — the HOLD-after-shrink timer that runs after the stabilization
  gate and shapes the magnitude of any released delta.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the
  anti-flap layer at the next level up; its
  `cross_stage_donor_anti_flap_cycles` must dominate every stage's
  `over_provisioned_streak_min_cycles`.
