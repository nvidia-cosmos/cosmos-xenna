# 11 — Growth-Mode State Machine

## TL;DR

Each stage carries a **three-mode** lifecycle —
`ACQUIRING`, `TRACKING`, `HOLD` — that records whether the
stage has ever shrunk. The capacity sizer drives scale-up
magnitude; the lifecycle is consumed as a **binary HOLD gate**:
`HOLD` blocks `SATURATED` grow during a post-shrink
stabilization window so a re-saturation blip cannot immediately
re-grow the stage. `SATURATED_CRITICAL` is always allowed.
Transitions are evaluated against the **executed delta**, not
the classifier output, so a recommendation suppressed by the
stabilization gate does not change the mode.

## Problem

A re-saturation blip immediately after a shrink would let the
sizer re-add the workers we just removed, oscillating the
cluster. The scheduler needs a per-stage cool-down window that
suppresses non-critical grow for a bounded number of cycles
after each shrink, yet still lets genuine bursts through when
the classifier escalates to `SATURATED_CRITICAL`.

A purely capacity-driven sizer (no cool-down) would correctly
identify the demand on every cycle but cannot distinguish "this
demand is structural" from "this demand is a transient blip from
a worker that just died". A cool-down keyed on the **executed
delta** records that distinction without observing transient
demand directly.

## Decision

Track a per-stage three-state machine —
[`GrowthMode`](../../../cosmos_xenna/pipelines/private/scheduling_py/state.py)
`ACQUIRING` / `TRACKING` / `HOLD` — and consume it as a binary
grow gate inside
[`compute_delta`](../../../cosmos_xenna/pipelines/private/scheduling_py/decisions.py).
The pure-function transition rule lives in
[`compute_growth_mode_transition`](../../../cosmos_xenna/pipelines/private/scheduling_py/growth_mode.py).

- **`ACQUIRING`** (initial state, no shrink ever observed). Grow
  is allowed; magnitude is the capacity sizer's shortfall
  bounded by `aggressive_growth_max_per_cycle`.
- **`TRACKING`** (post-first-shrink, ceiling discovered). Same
  grow contract as `ACQUIRING` — the lifecycle simply records
  that at least one shrink has happened.
- **`HOLD`** (post-shrink stabilization). Blocks `SATURATED`
  grow for `stabilization_window_cycles_down` cycles;
  `SATURATED_CRITICAL` grow is always allowed so a real burst
  is not ignored. A re-shrink while in `HOLD` restarts the
  timer.

Crucially, transitions are evaluated against the **post-commit
executed delta** the cycle actually applied (see
[07 — Streak stabilization](07-streak-stabilization.md)), not
the classifier output and not the pre-commit recommendation. The
scheduler computes the executed delta as
`post_phase_d_count − pre_phase_c_count` for every stage that
participated in the cycle's intent computation, then calls
`record_executed_delta` once per stage. That separation matters
in three places:

1. **Stabilization gate suppression.** A recommended shrink that
   the asymmetric gate refused never landed in the cluster, so
   the executed delta is `0` and the stage stays in its current
   mode.
2. **Hard caps / fractional clamps.** A recommended `+5` that
   the per-stage hard worker cap throttled to `+2` advances the
   timer by `+2`, not by the recommended `+5`. Likewise, a
   recommended `-5` that `max_scale_down_fraction_per_cycle`
   clamps to `-2` is recorded as `-2`.
3. **Allocation failures.** A recommended `+3` that aborted
   after one successful add (with the rest of the cycle failing
   the absorb path) reflects `+1` into the timer, not the full
   `+3` recommendation.

The mode therefore stays consistent with what the workers
actually saw.

Every grow path also runs through the
`aggressive_growth_max_per_cycle` cap, so a single cycle's
shortfall cannot single-handedly overshoot the cluster.

**Trade-off.** One extra integer of per-stage state plus a
streak counter, plus one knob
(`stabilization_window_cycles_down`) operators must understand.
In exchange we get bounded oscillation after over-provisioning
without giving up burst response.

```
   delta_executed >= 0  —  grow / no-op
   delta_executed <  0  —  shrink (workers actually removed)


   ┌──────────────────┐
   │    ACQUIRING     │ ◄── grow / no-op:
   │    initial,      │     stay, streak += 1
   │    no shrink yet │
   └──────────────────┘
            │
            │  first shrink
            ▼
   ┌──────────────────┐
   │     TRACKING     │ ◄── grow / no-op:
   │     ceiling      │     stay, streak += 1
   │     observed     │
   └──────────────────┘
        ▲       │
        │       │  later shrink
        │       ▼
        │  ┌──────────────────┐
        │  │       HOLD       │ ◄── grow / no-op,
        │  │   post-shrink    │     window not expired:
        │  │   stabilization, │     stay, streak += 1
        │  │   SATURATED grow │
        │  │   blocked        │ ◄── re-shrink:
        │  └──────────────────┘     stay HOLD, streak = 1
        │       │                   (restart timer)
        │       │
        │       │  HOLD timer expires
        │       │  (streak ≥ stabilization_window_cycles_down)
        └───────┘
```

Grow-gate effect per `(mode, classifier)` cell. The magnitude
itself comes from the capacity sizer (see
[`28-capacity-sizer.md`](28-capacity-sizer.md)); the mode is a
binary gate on top of it:

| Mode        | `SATURATED`        | `SATURATED_CRITICAL` |
|---|---|---|
| `ACQUIRING` | grow allowed       | grow allowed         |
| `TRACKING`  | grow allowed       | grow allowed         |
| `HOLD`      | grow blocked (0)   | grow allowed         |

`OVER_PROVISIONED` shrink magnitude is independent of mode;
`NORMAL` always emits zero regardless of mode. The per-cycle
shortfall and excess against the capacity target are clamped by
`aggressive_growth_max_per_cycle` and
`max_scale_down_fraction_per_cycle` respectively.

## How it works

The cycle owner calls `compute_growth_mode_transition` once per
stage at the end of the per-stage decision pipeline, passing
the executed delta. The rule is exhaustive over
`(prev_mode, sign(delta_executed))` and the `HOLD` timer:

| `prev_mode`  | Condition                                                              | `next_mode` | `next_streak` |
|---|---|---|---|
| `ACQUIRING`  | `delta_executed < 0` (first shrink — ceiling discovered)               | `TRACKING`  | `1`           |
| `TRACKING`   | `delta_executed < 0` (later shrink — enter stabilization)              | `HOLD`      | `1`           |
| `HOLD`       | `delta_executed < 0` (re-shrink — restart timer)                       | `HOLD`      | `1`           |
| `HOLD`       | `delta_executed >= 0` and `streak >= stabilization_window_cycles_down` | `TRACKING`  | `1`           |
| `ACQUIRING`  | `delta_executed >= 0`                                                  | `ACQUIRING` | `streak + 1`  |
| `TRACKING`   | `delta_executed >= 0`                                                  | `TRACKING`  | `streak + 1`  |
| `HOLD`       | `delta_executed >= 0` and `streak <  stabilization_window_cycles_down` | `HOLD`      | `streak + 1`  |

The mode is consumed by `compute_delta` on the *next* cycle as
a binary grow-gate. There is no path back from `TRACKING` or
`HOLD` to `ACQUIRING`: once the ceiling is observed, the system
remains in the post-discovery regime. Cold-start stages begin in
`ACQUIRING` with `growth_streak = 0` (the sentinel set by
`_StageRuntimeState`); the first cycle that records a non-shrink
delta promotes the streak to `1`.

## Knobs

All knobs live on `SaturationAwareStageConfig` in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Field                              | Default | Role                                                                                     |
|---|---|---|
| `enable_growth_mode_state_machine` | `True`  | When `False`, HOLD is neutralised so SATURATED grow proceeds during the stabilization window. |
| `aggressive_growth_max_per_cycle`  | `4`     | Hard ceiling on per-cycle additions; bounds the blast radius when capacity demand jumps.   |
| `stabilization_window_cycles_down` | `30`    | Cycles `HOLD` must persist with no further shrink before returning to `TRACKING`.          |

Operators tune the per-cycle blast radius via
`aggressive_growth_max_per_cycle`. If a stage flaps between
`HOLD` and growth, widen `stabilization_window_cycles_down` to
extend the cooldown.

## See also

- [00 — Per-cycle overview](00-overview.md) — where the
  growth-mode transition sits relative to Phases A–D.
- [05 — State classifier](05-state-classifier.md) — the
  `SATURATED` / `SATURATED_CRITICAL` / `OVER_PROVISIONED`
  inputs that `compute_delta` reads alongside the mode.
- [07 — Streak stabilization](07-streak-stabilization.md) —
  the gate that turns a recommended delta into the *executed*
  delta the transition rule consumes.
- [10 — Slow-start mechanisms](10-slow-start-mechanisms.md) —
  the three cold-start safety layers; HOLD's binary
  grow gate is one of them.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the
  saturation-mode donor protocol whose Layer 2 rejects donors
  in `HOLD` for the same anti-flap reason this state machine
  enforces internally.
- [15 — Idle-first scale-down](15-idle-first-scale-down.md) —
  the Phase D shrink that transitions `TRACKING` into `HOLD`.
