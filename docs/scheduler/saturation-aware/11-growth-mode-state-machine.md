# 11 — Growth-Mode State Machine

## TL;DR

Each stage carries a **three-mode** growth controller —
`ACQUIRING`, `TRACKING`, `HOLD` — that shapes how aggressively
the scheduler scales it up for the same classifier output. Mode
transitions are driven by the **executed delta**, not the
classifier output alone, so a recommendation suppressed by the
stabilization gate does not change the mode. `HOLD` is a
post-shrink stabilization window that throttles non-critical
growth while still allowing minimal burst response.

## Problem

The classifier emits a discrete zone (`SATURATED`,
`SATURATED_CRITICAL`, ...) but does not say *how much* to grow.
The right magnitude depends on how much the scheduler already
knows about the stage:

- **Cold start, no ceiling discovered.** Adding one worker per
  cycle leaves a stage that ultimately needs 64 workers
  undersized for most of the run; we want to find the ceiling
  fast.
- **After the first over-provisioned event.** We have a
  ceiling. Aggressive multiplicative growth from there
  overshoots it, triggers another shrink, and the stage flaps.
- **Immediately after a shrink.** Resuming aggressive growth
  on the next saturated signal re-creates the same
  over-provision. We need a cool-down window — but it must
  still let genuine bursts through.

A single fixed growth policy cannot serve all three regimes.
Picking the most aggressive policy wastes capacity by flapping;
picking the safest one starves stages during cold start.

## Decision

Track a per-stage three-state machine —
[`GrowthMode`](../../../cosmos_xenna/pipelines/private/scheduling_py/state.py)
`ACQUIRING` / `TRACKING` / `HOLD` — that drives the magnitude
selection inside
[`compute_delta`](../../../cosmos_xenna/pipelines/private/scheduling_py/decisions.py).
The pure-function transition rule lives in
[`compute_growth_mode_transition`](../../../cosmos_xenna/pipelines/private/scheduling_py/growth_mode.py).

- **`ACQUIRING`** (initial state). Grow **multiplicatively**:
  `delta = ceil(factor * current_workers)`. Discovers the
  ceiling fast. Conceptual analog of TCP slow-start.
- **`TRACKING`** (post-first-shrink). Grow **additively**:
  fixed `+1` or `+2`. The ceiling is known; do not overshoot
  it.
- **`HOLD`** (post-shrink stabilization). Suppress non-critical
  growth for `stabilization_window_cycles_down` cycles; still
  emit `+1` on `SATURATED_CRITICAL` so a real burst is not
  ignored. A re-shrink while in `HOLD` restarts the timer.

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
   the per-stage hard worker cap or `max_scale_down_fraction_per_cycle`
   throttled to `+2` advances the timer by `+2`, not by the
   recommended `+5`.
3. **Allocation failures.** A recommended `+3` that aborted
   after one successful add (with the rest of the cycle failing
   the absorb path) reflects `+1` into the timer, not the full
   `+3` recommendation.

The mode therefore stays consistent with what the workers
actually saw.

Every growth path also runs through a hard
`aggressive_growth_max_per_cycle` cap, so a multiplicative
`ACQUIRING` step on a large stage cannot single-handedly
overshoot the cluster.

**Trade-off.** One extra integer of per-stage state plus a
streak counter, plus an extra knob
(`stabilization_window_cycles_down`) operators must understand.
In exchange we get fast cold-start convergence, stable additive
steady-state, and bounded oscillation after over-provisioning.

```
   delta_executed >= 0  —  grow / no-op
   delta_executed <  0  —  shrink (workers actually removed)


   ┌──────────────────┐
   │   ACQUIRING      │ ◄── grow / no-op:
   │   cold start,    │     stay, streak += 1
   │   multiplicative │
   └──────────────────┘
            │
            │  first shrink
            ▼
   ┌──────────────────┐
   │    TRACKING      │ ◄── grow / no-op:
   │    ceiling       │     stay, streak += 1
   │    known,        │
   │    additive      │
   └──────────────────┘
        ▲       │
        │       │  later shrink
        │       ▼
        │  ┌──────────────────┐
        │  │      HOLD        │ ◄── grow / no-op,
        │  │   post-shrink    │     window not expired:
        │  │   stabilization, │     stay, streak += 1
        │  │   burst-only     │
        │  │   growth         │ ◄── re-shrink:
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
  the three cold-start safety layers; the `ACQUIRING` mode is
  one of them.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the
  saturation-mode donor protocol whose Layer 2 rejects donors
  in `HOLD` for the same anti-flap reason this state machine
  enforces internally.
- [15 — Idle-first scale-down](15-idle-first-scale-down.md) —
  the Phase D shrink that transitions `TRACKING` into `HOLD`.
