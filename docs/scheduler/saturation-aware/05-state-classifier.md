# 05 — State Classifier

## TL;DR

Each cycle the scheduler maps every stage's EWMA-smoothed empty-slot
ratio into one of **five operational zones** — `NORMAL`, `STARVED`,
`SATURATED`, `SATURATED_CRITICAL`, `OVER_PROVISIONED` — with
per-zone hysteresis bands so the zone does not flap when the signal
sits on a threshold boundary. Five **discrete** zones, not a
continuous controller, because every action that follows (grow,
shrink, no-op, wait) gates on a discrete state crossed for a streak
of cycles.

## Problem

Phase C (grow) and Phase D (shrink) do not consume a scalar
utilisation number — they consume a categorical zone plus a streak
counter ("this stage has been `SATURATED` for the last 2 cycles,
time to grow"). The streak counter has no natural definition over a
continuous signal ("this stage has been at `0.32` for 2 cycles"
when the next sample is `0.34`), so the classifier output must be
**discrete and labelled**.

Once the output is discrete, two failure modes appear:

- **Edge oscillation.** The smoothed signal drifts across a fixed
  threshold every other cycle. The state flips every cycle, the
  streak counter resets every cycle, no decision is ever taken.
- **STARVED vs OVER_PROVISIONED ambiguity.** "Empty slots" alone
  does not mean "too many workers". A stage with empty slots and an
  empty input queue is starved by an upstream bottleneck, not
  over-provisioned. Acting on the slot signal alone would shrink
  the wrong stage every time the bottleneck moves.

## Decision

Adopt a **five-state classifier with per-zone asymmetric
hysteresis**, implemented as a **pure function** of
`(slots_empty_ratio_ewma, input_queue_depth, prev_state,
saturation_threshold, activation_threshold, config)`. See
[`classifier.classify`](../../../cosmos_xenna/pipelines/private/scheduling_py/classifier.py)
and the `StageState` enum in
[`state.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/state.py).

```
 empty-slot ratio (EWMA smoothed); 0.0 = every slot busy, 1.0 = every slot free

   0.0      activation_thr   saturation_thr   over_provisioned_thr      1.0
    │             │                │                   │                 │
    ├─────────────┼────────────────┼───────────────────┼─────────────────┤
    │  SATURATED  │   SATURATED    │      NORMAL       │   OVER_PROV     │
    │  _CRITICAL  │                │                   │       or        │
    │             │                │                   │    STARVED      │
    └─────────────┴────────────────┴───────────────────┴─────────────────┘
       no                                                                 
       hysteresis  ╔══ sat_deadband_pct ══╗  ╔══ over_provisioned_     ══╗
                   ║ inflates the boundary║  ║ deadband_pct deflates    ║
                   ║ on exit from         ║  ║ the boundary on exit     ║
                   ║ SATURATED /          ║  ║ from OVER_PROVISIONED    ║
                   ║ SATURATED_CRITICAL   ║  ║                          ║
                   ╚══════════════════════╝  ╚══════════════════════════╝

   STARVED vs OVER_PROVISIONED at ratio ≥ over_provisioned_boundary:
        input_queue_depth == 0  ─▶  STARVED          (upstream bottleneck)
        input_queue_depth >  0  ─▶  OVER_PROVISIONED (true over-provision)
```

Key properties:

- **Five zones, not three or a continuous score.** Phase C grow,
  Phase D shrink, and the streak gate each need a label they can
  count cycles against. A continuous controller would re-classify
  every cycle and oscillate around the decision boundary.
- **Per-zone asymmetric hysteresis.** Hysteresis is applied **only
  on exit from a state**, not on entry. From `SATURATED` /
  `SATURATED_CRITICAL` the saturation boundary is inflated by
  `saturation_deadband_pct`; from `OVER_PROVISIONED` the
  over-provisioned boundary is deflated by
  `over_provisioned_deadband_pct`. The over-provisioned band is
  conventionally wider than the saturation-side band so scale-down
  requires stronger evidence than scale-up.
- **No hysteresis on `SATURATED_CRITICAL` or `STARVED`.**
  `CRITICAL` is a burst signal that must respond on the cycle it
  fires; `STARVED` is driven by `input_queue_depth == 0`, which is
  a discrete predicate that can legitimately flip every cycle as
  the upstream stage's production rate ebbs and flows.
- **Queue-depth tiebreaker.** Above the over-provisioned boundary,
  `input_queue_depth == 0` routes to `STARVED` (no local scale
  action helps — the upstream stage is the bottleneck), while a
  non-empty queue routes to `OVER_PROVISIONED` (sustained
  scale-down signal).
- **Pure function.** No scheduler state, no I/O, no logging side
  effects. The classifier is unit-testable in isolation and reusable
  by simulators and replay harnesses.

**Trade-off.** A pin-sharp transition is replaced by a deadband, so
the classifier can stay in `SATURATED` (or `OVER_PROVISIONED`) for
one or two extra cycles after the signal first crosses the
threshold. The cost is bounded — the streak gate already imposes a
minimum cycle count — and the benefit is a reliable streak counter.

## How it works

For each stage, the classifier evaluates the rules below in order
and returns the first matching zone:

| Test (in order)                                          | Result                            |
|---|---|
| `ratio < activation_threshold`                           | `SATURATED_CRITICAL`              |
| `ratio < saturation_boundary`                            | `SATURATED`                       |
| `ratio ≥ over_provisioned_boundary` and `queue == 0`    | `STARVED`                         |
| `ratio ≥ over_provisioned_boundary` and `queue > 0`     | `OVER_PROVISIONED`                |
| otherwise                                                | `NORMAL`                          |

The two boundaries depend on `prev_state`:

```
saturation_boundary = saturation_threshold
if prev_state in {SATURATED, SATURATED_CRITICAL}:
    saturation_boundary *= 1 + saturation_deadband_pct

over_provisioned_boundary = over_provisioned_threshold
if prev_state == OVER_PROVISIONED:
    over_provisioned_boundary *= 1 - over_provisioned_deadband_pct
```

`saturation_threshold` and `activation_threshold` are either
operator-pinned or auto-derived per stage; see
[08 — Auto-derived thresholds](08-auto-derived-thresholds.md). The
`over_provisioned_threshold` is fixed per config — it sits in the
flat tail of the M/M/c response-time curve and is largely
`c`-insensitive, so a single default works for almost any stage.

The EWMA smoothing that feeds `slots_empty_ratio_ewma` is owned by
[`update_ewma`](../../../cosmos_xenna/pipelines/private/scheduling_py/state.py)
and described in
[07 — Streak stabilization](07-streak-stabilization.md); the
classifier itself never sees a raw per-cycle sample.

## Knobs

All fields live on `SaturationAwareStageConfig` in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Field                            | Default       | Role                                                                       |
|---|---|---|
| `saturation_threshold`           | `None` (auto) | Boundary between `SATURATED` and `NORMAL`                                  |
| `activation_threshold`           | `None` (auto) | Boundary between `SATURATED` and `SATURATED_CRITICAL`                      |
| `over_provisioned_threshold`     | `0.50`        | Boundary between `NORMAL` and `OVER_PROVISIONED` / `STARVED`               |
| `saturation_deadband_pct`        | `0.15`        | Width of the band held on exit from `SATURATED` / `SATURATED_CRITICAL`     |
| `over_provisioned_deadband_pct`  | `0.30`        | Width of the band held on exit from `OVER_PROVISIONED`                     |

Operator-pinned thresholds must satisfy
`activation < saturation < over_provisioned`. The cross-field
validator on `SaturationAwareStageConfig.__attrs_post_init__`
enforces the invariant at construction time, and the auto-threshold
resolver re-checks it on the resolved values.

## See also

- [00 — Per-cycle overview](00-overview.md) — where the classifier
  sits in the four-phase cycle.
- [06 — Backlog-time signal](06-backlog-time-signal.md) — the
  compound AND-criterion that gates whether a `SATURATED` zone
  actually emits a positive intent.
- [07 — Streak stabilization](07-streak-stabilization.md) — the
  EWMA and asymmetric streak counters that consume the zone.
- [08 — Auto-derived thresholds](08-auto-derived-thresholds.md) —
  how `saturation_threshold` and `activation_threshold` are
  resolved per stage from `K / sqrt(c)`.
