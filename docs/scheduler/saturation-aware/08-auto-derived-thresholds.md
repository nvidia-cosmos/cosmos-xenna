# 08 — Auto-Derived Thresholds

## TL;DR

The classifier's `saturation_threshold` and `activation_threshold`
are resolved **lazily per stage** from each stage's runtime
`slots_per_actor` via the Halfin-Whitt heavy-traffic formula
`K / √c`, with `K = saturation_aggressiveness` (default `0.30`). A
single fixed default would be silently wrong across stages with
different per-actor concurrency `c`.

## Problem

The classifier triggers `SATURATED` when the **empty-slot fraction**
drops below `saturation_threshold`. The "correct" trigger value is
not a constant: an Erlang-C M/M/c queue's response-time knee shifts
**left in empty-slot space** as the number of concurrent servers `c`
grows — more servers means a healthy queue runs at higher utilisation
and a smaller empty-slot fraction. The Halfin-Whitt heavy-traffic
regime (Halfin & Whitt 1981) makes this precise: at the knee, the
empty-slot fraction `1 − ρ` scales as `β / √c`.

A single fixed default cannot serve both ends of the range:

- Tuned for high `c` (e.g. `0.05`) — `SATURATED` never fires on a
  `c=1` stage until it is 95% loaded, far past the knee.
- Tuned for low `c` (e.g. `0.30`) — `SATURATED` fires on a `c=64`
  stage on a single transient empty slot.

A single fixed default such as `0.15` happens to be reasonable for
`c ≈ 4` and is silently mis-calibrated for every other stage on the
same pipeline.

## Decision

Resolve `saturation_threshold` from `slots_per_actor` via the
Halfin-Whitt formula, clamped to a safety band; derive
`activation_threshold` from the resolved saturation:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   saturation = clamp( K / √c , auto_min , auto_max )         │
│   activation = saturation × activation_to_saturation_ratio   │
│                                                              │
│   where  K = saturation_aggressiveness  (default 0.30)       │
│          c = slots_per_actor            (per-stage runtime)  │
│          auto_min = 0.02   auto_max = 0.45                   │
│          activation_to_saturation_ratio = 0.33               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Example resolution at `K = 0.30` (the canonical defaults):

| `c` | `K / √c` | clamped `saturation` | derived `activation` |
|----:|---------:|---------------------:|---------------------:|
|   1 |    0.300 |               0.300  |               0.099  |
|   2 |    0.212 |               0.212  |               0.070  |
|   4 |    0.150 |               0.150  |               0.050  |
|   8 |    0.106 |               0.106  |               0.035  |
|  16 |    0.075 |               0.075  |               0.025  |
|  32 |    0.053 |               0.053  |               0.018  |
|  64 |    0.038 |               0.038  |               0.012  |

**Trade-off.** A textbook M/M/c queue uses `c = slots_per_actor ×
num_workers` (total concurrent servers across the stage). The
resolver deliberately uses `slots_per_actor` alone instead. The
trade-off is precision vs. stability: total-`c` is mathematically
tighter, but it shifts every time the autoscaler grows or shrinks
the stage. The EWMA decay, the asymmetric streak counters, and the
growth-mode state are all calibrated against the threshold band in
force at their first observation; a threshold that moves under them
would invalidate the consensus signal they feed. Per-actor
concurrency is the only `c` that is **stable for the lifetime of
the stage**, so it is the only `c` that produces a stable threshold.

## How it works

```
                first autoscale() cycle for stage S
                                 │
                                 ▼
              ┌──────────────────────────────────────────────┐
              │  runtime stage S has slots_per_actor = c     │
              │  (from ProblemStageState, not config)        │
              └──────────────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────────┐
              │  _resolve_auto_thresholds(stage_cfg, c, K)   │
              │  ─ pinned saturation?   use it               │
              │    else                 K/√c, clamped to     │
              │                         [auto_min, auto_max] │
              │  ─ pinned activation?   use it               │
              │    else                 saturation × ratio   │
              │  ─ verify zone ordering:                     │
              │       activation < saturation                │
              │                 < over_provisioned_threshold │
              │       (ValueError otherwise)                 │
              └──────────────────────────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────────┐
              │  ResolvedThresholds cached on                │
              │  runtime.resolved_thresholds                 │
              │  ─ one INFO log per stage with provenance    │
              │  ─ later cycles short-circuit on the cache   │
              │  ─ mid-run slots_per_actor changes do NOT    │
              │    re-resolve                                │
              └──────────────────────────────────────────────┘
```

**Override hierarchy.** Each threshold is independently auto-derived
or pinned; an explicit float on the config bypasses the formula for
that field only:

| `saturation_threshold` | `activation_threshold` | Resolved behaviour |
|---|---|---|
| `None` (auto)  | `None` (auto)  | both from formula; `activation = saturation × ratio` |
| pinned float   | `None` (auto)  | pin saturation, derive activation from it           |
| `None` (auto)  | pinned float   | derive saturation from formula, pin activation      |
| pinned float   | pinned float   | both pinned; formula unused                         |

The frozen `ResolvedThresholds` record carries
`saturation_threshold_was_overridden` and
`activation_threshold_was_overridden` flags so the per-stage startup
log distinguishes "auto" from "manual override" without guesswork.

**Why lazy + one-shot.** Resolution runs on the first `autoscale()`
cycle because that is the earliest cycle at which Xenna populates
the runtime `slots_per_actor` (config defaults may differ).
Re-resolving on every cycle would silently invalidate the EWMA
decay, the asymmetric streak counters, and the growth-mode state
machine, all of which are calibrated against the threshold band in
force at their first observation. The single sanctioned
re-resolution path is a Halfin-Whitt regime transition, which
intentionally rebases the threshold-relative history through the
same code path — see `09-regime-aware-aggressiveness.md`. Operators
who deliberately reshape a stage and want fresh thresholds must
restart the pipeline.

## Knobs

All knobs live on `SaturationAwareStageConfig` in
[`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).
The resolver itself is `_resolve_auto_thresholds` in
[`cosmos-xenna/cosmos_xenna/pipelines/private/scheduling_py/auto_thresholds.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/auto_thresholds.py).

| Field | Default | Effect |
|---|---|---|
| `saturation_aggressiveness`       | `0.30` | The `K` in `K/√c` — Halfin-Whitt `β`. Higher fires `SATURATED` sooner. Range `[0.10, 0.60]`. Shared with regime-aware lift. |
| `auto_threshold_min`              | `0.02` | Safety floor on the auto-derived `saturation_threshold`. Prevents single-slot transients in 100+ slot stages from firing `SATURATED`. |
| `auto_threshold_max`              | `0.45` | Safety ceiling. Must be strictly below `over_provisioned_threshold` so the auto-derived saturation never collides with the over-provisioned zone (cross-field validator). |
| `activation_to_saturation_ratio`  | `0.33` | Fraction of resolved saturation at which `SATURATED_CRITICAL` begins. Default reproduces the `0.05 / 0.15 = 0.33` ratio used before auto-derivation shipped. |
| `saturation_threshold`            | `None` | `None` → auto-derive; explicit float → pin this field, formula is bypassed for it. |
| `activation_threshold`            | `None` | Same override semantics as above; pinned independently of saturation. |

To reproduce pre-auto-derivation behaviour exactly, pin both
`saturation_threshold = 0.15` and `activation_threshold = 0.05`.

## See also

- [09 — Regime-aware aggressiveness](09-regime-aware-aggressiveness.md) —
  the regime detector lifts the effective `K` when the cluster enters
  super-Halfin-Whitt, and is the only mechanism that re-runs
  resolution mid-pipeline.
- [05 — State classifier](05-state-classifier.md) — the four-zone
  classifier that consumes the resolved thresholds.
- [02 — Configuration model](02-configuration-model.md) — where
  `SaturationAwareStageConfig` is composed and how the three-tier
  resolver merges per-stage overrides.
- Halfin & Whitt (1981), *Heavy-traffic limits for queues with many
  exponential servers*, *Operations Research* 29(3) — the original
  `1 − β/√c` heavy-traffic regime that justifies the `K/√c` shape.
- Lazowska, Zahorjan, Graham, Sevcik (1984), *Quantitative System
  Performance: Computer System Analysis Using Queueing Network
  Models* — textbook M/M/c queues and the response-time knee.
