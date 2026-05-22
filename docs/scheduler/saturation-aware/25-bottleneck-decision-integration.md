# 25 — Bottleneck Decision Integration

## TL;DR

The Forced-Flow-Law per-stage service demand `D_k` introduced as a
pure-observability metric in
[23 — Bottleneck score metric](23-bottleneck-score-metric.md) is now a
first-class scheduler input. Each cycle the autoscaler computes an
EWMA-smoothed `D_k_ewma[k]`, asks
[`identify_bottleneck`](../../../cosmos_xenna/pipelines/private/scheduling_py/bottleneck.py)
whether the cluster is heterogeneous enough to commit to a single
bottleneck stage (`max / median` for `n ≥ 3` stages, `max / min` for
`n = 2`, threshold `bottleneck_heterogeneity_threshold`), and feeds the
result into two decisions:

1. **Phase C grow priority.** When the gate is engaged, stages are
   walked in `D_k_ewma` descending order so a scarce placement slot
   lands on the true bottleneck regardless of DAG depth.
2. **Phase D shrink protection.** When the gate is engaged, the
   bottleneck stage is **never shrunk** by negative intent alone; only
   a hard-cap ceiling overflow can remove its workers.

Both behaviours are gated by individual config toggles
(`enable_bottleneck_priority_growth`, `enable_bottleneck_shrink_protection`,
both default `True`) so they can be turned off independently without a
redeploy. A debounced INFO log fires when the engagement state persists
past `bottleneck_engagement_persistence_cycles` cycles, so operators see
the gate flip at most once per sustained transition.

## Problem

[23 — Bottleneck score metric](23-bottleneck-score-metric.md) gave
operators a single number that ranks stages by bottleneck candidacy,
but the autoscaler did not consume it. Two failure modes survived:

- **Wrong-stage-first growth under multi-saturated transients.** Phase
  C's downstream-first DAG-depth ordering
  ([12 — Multi-target DAG growth](12-multi-target-dag-growth.md))
  matches the steady-state intuition "scale the tail first", but in
  pipelines where the bottleneck is **mid-DAG** (for example a
  captioning stage between cheap download and cheap upload) the deepest
  stage is not the slowest. When the cluster runs out of placement
  slots during a multi-saturated burst, the contended slot lands on
  whichever stage happens to be deepest, not the one whose `D_k_ewma`
  actually bounds throughput. The pipeline's `1 / max_k D_k` ceiling
  does not move, even though Phase C "did the work".
- **Mid-task shrink of the bottleneck on transient idle.** The
  five-zone classifier classifies a stage `OVER_PROVISIONED` once the
  smoothed slots-empty ratio exceeds the over-provisioned threshold
  for `over_provisioned_streak_min_cycles` cycles. The classifier is
  intentionally myopic per-stage; it cannot tell that the
  "over-provisioned" cluster is actually idle **because the bottleneck
  is briefly idle** (model reload, GPU stall, brief slot drop). Phase
  D shrinks the bottleneck on the streak; re-growing it after recovery
  costs a full `worker_warmup_measurement_grace_s` window
  (`60 s` default), capping pipeline throughput for the entire warmup.

The solution is to feed `D_k_ewma`, the same metric the operator
already trusts on the dashboard, back into Phase C and Phase D so the
scheduler's decisions match the operator's mental model.

## Decision

Promote `D_k_ewma` from a pure-observability gauge to a first-class
scheduler input by adding two narrow consumption points and one
shared identification helper. The toggles default to `True`, so the
behaviour is on for every pipeline; setting either to `False` disables
the matching phase's bottleneck awareness.

```
                       autoscale() cycle (ordered)
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Phase A : pre-flight           │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Phase B : floor enforcement    │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  bottleneck calculation         │
                  │  ───────────────────────────    │
                  │  service_times_s = consume()    │
                  │  _update_d_k_ewma(samples)      │
                  │  meta = identify_bottleneck(    │
                  │      d_k_ewma=_d_k_ewma,        │
                  │      heterogeneity_threshold=H) │
                  │  maybe_log_engagement(meta)     │
                  │                                 │
                  │  meta.engaged is the gate that  │
                  │  Phase C and Phase D consult.   │
                  └─────────────────────────────────┘
                          │                │
                          ▼                ▼
              ┌─────────────────┐  ┌─────────────────┐
              │   Phase C grow  │  │  Phase D shrink │
              │   priority      │  │   protection    │
              ├─────────────────┤  ├─────────────────┤
              │ engaged ?       │  │ engaged ?       │
              │   yes → sort    │  │   yes →         │
              │     by D_k desc │  │     skip stage  │
              │     (DAG depth  │  │     when stage  │
              │      tie-break) │  │     == argmax_k │
              │   no  → DAG     │  │     and intent<0│
              │     depth desc  │  │     and ceiling │
              │     (or problem │  │     excess == 0 │
              │     order if    │  │   no  →         │
              │     toggle off) │  │     unchanged   │
              └─────────────────┘  └─────────────────┘
```

- **`identify_bottleneck` is a pure helper.** It reads a
  `dict[str, float]` of EWMA-smoothed `D_k` values and returns a
  frozen `BottleneckIdentity` with the engagement flag, argmax stage,
  `max`, `median`, and heterogeneity ratio. The ratio uses
  `max / median` when at least three stages have finite values and
  `max / min` when exactly two do; a single finite stage cannot be a
  bottleneck because the metric is comparative. Near-tie argmax is
  resolved lexicographically so a small ratio does not flip the
  bottleneck identity every cycle.
- **EWMA smoothing keeps the gate stable.** `D_k` raw samples vary
  with batch size and warm-up timing; an EWMA with
  `bottleneck_d_k_smoothing_level` (default `0.20`, the same alpha
  as the slot-EWMA) cushions the per-cycle noise. Cold-start (a
  stage with no completed task yet) seeds with the first finite
  sample directly; missed samples preserve the previous value so
  the gate does not regress when one cycle is empty.
- **Debounced engagement logging.** A `BottleneckEngagementState`
  tracks the consecutive cycles the engagement flag has held its
  current value. The INFO log fires only when the state persists
  past `bottleneck_engagement_persistence_cycles` cycles, so
  short-lived flips during regime transitions are silent.
- **Ceiling overflow always wins.** Bottleneck shrink protection
  protects a stage from negative intent only. A hard-cap ceiling
  overflow (`ceiling_excess > 0`), because the operator just lowered
  `max_workers` or the per-node cap binds, always shrinks even on
  the bottleneck. The protection is for transient idle, not for
  operator-driven contraction.

**Trade-off.** Two new decision dependencies on `D_k_ewma` and one
new mutable scheduler attribute (`_d_k_ewma`). Each lives entirely
inside `autoscale()`'s single-threaded body, so no new lock is
required; the existing `_completed_counts` and
`_completed_service_time_sums` locks already cover the
`update_with_measurements`-side concurrency. The cost is roughly
`O(num_stages)` extra work per cycle (one EWMA update, one
`statistics.median` call); the benefit is that Phase C and Phase D
no longer make wrong-stage-first decisions on heterogeneous
pipelines.

## What gets grown vs. what gets ordered

A common mis-reading of "bottleneck-priority growth" is that the
gate **decides which stages to grow**. It does not. The candidate
set for Phase C is filtered by the classifier *before* this doc's
priority sort runs. The bottleneck gate only orders the visit
sequence among stages the classifier already marked saturated.

```
              ┌──────────────────────────────────────────────────┐
              │  STEP 1 — Classifier (per-stage, evidence-only)  │
              │  intent[k] is a function of stage k's own slot   │
              │  occupancy, queue depth, and streak counters.    │
              │  No cross-stage prediction. No DAG awareness.    │
              │  No D_k input.                                   │
              └──────────────────────────────────────────────────┘
                                    │
                                    ▼
                    candidate_set = { k : intent[k] > 0 }
                                    │
                                    ▼
              ┌──────────────────────────────────────────────────┐
              │  STEP 2 — Phase C grow priority (this doc)       │
              │  Orders ONLY the candidate set for visit         │
              │  sequence under cluster-headroom contention.     │
              │  Does NOT add stages. Does NOT filter the set.   │
              └──────────────────────────────────────────────────┘
                                    │
                                    ▼
                  for each candidate (in priority order):
                      try_add_worker(stage, intent[k] times)
                      stop when cluster headroom exhausted
```

The classifier is **evidence-only**: it does not predict, does not
look at static capacity hints, and does not anticipate the effect
of upstream growth. A stage with low slot-occupancy and a short
queue stays NORMAL with `intent = 0` no matter how saturated its
upstream is, so the candidate set never includes "fast enough"
downstream stages.

### Worked example: `A → B → C → D → E → F` with B as bottleneck

Capacities: `A=0.5, B=0.1, C=0.2, D=0.5, E=1.0, F=1.0`.
B is the bottleneck. Steady-state classifier output:

| Stage | What the classifier observes | Zone | Intent |
|---|---|---|---|
| A | Slot occupancy ≪ 1 (throttled by B) | OVER_PROVISIONED | ≤ 0 |
| B | Slot occupancy = 1 (saturated) | SATURATED | `+N` |
| C | Idle slots, drained queue (starved by B) | OVER_PROVISIONED | 0 (until streak fires shrink) |
| D | Idle slots, drained queue | OVER_PROVISIONED | 0 (until streak fires shrink) |
| E | Idle slots, short queue | NORMAL | 0 |
| F | Idle slots, short queue | NORMAL | 0 |

Candidate set is `{ B }`. Phase C grows **only B**. The bottleneck
priority sort is a no-op for a single candidate; downstream fast
stages (D, E, F) are not in the set, so the gate cannot grow them
even when engaged.

After B grows enough to push the bottleneck downstream (say B's
effective capacity rises to `0.2`, matching C), the **next** cycle
re-runs the classifier with fresh measurements. Now C may report
SATURATED in its own right, joining the candidate set:

```
   cycle N      candidate_set = { B }            grow B
                                ─────────────
   cycle N+k    candidate_set = { B, C }         multi-saturated:
                                ─────────────    bottleneck gate
                                                 picks B over C
                                                 (B still slightly
                                                 slower per
                                                 D_k_ewma)

   cycle N+m    candidate_set = { C }            B no longer
                                ─────────────    saturated;
                                                 bottleneck has
                                                 shifted to C
```

The bottleneck "chases" downstream one stage at a time, **driven
by observation, not by prediction**. There is no speculative
growth — every Phase C add is backed by classifier evidence on
that specific stage in the previous cycle.

### Where the priority sort actually matters

The sort is meaningful only when the candidate set has more than
one entry **and** the cluster has less headroom than the sum of
intents. Both conditions must hold; either alone makes the sort
a no-op.

| Scenario | Candidate set | Cluster headroom | Priority sort effect |
|---|---|---|---|
| Single bottleneck, steady state | `{ B }` | Any | No-op (one candidate). |
| Single bottleneck, headroom unlimited | `{ B }` | ≥ intent[B] | No-op. |
| Multi-saturated burst, headroom unlimited | `{ B, C }` | ≥ intent[B] + intent[C] | No-op (both grow fully). |
| Multi-saturated burst, headroom scarce | `{ B, C }` | < intent[B] + intent[C] | **Bottleneck wins** the contended slot when engaged; otherwise DAG-deepest wins. |
| Cold-start ramp (every stage saturates briefly) | `{ A, B, C, D, E, F }` | Typically scarce | **B (highest D_k) wins**, not F (deepest in DAG). |

In single-bottleneck steady state — the most common case — the
gate is observably idle; it neither grows extra stages nor blocks
the natural one. Its value shows up exactly in the burst /
cold-start / shifting-bottleneck cases that the prior DAG-only
ordering handled wrong.

## How it works

The integration is three short blocks: a helper, a pair of state
attributes, and a reordered `autoscale()` body that calls them in
the right sequence. The helper lives next to the metric it consumes
([`bottleneck.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/bottleneck.py));
the state attributes live on
[`SaturationAwareScheduler`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py);
the reorder happens inside `autoscale()` so Phase C and Phase D
both observe **fresh** bottleneck data for the current cycle.

```
      ┌────────────────────────────────────────────────────────────┐
      │  bottleneck.py                                             │
      │                                                            │
      │  @attrs.define(frozen=True)                                │
      │  class BottleneckIdentity:                                 │
      │      engaged:              bool                            │
      │      stage_name:           str | None                      │
      │      max_d_k:              float                           │
      │      median_d_k:           float                           │
      │      heterogeneity_ratio:  float                           │
      │                                                            │
      │  identify_bottleneck(                                      │
      │      d_k_ewma:              Mapping[str, float],           │
      │      heterogeneity_threshold: float,                       │
      │  ) → BottleneckIdentity                                    │
      │                                                            │
      │  • ratio = max / median   if n ≥ 3                         │
      │  • ratio = max / min      if n == 2                        │
      │  • engaged = (n ≥ 2) and ratio ≥ threshold                 │
      │  • argmax tie-break: lexicographic stage_name              │
      │                                                            │
      │  maybe_log_bottleneck_engagement(                          │
      │      identity, state, persistence_cycles, pipeline_name,   │
      │  )                                                         │
      │  • streak update + INFO once per persistent transition     │
      └────────────────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────────────────┐
      │  saturation_aware.py  (single-threaded autoscale() body)   │
      │                                                            │
      │  _d_k_ewma:               dict[str, float]                 │
      │  _last_bottleneck_meta:   BottleneckIdentity | None        │
      │  _bottleneck_engagement_state: BottleneckEngagementState   │
      │                                                            │
      │  setup() seeds _d_k_ewma to {name: math.nan} for every     │
      │  stage and resets engagement state.                        │
      │                                                            │
      │  autoscale() runs (in this order):                         │
      │     Phase A pre-flight                                     │
      │     Phase B floor enforcement                              │
      │                                                            │
      │     # bottleneck calculation: ONCE per cycle, BEFORE C/D   │
      │     service_times_s = self._consume_service_time_samples() │
      │     self._update_d_k_ewma(service_times_s)                 │
      │     self._last_bottleneck_meta = identify_bottleneck(...)  │
      │     maybe_log_bottleneck_engagement(...)                   │
      │                                                            │
      │     Phase C grow   → consults _last_bottleneck_meta        │
      │     Phase D shrink → consults _last_bottleneck_meta        │
      └────────────────────────────────────────────────────────────┘
```

### Phase C: grow-priority ordering

A unified
[`compute_grow_priority_order`](../../../cosmos_xenna/pipelines/private/scheduling_py/dag_priority.py)
implements a three-level hierarchy:

```
   bottleneck_engaged ?
     │
     ├── yes → sort by D_k_ewma DESC (NaN last);
     │         tie-break by topological depth DESC;
     │         final tie-break by problem index ASC.
     │
     └── no
            │
            └── enable_dag_priority_growth ?
                  │
                  ├── yes → sort by topological depth DESC.
                  │
                  └── no  → use problem (upstream-first) order.
```

The bottleneck path keeps the existing DAG-depth tiebreaker so two
stages with identical `D_k_ewma` retain the
[12](12-multi-target-dag-growth.md) downstream-first ordering. NaN
`D_k_ewma` (cold-start stages) sort to the back so they cannot win a
contended slot before they have observed a single completed task.

When the toggle `enable_bottleneck_priority_growth` is `False`,
Phase C falls through directly to the DAG-depth path regardless of
engagement, matching the [12](12-multi-target-dag-growth.md) ordering
exactly so the toggle is a clean rollback knob.

### Phase D: bottleneck shrink protection

When the gate engages and the current stage **is** the argmax `D_k`
stage and the intent is negative and there is no ceiling overflow
(`ceiling_excess == 0`), the per-stage shrink loop **skips** the
stage entirely:

```
   for stage_index in stage_order:
       intent          = self._last_intent_deltas.get(stage_name, 0)
       ceiling         = stage_ceilings[stage_index]
       ceiling_excess  = max(0, current - ceiling) if ceiling else 0

       bottleneck_meta = self._last_bottleneck_meta
       if (
           self._config.enable_bottleneck_shrink_protection
           and bottleneck_meta is not None
           and bottleneck_meta.engaged
           and stage_name == bottleneck_meta.stage_name
           and intent < 0
           and ceiling_excess == 0
       ):
           logger.info(
               f"phase D bottleneck shrink protected: stage {stage_name!r} "
               f"intent={intent} but D_k={bottleneck_meta.max_d_k:.2f}s is "
               f"argmax (ratio={bottleneck_meta.heterogeneity_ratio:.2f}); "
               "skipping shrink to preserve throughput across transient idle"
           )
           continue
       # ... existing floor / fraction-cap / warmup-grace path ...
```

The skip is per-cycle and per-stage; the next cycle re-evaluates
both the bottleneck identity and the intent, so a stage that
**stops** being the bottleneck (its `D_k_ewma` drops below another
stage's) will shrink as expected one cycle later. The classifier
state and streak counters are unchanged by the skip; they keep
ticking, so re-entering the OVER_PROVISIONED state on the same
stage still requires a fresh `over_provisioned_streak_min_cycles`
streak after the bottleneck moves away.

Ceiling overflow always wins. If the operator lowered `max_workers`
mid-run, the contraction proceeds even on the bottleneck stage; the
goal of shrink protection is to absorb transient classifier noise,
not to override operator-driven configuration.

## Knobs

All five knobs live on
[`SaturationAwareConfig`](../../../cosmos_xenna/pipelines/private/specs.py).
None are per-stage: bottleneck identification is a cluster-level
property, so per-stage overrides for these fields would be confusing
(every stage would see the same engagement gate regardless).

| Field | Default | Range | Effect |
|---|---|---|---|
| `enable_bottleneck_priority_growth` | `True` | bool | Phase C ordering. `True` engages the bottleneck-first sort when the gate engages; `False` uses [12](12-multi-target-dag-growth.md) DAG-depth ordering exactly. |
| `enable_bottleneck_shrink_protection` | `True` | bool | Phase D protection. `True` skips negative intent on the argmax `D_k` stage when engaged; `False` shrinks unconditionally. Ceiling overflow always shrinks. |
| `bottleneck_d_k_smoothing_level` | `0.20` | `(0.0, 1.0]` | EWMA alpha applied to per-cycle `D_k` samples. Lower smooths more (slower bottleneck moves); higher reacts faster. `1.0` disables smoothing (raw passthrough). |
| `bottleneck_heterogeneity_threshold` | `2.0` | `> 1.0` | Engagement floor. The cluster engages the gate only when the heterogeneity ratio is at least this value. `2.0` means "the bottleneck is at least twice the median (or twice the min in 2-stage pipelines)". Lower values engage on milder asymmetry; raise toward `4.0` for very long pipelines where the median is naturally pulled down by cheap I/O stages. |
| `bottleneck_engagement_persistence_cycles` | `2` | `≥ 1` | Streak gate for the engagement INFO log. The log fires only after the engagement flag has held its new value for this many consecutive cycles, so a single-cycle flip during regime transitions is silent. Does not affect the gate itself; the gate is per-cycle. |

## See also

- [00 — Per-cycle overview](00-overview.md) — where the bottleneck
  calculation block sits in the four-phase cycle.
- [12 — Multi-target DAG growth](12-multi-target-dag-growth.md) —
  the DAG-depth ordering Phase C uses when the bottleneck gate is
  disengaged or its toggle is `False`.
- [15 — Idle-first scale-down](15-idle-first-scale-down.md) — the
  per-worker victim selection that runs after Phase D's per-stage
  protection decides which stages may shrink.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — the
  catalogue that already exposes `xenna_stage_bottleneck_score`
  (raw `D_k`) and the cluster heterogeneity ratio used by
  `identify_bottleneck`.
- [23 — Bottleneck score metric](23-bottleneck-score-metric.md) —
  the Forced-Flow-Law derivation `D_k = V_k * S_k` that produces
  the per-cycle samples this doc smooths and consumes.
- Lazowska, Zahorjan, Graham, Sevcik (1984), *Quantitative System
  Performance: Computer System Analysis Using Queueing Network
  Models*, Chapter 3 — the Forced Flow Law and the bottleneck
  identification it implies for separable queueing networks.
