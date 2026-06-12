# 01: Capacity model

> **Where this fits.** This note is the **sizing** layer: it answers *how many*
> workers a stage needs. *Which* stage to grow or shrink is a separate decision,
> read from queue occupancy ([02](02-bottleneck-selection.md),
> [05](05-scale-down-floor.md)). Speed never *selects* the bottleneck; it only
> sizes the stage the queue gradient already picked - the one exception is the
> cold / balanced fallback in [02](02-bottleneck-selection.md). Speed is needed
> because naming the constraint is not a worker count, and because the wrapped
> `FRAG` solver is throughput-driven: a per-worker speed is the only way to turn
> "this stage is the bottleneck" into "give it N workers."

## The problem

Growth and shrink must agree on **one** view of the pipeline. If growth sizes a
stage from its local backlog while shrink sizes it from raw throughput, the two
fight: one adds a worker the other removes next cycle, churning warm state for
no throughput gain.

For a linear pipeline the truth is simple: sustainable end-to-end throughput
equals the **slowest stage's** rate. Over-provisioning any other stage cannot
raise throughput; the extra workers just push more inventory onto the same
downstream queue. So every stage's *useful* size has to be derived from that one
rate, computed once per cycle and shared by every decision.

## What we do

One model (`capacity.py`) computes, per stage, a **source-rate capacity** and
two worker targets. Everything downstream reads these numbers; nothing
recomputes them.

```
  cap_src[k] = workers[k] × target_speed[k] / chain[k]     (source items / sec)
```

- `target_speed[k]`: the smoothed per-worker service speed (see below); `0`
  while the stage is still cold/untrusted, which excludes it from the model.
- `chain[k]`: the fan-out from the source to stage `k`, i.e. how many stage-`k`
  items one source item turns into. If one source item fans out into 100
  stage-`k` items, that stage has `chain = 100`. Dividing by it puts every
  stage's rate in the same **source** items/sec unit, so they can be compared
  directly.

From the per-stage `cap_src` the model derives the rates and targets used
everywhere else:

| Symbol | Meaning |
|---|---|
| `bottleneck_rate` | the pipeline sizing rate, always the **slowest measured** `cap_src` (a serial pipeline cannot run faster than its slowest stage). |
| `next_bottleneck_rate` | the second-slowest `cap_src`, the rate the bottleneck stage may climb toward. |
| `w_sustain` | workers needed to **hold** the current bottleneck rate (no slack); the scale-down target. |
| `w_target` | the per-cycle **grow** ceiling. |

`w_target` is what makes the model self-correcting: only the bottleneck stage's
target climbs (toward `next_bottleneck_rate`, the move that can actually raise
pipeline speed); every other stage is bounded to
`bottleneck_rate × (1 + capacity_headroom)`, a small read-ahead, never free
growth.

![Bar chart of per-stage capacity in source items per second. S0 and S1 are long bars, S2 is the shortest and marked as the bottleneck, S3 is the next bottleneck. A dashed line at the S2 level marks the pipeline pace.](assets/01-capacity-bottleneck.png)

*The slowest measured `cap_src` sets `bottleneck_rate`, so the whole pipeline
runs at the S2 pace. Only the bottleneck is sized to climb toward the
next-slowest rate; every other stage sits above the line and may shrink toward
`w_sustain`.*

![Animated capacity bars where the slowest measured stage sets the pipeline rate while the bottleneck grows toward the next-slowest stage.](assets/01-capacity-cycle.gif)

*Across cycles, the same model feeds both directions: the slowest measured
`cap_src` holds the pipeline rate, the bottleneck's `w_target` climbs toward the
next-slowest rate, and non-bottleneck stages remain bounded by `w_sustain`.
These bars are the **sizing** view; the bottleneck's identity is chosen from
the queue gradient ([02](02-bottleneck-selection.md)), not from these speeds.*

## The speed signal it rests on

`cap_src` is only as trustworthy as `target_speed`, so the model smooths the raw
per-worker speed carefully (`estimator.py`):

- **Asymmetric EWMA.** `target_speed` rises with a modest weight
  (`speed_alpha_up`) and falls with a protective one (`speed_alpha_down`), so a
  single fast "skip" task cannot collapse the worker target and a one-cycle dip
  cannot flap the bottleneck. These weights are deliberately distinct from the
  arrival-rate alphas.

  ![Line chart contrasting a noisy raw-speed line with a smoothed target-speed line that climbs quickly on rises and descends slowly on dips.](assets/01-asymmetric-ewma.png)

- **In-flight aging.** Raw speed (`1 / mean(duration)`) only updates on task
  completion, so a stage grinding on one long in-flight task would report a
  frozen, stale-high rate. The estimator caps the rate at one completion per
  elapsed-since-last-completion while work is in flight, so a stalled stage's
  `cap_src` falls instead of lying high.

  ![Line chart: while one long task is in flight with no completions, a naive estimate stays frozen high while the aged rate decays downward over time.](assets/01-inflight-aging.png)

- **Stall snap.** When a busy stage is overdue past `speed_stale_multiple ×` its
  mean service time, the protective damping is bypassed: `target_speed` snaps
  down, `w_sustain` is held at current workers, and growth is bounded to
  `+speed_stale_growth_step` so the collapsing rate cannot explode the divisive
  growth target into a node-filling request before completions resume.

  ![Line chart: target speed holds level, then snaps sharply down at an overdue threshold, after which growth is limited.](assets/01-stall-snap.png)

- **Skip exclusion.** A sample that produced no output **and** finished faster
  than `speed_estimation_min_task_duration_s` is a degenerate skip, not a
  service-rate observation; it is dropped from the speed window and the
  trusted-sample count. Real zero-output filter stages, which take real time,
  are still measured.

A stage's speed is **trusted** only after it clears
`speed_estimation_min_data_points` samples; until then `cap_src = 0` and the
stage is invisible to bottleneck selection.

## Trade-offs

| Cost | Benefit |
|---|---|
| One model owns all throughput reasoning, so every consumer must read it, not recompute. | Growth and shrink can never disagree about the pipeline's rate. |
| Protective `speed_alpha_down` lags a genuine slowdown by a few cycles. | A noisy fast cycle cannot flap the bottleneck or shrink a capable stage. |
| `bottleneck_rate` excludes cold stages (`cap_src = 0`). | A just-restarted stage's `0` rate cannot collapse every stage to `min_workers`. |

## Implementation pointer

- `capacity.py`: `cap_src`, `bottleneck_rate`, `next_bottleneck_rate`,
  `w_sustain`, `w_target`, and the `CapacityState` carried across cycles.
- `estimator.py`: per-worker raw speed, in-flight aging, stall detection,
  skip exclusion.
- `chain.py`: the source-to-stage fan-out factor used to normalize rates.
- Config: `speed_alpha_up`, `speed_alpha_down`, `speed_stale_multiple`,
  `speed_estimation_*`, `capacity_headroom` (see `tuning.md`).
