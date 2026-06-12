# Saturation-Aware Scheduler (`SATURATION_AWARE`)

A pure-Python, backlog-aware streaming autoscaler that biases the inputs of the
shared Rust fragmentation solver and post-processes its `Solution`. It changes no
fragmentation-scheduler code; the solver is always called read-only.

The full design rationale lives in the parent repository at
`docs/curator/design/saturation-aware-scheduler.md`. This README is the
submodule-local, implementation-facing summary; module docstrings hold the
authoritative per-module contracts.

## Enabling

Select the scheduler on the streaming-mode spec (`specs.StreamingSpecificSpec`):

- `scheduler = SchedulerKind.SATURATION_AWARE`
- optional `saturation_aware = SaturationAwareConfig(...)` (`None` uses defaults)

It is ignored in `BATCH` execution mode.

## Per-cycle flow

Each `autoscale` pass (`scheduler.py`) composes these modules around the
read-only solver:

```
snapshot --> capacity --> demand --> solve --> ramp --> floor --> commit
(trusted    (rates,      (mult to   (FRAG,    (cap to  (clamp     (write
 speed)      targets)     w_target)  read-     w_target deletes    edited
                                     only)     + cold)  to         Solution)
                                                        w_sustain)
```

- `capacity.py`: per-stage throughput / capacity model; selects the bottleneck
  and derives the hold target `w_sustain` and growth target `w_target`.
- `sizing.py`: turns `w_target` into a solver growth multiplier, gated by
  `has_local_input`.
- `chain.py`: cumulative fan-out factors and whole-chain stock in source units.
- `ramp.py`: per-cycle growth clamp - not-yet-trusted stages grow +1 worker per
  cycle; a trusted stage is capped at its capacity target `w_target` (the growth
  ceiling), unless `w_target` is a placeholder (`w_target_is_real == False`: cold
  speed, no measured bottleneck, or collapsed source fan-out) - then the solver
  owns growth.
- `floor.py`: scale-down floor; clamps deletes to `w_sustain` while stock is
  still in flight.

## Runtime-signal timing

`SATURATION_AWARE` is runtime-aware: `streaming.py` defers signal submission
until **after** completed-task transfer/dispatch, so the per-stage queue depths
it reads are post-transfer (a downstream input queue already reflects upstream
output). `RuntimeSignals` carries queue depth (used for growth) plus pool-queued
and in-flight task counts (used by the release gate). `inflight_slots` counts
logical in-flight tasks, not raw actor-rank slots, so an SPMD worker group (one
logical task spread across `world_size` rank actors) is not over-counted.

## Queue-gradient target state

The target steady state is a self-balancing pipeline: every inter-stage buffer
stays populated, the bottleneck runs near 100%, and non-bottleneck stages run
below 100% (and may block on queue-put under backpressure). `classify_stages`
labels each stage from its own and its downstream's queue occupancy:

- `STARVED`: input queue below one batch (waiting on upstream).
- `BOTTLENECK`: input populated but the downstream input queue is starved (this
  stage cannot feed downstream fast enough); for the last stage, input populated
  with no ready worker.
- `BUFFERED`: input populated and downstream input also populated; not the
  constraint.
- `BALANCED`: last stage, input populated, with ready workers to spare.

The bottleneck is the deepest stage with a full input queue and an under-fed
consumer. It owns growth (sticky identity), but the pipeline-sizing rate
`bottleneck_rate` is always the slowest MEASURED source capacity `cap_src`:

- a serial pipeline runs no faster than its slowest stage, so the sizing rate
  is the measured minimum. A warm queue-cliff candidate that is the genuine
  constraint already IS this minimum, so the common case is unchanged.
- a chain-factor collapse can only inflate a stage's `cap_src` (it is the
  reciprocal of a near-zero fan-out), so the measured minimum is structurally
  immune and clamps a transiently corrupted candidate before its impossible
  rate poisons every stage's smoothed arrival.
- a cold candidate (`cap_src == 0.0`) is excluded from the minimum, so its
  `0.0` rate never collapses sizing to `min_workers`. The candidate's own
  (possibly inflated) `cap_src` stays visible on `bottleneck_candidate_rate`,
  which exceeds `bottleneck_rate` whenever the candidate is not itself the
  slowest measured stage (a fast warm candidate fed by a slower upstream, or a
  chain-factor collapse).

`cap_src` is built from each stage's per-worker speed smoothed with dedicated
EWMA weights (`speed_alpha_up` modest, `speed_alpha_down` protective), so one
transient fast or slow task cannot swing the sizing rate. The underlying speed
is averaged over `speed_estimation_averaging_samples` completed tasks, decoupled
from the smaller `speed_estimation_min_data_points` cold-start trust gate.

The windowed speed only updates on task completion, so a stage stuck on a long
in-flight task is aged: while `inflight > 0` the per-worker rate is capped at
one completion per elapsed-since-last-completion, so a stalled feeder is no
longer reported as fast. A genuine stall (`rate_is_stale`: busy and overdue past
`speed_stale_multiple` times its mean service time) bypasses the protective
down-damping and snaps `target_speed` down to the aged rate; its hold target is
held at the current worker count and its growth target is bounded to
`workers + speed_stale_growth_step`, so a collapsing `target_speed` cannot
explode the divisive worker targets. Normal completion variance keeps the
protective damping (no churn).

## One-batch boundary

"Has at least one batch of work" uses one boundary everywhere:

- growth: `local_pending >= batch_size` (`_Cycle.has_local_input`).
- floor: `stock_src >= source_stock_threshold` for a positive threshold; for a
  collapsed (`0.0`) threshold from zero / sub-`MIN_CHAIN_FACTOR` fan-out, only
  strictly positive stock counts, so a fully drained stage can still release.

## Composition with the legacy backlog-aware guard

The Rust `enable_backlog_aware_scaledown` guard is independent of this scheduler.
If both are enabled, the two scale-down clamps compose; exercise that
combination deliberately when tuning.
