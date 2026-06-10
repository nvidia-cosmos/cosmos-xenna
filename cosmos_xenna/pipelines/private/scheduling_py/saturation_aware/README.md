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
(trusted    (rates,      (mult to   (FRAG,    (cap     (clamp     (write
 speed)      targets)     w_target)  read-     cold)    deletes    edited
                                     only)              to         Solution)
                                                        w_sustain)
```

- `capacity.py`: per-stage throughput / capacity model; selects the bottleneck
  and derives the hold target `w_sustain` and growth target `w_target`.
- `sizing.py`: turns `w_target` into a solver growth multiplier, gated by
  `has_local_input`.
- `chain.py`: cumulative fan-out factors and whole-chain stock in source units.
- `ramp.py`: cold-start clamp (+1 worker per cycle for not-yet-trusted stages).
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
consumer. Its rate drives `bottleneck_rate`, always sized from one stable rate,
the smoothed source capacity `cap_src`:

- warm queue cliff: use the candidate's own `cap_src` (its input is full by
  construction, so it is the live constraint).
- cold cliff / no cliff: fall back to the slowest measured `cap_src`, so a cold
  candidate's `0.0` rate never collapses sizing to `min_workers`.

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
