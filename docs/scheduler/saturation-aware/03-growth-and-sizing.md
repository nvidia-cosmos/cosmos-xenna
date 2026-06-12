# 03: Growth and sizing

## The problem

SAT must grow the **right** stage (the bottleneck) without editing the Rust
solver, which sizes each stage from the speed SAT reports for it. Two things can go wrong if we are naive:

1. **Grow the wrong stage.** Feeding the solver raw speeds lets it grow whatever looks busy, which over-provisions non-bottleneck stages for no throughput gain (their extra workers just fill the next queue);
2. **Over-grow a fast-fed stage.** A stage sitting behind a fast upstream producer has a deep local queue every cycle, so any local-backlog growth rule would keep inflating it even though the real supply cliff is elsewhere.

## What we do

Grow by **biasing the solver's input**, bounded by the capacity model so the
bias can never exceed what the bottleneck can absorb.

Per stage, the demand sizer (`sizing.py`) computes a multiplier `m`:

```
  if stage is below w_target AND has local input:
      m = w_target / max(workers, 1)
  else:
      m = 1.0                       (ask for nothing extra)

  the solver receives  Estimate(speed / m, num_returns)
```

![SAT feeds the FRAG solver a deflated speed (speed divided by m); the solver responds by allocating more workers to that stage.](assets/03-input-bias.png)

*Growth by input bias: SAT lowers the per-worker speed it reports for the
bottleneck (`speed / m`). The unmodified solver reads that as a slower stage and allocates it proportionally more workers.*

A deflated speed makes the solver allocate proportionally more workers to that
stage. Because `w_target` is bounded by the pipeline bottleneck (see
[01](01-capacity-model.md)):

- **only the bottleneck** stage has a `w_target` that climbs (toward
  `next_bottleneck_rate`), so it is the only stage the multiplier actually
  grows;
- every other stage is at or above its target, so `m = 1.0` and it asks for
  nothing extra;
- a stage with no local input is not grown speculatively; there is nothing to
  feed a new worker.

There is no separate backlog cap to tune and no per-cycle backlog division: the
multiplier cannot over-grow a stage past the bottleneck because the target it
chases is itself bounded by the bottleneck rate.

![Per-stage demand multiplier table. S2 the bottleneck has 2 workers and w_target 3, so m = 1.5 and it grows from 2 to 3 workers. S0 (8 workers, target 2) and S3 (starved, no input) both have m = 1.0 and do not grow.](assets/03-demand-multiplier.png)

*Only the bottleneck gets a multiplier above 1, so only it grows: S2 has
`w_target = 3` against 2 workers, so `m = 3/2 = 1.5` and the deflated speed sizes
it at `2 -> 3`. A stage at or above its target (S0), or with no local input
(S3), gets `m = 1.0` and asks for nothing extra.*

![Animated SAT to FRAG flow where SAT reports speed divided by m, FRAG accepts the biased ask, and the bottleneck stage grows from two workers to three.](assets/03-w-target-growth.gif)

*The motion is the whole trick: SAT changes the **input estimate** (`speed / m`),
FRAG still owns placement, and the result is bounded growth from the current
worker count toward `w_target` (`2 -> 3` here).*

Real limits still apply *after* the bias: GPU capacity, whole-GPU placement, and fragmentation all bound what the solver actually creates. SAT only changes the *ask*; the solver decides what is feasible, and the cold-start ramp
([04](04-cold-start-ramp.md)) then trims the result.

## Trade-offs

| Cost | Benefit |
|---|---|
| Growth is expressed indirectly, as a deflated speed the solver consumes. | The solver stays unmodified and still owns feasibility and placement. |
| Only one stage grows per cycle's bottleneck. | No non-bottleneck stage can over-grow and starve the real constraint. |
| A stage with an empty local queue is never grown, even mid-burst. | Speculative growth from placeholder/empty signals is structurally impossible. |

## Implementation pointer

- `sizing.py`: `size_pipeline` over per-stage `size_stage`; computes the
  per-stage multiplier `m` and the `has_local_input` gate.
- `scheduler.py`: builds the per-stage snapshot, applies `m` to the speed
  estimate handed to `run_fragmentation_autoscaler`.
- `capacity.py`: supplies `w_target` (bounded by `bottleneck_rate` /
  `next_bottleneck_rate`), the ceiling the multiplier chases.
