# 04: Cold-start ramp

## The problem

On the first cycle no stage has completed a task, so every stage reports the
solver's **default placeholder speed**. Sizing from that guess is dangerous. The
original failure mode was GPU fragmentation: a cheap fractional-GPU stage (say a
0.25-GPU stage) won the first fill, scattered quarter-GPU workers across
every GPU, and blocked a whole-GPU stage from ever placing a worker. The
general problem is broader: **any** unmeasured stage can be over-grown before it has evidence a new worker will be used.

![Eight GPUs, each holding a small fractional worker in one corner so no GPU is fully free. A whole-GPU stage above them is blocked and cannot be placed.](assets/04-cold-start-fragmentation.png)

*Cold-start fragmentation: sized from a placeholder speed, cheap fractional-GPU workers scatter across every GPU and leave none fully free, so a whole-GPU stage can never place. The ramp caps an untrusted stage to +1 worker per cycle to prevent this.*

## What "trusted" means

A stage's per-worker speed is **trusted** once it has recorded at least
`speed_estimation_min_data_points` completed-task samples (default **5**). Until
then it is **untrusted** and the ramp owns its growth. Untrusted has two
sub-states:

- **cold**: zero samples. There is no measured speed at all, so `cap_src = 0`
  and the stage is invisible to bottleneck selection ([01](01-capacity-model.md)).
- **warming**: at least one sample but fewer than `min_data_points`. A speed
  exists, but it is not yet stable enough to size from.

![Trust lifecycle along a sample-count axis: cold at zero samples, warming from one to four, trusted at five or more (min_data_points). Cold and warming are untrusted and ramped at +1 worker per cycle.](assets/04-trust-lifecycle.png)

*Trust accrues from **task completions, not clock time**: a fast stage trusts in
a cycle or two, a slow one takes longer. The scheduler re-reads each stage's
sample count every `interval_s` (default **10 s**), and the underlying speed is
averaged over a `speed_estimation_window_s` window (default **60 s**).*

`min_data_points` is the **trust** threshold only. It is deliberately separate
from `speed_estimation_averaging_samples` (default **10**), the number of recent
samples the `1 / mean(duration)` estimate averages over for stability. Trust can
fire at 5 samples while the average still smooths over up to 10, so cold-start
stays quick without a noisy per-worker speed (the config enforces
`averaging_samples >= min_data_points`). The same `speed_estimation_window_s`
also bounds the **slow-starter release** below.

## What we do

Cap a **not-yet-trusted** stage's post-solve worker count by trimming the solver's proposed new workers. One generic rule covers every resource shape (CPU, fractional-GPU, whole-GPU all behave identically):

> A not-yet-trusted stage may grow by **at most one worker per cycle**, and only
> when it has its **own pending work** to feed the new worker.

```
  stage state                              ramp decision
  ───────────                              ─────────────
  trusted, capacity has a target           cap at w_target  (the growth ceiling)
  trusted, no measured bottleneck yet      uncapped (no target this cycle; solver grows)
  warming (0 < samples < min_data_points)  +1/cycle if it has pending work, else hold
  cold,  pending work present              +1/cycle (warm a worker before 1st sample)
  cold,  no pending work                   cap at 1 worker
  cold,  window elapsed + work waiting     uncapped → slow-starter release (below)
```

![Step chart: the solver would fill to N workers at once, but the ramp cap rises by only one worker per cycle.](assets/04-ramp-step.png)

*An untrusted stage grows by at most one worker per cycle, no matter how large
the solver's proposal, until it has a measured speed.*

![Animated cold-start ramp where a large solver proposal is trimmed to one additional worker per cycle until enough task completions make the stage trusted.](assets/04-ramp-warmup.gif)

*The ramp treats trust as earned evidence: before enough completions arrive, a
large placeholder-speed proposal is trimmed to `+1` per cycle; once trust is
reached, the normal `w_target` cap takes over.*

The growth step is a fixed `+1`, never scaled by sample count or by the solver's proposal, so a not-yet-trusted stage can **never** convert a large placeholder-driven proposal into a first-cycle burst. A locally dry stage is never grown speculatively; queue-gradient capacity ([02](02-bottleneck-selection.md))
instead grows the upstream producer.

**Slow-starter release.** The +1 cap assumes a stage produces its first sample within a cycle or two. A heavy stage whose first completion lands far in the future (a stage whose model load and `torch.compile` dwarf the estimation window) would otherwise stay pinned at one worker for the whole warmup, loading a single model while the rest of its budget sits idle. So once a full `speed_estimation_window_s` has elapsed with **zero** samples and the stage still has **work waiting**, it is treated as a confirmed slow-starter and released to the solver: all its workers spawn now and their models load in parallel. The "work waiting" gate is essential: it distinguishes a slow-warming stage with a real backlog (which needs all its workers) from a merely *starved* stage (which would otherwise scatter sub-GPU workers from placeholder throughput, the exact fragmentation the cap prevents).

The ramp **only trims additions**: it never adds a worker and never blocks a shrink. **Pinned** stages (operator-declared `num_workers`) are exempt: there is no evidence to ramp toward, so the solver may take them straight to the requested size.

## Trade-offs

| Cost | Benefit |
|---|---|
| A trusted stage's first ramp is slower than the solver's one-shot fill. | Resource-shape-agnostic: no fractional-GPU stage can fragment the cluster on cycle one. |
| Slow-starter release needs a full window of zero samples before firing. | A heavy stage warms all its models in parallel instead of one at a time. |
| The cap is a fixed `+1`, ignoring how large the solver's proposal was. | A placeholder-driven over-spawn is structurally impossible. |

## Implementation pointer

- `ramp.py::decide`: the pure, per-stage ramp decision (cold / warming /
  slow-start / trusted branches).
- `scheduler.py::_apply_cold_start_ramp`: feeds `has_pending_work`, sample
  count, and window age into the ramp; trims new workers via the
  `SolutionEditor`.
- Config: `speed_estimation_min_data_points` (trust threshold),
  `speed_estimation_averaging_samples` (averaging depth),
  `speed_estimation_window_s` (estimation + slow-starter window), `interval_s`
  (decide cadence) (see `tuning.md`).
