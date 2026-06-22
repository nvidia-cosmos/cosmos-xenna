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

> **What counts as "pending work".** The gate is the stage's **active depth**
> being greater than zero, not the input queue alone. Any queued input, any
> pool-queued batch, or any in-flight task all count
> (`activity.py::StageActivity.active_depth` is
> `queue + (pool + inflight) * batch_size`). So a stage still draining in-flight
> work is not "dry", and **any** nonzero amount trips the gate, not a full batch.
> The first worker is always allowed (a 0-worker stage is capped at 1 even with
> no work, so it can start and record a first sample); this gate guards the **+1
> on each later cycle**. It applies only while the stage is untrusted; a trusted
> stage is sized by `w_target` ([01](01-capacity-model.md)) regardless.

```
  stage state                              ramp decision
  ───────────                              ─────────────
  trusted, capacity has a target           cap at w_target  (the growth ceiling)
  trusted, pipeline still warming          cap at current + pipeline_warmup_growth_step
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

The growth step is a fixed `+1`, never scaled by sample count or by the solver's proposal, so a not-yet-trusted stage can **never** convert a large placeholder-driven proposal into a first-cycle burst. A stage with no active work (empty input, nothing pooled, nothing in flight) is never grown speculatively; queue-gradient capacity ([02](02-bottleneck-selection.md))
instead grows the upstream producer.

**Slow-starter release.** The +1 cap assumes a stage produces its first sample within a cycle or two. A heavy stage whose first completion lands far in the future (a stage whose model load and `torch.compile` dwarf the estimation window) would otherwise stay pinned at one worker for the whole warmup, loading a single model while the rest of its budget sits idle. So once a full `speed_estimation_window_s` has elapsed with **zero** samples and the stage still has **work waiting**, it is treated as a confirmed slow-starter and released to the solver: all its workers spawn now and their models load in parallel. The "work waiting" gate is essential: it distinguishes a slow-warming stage with a real backlog (which needs all its workers) from a merely *starved* stage (which would otherwise scatter sub-GPU workers from placeholder throughput, the exact fragmentation the cap prevents).

The ramp **only trims additions**: it never adds a worker and never blocks a shrink. **Pinned** stages (operator-declared `num_workers`) are exempt: there is no evidence to ramp toward, so the solver may take them straight to the requested size.

## Pipeline-warming growth gate

The cap above governs an *untrusted* stage. A different over-growth happens to a
**trusted** stage while the rest of the pipeline is still cold. `bottleneck_rate`
is the slowest *measured* `cap_src`, and cold stages (`cap_src = 0`) are
deliberately excluded so a not-yet-warm stage cannot collapse the rate to zero
and tear down the feeders keeping it supplied ([01](01-capacity-model.md)). The
side effect is that while any stage is still cold, `bottleneck_rate` - and every
`w_target` derived from it - is **provisional and biased high**: it reflects only
the fast stages that happened to warm first.

A trusted upstream stage sized from that provisional rate can compute a
node-filling `w_target` and, with nothing bounding the trusted growth path, leap
to it in a single cycle. In production run `f9fa2dde` a CPU frame-extraction
stage jumped from 2 to 57 workers on its first trusted cycle, exhausted the
node's CPUs, and starved a downstream whole-GPU caption stage that needed those
CPUs to place its workers - a **priority inversion** where a cheap upstream stage
denied a resource the expensive downstream stage required, leaving GPUs idle.

The gate: while the pipeline is **still warming** - any work-bearing stage is
untrusted, so `bottleneck_rate` is provisional - a trusted stage's growth is
bounded to `current + pipeline_warmup_growth_step` (default **4**) per cycle
instead of jumping straight to `w_target`. `w_target` remains the ceiling, so a
stage that does not want to grow is unaffected, and the cap only trims
additions, never forcing a shrink.

![Two-row infographic "pipeline-warming growth gate". Without the gate, an upstream CPU stage jumps from 2 to 57 workers in one cycle, the node-CPU bar is exhausted, and a downstream whole-GPU stage cannot place so its GPUs sit idle. With the gate, the upstream stage grows 2 to 6 to 10 by a small step per cycle, the node-CPU bar stays available, and the downstream whole-GPU stage places and warms in parallel. Banner: self-releasing - opens once every work-bearing stage is trusted.](assets/04-pipeline-warming-gate.png)

*While the pipeline is still warming, `bottleneck_rate` is provisional and biased
high, so a trusted upstream stage sized from it would jump to a node-filling
`w_target` in one cycle and starve the shared resource a still-cold downstream
stage needs. Bounding trusted growth to `pipeline_warmup_growth_step` per cycle
keeps that resource available so the downstream stage can place and warm.*

The gate is **self-releasing**: it reads one pipeline-wide signal each cycle and
opens automatically the moment every work-bearing stage is trusted, after which
the full `w_target` cap takes over with no behavioral change. It never touches
`bottleneck_rate`, so the cold-exclusion guard above is preserved. It is
resource-shape-agnostic - the same rule bounds a CPU, fractional-GPU, or
whole-GPU stage - so it special-cases no stage, resource, or model.

## Trade-offs

| Cost | Benefit |
|---|---|
| A trusted stage's first ramp is slower than the solver's one-shot fill. | Resource-shape-agnostic: no fractional-GPU stage can fragment the cluster on cycle one. |
| Slow-starter release needs a full window of zero samples before firing. | A heavy stage warms all its models in parallel instead of one at a time. |
| The cap is a fixed `+1`, ignoring how large the solver's proposal was. | A placeholder-driven over-spawn is structurally impossible. |
| While the pipeline warms, a trusted source-bound stage grows by only `pipeline_warmup_growth_step`/cycle instead of one shot. | A trusted stage cannot over-claim a shared resource off a provisional cold-start rate and starve a still-cold downstream stage. |

## Implementation pointer

- `ramp.py::decide`: the pure, per-stage ramp decision (cold / warming /
  slow-start / trusted / trusted-while-warming branches).
- `scheduler.py::_apply_cold_start_ramp`: feeds `has_pending_work`, sample
  count, window age, and the pipeline-wide `pipeline_warming` flag into the
  ramp; trims new workers via the `SolutionEditor`.
- Config: `speed_estimation_min_data_points` (trust threshold),
  `speed_estimation_averaging_samples` (averaging depth),
  `speed_estimation_window_s` (estimation + slow-starter window),
  `pipeline_warmup_growth_step` (trusted-stage growth bound while warming),
  `interval_s` (decide cadence) (see `tuning.md`).
