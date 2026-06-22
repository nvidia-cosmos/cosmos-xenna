# Operator Tuning Guide

The operator's quick-reference for the saturation-aware scheduler. The six
concept notes (`01` to `05`) plus [`README.md`](README.md) explain **why** each
mechanism exists; this guide explains **when to choose the scheduler** and
**which knob to turn**.

> **Source of truth**: every default below is mirrored on `SaturationAwareConfig`
> in [`config.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware/config.py).
> When this doc and that file disagree, `config.py` wins.

## When to use it

`FRAGMENTATION_BASED` is the production default and the right choice for most
pipelines. Prefer `SATURATION_AWARE` when a pipeline has **bursty input or
upstream bottlenecks that transiently starve an expensive downstream GPU stage**
(for example, a stage that loads a large model), and the default scheduler
scales that stage down during the lull, paying a cold-start reload cost when
work resumes.

Do **not** reach for it to fix placement, fragmentation, or a stage that is
simply slow; see [README: Non-goals](README.md#6-non-goals).

## How to select it

The scheduler is chosen per run on the streaming spec
(`specs.py::StreamingSpecificSpec`):

```python
from cosmos_xenna.pipelines.private.specs import SchedulerKind, SaturationAwareConfig

spec.scheduler = SchedulerKind.SATURATION_AWARE
spec.saturation_aware = SaturationAwareConfig()   # None → all defaults
```

The two schedulers are fully isolated; selecting one has no effect on the other.

## The knobs

Every field has a working default; a normal run touches none of them. There are
only a handful, grouped by what they affect.

### Cadence and read-ahead

| Field | Default | When to adjust |
|---|---|---|
| `interval_s` | `10.0` | Measure-and-decide cadence. Lower for tighter control on small clusters; raise on very large clusters where per-cycle work competes with pipeline traffic. It is **not** a backlog catch-up horizon. |
| `capacity_headroom` | `0.10` | Spare-rate read-ahead added on top of the bottleneck rate for a non-bottleneck stage's `w_target`. Raise slightly to keep buffers fuller; lower to pack tighter. |

### Speed estimation (rate stability)

| Field | Default | When to adjust |
|---|---|---|
| `speed_estimation_window_s` | `60.0` | Throughput-estimator window. Also the wait before a zero-sample stage is treated as a slow-starter ([04](04-cold-start-ramp.md)). |
| `speed_estimation_min_data_points` | `5` | Trust threshold below which a stage is cold/unmeasured (`cap_src=0`). |
| `speed_estimation_averaging_samples` | `10` | Samples retained for the `1/mean(duration)` average; raise for very heterogeneous task durations. Must be `>= speed_estimation_min_data_points`. |
| `speed_alpha_up` / `speed_alpha_down` | `0.3` / `0.1` | Asymmetric EWMA on per-worker speed. `up` modest so a fast skip can't collapse the target; `down` protective so a one-cycle dip can't flap the bottleneck. |
| `speed_stale_multiple` | `3.0` | Overdue factor (× mean service time) past which a busy, non-completing stage is treated as stalled and its rate snaps down. |
| `speed_stale_growth_step` | `1` | Max workers a stalled stage may add per cycle. |
| `speed_estimation_min_task_duration_s` | `1e-3` | Service-time floor separating a degenerate empty skip from real work. |
| `pipeline_warmup_growth_step` | `4` | Max workers a **trusted** stage may add per cycle while the pipeline is still warming (some stage is cold, so `bottleneck_rate` and the derived `w_target` are provisional). Bounds an upstream stage from leaping to a node-filling size and starving a still-cold downstream stage of a shared resource ([04](04-cold-start-ramp.md)). Lower is safer; raise to ramp a genuinely source-bound pipeline faster. No effect once every stage is trusted. |

### Scale-down release (anti-shrink)

| Field | Default | When to adjust |
|---|---|---|
| `scale_down_release_cycles` | `6` | Base release speed: `alpha_down = 1 / (scale_down_release_cycles × scale_down_release_slowdown)`. Larger holds a stage warm longer through a lull. |
| `scale_down_release_slowdown` | `4.0` | Uniform extra hold multiplier. Set to `1.0` to restore the fast base release for every stage. |
| `reclaim_confirm_cycles` | `6` | Consecutive cycles an idle, over-provisioned downstream stage must look beneficial-to-reclaim (its freed resource would help an under-capacity stage grow) before its warm pin releases. Lower to return stranded resources sooner; raise to be more conservative against a transient demand spike. |

## Symptom → knob index

| Symptom | First thing to check / turn |
|---|---|
| Expensive GPU stage shrinks during a brief upstream lull, then cold-starts | Raise `scale_down_release_cycles` or `scale_down_release_slowdown` (hold longer). First confirm the lull is transient, not a sustained upstream bottleneck. |
| Bottleneck identity flaps between two stages cycle to cycle | Lower `speed_alpha_up` toward `0.2` so a transient fast task doesn't swing `target_speed`. |
| A stage stuck on a long task reports a frozen-high rate | Lower `speed_stale_multiple` toward `2.0` so the stall is detected sooner. |
| Heavy stage warms one model at a time, budget idle | It should auto-release after `speed_estimation_window_s` with work waiting ([04](04-cold-start-ramp.md)); shorten the window only if warmup genuinely needs it. |
| Upstream stage leaps to a node-filling size on its first trusted cycle and starves a downstream stage of a shared resource (e.g. CPU) | The pipeline-warming gate bounds this to `pipeline_warmup_growth_step`/cycle while any stage is still cold ([04](04-cold-start-ramp.md)); lower it if the leap still strands a downstream stage during warmup. |
| Expensive GPU stage idle with `qstate=starved`, `local_pending=0` | **Mostly not a knob.** It is downstream of the bottleneck and underfed; the floor now reclaims its stranded resource for a blocked same-resource grower (see `reclaim_confirm_cycles`), but the durable fix is the upstream feeder ([02](02-bottleneck-selection.md)). |
| Over-provisioned idle downstream stage strands CPU/GPU while another stage wants to grow | Lower `reclaim_confirm_cycles` so the floor releases the stranded resource sooner. A stage is reclaimed only when every resource it reserves is one the growing stage also uses, so a stage holding a GPU is never torn down to feed a CPU-only grower. |
| Reaction feels slow on a small cluster | Lower `interval_s`; watch that per-cycle work stays well under the interval. |

## Tuning discipline

- **Wait for warm.** Most "first-impression" symptoms are cold-start artefacts
  the ramp clears on its own; let the pipeline run past
  `speed_estimation_window_s` before tuning.
- **One knob at a time.** Change one field, run, observe, then change the next.
- **Don't tune around a non-goal.** If the bottleneck is genuinely slow at high
  worker counts, that is the stage's throughput, not a scheduler setting
  ([README: Non-goals](README.md#6-non-goals)).
