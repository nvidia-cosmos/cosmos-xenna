# 12 — Multi-Target DAG Growth

## TL;DR

Phase C of the autoscale cycle attempts a worker addition for
**every** stage that has positive intent this cycle, walking
stages in **downstream-first DAG-depth order**. The
fragmentation-based scheduler grows only the single most-saturated
stage per cycle, so an N-stage chain spends N cycles ramping under
sustained pressure.

## Problem

Cosmos-Xenna's fragmentation-based scheduler — the scheduler kind
the saturation-aware autoscaler replaces — picked one "most
saturated" stage per cycle and tried to grow it. The cycle never
attempted growth for any other stage, even when several stages
were queue-bound at the same time.

That is fine when a single stage is the bottleneck. It is
painful in any pipeline where multiple stages saturate
concurrently:

- **N cycles to ramp**. An N-stage chain with N saturated stages
  needed N cycles before the last one received its first added
  worker. At the default `autoscale_interval_s`
  (`60 * 3.0` seconds) that is multiple minutes per added worker
  per stage — minutes of wasted compute on production runs.
- **Wrong-stage-first**. Greedy "most saturated wins" routinely
  picked an upstream stage whose output was already piling up
  behind a saturated downstream stage. Adding capacity there
  just deepens the backlog: the downstream stage still cannot
  consume the extra work, so the new upstream worker waits on
  the same queue as the old ones.

## Decision

Phase C **attempts every stage with positive intent in the same
cycle**, ordered by DAG depth (downstream-first). One stage's
cluster-full failure never blocks another stage from being
attempted.

```
   Before — fragmentation-based scheduler (single-target growth)

   cycle k                       cycle k+1                     cycle k+2
   ┌───┐  ┌───┐  ┌───┐  ┌───┐    ┌───┐  ┌───┐  ┌───┐  ┌───┐    ┌───┐  ┌───┐  ┌───┐  ┌───┐
   │ A │→ │ B*│→ │ C*│→ │ D*│    │ A │→ │ B*│→ │ C*│→ │ D⁺│    │ A │→ │ B*│→ │ C⁺│→ │ D │
   └───┘  └───┘  └───┘  └───┘    └───┘  └───┘  └───┘  └───┘    └───┘  └───┘  └───┘  └───┘
                                                      ▲                         ▲
                                               most-saturated           next-most-saturated
                                                 stage grows                 stage grows


      After — saturation-aware Phase C (multi-target, DAG-priority growth)

                                  cycle k
                          ┌───┐  ┌───┐  ┌───┐  ┌───┐
                          │ A │→ │ B⁺│→ │ C⁺│→ │ D⁺│
                          └───┘  └───┘  └───┘  └───┘
                                   ▲      ▲      ▲
                                   3      2      1   ◄── downstream-first
                                                          attempt order

                * = saturated stage (positive intent)
                ⁺ = +1 worker placed this cycle
```

Downstream-first matters because clearing the bottleneck at the
tail lets upstream queues drain naturally. Growing an upstream
stage whose downstream is still saturated just builds more
in-flight work behind the same wall — the new upstream worker
parks its output on the queue and the bottleneck is unchanged.

**Trade-off**. Phase C now does one `try_add_worker` attempt per
saturated stage per cycle instead of one per cycle total, plus
the per-attempt clamp against the per-stage hard cap. The win is
the ramp completes in one cycle instead of N, so the autoscale
cadence is no longer the dominant ramp latency on multi-stage
pipelines. The work per cycle is bounded by the per-stage
growth cap and the cluster's available placement budget, so
worst-case Phase C work scales as the number of saturated
stages, not the number of workers added.

## How it works

[`_run_phase_c_grow`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
walks stages in the order returned by the unified
[`compute_grow_priority_order`](../../../cosmos_xenna/pipelines/private/scheduling_py/dag_priority.py)
helper. The helper implements a three-level hierarchy whose first
non-trivial branch wins:

1. **Bottleneck-engaged path** (when the
   [25 — bottleneck-decision gate](25-bottleneck-decision-integration.md)
   has engaged and `enable_bottleneck_priority_growth=True`): order
   stages by EWMA-smoothed `D_k` descending, with topological depth
   descending as the tiebreaker.
2. **DAG-priority path** (default fallback): order stages by
   topological depth descending. Xenna pipelines are linear
   streaming chains, so DAG depth equals the stage's position in
   `problem.rust.stages`; downstream-first order is just the
   reversed stage list.
3. **Problem order** (when `enable_dag_priority_growth=False` and
   the bottleneck gate is disengaged): walk stages in problem
   (upstream-first) order.

For every non-finished stage in that order:

1. **Read the intent** from `_last_intent_deltas`. Non-positive
   intent (NORMAL, STARVED, OVER_PROVISIONED) is a no-op for
   Phase C; the stuck-plan counter for the stage is reset to 0
   and the loop moves on.
2. **Clamp to remaining headroom** under the per-stage ceiling
   `min(max_workers, max_workers_per_node * N)`. If the clamp
   reduces the request to zero, skip and reset the stuck counter.
3. **Attempt placement** with `ctx.try_add_worker(stage_index)`
   up to the clamped intent.
4. **On cluster-full** (`try_add_worker` returns `None`), the
   loop delegates to the
   [cross-stage donor](13-cross-stage-donor.md) fallback. If
   the donor protocol frees a placement, one retry attempts to
   spend it for this receiver; otherwise the stage takes a
   deficit for the cycle.
5. **Track unmet intent** by incrementing
   `_stuck_plan_counters[stage_name]` whenever added < intent
   (and resetting it on full satisfaction). The pipeline-level
   `stuck_plan_detection_cycles` watchdog consumes this counter.

The loop does **not** short-circuit when one stage hits its cap
or exhausts cluster capacity: each saturated stage is attempted
independently. Cluster placement budget is consumed in
downstream-first order, so the bottleneck stage gets first dibs
on a scarce slot.

## Knobs

| Field | Where | Effect |
|---|---|---|
| `enable_dag_priority_growth` | `SaturationAwareConfig` | `True` (default) walks stages downstream-first via `compute_grow_priority_order` when the bottleneck gate is disengaged; `False` walks them in problem order. Multi-target growth itself is always on; the toggle controls *order* only. |
| `enable_bottleneck_priority_growth` | `SaturationAwareConfig` | `True` (default) lets [25 — bottleneck decision integration](25-bottleneck-decision-integration.md) override DAG depth with `D_k` descending order when the heterogeneity gate engages; `False` keeps the DAG-depth-only ordering described above regardless of `D_k`. |
| `max_workers` | `SaturationAwareStageConfig` | Per-stage worker ceiling. Clamps positive intent before any `try_add_worker` call; see [16 — hard caps and floors](16-hard-caps-and-floors.md). |
| `max_workers_per_node` | `SaturationAwareStageConfig` | Combined with the cluster's node count to compute the effective per-stage cap that clamps the request. |

Phase C also consumes the cross-stage donor toggles on
`SaturationAwareConfig` when the cluster is full but a
downstream stage still has positive intent. See
[13 — cross-stage donor](13-cross-stage-donor.md) for the
five-layer anti-flap protocol that gates donor selection.

## See also

- [00 — Per-cycle overview](00-overview.md) — where Phase C
  fits in the four-phase autoscale cycle.
- [11 — Growth-mode state machine](11-growth-mode-state-machine.md)
  — how positive intent is generated before this loop runs.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the
  cluster-full path that Phase C delegates into when
  `try_add_worker` returns `None`.
- [16 — Hard caps and floors](16-hard-caps-and-floors.md) —
  the per-stage ceiling that clamps intent before growth is
  attempted.
- [25 — Bottleneck decision integration](25-bottleneck-decision-integration.md)
  — the Forced-Flow-Law `D_k`-driven override on top of this
  ordering and the matching Phase D shrink protection.
