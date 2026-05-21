# 21 — Allocation Error Tolerance

## TL;DR

Phase C grow treats a `None` return from
`AutoscalePlanContext.try_add_worker` as a **transient** allocation
failure: the partial result is recorded, the per-stage
`_stuck_plan_counters` entry is ticked by one, and the unsatisfied
intent is retried next cycle. Phase B floor enforcement is the only
place that escalates allocation failure to a hard `RuntimeError`, and
only after `floor_stuck_grace_cycles` of repeated no-progress
no-donor misses. The split routes cluster-fragmentation noise away
from the operator while keeping a monotonic counter on hand for the
genuinely stuck plans.

## Problem

The Rust planner that backs `AutoscalePlanContext` (Fragmentation
Gradient Descent) routinely refuses a placement when the cluster is
temporarily full or the receiver's worker shape does not match the
free fragments. Three causes recur in production:

- The cluster placement budget is briefly exhausted because a peer
  stage absorbed the free slots earlier in the same cycle's
  DAG-priority walk.
- The receiver's worker shape (multi-GPU, fractional GPU, NVDEC
  reservation) does not fit the surviving fragmentation pattern on
  the working snapshot.
- A donor was selected via the saturation-mode cross-stage donor,
  but the post-donation `try_add_worker` retry came back empty
  because the freed slot does not satisfy the receiver shape.

Raising `RuntimeError` on each of these would tear down a pipeline
that is otherwise healthy and would re-fire seconds later under
identical conditions. Silently dropping the intent without
bookkeeping would hide a permanently infeasible stage behind cycles
of "almost worked" logs. The planner must therefore stay neutral on
policy — return `None` and let the scheduler decide.

## Decision

Phase C absorbs the `None` and increments per-stage state; Phase B
escalates after a configurable grace. The planner contract carries
the policy: `try_add_worker` returns `None` on transient failure
rather than raising, so the planner never decides cycle-level
behaviour.

```
        Cycle N                Cycle N+1               Cycle N+2
        ─────────              ─────────               ─────────

  intent[S] = +2          intent[S] = +2          intent[S] =  0
  (SATURATED streak)      Phase D in cycle N      stage settles
                          freed peer slots        below saturation
                          elsewhere               boundary

  try_add → ✓ (1 fits)    try_add → ✓             no positive intent
  try_add → ✗ (full)      try_add → ✓             for S this cycle
  donor   → ✗  break
  ───────────────────     ───────────────────     ───────────────────
  added = 1 < intent      added = 2 = intent      added = intent = 0
  counter[S]: 0 → 1       counter[S]: 1 → 0       counter[S]: 0 → 0
  WARN: partial add       (silent)                (silent)
  invariant: 0 → 1 ✓     invariant: 1 → 0 ✓     invariant: 0 → 0 ✓
```

**Trade-off.** A stage that is structurally infeasible — no worker
shape will ever fit on this cluster — keeps logging WARN until the
operator notices, instead of failing fast. That cost is paid for the
dominant case (transient fragmentation), where the next cycle
recovers without a pipeline restart and without operator action.

## How it works

`_run_phase_c_grow` walks the per-stage positive intent in
DAG-depth-descending order. For each stage it calls
`ctx.try_add_worker(stage_index)` up to `intent` times. On the first
`None` return the saturation-mode cross-stage donor fallback
(`_attempt_cross_stage_donation`) is attempted; on donor selection
failure the inner loop breaks and the cycle moves on to the next
stage with a single per-stage WARNING.

Post-donation retry failure is treated differently. The donor was
already removed from the planner snapshot via `ctx.try_remove_worker`
before the receiver retry runs, and that removal cannot be rolled
back safely (the cluster snapshot, FGD reuse map, and worker-age
map all already reflect the removal). A `None` retry response would
otherwise leave the cluster with one-sided shrink (donor gone,
receiver did not grow). The scheduler therefore surfaces the failure
as a synthetic `RuntimeError("donor-retry-failed: ...")` and routes
it through `_absorb_allocation_failure`, which:

- increments the standard allocation-failure Counter so the
  one-sided shrink is operator-visible alongside other Phase C
  allocation failures;
- emits the per-GPU fragmentation snapshot at ERROR level, naming
  both the receiver and the donor stage;
- aborts the rest of Phase C for the cycle (under the default
  `skip_cycle_on_allocation_error=True`) or re-raises (when the
  kill switch is off);
- leaves `_record_donation_success` un-called, so donor cooldown
  and the per-cycle receiver counter remain at their pre-donation
  values; the donor stage stays eligible to be revisited next
  cycle without penalty.

At the bottom of the per-stage loop the counter is updated in one of
three ways:

- `intent <= 0` (no positive intent, or the hard worker-cap ceiling
  clamped the request to zero headroom) → `counter = 0`.
- `added == intent` (request fully satisfied this cycle) →
  `counter = 0`.
- `added < intent` (partial add) →
  `counter = counter + 1`.

The transition shape is enforced at the end of every cycle by
`check_stuck_plan_monotonicity` in
[`invariants.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/invariants.py).
Only `curr == 0` (reset) or `curr == prev + 1` (strict increment)
are legal; any other shape raises `SchedulerInvariantError`. The
pre-Phase-C snapshot stored in `prev_stuck_plan_counters` is the
input to that check, so a Phase C bug that double-increments or
skips a reset is caught before the plan reaches the actor pool.

Phase B floor enforcement (`_run_phase_b_floor`) uses a parallel but
stricter mechanism in `_floor_stuck_counters`: an allocation miss
with no eligible donor and no in-cycle progress increments the
counter; any in-cycle progress resets it. Once the counter exceeds
`floor_stuck_grace_cycles` the scheduler raises `RuntimeError` from
`_on_floor_stuck`. A post-donation retry miss inside Phase B raises
immediately because the donor removal has already been staged and
cannot be rolled back safely. The asymmetry between B and C reflects
intent: a missed floor is a structural requirement (the stage cannot
run); a missed Phase C grow is a throughput dip.

A second concern is *measurement* feedback rather than placement
feedback. Stages with multi-minute model load (large checkpoints,
lengthy initialisation) need queue depth to build ahead of
`stage_setup` completing, otherwise the dispatcher's measurement
loop reads "empty queue" right when a fresh actor becomes ready
and starves the stage Phase C just grew. The `setup_aware_max_queued` field on
`SaturationAwareStageConfig` is the policy hook that keeps queue
credit reserved for stages whose actors are still warming,
decoupling the "we asked for +2" decision from the "we observed an
empty queue" feedback that arrives one or two cycles later.

### Defense-in-depth for raised exceptions

The default Phase C path consumes the documented `None` return
contract directly. As a second line of defense for any future
planner-internal raise of `AllocationError`,
`_run_phase_c_grow` calls `try_add_worker` through
`_try_add_worker_with_defense`. The defense layer catches
`AllocationError` only; any other exception (e.g.
`SchedulerInvariantError`, `KeyError`, `IndexError` from a planner
bug) propagates out of `autoscale()` so the operator sees the
real defect instead of having it silently re-classified as a
transient allocation failure. The absorbed `AllocationError`
branch routes through `_absorb_allocation_failure`, which:

- emits an ERROR log carrying a per-GPU fragmentation snapshot
  (`(node, gpu_index, used_fraction, free_fraction)` for every GPU
  in the cluster, ordered by `(node_id, gpu.index)` so two snapshots
  compare as identical strings);
- increments the
  `xenna_scheduler_allocation_failures_total{stage,pipeline}`
  Counter;
- when `skip_cycle_on_allocation_error=True` (default), sets a
  per-cycle skip flag so the Phase C loop returns early; when
  `False`, re-raises the original exception so the cycle dies loud.

## Knobs

All fields live in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Field | Class | Default | Role |
|---|---|---|---|
| `floor_stuck_grace_cycles` | `SaturationAwareConfig` | `6` | Cycles a Phase B floor miss may persist without progress before `RuntimeError` |
| `stuck_plan_detection_cycles` | `SaturationAwareConfig` | `18` | Threshold against which `_stuck_plan_counters` is read by the operator-visible stuck-plan watchdog |
| `skip_cycle_on_allocation_error` | `SaturationAwareConfig` | `True` | When True, an absorbed Phase C `try_add_worker` exception logs the fragmentation snapshot, bumps the Counter, and skips the rest of Phase C; when False, the exception propagates so the run-loop dies loud |
| `enable_cross_stage_donor` | `SaturationAwareConfig` | `True` | Master toggle for the donor fallback Phase C consults before recording a partial add |
| `setup_aware_max_queued` | `SaturationAwareStageConfig` | `True` | Reserve queue credit for stages whose actors are still warming, so a Phase C add is not starved by the dispatcher's measurement loop |

## See also

- [00 — Per-cycle overview](00-overview.md) — where Phase C sits in
  the four-phase cycle and how Phase D rebalances the cluster
  between consecutive Phase C calls.
- [03 — Planning context](03-planning-context.md) — the
  `AutoscalePlanContext.try_add_worker` return contract that
  Phase C depends on.
- [13 — Cross-stage donor](13-cross-stage-donor.md) — the donor
  fallback Phase C invokes before recording a partial add.
- [16 — Hard caps and floors](16-hard-caps-and-floors.md) — the
  Phase B floor whose miss escalates to `RuntimeError` after
  `floor_stuck_grace_cycles`.
- [19 — Phase invariants](19-phase-invariants.md) — the
  monotonicity invariant on `_stuck_plan_counters`.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — operator
  visibility into stuck-plan counter timeseries.
- [26 — Stuck-plan detector](26-stuck-plan-detector.md) — the
  WARN-to-INFO latch and gauge / counter instrumentation that
  this doc's counter feeds into.
- [`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
  — `_run_phase_c_grow`, `_run_phase_b_floor`,
  `_attempt_cross_stage_donation`, `_record_donation_success`.
