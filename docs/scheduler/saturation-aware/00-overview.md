# 00 — Per-Cycle Overview FAQ

This doc explains the *structure of one autoscale cycle*. The
big-picture rationale for the scheduler as a whole — what problem
it solves, why discrete zones, why a separate bottleneck signal —
lives in [README.md](README.md).

If you need to find a single feature on a diagram, open the
[feature map](README.md#what-is-the-high-level-architecture).
If you need to triage a slow cycle or a `SchedulerInvariantError`,
[18 — Loop watchdog](18-loop-watchdog.md) and
[19 — Phase invariants](19-phase-invariants.md) are the answer.

---

## What does one autoscale cycle do?

Every cycle the scheduler runs a **fixed four-phase pipeline**
over a per-cycle `AutoscalePlanContext`, with named invariants
between every phase, and freezes the staged plan into a `Solution`
at the end.

The four phases are **A** (manual operator intent), **B** (floor
enforcement), **C** (saturation-driven grow), and **D**
(saturation-driven shrink). Pre-flight runs before A; bottleneck
calculation and intent classification run between B and C; the
plan is frozen and emitted after D crosses its invariants.

```
                       autoscale(time, problem_state)
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Pre-flight                                  │
              │  ─ shape check (problem ↔ problem_state)     │
              │  ─ cycle counter += 1                        │
              │  ─ refresh per-worker READY timestamps       │
              │  ─ snapshot stuck-plan counters              │
              │  ─ regime detector + lift effective K        │
              │  ─ resolve auto-thresholds (lazy, per stage) │
              │  ─ build AutoscalePlanContext (FGD bridge)   │
              │  ─ build donor-warmup-excluded set           │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase A — Manual operator intent            │
              │  ─ delete excess workers (manual stages)     │
              │  ─ add up to requested_num_workers           │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓
              ┌──────────────────────────────────────────────┐
              │  Phase B — Floor enforcement                 │
              │  ─ ensure current_workers ≥ floor for every  │
              │    non-finished stage                        │
              │  ─ on cluster-full failure: floor-mode       │
              │    cross-stage donor (youngest-first)        │
              │  ─ raises RuntimeError after grace exhausted │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓
              ┌──────────────────────────────────────────────┐
              │  Bottleneck calculation                      │
              │  ─ consume per-cycle service-time samples    │
              │  ─ update S_k EWMA per stage                 │
              │  ─ compute D_k = S_k / c_k from live cap.    │
              │  ─ identify_bottleneck (heterogeneity ratio: │
              │    max/median for n>=3, max/min for n=2)     │
              │  ─ debounced engagement INFO log             │
              │  ─ feeds Phase C grow priority + Phase D     │
              │    shrink protection (see doc 25)            │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Compute per-stage intent deltas             │
              │  ─ sample per-cycle throughput               │
              │  ─ aggregate slot signals (warmup-excluded)  │
              │  ─ EWMA smoothing                            │
              │  ─ classify into 4 zones (with hysteresis)   │
              │  ─ asymmetric streak counters                │
              │  ─ trust gate (min_data_points)              │
              │  ─ stabilization-window consensus            │
              │  ─ growth-mode state machine                 │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Phase C — Saturation-driven grow            │
              │  ─ grow-priority loop over positive intents  │
              │    (D_k descending when bottleneck engaged;  │
              │    DAG depth descending otherwise)           │
              │  ─ saturation-mode cross-stage donor when    │
              │    cluster is full but a downstream stage    │
              │    needs to grow                             │
              │  ─ stuck-plan counters tick                  │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓ + NaN check
              ┌──────────────────────────────────────────────┐
              │  Phase D — Saturation-driven shrink          │
              │  ─ apply negative intents                    │
              │  ─ skip bottleneck stage when engaged and    │
              │    no ceiling overflow (see doc 25)          │
              │  ─ idle-first + GPU-consolidation ordering   │
              │  ─ skip workers inside donor-warmup grace    │
              │  ─ never drop below per-stage / per-node     │
              │    floor                                     │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼  invariants  ✓ + floor check + monotonicity
              ┌──────────────────────────────────────────────┐
              │  Freeze plan → Solution                      │
              │  ─ ctx.into_solution()                       │
              │  ─ persist worker ages (cycle-counted)       │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
                                 Solution
```

The full flow is implemented in
[`SaturationAwareScheduler.autoscale`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py).
Each phase is a private method on the scheduler; helper modules
under
[`scheduling_py/`](../../../cosmos_xenna/pipelines/private/scheduling_py/)
own the pure-function decision primitives (classifier, EWMA,
streak, growth mode, donor selection, scale-down ordering).

---

## Why a fixed four-phase pipeline instead of a free-form decision step?

A streaming autoscaler that simply emits "current best worker
counts" each cycle has three failure modes that show up in
production.

- **Manual versus classifier collision.** Per-stage
  `requested_num_workers` lets an operator pin a stage. In a
  free-form decision step, whichever code path runs last wins; an
  operator setting `requested_num_workers = 4` while the
  classifier wants `+2` produces non-deterministic outcomes
  depending on internal call order.
- **Floor versus cluster-capacity collision.** Floor enforcement
  (`min_workers`, `min_workers_per_node`) collides with cluster
  capacity. A stage gets stuck at zero workers because the
  planner has nothing to give it; the classifier has no slot
  signal to react to.
- **Silent corruption from a single bug.** A bug in any one
  decision step — NaN ratio, negative count, off-by-one in floor
  math — emits a corrupted plan to the cluster. The pipeline
  drifts before anyone notices.

The fix is to treat one autoscale call as a closed-contract
four-phase pipeline with **per-phase invariants** between each
phase. Bugs surface as `SchedulerInvariantError` exceptions
*before* the plan reaches the planner, not as silent drift in
production.

## Why this exact phase order (A -> B -> C -> D)?

Each phase's position is structural, not accidental.

- **Phase A runs first** because manual deletions free placement
  slots that Phase B / Phase C can reuse the same cycle. If
  manual ran after the classifier, the freed slot would not be
  visible until the next cycle and operator changes would
  converge in two cycles instead of one.
- **Phase B runs before the classifier** because zero-worker
  stages have no slot signal to classify against. Trying to
  reason about a stage that has no workers is meaningless; the
  floor enforcement must run first to guarantee every
  non-finished stage has at least one worker the classifier can
  observe.
- **Phase C and Phase D are separated** so a single cycle never
  both grows and shrinks the same stage. Splitting positive and
  negative intent into separate phases eliminates a whole class
  of within-cycle oscillation.
- **Bottleneck calculation sits between Phase B and the intent
  loop** so the per-stage decision pipeline observes a populated
  `cycle_bottleneck_context` and Phase C / Phase D see fresh
  `_d_k_now` / `_s_k_ewma` values for the current cycle. See
  [25 — Bottleneck decision integration](25-bottleneck-decision-integration.md).

## What happens in pre-flight?

Pre-flight runs **before** Phase A and prepares the per-cycle
state that every phase will read.

- **Cycle counter** increments; downstream code references the
  current cycle number for stuck-plan and worker-age bookkeeping.
- **Per-worker READY timestamps** refresh from the live problem
  state so warmup-grace checks use up-to-date ages.
- **Stuck-plan counter snapshot** captures the prior cycle's
  values for delta-based promotion ([26](26-stuck-plan-detector.md)).
- **Regime detection** runs the Halfin-Whitt cluster signal and
  lifts the effective aggressiveness `K` when packed
  ([09](09-regime-aware-aggressiveness.md)).
- **Auto-threshold resolution** lazily computes per-stage
  saturation and activation thresholds from `K / sqrt(c)`
  ([08](08-auto-derived-thresholds.md)).
- **`AutoscalePlanContext`** is built — the mutable bridge that
  Phases A-D mutate and the Rust planner reads
  ([03](03-planning-context.md)).
- **Donor-warmup-excluded set** is computed once so Phase B / C
  donor selection skips workers younger than
  `donor_warmup_grace_s` ([10](10-slow-start-mechanisms.md)).

## What does each phase actually do?

- **Phase A — manual.** Delete excess workers from manual stages
  first (returning slots to the cluster), then add workers up to
  `requested_num_workers` for the same stages. The two-step
  ordering is what lets a manual deletion converge in one cycle.
- **Phase B — floor.** For every non-finished stage, ensure
  `current_workers >= max(min_workers, min_workers_per_node)`.
  On cluster-full failure, invoke the floor-mode cross-stage
  donor (youngest-first); after `floor_stuck_grace_cycles`, raise
  `RuntimeError` so the operator sees the deadlock instead of a
  silently degraded pipeline. See [13](13-cross-stage-donor.md)
  and [16](16-hard-caps-and-floors.md).
- **Bottleneck calculation.** Consume the cycle's service-time
  samples into `S_k` EWMA, divide by the live capacity `c_k` to
  produce `D_k`, identify the bottleneck stage via
  `identify_bottleneck`, emit the per-stage gauge, and emit the
  debounced engagement INFO log ([23](23-bottleneck-score-metric.md)).
- **Intent calculation.** Sample per-cycle throughput, aggregate
  the slot signals (excluding warmup-grace workers), EWMA-smooth
  them, classify each stage into one of four zones, run the
  asymmetric streak counter, gate on trust (`min_data_points`),
  check the stabilization window consensus, and feed the
  growth-mode state machine ([05](05-state-classifier.md),
  [07](07-streak-stabilization.md), [11](11-growth-mode-state-machine.md)).
- **Phase C — grow.** Walk the positive-intent stages in
  bottleneck-`D_k`-descending order when the gate is engaged,
  otherwise DAG-depth-descending ([12](12-multi-target-dag-growth.md)).
  For each stage, call `try_add_worker`; on cluster-full, invoke
  the saturation-mode cross-stage donor with the four anti-flap
  layers and the bounded resource-fit search
  ([13](13-cross-stage-donor.md), [29](29-cross-stage-donor-resource-fit.md)).
  Tick the stuck-plan counter for stages that could not grow.
- **Phase D — shrink.** Walk the negative-intent stages, **skip
  the bottleneck stage** when the gate is engaged and there is
  no ceiling overflow ([25](25-bottleneck-decision-integration.md)),
  pick victims using idle-first + GPU-consolidation ordering
  ([15](15-idle-first-scale-down.md)), skip workers inside the
  donor-warmup grace, and never drop below the floor.
- **Freeze.** Call `ctx.into_solution()` to produce the
  immutable `Solution`, persist the cycle-counted worker ages,
  and return.

## Why per-phase invariants?

Invariants run at every phase boundary. A structural violation
(plan references a stage that does not exist, count is negative,
floor is below the per-stage minimum, monotonicity is broken
between Phase C and Phase D) raises `SchedulerInvariantError` and
refuses to emit a `Solution`. The trade-off is a few microseconds
of pure-Python per phase against fail-loud behaviour on internal
corruption; in production, the invariants surface bugs before they
mutate the cluster. See [19 — Phase invariants](19-phase-invariants.md).

## Why a dedicated bottleneck-calculation step before intent and Phase C?

Phase C's grow priority is the gate that decides which stage gets
a scarce placement slot when several stages want one. The
classifier alone cannot rank stages by **pipeline-wide**
throughput contribution; it only knows each stage's slot ratio.
The Forced-Flow-Law `D_k = S_k / c_k` is the cross-stage ranking
that fills the gap: argmax `D_k` is the throughput bottleneck;
adding a worker there raises `1 / max_k D_k`, adding one anywhere
else does not.

Running the bottleneck step **before** the intent loop — not
inside the classifier — means the per-stage decision pipeline
already sees a populated `cycle_bottleneck_context` while it
runs, and the same `D_k` mapping flows into Phase C grow
priority and Phase D shrink protection in one cycle, so the
planner and the operator dashboard never disagree on which stage
is the bottleneck. See
[25 — Bottleneck decision integration](25-bottleneck-decision-integration.md).

## What knobs affect cycle behaviour, and where do they live?

The cycle **structure** itself is not configurable. The four-phase
order, the invariants between phases, and the pre-flight steps
are fixed. Knobs that affect *how each phase behaves* live on
`SaturationAwareConfig` (cluster-level) and
`SaturationAwareStageConfig` (per-stage) in
[`specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).

| Phase | Key knobs (linked to feature docs) |
|---|---|
| Pre-flight | [auto-derived thresholds](08-auto-derived-thresholds.md), [regime-aware lift](09-regime-aware-aggressiveness.md) |
| Phase A | none (operator intent only) |
| Phase B | [hard caps and floors](16-hard-caps-and-floors.md), [cross-stage donor](13-cross-stage-donor.md) |
| Intent compute | [state classifier](05-state-classifier.md), [backlog-time](06-backlog-time-signal.md), [streak stabilization](07-streak-stabilization.md), [slow-start](10-slow-start-mechanisms.md), [growth mode](11-growth-mode-state-machine.md) |
| Bottleneck calc | [bottleneck score metric](23-bottleneck-score-metric.md), [bottleneck decision integration](25-bottleneck-decision-integration.md) |
| Phase C | [multi-target DAG growth](12-multi-target-dag-growth.md), [cross-stage donor](13-cross-stage-donor.md), [worker age tracking](14-worker-age-tracking.md), [bottleneck decision integration](25-bottleneck-decision-integration.md) |
| Phase D | [idle-first scale-down](15-idle-first-scale-down.md), [bottleneck decision integration](25-bottleneck-decision-integration.md) |
| Invariants | [phase invariants](19-phase-invariants.md) |
| Cycle | [loop watchdog](18-loop-watchdog.md), [memory-pressure gate](20-memory-pressure-gate.md), [allocation-error tolerance](21-allocation-error-tolerance.md) |

The symptom-to-knob index for operators is the
[Operator Tuning Guide](tuning.md).

## What can go wrong in a cycle, and how would I see it?

Four observable failure modes, all surfaced by the safeguard
layer:

- **Slow cycle.** The loop watchdog histograms cycle wall time
  and WARNs when it crosses a fraction of `interval_s`. The
  warning identifies which phase dominated.
  See [18 — Loop watchdog](18-loop-watchdog.md).
- **Internal corruption.** A `SchedulerInvariantError` is raised
  *before* the `Solution` is frozen; the exception names the
  invariant that failed and the phase that violated it.
  See [19 — Phase invariants](19-phase-invariants.md).
- **Stage cannot place new workers.** The stuck-plan counter
  promotes to a one-shot INFO log per stage with active /
  cycles_total Prometheus gauges. Useful when Phase C grow
  cannot find a placement.
  See [26 — Stuck plan detector](26-stuck-plan-detector.md).
- **Cluster memory exhaustion.** The memory-pressure gate freezes
  Phase C growth when Ray object-store usage crosses
  `memory_pressure_critical_threshold`. Floor enforcement and
  Phase D continue.
  See [20 — Memory-pressure gate](20-memory-pressure-gate.md).

## Where do I read next?

- [01 — Scheduler selection](01-scheduler-selection.md) — how a
  pipeline opts into the saturation-aware scheduler.
- [02 — Configuration model](02-configuration-model.md) — the
  config classes and three-tier resolver.
- [03 — Planning context](03-planning-context.md) — the Rust
  planner bridge that Phases A-D mutate.
- [04 — Per-cycle pipeline](04-per-cycle-pipeline.md) — drill-down
  on what each phase actually does in code.
- [19 — Phase invariants](19-phase-invariants.md) — what each
  invariant check enforces.
- [README.md](README.md) — the design and operations FAQ for the
  scheduler as a whole.
