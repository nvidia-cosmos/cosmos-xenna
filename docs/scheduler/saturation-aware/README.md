# Saturation-Aware Scheduler — Design and Operations FAQ

This folder explains the streaming-mode autoscaler that decides how
many workers each pipeline stage gets every cycle. Instead of a
top-down narrative, every section below is a question an engineer or
operator would actually ask, with an answer that names the chosen
design, the alternatives we rejected, and the file you read for the
full detail.

If you only have two minutes, skip to **["What is the high-level
architecture?"](#what-is-the-high-level-architecture)** and look at
[`feature-map.png`](feature-map.png).

---

## Why this scheduler exists

### What problem does this scheduler solve?

A Cosmos-Xenna pipeline is a streaming directed-acyclic graph of
stages with **heterogeneous resource shapes** (CPU-only, fractional
GPU, whole GPU, multi-node). Each stage owns a pool of workers. The
scheduler decides every few seconds how many workers each stage
should run.

Three facts drive the design.

1. Pipeline throughput is bounded by `1 / max_k D_k`, where
   `D_k = S_k / c_k` is the Forced-Flow-Law service demand for
   stage `k` (per-task service time `S_k` divided by stage capacity
   `c_k`). The *one slowest stage* caps the whole pipeline; over-
   provisioning any other stage cannot raise throughput.
2. Warm GPU state is expensive: a wrong scale-down costs a
   `worker_warmup_measurement_grace_s` window (60 s default) of
   throughput while the replacement actor reloads its model.
3. The cluster is finite. When it is fully booked, growing a
   newly-saturated stage requires **rebalancing** workers off a
   donor stage, not waiting for fresh placement.

The scheduler's job is to keep `max_k D_k` low while spending as
little warmup penalty as possible and never silently corrupting the
plan that reaches the cluster.

### Why is this problem hard?

Per-stage signals are noisy at the cycle scale (slot occupancy
swings, queue depth changes in bursts, service-time samples vary
per task), so a naive controller flaps. The bottleneck can move
between stages mid-run as load shifts. Rebalancing under capacity
pressure has its own failure modes — flicker, shape mismatch,
donor flip — that a "take a worker, give a worker" loop introduces.
And every wrong decision compounds: the replacement worker is cold
for at least one EWMA window before it produces a trustworthy
signal again.

The autoscaler must also be debuggable. Operators inspect the
metrics, the logs, and occasionally the source; a free-form
decision blob that emits "current best worker counts" without
naming **which** classifier zone, **which** phase, and **which**
constraint drove each delta is impossible to triage in production.

### What constraints shaped the design?

| Constraint | Implication |
|---|---|
| Streaming pipeline; no batch boundary | Scheduler runs on a fixed interval; no "end-of-job" recompute. |
| Warm GPU state is expensive | Asymmetric streaks; donor selection prefers the youngest worker. |
| Cluster placement is owned by the Rust planner | Python decides intent; Rust commits placement. The boundary is the `AutoscalePlanContext` (see [03](03-planning-context.md)). |
| Operator must trust and debug the autoscaler | Discrete labelled zones, structured per-decision INFO logs, stable Prometheus catalogue. |
| Per-stage signals are noisy | EWMA smoothing + asymmetric streak counters before any state change. |
| Heterogeneous resource shapes | Donor probing must check **actual** placement feasibility; shape mismatch is a real production regression. |
| Failure modes must fail loud, not drift | Per-phase invariants raise `SchedulerInvariantError` rather than emit a corrupt `Solution`. |

### What did we consider and reject?

- **Continuous PID / proportional controller on utilization.**
  Rejected because streaks need stable discrete labels — a
  continuous score reclassifies every cycle and oscillates around
  the decision boundary. See [05 — State classifier](05-state-classifier.md).
- **Single-signal autoscaling on slot occupancy alone.** Rejected
  because slots-empty-ratio cannot distinguish "stage is idle
  because input is drained" from "stage is idle because it is
  downstream of a stuck bottleneck". The pressure signal
  (utilisation × normalised backlog) is the AND-criterion fix; see
  [06 — Backlog-time signal](06-backlog-time-signal.md).
- **Fixed saturation thresholds.** Rejected because the correct
  cutoff depends on how many slots each actor has. A stage with
  one slot saturates at a different empty-ratio than a stage with
  32 slots. See [08 — Auto-derived thresholds](08-auto-derived-thresholds.md).
- **No cross-stage rebalancing.** Rejected because a downstream
  burst against a static upstream allocation deadlocks the
  pipeline under capacity pressure. The trade-off was the four
  anti-flap layers and the resource-fit search in
  [13](13-cross-stage-donor.md) + [29](29-cross-stage-donor-resource-fit.md).
- **Monolithic single-pass decision step.** Rejected because
  manual operator intent, floor enforcement, classifier growth,
  and shrink can collide; a free-form blob makes the precedence
  implicit and silently overwrites operator intent.

---

## How the solution works end-to-end

### What is the high-level architecture?

![Saturation-Aware Scheduler — Feature Map](feature-map.png)

The diagram above is a one-page summary of every feature in this
folder. There are five categories, every category anchored to a
position in the autoscale cycle.

- **Pre-flight (blue)** smooths signals, detects the cluster regime,
  and builds the planning context that Phases A-D mutate.
- **Classification (purple)** maps each stage's empty-slot ratio,
  queue depth, and pressure into one of four discrete zones
  (`NORMAL`, `SATURATED`, `SATURATED_CRITICAL`, `OVER_PROVISIONED`)
  with asymmetric hysteresis.
- **Growth & scale (green)** decides which stages grow, by how
  much, in what order, and which stage donates a worker when the
  cluster is full.
- **Safeguards (red)** wrap every phase — hard caps, floors, phase
  invariants, the loop watchdog, the memory-pressure gate, and the
  allocation-error tolerance layer.
- **Observability (teal)** emits a stable Prometheus catalogue, a
  per-stage bottleneck gauge that the scheduler **also consumes**
  to bias growth and protect the bottleneck from shrink, and
  per-decision INFO logs.

### How does one autoscale cycle work?

Every cycle the scheduler runs four phases — **A** (manual intent),
**B** (floor enforcement), **C** (saturation-driven grow), **D**
(saturation-driven shrink) — over a per-cycle `AutoscalePlanContext`,
with named invariants between each phase.

```
   autoscale(time, problem_state)
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Pre-flight  : refresh worker ages, detect regime,   │
   │               resolve thresholds, build context     │
   └─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Phase A     : apply manual operator intent first    │
   └─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Phase B     : enforce per-stage / per-node floors   │
   └─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Bottleneck  : compute D_k = S_k_ewma / c_k          │
   │               (updates S_k EWMA per stage)          │
   └─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Intent calc : classify each stage into one of 4     │
   │               zones; emit growth / shrink intents   │
   └─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Phase C     : grow stages with positive intent      │
   └─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────┐
   │ Phase D     : shrink stages with negative intent    │
   └─────────────────────────────────────────────────────┘
                │
                ▼
              Invariants ✓  →  freeze plan  →  Solution
```

The full diagram lives in [00 — Per-cycle overview](00-overview.md);
each phase's specific responsibilities are in
[04 — Per-cycle pipeline](04-per-cycle-pipeline.md).

### Where does each decision actually live?

| Phase / step | Owning module | Doc |
|---|---|---|
| Pre-flight smoothing + regime detection | [`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py) | [09](09-regime-aware-aggressiveness.md), [08](08-auto-derived-thresholds.md) |
| Phase A (manual) | `SaturationAwareScheduler._run_phase_a_delete` + `_run_phase_a_grow` | [04](04-per-cycle-pipeline.md) |
| Phase B (floor) | `SaturationAwareScheduler._run_phase_b_floor` | [16](16-hard-caps-and-floors.md) |
| Bottleneck score | [`bottleneck.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/bottleneck.py) | [23](23-bottleneck-score-metric.md), [25](25-bottleneck-decision-integration.md) |
| Intent classification | [`classifier.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/classifier.py) | [05](05-state-classifier.md), [06](06-backlog-time-signal.md), [07](07-streak-stabilization.md) |
| Capacity sizer (how many to add) | `SaturationAwareScheduler` in [`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py) | [28](28-capacity-sizer.md) |
| Phase C (grow) + donor | `SaturationAwareScheduler._run_phase_c_grow`, [`donor.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/donor.py) | [12](12-multi-target-dag-growth.md), [13](13-cross-stage-donor.md), [29](29-cross-stage-donor-resource-fit.md) |
| Phase D (shrink) | `SaturationAwareScheduler._run_phase_d_shrink` | [15](15-idle-first-scale-down.md), [25](25-bottleneck-decision-integration.md) |
| Phase invariants | [`invariants.py::check_invariants_after_phase`](../../../cosmos_xenna/pipelines/private/scheduling_py/invariants.py) | [19](19-phase-invariants.md) |

---

## Why each major component exists

### Why discrete zones instead of a continuous score?

Phase C (grow) and Phase D (shrink) do not consume a scalar — they
consume **a categorical zone plus a streak counter** ("this stage
has been `SATURATED` for the last 2 cycles, time to grow"). A streak
counter has no natural definition over a continuous signal: if the
smoothed empty-slot ratio is `0.32` this cycle and `0.34` next, has
it "been" `0.32` for one cycle or zero? Discrete labels remove the
ambiguity and let the streak gate every state change on stable
evidence.

The four zones also let asymmetric hysteresis apply only on the
exit edge, so a stage flips into `SATURATED_CRITICAL` on the first
burst cycle but exits only after the slot-busy signal has eased
past an inflated bound for the full streak. See
[05 — State classifier](05-state-classifier.md).

### Why a separate bottleneck signal on top of the per-stage classifier?

The classifier is **per-stage and myopic** — it knows only its own
slot ratio, queue depth, and pressure. It cannot distinguish "this
stage is the throughput bottleneck of the pipeline" from "this
stage happens to be busy". The most common autoscaler tuning
mistake is scaling the busy-looking stage; adding workers to a
non-bottleneck stage cannot raise total pipeline throughput because
the new workers just push more inventory onto the same downstream
queue.

The Forced-Flow-Law service demand `D_k = S_k / c_k` is the
**cross-stage** ranking that the per-stage classifier cannot
compute. The largest `D_k` is the bottleneck; the inverse,
`1 / max_k D_k`, is the pipeline throughput ceiling. The scheduler
emits the per-stage gauge `xenna_stage_bottleneck_score` for
operator observability ([23](23-bottleneck-score-metric.md)) and
also consumes the EWMA-smoothed version to bias Phase C grow
priority and protect the bottleneck stage from Phase D shrink
([25](25-bottleneck-decision-integration.md)).

### Why EWMA smoothing plus asymmetric streak counters?

A single cycle's slot-empty ratio can swing dramatically on noise
alone — one slow task in a small worker pool drops the ratio for a
cycle and then recovers. Three layers absorb this noise:

1. **EWMA smoothing** of the slot signal (`slots_empty_ratio_smoothing_level`)
   so the classifier reads a low-pass-filtered value.
2. **Asymmetric state streaks** so acting on `SATURATED` (which
   triggers scale-up) requires a short streak
   (`saturated_streak_min_cycles`, default `2`) while acting on
   `OVER_PROVISIONED` (which triggers scale-down) requires a long
   streak in that state (`over_provisioned_streak_min_cycles`,
   default `30`).
3. **Recommendation-history windows** so adjacent cycles must agree
   on direction before a delta commits.

The asymmetry is by design: a wrong scale-up is cheap (the extra
worker shrinks back in a few cycles), but a wrong scale-down kills
warm GPU state and costs a full warmup window. See
[07 — Streak stabilization](07-streak-stabilization.md).

### Why auto-derived thresholds (`K / sqrt(c)`)?

The "right" saturation cutoff is not a constant. A stage with one
slot per actor saturates at a different empty-slot ratio than a
stage with 32 slots per actor. The classifier auto-derives
`saturation_threshold = K / sqrt(slots_per_actor)` per stage from
the single primary knob `saturation_aggressiveness` (the
Halfin-Whitt `β`). This lets the operator tune one number for the
whole pipeline and have it Do The Right Thing per stage. See
[08 — Auto-derived thresholds](08-auto-derived-thresholds.md).

### Why cross-stage donation when the cluster is full?

A streaming autoscaler that can grow on fresh placement but never
rebalance has two failure modes seen in production:

- **Floor stuck**: a stage with `min_workers >= 1` has zero live
  workers because the cluster is fully booked by earlier-started
  stages. The classifier has no slot signal to act on; the
  pipeline cannot make forward progress.
- **Saturation stuck**: a downstream bottleneck wants `+N`
  workers, but an upstream stage that classified `SATURATED`
  earlier in the run is holding the cluster. Throughput is bound
  on the wrong stage.

A naive "take a worker, give a worker" loop introduces three new
failures: **flicker** (two stages rotate the same worker every
cycle), **shape mismatch** (freeing a CPU-only worker does not
unblock a whole-GPU receiver), and **donor flip** (the freed donor
becomes the new bottleneck). The design pays for those failures
with a four-layer eligibility funnel ([13](13-cross-stage-donor.md))
plus a bounded multi-donor resource-fit search and a
throughput-first economic gate
([29](29-cross-stage-donor-resource-fit.md)).

### Why phase invariants, a loop watchdog, and a memory-pressure gate?

Three failure surfaces that the per-phase decision code cannot
detect on its own:

- **Internal corruption**: a bug in Phase D (off-by-one in floor
  math, NaN ratio, negative count) can leave the planning context
  structurally invalid. Per-phase invariants raise
  `SchedulerInvariantError` rather than freeze the corrupt plan
  into a `Solution`. See [19](19-phase-invariants.md).
- **Slow cycles**: an autoscale cycle that takes 30 s on a 5 s
  interval silently doubles the response latency. The loop
  watchdog histograms cycle wall time and WARNs when it crosses a
  fraction of `interval_s`. See [18](18-loop-watchdog.md).
- **Cluster-wide OOM**: per-stage signals cannot see the sum of
  Ray object-store usage across stages. The memory-pressure gate
  freezes Phase C growth (keeping floors and shrink alive) when
  the cluster fraction crosses
  `memory_pressure_critical_threshold`. See [20](20-memory-pressure-gate.md).

---

## Operating the scheduler

### How do I tell if my pipeline is well-balanced?

The pipeline is balanced when every stage has roughly the same
service demand `D_k = S_k / c_k`. Three signals tell you where you
sit.

- **Per-stage gauge `xenna_stage_bottleneck_score`** — the
  `D_k` value per stage. The argmax stage is the throughput
  bottleneck; pipeline throughput is bounded by `1 / max_k D_k`.
- **Cluster gauge `xenna_scheduler_cluster_heterogeneity_ratio`**
  — `max_k D_k / min_k D_k` across stages with a finite `D_k`. A
  value near `1.0` is perfectly balanced; large values mean one or
  more stages are much slower than the rest and the pipeline is
  leaving capacity on the floor. The ratio is `NaN` until at least
  two stages have a finite `D_k` (i.e. each has completed at least
  one task since `setup()` and still has ready capacity).
- **Bottleneck INFO log line** — one structured line per cycle
  naming the engaged bottleneck (debounced via
  `bottleneck_engagement_persistence_cycles`). `grep
  "bottleneck stage" scheduler.log` is enough to identify which
  stage is bounding throughput right now without opening Grafana.

All three share the same per-cycle `D_k` mapping, so the dashboard
and the scheduler never disagree on which stage is the bottleneck.
See [23 — Bottleneck score metric](23-bottleneck-score-metric.md),
[25 — Bottleneck decision integration](25-bottleneck-decision-integration.md),
and [22 — Prometheus metrics](22-prometheus-metrics.md).

### What is the first knob to turn?

`saturation_aggressiveness` (per-stage, default `0.30`). This is
the Halfin-Whitt `β` in the auto-derived `K / sqrt(c)` formula.

- Increase toward `0.45` when scale-up consistently lags burst
  arrivals (queue depth grows before the autoscaler reacts).
- Decrease toward `0.20` when stages oscillate despite long streak
  counters (the threshold is firing on transients).

The full symptom-to-knob index, with workload-class example
configs, is in the [Operator Tuning Guide](tuning.md).

### How do I read the autoscaler logs?

Every decision emits a structured INFO line with phase, stage,
zone, intent delta, and the reason a guardrail engaged. See
[24 — Structured logging](24-structured-logging.md) for the field
catalogue and [26 — Stuck plan detector](26-stuck-plan-detector.md)
for the WARN-to-INFO promotion that signals a stage cannot place
new workers.

### When should I tune at all?

Only after the pipeline has run for at least
`over_provisioned_streak_min_cycles + stabilization_window_cycles_down`
cycles at steady state. Most "first-impression" symptoms are
cold-start artefacts that the slow-start mechanisms
([10](10-slow-start-mechanisms.md)) clear automatically. Single-
cycle anomalies should be absorbed by hysteresis, EWMA, and
streaks without intervention.

---

## Tradeoffs and complexity

### What tradeoffs did we accept?

| Cost | Benefit | Doc |
|---|---|---|
| O(stages) pure-Python invariant checks per phase boundary | Fail-loud refusal to emit a corrupt `Solution` | [19](19-phase-invariants.md) |
| Slow ramp on cold start | One-cycle ramp eliminated; no overshoot from missing EWMA signal | [10](10-slow-start-mechanisms.md) |
| Occasional Phase C growth pause | Cluster never OOMs the Ray object store | [20](20-memory-pressure-gate.md) |
| One extra per-stage state (`ACQUIRING`/`TRACKING`/`HOLD`) | Bounds post-shrink oscillation | [11](11-growth-mode-state-machine.md) |
| Bounded combinatorial donor search | Avoids flicker, shape mismatch, donor flip | [29](29-cross-stage-donor-resource-fit.md) |
| One extra Prometheus gauge per stage | Single ranked bottleneck signal; "scale the busy-looking stage" mistake prevented | [23](23-bottleneck-score-metric.md) |

### Which complexity is essential, and which could be simplified later?

**Essential** — removing any of these reintroduces a failure mode
documented above:

- Four-zone classifier with asymmetric hysteresis.
- EWMA + streak counters.
- Phase invariants.
- Bottleneck-aware Phase C / Phase D integration.
- Cross-stage donor primitive (donor flip and shape mismatch are
  real production regressions).

**Could be simplified later** — kept because the cost is small
and the question is open, not because we know the simpler form
fails:

- The bounded multi-donor resource-fit combinatorial search
  ([29](29-cross-stage-donor-resource-fit.md)) could collapse to
  a single-donor heuristic if all production pipelines stay in
  the linear-DAG regime where the throughput-first gate is
  rarely tight.
- The Halfin-Whitt regime detector ([09](09-regime-aware-aggressiveness.md))
  could be removed if cluster sizes prove small enough that the
  packed regime is always the operating regime.
- The stuck-plan WARN-to-INFO promotion ([26](26-stuck-plan-detector.md))
  could move into an alerting rule once external monitoring is
  authoritative.

### What are the known limitations and open questions?

- The Forced-Flow bottleneck score assumes a **linear streaming
  DAG** (`V_k = 1`). Branching DAGs would need per-edge visit
  counts, which neither the scheduler nor the planner currently
  track.
- Phase C grow-priority is **one-bottleneck-at-a-time** — the
  scheduler grows the argmax `D_k` stage first. Multi-bottleneck
  workloads (two stages with near-identical `D_k`) rely on the
  heterogeneity threshold to fall back to DAG-depth ordering.
- The cross-stage donor's resource-fit search is bounded by
  `cross_stage_donor_max_plan_size` and
  `cross_stage_donor_max_plan_combinations`; pipelines with many
  heterogeneous shapes may hit the bound and skip otherwise-valid
  donations.
- Phase D shrink protection can hold the bottleneck stage warm
  during prolonged idle (model reload, GPU stall). This is
  intentional — re-growing it after recovery costs a full
  `worker_warmup_measurement_grace_s` window — but it does mean
  the bottleneck stage occupies cluster capacity it is not
  actively using.

---

## Where do I learn more?

If you have never touched this scheduler before, read the
**Architecture and configuration** docs in order — [00](00-overview.md),
[01](01-scheduler-selection.md), [02](02-configuration-model.md),
[04](04-per-cycle-pipeline.md) — and then dive into whichever
decision below is relevant to the question you have. Each numbered
doc is independent after the overview tier.

### Architecture and configuration

| Doc | Topic |
|---|---|
| [01-scheduler-selection.md](01-scheduler-selection.md) | Feature flag + dispatcher (`SchedulerKind`) |
| [02-configuration-model.md](02-configuration-model.md) | `SaturationAwareConfig` + `SaturationAwareStageConfig` + 3-tier resolver |
| [03-planning-context.md](03-planning-context.md) | `AutoscalePlanContext` Rust <-> Python planner bridge |
| [04-per-cycle-pipeline.md](04-per-cycle-pipeline.md) | Phase A / B / C / D orchestration inside `autoscale()` |

### Classification

| Doc | Topic |
|---|---|
| [05-state-classifier.md](05-state-classifier.md) | Four-zone classifier with hysteresis |
| [06-backlog-time-signal.md](06-backlog-time-signal.md) | Compound AND-criterion (utilisation + queue-time) |
| [07-streak-stabilization.md](07-streak-stabilization.md) | EWMA smoothing + asymmetric streak counters |
| [08-auto-derived-thresholds.md](08-auto-derived-thresholds.md) | `K / sqrt(c)` aggressiveness formula |
| [09-regime-aware-aggressiveness.md](09-regime-aware-aggressiveness.md) | Halfin-Whitt regime detector + lift |

### Growth and scale

| Doc | Topic |
|---|---|
| [10-slow-start-mechanisms.md](10-slow-start-mechanisms.md) | Three layers that suppress cold-start overshoot |
| [11-growth-mode-state-machine.md](11-growth-mode-state-machine.md) | `ACQUIRING` -> `TRACKING` -> `HOLD` |
| [12-multi-target-dag-growth.md](12-multi-target-dag-growth.md) | Why all saturated stages grow per cycle |
| [13-cross-stage-donor.md](13-cross-stage-donor.md) | Four-layer anti-flap donor eligibility funnel |
| [14-worker-age-tracking.md](14-worker-age-tracking.md) | Youngest-first donor selection substrate |
| [15-idle-first-scale-down.md](15-idle-first-scale-down.md) | Phase D consolidation-aware victim ordering |
| [28-capacity-sizer.md](28-capacity-sizer.md) | Closed-form target worker count (`Little's Law` + `Forced Flow Law`) |
| [29-cross-stage-donor-resource-fit.md](29-cross-stage-donor-resource-fit.md) | Bounded multi-donor resource-fit search + throughput-first commit gate |

### Safeguards

| Doc | Topic |
|---|---|
| [16-hard-caps-and-floors.md](16-hard-caps-and-floors.md) | Per-stage / per-node `min_workers`, `max_workers` |
| [17-config-validation.md](17-config-validation.md) | Field + cross-field validators |
| [18-loop-watchdog.md](18-loop-watchdog.md) | Cycle-time monitoring |
| [19-phase-invariants.md](19-phase-invariants.md) | `SchedulerInvariantError` between phases |
| [20-memory-pressure-gate.md](20-memory-pressure-gate.md) | Cluster-wide OOM defence |
| [21-allocation-error-tolerance.md](21-allocation-error-tolerance.md) | Transient-failure recovery |

### Observability

| Doc | Topic |
|---|---|
| [22-prometheus-metrics.md](22-prometheus-metrics.md) | The full metrics catalogue |
| [23-bottleneck-score-metric.md](23-bottleneck-score-metric.md) | Forced-Flow-Law bottleneck gauge |
| [24-structured-logging.md](24-structured-logging.md) | Per-decision INFO logging contract |
| [25-bottleneck-decision-integration.md](25-bottleneck-decision-integration.md) | `D_k = S_k_ewma / c_k` driving Phase C priority and Phase D protection |
| [26-stuck-plan-detector.md](26-stuck-plan-detector.md) | Per-stage WARN-to-INFO latch + Prometheus instrumentation |
| [27-topology-aware-classifier.md](27-topology-aware-classifier.md) | Topology-aware classifier |

### Operator quick reference

- [Operator Tuning Guide](tuning.md) — primary knobs, symptom-to-knob
  index, workload-class example configs.

### How are these docs written?

- **One question per section, one answer per question.** Each doc
  is meant to be readable in roughly two minutes; the source-file
  docstrings carry the formal contracts.
- **Diagrams use Unicode box-drawing characters**
  (`┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ → ⇒ ● ○`). They are the primary teaching
  aid; prose around a diagram is captions.
- **Source links** point at files under
  [`cosmos-xenna/cosmos_xenna/pipelines/private/scheduling_py/`](../../../cosmos_xenna/pipelines/private/scheduling_py/).
  Open the source for exact types and defaults.
- **Configuration field names** are written verbatim in
  backticks and resolve to fields on `SaturationAwareConfig` or
  `SaturationAwareStageConfig` in
  [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py).
  When this folder and `specs.py` disagree, `specs.py` wins.
- **External concepts** (Halfin-Whitt regime, Forced Flow Law,
  Little's Law, TCP slow-start, K8s topology spread, Cloud
  Dataflow Streaming Engine backlog, HPA tolerance) are cited by
  name; follow the public references in each doc for the academic
  primary source.
