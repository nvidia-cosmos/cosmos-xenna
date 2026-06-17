# Saturation-Aware Scheduler

The entry point for an engineer new to the `SATURATION_AWARE` streaming
autoscaler. Read this top-to-bottom in ~10 minutes and you will know what it
does, why it is shaped this way, and where to look for more detail. The five numbered
**concept notes** go one layer deeper into each topic; **`tuning.md`** is for
operators.

---

## 1. Two-minute pitch

A Cosmos-Xenna pipeline is a streaming DAG of stages. Each stage owns a pool of
workers; the cluster has a finite resource budget. The scheduler runs on a fixed
interval (~10 s) and emits, for every stage, a target worker count.

`SATURATION_AWARE` is a thin **pure-Python layer that wraps the Rust
fragmentation solver** (`FRAGMENTATION_BASED`, the default). It leaves the
solver's code untouched and instead does two things around it each cycle:

- **rewrites the solver's input**: it hands the solver a *deflated* per-worker
  speed for the stage it wants to grow, so the solver allocates more workers
  there (this is the growth driver);
- **bounds the solver's output into a per-stage `[floor, cap]` band**: an upper
  **cap** (the capacity target `w_target`, or +1 worker/cycle while a stage is
  still cold) that trims over-eager additions, and a lower **floor**
  (`w_sustain`, down to `min_workers` on release) that vetoes over-eager deletes.

SAT is the component that **decides how big each stage should be**: its capacity
model computes a per-stage target band `[w_sustain, w_target]` from the
slowest-stage rate (see [01](01-capacity-model.md)). The two steps above are how
it *enforces* that decision rather than how it makes it. What SAT does **not**
do is write the counts itself. It never sets a worker count, adds a worker, or
forces a delete directly; it realizes its sizing through the solver (growth
steered by the input bias and trimmed to the cap, shrink produced by the solver
and vetoed below the floor). So **SAT owns the sizing and the solver owns
placement**. The net job: keep each stage sized to the rate the pipeline's
**slowest stage** can sustain, and keep expensive warm stages (e.g. a GPU
stage with a costly model to load) warm across short upstream lulls.

`FRAGMENTATION_BASED` stays the production default. `SATURATION_AWARE` is
opt-in per run; the two are fully isolated and never affect each other.

### Two signals: queues decide *where*, speed decides *how much*

SAT reads two kinds of signal, and they do different jobs - only one of them is
what makes it "saturation-aware":

- **Queues choose the target.** Which stage is the constraint, whether a stage
  is even eligible to grow, and when an expensive stage may shrink are all read
  from inter-stage **queue occupancy** ([02](02-bottleneck-selection.md),
  [05](05-scale-down-floor.md)). A stage that looks slow only because it is
  starved of input is a *symptom*, never the bottleneck. This queue-gradient
  view is SAT's distinctive contribution.
- **Speed sizes the target.** Naming the stage is not enough; SAT still needs a
  number - *how many* workers hold the pipeline rate. That takes a per-worker
  **throughput** estimate, which the capacity model turns into
  `cap_src = workers × speed / chain` and the targets `w_sustain` / `w_target`
  ([01](01-capacity-model.md)). Speed is also the only language the wrapped
  solver speaks: SAT steers `FRAG` by handing it a *deflated* speed, never by
  editing counts ([03](03-growth-and-sizing.md)).

The invariant that keeps them from fighting: **speed never selects a bottleneck, it only sizes the one the queues already picked.** Letting a low measured speed pick the constraint is the speed-argmin trap ([02](02-bottleneck-selection.md)) that queue-gradient selection exists to remove. The single exception is a deliberate fallback - when no queue cliff is visible at all (cold start, or a fully balanced pipeline where every buffer is populated), SAT sizes from the slowest smoothed `cap_src` so the rate stays stable until a cliff reappears.

![Split diagram. Left half labeled queues to where: a four-stage pipeline S0 to S3 with a full input buffer feeding S2 and a near-empty buffer feeding S3, marking S2 as the bottleneck at the queue cliff. Right half labeled speed to how much: the selected stage S2 with a speed gauge, the formula cap_src equals workers times speed over chain, and worker count growing from two to three. A bottom banner reads speed never selects, it only sizes.](assets/readme-queues-vs-speed.png)

*Queues pick the stage at the cliff (S2, whose full input feeds an under-fed S3); speed then sizes only that stage via `cap_src` and the worker targets. The banner is the invariant: a low speed at a populated-input stage never makes it the bottleneck.*

---

## 2. The problem this scheduler solves

The default fragmentation solver re-optimizes placement every tick and packs
workers onto GPUs very well. Two failure modes hurt streaming pipelines with an
expensive downstream stage:

1. **It scales down a transiently-starved stage.** When an upstream stage
   briefly falls behind, a downstream GPU stage sees an empty input queue and
   near-zero throughput. A throughput-only view reads that as "over-provisioned"
   and shrinks it, but re-acquiring that worker later costs a full model reload
   (minutes of GPU time), a net loss for a few-second lull.
2. **Cold-start fragmentation.** On the first cycle no stage has a measured
   speed, so every stage reports the solver's placeholder speed. Sized from that
   guess, cheap fractional-GPU stages can scatter quarter-GPU workers across
   every GPU and block a whole-GPU stage from ever placing.

![Two worker-count timelines over the same input lull. The default solver drops to zero during the dip and then pays a long model-reload cost when work resumes; the saturation-aware layer holds the count steady through the dip and resumes instantly.](assets/readme-warm-vs-shrink.png)

*Failure mode 1. A throughput-only view reads a transient lull as
over-provisioning and shrinks the stage, then pays a full model reload when work
returns. SAT holds the stage warm across the dip.*

The fix, in one sentence: **rewrite the solver's input to drive useful growth,
then bound its output (an upper cap and a lower floor) to protect
must-not-shrink stages**, all without changing the solver itself.

---

## 3. The shape of one cycle

Everything flows from **one capacity model** that is the single source of truth
for both growth and shrink (see [01](01-capacity-model.md)). The cycle is a flat
sequence that wraps the unchanged solver in an input bias and two output bounds
(an upper cap and a lower floor):

![SAT decision cycle from signals through capacity, demand, FRAG solve, ramp, floor, to commit. SAT applies an input bias before the solver and output bounds after it.](assets/readme-cycle-flow.png)

*One cycle, left to right: SAT rewrites the solver's **input** (the bias on
`demand`) and bounds its **output** (the `ramp` cap and the `floor`). The shaded
`FRAG solve` step is the unchanged Rust solver SAT wraps.*

The solver still owns **placement** and any downward **degradation** under real
resource limits. SAT only rewrites *how much* it is asked to grow (input bias)
and bounds *how far* the result may grow or shrink (the `[floor, cap]` band).

---

## 4. Main concepts

Five conceptual areas, one note each. Each is a self-contained deep dive, a
page or two.

| # | Concept | One-liner |
|---|---|---|
| **01** | [Capacity model](01-capacity-model.md) | The single source of truth: `cap_src`, the pipeline rates, `w_sustain` / `w_target`, and the smoothed speed signal they rest on. |
| **02** | [Bottleneck selection](02-bottleneck-selection.md) | Pick the bottleneck from the **queue gradient**, not from speed, and why a starved downstream stage is expected to idle. |
| **03** | [Growth and sizing](03-growth-and-sizing.md) | Grow only the bottleneck by handing the solver a deflated speed; bound everyone else to the bottleneck rate. |
| **04** | [Cold-start ramp](04-cold-start-ramp.md) | Cap an unmeasured stage to +1 worker/cycle so a placeholder speed can never trigger a fragmentation burst. |
| **05** | [Scale-down floor](05-scale-down-floor.md) | A shrink-veto that holds expensive stages warm through transient lulls; releases on a confirmed drain, or when an over-provisioned idle downstream stage can return resources the bottleneck needs. |

---

## 5. Expected behavior (the steady-state contract)

A balanced pipeline does **not** run every stage at 100%. The target is:

- every inter-stage buffer stays populated (bounded, at/above its dispatch
  threshold);
- the **bottleneck** stage runs near 100%;
- **non-bottleneck stages run below 100% by definition**. That spare capacity is
  exactly what makes them not the bottleneck.

A stage **downstream of the bottleneck** can only go as fast as the bottleneck
feeds it, so it will sit with an empty input queue and low utilization. **This is
designed behavior, not a defect**. See
[02: Bottleneck selection](02-bottleneck-selection.md#why-downstream-stages-idle-by-design).

---

## 6. Non-goals

The saturation-aware layer deliberately does **not** try to:

- **Replace placement.** The Rust solver owns where workers go; SAT never
  re-implements bin-packing or GPU placement.
- **Defragment GPUs with a second pass.** The solver already picks deletion
  victims to minimize fragmentation; a Python re-chooser would be cruder and
  could desync from the solver's placements.
- **Balance branching DAGs.** The rate math assumes a linear streaming pipeline
  (one path, fan-out via `chain`). Branching topologies are out of scope.
- **Fix feeder service capacity.** SAT sizes stages to the measured bottleneck
  rate; it cannot make a stage faster than its code allows. A bottleneck that
  stays saturated at high worker counts with low per-worker speed is limited by
  that stage's throughput, not the scheduler.
- **Expose a knob per failure mode.** Hold-vs-release is governed by the capacity
  model plus a single release default, not a thicket of per-stage flags.

---

## 7. Where each decision lives (code map)

All paths are relative to
`cosmos_xenna/pipelines/private/scheduling_py/saturation_aware/`.

| Concept | Owning module | Note |
|---|---|---|
| Per-cycle orchestration facade | `scheduler.py::SaturationAwareScheduler` | - |
| Capacity model (rates, `w_sustain`, `w_target`) | `capacity.py` | 01 |
| Speed estimator (per-worker rate, in-flight aging) | `estimator.py` | 01 |
| Queue-gradient bottleneck selection | `capacity.py::classify_stages`, `_select_bottleneck_by_queue` | 02 |
| Demand multiplier (input bias) | `sizing.py` | 03 |
| Cold-start ramp | `ramp.py` | 04 |
| Scale-down floor + chain math | `floor.py`, `chain.py` | 05 |
| Buffered write-back to the solver result | `solution_editor.py` | - |
| Best-effort pinned worker counts | `problem_template.py` | - |
| Configuration | `config.py::SaturationAwareConfig` | `tuning.md` |
| Selection / opt-in | `SchedulerKind.SATURATION_AWARE` in `specs.py` | `tuning.md` |

---

## 8. Further reading

- The five concept notes above, in order.
- **`tuning.md`**: when to choose `SATURATION_AWARE`, how to select it, and the
  handful of knobs worth turning.
- The deep design spec (curator repo) at
  `docs/curator/design/saturation-aware-scheduler.md` gives exhaustive
  field-level rationale; this note set is the fast on-ramp.
- The Rust solver SAT wraps:
  `cosmos_xenna/pipelines/private/autoscaling_algorithms.py`
  (`run_fragmentation_autoscaler`).
