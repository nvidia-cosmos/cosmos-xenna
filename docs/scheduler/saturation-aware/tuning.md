# Operator Tuning Guide

This is the operator's quick-reference for tuning the saturation-aware
scheduler in production. The six numbered concept notes
(`01-signals-and-classification.md` through `06-safeguards.md`) plus
[`README.md`](README.md) explain **why** each knob exists; this guide
explains **which knob to turn when**.

> **Source of truth**: every default and range below is mirrored in
> [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py)
> on `SaturationAwareConfig` (cluster-level) or
> `SaturationAwareStageConfig` (per-stage). When this doc and that
> file disagree, `specs.py` wins.

## When to tune

Tune scheduler knobs when:

- A specific operational symptom (oscillation, slow ramp, mid-task
  shrink, stuck plan, regime flap) is reproducible and you have
  identified the responsible knob via the symptom index below.
- A stage has a materially different signal profile than the
  cluster default — multi-minute model warmup, latency-critical
  serving, or extreme task-size variance — and the per-stage
  override path documented in
  [README — Where each decision lives](README.md#9-where-each-decision-lives-code-map) gives you
  scoped control.

**Do not tune** when:

- The pipeline has not yet run for at least
  `over_provisioned_streak_min_cycles + stabilization_window_cycles_down`
  steady-state cycles — most "first-impression" symptoms are
  cold-start artefacts that the slow-start mechanisms
  ([decisions](02-decisions-and-growth.md)) clear automatically.
- The symptom is a single-cycle anomaly. The hysteresis,
  EWMA smoothing, and streak counters are designed to absorb
  one-off noise without operator intervention.

## Primary knobs (the ~80% of tuning that matters)

### `saturation_aggressiveness`

| Field | Default | Range | When to adjust |
|---|---|---|---|
| `SaturationAwareStageConfig.saturation_aggressiveness` | `0.25` | `[0.10, 0.60]` | Whole-pipeline knee tuning. |

The single primary knob. It is the Halfin-Whitt `β` in the
`K / √c` formula that auto-derives `saturation_threshold` per
stage. Higher values fire `SATURATED` sooner. Lower values are
more conservative.

- Increase toward `0.45` when scale-up consistently lags burst
  arrivals (operators see queue depth grow before the autoscaler
  reacts).
- Decrease toward `0.20` when stages oscillate between adding and
  removing workers despite long streak counters — the threshold
  is firing on transients.

See [01 — Signals and classification](01-signals-and-classification.md).

### Streak counters (asymmetric stabilization)

| Field | Default | Range | When to adjust |
|---|---|---|---|
| `saturated_streak_min_cycles` | `2` | `[1, 8]` typical | Burst response. Higher → slower ramp; lower → noisier ramp. |
| `over_provisioned_streak_min_cycles` | `10` | `[10, 60]` typical | Scale-down patience. Higher → less churn; lower → reclaim (and donate) resources sooner. |
| `saturated_critical_streak_min_cycles` | `1` | `[1, 4]` | How fast `SATURATED_CRITICAL` fires. Lower means burst response in one cycle. |

Cross-field invariant: `over_provisioned_streak_min_cycles >
saturated_streak_min_cycles` (validated; see
[README — Where each decision lives](README.md#9-where-each-decision-lives-code-map)). The
asymmetry is by design: a wrong scale-up is cheap; a wrong
scale-down kills warm GPU state.

See [02 — Decisions and growth](02-decisions-and-growth.md).

### Stabilization windows (recommendation consensus)

| Field | Default | Range | When to adjust |
|---|---|---|---|
| `stabilization_window_cycles_up` | `1` | `[1, 4]` typical | Grow responsiveness. Higher → slower ramp after a hold. |
| `stabilization_window_cycles_down` | `40` | `[30, 60]` typical | Shrink patience. Higher → less churn on transient idle; lower → reclaim sooner. |

Cross-field invariant: `stabilization_window_cycles_down >
stabilization_window_cycles_up` (validated). At default `interval_s=10s`,
``40`` down cycles ≈ **400s** of consecutive shrink recommendations
before Phase D can commit a removal.

See [02 — Decisions and growth](02-decisions-and-growth.md).

### Worker bounds (`min_workers`, `max_workers`)

| Field | Default | When to set |
|---|---|---|
| `min_workers` | `None` (implicit floor 1) | Pre-warm a stage with multi-minute model load. |
| `min_workers_per_node` | `None` | Stage with `WholeNumberedGpu` shape; one worker per node. |
| `max_workers` | `None` (cluster cap) | Cap an over-eager stage from monopolising the cluster. |
| `max_workers_per_node` | `None` | K8s topology-spread analog; even distribution. |

`min_workers` is enforced **structurally** in Phase B every
cycle. `max_workers` clamps Phase C grow. See
[06 — Safeguards](06-safeguards.md).

### Warmup grace windows

| Field | Default | When to adjust |
|---|---|---|
| `worker_warmup_measurement_grace_s` | `60.0 s` | Increase when a stage's actor `setup()` is consistently slow (large model load, lengthy initialisation). |
| `donor_warmup_grace_s` | `180.0 s` | Same trigger; donor grace must `>= worker_warmup_measurement_grace_s` (cross-field validator). |

Workers younger than these windows are excluded from EWMA
contribution and donor selection respectively. See
[02 — Decisions and growth](02-decisions-and-growth.md).

### Cluster-level cycle period

| Field | Default | When to adjust |
|---|---|---|
| `SaturationAwareConfig.interval_s` | `10.0 s` | Lower for tighter control on small clusters; raise on 10k-node clusters where the autoscaler's per-cycle work begins to compete with pipeline traffic. |

### Bottleneck decision integration

The Forced-Flow-Law `D_k`-driven Phase C grow priority and Phase D
shrink protection (see
[04 — Bottleneck awareness](04-bottleneck-awareness.md))
add five cluster-wide knobs. Defaults are set so the gate self-disables
on homogeneous clusters; most pipelines never need to tune these.

| Field | Default | When to adjust |
|---|---|---|
| `enable_bottleneck_priority_growth` | `True` | **Escape hatch.** Set `False` to keep Phase C on DAG-depth-only ordering when the bottleneck gate's grow-priority override produces a worse outcome than DAG depth (rare; usually a sign that `D_k` samples are too noisy). |
| `enable_bottleneck_shrink_protection` | `True` | **Escape hatch.** Set `False` to allow Phase D to shrink unconditionally when an operator-driven mid-run capacity reduction must shrink the bottleneck stage. Hard-cap ceiling overflow always shrinks regardless of this toggle. |
| `bottleneck_d_k_smoothing_level` | `0.20` | Lower toward `0.10` if `xenna_stage_bottleneck_score` is noisy and the gate's bottleneck identity flips between cycles. Raise toward `0.40` for short pipeline runs where the EWMA's warmup is itself the dominant effect. |
| `bottleneck_heterogeneity_threshold` | `2.0` | Raise toward `4.0` for very long pipelines where the median `D_k` is naturally pulled down by cheap I/O stages, so a `2.0` ratio is the steady state rather than a true bottleneck. Lower toward `1.5` only when the gate fails to engage on a known bottleneck (verify `D_k` panels first). |
| `bottleneck_engagement_persistence_cycles` | `2` | Raise toward `4` if the engagement INFO log fires noisily during regime transitions. Does not affect the gate's per-cycle decisions, only the log debouncer. |

### Backlog-time pressure gate

The classifier's compound pressure signal documented in
[01 — Signals and classification](01-signals-and-classification.md) has six
per-stage knobs. Most pipelines run with the defaults; the only one
operators typically tune is `target_backlog_seconds`.

| Field | Default | When to adjust |
|---|---|---|
| `target_backlog_seconds` | `30.0 s` | **Operator-facing primary knob.** The acceptable queue drain-time at which `normalized_backlog == 1.0`. Increase for latency-tolerant pipelines that can absorb a longer backlog before scaling up; decrease for latency-critical stages where queue depth must drain inside `target_backlog_seconds`. |
| `pressure_smoothing_level` | `0.20` | EWMA alpha on the composite pressure scalar. Increase toward `0.40` if the throughput sample is steady and the operator wants the pressure gauge to react faster; decrease toward `0.10` if `xenna_stage_pressure_ewma` shows cycle-to-cycle noise that masks the trend. |
| `pressure_critical_threshold` | `2.0` | Operator-pinned override. Increase only if a stage produces high-pressure bursts that should not trigger the burst-response (`SATURATED_CRITICAL`) path. The hard cap is `BACKLOG_CAP=3.0`. |
| `pressure_saturation_threshold` | `1.0` | Operator-pinned override. Lowering toward `0.6` makes the AND-criterion more permissive (slot pin still demoted if pressure is below `0.6`), which can be useful when measurements consistently underestimate true throughput. |
| `pressure_normal_threshold` | `0.3` | Operator-pinned override. Decrease (toward `0.1`) when an idle stage with a consistently long downstream queue should NOT scale down. Increase (toward `0.6`) when the operator wants the demotion gate to ignore mild pressure and shrink the stage anyway. |

Tuning workflow:

1. **Watch the gauges first.** [README — Trade-offs and observability](README.md#11-trade-offs-and-known-limitations)
   exposes `xenna_stage_observed_throughput`, `xenna_stage_backlog_time`,
   and `xenna_stage_pressure_ewma`. If `pressure_ewma` consistently sits
   in the `[1.5, 3.0]` band during steady-state operation but the
   pipeline is meeting SLO, increase `target_backlog_seconds` so the
   gauge's normal range shifts down.
2. **Tune `target_backlog_seconds` before the thresholds.** All three
   pressure thresholds scale against the normalised backlog; moving the
   single primary knob is preferred over tuning three thresholds at
   once.
3. **Use per-stage overrides for outlier stages.** A stage with
   variable batch sizes (Gaussian throughput) needs a higher
   `pressure_smoothing_level` than a stage with steady inference
   throughput.

The effective response time is `interval_s * streak_min_cycles`.
Lowering `interval_s` makes the autoscaler more reactive but also
amplifies the cost of the per-cycle work. The watchdog
([06 — Safeguards](06-safeguards.md)) makes excessive
runtime visible.

## Symptom-to-knob index

| Symptom | First knob to try | Decision doc |
|---|---|---|
| Stages oscillate between adding and removing workers | Increase `over_provisioned_streak_min_cycles` (longer scale-down patience) | [decisions](02-decisions-and-growth.md) |
| Phase D shrinks a freshly-warmed worker | Increase `worker_warmup_measurement_grace_s` and `donor_warmup_grace_s` | [decisions](02-decisions-and-growth.md) |
| New stage takes minutes to ramp under sustained load | Increase `saturation_aggressiveness` (`0.30` → `0.45`); confirm `enable_dag_priority_growth=True` | [signals](01-signals-and-classification.md), [bottleneck](04-bottleneck-awareness.md) |
| Mid-DAG bottleneck stays last in the grow order | Confirm `enable_bottleneck_priority_growth=True` and inspect `xenna_stage_bottleneck_score`: if the gate is silent, the heterogeneity ratio is below `bottleneck_heterogeneity_threshold` (default `2.0`); raise the metric (or lower the threshold toward `1.5`) so the gate engages | [bottleneck](04-bottleneck-awareness.md) |
| Bottleneck stage shrinks during transient idle | Confirm `enable_bottleneck_shrink_protection=True`; inspect Phase D INFO logs for `bottleneck shrink protected`. If absent, the gate is disengaged (cluster too homogeneous, see threshold guidance above) | [bottleneck](04-bottleneck-awareness.md) |
| Slot-pin SATURATED fires but no scale-up happens | Inspect `xenna_stage_pressure_ewma`; if `< pressure_saturation_threshold` (1.0) the demotion gate is correctly suppressing transient bursts. Decrease `target_backlog_seconds` if the pipeline truly is latency-critical and the queue drains too fast for the gate. | [signals](01-signals-and-classification.md), [README](README.md#11-trade-offs-and-known-limitations) |
| OVER_PROVISIONED stage refuses to scale down | Inspect `xenna_stage_pressure_ewma`; values `> pressure_normal_threshold` (0.3) are demoting to NORMAL because the queue is stuck downstream. Fix the downstream bottleneck first; only as a last resort raise `pressure_normal_threshold`. | [signals](01-signals-and-classification.md) |
| `xenna_stage_pressure_ewma` is unexpectedly noisy | Lower `pressure_smoothing_level` (`0.20` → `0.10`); confirm `xenna_stage_observed_throughput` itself is not noisy first | [signals](01-signals-and-classification.md) |
| Cluster-full pipeline never bootstraps a new stage | Confirm `floor_stuck_grace_cycles` is non-zero (default `6`); check Phase B donor logs | [donor](03-cross-stage-rebalancing.md) |
| Cross-stage donor rotates the same worker every cycle | Increase `cross_stage_donor_anti_flap_cycles` (default `30`) | [donor](03-cross-stage-rebalancing.md) |
| Regime detector enters / exits SUPER_HW every few cycles | Increase `regime_transition_streak_cycles` (default `3` → `5`) | [bottleneck](04-bottleneck-awareness.md) |
| Cycle p95 exceeds the watchdog WARN line | Profile first; tune `interval_s` only after profiling rules out an algorithmic issue | [safeguards](06-safeguards.md) |
| Object-store full → autoscaler still tries to grow | Lower `memory_pressure_critical_threshold` (default `0.85`) | [safeguards](06-safeguards.md) |
| Stage spec override silently ignored | Verify the resolver tier path (`StageSpec.saturation_aware` highest) | [README](README.md) |
| Stuck-plan counter ticks every cycle | Likely cluster-full; check Phase C / donor logs and consider raising the matching stage's `max_workers` | [safeguards](06-safeguards.md) |
| Pipeline hangs on cold start, no workers placed | Check `floor_stuck_grace_cycles` exhausted → `RuntimeError`; the cluster is genuinely too small to satisfy `min_workers` | [safeguards](06-safeguards.md) |

## Workload-class example configurations

These are starting points, not prescriptions. Always tune from the
default and revert if you cannot point at a specific symptom that
required the override.

### Throughput-oriented (long per-task work, latency-tolerant)

Default knee values are slightly conservative. Raise burst
response and accept a wider scale-down deadband.

    SaturationAwareConfig(
        interval_s=10.0,
        stage_defaults=SaturationAwareStageConfig(
            saturation_aggressiveness=0.35,
            saturated_streak_min_cycles=2,
            over_provisioned_streak_min_cycles=30,
            stabilization_window_cycles_down=30,
        ),
    )

### Latency-sensitive (low per-task latency target)

Faster reaction; tighter ramp; smaller stabilization windows.

    SaturationAwareConfig(
        interval_s=5.0,
        stage_defaults=SaturationAwareStageConfig(
            saturation_aggressiveness=0.45,
            saturated_streak_min_cycles=1,
            over_provisioned_streak_min_cycles=20,
            stabilization_window_cycles_down=15,
        ),
    )

### Stage with multi-minute model warmup (mixed pipeline)

Use a per-stage override on `StageSpec.saturation_aware` for the
slow-warming stage; keep the cluster default for the rest.

    slow_warming_stage_spec = StageSpec(
        ...,
        saturation_aware=SaturationAwareStageConfig(
            min_workers=4,                          # pre-warm 4 actors
            worker_warmup_measurement_grace_s=180,  # slow setup
            donor_warmup_grace_s=480,               # very slow setup
            over_provisioned_streak_min_cycles=60,  # patient shrink
        ),
    )

The other stages on the same pipeline get
`SaturationAwareConfig.stage_defaults` unchanged.

### GPU-stage with strict topology spread

One worker per node, exactly. Combine with `WholeNumberedGpu`
shape on the stage's resource spec.

    SaturationAwareStageConfig(
        min_workers_per_node=1,
        max_workers_per_node=1,
    )

## Tuning anti-patterns

- **Tuning two correlated knobs at once.** Always change one
  thing, run for at least
  `over_provisioned_streak_min_cycles + stabilization_window_cycles_down`
  cycles, observe, then change the next.
- **Tuning before profiling.** A slow autoscale cycle that the
  watchdog flags is rarely fixed by changing `interval_s`; it is
  usually a hot path inside the cycle that benefits from a
  Python-side optimisation in the relevant decision doc's
  source file.
- **Pinning `saturation_threshold` directly when
  `saturation_aggressiveness` would do.** The aggressiveness
  knob preserves the `K / √c` shape across stages with different
  `slots_per_worker`; pinning the threshold sets the same value
  for every stage and silently mis-calibrates most of them.
- **Setting `min_workers` to "just be safe".** The implicit
  floor of 1 is correct in almost every case. Use `min_workers`
  only when the *cost of being below the floor* is concrete
  (warm engine state, redundancy, latency SLO).

## Debug one unexpected decision

Start with the stage and the cycle number from the cycle-summary
INFO log line.

1. **Classifier state.** Was the stage `SATURATED`,
   `SATURATED_CRITICAL`, `NORMAL`, or `OVER_PROVISIONED`?
   Inspect `xenna_stage_classifier_state{stage,pipeline}` at the
   cycle in question.
2. **Pressure signal.** Were slots full and was queue pressure
   above threshold? See
   `xenna_stage_slots_empty_ratio_ewma` and
   `xenna_stage_backlog_time_seconds`.
3. **Stabilizers.** Did EWMA smoothing, streak counters,
   warmup grace, or HOLD-mode suppress the recommendation? Read
   `xenna_stage_streak_cycles` and the stage's classifier-streak
   in the cycle-summary log.
4. **Placement feasibility.** Did Rust placement reject a
   `try_add_worker` because the cluster was full or fragmented?
   Look for `xenna_scheduler_allocation_failures_total` increments
   and the structured `phase_c` allocation-failure log line.
5. **Safety gates.** Did floors, caps, memory pressure, donor
   anti-flap, or bottleneck protection override the action?
   `xenna_scheduler_memory_pressure_active`,
   `xenna_scheduler_stuck_plan_active`, and the structured
   `phase_d` shrink-outcome log line cover these cases.

If the answer is not visible from these metrics and logs alone,
that is an observability bug — open it as a follow-up rather
than re-running the pipeline under a profiler.

## Worst-case cycle duration

The autoscale cycle runs every `interval_s` seconds. If the
cycle itself takes too long, the autoscaler falls behind
real-time signal collection and recommendations lag.

Watch `xenna_scheduler_cycle_duration_seconds` (p95). If p95
approaches `interval_s`, either:

- raise `interval_s` (cluster-wide), accepting slower reaction; or
- shrink the active stage set (the cycle cost scales with the
  number of non-finished stages).

The `loop_watchdog` emits a WARN when one cycle exceeds
`cycle_time_warn_threshold * interval_s` (default `0.5 *
interval_s`). One isolated WARN is not actionable; sustained
WARNs are.

## Advanced tuning fields

The "Primary knobs" section above covers ~80% of operator tuning. The
categories below are expert-tier: they exist for diagnosis, niche
workloads, or A/B comparison. Defaults match the recommended production
behaviour, so reach for an expert field only when a documented symptom
matches one of the categories below; otherwise leave the default in
place.

This section points to the right `specs.py` field block and concept
doc per feature. It does NOT duplicate the per-field defaults, ranges,
or rationale — those live in the field comments and validators on
[`SaturationAwareConfig`](../../../cosmos_xenna/pipelines/private/specs.py)
and `SaturationAwareStageConfig`. When this section and `specs.py`
disagree, `specs.py` wins.

### Cross-stage donor (cluster rebalancing)

Anti-flap, eligibility, scoring weights, multi-donor resource-fit
search, and post-plan economic gate for the saturation-mode donor
fallback. The floor-mode donor reuses the same selection /
resource-fit pipeline but skips the economic gate.

- `specs.py` fields: `SaturationAwareConfig.enable_cross_stage_donor`,
  `donor_must_be_strictly_upstream`,
  `cross_stage_donor_require_over_provisioned`,
  `cross_stage_donor_exclude_hold_state`,
  `cross_stage_donor_anti_flap_cycles`,
  `cross_stage_donor_streak_bonus`,
  `cross_stage_donor_bottleneck_weight`,
  `cross_stage_donor_intent_weight`,
  `cross_stage_donor_streak_cap`,
  `cross_stage_donor_spread_threshold`,
  `cross_stage_donor_throughput_tolerance`,
  `cross_stage_donor_donor_flip_tolerance`,
  `cross_stage_donor_max_plan_size`,
  `cross_stage_donor_max_plan_combinations`.
- Decision doc: [03 — Cross-stage rebalancing](03-cross-stage-rebalancing.md).

### Regime detection (aggressiveness adaptation)

Lifts `saturation_aggressiveness` when the cluster sits in the
super-Halfin-Whitt regime so scale-up reacts faster, with hysteresis
to prevent flapping. Disable for queueing-theory-purity A/B comparisons
or when the cluster-wide idle-fraction signal is too noisy to trust.

- `specs.py` fields:
  `SaturationAwareConfig.enable_regime_aware_aggressiveness`,
  `super_halfin_whitt_aggressiveness_lift`,
  `regime_transition_streak_cycles`.
- Decision doc: [04 — Bottleneck awareness](04-bottleneck-awareness.md).

### Growth mode state machine (per-stage HOLD)

Per-stage state machine that prevents growth-after-shrink ping-pong by
holding a stage in HOLD mode for the standard post-shrink window. The
field is a single enable flag; disable only for diagnosis or to
isolate the contribution of streak counters and the stabilization
window to scale-up behaviour.

- `specs.py` field:
  `SaturationAwareStageConfig.enable_growth_mode_state_machine`.
- Decision doc: [02 — Decisions and growth](02-decisions-and-growth.md).

### Bottleneck scoring and engagement (Forced-Flow-Law `D_k`)

The primary section above covers the three operator-facing bottleneck
knobs (`bottleneck_d_k_smoothing_level`,
`bottleneck_heterogeneity_threshold`,
`bottleneck_engagement_persistence_cycles`). The fields below are the
cluster-heterogeneity warn-streak and warn-threshold that drive the
"recommend lowering capacity on cheap stages" operator hint.

- `specs.py` fields:
  `SaturationAwareConfig.cluster_heterogeneity_warn_threshold`,
  `cluster_heterogeneity_warn_streak`.
- Decision doc: [04 — Bottleneck awareness](04-bottleneck-awareness.md).

### Memory-pressure gate

Cluster-wide kill switch that freezes Phase C grow when the Ray
object-store `used_fraction` exceeds the configured threshold.
Phase A (manual), Phase B (floor), and Phase D (shrink) keep running.
Lower the threshold to be more conservative; disable only for
A/B comparison.

- `specs.py` fields: `SaturationAwareConfig.enable_memory_pressure_gate`,
  `memory_pressure_critical_threshold`,
  `memory_pressure_polling_interval_s`.
- Decision doc: [06 — Safeguards](06-safeguards.md).

### Streak counters, EWMA smoothing, stabilization windows

The primary section above covers the three streak-count knobs that
operators usually tune (`saturated_streak_min_cycles`,
`over_provisioned_streak_min_cycles`,
`saturated_critical_streak_min_cycles`). The fields below are the
smoothing-level, window-size, and data-sufficiency knobs that almost
never need adjustment outside of diagnosing a specific signal-noise
problem.

- `specs.py` fields (signal smoothing):
  `SaturationAwareStageConfig.slots_empty_ratio_smoothing_level`,
  `classifier_signal_noise_smoothing_level`,
  `pressure_smoothing_level` (covered in primary above).
- `specs.py` fields (asymmetric stabilization windows):
  `SaturationAwareStageConfig.stabilization_window_cycles_up`,
  `stabilization_window_cycles_down`.
- `specs.py` fields (data sufficiency gate):
  `SaturationAwareStageConfig.min_data_points`,
  `setup_phase_quiescence_enabled`.
- `specs.py` fields (classifier zone deadbands):
  `SaturationAwareStageConfig.saturation_deadband_pct`.
- Decision doc: [02 — Decisions and growth](02-decisions-and-growth.md).

### Auto-derived thresholds (`K / √c` formula)

Per-stage `saturation_threshold` and `activation_threshold` auto-derive
from `saturation_aggressiveness` on the first `autoscale()` cycle.
Operators rarely touch these directly — change
`saturation_aggressiveness` (covered in primary above) instead. The
fields below are the floor / ceiling clamps and the
`activation_to_saturation_ratio` that operators only touch for
extreme `c` values where the formula collides with a real-world floor.

- `specs.py` fields:
  `SaturationAwareStageConfig.saturation_threshold`,
  `activation_threshold`,
  `auto_threshold_min`,
  `auto_threshold_max`,
  `activation_to_saturation_ratio`,
  `over_provisioned_threshold`.
- Decision doc: [01 — Signals and classification](01-signals-and-classification.md).

### Stuck-plan detector and allocation-error tolerance

Detects pathological stages whose intent stays non-zero but never
satisfied for `stuck_plan_detection_cycles` cycles (typically because
the cluster is full and donor fallback is exhausted). Lower the
detection threshold to trip the operator alert sooner.
`skip_cycle_on_allocation_error` controls whether the Phase C
allocation-failure absorb path skips the rest of the cycle or
re-raises.

- `specs.py` fields:
  `SaturationAwareConfig.stuck_plan_detection_cycles`,
  `skip_cycle_on_allocation_error`,
  `floor_stuck_grace_cycles`.
- Decision doc: [06 — Safeguards](06-safeguards.md).

### Loop watchdog and cycle-time alerting

The autoscaler watchdog that fires a WARN when a single cycle exceeds
`cycle_time_warn_threshold * interval_s`. Covered in the primary
"Worst-case cycle duration" section above; included here for
completeness so this section carries the full expert-tier surface.

- `specs.py` field: `SaturationAwareConfig.cycle_time_warn_threshold`.
- Decision doc: [06 — Safeguards](06-safeguards.md).

## See also

- [README](README.md) — index of all concept docs and the
  notation table.
- [README — Where each decision lives](README.md#9-where-each-decision-lives-code-map)
  — owning module per concept, plus the three-tier override
  resolver and what is validated at startup vs at runtime.
- [README — Trade-offs and known limitations](README.md#11-trade-offs-and-known-limitations)
  — how to observe whether a tune actually changed behaviour.
