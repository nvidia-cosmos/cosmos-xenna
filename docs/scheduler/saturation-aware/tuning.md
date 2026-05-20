# Operator Tuning Guide

This is the operator's quick-reference for tuning the saturation-aware
scheduler in production. The per-feature decision docs (`00-overview`
through `24-structured-logging` in this folder) explain **why** each
knob exists; this guide explains **which knob to turn when**.

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
  [02 — Configuration model](02-configuration-model.md) gives you
  scoped control.

**Do not tune** when:

- The pipeline has not yet run for at least
  `over_provisioned_streak_min_cycles + stabilization_window_cycles_down`
  steady-state cycles — most "first-impression" symptoms are
  cold-start artefacts that the slow-start mechanisms
  ([10](10-slow-start-mechanisms.md)) clear automatically.
- The symptom is a single-cycle anomaly. The hysteresis,
  EWMA smoothing, and streak counters are designed to absorb
  one-off noise without operator intervention.

## Primary knobs (the ~80% of tuning that matters)

### `saturation_aggressiveness`

| Field | Default | Range | When to adjust |
|---|---|---|---|
| `SaturationAwareStageConfig.saturation_aggressiveness` | `0.30` | `[0.10, 0.60]` | Whole-pipeline knee tuning. |

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

See [08 — Auto-derived thresholds](08-auto-derived-thresholds.md).

### Streak counters (asymmetric stabilization)

| Field | Default | Range | When to adjust |
|---|---|---|---|
| `saturated_streak_min_cycles` | `2` | `[1, 8]` typical | Burst response. Higher → slower ramp; lower → noisier ramp. |
| `over_provisioned_streak_min_cycles` | `30` | `[15, 60]` typical | Scale-down patience. Higher → less churn; lower → reclaim resources sooner. |
| `saturated_critical_streak_min_cycles` | `1` | `[1, 4]` | How fast `SATURATED_CRITICAL` fires. Lower means burst response in one cycle. |
| `starved_streak_min_cycles` | `6` | `[3, 12]` | How long before the upstream-bottleneck warning fires. |

Cross-field invariant: `over_provisioned_streak_min_cycles >
saturated_streak_min_cycles` (validated; see
[17 — Config validation](17-config-validation.md)). The
asymmetry is by design: a wrong scale-up is cheap; a wrong
scale-down kills warm GPU state.

See [07 — Streak stabilization](07-streak-stabilization.md).

### Worker bounds (`min_workers`, `max_workers`)

| Field | Default | When to set |
|---|---|---|
| `min_workers` | `None` (implicit floor 1) | Pre-warm a stage with multi-minute model load. |
| `min_workers_per_node` | `None` | Stage with `WholeNumberedGpu` shape; one worker per node. |
| `max_workers` | `None` (cluster cap) | Cap an over-eager stage from monopolising the cluster. |
| `max_workers_per_node` | `None` | K8s topology-spread analog; even distribution. |

`min_workers` is enforced **structurally** in Phase B every
cycle. `max_workers` clamps Phase C grow. See
[16 — Hard caps and floors](16-hard-caps-and-floors.md).

### Warmup grace windows

| Field | Default | When to adjust |
|---|---|---|
| `worker_warmup_measurement_grace_s` | `60.0 s` | Increase when a stage's actor `setup()` is consistently slow (large model load, lengthy initialisation). |
| `donor_warmup_grace_s` | `180.0 s` | Same trigger; donor grace must `>= worker_warmup_measurement_grace_s` (cross-field validator). |

Workers younger than these windows are excluded from EWMA
contribution and donor selection respectively. See
[10 — Slow-start mechanisms](10-slow-start-mechanisms.md).

### Cluster-level cycle period

| Field | Default | When to adjust |
|---|---|---|
| `SaturationAwareConfig.interval_s` | `10.0 s` | Lower for tighter control on small clusters; raise on 10k-node clusters where the autoscaler's per-cycle work begins to compete with pipeline traffic. |

The effective response time is `interval_s * streak_min_cycles`.
Lowering `interval_s` makes the autoscaler more reactive but also
amplifies the cost of the per-cycle work. The watchdog
([18 — Loop watchdog](18-loop-watchdog.md)) makes excessive
runtime visible.

## Symptom-to-knob index

| Symptom | First knob to try | Decision doc |
|---|---|---|
| Stages oscillate between adding and removing workers | Increase `over_provisioned_streak_min_cycles` (longer scale-down patience) | [07](07-streak-stabilization.md) |
| Phase D shrinks a freshly-warmed worker | Increase `worker_warmup_measurement_grace_s` and `donor_warmup_grace_s` | [10](10-slow-start-mechanisms.md), [15](15-idle-first-scale-down.md) |
| New stage takes minutes to ramp under sustained load | Increase `saturation_aggressiveness` (`0.30` → `0.45`); confirm `enable_dag_priority_growth=True` | [08](08-auto-derived-thresholds.md), [12](12-multi-target-dag-growth.md) |
| Cluster-full pipeline never bootstraps a new stage | Confirm `floor_stuck_grace_cycles` is non-zero (default `60`); check Phase B donor logs | [13](13-cross-stage-donor.md) |
| Cross-stage donor rotates the same worker every cycle | Increase `cross_stage_donor_anti_flap_cycles` (default `30`) | [13](13-cross-stage-donor.md) |
| Regime detector enters / exits SUPER_HW every few cycles | Increase `regime_transition_streak_cycles` (default `3` → `5`) | [09](09-regime-aware-aggressiveness.md) |
| Cycle p95 exceeds the watchdog WARN line | Profile first; tune `interval_s` only after profiling rules out an algorithmic issue | [18](18-loop-watchdog.md) |
| Object-store full → autoscaler still tries to grow | Lower `memory_pressure_critical_threshold` (default `0.85`) | [20](20-memory-pressure-gate.md) |
| Stage spec override silently ignored | Verify the resolver tier path (`StageSpec.saturation_aware` highest) | [02](02-configuration-model.md), [17](17-config-validation.md) |
| Stuck-plan counter ticks every cycle | Likely cluster-full; check Phase C / donor logs and consider raising the matching stage's `max_workers` | [21](21-allocation-error-tolerance.md) |
| Pipeline hangs on cold start, no workers placed | Check `floor_stuck_grace_cycles` exhausted → `RuntimeError`; the cluster is genuinely too small to satisfy `min_workers` | [16](16-hard-caps-and-floors.md) |

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
  `slots_per_actor`; pinning the threshold sets the same value
  for every stage and silently mis-calibrates most of them.
- **Setting `min_workers` to "just be safe".** The implicit
  floor of 1 is correct in almost every case. Use `min_workers`
  only when the *cost of being below the floor* is concrete
  (warm engine state, redundancy, latency SLO).

## See also

- [README](README.md) — index of all decision-rationale docs.
- [02 — Configuration model](02-configuration-model.md) — the
  three-tier resolver that decides which override wins.
- [17 — Config validation](17-config-validation.md) — what fails
  at startup vs at runtime, and which cross-field invariants are
  enforced.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — how to
  observe whether a tune actually changed behaviour.
