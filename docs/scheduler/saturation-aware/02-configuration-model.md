# 02 — Configuration Model

## TL;DR

Scheduler configuration lives on two classes:
``SaturationAwareConfig`` (cluster-wide) and
``SaturationAwareStageConfig`` (per-stage). The effective per-stage
config is resolved by a strict three-tier precedence chain —
``StageSpec.saturation_aware`` > ``per_stage_overrides[name]`` >
``stage_defaults`` — validated eagerly in the scheduler constructor
and locked behind a read-only ``MappingProxyType`` view for the rest
of the run.

## Problem

Bolting every scheduler knob onto a single flat config class produces
three failure modes that surface in heterogeneous pipelines:

- **Shared knobs hide workload diversity.** A stage with a
  multi-minute model warmup needs a long donor warmup grace; a stage
  that warms in seconds needs a short one. A single global value
  forces every stage to compromise on whichever workload shouts
  loudest during deployment.
- **Cluster-wide invariants are bypassed silently.** Cross-stage
  donor anti-flap (``cross_stage_donor_anti_flap_cycles``) must
  dominate the longest ``over_provisioned_streak_min_cycles`` across
  every stage. That invariant cannot be checked on a per-stage knob
  in isolation; the cluster needs visibility into every stage at once.
- **Misconfiguration surfaces mid-cycle.** A bad per-stage override
  that passes every per-field validator can run for many cycles and
  only crash when the autoscaler reaches the path that consumes the
  bad value — under load, when the cluster least wants a surprise
  ``RuntimeError`` or ``AssertionError``.

## Decision

Split configuration along the natural axis of concern, then resolve
per-stage knobs with a strict three-tier precedence chain that is
validated once and frozen.

- **Two classes, not one.** Cluster-wide concerns (autoscale cycle
  period, regime detector, cross-stage donor anti-flap, memory
  pressure gate, stuck-plan watchdog) live on
  ``SaturationAwareConfig``. Per-stage concerns (classifier
  thresholds, streak counters, EWMA alpha, growth-mode parameters,
  slow-start grace windows, per-stage worker caps and floors) live on
  ``SaturationAwareStageConfig``. The split mirrors the
  "controller vs target" separation Kubernetes HPA uses for the same
  reason: one controller, many targets, different tuning surfaces.
- **Three-tier resolver, first match wins.** Per-stage lookup walks
  three sources in order: ``StageSpec.saturation_aware``
  (programmatic per-stage override; highest precedence) >
  ``SaturationAwareConfig.per_stage_overrides[stage_name]``
  (cluster-level registry keyed by stage name) >
  ``SaturationAwareConfig.stage_defaults`` (cluster-wide default,
  always populated). The chain is implemented in the five-line
  ``SaturationAwareConfig.get_effective_stage_config``.
- **Eager validation in the scheduler constructor.**
  ``SaturationAwareScheduler.__init__`` collects the runtime
  ``StageSpec.saturation_aware`` overrides and re-runs every
  cross-stage invariant via
  ``SaturationAwareConfig.validate_effective_stage_configs`` with the
  tier-1 overrides included. A misconfigured override raises
  ``ValueError`` at build time, never inside ``autoscale()``.
- **Lock the override map after validation.** The collected runtime
  overrides are copied into a fresh ``dict``, wrapped in
  ``types.MappingProxyType``, and stored as
  ``Mapping[str, SaturationAwareStageConfig]``. Post-construction
  caller mutation raises ``TypeError`` at the language layer; mypy
  rejects mutation at the type layer; the hot path never sees a
  tampered override.

```
              get_effective_stage_config(stage_name, spec_override)
                                     │
                                     │  first match wins
                                     ▼
              ┌────────────────────────────────────────────────────┐
              │  1. StageSpec.saturation_aware                     │
              │     (per-stage spec; highest precedence; passed    │
              │      to the resolver as ``spec_override``)         │
              └──────────────────────────┬─────────────────────────┘
                                     │   None
                                     ▼
              ┌────────────────────────────────────────────────────┐
              │  2. SaturationAwareConfig.per_stage_overrides      │
              │     [stage_name]   (cluster-level, keyed by name)  │
              └──────────────────────────┬─────────────────────────┘
                                     │   miss
                                     ▼
              ┌────────────────────────────────────────────────────┐
              │  3. SaturationAwareConfig.stage_defaults           │
              │     (cluster-wide default; always populated)       │
              └──────────────────────────┬─────────────────────────┘
                                     ▼
                       SaturationAwareStageConfig
                (used by thresholds, streaks, caps, donor logic ...)
```

Trade-offs: two config classes add a second surface for operators to
learn — accepted because each class is single-purpose and the
boundary maps directly to "cluster controller vs per-stage target".
Three tiers add five lines of resolver logic — accepted because the
alternative is editing a global default to "fix" one stage and
silently affecting every other stage. Eager validation pays a one-off
cost at build time to remove an entire class of mid-cycle failures.

## How it works

The resolver itself is the five-line
``SaturationAwareConfig.get_effective_stage_config(stage_name,
spec_override)``. Inside the scheduler, every consumer goes through
the ``SaturationAwareScheduler._stage_cfg(stage_name)`` helper, which
calls the resolver with the locked override map as the
``spec_override`` argument. Phase B floor enforcement, Phase C grow,
Phase D shrink, donor selection, and threshold resolution all observe
the same precedence chain — no code path can accidentally read
``stage_defaults`` while another reads ``per_stage_overrides``.

Validation runs in two passes against the same set of cross-stage
invariants:

1. ``SaturationAwareConfig.__attrs_post_init__`` checks
   ``stage_defaults`` plus ``per_stage_overrides`` at construction
   of the cluster config itself, when only the two lower tiers are
   knowable.
2. ``SaturationAwareScheduler.__init__`` re-runs
   ``validate_effective_stage_configs`` with the collected
   ``StageSpec.saturation_aware`` overrides included, so the
   highest-precedence tier cannot weaken a cluster-wide guardrail
   (today: ``cross_stage_donor_anti_flap_cycles >= max
   over_provisioned_streak_min_cycles`` across every effective stage
   config).

After validation, ``_stage_spec_overrides`` is wrapped in
``types.MappingProxyType``. From that point the scheduler instance is
immutable wiring: there is no public setter and no way for the
orchestrator (or a future refactor) to introduce a hot-path race or a
"forgot to install overrides" regression.

## Knobs

The two classes themselves are the configuration surface; each knob
group is rationalised in a sibling document.

| Class | Knob group | Sibling doc |
|---|---|---|
| ``SaturationAwareConfig`` | Cycle period (``interval_s``) | [00 — Per-cycle overview](00-overview.md) |
| ``SaturationAwareConfig`` | DAG priority + cross-stage donor anti-flap | [13 — Cross-stage donor](13-cross-stage-donor.md) |
| ``SaturationAwareConfig`` | Regime detector | [09 — Regime-aware aggressiveness](09-regime-aware-aggressiveness.md) |
| ``SaturationAwareConfig`` | Memory pressure gate | [20 — Memory pressure gate](20-memory-pressure-gate.md) |
| ``SaturationAwareConfig`` | Loop / stuck-plan watchdog | [18 — Loop watchdog](18-loop-watchdog.md) |
| ``SaturationAwareStageConfig`` | Trust gate (``min_data_points``) | [05 — State classifier](05-state-classifier.md) |
| ``SaturationAwareStageConfig`` | Classifier thresholds | [05 — State classifier](05-state-classifier.md), [08 — Auto-derived thresholds](08-auto-derived-thresholds.md) |
| ``SaturationAwareStageConfig`` | Backlog-time pressure gate (6 knobs) | [06 — Backlog-time pressure signal](06-backlog-time-signal.md) |
| ``SaturationAwareStageConfig`` | Streak counters | [07 — Streak stabilization](07-streak-stabilization.md) |
| ``SaturationAwareStageConfig`` | Growth mode + slow-start (``enable_growth_mode_state_machine``) | [10 — Slow-start mechanisms](10-slow-start-mechanisms.md), [11 — Growth-mode state machine](11-growth-mode-state-machine.md) |
| ``SaturationAwareStageConfig`` | Per-stage caps and floors | [16 — Hard caps and floors](16-hard-caps-and-floors.md) |

### Dispatcher cadence wiring

``SaturationAwareConfig.interval_s`` (default ``10.0`` s) is now the
single source of truth for the streaming dispatcher's autoscale
cadence under ``SchedulerKind.SATURATION_AWARE``. The shared
``effective_autoscale_interval(pipeline_spec)`` helper in
``streaming.py`` branches on ``mode_specific.scheduler``:

```
SchedulerKind.SATURATION_AWARE   -> mode_specific.saturation_aware.interval_s
SchedulerKind.FRAGMENTATION_BASED -> mode_specific.autoscale_interval_s (180 s)
```

The fragmentation-based path intentionally keeps the 180 s cadence
because its Rust-backed solver is expensive; saturation-aware's
watchdog and growth windows are sized for 10 s cycles. The helper
emits one ``INFO`` log per pipeline naming the resolved cadence and
the source field so deployment-time changes are auditable.

### Trust gate and growth-mode kill switch

Two per-stage knobs gate the classifier-driven action loop:

- ``min_data_points`` (default ``5``) - the trust gate. The scheduler
  keeps the EWMA and classifier history flowing, but clamps any
  non-zero recommendation to ``0`` until the stage has accumulated
  ``min_data_points`` consecutive valid samples. A "valid" sample is
  one in which the warmup filter did not drop every contribution
  (``num_used_slots + num_empty_slots > 0``). The gate cannot starve
  a zero-worker stage because Phase B floor enforcement runs outside
  the gate.
- ``enable_growth_mode_state_machine`` (default ``True``) - the
  growth-mode kill switch. When ``False``, ``compute_delta`` always
  uses ``GrowthMode.TRACKING`` magnitudes (additive +1 / +2) and
  ``record_executed_delta`` skips the state machine update, so the
  per-stage runtime state stays frozen at its construction-time
  defaults. Re-enabling the flag mid-run resumes from ACQUIRING.

### Backlog-time pressure gate

Six per-stage knobs control the compound pressure classifier introduced
in [06 — Backlog-time pressure signal](06-backlog-time-signal.md).
These extend the slot-ratio gate with a smoothed compound
`pressure = utilisation * normalized_backlog` scalar, used as a
demotion gate inside each slot-pin branch:

| Knob | Default | Effect |
|---|---|---|
| ``target_backlog_seconds`` | ``30.0`` | Operator-facing primary knob: queue drain-time at which ``normalized_backlog == 1.0``. Higher = more conservative (longer queue accepted before scale-up); lower = more aggressive. |
| ``pressure_smoothing_level`` | ``0.20`` | EWMA alpha applied to the composite pressure scalar. Lower = smoother (filters throughput noise); higher = reacts faster. Bounded ``(0.0, 1.0]``. |
| ``pressure_critical_threshold`` | ``2.0`` | Smoothed pressure above which a slot-pin ``SATURATED_CRITICAL`` actually fires. Strictly larger than ``pressure_saturation_threshold`` and ``≤ BACKLOG_CAP`` (``3.0``). |
| ``pressure_saturation_threshold`` | ``1.0`` | Pressure above which a slot-pin ``SATURATED`` actually fires. Strictly larger than ``pressure_normal_threshold``. |
| ``pressure_normal_threshold`` | ``0.3`` | Pressure above which a slot-pin ``OVER_PROVISIONED`` is demoted to ``NORMAL`` (queue is stuck downstream; shrinking would worsen the bottleneck). |
| ``enable_backlog_time_classifier`` | ``True`` | Escape hatch: ``False`` reverts the stage to legacy slot-only behaviour (no demotion, no pressure refresh). |

Cross-field invariants enforced at construction time:

```
pressure_critical_threshold > pressure_saturation_threshold > pressure_normal_threshold
pressure_critical_threshold ≤ BACKLOG_CAP   (= 3.0)
```

## See also

- [00 — Per-cycle overview](00-overview.md) — how the scheduler
  consumes the resolved config each cycle.
- [01 — Scheduler selection](01-scheduler-selection.md) — how a
  pipeline opts into this scheduler in the first place.
- [17 — Config validation](17-config-validation.md) — the
  single-field and cross-field validators that the resolver
  delegates to.
- [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py)
  — ``SaturationAwareConfig``, ``SaturationAwareStageConfig``,
  ``get_effective_stage_config``, and
  ``validate_effective_stage_configs``.
- [`cosmos-xenna/cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
  — ``SaturationAwareScheduler.__init__`` and ``_stage_cfg``.
