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
| ``SaturationAwareConfig`` | Cycle period | [00 — Per-cycle overview](00-overview.md) |
| ``SaturationAwareConfig`` | DAG priority + cross-stage donor anti-flap | [13 — Cross-stage donor](13-cross-stage-donor.md) |
| ``SaturationAwareConfig`` | Regime detector | [09 — Regime-aware aggressiveness](09-regime-aware-aggressiveness.md) |
| ``SaturationAwareConfig`` | Memory pressure gate | [20 — Memory pressure gate](20-memory-pressure-gate.md) |
| ``SaturationAwareConfig`` | Loop / stuck-plan watchdog | [18 — Loop watchdog](18-loop-watchdog.md) |
| ``SaturationAwareStageConfig`` | Classifier thresholds | [05 — State classifier](05-state-classifier.md), [08 — Auto-derived thresholds](08-auto-derived-thresholds.md) |
| ``SaturationAwareStageConfig`` | Streak counters | [07 — Streak stabilization](07-streak-stabilization.md) |
| ``SaturationAwareStageConfig`` | Growth mode + slow-start | [10 — Slow-start mechanisms](10-slow-start-mechanisms.md), [11 — Growth-mode state machine](11-growth-mode-state-machine.md) |
| ``SaturationAwareStageConfig`` | Per-stage caps and floors | [16 — Hard caps and floors](16-hard-caps-and-floors.md) |

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
