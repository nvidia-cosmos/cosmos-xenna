# 17 — Config Validation

## TL;DR

Config invariants are enforced in two layers — single-field
predicates via `attrs.field(validator=...)` and cross-field
invariants in `__attrs_post_init__` — and the scheduler constructor
re-runs the cross-field pass against every `StageSpec.saturation_aware`
override before the first autoscale cycle. A misconfigured pipeline
fails at build time, in the constructor call that produced the bad
value, never mid-cycle.

## Problem

An autoscaler that trusts its configuration at first use exhibits
three failure modes:

- **Silent field-level corruption.** A negative streak count or a
  fraction outside `[0, 1]` passes Python's type checker but breaks
  the classifier on the first cycle that reads it — usually under
  load, when the cluster least wants a surprise.
- **Cross-field invariants are unreachable from a single-field
  predicate.** Threshold ordering
  (`activation < saturation < over_provisioned`), asymmetric streaks
  (`over_provisioned_streak_min_cycles > saturated_streak_min_cycles`),
  asymmetric stabilization windows, `min_workers <= max_workers`, and
  `donor_warmup_grace_s >= worker_warmup_measurement_grace_s` each
  span two or more fields. A validator that sees only one field at a
  time cannot reject any of them.
- **Bad per-stage overrides escape cluster-level validation.**
  `SaturationAwareConfig.__attrs_post_init__` only sees the two
  lower tiers it owns (`stage_defaults` plus `per_stage_overrides`).
  The tier-1 `StageSpec.saturation_aware` overrides are collected
  later, by `SaturationAwareScheduler.__init__`. A bad tier-1
  override that weakens the cluster-wide
  `cross_stage_donor_anti_flap_cycles >=
  max(over_provisioned_streak_min_cycles)` invariant slips past every
  per-class post-init and surfaces only when Phase B reaches for the
  donor under floor pressure.

## Decision

Two validation layers on the config classes, plus an eager re-run
of Layer 2 at the scheduler-constructor boundary.

- **Layer 1 — single-field validators on `attrs.field(...)`.** Every
  field with a numeric, range, or type contract is decorated with a
  stdlib predicate (`attrs.validators.ge`, `gt`, `le`, `in_`,
  `instance_of`, composed via `attrs.validators.and_` /
  `attrs.validators.optional`) or with a contract-named helper from
  [`cosmos-xenna/cosmos_xenna/utils/attrs_utils.py`](../../../cosmos_xenna/utils/attrs_utils.py)
  such as `validate_positive_int` and
  `validate_optional_positive_int`. Predicates are named after the
  contract, not the field, so one helper serves every field with the
  same constraint and the error message identifies the failing
  attribute dynamically via `attrs.Attribute.name`.
- **Layer 2 — `__attrs_post_init__` for cross-field invariants.**
  `SaturationAwareStageConfig.__attrs_post_init__` enforces threshold
  ordering, asymmetric streak ordering, asymmetric stabilization
  window ordering, auto-threshold clamp ordering, slow-start grace
  ordering, and `min <= max` on both cluster-wide and per-node worker
  caps. Each invariant raises `ValueError` with a message that names
  every offending field and its concrete value so the operator can
  correlate the failure with the spec they wrote.
- **Eager scheduler-constructor re-run.** `SaturationAwareScheduler.__init__`
  calls `self._config.validate_effective_stage_configs(spec_overrides=...)`
  with the runtime `StageSpec.saturation_aware` overrides included
  before any cycle runs. The cluster-wide invariant (donor anti-flap
  dominates the longest shrink streak across every effective config)
  is re-checked against the union of all three tiers, so a
  higher-precedence override cannot silently weaken a cluster
  guardrail. A misconfigured override raises synchronously from
  `SaturationAwareScheduler.__init__`; the planner thread never sees it.

```
        SaturationAwareConfig(...)                SaturationAwareScheduler(
        SaturationAwareStageConfig(...)               config=cfg,
                    │                                 stage_spec_overrides={...},
                    │ construction                )
                    ▼                                     │
        ┌────────────────────────────────┐                │ constructor
        │  Layer 1                       │                ▼
        │  attrs.field(validator=...)    │   ┌────────────────────────────────┐
        │  ─ attrs.validators.ge / gt /  │   │  Eager re-run of Layer 2       │
        │    le / in_ / instance_of /    │   │  on the full three-tier set    │
        │    and_ / optional             │   │                                │
        │  ─ attrs_utils.validate_       │   │  validate_effective_stage_     │
        │    positive_int / optional_    │   │  configs(spec_overrides=...)   │
        │    positive_int                │   │                                │
        │  one predicate per field;      │   │  stage_defaults                │
        │  reusable across fields with   │   │   + per_stage_overrides        │
        │  the same constraint           │   │   + tier-1 spec overrides      │
        └──────────────┬─────────────────┘   └─────────────────┬──────────────┘
                       │                                       │
                       ▼                                       │
        ┌────────────────────────────────┐                     │
        │  Layer 2                       │                     │
        │  __attrs_post_init__           │                     │
        │  cross-field invariants:       │                     │
        │  ─ threshold ordering          │                     │
        │  ─ asymmetric streaks          │                     │
        │  ─ asymmetric stabilization    │                     │
        │  ─ auto-threshold clamps       │                     │
        │  ─ slow-start grace ordering   │                     │
        │  ─ min <= max (caps & floors)  │                     │
        └──────────────┬─────────────────┘                     │
                       │                                       │
                       ▼                                       ▼
              ┌───────────────────────────────────────────────────────┐
              │  raise ValueError(operator-actionable message naming  │
              │                   every offending field and value)    │
              └───────────────────────────────────────────────────────┘
```

**Trade-off.** Validation costs a few microseconds of build-time CPU
and one extra method on each config class. In exchange, every
misconfiguration surfaces in the same Python call that produced it,
with a stack trace pointing at the offending field. There is no
per-cycle cost — once construction succeeds, the scheduler never
re-validates.

## How it works

Each layer runs at the lifecycle point where it is the cheapest
validator that can see its inputs.

- **Layer 1** fires inside the attrs-generated `__init__`, before any
  user code observes the field. Implementations come from
  `attrs.validators.*` (range, type, container predicates) and from
  `cosmos_xenna/utils/attrs_utils.py`. Validator names describe
  contracts, not fields: a single `validate_positive_int` serves
  every `>= 1` integer field and the runtime error message identifies
  the failing field via `attrs.Attribute.name`.
- **Layer 2** fires immediately after attrs sets every field, at the
  point where the class first holds a self-consistent state to
  inspect. `SaturationAwareStageConfig.__attrs_post_init__` checks
  the per-stage invariants in source order;
  `SaturationAwareConfig.__attrs_post_init__` checks the cluster-wide
  invariant by delegating to `validate_effective_stage_configs(spec_overrides=())`
  against the two lower tiers it can see at construction time
  (`stage_defaults` plus `per_stage_overrides`).
- **Eager re-run** fires in `SaturationAwareScheduler.__init__`. The
  constructor calls
  `self._config.validate_effective_stage_configs(tuple(stage_spec_overrides.values()))`,
  then locks the override map behind a `types.MappingProxyType`. The
  empty-input fast path skips the validator call to keep the
  no-override common case free. Layers 1 and 2 already fired during
  `SaturationAwareConfig` construction; the re-run extends Layer 2 to
  the strictly larger three-tier set now that the tier-1 overrides
  are visible. The invariant is monotone in the set of stage configs
  (adding more configs can only raise the longest observed streak),
  so the second pass strictly extends the first.

The full source-order list of invariants enforced by Layer 2 lives
in [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py)
under `SaturationAwareStageConfig.__attrs_post_init__`; the
cluster-wide invariant enforced by the eager re-run lives in
`SaturationAwareConfig.validate_effective_stage_configs`. Both
methods carry the canonical error-message wording and are the single
source of truth.

## Knobs

None. The validators themselves are not configurable: they exist to
enforce contracts that the rest of the scheduler relies on, and
weakening them at runtime would defeat their purpose. The
constraints themselves (per-field bounds, cross-field orderings,
cluster-wide dominance rules) are listed inline next to each field
in [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py);
add or change a constraint there if a future workload needs it,
never by patching the validator chain.

## See also

- [02 — Configuration model](02-configuration-model.md) — the two
  config classes and the three-tier resolver that this validation
  layer protects.
- [16 — Hard caps and floors](16-hard-caps-and-floors.md) — the
  `min_workers` / `max_workers` and per-node variants that the
  `min <= max` cross-field invariants enforce.
- [`cosmos-xenna/cosmos_xenna/utils/attrs_utils.py`](../../../cosmos_xenna/utils/attrs_utils.py)
  — `validate_positive_int` and `validate_optional_positive_int`.
- [`cosmos-xenna/cosmos_xenna/pipelines/private/specs.py`](../../../cosmos_xenna/pipelines/private/specs.py)
  — `SaturationAwareStageConfig`, `SaturationAwareConfig`, and the
  per-class `__attrs_post_init__` plus
  `validate_effective_stage_configs` source.
- [`cosmos-xenna/cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
  — `SaturationAwareScheduler.__init__` invokes the eager re-run.
