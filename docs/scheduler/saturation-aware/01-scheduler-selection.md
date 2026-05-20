# 01 — Scheduler Selection

## TL;DR

A frozen-at-init feature flag (`SchedulerKind`) selects between the
Rust `FragmentationBasedAutoscaler` and the
`SaturationAwareScheduler` per pipeline. `FRAGMENTATION_BASED` is the
default, so any pipeline that never touches the field keeps the
same behaviour it had before the saturation-aware scheduler shipped.

## Problem

Shipping a second autoscaler alongside the production one creates
four observable risks:

- A pipeline accidentally picking the new scheduler would change
  worker counts under the same workload and silently regress
  throughput before anyone reads a dashboard.
- The two algorithms keep incompatible per-cycle state (EWMA values,
  asymmetric streak counters, growth-mode flags on the saturation-aware
  scheduler vs. Rust max-min bookkeeping on the fragmentation-based
  one); swapping mid-run would orphan that state and corrupt the next
  decision.
- The selection has to round-trip through YAML / JSON specs unchanged
  even when Python class names move during a future refactor.
- The `autoscale_*` knobs on `StreamingSpecificSpec` drive the
  Rust solver only; an operator who picks the saturation-aware
  scheduler and still tunes those fields gets a silent no-op that
  is invisible from the spec alone.

## Decision

Add a `SchedulerKind` enum on `StreamingSpecificSpec.scheduler`, read
once at `Autoscaler.__init__` via `_make_scheduler_algorithm`, and
frozen for the lifetime of the run.

- Default `SchedulerKind.FRAGMENTATION_BASED` — every pipeline that
  has never touched the field gets the exact `FragmentationBasedAutoscaler`
  code path: no per-stage overrides collected, no new constructor call
  made, and no observable change in worker counts.
- Selection is frozen at init: the dispatcher binds the chosen
  algorithm to `Autoscaler._algorithm` and never re-reads the field.
  Hot-swapping would have to drain streak counters, EWMA history,
  and the worker-age map; freezing them in place removes a whole
  class of state-corruption bugs at the price of one trade-off --
  changing schedulers now requires re-running the pipeline.
- `SchedulerKind` is a `str`-valued `enum.Enum`
  (`"fragmentation_based"`, `"saturation_aware"`); the wire form is
  the string literal, so YAML / JSON specs do not depend on Python
  class identity and survive future class renames.
- The dispatcher emits a `WARN` log when a Rust-only `autoscale_*`
  field on `StreamingSpecificSpec` is set to a non-default value
  while `SchedulerKind.SATURATION_AWARE` is selected, so the silent
  mis-tune surfaces at startup instead of in a postmortem.

```
                  StreamingSpecificSpec.scheduler
                       (SchedulerKind enum)
                              │
                              │ read once at
                              │ Autoscaler.__init__
                              ▼
              ┌───────────────────────────────────┐
              │  _make_scheduler_algorithm        │
              │  (streaming.py dispatcher)        │
              └───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        FRAGMENTATION_BASED            SATURATION_AWARE
        (default)                      (opt-in)
              │                               │
              ▼                               ▼
   ┌──────────────────────┐        ┌──────────────────────┐
   │ Fragmentation        │        │ Saturation           │
   │ BasedAutoscaler      │        │ AwareScheduler       │
   │ (Rust solver)        │        │ (pure Python)        │
   └──────────────────────┘        └──────────────────────┘
```

## How it works

The dispatcher reads `pipeline_spec.config.mode_specific.scheduler`
once and constructs one of the two implementations. The
`FRAGMENTATION_BASED` branch is a pure passthrough of the
`autoscale_speed_estimation_*` fields the Rust solver consumes. The `SATURATION_AWARE` branch
additionally collects every `StageSpec.saturation_aware` override
via `_collect_saturation_aware_stage_overrides` and injects them
through the constructor; this is the single integration point for
spec-side per-stage overrides. An unrecognised `SchedulerKind` value
raises `ValueError` so a future enum addition cannot silently fall
through. Both implementations satisfy the `_SchedulerAlgorithm`
Protocol (`setup`, `update_with_measurements`, `autoscale`), so
`Autoscaler` — worker allocator, threading, measurement
aggregation — is agnostic to which one was chosen.

```
   Autoscaler.__init__(pipeline_spec, ...)
                         │  reads .scheduler once
                         ▼
   _make_scheduler_algorithm(pipeline_spec)
                         │  picks branch, builds algorithm,
                         │  binds it to self._algorithm
                         ▼
   self._algorithm.setup(problem)        ─── one-shot
                         │
                         ▼
   loop per autoscale_interval_s:
        update_with_measurements(...)
        autoscale(time, problem_state)   ─── per-cycle
                         │
                         ▼
   Autoscaler.__exit__ -> executor.shutdown()
```

## Knobs

- `StreamingSpecificSpec.scheduler` — the `SchedulerKind` enum;
  defaults to `FRAGMENTATION_BASED`.
- `StreamingSpecificSpec.saturation_aware` — `SaturationAwareConfig`;
  has no effect when `scheduler == FRAGMENTATION_BASED`.
- `StageSpec.saturation_aware` — optional per-stage
  `SaturationAwareStageConfig`; honoured only when
  `scheduler == SATURATION_AWARE`.
- `StreamingSpecificSpec.autoscale_speed_estimation_*` and
  `StageSpec.over_provision_factor` — fields consumed only by the
  Rust solver; non-default values trigger the `WARN` log above when
  `SATURATION_AWARE` is selected.

## See also

- [00 — Per-cycle overview](00-overview.md) — the saturation-aware
  cycle this flag opts a pipeline into.
- [02 — Configuration model](02-configuration-model.md) — the
  three-tier resolver that the dispatcher's
  `_collect_saturation_aware_stage_overrides` feeds.
- [03 — Planning context](03-planning-context.md) — the per-cycle
  bridge constructed by the chosen scheduler.
- [16 — Hard caps and floors](16-hard-caps-and-floors.md) — the
  per-stage caps that gate each cycle's plan, independent of which
  scheduler emits it.
- [17 — Config validation](17-config-validation.md) — the
  cross-field validators that the saturation-aware constructor runs
  before any worker is allocated.
