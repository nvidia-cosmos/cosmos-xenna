# 24 — Structured Logging

## TL;DR

Every saturation-aware decision — regime transition, resolved
thresholds, donor donation, Phase A / B / C / D action, allocation
failure — emits a structured `INFO` log line with a **stable field
schema** (`stage`, `cycle`, `intent`, `current`, `ceiling`, `floor`,
`deficit`, `donor_stage`, `donor_worker_id`, `donor_age`,
`threshold`, `effective_aggressiveness`, ...). Recovery and
capacity-exhaustion paths upgrade to `WARNING`. Per-cycle internals
that fire on every stage every cycle (cold-start quiescent skips,
hot-pending intent clamps) stay at `DEBUG`. The level ladder is
gated by the project-wide `PYTHON_LOG` environment variable; the
per-decision logging itself is fixed in source.

## Problem

The saturation-aware autoscaler runs a dense decision pipeline every
cycle: classify each stage into one of five zones, smooth signals
through EWMA and asymmetric streak counters, gate growth through a
state machine, attempt cross-stage donations, run four ordered
phases, and finally clamp every result against hard caps and floors.
When something goes wrong — a stage that will not grow, a donor
that flaps, a regime that should have lifted but did not — the
operator wants to reconstruct **which decisions fired and why**
without re-running the pipeline under a profiler.

The simplest channel for that reconstruction is the log stream. But
three failure modes recur when logging is left to the author of each
decision in isolation:

- **Free-form prose.** Each call site invents its own format, so a
  Grafana panel or `grep` recipe written against one decision breaks
  the next time the message is reworded for clarity.
- **Volume blow-up.** A line per stage per cycle at `INFO` saturates
  the log buffer on a 200-stage pipeline; the decision that mattered
  scrolls out before the operator finishes reading.
- **Hot-loop overhead.** Logging inside the per-stage intent
  pipeline without a level check pays the f-string construction cost
  on every cycle even when the sink filters the line away.

## Decision

Treat the **field schema** as the contract, not the prose. Every
algorithmic decision emits one structured line whose field names are
stable across scheduler refactors; the underlying decision can
change behind them, but `stage`, `cycle`, `intent`, `current`,
`ceiling`, `floor`, `donor_stage`, `donor_worker_id`, `donor_age`,
`threshold`, and `effective_aggressiveness` keep their meanings.
Three levels divide the noise:

```
┌──────────┬───────────────────────────────────────────────────────┐
│  INFO    │  every algorithmic decision and every regime /        │
│          │  threshold change; safe for production logs           │
├──────────┼───────────────────────────────────────────────────────┤
│  WARNING │  allocation failure approaching grace, stuck floor,   │
│          │  donor refused, watchdog over-budget                  │
├──────────┼───────────────────────────────────────────────────────┤
│  DEBUG   │  per-cycle internals (cold-start quiescent skip,      │
│          │  hot-pending intent clamp, EWMA / streak detail);     │
│          │  enabled only when investigating a single incident    │
└──────────┴───────────────────────────────────────────────────────┘
```

**Trade-off.** A single `INFO` firehose carrying every per-stage
per-cycle signal would be richer but would drown the operator on a
real-sized pipeline. A single `ERROR`-only stream would be quieter
but would force re-running the pipeline whenever a benign decision
needs to be reconstructed. The three-level ladder splits the cost:
production sees only decisions (and warnings about decisions that
could not complete), and adding `DEBUG` for an incident is a one-line
environment-variable change rather than a redeploy.

## How it works

The autoscaler imports the project's loguru wrapper and calls the
re-exported level methods directly:

```python
from cosmos_xenna.utils import python_log as logger

logger.info(
    f"saturation-aware scale-up: stage {stage_name!r} intent "
    f"+{intent} workers; hard worker cap left {headroom} "
    f"(current={current}, ceiling={ceiling})."
)
```

The schema is encoded as `name=value` pairs inside the parenthesised
tail. The leading prose names the decision family
(`saturation-aware scale-up`, `saturation-aware scale-down`,
`saturation-mode donation`, `scheduler regime transition`,
`scheduler resolved auto thresholds`, ...), so an operator can
filter on the family and a stable field name in one pipeline:

```
grep "saturation-aware scale-up" autoscaler.log | grep "stage 'caption'"
```

The level for each decision is fixed in source — the operator
cannot promote a `DEBUG` line to `INFO` (or vice versa) without
editing
[`saturation_aware.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/saturation_aware.py)
and redeploying:

| Decision                                | Level     | Sample line |
| --- | --- | --- |
| Regime transition                       | `INFO`    | `scheduler regime transition: -> SUPER_HALFIN_WHITT (total_workers=128, cluster_idle_fraction=0.0312, threshold=0.0500, effective_aggressiveness=0.45)` |
| Resolved auto thresholds (per stage)    | `INFO`    | `scheduler resolved auto thresholds for stage 'caption': slots_per_actor=4, saturation_aggressiveness=0.30, saturation_threshold=0.150000 (auto), activation_threshold=0.049500 (auto), over_provisioned_threshold=0.6 (config; not auto-derived)` |
| Phase B cross-stage floor donation      | `INFO`    | `[scheduler] 'caption': cross-stage minimum-floor donor accepted (donor_stage_index=3, donor_worker_id='w-12', donor_age=18)` |
| Phase C donor donation accepted         | `INFO`    | `[scheduler] saturation-mode donation: donor stage 'preprocess' worker 'w-12' (age=18) -> receiver stage 'caption' at cycle 47 (pending retry).` |
| Phase C ceiling clamp                   | `INFO`    | `saturation-aware scale-up: stage 'caption' intent +6 workers; hard worker cap left 2 (current=14, ceiling=16).` |
| Phase D shrink clamp (floor / fraction) | `INFO`    | `saturation-aware scale-down: stage 'caption' intent -4 workers; floor cap left 2 removed (deficit=2, current=4, floor=2).` |
| Stuck-plan threshold breached           | `INFO`    | `saturation-aware stuck plan: stage 'caption' stuck for 18 cycles (threshold=18, last_intent=2); growth blocked by cluster placement and donor selection.` |
| Stuck-plan recovery                     | `INFO`    | `saturation-aware stuck plan: stage 'caption' recovered; growth is no longer blocked.` |
| Memory-pressure gate engaged            | `WARNING` | `memory pressure gate: ACTIVE - cluster object-store used_fraction=0.91 exceeds critical_threshold=0.85; Phase C scale-up will be frozen until pressure clears.` |
| Memory-pressure gate cleared            | `INFO`    | `memory pressure gate: CLEARED - cluster object-store used_fraction=0.62 now within critical_threshold=0.85; Phase C scale-up resumes.` |
| Phase A manual placement exhausted      | `WARNING` | `manual grow: stage 'caption' requested 8 workers; cluster placement exhausted at 3 (deficit=5); manual request remains partially satisfied this cycle.` |
| Phase C cluster placement exhausted     | `WARNING` | `saturation-aware scale-up: stage 'caption' intent 6 workers; cluster placement exhausted after 3 (deficit=3); request remains partially satisfied this cycle.` |
| Donor refused by planner                | `WARNING` | `[scheduler] saturation-mode donor: stage 'preprocess' worker 'w-12' selected by donor helper but planner refused removal; donation cancelled and receiver retry skipped.` |
| Floor stuck, approaching grace          | `WARNING` | `[scheduler] 'caption': minimum-worker floor stuck (3/5 grace cycles); target_min=4, achieved=2, no eligible cross-stage donor; will raise after 2 more consecutive failed cycles.` |
| Loop watchdog over budget               | `WARNING` | `saturation-aware loop watchdog: autoscale cycle took 6.40s (threshold=5.00s = 0.5 * interval_s=10.0)` |
| Allocation failure absorbed             | `ERROR`   | `saturation-aware allocation failure: stage 'caption' raised AllocationError: ...; Per-GPU fragmentation snapshot: [{...}, {...}, ...]` |
| Per-cycle summary                       | `DEBUG`   | `saturation-aware cycle 47 summary: regime=SUB_HALFIN_WHITT, heterogeneity_streak=0, heterogeneity_fired=False, phase_c_allocation_failure=False` |
| Cold-start / hot-pending intent skip    | `DEBUG`   | `saturation-aware: stage 'caption' cold-start quiescent (pending=2, ready=0); skipping intent pipeline.` |

The loguru wrapper at
[`cosmos_xenna/utils/python_log.py`](../../../cosmos_xenna/utils/python_log.py)
parses `PYTHON_LOG` with `RUST_LOG`-style semantics: a default global
level plus optional per-module overrides matched by fnmatch glob,
most-specific match wins. Three canonical recipes cover the common
cases:

```
PYTHON_LOG=info
PYTHON_LOG=debug,cosmos_xenna.pipelines.private.streaming=warning
PYTHON_LOG=*=info,cosmos_xenna.pipelines.private.scheduling_py.*=debug
```

The first form sets every module to `INFO`. The second drops the
noisy streaming module to `WARNING` while leaving everything else at
`DEBUG`. The third is the "investigate one incident" recipe --
quiet the rest of the process and enable the scheduling internals.

**Don't-log rules.** Three patterns are out of bounds at `INFO` and
do not appear in source:

- **No per-cycle per-stage `INFO` lines.** A 200-stage pipeline emits
  200 lines every `interval_s` seconds; the decision that mattered
  scrolls off. Those signals belong at `DEBUG` — the cold-start and
  hot-pending quiescent skips above are exactly this pattern.
- **No log call inside a hot inner loop without a level check.** The
  f-string is built even when the sink would filter the record out,
  and on a per-worker or per-frame loop the construction dominates
  the loop. Aggregate first, log the decision once.
- **No raw signal arrays in the log stream.** Per-cycle slot
  signals, EWMA buffers, and streak histories are large and almost
  unreadable inline; they belong in the Prometheus gauges documented
  in [22-prometheus-metrics.md](22-prometheus-metrics.md) or in an
  offline trace.

## Knobs

There is exactly one knob, and it lives outside the scheduler: the
`PYTHON_LOG` environment variable read by the loguru wrapper at
process start. **Per-decision logging is fixed in source** — the
level for any specific decision cannot be retuned without editing
`saturation_aware.py` and redeploying. This is deliberate: the
schema is a public contract, and a per-deployment override would
break the very dashboards and grep recipes the schema exists to
protect.

| Knob | Default | Effect |
| --- | --- | --- |
| `PYTHON_LOG` | `INFO` (when unset) | Global default level. `OFF` silences the scheduler entirely; `DEBUG` enables the per-cycle internals. |
| `PYTHON_LOG=<glob>=<level>` | n/a | Per-module overrides, fnmatch glob against the dotted module path. Use `cosmos_xenna.pipelines.private.scheduling_py.*=debug` to enable scheduler `DEBUG` only. |

The default `INFO` level is the production target: every decision
and every warning is preserved, while per-cycle internals are
suppressed so the volume stays bounded by the number of *decisions
taken* and not by the number of *stages * cycles*.

## See also

- [00 — Per-cycle overview](00-overview.md) — the four phases that
  the per-decision `INFO` lines bracket.
- [18 — Loop watchdog](18-loop-watchdog.md) — the cycle-duration
  `WARNING` that complements the per-decision lines by reporting on
  the cycle envelope rather than the decisions inside it.
- [22 — Prometheus metrics](22-prometheus-metrics.md) — the
  numerical counterpart of these log lines; per-cycle internals
  (EWMA, streak, intent) and aggregated decision counts are exposed
  as gauges and counters there. Logs are for "which decision fired
  for this stage at this cycle"; metrics are for "what is the
  distribution across all stages and cycles".
- [loguru](https://loguru.readthedocs.io) — the underlying logger;
  level names, format strings, and sink semantics. The `PYTHON_LOG`
  parser at `cosmos_xenna/utils/python_log.py` is a project wrapper
  that adds `RUST_LOG`-style per-module filtering on top.
