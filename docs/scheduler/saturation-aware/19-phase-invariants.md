# 19 — Phase Invariants

## TL;DR

Every cycle the scheduler runs a thin structural invariant suite at
each named phase boundary. A violation raises
`SchedulerInvariantError` (a real exception, not an `assert`), so
the check survives `python -O` in production and the offending plan
is refused before it ever reaches `Solution`. The error message
names the boundary — operators read "After PHASE_C: ..." instead of
"somewhere deep in autoscale()".

## Problem

The four-phase cycle (A manual → B floor → C grow → D shrink) is a
sequence of pure-Python mutations on a shared
`AutoscalePlanContext`. Each phase touches per-stage
`pending_add_count` and `pending_remove_count`, the classifier's
per-stage EWMA state, and the per-stage worker map (indirectly via
FFI to the Rust planner). A bug in any single phase can leave the
context in a state that is **structurally invalid** — negative
pending counter, NaN EWMA, stage-count drift, post-Phase-D floor
violation, stuck-plan counter regression — yet still passes through
`ctx.into_solution()` and emits a `Solution`. The streaming executor
consumes that `Solution` positionally, so a shape mismatch silently
misroutes worker mutations across stages.

Two specific failure modes drive the design:

- A defensive `assert` placed inside `autoscale()` would catch most
  of these, but `assert` is stripped under `python -O`, the
  supported production interpreter mode. The check disappears
  exactly where it is needed most, turning a loud failure into a
  silent corruption.
- An exception thrown deep inside Phase C with a message like
  "negative pending counter" gives an operator no anchor for which
  phase actually introduced the violation. Triage starts from the
  stack trace, not from the algorithm's structure.

## Decision

Run a thin invariant suite at every phase boundary and raise
`SchedulerInvariantError` — a real `RuntimeError` subclass defined
in
[`scheduling_py/errors.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/errors.py)
— on violation. Each check identifies its call site by a stable
[`PhaseBoundary`](../../../cosmos_xenna/pipelines/private/scheduling_py/invariants.py)
enum value (`PHASE_A`, `PHASE_B`, `PHASE_C`, `PHASE_D`,
`INTO_SOLUTION`) that ends up verbatim in the error message and in
a paired `logger.error(...)` line emitted just before raise — so a
higher-level supervisor that catches and re-wraps the exception
cannot drop the diagnostic.

```
            ProblemState ──▶ autoscale(time, problem_state)
                                    │
                                    ▼
                ┌─────────────────────────────────────────────┐
                │  Pre-flight gate                            │
                │  _check_problem_state_shape_before_phase_a  │  fail
                │    problem.stages ↔ problem_state.stages    │ ───┐
                │    counts + name-at-index                   │    │
                └─────────────────────────────────────────────┘    │
                                    │ pass                         │
                                    ▼                              │
                ┌─────────────────────────────────────────────┐    │
                │  Phase A — manual operator intent           │    │
                └─────────────────────────────────────────────┘    │
                                    │                              │
                ┌─────────────────────────────────────────────┐    │
                │  check_invariants_after_phase(PHASE_A)      │    │
                │    ctx.num_stages() == len(problem.stages)  │    │
                │    pending_add_count   ≥ 0 (per stage)      │ ───┤
                │    pending_remove_count ≥ 0 (per stage)     │    │
                └─────────────────────────────────────────────┘    │
                                    │ pass                         │
                                    ▼                              │
                ┌─────────────────────────────────────────────┐    │
                │  Phase B — floor enforcement                │    │
                └─────────────────────────────────────────────┘    │
                                    │                              │
                ┌─────────────────────────────────────────────┐    │
                │  check_invariants_after_phase(PHASE_B)      │ ───┤
                └─────────────────────────────────────────────┘    │
                                    │ pass                         │
                                    ▼                              │
                ┌─────────────────────────────────────────────┐    │
                │  Phase C — saturation-driven grow           │    │
                └─────────────────────────────────────────────┘    │
                                    │                              │
                ┌─────────────────────────────────────────────┐    │
                │  check_invariants_after_phase(PHASE_C)      │    │
                │  check_no_nan_in_classifier_state(PHASE_C)  │    │
                │    slots_empty_ratio_ewma   is finite       │ ───┤
                │    last_valid_slots_empty_ratio_ewma finite │    │
                └─────────────────────────────────────────────┘    │
                                    │ pass                         │
                                    ▼                              │
                snapshot pre_phase_d_worker_counts                 │
                                    │                              │
                                    ▼                              │
                ┌─────────────────────────────────────────────┐    │
                │  Phase D — saturation-driven shrink         │    │
                └─────────────────────────────────────────────┘    │
                                    │                              │
                ┌─────────────────────────────────────────────┐    │
                │  check_invariants_after_phase(PHASE_D)      │    │
                │  check_floor_after_phase_d(PHASE_D)         │    │
                │    min(pre_d, floor) ≤ now ≤ pre_d          │ ───┤
                │  check_stuck_plan_monotonicity              │    │
                │    curr == 0  ∨  curr == prev + 1           │    │
                └─────────────────────────────────────────────┘    │
                                    │ pass                         │
                                    ▼                              │
                ctx.into_solution()                                │
                                    │                              │
                ┌─────────────────────────────────────────────┐    │
                │  check_solution_shape(INTO_SOLUTION)        │ ───┤
                │    len(solution.stages) == len(problem)     │    │
                └─────────────────────────────────────────────┘    │
                                    │ pass                         ▼
                                    ▼                  SchedulerInvariantError
                              return Solution           ─ log @ ERROR
                                                        ─ message names boundary
                                                        ─ caller refuses to apply
```

**Trade-off.** A few microseconds per cycle for fail-loud behaviour
on internal corruption. The shape check is O(1); the per-stage
counter sweep is O(stages) with two FFI hops per stage; Phase C
adds an O(stages) classifier-state finiteness sweep; Phase D adds
an O(stages) floor sweep and an O(stages) stuck-plan monotonicity
sweep. All passes are pure-Python and allocate at most a shallow
dict view, so the per-cycle budget remains negligible for clusters
up to several thousand stages.

## How it works

Six named gates wrap one cycle. Each gate is a free function in
[`invariants.py`](../../../cosmos_xenna/pipelines/private/scheduling_py/invariants.py)
(except the pre-Phase-A shape check, which lives next to its only
caller as
`SaturationAwareScheduler._check_problem_state_shape_before_phase_a`).

| Boundary | Function | What it pins |
|---|---|---|
| (before PHASE_A) | `_check_problem_state_shape_before_phase_a` | `problem.stages` and `problem_state.stages` agree on count and name-at-index |
| PHASE_A, PHASE_B, PHASE_C, PHASE_D | `check_invariants_after_phase` | `ctx.num_stages() == len(problem.stages)`; every stage's `pending_add_count` and `pending_remove_count` is `≥ 0` |
| PHASE_C | `check_no_nan_in_classifier_state` | every stage's `slots_empty_ratio_ewma` and `last_valid_slots_empty_ratio_ewma` is finite (`None` is honoured as a cold-start sentinel) |
| PHASE_D | `check_floor_after_phase_d` | every non-manual non-finished stage `i` satisfies `min(pre_phase_d_worker_counts[i], stage_floors[i]) ≤ current ≤ pre_phase_d_worker_counts[i]` |
| PHASE_D | `check_stuck_plan_monotonicity` | per-stage stuck-plan counter transition is `curr == 0` (reset) or `curr == prev + 1` (strict increment); no other transition is legal |
| INTO_SOLUTION | `check_solution_shape` | `len(solution.stages) == len(problem.stages)` — callers consume the `Solution` positionally |

Two design choices deserve explicit calls out, because both protect
against a class of bug that the "obvious" implementation would miss:

- **Why the pre-Phase-A shape check is separate.** The four
  `check_invariants_after_phase` callers compare `ctx.num_stages()`
  against the captured `Problem`, but the cycle's `problem_state`
  is the operator-supplied runtime snapshot. If the snapshot
  disagrees with the captured `Problem` — different stage names or
  counts because the input spec mutated mid-run — the
  phase-specific indexing inside `_run_phase_a_delete` would raise
  an opaque `IndexError`. The pre-Phase-A check converts that into
  a labelled `SchedulerInvariantError` whose message identifies
  exactly which stage index disagrees, before any phase code runs.

- **Why Phase D captures its own pre-counts.** Phase B can
  legitimately leave a stage below its floor for a single grace
  cycle (the floor donor protocol exhausts grace and surfaces
  failure as a `RuntimeError`; until then the stage stays under
  floor while the next cycle places more workers). Phase D must
  respect that state, not "fix" it by mistake. The floor invariant
  therefore compares `current` against `min(pre_phase_d_count,
  floor)`: a `current` below `pre_phase_d_count` and below the
  floor unambiguously means Phase D dropped it (a defect), while a
  count that was already low at entry passes through. The upper
  bound `current ≤ pre_phase_d_count` separately pins Phase D as
  remove-only — growing a stage during shrink is also a defect.

Every error message names the boundary, the stage (when relevant),
the offending value, and the words "scheduler defect" — so an
operator reading the log line cannot mistake an invariant violation
for a configuration error.

## Knobs

None. Invariants are not configurable, and there is no operator
action that would make it correct to disable a gate. The per-cycle
cost is fixed and bounded (one shape check, one O(stages) counter
sweep per phase boundary, one O(stages) classifier-state sweep at
C, one O(stages) floor sweep and one O(stages) monotonicity sweep
at D); a violation is always a scheduler bug; the only valid
response is "refuse this plan and surface it".

## See also

- [00 — Per-cycle overview](00-overview.md) — where each gate sits
  relative to the four phases, and the cycle-level pre-flight
  preamble that builds the planner context.
- [04 — Per-cycle pipeline](04-per-cycle-pipeline.md) — drill-down
  on what each phase actually mutates on the planning context, so
  the invariant set in this doc lines up with the work each gate is
  validating against.
