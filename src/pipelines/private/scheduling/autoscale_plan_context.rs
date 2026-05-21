// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Per-cycle autoscale planning context exposed to Python.
//!
//! `AutoscalePlanContext` is a `pyclass` that owns a working copy of the
//! cluster snapshot plus per-stage staged adds and removes. Its construction
//! path, worker-add path, worker-remove path, and final `into_solution`
//! conversion are live today.
//!
//! Lifecycle (one instance per autoscale cycle):
//!
//!   1. `from_problem_state(problem, state, *, worker_ages=None)` -
//!      seeds the working cluster with all currently-allocated workers
//!      from the input snapshot; initialises empty per-stage
//!      `pending_adds` / `pending_removes` maps; computes the workload
//!      estimate used by FGD; seeds the per-worker age map from the
//!      optional `worker_ages` kwarg (each entry is the age in cycles
//!      already incremented by the caller for the new cycle, defaulting
//!      to 0 for ids the scheduler has never observed before).
//!   2. `try_add_worker(stage_index)` - runs Fragmentation Gradient
//!      Descent against the working cluster, mutates `cluster` to
//!      reflect the new allocation, appends the placement to
//!      `pending_adds[stage_name]`, and returns the placed worker.
//!      Reuses a worker already staged for removal when its placement
//!      is the best candidate (cancels the pending remove instead of
//!      adding to `pending_adds`). Returns `None` when the cluster
//!      cannot satisfy the request.
//!   3. `try_remove_worker(stage_index, worker_id)` - removes a worker
//!      from the working cluster by id and stages it in
//!      `pending_removes[stage_name]`; returns `false` for unknown ids.
//!      A worker added earlier in this same cycle is symmetric: the
//!      pending add is cancelled rather than pushed to pending_removes.
//!   4. `into_solution()` - drains `pending_adds` / `pending_removes`
//!      into one ordered `StageSolution` per stage (parallel to
//!      `Problem.stages`); preserves the per-stage `slots_per_worker`
//!      captured at construction time. Workers reused inside the
//!      cycle (remove cancelled by a later add, or add cancelled by
//!      a later remove) appear in NEITHER `new_workers` NOR
//!      `deleted_workers`. The drain is in-place: a second call on
//!      the same context returns a `Solution` whose per-stage lists
//!      are all empty.
//!
//! Callers MUST construct a fresh `AutoscalePlanContext` for each cycle;
//! mutations are scoped to the context instance and the per-stage
//! pending lists are drained by `into_solution`.
//!

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;

use super::autoscaling_algorithms::{
    DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT, WorkerIdFactory, make_workload_from_state,
};
use super::data_structures as ds;
use super::fragmentation_allocation_algorithms as frag;
use super::resources as rds;

/// Per-cycle autoscale planning context.
///
/// Owns a working copy of the cluster snapshot seeded with current worker
/// allocations, plus per-stage maps of pending adds and removes. Planning
/// methods mutate this private snapshot; callers build one context per
/// autoscale cycle.
///
/// # Fields
/// * `cluster` - Working cluster snapshot. Cloned from the input
///   `Problem.cluster_resources` and pre-allocated with all workers
///   currently in `ProblemState`. Mutated during planning by
///   `try_add_worker` / `try_remove_worker`.
/// * `stages` - Cloned ordered list of `ProblemStage` from the input
///   problem. `try_add_worker(stage_index)` indexes this vector to
///   recover the stage name and worker shape needed by FGD without
///   forcing the caller to re-pass the original `Problem`.
/// * `pending_adds` - Per-stage list of workers staged for addition this
///   cycle. Maps stage_name -> Vec of `ProblemWorkerGroupState` to add.
///   Reused workers (those whose pending-remove was cancelled by a
///   subsequent `try_add_worker`) are deliberately NOT pushed here:
///   they were never structurally removed from the live set, only
///   un-staged.
/// * `pending_removes` - Per-stage list of workers staged for removal this
///   cycle. Maps stage_name -> Vec of `ProblemWorkerGroupState` to remove.
/// * `current_workers` - Per-stage current non-SPMD workers in the mutable
///   planning snapshot. Fresh adds are inserted here and removals pop from
///   here, so multiple planning operations compose inside one cycle.
/// * `current_worker_groups` - Same mutable snapshot for multi-allocation
///   SPMD worker groups.
/// * `reserved_worker_ids` - Every worker id observed (seeded from
///   `ProblemState`) or minted (by `make_unique_worker_id`) during this
///   context's cycle. The id factory consults this set and regenerates
///   on collision so a freshly minted id can never alias a seeded live
///   worker, a pending-add already staged earlier in the cycle, or a
///   pending-remove waiting to be drained.
/// * `slots_per_worker_by_stage` - `slots_per_worker` value snapshotted
///   from each `ProblemStageState` at construction time. Indexed by
///   stage position so `into_solution` can produce a `Solution` whose
///   `StageSolution.slots_per_worker` round-trips the input cycle's
///   value (the planner does not change `slots_per_worker`; it is a
///   per-stage configuration property).
/// * `worker_id_factory` - Generates unique IDs for newly allocated workers.
/// * `workload_estimate` - Per-stage workload weights consumed by FGD when
///   choosing among candidate placements.
/// * `worker_reuse_fragmentation_equivalent` - FGD reward parameter that
///   biases the search toward reusing a recently removed placement (avoids
///   thrash when a stage shrinks then re-grows in the same cycle).
/// * `is_drained` - Set to `true` by `into_solution()` to mark the context
///   as terminal. Subsequent calls to `try_add_worker` /
///   `try_remove_worker` raise `PyRuntimeError` instead of corrupting
///   the just-emitted plan. Read-only accessors stay valid (they just
///   report 0 for the now-empty pending maps), and `into_solution()`
///   itself remains idempotent so callers can re-extract the same
///   empty `Solution` shape if needed for tests / introspection.
/// * `worker_ages` - Per-worker age (in autoscale cycles) for every worker
///   present in this cycle's planning snapshot. Seeded from the optional
///   `worker_ages` constructor argument (the caller's previous-cycle
///   snapshot, with each value already incremented for the new cycle);
///   missing seed entries default to age 0. Updated by `try_add_worker`
///   (fresh placements get age 0; reuse paths keep the existing age) and
///   by `try_remove_worker` (only the cancel-pending-add branch drops
///   the entry, since stage-for-removal workers may still be reused
///   later in the same cycle via the FGD reuse path). Callers that need
///   a youngest-first index across stages build it in O(W) by joining
///   this map against the per-stage current-worker maps. Stale ids --
///   workers that were in the seed but are no longer in
///   `state.stages.worker_groups` -- are silently dropped.
#[pyclass]
pub struct AutoscalePlanContext {
    cluster: rds::ClusterResources,
    stages: Vec<ds::ProblemStage>,
    pending_adds: HashMap<String, Vec<ds::ProblemWorkerGroupState>>,
    pending_removes: HashMap<String, Vec<ds::ProblemWorkerGroupState>>,
    current_workers: HashMap<String, HashMap<String, ds::ProblemWorkerGroupState>>,
    current_worker_groups: HashMap<String, HashMap<String, ds::ProblemWorkerGroupState>>,
    reserved_worker_ids: HashSet<String>,
    slots_per_worker_by_stage: Vec<usize>,
    worker_ages: HashMap<String, u64>,
    worker_id_factory: WorkerIdFactory,
    workload_estimate: frag::Workload,
    worker_reuse_fragmentation_equivalent: f32,
    is_drained: bool,
}

#[pymethods]
impl AutoscalePlanContext {
    /// Construct a planning context seeded with the current cluster state.
    ///
    /// The returned object owns a private mutable copy of
    /// `Problem.cluster_resources` with every worker in `ProblemState`
    /// already allocated, so subsequent `try_add_worker` / `try_remove_worker`
    /// calls respect existing placements and fragmentation constraints.
    ///
    /// # Arguments
    /// * `problem` - Static autoscaling input (cluster shape + per-stage
    ///   shape definitions).
    /// * `state` - Runtime snapshot (per-stage current workers, slots,
    ///   finished flag).
    /// * `worker_ages` - Optional previous-cycle worker-age snapshot.
    ///   Each entry maps a worker id to its age in autoscale cycles.
    ///   Workers present in `state.stages.worker_groups` but absent from
    ///   this map default to age 0 (treated as freshly observed).
    ///   Workers present in this map but absent from the new
    ///   `ProblemState` are silently dropped (the worker died between
    ///   cycles). Callers are responsible for incrementing ages by 1 for
    ///   surviving workers between cycles before passing the map back
    ///   in. When omitted, every seeded worker starts at age 0
    ///   (cold-start); callers that do not maintain cross-cycle age
    ///   state pass `None`.
    ///
    /// # Errors
    /// Returns a `PyRuntimeError` if seeding fails because a current worker
    /// cannot be allocated on the cluster (would indicate a corrupted
    /// snapshot).
    #[new]
    #[pyo3(signature = (problem, state, *, worker_ages=None))]
    pub fn from_problem_state(
        problem: &ds::Problem,
        state: &ds::ProblemState,
        worker_ages: Option<HashMap<String, u64>>,
    ) -> PyResult<Self> {
        if problem.stages.len() != state.stages.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "AutoscalePlanContext: stage count mismatch - \
                 problem has {} stages, state has {}",
                problem.stages.len(),
                state.stages.len(),
            )));
        }

        // Initialise per-stage staged-adds/removes and current-worker maps
        // with one entry per pipeline stage so planning methods can index by
        // stage name without a None-check. Stage names must be unique across
        // the problem; duplicates would silently collapse into a single
        // HashMap entry and corrupt subsequent planning.
        let mut pending_adds: HashMap<String, Vec<ds::ProblemWorkerGroupState>> = HashMap::new();
        let mut pending_removes: HashMap<String, Vec<ds::ProblemWorkerGroupState>> = HashMap::new();
        let mut current_workers: HashMap<String, HashMap<String, ds::ProblemWorkerGroupState>> =
            HashMap::new();
        let mut current_worker_groups: HashMap<
            String,
            HashMap<String, ds::ProblemWorkerGroupState>,
        > = HashMap::new();
        let mut reserved_worker_ids: HashSet<String> = HashSet::new();
        let mut seen_stage_names: HashSet<&str> = HashSet::new();
        for stage in &problem.stages {
            if !seen_stage_names.insert(stage.name.as_str()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "AutoscalePlanContext: duplicate stage name {} in problem.stages",
                    stage.name,
                )));
            }
            pending_adds.insert(stage.name.clone(), Vec::new());
            pending_removes.insert(stage.name.clone(), Vec::new());
            current_workers.insert(stage.name.clone(), HashMap::new());
            current_worker_groups.insert(stage.name.clone(), HashMap::new());
        }

        // Clone the cluster shape and seed it with every worker currently
        // allocated to a stage. After this loop the working cluster reflects
        // the live placement, so FGD calls in subsequent iterations find
        // available capacity correctly.
        let mut cluster = problem.cluster_resources.clone();
        for (stage_idx, (stage, stage_state)) in
            problem.stages.iter().zip(state.stages.iter()).enumerate()
        {
            if stage.name != stage_state.stage_name {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "AutoscalePlanContext: stage name mismatch at position {} - \
                     problem stage is {}, state stage is {}",
                    stage_idx, stage.name, stage_state.stage_name,
                )));
            }

            for w in &stage_state.worker_groups {
                if !reserved_worker_ids.insert(w.id.clone()) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "AutoscalePlanContext: duplicate worker id {} in seed for stage {}",
                        w.id, stage.name,
                    )));
                }
                match &stage.worker_shape {
                    rds::WorkerShape::SpmdNodeMultiple(_) => {
                        validate_seed_resources(&cluster, &stage.name, w, None)?;
                        let group = w.to_worker_group(stage.name.clone());
                        cluster.allocate_multiple(&group.allocations).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "AutoscalePlanContext: failed to seed cluster with \
                                 SPMD worker group {} for stage {}: {:?}",
                                w.id, stage.name, e
                            ))
                        })?;
                        current_worker_groups
                            .get_mut(&stage.name)
                            .ok_or_else(|| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "AutoscalePlanContext: current_worker_groups entry \
                                     missing for stage {}",
                                    stage.name
                                ))
                            })?
                            .insert(w.id.clone(), w.clone());
                    }
                    _ => {
                        validate_seed_resources(&cluster, &stage.name, w, Some(1))?;
                        let worker = w.to_worker(stage.name.clone())?;
                        cluster.allocate(&worker.allocation).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "AutoscalePlanContext: failed to seed cluster with \
                                 worker {} for stage {}: {:?}",
                                w.id, stage.name, e
                            ))
                        })?;
                        current_workers
                            .get_mut(&stage.name)
                            .ok_or_else(|| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "AutoscalePlanContext: current_workers entry missing \
                                     for stage {}",
                                    stage.name
                                ))
                            })?
                            .insert(w.id.clone(), w.clone());
                    }
                }
            }
        }

        // Build the workload estimate FGD consumes when ranking candidate
        // placements. Use the legacy autoscaler helper directly so both
        // planning paths share the requested-worker/current-worker weighting
        // contract.
        let workload_estimate = make_workload_from_state(state, problem);

        // Snapshot the stage definitions so `try_add_worker(stage_index)`
        // can recover the stage name and worker shape without holding a
        // reference to the original `Problem`. ProblemStage is `Clone`
        // (small struct; cloning the whole vec is O(S) one-time work).
        let stages = problem.stages.clone();

        // Snapshot per-stage `slots_per_worker` so `into_solution` can
        // produce a `Solution` whose `StageSolution.slots_per_worker`
        // round-trips the input cycle's value. The planner does not
        // change `slots_per_worker`; it is a per-stage configuration
        // property carried through unchanged.
        let slots_per_worker_by_stage: Vec<usize> = state
            .stages
            .iter()
            .map(|stage_state| stage_state.slots_per_worker)
            .collect();

        // Seed the per-worker ages map. Iterate over the seeded workers
        // (`reserved_worker_ids`) rather than the input map so stale ids
        // - workers that were alive in a prior cycle but no longer
        // appear in `state.stages.worker_groups` - are silently dropped.
        // Missing entries in the input map default to age 0 (cold-start
        // semantics: a worker the scheduler has never seen before is
        // treated as freshly observed).
        let seed_ages: HashMap<String, u64> = worker_ages.unwrap_or_default();
        let worker_ages_map: HashMap<String, u64> = reserved_worker_ids
            .iter()
            .map(|id| {
                let age = seed_ages.get(id).copied().unwrap_or(0);
                (id.clone(), age)
            })
            .collect();

        Ok(Self {
            cluster,
            stages,
            pending_adds,
            pending_removes,
            current_workers,
            current_worker_groups,
            reserved_worker_ids,
            slots_per_worker_by_stage,
            worker_ages: worker_ages_map,
            worker_id_factory: WorkerIdFactory::default(),
            workload_estimate,
            worker_reuse_fragmentation_equivalent: DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT,
            is_drained: false,
        })
    }

    /// Try to place a new worker for `stage_index` using Fragmentation
    /// Gradient Descent (FGD) against the working cluster snapshot.
    ///
    /// Three outcomes:
    ///
    ///   * **Fresh allocation** - FGD found a placement that fits in the
    ///     working snapshot. A new `ProblemWorkerGroupState` is built
    ///     (id from `worker_id_factory`, allocation from FGD), the cluster
    ///     is mutated to reflect the allocation, and the placement is
    ///     pushed to `pending_adds[stage_name]`. Returns `Some(state)`.
    ///   * **Reuse** - FGD found that an already-staged-for-removal worker
    ///     is the best candidate (its placement scores at least as well
    ///     as any fresh placement once the
    ///     `worker_reuse_fragmentation_equivalent` bonus is applied). The
    ///     reused worker is popped from `pending_removes[stage_name]`
    ///     (cancelling the pending remove), the cluster is re-allocated
    ///     for that placement, and the original state is returned.
    ///     `pending_adds` is intentionally NOT touched: the worker was
    ///     never structurally removed from the live set, only un-staged.
    ///   * **AllocationError** - FGD could not find any placement that
    ///     fits. Returns `None` with no mutation. Caller is responsible
    ///     for fallback strategy (donor logic, grace counter).
    ///
    /// # Arguments
    /// * `stage_index` - Position of the stage in `Problem.stages`
    ///   (and therefore in `ctx.stages`).
    ///
    /// # Errors
    /// * `PyIndexError` - `stage_index >= num_stages()`. The plan context
    ///   intentionally enforces a bounds check rather than panicking
    ///   inside a Python-visible method.
    /// * `PyRuntimeError` - The cluster allocation step failed despite
    ///   FGD reporting a feasible placement (would indicate a bug in
    ///   FGD or a corrupted snapshot), OR the context has already been
    ///   drained by `into_solution()` (any further staging is a caller
    ///   bug, since the plan for this cycle has already been emitted).
    pub fn try_add_worker(
        &mut self,
        stage_index: usize,
    ) -> PyResult<Option<ds::ProblemWorkerGroupState>> {
        self.ensure_not_drained("try_add_worker")?;
        if stage_index >= self.stages.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "AutoscalePlanContext.try_add_worker: stage_index {} out of range \
                 (num_stages={})",
                stage_index,
                self.stages.len(),
            )));
        }

        // Clone the stage definition up front so the immutable borrow on
        // `self.stages[stage_index]` does not collide with the later
        // mutable borrows of `self.cluster` / `self.pending_adds` /
        // `self.pending_removes`. ProblemStage is `Clone` and small
        // (a name, a usize, a small enum, two Options) so the cost is
        // negligible.
        let stage = self.stages[stage_index].clone();
        let stage_name = stage.name;
        let stage_shape = stage.worker_shape;

        match stage_shape {
            rds::WorkerShape::SpmdNodeMultiple(spmd_shape) => {
                let result =
                    frag::find_best_allocation_for_spmd_node_multiple(&self.cluster, &spmd_shape);
                if result.worker_allocations.is_empty() {
                    return Ok(None);
                }

                if let Some(reused) = self.take_pending_remove_matching_allocations(
                    &stage_name,
                    &result.worker_allocations,
                ) {
                    // Reuse a previously staged-for-removal SPMD group:
                    // try the cluster commit first, and if it fails restore
                    // `pending_removes` so the planner state is identical
                    // to its pre-call shape. Without this restore, a failed
                    // `allocate_multiple` would lose the worker from BOTH
                    // `pending_removes` (consumed above by
                    // `take_pending_remove_matching_allocations`) AND the
                    // cluster (allocate_multiple did not commit), creating
                    // an undetectable leak inside the cycle.
                    let worker_group = reused.to_worker_group(stage_name.clone());
                    match self.cluster.allocate_multiple(&worker_group.allocations) {
                        Ok(()) => {
                            self.current_worker_groups
                                .entry(stage_name)
                                .or_default()
                                .insert(reused.id.clone(), reused.clone());
                            return Ok(Some(reused));
                        }
                        Err(e) => {
                            let err_msg = format!(
                                "AutoscalePlanContext.try_add_worker: SPMD reuse path \
                                 failed to re-allocate cluster for worker group {} on \
                                 stage {}: {:?}",
                                reused.id, stage_name, e
                            );
                            self.pending_removes
                                .entry(stage_name)
                                .or_default()
                                .push(reused);
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(err_msg));
                        }
                    }
                }

                let new_id = self.make_unique_worker_id();
                let placement = ds::ProblemWorkerGroupState {
                    id: new_id,
                    resources: result.worker_allocations,
                    num_used_slots: 0,
                };
                // Convert to a WorkerGroup so the cluster's
                // allocate_multiple API accepts the allocations slice.
                let worker_group = placement.to_worker_group(stage_name.clone());
                self.cluster
                    .allocate_multiple(&worker_group.allocations)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "AutoscalePlanContext.try_add_worker: SPMD allocation \
                             reported feasible but cluster.allocate_multiple failed \
                             for stage {}: {:?}",
                            stage_name, e
                        ))
                    })?;

                self.pending_adds
                    .entry(stage_name)
                    .or_default()
                    .push(placement.clone());
                self.current_worker_groups
                    .entry(worker_group.stage_name)
                    .or_default()
                    .insert(placement.id.clone(), placement.clone());
                // Fresh placement: age 0 because the worker is brand-new
                // this cycle. The reuse path above does NOT touch
                // worker_ages, so the original seeded age is preserved.
                self.worker_ages.insert(placement.id.clone(), 0);
                Ok(Some(placement))
            }
            _ => {
                // Build the reuse map FGD expects: stage's currently
                // staged-for-removal workers indexed by id. Conversion is
                // O(R) where R = pending removes for this stage; small
                // by construction (one removed worker per scale-down
                // event).
                let reuse_map: HashMap<String, rds::Worker> =
                    match self.pending_removes.get(&stage_name) {
                        Some(list) => list
                            .iter()
                            .map(|p| p.to_worker(stage_name.clone()).map(|w| (p.id.clone(), w)))
                            .collect::<PyResult<HashMap<_, _>>>()?,
                        None => HashMap::new(),
                    };

                let allocation = frag::find_best_allocation_using_fragmentation_gradient_descent(
                    &self.cluster,
                    &self.workload_estimate,
                    &stage_shape,
                    Some(&reuse_map),
                    self.worker_reuse_fragmentation_equivalent,
                );

                if !allocation.did_allocate {
                    return Ok(None);
                }

                if let Some(reused_id) = allocation.reused_worker_id {
                    // FGD selected an already-staged-for-removal placement.
                    // Cancel that pending remove by popping it out of the
                    // list, re-allocate the cluster for the placement, and
                    // return the original state. Do NOT push to
                    // `pending_adds`: the worker was never structurally
                    // removed from the live set, just un-staged.
                    //
                    // Transactional ordering: we consume the pending-remove
                    // entry, then attempt the cluster commit. If the commit
                    // fails we restore `pending_removes` to its pre-call
                    // shape so the planning state stays consistent. Without
                    // the restore, a failed `cluster.allocate` would lose
                    // the worker from BOTH `pending_removes` (consumed) AND
                    // the cluster (allocate did not commit).
                    let reused =
                        {
                            let removes =
                                self.pending_removes.get_mut(&stage_name).ok_or_else(|| {
                                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                                        "AutoscalePlanContext.try_add_worker: pending_removes \
                                     entry missing for stage {}",
                                        stage_name
                                    ))
                                })?;
                            let pos = removes.iter().position(|p| p.id == reused_id).ok_or_else(
                                || {
                                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                                        "AutoscalePlanContext.try_add_worker: reused worker {} \
                                 reported by FGD is not in pending_removes for stage {}",
                                        reused_id, stage_name
                                    ))
                                },
                            )?;
                            removes.remove(pos)
                        };
                    let worker = reused.to_worker(stage_name.clone())?;
                    match self.cluster.allocate(&worker.allocation) {
                        Ok(()) => {
                            self.current_workers
                                .entry(stage_name)
                                .or_default()
                                .insert(reused.id.clone(), reused.clone());
                            Ok(Some(reused))
                        }
                        Err(e) => {
                            let err_msg = format!(
                                "AutoscalePlanContext.try_add_worker: reuse path \
                                 failed to re-allocate cluster for worker {} on \
                                 stage {}: {:?}",
                                reused_id, stage_name, e
                            );
                            self.pending_removes
                                .entry(stage_name)
                                .or_default()
                                .push(reused);
                            Err(pyo3::exceptions::PyRuntimeError::new_err(err_msg))
                        }
                    }
                } else if let Some(allocation_resources) = allocation.resources {
                    // Fresh placement. Build a new ProblemWorkerGroupState
                    // with a freshly-minted id and the FGD-chosen
                    // allocation.
                    let new_id = self.make_unique_worker_id();
                    let placement = ds::ProblemWorkerGroupState {
                        id: new_id,
                        resources: vec![allocation_resources.clone()],
                        num_used_slots: 0,
                    };
                    self.cluster.allocate(&allocation_resources).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "AutoscalePlanContext.try_add_worker: fresh path \
                             failed to allocate cluster for stage {}: {:?}",
                            stage_name, e
                        ))
                    })?;
                    self.pending_adds
                        .entry(stage_name.clone())
                        .or_default()
                        .push(placement.clone());
                    self.current_workers
                        .entry(stage_name)
                        .or_default()
                        .insert(placement.id.clone(), placement.clone());
                    // Fresh placement: age 0 because the worker is brand-
                    // new this cycle. The reuse path above does NOT touch
                    // worker_ages, so the original seeded age is
                    // preserved when the FGD reused-worker branch fires.
                    self.worker_ages.insert(placement.id.clone(), 0);
                    Ok(Some(placement))
                } else {
                    // FGD says did_allocate=true but neither a reused id
                    // nor resources came back. Defensive: surface as a
                    // RuntimeError rather than silently returning None.
                    Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "AutoscalePlanContext.try_add_worker: FGD reported \
                         did_allocate=true but produced neither reused_worker_id \
                         nor resources for stage {}",
                        stage_name
                    )))
                }
            }
        }
    }

    /// Number of workers staged for addition for `stage_index` so far
    /// this cycle. Returns 0 for stages that have not been touched.
    ///
    /// After `into_solution()` drains the per-stage pending lists,
    /// this method reports 0 for every stage regardless of how many
    /// adds were staged earlier in the cycle. Use it as a planning-
    /// time invariant guard rather than a post-drain audit.
    ///
    /// # Errors
    /// * `PyIndexError` - `stage_index >= num_stages()`.
    pub fn pending_add_count(&self, stage_index: usize) -> PyResult<usize> {
        if stage_index >= self.stages.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "AutoscalePlanContext.pending_add_count: stage_index {} out of \
                 range (num_stages={})",
                stage_index,
                self.stages.len(),
            )));
        }
        Ok(self
            .pending_adds
            .get(&self.stages[stage_index].name)
            .map_or(0, |v| v.len()))
    }

    /// Number of workers staged for removal for `stage_index` so far
    /// this cycle. Returns 0 for stages that have not been touched.
    ///
    /// After `into_solution()` drains the per-stage pending lists,
    /// this method reports 0 for every stage regardless of how many
    /// removes were staged earlier in the cycle. Use it as a planning-
    /// time invariant guard rather than a post-drain audit.
    ///
    /// # Errors
    /// * `PyIndexError` - `stage_index >= num_stages()`.
    pub fn pending_remove_count(&self, stage_index: usize) -> PyResult<usize> {
        if stage_index >= self.stages.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "AutoscalePlanContext.pending_remove_count: stage_index {} out \
                 of range (num_stages={})",
                stage_index,
                self.stages.len(),
            )));
        }
        Ok(self
            .pending_removes
            .get(&self.stages[stage_index].name)
            .map_or(0, |v| v.len()))
    }

    /// Remove a specific worker (by id) from the working snapshot, freeing
    /// its resources for reuse by a subsequent `try_add_worker` call. Returns
    /// True on success, False if the worker was not found in this stage's
    /// current set. If the worker was freshly added earlier in the same
    /// cycle, the pending add is cancelled instead of staging a delete for a
    /// worker that was not live at cycle start.
    ///
    /// # Errors
    /// * `PyIndexError` - `stage_index >= num_stages()`.
    /// * `PyRuntimeError` - Releasing resources from the working cluster
    ///   failed, which indicates a corrupted planning snapshot, OR the
    ///   context has already been drained by `into_solution()` (any
    ///   further staging is a caller bug).
    pub fn try_remove_worker(&mut self, stage_index: usize, worker_id: &str) -> PyResult<bool> {
        self.ensure_not_drained("try_remove_worker")?;
        if stage_index >= self.stages.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "AutoscalePlanContext.try_remove_worker: stage_index {} out of range \
                 (num_stages={})",
                stage_index,
                self.stages.len(),
            )));
        }

        let stage = self.stages[stage_index].clone();
        let stage_name = stage.name;

        match stage.worker_shape {
            rds::WorkerShape::SpmdNodeMultiple(_) => {
                let groups = self
                    .current_worker_groups
                    .get_mut(&stage_name)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "AutoscalePlanContext.try_remove_worker: current_worker_groups \
                         entry missing for stage {}",
                            stage_name
                        ))
                    })?;
                let Some(removed) = groups.remove(worker_id) else {
                    return Ok(false);
                };

                self.cluster
                    .release_allocations(&removed.resources)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "AutoscalePlanContext.try_remove_worker: failed to release SPMD \
                         worker group {} for stage {}: {:?}",
                            worker_id, stage_name, e
                        ))
                    })?;
                if self.cancel_pending_add(&stage_name, worker_id) {
                    // Cancel-pending-add: the worker was added earlier in
                    // this same cycle (so its worker_ages entry was age 0)
                    // and is now being un-staged. Drop the age entry --
                    // the worker never structurally existed before this
                    // cycle and will not appear in any next-cycle seed.
                    self.worker_ages.remove(worker_id);
                    return Ok(true);
                }
                // Stage-for-removal: KEEP the age entry. The worker is
                // moving from current_worker_groups to pending_removes,
                // but a subsequent try_add_worker may resurrect it via
                // the FGD reuse path; preserving the age lets any
                // age-aware donor selector compare candidates by their
                // true age rather than collapsing all reused workers
                // to age 0.
                self.pending_removes
                    .entry(stage_name)
                    .or_default()
                    .push(removed);
                Ok(true)
            }
            _ => {
                let workers = self.current_workers.get_mut(&stage_name).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "AutoscalePlanContext.try_remove_worker: current_workers entry \
                         missing for stage {}",
                        stage_name
                    ))
                })?;
                let Some(removed) = workers.remove(worker_id) else {
                    return Ok(false);
                };

                let Some(resource) = removed.resources.first() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "AutoscalePlanContext.try_remove_worker: worker {} for stage {} \
                         has no resource allocation",
                        worker_id, stage_name
                    )));
                };
                self.cluster.release_allocation(resource).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "AutoscalePlanContext.try_remove_worker: failed to release worker \
                         {} for stage {}: {:?}",
                        worker_id, stage_name, e
                    ))
                })?;
                if self.cancel_pending_add(&stage_name, worker_id) {
                    // Cancel-pending-add: the worker was added earlier in
                    // this same cycle (so its worker_ages entry was age 0)
                    // and is now being un-staged. Drop the age entry --
                    // the worker never structurally existed before this
                    // cycle and will not appear in any next-cycle seed.
                    self.worker_ages.remove(worker_id);
                    return Ok(true);
                }
                // Stage-for-removal: KEEP the age entry. The worker is
                // moving from current_workers to pending_removes, but a
                // subsequent try_add_worker may resurrect it via the FGD
                // reuse path; preserving the age lets any age-aware
                // donor selector compare candidates by their true age
                // rather than collapsing all reused workers to age 0.
                self.pending_removes
                    .entry(stage_name)
                    .or_default()
                    .push(removed);
                Ok(true)
            }
        }
    }

    /// Build a `Solution` from the staged plan.
    ///
    /// Drains the per-stage `pending_adds` / `pending_removes` lists into
    /// the matching `StageSolution` entries (one per stage, ordered by
    /// stage position). The per-stage `slots_per_worker` is the value
    /// captured at construction time -- the planner does not change
    /// `slots_per_worker`; it is a per-stage configuration property
    /// preserved unchanged.
    ///
    /// Reused workers (those whose pending-remove was cancelled by a
    /// later `try_add_worker`) appear in NEITHER `new_workers` NOR
    /// `deleted_workers`: they were never structurally removed, only
    /// un-staged. Pure-add stages have empty `deleted_workers`; pure-
    /// remove stages have empty `new_workers`; idle stages (no
    /// planning calls this cycle) have both empty.
    ///
    /// The method takes `&mut self` rather than `self` because PyO3
    /// pyclasses are reference-counted on the Python side and cannot
    /// be consumed across the FFI boundary. After calling
    /// `into_solution()` the context is marked drained: a second call
    /// is idempotent (returns a `Solution` with empty per-stage lists),
    /// but any further `try_add_worker` / `try_remove_worker` raises
    /// `PyRuntimeError`. Callers must build a fresh
    /// `AutoscalePlanContext` for each cycle.
    ///
    pub fn into_solution(&mut self) -> PyResult<ds::Solution> {
        let mut stage_solutions: Vec<ds::StageSolution> = Vec::with_capacity(self.stages.len());
        // Drain semantics: `HashMap::remove(...).unwrap_or_default()`
        // is intentional. After the first pass every stage's entry is
        // gone and the second call returns an empty `Vec` rather than
        // panicking, which preserves `into_solution`'s idempotency
        // contract (mutating entrypoints, not this read-out, are the
        // ones guarded against drain).
        for (stage_idx, stage) in self.stages.iter().enumerate() {
            let slots_per_worker = self.slots_per_worker_by_stage[stage_idx];
            let new_workers = self.pending_adds.remove(&stage.name).unwrap_or_default();
            let deleted_workers = self.pending_removes.remove(&stage.name).unwrap_or_default();
            stage_solutions.push(ds::StageSolution {
                slots_per_worker,
                new_workers,
                deleted_workers,
            });
        }
        self.is_drained = true;
        Ok(ds::Solution {
            stages: stage_solutions,
        })
    }

    /// Read-only number of stages this context is tracking.
    ///
    /// Useful for invariant checks (callers can confirm
    /// `len(stage_solutions) == ctx.num_stages()`) and for tests that need
    /// to verify the seeding round-tripped the input shape correctly.
    /// Backed by `stages.len()` (the authoritative ordered list); the
    /// per-stage hashmaps `pending_adds` / `pending_removes` always have
    /// the same cardinality by construction.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Snapshot of the per-worker age map.
    ///
    /// Keys are every worker id present in this cycle's planning
    /// snapshot (initial seed plus mid-cycle additions); values are
    /// each worker's age in autoscale cycles (0 for workers placed
    /// fresh this cycle, positive integers for workers carried over
    /// from previous cycles). Workers staged for removal are still
    /// included -- they may be reused via the FGD reuse path before
    /// `into_solution` runs -- so callers building an age-aware index
    /// must intersect this map with the per-stage current-worker maps
    /// to filter out scheduled-for-removal entries. After
    /// `into_solution` drains `pending_removes`, callers should
    /// further filter against `solution.deleted_workers` before
    /// persisting the map for the next cycle.
    ///
    /// Returns a clone (`HashMap` is cheaply clonable for the small
    /// W << 100k cardinalities we see in practice). Safe to call
    /// after `into_solution()` drains the plan; the read accessors
    /// deliberately bypass the drained-state guard so the caller can
    /// persist the post-cycle age map for the next cycle.
    pub fn worker_ages(&self) -> HashMap<String, u64> {
        self.worker_ages.clone()
    }

    /// Age of a single worker (`None` if not present in the planning
    /// snapshot).
    ///
    /// Cheap O(1) lookup against the same map exposed in bulk by
    /// `worker_ages`. Use this when you have a specific worker id and
    /// want its age without cloning the whole map. Safe to call after
    /// `into_solution()` for the same reason as `worker_ages`.
    pub fn worker_age(&self, worker_id: &str) -> Option<u64> {
        self.worker_ages.get(worker_id).copied()
    }

    /// Live worker ids per stage in the current planning snapshot.
    ///
    /// Returns a vector indexed by stage position (matching the order
    /// of `stages` in the constructor); each entry is the list of
    /// worker ids currently held by that stage in the planner's
    /// working snapshot. Includes both non-SPMD workers (from
    /// `current_workers`) and SPMD groups (from
    /// `current_worker_groups`); reflects every `try_add_worker`
    /// success and every `try_remove_worker` success applied so far
    /// in this cycle.
    ///
    /// Workers staged for removal are NOT present here -- they have
    /// been moved to `pending_removes`. Workers staged as fresh
    /// adds ARE present (the planner inserts them into
    /// `current_workers` / `current_worker_groups` on success).
    /// Per-stage ids are sorted lexicographically so the output is
    /// deterministic across calls; ordering aids reproducibility in
    /// callers that drive donor-style selection.
    ///
    /// Safe to call after `into_solution()` for the same reason as
    /// `worker_ages`: read accessors deliberately bypass the
    /// drained-state guard. The returned vector clones the live
    /// state; mutating it does not affect the planner.
    pub fn worker_ids_by_stage(&self) -> Vec<Vec<String>> {
        self.stages
            .iter()
            .map(|stage| {
                let mut ids: Vec<String> = Vec::new();
                if let Some(non_spmd) = self.current_workers.get(&stage.name) {
                    ids.extend(non_spmd.keys().cloned());
                }
                if let Some(spmd) = self.current_worker_groups.get(&stage.name) {
                    ids.extend(spmd.keys().cloned());
                }
                ids.sort();
                ids
            })
            .collect()
    }
}

impl AutoscalePlanContext {
    fn ensure_not_drained(&self, op: &str) -> PyResult<()> {
        if self.is_drained {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "AutoscalePlanContext.{}: context is drained (into_solution() \
                 was already called this cycle); build a fresh \
                 AutoscalePlanContext from a new ProblemState before \
                 staging more deltas",
                op,
            )));
        }
        Ok(())
    }

    fn make_unique_worker_id(&mut self) -> String {
        loop {
            let id = self.worker_id_factory.make_new_id();
            if self.reserved_worker_ids.insert(id.clone()) {
                return id;
            }
        }
    }

    fn cancel_pending_add(&mut self, stage_name: &str, worker_id: &str) -> bool {
        let Some(adds) = self.pending_adds.get_mut(stage_name) else {
            return false;
        };
        let Some(pos) = adds.iter().position(|p| p.id == worker_id) else {
            return false;
        };
        adds.remove(pos);
        true
    }

    fn take_pending_remove_matching_allocations(
        &mut self,
        stage_name: &str,
        allocations: &[rds::WorkerResources],
    ) -> Option<ds::ProblemWorkerGroupState> {
        let removes = self.pending_removes.get_mut(stage_name)?;
        let pos = removes
            .iter()
            .position(|p| allocation_sets_match(&p.resources, allocations))?;
        Some(removes.remove(pos))
    }
}

/// Validate current-worker resources before calling lower-level allocation
/// helpers that assume well-formed snapshots and use `unwrap()`.
fn validate_seed_resources(
    cluster: &rds::ClusterResources,
    stage_name: &str,
    worker: &ds::ProblemWorkerGroupState,
    expected_count: Option<usize>,
) -> PyResult<()> {
    if let Some(expected) = expected_count {
        if worker.resources.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "AutoscalePlanContext: worker {} for stage {} must have \
                 exactly {} resource allocation(s), got {}",
                worker.id,
                stage_name,
                expected,
                worker.resources.len(),
            )));
        }
    } else if worker.resources.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "AutoscalePlanContext: worker {} for stage {} must have at \
             least one resource allocation",
            worker.id, stage_name,
        )));
    }

    for resource in &worker.resources {
        let Some(node) = cluster.nodes.get(&resource.node) else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "AutoscalePlanContext: worker {} for stage {} references \
                 unknown node {}",
                worker.id, stage_name, resource.node,
            )));
        };

        for gpu in &resource.gpus {
            if gpu.offset >= node.gpus.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "AutoscalePlanContext: worker {} for stage {} references \
                     GPU offset {} on node {}, but the node has {} GPU(s)",
                    worker.id,
                    stage_name,
                    gpu.offset,
                    resource.node,
                    node.gpus.len(),
                )));
            }
        }
    }

    Ok(())
}

/// Order-agnostic multiset equality on per-worker resource allocations.
///
/// Returns `true` iff `left` and `right` have the same length and the
/// same multiset of `WorkerResources` entries (each entry in `left` is
/// matched against a unique, distinct entry in `right`). Element order
/// is irrelevant; duplicate entries are honoured (each occurrence on
/// the left consumes a distinct occurrence on the right).
///
/// # Complexity
///
/// O(L * R) with L = `left.len()` and R = `right.len()`. The function
/// is only called from the SPMD reuse detection path
/// (`take_pending_remove_matching_allocations`) where each input is the
/// allocation list of a single SPMD worker group. SPMD group cardinality
/// is bounded by `SpmdNodeMultiple.num_workers_per_node *
/// num_nodes_per_worker`, which is typically <= 64 in production
/// pipelines (single-node multi-GPU groups) and never exceeds the
/// physical-GPU-count of the cluster. The quadratic factor is therefore
/// bounded by a small constant in practice; sorting-based comparison
/// would require deriving `Ord` on `WorkerResources` and its nested
/// allocation types, which has wider blast radius than warranted.
fn allocation_sets_match(left: &[rds::WorkerResources], right: &[rds::WorkerResources]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    let mut matched = vec![false; right.len()];
    for left_resource in left {
        let Some(pos) = right
            .iter()
            .enumerate()
            .position(|(idx, right_resource)| !matched[idx] && right_resource == left_resource)
        else {
            return false;
        };
        matched[pos] = true;
    }
    true
}

/// Module initialisation: register `AutoscalePlanContext` as a Python class
/// under the scheduling submodule.
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::utils::module_builders::ImportablePyModuleBuilder;
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<AutoscalePlanContext>()?
        .finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_empty_cluster(num_nodes: usize, cpus_per_node: f32) -> rds::ClusterResources {
        let mut nodes: HashMap<String, rds::NodeResources> = HashMap::new();
        for i in 0..num_nodes {
            nodes.insert(
                format!("node{i}"),
                rds::NodeResources {
                    used_cpus: rds::FixedUtil::ZERO,
                    total_cpus: rds::FixedUtil::from_num(cpus_per_node),
                    gpus: Vec::new(),
                    name: format!("node{i}").into(),
                },
            );
        }
        rds::ClusterResources { nodes }
    }

    fn make_cpu_stage(name: &str) -> ds::ProblemStage {
        ds::ProblemStage {
            name: name.to_string(),
            stage_batch_size: 1,
            worker_shape: rds::WorkerShape::CpuOnly(rds::CpuOnly {
                num_cpus: rds::FixedUtil::ONE,
            }),
            requested_num_workers: None,
            over_provision_factor: None,
        }
    }

    /// Helper: empty stage state for `name` with no current workers.
    fn make_empty_stage_state(name: &str) -> ds::ProblemStageState {
        ds::ProblemStageState {
            stage_name: name.to_string(),
            worker_groups: Vec::new(),
            slots_per_worker: 1,
            is_finished: false,
            ..Default::default()
        }
    }

    /// Helper: stage state with `num_workers` CPU workers all placed on
    /// the specified node, each consuming one CPU. Used to verify that
    /// `from_problem_state` actually drains capacity from the seeded
    /// cluster snapshot when current workers exist.
    fn make_cpu_stage_state_with_workers(
        name: &str,
        num_workers: usize,
        node_id: &str,
    ) -> ds::ProblemStageState {
        let worker_groups = (0..num_workers)
            .map(|i| ds::ProblemWorkerGroupState {
                id: format!("{name}_worker_{i}"),
                resources: vec![rds::WorkerResources {
                    node: node_id.to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: Vec::new(),
                }],
                num_used_slots: 0,
            })
            .collect();
        ds::ProblemStageState {
            stage_name: name.to_string(),
            worker_groups,
            slots_per_worker: 1,
            is_finished: false,
            ..Default::default()
        }
    }

    /// Helper: SPMD multi-node stage requiring 2 GPU actors, each on
    /// its own node, 1 CPU per actor. Exercises the `allocate_multiple`
    /// branch in `from_problem_state` when seeded with a current group.
    fn make_spmd_stage(name: &str) -> ds::ProblemStage {
        ds::ProblemStage {
            name: name.to_string(),
            stage_batch_size: 1,
            worker_shape: rds::WorkerShape::SpmdNodeMultiple(rds::SpmdNodeMultiple {
                num_gpu_actors_in_group: 2,
                num_cpus_per_actor: rds::FixedUtil::ONE,
                num_gpus_in_node: 1,
            }),
            requested_num_workers: None,
            over_provision_factor: None,
        }
    }

    /// Helper: cluster with `num_nodes` nodes, each owning `cpus_per_node`
    /// CPUs and one whole GPU. Used by SPMD / GPU seeding tests.
    fn make_gpu_cluster(num_nodes: usize, cpus_per_node: f32) -> rds::ClusterResources {
        let mut nodes: HashMap<String, rds::NodeResources> = HashMap::new();
        for i in 0..num_nodes {
            nodes.insert(
                format!("node{i}"),
                rds::NodeResources {
                    used_cpus: rds::FixedUtil::ZERO,
                    total_cpus: rds::FixedUtil::from_num(cpus_per_node),
                    gpus: vec![rds::GpuResources {
                        index: i as u8,
                        uuid_: uuid::Uuid::new_v4(),
                        used_fraction: rds::FixedUtil::ZERO,
                    }],
                    name: format!("node{i}").into(),
                },
            );
        }
        rds::ClusterResources { nodes }
    }

    fn assert_workload_matches_legacy_autoscaler(
        actual: &frag::Workload,
        problem: &ds::Problem,
        state: &ds::ProblemState,
    ) {
        let legacy = super::super::autoscaling_algorithms::make_workload_from_state(state, problem);

        assert_eq!(
            actual.stages.len(),
            legacy.stages.len(),
            "plan context and legacy autoscaler must produce the same number of workload stages",
        );
        for (idx, (actual_stage, legacy_stage)) in
            actual.stages.iter().zip(legacy.stages.iter()).enumerate()
        {
            assert!(
                (actual_stage.frequency - legacy_stage.frequency).abs() < 1e-6,
                "workload frequency mismatch at stage {idx}: context={} legacy={}",
                actual_stage.frequency,
                legacy_stage.frequency,
            );
            assert_eq!(
                actual_stage.shape.to_pool(),
                legacy_stage.shape.to_pool(),
                "workload shape pool mismatch at stage {idx}",
            );
        }
    }

    #[test]
    fn from_problem_state_with_zero_stages_succeeds() {
        // Boundary: a Problem with no stages is degenerate but valid;
        // the constructor must not panic on the empty zip iteration and
        // must return an empty workload estimate (no division-by-zero
        // in the uniform-weighting branch).
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: Vec::new(),
        };
        let state = ds::ProblemState { stages: Vec::new() };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None)
            .expect("zero-stage pipeline is a valid empty input");

        assert_eq!(ctx.num_stages(), 0, "no stages tracked");
        assert!(
            ctx.workload_estimate.stages.is_empty(),
            "workload estimate has no entries when there are no stages"
        );
    }

    #[test]
    fn from_problem_state_rejects_when_state_has_more_stages_than_problem() {
        // The opposite mismatch direction from the existing test: an
        // overly-long state slice must also be rejected, otherwise the
        // zip in the seeding loop would silently truncate and lose the
        // tail stages.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
            ],
        };

        let result = AutoscalePlanContext::from_problem_state(&problem, &state, None);
        assert!(
            result.is_err(),
            "stage count mismatch in the state>problem direction must error"
        );
    }

    #[test]
    fn from_problem_state_with_three_stages_keys_pending_maps_for_every_stage() {
        // Multi-stage success path: every stage name from the input
        // must appear as a key in both pending maps, with empty Vec
        // values. Catches off-by-one errors in the keying loop.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(2, 4.0),
            stages: vec![
                make_cpu_stage("stage_a"),
                make_cpu_stage("stage_b"),
                make_cpu_stage("stage_c"),
            ],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
                make_empty_stage_state("stage_c"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert_eq!(ctx.num_stages(), 3);
        for name in ["stage_a", "stage_b", "stage_c"] {
            assert!(
                ctx.pending_adds.contains_key(name),
                "pending_adds keyed for {name}"
            );
            assert!(
                ctx.pending_adds[name].is_empty(),
                "pending_adds[{name}] starts empty"
            );
            assert!(
                ctx.pending_removes.contains_key(name),
                "pending_removes keyed for {name}"
            );
            assert!(
                ctx.pending_removes[name].is_empty(),
                "pending_removes[{name}] starts empty"
            );
        }
    }

    #[test]
    fn from_problem_state_seeds_existing_cpu_worker_consuming_capacity() {
        // Verifies the seeding loop actually calls `cluster.allocate(..)`
        // for every current worker. We seed two 1-cpu workers on a
        // 4-cpu node and assert the working cluster reflects 2 used cpus.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 2, "node0")],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let used = ctx.cluster.nodes["node0"].used_cpus.to_num::<f32>();
        assert!(
            (used - 2.0).abs() < 1e-6,
            "two 1-cpu workers must consume 2 cpus on node0 (got {used})"
        );
    }

    #[test]
    fn from_problem_state_returns_runtime_error_when_workers_overflow_cluster() {
        // The seeding loop must surface allocation failures as a
        // PyRuntimeError instead of silently dropping workers - the
        // saturation snapshot would otherwise drift out of sync with
        // the live placement.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0), // only 1 cpu
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 2, "node0")],
        };

        let result = AutoscalePlanContext::from_problem_state(&problem, &state, None);
        assert!(
            result.is_err(),
            "seeding two cpu workers on a 1-cpu cluster must error"
        );
    }

    #[test]
    fn from_problem_state_seeds_spmd_multi_node_worker_via_allocate_multiple() {
        // Exercises the `WorkerShape::SpmdNodeMultiple` branch in the
        // seeding loop, which goes through `allocate_multiple` instead
        // of `allocate`. Without this test that branch would be dead
        // from a unit-test perspective.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 2.0),
            stages: vec![make_spmd_stage("stage_spmd")],
        };
        // One SPMD group occupying both nodes (1 cpu + 1 full gpu each).
        let group = ds::ProblemWorkerGroupState {
            id: "spmd_group_0".to_string(),
            resources: vec![
                rds::WorkerResources {
                    node: "node0".to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::ONE,
                    }],
                },
                rds::WorkerResources {
                    node: "node1".to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::ONE,
                    }],
                },
            ],
            num_used_slots: 0,
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_spmd".to_string(),
                worker_groups: vec![group],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None)
            .expect("seeding an SPMD group must go through allocate_multiple cleanly");

        // Both nodes contributed one cpu and one whole gpu fraction.
        for node_id in ["node0", "node1"] {
            let used_cpus = ctx.cluster.nodes[node_id].used_cpus.to_num::<f32>();
            assert!(
                (used_cpus - 1.0).abs() < 1e-6,
                "{node_id} consumed 1 cpu by the SPMD seed (got {used_cpus})"
            );
            let gpu = &ctx.cluster.nodes[node_id].gpus[0];
            assert_eq!(
                gpu.used_fraction,
                rds::FixedUtil::ONE,
                "{node_id} GPU should be fully reserved by the SPMD seed"
            );
        }
    }

    #[test]
    fn workload_estimate_is_uniform_when_no_requested_or_current_workers() {
        // Cold-start branch: no stage has requested or current workers,
        // so frequencies must default to 1/N to keep FGD progressing.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![
                make_cpu_stage("stage_a"),
                make_cpu_stage("stage_b"),
                make_cpu_stage("stage_c"),
            ],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
                make_empty_stage_state("stage_c"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let expected = 1.0_f32 / 3.0;
        for s in &ctx.workload_estimate.stages {
            assert!(
                (s.frequency - expected).abs() < 1e-6,
                "uniform branch: every stage frequency must equal 1/N (got {})",
                s.frequency
            );
        }
    }

    #[test]
    fn workload_estimate_is_proportional_to_requested_num_workers() {
        // When `requested_num_workers` is set on every stage, frequencies
        // must be the per-stage ratio. Here: stage_a wants 1 worker,
        // stage_b wants 3 -> frequencies are 0.25 and 0.75.
        let mut stage_a = make_cpu_stage("stage_a");
        stage_a.requested_num_workers = Some(1);
        let mut stage_b = make_cpu_stage("stage_b");
        stage_b.requested_num_workers = Some(3);

        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(2, 4.0),
            stages: vec![stage_a, stage_b],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let f_a = ctx.workload_estimate.stages[0].frequency;
        let f_b = ctx.workload_estimate.stages[1].frequency;
        assert!(
            (f_a - 0.25).abs() < 1e-6,
            "stage_a frequency = 1/(1+3) = 0.25 (got {f_a})"
        );
        assert!(
            (f_b - 0.75).abs() < 1e-6,
            "stage_b frequency = 3/(1+3) = 0.75 (got {f_b})"
        );
    }

    #[test]
    fn workload_estimate_falls_back_to_current_workers_when_other_stage_is_empty() {
        // Fallback rule: when `requested_num_workers` is None, the helper
        // falls back to the current worker count from the matched
        // ProblemStageState. Two current workers in stage_a vs none in
        // stage_b -> stage_a gets all the weight (1.0 / 0.0).
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_cpu_stage_state_with_workers("stage_a", 2, "node0"),
                make_empty_stage_state("stage_b"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let f_a = ctx.workload_estimate.stages[0].frequency;
        let f_b = ctx.workload_estimate.stages[1].frequency;
        assert!(
            (f_a - 1.0).abs() < 1e-6,
            "stage_a frequency derives from 2 current workers (got {f_a})"
        );
        assert!(
            f_b.abs() < 1e-6,
            "stage_b frequency is zero with no current and no requested workers (got {f_b})"
        );
    }

    #[test]
    fn workload_estimate_falls_back_to_current_workers_proportionally() {
        // Fallback rule (proportional case): both stages have current
        // workers but no `requested_num_workers`. Frequencies must be
        // proportional to the current worker count.
        // Two current workers in stage_a, three in stage_b -> 2/(2+3) = 0.4
        // and 3/(2+3) = 0.6, locking down both the fallback and the
        // normaliser used by the shared workload helper.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 8.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_cpu_stage_state_with_workers("stage_a", 2, "node0"),
                make_cpu_stage_state_with_workers("stage_b", 3, "node0"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let f_a = ctx.workload_estimate.stages[0].frequency;
        let f_b = ctx.workload_estimate.stages[1].frequency;
        assert!(
            (f_a - 0.4).abs() < 1e-6,
            "stage_a frequency = 2/(2+3) = 0.4 (got {f_a})"
        );
        assert!(
            (f_b - 0.6).abs() < 1e-6,
            "stage_b frequency = 3/(2+3) = 0.6 (got {f_b})"
        );
    }

    #[test]
    fn workload_estimate_matches_legacy_autoscaler_for_mixed_requested_and_current_workers() {
        // Parity contract: AutoscalePlanContext must weight stages exactly
        // like run_fragmentation_autoscaler. Manual requests override the
        // observed current-worker count for that stage; stages without a
        // request fall back to their current worker count.
        let mut stage_a = make_cpu_stage("stage_a");
        stage_a.requested_num_workers = Some(4);
        let stage_b = make_cpu_stage("stage_b");
        let mut stage_c = make_cpu_stage("stage_c");
        stage_c.requested_num_workers = Some(0);

        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 10.0),
            stages: vec![stage_a, stage_b, stage_c],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_cpu_stage_state_with_workers("stage_a", 1, "node0"),
                make_cpu_stage_state_with_workers("stage_b", 2, "node0"),
                make_cpu_stage_state_with_workers("stage_c", 3, "node0"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert_workload_matches_legacy_autoscaler(&ctx.workload_estimate, &problem, &state);
        let frequencies: Vec<f32> = ctx
            .workload_estimate
            .stages
            .iter()
            .map(|stage| stage.frequency)
            .collect();
        assert_eq!(frequencies.len(), 3);
        assert!(
            (frequencies[0] - (4.0 / 6.0)).abs() < 1e-6,
            "stage_a uses requested_num_workers=4, not its one current worker",
        );
        assert!(
            (frequencies[1] - (2.0 / 6.0)).abs() < 1e-6,
            "stage_b falls back to its two current workers",
        );
        assert!(
            frequencies[2].abs() < 1e-6,
            "stage_c requested_num_workers=0 intentionally contributes no workload weight",
        );
    }

    #[test]
    fn workload_estimate_matches_legacy_autoscaler_for_cold_start_uniform_weights() {
        // Cold start has no current workers and no manual requests. Both
        // Rust planning paths must use the same uniform fallback so FGD
        // starts from a non-degenerate workload vector.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(3, 4.0),
            stages: vec![
                make_cpu_stage("stage_a"),
                make_cpu_stage("stage_b"),
                make_cpu_stage("stage_c"),
            ],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
                make_empty_stage_state("stage_c"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert_workload_matches_legacy_autoscaler(&ctx.workload_estimate, &problem, &state);
        for stage in &ctx.workload_estimate.stages {
            assert!(
                (stage.frequency - (1.0 / 3.0)).abs() < 1e-6,
                "cold-start workload should be uniform (got {})",
                stage.frequency,
            );
        }
    }

    #[test]
    fn worker_reuse_bonus_is_shared_with_legacy_autoscaler() {
        // The context drives one-stage-at-a-time mutations but delegates
        // placement scoring to the same FGD routine as the legacy
        // autoscaler. Keep the reuse reward shared so a future tuning pass
        // cannot make the two paths prefer different reuse/fresh choices.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 1.0);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert_eq!(
            ctx.worker_reuse_fragmentation_equivalent,
            DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT,
            "plan context and legacy autoscaler must use the same FGD reuse bonus",
        );
        assert!(
            ctx.worker_reuse_fragmentation_equivalent > 0.0,
            "positive reuse bonus is required to prefer cancelling a pending remove over fresh placement",
        );
    }

    #[test]
    fn from_problem_state_rejects_stage_name_mismatch() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("wrong_stage")],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state, None).is_err(),
            "stage-name mismatch must be rejected before seeding"
        );
    }

    #[test]
    fn from_problem_state_rejects_empty_non_spmd_worker_resources() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: vec![ds::ProblemWorkerGroupState {
                    id: "empty".to_string(),
                    resources: Vec::new(),
                    num_used_slots: 0,
                }],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state, None).is_err(),
            "empty non-SPMD worker resources must not panic"
        );
    }

    #[test]
    fn from_problem_state_rejects_multi_allocation_non_spmd_worker() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(2, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: vec![ds::ProblemWorkerGroupState {
                    id: "multi".to_string(),
                    resources: vec![
                        rds::WorkerResources {
                            node: "node0".to_string(),
                            cpus: rds::FixedUtil::ONE,
                            gpus: Vec::new(),
                        },
                        rds::WorkerResources {
                            node: "node1".to_string(),
                            cpus: rds::FixedUtil::ONE,
                            gpus: Vec::new(),
                        },
                    ],
                    num_used_slots: 0,
                }],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state, None).is_err(),
            "multi-allocation non-SPMD worker resources must not panic"
        );
    }

    #[test]
    fn from_problem_state_rejects_unknown_worker_node() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node9")],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state, None).is_err(),
            "unknown node must be rejected before allocation unwraps"
        );
    }

    #[test]
    fn from_problem_state_rejects_invalid_gpu_offset() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: vec![ds::ProblemWorkerGroupState {
                    id: "bad-gpu".to_string(),
                    resources: vec![rds::WorkerResources {
                        node: "node0".to_string(),
                        cpus: rds::FixedUtil::ONE,
                        gpus: vec![rds::GpuAllocation {
                            offset: 0,
                            used_fraction: rds::FixedUtil::ONE,
                        }],
                    }],
                    num_used_slots: 0,
                }],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state, None).is_err(),
            "invalid GPU offset must be rejected before allocation unwraps"
        );
    }

    #[test]
    fn from_problem_state_seeds_empty_cluster() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(2, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: Vec::new(),
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None)
            .expect("from_problem_state should succeed on a valid empty snapshot");

        assert_eq!(
            ctx.num_stages(),
            1,
            "context tracks one stage matching the input"
        );
        assert!(
            ctx.pending_adds.contains_key("stage_a"),
            "pending_adds is keyed by stage name"
        );
        assert!(
            ctx.pending_removes.contains_key("stage_a"),
            "pending_removes is keyed by stage name"
        );
        assert!(
            ctx.pending_adds["stage_a"].is_empty(),
            "no adds staged at construction"
        );
        assert!(
            ctx.pending_removes["stage_a"].is_empty(),
            "no removes staged at construction"
        );
        assert_eq!(
            ctx.workload_estimate.stages.len(),
            1,
            "workload estimate has one entry per stage"
        );
        assert_eq!(
            ctx.worker_reuse_fragmentation_equivalent,
            DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT,
            "default reuse parameter matches the in-tree autoscaler constant"
        );
    }

    #[test]
    fn from_problem_state_rejects_stage_count_mismatch() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: Vec::new(),
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        let result = AutoscalePlanContext::from_problem_state(&problem, &state, None);
        assert!(
            result.is_err(),
            "stage count mismatch must produce a clear error"
        );
    }

    /// Build a one-stage CPU problem + empty state on a cluster with
    /// `num_nodes` nodes of `cpus_per_node` CPUs each. Used as the
    /// baseline for try_add_worker tests; each fresh allocation
    /// consumes one CPU from one node.
    fn one_stage_cpu_problem_and_state(
        num_nodes: usize,
        cpus_per_node: f32,
    ) -> (ds::Problem, ds::ProblemState) {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(num_nodes, cpus_per_node),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_a")],
        };
        (problem, state)
    }

    #[test]
    fn try_add_worker_on_empty_cluster_places_one_cpu_worker() {
        // Smoke test: cluster has 1 CPU free; try_add_worker for the only
        // stage must return Some, allocate the CPU, push a placement to
        // pending_adds, and the placement must reference the only node.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 1.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap();
        let placed = placed.expect("FGD must find a placement on an empty 1-CPU cluster");

        assert_eq!(
            placed.resources.len(),
            1,
            "CPU-only worker must own exactly one resource allocation"
        );
        assert_eq!(
            placed.resources[0].node, "node0",
            "the only feasible placement is the only node"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            1,
            "fresh placement must be pushed to pending_adds[stage_a]"
        );
        assert_eq!(
            ctx.pending_remove_count(0).unwrap(),
            0,
            "no removes were staged"
        );
    }

    #[test]
    fn try_add_worker_returns_none_when_cluster_is_full() {
        // The cluster has exactly enough capacity for one CPU worker.
        // The first call succeeds; the second must return Ok(None) (FGD
        // reports no feasible placement) without mutating anything.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 1.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _first = ctx.try_add_worker(0).unwrap().unwrap();
        let second = ctx.try_add_worker(0).unwrap();

        assert!(
            second.is_none(),
            "second add must fail with Ok(None) when cluster is at capacity"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            1,
            "only the first add was committed to pending_adds"
        );
    }

    #[test]
    fn try_add_worker_consumes_capacity_across_sequential_adds() {
        // 2 nodes x 1 CPU each = 2 worker slots total. Three sequential
        // adds: first two succeed and place on different nodes (FGD
        // spreads across nodes when capacity is symmetric), third
        // returns None.
        let (problem, state) = one_stage_cpu_problem_and_state(2, 1.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed1 = ctx.try_add_worker(0).unwrap().unwrap();
        let placed2 = ctx.try_add_worker(0).unwrap().unwrap();
        let placed3 = ctx.try_add_worker(0).unwrap();

        assert!(placed3.is_none(), "third add must run out of capacity");
        // Each placement must be on a real node and consume one CPU.
        assert!(["node0", "node1"].contains(&placed1.resources[0].node.as_str()));
        assert!(["node0", "node1"].contains(&placed2.resources[0].node.as_str()));
        // FGD spreads across the two symmetric nodes, so the two
        // placements must NOT collide on the same node.
        assert_ne!(
            placed1.resources[0].node, placed2.resources[0].node,
            "FGD must spread CPU workers across symmetric nodes \
             (fragmentation gradient descent prefers an unallocated \
             node over a partially-allocated one)"
        );
        assert_eq!(ctx.pending_add_count(0).unwrap(), 2);
    }

    #[test]
    fn try_add_worker_assigns_unique_worker_ids() {
        // The context reserves every minted id; sequential placements
        // must therefore have distinct ids.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 4.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let p1 = ctx.try_add_worker(0).unwrap().unwrap();
        let p2 = ctx.try_add_worker(0).unwrap().unwrap();
        let p3 = ctx.try_add_worker(0).unwrap().unwrap();

        assert_ne!(p1.id, p2.id);
        assert_ne!(p2.id, p3.id);
        assert_ne!(p1.id, p3.id);
    }

    #[test]
    fn try_add_worker_skips_seeded_numeric_worker_ids() {
        // A fresh context starts its local WorkerIdFactory at "0", but
        // ProblemState may already contain workers from prior cycles
        // with numeric ids. Fresh adds must skip those ids rather than
        // overwriting the seeded current-worker map entry.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 3.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: vec![
                    ds::ProblemWorkerGroupState {
                        id: "0".to_string(),
                        resources: vec![rds::WorkerResources {
                            node: "node0".to_string(),
                            cpus: rds::FixedUtil::ONE,
                            gpus: Vec::new(),
                        }],
                        num_used_slots: 0,
                    },
                    ds::ProblemWorkerGroupState {
                        id: "1".to_string(),
                        resources: vec![rds::WorkerResources {
                            node: "node0".to_string(),
                            cpus: rds::FixedUtil::ONE,
                            gpus: Vec::new(),
                        }],
                        num_used_slots: 0,
                    },
                ],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap().unwrap();

        assert_eq!(
            placed.id, "2",
            "fresh id must advance past seeded ids 0 and 1"
        );
        assert_eq!(
            ctx.current_workers["stage_a"].len(),
            3,
            "fresh add must not overwrite a seeded worker with the same id"
        );
    }

    #[test]
    fn try_add_worker_rejects_out_of_bounds_stage_index() {
        // PyIndexError is the documented failure mode for an out-of-range
        // stage_index. The check must fire before any FGD call so a buggy
        // caller cannot accidentally mutate `pending_adds` for a stage
        // that does not exist.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 1.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        // Only stage 0 exists; stage 1 must be rejected.
        let result = ctx.try_add_worker(1);
        assert!(
            result.is_err(),
            "out-of-range stage_index must return PyIndexError"
        );
    }

    #[test]
    fn try_add_worker_respects_seeded_cluster_capacity() {
        // The cluster has 2 CPUs total but ProblemState already has 1
        // current worker pre-seeded onto node0. The remaining capacity
        // is 1 CPU; only one fresh add should succeed before the second
        // returns None.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let p1 = ctx.try_add_worker(0).unwrap();
        let p2 = ctx.try_add_worker(0).unwrap();

        assert!(p1.is_some(), "1 CPU of capacity remains after seeding");
        assert!(
            p2.is_none(),
            "second add must fail; pre-seeded worker already consumed 1 of 2 CPUs"
        );
    }

    #[test]
    fn try_add_worker_places_spmd_multi_node_group_via_allocate_multiple() {
        // SPMD shape needs 2 GPU actors, one per node. With 2 GPU nodes
        // the placement must succeed and the returned state must own
        // exactly two resource allocations (one per node), each with a
        // whole GPU consumed.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 4.0),
            stages: vec![make_spmd_stage("stage_spmd")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_spmd")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap();
        let placed = placed.expect("2-actor SPMD group fits on a 2-node GPU cluster");

        assert_eq!(
            placed.resources.len(),
            2,
            "SPMD group must own one resource allocation per actor node"
        );
        let nodes: std::collections::HashSet<_> =
            placed.resources.iter().map(|r| r.node.as_str()).collect();
        assert_eq!(
            nodes.len(),
            2,
            "the two allocations must land on distinct nodes"
        );
    }

    #[test]
    fn try_add_worker_returns_none_for_spmd_when_node_count_insufficient() {
        // Only 1 GPU node available but SPMD needs 2. FGD must report
        // no allocation possible.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(1, 4.0),
            stages: vec![make_spmd_stage("stage_spmd")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_spmd")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap();
        assert!(
            placed.is_none(),
            "SPMD group requiring 2 nodes cannot fit on 1-node cluster"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            0,
            "no placement was committed"
        );
    }

    #[test]
    fn try_add_worker_reuses_worker_from_pending_removes() {
        // Pre-seed a single worker onto node0, then manually inject it
        // into pending_removes (simulating the state after a
        // try_remove_worker call). The next try_add_worker call must
        // reuse that exact placement (the
        // FGD reuse bonus dominates over any fresh fragment cost),
        // pop the entry from pending_removes, and NOT push to
        // pending_adds.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_a")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        // Manually stage a removal: a worker that "was removed earlier
        // this cycle" and whose placement is now free in `cluster`.
        let removed_id = "worker_being_removed".to_string();
        let removed_placement = ds::ProblemWorkerGroupState {
            id: removed_id.clone(),
            resources: vec![rds::WorkerResources {
                node: "node0".to_string(),
                cpus: rds::FixedUtil::ONE,
                gpus: Vec::new(),
            }],
            num_used_slots: 0,
        };
        ctx.pending_removes
            .get_mut("stage_a")
            .unwrap()
            .push(removed_placement.clone());

        let placed = ctx.try_add_worker(0).unwrap();
        let placed = placed.expect("FGD must reuse the pending-remove placement");

        assert_eq!(
            placed.id, removed_id,
            "the reused placement must carry the original worker id"
        );
        assert_eq!(
            ctx.pending_remove_count(0).unwrap(),
            0,
            "reuse must pop the entry out of pending_removes (cancel the remove)"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            0,
            "reused workers must NOT be pushed to pending_adds (they were never \
             structurally removed; the staged remove was simply un-staged)"
        );
        // Verify the third invariant of the reuse path: the cluster
        // was re-allocated for the placement. Without this assertion
        // a regression that pops pending_removes but skips
        // `cluster.allocate(...)` would silently double-account the
        // resources.
        let node0_used_cpus = ctx.cluster.nodes.get("node0").unwrap().used_cpus;
        assert_eq!(
            node0_used_cpus,
            rds::FixedUtil::ONE,
            "reuse path must re-allocate the cluster (1 cpu consumed on node0)",
        );
    }

    #[test]
    fn try_add_worker_does_not_mutate_state_when_returning_none() {
        // Failure cases (cluster full) must leave pending_adds /
        // pending_removes / cluster untouched so the caller can fall
        // back to donor logic without rolling back side effects.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 1.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _first = ctx.try_add_worker(0).unwrap().unwrap();
        let pending_adds_before = ctx.pending_add_count(0).unwrap();
        let pending_removes_before = ctx.pending_remove_count(0).unwrap();
        let cluster_used_before = ctx.cluster.nodes.get("node0").unwrap().used_cpus;

        let second = ctx.try_add_worker(0).unwrap();

        assert!(second.is_none());
        assert_eq!(ctx.pending_add_count(0).unwrap(), pending_adds_before);
        assert_eq!(ctx.pending_remove_count(0).unwrap(), pending_removes_before);
        // Cluster snapshot must not have shifted for a failed
        // allocation: the caller depends on this to drive donor logic
        // without rolling back side effects.
        assert_eq!(
            ctx.cluster.nodes.get("node0").unwrap().used_cpus,
            cluster_used_before,
            "failed try_add_worker must leave cluster used_cpus unchanged",
        );
    }

    #[test]
    fn try_add_worker_isolates_pending_adds_per_stage() {
        // Two stages share a cluster. An add on stage_a must not appear
        // in stage_b's pending_adds count and vice versa.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(2, 1.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
            ],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _ = ctx.try_add_worker(0).unwrap().unwrap();

        assert_eq!(ctx.pending_add_count(0).unwrap(), 1);
        assert_eq!(
            ctx.pending_add_count(1).unwrap(),
            0,
            "stage_b must not see stage_a's pending add"
        );
    }

    #[test]
    fn try_remove_worker_stages_seeded_cpu_worker_and_reuse_cancels_remove() {
        // Seed one existing worker into a 1-CPU cluster. Removing it must
        // release that CPU from the working cluster and append the original
        // worker state to pending_removes. A subsequent add for the same
        // shape should reuse that exact worker id and cancel the remove
        // instead of creating a new pending add.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        let worker_id = "stage_a_worker_0";

        assert!(
            ctx.try_add_worker(0).unwrap().is_none(),
            "seeded worker consumes the only CPU before removal"
        );
        assert!(
            ctx.try_remove_worker(0, worker_id).unwrap(),
            "existing worker id must be removable"
        );
        assert_eq!(
            ctx.pending_remove_count(0).unwrap(),
            1,
            "successful remove is staged in pending_removes"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            0,
            "removing an existing worker must not create a pending add"
        );

        let reused = ctx.try_add_worker(0).unwrap().unwrap();

        assert_eq!(
            reused.id, worker_id,
            "FGD should reuse the exact worker staged for removal"
        );
        assert_eq!(
            ctx.pending_remove_count(0).unwrap(),
            0,
            "reuse cancels the pending remove"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            0,
            "reused worker was already live, so no add is staged"
        );
    }

    #[test]
    fn try_remove_worker_returns_false_for_unknown_id_without_mutation() {
        // Unknown worker ids are expected during stale-snapshot handling:
        // callers can attempt a conservative remove and continue when it
        // returns false. The method must not mutate any pending counts.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let removed = ctx.try_remove_worker(0, "missing_worker").unwrap();

        assert!(!removed, "unknown worker id must return false");
        assert_eq!(ctx.pending_remove_count(0).unwrap(), 0);
        assert_eq!(ctx.pending_add_count(0).unwrap(), 0);
    }

    #[test]
    fn try_remove_worker_rejects_out_of_bounds_stage_index() {
        // Mirror try_add_worker / pending count bounds checks so Python
        // callers get IndexError instead of a Rust panic.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 1.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let result = ctx.try_remove_worker(1, "any");

        assert!(
            result.is_err(),
            "out-of-range stage_index must return PyIndexError"
        );
    }

    #[test]
    fn try_remove_worker_second_remove_of_same_id_returns_false() {
        // Once a worker is removed from the current snapshot, attempting
        // to remove it again in the same planning cycle should be a
        // no-op false, not a duplicate pending remove.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());
        assert!(!ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());

        assert_eq!(
            ctx.pending_remove_count(0).unwrap(),
            1,
            "duplicate remove must not duplicate pending_removes"
        );
    }

    #[test]
    fn try_remove_worker_isolates_pending_removes_per_stage() {
        // Two stages each have one live worker. Removing stage_a's worker
        // must not affect stage_b's pending_remove count.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_cpu_stage_state_with_workers("stage_a", 1, "node0"),
                make_cpu_stage_state_with_workers("stage_b", 1, "node0"),
            ],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());

        assert_eq!(ctx.pending_remove_count(0).unwrap(), 1);
        assert_eq!(
            ctx.pending_remove_count(1).unwrap(),
            0,
            "stage_b must not see stage_a's pending remove"
        );
    }

    #[test]
    fn try_remove_worker_cancels_freshly_added_worker() {
        // The plan context treats newly added workers as part of the
        // current working snapshot. Removing one before it becomes live
        // cancels the pending add instead of staging an impossible delete
        // for a worker that did not exist at cycle start.
        let (problem, state) = one_stage_cpu_problem_and_state(1, 2.0);
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        let first = ctx.try_add_worker(0).unwrap().unwrap();
        let second = ctx.try_add_worker(0).unwrap().unwrap();

        assert!(ctx.try_remove_worker(0, &first.id).unwrap());
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            1,
            "removing a fresh add cancels that pending add"
        );
        assert_eq!(
            ctx.pending_remove_count(0).unwrap(),
            0,
            "fresh adds were not live at cycle start, so removal must not stage a delete"
        );
        let replacement = ctx
            .try_add_worker(0)
            .unwrap()
            .expect("cancelling the fresh add frees capacity for a replacement");
        let exhausted = ctx.try_add_worker(0).unwrap();

        assert_ne!(
            replacement.id, first.id,
            "cancelled fresh-add ids are not revived through pending_removes"
        );
        assert!(
            exhausted.is_none(),
            "cluster returns to full capacity after the replacement is added"
        );
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            2,
            "the replacement restores the two net pending adds"
        );
        assert_ne!(
            first.id, second.id,
            "test setup requires two distinct fresh worker ids"
        );
    }

    #[test]
    fn try_remove_worker_stages_spmd_group_and_readd_uses_freed_capacity() {
        // SPMD worker groups carry multiple resource allocations. Removal
        // must release every allocation, not just the first one, so a
        // subsequent SPMD add can fit on the same two-node GPU cluster.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 2.0),
            stages: vec![make_spmd_stage("stage_spmd")],
        };
        let group = ds::ProblemWorkerGroupState {
            id: "spmd_group_0".to_string(),
            resources: vec![
                rds::WorkerResources {
                    node: "node0".to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::ONE,
                    }],
                },
                rds::WorkerResources {
                    node: "node1".to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::ONE,
                    }],
                },
            ],
            num_used_slots: 0,
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_spmd".to_string(),
                worker_groups: vec![group],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert!(
            ctx.try_add_worker(0).unwrap().is_none(),
            "seeded SPMD group consumes both GPU nodes before removal"
        );
        assert!(ctx.try_remove_worker(0, "spmd_group_0").unwrap());
        assert_eq!(ctx.pending_remove_count(0).unwrap(), 1);

        let readded = ctx
            .try_add_worker(0)
            .unwrap()
            .expect("SPMD removal must release all group allocations for a later add");

        assert_eq!(
            readded.id, "spmd_group_0",
            "SPMD add should cancel the matching pending remove instead of minting a new id"
        );
        assert_eq!(ctx.pending_remove_count(0).unwrap(), 0);
        assert_eq!(
            ctx.pending_add_count(0).unwrap(),
            0,
            "reused SPMD group was already live, so no fresh add is staged"
        );
    }

    #[test]
    fn into_solution_for_idle_context_returns_empty_per_stage_lists() {
        // No try_add_worker / try_remove_worker calls -> every stage's
        // new_workers and deleted_workers must be empty, but the
        // Solution must still have one StageSolution per stage with
        // slots_per_worker preserved from the input.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                ds::ProblemStageState {
                    stage_name: "stage_a".to_string(),
                    worker_groups: Vec::new(),
                    slots_per_worker: 4,
                    is_finished: false,
                    ..Default::default()
                },
                ds::ProblemStageState {
                    stage_name: "stage_b".to_string(),
                    worker_groups: Vec::new(),
                    slots_per_worker: 7,
                    is_finished: false,
                    ..Default::default()
                },
            ],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let solution = ctx.into_solution().unwrap();

        assert_eq!(solution.stages.len(), 2);
        assert_eq!(solution.stages[0].slots_per_worker, 4);
        assert_eq!(solution.stages[1].slots_per_worker, 7);
        assert!(solution.stages[0].new_workers.is_empty());
        assert!(solution.stages[0].deleted_workers.is_empty());
        assert!(solution.stages[1].new_workers.is_empty());
        assert!(solution.stages[1].deleted_workers.is_empty());
    }

    #[test]
    fn into_solution_routes_fresh_add_to_new_workers() {
        // try_add_worker on stage 0 -> the placement appears in
        // solution.stages[0].new_workers; stage 1 stays empty.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(2, 1.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
            ],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap().unwrap();
        let solution = ctx.into_solution().unwrap();

        assert_eq!(solution.stages.len(), 2);
        assert_eq!(solution.stages[0].new_workers.len(), 1);
        assert_eq!(solution.stages[0].new_workers[0].id, placed.id);
        assert!(solution.stages[0].deleted_workers.is_empty());
        assert!(solution.stages[1].new_workers.is_empty());
        assert!(solution.stages[1].deleted_workers.is_empty());
    }

    #[test]
    fn into_solution_routes_remove_to_deleted_workers() {
        // Pre-seed a worker, remove it, then call into_solution. The
        // worker appears in deleted_workers; new_workers stays empty.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        // Worker id matches the helper's naming convention.
        let removed = ctx.try_remove_worker(0, "stage_a_worker_0").unwrap();
        assert!(removed);

        let solution = ctx.into_solution().unwrap();

        assert_eq!(solution.stages[0].new_workers.len(), 0);
        assert_eq!(solution.stages[0].deleted_workers.len(), 1);
        assert_eq!(solution.stages[0].deleted_workers[0].id, "stage_a_worker_0");
    }

    #[test]
    fn into_solution_routes_mixed_adds_and_removes_to_matching_stages() {
        // Mixed deltas are the common scheduler output: one stage can
        // shrink while another grows. The Solution vector has no stage
        // names, so routing relies entirely on preserving Problem.stages
        // order and attaching each stage's pending lists to the matching
        // index.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 3.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_cpu_stage_state_with_workers("stage_a", 1, "node0"),
                make_empty_stage_state("stage_b"),
            ],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());
        let added = ctx.try_add_worker(1).unwrap().unwrap();

        let solution = ctx.into_solution().unwrap();

        assert_eq!(solution.stages.len(), 2);
        assert!(solution.stages[0].new_workers.is_empty());
        assert_eq!(solution.stages[0].deleted_workers.len(), 1);
        assert_eq!(solution.stages[0].deleted_workers[0].id, "stage_a_worker_0");
        assert_eq!(solution.stages[1].new_workers.len(), 1);
        assert_eq!(solution.stages[1].new_workers[0].id, added.id);
        assert!(solution.stages[1].deleted_workers.is_empty());
    }

    #[test]
    fn into_solution_omits_reused_worker_from_both_lists() {
        // Pre-seed a worker, remove it (-> pending_removes), then
        // try_add_worker which reuses the placement (-> cancels remove).
        // The Solution must contain neither an add nor a delete for
        // that worker -- it stays in the live set throughout the cycle.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());
        let reused = ctx.try_add_worker(0).unwrap().unwrap();
        assert_eq!(reused.id, "stage_a_worker_0");

        let solution = ctx.into_solution().unwrap();

        assert!(
            solution.stages[0].new_workers.is_empty(),
            "reused worker must NOT appear in new_workers"
        );
        assert!(
            solution.stages[0].deleted_workers.is_empty(),
            "reused worker must NOT appear in deleted_workers \
             (the staged remove was cancelled by the reuse)"
        );
    }

    #[test]
    fn into_solution_preserves_stage_order_from_stages_vec() {
        // Build a 3-stage pipeline; stage_b is the only one that gets
        // an add. solution.stages must preserve the original ordering
        // (stage_a, stage_b, stage_c), and stage_b must own the only
        // non-empty new_workers list.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(3, 1.0),
            stages: vec![
                make_cpu_stage("stage_a"),
                make_cpu_stage("stage_b"),
                make_cpu_stage("stage_c"),
            ],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("stage_a"),
                make_empty_stage_state("stage_b"),
                make_empty_stage_state("stage_c"),
            ],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _ = ctx.try_add_worker(1).unwrap().unwrap();
        let solution = ctx.into_solution().unwrap();

        assert_eq!(solution.stages.len(), 3);
        assert_eq!(solution.stages[0].new_workers.len(), 0);
        assert_eq!(solution.stages[1].new_workers.len(), 1);
        assert_eq!(solution.stages[2].new_workers.len(), 0);
    }

    #[test]
    fn into_solution_drains_pending_lists_so_second_call_is_empty() {
        // into_solution() finalizes and drains staged deltas, so a
        // second call returns a fully-empty Solution. Pinning this
        // prevents a future refactor from accidentally returning the
        // same list twice (which would cause the streaming layer to
        // try to add the same worker twice in two consecutive cycles).
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_a")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        ctx.try_add_worker(0).unwrap().unwrap();

        let first = ctx.into_solution().unwrap();
        let second = ctx.into_solution().unwrap();

        assert_eq!(first.stages[0].new_workers.len(), 1);
        assert_eq!(
            second.stages[0].new_workers.len(),
            0,
            "into_solution drains pending_adds; a second call must produce an empty Solution"
        );
    }

    #[test]
    fn into_solution_drains_pending_counts() {
        // pending_*_count are planning-time accessors. After
        // into_solution drains the staged deltas, both counts must read
        // as zero so callers cannot accidentally audit stale pending
        // lists after finalization.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _ = ctx.try_add_worker(0).unwrap().unwrap();
        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());
        assert_eq!(ctx.pending_add_count(0).unwrap(), 1);
        assert_eq!(ctx.pending_remove_count(0).unwrap(), 1);

        let solution = ctx.into_solution().unwrap();

        assert_eq!(solution.stages[0].new_workers.len(), 1);
        assert_eq!(solution.stages[0].deleted_workers.len(), 1);
        assert_eq!(ctx.pending_add_count(0).unwrap(), 0);
        assert_eq!(ctx.pending_remove_count(0).unwrap(), 0);
    }

    #[test]
    fn try_add_worker_after_into_solution_returns_runtime_error() {
        // Once `into_solution` has emitted the plan for this cycle, any
        // further `try_add_worker` is a programming bug: the staged
        // deltas have already been handed to the caller and mutating
        // the (now-empty) pending maps would silently produce a stale
        // plan on a second `into_solution` call. The drained-state
        // guard must convert this misuse into a loud `PyRuntimeError`.
        // Asserting on `is_err()` (not the formatted message) keeps the
        // test free of the Python GIL initialization that
        // `PyErr::to_string` requires.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![ds::ProblemStageState {
                stage_name: "stage_a".to_string(),
                worker_groups: Vec::new(),
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _ = ctx.into_solution().unwrap();

        assert!(
            ctx.try_add_worker(0).is_err(),
            "try_add_worker on drained context must return an error"
        );
    }

    #[test]
    fn try_remove_worker_after_into_solution_returns_runtime_error() {
        // Same drained-state contract as above, applied to the remove
        // entrypoint. After draining the plan, removing more workers
        // would corrupt the next cycle's accounting; the guard must
        // surface the misuse instead of silently mutating a stale
        // snapshot.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let _ = ctx.into_solution().unwrap();

        assert!(
            ctx.try_remove_worker(0, "stage_a_worker_0").is_err(),
            "try_remove_worker on drained context must return an error"
        );
    }

    #[test]
    fn allocation_sets_match_handles_empty_lists_as_equal() {
        // Two empty allocation lists are trivially equal. This is the
        // degenerate case where a stage has no reservation-bearing SPMD
        // workers; the helper must not require iteration to confirm.
        assert!(allocation_sets_match(&[], &[]));
    }

    #[test]
    fn allocation_sets_match_returns_false_for_mismatched_lengths() {
        // Length mismatch is the cheapest disqualifier and must short-
        // circuit before any element comparison runs.
        let one = vec![rds::WorkerResources {
            node: "node0".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        }];
        assert!(!allocation_sets_match(&one, &[]));
        assert!(!allocation_sets_match(&[], &one));
    }

    #[test]
    fn allocation_sets_match_returns_true_for_same_set_in_different_order() {
        // SPMD reuse depends on this property: FGD may report worker
        // allocations in a different order than the originally-staged
        // pending-remove (different node iteration order, etc.). The
        // helper must treat allocation lists as multisets, not lists.
        let a = rds::WorkerResources {
            node: "node0".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        };
        let b = rds::WorkerResources {
            node: "node1".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        };
        assert!(allocation_sets_match(&[a.clone(), b.clone()], &[b, a]));
    }

    #[test]
    fn allocation_sets_match_honours_duplicates() {
        // Two workers occupying the same node + same fractional CPU is
        // a legitimate SPMD configuration (e.g. two TP ranks on the
        // same physical machine). Duplicate-aware multiset matching
        // must consume distinct positions in `right` for each
        // duplicate in `left`.
        let dup = rds::WorkerResources {
            node: "node0".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        };
        let single = rds::WorkerResources {
            node: "node1".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        };
        // [dup, dup] != [dup, single] because the second `dup` cannot
        // pair with the unique `single` slot.
        assert!(!allocation_sets_match(
            &[dup.clone(), dup.clone()],
            &[dup.clone(), single],
        ));
        // [dup, dup] == [dup, dup] (both slots line up).
        assert!(allocation_sets_match(
            &[dup.clone(), dup.clone()],
            &[dup.clone(), dup]
        ));
    }

    #[test]
    fn allocation_sets_match_returns_false_when_node_differs() {
        // Same CPU count, different node ids: the planner must NOT
        // reuse such a placement because the cluster commit would
        // target a different physical machine. This is a critical
        // correctness boundary for SPMD reuse.
        let on_node0 = rds::WorkerResources {
            node: "node0".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        };
        let on_node1 = rds::WorkerResources {
            node: "node1".to_string(),
            cpus: rds::FixedUtil::ONE,
            gpus: Vec::new(),
        };
        assert!(!allocation_sets_match(&[on_node0], &[on_node1]));
    }

    #[test]
    fn into_solution_for_zero_stage_pipeline_returns_empty_solution() {
        // Boundary: a zero-stage pipeline is a valid degenerate input.
        // The Solution must have no stages and the call must not panic.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: Vec::new(),
        };
        let state = ds::ProblemState { stages: Vec::new() };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let solution = ctx.into_solution().unwrap();

        assert!(solution.stages.is_empty());
    }

    #[test]
    fn worker_ages_default_to_zero_when_no_seed_supplied() {
        // Cold start: when the constructor is called with no
        // `worker_ages` kwarg, every seeded worker reports age 0. This
        // is the entry path on the very first autoscale cycle, before
        // any caller has a previous-cycle snapshot to pass back in.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 2, "node0")],
        };
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let ages = ctx.worker_ages();
        assert_eq!(ages.len(), 2);
        assert_eq!(ages.get("stage_a_worker_0"), Some(&0u64));
        assert_eq!(ages.get("stage_a_worker_1"), Some(&0u64));
    }

    #[test]
    fn worker_ages_round_trip_seed_values_through_constructor() {
        // The caller passes the previous cycle's age map (already
        // incremented for the new cycle) into the constructor, and the
        // context exposes those same values back through
        // `worker_ages()`. This is the cycle-to-cycle persistence
        // contract the donor logic relies on.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 2, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 5);
        seed.insert("stage_a_worker_1".to_string(), 12);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        let ages = ctx.worker_ages();
        assert_eq!(ages.get("stage_a_worker_0"), Some(&5u64));
        assert_eq!(ages.get("stage_a_worker_1"), Some(&12u64));
        // Single-worker accessor must agree with the bulk map.
        assert_eq!(ctx.worker_age("stage_a_worker_1"), Some(12));
    }

    #[test]
    fn worker_ages_drop_stale_seed_ids_not_in_state() {
        // Workers that died between the previous cycle and the new
        // cycle still appear in the caller's saved age map (the caller
        // does not necessarily know which ids survived). The
        // constructor must silently filter them out so the in-context
        // age map mirrors the live worker set.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 3);
        seed.insert("worker_that_died".to_string(), 99);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        let ages = ctx.worker_ages();
        assert_eq!(ages.len(), 1, "stale seed id must be filtered out");
        assert_eq!(ages.get("stage_a_worker_0"), Some(&3u64));
        assert_eq!(ctx.worker_age("worker_that_died"), None);
    }

    #[test]
    fn worker_ages_default_to_zero_for_seed_missing_known_worker() {
        // Mixed case: caller passes a partial seed (some workers were
        // observed in prior cycles, others are brand-new this cycle
        // because they were just created externally between cycles).
        // Missing entries default to 0 -- treat unknown ids as freshly
        // observed rather than rejecting the call.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 2, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 7);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        let ages = ctx.worker_ages();
        assert_eq!(ages.get("stage_a_worker_0"), Some(&7u64));
        assert_eq!(
            ages.get("stage_a_worker_1"),
            Some(&0u64),
            "worker missing from seed defaults to age 0"
        );
    }

    #[test]
    fn try_add_worker_assigns_age_zero_to_fresh_placement() {
        // Fresh placements register age 0 on the non-SPMD code path so
        // any age-aware donor selector treats freshly-placed
        // candidates as the youngest possible.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_a")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap().unwrap();
        let ages = ctx.worker_ages();
        assert_eq!(
            ages.get(&placed.id),
            Some(&0u64),
            "fresh CPU placement must register age 0"
        );
    }

    #[test]
    fn try_add_worker_assigns_age_zero_to_fresh_spmd_placement() {
        // Companion to the non-SPMD test. SPMD groups go through
        // `find_best_allocation_for_spmd_node_multiple` rather than
        // FGD, so they exercise a different fresh-add code path that
        // must also register age 0.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 1.0),
            stages: vec![make_spmd_stage("spmd_stage")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("spmd_stage")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap().unwrap();
        let ages = ctx.worker_ages();
        assert_eq!(
            ages.get(&placed.id),
            Some(&0u64),
            "fresh SPMD placement must register age 0"
        );
    }

    #[test]
    fn try_remove_worker_cancel_pending_spmd_add_drops_age_entry() {
        // Same cancel-pending-add invariant as the CPU path, applied to
        // a fresh SPMD group. SPMD removals go through
        // `current_worker_groups` and `release_allocations`, so pin
        // that they also remove the age entry when the add is retracted
        // before it ever becomes live.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 1.0),
            stages: vec![make_spmd_stage("spmd_stage")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("spmd_stage")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap().unwrap();
        assert_eq!(ctx.worker_age(&placed.id), Some(0));

        assert!(ctx.try_remove_worker(0, &placed.id).unwrap());

        assert_eq!(
            ctx.worker_age(&placed.id),
            None,
            "cancel-pending-add must drop the SPMD worker_ages entry"
        );
    }

    #[test]
    fn worker_ages_sort_oldest_first_returns_correct_order() {
        // Caller-side sorting is the documented pattern (callers sort
        // in Python on the (worker_id, age) pairs the context exposes).
        // Pin that the exposed pairs are sufficient for a deterministic
        // descending sort.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 3.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 3, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 5);
        seed.insert("stage_a_worker_1".to_string(), 0);
        seed.insert("stage_a_worker_2".to_string(), 10);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        let mut entries: Vec<(String, u64)> = ctx.worker_ages().into_iter().collect();
        // Descending sort by age (.1) via std::cmp::Reverse — equivalent to
        // |a, b| b.1.cmp(&a.1) but lint-clean (clippy::unnecessary_sort_by).
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        let ids: Vec<&str> = entries.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["stage_a_worker_2", "stage_a_worker_0", "stage_a_worker_1",],
            "descending sort must place the oldest worker first"
        );
    }

    #[test]
    fn worker_ages_sort_youngest_first_returns_correct_order() {
        // Donor selection picks the youngest eligible donor across
        // stages so that long-running workers are not preferentially
        // evicted; pin that an ascending sort on the exposed pairs
        // gives the right ordering.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 3.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 3, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 5);
        seed.insert("stage_a_worker_1".to_string(), 0);
        seed.insert("stage_a_worker_2".to_string(), 10);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        let mut entries: Vec<(String, u64)> = ctx.worker_ages().into_iter().collect();
        entries.sort_by_key(|(_, age)| *age);
        let ids: Vec<&str> = entries.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["stage_a_worker_1", "stage_a_worker_0", "stage_a_worker_2",],
            "ascending sort must place the youngest worker first"
        );
    }

    #[test]
    fn try_remove_worker_cancel_pending_add_drops_age_entry() {
        // Worker added mid-cycle (age 0) and then immediately removed:
        // cancel_pending_add fires, the worker disappears from the
        // plan entirely, and its age entry must be dropped so a
        // subsequent reuse (would be impossible here, but defensively
        // we drop) cannot resurface a stale age 0 record.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("stage_a")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let placed = ctx.try_add_worker(0).unwrap().unwrap();
        assert!(ctx.worker_ages().contains_key(&placed.id));

        assert!(ctx.try_remove_worker(0, &placed.id).unwrap());
        assert!(
            !ctx.worker_ages().contains_key(&placed.id),
            "cancel-pending-add must drop the worker_ages entry"
        );
    }

    #[test]
    fn try_remove_worker_stage_for_removal_keeps_age_entry() {
        // A seeded worker (age >= 1) staged for removal still occupies
        // a slot in pending_removes; a later try_add_worker may revive
        // it via the FGD reuse path. Keep its age entry so the donor
        // logic compares revived workers by their true age, not 0.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 4);
        let mut ctx =
            AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());

        assert_eq!(
            ctx.worker_age("stage_a_worker_0"),
            Some(4),
            "stage-for-removal must NOT drop the worker_ages entry; \
             pending_removes may still be reused this cycle"
        );
    }

    #[test]
    fn try_add_worker_reuse_path_preserves_original_age() {
        // Seeded worker has a non-zero age; remove it (-> pending_removes,
        // age preserved); add another worker with the same shape on
        // the same stage (-> FGD reuse path, age preserved). The reused
        // worker must still report its original age, NOT 0. This is the
        // core invariant that makes age-based donor selection meaningful
        // when the same cycle shrinks then re-grows a stage.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 8);
        let mut ctx =
            AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        assert!(ctx.try_remove_worker(0, "stage_a_worker_0").unwrap());
        let reused = ctx.try_add_worker(0).unwrap().unwrap();
        assert_eq!(
            reused.id, "stage_a_worker_0",
            "reuse path consumed pending remove"
        );
        assert_eq!(
            ctx.worker_age(&reused.id),
            Some(8),
            "reuse path must preserve original age, not collapse to 0"
        );
    }

    #[test]
    fn try_add_worker_reuse_path_preserves_age_for_spmd_groups() {
        // Same invariant as above, applied to the SPMD reuse path
        // (`take_pending_remove_matching_allocations`). SPMD groups go
        // through a different reuse code path than FGD; the age
        // contract must hold for both.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 1.0),
            stages: vec![make_spmd_stage("spmd_stage")],
        };
        let seeded_id = "spmd_worker".to_string();
        let stage_state = ds::ProblemStageState {
            stage_name: "spmd_stage".to_string(),
            worker_groups: vec![ds::ProblemWorkerGroupState {
                id: seeded_id.clone(),
                resources: vec![
                    rds::WorkerResources {
                        node: "node0".to_string(),
                        cpus: rds::FixedUtil::ONE,
                        gpus: vec![rds::GpuAllocation {
                            offset: 0,
                            used_fraction: rds::FixedUtil::ONE,
                        }],
                    },
                    rds::WorkerResources {
                        node: "node1".to_string(),
                        cpus: rds::FixedUtil::ONE,
                        gpus: vec![rds::GpuAllocation {
                            offset: 0,
                            used_fraction: rds::FixedUtil::ONE,
                        }],
                    },
                ],
                num_used_slots: 0,
            }],
            slots_per_worker: 1,
            is_finished: false,
            ..Default::default()
        };
        let state = ds::ProblemState {
            stages: vec![stage_state],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert(seeded_id.clone(), 6);
        let mut ctx =
            AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();

        assert!(ctx.try_remove_worker(0, &seeded_id).unwrap());
        let reused = ctx.try_add_worker(0).unwrap().unwrap();
        assert_eq!(
            reused.id, seeded_id,
            "SPMD reuse path consumed pending remove"
        );
        assert_eq!(
            ctx.worker_age(&reused.id),
            Some(6),
            "SPMD reuse path must preserve original age"
        );
    }

    #[test]
    fn worker_ages_increment_cycle_to_cycle_via_caller() {
        // Age increments cycle-to-cycle: build context for cycle 1
        // (cold start), drain to Solution, increment all surviving
        // ages by 1 (caller's job between cycles), build context for
        // cycle 2 with the incremented map. The same worker now
        // reports age 1.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state_cycle_1 = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let ctx_cycle_1 =
            AutoscalePlanContext::from_problem_state(&problem, &state_cycle_1, None).unwrap();
        // Cold start: every seeded worker is age 0.
        assert_eq!(ctx_cycle_1.worker_age("stage_a_worker_0"), Some(0));

        // Caller advances the clock between cycles by incrementing
        // every surviving worker's age by 1.
        let mut next_cycle_ages = ctx_cycle_1.worker_ages();
        for v in next_cycle_ages.values_mut() {
            *v += 1;
        }

        // Cycle 2: same physical worker is still alive, the seeded age
        // map carries the incremented value, and the new context
        // reports age 1.
        let state_cycle_2 = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let ctx_cycle_2 = AutoscalePlanContext::from_problem_state(
            &problem,
            &state_cycle_2,
            Some(next_cycle_ages),
        )
        .unwrap();
        assert_eq!(
            ctx_cycle_2.worker_age("stage_a_worker_0"),
            Some(1),
            "age must increment cycle-to-cycle when the caller \
             passes the previous cycle's incremented map"
        );
    }

    #[test]
    fn worker_ages_zero_stage_pipeline_yields_empty_map_without_seed() {
        // Boundary: zero-stage pipeline reports an empty worker_ages
        // map when the constructor is called without a seed.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: Vec::new(),
        };
        let state = ds::ProblemState { stages: Vec::new() };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        assert!(ctx.worker_ages().is_empty());
    }

    #[test]
    fn worker_ages_zero_stage_pipeline_drops_every_seed_entry_as_stale() {
        // Boundary: zero-stage pipeline with a non-empty seed yields
        // an empty map -- no stage means no workers, so every seed
        // entry is stale and silently filtered.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: Vec::new(),
        };
        let state = ds::ProblemState { stages: Vec::new() };

        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("ghost_worker".to_string(), 99);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();
        assert!(ctx.worker_ages().is_empty());
    }

    #[test]
    fn try_add_worker_returning_none_does_not_mutate_worker_ages() {
        // Mutation guard: when try_add_worker returns None (cluster
        // full) the worker_ages map must be byte-identical to its
        // pre-call state. A leak here would let the donor logic
        // observe a phantom worker as age 0 after a failed add.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 4);
        let mut ctx =
            AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();
        let ages_before = ctx.worker_ages();

        // 1-cpu cluster already saturated by the seeded worker; a
        // second 1-cpu add cannot fit and the planner returns None.
        assert!(ctx.try_add_worker(0).unwrap().is_none());
        assert_eq!(ctx.worker_ages(), ages_before);
    }

    #[test]
    fn try_remove_worker_returning_false_does_not_mutate_worker_ages() {
        // Mutation guard: when try_remove_worker returns false (no
        // such worker in the planning snapshot) the worker_ages map
        // must be byte-identical to its pre-call state. A leak here
        // would silently drop a live worker's age and collapse it to
        // 0 on a future cycle.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 6);
        let mut ctx =
            AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();
        let ages_before = ctx.worker_ages();

        assert!(!ctx.try_remove_worker(0, "nonexistent").unwrap());
        assert_eq!(ctx.worker_ages(), ages_before);
    }

    #[test]
    fn worker_ages_remain_valid_after_into_solution_drained() {
        // Read accessors stay valid after drain: the documented caller
        // pattern reads `worker_ages()` AFTER `into_solution()` to
        // filter against `solution.deleted_workers` for the next cycle.
        // Mutating entrypoints are guarded; reads are not.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 2.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 5);
        let mut ctx =
            AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();
        let added = ctx.try_add_worker(0).unwrap().unwrap();

        // Drain the plan; the read accessors must keep working.
        let _solution = ctx.into_solution().unwrap();
        let ages_after_drain = ctx.worker_ages();
        assert_eq!(ages_after_drain.get("stage_a_worker_0"), Some(&5u64));
        assert_eq!(ages_after_drain.get(&added.id), Some(&0u64));
        assert_eq!(ctx.worker_age("stage_a_worker_0"), Some(5));
    }

    #[test]
    fn worker_ages_returns_a_clone_so_caller_mutations_do_not_leak() {
        // The contract documented at the Python boundary requires that
        // `worker_ages()` returns a fresh map; mutating it must not
        // affect the underlying context state. Pin this so a future
        // refactor that exposes a Rust reference cannot silently break
        // the encapsulation.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut seed: HashMap<String, u64> = HashMap::new();
        seed.insert("stage_a_worker_0".to_string(), 4);
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, Some(seed)).unwrap();
        let mut snapshot_a = ctx.worker_ages();
        snapshot_a.insert("stage_a_worker_0".to_string(), 999);
        snapshot_a.insert("phantom".to_string(), 12345);
        let snapshot_b = ctx.worker_ages();
        assert_eq!(snapshot_b.get("stage_a_worker_0"), Some(&4u64));
        assert!(!snapshot_b.contains_key("phantom"));
    }

    #[test]
    fn worker_ids_by_stage_returns_seeded_workers_in_stage_order() {
        // Two stages with 2 and 3 seeded workers respectively; the
        // accessor returns one entry per stage in the same order as
        // `Problem.stages`, with per-stage ids sorted lexicographically.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 8.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_cpu_stage_state_with_workers("stage_a", 2, "node0"),
                make_cpu_stage_state_with_workers("stage_b", 3, "node0"),
            ],
        };

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        let by_stage = ctx.worker_ids_by_stage();
        assert_eq!(
            by_stage,
            vec![
                vec![
                    "stage_a_worker_0".to_string(),
                    "stage_a_worker_1".to_string()
                ],
                vec![
                    "stage_b_worker_0".to_string(),
                    "stage_b_worker_1".to_string(),
                    "stage_b_worker_2".to_string(),
                ],
            ]
        );
    }

    #[test]
    fn worker_ids_by_stage_reflects_try_add_and_try_remove_within_cycle() {
        // The accessor must return the LIVE state, not the seed: a
        // successful add appears in the entry; a successful remove
        // disappears from it.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();

        let added = ctx
            .try_add_worker(0)
            .unwrap()
            .expect("cluster has 4 cpus, add succeeds");
        let added_id = added.id.clone();
        let after_add = ctx.worker_ids_by_stage();
        assert_eq!(after_add.len(), 1);
        let mut expected_after_add = vec!["stage_a_worker_0".to_string(), added_id.clone()];
        expected_after_add.sort();
        assert_eq!(after_add[0], expected_after_add);

        let removed = ctx.try_remove_worker(0, "stage_a_worker_0").unwrap();
        assert!(removed);
        let after_remove = ctx.worker_ids_by_stage();
        assert_eq!(after_remove[0], vec![added_id]);
    }

    #[test]
    fn worker_ids_by_stage_remains_valid_after_into_solution() {
        // Read accessors deliberately bypass the drained-state guard so
        // callers can persist the post-cycle state for the next cycle.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_cpu_stage_state_with_workers("stage_a", 1, "node0")],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        let _ = ctx.into_solution();

        let by_stage = ctx.worker_ids_by_stage();
        assert_eq!(by_stage, vec![vec!["stage_a_worker_0".to_string()]]);
    }

    #[test]
    fn worker_ids_by_stage_zero_stage_pipeline_returns_empty_outer_vec() {
        // Boundary: zero-stage pipeline yields an empty outer vector
        // (no per-stage entries, not a one-element vector with an
        // empty inner list).
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
            stages: Vec::new(),
        };
        let state = ds::ProblemState { stages: Vec::new() };
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        assert!(ctx.worker_ids_by_stage().is_empty());
    }

    #[test]
    fn worker_ids_by_stage_unions_spmd_and_non_spmd_workers() {
        // SPMD workers live in `current_worker_groups`; non-SPMD live in
        // `current_workers`. The accessor must surface both and merge them
        // into one per-stage list.
        let problem = ds::Problem {
            cluster_resources: make_gpu_cluster(2, 2.0),
            stages: vec![make_spmd_stage("stage_spmd"), make_cpu_stage("stage_cpu")],
        };
        let spmd_group = ds::ProblemWorkerGroupState {
            id: "spmd_group_0".to_string(),
            resources: vec![
                rds::WorkerResources {
                    node: "node0".to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::ONE,
                    }],
                },
                rds::WorkerResources {
                    node: "node1".to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::ONE,
                    }],
                },
            ],
            num_used_slots: 0,
        };
        let state = ds::ProblemState {
            stages: vec![
                ds::ProblemStageState {
                    stage_name: "stage_spmd".to_string(),
                    worker_groups: vec![spmd_group],
                    slots_per_worker: 1,
                    is_finished: false,
                    ..Default::default()
                },
                make_cpu_stage_state_with_workers("stage_cpu", 1, "node0"),
            ],
        };
        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state, None).unwrap();
        let by_stage = ctx.worker_ids_by_stage();
        assert_eq!(
            by_stage,
            vec![
                vec!["spmd_group_0".to_string()],
                vec!["stage_cpu_worker_0".to_string()],
            ]
        );
    }

    /// Helper: stage state with a single CPU worker on the given node
    /// whose id is supplied by the caller, so callers can construct
    /// collisions on demand.
    fn make_stage_state_with_named_workers(
        stage_name: &str,
        worker_ids: &[&str],
        node_id: &str,
    ) -> ds::ProblemStageState {
        let worker_groups = worker_ids
            .iter()
            .map(|id| ds::ProblemWorkerGroupState {
                id: id.to_string(),
                resources: vec![rds::WorkerResources {
                    node: node_id.to_string(),
                    cpus: rds::FixedUtil::ONE,
                    gpus: Vec::new(),
                }],
                num_used_slots: 0,
            })
            .collect();
        ds::ProblemStageState {
            stage_name: stage_name.to_string(),
            worker_groups,
            slots_per_worker: 1,
            is_finished: false,
            ..Default::default()
        }
    }

    #[test]
    fn from_problem_state_rejects_duplicate_stage_names() {
        // Two stages sharing the same name would silently collapse into
        // a single HashMap entry for pending_adds / pending_removes /
        // current_workers / current_worker_groups, corrupting all
        // subsequent planning. The constructor must reject the input.
        // Asserting `is_err()` (not the formatted message) keeps the
        // test free of the Python GIL initialization that
        // `PyErr::Display` requires.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("dup_stage"), make_cpu_stage("dup_stage")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_empty_stage_state("dup_stage"),
                make_empty_stage_state("dup_stage"),
            ],
        };

        let result = AutoscalePlanContext::from_problem_state(&problem, &state, None);
        assert!(
            result.is_err(),
            "duplicate stage name must be rejected before seeding"
        );
    }

    #[test]
    fn from_problem_state_rejects_duplicate_worker_ids_within_stage() {
        // Two workers in the same stage sharing one id previously
        // silently overwrote each other in `current_workers`, leaving
        // the cluster snapshot double-allocated. Constructor must
        // reject before any cluster allocation runs.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_stage_state_with_named_workers(
                "stage_a",
                &["dup_w", "dup_w"],
                "node0",
            )],
        };

        let result = AutoscalePlanContext::from_problem_state(&problem, &state, None);
        assert!(
            result.is_err(),
            "duplicate worker id within a stage must be rejected"
        );
    }

    #[test]
    fn from_problem_state_rejects_duplicate_worker_ids_across_stages() {
        // Worker ids must be globally unique within a planning cycle.
        // A collision across stages would let one stage's worker
        // accidentally satisfy another stage's lookup and corrupt the
        // reuse path.
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a"), make_cpu_stage("stage_b")],
        };
        let state = ds::ProblemState {
            stages: vec![
                make_stage_state_with_named_workers("stage_a", &["shared_id"], "node0"),
                make_stage_state_with_named_workers("stage_b", &["shared_id"], "node0"),
            ],
        };

        let result = AutoscalePlanContext::from_problem_state(&problem, &state, None);
        assert!(
            result.is_err(),
            "duplicate worker id across stages must be rejected"
        );
    }
}
