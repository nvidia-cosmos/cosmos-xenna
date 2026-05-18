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
//!   1. `from_problem_state(problem, state)` - seeds the working cluster
//!      with all currently-allocated workers from the input snapshot;
//!      initialises empty per-stage `pending_adds` / `pending_removes`
//!      maps; computes the workload estimate used by FGD.
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

use super::autoscaling_algorithms::WorkerIdFactory;
use super::data_structures as ds;
use super::fragmentation_allocation_algorithms as frag;
use super::resources as rds;

/// Default fragmentation-equivalent reward for reusing a recently removed
/// worker placement during FGD search. Matches the value used by
/// `run_fragmentation_autoscaler` so the plan context keeps the same reuse
/// preference when its mutation methods are implemented.
const DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT: f32 = 10.0;

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
    ///
    /// # Errors
    /// Returns a `PyRuntimeError` if seeding fails because a current worker
    /// cannot be allocated on the cluster (would indicate a corrupted
    /// snapshot).
    #[new]
    pub fn from_problem_state(problem: &ds::Problem, state: &ds::ProblemState) -> PyResult<Self> {
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
        // stage name without a None-check.
        let mut pending_adds: HashMap<String, Vec<ds::ProblemWorkerGroupState>> = HashMap::new();
        let mut pending_removes: HashMap<String, Vec<ds::ProblemWorkerGroupState>> = HashMap::new();
        let mut current_workers: HashMap<String, HashMap<String, ds::ProblemWorkerGroupState>> =
            HashMap::new();
        let mut current_worker_groups: HashMap<
            String,
            HashMap<String, ds::ProblemWorkerGroupState>,
        > = HashMap::new();
        let mut reserved_worker_ids: HashSet<String> = HashSet::new();
        for stage in &problem.stages {
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
                reserved_worker_ids.insert(w.id.clone());
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
                        let worker = w.to_worker(stage.name.clone());
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
        // placements; uniform weighting when no stage has requested workers
        // (matches the seed convention in run_fragmentation_autoscaler).
        let workload_estimate = build_workload_estimate(problem, state);

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

        Ok(Self {
            cluster,
            stages,
            pending_adds,
            pending_removes,
            current_workers,
            current_worker_groups,
            reserved_worker_ids,
            slots_per_worker_by_stage,
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
                Ok(Some(placement))
            }
            _ => {
                // Build the reuse map FGD expects: stage's currently
                // staged-for-removal workers indexed by id. Conversion is
                // O(R) where R = pending removes for this stage; small
                // by construction (one removed worker per scale-down
                // event).
                let reuse_map: HashMap<String, rds::Worker> = self
                    .pending_removes
                    .get(&stage_name)
                    .map(|list| {
                        list.iter()
                            .map(|p| (p.id.clone(), p.to_worker(stage_name.clone())))
                            .collect()
                    })
                    .unwrap_or_default();

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
                    let worker = reused.to_worker(stage_name.clone());
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
                    return Ok(true);
                }
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
                    return Ok(true);
                }
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

/// Build the workload estimate consumed by FGD when ranking candidate
/// placements. Mirrors the equivalent helper in `autoscaling_algorithms.rs`
/// (`make_workload_from_state`) so both code paths produce the same
/// per-stage frequency weighting.
///
/// When no stage has any current workers and no manual `requested_num_workers`
/// is set, the workload defaults to uniform weighting across stages so FGD
/// has a non-degenerate weighting to progress on a cold cluster.
fn build_workload_estimate(problem: &ds::Problem, state: &ds::ProblemState) -> frag::Workload {
    let mut total_requested: usize = 0;
    let mut per_stage_requested: Vec<usize> = Vec::with_capacity(problem.stages.len());
    for (stage_problem, stage_state) in problem.stages.iter().zip(state.stages.iter()) {
        let n = stage_state.num_workers();
        let req = stage_problem.requested_num_workers.unwrap_or(n);
        per_stage_requested.push(req);
        total_requested += req;
    }

    if total_requested == 0 {
        let freq = if problem.stages.is_empty() {
            0.0
        } else {
            1.0 / (problem.stages.len() as f32)
        };
        return frag::Workload {
            stages: problem
                .stages
                .iter()
                .map(|s| frag::Stage {
                    frequency: freq,
                    shape: s.worker_shape.clone(),
                })
                .collect(),
        };
    }

    frag::Workload {
        stages: problem
            .stages
            .iter()
            .zip(per_stage_requested)
            .map(|(s, req)| frag::Stage {
                frequency: (req as f32) / (total_requested as f32),
                shape: s.worker_shape.clone(),
            })
            .collect(),
    }
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

// --------------------
// Tests (pure Rust)
// --------------------

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

    // ----- New corner-case coverage starts here -----

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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state)
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

        let result = AutoscalePlanContext::from_problem_state(&problem, &state);
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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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

        let result = AutoscalePlanContext::from_problem_state(&problem, &state);
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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state)
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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        // normaliser used by `build_workload_estimate`.
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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
    fn from_problem_state_rejects_stage_name_mismatch() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 4.0),
            stages: vec![make_cpu_stage("stage_a")],
        };
        let state = ds::ProblemState {
            stages: vec![make_empty_stage_state("wrong_stage")],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state).is_err(),
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
                }],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state).is_err(),
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
                }],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state).is_err(),
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
            AutoscalePlanContext::from_problem_state(&problem, &state).is_err(),
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
                }],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };

        assert!(
            AutoscalePlanContext::from_problem_state(&problem, &state).is_err(),
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

        let ctx = AutoscalePlanContext::from_problem_state(&problem, &state)
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

        let result = AutoscalePlanContext::from_problem_state(&problem, &state);
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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
                    },
                    ds::ProblemWorkerGroupState {
                        id: "1".to_string(),
                        resources: vec![rds::WorkerResources {
                            node: "node0".to_string(),
                            cpus: rds::FixedUtil::ONE,
                            gpus: Vec::new(),
                        }],
                    },
                ],
                slots_per_worker: 1,
                is_finished: false,
                ..Default::default()
            }],
        };
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();
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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();
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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();
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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

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
        let mut ctx = AutoscalePlanContext::from_problem_state(&problem, &state).unwrap();

        let solution = ctx.into_solution().unwrap();

        assert!(solution.stages.is_empty());
    }
}
