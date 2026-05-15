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

//! Per-cycle planning context for pure-Python schedulers.
//!
//! The `AutoscalePlanContext` is a `pyclass` that owns a working copy of the
//! cluster snapshot plus per-stage staged adds and removes. It is the
//! Python-callable equivalent of the private `AutoscaleContext` used by
//! `run_fragmentation_autoscaler`; lifting it into a `pyclass` lets a pure-
//! Python scheduler (the saturation-aware path) plan multi-add cycles using
//! the same FGD allocator without re-deriving the algorithm in Python.
//!
//! Lifecycle (one instance per autoscale cycle):
//!
//!   1. `from_problem_state(problem, state)` - seeds the working cluster
//!      with all currently-allocated workers from the input snapshot;
//!      initialises empty per-stage `pending_adds` / `pending_removes`
//!      maps; computes the workload estimate used by FGD.
//!   2. `try_add_worker(stage_index)` - runs FGD on the working cluster
//!      and stages the resulting placement.
//!   3. `try_remove_worker(stage_index, worker_id)` - frees a placement
//!      and stages the removal.
//!   4. `into_solution()` - consumes self, drains the staged adds and
//!      removes into a `Solution`.
//!
//! Each cycle constructs a fresh `AutoscalePlanContext`; mutations DO NOT
//! persist across cycles (the consuming `into_solution` enforces this).
//!

use std::collections::HashMap;

use pyo3::prelude::*;

use super::autoscaling_algorithms::WorkerIdFactory;
use super::data_structures as ds;
use super::fragmentation_allocation_algorithms as frag;
use super::resources as rds;

/// Default fragmentation-equivalent reward for reusing a recently removed
/// worker placement during FGD search. Mirrors the constant baked into
/// `run_fragmentation_autoscaler`'s call site (autoscaling_algorithms.rs:1396).
const DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT: f32 = 1.5;

/// Per-cycle planning context for the saturation-aware scheduler.
///
/// Owns a working copy of the cluster snapshot seeded with current worker
/// allocations, plus per-stage maps of pending adds and removes. The
/// scheduler manipulates this context once per cycle and then consumes it
/// via `into_solution()` (lands in 1a-iv).
///
/// # Fields
/// * `cluster` - Working cluster snapshot. Cloned from the input
///   `Problem.cluster_resources` and pre-allocated with all workers
///   currently in `ProblemState`. Mutated during planning by
///   `try_add_worker` / `try_remove_worker`.
/// * `pending_adds` - Per-stage list of workers staged for addition this
///   cycle. Maps stage_name -> Vec of `ProblemWorkerGroupState` to add.
/// * `pending_removes` - Per-stage list of workers staged for removal this
///   cycle. Maps stage_name -> Vec of `ProblemWorkerGroupState` to remove.
/// * `worker_id_factory` - Generates unique IDs for newly allocated workers.
/// * `workload_estimate` - Per-stage workload weights consumed by FGD when
///   choosing among candidate placements.
/// * `worker_reuse_fragmentation_equivalent` - FGD reward parameter that
///   biases the search toward reusing a recently removed placement (avoids
///   thrash when a stage shrinks then re-grows in the same cycle).
#[pyclass]
// All fields below are seeded by `from_problem_state` (1a-i) and consumed by
// the methods that land in subsequent sub-iterations (`try_add_worker` in
// 1a-ii, `try_remove_worker` in 1a-iii, `into_solution` in 1a-iv). The
// `dead_code` allow vanishes once those methods replace their `unimplemented!`
// stubs; suppressing it explicitly keeps the build clean during the
// staged rollout.
#[allow(dead_code)]
pub struct AutoscalePlanContext {
    cluster: rds::ClusterResources,
    pending_adds: HashMap<String, Vec<ds::ProblemWorkerGroupState>>,
    pending_removes: HashMap<String, Vec<ds::ProblemWorkerGroupState>>,
    worker_id_factory: WorkerIdFactory,
    workload_estimate: frag::Workload,
    worker_reuse_fragmentation_equivalent: f32,
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

        // Clone the cluster shape and seed it with every worker currently
        // allocated to a stage. After this loop the working cluster reflects
        // the live placement, so FGD calls in subsequent iterations find
        // available capacity correctly.
        let mut cluster = problem.cluster_resources.clone();
        for (stage, stage_state) in problem.stages.iter().zip(state.stages.iter()) {
            for w in &stage_state.worker_groups {
                match &stage.worker_shape {
                    rds::WorkerShape::SpmdNodeMultiple(_) => {
                        let group = w.to_worker_group(stage.name.clone());
                        cluster.allocate_multiple(&group.allocations).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "AutoscalePlanContext: failed to seed cluster with \
                                 SPMD worker group {} for stage {}: {:?}",
                                w.id, stage.name, e
                            ))
                        })?;
                    }
                    _ => {
                        let worker = w.to_worker(stage.name.clone());
                        cluster.allocate(&worker.allocation).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "AutoscalePlanContext: failed to seed cluster with \
                                 worker {} for stage {}: {:?}",
                                w.id, stage.name, e
                            ))
                        })?;
                    }
                }
            }
        }

        // Initialise per-stage staged-adds and staged-removes maps with one
        // entry per pipeline stage so callers can index by stage name without
        // a None-check.
        let mut pending_adds: HashMap<String, Vec<ds::ProblemWorkerGroupState>> = HashMap::new();
        let mut pending_removes: HashMap<String, Vec<ds::ProblemWorkerGroupState>> = HashMap::new();
        for stage in &problem.stages {
            pending_adds.insert(stage.name.clone(), Vec::new());
            pending_removes.insert(stage.name.clone(), Vec::new());
        }

        // Build the workload estimate FGD consumes when ranking candidate
        // placements; uniform weighting when no stage has requested workers
        // (matches the seed convention in run_fragmentation_autoscaler).
        let workload_estimate = build_workload_estimate(problem, state);

        Ok(Self {
            cluster,
            pending_adds,
            pending_removes,
            worker_id_factory: WorkerIdFactory::default(),
            workload_estimate,
            worker_reuse_fragmentation_equivalent: DEFAULT_WORKER_REUSE_FRAGMENTATION_EQUIVALENT,
        })
    }

    /// Try to place a new worker for `stage_index` using FGD on the working
    /// snapshot. Returns the placed worker on success (and mutates the
    /// snapshot to reflect the allocation); returns `None` on
    /// `AllocationError`.
    ///
    /// Lands in granular sub-iteration `1a-ii`.
    pub fn try_add_worker(
        &mut self,
        _stage_index: usize,
    ) -> PyResult<Option<ds::ProblemWorkerGroupState>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AutoscalePlanContext.try_add_worker: lands in granular sub-iteration 1a-ii",
        ))
    }

    /// Remove a specific worker (by id) from the working snapshot, freeing
    /// its resources for reuse by a subsequent `try_add_worker` call. Returns
    /// True on success, False if the worker was not found in this stage's
    /// current set.
    ///
    /// Lands in granular sub-iteration `1a-iii`.
    pub fn try_remove_worker(&mut self, _stage_index: usize, _worker_id: &str) -> PyResult<bool> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AutoscalePlanContext.try_remove_worker: lands in granular sub-iteration 1a-iii",
        ))
    }

    /// Freeze the plan into a `Solution`. Consumes self; further mutations
    /// are not allowed (caller must construct a new `AutoscalePlanContext`
    /// each cycle).
    ///
    /// Lands in granular sub-iteration `1a-iv`.
    pub fn into_solution(&self) -> PyResult<ds::Solution> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AutoscalePlanContext.into_solution: lands in granular sub-iteration 1a-iv",
        ))
    }

    /// Read-only number of stages this context is tracking.
    ///
    /// Useful for invariant checks (callers can confirm
    /// `len(stage_solutions) == ctx.num_stages()`) and for tests that need
    /// to verify the seeding round-tripped the input shape correctly.
    pub fn num_stages(&self) -> usize {
        self.pending_adds.len()
    }
}

/// Build the workload estimate consumed by FGD when ranking candidate
/// placements. Mirrors the equivalent helper in `autoscaling_algorithms.rs`
/// (`make_workload_from_state`) so both code paths produce the same
/// per-stage frequency weighting.
///
/// When no stage has any current workers and no manual `requested_num_workers`
/// is set, the workload defaults to uniform weighting across stages - this
/// matches the seed convention that keeps FGD progressing on a cold cluster.
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
            "default reuse parameter matches the legacy autoscaler constant"
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

    #[test]
    fn other_methods_stub_to_not_implemented() {
        let problem = ds::Problem {
            cluster_resources: make_empty_cluster(1, 1.0),
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

        assert!(
            ctx.try_add_worker(0).is_err(),
            "try_add_worker stubbed; lands in 1a-ii"
        );
        assert!(
            ctx.try_remove_worker(0, "fake_id").is_err(),
            "try_remove_worker stubbed; lands in 1a-iii"
        );
        assert!(
            ctx.into_solution().is_err(),
            "into_solution stubbed; lands in 1a-iv"
        );
    }
}
