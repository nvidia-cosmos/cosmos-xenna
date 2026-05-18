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
//! path is live today; the mutation methods are Python-visible stubs until
//! the Fragmentation Gradient Descent (FGD) planning operations land.
//!
//! Lifecycle (one instance per autoscale cycle):
//!
//!   1. `from_problem_state(problem, state)` - seeds the working cluster
//!      with all currently-allocated workers from the input snapshot;
//!      initialises empty per-stage `pending_adds` / `pending_removes`
//!      maps; computes the workload estimate used by FGD.
//!   2. `try_add_worker(stage_index)` - currently raises
//!      `NotImplementedError`.
//!   3. `try_remove_worker(stage_index, worker_id)` - currently raises
//!      `NotImplementedError`.
//!   4. `into_solution()` - currently raises `NotImplementedError`.
//!
//! Callers should construct a fresh `AutoscalePlanContext` for each cycle;
//! mutations are scoped to the context instance.
//!

use std::collections::HashMap;

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
/// allocations, plus per-stage maps of pending adds and removes. The
/// construction path is implemented now; mutation methods are exposed as
/// stubs until the planner can stage adds and removes.
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
pub struct AutoscalePlanContext {
    #[allow(dead_code)]
    cluster: rds::ClusterResources,
    pending_adds: HashMap<String, Vec<ds::ProblemWorkerGroupState>>,
    #[allow(dead_code)]
    pending_removes: HashMap<String, Vec<ds::ProblemWorkerGroupState>>,
    #[allow(dead_code)]
    worker_id_factory: WorkerIdFactory,
    #[allow(dead_code)]
    workload_estimate: frag::Workload,
    #[allow(dead_code)]
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
    /// Not yet implemented.
    pub fn try_add_worker(
        &mut self,
        _stage_index: usize,
    ) -> PyResult<Option<ds::ProblemWorkerGroupState>> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AutoscalePlanContext.try_add_worker is not yet implemented",
        ))
    }

    /// Remove a specific worker (by id) from the working snapshot, freeing
    /// its resources for reuse by a subsequent `try_add_worker` call. Returns
    /// True on success, False if the worker was not found in this stage's
    /// current set.
    ///
    /// Not yet implemented.
    pub fn try_remove_worker(&mut self, _stage_index: usize, _worker_id: &str) -> PyResult<bool> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AutoscalePlanContext.try_remove_worker is not yet implemented",
        ))
    }

    /// Build a `Solution` from the staged plan.
    ///
    /// Not yet implemented.
    pub fn into_solution(&self) -> PyResult<ds::Solution> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "AutoscalePlanContext.into_solution is not yet implemented",
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
            "try_add_worker is currently a NotImplementedError stub"
        );
        assert!(
            ctx.try_remove_worker(0, "fake_id").is_err(),
            "try_remove_worker is currently a NotImplementedError stub"
        );
        assert!(
            ctx.into_solution().is_err(),
            "into_solution is currently a NotImplementedError stub"
        );
    }
}
