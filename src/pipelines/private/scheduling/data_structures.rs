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

//! Data structures used by autoscaling algorithms and simulations.
//!
//! This module presents an interface for autoscaling algorithms. This interface formulates the autoscaling information as
//! a "Problem" and "Solution". It provides data structures for representing resource allocation problems and their
//! solutions in a distributed computing environment.

use crate::utils::module_builders::ImportablePyModuleBuilder;

use super::resources;
use comfy_table::{ContentArrangement, Table};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter, Result as FmtResult};

// --------------------
// Problem description
// --------------------

/// Represents a single stage in the allocation problem.
///
/// A stage represents a discrete step in the processing pipeline that requires
/// specific resource allocations.
///
/// # Attributes
/// * `name` - A unique identifier for the stage.
/// * `worker_shape` - Resource requirements for each worker in this stage.
/// * `requested_num_workers` - Optional explicitly requested number of workers.
///     If specified, this is the exact number of workers requested for the stage.
///     If None, the number of workers will be determined by the autoscaling algorithm.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct ProblemStage {
    pub name: String,
    pub stage_batch_size: usize,
    pub worker_shape: resources::WorkerShape,
    pub requested_num_workers: Option<usize>,
    pub over_provision_factor: Option<f32>,
}

#[pymethods]

impl ProblemStage {
    #[new]
    pub fn new(
        name: String,
        stage_batch_size: usize,
        worker_shape: resources::WorkerShape,
        requested_num_workers: Option<usize>,
        over_provision_factor: Option<f32>,
    ) -> Self {
        Self {
            name,
            stage_batch_size,
            worker_shape,
            requested_num_workers,
            over_provision_factor,
        }
    }
}

/// Represents the state of a worker group in the system.
///
/// # Attributes
/// * `id` - Unique identifier for the worker group.
/// * `resources` - Per-allocation resource list.
/// * `num_used_slots` - Number of task slots currently occupied on this worker
///   at sample time. Defaults to 0; consumers that do not populate this
///   field treat the default as "no signal" (any worker is equally
///   eligible for selection).
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProblemWorkerGroupState {
    pub id: String,
    pub resources: Vec<resources::WorkerResources>,
    pub num_used_slots: usize,
}

#[pymethods]
impl ProblemWorkerGroupState {
    /// Creates a ProblemWorkerState from a Worker instance.
    ///
    /// # Arguments
    /// * `state` - Worker instance containing worker state information.
    ///
    /// # Returns
    /// A new ProblemWorkerState instance.
    #[staticmethod]
    pub fn make_from_worker_group_state(state: resources::WorkerGroup) -> Self {
        Self {
            id: state.id,
            resources: state.allocations,
            num_used_slots: 0,
        }
    }

    #[staticmethod]
    pub fn make_from_worker_state(state: resources::Worker) -> Self {
        Self {
            id: state.id,
            resources: vec![state.allocation],
            num_used_slots: 0,
        }
    }

    /// Construct a `ProblemWorkerGroupState`.
    ///
    /// The `num_used_slots` field is an optional keyword argument
    /// defaulting to 0 so existing call sites that built
    /// `ProblemWorkerGroupState` from two positional arguments continue to
    /// compile unchanged.
    #[new]
    #[pyo3(signature = (id, resources, num_used_slots = 0))]
    pub fn py_new(
        id: String,
        resources: Vec<resources::WorkerResources>,
        num_used_slots: usize,
    ) -> Self {
        Self {
            id,
            resources,
            num_used_slots,
        }
    }

    /// Converts this state to a Worker instance.
    ///
    /// # Arguments
    /// * `stage_name` - Name of the stage this worker belongs to.
    ///
    /// # Returns
    /// A Worker instance representing this state.
    pub fn to_worker_group(&self, stage_name: String) -> resources::WorkerGroup {
        resources::WorkerGroup {
            id: self.id.clone(),
            stage_name,
            allocations: self.resources.clone(),
        }
    }

    /// Convert this state into a single-allocation `Worker`.
    ///
    /// Returns `PyValueError` if `resources` does not have exactly
    /// one allocation. The previous implementation panicked on
    /// multi-allocation input, which propagated as a process-level
    /// crash through PyO3 instead of a recoverable Python exception.
    pub fn to_worker(&self, stage_name: String) -> PyResult<resources::Worker> {
        if self.resources.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ProblemWorkerGroupState.to_worker: worker {} for stage {} \
                 must have exactly one resource allocation, got {}",
                self.id,
                stage_name,
                self.resources.len(),
            )));
        }
        Ok(resources::Worker {
            id: self.id.clone(),
            stage_name,
            allocation: self.resources[0].clone(),
        })
    }

    /// Serialize this state to a JSON string.
    ///
    /// Returns `PyRuntimeError` if the serializer fails. Serde rarely
    /// fails on flat owned structs, but we surface any failure as a
    /// Python exception instead of panicking through PyO3.
    pub fn serialize(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ProblemWorkerGroupState serialize failed: {err}"
            ))
        })
    }

    /// Deserialize a JSON string into a `ProblemWorkerGroupState`.
    ///
    /// Returns `PyValueError` on malformed JSON instead of panicking
    /// through PyO3.
    #[staticmethod]
    pub fn deserialize(data: &str) -> PyResult<Self> {
        serde_json::from_str(data).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "ProblemWorkerGroupState deserialize failed: {err}"
            ))
        })
    }
}

/// Represents the current state of a stage including its workers.
///
/// # Attributes
/// * `stage_name` - Name identifier for this stage.
/// * `worker_groups` - List of worker groups currently assigned to this stage.
/// * `slots_per_worker` - Number of task slots available per worker.
/// * `is_finished` - Boolean indicating if this stage has completed processing.
/// * `num_used_slots` - Number of task slots currently occupied across all
///   workers in the stage at sample time. Defaults to 0; consumers that do
///   not populate this field treat the default as "no signal".
/// * `num_empty_slots` - Number of task slots currently free across all
///   workers in the stage at sample time. Combined with `num_used_slots`,
///   gives the total in-stage slot capacity at sample time. Defaults to 0.
/// * `input_queue_depth` - Number of pre-batch tasks queued upstream of this
///   stage at sample time. Defaults to 0.
/// * `num_pending_actors` - Number of actors currently in setup phases (not
///   yet ready to process tasks). Defaults to 0; the saturation-aware
///   scheduler combines this with `worker_groups.len()` (the ready-actor
///   count) to detect the setup-phase quiescence condition described by the
///   `setup_phase_quiescence_enabled` configuration field.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Default)]
pub struct ProblemStageState {
    pub stage_name: String,
    pub worker_groups: Vec<ProblemWorkerGroupState>,
    pub slots_per_worker: usize,
    pub is_finished: bool,
    pub num_used_slots: usize,
    pub num_empty_slots: usize,
    pub input_queue_depth: usize,
    pub num_pending_actors: usize,
}

#[pymethods]
impl ProblemStageState {
    /// Construct a `ProblemStageState`.
    ///
    /// The slot-signal fields (`num_used_slots`, `num_empty_slots`,
    /// `input_queue_depth`) and the setup-phase quiescence signal
    /// (`num_pending_actors`) are optional keyword arguments defaulting to 0
    /// so existing call sites that built `ProblemStageState` from four
    /// positional arguments continue to compile unchanged.
    #[new]
    #[pyo3(signature = (
        stage_name,
        worker_groups,
        slots_per_worker,
        is_finished,
        num_used_slots = 0,
        num_empty_slots = 0,
        input_queue_depth = 0,
        num_pending_actors = 0,
    ))]
    pub fn py_new(
        stage_name: String,
        worker_groups: Vec<ProblemWorkerGroupState>,
        slots_per_worker: usize,
        is_finished: bool,
        num_used_slots: usize,
        num_empty_slots: usize,
        input_queue_depth: usize,
        num_pending_actors: usize,
    ) -> Self {
        Self {
            stage_name,
            worker_groups,
            slots_per_worker,
            is_finished,
            num_used_slots,
            num_empty_slots,
            input_queue_depth,
            num_pending_actors,
        }
    }

    /// Returns the current number of workers in this stage.
    pub fn num_workers(&self) -> usize {
        self.worker_groups.len()
    }
}

/// Represents the complete current state of the allocation problem.
///
/// Provides a snapshot of all stages and their current resource allocations.
///
/// # Attributes
/// * `stages` - List of all stage states in the system.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct ProblemState {
    pub stages: Vec<ProblemStageState>,
}

#[pymethods]
impl ProblemState {
    #[new]
    pub fn new(stages: Vec<ProblemStageState>) -> Self {
        Self { stages }
    }
}

impl Display for ProblemState {
    /// Returns a formatted string representation of the problem state.
    ///
    /// # Returns
    /// A string containing a tabulated view of all stages and their resource allocations.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut table = Table::new();
        table
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Stage", "Worker ID", "Node", "CPUs", "GPUs"]);

        for (stage_idx, stage) in self.stages.iter().enumerate() {
            for w in &stage.worker_groups {
                // WorkerGroup can have multiple resource allocations
                for (alloc_idx, resource) in w.resources.iter().enumerate() {
                    let gpu_alloc = resource
                        .gpus
                        .iter()
                        .map(|g| format!("{}:{:.2}", g.offset, g.used_fraction))
                        .collect::<Vec<_>>()
                        .join(", ");

                    let worker_id = if w.resources.len() > 1 {
                        format!("{}-{}", w.id, alloc_idx)
                    } else {
                        w.id.clone()
                    };

                    table.add_row(vec![
                        stage_idx.to_string(),
                        worker_id,
                        resource.node.clone(),
                        format!("{:.2}", resource.cpus.to_num::<f32>()),
                        gpu_alloc,
                    ]);
                }
            }
        }
        write!(f, "{}", table)
    }
}

/// Represents the complete allocation problem to be solved.
///
/// This class encapsulates all information needed to solve the resource
/// allocation problem, including cluster resources and stage definitions.
///
/// # Attributes
/// * `cluster_resources` - Total available resources in the cluster.
/// * `stages` - List of all stages that need resource allocation.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Problem {
    pub cluster_resources: resources::ClusterResources,
    pub stages: Vec<ProblemStage>,
}

#[pymethods]
impl Problem {
    #[new]
    pub fn py_new(
        cluster_resources: resources::ClusterResources,
        stages: Vec<ProblemStage>,
    ) -> Self {
        Self {
            cluster_resources,
            stages,
        }
    }
}

// --------------------
// Solution
// --------------------

/// Represents the allocation result for a single stage.
///
/// Contains information about resource allocation changes for a specific stage.
///
/// # Attributes
/// * `slots_per_worker` - Number of task slots to allocate per worker.
/// * `new_workers` - List of workers to be added to the stage.
/// * `deleted_workers` - List of workers to be removed from the stage.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct StageSolution {
    pub slots_per_worker: usize,
    pub new_workers: Vec<ProblemWorkerGroupState>,
    pub deleted_workers: Vec<ProblemWorkerGroupState>,
}

impl StageSolution {
    pub fn new(slots_per_worker: usize) -> Self {
        Self {
            slots_per_worker,
            new_workers: Vec::new(),
            deleted_workers: Vec::new(),
        }
    }
}

#[pymethods]
impl StageSolution {
    /// Construct an empty StageSolution from Python.
    ///
    /// Used by pure-Python schedulers that need to produce
    /// `Solution` outputs without going through the Rust autoscaler.
    /// Callers populate `new_workers` and `deleted_workers` via the
    /// `set_all`-exposed setters; mirrors the existing `#[new]`
    /// constructors on `ProblemWorkerGroupState`, `ProblemStageState`,
    /// `ProblemState`, and `Problem`.
    #[new]
    pub fn py_new(slots_per_worker: usize) -> Self {
        Self::new(slots_per_worker)
    }
}

/// Represents the complete result of the allocation problem.
///
/// Contains the complete set of changes to be applied to the system.
///
/// # Attributes
/// * `stages` - List of solutions for each stage in the system.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Default)]
pub struct Solution {
    pub stages: Vec<StageSolution>,
}

#[pymethods]
impl Solution {
    /// Construct an empty Solution from Python.
    ///
    /// Used by pure-Python schedulers that need to assemble a
    /// `Solution` from a list of `StageSolution`s without going
    /// through the Rust autoscaler. Callers populate `stages` via
    /// the `set_all`-exposed setter.
    #[new]
    pub fn py_new() -> Self {
        Self::default()
    }
    pub fn num_new_workers_per_stage(&self) -> Vec<usize> {
        self.stages.iter().map(|x| x.new_workers.len()).collect()
    }
    pub fn num_deleted_workers_per_stage(&self) -> Vec<usize> {
        self.stages
            .iter()
            .map(|x| x.deleted_workers.len())
            .collect()
    }
}

impl Display for Solution {
    /// Returns a formatted string representation of the solution.
    ///
    /// # Returns
    /// A string containing a tabulated view of all resource allocation changes.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if self.stages.is_empty() {
            return write!(f, "No changes in allocation");
        }
        let mut table = Table::new();
        table
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["Stage", "Action", "Worker ID", "Node", "CPUs", "GPUs"]);

        for (stage_idx, stage) in self.stages.iter().enumerate() {
            for w in &stage.new_workers {
                // WorkerGroup can have multiple resource allocations
                for (alloc_idx, resource) in w.resources.iter().enumerate() {
                    let gpu_alloc = resource
                        .gpus
                        .iter()
                        .map(|g| format!("{}:{:.2}", g.offset, g.used_fraction))
                        .collect::<Vec<_>>()
                        .join(", ");

                    let worker_id = if w.resources.len() > 1 {
                        format!("{}-{}", w.id, alloc_idx)
                    } else {
                        w.id.clone()
                    };

                    table.add_row(vec![
                        stage_idx.to_string(),
                        "New".to_string(),
                        worker_id,
                        resource.node.clone(),
                        format!("{:.2}", resource.cpus.to_num::<f32>()),
                        gpu_alloc,
                    ]);
                }
            }
            for w in &stage.deleted_workers {
                // WorkerGroup can have multiple resource allocations
                for (alloc_idx, resource) in w.resources.iter().enumerate() {
                    let gpu_alloc = resource
                        .gpus
                        .iter()
                        .map(|g| format!("{}:{:.2}", g.offset, g.used_fraction))
                        .collect::<Vec<_>>()
                        .join(", ");

                    let worker_id = if w.resources.len() > 1 {
                        format!("{}-{}", w.id, alloc_idx)
                    } else {
                        w.id.clone()
                    };

                    table.add_row(vec![
                        stage_idx.to_string(),
                        "Deleted".to_string(),
                        worker_id,
                        resource.node.clone(),
                        format!("{:.2}", resource.cpus.to_num::<f32>()),
                        gpu_alloc,
                    ]);
                }
            }
        }
        write!(f, "{}", table)
    }
}

// --------------------
// ProblemState + Solution bundle
// --------------------

/// Represents both the current state and solution of the allocation problem.
///
/// This class combines both the current state of the system and the proposed
/// changes, allowing for complete context when reviewing allocation decisions.
///
/// # Attributes
/// * `state` - Current state of the system.
/// * `result` - Proposed changes to the system.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ProblemStateAndSolution {
    pub state: ProblemState,
    pub result: Solution,
}

impl Display for ProblemStateAndSolution {
    /// Returns a formatted string representation of both state and solution.
    ///
    /// # Returns
    /// A string containing both the current state and proposed changes.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "Problem State and Result:")?;
        writeln!(f, "State:")?;
        writeln!(f, "{}", self.state)?;
        writeln!(f, "Result:")?;
        write!(f, "{}", self.result)
    }
}

// --------------------
// Measurements
// --------------------

/// Contains timing measurements for a single task.
///
/// # Attributes
/// * `start_time` - Time when the task started processing.
/// * `end_time` - Time when the task completed processing.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Copy)]
pub struct TaskMeasurement {
    pub start_time: f64,
    pub end_time: f64,
    pub num_returns: u32,
}

#[pymethods]
impl TaskMeasurement {
    #[new]
    pub fn new(start_time: f64, end_time: f64, num_returns: u32) -> Self {
        Self {
            start_time,
            end_time,
            num_returns,
        }
    }
    /// Calculates the duration of the task.
    ///
    /// # Returns
    /// The duration of the task in seconds.
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// Contains measurements for a single stage.
///
/// # Attributes
/// * `task_measurements` - List of measurements for individual tasks in this stage.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Default)]
pub struct StageMeasurements {
    pub task_measurements: Vec<TaskMeasurement>,
}

#[pymethods]
impl StageMeasurements {
    #[new]
    pub fn new(task_measurements: Vec<TaskMeasurement>) -> Self {
        Self { task_measurements }
    }
}

/// Contains measurements across multiple stages.
///
/// These measurements can be used by the auto-scaling algorithm to estimate
/// the processing rate of the stages.
///
/// # Attributes
/// * `time` - Timestamp when these measurements were taken.
/// * `stages` - List of measurements for each stage.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct Measurements {
    pub time: f64,
    pub stages: Vec<StageMeasurements>,
}

#[pymethods]
impl Measurements {
    #[new]
    pub fn new(time: f64, stages: Vec<StageMeasurements>) -> Self {
        Self { time, stages }
    }
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<Problem>()?
        .add_class::<ProblemState>()?
        .add_class::<ProblemStage>()?
        .add_class::<ProblemStageState>()?
        .add_class::<ProblemWorkerGroupState>()?
        .add_class::<Solution>()?
        .add_class::<StageSolution>()?
        .add_class::<TaskMeasurement>()?
        .add_class::<StageMeasurements>()?
        .add_class::<Measurements>()?
        .finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Boundary tests for the Python-facing data structures.
    //!
    //! These tests pin the PyO3 contract: malformed inputs must surface
    //! as recoverable Python exceptions (returned as `Err(PyErr)`),
    //! never as Rust panics propagated through PyO3 (which abort the
    //! autoscaler thread). Tests check `is_err()` only; formatting the
    //! `PyErr` value requires an initialized Python interpreter that
    //! the Rust unit-test harness does not bring up.
    use super::*;
    use crate::pipelines::private::scheduling::resources as rds;

    fn make_single_alloc(node: &str, cpus: f32) -> rds::WorkerResources {
        rds::WorkerResources {
            node: node.to_string(),
            cpus: rds::FixedUtil::from_num(cpus),
            gpus: Vec::new(),
        }
    }

    #[test]
    fn to_worker_returns_err_for_multi_allocation_state() {
        // ProblemWorkerGroupState built with two allocations represents
        // an SPMD worker group; converting it to a flat single-allocation
        // Worker would silently drop one allocation. Previously panicked
        // through PyO3 and aborted the autoscaler thread; now must
        // surface as a recoverable error.
        let state = ProblemWorkerGroupState {
            id: "w0".to_string(),
            resources: vec![
                make_single_alloc("node0", 1.0),
                make_single_alloc("node1", 1.0),
            ],
            num_used_slots: 0,
        };

        assert!(
            state.to_worker("stage_a".to_string()).is_err(),
            "multi-allocation conversion must error, not panic"
        );
    }

    #[test]
    fn to_worker_returns_err_for_empty_allocation_state() {
        // An empty resources vector previously triggered an out-of-bounds
        // index panic on `self.resources[0]`. The new contract is a
        // recoverable error.
        let state = ProblemWorkerGroupState {
            id: "w_empty".to_string(),
            resources: Vec::new(),
            num_used_slots: 0,
        };

        assert!(
            state.to_worker("stage_a".to_string()).is_err(),
            "empty-allocation conversion must error, not panic"
        );
    }

    #[test]
    fn to_worker_succeeds_for_single_allocation_state() {
        // Happy path: a single allocation produces a single-allocation
        // Worker carrying the same id/stage name/allocation.
        let state = ProblemWorkerGroupState {
            id: "w_solo".to_string(),
            resources: vec![make_single_alloc("node0", 2.0)],
            num_used_slots: 3,
        };

        let worker = state
            .to_worker("stage_b".to_string())
            .expect("single allocation must succeed");

        assert_eq!(worker.id, "w_solo");
        assert_eq!(worker.stage_name, "stage_b");
        assert_eq!(worker.allocation.node, "node0");
    }

    #[test]
    fn deserialize_returns_err_for_malformed_json() {
        // Malformed JSON used to panic through
        // `serde_json::from_str().unwrap()`; the new contract is a
        // recoverable error.
        assert!(
            ProblemWorkerGroupState::deserialize("not-json").is_err(),
            "malformed JSON must error, not panic"
        );
    }

    #[test]
    fn deserialize_returns_err_for_empty_string() {
        assert!(
            ProblemWorkerGroupState::deserialize("").is_err(),
            "empty input must error, not panic"
        );
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        // Happy path covering the renamed PyResult signatures: serialize
        // returns Ok(String) for a flat owned struct; deserialize round
        // trips the value back to the original state.
        let original = ProblemWorkerGroupState {
            id: "w_round".to_string(),
            resources: vec![make_single_alloc("node0", 1.5)],
            num_used_slots: 7,
        };

        let serialized = original.serialize().expect("serialize must succeed");
        let restored =
            ProblemWorkerGroupState::deserialize(&serialized).expect("deserialize must succeed");

        assert_eq!(restored.id, original.id);
        assert_eq!(restored.num_used_slots, original.num_used_slots);
        assert_eq!(restored.resources.len(), original.resources.len());
    }
}
