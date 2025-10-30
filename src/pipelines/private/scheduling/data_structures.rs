// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/// * `workers` - List of workers in this group.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemWorkerGroupState {
    pub id: String,
    pub resources: Vec<resources::WorkerResources>,
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
        }
    }

    #[staticmethod]
    pub fn make_from_worker_state(state: resources::Worker) -> Self {
        Self {
            id: state.id,
            resources: vec![state.allocation],
        }
    }

    #[new]
    pub fn py_new(id: String, resources: Vec<resources::WorkerResources>) -> Self {
        Self { id, resources }
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

    pub fn to_worker(&self, stage_name: String) -> resources::Worker {
        if self.resources.len() > 1 {
            panic!(
                "Cannot convert ProblemWorkerGroupState to Worker if there are multiple allocations"
            );
        }
        resources::Worker {
            id: self.id.clone(),
            stage_name,
            allocation: self.resources[0].clone(),
        }
    }

    pub fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    #[staticmethod]
    pub fn deserialize(data: &str) -> Self {
        serde_json::from_str(data).unwrap()
    }
}

/// Represents the current state of a stage including its workers.
///
/// # Attributes
/// * `stage_name` - Name identifier for this stage.
/// * `workers` - List of workers currently assigned to this stage.
/// * `slots_per_worker` - Number of task slots available per worker.
/// * `is_finished` - Boolean indicating if this stage has completed processing.
#[pyclass(get_all, set_all)]
#[derive(Debug, Clone)]
pub struct ProblemStageState {
    pub stage_name: String,
    pub worker_groups: Vec<ProblemWorkerGroupState>,
    pub slots_per_worker: usize,
    pub is_finished: bool,
}

#[pymethods]
impl ProblemStageState {
    #[new]
    pub fn py_new(
        stage_name: String,
        worker_groups: Vec<ProblemWorkerGroupState>,
        slots_per_worker: usize,
        is_finished: bool,
    ) -> Self {
        Self {
            stage_name,
            worker_groups,
            slots_per_worker,
            is_finished,
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
#[pyclass]
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
