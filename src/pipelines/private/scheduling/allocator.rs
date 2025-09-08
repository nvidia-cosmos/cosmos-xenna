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

//! Resource allocation manager for distributed pipeline workers.
//!
//! This module provides resource allocation and tracking capabilities for a distributed
//! pipeline system. It ensures safe and efficient distribution of compute resources
//! (CPU, GPU, NVDEC, NVENC) across multiple nodes while maintaining pipeline stage
//! organization.
//!
//! The WorkerAllocator tracks both the physical allocation of resources across nodes
//! and the logical organization of workers into pipeline stages. It prevents resource
//! oversubscription and provides utilities for monitoring resource utilization.
//!
//! Typical usage:
//! ```rust
//! // Create allocator with cluster resources
//! let allocator = WorkerAllocator::new(cluster_resources, None)?;
//!
//! // Add workers for different pipeline stages
//! allocator.add_worker(Worker::new("worker1".into(), "stage1".into(), resources))?;
//! allocator.add_worker(Worker::new("worker2".into(), "stage1".into(), resources))?;
//!
//! // Monitor resource usage
//! println!("{}", allocator.make_detailed_utilization_table());
//! ```

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::utils::module_builders::ImportablePyModuleBuilder;

use super::resources::{AllocationError, ClusterResources, Worker};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

impl From<WorkerAllocatorError> for PyErr {
    fn from(err: WorkerAllocatorError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Container for workers allocated to a specific node.
///
/// # Attributes
/// * `by_id` - Dictionary mapping worker IDs to Worker instances for this node.
#[derive(Debug, Default, Clone)]
pub struct NodeWorkers {
    pub by_id: HashMap<String, Worker>,
}

/// Container for workers assigned to a specific pipeline stage.
///
/// # Attributes
/// * `by_id` - Dictionary mapping worker IDs to Worker instances for this stage.
#[derive(Debug, Default, Clone)]
pub struct StageWorkers {
    pub by_id: HashMap<String, Worker>,
}

#[derive(Error, Debug)]
pub enum WorkerAllocatorError {
    #[error("Worker id already exists: {0}")]
    DuplicateWorkerId(String),
    #[error("Worker not found: {0}")]
    WorkerNotFound(String),
    #[error("Allocation error: {0}")]
    Allocation(#[from] AllocationError),
}

/// Manages resource allocation for distributed pipeline workers across nodes.
///
/// This class is responsible for:
/// 1. Tracking available compute resources (CPU, GPU, NVDEC, NVENC) across nodes
/// 2. Managing worker allocation to both nodes and pipeline stages
/// 3. Preventing resource oversubscription
/// 4. Providing utilization monitoring and reporting
///
/// The allocator maintains both physical (node-based) and logical (stage-based)
/// views of worker allocation to support pipeline execution while ensuring
/// safe resource usage.
///
/// # Attributes
/// * `num_nodes` - Number of nodes in the cluster.
/// * `totals` - Total available resources across all nodes.
/// * `available_resources` - Currently unallocated resources across all nodes.
#[pyclass]
#[derive(Debug, Clone)]
pub struct WorkerAllocator {
    pub cluster_resources: ClusterResources,
    nodes_state: HashMap<String, NodeWorkers>,
    pub stages_state: HashMap<String, StageWorkers>,
}

impl WorkerAllocator {
    /// Initialize the WorkerAllocator.
    ///
    /// # Arguments
    /// * `cluster_resources` - Available resources across all nodes.
    /// * `workers` - Optional list of pre-existing workers to track.
    pub fn new(
        cluster_resources: ClusterResources,
        workers: Option<Vec<Worker>>,
    ) -> Result<Self, WorkerAllocatorError> {
        let mut nodes_state: HashMap<String, NodeWorkers> = HashMap::new();
        for node_id in cluster_resources.nodes.keys() {
            nodes_state.insert(node_id.clone(), NodeWorkers::default());
        }

        let mut this = Self {
            cluster_resources,
            nodes_state,
            stages_state: HashMap::new(),
        };

        if let Some(initial_workers) = workers {
            this.add_workers(initial_workers.into_iter())?;
        }
        Ok(this)
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes_state.len()
    }

    fn ensure_worker_id_absent(&self, worker_id: &str) -> Result<(), WorkerAllocatorError> {
        if self.get_worker_if_exists(worker_id).is_some() {
            return Err(WorkerAllocatorError::DuplicateWorkerId(
                worker_id.to_string(),
            ));
        }
        Ok(())
    }

    /// Adds a single worker to the allocation tracking.
    ///
    /// The worker will be tracked both by its assigned node and pipeline stage.
    /// Validates resource allocation and prevents oversubscription.
    ///
    /// # Arguments
    /// * `worker` - Worker instance to add.
    ///
    /// # Errors
    /// Returns `WorkerAllocatorError::DuplicateWorkerId` if worker ID already exists.
    /// Returns `WorkerAllocatorError::OverAllocated` if adding worker would exceed available resources.
    pub fn add_worker(&mut self, worker: Worker) -> Result<(), WorkerAllocatorError> {
        self.ensure_worker_id_absent(&worker.id)?;

        // Allocate resources on the node
        self.cluster_resources.allocate(&worker.allocation)?;

        // Track via node state
        let node_state = self
            .nodes_state
            .get_mut(&worker.allocation.node)
            .expect("node exists");
        node_state.by_id.insert(worker.id.clone(), worker.clone());

        // Track in stage index
        self.stages_state
            .entry(worker.stage_name.clone())
            .or_default()
            .by_id
            .insert(worker.id.clone(), worker);
        Ok(())
    }

    /// Adds multiple workers to allocation tracking.
    ///
    /// # Arguments
    /// * `workers` - Iterable of Worker instances to add.
    ///
    /// # Errors
    /// Returns `WorkerAllocatorError::DuplicateWorkerId` if any worker ID already exists.
    /// Returns `WorkerAllocatorError::OverAllocated` if adding workers would exceed available resources.
    pub fn add_workers<I>(&mut self, workers: I) -> Result<(), WorkerAllocatorError>
    where
        I: IntoIterator<Item = Worker>,
    {
        // Collect workers so we can pre-validate and also roll back if needed
        let workers_vec: Vec<Worker> = workers.into_iter().collect();

        // Fast duplicate detection within the provided batch
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for w in &workers_vec {
            if !seen_ids.insert(w.id.clone()) {
                return Err(WorkerAllocatorError::DuplicateWorkerId(w.id.clone()));
            }
        }

        // Ensure none of these IDs already exist in the allocator
        for w in &workers_vec {
            self.ensure_worker_id_absent(&w.id)?;
        }

        // Try to add each worker; if any fail, roll back previously added ones
        let mut added_ids: Vec<String> = Vec::new();
        for w in workers_vec {
            let id = w.id.clone();
            if let Err(e) = self.add_worker(w) {
                // Rollback already-added workers
                for added_id in added_ids.iter() {
                    let _ = self.remove_worker(added_id);
                }
                return Err(e);
            }
            added_ids.push(id);
        }

        Ok(())
    }

    /// Retrieves a worker by ID.
    ///
    /// # Arguments
    /// * `worker_id` - ID of the worker to retrieve.
    ///
    /// # Returns
    /// The requested Worker instance.
    ///
    /// # Errors
    /// Returns `WorkerAllocatorError::WorkerNotFound` if no worker exists with the given ID.
    pub fn get_worker(&self, worker_id: &str) -> Result<Worker, WorkerAllocatorError> {
        self.get_worker_if_exists(worker_id)
            .ok_or_else(|| WorkerAllocatorError::WorkerNotFound(worker_id.to_string()))
    }

    /// Return the worker or None, if it does not exist.
    pub fn get_worker_if_exists(&self, worker_id: &str) -> Option<Worker> {
        for node in self.nodes_state.values() {
            if let Some(found) = node.by_id.get(worker_id) {
                return Some(found.clone());
            }
        }
        None
    }

    pub fn remove_worker(&mut self, worker_id: &str) -> Result<Worker, WorkerAllocatorError> {
        let worker = self.get_worker(worker_id)?;
        self.cluster_resources
            .release_allocation(&worker.allocation)?;

        let node_state = self
            .nodes_state
            .get_mut(&worker.allocation.node)
            .expect("node exists");
        node_state.by_id.remove(worker_id);

        let stage_state = self
            .stages_state
            .get_mut(&worker.stage_name)
            .expect("stage exists");
        stage_state.by_id.remove(worker_id);

        Ok(worker)
    }

    pub fn delete_workers(&mut self, worker_ids: &[String]) -> Result<(), WorkerAllocatorError> {
        // Delete each worker; if any fail, re-add those already deleted and return the error
        let mut deleted_workers: Vec<Worker> = Vec::new();
        for worker_id in worker_ids {
            match self.remove_worker(worker_id) {
                Ok(w) => deleted_workers.push(w),
                Err(e) => {
                    // Rollback previously deleted workers
                    for w in deleted_workers.into_iter() {
                        let _ = self.add_worker(w);
                    }
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub fn get_mut_cluster_resources(&mut self) -> &mut ClusterResources {
        &mut self.cluster_resources
    }

    pub fn get_cluster_resources(&self) -> &ClusterResources {
        &self.cluster_resources
    }

    pub fn get_workers(&self) -> Vec<Worker> {
        let mut out = Vec::new();
        for stage in self.stages_state.values() {
            out.extend(stage.by_id.values().cloned());
        }
        out
    }

    pub fn get_num_workers_per_stage(&self) -> HashMap<String, usize> {
        let mut out: HashMap<String, usize> = HashMap::new();
        for (stage, workers) in &self.stages_state {
            out.insert(stage.clone(), workers.by_id.len());
        }
        out
    }

    /// Returns worker IDs sorted by their node's CPU utilization.
    ///
    /// Useful for load balancing and resource optimization decisions.
    ///
    /// # Arguments
    /// * `workers_ids_to_consider` - Optional set of worker IDs to limit consideration to.
    ///
    /// # Returns
    /// List of tuples (cpu_utilization, worker_id) sorted by utilization.
    pub fn worker_ids_and_node_cpu_utilizations(
        &self,
        workers_ids_to_consider: Option<&HashSet<String>>,
    ) -> Vec<(f32, String)> {
        let node_utils = self.calculate_node_cpu_utilizations();
        let mut out: Vec<(f32, String)> = Vec::new();

        for (node_id, node_workers) in &self.nodes_state {
            let util = node_utils.get(node_id).copied().unwrap_or(0.0);
            for worker_id in node_workers.by_id.keys() {
                if workers_ids_to_consider
                    .map(|s| s.contains(worker_id))
                    .unwrap_or(true)
                {
                    out.push((util, worker_id.clone()));
                }
            }
        }
        out
    }

    pub fn calculate_lowest_allocated_node_by_cpu(&self) -> Option<String> {
        let utils = self.calculate_node_cpu_utilizations();
        utils
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k)
    }

    /// Calculate the current CPU utilization for each node.
    ///
    /// # Returns
    /// HashMap mapping node IDs to CPU utilization ratios for each node.
    pub fn calculate_node_cpu_utilizations(&self) -> HashMap<String, f32> {
        let mut utilizations: HashMap<String, f32> = HashMap::new();

        for (node_id, node) in self.cluster_resources.nodes.iter() {
            let utilization = if node.total_cpus > 0.0 {
                node.used_cpus.to_num::<f32>() / node.total_cpus.to_num::<f32>()
            } else {
                0.0
            };
            utilizations.insert(node_id.clone(), utilization);
        }
        utilizations
    }

    /// Generates a human-readable table showing resource utilization.
    ///
    /// Creates an ASCII table showing CPU, GPU, NVDEC, and NVENC utilization
    /// for each node in the cluster. Uses bar charts to visualize usage levels.
    ///
    /// # Returns
    /// Formatted string containing the utilization table.
    pub fn make_detailed_utilization_table(&self) -> String {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![Cell::new("Component"), Cell::new("Utilization")]);

        for (node_index, node_resources) in self.cluster_resources.nodes.values().enumerate() {
            let cpu_bar = create_bar_chart(
                node_resources.used_cpus.to_num::<f32>(),
                node_resources.total_cpus.to_num::<f32>(),
                20,
            );
            table.add_row(vec![
                Cell::new(format!("Node {}", node_index)),
                Cell::new(format!("CPUs: {}", cpu_bar)),
            ]);

            for (i, gpu) in node_resources.gpus.iter().enumerate() {
                let gpu_bar = create_bar_chart(gpu.used_fraction.to_num::<f32>(), 1.0, 20);
                table.add_row(vec![
                    Cell::new(format!("  GPU {}", i)),
                    Cell::new(format!("GPU: {}", gpu_bar)),
                ]);
            }
        }

        table.to_string()
    }
}

/// Creates an ASCII bar chart showing resource utilization.
///
/// # Arguments
/// * `used` - Amount of resource currently in use.
/// * `total` - Total amount of resource available.
/// * `width` - Width of the bar chart in characters.
///
/// # Returns
/// String representation of a bar chart showing utilization.
fn create_bar_chart(used: f32, total: f32, width: usize) -> String {
    if total <= 0.0 {
        return format!("[{}] {used:.2}/{total:.2}", "-".repeat(width));
    }
    let filled = ((used / total).clamp(0.0, 1.0) * width as f32) as usize;
    let bar = format!(
        "[{}{}] {used:.2}/{total:.2}",
        "#".repeat(filled),
        "-".repeat(width - filled)
    );
    bar
}

// --------------------
// PyO3 methods on WorkerAllocator
// --------------------

#[pymethods]
impl WorkerAllocator {
    #[new]
    pub fn py_new(cluster_resources: ClusterResources) -> Self {
        // Initialize with no workers; should not fail
        Self::new(cluster_resources, None).expect("failed to initialize WorkerAllocator")
    }

    // #[pyo3(name = "totals")]
    // pub fn py_totals(&self) -> ClusterResources {
    //     self.totals().clone()
    // }

    // #[pyo3(name = "available_resources")]
    // pub fn py_available_resources(&self) -> ClusterResources {
    //     self.available_resources().clone()
    // }

    pub fn get_gpu_index(&self, node_id: &str, gpu_offset: usize) -> usize {
        self.cluster_resources
            .nodes
            .get(node_id)
            .expect("node not found")
            .gpus
            .get(gpu_offset)
            .expect("gpu not found")
            .index as usize
    }

    /// Retrieves all workers assigned to a pipeline stage.
    ///
    /// # Arguments
    /// * `stage_name` - Name of the pipeline stage.
    ///
    /// # Returns
    /// List of Worker instances assigned to the stage.
    pub fn get_workers_in_stage(&self, stage_name: &str) -> Vec<Worker> {
        self.stages_state
            .get(stage_name)
            .map(|s| s.by_id.values().cloned().collect())
            .unwrap_or_default()
    }

    #[pyo3(name = "num_nodes")]
    pub fn py_num_nodes(&self) -> usize {
        self.num_nodes()
    }

    #[pyo3(name = "add_worker")]
    pub fn py_add_worker(&mut self, worker: Worker) -> PyResult<()> {
        self.add_worker(worker)?;
        Ok(())
    }

    #[pyo3(name = "add_workers")]
    pub fn py_add_workers(&mut self, workers: Vec<Worker>) -> PyResult<()> {
        self.add_workers(workers)?;
        Ok(())
    }

    #[pyo3(name = "remove_worker")]
    pub fn py_remove_worker(&mut self, worker_id: String) -> PyResult<Worker> {
        self.remove_worker(&worker_id).map_err(Into::into)
    }

    #[pyo3(name = "delete_workers")]
    pub fn py_delete_workers(&mut self, worker_ids: Vec<String>) -> PyResult<()> {
        self.delete_workers(&worker_ids).map_err(Into::into)
    }

    #[pyo3(name = "get_worker")]
    pub fn py_get_worker(&self, worker_id: String) -> Option<Worker> {
        self.get_worker_if_exists(&worker_id)
    }

    #[pyo3(name = "get_workers")]
    pub fn py_get_workers(&self) -> Vec<Worker> {
        self.get_workers()
    }

    #[pyo3(name = "get_num_workers_per_stage")]
    pub fn py_get_num_workers_per_stage(&self) -> HashMap<String, usize> {
        self.get_num_workers_per_stage()
    }

    #[pyo3(name = "calculate_lowest_allocated_node_by_cpu")]
    pub fn py_calculate_lowest_allocated_node_by_cpu(&self) -> Option<String> {
        self.calculate_lowest_allocated_node_by_cpu()
    }

    #[pyo3(name = "worker_ids_and_node_cpu_utilizations")]
    pub fn py_worker_ids_and_node_cpu_utilizations(
        &self,
        workers_ids_to_consider: Option<Vec<String>>,
    ) -> Vec<(f32, String)> {
        let set_opt = workers_ids_to_consider
            .map(|v| v.into_iter().collect::<std::collections::HashSet<_>>());
        self.worker_ids_and_node_cpu_utilizations(set_opt.as_ref())
    }

    #[pyo3(name = "make_detailed_utilization_table")]
    pub fn py_make_detailed_utilization_table(&self) -> String {
        self.make_detailed_utilization_table()
    }
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<WorkerAllocator>()?
        .finish();
    Ok(())
}

// --------------------
// Tests
// --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::private::scheduling::resources as rds;
    use std::collections::HashMap;

    fn make_simple_cluster() -> rds::ClusterResources {
        let mut nodes: HashMap<String, rds::NodeResources> = HashMap::new();
        let node0 = rds::NodeResources {
            used_cpus: rds::FixedUtil::ZERO,
            total_cpus: rds::FixedUtil::from_num(8.0),
            gpus: vec![
                rds::GpuResources {
                    index: 0,
                    uuid_: uuid::Uuid::new_v4(),
                    used_fraction: rds::FixedUtil::ZERO,
                },
                rds::GpuResources {
                    index: 1,
                    uuid_: uuid::Uuid::new_v4(),
                    used_fraction: rds::FixedUtil::ZERO,
                },
            ],
            name: None,
        };
        let node1 = rds::NodeResources {
            used_cpus: rds::FixedUtil::ZERO,
            total_cpus: rds::FixedUtil::from_num(4.0),
            gpus: vec![rds::GpuResources {
                index: 0,
                uuid_: uuid::Uuid::new_v4(),
                used_fraction: rds::FixedUtil::ZERO,
            }],
            name: None,
        };
        nodes.insert("0".to_string(), node0);
        nodes.insert("1".to_string(), node1);
        rds::ClusterResources::new(Some(nodes))
    }

    fn make_allocator() -> WorkerAllocator {
        WorkerAllocator::new(make_simple_cluster(), None).expect("init allocator")
    }

    fn wr(node: &str, cpus: f32, gpus: Vec<(usize, f32)>) -> rds::WorkerResources {
        let gpu_allocs: Vec<rds::GPUAllocation> = gpus
            .into_iter()
            .map(|(idx, frac)| rds::GPUAllocation {
                index: idx,
                used_fraction: rds::FixedUtil::from_num(frac),
            })
            .collect();
        rds::WorkerResources {
            node: node.to_string(),
            cpus: rds::FixedUtil::from_num(cpus),
            gpus: gpu_allocs,
        }
    }

    #[test]
    fn test_init() {
        let allocator = make_allocator();
        assert_eq!(allocator.num_nodes(), 2);
    }

    #[test]
    fn test_add_worker() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        allocator.add_worker(worker.clone()).expect("add");
        let fetched = allocator.get_worker("w1").expect("get");
        assert_eq!(fetched.id, "w1");
        let map = allocator.get_num_workers_per_stage();
        assert_eq!(map.get("stage1").copied().unwrap_or_default(), 1);
    }

    #[test]
    fn test_add_workers() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)])),
            rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add workers");
        assert!(allocator.get_worker("w1").is_ok());
        assert!(allocator.get_worker("w2").is_ok());
    }

    #[test]
    fn test_delete_workers() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)])),
            rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add workers");
        allocator
            .delete_workers(&vec!["w1".to_string()])
            .expect("delete workers");
        assert!(allocator.get_worker("w1").is_err());
        assert!(allocator.get_worker("w2").is_ok());
    }

    #[test]
    fn test_delete_non_existent_worker() {
        let mut allocator = make_allocator();
        let err = allocator
            .delete_workers(&vec!["non_existent".to_string()])
            .unwrap_err();
        match err {
            WorkerAllocatorError::WorkerNotFound(id) => assert_eq!(id, "non_existent"),
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_make_detailed_utilization_table() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)])),
            rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add workers");
        let table = allocator.make_detailed_utilization_table();
        assert!(table.contains("Node 0"));
        assert!(table.contains("Node 1"));
    }

    #[test]
    fn test_overallocation() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 10.0, vec![]));
        let err = allocator.add_worker(worker).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NotEnoughResources { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_overallocation_single_gpu() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.5)])),
            rds::Worker::new("w2".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.7)])),
        ];
        let err = allocator.add_workers(workers).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NotEnoughResources { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_overallocation_single_gpu_separate_calls() {
        let mut allocator = make_allocator();
        let w1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.5)]));
        let w2 = rds::Worker::new("w2".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.7)]));
        allocator.add_worker(w1).expect("add first");
        let err = allocator.add_worker(w2).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NotEnoughResources { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_adding_workers_with_existing_ids_raises() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new(
                "1".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "0".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GPUAllocation::new(0, 1.0)],
                },
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GPUAllocation::new(0, 0.7)],
                },
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GPUAllocation::new(0, 0.31)],
                },
            ),
        ];
        let err = allocator.add_workers(workers).unwrap_err();
        match err {
            WorkerAllocatorError::DuplicateWorkerId(id) => assert_eq!(id, "2"),
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_overallocation_with_fractional_resources() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new(
                "1".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "0".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GPUAllocation::new(0, 1.0)],
                },
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GPUAllocation::new(0, 0.7)],
                },
            ),
            rds::Worker::new(
                "3".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GPUAllocation::new(0, 0.31)],
                },
            ),
        ];
        let err = allocator.add_workers(workers).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NotEnoughResources { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_gpu_allocation_limit() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new(
            "w1".into(),
            "stage1".into(),
            rds::WorkerResources {
                node: "0".into(),
                cpus: rds::FixedUtil::from_num(1.0),
                gpus: vec![rds::GPUAllocation::new(0, 1.5)],
            },
        );
        let err = allocator.add_worker(worker).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NotEnoughResources { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_get_worker() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        allocator.add_worker(worker).expect("add");
        let retrieved = allocator.get_worker("w1").expect("get");
        assert_eq!(retrieved.id, "w1");
        assert_eq!(retrieved.stage_name, "stage1");
    }

    #[test]
    fn test_get_nonexistent_worker() {
        let allocator = make_allocator();
        let err = allocator.get_worker("nonexistent").unwrap_err();
        match err {
            WorkerAllocatorError::WorkerNotFound(id) => assert_eq!(id, "nonexistent"),
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_delete_worker() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        allocator.add_worker(worker).expect("add");
        allocator.remove_worker("w1").expect("delete");
        assert!(allocator.get_worker("w1").is_err());
    }

    #[test]
    fn test_worker_ids_and_node_cpu_utilizations() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 4.0, vec![])),
            rds::Worker::new("w2".into(), "stage1".into(), wr("0", 2.0, vec![])),
            rds::Worker::new("w3".into(), "stage2".into(), wr("1", 2.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add");
        let v = allocator.worker_ids_and_node_cpu_utilizations(None);
        assert_eq!(v.len(), 3);
        let ids: std::collections::HashSet<_> = v.iter().map(|(_, id)| id.as_str()).collect();
        assert!(ids.contains("w1") && ids.contains("w2") && ids.contains("w3"));
    }

    #[test]
    fn test_worker_ids_and_node_cpu_utilizations_with_subset() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 4.0, vec![])),
            rds::Worker::new("w2".into(), "stage1".into(), wr("0", 2.0, vec![])),
            rds::Worker::new("w3".into(), "stage2".into(), wr("1", 2.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add");
        let subset: std::collections::HashSet<String> =
            ["w1".to_string(), "w3".to_string()].into_iter().collect();
        let v = allocator.worker_ids_and_node_cpu_utilizations(Some(&subset));
        assert_eq!(v.len(), 2);
        let ids: std::collections::HashSet<_> = v.iter().map(|(_, id)| id.as_str()).collect();
        assert!(ids.contains("w1") && ids.contains("w3"));
    }

    #[test]
    fn test_calculate_node_cpu_utilizations() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 4.0, vec![])),
            rds::Worker::new("w2".into(), "stage2".into(), wr("1", 2.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add");
        let utils = allocator.calculate_node_cpu_utilizations();
        assert_eq!(utils.len(), 2);
        let u0 = utils.get("0").copied().unwrap_or_default();
        let u1 = utils.get("1").copied().unwrap_or_default();
        assert!((u0 - 0.5).abs() < 1e-6);
        assert!((u1 - 0.5).abs() < 1e-6);
    }
}
