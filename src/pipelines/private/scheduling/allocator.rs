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
//! use _cosmos_xenna::pipelines::private::scheduling::allocator::WorkerAllocator;
//! use _cosmos_xenna::pipelines::private::scheduling::resources::{ClusterResources, Worker, WorkerGroup, WorkerResources, FixedUtil};
//! use std::collections::HashMap;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create cluster resources
//!     let cluster_resources = ClusterResources {
//!         nodes: HashMap::new(),
//!     };
//!
//!     // Create allocator with cluster resources
//!     let mut allocator = WorkerAllocator::new(cluster_resources, None)?;
//!
//!     // Create worker resources
//!     let resources = WorkerResources {
//!         node: "node1".to_string(),
//!         cpus: FixedUtil::from_num(1.0),
//!         gpus: vec![],
//!     };
//!
//!     // Add workers for different pipeline stages
//!     let worker1 = Worker::new("worker1".into(), "stage1".into(), resources.clone());
//!     let worker2 = Worker::new("worker2".into(), "stage1".into(), resources);
//!     allocator.add_worker(WorkerGroup::from_worker(worker1))?;
//!     allocator.add_worker(WorkerGroup::from_worker(worker2))?;
//!
//!     // Monitor resource usage
//!     println!("{}", allocator.make_detailed_utilization_table());
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;

use thiserror::Error;

use crate::utils::module_builders::ImportablePyModuleBuilder;

use super::resources::{AllocationError, ClusterResources, WorkerGroup};
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
    pub by_id: HashMap<String, WorkerGroup>,
}

/// Container for workers assigned to a specific pipeline stage.
///
/// # Attributes
/// * `by_id` - Dictionary mapping worker IDs to Worker instances for this stage.
#[derive(Debug, Default, Clone)]
pub struct StageWorkers {
    pub by_id: HashMap<String, WorkerGroup>,
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
        workers: Option<Vec<WorkerGroup>>,
    ) -> Result<Self, WorkerAllocatorError> {
        let mut this = Self {
            cluster_resources,
            stages_state: HashMap::new(),
        };

        if let Some(initial_workers) = workers {
            this.add_workers(initial_workers.into_iter())?;
        }
        Ok(this)
    }

    pub fn num_nodes(&self) -> usize {
        self.cluster_resources.nodes.len()
    }

    fn ensure_worker_id_absent(&self, worker_id: &str) -> Result<(), WorkerAllocatorError> {
        // TODO: searrcht throught the states
        for stage in self.stages_state.values() {
            if stage.by_id.contains_key(worker_id) {
                return Err(WorkerAllocatorError::DuplicateWorkerId(
                    worker_id.to_string(),
                ));
            }
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
    pub fn add_worker(&mut self, worker: WorkerGroup) -> Result<(), WorkerAllocatorError> {
        self.ensure_worker_id_absent(&worker.id)?;

        // Allocate resources on the node(s)
        self.cluster_resources
            .allocate_multiple(&worker.allocations)?;

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
        I: IntoIterator<Item = WorkerGroup>,
    {
        // Collect workers so we can pre-validate and also roll back if needed
        let workers_vec: Vec<WorkerGroup> = workers.into_iter().collect();

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
    /// The requested WorkerGroup instance.
    ///
    /// # Errors
    /// Returns `WorkerAllocatorError::WorkerNotFound` if no worker exists with the given ID.
    pub fn get_worker(&self, worker_id: &str) -> Result<&WorkerGroup, WorkerAllocatorError> {
        for stage in self.stages_state.values() {
            if let Some(found) = stage.by_id.get(worker_id) {
                return Ok(found);
            }
        }
        Err(WorkerAllocatorError::WorkerNotFound(worker_id.to_string()))
    }

    pub fn remove_worker(&mut self, worker_id: &str) -> Result<WorkerGroup, WorkerAllocatorError> {
        // Get worker info without borrowing self mutably
        let (stage_name, allocations) = {
            let worker: &WorkerGroup = self.get_worker(worker_id)?;
            (worker.stage_name.clone(), worker.allocations.clone())
        };

        // Release allocations
        self.cluster_resources.release_allocations(&allocations)?;

        // Remove from stage state
        let stage_state = self
            .stages_state
            .get_mut(&stage_name)
            .expect("stage exists");
        Ok(stage_state.by_id.remove(worker_id).expect("worker exists"))
    }

    pub fn delete_workers(&mut self, worker_ids: &[String]) -> Result<(), WorkerAllocatorError> {
        // Delete each worker; if any fail, re-add those already deleted and return the error
        let mut deleted_workers: Vec<WorkerGroup> = Vec::new();
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

    pub fn get_workers(&self) -> Vec<WorkerGroup> {
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
    pub fn get_workers_in_stage(&self, stage_name: &str) -> Vec<WorkerGroup> {
        self.stages_state
            .get(stage_name)
            .map(|s| s.by_id.values().cloned().collect())
            .unwrap_or_default()
    }

    #[pyo3(name = "add_worker")]
    pub fn py_add_worker(&mut self, worker: WorkerGroup) -> PyResult<()> {
        self.add_worker(worker)?;
        Ok(())
    }

    #[pyo3(name = "remove_worker")]
    pub fn py_remove_worker(&mut self, worker_id: String) -> PyResult<WorkerGroup> {
        self.remove_worker(&worker_id).map_err(Into::into)
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
        rds::ClusterResources { nodes: nodes }
    }

    fn make_allocator() -> WorkerAllocator {
        WorkerAllocator::new(make_simple_cluster(), None).expect("init allocator")
    }

    fn wr(node: &str, cpus: f32, gpus: Vec<(usize, f32)>) -> rds::WorkerResources {
        let gpu_allocs: Vec<rds::GpuAllocation> = gpus
            .into_iter()
            .map(|(idx, frac)| rds::GpuAllocation {
                offset: idx,
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
        let worker_group = rds::WorkerGroup::from_worker(worker);
        allocator.add_worker(worker_group.clone()).expect("add");
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
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        allocator.add_workers(worker_groups).expect("add workers");
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
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        allocator.add_workers(worker_groups).expect("add workers");
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
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        allocator.add_workers(worker_groups).expect("add workers");
        let table = allocator.make_detailed_utilization_table();
        assert!(table.contains("Node 0"));
        assert!(table.contains("Node 1"));
    }

    #[test]
    fn test_overallocation() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 10.0, vec![]));
        let worker_group = rds::WorkerGroup::from_worker(worker);
        let err = allocator.add_worker(worker_group).unwrap_err();
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
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        let err = allocator.add_workers(worker_groups).unwrap_err();
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
        let w1_group = rds::WorkerGroup::from_worker(w1);
        let w2_group = rds::WorkerGroup::from_worker(w2);
        allocator.add_worker(w1_group).expect("add first");
        let err = allocator.add_worker(w2_group).unwrap_err();
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
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::from_num(1.0),
                    }],
                },
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::from_num(0.7),
                    }],
                },
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::from_num(0.31),
                    }],
                },
            ),
        ];
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        let err = allocator.add_workers(worker_groups).unwrap_err();
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
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::from_num(1.0),
                    }],
                },
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::from_num(0.7),
                    }],
                },
            ),
            rds::Worker::new(
                "3".into(),
                "1".into(),
                rds::WorkerResources {
                    node: "1".into(),
                    cpus: rds::FixedUtil::ZERO,
                    gpus: vec![rds::GpuAllocation {
                        offset: 0,
                        used_fraction: rds::FixedUtil::from_num(0.31),
                    }],
                },
            ),
        ];
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        let err = allocator.add_workers(worker_groups).unwrap_err();
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
                gpus: vec![rds::GpuAllocation {
                    offset: 0,
                    used_fraction: rds::FixedUtil::from_num(1.5),
                }],
            },
        );
        let worker_group = rds::WorkerGroup::from_worker(worker);
        let err = allocator.add_worker(worker_group).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NotEnoughResources { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_get_worker() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker_group = rds::WorkerGroup::from_worker(worker);
        allocator.add_worker(worker_group).expect("add");
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
        let worker_group = rds::WorkerGroup::from_worker(worker);
        allocator.add_worker(worker_group).expect("add");
        allocator.remove_worker("w1").expect("delete");
        assert!(allocator.get_worker("w1").is_err());
    }

    #[test]
    fn test_calculate_node_cpu_utilizations() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 4.0, vec![])),
            rds::Worker::new("w2".into(), "stage2".into(), wr("1", 2.0, vec![])),
        ];
        let worker_groups: Vec<rds::WorkerGroup> = workers
            .into_iter()
            .map(rds::WorkerGroup::from_worker)
            .collect();
        allocator.add_workers(worker_groups).expect("add");
        let utils = allocator.calculate_node_cpu_utilizations();
        assert_eq!(utils.len(), 2);
        let u0 = utils.get("0").copied().unwrap_or_default();
        let u1 = utils.get("1").copied().unwrap_or_default();
        assert!((u0 - 0.5).abs() < 1e-6);
        assert!((u1 - 0.5).abs() < 1e-6);
    }

    // ============================================================================
    // WorkerGroup-specific Tests
    // ============================================================================

    #[test]
    fn test_workergroup_creation_from_worker() {
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker_group = rds::WorkerGroup::from_worker(worker.clone());

        assert_eq!(worker_group.id, "w1");
        assert_eq!(worker_group.stage_name, "stage1");
        assert_eq!(worker_group.allocations.len(), 1);
        assert_eq!(worker_group.allocations[0].node, worker.allocation.node);
        assert_eq!(worker_group.allocations[0].cpus, worker.allocation.cpus);
        assert_eq!(worker_group.allocations[0].gpus, worker.allocation.gpus);
    }

    #[test]
    fn test_workergroup_with_multiple_allocations() {
        let allocation1 = wr("0", 2.0, vec![(0, 0.5)]);
        let allocation2 = wr("1", 1.0, vec![(0, 0.25)]);

        let worker_group = rds::WorkerGroup {
            id: "multi_node_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![allocation1.clone(), allocation2.clone()],
        };

        assert_eq!(worker_group.id, "multi_node_worker");
        assert_eq!(worker_group.stage_name, "stage1");
        assert_eq!(worker_group.allocations.len(), 2);
        assert_eq!(worker_group.allocations[0].node, "0");
        assert_eq!(worker_group.allocations[1].node, "1");
    }

    #[test]
    fn test_workergroup_allocator_add_single() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker_group = rds::WorkerGroup::from_worker(worker);

        // Check initial state - no resources allocated
        let node0_initial = allocator.cluster_resources.nodes.get("0").unwrap();
        assert_eq!(node0_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);

        allocator
            .add_worker(worker_group.clone())
            .expect("add worker group");

        // Verify worker group was added
        let retrieved = allocator.get_worker("w1").expect("get worker group");
        assert_eq!(retrieved.id, "w1");
        assert_eq!(retrieved.stage_name, "stage1");
        assert_eq!(retrieved.allocations.len(), 1);

        // Verify resources were actually allocated
        let node0_after = allocator.cluster_resources.nodes.get("0").unwrap();
        assert_eq!(node0_after.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0_after.gpus[0].used_fraction.to_num::<f32>(), 0.5);
    }

    #[test]
    fn test_workergroup_allocator_add_multiple_allocations() {
        let mut allocator = make_allocator();
        let allocation1 = wr("0", 2.0, vec![(0, 0.5)]);
        let allocation2 = wr("1", 1.0, vec![(0, 0.25)]);

        // Check initial state - no resources allocated
        let node0_initial = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_initial = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);
        assert_eq!(node1_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node1_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);

        let worker_group = rds::WorkerGroup {
            id: "multi_node_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![allocation1, allocation2],
        };

        allocator
            .add_worker(worker_group.clone())
            .expect("add multi-allocation worker group");

        // Verify worker group was added
        let retrieved = allocator
            .get_worker("multi_node_worker")
            .expect("get worker group");
        assert_eq!(retrieved.id, "multi_node_worker");
        assert_eq!(retrieved.allocations.len(), 2);
        assert_eq!(retrieved.allocations[0].node, "0");
        assert_eq!(retrieved.allocations[1].node, "1");

        // Verify resources were allocated on both nodes
        let node0_after = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_after.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0_after.gpus[0].used_fraction.to_num::<f32>(), 0.5);
        assert_eq!(node1_after.used_cpus.to_num::<f32>(), 1.0);
        assert_eq!(node1_after.gpus[0].used_fraction.to_num::<f32>(), 0.25);
    }

    #[test]
    fn test_workergroup_allocator_remove() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker_group = rds::WorkerGroup::from_worker(worker);

        // Add worker and verify resources are allocated
        allocator
            .add_worker(worker_group)
            .expect("add worker group");

        let node0_after_add = allocator.cluster_resources.nodes.get("0").unwrap();
        assert_eq!(node0_after_add.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0_after_add.gpus[0].used_fraction.to_num::<f32>(), 0.5);

        // Remove worker
        let removed = allocator.remove_worker("w1").expect("remove worker group");
        assert_eq!(removed.id, "w1");
        assert_eq!(removed.allocations.len(), 1);

        // Verify it's actually removed from allocator
        assert!(allocator.get_worker("w1").is_err());

        // Verify resources were actually deallocated
        let node0_after_remove = allocator.cluster_resources.nodes.get("0").unwrap();
        assert_eq!(node0_after_remove.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(
            node0_after_remove.gpus[0].used_fraction.to_num::<f32>(),
            0.0
        );
    }

    #[test]
    fn test_workergroup_allocator_remove_multi_allocation() {
        let mut allocator = make_allocator();
        let allocation1 = wr("0", 2.0, vec![(0, 0.5)]);
        let allocation2 = wr("1", 1.0, vec![(0, 0.25)]);

        let worker_group = rds::WorkerGroup {
            id: "multi_node_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![allocation1, allocation2],
        };

        // Add worker and verify resources are allocated on both nodes
        allocator
            .add_worker(worker_group)
            .expect("add multi-allocation worker group");

        let node0_after_add = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after_add = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_after_add.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0_after_add.gpus[0].used_fraction.to_num::<f32>(), 0.5);
        assert_eq!(node1_after_add.used_cpus.to_num::<f32>(), 1.0);
        assert_eq!(node1_after_add.gpus[0].used_fraction.to_num::<f32>(), 0.25);

        // Remove worker
        let removed = allocator
            .remove_worker("multi_node_worker")
            .expect("remove worker group");
        assert_eq!(removed.id, "multi_node_worker");
        assert_eq!(removed.allocations.len(), 2);

        // Verify it's actually removed from allocator
        assert!(allocator.get_worker("multi_node_worker").is_err());

        // Verify resources were deallocated on both nodes
        let node0_after_remove = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after_remove = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_after_remove.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(
            node0_after_remove.gpus[0].used_fraction.to_num::<f32>(),
            0.0
        );
        assert_eq!(node1_after_remove.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(
            node1_after_remove.gpus[0].used_fraction.to_num::<f32>(),
            0.0
        );
    }

    #[test]
    fn test_workergroup_allocator_add_multiple_worker_groups() {
        let mut allocator = make_allocator();

        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![]));

        // Check initial state
        let node0_initial = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_initial = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);
        assert_eq!(node1_initial.used_cpus.to_num::<f32>(), 0.0);

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        allocator
            .add_workers(worker_groups)
            .expect("add multiple worker groups");

        // Verify workers were added
        assert!(allocator.get_worker("w1").is_ok());
        assert!(allocator.get_worker("w2").is_ok());

        let all_workers = allocator.get_workers();
        assert_eq!(all_workers.len(), 2);

        // Verify resources were allocated on both nodes
        let node0_after = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_after.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0_after.gpus[0].used_fraction.to_num::<f32>(), 0.5);
        assert_eq!(node1_after.used_cpus.to_num::<f32>(), 1.0);
    }

    #[test]
    fn test_workergroup_allocator_delete_multiple() {
        let mut allocator = make_allocator();

        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![]));

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        // Add workers and verify resources are allocated
        allocator
            .add_workers(worker_groups)
            .expect("add multiple worker groups");

        let node0_after_add = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after_add = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_after_add.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0_after_add.gpus[0].used_fraction.to_num::<f32>(), 0.5);
        assert_eq!(node1_after_add.used_cpus.to_num::<f32>(), 1.0);

        // Delete one worker
        allocator
            .delete_workers(&vec!["w1".to_string()])
            .expect("delete worker group");

        // Verify w1 is removed but w2 remains
        assert!(allocator.get_worker("w1").is_err());
        assert!(allocator.get_worker("w2").is_ok());

        // Verify only w1's resources were deallocated
        let node0_after_delete = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after_delete = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_after_delete.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(
            node0_after_delete.gpus[0].used_fraction.to_num::<f32>(),
            0.0
        );
        assert_eq!(node1_after_delete.used_cpus.to_num::<f32>(), 1.0); // w2's resources remain
    }

    #[test]
    fn test_workergroup_cross_node_allocation() {
        let mut allocator = make_allocator();

        // Create a worker group that spans multiple nodes
        let allocation1 = wr("0", 2.0, vec![(0, 0.5)]);
        let allocation2 = wr("1", 1.0, vec![(0, 0.25)]);

        let worker_group = rds::WorkerGroup {
            id: "cross_node_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![allocation1, allocation2],
        };

        allocator
            .add_worker(worker_group.clone())
            .expect("add cross-node worker group");

        let retrieved = allocator
            .get_worker("cross_node_worker")
            .expect("get worker group");
        assert_eq!(retrieved.allocations.len(), 2);

        // Verify resources are allocated on both nodes
        let node0 = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1 = allocator.cluster_resources.nodes.get("1").unwrap();

        assert_eq!(node0.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0.gpus[0].used_fraction.to_num::<f32>(), 0.5);

        assert_eq!(node1.used_cpus.to_num::<f32>(), 1.0);
        assert_eq!(node1.gpus[0].used_fraction.to_num::<f32>(), 0.25);
    }

    #[test]
    fn test_workergroup_cross_node_allocation_rollback() {
        let mut allocator = make_allocator();

        // Check initial state
        let node0_initial = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_initial = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);
        assert_eq!(node1_initial.used_cpus.to_num::<f32>(), 0.0);

        // Create a worker group that spans multiple nodes, but one allocation will fail
        let allocation1 = wr("0", 2.0, vec![(0, 0.5)]);
        let allocation2 = wr("1", 10.0, vec![]); // This will fail - not enough CPUs on node1 (only 4 available)

        let worker_group = rds::WorkerGroup {
            id: "cross_node_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![allocation1, allocation2],
        };

        let result = allocator.add_worker(worker_group);
        assert!(result.is_err());

        // Verify that no resources were allocated (rollback worked)
        let node0_after = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after = allocator.cluster_resources.nodes.get("1").unwrap();

        assert_eq!(node0_after.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0_after.gpus[0].used_fraction.to_num::<f32>(), 0.0);
        assert_eq!(node1_after.used_cpus.to_num::<f32>(), 0.0);

        // Verify the worker group was not added to the allocator
        assert!(allocator.get_worker("cross_node_worker").is_err());
    }

    #[test]
    fn test_workergroup_empty_allocations() {
        let worker_group = rds::WorkerGroup {
            id: "empty_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![],
        };

        assert_eq!(worker_group.id, "empty_worker");
        assert_eq!(worker_group.allocations.len(), 0);
    }

    #[test]
    fn test_workergroup_allocator_with_empty_allocations() {
        let mut allocator = make_allocator();

        let worker_group = rds::WorkerGroup {
            id: "empty_worker".to_string(),
            stage_name: "stage1".to_string(),
            allocations: vec![],
        };

        // This should succeed - empty allocations don't require resources
        allocator
            .add_worker(worker_group.clone())
            .expect("add empty worker group");

        let retrieved = allocator
            .get_worker("empty_worker")
            .expect("get worker group");
        assert_eq!(retrieved.allocations.len(), 0);
    }

    #[test]
    fn test_workergroup_duplicate_id_detection() {
        let mut allocator = make_allocator();

        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w1".into(), "stage2".into(), wr("1", 1.0, vec![])); // Same ID

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        let result = allocator.add_workers(worker_groups);
        assert!(result.is_err());

        match result.unwrap_err() {
            WorkerAllocatorError::DuplicateWorkerId(id) => assert_eq!(id, "w1"),
            _ => panic!("Expected DuplicateWorkerId error"),
        }
    }

    #[test]
    fn test_workergroup_get_workers() {
        let mut allocator = make_allocator();

        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![]));

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        allocator
            .add_workers(worker_groups)
            .expect("add worker groups");

        let all_workers = allocator.get_workers();
        assert_eq!(all_workers.len(), 2);

        let ids: Vec<&String> = all_workers.iter().map(|w| &w.id).collect();
        assert!(ids.contains(&&"w1".to_string()));
        assert!(ids.contains(&&"w2".to_string()));
    }

    #[test]
    fn test_workergroup_get_num_workers_per_stage() {
        let mut allocator = make_allocator();

        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w2".into(), "stage1".into(), wr("0", 1.0, vec![]));
        let worker3 = rds::Worker::new("w3".into(), "stage2".into(), wr("1", 1.0, vec![]));

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
            rds::WorkerGroup::from_worker(worker3),
        ];

        allocator
            .add_workers(worker_groups)
            .expect("add worker groups");

        let counts = allocator.get_num_workers_per_stage();
        assert_eq!(counts.get("stage1").copied().unwrap_or_default(), 2);
        assert_eq!(counts.get("stage2").copied().unwrap_or_default(), 1);
    }

    #[test]
    fn test_workergroup_initialization_with_workers() {
        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![]));

        let initial_workers = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        let allocator = WorkerAllocator::new(make_simple_cluster(), Some(initial_workers))
            .expect("create allocator with initial workers");

        // Verify workers were added
        assert!(allocator.get_worker("w1").is_ok());
        assert!(allocator.get_worker("w2").is_ok());

        let counts = allocator.get_num_workers_per_stage();
        assert_eq!(counts.get("stage1").copied().unwrap_or_default(), 1);
        assert_eq!(counts.get("stage2").copied().unwrap_or_default(), 1);

        // Verify resources were allocated during initialization
        let node0 = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1 = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0.used_cpus.to_num::<f32>(), 2.0);
        assert_eq!(node0.gpus[0].used_fraction.to_num::<f32>(), 0.5);
        assert_eq!(node1.used_cpus.to_num::<f32>(), 1.0);
    }

    #[test]
    fn test_workergroup_initialization_with_duplicate_workers() {
        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w1".into(), "stage2".into(), wr("1", 1.0, vec![])); // Same ID

        let initial_workers = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        let result = WorkerAllocator::new(make_simple_cluster(), Some(initial_workers));
        assert!(result.is_err());

        match result.unwrap_err() {
            WorkerAllocatorError::DuplicateWorkerId(id) => assert_eq!(id, "w1"),
            _ => panic!("Expected DuplicateWorkerId error"),
        }
    }

    #[test]
    fn test_workergroup_complex_multi_node_scenario() {
        let mut allocator = make_allocator();

        // Create a complex scenario with multiple worker groups across different nodes
        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.3)]));
        let worker2 = rds::Worker::new("w2".into(), "stage1".into(), wr("0", 1.0, vec![(1, 0.4)]));

        // Multi-allocation worker group spanning both nodes
        let allocation1 = wr("0", 1.0, vec![(0, 0.2)]);
        let allocation2 = wr("1", 1.0, vec![(0, 0.3)]);
        let multi_node_worker = rds::WorkerGroup {
            id: "multi_node".to_string(),
            stage_name: "stage2".to_string(),
            allocations: vec![allocation1, allocation2],
        };

        // Check initial state
        let node0_initial = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_initial = allocator.cluster_resources.nodes.get("1").unwrap();
        assert_eq!(node0_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);
        assert_eq!(node0_initial.gpus[1].used_fraction.to_num::<f32>(), 0.0);
        assert_eq!(node1_initial.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node1_initial.gpus[0].used_fraction.to_num::<f32>(), 0.0);

        // Add all workers
        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
            multi_node_worker,
        ];

        allocator
            .add_workers(worker_groups)
            .expect("add complex worker groups");

        // Verify all workers were added
        assert!(allocator.get_worker("w1").is_ok());
        assert!(allocator.get_worker("w2").is_ok());
        assert!(allocator.get_worker("multi_node").is_ok());

        // Verify resource allocation across nodes
        let node0_after = allocator.cluster_resources.nodes.get("0").unwrap();
        let node1_after = allocator.cluster_resources.nodes.get("1").unwrap();

        // Node 0: w1 (1.0 CPU, 0.3 GPU0) + w2 (1.0 CPU, 0.4 GPU1) + multi_node (1.0 CPU, 0.2 GPU0)
        assert_eq!(node0_after.used_cpus.to_num::<f32>(), 3.0);
        assert!((node0_after.gpus[0].used_fraction.to_num::<f32>() - 0.5).abs() < 1e-4); // 0.3 + 0.2
        assert!((node0_after.gpus[1].used_fraction.to_num::<f32>() - 0.4).abs() < 1e-4);

        // Node 1: multi_node (1.0 CPU, 0.3 GPU0)
        assert_eq!(node1_after.used_cpus.to_num::<f32>(), 1.0);
        assert!((node1_after.gpus[0].used_fraction.to_num::<f32>() - 0.3).abs() < 1e-4);

        // Verify stage counts
        let counts = allocator.get_num_workers_per_stage();
        assert_eq!(counts.get("stage1").copied().unwrap_or_default(), 2);
        assert_eq!(counts.get("stage2").copied().unwrap_or_default(), 1);
    }

    #[test]
    fn test_workergroup_gpu_overallocation_detection() {
        let mut allocator = make_allocator();

        // First worker takes 0.8 of GPU 0
        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.8)]));

        // Second worker tries to take 0.5 of the same GPU (total would be 1.3 > 1.0)
        let worker2 = rds::Worker::new("w2".into(), "stage1".into(), wr("0", 1.0, vec![(0, 0.5)]));

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        let result = allocator.add_workers(worker_groups);
        assert!(result.is_err());

        // Verify that no resources were allocated (rollback worked)
        let node0 = allocator.cluster_resources.nodes.get("0").unwrap();
        assert_eq!(node0.used_cpus.to_num::<f32>(), 0.0);
        assert_eq!(node0.gpus[0].used_fraction.to_num::<f32>(), 0.0);

        // Verify no workers were added
        assert!(allocator.get_worker("w1").is_err());
        assert!(allocator.get_worker("w2").is_err());
    }

    #[test]
    fn test_workergroup_cpu_overallocation_detection() {
        let mut allocator = make_allocator();

        // First worker takes 6.0 CPUs (node0 has 8.0 total)
        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 6.0, vec![]));

        // Second worker tries to take 3.0 CPUs (total would be 9.0 > 8.0)
        let worker2 = rds::Worker::new("w2".into(), "stage1".into(), wr("0", 3.0, vec![]));

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        let result = allocator.add_workers(worker_groups);
        assert!(result.is_err());

        // Verify that no resources were allocated (rollback worked)
        let node0 = allocator.cluster_resources.nodes.get("0").unwrap();
        assert_eq!(node0.used_cpus.to_num::<f32>(), 0.0);

        // Verify no workers were added
        assert!(allocator.get_worker("w1").is_err());
        assert!(allocator.get_worker("w2").is_err());
    }

    #[test]
    fn test_workergroup_utilization_table_with_workergroups() {
        let mut allocator = make_allocator();

        let worker1 = rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)]));
        let worker2 = rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![]));

        let worker_groups = vec![
            rds::WorkerGroup::from_worker(worker1),
            rds::WorkerGroup::from_worker(worker2),
        ];

        allocator
            .add_workers(worker_groups)
            .expect("add worker groups");

        let table = allocator.make_detailed_utilization_table();
        assert!(table.contains("Node 0"));
        assert!(table.contains("Node 1"));

        // Verify the table shows some utilization (not all zeros)
        assert!(table.contains("2.00")); // CPU usage
        assert!(table.contains("0.50")); // GPU usage
    }
}
