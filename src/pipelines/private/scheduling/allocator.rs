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

use super::resources::{AllocationError, ClusterResources, NodeResources, Worker};
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
    #[error("Cluster is over-allocated: {0}")]
    OverAllocated(String),
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
    cluster_resources: ClusterResources,
    nodes_state: HashMap<String, NodeWorkers>,
    stages_state: HashMap<String, StageWorkers>,
    available_resources: ClusterResources,
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
            available_resources: cluster_resources.clone(),
            cluster_resources,
            nodes_state,
            stages_state: HashMap::new(),
        };

        if let Some(initial_workers) = workers {
            this.add_workers(initial_workers.into_iter())?;
        } else {
            // Ensure available_resources is consistent
            this.recalculate_available_resources()?;
        }

        Ok(this)
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes_state.len()
    }

    pub fn totals(&self) -> &ClusterResources {
        &self.cluster_resources
    }

    pub fn available_resources(&self) -> &ClusterResources {
        &self.available_resources
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
        worker.allocation.validate();
        self.ensure_worker_id_absent(&worker.id)?;

        // Ensure node exists; do not create implicitly
        let Some(container) = self.nodes_state.get_mut(&worker.allocation.node) else {
            return Err(WorkerAllocatorError::Allocation(
                AllocationError::NodeNotFound(worker.allocation.node.clone()),
            ));
        };
        container.by_id.insert(worker.id.clone(), worker.clone());

        // Track in stage index
        self.stages_state
            .entry(worker.stage_name.clone())
            .or_default()
            .by_id
            .insert(worker.id.clone(), worker);

        self.recalculate_available_resources()?;
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
        // Validate first (IDs and allocations)
        let mut to_insert: Vec<Worker> = Vec::new();
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        for w in workers {
            w.allocation.validate();
            self.ensure_worker_id_absent(&w.id)?;
            // Detect duplicates within the same batch
            if !seen_ids.insert(w.id.clone()) {
                return Err(WorkerAllocatorError::DuplicateWorkerId(w.id.clone()));
            }
            if !self.nodes_state.contains_key(&w.allocation.node) {
                return Err(WorkerAllocatorError::Allocation(
                    AllocationError::NodeNotFound(w.allocation.node.clone()),
                ));
            }
            to_insert.push(w);
        }

        // Insert
        for worker in to_insert.into_iter() {
            let container = self
                .nodes_state
                .get_mut(&worker.allocation.node)
                .expect("validated above that node exists");
            container.by_id.insert(worker.id.clone(), worker.clone());

            self.stages_state
                .entry(worker.stage_name.clone())
                .or_default()
                .by_id
                .insert(worker.id.clone(), worker);
        }

        self.recalculate_available_resources()?;
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

    pub fn delete_worker(&mut self, worker_id: &str) -> Result<Worker, WorkerAllocatorError> {
        let worker = self.get_worker(worker_id)?;
        if let Some(container) = self.nodes_state.get_mut(&worker.allocation.node) {
            container.by_id.remove(worker_id);
        }
        if let Some(stage_map) = self.stages_state.get_mut(&worker.stage_name) {
            stage_map.by_id.remove(worker_id);
        }
        self.recalculate_available_resources()?;
        Ok(worker)
    }

    pub fn delete_workers(&mut self, worker_ids: &[String]) -> Result<(), WorkerAllocatorError> {
        // Collect workers first to avoid partial state if any are missing
        let mut workers: Vec<Worker> = Vec::with_capacity(worker_ids.len());
        {
            let mut seen: HashSet<&str> = HashSet::new();
            for id in worker_ids {
                if !seen.insert(id.as_str()) {
                    // Duplicates in input are not allowed, mirror Python assertion
                    return Err(WorkerAllocatorError::DuplicateWorkerId(id.clone()));
                }
                workers.push(self.get_worker(id)?);
            }
        }

        for worker in workers.into_iter() {
            if let Some(container) = self.nodes_state.get_mut(&worker.allocation.node) {
                container.by_id.remove(&worker.id);
            }
            if let Some(stage_map) = self.stages_state.get_mut(&worker.stage_name) {
                stage_map.by_id.remove(&worker.id);
            }
        }
        self.recalculate_available_resources()?;
        Ok(())
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

    /// Updates tracking of remaining available resources across nodes.
    ///
    /// This method recalculates available resources by subtracting all allocated
    /// resources from the total available resources. It tracks CPU, GPU, NVDEC,
    /// and NVENC allocations.
    ///
    /// # Errors
    /// Returns `WorkerAllocatorError::OverAllocated` if current allocation exceeds available resources.
    fn recalculate_available_resources(&mut self) -> Result<(), WorkerAllocatorError> {
        // Start from the full cluster resources
        let mut remaining = self.cluster_resources.clone();

        // Subtract all allocations
        for node_workers in self.nodes_state.values() {
            for worker in node_workers.by_id.values() {
                // Allocate on the correct node
                let Some(node) = remaining.nodes.get_mut(&worker.allocation.node) else {
                    return Err(
                        AllocationError::NodeNotFound(worker.allocation.node.clone()).into(),
                    );
                };
                Self::allocate_on_node(node, &worker).map_err(WorkerAllocatorError::Allocation)?;
            }
        }

        if remaining.is_overallocated() {
            return Err(WorkerAllocatorError::OverAllocated(
                remaining.make_overallocated_message(&self.cluster_resources),
            ));
        }

        self.available_resources = remaining;
        Ok(())
    }

    fn allocate_on_node(node: &mut NodeResources, worker: &Worker) -> Result<(), AllocationError> {
        node.allocate(&worker.allocation)
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
        let node_ids: HashSet<String> = self
            .cluster_resources
            .nodes
            .keys()
            .cloned()
            .chain(self.available_resources.nodes.keys().cloned())
            .collect();

        for node_id in node_ids {
            let total = self
                .cluster_resources
                .nodes
                .get(&node_id)
                .expect("node id must exist in cluster_resources");
            let remaining = self
                .available_resources
                .nodes
                .get(&node_id)
                .expect("node id must exist in available_resources");
            let used_cpus = total.cpus - remaining.cpus;
            let utilization = if total.cpus > 0.0 {
                used_cpus / total.cpus
            } else {
                0.0
            };
            utilizations.insert(node_id, utilization);
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
        let mut node_ids: Vec<String> = self.cluster_resources.nodes.keys().cloned().collect();
        for nid in self.nodes_state.keys() {
            if !node_ids.contains(nid) {
                node_ids.push(nid.clone());
            }
        }

        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Component"),
                Cell::new("Utilization"),
                Cell::new("NVDEC"),
                Cell::new("NVENC"),
            ]);

        for (node_index, node_id) in node_ids.iter().enumerate() {
            let total = self
                .cluster_resources
                .nodes
                .get(node_id)
                .expect("node id must exist in cluster_resources");
            let node_workers = self.nodes_state.get(node_id);

            let mut cpu_usage: f32 = 0.0;
            let mut gpu_usage: Vec<f32> = vec![0.0; total.gpus.len()];
            let mut nvdec_usage: Vec<f32> = vec![0.0; total.gpus.len()];
            let mut nvenc_usage: Vec<f32> = vec![0.0; total.gpus.len()];

            if let Some(nw) = node_workers {
                for worker in nw.by_id.values() {
                    cpu_usage += worker.allocation.cpus;
                    for gpu in &worker.allocation.gpus {
                        if let Some(slot) = gpu_usage.get_mut(gpu.gpu_index) {
                            *slot += gpu.fraction;
                        }
                    }
                    for nvdec in &worker.allocation.nvdecs {
                        if let Some(slot) = nvdec_usage.get_mut(nvdec.gpu_index) {
                            *slot += 1.0;
                        }
                    }
                    for nvenc in &worker.allocation.nvencs {
                        if let Some(slot) = nvenc_usage.get_mut(nvenc.gpu_index) {
                            *slot += 1.0;
                        }
                    }
                }
            }

            let cpu_bar = create_bar_chart(cpu_usage, total.cpus, 20);
            table.add_row(vec![
                Cell::new(format!("Node {}", node_index)),
                Cell::new(format!("CPUs: {}", cpu_bar)),
                Cell::new(""),
                Cell::new(""),
            ]);

            for (i, gpu) in total.gpus.iter().enumerate() {
                let gpu_bar = create_bar_chart(gpu_usage[i], 1.0, 20);
                let nvdec_bar = create_bar_chart(nvdec_usage[i], gpu.num_nvdecs() as f32, 20);
                let nvenc_bar = create_bar_chart(nvenc_usage[i], gpu.num_nvencs() as f32, 20);
                table.add_row(vec![
                    Cell::new(format!("  GPU {}", i)),
                    Cell::new(format!("GPU: {}", gpu_bar)),
                    Cell::new(format!("NVDEC: {}", nvdec_bar)),
                    Cell::new(format!("NVENC: {}", nvenc_bar)),
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

    #[pyo3(name = "totals")]
    pub fn py_totals(&self) -> ClusterResources {
        self.totals().clone()
    }

    #[pyo3(name = "available_resources")]
    pub fn py_available_resources(&self) -> ClusterResources {
        self.available_resources().clone()
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

    #[pyo3(name = "delete_worker")]
    pub fn py_delete_worker(&mut self, worker_id: String) -> PyResult<Worker> {
        self.delete_worker(&worker_id).map_err(Into::into)
    }

    #[pyo3(name = "delete_workers")]
    pub fn py_delete_workers(&mut self, worker_ids: Vec<String>) -> PyResult<()> {
        self.delete_workers(&worker_ids).map_err(Into::into)
    }

    #[pyo3(name = "get_worker")]
    pub fn py_get_worker(&self, worker_id: String) -> Option<Worker> {
        self.get_worker_if_exists(&worker_id)
    }

    #[pyo3(name = "get_workers_in_stage")]
    pub fn py_get_workers_in_stage(&self, stage_name: String) -> Vec<Worker> {
        self.get_workers_in_stage(&stage_name)
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
        let node0 = rds::NodeResources::new(
            8.0,
            Some(vec![
                rds::GpuResources::new(0, uuid::Uuid::new_v4(), 1.0, vec![0, 1], vec![0, 1]),
                rds::GpuResources::new(1, uuid::Uuid::new_v4(), 1.0, vec![0, 1], vec![0, 1]),
            ]),
            None,
        );
        let node1 = rds::NodeResources::new(
            4.0,
            Some(vec![rds::GpuResources::new(
                0,
                uuid::Uuid::new_v4(),
                1.0,
                vec![0],
                vec![1],
            )]),
            None,
        );
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
            .map(|(idx, frac)| rds::GPUAllocation::new(idx, frac))
            .collect();
        rds::WorkerResources::new(node.to_string(), cpus, Some(gpu_allocs), None, None)
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
    fn test_calculate_remaining_resources() {
        let mut allocator = make_allocator();
        let workers = vec![
            rds::Worker::new("w1".into(), "stage1".into(), wr("0", 2.0, vec![(0, 0.5)])),
            rds::Worker::new("w2".into(), "stage2".into(), wr("1", 1.0, vec![])),
        ];
        allocator.add_workers(workers).expect("add workers");
        let remaining = allocator.available_resources().clone();
        assert_eq!(remaining.nodes.get("0").unwrap().cpus, 6.0);
        assert!((remaining.nodes.get("0").unwrap().gpus[0].gpu_fraction - 0.5).abs() < 1e-6);
        assert_eq!(remaining.nodes.get("1").unwrap().cpus, 3.0);
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
            WorkerAllocatorError::OverAllocated(_) => {}
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
            WorkerAllocatorError::OverAllocated(_) => {}
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
            WorkerAllocatorError::OverAllocated(_) => {}
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
                rds::WorkerResources::new(
                    "0".into(),
                    0.0,
                    Some(vec![rds::GPUAllocation::new(0, 1.0)]),
                    None,
                    None,
                ),
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources::new(
                    "1".into(),
                    0.0,
                    Some(vec![rds::GPUAllocation::new(0, 0.7)]),
                    None,
                    None,
                ),
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources::new(
                    "1".into(),
                    0.0,
                    Some(vec![rds::GPUAllocation::new(0, 0.31)]),
                    None,
                    None,
                ),
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
                rds::WorkerResources::new(
                    "0".into(),
                    0.0,
                    Some(vec![rds::GPUAllocation::new(0, 1.0)]),
                    None,
                    None,
                ),
            ),
            rds::Worker::new(
                "2".into(),
                "1".into(),
                rds::WorkerResources::new(
                    "1".into(),
                    0.0,
                    Some(vec![rds::GPUAllocation::new(0, 0.7)]),
                    None,
                    None,
                ),
            ),
            rds::Worker::new(
                "3".into(),
                "1".into(),
                rds::WorkerResources::new(
                    "1".into(),
                    0.0,
                    Some(vec![rds::GPUAllocation::new(0, 0.31)]),
                    None,
                    None,
                ),
            ),
        ];
        let err = allocator.add_workers(workers).unwrap_err();
        match err {
            WorkerAllocatorError::OverAllocated(_) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    #[test]
    fn test_gpu_allocation_limit() {
        let mut allocator = make_allocator();
        let worker = rds::Worker::new(
            "w1".into(),
            "stage1".into(),
            rds::WorkerResources::new(
                "0".into(),
                1.0,
                Some(vec![rds::GPUAllocation::new(0, 1.5)]),
                None,
                None,
            ),
        );
        let err = allocator.add_worker(worker).unwrap_err();
        match err {
            WorkerAllocatorError::OverAllocated(_) => {}
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
        allocator.delete_worker("w1").expect("delete");
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
    fn test_overallocation_with_nvdec_nvenc() {
        let mut allocator = make_allocator();
        let w1 = rds::Worker::new(
            "w1".into(),
            "stage1".into(),
            rds::WorkerResources::new(
                "0".into(),
                1.0,
                Some(vec![rds::GPUAllocation::new(0, 0.5)]),
                Some(vec![
                    rds::CodecAllocation::new(0, 0),
                    rds::CodecAllocation::new(0, 1),
                ]),
                Some(vec![
                    rds::CodecAllocation::new(0, 0),
                    rds::CodecAllocation::new(0, 1),
                ]),
            ),
        );
        let w2 = rds::Worker::new(
            "w2".into(),
            "stage2".into(),
            rds::WorkerResources::new(
                "0".into(),
                1.0,
                Some(vec![rds::GPUAllocation::new(0, 0.5)]),
                Some(vec![
                    rds::CodecAllocation::new(0, 0),
                    rds::CodecAllocation::new(0, 1),
                ]),
                Some(vec![
                    rds::CodecAllocation::new(0, 0),
                    rds::CodecAllocation::new(0, 1),
                ]),
            ),
        );
        let err = allocator.add_workers(vec![w1, w2]).unwrap_err();
        match err {
            WorkerAllocatorError::Allocation(AllocationError::NvdecUnavailable { .. }) => {}
            WorkerAllocatorError::Allocation(AllocationError::NvencUnavailable { .. }) => {}
            _ => panic!("unexpected error variant: {err:?}"),
        }
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