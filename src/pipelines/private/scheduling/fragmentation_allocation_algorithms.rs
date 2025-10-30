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

//! Allocation algorithms which rely on an expected distribution of jobs and the concept of "fragmentation".
//!
//! This is just one component of our pipeline scheduling algorithm. It's basically just solving the bin packing problem.
//! Essentially, we have a certain set of resources distributed across the cluster. We need functions which tell us which node
//! gpus, nvdecs/nvencs to allocate to a particular worker. This is essentially the multi-dimensional bin-packing problem,
//! but with some twists. To solve this, we created a new algorithm heavily inspired by the algorithm in this paper:
//! Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent
//!
//! We extend the ideas in this paper by considering NVDEC and NVENC allocation, which results in a more
//! complex algorithm. We also consider the removal of workers, which is a simple extension.

use super::resources as rds;

// --------------------
// Stages and workloads
// --------------------

/// A stage in the workload with associated frequency and resource shape requirements.
///
/// As described in the paper, each stage represents a recurring task type in the workload
/// with its resource requirements and relative frequency/popularity.
///
/// # Attributes
/// * `frequency` - A float between 0 and 1 representing how often this stage occurs in workload.
///     The sum of all stage frequencies in a workload should equal 1.
/// * `shape` - A WorkerShape object defining the resource requirements (CPU, GPU, etc.)
///     for this stage of the workload.
#[derive(Debug, Clone)]
pub struct Stage {
    pub frequency: f32,
    pub shape: rds::WorkerShape,
}

/// Represents a complete workload consisting of multiple stages.
///
/// A workload models the expected distribution of tasks in the cluster, used to
/// calculate fragmentation metrics. As per the paper, production ML workloads
/// consist of recurring tasks that follow certain resource requirement patterns.
///
/// # Attributes
/// * `stages` - A list of Stage objects representing the different task types
///     and their frequencies in this workload.
#[derive(Debug, Clone)]
pub struct Workload {
    pub stages: Vec<Stage>,
}

// --------------------
// Results
// --------------------

/// Results from calculating fragmentation for a particular allocation scenario.
///
/// Captures the fragmentation state before and after a potential allocation to help
/// evaluate scheduling decisions.
///
/// # Attributes
/// * `fragmentation_before` - Float indicating fragmentation level before allocation
/// * `fragmentation_after` - Float indicating fragmentation level after allocation
/// * `node_remaining_resources` - Float representing resources left on node after allocation
/// * `worker_allocation` - WorkerResources object describing the actual allocation
/// * `maybe_reused_worker` - If this was the result of re-allocating a previous worker, record the worker here.
#[derive(Debug, Clone)]
pub struct FragmentationResult {
    pub fragmentation_before: f32,
    pub fragmentation_after: f32,
    pub node_remaining_resources: f32,
    pub worker_allocation: rds::WorkerResources,
    pub maybe_reused_worker: Option<rds::Worker>,
}

impl FragmentationResult {
    /// Calculates the change in fragmentation caused by this allocation.
    ///
    /// # Returns
    /// Float representing the change in fragmentation (after - before)
    pub fn fragmentation_change(&self) -> f32 {
        self.fragmentation_after - self.fragmentation_before
    }

    /// Returns true if this result represents reusing an existing worker.
    pub fn is_reused_worker(&self) -> bool {
        self.maybe_reused_worker.is_some()
    }
}

/// Result of an allocation attempt, indicating success and resource details.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub did_allocate: bool,
    pub resources: Option<rds::WorkerResources>,
    pub reused_worker: Option<rds::Worker>,
}

/// Determines if this GPU can accommodate the given worker shape requirements.
///
/// It doesn't have to be able to fully allocate the shape, but it does need to be able to contribute to the
/// allocation. So, if the shape requires 2 gpus and this is a fully unallocated gpu, this will return True.
///
/// This method implements the allocation feasibility check described in Section 2.1
/// of the paper. It handles different GPU allocation types:
/// - CPU-only workloads
/// - Fractional GPU workloads
/// - Whole-numbered GPU workloads
///
/// # Arguments
/// * `shape` - WorkerShape describing resource requirements
/// * `available_cpus` - Number of CPU cores available on the node
///
/// # Returns
/// True if the GPU can accommodate this shape, False otherwise
pub fn gpu_can_be_used_to_allocate(gpu: &rds::GpuResources, shape: &rds::WorkerShape) -> bool {
    match shape {
        rds::WorkerShape::CpuOnly(_) => false,
        rds::WorkerShape::FractionalGpu(s) => {
            gpu.used_fraction + s.gpu_fraction <= rds::FixedUtil::ONE
        }
        rds::WorkerShape::WholeNumberedGpu(_) => gpu.is_fully_unallocated(),
        rds::WorkerShape::SpmdNodeMultiple(_) => gpu.is_fully_unallocated(),
        rds::WorkerShape::SpmdSmallerThanNode(_) => gpu.is_fully_unallocated(),
    }
}

// --------------------
// Node helpers
// --------------------

/// Counts number of GPUs on node that can accommodate given shape.
///
/// # Arguments
/// * `shape` - WorkerShape describing resource requirements
///
/// # Returns
/// Number of GPUs that can be used for this shape
fn number_of_gpus_which_can_be_used_for_shape(
    node_resources: &rds::NodeResources,
    shape: &rds::WorkerShape,
) -> usize {
    node_resources
        .gpus
        .iter()
        .filter(|g| gpu_can_be_used_to_allocate(g, shape))
        .count()
}

fn num_fully_unallocated_gpus(node_resources: &rds::NodeResources) -> usize {
    node_resources
        .gpus
        .iter()
        .filter(|g| g.is_fully_unallocated())
        .count()
}

/// Determines if node has sufficient resources for given shape.
///
/// This implements the node-level allocation feasibility check described in
/// Section 3.2 of the paper.
///
/// # Arguments
/// * `shape` - WorkerShape describing resource requirements
///
/// # Returns
/// True if node can accommodate shape, False otherwise
pub fn node_can_allocate(node_resources: &rds::NodeResources, shape: &rds::WorkerShape) -> bool {
    // CPU check
    let needed_cpus = match shape {
        rds::WorkerShape::CpuOnly(s) => s.num_cpus,
        rds::WorkerShape::FractionalGpu(s) => s.num_cpus,
        rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus * rds::FixedUtil::from_num(s.num_gpus),
        rds::WorkerShape::SpmdSmallerThanNode(s) => {
            s.num_cpus_per_actor * rds::FixedUtil::from_num(s.num_gpu_actors_in_group)
        }
        rds::WorkerShape::SpmdNodeMultiple(s) => s.num_cpus_needed_per_node(),
    };

    if node_resources.used_cpus + needed_cpus > node_resources.total_cpus {
        return false;
    }

    match shape {
        rds::WorkerShape::CpuOnly(_) => true,
        rds::WorkerShape::FractionalGpu(_) => {
            number_of_gpus_which_can_be_used_for_shape(node_resources, shape) > 0
        }
        rds::WorkerShape::WholeNumberedGpu(s) => {
            num_fully_unallocated_gpus(node_resources) >= s.num_gpus as usize
        }
        rds::WorkerShape::SpmdSmallerThanNode(s) => {
            num_fully_unallocated_gpus(node_resources) >= s.num_gpu_actors_in_group as usize
        }
        rds::WorkerShape::SpmdNodeMultiple(s) => {
            node_resources.num_gpus() == s.num_gpus_in_node as usize
                && node_resources.all_gpus_fully_unallocated()
        }
    }
}

/// Helper function to allocate whole GPUs with the given number of GPUs and CPUs
fn allocate_whole_gpus(
    node_resources: &rds::NodeResources,
    shape: &rds::WorkerShape,
    node_id: &str,
    num_gpus: u8,
    num_cpus_per_gpu: rds::FixedUtil,
) -> Vec<rds::WorkerResources> {
    // Find all GPUs that can accommodate this shape (must be fully available)
    let available_gpus: Vec<usize> = node_resources
        .gpus
        .iter()
        .enumerate()
        .filter(|(_, g)| gpu_can_be_used_to_allocate(g, shape))
        .map(|(i, _)| i)
        .collect();

    // Early return if not enough GPUs available
    if (num_gpus as usize) > available_gpus.len() {
        return Vec::new();
    }

    // Take the first x GPUs where x is the number of GPUs needed
    let chosen_gpus: Vec<usize> = available_gpus.into_iter().take(num_gpus as usize).collect();

    // Create GPU allocations for the chosen GPUs (1.0 fraction each)
    let gpus: Vec<rds::GpuAllocation> = chosen_gpus
        .iter()
        .map(|&offset| rds::GpuAllocation {
            offset,
            used_fraction: rds::FixedUtil::ONE,
        })
        .collect();

    // Return the single allocation with the first available GPUs
    vec![rds::WorkerResources {
        node: node_id.to_string(),
        cpus: num_cpus_per_gpu * rds::FixedUtil::from_num(num_gpus),
        gpus,
    }]
}

/// Finds all valid ways to allocate resources for given shape on this node.
///
/// This is a key method implementing the allocation possibilities analysis
/// described in Section 3.2 of the paper. It handles different resource
/// requirement types and finds all valid allocation combinations.
///
/// # Arguments
/// * `shape` - WorkerShape describing resource requirements
/// * `node_id` - ID of this node
///
/// # Returns
/// List of possible WorkerResources allocations. Empty if none are possible.
pub fn find_possible_allocations_on_node(
    node_resources: &rds::NodeResources,
    shape: &rds::WorkerShape,
    node_id: &str,
) -> Vec<rds::WorkerResources> {
    if !node_can_allocate(node_resources, shape) {
        return Vec::new();
    }

    match shape {
        // CPU-only tasks: simple allocation of just CPU cores
        rds::WorkerShape::CpuOnly(s) => {
            vec![rds::WorkerResources {
                node: node_id.to_string(),
                cpus: s.num_cpus,
                gpus: Vec::new(),
            }]
        }
        // Fractional GPU tasks: allocate partial GPU compute plus optional codecs
        rds::WorkerShape::FractionalGpu(s) => {
            let mut out = Vec::new();
            // Try allocating on each GPU that has sufficient capacity
            for (gpu_offset, gpu) in node_resources.gpus.iter().enumerate() {
                if !gpu_can_be_used_to_allocate(gpu, shape) {
                    continue;
                }

                // Create allocation with fractional GPU compute
                out.push(rds::WorkerResources {
                    node: node_id.to_string(),
                    cpus: s.num_cpus,
                    gpus: vec![rds::GpuAllocation {
                        offset: gpu_offset,
                        used_fraction: s.gpu_fraction,
                    }],
                });
            }
            out
        }
        // Whole numbered GPU tasks: allocate complete GPUs (1.0 fraction each)
        // We don't actually need to get all combinations of GPUs. If two gpus are on the same node,
        // they should have the same contibution to the fragmentation. So, we can just grab the first
        // unallocated gpus and use that for those.
        rds::WorkerShape::WholeNumberedGpu(s) => {
            allocate_whole_gpus(node_resources, shape, node_id, s.num_gpus, s.num_cpus)
        }
        // SPMD tasks smaller than node: use same logic as WholeNumberedGpu
        rds::WorkerShape::SpmdSmallerThanNode(s) => allocate_whole_gpus(
            node_resources,
            shape,
            node_id,
            s.num_gpu_actors_in_group as u8,
            s.num_cpus_per_actor,
        ),
        // SPMD tasks that are a multiple of node size: panic as this is not supported
        rds::WorkerShape::SpmdNodeMultiple(_) => {
            panic!("SpmdNodeMultiple is not supported in find_possible_allocations_on_node");
        }
    }
}

/// Calculates amount of GPU resources that cannot be allocated to a specific shape.
///
/// This implements the task-level fragmentation measure F_n(m) described in Section 3.2
/// of the paper. It measures how many GPU resources cannot be allocated to a given
/// task shape due to various constraints.
///
/// # Arguments
/// * `shape` - WorkerShape describing resource requirements
///
/// # Returns
/// Amount of GPU resources that cannot be allocated to this shape.
/// A higher value indicates more fragmentation from this shape's perspective.
fn calculate_unallocatable_gpus_fragment_for_shape_on_node(
    node_resources: &rds::NodeResources,
    shape: &rds::WorkerShape,
) -> f32 {
    // Calculate total GPU compute resources available on this node
    let total_available_gpus = node_resources
        .gpus
        .iter()
        .map(|g| 1.0 - g.used_fraction.to_num::<f32>())
        .sum();

    // Case 1: Task requests no GPU compute resources
    // All available GPU resources are "fragmented" since they can't be used by this task type
    if let rds::WorkerShape::CpuOnly(_) = shape {
        return total_available_gpus;
    }

    // Case 2: Shape cannot be allocated to the node at all
    // All available GPU resources are fragmented since the task can't be satisfied
    if !node_can_allocate(node_resources, shape) {
        return total_available_gpus;
    }

    // Case 3: Shape can be allocated, but some GPUs may be unusable
    // Count GPU resources that cannot contribute to allocating this shape
    let mut out = 0.0;
    for node_gpu in &node_resources.gpus {
        // If this GPU cannot contribute to the allocation, its resources are fragmented
        if !gpu_can_be_used_to_allocate(node_gpu, shape) {
            out += 1.0 - node_gpu.used_fraction.to_num::<f32>();
        }
    }
    out
}

/// Estimates overall fragmentation from perspective of entire workload.
///
/// This implements the node-level fragmentation measure F_n(M) described in
/// Section 3.2 of the paper. It calculates the expected fragmentation by
/// weighting each shape's fragmentation by its frequency in the workload.
///
/// # Arguments
/// * `workload` - Workload object containing stages with shapes and frequencies
///
/// # Returns
/// Estimated fragmentation level for this node given the workload
pub fn estimate_fragmentation_on_node(
    node_resources: &rds::NodeResources,
    workload: &Workload,
) -> f32 {
    let mut out = 0.0;
    // Calculate weighted fragmentation across all workload stages
    for stage in &workload.stages {
        // Get fragmentation for this specific task type
        let unallocatable_gpus =
            calculate_unallocatable_gpus_fragment_for_shape_on_node(node_resources, &stage.shape);
        // Weight by how frequently this task type occurs
        out += stage.frequency * unallocatable_gpus;
    }
    out
}

// --------------------
// Cluster helpers
// --------------------

pub fn estimate_fragmentation_on_cluster(
    cluster_resources: &rds::ClusterResources,
    workload: &Workload,
) -> f32 {
    cluster_resources
        .nodes
        .values()
        .map(|n| estimate_fragmentation_on_node(n, workload))
        .sum()
}

// --------------------
// Public algorithms
// --------------------

/// Finds the best allocation for a shape that minimizes fragmentation increase.
///
/// This implements the Fragmentation Gradient Descent (FGD) algorithm described
/// in Section 4.2 of the paper. It tries all possible allocations and chooses
/// the one that causes the minimum increase in fragmentation.
///
/// # Arguments
/// * `cluster` - Cluster resource helper
/// * `workload` - Workload object describing expected task distribution
/// * `shape` - WorkerShape to be allocated
/// * `reusable_workers` - Workers we could potentially re-use. This is helpful to avoid thrashing in our auto-scaling
///     algorithm. We assume these are the same shape as "shape", but do not check this.
/// * `worker_reuse_fragmentation_equivalent` - A reward for re-using workers.
///
/// # Returns
/// WorkerResources describing best allocation, or None if no allocation possible
pub fn find_best_allocation_using_fragmentation_gradient_descent(
    cluster: &mut rds::ClusterResources,
    workload: &Workload,
    shape: &rds::WorkerShape,
    reusable_workers: Option<&std::collections::HashMap<String, rds::Worker>>,
    worker_reuse_fragmentation_equivalent: f32,
) -> AllocationResult {
    // Shape::SpmdNodeMultiple is not implemented. This code path should not be reached.
    if let rds::WorkerShape::SpmdNodeMultiple(_) = shape {
        panic!("SpmdNodeMultiple is not implemented. This code path should not be reached.");
    }

    // Store all possible allocation options with their fragmentation impact
    let mut results: Vec<FragmentationResult> = Vec::new();

    // First, try reusing recently removed workers to avoid allocation thrashing
    // This helps prevent oscillation between creating and destroying workers
    if let Some(reuse_map) = reusable_workers {
        for worker in reuse_map.values() {
            let node = cluster
                .nodes
                .get_mut(&worker.allocation.node)
                .expect("node");
            // Check if this worker's allocation is still feasible
            if !node.can_allocate(&worker.allocation) {
                continue;
            }

            // Calculate fragmentation impact of reusing this worker
            let current_frag = estimate_fragmentation_on_node(node, workload);

            node.allocate(&worker.allocation).expect("allocate");
            let new_frag = estimate_fragmentation_on_node(node, workload);
            let new_remaining_resources = node.used_pool().total_num();
            node.release_allocation(&worker.allocation);

            // Record this reuse option for comparison
            results.push(FragmentationResult {
                fragmentation_before: current_frag,
                fragmentation_after: new_frag,
                node_remaining_resources: new_remaining_resources,
                worker_allocation: worker.allocation.clone(),
                maybe_reused_worker: Some(worker.clone()),
            });
        }
    }

    // Now explore all possible fresh allocations across the cluster
    for (node_id, node) in &mut cluster.nodes {
        // Skip nodes that cannot accommodate this shape
        if !node_can_allocate(node, shape) {
            continue;
        }

        // Calculate current fragmentation level for this node
        let current_frag = estimate_fragmentation_on_node(node, workload);

        // Find all possible ways to allocate this shape on this node
        let possible_allocations = find_possible_allocations_on_node(node, shape, node_id);

        // Evaluate the fragmentation impact of each possible allocation
        for allocation in possible_allocations {
            node.allocate(&allocation).expect("allocate");
            let new_frag = estimate_fragmentation_on_node(node, workload);
            let new_remaining_resources = node.free_pool().total_num();
            node.release_allocation(&allocation);
            // Record this allocation option for comparison
            results.push(FragmentationResult {
                fragmentation_before: current_frag,
                fragmentation_after: new_frag,
                node_remaining_resources: new_remaining_resources,
                worker_allocation: allocation,
                maybe_reused_worker: None, // This is a fresh allocation
            });
        }
    }

    // Return failure if no allocations are possible
    if results.is_empty() {
        return AllocationResult {
            did_allocate: false,
            resources: None,
            reused_worker: None,
        };
    }

    /// Cost function for comparing allocation options.
    ///
    /// Returns a tuple (fragmentation_change, -remaining_resources) for lexicographic ordering.
    /// Lower fragmentation change is preferred, with more remaining resources as tiebreaker.
    fn cost(x: &FragmentationResult, worker_reuse_fragmentation_equivalent: f32) -> (f32, f32) {
        let mut fragmentation_change = x.fragmentation_change();

        // Apply reuse bonus: reusing workers gets an equivalent fragmentation reduction
        // This helps prevent thrashing between allocation and deallocation
        if x.is_reused_worker() {
            fragmentation_change -= worker_reuse_fragmentation_equivalent;
        }

        // Return (primary_cost, secondary_cost) where:
        // - primary_cost: fragmentation change (lower is better)
        // - secondary_cost: negative remaining resources (higher remaining is better)
        (fragmentation_change, -x.node_remaining_resources)
    }

    // Select the allocation option with the best cost (minimum fragmentation increase)
    let best = results
        .into_iter()
        .min_by(|a, b| {
            let ca = cost(a, worker_reuse_fragmentation_equivalent);
            let cb = cost(b, worker_reuse_fragmentation_equivalent);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    AllocationResult {
        did_allocate: true,
        resources: Some(best.worker_allocation.clone()),
        reused_worker: best.maybe_reused_worker.clone(),
    }
}

/// Identifies best worker to remove to minimize resulting fragmentation.
///
/// This implements the worker removal strategy using FGD principles. It evaluates
/// removing each candidate worker and chooses the one that results in minimum
/// fragmentation increase.
///
/// # Arguments
/// * `cluster` - Cluster resource helper
/// * `workload` - Workload object describing expected task distribution  
/// * `potential_workers` - List of workers that could be removed
///
/// # Returns
/// Worker that should be removed to minimize fragmentation impact
pub fn find_worker_to_delete_using_fragmentation_gradient_descent(
    cluster: &mut rds::ClusterResources,
    workload: &Workload,
    potential_workers: &std::collections::HashMap<String, rds::Worker>,
) -> String {
    assert!(!potential_workers.is_empty());

    #[derive(Debug, Clone)]
    struct CandidateCost {
        frag_delta: f32,
        used_resources_before: f32,
        worker_id: String,
    }

    let mut changes: Vec<CandidateCost> = Vec::new();

    // Evaluate the fragmentation impact of removing each candidate worker
    for (worker_id, worker) in potential_workers.iter() {
        let node = cluster
            .nodes
            .get_mut(&worker.allocation.node)
            .expect("node");
        // Fragmentation before on this node
        let frag_before = estimate_fragmentation_on_node(node, workload);
        let used_resources_before = node.used_pool().total_num();

        // Simulate releasing the worker on a cloned NodeResources
        node.release_allocation(&worker.allocation);
        // Fragmentation after on this node
        let frag_after = estimate_fragmentation_on_node(node, workload);
        // Re-allocate the worker so that we do not actually mutate the cluster
        node.allocate(&worker.allocation).expect("allocate");

        changes.push(CandidateCost {
            frag_delta: frag_after - frag_before,
            used_resources_before,
            worker_id: worker_id.clone(),
        });
    }

    log::trace!("changes: {:?}", changes);
    // Select the worker whose removal results in the lowest cluster fragmentation
    // If multiple workers result in the same fragmentation, prefer removing the one
    // from the highest-utilized node.
    changes
        .into_iter()
        .min_by(|a, b| {
            // Cost tuples: (frag_delta, -remaining_resources_after)
            // Lower delta is preferred; if equal, prefer higher-utilized nodes.
            let ka = (a.frag_delta, -a.used_resources_before);
            let kb = (b.frag_delta, -b.used_resources_before);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("non-empty")
        .worker_id
}

#[derive(Debug, Clone)]
pub struct SpmdNodeMultipleAllocation {
    pub worker_allocations: Vec<rds::WorkerResources>,
}

/// Simple allocation logic for SPMDNodeMultiple.
///
/// SPMDNodeMultiple does not require any intensive search or fragmentation analysis.
/// This is because we know that the number of gpus per worker group is a multiple of the number of gpus per node.
/// Therefore, we can just allocate the first x nodes that are fully available.
pub fn find_best_allocation_for_spmd_node_multiple(
    cluster: &rds::ClusterResources,
    shape: &rds::SpmdNodeMultiple,
    // reusable_workers: Option<&std::collections::HashMap<String, rds::Worker>>,
) -> SpmdNodeMultipleAllocation {
    let num_nodes_needed = shape.num_gpu_actors_in_group as usize / shape.num_gpus_in_node as usize;

    // Get all nodes that can allocate the given shape
    // This will be nodes completely unallocated by
    let mut potential_nodes: Vec<&rds::NodeResources> = cluster
        .nodes
        .values()
        .filter(|n| node_can_allocate(n, &rds::WorkerShape::SpmdNodeMultiple(*shape)))
        .collect();

    if potential_nodes.len() < num_nodes_needed {
        return SpmdNodeMultipleAllocation {
            worker_allocations: Vec::new(),
        };
    }

    // Sort by node names and take the first num_nodes_needed
    potential_nodes.sort_by(|a, b| a.name.cmp(&b.name));
    let selected_nodes = &potential_nodes[..num_nodes_needed as usize];

    // Allocate the first num_nodes_needed nodes
    let mut worker_allocations = Vec::new();
    for node in selected_nodes {
        let allocation = rds::WorkerResources {
            node: node.name.clone().unwrap(),
            cpus: shape.num_cpus_needed_per_node(),
            gpus: node
                .gpus
                .iter()
                .enumerate()
                .map(|(offset, _g)| rds::GpuAllocation {
                    offset,
                    used_fraction: rds::FixedUtil::ONE,
                })
                .collect(),
        };
        worker_allocations.push(allocation);
    }

    SpmdNodeMultipleAllocation { worker_allocations }
}

// Simple deletion logic for SPMDNodeMultiple.
//
// This is the companion to `find_best_allocation_for_spmd_node_multiple`. SPMDNodeMultiple is a simple shape case so
// we can just delete any worker group really. No need to do any fancy fragmentation analysis.
// This is broken out as a seperate function as it will get more complex in the future with GB200.
pub fn find_worker_group_to_delete_for_spmd_node_multiple(
    _cluster: &mut rds::ClusterResources,
    _shape: &rds::SpmdNodeMultiple,
    potential_worker_groups: &std::collections::HashMap<String, rds::WorkerGroup>,
) -> String {
    // Select a random worker group to delete
    let worker_group_id = potential_worker_groups
        .keys()
        .next()
        .expect("non-empty")
        .clone();
    worker_group_id
}

// --------------------
// Tests (pure Rust)
// --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::private::scheduling::resources::{self as rds};
    use std::collections::HashMap;

    // --------------------
    // Test Helpers
    // --------------------

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    /// Creates a cluster from a list of nodes with sequential string IDs
    fn cluster_from_nodes(nodes: Vec<rds::NodeResources>) -> rds::ClusterResources {
        let mut map: HashMap<String, rds::NodeResources> = HashMap::new();
        for (i, n) in nodes.into_iter().enumerate() {
            map.insert(i.to_string(), n);
        }
        rds::ClusterResources { nodes: map }
    }

    /// Creates a uniform cluster with identical nodes
    fn uniform_cluster(
        num_nodes: usize,
        cpus_per_node: f32,
        gpus_per_node: usize,
    ) -> rds::ClusterResources {
        let nodes: Vec<rds::NodeResources> = (0..num_nodes)
            .map(|i| fresh_node(&format!("node-{}", i), cpus_per_node, gpus_per_node))
            .collect();
        cluster_from_nodes(nodes)
    }

    /// Creates a fresh node with all resources available
    fn fresh_node(name: &str, total_cpus: f32, num_gpus: usize) -> rds::NodeResources {
        let gpus: Vec<rds::GpuResources> = (0..num_gpus).map(|i| fresh_gpu(i as u8)).collect();

        rds::NodeResources {
            used_cpus: rds::FixedUtil::ZERO,
            total_cpus: rds::FixedUtil::from_num(total_cpus),
            gpus,
            name: Some(name.to_string()),
        }
    }

    /// Creates a node with specific resource usage
    fn node_with_usage(
        name: &str,
        used_cpus: f32,
        total_cpus: f32,
        gpus: Vec<rds::GpuResources>,
    ) -> rds::NodeResources {
        rds::NodeResources {
            used_cpus: rds::FixedUtil::from_num(used_cpus),
            total_cpus: rds::FixedUtil::from_num(total_cpus),
            gpus,
            name: Some(name.to_string()),
        }
    }

    /// Creates a fresh GPU with no allocation
    fn fresh_gpu(index: u8) -> rds::GpuResources {
        rds::GpuResources {
            index,
            uuid_: uuid::Uuid::new_v4(),
            used_fraction: rds::FixedUtil::ZERO,
        }
    }

    /// Creates a GPU with specific usage
    fn gpu_with_usage(index: u8, used_fraction: f32) -> rds::GpuResources {
        rds::GpuResources {
            index,
            uuid_: uuid::Uuid::new_v4(),
            used_fraction: rds::FixedUtil::from_num(used_fraction),
        }
    }

    /// Creates a test worker
    fn test_worker(
        id: &str,
        stage: &str,
        node: &str,
        cpus: f32,
        gpu_allocs: &[(usize, f32)],
    ) -> rds::Worker {
        let gpus: Vec<rds::GpuAllocation> = gpu_allocs
            .iter()
            .map(|&(offset, fraction)| rds::GpuAllocation {
                offset,
                used_fraction: rds::FixedUtil::from_num(fraction),
            })
            .collect();

        rds::Worker::new(
            id.to_string(),
            stage.to_string(),
            rds::WorkerResources {
                node: node.to_string(),
                cpus: rds::FixedUtil::from_num(cpus),
                gpus,
            },
        )
    }

    /// Creates a simple workload for testing
    fn simple_workload(cluster: &rds::ClusterResources) -> Workload {
        Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.5,
                        cpus: 2.0,
                        is_spmd: false,
                    }
                    .to_shape(cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(cluster)
                    .unwrap(),
                },
            ],
        }
    }

    /// Asserts that two floats are approximately equal
    fn assert_float_eq(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() < tolerance,
            "Expected {}, got {} (tolerance: {})",
            expected,
            actual,
            tolerance
        );
    }

    /// Asserts that two floats are approximately equal with default tolerance
    fn assert_approx_eq(actual: f32, expected: f32) {
        assert_float_eq(actual, expected, 1e-4); // More lenient tolerance for FixedUtil precision
    }

    // --------------------
    // Fragmentation Calculation Tests
    // --------------------

    #[test]
    fn test_fragmentation_calculation_cpu_only_shape() {
        // CPU-only shapes should consider all GPU resources as fragmented
        let node = node_with_usage(
            "test-node",
            2.0,                                                  // 2 CPUs used
            8.0,                                                  // 8 CPUs total
            vec![gpu_with_usage(0, 0.3), gpu_with_usage(1, 0.0)], // 0.7 + 1.0 = 1.7 available
        );
        let cluster = cluster_from_nodes(vec![node]);

        let cpu_only_shape = rds::Resources {
            gpus: 0.0,
            cpus: 2.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let fragmentation = calculate_unallocatable_gpus_fragment_for_shape_on_node(
            &cluster.nodes["0"],
            &cpu_only_shape,
        );

        // All available GPU resources (0.7 + 1.0 = 1.7) should be considered fragmented
        assert_approx_eq(fragmentation, 1.7);
    }

    #[test]
    fn test_fragmentation_calculation_fractional_gpu_shape() {
        // Test fragmentation calculation for fractional GPU requirements
        let node = node_with_usage(
            "test-node",
            0.0, // No CPUs used
            8.0, // 8 CPUs total
            vec![
                gpu_with_usage(0, 0.5), // 0.5 available
                gpu_with_usage(1, 0.0), // 1.0 available
                gpu_with_usage(2, 0.8), // 0.2 available
            ],
        );
        let cluster = cluster_from_nodes(vec![node]);

        let fractional_shape = rds::Resources {
            gpus: 0.6, // Requires 0.6 GPU fraction
            cpus: 2.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let fragmentation = calculate_unallocatable_gpus_fragment_for_shape_on_node(
            &cluster.nodes["0"],
            &fractional_shape,
        );

        // GPU 0 (0.5 available) and GPU 2 (0.2 available) cannot accommodate 0.6 fraction
        // Only GPU 1 (1.0 available) can be used, so 0.5 + 0.2 = 0.7 is fragmented
        assert_approx_eq(fragmentation, 0.7);
    }

    #[test]
    fn test_fragmentation_calculation_whole_gpu_shape() {
        // Test fragmentation for whole GPU requirements
        let node = node_with_usage(
            "test-node",
            0.0,
            16.0,
            vec![
                gpu_with_usage(0, 0.5), // Partially used, can't be used for whole GPU
                gpu_with_usage(1, 0.0), // Fully available
                gpu_with_usage(2, 0.0), // Fully available
            ],
        );
        let cluster = cluster_from_nodes(vec![node]);

        let whole_gpu_shape = rds::Resources {
            gpus: 1.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let fragmentation = calculate_unallocatable_gpus_fragment_for_shape_on_node(
            &cluster.nodes["0"],
            &whole_gpu_shape,
        );

        // GPU 0 has 0.5 available but can't be used for whole GPU allocation
        assert_approx_eq(fragmentation, 0.5);
    }

    #[test]
    fn test_fragmentation_calculation_insufficient_resources() {
        // Test when node cannot allocate the shape at all
        let node = node_with_usage(
            "test-node",
            7.0, // Only 1 CPU available
            8.0,
            vec![gpu_with_usage(0, 0.0)], // 1.0 GPU available
        );
        let cluster = cluster_from_nodes(vec![node]);

        let large_shape = rds::Resources {
            gpus: 1.0,
            cpus: 4.0, // Requires 4 CPUs but only 1 available
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let fragmentation = calculate_unallocatable_gpus_fragment_for_shape_on_node(
            &cluster.nodes["0"],
            &large_shape,
        );

        // All available GPU resources should be considered fragmented
        assert_approx_eq(fragmentation, 1.0);
    }

    #[test]
    fn test_workload_based_fragmentation_estimation() {
        // Test fragmentation estimation with a realistic workload
        let node = node_with_usage(
            "test-node",
            2.0, // 6 CPUs available
            8.0,
            vec![
                gpu_with_usage(0, 0.3), // 0.7 available
                gpu_with_usage(1, 0.0), // 1.0 available
            ],
        );
        let cluster = cluster_from_nodes(vec![node]);

        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.7, // 70% of tasks need 0.5 GPU
                    shape: rds::Resources {
                        gpus: 0.5,
                        cpus: 2.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.3, // 30% of tasks need 1.0 GPU
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let fragmentation = estimate_fragmentation_on_node(&cluster.nodes["0"], &workload);

        // For 0.5 GPU shape: both GPUs can be used, no fragmentation
        // For 1.0 GPU shape: GPU 0 (0.7 available) cannot be used, so 0.7 is fragmented
        // Weighted average: 0.7 * 0.0 + 0.3 * 0.7 = 0.21
        assert_approx_eq(fragmentation, 0.21);
    }

    #[test]
    fn test_cluster_fragmentation_aggregation() {
        // Test that cluster fragmentation correctly aggregates node fragmentations
        let cluster = cluster_from_nodes(vec![
            node_with_usage("node-0", 0.0, 8.0, vec![gpu_with_usage(0, 0.5)]), // 0.5 available
            node_with_usage("node-1", 0.0, 8.0, vec![gpu_with_usage(0, 0.3)]), // 0.7 available
        ]);

        let workload = Workload {
            stages: vec![Stage {
                frequency: 1.0,
                shape: rds::Resources {
                    gpus: 1.0, // Requires whole GPU
                    cpus: 2.0,
                    is_spmd: false,
                }
                .to_shape(&cluster)
                .unwrap(),
            }],
        };

        let cluster_fragmentation = estimate_fragmentation_on_cluster(&cluster, &workload);

        // Node 0: 0.5 fragmented (partial GPU can't be used for whole GPU)
        // Node 1: 0.7 fragmented (partial GPU can't be used for whole GPU)
        // Total: 0.5 + 0.7 = 1.2
        assert_approx_eq(cluster_fragmentation, 1.2);
    }

    // --------------------
    // Allocation Algorithm Tests
    // --------------------

    #[test]
    fn test_allocation_success_fractional_gpu() {
        // Test successful allocation of fractional GPU workload
        let mut cluster = uniform_cluster(1, 16.0, 2);
        let workload = simple_workload(&cluster);

        let shape = rds::Resources {
            gpus: 0.5,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            0.0, // No reuse bonus
        );

        assert!(result.did_allocate, "Should successfully allocate");
        let allocation = result.resources.unwrap();
        assert_eq!(allocation.node, "0");
        assert_eq!(allocation.cpus, rds::FixedUtil::from_num(4.0));
        assert_eq!(allocation.gpus.len(), 1);
        assert_eq!(
            allocation.gpus[0].used_fraction,
            rds::FixedUtil::from_num(0.5)
        );
    }

    #[test]
    fn test_allocation_success_whole_gpu() {
        // Test successful allocation of whole GPU workload
        let mut cluster = uniform_cluster(1, 16.0, 2);
        let workload = simple_workload(&cluster);

        let shape = rds::Resources {
            gpus: 1.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should successfully allocate");
        let allocation = result.resources.unwrap();
        assert_eq!(allocation.cpus, rds::FixedUtil::from_num(4.0));
        assert_eq!(allocation.gpus.len(), 1);
        assert_eq!(allocation.gpus[0].used_fraction, rds::FixedUtil::ONE);
    }

    #[test]
    fn test_allocation_success_cpu_only() {
        // Test successful allocation of CPU-only workload
        let mut cluster = uniform_cluster(1, 16.0, 2);
        let workload = simple_workload(&cluster);

        let shape = rds::Resources {
            gpus: 0.0,
            cpus: 8.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should successfully allocate");
        let allocation = result.resources.unwrap();
        assert_eq!(allocation.cpus, rds::FixedUtil::from_num(8.0));
        assert!(
            allocation.gpus.is_empty(),
            "CPU-only should have no GPU allocation"
        );
    }

    #[test]
    fn test_allocation_failure_insufficient_resources() {
        // Test allocation failure when resources are insufficient
        let mut cluster = uniform_cluster(1, 8.0, 1);
        let workload = simple_workload(&cluster);

        // Allocate most resources first
        let existing_worker = test_worker("existing", "stage", "0", 6.0, &[(0, 0.8)]);
        cluster.allocate(&existing_worker.allocation).unwrap();

        // Try to allocate more than what's available
        let shape = rds::Resources {
            gpus: 0.5,
            cpus: 4.0, // Only 2 CPUs left
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            0.0,
        );

        assert!(!result.did_allocate, "Should fail to allocate");
        assert!(result.resources.is_none());
    }

    #[test]
    fn test_allocation_prefers_lower_fragmentation() {
        // Test that allocation chooses the option with lower fragmentation impact
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage("node-0", 0.0, 16.0, vec![gpu_with_usage(0, 0.3)]), // 0.7 available
            node_with_usage("node-1", 0.0, 16.0, vec![gpu_with_usage(0, 0.0)]), // 1.0 available
        ]);

        // Workload that heavily weights whole GPU tasks
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.1,
                    shape: rds::Resources {
                        gpus: 0.5,
                        cpus: 2.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.9, // Most tasks need whole GPU
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let shape = rds::Resources {
            gpus: 0.5,
            cpus: 2.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            0.0,
        );

        let allocation = result.resources.unwrap();
        // Should prefer node-0 because allocating 0.5 on node-1 would fragment the whole GPU
        // that's more valuable for the workload
        assert_eq!(allocation.node, "0");
    }

    #[test]
    fn test_worker_reuse_functionality() {
        // Test that worker reuse works correctly and gets preference
        let mut cluster = uniform_cluster(1, 16.0, 2);
        let workload = simple_workload(&cluster);

        // Create a reusable worker
        let reusable_worker = test_worker("reusable", "stage", "0", 4.0, &[(0, 0.5)]);
        let mut reuse_map = HashMap::new();
        reuse_map.insert("reusable".to_string(), reusable_worker.clone());

        let shape = rds::Resources {
            gpus: 0.5,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            Some(&reuse_map),
            1.0, // Reuse bonus
        );

        assert!(result.did_allocate, "Should successfully allocate");
        assert!(result.reused_worker.is_some(), "Should reuse the worker");
        let reused = result.reused_worker.unwrap();
        assert_eq!(reused.id, "reusable");
    }

    #[test]
    fn test_worker_reuse_vs_fresh_allocation() {
        // Test that reuse bonus affects allocation decisions
        let mut cluster = uniform_cluster(2, 16.0, 1);
        let workload = simple_workload(&cluster);

        // Pre-allocate something on node 0 to make it less attractive
        let existing = test_worker("existing", "stage", "0", 8.0, &[(0, 0.5)]);
        cluster.allocate(&existing.allocation).unwrap();

        // Create reusable worker on the less attractive node 0
        let reusable_worker = test_worker("reusable", "stage", "0", 4.0, &[]);
        let mut reuse_map = HashMap::new();
        reuse_map.insert("reusable".to_string(), reusable_worker.clone());

        let shape = rds::Resources {
            gpus: 0.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        // Without reuse bonus, should prefer fresh node
        let result_no_bonus = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            Some(&reuse_map),
            0.0, // No reuse bonus
        );

        // With significant reuse bonus, should prefer reuse
        let result_with_bonus = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            Some(&reuse_map),
            10.0, // Large reuse bonus
        );

        // The behavior should be different based on reuse bonus
        let no_bonus_reused = result_no_bonus.reused_worker.is_some();
        let with_bonus_reused = result_with_bonus.reused_worker.is_some();

        // With a large bonus, reuse should be more likely
        assert!(
            !no_bonus_reused || with_bonus_reused,
            "Reuse bonus should make reuse more attractive"
        );
    }

    // --------------------
    // Worker Deletion Tests
    // --------------------

    #[test]
    fn test_worker_deletion_chooses_least_impactful() {
        init();

        // Test that deletion algorithm chooses the worker with least fragmentation impact
        let mut cluster = uniform_cluster(1, 16.0, 2);

        // Create workers with different fragmentation impacts
        let workers = vec![
            test_worker("worker1", "stage1", "0", 4.0, &[(0, 0.7)]), // Leaves 0.3 on GPU 0
            test_worker("worker2", "stage2", "0", 4.0, &[(1, 0.9)]), // Leaves 0.1 on GPU 1
        ];

        // Allocate both workers
        for worker in &workers {
            cluster.allocate(&worker.allocation).unwrap();
        }

        // Workload that heavily prefers partial GPU allocations
        let workload = Workload {
            stages: vec![Stage {
                frequency: 1.0,
                shape: rds::Resources {
                    gpus: 0.3,
                    cpus: 4.0,
                    is_spmd: false,
                }
                .to_shape(&cluster)
                .unwrap(),
            }],
        };

        let worker_map: HashMap<String, rds::Worker> =
            workers.into_iter().map(|w| (w.id.clone(), w)).collect();

        let to_delete = find_worker_to_delete_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &worker_map,
        );

        // Should delete worker2 because removing it removes more of a GPU
        assert_eq!(to_delete, "worker2");
    }

    #[test]
    fn test_worker_deletion_prefers_high_utilization_nodes() {
        // Test tiebreaker: prefer deleting from higher-utilization nodes
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage("node-0", 12.0, 16.0, vec![gpu_with_usage(0, 0.5)]), // High CPU usage
            node_with_usage("node-1", 4.0, 16.0, vec![gpu_with_usage(0, 0.5)]),  // Low CPU usage
        ]);

        let workers = vec![
            test_worker("worker1", "stage", "0", 2.0, &[(0, 0.3)]), // On high-usage node
            test_worker("worker2", "stage", "1", 2.0, &[(0, 0.3)]), // On low-usage node
        ];

        for worker in &workers {
            cluster.allocate(&worker.allocation).unwrap();
        }

        let workload = simple_workload(&cluster);
        let worker_map: HashMap<String, rds::Worker> =
            workers.into_iter().map(|w| (w.id.clone(), w)).collect();

        let to_delete = find_worker_to_delete_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &worker_map,
        );

        // Should prefer deleting from the higher-utilization node
        assert_eq!(to_delete, "worker1");
    }

    // --------------------
    // Edge Case and Helper Function Tests
    // --------------------

    #[test]
    fn test_gpu_can_be_used_to_allocate_edge_cases() {
        // Test edge cases for GPU allocation feasibility
        let cluster = uniform_cluster(1, 16.0, 1);

        // Test cases: (gpu_usage, shape_gpus, expected_result)
        let test_cases = vec![
            (0.0, 0.5, true),  // Fresh GPU can accommodate fractional
            (0.5, 0.5, true),  // Exactly fits remaining capacity
            (0.6, 0.5, false), // Insufficient remaining capacity
            (0.0, 1.0, true),  // Fresh GPU can accommodate whole GPU
            (0.1, 1.0, false), // Partially used GPU cannot accommodate whole GPU
            (1.0, 0.5, false), // Fully used GPU cannot accommodate anything
        ];

        for (gpu_usage, shape_gpus, expected) in test_cases {
            let gpu = gpu_with_usage(0, gpu_usage);
            let shape = rds::Resources {
                gpus: shape_gpus,
                cpus: 2.0,
                is_spmd: false,
            }
            .to_shape(&cluster)
            .unwrap();

            let result = gpu_can_be_used_to_allocate(&gpu, &shape);
            assert_eq!(
                result, expected,
                "GPU usage: {}, Shape GPUs: {}, Expected: {}, Got: {}",
                gpu_usage, shape_gpus, expected, result
            );
        }
    }

    #[test]
    fn test_node_can_allocate_cpu_constraints() {
        // Test CPU constraint checking in node allocation feasibility
        let node = node_with_usage("test", 6.0, 8.0, vec![fresh_gpu(0)]); // 2 CPUs available
        let cluster = cluster_from_nodes(vec![node]);

        let test_cases = vec![
            (1.0, 0.5, true),  // 1 CPU + 0.5 GPU - should fit
            (2.0, 0.5, true),  // 2 CPUs + 0.5 GPU - exactly fits
            (3.0, 0.5, false), // 3 CPUs + 0.5 GPU - too many CPUs
            (1.0, 1.0, true),  // 1 CPU + 1 GPU - should fit
            (2.0, 1.0, true),  // 2 CPUs + 1 GPU - exactly fits
            (3.0, 1.0, false), // 3 CPUs + 1 GPU - too many CPUs
        ];

        for (cpus, gpus, expected) in test_cases {
            let shape = rds::Resources {
                gpus,
                cpus,
                is_spmd: false,
            }
            .to_shape(&cluster)
            .unwrap();

            let result = node_can_allocate(&cluster.nodes["0"], &shape);
            assert_eq!(
                result, expected,
                "CPUs: {}, GPUs: {}, Expected: {}, Got: {}",
                cpus, gpus, expected, result
            );
        }
    }

    #[test]
    fn test_find_possible_allocations_comprehensive() {
        // Test allocation possibility finding for different scenarios
        let node = node_with_usage(
            "test",
            0.0,
            16.0,
            vec![
                gpu_with_usage(0, 0.3), // 0.7 available
                fresh_gpu(1),           // 1.0 available
                gpu_with_usage(2, 0.9), // 0.1 available
            ],
        );
        let cluster = cluster_from_nodes(vec![node]);

        // CPU-only allocation
        let cpu_shape = rds::Resources {
            gpus: 0.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();
        let cpu_allocs = find_possible_allocations_on_node(&cluster.nodes["0"], &cpu_shape, "0");
        assert_eq!(cpu_allocs.len(), 1);
        assert!(cpu_allocs[0].gpus.is_empty());

        // Fractional GPU allocation - should find allocations on GPUs 0 and 1
        let frac_shape = rds::Resources {
            gpus: 0.5,
            cpus: 2.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();
        let frac_allocs = find_possible_allocations_on_node(&cluster.nodes["0"], &frac_shape, "0");
        assert_eq!(frac_allocs.len(), 2); // Can allocate on GPU 0 and GPU 1

        // Whole GPU allocation - should only find allocation on GPU 1
        let whole_shape = rds::Resources {
            gpus: 1.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();
        let whole_allocs =
            find_possible_allocations_on_node(&cluster.nodes["0"], &whole_shape, "0");
        assert_eq!(whole_allocs.len(), 1); // Only GPU 1 is fully available
        assert_eq!(whole_allocs[0].gpus[0].offset, 1);
    }

    #[test]
    fn test_allocation_with_no_available_nodes() {
        // Test allocation when no nodes can satisfy the request
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage("node-0", 15.0, 16.0, vec![gpu_with_usage(0, 1.0)]), // Almost full
            node_with_usage("node-1", 14.0, 16.0, vec![gpu_with_usage(0, 0.9)]), // Almost full
        ]);

        let workload = simple_workload(&cluster);
        let large_shape = rds::Resources {
            gpus: 1.0,
            cpus: 8.0, // More CPUs than available on any node
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &large_shape,
            None,
            0.0,
        );

        assert!(
            !result.did_allocate,
            "Should fail when no nodes can satisfy request"
        );
        assert!(result.resources.is_none());
        assert!(result.reused_worker.is_none());
    }

    #[test]
    fn test_empty_workload_fragmentation() {
        // Test fragmentation calculation with empty workload
        let node = fresh_node("test", 8.0, 2);
        let cluster = cluster_from_nodes(vec![node]);
        let empty_workload = Workload { stages: vec![] };

        let fragmentation = estimate_fragmentation_on_node(&cluster.nodes["0"], &empty_workload);
        assert_approx_eq(fragmentation, 0.0);

        let cluster_fragmentation = estimate_fragmentation_on_cluster(&cluster, &empty_workload);
        assert_approx_eq(cluster_fragmentation, 0.0);
    }

    #[test]
    fn test_allocation_prefers_less_utilized_nodes() {
        // Test that allocation prefers nodes with more available resources
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage("node-0", 12.0, 16.0, vec![fresh_gpu(0)]), // High CPU usage
            node_with_usage("node-1", 2.0, 16.0, vec![fresh_gpu(0)]),  // Low CPU usage
        ]);

        let workload = simple_workload(&cluster);
        let shape = rds::Resources {
            gpus: 0.0,
            cpus: 2.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            0.0,
        );

        let allocation = result.resources.unwrap();
        // Should prefer the less utilized node (node-1)
        assert_eq!(allocation.node, "1");
    }

    // --------------------
    // Weird Shapes and Complex Fragmentation Tests
    // --------------------

    #[test]
    fn test_fragmentation_with_tiny_fractional_shapes() {
        init();

        // Test fragmentation decisions with very small fractional GPU requirements
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.95), // Only 0.05 available
                    gpu_with_usage(1, 0.0),  // 1.0 available
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.0), // 1.0 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
        ]);

        // Workload heavily weighted toward tiny allocations
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.8,
                    shape: rds::Resources {
                        gpus: 0.05, // Very small allocation
                        cpus: 1.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.2,
                    shape: rds::Resources {
                        gpus: 1.0, // Full GPU
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let tiny_shape = rds::Resources {
            gpus: 0.05,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &tiny_shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should allocate tiny shape");
        let allocation = result.resources.unwrap();

        // Should prefer node-0 GPU-0 because it can exactly fit the tiny requirement
        // without fragmenting a full GPU that's more valuable for the workload
        assert_eq!(allocation.node, "0");
        assert_eq!(allocation.gpus[0].offset, 0);
    }

    #[test]
    fn test_fragmentation_with_large_fractional_shapes() {
        init();

        // Test with shapes that require most of a GPU but not all
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.1), // 0.9 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.0), // 1.0 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
        ]);

        // Workload that prefers whole GPUs
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.2,
                    shape: rds::Resources {
                        gpus: 0.9, // Large fractional
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.8,
                    shape: rds::Resources {
                        gpus: 1.0, // Whole GPU
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let large_frac_shape = rds::Resources {
            gpus: 0.9,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &large_frac_shape,
            None,
            0.0,
        );

        assert!(
            result.did_allocate,
            "Should allocate large fractional shape"
        );
        let allocation = result.resources.unwrap();

        // Should prefer node-0 GPU-0 because it already has some usage,
        // preserving the full GPUs for whole GPU allocations
        assert_eq!(allocation.node, "0");
        assert_eq!(allocation.gpus[0].offset, 0);
    }

    #[test]
    fn test_deletion_with_asymmetric_gpu_usage() {
        init();

        // Test deletion decisions when workers have very different GPU usage patterns
        let mut cluster = uniform_cluster(1, 32.0, 4);

        let workers = vec![
            test_worker("tiny_user", "stage", "0", 2.0, &[(0, 0.1)]), // Uses 10% of GPU 0
            test_worker("medium_user", "stage", "0", 4.0, &[(1, 0.5)]), // Uses 50% of GPU 1
            test_worker("large_user", "stage", "0", 6.0, &[(2, 0.9)]), // Uses 90% of GPU 2
            test_worker("multi_user", "stage", "0", 8.0, &[(3, 0.3), (0, 0.2)]), // Uses multiple GPUs
        ];

        for worker in &workers {
            cluster.allocate(&worker.allocation).unwrap();
        }

        // Workload that heavily prefers large allocations
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.1,
                    shape: rds::Resources {
                        gpus: 0.2,
                        cpus: 2.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.9,
                    shape: rds::Resources {
                        gpus: 0.8,
                        cpus: 6.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let worker_map: HashMap<String, rds::Worker> =
            workers.into_iter().map(|w| (w.id.clone(), w)).collect();

        let to_delete = find_worker_to_delete_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &worker_map,
        );

        // Should delete one of the workers - verify it's a valid choice
        assert!(
            ["tiny_user", "medium_user", "large_user", "multi_user"].contains(&to_delete.as_str()),
            "Should delete a valid worker, got: {}",
            to_delete
        );
    }

    #[test]
    fn test_allocation_with_fragmented_multi_gpu_requirements() {
        init();

        // Test allocation of multi-GPU shapes when cluster is fragmented
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                32.0,
                vec![
                    gpu_with_usage(0, 0.5), // 0.5 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                    gpu_with_usage(2, 0.8), // 0.2 available
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                32.0,
                vec![
                    gpu_with_usage(0, 0.0), // 1.0 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                    gpu_with_usage(2, 0.0), // 1.0 available
                ],
            ),
        ]);

        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.3,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.7,
                    shape: rds::Resources {
                        gpus: 2.0,
                        cpus: 8.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let multi_gpu_shape = rds::Resources {
            gpus: 2.0,
            cpus: 8.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &multi_gpu_shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should allocate multi-GPU shape");
        let allocation = result.resources.unwrap();

        // Should prefer node-1 because it has more contiguous whole GPUs available
        assert_eq!(allocation.node, "1");
        assert_eq!(allocation.gpus.len(), 2);

        // Both GPUs should be fully allocated
        for gpu_alloc in &allocation.gpus {
            assert_eq!(gpu_alloc.used_fraction, rds::FixedUtil::ONE);
        }
    }

    #[test]
    fn test_cpu_heavy_vs_gpu_heavy_workload_decisions() {
        init();

        // Test allocation decisions with workloads that have very different CPU/GPU ratios
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage("cpu-heavy", 8.0, 64.0, vec![gpu_with_usage(0, 0.0)]), // Lots of CPU available
            node_with_usage("gpu-heavy", 60.0, 64.0, vec![gpu_with_usage(0, 0.0)]), // Little CPU available
        ]);

        // CPU-heavy workload
        let cpu_workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.8,
                    shape: rds::Resources {
                        gpus: 0.1,
                        cpus: 16.0, // High CPU requirement
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.2,
                    shape: rds::Resources {
                        gpus: 0.0,
                        cpus: 32.0, // CPU-only
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let cpu_shape = rds::Resources {
            gpus: 0.1,
            cpus: 16.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &cpu_workload,
            &cpu_shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should allocate CPU-heavy shape");
        let allocation = result.resources.unwrap();

        // Should allocate to one of the available nodes
        assert_eq!(allocation.node, "0");
    }

    #[test]
    fn test_spmd_smaller_than_node_fragmentation() {
        init();

        // Test SPMD allocation that uses part of a node
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                32.0,
                vec![
                    gpu_with_usage(0, 0.5), // Partially used
                    gpu_with_usage(1, 0.0), // Available
                    gpu_with_usage(2, 0.0), // Available
                    gpu_with_usage(3, 0.0), // Available
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                32.0,
                vec![
                    gpu_with_usage(0, 0.0), // All available
                    gpu_with_usage(1, 0.0),
                    gpu_with_usage(2, 0.0),
                    gpu_with_usage(3, 0.0),
                ],
            ),
        ]);

        // Workload with mix of SPMD and regular tasks
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::SpmdSmallerThanNode(
                        rds::SpmdSmallerThanNodeResources {
                            num_gpu_actors_in_group: 2,
                            num_gpus_in_node: 4,
                            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
                        },
                    ),
                },
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let spmd_shape = rds::WorkerShape::SpmdSmallerThanNode(rds::SpmdSmallerThanNodeResources {
            num_gpu_actors_in_group: 2,
            num_gpus_in_node: 4,
            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
        });

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &spmd_shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should allocate SPMD shape");
        let allocation = result.resources.unwrap();

        // SPMD smaller than node allocates the requested number of GPUs
        assert!(
            allocation.gpus.len() >= 2,
            "Should allocate at least 2 GPUs, got: {}",
            allocation.gpus.len()
        );
        // Should allocate to one of the available nodes
        assert_eq!(allocation.node, "1");
    }

    #[test]
    fn test_extreme_workload_skew_allocation_decisions() {
        init();

        // Test with extremely skewed workload frequencies
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.1), // 0.9 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.0), // 1.0 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
        ]);

        // Extremely skewed workload - 99% tiny tasks, 1% large tasks
        let skewed_workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.99,
                    shape: rds::Resources {
                        gpus: 0.05,
                        cpus: 1.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.01,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        let tiny_shape = rds::Resources {
            gpus: 0.05,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &skewed_workload,
            &tiny_shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should allocate with skewed workload");
        let allocation = result.resources.unwrap();

        // With such extreme skew toward tiny tasks, should prefer using the already
        // partially allocated GPU rather than fragmenting a fresh one
        assert_eq!(allocation.node, "0");
        assert_eq!(allocation.gpus[0].offset, 0);
    }

    // --------------------
    // Multi-GPU and SPMD Tests
    // --------------------

    #[test]
    fn test_multi_gpu_allocation() {
        // Test allocation of multi-GPU workloads
        let mut cluster = uniform_cluster(1, 32.0, 4);
        let workload = simple_workload(&cluster);

        let multi_gpu_shape = rds::Resources {
            gpus: 2.0,
            cpus: 8.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &multi_gpu_shape,
            None,
            0.0,
        );

        assert!(
            result.did_allocate,
            "Should successfully allocate multi-GPU workload"
        );
        let allocation = result.resources.unwrap();
        assert_eq!(allocation.gpus.len(), 2);
        for gpu_alloc in &allocation.gpus {
            assert_eq!(gpu_alloc.used_fraction, rds::FixedUtil::ONE);
        }
    }

    #[test]
    fn test_spmd_node_multiple_allocation() {
        // Test SPMD allocation that spans multiple nodes
        let cluster = uniform_cluster(4, 32.0, 2); // 4 nodes, 2 GPUs each

        let spmd_shape = rds::SpmdNodeMultiple {
            num_gpu_actors_in_group: 4, // Total 4 GPUs needed
            num_gpus_in_node: 2,        // 2 GPUs per node
            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
        };

        let result = find_best_allocation_for_spmd_node_multiple(&cluster, &spmd_shape);

        assert_eq!(result.worker_allocations.len(), 2); // Should use 2 nodes
        for allocation in &result.worker_allocations {
            assert_eq!(allocation.gpus.len(), 2); // Each node contributes 2 GPUs
            assert_eq!(allocation.cpus, rds::FixedUtil::from_num(8.0)); // 4 CPUs per GPU * 2 GPUs
        }
    }

    #[test]
    fn test_spmd_node_multiple_insufficient_nodes() {
        // Test SPMD allocation when insufficient nodes are available
        let cluster = uniform_cluster(1, 32.0, 2); // Only 1 node available

        let spmd_shape = rds::SpmdNodeMultiple {
            num_gpu_actors_in_group: 4, // Needs 4 GPUs total
            num_gpus_in_node: 2,        // 2 GPUs per node, so needs 2 nodes
            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
        };

        let result = find_best_allocation_for_spmd_node_multiple(&cluster, &spmd_shape);

        assert!(
            result.worker_allocations.is_empty(),
            "Should fail when insufficient nodes"
        );
    }

    #[test]
    fn test_spmd_smaller_than_node_allocation_uses_correct_gpu_count() {
        // Test that SpmdSmallerThanNode uses num_gpu_actors_in_group, not num_gpus_in_node
        // This is a regression test for a bug where it was using num_gpus_in_node incorrectly

        // Create a node with 8 GPUs (simulating the bug scenario)
        let node = fresh_node("test-node", 32.0, 8);
        let cluster = cluster_from_nodes(vec![node]);

        // Create an SPMD shape requesting only 2 GPUs (not all 8)
        let spmd_shape = rds::WorkerShape::SpmdSmallerThanNode(rds::SpmdSmallerThanNodeResources {
            num_gpu_actors_in_group: 2, // Request only 2 GPUs
            num_gpus_in_node: 8,        // Node has 8 GPUs total
            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
        });

        // Find possible allocations
        let allocations = find_possible_allocations_on_node(&cluster.nodes["0"], &spmd_shape, "0");

        // Should get exactly one allocation
        assert_eq!(allocations.len(), 1, "Should find exactly one allocation");

        let allocation = &allocations[0];

        // Critical assertion: Should only allocate 2 GPUs, not 8
        assert_eq!(
            allocation.gpus.len(),
            2,
            "Should allocate exactly 2 GPUs (num_gpu_actors_in_group), not 8 (num_gpus_in_node)"
        );

        // Verify the GPUs are fully allocated
        for gpu_alloc in &allocation.gpus {
            assert_eq!(
                gpu_alloc.used_fraction,
                rds::FixedUtil::ONE,
                "Each GPU should be fully allocated"
            );
        }

        // Verify CPU allocation (2 GPUs * 4 CPUs per actor = 8 CPUs)
        assert_eq!(
            allocation.cpus,
            rds::FixedUtil::from_num(8.0),
            "Should allocate 8 CPUs (2 actors * 4 CPUs per actor)"
        );

        // Verify the allocated GPUs are the first ones (indices 0 and 1)
        assert_eq!(allocation.gpus[0].offset, 0);
        assert_eq!(allocation.gpus[1].offset, 1);
    }

    #[test]
    fn test_spmd_smaller_than_node_gradient_descent_allocation() {
        // Test that the full fragmentation gradient descent algorithm correctly allocates
        // SpmdSmallerThanNode with the right number of GPUs

        // Create a cluster with one node having 8 GPUs
        let mut cluster = uniform_cluster(1, 64.0, 8);

        // Create a workload that includes SPMD tasks
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.5,
                    shape: rds::WorkerShape::SpmdSmallerThanNode(
                        rds::SpmdSmallerThanNodeResources {
                            num_gpu_actors_in_group: 2,
                            num_gpus_in_node: 8,
                            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
                        },
                    ),
                },
                Stage {
                    frequency: 0.5,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        // Create the SPMD shape to allocate
        let spmd_shape = rds::WorkerShape::SpmdSmallerThanNode(rds::SpmdSmallerThanNodeResources {
            num_gpu_actors_in_group: 2, // Request only 2 GPUs
            num_gpus_in_node: 8,        // Node has 8 GPUs total
            num_cpus_per_actor: rds::FixedUtil::from_num(4.0),
        });

        // Use the fragmentation gradient descent algorithm
        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &spmd_shape,
            None,
            0.0,
        );

        // Should successfully allocate
        assert!(
            result.did_allocate,
            "Should successfully allocate SPMD shape"
        );

        let allocation = result.resources.unwrap();

        // Critical assertion: Should allocate exactly 2 GPUs, not 8
        assert_eq!(
            allocation.gpus.len(),
            2,
            "Should allocate exactly 2 GPUs (num_gpu_actors_in_group), not 8 (num_gpus_in_node)"
        );

        // Verify each GPU is fully allocated
        for gpu_alloc in &allocation.gpus {
            assert_eq!(
                gpu_alloc.used_fraction,
                rds::FixedUtil::ONE,
                "Each GPU should be fully allocated for SPMD"
            );
        }

        // Verify CPU allocation
        assert_eq!(
            allocation.cpus,
            rds::FixedUtil::from_num(8.0),
            "Should allocate 8 CPUs (2 actors * 4 CPUs per actor)"
        );

        // Now allocate the worker and verify we can still allocate more
        cluster.allocate(&allocation).unwrap();

        // Should still have 6 GPUs available
        let remaining_gpus = cluster.nodes["0"]
            .gpus
            .iter()
            .filter(|g| g.is_fully_unallocated())
            .count();
        assert_eq!(
            remaining_gpus, 6,
            "Should have 6 GPUs remaining after allocating 2"
        );

        // Try to allocate another 2-GPU SPMD task
        let result2 = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &spmd_shape,
            None,
            0.0,
        );

        assert!(
            result2.did_allocate,
            "Should be able to allocate another 2-GPU SPMD task"
        );

        let allocation2 = result2.resources.unwrap();
        assert_eq!(
            allocation2.gpus.len(),
            2,
            "Second allocation should also use exactly 2 GPUs"
        );
    }

    #[test]
    fn test_complex_mixed_workload_scenario() {
        // Integration test with a complex, realistic scenario
        let mut cluster = uniform_cluster(3, 64.0, 4);

        // Pre-allocate some resources to create fragmentation
        let existing_workers = vec![
            test_worker("existing1", "stage", "0", 16.0, &[(0, 0.3), (1, 0.7)]),
            test_worker("existing2", "stage", "1", 32.0, &[(0, 1.0), (1, 1.0)]),
            test_worker("existing3", "stage", "2", 8.0, &[(0, 0.5)]),
        ];

        for worker in &existing_workers {
            cluster.allocate(&worker.allocation).unwrap();
        }

        // Complex workload with multiple stage types
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.5,
                        cpus: 4.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.3,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 8.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.2,
                    shape: rds::Resources {
                        gpus: 2.0,
                        cpus: 16.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.1,
                    shape: rds::Resources {
                        gpus: 0.0,
                        cpus: 8.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        // Try to allocate a new fractional GPU workload
        let new_shape = rds::Resources {
            gpus: 0.7,
            cpus: 6.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &new_shape,
            None,
            0.0,
        );

        assert!(
            result.did_allocate,
            "Should find allocation in complex scenario"
        );
        let allocation = result.resources.unwrap();

        // Verify the allocation is valid
        assert_eq!(allocation.cpus, rds::FixedUtil::from_num(6.0));
        assert_eq!(allocation.gpus.len(), 1);
        assert_eq!(
            allocation.gpus[0].used_fraction,
            rds::FixedUtil::from_num(0.7)
        );

        // Should allocate to one of the available nodes
        assert!(
            allocation.node == "0" || allocation.node == "1" || allocation.node == "2",
            "Should allocate to a valid node, got: {}",
            allocation.node
        );
    }

    #[test]
    fn test_worker_reuse_with_fragmentation_considerations() {
        init();

        // Test that worker reuse considers fragmentation impact
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.8), // 0.2 available - bad for most workloads
                    gpu_with_usage(1, 0.0), // 1.0 available - good for most workloads
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.0), // 1.0 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
        ]);

        // Workload that prefers larger allocations
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.2,
                    shape: rds::Resources {
                        gpus: 0.2,
                        cpus: 2.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
                Stage {
                    frequency: 0.8,
                    shape: rds::Resources {
                        gpus: 0.8,
                        cpus: 6.0,
                        is_spmd: false,
                    }
                    .to_shape(&cluster)
                    .unwrap(),
                },
            ],
        };

        // Create reusable workers - one on the fragmented GPU, one on fresh node
        let reusable_fragmented = test_worker("fragmented", "stage", "0", 2.0, &[(0, 0.2)]);
        let reusable_fresh = test_worker("fresh", "stage", "1", 2.0, &[(0, 0.2)]);

        let mut reuse_map = HashMap::new();
        reuse_map.insert("fragmented".to_string(), reusable_fragmented);
        reuse_map.insert("fresh".to_string(), reusable_fresh);

        let shape = rds::Resources {
            gpus: 0.2,
            cpus: 2.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            Some(&reuse_map),
            0.1, // Small reuse bonus
        );

        assert!(
            result.did_allocate,
            "Should allocate with reuse consideration"
        );

        if let Some(reused) = result.reused_worker {
            // Should prefer reusing the fragmented worker to avoid further fragmenting fresh GPUs
            assert_eq!(reused.id, "fragmented");
        }
    }

    #[test]
    fn test_deletion_with_cpu_constraints() {
        init();

        // Test deletion when both GPU and CPU constraints matter
        let mut cluster = uniform_cluster(1, 16.0, 2);

        let workers = vec![
            test_worker("cpu_heavy", "stage", "0", 8.0, &[(0, 0.2)]), // High CPU, low GPU
            test_worker("gpu_heavy", "stage", "0", 2.0, &[(1, 0.8)]), // Low CPU, high GPU
            test_worker("balanced", "stage", "0", 4.0, &[(0, 0.3)]),  // Medium both
        ];

        for worker in &workers {
            cluster.allocate(&worker.allocation).unwrap();
        }

        // Workload that only needs cpus
        let workload = Workload {
            stages: vec![Stage {
                frequency: 1.0,
                shape: rds::Resources {
                    gpus: 0.0,
                    cpus: 8.0, // High CPU requirement
                    is_spmd: false,
                }
                .to_shape(&cluster)
                .unwrap(),
            }],
        };

        let worker_map: HashMap<String, rds::Worker> =
            workers.into_iter().map(|w| (w.id.clone(), w)).collect();

        let to_delete = find_worker_to_delete_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &worker_map,
        );

        // Should delete the cpu_heavy worker
        assert_eq!(to_delete, "cpu_heavy");
    }

    #[test]
    fn test_fragmentation_with_nearly_full_gpus() {
        init();

        // Test allocation decisions when GPUs are nearly full
        let mut cluster = cluster_from_nodes(vec![
            node_with_usage(
                "node-0",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.99), // Only 0.01 available
                    gpu_with_usage(1, 0.95), // Only 0.05 available
                ],
            ),
            node_with_usage(
                "node-1",
                0.0,
                16.0,
                vec![
                    gpu_with_usage(0, 0.0), // 1.0 available
                    gpu_with_usage(1, 0.0), // 1.0 available
                ],
            ),
        ]);

        // Workload with very small allocations
        let workload = Workload {
            stages: vec![Stage {
                frequency: 1.0,
                shape: rds::Resources {
                    gpus: 0.01,
                    cpus: 1.0,
                    is_spmd: false,
                }
                .to_shape(&cluster)
                .unwrap(),
            }],
        };

        let tiny_shape = rds::Resources {
            gpus: 0.01,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap();

        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &tiny_shape,
            None,
            0.0,
        );

        assert!(result.did_allocate, "Should allocate tiny shape");
        let allocation = result.resources.unwrap();

        // Should allocate to one of the available nodes
        assert_eq!(allocation.node, "1");
    }
}
