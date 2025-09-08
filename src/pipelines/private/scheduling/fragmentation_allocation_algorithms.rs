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

use itertools::Itertools;

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
pub fn gpu_can_be_used_to_allocate(
    gpu: &rds::GpuResources,
    shape: &rds::WorkerShape,
    available_cpus: rds::FixedUtil,
) -> bool {
    // Not required to fully satisfy; only needs to contribute
    let needed_cpus = match shape {
        rds::WorkerShape::CpuOnly(s) => s.num_cpus,
        rds::WorkerShape::FractionalGpu(s) => s.num_cpus,
        rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus,
    };

    if needed_cpus > available_cpus {
        return false;
    }

    match shape {
        rds::WorkerShape::CpuOnly(_) => false,
        rds::WorkerShape::FractionalGpu(s) => {
            gpu.used_fraction + s.gpu_fraction <= rds::FixedUtil::ONE
        }
        rds::WorkerShape::WholeNumberedGpu(_) => gpu.is_fully_unallocated(),
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
    num_cpus: rds::FixedUtil,
) -> usize {
    node_resources
        .gpus
        .iter()
        .filter(|g| gpu_can_be_used_to_allocate(g, shape, num_cpus))
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
        rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus,
    };

    match shape {
        rds::WorkerShape::CpuOnly(_) => {
            node_resources.used_cpus + needed_cpus <= node_resources.total_cpus
        }
        rds::WorkerShape::FractionalGpu(_) => {
            number_of_gpus_which_can_be_used_for_shape(node_resources, shape, needed_cpus) > 0
        }
        rds::WorkerShape::WholeNumberedGpu(_) => num_fully_unallocated_gpus(node_resources) > 0,
    }
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
            for (gpu_index, gpu) in node_resources.gpus.iter().enumerate() {
                if !gpu_can_be_used_to_allocate(
                    gpu,
                    shape,
                    node_resources.total_cpus - node_resources.used_cpus,
                ) {
                    continue;
                }

                // Create allocation with fractional GPU compute
                out.push(rds::WorkerResources {
                    node: node_id.to_string(),
                    cpus: s.num_cpus,
                    gpus: vec![rds::GPUAllocation {
                        index: gpu_index,
                        used_fraction: s.gpu_fraction,
                    }],
                });
            }
            out
        }
        // Whole numbered GPU tasks: allocate complete GPUs (1.0 fraction each)
        rds::WorkerShape::WholeNumberedGpu(s) => {
            // Find all GPUs that can accommodate this shape (must be fully available)
            let available_gpus: Vec<usize> = node_resources
                .gpus
                .iter()
                .enumerate()
                .filter(|(_, g)| {
                    gpu_can_be_used_to_allocate(
                        g,
                        shape,
                        node_resources.total_cpus - node_resources.used_cpus,
                    )
                })
                .map(|(i, _)| i)
                .collect();
            let mut out = Vec::new();

            // Early return if not enough GPUs available
            if (s.num_gpus as usize) > available_gpus.len() {
                return out;
            }

            // Use itertools to generate all combinations of the required number of GPUs
            for chosen_gpus in available_gpus.iter().combinations(s.num_gpus as usize) {
                // Create GPU allocations for the chosen GPUs (1.0 fraction each)
                let gpus: Vec<rds::GPUAllocation> = chosen_gpus
                    .iter()
                    .map(|&&gpu_index| rds::GPUAllocation {
                        index: gpu_index,
                        used_fraction: rds::FixedUtil::ONE,
                    })
                    .collect();

                // If we successfully allocated all required resources, add this combination
                out.push(rds::WorkerResources {
                    node: node_id.to_string(),
                    cpus: s.num_cpus,
                    gpus,
                });
            }
            out
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
        .map(|g| g.used_fraction.to_num::<f32>())
        .sum();

    // Determine how many GPU resources this shape requires
    let shape_num_gpus: rds::FixedUtil = match shape {
        rds::WorkerShape::CpuOnly(_) => rds::FixedUtil::ZERO, // CPU-only tasks use no GPU
        rds::WorkerShape::FractionalGpu(s) => s.gpu_fraction,
        rds::WorkerShape::WholeNumberedGpu(s) => rds::FixedUtil::from_num(s.num_gpus),
    };

    // Case 1: Task requests no GPU compute resources
    // All available GPU resources are "fragmented" since they can't be used by this task type
    if shape_num_gpus == rds::FixedUtil::ZERO {
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
    for gpu in &node_resources.gpus {
        // If this GPU cannot contribute to the allocation, its resources are fragmented
        if !gpu_can_be_used_to_allocate(
            gpu,
            shape,
            node_resources.total_cpus - node_resources.used_cpus,
        ) {
            out += 1.0 - gpu.used_fraction.to_num::<f32>();
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

// --------------------
// Tests (pure Rust)
// --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::private::scheduling::resources as rds;
    use std::collections::HashMap;

    fn make_cluster(nodes: Vec<rds::NodeResources>) -> rds::ClusterResources {
        let mut map: HashMap<String, rds::NodeResources> = HashMap::new();
        for (i, n) in nodes.into_iter().enumerate() {
            map.insert(i.to_string(), n);
        }
        rds::ClusterResources::new(Some(map))
    }

    fn make_cluster_resources(
        num_nodes: usize,
        cpus_per_node: f32,
        gpus_per_node: usize,
    ) -> rds::ClusterResources {
        let mut map: HashMap<String, rds::NodeResources> = HashMap::new();
        for i in 0..num_nodes {
            let mut gpus = Vec::with_capacity(gpus_per_node);
            for i in 0..gpus_per_node {
                gpus.push(rds::GpuResources {
                    index: i as u8,
                    uuid_: uuid::Uuid::new_v4(),
                    used_fraction: rds::FixedUtil::ZERO,
                });
            }
            map.insert(
                i.to_string(),
                rds::NodeResources {
                    used_cpus: rds::FixedUtil::ZERO,
                    total_cpus: rds::FixedUtil::from_num(cpus_per_node),
                    gpus: gpus,
                    name: None,
                },
            );
        }
        rds::ClusterResources::new(Some(map))
    }

    fn make_gpu(index: u8, available_fraction: f32) -> rds::GpuResources {
        rds::GpuResources {
            index: index,
            uuid_: uuid::Uuid::new_v4(),
            used_fraction: rds::FixedUtil::from_num(1.0 - available_fraction),
        }
    }

    fn make_node(
        available_cpus: f32,
        total_cpus: f32,
        gpus: Vec<rds::GpuResources>,
    ) -> rds::NodeResources {
        rds::NodeResources {
            used_cpus: rds::FixedUtil::from_num(total_cpus - available_cpus),
            total_cpus: rds::FixedUtil::from_num(total_cpus),
            gpus: gpus,
            name: None,
        }
    }

    fn make_worker(
        id: &str,
        stage: &str,
        node: &str,
        cpus: f32,
        gpu_allocs: &[(usize, f32)],
    ) -> rds::Worker {
        let gpus: Vec<rds::GPUAllocation> = gpu_allocs
            .iter()
            .copied()
            .map(|(idx, frac)| rds::GPUAllocation::new(idx, frac))
            .collect();
        rds::Worker::new(
            id.to_string(),
            stage.to_string(),
            rds::WorkerResources {
                node: node.to_string(),
                cpus: rds::FixedUtil::from_num(cpus),
                gpus: gpus,
            },
        )
    }

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_calculate_unallocatable_gpus_fragment_for_shape() {
        // Test the core fragmentation calculation logic with a partially allocated node
        let resources = make_node(4.0, 8.0, vec![make_gpu(0, 0.5), make_gpu(1, 1.0)]);
        let shape = rds::Resources {
            gpus: 0.7,
            cpus: 2.0,
        }
        .to_shape()
        .unwrap();
        let result = calculate_unallocatable_gpus_fragment_for_shape_on_node(&resources, &shape);
        assert!(float_eq(result, 0.5), "Expected 0.5, got {result}");
    }

    #[test]
    fn test_estimate_node_fragmentation() {
        // Test workload-based fragmentation estimation with multiple task types
        let resources = make_node(4.0, 8.0, vec![make_gpu(0, 0.5), make_gpu(1, 1.0)]);
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.7,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.3,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let result = estimate_fragmentation_on_node(&resources, &workload);
        assert!(
            result >= 0.3 && result <= 0.5,
            "Expected 0.3..0.5, got {result}"
        );
    }

    #[test]
    fn test_calculate_cluster_fragmentation() {
        let resources = make_cluster(vec![
            make_node(4.0, 8.0, vec![make_gpu(0, 0.5), make_gpu(1, 1.0)]),
            make_node(8.0, 8.0, vec![make_gpu(0, 0.7), make_gpu(1, 0.3)]),
        ]);
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.7,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.3,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let result = estimate_fragmentation_on_cluster(&resources, &workload);
        assert!(result >= 0.2 && result <= 1.0, "cluster frag {result}");
    }

    #[test]
    fn test_maybe_allocate_worker_using_fragmentation_gradient_descent() {
        // Integration test for the main fragmentation gradient descent allocation algorithm
        let mut cluster_resources = make_cluster(vec![make_node(
            4.0,
            8.0,
            vec![make_gpu(0, 0.5), make_gpu(1, 1.0)],
        )]);

        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.7,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.3,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let shape = rds::Resources {
            gpus: 0.7,
            cpus: 2.0,
        }
        .to_shape()
        .unwrap();
        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster_resources,
            &workload,
            &shape,
            None,
            10.0,
        );
        let res = result.resources.expect("allocation");
        assert_eq!(res.node, "0");
        assert!(
            res.gpus
                .iter()
                .any(|g| g.used_fraction == rds::FixedUtil::from_num(0.7))
        );
    }

    #[test]
    fn test_maybe_allocate_worker_various_shapes() {
        // Comprehensive test of allocation algorithm with different worker shape types
        let mut cluster_resources = make_cluster_resources(2, 8.0, 2);
        let workers = vec![
            make_worker("worker1", "stage1", "0", 4.0, &[(0, 0.5)]),
            make_worker("worker2", "stage1", "1", 2.0, &[(0, 0.3), (1, 0.2)]),
        ];
        cluster_resources
            .allocate(&workers[0].allocation)
            .expect("allocate");
        cluster_resources
            .allocate(&workers[1].allocation)
            .expect("allocate");
        let original_cluster_resources = cluster_resources.clone();
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.7,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.3,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };

        let cases: Vec<(rds::WorkerShape, bool)> = vec![
            (
                rds::Resources {
                    gpus: 0.5,
                    cpus: 2.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
            (
                rds::Resources {
                    gpus: 0.8,
                    cpus: 4.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
            (
                rds::Resources {
                    gpus: 1.0,
                    cpus: 6.0,
                }
                .to_shape()
                .unwrap(),
                false,
            ),
            (
                rds::Resources {
                    gpus: 1.0,
                    cpus: 2.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
            (
                rds::Resources {
                    gpus: 2.0,
                    cpus: 4.0,
                }
                .to_shape()
                .unwrap(),
                false,
            ),
            (
                rds::Resources {
                    gpus: 1.0,
                    cpus: 2.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
            (
                rds::Resources {
                    gpus: 0.0,
                    cpus: 2.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
            (
                rds::Resources {
                    gpus: 0.0,
                    cpus: 8.0,
                }
                .to_shape()
                .unwrap(),
                false,
            ),
            (
                rds::Resources {
                    gpus: 0.0,
                    cpus: 1.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
            (
                rds::Resources {
                    gpus: 0.0,
                    cpus: 1.0,
                }
                .to_shape()
                .unwrap(),
                true,
            ),
        ];

        for (shape, expected_ok) in cases.into_iter() {
            println!("shape: {:?}", shape);
            println!(
                "cluster_resources:\n{}",
                cluster_resources.make_detailed_utilization_table()
            );
            let result = find_best_allocation_using_fragmentation_gradient_descent(
                &mut cluster_resources,
                &workload,
                &shape,
                None,
                10.0,
            );
            assert_eq!(
                cluster_resources, original_cluster_resources,
                "cluster_resources should not change"
            );
            if expected_ok {
                let res = result.resources.expect("expected allocation");
                assert!(res.node == "0" || res.node == "1");
                // CPUs equal to request
                let needed_cpus = match shape {
                    rds::WorkerShape::CpuOnly(s) => s.num_cpus,
                    rds::WorkerShape::FractionalGpu(s) => s.num_cpus,
                    rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus,
                };
                assert!(res.cpus == needed_cpus);
            } else {
                println!("shape: {:?}", shape);
                println!("result: {:?}", result);
                assert!(result.resources.is_none());
            }
        }
    }

    #[test]
    fn test_find_worker_to_delete_using_fragmentation_gradient_descent() {
        let mut cluster_resources = make_cluster(vec![make_node(
            8.0,
            8.0,
            vec![make_gpu(0, 1.0), make_gpu(1, 1.0)],
        )]);
        let workers = vec![
            make_worker("worker1", "stage1", "0", 2.0, &[(0, 0.5)]),
            make_worker("worker2", "stage1", "0", 2.0, &[(1, 0.7)]),
        ];
        cluster_resources.allocate(&workers[0].allocation).unwrap();
        cluster_resources.allocate(&workers[1].allocation).unwrap();
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.7,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.3,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let worker_map: std::collections::HashMap<String, rds::Worker> =
            workers.into_iter().map(|w| (w.id.clone(), w)).collect();
        let result_id = find_worker_to_delete_using_fragmentation_gradient_descent(
            &mut cluster_resources,
            &workload,
            &worker_map,
        );
        assert!(result_id == "worker1" || result_id == "worker2");
    }

    #[test]
    fn test_prefer_lower_allocated_nodes() {
        // Two nodes, 8 GPUs and 240 CPUs each
        let mut cluster = make_cluster(vec![
            make_node(240.0, 240.0, (0..8).map(|_| make_gpu(0, 1.0)).collect()),
            make_node(240.0, 240.0, (0..8).map(|_| make_gpu(0, 1.0)).collect()),
        ]);
        let workers = vec![
            // Node 0: highly allocated CPUs
            make_worker("worker1", "stage1", "0", 100.0, &[]),
            // Node 1: lightly allocated CPUs
            make_worker("worker2", "stage2", "1", 10.0, &[]),
        ];
        cluster.allocate(&workers[0].allocation).unwrap();
        cluster.allocate(&workers[1].allocation).unwrap();

        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.5,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.5,
                    shape: rds::Resources {
                        gpus: 0.0,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let shape = rds::Resources {
            gpus: 0.0,
            cpus: 1.0,
        }
        .to_shape()
        .unwrap();
        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster,
            &workload,
            &shape,
            None,
            10.0,
        );
        let res = result.resources.expect("allocation");
        assert_eq!(res.node, "1");
        assert!(res.cpus == rds::FixedUtil::ONE);
    }

    #[test]
    fn test_gpu_resource_helpers_can_be_used_to_allocate() {
        let cases = vec![
            (
                make_gpu(0, 1.0),
                rds::Resources {
                    gpus: 1.0,
                    cpus: 4.0,
                },
                16.0,
                true,
            ),
            (
                make_gpu(0, 0.5),
                rds::Resources {
                    gpus: 1.0,
                    cpus: 4.0,
                },
                16.0,
                false,
            ),
            (
                make_gpu(0, 1.0),
                rds::Resources {
                    gpus: 0.5,
                    cpus: 4.0,
                },
                16.0,
                true,
            ),
            (
                make_gpu(0, 0.4),
                rds::Resources {
                    gpus: 0.5,
                    cpus: 4.0,
                },
                16.0,
                false,
            ),
        ];

        for (gpu_res, r, available_cpus, expected) in cases {
            let ok = gpu_can_be_used_to_allocate(
                &gpu_res,
                &r.to_shape().unwrap(),
                rds::FixedUtil::from_num(available_cpus),
            );
            assert_eq!(ok, expected, "case failed");
        }
    }

    #[test]
    fn test_node_resource_helpers_find_possible_allocations() {
        let cluster = make_cluster_resources(1, 16.0, 2);
        let node = cluster.nodes.get("0").unwrap();

        // CPU-only
        let shape_cpu = rds::Resources {
            gpus: 0.0,
            cpus: 4.0,
        }
        .to_shape()
        .unwrap();
        let allocs = find_possible_allocations_on_node(node, &shape_cpu, "0");
        assert_eq!(allocs.len(), 1);

        // Fractional GPU: one per GPU
        let shape_frac = rds::Resources {
            gpus: 0.5,
            cpus: 4.0,
        }
        .to_shape()
        .unwrap();
        let allocs_frac = find_possible_allocations_on_node(node, &shape_frac, "0");
        assert_eq!(allocs_frac.len(), 2);
    }

    #[test]
    fn test_mixed_resource_scenarios() {
        // Case 1
        let mut cluster1 = make_cluster_resources(1, 16.0, 3);
        let workers1 = vec![
            make_worker("worker1", "stage1", "0", 4.0, &[(0, 1.0)]),
            make_worker("worker2", "stage1", "0", 4.0, &[(1, 0.5)]),
        ];
        cluster1.allocate(&workers1[0].allocation).unwrap();
        cluster1.allocate(&workers1[1].allocation).unwrap();
        let workload1 = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::Resources {
                        gpus: 0.7,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::Resources {
                        gpus: 0.3,
                        cpus: 1.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let shape1 = rds::Resources {
            gpus: 0.5,
            cpus: 2.0,
        }
        .to_shape()
        .unwrap();
        let result1 = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster1,
            &workload1,
            &shape1,
            None,
            10.0,
        )
        .resources;
        assert!(result1.is_some());

        // Case 2
        let mut cluster2 = make_cluster_resources(1, 8.0, 2);
        let workers2 = vec![make_worker(
            "worker1",
            "stage1",
            "0",
            6.0,
            &[(0, 0.8), (1, 0.8)],
        )];
        cluster2.allocate(&workers2[0].allocation).unwrap();
        let workload2 = Workload {
            stages: vec![
                Stage {
                    frequency: 0.5,
                    shape: rds::Resources {
                        gpus: 0.5,
                        cpus: 2.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
                Stage {
                    frequency: 0.5,
                    shape: rds::Resources {
                        gpus: 1.0,
                        cpus: 4.0,
                    }
                    .to_shape()
                    .unwrap(),
                },
            ],
        };
        let shape2 = rds::Resources {
            gpus: 1.0,
            cpus: 3.0,
        }
        .to_shape()
        .unwrap();
        let result2 = find_best_allocation_using_fragmentation_gradient_descent(
            &mut cluster2,
            &workload2,
            &shape2,
            None,
            10.0,
        )
        .resources;
        assert!(result2.is_none());
    }
}
