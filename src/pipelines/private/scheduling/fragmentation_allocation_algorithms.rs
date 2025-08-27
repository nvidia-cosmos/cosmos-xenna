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

use std::collections::HashMap;

use itertools::Itertools;

use super::{allocator::WorkerAllocator, resources as rds};

// --------------------
// Float helpers
// --------------------

/// Checks if two floating point numbers are approximately equal.
///
/// Uses a small epsilon (1e-6) to handle floating point precision issues
/// that can occur during resource calculations.
fn float_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6
}

/// Checks if a is strictly less than b with floating point tolerance.
///
/// Accounts for floating point precision by using epsilon comparison.
fn float_lt(a: f32, b: f32) -> bool {
    a < b - 1e-6
}

/// Checks if a is less than or equal to b with floating point tolerance.
///
/// Accounts for floating point precision by using epsilon comparison.
fn float_lte(a: f32, b: f32) -> bool {
    a <= b + 1e-6
}

/// Checks if a is greater than or equal to b with floating point tolerance.
///
/// Accounts for floating point precision by using epsilon comparison.
fn float_gte(a: f32, b: f32) -> bool {
    a + 1e-6 >= b
}

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

// --------------------
// GPU helpers
// --------------------

/// Helper class for managing GPU-specific resource calculations and checks.
///
/// This class encapsulates logic for determining if GPUs have sufficient resources
/// for different types of workloads. It handles the complexity of different GPU
/// allocation types (fractional, whole, codecs, etc.) as described in Section 2.1
/// of the paper.
///
/// # Attributes
/// * `available` - Current available GPU resources
/// * `totals` - Total GPU resources on the node
#[derive(Debug, Clone)]
pub struct GpuResourceHelpers {
    available: rds::GpuResources,
    totals: rds::GpuResources,
}

impl GpuResourceHelpers {
    pub fn new(available: rds::GpuResources, totals: rds::GpuResources) -> Self {
        Self { available, totals }
    }

    /// Checks if GPU is completely free with all resources available.
    ///
    /// A GPU is considered fully unallocated if it has:
    /// - 100% of compute capacity available (gpu_fraction_available == 1.0)
    /// - All hardware video decoders (nvdecs) available
    /// - All hardware video encoders (nvencs) available
    ///
    /// # Returns
    /// True if GPU is completely unused, False otherwise
    pub fn is_fully_unallocated(&self) -> bool {
        float_eq(self.available.gpu_fraction, 1.0)
            && (self.available.num_nvdecs() == self.totals.num_nvdecs())
            && (self.available.num_nvencs() == self.totals.num_nvencs())
    }

    /// Determines if this GPU can accommodate the given worker shape requirements.
    ///
    /// It doesn't have to be able to fully allocate the shape, but it does need to be able to contribute to the
    /// allocation. So, if the shape requires 2 gpus and this is a fully unallocated gpu, this will return True.
    ///
    /// This method implements the allocation feasibility check described in Section 2.1
    /// of the paper. It handles different GPU allocation types:
    /// - CPU-only workloads
    /// - Codec (video encoder/decoder) workloads
    /// - Fractional GPU workloads
    /// - Whole-numbered GPU workloads
    /// - Entire GPU workloads
    ///
    /// # Arguments
    /// * `shape` - WorkerShape describing resource requirements
    /// * `available_cpus` - Number of CPU cores available on the node
    ///
    /// # Returns
    /// True if the GPU can accommodate this shape, False otherwise
    pub fn can_be_used_to_allocate(&self, shape: &rds::WorkerShape, available_cpus: f32) -> bool {
        // Not required to fully satisfy; only needs to contribute
        let needed_cpus = match shape {
            rds::WorkerShape::CpuOnly(s) => s.num_cpus,
            rds::WorkerShape::Codec(s) => s.num_cpus,
            rds::WorkerShape::FractionalGpu(s) => s.num_cpus,
            rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus,
            rds::WorkerShape::EntireGpu(s) => s.num_cpus,
        };
        if float_lt(available_cpus, needed_cpus) {
            return false;
        }

        match shape {
            rds::WorkerShape::CpuOnly(_) => false,
            rds::WorkerShape::Codec(s) => {
                (self.available.num_nvdecs() as u32 >= s.num_nvdecs as u32)
                    && (self.available.num_nvencs() as u32 >= s.num_nvencs as u32)
            }
            rds::WorkerShape::FractionalGpu(s) => {
                float_gte(self.available.gpu_fraction, s.num_gpus)
                    && (self.available.num_nvdecs() as u32 >= s.num_nvdecs as u32)
                    && (self.available.num_nvencs() as u32 >= s.num_nvencs as u32)
            }
            rds::WorkerShape::WholeNumberedGpu(s) => {
                float_eq(self.available.gpu_fraction, 1.0)
                    && (self.available.num_nvdecs() as u32 >= s.num_nvdecs_per_gpu as u32)
                    && (self.available.num_nvencs() as u32 >= s.num_nvencs_per_gpu as u32)
            }
            rds::WorkerShape::EntireGpu(_) => self.is_fully_unallocated(),
        }
    }
}

// --------------------
// Node helpers
// --------------------

/// Helper class for node-level resource management and fragmentation calculation.
///
/// This class implements the core fragmentation analysis described in Section 3 of the paper.
/// It helps determine how fragmented a node's resources are from the perspective of different
/// workload types.
///
/// # Attributes
/// * `available` - Current available resources on the node
/// * `totals` - Total resources on the node
/// * `gpus` - List of GpuResourceHelpers for each GPU on the node
#[derive(Debug, Clone)]
pub struct NodeResourceHelpers {
    pub available: rds::NodeResources,
    pub totals: rds::NodeResources,
    pub gpus: Vec<GpuResourceHelpers>,
}

impl NodeResourceHelpers {
    pub fn new(available: rds::NodeResources, totals: rds::NodeResources) -> Self {
        let gpus = available
            .gpus
            .iter()
            .cloned()
            .zip(totals.gpus.iter().cloned())
            .map(|(a, t)| GpuResourceHelpers::new(a, t))
            .collect();
        Self {
            available,
            totals,
            gpus,
        }
    }

    /// Counts number of GPUs on node that can accommodate given shape.
    ///
    /// # Arguments
    /// * `shape` - WorkerShape describing resource requirements
    ///
    /// # Returns
    /// Number of GPUs that can be used for this shape
    fn number_of_gpus_which_can_be_used_for_shape(&self, shape: &rds::WorkerShape) -> usize {
        self.gpus
            .iter()
            .filter(|g| g.can_be_used_to_allocate(shape, self.available.cpus))
            .count()
    }
}

impl NodeResourceHelpers {
    /// Finds all possible ways to allocate video encoders/decoders across GPUs.
    ///
    /// Uses backtracking to find all valid combinations of codec allocations.
    /// This implements the codec allocation strategy described in the paper where
    /// we need to find all valid ways to distribute codec resources across GPUs.
    ///
    /// # Arguments
    /// * `num_codecs` - Number of encoders/decoders needed
    /// * `codec_is_nvdec` - True for nvdecs, false for nvencs
    ///
    /// # Returns
    /// List of possible codec allocation combinations
    fn find_codec_allocations_internal(
        &self,
        num_codecs: usize,
        codec_is_nvdec: bool,
    ) -> Vec<Vec<rds::CodecAllocation>> {
        // Match Python semantics: choose counts per GPU and always take the lowest-index
        // available codecs on that GPU for the chosen count.
        let mut out: Vec<Vec<rds::CodecAllocation>> = Vec::new();
        let mut current: Vec<rds::CodecAllocation> = Vec::new();
        // Working copy of available resources to track allocations during backtracking
        let mut working = self.available.clone();

        /// Helper function to greedily take the first k available codecs from a specific GPU.
        ///
        /// This function attempts to allocate k codec units from the specified GPU,
        /// taking them in order from lowest index to highest. It validates each
        /// allocation by checking if it can be successfully allocated on the working
        /// resource set.
        ///
        /// # Arguments
        /// * `working` - Mutable reference to current resource state for validation
        /// * `totals` - Total resources to determine codec limits
        /// * `gpu_index` - Index of the GPU to allocate codecs from  
        /// * `k` - Number of codecs to allocate
        /// * `codec_is_nvdec` - True for NVDEC allocation, false for NVENC
        ///
        /// # Returns
        /// Some(Vec<CodecAllocation>) if k codecs could be allocated, None otherwise
        fn greedy_take_first_k_on_gpu(
            working: &mut rds::NodeResources,
            totals: &rds::NodeResources,
            gpu_index: usize,
            k: usize,
            codec_is_nvdec: bool,
        ) -> Option<Vec<rds::CodecAllocation>> {
            // Determine the total number of this codec type available on the GPU
            let limit = if codec_is_nvdec {
                totals.gpus[gpu_index].num_nvdecs()
            } else {
                totals.gpus[gpu_index].num_nvencs()
            };

            // Early return if request is impossible (0 codecs or more than available)
            if k == 0 || k > limit {
                return None;
            }

            let mut picked: Vec<rds::CodecAllocation> = Vec::new();

            // Try to allocate codecs in order from index 0 upward
            for codec_index in 0..limit {
                if picked.len() >= k {
                    break;
                }

                // Create a test allocation to verify this codec is available
                let attempt = if codec_is_nvdec {
                    rds::WorkerResources::new(
                        "".to_string(),
                        0.0,
                        None,
                        Some(vec![rds::CodecAllocation::new(gpu_index, codec_index)]),
                        None,
                    )
                } else {
                    rds::WorkerResources::new(
                        "".to_string(),
                        0.0,
                        None,
                        None,
                        Some(vec![rds::CodecAllocation::new(gpu_index, codec_index)]),
                    )
                };

                // If the allocation succeeds, add this codec to our selection
                if working.allocate(&attempt).is_ok() {
                    picked.push(rds::CodecAllocation::new(gpu_index, codec_index));
                }
            }

            // Return the allocations only if we got exactly k codecs
            if picked.len() == k {
                Some(picked)
            } else {
                None
            }
        }

        /// Recursive backtracking function to find all valid codec allocation combinations.
        ///
        /// This implements a backtracking algorithm to explore all possible ways to
        /// distribute the required number of codecs across available GPUs. It tries
        /// different allocation counts per GPU and recursively explores remaining
        /// GPUs until all codecs are allocated.
        ///
        /// # Arguments
        /// * `working` - Mutable reference to current resource state for allocation tracking
        /// * `totals` - Total resources to determine limits
        /// * `codec_is_nvdec` - True for NVDEC allocation, false for NVENC
        /// * `start_gpu` - GPU index to start searching from (for ordered allocation)
        /// * `remaining` - Number of codecs still needing allocation
        /// * `current` - Current partial allocation being built
        /// * `out` - Output vector to store complete valid allocations
        fn backtrack(
            working: &mut rds::NodeResources,
            totals: &rds::NodeResources,
            codec_is_nvdec: bool,
            start_gpu: usize,
            remaining: usize,
            current: &mut Vec<rds::CodecAllocation>,
            out: &mut Vec<Vec<rds::CodecAllocation>>,
        ) {
            // Base case: all codecs allocated, save this combination
            if remaining == 0 {
                out.push(current.clone());
                return;
            }

            // Try allocating on each remaining GPU
            for gpu_index in start_gpu..working.gpus.len() {
                // Check how many of this codec type are available on this GPU
                let avail_count = if codec_is_nvdec {
                    working.gpus[gpu_index].num_nvdecs()
                } else {
                    working.gpus[gpu_index].num_nvencs()
                };

                // Skip this GPU if no codecs are available
                if avail_count == 0 {
                    continue;
                }

                // Try taking different numbers of codecs from this GPU
                let max_take = remaining.min(avail_count);
                for take in 1..=max_take {
                    // Attempt to allocate 'take' codecs from this GPU
                    if let Some(taken) =
                        greedy_take_first_k_on_gpu(working, totals, gpu_index, take, codec_is_nvdec)
                    {
                        // Add these allocations to our current combination
                        current.extend(taken.iter().cloned());

                        // Recursively try to allocate remaining codecs on subsequent GPUs
                        backtrack(
                            working,
                            totals,
                            codec_is_nvdec,
                            gpu_index + 1, // Start from next GPU to maintain order
                            remaining - take,
                            current,
                            out,
                        );

                        // Backtrack: release the allocations we just made
                        let to_release = if codec_is_nvdec {
                            rds::WorkerResources::new(
                                "".to_string(),
                                0.0,
                                None,
                                Some(taken.clone()),
                                None,
                            )
                        } else {
                            rds::WorkerResources::new(
                                "".to_string(),
                                0.0,
                                None,
                                None,
                                Some(taken.clone()),
                            )
                        };
                        working.release_allocation(&to_release);

                        // Remove the allocations from our current combination
                        for _ in 0..take {
                            current.pop();
                        }
                    }
                }
            }
        }

        backtrack(
            &mut working,
            &self.totals,
            codec_is_nvdec,
            0,
            num_codecs,
            &mut current,
            &mut out,
        );

        out
    }

    fn num_fully_unallocated_gpus(&self) -> usize {
        self.gpus
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
    pub fn can_allocate(&self, shape: &rds::WorkerShape) -> bool {
        // CPU check
        let needed_cpus = match shape {
            rds::WorkerShape::CpuOnly(s) => s.num_cpus,
            rds::WorkerShape::Codec(s) => s.num_cpus,
            rds::WorkerShape::FractionalGpu(s) => s.num_cpus,
            rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus,
            rds::WorkerShape::EntireGpu(s) => s.num_cpus,
        };
        if float_lt(self.available.cpus, needed_cpus) {
            return false;
        }

        let total_available = self.available.totals();
        match shape {
            rds::WorkerShape::CpuOnly(_) => true,
            rds::WorkerShape::Codec(s) => {
                (s.num_nvdecs as f32) <= total_available.nvdecs
                    && (s.num_nvencs as f32) <= total_available.nvencs
            }
            rds::WorkerShape::FractionalGpu(_) => {
                self.number_of_gpus_which_can_be_used_for_shape(shape) > 0
            }
            rds::WorkerShape::WholeNumberedGpu(s) => float_lte(
                s.num_gpus as f32,
                self.number_of_gpus_which_can_be_used_for_shape(shape) as f32,
            ),
            rds::WorkerShape::EntireGpu(s) => {
                float_lte(s.num_gpus as f32, self.num_fully_unallocated_gpus() as f32)
            }
        }
    }

    /// Determines if these node resources can accommodate the requested allocation.
    ///
    /// This method is similar to the NodeResources.allocate() method but only checks feasibility
    /// without modifying state. It uses the floating point comparison functions for numerical stability.
    ///
    /// # Arguments
    /// * `resources` - WorkerResources object describing the requested allocation
    ///
    /// # Returns
    /// True if allocation is possible, False otherwise
    pub fn can_allocate_resources(&self, resources: &rds::WorkerResources) -> bool {
        self.available.copy_and_allocate(resources).is_ok()
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
    pub fn find_possible_allocations(
        &self,
        shape: &rds::WorkerShape,
        node_id: &str,
    ) -> Vec<rds::WorkerResources> {
        if !self.can_allocate(shape) {
            return Vec::new();
        }

        match shape {
            // CPU-only tasks: simple allocation of just CPU cores
            rds::WorkerShape::CpuOnly(s) => {
                vec![rds::WorkerResources::new(
                    node_id.to_string(),
                    s.num_cpus,
                    None,
                    None,
                    None,
                )]
            }
            // Codec-only tasks: allocate video encoders/decoders without GPU compute
            rds::WorkerShape::Codec(s) => {
                // Find all possible ways to allocate the required NVDEC units
                let nvdec_allocs =
                    self.find_codec_allocations_internal(s.num_nvdecs as usize, true);
                // Find all possible ways to allocate the required NVENC units
                let nvenc_allocs =
                    self.find_codec_allocations_internal(s.num_nvencs as usize, false);

                let mut out = Vec::new();
                // Generate all combinations of NVDEC and NVENC allocations
                for decs in &nvdec_allocs {
                    for encs in &nvenc_allocs {
                        out.push(rds::WorkerResources::new(
                            node_id.to_string(),
                            s.num_cpus,
                            None, // No GPU compute allocation
                            Some(decs.clone()),
                            Some(encs.clone()),
                        ));
                    }
                }
                out
            }
            // Fractional GPU tasks: allocate partial GPU compute plus optional codecs
            rds::WorkerShape::FractionalGpu(s) => {
                let mut out = Vec::new();
                // Try allocating on each GPU that has sufficient capacity
                for (gpu_index, gpu) in self.gpus.iter().enumerate() {
                    if !gpu.can_be_used_to_allocate(shape, self.available.cpus) {
                        continue;
                    }

                    // Allocate required NVDEC units from this GPU (if any needed)
                    let mut nvdecs: Vec<rds::CodecAllocation> = Vec::new();
                    if s.num_nvdecs > 0 {
                        let mut temp = self.available.clone();
                        // Try to allocate codecs greedily from lowest index upward
                        for codec_index in 0..self.totals.gpus[gpu_index].num_nvdecs() {
                            if nvdecs.len() >= s.num_nvdecs as usize {
                                break;
                            }
                            // Test if this codec is available
                            let attempt = rds::WorkerResources::new(
                                "".to_string(),
                                0.0,
                                None,
                                Some(vec![rds::CodecAllocation::new(gpu_index, codec_index)]),
                                None,
                            );
                            if temp.allocate(&attempt).is_ok() {
                                nvdecs.push(rds::CodecAllocation::new(gpu_index, codec_index));
                            }
                        }
                        // Skip this GPU if we couldn't get enough NVDECs
                        if nvdecs.len() < s.num_nvdecs as usize {
                            continue;
                        }
                    }

                    // Allocate required NVENC units from this GPU (if any needed)
                    let mut nvencs: Vec<rds::CodecAllocation> = Vec::new();
                    if s.num_nvencs > 0 {
                        let mut temp = self.available.clone();
                        // Try to allocate codecs greedily from lowest index upward
                        for codec_index in 0..self.totals.gpus[gpu_index].num_nvencs() {
                            if nvencs.len() >= s.num_nvencs as usize {
                                break;
                            }
                            // Test if this codec is available
                            let attempt = rds::WorkerResources::new(
                                "".to_string(),
                                0.0,
                                None,
                                None,
                                Some(vec![rds::CodecAllocation::new(gpu_index, codec_index)]),
                            );
                            if temp.allocate(&attempt).is_ok() {
                                nvencs.push(rds::CodecAllocation::new(gpu_index, codec_index));
                            }
                        }
                        // Skip this GPU if we couldn't get enough NVENCs
                        if nvencs.len() < s.num_nvencs as usize {
                            continue;
                        }
                    }

                    // Create allocation with fractional GPU compute + codecs
                    out.push(rds::WorkerResources::new(
                        node_id.to_string(),
                        s.num_cpus,
                        Some(vec![rds::GPUAllocation::new(gpu_index, s.num_gpus)]),
                        Some(nvdecs),
                        Some(nvencs),
                    ));
                }
                out
            }
            // Whole numbered GPU tasks: allocate complete GPUs (1.0 fraction each)
            rds::WorkerShape::WholeNumberedGpu(s) => {
                // Find all GPUs that can accommodate this shape (must be fully available)
                let available_gpus: Vec<usize> = self
                    .gpus
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| g.can_be_used_to_allocate(shape, self.available.cpus))
                    .map(|(i, _)| i)
                    .collect();
                let mut out = Vec::new();

                // Early return if not enough GPUs available
                if (s.num_gpus as usize) > available_gpus.len() {
                    return out;
                }

                // Use itertools to generate all combinations of the required number of GPUs
                // This is much cleaner than manual lexicographic ordering!
                for chosen_gpus in available_gpus.iter().combinations(s.num_gpus as usize) {
                    // Create GPU allocations for the chosen GPUs (1.0 fraction each)
                    let gpus: Vec<rds::GPUAllocation> = chosen_gpus
                        .iter()
                        .map(|&&gpu_index| rds::GPUAllocation::new(gpu_index, 1.0))
                        .collect();

                    // Try to allocate required codecs for each chosen GPU
                    let mut nvdecs: Vec<rds::CodecAllocation> = Vec::new();
                    let mut nvencs: Vec<rds::CodecAllocation> = Vec::new();
                    let mut allocation_valid = true; // Track if we can satisfy codec requirements

                    for &&gpu_index in chosen_gpus.iter() {
                        // Allocate required NVDEC units per GPU
                        if s.num_nvdecs_per_gpu > 0 {
                            let available_nvdecs = self.available.gpus[gpu_index].num_nvdecs();
                            if available_nvdecs < s.num_nvdecs_per_gpu as usize {
                                allocation_valid = false;
                                break;
                            }
                            // Take the first N available NVDECs from this GPU
                            for codec_index in 0..s.num_nvdecs_per_gpu as usize {
                                nvdecs.push(rds::CodecAllocation::new(gpu_index, codec_index));
                            }
                        }

                        // Allocate required NVENC units per GPU
                        if s.num_nvencs_per_gpu > 0 {
                            let available_nvencs = self.available.gpus[gpu_index].num_nvencs();
                            if available_nvencs < s.num_nvencs_per_gpu as usize {
                                allocation_valid = false;
                                break;
                            }
                            // Take the first N available NVENCs from this GPU
                            for codec_index in 0..s.num_nvencs_per_gpu as usize {
                                nvencs.push(rds::CodecAllocation::new(gpu_index, codec_index));
                            }
                        }
                    }

                    // If we successfully allocated all required resources, add this combination
                    if allocation_valid {
                        out.push(rds::WorkerResources::new(
                            node_id.to_string(),
                            s.num_cpus,
                            Some(gpus),
                            Some(nvdecs),
                            Some(nvencs),
                        ));
                    }
                }
                out
            }
            // Entire GPU tasks: allocate completely unallocated GPUs with all their resources
            rds::WorkerShape::EntireGpu(s) => {
                // Find all GPUs that are completely unallocated (no partial usage)
                let fully: Vec<usize> = self
                    .gpus
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| g.is_fully_unallocated())
                    .map(|(i, _)| i)
                    .collect();
                let mut out = Vec::new();
                let k = s.num_gpus as usize;

                // Early return if impossible (need 0 GPUs or more than available)
                if k == 0 || k > fully.len() {
                    return out;
                }

                // Use itertools to generate all combinations of fully unallocated GPUs
                // Much simpler than manual combination generation!
                for chosen_gpus in fully.iter().combinations(k) {
                    // Allocate entire GPUs (1.0 fraction each)
                    let gpus: Vec<rds::GPUAllocation> = chosen_gpus
                        .iter()
                        .map(|&&gpu_index| rds::GPUAllocation::new(gpu_index, 1.0))
                        .collect();

                    // For entire GPU allocation, we get ALL codecs on each GPU
                    let mut nvdecs: Vec<rds::CodecAllocation> = Vec::new();
                    let mut nvencs: Vec<rds::CodecAllocation> = Vec::new();
                    for &&gpu_index in chosen_gpus.iter() {
                        // Allocate all NVDECs on this GPU
                        for i in 0..self.totals.gpus[gpu_index].num_nvdecs() {
                            nvdecs.push(rds::CodecAllocation::new(gpu_index, i));
                        }
                        // Allocate all NVENCs on this GPU
                        for i in 0..self.totals.gpus[gpu_index].num_nvencs() {
                            nvencs.push(rds::CodecAllocation::new(gpu_index, i));
                        }
                    }

                    out.push(rds::WorkerResources::new(
                        node_id.to_string(),
                        s.num_cpus,
                        Some(gpus),
                        Some(nvdecs),
                        Some(nvencs),
                    ));
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
    fn calculate_unallocatable_gpus_fragment_for_shape(&self, shape: &rds::WorkerShape) -> f32 {
        // Calculate total GPU compute resources available on this node
        let total_available_gpus: f32 = self.available.gpus.iter().map(|g| g.gpu_fraction).sum();

        // Determine how many GPU resources this shape requires
        let shape_num_gpus: f32 = match shape {
            rds::WorkerShape::CpuOnly(_) => 0.0, // CPU-only tasks use no GPU
            rds::WorkerShape::Codec(_) => 0.0,   // Codec tasks use no GPU compute
            rds::WorkerShape::FractionalGpu(s) => s.num_gpus,
            rds::WorkerShape::WholeNumberedGpu(s) => s.num_gpus as f32,
            rds::WorkerShape::EntireGpu(s) => s.num_gpus as f32,
        };

        // Case 1: Task requests no GPU compute resources
        // All available GPU resources are "fragmented" since they can't be used by this task type
        if float_eq(shape_num_gpus, 0.0) {
            return total_available_gpus;
        }

        // Case 2: Shape cannot be allocated to the node at all
        // All available GPU resources are fragmented since the task can't be satisfied
        if !self.can_allocate(shape) {
            return total_available_gpus;
        }

        // Case 3: Shape can be allocated, but some GPUs may be unusable
        // Count GPU resources that cannot contribute to allocating this shape
        let mut out = 0.0;
        for gpu in &self.gpus {
            // If this GPU cannot contribute to the allocation, its resources are fragmented
            if !gpu.can_be_used_to_allocate(shape, self.available.cpus) {
                out += gpu.available.gpu_fraction;
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
    pub fn estimate_fragmentation(&self, workload: &Workload) -> f32 {
        let mut out = 0.0;
        // Calculate weighted fragmentation across all workload stages
        for stage in &workload.stages {
            // Get fragmentation for this specific task type
            let unallocatable_gpus =
                self.calculate_unallocatable_gpus_fragment_for_shape(&stage.shape);
            // Weight by how frequently this task type occurs
            out += stage.frequency * unallocatable_gpus;
        }
        out
    }
}

// --------------------
// Cluster helpers
// --------------------

#[derive(Debug, Clone)]
pub struct ClusterResourceHelpers {
    pub nodes: HashMap<String, NodeResourceHelpers>,
}

impl ClusterResourceHelpers {
    pub fn make_from_allocator(allocator: &WorkerAllocator) -> Self {
        Self::new(
            allocator.available_resources().clone(),
            allocator.totals().clone(),
        )
    }

    pub fn new(available: rds::ClusterResources, totals: rds::ClusterResources) -> Self {
        let mut nodes: HashMap<String, NodeResourceHelpers> = HashMap::new();
        for (name, avail_node) in &available.nodes {
            if let Some(total_node) = totals.nodes.get(name) {
                nodes.insert(
                    name.clone(),
                    NodeResourceHelpers::new(avail_node.clone(), total_node.clone()),
                );
            }
        }
        Self { nodes }
    }

    pub fn copy_and_allocate(
        &self,
        resources: &rds::WorkerResources,
    ) -> Result<Self, rds::AllocationError> {
        let mut new = self.clone();
        if let Some(node) = new.nodes.get_mut(&resources.node) {
            node.available.allocate(resources)?;
        }
        Ok(new)
    }

    pub fn copy_and_release_allocation(
        &self,
        resources: &rds::WorkerResources,
    ) -> Result<Self, rds::AllocationError> {
        let mut new = self.clone();
        if let Some(node) = new.nodes.get_mut(&resources.node) {
            node.available.release_allocation(resources);
            Ok(new)
        } else {
            Err(rds::AllocationError::NodeNotFound(resources.node.clone()))
        }
    }

    pub fn estimate_fragmentation(&self, workload: &Workload) -> f32 {
        self.nodes
            .values()
            .map(|n| n.estimate_fragmentation(workload))
            .sum()
    }
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
    cluster: &ClusterResourceHelpers,
    workload: &Workload,
    shape: &rds::WorkerShape,
    reusable_workers: Option<&std::collections::HashSet<rds::Worker>>,
    worker_reuse_fragmentation_equivalent: f32,
) -> AllocationResult {
    // Store all possible allocation options with their fragmentation impact
    let mut results: Vec<FragmentationResult> = Vec::new();

    // First, try reusing recently removed workers to avoid allocation thrashing
    // This helps prevent oscillation between creating and destroying workers
    if let Some(reuse_set) = reusable_workers {
        for worker in reuse_set {
            if let Some(node) = cluster.nodes.get(&worker.allocation.node) {
                // Check if this worker's allocation is still feasible
                if !node.can_allocate_resources(&worker.allocation) {
                    continue;
                }

                // Calculate fragmentation impact of reusing this worker
                let current_frag = node.estimate_fragmentation(workload);
                if let Ok(new_remaining_node_resources) =
                    node.available.copy_and_allocate(&worker.allocation)
                {
                    // Create new node state after this allocation
                    let new_node = NodeResourceHelpers::new(
                        new_remaining_node_resources.clone(),
                        node.totals.clone(),
                    );
                    let new_frag = new_node.estimate_fragmentation(workload);

                    // Record this reuse option for comparison
                    results.push(FragmentationResult {
                        fragmentation_before: current_frag,
                        fragmentation_after: new_frag,
                        node_remaining_resources: new_remaining_node_resources.totals().total_num(),
                        worker_allocation: worker.allocation.clone(),
                        maybe_reused_worker: Some(worker.clone()),
                    });
                }
            }
        }
    }

    // Now explore all possible fresh allocations across the cluster
    for (node_id, node) in &cluster.nodes {
        // Skip nodes that cannot accommodate this shape
        if !node.can_allocate(shape) {
            continue;
        }

        // Calculate current fragmentation level for this node
        let current_frag = node.estimate_fragmentation(workload);

        // Find all possible ways to allocate this shape on this node
        let possible_allocations = node.find_possible_allocations(shape, node_id);

        // Evaluate the fragmentation impact of each possible allocation
        for allocation in possible_allocations {
            if let Ok(new_remaining_node_resources) = node.available.copy_and_allocate(&allocation)
            {
                // Create node state after this allocation
                let new_node = NodeResourceHelpers::new(
                    new_remaining_node_resources.clone(),
                    node.totals.clone(),
                );

                // Calculate new fragmentation level
                let new_frag = new_node.estimate_fragmentation(workload);

                // Record this allocation option for comparison
                results.push(FragmentationResult {
                    fragmentation_before: current_frag,
                    fragmentation_after: new_frag,
                    node_remaining_resources: new_remaining_node_resources.totals().total_num(),
                    worker_allocation: allocation,
                    maybe_reused_worker: None, // This is a fresh allocation
                });
            }
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

#[derive(Debug, Clone)]
struct FragmentationDeleteResult {
    // Cost to minimize: delta in fragmentation from the perspective of the node
    // hosting the worker (after - before). Lower is better.
    frag_delta: f32,
    // Tie-breaker: remaining resources on the node AFTER releasing the worker.
    // Lower is better (preserve more cluster-wide headroom).
    node_remaining_resources_after: f32,
    worker: rds::Worker,
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
    cluster: &ClusterResourceHelpers,
    workload: &Workload,
    potential_workers: &[rds::Worker],
) -> rds::Worker {
    assert!(!potential_workers.is_empty());
    let mut changes: Vec<FragmentationDeleteResult> = Vec::new();

    // Evaluate the fragmentation impact of removing each candidate worker
    for worker in potential_workers {
        // Compute node-local fragmentation delta instead of cluster-wide.
        if let Some(node) = cluster.nodes.get(&worker.allocation.node) {
            // Fragmentation before on this node
            let frag_before = node.estimate_fragmentation(workload);

            // Simulate releasing the worker on a cloned NodeResources
            let mut new_available = node.available.clone();
            new_available.release_allocation(&worker.allocation);
            let new_node = NodeResourceHelpers::new(new_available.clone(), node.totals.clone());

            // Fragmentation after on this node
            let frag_after = new_node.estimate_fragmentation(workload);

            // Remaining resources on the node AFTER release
            let node_remaining_after = new_available.totals().total_num();

            // Record this deletion option for comparison
            changes.push(FragmentationDeleteResult {
                frag_delta: frag_after - frag_before,
                node_remaining_resources_after: node_remaining_after,
                worker: worker.clone(),
            });
        }
    }

    // Select the worker whose removal results in the lowest cluster fragmentation
    // If multiple workers result in the same fragmentation, prefer removing the one
    // that frees fewer resources (to preserve more capacity for future allocations)
    changes
        .into_iter()
        .min_by(|a, b| {
            // Cost tuples: (frag_delta, -remaining_resources_after)
            // Lower delta is preferred; if equal, prefer freeing fewer resources.
            let ka = (a.frag_delta, -a.node_remaining_resources_after);
            let kb = (b.frag_delta, -b.node_remaining_resources_after);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.worker)
        .expect("changes must be non-empty")
}

// --------------------
// Tests (pure Rust)
// --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::private::scheduling::{allocator::WorkerAllocator, resources as rds};
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
        num_nvdecs_per_gpu: u8,
        num_nvencs_per_gpu: u8,
    ) -> rds::ClusterResources {
        let mut map: HashMap<String, rds::NodeResources> = HashMap::new();
        for i in 0..num_nodes {
            let mut gpus = Vec::with_capacity(gpus_per_node);
            for i in 0..gpus_per_node {
                gpus.push(rds::GpuResources::make_from_num_codecs(
                    i as u8,
                    uuid::Uuid::new_v4(),
                    1.0,
                    num_nvdecs_per_gpu,
                    num_nvencs_per_gpu,
                ));
            }
            map.insert(
                i.to_string(),
                rds::NodeResources::new(cpus_per_node, Some(gpus), None),
            );
        }
        rds::ClusterResources::new(Some(map))
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
            rds::WorkerResources::new(node.to_string(), cpus, Some(gpus), None, None),
        )
    }

    #[test]
    fn test_calculate_unallocatable_gpus_fragment_for_shape() {
        // Test the core fragmentation calculation logic with a partially allocated node
        let available = rds::NodeResources::new(
            4.0,
            Some(vec![
                rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 0.5, 0, 0),
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
            ]),
            None,
        );
        let totals = rds::NodeResources::new(
            8.0,
            Some(vec![
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
            ]),
            None,
        );
        let shape = rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
            num_gpus: 0.7,
            num_cpus: 2.0,
            num_nvdecs: 0,
            num_nvencs: 0,
        });
        let node = NodeResourceHelpers::new(available, totals);
        let result = node.calculate_unallocatable_gpus_fragment_for_shape(&shape);
        assert!(float_eq(result, 0.5), "Expected 0.5, got {result}");
    }

    #[test]
    fn test_estimate_node_fragmentation() {
        // Test workload-based fragmentation estimation with multiple task types
        let available = rds::NodeResources::new(
            4.0,
            Some(vec![
                rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 0.5, 0, 0),
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
            ]),
            None,
        );
        let totals = rds::NodeResources::new(
            8.0,
            Some(vec![
                rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 1.0, 0, 0),
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
            ]),
            None,
        );
        let node = NodeResourceHelpers::new(available, totals);
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.7,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.3,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
            ],
        };
        let result = node.estimate_fragmentation(&workload);
        assert!(
            result >= 0.3 && result <= 0.5,
            "Expected 0.3..0.5, got {result}"
        );
    }

    #[test]
    fn test_calculate_cluster_fragmentation() {
        let available = make_cluster(vec![
            rds::NodeResources::new(
                4.0,
                Some(vec![
                    rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 0.5, 0, 0),
                    rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
                ]),
                None,
            ),
            rds::NodeResources::new(
                8.0,
                Some(vec![
                    rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 0.7, 0, 0),
                    rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 0.3, 0, 0),
                ]),
                None,
            ),
        ]);
        let totals = make_cluster(vec![
            rds::NodeResources::new(
                8.0,
                Some(vec![
                    rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 1.0, 0, 0),
                    rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
                ]),
                None,
            ),
            rds::NodeResources::new(
                8.0,
                Some(vec![
                    rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 1.0, 0, 0),
                    rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
                ]),
                None,
            ),
        ]);
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.7,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.3,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
            ],
        };
        let cluster_helpers = ClusterResourceHelpers::new(available, totals);
        let result = cluster_helpers.estimate_fragmentation(&workload);
        assert!(result >= 0.2 && result <= 1.0, "cluster frag {result}");
    }

    #[test]
    fn test_maybe_allocate_worker_using_fragmentation_gradient_descent() {
        // Integration test for the main fragmentation gradient descent allocation algorithm
        let cluster_resources = make_cluster(vec![rds::NodeResources::new(
            8.0,
            Some(vec![
                rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 1.0, 0, 0),
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
            ]),
            None,
        )]);
        let workers = vec![make_worker("worker1", "stage1", "0", 4.0, &[(0, 0.5)])];
        let alloc = WorkerAllocator::new(cluster_resources, Some(workers)).expect("allocator");

        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.7,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.3,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
            ],
        };
        let shape = rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
            num_gpus: 0.7,
            num_cpus: 2.0,
            num_nvdecs: 0,
            num_nvencs: 0,
        });
        let cluster_helpers = ClusterResourceHelpers::make_from_allocator(&alloc);
        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &cluster_helpers,
            &workload,
            &shape,
            None,
            10.0,
        );
        let res = result.resources.expect("allocation");
        assert_eq!(res.node, "0");
        assert!(res.gpus.iter().any(|g| float_eq(g.fraction, 0.7)));
    }

    #[test]
    fn test_maybe_allocate_worker_various_shapes() {
        // Comprehensive test of allocation algorithm with different worker shape types
        let cluster_resources = make_cluster_resources(2, 8.0, 2, 1, 1);
        let workers = vec![
            make_worker("worker1", "stage1", "0", 4.0, &[(0, 0.5)]),
            make_worker("worker2", "stage1", "1", 2.0, &[(0, 0.3), (1, 0.2)]),
        ];
        let alloc = WorkerAllocator::new(cluster_resources, Some(workers)).expect("allocator");
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.7,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.3,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
            ],
        };
        let cluster_helpers = ClusterResourceHelpers::make_from_allocator(&alloc);

        let cases: Vec<(rds::WorkerShape, bool)> = vec![
            (
                rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                    num_gpus: 0.5,
                    num_cpus: 2.0,
                    num_nvdecs: 0,
                    num_nvencs: 0,
                }),
                true,
            ),
            (
                rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                    num_gpus: 0.8,
                    num_cpus: 4.0,
                    num_nvdecs: 0,
                    num_nvencs: 0,
                }),
                true,
            ),
            (
                rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                    num_gpus: 1.0,
                    num_cpus: 6.0,
                    num_nvdecs: 0,
                    num_nvencs: 0,
                }),
                false,
            ),
            (
                rds::WorkerShape::WholeNumberedGpu(rds::WholeNumberedGpu {
                    num_gpus: 1,
                    num_cpus: 2.0,
                    num_nvdecs_per_gpu: 0,
                    num_nvencs_per_gpu: 0,
                }),
                true,
            ),
            (
                rds::WorkerShape::WholeNumberedGpu(rds::WholeNumberedGpu {
                    num_gpus: 2,
                    num_cpus: 4.0,
                    num_nvdecs_per_gpu: 0,
                    num_nvencs_per_gpu: 0,
                }),
                false,
            ),
            (
                rds::WorkerShape::EntireGpu(rds::EntireGpu {
                    num_gpus: 1,
                    num_cpus: 2.0,
                }),
                true,
            ),
            (
                rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 2.0 }),
                true,
            ),
            (
                rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 8.0 }),
                false,
            ),
            (
                rds::WorkerShape::Codec(rds::Codec {
                    num_cpus: 1.0,
                    num_nvdecs: 1,
                    num_nvencs: 0,
                }),
                true,
            ),
            (
                rds::WorkerShape::Codec(rds::Codec {
                    num_cpus: 1.0,
                    num_nvdecs: 10,
                    num_nvencs: 2,
                }),
                false,
            ),
        ];

        for (shape, expected_ok) in cases.into_iter() {
            let result = find_best_allocation_using_fragmentation_gradient_descent(
                &cluster_helpers,
                &workload,
                &shape,
                None,
                10.0,
            );
            if expected_ok {
                let res = result.resources.expect("expected allocation");
                assert!(res.node == "0" || res.node == "1");
                // CPUs equal to request
                let needed_cpus = match shape {
                    rds::WorkerShape::CpuOnly(s) => s.num_cpus,
                    rds::WorkerShape::Codec(s) => s.num_cpus,
                    rds::WorkerShape::FractionalGpu(s) => s.num_cpus,
                    rds::WorkerShape::WholeNumberedGpu(s) => s.num_cpus,
                    rds::WorkerShape::EntireGpu(s) => s.num_cpus,
                };
                assert!(float_eq(res.cpus, needed_cpus));
            } else {
                assert!(result.resources.is_none());
            }
        }
    }

    #[test]
    fn test_find_worker_to_delete_using_fragmentation_gradient_descent() {
        let cluster_resources = make_cluster(vec![rds::NodeResources::new(
            8.0,
            Some(vec![
                rds::GpuResources::make_from_num_codecs(0, uuid::Uuid::new_v4(), 1.0, 0, 0),
                rds::GpuResources::make_from_num_codecs(1, uuid::Uuid::new_v4(), 1.0, 0, 0),
            ]),
            None,
        )]);
        let workers = vec![
            make_worker("worker1", "stage1", "0", 2.0, &[(0, 0.5)]),
            make_worker("worker2", "stage1", "0", 2.0, &[(1, 0.7)]),
        ];
        let alloc = WorkerAllocator::new(cluster_resources, Some(workers)).expect("allocator");
        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.7,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.3,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
            ],
        };
        let result = find_worker_to_delete_using_fragmentation_gradient_descent(
            &ClusterResourceHelpers::make_from_allocator(&alloc),
            &workload,
            &alloc.get_workers(),
        );
        let id = result.id;
        assert!(id == "worker1" || id == "worker2");
    }

    #[test]
    fn test_prefer_lower_allocated_nodes() {
        // Two nodes, 8 GPUs and 240 CPUs each
        let cluster = make_cluster(vec![
            rds::NodeResources::new(
                240.0,
                Some(
                    (0..8)
                        .map(|_| {
                            rds::GpuResources::make_from_num_codecs(
                                0,
                                uuid::Uuid::new_v4(),
                                1.0,
                                0,
                                0,
                            )
                        })
                        .collect(),
                ),
                None,
            ),
            rds::NodeResources::new(
                240.0,
                Some(
                    (0..8)
                        .map(|_| {
                            rds::GpuResources::make_from_num_codecs(
                                0,
                                uuid::Uuid::new_v4(),
                                1.0,
                                0,
                                0,
                            )
                        })
                        .collect(),
                ),
                None,
            ),
        ]);
        let workers = vec![
            // Node 0: highly allocated CPUs
            make_worker("worker1", "stage1", "0", 100.0, &[]),
            // Node 1: lightly allocated CPUs
            make_worker("worker2", "stage2", "1", 10.0, &[]),
        ];
        let alloc = WorkerAllocator::new(cluster, Some(workers)).expect("allocator");

        let workload = Workload {
            stages: vec![
                Stage {
                    frequency: 0.5,
                    shape: rds::WorkerShape::WholeNumberedGpu(rds::WholeNumberedGpu {
                        num_cpus: 1.0,
                        num_gpus: 1,
                        num_nvdecs_per_gpu: 0,
                        num_nvencs_per_gpu: 0,
                    }),
                },
                Stage {
                    frequency: 0.5,
                    shape: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                },
            ],
        };
        let shape = rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 });
        let cluster_helpers = ClusterResourceHelpers::make_from_allocator(&alloc);
        let result = find_best_allocation_using_fragmentation_gradient_descent(
            &cluster_helpers,
            &workload,
            &shape,
            None,
            10.0,
        );
        let res = result.resources.expect("allocation");
        assert_eq!(res.node, "1");
        assert!(float_eq(res.cpus, 1.0));
    }

    #[test]
    fn test_node_resources_totals() {
        let cluster = make_cluster_resources(1, 16.0, 2, 2, 2);
        let helpers = ClusterResourceHelpers::new(cluster.clone(), cluster);
        let node = helpers.nodes.get("0").expect("node 0");
        assert_eq!(node.gpus.len(), 2);
        assert!(float_eq(node.available.cpus, 16.0));
        let sum_gpu: f32 = node.gpus.iter().map(|g| g.available.gpu_fraction).sum();
        assert!(float_eq(sum_gpu, 2.0));
        let sum_nvdec: usize = node.gpus.iter().map(|g| g.available.num_nvdecs()).sum();
        let sum_nvenc: usize = node.gpus.iter().map(|g| g.available.num_nvencs()).sum();
        assert_eq!(sum_nvdec, 4);
        assert_eq!(sum_nvenc, 4);
    }

    #[test]
    fn test_gpu_resource_helpers_can_be_used_to_allocate() {
        let cases = vec![
            (
                rds::GpuResources::new(0, uuid::Uuid::new_v4(), 1.0, vec![0, 1], vec![0, 1]),
                rds::WorkerShape::EntireGpu(rds::EntireGpu {
                    num_gpus: 1,
                    num_cpus: 4.0,
                }),
                16.0,
                true,
            ),
            (
                rds::GpuResources::new(0, uuid::Uuid::new_v4(), 0.5, vec![0, 1], vec![0, 1]),
                rds::WorkerShape::EntireGpu(rds::EntireGpu {
                    num_gpus: 1,
                    num_cpus: 4.0,
                }),
                16.0,
                false,
            ),
            (
                rds::GpuResources::new(0, uuid::Uuid::new_v4(), 1.0, vec![0, 1], vec![0, 1]),
                rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                    num_gpus: 0.5,
                    num_cpus: 4.0,
                    num_nvdecs: 1,
                    num_nvencs: 1,
                }),
                16.0,
                true,
            ),
            (
                rds::GpuResources::new(0, uuid::Uuid::new_v4(), 0.4, vec![0, 1], vec![0, 1]),
                rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                    num_gpus: 0.5,
                    num_cpus: 4.0,
                    num_nvdecs: 1,
                    num_nvencs: 1,
                }),
                16.0,
                false,
            ),
            (
                rds::GpuResources::new(0, uuid::Uuid::new_v4(), 1.0, vec![0], vec![0]),
                rds::WorkerShape::Codec(rds::Codec {
                    num_cpus: 4.0,
                    num_nvdecs: 2,
                    num_nvencs: 2,
                }),
                16.0,
                false,
            ),
        ];

        for (gpu_res, shape, available_cpus, expected) in cases {
            let helper = GpuResourceHelpers::new(gpu_res.clone(), gpu_res);
            let ok = helper.can_be_used_to_allocate(&shape, available_cpus);
            assert_eq!(ok, expected, "case failed");
        }
    }

    #[test]
    fn test_node_resource_helpers_find_possible_allocations() {
        let cluster = make_cluster_resources(1, 16.0, 2, 2, 2);
        let node = NodeResourceHelpers::new(
            cluster.nodes.get("0").unwrap().clone(),
            cluster.nodes.get("0").unwrap().clone(),
        );

        // CPU-only
        let shape_cpu = rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 4.0 });
        let allocs = node.find_possible_allocations(&shape_cpu, "0");
        assert_eq!(allocs.len(), 1);

        // Codec: expect 9 (3 distributions for nvdec x 3 for nvenc)
        let shape_codec = rds::WorkerShape::Codec(rds::Codec {
            num_cpus: 4.0,
            num_nvdecs: 2,
            num_nvencs: 2,
        });
        let allocs_codec = node.find_possible_allocations(&shape_codec, "0");
        assert_eq!(allocs_codec.len(), 9);

        // Fractional GPU: one per GPU
        let shape_frac = rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
            num_cpus: 4.0,
            num_gpus: 0.5,
            num_nvdecs: 1,
            num_nvencs: 1,
        });
        let allocs_frac = node.find_possible_allocations(&shape_frac, "0");
        assert_eq!(allocs_frac.len(), 2);
    }

    #[test]
    fn test_mixed_resource_scenarios() {
        // Case 1
        let cluster1 = make_cluster_resources(1, 16.0, 3, 2, 2);
        let workers1 = vec![
            make_worker("worker1", "stage1", "0", 4.0, &[(0, 1.0)]),
            make_worker("worker2", "stage1", "0", 4.0, &[(1, 0.5)]),
        ];
        let alloc1 = WorkerAllocator::new(cluster1, Some(workers1)).expect("alloc1");
        let workload1 = Workload {
            stages: vec![
                Stage {
                    frequency: 0.6,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.7,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.4,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.3,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
            ],
        };
        let shape1 = rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
            num_cpus: 2.0,
            num_gpus: 0.5,
            num_nvdecs: 1,
            num_nvencs: 1,
        });
        let result1 = find_best_allocation_using_fragmentation_gradient_descent(
            &ClusterResourceHelpers::make_from_allocator(&alloc1),
            &workload1,
            &shape1,
            None,
            10.0,
        )
        .resources;
        assert!(result1.is_some());

        // Case 2
        let cluster2 = make_cluster_resources(1, 8.0, 2, 1, 1);
        let workers2 = vec![make_worker(
            "worker1",
            "stage1",
            "0",
            6.0,
            &[(0, 0.8), (1, 0.8)],
        )];
        let alloc2 = WorkerAllocator::new(cluster2, Some(workers2)).expect("alloc2");
        let workload2 = Workload {
            stages: vec![
                Stage {
                    frequency: 0.5,
                    shape: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.5,
                        num_cpus: 2.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                Stage {
                    frequency: 0.5,
                    shape: rds::WorkerShape::WholeNumberedGpu(rds::WholeNumberedGpu {
                        num_cpus: 4.0,
                        num_gpus: 1,
                        num_nvdecs_per_gpu: 0,
                        num_nvencs_per_gpu: 0,
                    }),
                },
            ],
        };
        let shape2 = rds::WorkerShape::WholeNumberedGpu(rds::WholeNumberedGpu {
            num_cpus: 3.0,
            num_gpus: 1,
            num_nvdecs_per_gpu: 0,
            num_nvencs_per_gpu: 0,
        });
        let result2 = find_best_allocation_using_fragmentation_gradient_descent(
            &ClusterResourceHelpers::make_from_allocator(&alloc2),
            &workload2,
            &shape2,
            None,
            10.0,
        )
        .resources;
        assert!(result2.is_none());
    }
}