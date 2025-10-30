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

//! Resource allocation optimizer for multi-stage streaming pipelines using linear programming.
//!
//! This module implements an optimization-based resource allocation system that determines
//! the optimal number of workers for each stage in a streaming pipeline while respecting
//! resource constraints and throughput objectives. It handles both automatically scaled
//! stages and manually configured stages.
//!
//! This module is "naive" in the sense that it does not consider bin-packing. As a result,
//! it is not a true optimal solution to Yotta's auto-scaling problem. It may end up
//! allocating miss-matched stages. It is a useful estimate, however.
//!
//! # Key Features
//!
//! - Linear programming-based optimization using the `good_lp` crate
//! - Support for multiple resource types (CPU, GPU, NVDEC, NVENC)
//! - Balanced throughput across stages
//! - Mixed manual and automatic worker allocation
//! - Resource constraint satisfaction
//! - Throughput maximization
//!
//! # Algorithm Overview
//!
//! The optimizer uses mixed-integer linear programming to solve a problem that:
//! 1. Maximizes minimum throughput across stages
//! 2. Balances stage throughputs to prevent bottlenecks
//! 3. Respects cluster resource limits
//! 4. Handles manually specified worker counts
//!
//! # Mathematical Formulation
//!
//! The optimization problem can be expressed as:
//!
//! **Maximize:** z - 0.001 * Σ(s_i)
//!
//! **Subject to:**
//! - t_i ≥ z                                    (minimum throughput constraint)
//! - t_i = x_i * samples_per_sample_i * bps_i * batch_size_i  (throughput definition)
//! - t_i = z + s_i                              (slack variable for excess capacity)
//! - Σ(x_i * resources_i) ≤ cluster_resources   (resource constraints)
//! - x_i ≥ 1, x_i ∈ ℤ                          (integer worker counts, minimum 1)
//!
//! Where:
//! - z: minimum throughput across all stages
//! - x_i: number of workers for stage i
//! - t_i: throughput of stage i
//! - s_i: slack variable measuring excess capacity above target throughput z
//! - samples_per_sample_i: normalization factor for inter-stage sample flow
//! - bps_i: batches per second per worker for stage i
//! - batch_size_i: number of inputs per batch for stage i

use good_lp::variable;
use good_lp::{
    Expression, ProblemVariables, Solution, SolverModel, Variable, constraint, default_solver,
};
use thiserror::Error;

use crate::pipelines::private::scheduling::resources;

/// Error types that can occur during resource allocation optimization.
///
/// These errors represent various failure modes of the allocation algorithm,
/// from mathematical infeasibility to invalid input configurations.
#[derive(Error, Debug)]
pub enum AllocationError {
    /// The optimization problem has no feasible solution.
    ///
    /// This typically occurs when:
    /// - Manual stage allocations exceed available cluster resources
    /// - Resource constraints are too restrictive for any valid allocation
    /// - The solver cannot find a solution that satisfies all constraints
    #[error("No feasible solution found for the given problem")]
    NoFeasibleSolution,

    /// A stage has invalid configuration parameters.
    ///
    /// This occurs when stage parameters violate basic requirements:
    /// - `batches_per_second_per_worker` must be positive
    /// - `num_returns_per_batch` must be positive  
    /// - `stage_batch_size` must be positive
    #[error("Invalid stage configuration for stage '{stage}': {reason}")]
    InvalidStage { stage: String, reason: String },
}

/// Represents a single stage in the resource allocation problem.
///
/// Each stage has specific resource requirements, processing speed characteristics,
/// and optionally a fixed number of workers. This struct captures all the information
/// needed to model a pipeline stage in the optimization problem.
///
/// # Examples
///
/// ```rust
/// use _cosmos_xenna::pipelines::private::scheduling::naiive_worker_allocation::AllocationProblemStage;
/// use _cosmos_xenna::pipelines::private::scheduling::resources::PoolOfResources;
///
/// // Create a stage that processes 2 batches per second per worker
/// let stage = AllocationProblemStage {
///     name: "video_decoder".to_string(),
///     batches_per_second_per_worker: 2.0,
///     num_returns_per_batch: 1.0,
///     stage_batch_size: 8,
///     resources_per_worker: PoolOfResources {
///         cpus: 2.0,
///         gpus: 1.0,
///     },
///     requested_num_workers: None, // Let optimizer decide
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AllocationProblemStage {
    /// Unique identifier for the stage.
    ///
    /// Used for debugging, error reporting, and result identification.
    pub name: String,

    /// Processing speed in batches per second per worker.
    ///
    /// This represents how many batches a single worker can process per second.
    /// Must be positive. Higher values indicate faster processing stages.
    pub batches_per_second_per_worker: f64,

    /// Average number of output items per input batch.
    ///
    /// This accounts for stages that may generate more or fewer outputs than inputs.
    /// For example:
    /// - A 1:1 transformation has `num_returns_per_batch = 1.0`
    /// - A stage that duplicates inputs has `num_returns_per_batch = 2.0`  
    /// - A filtering stage might have `num_returns_per_batch = 0.5`
    /// Must be positive.
    pub num_returns_per_batch: f64,

    /// Number of input items processed together in a single batch.
    ///
    /// Larger batch sizes can improve throughput but may increase latency.
    /// Must be positive.
    pub stage_batch_size: u32,

    /// Resource requirements per worker instance.
    ///
    /// Specifies how many CPUs, GPUs, NVDECs, and NVENCs each worker needs.
    /// The optimizer will ensure total resource usage doesn't exceed cluster capacity.
    pub resources_per_worker: resources::PoolOfResources,

    /// Optional fixed number of workers for this stage.
    ///
    /// If `Some(n)`, exactly `n` workers will be allocated to this stage.
    /// If `None`, the optimizer will determine the optimal number of workers.
    /// Manual specifications are honored exactly and their resources are reserved
    /// before optimizing remaining stages.
    pub requested_num_workers: Option<usize>,
}

impl AllocationProblemStage {
    /// Check if this stage has a manually specified worker count.
    ///
    /// # Returns
    ///
    /// `true` if `requested_num_workers` is `Some`, meaning the number of workers
    /// for this stage is fixed and should not be optimized. `false` if the optimizer
    /// should determine the optimal worker count.
    pub fn was_manually_specified(&self) -> bool {
        self.requested_num_workers.is_some()
    }

    /// Validate that all stage parameters are within acceptable ranges.
    ///
    /// Checks that:
    /// - `batches_per_second_per_worker` is positive
    /// - `num_returns_per_batch` is positive
    /// - `stage_batch_size` is positive
    ///
    /// # Returns
    ///
    /// `Ok(())` if all parameters are valid, otherwise `Err(AllocationError::InvalidStage)`
    /// with details about which parameter is invalid.
    fn validate(&self) -> Result<(), AllocationError> {
        // Processing speed must be positive - zero or negative speed is meaningless
        if self.batches_per_second_per_worker <= 0.0 {
            return Err(AllocationError::InvalidStage {
                stage: self.name.clone(),
                reason: "batches_per_second_per_worker must be positive".into(),
            });
        }

        // Return count must be positive - stages must produce some output
        if self.num_returns_per_batch <= 0.0 {
            return Err(AllocationError::InvalidStage {
                stage: self.name.clone(),
                reason: "num_returns_per_batch must be positive".into(),
            });
        }

        // Batch size must be positive - cannot process zero items per batch
        if self.stage_batch_size == 0 {
            return Err(AllocationError::InvalidStage {
                stage: self.name.clone(),
                reason: "stage_batch_size must be positive".into(),
            });
        }

        Ok(())
    }
}

/// Defines a complete resource allocation problem for optimization.
///
/// Contains all information needed to solve the worker allocation problem:
/// - Stage definitions with resource requirements
/// - Available cluster resources  
/// - Processing speed characteristics
/// - Manual worker count specifications
///
/// The problem represents a pipeline of stages that need to be allocated workers
/// such that throughput is maximized while respecting resource constraints.
#[derive(Debug, Clone)]
pub struct AllocationProblem {
    /// List of all pipeline stages to allocate resources for.
    ///
    /// Stages are processed in the order they appear in this vector, which affects
    /// the calculation of inter-stage sample flow normalization factors.
    pub stages: Vec<AllocationProblemStage>,

    /// Total available resources in the cluster.
    ///
    /// The optimizer will ensure that the sum of all allocated worker resources
    /// does not exceed these limits for any resource type (CPU, GPU, NVDEC, NVENC).
    pub cluster_resources: resources::PoolOfResources,
}

/// Results of resource allocation for a single pipeline stage.
///
/// Tracks both the original problem definition and the solved allocation
/// for a stage, providing methods to calculate resource usage and throughput.
/// This combines the input stage specification with the optimizer's solution.
#[derive(Debug, Clone)]
pub struct AllocationResultStage {
    /// Original stage definition from the allocation problem.
    ///
    /// Contains all the input parameters for this stage including resource
    /// requirements, processing speeds, and manual worker specifications.
    pub problem: AllocationProblemStage,

    /// Number of workers allocated to this stage by the optimizer.
    ///
    /// For manually specified stages, this equals `problem.requested_num_workers.unwrap()`.
    /// For auto-scaling stages, this is the optimal number determined by the optimizer.
    pub num_workers: usize,

    /// Normalization factor for inter-stage sample flow.
    ///
    /// Represents how many first-stage input samples are needed to generate
    /// one input sample for this stage. This accounts for stages that may
    /// produce more or fewer outputs than inputs (via `num_returns_per_batch`
    /// and `stage_batch_size`). Used to normalize throughput calculations
    /// across stages with different sample flow characteristics.
    pub input_samples_per_sample: f64,
}

impl AllocationResultStage {
    /// Get the stage identifier.
    ///
    /// # Returns
    ///
    /// The unique name of the stage from the original problem definition.
    pub fn name(&self) -> &str {
        &self.problem.name
    }

    /// Calculate per-worker throughput for this stage.
    ///
    /// Computes how many output items a single worker produces per second,
    /// taking into account both the batch processing rate and the number
    /// of outputs generated per batch.
    ///
    /// # Returns
    ///
    /// Output items processed per second per worker.
    /// Formula: `batches_per_second_per_worker * num_returns_per_batch`
    pub fn throughput_per_worker(&self) -> f64 {
        self.problem.batches_per_second_per_worker * self.problem.num_returns_per_batch
    }

    /// Calculate total stage throughput across all allocated workers.
    ///
    /// This represents the maximum number of output items this stage can
    /// produce per second with its current worker allocation.
    ///
    /// # Returns
    ///
    /// Total output items processed per second for the entire stage.
    /// Formula: `throughput_per_worker() * num_workers`
    pub fn total_throughput(&self) -> f64 {
        self.throughput_per_worker() * self.num_workers as f64
    }

    /// Calculate total resources consumed by all workers in this stage.
    ///
    /// Multiplies the per-worker resource requirements by the number of
    /// allocated workers to get the total resource consumption for this stage.
    ///
    /// # Returns
    ///
    /// Combined resource usage across all workers in this stage.
    pub fn total_resource_usage(&self) -> resources::PoolOfResources {
        self.problem
            .resources_per_worker
            .multiply_by(self.num_workers as f32)
    }
}

/// Complete results of the resource allocation optimization.
///
/// Contains the solved allocation for all stages and provides methods
/// to analyze and display the results. This represents the final output
/// of the optimization algorithm with worker allocations for each stage.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Allocation results for each pipeline stage.
    ///
    /// Contains both the original problem definition and the solution
    /// (number of workers allocated) for each stage in the pipeline.
    /// Stages appear in the same order as in the original problem.
    pub stages: Vec<AllocationResultStage>,

    /// Total available cluster resources.
    ///
    /// Copy of the cluster resource constraints from the original problem.
    /// Useful for analyzing resource utilization in the final allocation.
    pub cluster_resources: resources::PoolOfResources,

    /// Achieved pipeline throughput (minimum across all stages).
    ///
    /// This represents the bottleneck throughput of the entire pipeline.
    /// The pipeline can process at most this many items per second end-to-end.
    /// Calculated as the minimum of all stage throughputs after normalization
    /// for inter-stage sample flow.
    pub throughput: f64,
}

impl AllocationResult {
    /// Generate a human-readable representation of the allocation results.
    ///
    /// Creates a formatted table showing key metrics for each stage including
    /// worker allocations, throughput calculations, and resource usage.
    /// This is useful for debugging optimization results and understanding
    /// how resources were distributed across stages.
    ///
    /// # Returns
    ///
    /// A formatted string containing:
    /// - Overall pipeline throughput
    /// - Per-stage breakdown showing:
    ///   - Stage name and processing characteristics
    ///   - Whether worker count was manually specified
    ///   - Number of allocated workers
    ///   - Throughput and resource usage metrics
    pub fn to_debug_str(&self) -> String {
        use comfy_table::{Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};

        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header([
                "Stage",                     // Stage name
                "Thpt/worker",               // Batches per second per worker
                "Input samples/sample",      // Normalization factor for sample flow
                "Input samples thpt/worker", // Normalized throughput per worker
                "Avg returns/batch",         // Output items per input batch
                "Batch size",                // Items processed per batch
                "Manual",                    // Whether worker count was manually specified
                "Workers",                   // Number of allocated workers
                "Throughput",                // Total stage throughput (output items/sec)
                "Input samples thpt",        // Normalized total throughput
                "CPUs",                      // Total CPU resources used
                "GPUs",                      // Total GPU resources used
            ]);

        // Add a row for each stage with its allocation details
        for s in &self.stages {
            let r = s.total_resource_usage();
            let thpt_worker = s.problem.batches_per_second_per_worker;
            let in_samples_per_sample = s.input_samples_per_sample;
            table.add_row([
                s.name().into(),
                format!("{:.6}", thpt_worker),
                format!("{:.6}", in_samples_per_sample),
                format!("{:.6}", thpt_worker * in_samples_per_sample),
                format!("{:.6}", s.problem.num_returns_per_batch),
                s.problem.stage_batch_size.to_string(),
                s.problem.was_manually_specified().to_string(),
                s.num_workers.to_string(),
                format!("{:.6}", s.total_throughput()),
                format!("{:.6}", s.total_throughput() * in_samples_per_sample),
                format!("{:.3}", r.cpus),
                format!("{:.3}", r.gpus),
            ]);
        }

        format!(
            "Allocation Result (Throughput: {:.3}):\n{}",
            self.throughput, table
        )
    }
}

// ----------------------------
// Public API
// ----------------------------

/// Solve the complete resource allocation problem, handling both manual and auto-scaling stages.
///
/// This function serves as the main entry point for resource allocation. It implements
/// a comprehensive optimization strategy that:
/// 1. Validates input parameters for all stages
/// 2. Separates manually specified stages from auto-scaling stages  
/// 3. Reserves resources for manual stages
/// 4. Optimizes remaining resources for auto-scaling stages using linear programming
/// 5. Combines results into a complete allocation
///
/// The process ensures that:
/// - Manual stage specifications are honored exactly
/// - Resource constraints are respected
/// - Remaining resources are optimally allocated to auto-scaling stages
/// - Pipeline throughput is maximized within constraints
///
/// # Arguments
///
/// * `problem` - Complete allocation problem definition including all stages and cluster resources
///
/// # Returns
///
/// * `Ok(AllocationResult)` - Complete allocation solution for all stages with optimal worker counts
/// * `Err(AllocationError)` - If the problem is infeasible or has invalid parameters
///
/// # Errors
///
/// * `AllocationError::InvalidStage` - If any stage has invalid parameters (non-positive speeds, batch sizes, etc.)
/// * `AllocationError::NoFeasibleSolution` - If manual allocations exceed available resources or no valid allocation exists
///
/// # Examples
///
/// ```rust
/// use _cosmos_xenna::pipelines::private::scheduling::naiive_worker_allocation::{AllocationProblem, AllocationProblemStage, solve_allocation};
/// use _cosmos_xenna::pipelines::private::scheduling::resources::PoolOfResources;
///
/// fn example() -> Result<(), Box<dyn std::error::Error>> {
///     let problem = AllocationProblem {
///         stages: vec![
///             AllocationProblemStage {
///                 name: "decode".to_string(),
///                 batches_per_second_per_worker: 2.0,
///                 num_returns_per_batch: 1.0,
///                 stage_batch_size: 4,
///                 resources_per_worker: PoolOfResources { cpus: 1.0, gpus: 1.0 },
///                 requested_num_workers: None,
///             },
///             AllocationProblemStage {
///                 name: "process".to_string(),
///                 batches_per_second_per_worker: 1.0,
///                 num_returns_per_batch: 1.0,
///                 stage_batch_size: 8,
///                 resources_per_worker: PoolOfResources { cpus: 2.0, gpus: 0.0 },
///                 requested_num_workers: Some(3), // Manually specified
///             },
///         ],
///         cluster_resources: PoolOfResources { cpus: 20.0, gpus: 8.0 },
///     };
///
///     let result = solve_allocation(problem)?;
///     println!("Pipeline throughput: {:.2} items/sec", result.throughput);
///     Ok(())
/// }
/// ```
pub fn solve_allocation(problem: AllocationProblem) -> Result<AllocationResult, AllocationError> {
    // Validate all stage parameters before proceeding with optimization
    // This catches invalid configurations early with clear error messages
    for s in &problem.stages {
        s.validate()?;
    }

    // Calculate normalization factors for inter-stage sample flow
    // This accounts for stages that produce different numbers of outputs than inputs
    // (e.g., a stage that filters or duplicates data)
    let input_samples_per_sample = calculate_input_samples_per_sample(
        &problem
            .stages
            .iter()
            .map(|s| s.stage_batch_size)
            .collect::<Vec<u32>>(),
        &problem
            .stages
            .iter()
            .map(|s| s.num_returns_per_batch)
            .collect::<Vec<f64>>(),
    );

    // Separate stages into two categories for different handling:
    // - Manual stages: worker count fixed by user, resources reserved
    // - Auto stages: worker count optimized by linear programming
    let mut manual_stages: Vec<&AllocationProblemStage> = Vec::new();
    let mut auto_stages: Vec<&AllocationProblemStage> = Vec::new();
    let mut auto_samples_per_sample: Vec<f64> = Vec::new();

    for (i, s) in problem.stages.iter().enumerate() {
        if s.requested_num_workers.is_some() {
            // Stage has fixed worker count - no optimization needed
            manual_stages.push(s);
        } else {
            // Stage worker count should be optimized
            auto_stages.push(s);
            // Keep normalization factor for this stage
            auto_samples_per_sample.push(input_samples_per_sample[i]);
        }
    }

    // Calculate total resources consumed by manually specified stages
    // These resources are reserved and cannot be used for auto-scaling stages
    let mut manual_resources_used = resources::PoolOfResources {
        cpus: 0.0,
        gpus: 0.0,
    };

    for s in &manual_stages {
        // Get the fixed worker count (guaranteed to be Some by filtering above)
        let n = s.requested_num_workers.unwrap_or(0) as f32;
        // Add this stage's resource consumption to the total
        manual_resources_used = manual_resources_used.add(&s.resources_per_worker.multiply_by(n));
    }

    // Verify that manual allocations don't exceed available cluster resources
    // If they do, the problem is immediately infeasible
    if !problem.cluster_resources.contains(&manual_resources_used) {
        return Err(AllocationError::NoFeasibleSolution);
    }

    // Calculate resources remaining for auto-scaling stages after reserving manual allocations
    let remaining_resources = problem.cluster_resources.sub(&manual_resources_used);

    // Optimize worker allocation for auto-scaling stages using linear programming
    let auto_result = if !auto_stages.is_empty() {
        // Create a sub-problem containing only auto-scaling stages and remaining resources
        let sub_problem = AllocationProblem {
            stages: auto_stages.into_iter().cloned().collect(),
            cluster_resources: remaining_resources,
        };
        // Solve the optimization problem for these stages
        solve_allocation_with_no_manual_stages(sub_problem, auto_samples_per_sample)?
    } else {
        // No auto-scaling stages - create empty result
        AllocationResult {
            stages: Vec::new(),
            cluster_resources: remaining_resources,
            throughput: 0.0,
        }
    };

    // Combine manual and auto allocation results, preserving original stage order
    // This is important because stage order affects pipeline semantics
    let mut stages_out: Vec<AllocationResultStage> = Vec::with_capacity(problem.stages.len());
    let mut auto_idx = 0usize; // Index into auto_result.stages

    for (i, s) in problem.stages.into_iter().enumerate() {
        let num_workers = if let Some(n) = s.requested_num_workers {
            // Manual stage - use the specified worker count
            n
        } else {
            // Auto stage - use the optimized worker count
            let n = auto_result.stages[auto_idx].num_workers;
            auto_idx += 1;
            n
        };

        stages_out.push(AllocationResultStage {
            problem: s,
            num_workers,
            input_samples_per_sample: input_samples_per_sample[i],
        });
    }

    // Calculate actual pipeline throughput as the minimum across all stages
    // The pipeline is only as fast as its slowest stage (bottleneck)
    let actual_throughput = stages_out
        .iter()
        .map(|s| s.total_throughput())
        .fold(f64::INFINITY, |a, b| a.min(b));

    Ok(AllocationResult {
        stages: stages_out,
        cluster_resources: problem.cluster_resources,
        throughput: actual_throughput,
    })
}

// ----------------------------
// Internals
// ----------------------------

/// Solve the worker allocation problem for automatically scaled stages using linear programming.
///
/// This function implements the core optimization algorithm that determines optimal
/// worker counts for pipeline stages when no manual worker specifications are present.
/// It formulates and solves a mixed-integer linear programming problem that:
///
/// 1. Maximizes the minimum *normalized* throughput across all stages
/// 2. Respects all resource constraints (CPU, GPU, NVDEC, NVENC)
/// 3. Minimizes excess capacity using slack variables to encourage balanced allocation
/// 4. Ensures at least 1 worker per stage (lower bound constraint)
///
/// # Mathematical Formulation
///
/// The optimization problem is:
///
/// **Maximize:** z - 0.001 * Σ(s_i)
///
/// **Subject to:**
/// - t_i ≥ z                                    (minimum throughput constraint)
/// - t_i = x_i * samples_per_sample_i * bps_i * batch_size_i  (throughput definition)  
/// - t_i = z + s_i                              (slack definition)
/// - Σ(x_i * resources_i) ≤ cluster_resources   (resource constraints for each type)
/// - x_i ≥ 1, x_i ∈ ℤ                          (integer worker counts, minimum 1)
///
/// Where:
/// - z: minimum normalized throughput across all stages (objective to maximize)
/// - x_i: number of workers for stage i (decision variables)
/// - t_i: normalized throughput of stage i
/// - s_i: slack variable measuring excess capacity above target throughput z
/// - samples_per_sample_i: normalization factor for stage i
/// - bps_i: batches per second per worker for stage i
/// - batch_size_i: items processed per batch for stage i
///
/// # Arguments
///
/// * `problem` - Allocation problem containing only auto-scaling stages
/// * `input_samples_per_sample` - Normalization factors for each stage
///
/// # Returns
///
/// * `Ok(AllocationResult)` - Optimal allocation with worker counts and achieved throughput
/// * `Err(AllocationError::NoFeasibleSolution)` - If no valid allocation exists within constraints
fn solve_allocation_with_no_manual_stages(
    problem: AllocationProblem,
    input_samples_per_sample: Vec<f64>,
) -> Result<AllocationResult, AllocationError> {
    assert_eq!(problem.stages.len(), input_samples_per_sample.len());

    // Initialize the mixed-integer linear programming problem using good_lp
    let mut vars = ProblemVariables::new();

    // z = minimum throughput across all stages (our primary objective to maximize)
    let z = vars.add(variable().name("z").min(0.0));

    // Create decision variables for each stage
    let mut x_vars: Vec<Variable> = Vec::with_capacity(problem.stages.len()); // Worker counts
    let mut t_vars: Vec<Variable> = Vec::with_capacity(problem.stages.len()); // Throughputs  
    let mut s_vars: Vec<Variable> = Vec::with_capacity(problem.stages.len()); // Slack variables

    for i in 0..problem.stages.len() {
        // x_i: number of workers for stage i (integer, minimum 1)
        // Must be integer since we can't have fractional workers
        // Minimum 1 ensures every stage has at least one worker
        let xi = vars.add(variable().integer().min(1.0).name(format!("x_{}", i)));

        // t_i: normalized throughput for stage i (continuous, non-negative)
        // Represents stage throughput in terms of first-stage input samples per second
        let ti = vars.add(variable().min(0.0).name(format!("t_{}", i)));

        // s_i: slack variable measuring excess capacity above minimum throughput z
        // Used to encourage balanced allocation and prevent over-allocation
        let si = vars.add(variable().min(0.0).name(format!("s_{}", i)));

        x_vars.push(xi);
        t_vars.push(ti);
        s_vars.push(si);
    }

    // Define the objective function: Maximize z - 0.001 * Σ(s_i)
    // Primary goal: maximize minimum throughput z
    // Secondary goal: minimize slack (excess capacity) to encourage balanced allocation
    let mut objective: Expression = z.into();
    let slack_penalty_coeff = 0.001; // Small coefficient ensures throughput remains primary objective
    let mut slack_sum: Expression = 0.0.into();
    for si in &s_vars {
        slack_sum = slack_sum + *si;
    }
    objective = objective - slack_penalty_coeff * slack_sum;

    // Create the optimization model with our objective
    let mut model = vars.maximise(objective).using(default_solver);

    // Add constraints for each stage to define the optimization problem
    for (i, stage) in problem.stages.iter().enumerate() {
        // Constraint 1: Each stage throughput must be at least the minimum throughput z
        // This ensures that z represents the true bottleneck throughput
        model = model.with(constraint!(t_vars[i] >= z));

        // Constraint 2: Define actual throughput for stage i
        // t_i = x_i * input_samples_per_sample_i * batches_per_second_per_worker * stage_batch_size
        // This calculates normalized throughput in terms of first-stage input samples per second
        let coeff = input_samples_per_sample[i]          // Normalization factor
            * stage.batches_per_second_per_worker        // Processing speed
            * stage.stage_batch_size as f64; // Batch size
        model = model.with(constraint!(t_vars[i] == x_vars[i] * coeff));

        // Constraint 3: Define slack variable relationship
        // t_i = z + s_i, which means s_i = t_i - z (excess capacity above minimum)
        // This allows stages to have more throughput than the minimum while penalizing excess
        model = model.with(constraint!(t_vars[i] == z + s_vars[i]));
    }

    // Resource constraints: ensure total resource usage doesn't exceed cluster capacity
    // These constraints are crucial for ensuring the solution is physically implementable

    // Add a tiny epsilon to right-hand side to mitigate floating-point rounding errors
    // that can occur when converting from f32 (resource pools) to f64 (solver precision)
    let numeric_epsilon: f64 = 1e-6;

    // Build expressions for total resource consumption across all stages
    let mut sum_cpus: Expression = 0.0.into();
    let mut sum_gpus: Expression = 0.0.into();

    for (i, stage) in problem.stages.iter().enumerate() {
        // For each stage, add: (number of workers) * (resources per worker)
        sum_cpus = sum_cpus + x_vars[i] * stage.resources_per_worker.cpus as f64;
        sum_gpus = sum_gpus + x_vars[i] * stage.resources_per_worker.gpus as f64;
    }

    // Add resource capacity constraints for each resource type
    model = model.with(constraint!(
        sum_cpus <= problem.cluster_resources.cpus as f64 + numeric_epsilon
    ));
    model = model.with(constraint!(
        sum_gpus <= problem.cluster_resources.gpus as f64 + numeric_epsilon
    ));

    // Solve the mixed-integer linear programming problem
    // The solver will find the optimal allocation that maximizes minimum throughput
    // while respecting all constraints, or return an error if no feasible solution exists
    let solution = model
        .solve()
        .map_err(|_| AllocationError::NoFeasibleSolution)?;

    // Extract the optimal worker allocations and build result stages
    let mut stages: Vec<AllocationResultStage> = Vec::with_capacity(problem.stages.len());
    for (i, stage) in problem.stages.into_iter().enumerate() {
        // Get the optimal number of workers for this stage
        let xi_val = solution.value(x_vars[i]);
        // Round to nearest integer (should already be integer due to MILP constraints)
        let num_workers = xi_val.round() as usize;

        stages.push(AllocationResultStage {
            problem: stage,
            num_workers,
            input_samples_per_sample: input_samples_per_sample[i],
        });
    }

    // Extract the achieved minimum throughput
    let throughput = solution.value(z);

    Ok(AllocationResult {
        stages,
        cluster_resources: problem.cluster_resources,
        throughput,
    })
}

/// Calculate normalization factors for inter-stage sample flow in a pipeline.
///
/// This function computes how many first-stage input samples are needed to generate
/// one input sample for each stage in the pipeline. This is necessary because stages
/// may produce different numbers of outputs than inputs (via filtering, duplication,
/// aggregation, etc.), which affects throughput calculations.
///
/// The calculation works by propagating the sample flow through the pipeline:
/// - Stage 0 (first stage): 1.0 first-stage samples per stage-0 sample (by definition)
/// - Stage i: samples_per_sample[i-1] * (batch_size[i-1] / returns_per_batch[i-1])
///
/// # Arguments
///
/// * `batch_sizes` - Number of input items processed per batch for each stage
/// * `num_returns_per_batch` - Number of output items produced per batch for each stage
///
/// # Returns
///
/// Vector where `result[i]` is the number of first-stage input samples needed
/// to generate one input sample for stage i.
///
/// # Examples
///
/// ```rust
/// use _cosmos_xenna::pipelines::private::scheduling::naiive_worker_allocation::calculate_input_samples_per_sample;
///
/// // Pipeline: decode -> filter -> process
/// let batch_sizes = vec![8, 4, 2];           // Batch sizes
/// let returns_per_batch = vec![1.0, 0.5, 1.0]; // Filter removes 50% of samples
///
/// let factors = calculate_input_samples_per_sample(&batch_sizes, &returns_per_batch);
/// // factors = [1.0, 8.0, 32.0]
/// // - Stage 0: 1 first-stage sample per stage-0 sample  
/// // - Stage 1: 8 first-stage samples per stage-1 sample (due to batching)
/// // - Stage 2: 32 first-stage samples per stage-2 sample (batching + filtering)
/// ```
///
/// # Panics
///
/// Panics if `batch_sizes` and `num_returns_per_batch` have different lengths.
pub fn calculate_input_samples_per_sample(
    batch_sizes: &[u32],
    num_returns_per_batch: &[f64],
) -> Vec<f64> {
    assert_eq!(batch_sizes.len(), num_returns_per_batch.len());
    let mut out = vec![0.0f64; batch_sizes.len()];

    // Handle empty pipeline case
    if out.is_empty() {
        return out;
    }

    // First stage: by definition, 1 first-stage sample per first-stage sample
    out[0] = 1.0;

    // For each subsequent stage, calculate the cumulative sample flow factor
    for i in 1..batch_sizes.len() {
        // Each input batch to stage i-1 contains batch_sizes[i-1] samples
        // and produces num_returns_per_batch[i-1] outputs
        // So the ratio tells us how the sample count changes between stages
        out[i] = out[i - 1] * (batch_sizes[i - 1] as f64 / num_returns_per_batch[i - 1]);
    }

    out
}

// (no local pool helpers; using PoolOfResources methods instead)

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(cpus: f32, gpus: f32) -> resources::PoolOfResources {
        resources::PoolOfResources { cpus, gpus }
    }

    fn stage(
        name: &str,
        bps: f64,
        returns: f64,
        batch: u32,
        res: resources::PoolOfResources,
        requested: Option<usize>,
    ) -> AllocationProblemStage {
        AllocationProblemStage {
            name: name.to_string(),
            batches_per_second_per_worker: bps,
            num_returns_per_batch: returns,
            stage_batch_size: batch,
            resources_per_worker: res,
            requested_num_workers: requested,
        }
    }

    fn assert_allocation_result(
        result: &AllocationResult,
        expected_workers: &[usize],
        expected_throughput: Option<f64>,
        tolerance: f64,
    ) {
        assert_eq!(result.stages.len(), expected_workers.len());
        for (s, &e) in result.stages.iter().zip(expected_workers.iter()) {
            assert_eq!(s.num_workers, e);
        }
        if let Some(t) = expected_throughput {
            assert!((result.throughput - t).abs() < tolerance);
        }
    }

    #[test]
    fn test_single_stage_allocation() {
        // Test basic allocation with a single stage
        // Expected: All available resources go to the single stage
        let problem = AllocationProblem {
            stages: vec![stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None)],
            cluster_resources: pool(5.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        // With 5 CPUs available and 1 CPU per worker, we should get 5 workers
        // Throughput = 5 workers * 1 batch/sec/worker * 1 return/batch * 1 item/batch = 5.0
        assert_allocation_result(&result, &[5], Some(5.0), 1e-6);
    }

    #[test]
    fn test_two_stage_equal_speed() {
        // Test allocation with two stages of equal processing speed
        // Expected: Resources split evenly to balance throughput
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 1.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(6.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        // With equal speeds, resources split evenly: 3 workers each
        // Bottleneck throughput = min(3, 3) = 3.0
        assert_allocation_result(&result, &[3, 3], Some(3.0), 1e-6);
    }

    #[test]
    fn test_two_stage_different_speed() {
        // Test allocation with stages of different processing speeds
        // Expected: Slower stage gets more workers to balance throughput
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None), // Slower: 1 batch/sec/worker
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), None), // Faster: 2 batch/sec/worker
            ],
            cluster_resources: pool(6.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        // Stage A needs more workers since it's slower: 4 workers @ 1.0 = 4.0 throughput
        // Stage B needs fewer workers since it's faster: 2 workers @ 2.0 = 4.0 throughput
        assert_allocation_result(&result, &[4, 2], Some(4.0), 1e-6);
    }

    #[test]
    fn test_three_stage_multiple_resources() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(0.5, 1.0), None),
                stage("B", 2.0, 1.0, 1, pool(1.0, 2.0), None),
                stage("C", 3.0, 1.0, 1, pool(1.5, 3.0), None),
            ],
            cluster_resources: pool(10.0, 20.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[6, 3, 2], Some(6.0), 1e-6);
    }

    #[test]
    fn test_minimum_one_worker_per_stage() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 10.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(3.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[2, 1], Some(2.0), 1e-6);
    }

    #[test]
    fn test_resource_limited_allocation() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 2.0), None),
                stage("B", 1.0, 1.0, 1, pool(2.0, 1.0), None),
            ],
            cluster_resources: pool(10.0, 10.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[3, 3], Some(3.0), 1e-1);
    }

    #[test]
    fn test_infeasible_problem() {
        let problem = AllocationProblem {
            stages: vec![stage("A", 1.0, 1.0, 1, pool(2.0, 0.0), None)],
            cluster_resources: pool(1.0, 0.0),
        };
        let err = solve_allocation(problem).unwrap_err();
        match err {
            AllocationError::NoFeasibleSolution => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn test_manual_and_auto_stages() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), Some(3)),
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(10.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[3, 7], Some(3.0), 1e-1);
        assert!(result.stages[0].problem.was_manually_specified());
        assert!(!result.stages[1].problem.was_manually_specified());
    }

    #[test]
    fn test_extreme_speed_differences() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 0.1, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 100.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(20.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[19, 1], Some(1.9), 1e-1);
    }

    #[test]
    fn test_one_stage_dominates_resources() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(0.1, 0.1), None),
                stage("B", 1.0, 1.0, 1, pool(10.0, 10.0), None),
            ],
            cluster_resources: pool(100.0, 100.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[9, 9], Some(9.0), 1e-1);
    }

    #[test]
    fn test_zero_speed_stage() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 0.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(100.0, 0.0),
        };
        let err = solve_allocation(problem).unwrap_err();
        match err {
            AllocationError::InvalidStage { .. } => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn test_very_small_resource_requirement() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(0.001, 0.0), None),
                stage("B", 1.0, 1.0, 1, pool(0.001, 0.0), None),
            ],
            cluster_resources: pool(1.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        println!("result: {}", result.to_debug_str());
        assert_allocation_result(&result, &[500, 500], Some(500.0), 1e-1);
    }

    #[test]
    fn test_very_large_resource_requirement() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1e6, 0.0), None),
                stage("B", 1.0, 1.0, 1, pool(1e6, 0.0), None),
            ],
            cluster_resources: pool(3e6 as f32, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[1, 1], Some(1.0), 1e-1);
    }

    #[test]
    fn test_exact_resource_match() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 2.0), None),
                stage("B", 1.0, 1.0, 1, pool(2.0, 1.0), None),
            ],
            cluster_resources: pool(9.0, 9.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[3, 3], Some(3.0), 1e-1);
    }

    #[test]
    fn test_fractional_resource_requirements() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(0.3, 0.7), None),
                stage("B", 1.0, 1.0, 1, pool(0.7, 0.3), None),
            ],
            cluster_resources: pool(10.0, 10.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[10, 10], Some(10.0), 1e-1);
    }

    #[test]
    fn test_overallocation_limited_by_single_resource() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.1), None),
                stage("B", 1.0, 1.0, 1, pool(1.0, 0.1), None),
            ],
            cluster_resources: pool(300.0, 15.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        println!("result: {}", result.to_debug_str());
        assert_allocation_result(&result, &[75, 75], Some(75.0), 1e-1);
    }

    #[test]
    fn test_single_manual_stage() {
        let problem = AllocationProblem {
            stages: vec![stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), Some(5))],
            cluster_resources: pool(10.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[5], Some(5.0), 1e-6);
        assert!(result.stages[0].problem.was_manually_specified());
    }

    #[test]
    fn test_multiple_manual_stages() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), Some(2)),
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), Some(3)),
                stage("C", 3.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(10.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[2, 3, 5], Some(2.0), 1e-1);
        assert!(result.stages[0].problem.was_manually_specified());
        assert!(result.stages[1].problem.was_manually_specified());
        assert!(!result.stages[2].problem.was_manually_specified());
    }

    #[test]
    fn test_manual_stages_exceed_resources() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(2.0, 0.0), Some(3)),
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), Some(5)),
            ],
            cluster_resources: pool(10.0, 0.0),
        };
        let err = solve_allocation(problem).unwrap_err();
        match err {
            AllocationError::NoFeasibleSolution => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn test_manual_stages_use_all_resources() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(2.0, 0.0), Some(3)),
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), Some(4)),
                stage("C", 3.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(10.0, 0.0),
        };
        let err = solve_allocation(problem).unwrap_err();
        match err {
            AllocationError::NoFeasibleSolution => {}
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn test_manual_stages_with_multiple_resources() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 2.0), Some(2)),
                stage("B", 2.0, 1.0, 1, pool(2.0, 1.0), None),
                stage("C", 3.0, 1.0, 1, pool(1.0, 1.0), None),
            ],
            cluster_resources: pool(10.0, 10.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[2, 3, 2], Some(2.0), 1e-1);
        assert!(result.stages[0].problem.was_manually_specified());
        assert!(!result.stages[1].problem.was_manually_specified());
        assert!(!result.stages[2].problem.was_manually_specified());
    }

    #[test]
    fn test_dont_overallocate() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 10.0, 1.0, 1, pool(1.0, 1.0), None),
            ],
            cluster_resources: pool(10000.0, 8.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[80, 8], Some(80.0), 1e-1);
    }

    #[test]
    fn test_dont_overallocate_multiple_stages() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("C", 3.0, 1.0, 1, pool(1.0, 1.0), None),
            ],
            cluster_resources: pool(1000.0, 10.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[30, 15, 10], Some(30.0), 1e-1);
    }

    #[test]
    fn test_dont_overallocate_limited_by_resources() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 2.0, 1.0, 1, pool(2.0, 1.0), None),
            ],
            cluster_resources: pool(60.0, 10.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[20, 10], Some(20.0), 1e-1);
    }

    #[test]
    fn test_unbalanced_batching() {
        let problem = AllocationProblem {
            stages: vec![
                stage("A", 1.0, 1.0, 1, pool(1.0, 0.0), None),
                stage("B", 1.0, 1000.0, 1, pool(1.0, 0.0), None),
                stage("C", 1.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(1000.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[1, 1, 998], None, 1e-1);
    }

    #[test]
    fn test_simple_batching() {
        let problem = AllocationProblem {
            stages: vec![
                // Stage A: 1 bps, batch size 10
                stage("A", 1.0, 1.0, 10, pool(1.0, 0.0), None),
                // Stage B: 1 bps, batch size 1
                stage("B", 1.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(1000.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[500, 500], None, 1e-1);
    }

    #[test]
    fn test_simple_batching_2() {
        let problem = AllocationProblem {
            stages: vec![
                // A: 2 bps, returns 1000, batch size 10
                stage("A", 2.0, 1000.0, 10, pool(1.0, 0.0), None),
                // B: 2 bps, returns 1, batch size 1
                stage("B", 2.0, 1.0, 1, pool(1.0, 0.0), None),
                // C: 2 bps, returns 10, batch size 1000
                stage("C", 2.0, 10.0, 1000, pool(1.0, 0.0), None),
                // D: 2 bps, returns 1, batch size 1
                stage("D", 2.0, 1.0, 1, pool(1.0, 0.0), None),
            ],
            cluster_resources: pool(1000.0, 0.0),
        };
        let result = solve_allocation(problem).expect("alloc");
        assert_allocation_result(&result, &[1, 988, 1, 10], None, 1e-1);
    }
}
