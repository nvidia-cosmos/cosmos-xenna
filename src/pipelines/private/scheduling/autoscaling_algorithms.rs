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

//! Algorithms for auto-scaling streaming pipeline workers with smart resource packing.
//!
//! This module implements an adaptive worker allocation system for streaming pipelines that optimizes
//! resource utilization while maintaining performance. The system handles multi-stage pipelines where
//! each stage has specific resource requirements and throughput characteristics.
//!
//! Key Features:
//! - Adaptive worker allocation based on real-time performance measurements
//! - Smart resource packing to minimize fragmentation
//! - Multi-stage pipeline support with different resource requirements per stage
//! - Automatic slot calculation to prevent work starvation
//! - Balanced worker distribution across stages based on throughput
//!
//! The main algorithm combines:
//! 1. Naive allocation to determine target worker counts
//! 2. Fragmentation-aware worker placement
//! 3. Priority-based stage scaling
//! 4. Performance-based slot adjustment
//!
//! The system continuously monitors stage performance and adjusts worker counts and slot allocations
//! to maintain optimal throughput while respecting resource constraints.

use std::collections::{HashMap, VecDeque};

use log::{debug, trace};
use pyo3::prelude::*;

use crate::utils::module_builders::ImportablePyModuleBuilder;

use super::{
    allocator::WorkerAllocator, data_structures as ds, fragmentation_allocation_algorithms as frag,
    resources as rds,
};

// --------------------
// Estimators
// --------------------

/// Generates unique worker IDs for new worker allocations.
///
/// This simple counter-based ID generator ensures each worker gets a unique string identifier.
/// The IDs are generated sequentially starting from 0.
#[derive(Debug, Default, Clone)]
#[pyclass(get_all, set_all)]
pub struct WorkerIdFactory {
    count: usize,
}

#[pymethods]
impl WorkerIdFactory {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a new unique worker ID.
    ///
    /// # Returns
    /// A unique string ID representing the worker
    fn make_new_id(&mut self) -> String {
        let id = self.count.to_string();
        self.count += 1;
        id
    }
}

/// Maintains rolling window estimates of the number of returns for pipeline stages.
#[derive(Debug, Clone)]
struct NumberOfReturnsEstimator {
    window_duration: f64,
    min_num_events: usize,
    events: VecDeque<(f64, u32)>,
}

impl NumberOfReturnsEstimator {
    fn new(window_duration: f64, min_num_events: usize) -> Self {
        Self {
            window_duration,
            min_num_events,
            events: VecDeque::new(),
        }
    }

    /// Removes old events based on time window and optionally min_num_events.
    fn remove_old(&mut self, now: f64) {
        while let Some(&(t, _)) = self.events.front() {
            let is_too_old = now - t > self.window_duration;
            let can_remove_based_on_count = self.events.len() > self.min_num_events;
            if is_too_old && can_remove_based_on_count {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Updates the estimator with a new duration and timestamp.
    ///
    /// Removes old entries based on the configured time window and `min_num_events` policy.
    ///
    /// # Arguments
    /// * `num_returns` - Number of returns for the event.
    /// * `current_time` - Optional timestamp for the event; defaults to current time.
    fn update(&mut self, num_returns: u32, current_time: f64) {
        self.events.push_back((current_time, num_returns));
        self.remove_old(current_time);
    }

    /// Calculates and returns the average number of returns, or None if insufficient data.
    ///
    /// Similar to `get_average_num_returns`, but returns None if no events are stored.
    ///
    /// # Arguments
    /// * `now` - Timestamp to use as 'now'.
    ///
    /// # Returns
    /// The average number of returns, or None if no events are stored.
    fn maybe_get_average_num_returns(&mut self, now: f64) -> Option<f64> {
        self.remove_old(now);
        if self.events.is_empty() {
            return None;
        }
        let sum: f64 = self.events.iter().map(|(_, n)| *n as f64).sum();
        Some(sum / (self.events.len() as f64))
    }
}

#[derive(Debug, Clone)]
struct RateEstimatorDuration {
    window_duration: f64,
    min_num_events: usize,
    events: VecDeque<(f64, f64)>, // (start_time, duration)
}

impl RateEstimatorDuration {
    fn new(window_duration: f64, min_num_events: usize) -> Self {
        Self {
            window_duration,
            min_num_events,
            events: VecDeque::new(),
        }
    }

    fn remove_old(&mut self, now: f64) {
        while let Some(&(t, _)) = self.events.front() {
            let is_too_old = now - t > self.window_duration;
            let can_remove_based_on_count = self.events.len() > self.min_num_events;
            if is_too_old && can_remove_based_on_count {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    fn update(&mut self, duration: f64, start_time: f64) {
        self.events.push_back((start_time, duration));
        self.remove_old(start_time);
    }

    fn maybe_get_rate(&mut self, now: f64) -> Option<f64> {
        self.remove_old(now);
        if self.events.is_empty() {
            return None;
        }
        let mean_duration =
            self.events.iter().map(|(_, d)| *d).sum::<f64>() / (self.events.len() as f64);
        if mean_duration <= 0.0 {
            None
        } else {
            Some(1.0 / mean_duration)
        }
    }
}

#[derive(Debug, Clone, Default)]
#[pyclass(get_all, set_all)]
pub struct Estimate {
    pub batches_per_second_per_worker: Option<f64>,
    pub num_returns_per_batch: Option<f64>,
}

#[pymethods]
impl Estimate {
    #[new]
    fn new(batches_per_second_per_worker: Option<f64>, num_returns_per_batch: Option<f64>) -> Self {
        Self {
            batches_per_second_per_worker,
            num_returns_per_batch,
        }
    }
}

#[derive(Debug, Clone, Default)]
#[pyclass(get_all, set_all)]
pub struct Estimates {
    pub stages: Vec<Estimate>,
}

#[pymethods]
impl Estimates {
    #[new]
    fn new(stages: Vec<Estimate>) -> Self {
        Self { stages }
    }
}

/// Maintains rolling window estimates of processing speeds for pipeline stages.
///
/// This class tracks the processing rate (tasks/second) for each stage in the pipeline
/// using a sliding time window. It maintains both current and historical speed estimates
/// to ensure stable scaling decisions.
///
/// # Arguments
/// * `num_stages` - Number of pipeline stages to track
/// * `window_duration` - Duration of the sliding window in seconds (default: 60 * 3)
#[derive(Debug, Clone)]
struct SpeedAndNumberOfReturnsEstimator {
    speed_estimators: Vec<RateEstimatorDuration>,
    num_returns_estimators: Vec<NumberOfReturnsEstimator>,
    last_valid_speeds: Vec<Option<f64>>,
    last_valid_num_returns: Vec<Option<f64>>,
}

impl SpeedAndNumberOfReturnsEstimator {
    fn new(num_stages: usize, window_duration: f64, min_num_events: usize) -> Self {
        Self {
            speed_estimators: (0..num_stages)
                .map(|_| RateEstimatorDuration::new(window_duration, min_num_events))
                .collect(),
            num_returns_estimators: (0..num_stages)
                .map(|_| NumberOfReturnsEstimator::new(window_duration, min_num_events))
                .collect(),
            last_valid_speeds: vec![None; num_stages],
            last_valid_num_returns: vec![None; num_stages],
        }
    }

    /// Update speed estimates with new task timing measurements.
    ///
    /// # Arguments
    /// * `measurements` - Collection of task timing measurements for each stage
    fn update_with_measurements(&mut self, measurements: &ds::Measurements) {
        for (stage_measurements, speed_estimator, num_returns_estimator) in itertools::izip!(
            &measurements.stages,
            &mut self.speed_estimators,
            &mut self.num_returns_estimators,
        ) {
            for m in &stage_measurements.task_measurements {
                speed_estimator.update(m.duration(), m.start_time);
                num_returns_estimator.update(m.num_returns, m.start_time);
            }
        }
    }

    fn get_estimates(&mut self, now: f64) -> Estimates {
        let maybe_speeds: Vec<Option<f64>> = self
            .speed_estimators
            .iter_mut()
            .map(|s| s.maybe_get_rate(now))
            .collect();
        let maybe_returns: Vec<Option<f64>> = self
            .num_returns_estimators
            .iter_mut()
            .map(|e| e.maybe_get_average_num_returns(now))
            .collect();
        for (i, (ms, mr)) in maybe_speeds
            .into_iter()
            .zip(maybe_returns.into_iter())
            .enumerate()
        {
            if ms.is_some() {
                self.last_valid_speeds[i] = ms;
            }
            if mr.is_some() {
                self.last_valid_num_returns[i] = mr;
            }
        }
        Estimates {
            stages: self
                .last_valid_speeds
                .iter()
                .zip(self.last_valid_num_returns.iter())
                .map(|(s, r)| Estimate {
                    batches_per_second_per_worker: *s,
                    num_returns_per_batch: *r,
                })
                .collect(),
        }
    }

    fn get_last_valid_estimates(&mut self, now: f64) -> Estimates {
        // Refresh last valid values
        let _ = self.get_estimates(now);
        Estimates {
            stages: self
                .last_valid_speeds
                .iter()
                .zip(self.last_valid_num_returns.iter())
                .map(|(s, r)| Estimate {
                    batches_per_second_per_worker: *s,
                    num_returns_per_batch: *r,
                })
                .collect(),
        }
    }
}

// --------------------
// Helper structures and functions
// --------------------

#[derive(Debug, Clone)]
struct StageInternal {
    name: String,
    current_workers: usize,
    speed_per_worker: Option<f64>,
    stage_batch_size: usize,
    num_returns_per_batch: Option<f64>,
    num_input_samples_per_sample: Option<f64>,
    shape: rds::WorkerShape,
    requested_num_workers: Option<usize>,
}

impl StageInternal {
    fn throughput(&self) -> f64 {
        let s = self.speed_per_worker.expect("speed_per_worker is None");
        let k = self
            .num_input_samples_per_sample
            .expect("num_input_samples_per_sample is None");
        (self.current_workers as f64) * s * k * (self.stage_batch_size as f64)
    }

    fn throughput_if_one_removed(&self) -> f64 {
        let s = self.speed_per_worker.expect("speed_per_worker is None");
        let k = self
            .num_input_samples_per_sample
            .expect("num_input_samples_per_sample is None");
        ((self.current_workers.saturating_sub(1)) as f64) * s * k * (self.stage_batch_size as f64)
    }

    fn throughput_if_one_added(&self) -> f64 {
        let s = self.speed_per_worker.expect("speed_per_worker is None");
        let k = self
            .num_input_samples_per_sample
            .expect("num_input_samples_per_sample is None");
        ((self.current_workers + 1) as f64) * s * k * (self.stage_batch_size as f64)
    }
}

fn make_workers_from_problem_state(state: &ds::ProblemState) -> Vec<rds::Worker> {
    let mut out = Vec::new();
    for stage in &state.stages {
        for w in &stage.workers {
            out.push(w.to_worker(&stage.stage_name));
        }
    }
    out
}

fn calculate_input_samples_per_sample(
    stage_batch_sizes: &[usize],
    num_returns_per_batch: &[f64],
) -> Vec<f64> {
    let mut out = Vec::with_capacity(stage_batch_sizes.len());
    let mut acc = 1.0f64;
    for i in 0..stage_batch_sizes.len() {
        if i == 0 {
            out.push(1.0);
        } else {
            let prev = num_returns_per_batch[i - 1];
            let denom = stage_batch_sizes[i - 1] as f64;
            if denom > 0.0 {
                acc *= prev / denom;
            }
            out.push(acc);
        }
    }
    out
}

fn calculate_num_slots_per_worker_for_all_stages(
    num_workers: &[usize],
    current_slots: &[usize],
    last_valid_speeds: &[Option<f64>],
    min_scheduling_algorithm_rate_hz: f64,
    min_slots: usize,
) -> Vec<usize> {
    assert_eq!(num_workers.len(), current_slots.len());
    assert_eq!(num_workers.len(), last_valid_speeds.len());

    if last_valid_speeds.iter().any(|x| x.is_none()) {
        return current_slots.to_vec();
    }
    if num_workers.is_empty() {
        return Vec::new();
    }

    let stage_speeds: Vec<f64> = num_workers
        .iter()
        .zip(last_valid_speeds.iter())
        .map(|(nw, s)| (*nw as f64) * s.unwrap())
        .collect();
    let pipeline_speed = stage_speeds
        .into_iter()
        .fold(f64::INFINITY, |acc, x| acc.min(x))
        .min(0.0_f64.max(f64::INFINITY));
    let pipeline_speed = if pipeline_speed.is_finite() {
        pipeline_speed
    } else {
        0.0
    };
    let num_tasks_per_slowest_loop = if min_scheduling_algorithm_rate_hz > 0.0 {
        pipeline_speed * (1.0 / min_scheduling_algorithm_rate_hz)
    } else {
        0.0
    };

    let mut out: Vec<usize> = Vec::with_capacity(num_workers.len());
    for (nw, current) in num_workers.iter().zip(current_slots.iter()) {
        let total_slots_needed = num_tasks_per_slowest_loop;
        let slots_needed_per_worker = if *nw > 0 {
            ((2.0 * total_slots_needed) / (*nw as f64)).ceil() as usize
        } else {
            *current
        };
        out.push(slots_needed_per_worker.max(*current).max(min_slots));
    }
    out
}

fn make_workload_from_state(state: &ds::ProblemState, problem: &ds::Problem) -> frag::Workload {
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
                    shape: s.worker_shape.clone().into(),
                })
                .collect(),
        };
    }

    frag::Workload {
        stages: problem
            .stages
            .iter()
            .zip(per_stage_requested.into_iter())
            .map(|(s, req)| frag::Stage {
                frequency: (req as f32) / (total_requested as f32),
                shape: s.worker_shape.clone().into(),
            })
            .collect(),
    }
}

// --------------------
// Main algorithm
// --------------------

fn format_summary(
    workers_to_add: &HashMap<String, Vec<rds::Worker>>,
    workers_to_remove: &HashMap<String, Vec<rds::Worker>>,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Additions summary
    let add_total: usize = workers_to_add.values().map(|v| v.len()).sum();
    if add_total > 0 {
        let mut stage_names: Vec<&String> = workers_to_add
            .iter()
            .filter(|(_, ws)| !ws.is_empty())
            .map(|(k, _)| k)
            .collect();
        stage_names.sort();
        let mut details: Vec<String> = Vec::new();
        for name in stage_names {
            let n = workers_to_add.get(name).map(|v| v.len()).unwrap_or(0);
            if n > 0 {
                details.push(format!("{}={}", name, n));
            }
        }
        if details.is_empty() {
            parts.push(format!("\nadd[total={}]", add_total));
        } else {
            parts.push(format!(
                "\nadd[total={}]: {}",
                add_total,
                details.join(", ")
            ));
        }
    } else {
        parts.push("\nadd[total=0]".to_string());
    }

    // Removals summary
    let remove_total: usize = workers_to_remove.values().map(|v| v.len()).sum();
    if remove_total > 0 {
        let mut stage_names: Vec<&String> = workers_to_remove
            .iter()
            .filter(|(_, ws)| !ws.is_empty())
            .map(|(k, _)| k)
            .collect();
        stage_names.sort();
        let mut details: Vec<String> = Vec::new();
        for name in stage_names {
            let n = workers_to_remove.get(name).map(|v| v.len()).unwrap_or(0);
            if n > 0 {
                details.push(format!("{}={}", name, n));
            }
        }
        if details.is_empty() {
            parts.push(format!("\nremove[total={}]", remove_total));
        } else {
            parts.push(format!(
                "\nremove[total={}]: {}",
                remove_total,
                details.join(", ")
            ));
        }
    } else {
        parts.push("\nremove[total=0]".to_string());
    }

    parts.join("; ")
}

// --------------------
// Metrics helpers
// --------------------

// Lightweight cumulative metrics tracker for add/remove operations within a single autoscale run.
struct AutoscalerOpMetrics {
    add_calls: usize,
    add_total_s: f64,
    remove_calls: usize,
    remove_total_s: f64,
}

impl AutoscalerOpMetrics {
    fn new() -> Self {
        Self {
            add_calls: 0,
            add_total_s: 0.0,
            remove_calls: 0,
            remove_total_s: 0.0,
        }
    }

    fn record_add(&mut self, seconds: f64) {
        self.add_calls += 1;
        self.add_total_s += seconds;
    }

    fn record_remove(&mut self, seconds: f64) {
        self.remove_calls += 1;
        self.remove_total_s += seconds;
    }

    fn cumulative_str(&self) -> String {
        let add_avg = if self.add_calls > 0 {
            self.add_total_s / (self.add_calls as f64)
        } else {
            0.0
        };
        let remove_avg = if self.remove_calls > 0 {
            self.remove_total_s / (self.remove_calls as f64)
        } else {
            0.0
        };
        format!(
            "metrics[add_calls={}, add_avg_s={:.6}, remove_calls={}, remove_avg_s={:.6}]",
            self.add_calls, add_avg, self.remove_calls, remove_avg
        )
    }
}
/// Runs the fragmentation-based autoscaling algorithm and returns a scheduling solution
/// describing new workers to add/delete per stage and the desired slots-per-worker.
///
/// Overview
/// - Builds an internal view of the cluster and current workers, then iteratively proposes
///   allocations/deallocations that improve pipeline throughput while respecting resource
///   fragmentation on heterogeneous nodes/GPUs.
/// - Stages with manually requested worker counts are satisfied first and treated as fixed.
/// - Ensures every non-manual stage has at least one worker (if resources allow).
/// - Balances throughput by repeatedly helping the slowest active stage, borrowing workers
///   from faster stages when it does not violate their own throughput constraints.
/// - Optionally over-allocates slower stages up to a multiplicative target relative to the
///   current slowest stage to create headroom (burst tolerance).
///
/// Inputs
/// - `problem`: Static pipeline description, including cluster resources, per-stage shapes
///   and optional `requested_num_workers`/`over_provision_factor`.
/// - `state`: Current dynamic state: existing workers per stage and slots per worker.
/// - `estimates`: Latest speed and return-count estimates per stage. Missing speeds are
///   treated as 1.0; missing or non-positive returns-per-batch default to `stage_batch_size`.
/// - `overallocation_target`: If > 1.0, slower stages can be allocated additional workers as
///   long as their throughput with one more worker does not exceed
///   `base_min_throughput * overallocation_target`.
/// - `worker_id_factory`: Mutable ID generator used to assign stable IDs to newly proposed
///   workers in the returned solution.
///
/// Algorithm phases
/// 1) Satisfy manual requests: For stages with `requested_num_workers`, add/delete until the
///    requested count is met. This can panic if resources cannot satisfy a hard manual request.
/// 2) Ensure minimum one worker: For non-manual stages, allocate a worker if none exists.
///    This can panic if the cluster cannot fit even a single required worker shape.
/// 3) Balance minimum throughput: Among active stages (non-manual with valid speeds), repeatedly
///    pick the slowest stage and try to allocate one worker. If allocation fails due to
///    fragmentation, attempt to remove one worker from a donor stage that can afford it (keeps
///    donor throughput above the current min). Donor selection prefers the stage whose
///    throughput remains highest after removal. Stops when no safe donor remains.
/// 4) Over-allocation: Using the minimum throughput at the start of this phase as a baseline,
///    repeatedly try to add workers to stages whose next-worker throughput is still below
///    `base_min * overallocation_target`, preferring slower stages first. Stops when no more
///    allocations fit or all candidates exceed the target.
///
/// Implementation notes
/// - Uses a local `WorkerAllocator` and fragmentation-aware search
///   (`find_best_allocation_using_fragmentation_gradient_descent` and
///   `find_worker_to_delete_using_fragmentation_gradient_descent`) so that add/delete decisions
///   account for heterogeneous resources and packing constraints.
/// - Never reduces any active, non-manual stage below one worker.
/// - Builds `slots_per_worker` for the output upfront from current measurements and last known
///   speeds via `calculate_num_slots_per_worker_for_all_stages`.
/// - Side-effect free: Only the returned `ds::Solution` communicates the desired changes.
///
/// Panics
/// - If a stage has `requested_num_workers` and the allocator cannot place the required number
///   of workers.
/// - If a non-manual stage has zero workers and the allocator cannot place at least one.
///
/// Returns
/// - `ds::Solution` where for each stage:
///   - `new_workers`: proposed workers (with concrete resource assignments) to create.
///   - `deleted_workers`: existing workers to remove.
///   - `slots_per_worker`: desired concurrency per worker as computed from current state.
///
/// Complexity
/// - Heuristic/iterative; each add/remove attempt runs a fragmentation-aware search. In
///   practice the number of iterations is bounded by the number of stages and cluster capacity.
///
/// Examples
/// ```rust
/// // Rust (internal usage)
/// let solution = run_fragmentation_autoscaler(
///     &problem,
///     &state,
///     &estimates,
///     1.5, // allow 50% headroom for slower stages
///     &mut worker_id_factory,
/// );
/// // Apply `solution` in the scheduler layer.
/// ```
///
/// ```python
/// # Python (via PyO3 export)
/// from cosmos_xenna import run_fragmentation_autoscaler
/// sol = run_fragmentation_autoscaler(problem, state, estimates, 1.5, worker_id_factory)
/// # `sol.stages[i].new_workers` and `.deleted_workers` describe desired changes.
/// ```
#[pyfunction]
pub fn run_fragmentation_autoscaler(
    problem: &ds::Problem,
    state: &ds::ProblemState,
    estimates: &Estimates,
    overallocation_target: f64,
    worker_id_factory: &mut WorkerIdFactory,
) -> ds::Solution {
    // Overall timer
    let overall_start = std::time::Instant::now();
    // Build an allocator view seeded with current workers so subsequent add/remove
    // operations respect existing placement and fragmentation constraints.
    let mut allocator = WorkerAllocator::new(
        problem.cluster_resources.clone(),
        Some(make_workers_from_problem_state(state)),
    )
    .expect("allocator init");
    // Prepare an internal per-stage snapshot that merges static shape/problem data,
    // dynamic state (current workers), and the latest performance/returns estimates.
    let mut stages: Vec<StageInternal> = Vec::new();
    for (stage_problem, stage_state, stage_estimate) in
        itertools::izip!(&problem.stages, &state.stages, &estimates.stages)
    {
        stages.push(StageInternal {
            name: stage_problem.name.clone(),
            current_workers: stage_state.num_workers(),
            speed_per_worker: stage_estimate.batches_per_second_per_worker,
            stage_batch_size: stage_problem.stage_batch_size,
            num_returns_per_batch: stage_estimate.num_returns_per_batch,
            num_input_samples_per_sample: None,
            shape: stage_problem.worker_shape.clone().into(),
            requested_num_workers: stage_problem.requested_num_workers,
        });
    }

    // Guardrails: if measurements are missing or invalid, fall back to conservative
    // defaults (speed = 1.0) to keep the algorithm progressing instead of stalling.
    for stage in &mut stages {
        if stage.speed_per_worker.is_none() {
            stage.speed_per_worker = Some(1.0);
        }
        if stage
            .num_returns_per_batch
            .map(|x| x <= 0.0)
            .unwrap_or(true)
        {
            stage.num_returns_per_batch = Some(stage.stage_batch_size as f64);
        }
    }

    // Estimate how many samples are required for one output sample at each stage. We need this to balance throughputs across stages.
    let stage_batch_sizes: Vec<usize> = problem.stages.iter().map(|s| s.stage_batch_size).collect();
    let num_returns_per_batch: Vec<f64> = stages
        .iter()
        .map(|s| s.num_returns_per_batch.unwrap())
        .collect();
    let input_samples_per_sample =
        calculate_input_samples_per_sample(&stage_batch_sizes, &num_returns_per_batch);
    for (stage, val) in stages.iter_mut().zip(input_samples_per_sample.into_iter()) {
        stage.num_input_samples_per_sample = Some(val);
    }

    // Derive desired slots-per-worker per stage using the most recent speeds and
    // current worker/slot counts. This is applied to the output regardless of
    // whether we add or remove workers.
    let num_workers_now: Vec<usize> = state.stages.iter().map(|s| s.num_workers()).collect();
    let current_slots: Vec<usize> = state.stages.iter().map(|s| s.slots_per_worker).collect();
    let last_valid_speeds: Vec<Option<f64>> = estimates
        .stages
        .iter()
        .map(|e| e.batches_per_second_per_worker)
        .collect();
    let num_slots_per_stage = calculate_num_slots_per_worker_for_all_stages(
        &num_workers_now,
        &current_slots,
        &last_valid_speeds,
        1.0,
        2,
    );

    let mut workers_to_add: HashMap<String, Vec<rds::Worker>> = HashMap::new();
    let mut workers_to_remove: HashMap<String, Vec<rds::Worker>> = HashMap::new();
    // Trade-off constant for the allocator search: encourages solutions that
    // reuse existing packing by treating reuse as worth this much fragmentation.
    let worker_reuse_fragmentation_equivalent: f32 = 10.0;

    // Cumulative operation metrics for this autoscaler run
    let mut metrics = AutoscalerOpMetrics::new();

    // Snapshot current workload characteristics to drive donor selection and deletions.
    let workload_estimate = make_workload_from_state(state, problem);

    // Attempt to add a worker to a using fragmentation-aware search and record
    // the side effects locally if successful.
    fn add_worker_fn(
        stage: &mut StageInternal,
        allocator: &mut WorkerAllocator,
        workload_estimate: &frag::Workload,
        workers_to_add_map: &mut HashMap<String, Vec<rds::Worker>>,
        worker_id_factory: &mut WorkerIdFactory,
        worker_reuse_fragmentation_equivalent: f32,
        metrics: &mut AutoscalerOpMetrics,
    ) -> bool {
        let t0 = std::time::Instant::now();
        let allocation = frag::find_best_allocation_using_fragmentation_gradient_descent(
            &frag::ClusterResourceHelpers::make_from_allocator(allocator),
            workload_estimate,
            &stage.shape,
            None,
            worker_reuse_fragmentation_equivalent,
        );
        let mut ok = false;
        if allocation.did_allocate {
            if let Some(resources) = allocation.resources.clone() {
                let worker = rds::Worker::new(
                    worker_id_factory.make_new_id(),
                    stage.name.clone(),
                    resources,
                );
                allocator.add_worker(worker.clone()).expect("add worker");
                workers_to_add_map
                    .entry(stage.name.clone())
                    .or_default()
                    .push(worker);
                stage.current_workers += 1;
                ok = true;
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        metrics.record_add(elapsed);
        ok
    }

    // Helper to remove the worker whose deletion least harms packing/throughput according to
    // the fragmentation-aware heuristic; update local bookkeeping.
    fn remove_best_worker_fn(
        stage: &mut StageInternal,
        allocator: &mut WorkerAllocator,
        workers_to_remove_map: &mut HashMap<String, Vec<rds::Worker>>,
        workload_estimate: &frag::Workload,
        metrics: &mut AutoscalerOpMetrics,
    ) {
        let t0 = std::time::Instant::now();
        let worker = frag::find_worker_to_delete_using_fragmentation_gradient_descent(
            &frag::ClusterResourceHelpers::make_from_allocator(allocator),
            &workload_estimate,
            &allocator.get_workers_in_stage(&stage.name),
        );
        allocator.delete_worker(&worker.id).expect("delete worker");
        workers_to_remove_map
            .entry(stage.name.clone())
            .or_default()
            .push(worker.clone());
        stage.current_workers = stage.current_workers.saturating_sub(1);
        let elapsed = t0.elapsed().as_secs_f64();
        metrics.record_remove(elapsed);
    }

    // Phase 1: Honor hard manual requests. These are treated as fixed constraints;
    // if they cannot be satisfied we intentionally panic to surface misconfiguration
    // or insufficient cluster capacity for the requested shape/count.
    log::debug!("Phase 1: satisfy manually requested counts");
    let phase1_start = std::time::Instant::now();
    for stage in &mut stages {
        if let Some(req) = stage.requested_num_workers {
            trace!(
                "Phase 1: satisfy manually requested counts for stage {}. Requested={}, Current={}",
                stage.name, req, stage.current_workers
            );
            while stage.current_workers < req {
                if !add_worker_fn(
                    stage,
                    &mut allocator,
                    &workload_estimate,
                    &mut workers_to_add,
                    worker_id_factory,
                    worker_reuse_fragmentation_equivalent,
                    &mut metrics,
                ) {
                    panic!(
                        "Unable to allocate requested workers for stage {}. Requested={}, Current={}",
                        stage.name, req, stage.current_workers
                    );
                }
            }
            while stage.current_workers > req {
                remove_best_worker_fn(
                    stage,
                    &mut allocator,
                    &mut workers_to_remove,
                    &workload_estimate,
                    &mut metrics,
                );
            }
        }
    }
    let phase1_duration_s = phase1_start.elapsed().as_secs_f64();
    log::debug!(
        "End of phase 1: {}\n{}\nphase_duration_s={:.6}",
        format_summary(&workers_to_add, &workers_to_remove),
        metrics.cumulative_str(),
        phase1_duration_s
    );

    // Phase 2: Ensure forward progress by guaranteeing at least one worker on
    // every non-manual stage. This may also panic if a single worker cannot fit.
    let phase2_start = std::time::Instant::now();
    for stage in &mut stages {
        if stage.requested_num_workers.is_some() {
            continue;
        }
        if stage.current_workers < 1 {
            if !add_worker_fn(
                stage,
                &mut allocator,
                &workload_estimate,
                &mut workers_to_add,
                worker_id_factory,
                worker_reuse_fragmentation_equivalent,
                &mut metrics,
            ) {
                panic!("Unable to allocate minimum worker for stage {}", stage.name);
            }
        }
    }
    let phase2_duration_s = phase2_start.elapsed().as_secs_f64();
    log::debug!(
        "End of phase 2: {}\n{}\nphase_duration_s={:.6}",
        format_summary(&workers_to_add, &workers_to_remove),
        metrics.cumulative_str(),
        phase2_duration_s
    );

    // Phase 3: Max-min balancing across active stages. Repeatedly try to help the
    // currently slowest stage; if a direct allocation fails, borrow from a donor
    // stage whose throughput remains above the current minimum after removal.
    let mut active: Vec<StageInternal> = stages
        .iter()
        .filter(|s| s.requested_num_workers.is_none() && s.speed_per_worker.is_some())
        .cloned()
        .collect();

    let phase3_start = std::time::Instant::now();
    if !active.is_empty() {
        loop {
            // Identify the current slowest stage by estimated throughput.
            let (min_idx, _) = active
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.throughput().partial_cmp(&b.throughput()).unwrap())
                .unwrap();
            let min_throughput = active[min_idx].throughput();

            // First, try to allocate one more worker to the slowest stage directly.
            if add_worker_fn(
                &mut active[min_idx],
                &mut allocator,
                &workload_estimate,
                &mut workers_to_add,
                worker_id_factory,
                worker_reuse_fragmentation_equivalent,
                &mut metrics,
            ) {
                continue;
            }

            // Otherwise, look for donor stages that can afford to give up one worker
            // while staying above the current minimum throughput.
            let mut removable: Vec<usize> = active
                .iter()
                .enumerate()
                .filter(|(i, s)| {
                    *i != min_idx
                        && s.current_workers > 1
                        && s.throughput_if_one_removed() > min_throughput
                })
                .map(|(i, _)| i)
                .collect();
            if removable.is_empty() {
                break;
            }
            // Prefer the donor that remains strongest after removal to preserve
            // headroom on faster stages.
            removable.sort_by(|&ia, &ib| {
                active[ia]
                    .throughput_if_one_removed()
                    .partial_cmp(&active[ib].throughput_if_one_removed())
                    .unwrap()
                    .reverse()
            });
            let donor_idx = removable[0];
            remove_best_worker_fn(
                &mut active[donor_idx],
                &mut allocator,
                &mut workers_to_remove,
                &workload_estimate,
                &mut metrics,
            );
        }
    }
    let phase3_duration_s = phase3_start.elapsed().as_secs_f64();
    log::debug!(
        "End of phase 3: {}\n{}\nphase_duration_s={:.6}",
        format_summary(&workers_to_add, &workers_to_remove),
        metrics.cumulative_str(),
        phase3_duration_s
    );

    // Phase 4: Controlled over-allocation to create headroom. Use the minimum
    // throughput at the start of the phase as a baseline and keep allocating to
    // stages whose next-worker throughput would still be below
    // `base_min * overallocation_target`.
    let phase4_start = std::time::Instant::now();
    if !active.is_empty() {
        // Find current slowest throughput as base
        let base_min = active
            .iter()
            .map(|s| s.throughput())
            .fold(f64::INFINITY, |acc, x| acc.min(x));
        let mut active_indices: Vec<usize> = (0..active.len()).collect();
        loop {
            let candidates: Vec<usize> = active_indices
                .iter()
                .copied()
                .filter(|&i| {
                    active[i].throughput_if_one_added() <= base_min * overallocation_target
                })
                .collect();
            if candidates.is_empty() {
                break;
            }

            // Prefer slower stages first to raise the minimum as quickly as possible.
            let mut sorted = candidates.clone();
            sorted.sort_by(|&ia, &ib| {
                active[ia]
                    .throughput()
                    .partial_cmp(&active[ib].throughput())
                    .unwrap()
            });

            let mut allocated_any = false;
            let mut to_remove: Vec<usize> = Vec::new();
            for idx in sorted {
                if add_worker_fn(
                    &mut active[idx],
                    &mut allocator,
                    &workload_estimate,
                    &mut workers_to_add,
                    worker_id_factory,
                    worker_reuse_fragmentation_equivalent,
                    &mut metrics,
                ) {
                    allocated_any = true;
                    break;
                } else {
                    to_remove.push(idx);
                }
            }
            if !allocated_any {
                break;
            }
            active_indices.retain(|i| !to_remove.contains(i));
        }
    }
    let phase4_duration_s = phase4_start.elapsed().as_secs_f64();
    log::debug!(
        "End of phase 4: {}\n{}\nphase_duration_s={:.6}",
        format_summary(&workers_to_add, &workers_to_remove),
        metrics.cumulative_str(),
        phase4_duration_s
    );

    // Build the final Solution: slots are set first; adds/deletes are populated
    // per stage from the allocator-local bookkeeping accumulated above.
    let mut out = ds::Solution::default();
    out.stages = num_slots_per_stage
        .into_iter()
        .map(|slots| ds::StageSolution::new(slots))
        .collect();

    for (idx, stage_problem) in problem.stages.iter().enumerate() {
        if let Some(stage) = out.stages.get_mut(idx) {
            let added = workers_to_add
                .remove(&stage_problem.name)
                .unwrap_or_default();
            let deleted = workers_to_remove
                .remove(&stage_problem.name)
                .unwrap_or_default();
            stage.new_workers = added
                .iter()
                .map(|w| ds::ProblemWorkerState::make_from_worker_state(w))
                .collect();
            stage.deleted_workers = deleted
                .iter()
                .map(|w| ds::ProblemWorkerState::make_from_worker_state(w))
                .collect();
        }
    }

    // Overall duration
    let overall_duration_s = overall_start.elapsed().as_secs_f64();
    log::debug!("Autoscaler total duration_s={:.6}", overall_duration_s);
    out
}

// --------------------
// Public interface implementation
// --------------------

#[pyclass]
pub struct FragmentationBasedAutoscaler {
    worker_id_factory: WorkerIdFactory,
    speed_estimation_window_duration_s: f64,
    speed_calculator: Option<SpeedAndNumberOfReturnsEstimator>,
    problem: Option<ds::Problem>,
}

impl Default for FragmentationBasedAutoscaler {
    fn default() -> Self {
        Self {
            worker_id_factory: WorkerIdFactory::new(),
            speed_estimation_window_duration_s: 60.0 * 3.0,
            speed_calculator: None,
            problem: None,
        }
    }
}

#[pymethods]
impl FragmentationBasedAutoscaler {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn name(&self) -> &str {
        "fragmentation_based_autoscaler"
    }

    fn setup(&mut self, problem: &ds::Problem) {
        self.problem = Some(problem.clone());
        self.speed_calculator = Some(SpeedAndNumberOfReturnsEstimator::new(
            problem.stages.len(),
            self.speed_estimation_window_duration_s,
            5,
        ));
    }

    fn update_with_measurements(&mut self, _time: f64, measurements: &ds::Measurements) {
        if let Some(calc) = &mut self.speed_calculator {
            calc.update_with_measurements(measurements);
        }
    }

    fn autoscale(&mut self, current_time: f64, state: &ds::ProblemState) -> ds::Solution {
        let problem = self
            .problem
            .as_ref()
            .expect("autoscaler must be setup before use");
        let calc = self
            .speed_calculator
            .as_mut()
            .expect("speed calculator not initialized");
        let mut estimates = calc.get_last_valid_estimates(current_time);

        // Apply over_provision_factor adjustments if it is set and we have a speed estimate
        for (stage_est, stage_problem) in estimates.stages.iter_mut().zip(problem.stages.iter()) {
            if let Some(f) = stage_problem.over_provision_factor {
                if let Some(bps) = stage_est.batches_per_second_per_worker.as_mut() {
                    *bps /= f as f64;
                }
            }
        }

        let out = run_fragmentation_autoscaler(
            problem,
            state,
            &estimates,
            1.5,
            &mut self.worker_id_factory,
        );
        debug!("Autoscaler result:\n{:?}", out);
        out
    }
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<FragmentationBasedAutoscaler>()?
        .add_class::<Estimate>()?
        .add_class::<Estimates>()?
        .add_class::<WorkerIdFactory>()?
        .add_function(wrap_pyfunction!(run_fragmentation_autoscaler, m)?)?
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

    fn make_cluster(
        num_nodes: usize,
        cpus_per_node: usize,
        gpus_per_node: usize,
        nvdecs_per_gpu: u8,
        nvencs_per_gpu: u8,
        heterogeneous: bool,
    ) -> rds::ClusterResources {
        let mut nodes: HashMap<String, rds::NodeResources> = HashMap::new();
        for i in 0..num_nodes {
            let mut cpus = cpus_per_node as f32;
            let mut gpus = gpus_per_node;
            if heterogeneous && (i % 2 == 0) {
                cpus = (cpus_per_node as f32) / 2.0;
                gpus = gpus_per_node / 2;
            }
            let mut gpu_vec: Vec<rds::GpuResources> = Vec::new();
            for _ in 0..gpus {
                gpu_vec.push(rds::GpuResources::make_from_num_codecs(
                    i as u8,
                    uuid::Uuid::new_v4(),
                    1.0,
                    nvdecs_per_gpu,
                    nvencs_per_gpu,
                ));
            }
            nodes.insert(
                format!("node{}", i),
                rds::NodeResources::new(cpus, Some(gpu_vec), None),
            );
        }
        rds::ClusterResources::new(Some(nodes))
    }

    fn make_default_state_for_stages(problem: &ds::Problem) -> ds::ProblemState {
        ds::ProblemState {
            stages: problem
                .stages
                .iter()
                .map(|s| ds::ProblemStageState {
                    stage_name: s.name.clone(),
                    workers: Vec::new(),
                    slots_per_worker: 2,
                    is_finished: false,
                })
                .collect(),
        }
    }

    fn estimates_from_speeds(speeds: &[Option<f64>]) -> Estimates {
        Estimates {
            stages: speeds
                .iter()
                .map(|s| Estimate {
                    batches_per_second_per_worker: *s,
                    num_returns_per_batch: Some(1.0),
                })
                .collect(),
        }
    }

    #[test]
    fn test_speed_estimator_basic() {
        let num_stages = 3usize;
        let mut estimator = SpeedAndNumberOfReturnsEstimator::new(num_stages, 10.0, 1);
        let current_time = 100.0;
        let measurements = ds::Measurements {
            time: current_time,
            stages: vec![
                ds::StageMeasurements {
                    task_measurements: vec![ds::TaskMeasurement {
                        start_time: 99.0,
                        end_time: 100.0,
                        num_returns: 1,
                    }],
                },
                ds::StageMeasurements {
                    task_measurements: vec![ds::TaskMeasurement {
                        start_time: 98.0,
                        end_time: 100.0,
                        num_returns: 1,
                    }],
                },
                ds::StageMeasurements {
                    task_measurements: vec![],
                },
            ],
        };
        estimator.update_with_measurements(&measurements);
        let estimates = estimator.get_estimates(current_time);
        let speeds: Vec<Option<f64>> = estimates
            .stages
            .iter()
            .map(|e| e.batches_per_second_per_worker)
            .collect();
        assert!((speeds[0].unwrap() - 1.0).abs() < 1e-3);
        assert!((speeds[1].unwrap() - 0.5).abs() < 1e-3);
        assert!(speeds[2].is_none());
    }

    #[test]
    fn test_speed_estimator_window() {
        let mut estimator = SpeedAndNumberOfReturnsEstimator::new(1, 10.0, 2);
        let measurements = ds::Measurements {
            time: 0.0,
            stages: vec![ds::StageMeasurements {
                task_measurements: vec![ds::TaskMeasurement {
                    start_time: 0.0,
                    end_time: 1.0,
                    num_returns: 1,
                }],
            }],
        };
        estimator.update_with_measurements(&measurements);
        let estimates = estimator.get_estimates(5.0);
        assert!(estimates.stages[0].batches_per_second_per_worker.is_some());
        assert!((estimates.stages[0].batches_per_second_per_worker.unwrap() - 1.0).abs() < 1e-3);

        let estimates2 = estimator.get_estimates(15.0);
        assert!(estimates2.stages[0].batches_per_second_per_worker.is_some());
    }

    #[test]
    fn test_calculate_slots_no_speeds() {
        let num_workers = vec![2usize, 3, 1];
        let current_slots = vec![4usize, 5, 6];
        let last_valid_speeds = vec![None, Some(1.0), None];
        let slots = calculate_num_slots_per_worker_for_all_stages(
            &num_workers,
            &current_slots,
            &last_valid_speeds,
            1.0,
            2,
        );
        assert_eq!(slots, current_slots);
    }

    #[test]
    fn test_calculate_slots_with_speeds() {
        let num_workers = vec![1usize, 1, 1];
        let current_slots = vec![1usize, 1, 1];
        let last_valid_speeds = vec![Some(1.0), Some(2.0), Some(0.5)];
        let slots = calculate_num_slots_per_worker_for_all_stages(
            &num_workers,
            &current_slots,
            &last_valid_speeds,
            0.1,
            2,
        );
        assert_eq!(slots, vec![10, 10, 10]);

        let slots_fast = calculate_num_slots_per_worker_for_all_stages(
            &num_workers,
            &current_slots,
            &last_valid_speeds,
            10.0,
            2,
        );
        assert_eq!(slots_fast, vec![2, 2, 2]);
    }

    #[test]
    fn test_calculate_slots_respects_minimum() {
        let num_workers = vec![1usize, 1, 1];
        let current_slots = vec![10usize, 20, 30];
        let last_valid_speeds = vec![Some(1.0), Some(1.0), Some(1.0)];
        let slots = calculate_num_slots_per_worker_for_all_stages(
            &num_workers,
            &current_slots,
            &last_valid_speeds,
            1.0,
            2,
        );
        assert_eq!(slots, current_slots);
    }

    #[test]
    fn test_simple_autoscaling_cpu_only() {
        let problem = ds::Problem {
            cluster_resources: make_cluster(1, 4, 0, 0, 0, false),
            stages: vec![ds::ProblemStage {
                name: "stage_0".to_string(),
                stage_batch_size: 1,
                worker_shape: rds::WorkerShapeWrapper {
                    inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                },
                requested_num_workers: None,
                over_provision_factor: None,
            }],
        };
        let state = make_default_state_for_stages(&problem);
        let estimates = Estimates {
            stages: vec![Estimate {
                batches_per_second_per_worker: Some(1.0),
                num_returns_per_batch: Some(1.0),
            }],
        };
        let mut worker_id_factory = WorkerIdFactory::new();
        let sol = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );
        assert_eq!(sol.num_new_workers_per_stage(), vec![4]);
        assert_eq!(sol.num_deleted_workers_per_stage(), vec![0]);
    }

    #[test]
    fn test_manual_worker_count_respected() {
        let problem = ds::Problem {
            cluster_resources: make_cluster(2, 24, 4, 0, 0, false),
            stages: vec![
                ds::ProblemStage {
                    name: "stage_0".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: Some(10),
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_1".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::WholeNumberedGpu(rds::WholeNumberedGpu {
                            num_gpus: 1,
                            num_cpus: 1.0,
                            num_nvdecs_per_gpu: 0,
                            num_nvencs_per_gpu: 0,
                        }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_2".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: Some(5),
                    over_provision_factor: None,
                },
            ],
        };
        let state = make_default_state_for_stages(&problem);
        let estimates = estimates_from_speeds(&[Some(1.0), Some(0.5), Some(10.0)]);
        let mut worker_id_factory = WorkerIdFactory::new();
        let sol = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );
        assert_eq!(sol.num_new_workers_per_stage(), vec![10, 8, 5]);
        assert_eq!(sol.num_deleted_workers_per_stage(), vec![0, 0, 0]);
    }

    #[test]
    fn test_manual_worker_count_respected_single_stage() {
        let problem = ds::Problem {
            cluster_resources: make_cluster(1, 24, 0, 0, 0, false),
            stages: vec![ds::ProblemStage {
                name: "stage_0".into(),
                stage_batch_size: 1,
                worker_shape: rds::WorkerShapeWrapper {
                    inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                },
                requested_num_workers: Some(10),
                over_provision_factor: None,
            }],
        };
        let state = make_default_state_for_stages(&problem);
        let estimates = estimates_from_speeds(&[Some(1.0)]);
        let mut worker_id_factory = WorkerIdFactory::new();
        let sol = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );
        assert_eq!(sol.num_new_workers_per_stage(), vec![10]);
        assert_eq!(sol.num_deleted_workers_per_stage(), vec![0]);
    }

    #[test]
    fn test_minimum_one_worker_per_stage() {
        // Mirror the original expectation: single node with 8 CPUs and three CPU-only stages
        let problem = ds::Problem {
            cluster_resources: make_cluster(1, 8, 0, 0, 0, false),
            stages: vec![
                ds::ProblemStage {
                    name: "stage_0".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_1".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_2".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
            ],
        };
        let state = make_default_state_for_stages(&problem);
        let estimates = estimates_from_speeds(&[Some(1.0), None, Some(0.5)]);
        let mut worker_id_factory = WorkerIdFactory::new();
        let sol = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );
        assert_eq!(sol.num_new_workers_per_stage(), vec![2, 2, 4]);
        assert_eq!(sol.num_deleted_workers_per_stage(), vec![0, 0, 0]);
    }

    #[test]
    fn test_gpu_fragmentation_awareness() {
        let problem = ds::Problem {
            cluster_resources: make_cluster(1, 8, 2, 2, 2, false),
            stages: vec![
                ds::ProblemStage {
                    name: "stage_0".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                            num_gpus: 0.5,
                            num_cpus: 1.0,
                            num_nvdecs: 0,
                            num_nvencs: 0,
                        }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_1".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::EntireGpu(rds::EntireGpu {
                            num_gpus: 1,
                            num_cpus: 1.0,
                        }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_2".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::Codec(rds::Codec {
                            num_cpus: 1.0,
                            num_nvdecs: 1,
                            num_nvencs: 0,
                        }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
            ],
        };
        let state = ds::ProblemState {
            stages: problem
                .stages
                .iter()
                .map(|s| ds::ProblemStageState {
                    stage_name: s.name.clone(),
                    workers: Vec::new(),
                    slots_per_worker: 1,
                    is_finished: false,
                })
                .collect(),
        };
        let estimates = estimates_from_speeds(&[Some(1.0), Some(1.0), Some(1.0)]);
        let mut worker_id_factory = WorkerIdFactory::new();
        let sol = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );
        assert_eq!(sol.num_new_workers_per_stage(), vec![2, 1, 1]);
        assert_eq!(sol.num_deleted_workers_per_stage(), vec![0, 0, 0]);
    }

    #[test]
    fn test_overallocation_target() {
        let problem = ds::Problem {
            cluster_resources: make_cluster(1, 1000, 8, 0, 0, false),
            stages: vec![
                ds::ProblemStage {
                    name: "stage_0".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_1".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::EntireGpu(rds::EntireGpu {
                            num_gpus: 1,
                            num_cpus: 1.0,
                        }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
                ds::ProblemStage {
                    name: "stage_2".into(),
                    stage_batch_size: 1,
                    worker_shape: rds::WorkerShapeWrapper {
                        inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
                    },
                    requested_num_workers: None,
                    over_provision_factor: None,
                },
            ],
        };
        let state = make_default_state_for_stages(&problem);
        let estimates = estimates_from_speeds(&[Some(0.1), Some(1.0), Some(0.05)]);
        let mut worker_id_factory = WorkerIdFactory::new();
        let sol = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );
        assert_eq!(sol.num_new_workers_per_stage(), vec![120, 8, 240]);
        assert_eq!(sol.num_deleted_workers_per_stage(), vec![0, 0, 0]);
    }

    // Turned off for normal tests because it takes too long when compiled without optimizations.
    // Run with:
    // cargo test --release --features release-tests
    #[test]
    #[cfg(feature = "release-tests")]
    fn test_large_number_of_nodes() {
        // Ported from python test. The original test had no assertions,
        // but it appears to be a stability test. We assert that a second
        // run of the autoscaler produces no changes.

        // Create a cluster with plenty of resources
        let cluster = make_cluster(100, 24, 8, 4, 4, false);

        let mut stages = Vec::new();
        let mut speeds = Vec::new();
        let mut cur_stage_idx = 0;

        stages.push(ds::ProblemStage {
            name: format!("stage_{}", cur_stage_idx),
            stage_batch_size: 1,
            worker_shape: rds::WorkerShapeWrapper {
                inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
            },
            requested_num_workers: None,
            over_provision_factor: None,
        });
        speeds.push(Some(1.0));
        cur_stage_idx += 1;

        for _ in 0..10 {
            stages.push(ds::ProblemStage {
                name: format!("stage_{}", cur_stage_idx),
                stage_batch_size: 1,
                worker_shape: rds::WorkerShapeWrapper {
                    inner: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.1,
                        num_cpus: 1.0,
                        num_nvdecs: 0,
                        num_nvencs: 0,
                    }),
                },
                requested_num_workers: None,
                over_provision_factor: None,
            });
            speeds.push(Some(0.5));
            cur_stage_idx += 1;
        }

        for _ in 0..4 {
            stages.push(ds::ProblemStage {
                name: format!("stage_{}", cur_stage_idx),
                stage_batch_size: 1,
                worker_shape: rds::WorkerShapeWrapper {
                    inner: rds::WorkerShape::FractionalGpu(rds::FractionalGpu {
                        num_gpus: 0.1,
                        num_cpus: 1.0,
                        num_nvdecs: 1,
                        num_nvencs: 2,
                    }),
                },
                requested_num_workers: None,
                over_provision_factor: None,
            });
            speeds.push(Some(0.5));
            cur_stage_idx += 1;
        }

        stages.push(ds::ProblemStage {
            name: format!("stage_{}", cur_stage_idx),
            stage_batch_size: 1,
            worker_shape: rds::WorkerShapeWrapper {
                inner: rds::WorkerShape::CpuOnly(rds::CpuOnly { num_cpus: 1.0 }),
            },
            requested_num_workers: None,
            over_provision_factor: None,
        });
        speeds.push(Some(1.0));

        let problem = ds::Problem {
            cluster_resources: cluster,
            stages,
        };

        // Create initial state with no workers
        let state = make_default_state_for_stages(&problem);
        let estimates = estimates_from_speeds(&speeds);

        let mut worker_id_factory = WorkerIdFactory::new();
        let solution1 = super::run_fragmentation_autoscaler(
            &problem,
            &state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );

        let new_state = ds::ProblemState {
            stages: problem
                .stages
                .iter()
                .zip(solution1.stages.iter())
                .map(|(p, s)| ds::ProblemStageState {
                    stage_name: p.name.clone(),
                    workers: s.new_workers.clone(),
                    slots_per_worker: s.slots_per_worker,
                    is_finished: false,
                })
                .collect(),
        };

        let solution2 = super::run_fragmentation_autoscaler(
            &problem,
            &new_state,
            &estimates,
            1.5,
            &mut worker_id_factory,
        );

        assert_eq!(
            solution2.num_new_workers_per_stage(),
            vec![0; problem.stages.len()]
        );
        assert_eq!(
            solution2.num_deleted_workers_per_stage(),
            vec![0; problem.stages.len()]
        );
    }
}