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

//! Data structures used to represent allocated/available resources on a cluster/node/gpu.
//!
//! Many of the classes in this module are "shapes". A shape is a fully specified resource requirement for something.
//! Shapes are meant to specified by users on a per-stage basis.

use approx::AbsDiffEq;
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};
use fixed::FixedU32;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::utils::module_builders::ImportablePyModuleBuilder;

// Used to track the utilization of a single GPU or all of the CPUs on a node. The value is 0 <=x < 2**16 with 16 bits of precision.
pub type FixedUtil = FixedU32<fixed::types::extra::U16>;

// These are the data-carrying variants of our enum
/// A shape which only requires a certain number of CPUs.
///
/// `num_cpus` can be a fraction. In means multiple workers can be allocated to the same cpu.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct CpuOnly {
    pub num_cpus: FixedUtil,
}

/// A shape which requires a fraction of a GPU.
///
/// Can also require cpus, nvdecs and nvencs.
///
/// `num_gpus` must be 0.0 < x < 1.0.
///
/// This enables multiple workers to be allocated on a single gpu.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct FractionalGpu {
    pub gpu_fraction: FixedUtil,
    pub num_cpus: FixedUtil,
}

/// A shape which requires a whole number GPU(s).
///
/// Can also require cpus, nvdecs and nvencs
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct WholeNumberedGpu {
    pub num_gpus: u8,
    pub num_cpus: FixedUtil,
}

/// A class representing the shape of compute resources for a worker.
///
/// This class encapsulates different types of compute resource configurations and
/// provides methods to query and manipulate these configurations. It supports
/// various resource types including CPU-only, codec, and different GPU
/// configurations.
///
/// Example:
/// ```rust
/// let cpu_config = CpuOnly { num_cpus: 4.0 };
/// let worker = WorkerShape::CpuOnly(cpu_config);
/// ```
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerShape {
    CpuOnly(CpuOnly),
    FractionalGpu(FractionalGpu),
    WholeNumberedGpu(WholeNumberedGpu),
}

impl WorkerShape {
    pub fn to_pool(&self) -> Result<PoolOfResources, ShapeError> {
        match self {
            WorkerShape::CpuOnly(cpu_config) => Ok(PoolOfResources {
                cpus: cpu_config.num_cpus.to_num::<f32>(),
                gpus: 0.0,
            }),
            WorkerShape::FractionalGpu(fractional_gpu_config) => Ok(PoolOfResources {
                cpus: fractional_gpu_config.num_cpus.to_num::<f32>(),
                gpus: fractional_gpu_config.gpu_fraction.to_num::<f32>(),
            }),
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => Ok(PoolOfResources {
                cpus: whole_numbered_gpu_config.num_cpus.to_num::<f32>(),
                gpus: whole_numbered_gpu_config.num_gpus.into(),
            }),
        }
    }
}

#[pymethods]
impl WorkerShape {
    #[staticmethod]
    pub fn deserialize(data: &str) -> WorkerShape {
        serde_json::from_str(data).unwrap()
    }

    pub fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn get_num_cpus(&self) -> f32 {
        match self {
            WorkerShape::CpuOnly(cpu_config) => cpu_config.num_cpus.to_num::<f32>(),
            WorkerShape::FractionalGpu(fractional_gpu_config) => {
                fractional_gpu_config.num_cpus.to_num::<f32>()
            }
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => {
                whole_numbered_gpu_config.num_cpus.to_num::<f32>()
            }
        }
    }

    fn get_num_gpus(&self) -> f32 {
        match self {
            WorkerShape::CpuOnly(_) => 0.0,
            WorkerShape::FractionalGpu(fractional_gpu_config) => {
                fractional_gpu_config.gpu_fraction.to_num::<f32>()
            }
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => {
                whole_numbered_gpu_config.num_gpus.into()
            }
        }
    }

    fn __repr__(&self) -> String {
        match self {
            WorkerShape::CpuOnly(c) => format!("WorkerShape::CpuOnly(num_cpus={})", c.num_cpus),
            WorkerShape::FractionalGpu(c) => {
                format!(
                    "WorkerShape::FractionalGpu(num_gpus={}, num_cpus={})",
                    c.gpu_fraction, c.num_cpus
                )
            }
            WorkerShape::WholeNumberedGpu(c) => {
                format!(
                    "WorkerShape::WholeNumberedGpu(num_gpus={}, num_cpus={})",
                    c.num_gpus, c.num_cpus
                )
            }
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum ShapeError {
    #[error("Invalid shape: {0:?}. Some values were negative.")]
    NegativeValues(Resources),
    #[error(
        "Invalid shape: {0:?}. Expected at least one value to be nonzero, but all values were zero."
    )]
    ZeroResources(Resources),
    #[error(
        "Invalid shape: {0:?}. If entire_gpu is set to True, self.gpus needs to be an integer > 0 (e.g. 1, 2, 3, 3.0)."
    )]
    EntireGpuNotInteger(Resources),
    #[error(
        "Invalid shape: {0:?}. If self.entire_gpu is True, nvdecs and nvencs can not be explictly asked for."
    )]
    EntireGpuWithCodecs(Resources),
    #[error(
        "Invalid shape: {0:?}. If self.gpus is greater than 1, self.gpus needs to be an integer (e.g. 1, 2, 3, 3.0)."
    )]
    GpuNotInteger(Resources),
    #[error(
        "Invalid shape: {0:?}. If self.gpus is less than 1, is also must be greater than 0. (e.g. 0.5, 0.25, 0.75)."
    )]
    FractionalGpuNotValid(Resources),
}

/// A user friendly way to specify the resources required for something.
///
/// This class provides an intuitive interface for specifying resource requirements
/// that get translated into more detailed internal worker shapes.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct Resources {
    #[pyo3(get, set)]
    pub cpus: f32,
    #[pyo3(get, set)]
    pub gpus: f32,
}

impl From<ShapeError> for PyErr {
    fn from(err: ShapeError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum AllocationError {
    #[error("GPU index {gpu_index} out of range for node resources")]
    GpuIndexOutOfRange { gpu_index: usize },
    #[error(
        "Not enough resources on node {node}. Requested: {resources:?}, available: {available:?}"
    )]
    NotEnoughResources {
        node: String,
        resources: PoolOfResources,
        available: PoolOfResources,
    },
    #[error("Node '{0}' not found in cluster resources")]
    NodeNotFound(String),
}

impl From<AllocationError> for PyErr {
    fn from(err: AllocationError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

// Round down to the nearest fixed point value
fn to_fixed_floor_f32(x: f32) -> FixedUtil {
    let f = FixedUtil::from_num(x);
    if f.to_num::<f32>() > x {
        FixedUtil::from_bits(f.to_bits().saturating_sub(1))
    } else {
        f
    }
}

#[pymethods]
impl Resources {
    #[new]
    pub fn new(cpus: f32, gpus: f32) -> Self {
        Self { cpus, gpus }
    }

    fn __repr__(&self) -> String {
        format!("Resources(cpus={}, gpus={})", self.cpus, self.gpus)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn to_pool(&self) -> Result<PoolOfResources, ShapeError> {
        self.to_shape()?.to_pool()
    }

    pub fn to_shape(&self) -> Result<WorkerShape, ShapeError> {
        // TODO: round down to the nearest fixed point value

        // Validation
        if self.cpus < 0.0 || self.gpus < 0.0 {
            return Err(ShapeError::NegativeValues(*self));
        }
        if self.cpus == 0.0 && self.gpus == 0.0 {
            return Err(ShapeError::ZeroResources(*self));
        }

        // CPU stage
        if self.cpus > 0.0 && self.gpus == 0.0 {
            return Ok(WorkerShape::CpuOnly(CpuOnly {
                num_cpus: to_fixed_floor_f32(self.cpus),
            }));
        }

        // Whole numbered GPU
        if self.gpus >= 1.0 - 1e-6 {
            if !self.gpus.abs_diff_eq(&self.gpus.round(), 1e-6) {
                return Err(ShapeError::GpuNotInteger(*self));
            }
            return Ok(WorkerShape::WholeNumberedGpu(WholeNumberedGpu {
                num_gpus: self.gpus.round() as u8,
                num_cpus: to_fixed_floor_f32(self.cpus),
            }));
        }

        // Fractional GPU
        if !(self.gpus > 0.0 && self.gpus < 1.0) {
            return Err(ShapeError::FractionalGpuNotValid(*self));
        } else {
            return Ok(WorkerShape::FractionalGpu(FractionalGpu {
                gpu_fraction: to_fixed_floor_f32(self.gpus),
                num_cpus: to_fixed_floor_f32(self.cpus),
            }));
        }
    }
}

// --------------------
// PoolOfResources
// --------------------
/// Represents the resources required by a worker or available on a node.
///
/// This is a way of reporting resources which doesn't keep track of the nuances around node/gpu boundaries. It can
/// be useful for user facing reporting and some simple allocation algorithms.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct PoolOfResources {
    /// Number of CPUs (can be fractional)
    #[pyo3(get, set)]
    pub cpus: f32,
    /// Number of GPUs (can be fractional)  
    #[pyo3(get, set)]
    pub gpus: f32,
}

#[pymethods]
impl PoolOfResources {
    #[new]
    pub fn new(cpus: f32, gpus: f32) -> Self {
        Self { cpus, gpus }
    }

    fn __repr__(&self) -> String {
        format!("PoolOfResources(cpus={}, gpus={})", self.cpus, self.gpus,)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn total_num(&self) -> f32 {
        self.cpus + self.gpus
    }

    pub fn multiply_by(&self, factor: f32) -> Self {
        Self {
            cpus: self.cpus * factor,
            gpus: self.gpus * factor,
        }
    }

    pub fn add(&self, other: &PoolOfResources) -> Self {
        Self {
            cpus: self.cpus + other.cpus,
            gpus: self.gpus + other.gpus,
        }
    }

    pub fn sub(&self, other: &PoolOfResources) -> Self {
        Self {
            cpus: self.cpus - other.cpus,
            gpus: self.gpus - other.gpus,
        }
    }

    pub fn div(&self, other: &PoolOfResources) -> Self {
        Self {
            cpus: if other.cpus != 0.0 {
                self.cpus / other.cpus
            } else {
                0.0
            },
            gpus: if other.gpus != 0.0 {
                self.gpus / other.gpus
            } else {
                0.0
            },
        }
    }

    pub fn contains(&self, other: &PoolOfResources) -> bool {
        self.cpus >= other.cpus && self.gpus >= other.gpus
    }

    pub fn to_dict(&self) -> std::collections::HashMap<String, f32> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("cpu".to_string(), self.cpus);
        map.insert("gpu".to_string(), self.gpus);
        map
    }
}

// --------------------
// GpuResources
// --------------------
/// Represents the state of allocation for a single GPU.
#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct GpuResources {
    #[pyo3(get, set)]
    pub index: u8,
    #[pyo3(get, set)]
    pub uuid_: uuid::Uuid,
    pub used_fraction: FixedUtil,
}

#[pymethods]
impl GpuResources {
    #[new]
    pub fn new(index: u8, uuid_: uuid::Uuid, used_fraction: f32) -> Self {
        Self {
            index,
            uuid_,
            used_fraction: FixedUtil::from_num(used_fraction),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuResources(index={}, uuid_={:?}, used_fraction={})",
            self.index, self.uuid_, self.used_fraction
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn is_fully_unallocated(&self) -> bool {
        self.used_fraction == FixedUtil::ZERO
    }

    pub fn used_pool(&self) -> PoolOfResources {
        PoolOfResources {
            cpus: 0.0,
            gpus: self.used_fraction.to_num::<f32>(),
        }
    }

    pub fn free_pool(&self) -> PoolOfResources {
        PoolOfResources {
            cpus: 0.0,
            gpus: 1.0 - self.used_fraction.to_num::<f32>(),
        }
    }
}

// --------------------
// GPUAllocation
// --------------------
/// Represents the allocation a worker is taking up for a given GPU.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct GPUAllocation {
    #[pyo3(get, set)]
    pub index: usize,
    pub used_fraction: FixedUtil,
}

#[pymethods]
impl GPUAllocation {
    #[new]
    pub fn new(index: usize, used_fraction: f32) -> Self {
        Self {
            index,
            used_fraction: FixedUtil::from_num(used_fraction),
        }
    }

    #[getter]
    fn get_used_fraction(&self) -> f32 {
        self.used_fraction.to_num::<f32>()
    }

    fn __repr__(&self) -> String {
        format!(
            "GPUAllocation(index={}, used_fraction={})",
            self.index, self.used_fraction
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
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
// WorkerResources
// --------------------
/// Represents all the resources allocated to a single worker.
#[pyclass]
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct WorkerResources {
    #[pyo3(get, set)]
    pub node: String,
    pub cpus: FixedUtil,
    #[pyo3(get, set)]
    pub gpus: Vec<GPUAllocation>,
}

#[pymethods]
impl WorkerResources {
    #[new]
    pub fn py_new(node: String, cpus: f32, gpus: Option<Vec<GPUAllocation>>) -> Self {
        Self {
            node,
            cpus: FixedUtil::from_num(cpus),
            gpus: gpus.unwrap_or_default(),
        }
    }

    #[getter]
    fn get_cpus(&self) -> f32 {
        self.cpus.to_num::<f32>()
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkerResources(node={}, cpus={}, gpus={:?})",
            self.node, self.cpus, self.gpus
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn to_pool(&self) -> PoolOfResources {
        let gpu_sum: f32 = self
            .gpus
            .iter()
            .map(|g| g.used_fraction.to_num::<f32>())
            .sum::<f32>();
        PoolOfResources {
            cpus: self.cpus.to_num::<f32>(),
            gpus: gpu_sum,
        }
    }
}

// --------------------
// NodeResources
// --------------------
/// Represents all the resources available on a single node in a cluster.
#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct NodeResources {
    pub used_cpus: FixedUtil,
    pub total_cpus: FixedUtil,
    #[pyo3(get, set)]
    pub gpus: Vec<GpuResources>,
    #[pyo3(get, set)]
    pub name: Option<String>,
}

#[pymethods]
impl NodeResources {
    #[new]
    pub fn py_new(
        used_cpus: f32,
        total_cpus: f32,
        gpus: Vec<GpuResources>,
        name: Option<String>,
    ) -> Self {
        Self {
            used_cpus: FixedUtil::from_num(used_cpus),
            total_cpus: FixedUtil::from_num(total_cpus),
            gpus: gpus,
            name,
        }
    }

    /// Make a "uniform" node. I.e. all the nodes have the same number of nvdecs and nvencs.
    #[staticmethod]
    pub fn make_uniform(num_cpus: u32, num_gpus: u32) -> Self {
        let mut gpus: Vec<GpuResources> = Vec::with_capacity(num_gpus as usize);
        for i in 0..num_gpus {
            gpus.push(GpuResources {
                index: i as u8,
                uuid_: uuid::Uuid::new_v4(),
                used_fraction: FixedUtil::ZERO,
            });
        }
        Self {
            used_cpus: FixedUtil::ZERO,
            total_cpus: FixedUtil::from_num(num_cpus),
            gpus,
            name: None,
        }
    }

    pub fn used_pool(&self) -> PoolOfResources {
        let mut out = PoolOfResources {
            cpus: self.used_cpus.to_num::<f32>(),
            gpus: 0.0,
        };
        for gpu in &self.gpus {
            out = out.add(&gpu.used_pool());
        }
        out
    }

    pub fn free_pool(&self) -> PoolOfResources {
        let mut out = PoolOfResources {
            cpus: self.total_cpus.to_num::<f32>() - self.used_cpus.to_num::<f32>(),
            gpus: 0.0,
        };

        for gpu in &self.gpus {
            out = out.add(&gpu.free_pool());
        }
        out
    }

    pub fn total_pool(&self) -> PoolOfResources {
        PoolOfResources {
            cpus: self.total_cpus.to_num::<f32>(),
            gpus: self.gpus.len() as f32,
        }
    }

    pub fn can_allocate(&self, resources: &WorkerResources) -> bool {
        // Check CPUs
        if self.used_cpus + resources.cpus > self.total_cpus {
            return false;
        }

        // Check GPUs
        for alloc in &resources.gpus {
            // Ensure GPU index exists
            let Some(node_gpu) = self.gpus.get(alloc.index) else {
                return false;
            };
            // Ensure the resulting allocation would not exceed 100%
            if node_gpu.used_fraction + alloc.used_fraction > FixedUtil::ONE {
                return false;
            }
        }

        true
    }

    pub fn allocate(&mut self, resources: &WorkerResources) -> Result<(), AllocationError> {
        // Check CPUs
        if self.used_cpus + resources.cpus > self.total_cpus {
            return Err(AllocationError::NotEnoughResources {
                node: self.name.clone().unwrap_or_default(),
                resources: resources.to_pool(),
                available: self.free_pool(),
            });
        }

        // Check GPUs
        for gpu in &resources.gpus {
            let node_gpu = self.gpus.get_mut(gpu.index).unwrap();
            if node_gpu.used_fraction + gpu.used_fraction > FixedUtil::ONE {
                return Err(AllocationError::NotEnoughResources {
                    node: self.name.clone().unwrap_or_default(),
                    resources: resources.to_pool(),
                    available: self.free_pool(),
                });
            }
        }
        self.used_cpus += resources.cpus;
        for gpu in &resources.gpus {
            let node_gpu = self.gpus.get_mut(gpu.index).unwrap();
            node_gpu.used_fraction += gpu.used_fraction;
        }
        Ok(())
    }

    pub fn release_allocation(&mut self, resources: &WorkerResources) {
        self.used_cpus -= resources.cpus;
        for gpu in &resources.gpus {
            let node_gpu = self.gpus.get_mut(gpu.index).unwrap();
            node_gpu.used_fraction -= gpu.used_fraction;
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeResources(used_cpus={}, total_cpus={}, gpus=len({}), name={:?})",
            self.used_cpus,
            self.total_cpus,
            self.gpus.len(),
            self.name
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// --------------------
// ClusterResources
// --------------------
/// Represents the total resources available in the entire cluster.
#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct ClusterResources {
    /// dict of all nodes in the cluster
    #[pyo3(get, set)]
    pub nodes: std::collections::HashMap<String, NodeResources>,
}

#[pymethods]
impl ClusterResources {
    #[new]
    pub fn new(nodes: Option<std::collections::HashMap<String, NodeResources>>) -> Self {
        Self {
            nodes: nodes.unwrap_or_default(),
        }
    }

    pub fn allocate(&mut self, worker: &WorkerResources) -> Result<(), AllocationError> {
        let node = self.nodes.get_mut(&worker.node).unwrap();
        node.allocate(worker)?;
        Ok(())
    }

    #[staticmethod]
    pub fn make_uniform(node_resources: &NodeResources, node_ids: Vec<String>) -> Self {
        let mut node_dict: std::collections::HashMap<String, NodeResources> = Default::default();
        for node_id in node_ids {
            node_dict.insert(node_id.clone(), node_resources.clone());
        }
        Self { nodes: node_dict }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_used_gpus(&self) -> usize {
        let mut out: usize = 0;
        for node in self.nodes.values() {
            out += node.gpus.len();
        }
        out
    }

    pub fn num_used_cpus(&self) -> f32 {
        self.nodes
            .values()
            .map(|n| n.used_cpus.to_num::<f32>())
            .sum()
    }

    pub fn num_total_cpus(&self) -> f32 {
        self.nodes
            .values()
            .map(|n| n.total_cpus.to_num::<f32>())
            .sum()
    }

    pub fn num_total_gpus(&self) -> usize {
        self.nodes.values().map(|n| n.gpus.len()).sum()
    }

    pub fn used_pool(&self) -> PoolOfResources {
        let mut out = PoolOfResources::default();
        for node in self.nodes.values() {
            out = out.add(&node.used_pool());
        }
        out
    }

    pub fn free_pool(&self) -> PoolOfResources {
        let mut out = PoolOfResources::default();
        for node in self.nodes.values() {
            out = out.add(&node.free_pool());
        }
        out
    }

    pub fn total_pool(&self) -> PoolOfResources {
        let mut out = PoolOfResources::default();
        for node in self.nodes.values() {
            out = out.add(&node.total_pool());
        }
        out
    }
    pub fn release_allocation(
        &mut self,
        resources: &WorkerResources,
    ) -> Result<(), AllocationError> {
        let Some(node) = self.nodes.get_mut(&resources.node) else {
            return Err(AllocationError::NodeNotFound(resources.node.clone()));
        };
        node.release_allocation(resources);
        Ok(())
    }

    pub fn make_detailed_utilization_table(&self) -> String {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                Cell::new("Component"),
                Cell::new("Used"),
                Cell::new("Free"),
                Cell::new("Total"),
                Cell::new("Utilization"),
            ]);

        // Cluster totals
        let used_cluster = self.used_pool();
        let free_cluster = self.free_pool();
        let total_cpus = self.num_total_cpus();
        let total_gpus = self.num_total_gpus() as f32;

        let cpu_bar = create_bar_chart(used_cluster.cpus, total_cpus, 20);
        let gpu_bar = create_bar_chart(used_cluster.gpus, total_gpus, 20);

        table.add_row(vec![
            Cell::new("Cluster CPUs"),
            Cell::new(format!("{:.2}", used_cluster.cpus)),
            Cell::new(format!("{:.2}", free_cluster.cpus)),
            Cell::new(format!("{:.2}", total_cpus)),
            Cell::new(cpu_bar),
        ]);
        table.add_row(vec![
            Cell::new("Cluster GPUs"),
            Cell::new(format!("{:.2}", used_cluster.gpus)),
            Cell::new(format!("{:.2}", free_cluster.gpus)),
            Cell::new(format!("{:.2}", total_gpus)),
            Cell::new(gpu_bar),
        ]);

        // Per-node breakdown (sorted by node id for stable output)
        let mut nodes: Vec<(&String, &NodeResources)> = self.nodes.iter().collect();
        nodes.sort_by(|a, b| a.0.cmp(b.0));

        for (node_id, node) in nodes {
            let used_cpus = node.used_cpus.to_num::<f32>();
            let total_cpus = node.total_cpus.to_num::<f32>();
            let free_cpus = (total_cpus - used_cpus).max(0.0);
            let cpu_bar = create_bar_chart(used_cpus, total_cpus, 20);

            let label = match &node.name {
                Some(name) if !name.is_empty() => format!("Node {} ({}) CPUs", node_id, name),
                _ => format!("Node {} CPUs", node_id),
            };
            table.add_row(vec![
                Cell::new(label),
                Cell::new(format!("{used_cpus:.2}")),
                Cell::new(format!("{free_cpus:.2}")),
                Cell::new(format!("{total_cpus:.2}")),
                Cell::new(cpu_bar),
            ]);

            for gpu in &node.gpus {
                let used = gpu.used_fraction.to_num::<f32>();
                let free = (1.0 - used).max(0.0);
                let bar = create_bar_chart(used, 1.0, 20);
                table.add_row(vec![
                    Cell::new(format!("  GPU {}", gpu.index)),
                    Cell::new(format!("{used:.2}")),
                    Cell::new(format!("{free:.2}")),
                    Cell::new("1.00"),
                    Cell::new(bar),
                ]);
            }
        }

        table.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ClusterResources(num_nodes={})", self.nodes.len())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// --------------------
// Worker
// --------------------
/// An allocated worker
#[pyclass]
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Worker {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub stage_name: String,
    #[pyo3(get, set)]
    pub allocation: WorkerResources,
}

#[pymethods]
impl Worker {
    #[new]
    pub fn new(id: String, stage_name: String, allocation: WorkerResources) -> Self {
        Self {
            id,
            stage_name,
            allocation,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Worker(id={}, stage_name={}, allocation={})",
            self.id,
            self.stage_name,
            self.allocation.__repr__()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getnewargs__(&self) -> (String, String, WorkerResources) {
        (
            self.id.clone(),
            self.stage_name.clone(),
            self.allocation.clone(),
        )
    }

    pub fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    #[staticmethod]
    pub fn deserialize(data: &str) -> Self {
        serde_json::from_str(data).unwrap()
    }
}

// --------------------
// WorkerMetadata
// --------------------
#[pyclass(get_all, set_all)]
#[derive(Debug, PartialEq, Clone)]
pub struct WorkerMetadata {
    pub worker_id: String,
    pub allocation: WorkerResources,
}

#[pymethods]
impl WorkerMetadata {
    #[new]
    pub fn new(worker_id: String, allocation: WorkerResources) -> Self {
        Self {
            worker_id,
            allocation,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkerMetadata(worker_id={}, allocation={})",
            self.worker_id,
            self.allocation.__repr__()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn make_mock() -> Self {
        Self {
            worker_id: "mock".to_string(),
            allocation: WorkerResources {
                node: "mock".to_string(),
                cpus: FixedUtil::from_num(1.0),
                gpus: vec![],
            },
        }
    }
}

// --------------------
// NodeInfo
// --------------------
#[pyclass(get_all, set_all)]
#[derive(Debug, PartialEq, Clone)]
pub struct NodeInfo {
    pub node_id: String,
}

#[pymethods]
impl NodeInfo {
    #[new]
    pub fn new(node_id: String) -> Self {
        Self { node_id }
    }

    fn __repr__(&self) -> String {
        format!("NodeInfo(node_id={})", self.node_id)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<Resources>()?
        .add_class::<Worker>()?
        .add_class::<WorkerResources>()?
        .add_class::<ClusterResources>()?
        .add_class::<NodeResources>()?
        .add_class::<GpuResources>()?
        .add_class::<GPUAllocation>()?
        .add_class::<WorkerMetadata>()?
        .add_class::<NodeInfo>()?
        .add_class::<PoolOfResources>()?
        .add_class::<CpuOnly>()?
        .add_class::<FractionalGpu>()?
        .add_class::<WholeNumberedGpu>()?
        .add_class::<NodeInfo>()?
        .add_class::<WorkerMetadata>()?
        .add_class::<WorkerShape>()?
        .finish();
    Ok(())
}
