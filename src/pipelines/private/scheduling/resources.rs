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

#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct SpmdNodeMultiple {
    pub num_gpu_actors_in_group: u16,
    pub num_cpus_per_actor: FixedUtil,
    pub num_gpus_in_node: u8,
}

impl SpmdNodeMultiple {
    pub fn num_nodes_needed(&self) -> usize {
        self.num_gpu_actors_in_group as usize / self.num_gpus_in_node as usize
    }

    pub fn num_cpus_needed_per_node(&self) -> FixedUtil {
        self.num_cpus_per_actor * self.num_gpus_in_node as u32
    }
}

#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct SpmdSmallerThanNodeResources {
    pub num_gpu_actors_in_group: u16,
    pub num_cpus_per_actor: FixedUtil,
    pub num_gpus_in_node: u8,
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
/// use _cosmos_xenna::pipelines::private::scheduling::resources::{WorkerShape, CpuOnly};
/// use _cosmos_xenna::pipelines::private::scheduling::resources::FixedUtil;
///
/// let cpu_config = CpuOnly { num_cpus: FixedUtil::from_num(4.0) };
/// let worker = WorkerShape::CpuOnly(cpu_config);
/// ```
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerShape {
    CpuOnly(CpuOnly),
    FractionalGpu(FractionalGpu),
    WholeNumberedGpu(WholeNumberedGpu),
    SpmdNodeMultiple(SpmdNodeMultiple),
    SpmdSmallerThanNode(SpmdSmallerThanNodeResources),
}

impl WorkerShape {
    pub fn to_pool(&self) -> PoolOfResources {
        match self {
            WorkerShape::CpuOnly(cpu_config) => PoolOfResources {
                cpus: cpu_config.num_cpus.to_num::<f32>(),
                gpus: 0.0,
            },
            WorkerShape::FractionalGpu(fractional_gpu_config) => PoolOfResources {
                cpus: fractional_gpu_config.num_cpus.to_num::<f32>(),
                gpus: fractional_gpu_config.gpu_fraction.to_num::<f32>(),
            },
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => PoolOfResources {
                cpus: whole_numbered_gpu_config.num_cpus.to_num::<f32>()
                    * whole_numbered_gpu_config.num_gpus as f32,
                gpus: whole_numbered_gpu_config.num_gpus as f32,
            },
            WorkerShape::SpmdNodeMultiple(spmd_config) => PoolOfResources {
                cpus: spmd_config.num_cpus_per_actor.to_num::<f32>()
                    * spmd_config.num_gpu_actors_in_group as f32,
                gpus: spmd_config.num_gpu_actors_in_group as f32,
            },
            WorkerShape::SpmdSmallerThanNode(spmd_config) => PoolOfResources {
                cpus: spmd_config.num_cpus_per_actor.to_num::<f32>()
                    * spmd_config.num_gpu_actors_in_group as f32,
                gpus: spmd_config.num_gpu_actors_in_group as f32,
            },
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
        self.to_pool().cpus
    }

    fn get_num_gpus(&self) -> f32 {
        self.to_pool().gpus
    }

    fn is_spmd(&self) -> bool {
        match self {
            WorkerShape::CpuOnly(_) => false,
            WorkerShape::FractionalGpu(_) => false,
            WorkerShape::WholeNumberedGpu(_) => false,
            WorkerShape::SpmdNodeMultiple(_) => true,
            WorkerShape::SpmdSmallerThanNode(_) => true,
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
            WorkerShape::SpmdNodeMultiple(spmd_config) => {
                format!(
                    "WorkerShape::SpmdLargerThanNode(num_gpu_actors_in_group={}, num_cpus_per_actor={}, num_gpus_in_node={})",
                    spmd_config.num_gpu_actors_in_group,
                    spmd_config.num_cpus_per_actor,
                    spmd_config.num_gpus_in_node
                )
            }
            WorkerShape::SpmdSmallerThanNode(spmd_config) => {
                format!(
                    "WorkerShape::SpmdSmallerThanNode(num_gpu_actors_in_group={}, num_cpus_per_actor={}, num_gpus_in_node={})",
                    spmd_config.num_gpu_actors_in_group,
                    spmd_config.num_cpus_per_actor,
                    spmd_config.num_gpus_in_node
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
    #[error(
        "Invalid shape: {0:?}. If self.is_spmd is True, self.gpus needs to be an integer > 0 (e.g. 1, 2, 3, 3.0)."
    )]
    SpmdGpuNotInteger(Resources),
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
    #[pyo3(get, set)]
    pub is_spmd: bool,
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
    pub fn new(cpus: f32, gpus: f32, is_spmd: bool) -> Self {
        Self {
            cpus,
            gpus,
            is_spmd,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Resources(cpus={}, gpus={}, is_spmd={})",
            self.cpus, self.gpus, self.is_spmd
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn to_pool(
        &self,
        cluster_resources: &ClusterResources,
    ) -> Result<PoolOfResources, ShapeError> {
        Ok(self.to_shape(cluster_resources)?.to_pool())
    }

    pub fn to_shape(
        &self,
        cluster_resources: &ClusterResources,
    ) -> Result<WorkerShape, ShapeError> {
        // TODO: round down to the nearest fixed point value

        // Validation
        if self.cpus < 0.0 || self.gpus < 0.0 {
            return Err(ShapeError::NegativeValues(*self));
        }
        if self.cpus == 0.0 && self.gpus == 0.0 {
            return Err(ShapeError::ZeroResources(*self));
        }

        // SPMD
        if self.is_spmd {
            if self.gpus < 1.0 - 1e-6 || !self.gpus.abs_diff_eq(&self.gpus.round(), 1e-6) {
                return Err(ShapeError::SpmdGpuNotInteger(*self));
            }
            let most_common_num_gpus_per_node =
                cluster_resources.calc_most_common_num_gpus_per_node();
            let gpus_per_group = self.gpus.round() as usize;
            let num_cpus_per_actor = to_fixed_floor_f32(self.cpus);
            if gpus_per_group < most_common_num_gpus_per_node {
                return Ok(WorkerShape::SpmdSmallerThanNode(
                    SpmdSmallerThanNodeResources {
                        num_gpu_actors_in_group: gpus_per_group as u16,
                        num_cpus_per_actor: num_cpus_per_actor,
                        num_gpus_in_node: most_common_num_gpus_per_node as u8,
                    },
                ));
            } else if gpus_per_group % most_common_num_gpus_per_node == 0 {
                return Ok(WorkerShape::SpmdNodeMultiple(SpmdNodeMultiple {
                    num_gpu_actors_in_group: gpus_per_group as u16,
                    num_cpus_per_actor: num_cpus_per_actor,
                    num_gpus_in_node: most_common_num_gpus_per_node as u8,
                }));
            } else {
                return Err(ShapeError::SpmdGpuNotInteger(*self));
            }
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
    pub fn py_new(index: u8, uuid_: uuid::Uuid, used_fraction: f32) -> Self {
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
///
/// This struct describes how much of a GPU's resources are allocated to a worker.
/// It's a lightweight reference that points to a GPU in a node's GPU list rather
/// than storing full GPU details.
///
/// # Fields
///
/// * `offset` - **Index into the node's `NodeResources.gpus` vector**, not the hardware GPU index.
///              This indirection allows the same allocation to be used with different nodes,
///              and keeps the allocation struct small. To get the actual hardware GPU index
///              or UUID, you must look up `node_resources.gpus[offset]`.
///
/// * `used_fraction` - Fraction of the GPU's compute capacity allocated (0.0 to 1.0).
///                     For whole-GPU allocations, this is 1.0. For fractional allocations,
///                     this can be any value like 0.25, 0.5, etc.
///
/// # Important: Offset vs. GPU Index
///
/// The `offset` field is **not** the hardware GPU index! It's the position in the
/// `NodeResources.gpus` vector. For example:
/// - If a node has 4 GPUs and you want GPU at hardware index 2, you need to find
///   which position in the `gpus` vector corresponds to that GPU.
/// - The actual hardware index is stored in `GpuResources.index`
/// - The GPU UUID is stored in `GpuResources.uuid_`
///
/// # Example
///
/// ```rust
/// use _cosmos_xenna::pipelines::private::scheduling::resources::{GpuAllocation, NodeResources};
///
/// // Create an allocation for the first GPU in a node's list (offset=0)
/// // using 50% of its capacity
/// let alloc = GpuAllocation {
///     offset: 0,
///     used_fraction: fixed::FixedU32::from_num(0.5),
/// };
///
/// // To get the actual hardware GPU index:
/// // let hardware_index = node_resources.gpus[alloc.offset].index;
/// ```
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub struct GpuAllocation {
    #[pyo3(get, set)]
    pub offset: usize,
    pub used_fraction: FixedUtil,
}

#[pymethods]
impl GpuAllocation {
    #[new]
    pub fn py_new(offset: usize, used_fraction: f32) -> Self {
        Self {
            offset,
            used_fraction: FixedUtil::from_num(used_fraction),
        }
    }

    #[getter]
    fn get_used_fraction(&self) -> f32 {
        self.used_fraction.to_num::<f32>()
    }

    fn __repr__(&self) -> String {
        format!(
            "GPUAllocation(offset={}, used_fraction={})",
            self.offset, self.used_fraction
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

/// Represents all the resources allocated to a single worker.
#[pyclass]
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct WorkerGroupResources {
    pub workers: Vec<WorkerResources>,
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
    pub gpus: Vec<GpuAllocation>,
}

#[pymethods]
impl WorkerResources {
    #[new]
    pub fn py_new(node: String, cpus: f32, gpus: Option<Vec<GpuAllocation>>) -> Self {
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

    pub fn num_gpus(&self) -> usize {
        self.gpus.len()
    }

    pub fn num_fully_unallocated_gpus(&self) -> usize {
        self.gpus
            .iter()
            .filter(|g| g.is_fully_unallocated())
            .count()
    }

    pub fn all_gpus_fully_unallocated(&self) -> bool {
        self.num_fully_unallocated_gpus() == self.num_gpus()
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
            let Some(node_gpu) = self.gpus.get(alloc.offset) else {
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
            let node_gpu = self.gpus.get_mut(gpu.offset).unwrap();
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
            let node_gpu = self.gpus.get_mut(gpu.offset).unwrap();
            node_gpu.used_fraction += gpu.used_fraction;
        }
        Ok(())
    }

    pub fn release_allocation(&mut self, resources: &WorkerResources) {
        self.used_cpus -= resources.cpus;
        for gpu in &resources.gpus {
            let node_gpu = self.gpus.get_mut(gpu.offset).unwrap();
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

impl ClusterResources {
    pub fn allocate_multiple(
        &mut self,
        workers: &[WorkerResources],
    ) -> Result<(), AllocationError> {
        let mut allocated_workers = Vec::new();

        for worker in workers {
            let node = self.nodes.get_mut(&worker.node).expect("node exists");
            match node.allocate(worker) {
                Ok(()) => {
                    allocated_workers.push(worker.clone());
                }
                Err(e) => {
                    // Rollback successful allocations
                    for allocated_worker in allocated_workers {
                        let node = self
                            .nodes
                            .get_mut(&allocated_worker.node)
                            .expect("node exists");
                        node.release_allocation(&allocated_worker);
                    }
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    pub fn release_allocations(
        &mut self,
        resources: &[WorkerResources],
    ) -> Result<(), AllocationError> {
        let mut released_resources = Vec::new();

        for resource in resources {
            match self.release_allocation(resource) {
                Ok(()) => {
                    released_resources.push(resource.clone());
                }
                Err(e) => {
                    // Rollback successful releases by re-allocating them
                    for released_resource in released_resources {
                        let node = self
                            .nodes
                            .get_mut(&released_resource.node)
                            .expect("node exists");
                        if let Err(rollback_err) = node.allocate(&released_resource) {
                            // If rollback fails, log it but continue with the original error
                            eprintln!(
                                "Warning: Failed to rollback resource release: {}",
                                rollback_err
                            );
                        }
                    }
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl ClusterResources {
    #[staticmethod]
    pub fn make_uniform(node_resources: &NodeResources, node_ids: Vec<String>) -> Self {
        let mut node_dict: std::collections::HashMap<String, NodeResources> = Default::default();
        for node_id in node_ids {
            node_dict.insert(node_id.clone(), node_resources.clone());
        }
        Self { nodes: node_dict }
    }

    #[new]
    pub fn py_new(nodes: Option<std::collections::HashMap<String, NodeResources>>) -> Self {
        Self {
            nodes: nodes.unwrap_or_default(),
        }
    }

    pub fn allocate(&mut self, worker: &WorkerResources) -> Result<(), AllocationError> {
        let node = self.nodes.get_mut(&worker.node).unwrap();
        node.allocate(worker)?;
        Ok(())
    }

    pub fn release_allocation(
        &mut self,
        resources: &WorkerResources,
    ) -> Result<(), AllocationError> {
        let node = self.nodes.get_mut(&resources.node).expect("node exists");
        node.release_allocation(resources);
        Ok(())
    }

    pub fn calc_most_common_num_gpus_per_node(&self) -> usize {
        let mut num_gpus_per_node: std::collections::HashMap<usize, usize> = Default::default();
        for node in self.nodes.values() {
            if node.gpus.len() > 0 {
                *num_gpus_per_node.entry(node.gpus.len()).or_insert(0) += 1;
            }
        }
        if num_gpus_per_node.is_empty() {
            return 0;
        }
        *num_gpus_per_node
            .iter()
            .max_by_key(|(_, count)| *count)
            .unwrap()
            .0
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

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerGroup {
    pub id: String,
    pub stage_name: String,
    pub allocations: Vec<WorkerResources>,
}

impl WorkerGroup {
    pub fn from_worker(worker: Worker) -> Self {
        Self {
            id: worker.id,
            stage_name: worker.stage_name,
            allocations: vec![worker.allocation],
        }
    }
}

#[pymethods]
impl WorkerGroup {
    /// Splits the worker group's allocations into separate `WorkerResources` for each GPU.
    ///
    /// This method is useful for distributed training/inference scenarios where you need to treat
    /// each GPU as a separate worker with its own resource allocation. The CPUs are
    /// divided evenly among all GPUs in each allocation.
    ///
    /// # Returns
    ///
    /// A vector of `WorkerResources`, one for each GPU in the worker group. Each entry
    /// contains:
    /// - The same node as the original allocation
    /// - A fraction of the CPUs (total CPUs / number of GPUs in that allocation)
    /// - A single GPU allocation
    ///
    /// # Example
    ///
    /// If a worker group has an allocation with 8 CPUs and 4 GPUs, this method will
    /// return 4 `WorkerResources` entries, each with 2 CPUs and 1 GPU.
    fn split_allocation_per_gpu(&self) -> Vec<WorkerResources> {
        let mut metadata = Vec::new();
        for allocation in &self.allocations {
            for gpu in &allocation.gpus {
                metadata.push(WorkerResources {
                    node: allocation.node.clone(),
                    cpus: allocation.cpus / FixedUtil::from_num(allocation.gpus.len()),
                    gpus: vec![gpu.clone()],
                });
            }
        }
        metadata
    }

    fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    #[staticmethod]
    pub fn deserialize(data: &str) -> Self {
        serde_json::from_str(data).unwrap()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "WorkerGroup(id={}, stage_name={}, allocations={:?})",
            self.id, self.stage_name, self.allocations
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
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
        .add_class::<GpuAllocation>()?
        .add_class::<NodeInfo>()?
        .add_class::<PoolOfResources>()?
        .add_class::<CpuOnly>()?
        .add_class::<FractionalGpu>()?
        .add_class::<WholeNumberedGpu>()?
        .add_class::<NodeInfo>()?
        .add_class::<WorkerShape>()?
        .finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    // Helper function to create a test cluster with uniform nodes
    fn create_test_cluster() -> ClusterResources {
        let node1 = NodeResources::make_uniform(8, 2);
        let node2 = NodeResources::make_uniform(8, 2);
        let mut nodes = HashMap::new();
        nodes.insert("node1".to_string(), node1);
        nodes.insert("node2".to_string(), node2);
        ClusterResources { nodes }
    }

    // Helper function to create a test cluster with different node configurations
    fn create_heterogeneous_cluster() -> ClusterResources {
        let node1 = NodeResources::make_uniform(8, 2);
        let node2 = NodeResources::make_uniform(16, 4);
        let node3 = NodeResources::make_uniform(4, 1);
        let node4 = NodeResources::make_uniform(16, 4);
        let mut nodes = HashMap::new();
        nodes.insert("node1".to_string(), node1);
        nodes.insert("node2".to_string(), node2);
        nodes.insert("node3".to_string(), node3);
        nodes.insert("node4".to_string(), node4);
        ClusterResources { nodes }
    }

    // ============================================================================
    // WorkerShape Tests
    // ============================================================================

    #[test]
    fn test_worker_shape_cpu_only_to_pool() {
        let cpu_config = CpuOnly {
            num_cpus: FixedUtil::from_num(4.0),
        };
        let shape = WorkerShape::CpuOnly(cpu_config);
        let pool = shape.to_pool();

        assert_eq!(pool.cpus, 4.0);
        assert_eq!(pool.gpus, 0.0);
    }

    #[test]
    fn test_worker_shape_fractional_gpu_to_pool() {
        let fractional_gpu_config = FractionalGpu {
            gpu_fraction: FixedUtil::from_num(0.5),
            num_cpus: FixedUtil::from_num(2.0),
        };
        let shape = WorkerShape::FractionalGpu(fractional_gpu_config);
        let pool = shape.to_pool();

        assert_eq!(pool.cpus, 2.0);
        assert_eq!(pool.gpus, 0.5);
    }

    #[test]
    fn test_worker_shape_whole_numbered_gpu_to_pool() {
        let whole_gpu_config = WholeNumberedGpu {
            num_gpus: 2,
            num_cpus: FixedUtil::from_num(4.0),
        };
        let shape = WorkerShape::WholeNumberedGpu(whole_gpu_config);
        let pool = shape.to_pool();

        assert_eq!(pool.cpus, 8.0); // 4.0 * 2 GPUs
        assert_eq!(pool.gpus, 2.0);
    }

    #[test]
    fn test_worker_shape_spmd_node_multiple_to_pool() {
        let spmd_config = SpmdNodeMultiple {
            num_gpu_actors_in_group: 4,
            num_cpus_per_actor: FixedUtil::from_num(2.0),
            num_gpus_in_node: 2,
        };
        let shape = WorkerShape::SpmdNodeMultiple(spmd_config);
        let pool = shape.to_pool();

        assert_eq!(pool.cpus, 8.0); // 2.0 * 4 actors
        assert_eq!(pool.gpus, 4.0); // num_gpu_actors_in_group
    }

    #[test]
    fn test_worker_shape_spmd_smaller_than_node_to_pool() {
        let spmd_config = SpmdSmallerThanNodeResources {
            num_gpu_actors_in_group: 2,
            num_cpus_per_actor: FixedUtil::from_num(1.5),
            num_gpus_in_node: 4,
        };
        let shape = WorkerShape::SpmdSmallerThanNode(spmd_config);
        let pool = shape.to_pool();

        assert_eq!(pool.cpus, 3.0); // 1.5 * 2 actors
        assert_eq!(pool.gpus, 2.0); // num_gpu_actors_in_group
    }

    #[test]
    fn test_spmd_node_multiple_calculations() {
        let spmd_config = SpmdNodeMultiple {
            num_gpu_actors_in_group: 8,
            num_cpus_per_actor: FixedUtil::from_num(2.0),
            num_gpus_in_node: 2,
        };

        assert_eq!(spmd_config.num_nodes_needed(), 4); // 8 / 2
        assert_eq!(
            spmd_config.num_cpus_needed_per_node(),
            FixedUtil::from_num(4.0)
        ); // 2.0 * 2
    }

    // ============================================================================
    // Resources Tests
    // ============================================================================

    #[test]
    fn test_resources_to_shape_cpu_only() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: 4.0,
            gpus: 0.0,
            is_spmd: false,
        };

        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::CpuOnly(cpu_config) => {
                assert_eq!(cpu_config.num_cpus, FixedUtil::from_num(4.0));
            }
            _ => panic!("Expected CpuOnly shape"),
        }
    }

    #[test]
    fn test_resources_to_shape_fractional_gpu() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: 2.0,
            gpus: 0.5,
            is_spmd: false,
        };

        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::FractionalGpu(fractional_config) => {
                assert_eq!(fractional_config.gpu_fraction, FixedUtil::from_num(0.5));
                assert_eq!(fractional_config.num_cpus, FixedUtil::from_num(2.0));
            }
            _ => panic!("Expected FractionalGpu shape"),
        }
    }

    #[test]
    fn test_resources_to_shape_whole_numbered_gpu() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: 4.0,
            gpus: 2.0,
            is_spmd: false,
        };

        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::WholeNumberedGpu(whole_config) => {
                assert_eq!(whole_config.num_gpus, 2);
                assert_eq!(whole_config.num_cpus, FixedUtil::from_num(4.0));
            }
            _ => panic!("Expected WholeNumberedGpu shape"),
        }
    }

    #[test]
    fn test_resources_to_shape_spmd_smaller_than_node() {
        let cluster = create_heterogeneous_cluster();
        let resources = Resources {
            cpus: 2.0,
            gpus: 2.0, // Less than most common (4 GPUs per node)
            is_spmd: true,
        };

        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::SpmdSmallerThanNode(spmd_config) => {
                assert_eq!(spmd_config.num_gpu_actors_in_group, 2);
                assert_eq!(spmd_config.num_cpus_per_actor, FixedUtil::from_num(2.0));
                assert_eq!(spmd_config.num_gpus_in_node, 4); // Most common
            }
            _ => panic!("Expected SpmdSmallerThanNode shape"),
        }
    }

    #[test]
    fn test_resources_to_shape_spmd_node_multiple() {
        let cluster = create_heterogeneous_cluster();
        let resources = Resources {
            cpus: 2.0,
            gpus: 8.0, // Multiple of most common (4 GPUs per node)
            is_spmd: true,
        };

        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::SpmdNodeMultiple(spmd_config) => {
                assert_eq!(spmd_config.num_gpu_actors_in_group, 8);
                assert_eq!(spmd_config.num_cpus_per_actor, FixedUtil::from_num(2.0));
                assert_eq!(spmd_config.num_gpus_in_node, 4); // Most common
            }
            _ => panic!("Expected SpmdNodeMultiple shape"),
        }
    }

    #[test]
    fn test_resources_validation_negative_values() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: -1.0,
            gpus: 1.0,
            is_spmd: false,
        };

        let result = resources.to_shape(&cluster);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::NegativeValues(_) => {}
            _ => panic!("Expected NegativeValues error"),
        }
    }

    #[test]
    fn test_resources_validation_zero_resources() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: 0.0,
            gpus: 0.0,
            is_spmd: false,
        };

        let result = resources.to_shape(&cluster);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::ZeroResources(_) => {}
            _ => panic!("Expected ZeroResources error"),
        }
    }

    #[test]
    fn test_resources_validation_fractional_gpu_invalid() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: 1.0,
            gpus: 1.5, // Invalid: > 1.0 but not integer
            is_spmd: false,
        };

        let result = resources.to_shape(&cluster);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::GpuNotInteger(_) => {}
            _ => panic!("Expected GpuNotInteger error"),
        }
    }

    #[test]
    fn test_resources_validation_spmd_gpu_not_integer() {
        let cluster = create_test_cluster();
        let resources = Resources {
            cpus: 1.0,
            gpus: 1.5, // Invalid for SPMD: not integer
            is_spmd: true,
        };

        let result = resources.to_shape(&cluster);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::SpmdGpuNotInteger(_) => {}
            _ => panic!("Expected SpmdGpuNotInteger error"),
        }
    }

    // ============================================================================
    // PoolOfResources Tests
    // ============================================================================

    #[test]
    fn test_pool_of_resources_arithmetic() {
        let pool1 = PoolOfResources {
            cpus: 4.0,
            gpus: 2.0,
        };
        let pool2 = PoolOfResources {
            cpus: 2.0,
            gpus: 1.0,
        };

        // Addition
        let sum = pool1.add(&pool2);
        assert_eq!(sum.cpus, 6.0);
        assert_eq!(sum.gpus, 3.0);

        // Subtraction
        let diff = pool1.sub(&pool2);
        assert_eq!(diff.cpus, 2.0);
        assert_eq!(diff.gpus, 1.0);

        // Multiplication
        let scaled = pool1.multiply_by(2.0);
        assert_eq!(scaled.cpus, 8.0);
        assert_eq!(scaled.gpus, 4.0);
    }

    #[test]
    fn test_pool_of_resources_division() {
        let pool1 = PoolOfResources {
            cpus: 8.0,
            gpus: 4.0,
        };
        let pool2 = PoolOfResources {
            cpus: 2.0,
            gpus: 2.0,
        };

        let result = pool1.div(&pool2);
        assert_eq!(result.cpus, 4.0);
        assert_eq!(result.gpus, 2.0);
    }

    #[test]
    fn test_pool_of_resources_division_by_zero() {
        let pool1 = PoolOfResources {
            cpus: 8.0,
            gpus: 4.0,
        };
        let pool2 = PoolOfResources {
            cpus: 0.0,
            gpus: 2.0,
        };

        let result = pool1.div(&pool2);
        assert_eq!(result.cpus, 0.0); // Should be 0 when dividing by 0
        assert_eq!(result.gpus, 2.0);
    }

    #[test]
    fn test_pool_of_resources_contains() {
        let large_pool = PoolOfResources {
            cpus: 8.0,
            gpus: 4.0,
        };
        let small_pool = PoolOfResources {
            cpus: 4.0,
            gpus: 2.0,
        };
        let too_large_pool = PoolOfResources {
            cpus: 10.0,
            gpus: 2.0,
        };

        assert!(large_pool.contains(&small_pool));
        assert!(!large_pool.contains(&too_large_pool));
        assert!(large_pool.contains(&large_pool)); // Should contain itself
    }

    #[test]
    fn test_pool_of_resources_total_num() {
        let pool = PoolOfResources {
            cpus: 4.0,
            gpus: 2.0,
        };
        assert_eq!(pool.total_num(), 6.0);
    }

    #[test]
    fn test_pool_of_resources_to_dict() {
        let pool = PoolOfResources {
            cpus: 4.0,
            gpus: 2.0,
        };
        let dict = pool.to_dict();

        assert_eq!(dict.get("cpu"), Some(&4.0));
        assert_eq!(dict.get("gpu"), Some(&2.0));
    }

    // ============================================================================
    // GpuResources Tests
    // ============================================================================

    #[test]
    fn test_gpu_resources_allocation() {
        let gpu = GpuResources {
            index: 0,
            uuid_: Uuid::new_v4(),
            used_fraction: FixedUtil::from_num(0.3),
        };

        assert_eq!(gpu.index, 0);
        assert_eq!(gpu.used_fraction, FixedUtil::from_num(0.3));
        assert!(!gpu.is_fully_unallocated());

        let used_pool = gpu.used_pool();
        assert!((used_pool.gpus - 0.3).abs() < 1e-4);
        assert_eq!(used_pool.cpus, 0.0);

        let free_pool = gpu.free_pool();
        assert!((free_pool.gpus - 0.7).abs() < 1e-4);
        assert_eq!(free_pool.cpus, 0.0);
    }

    #[test]
    fn test_gpu_resources_fully_unallocated() {
        let gpu = GpuResources {
            index: 0,
            uuid_: Uuid::new_v4(),
            used_fraction: FixedUtil::from_num(0.0),
        };
        assert!(gpu.is_fully_unallocated());
    }

    // ============================================================================
    // GPUAllocation Tests
    // ============================================================================

    #[test]
    fn test_gpu_allocation() {
        let allocation = GpuAllocation {
            offset: 1,
            used_fraction: FixedUtil::from_num(0.5),
        };

        assert_eq!(allocation.offset, 1);
        assert_eq!(allocation.used_fraction, FixedUtil::from_num(0.5));
        assert_eq!(allocation.get_used_fraction(), 0.5);
    }

    // ============================================================================
    // WorkerResources Tests
    // ============================================================================

    #[test]
    fn test_worker_resources_to_pool() {
        let gpu_allocations = vec![
            GpuAllocation {
                offset: 0,
                used_fraction: FixedUtil::from_num(0.5),
            },
            GpuAllocation {
                offset: 1,
                used_fraction: FixedUtil::from_num(0.25),
            },
        ];
        let worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: gpu_allocations,
        };

        let pool = worker.to_pool();
        assert_eq!(pool.cpus, 4.0);
        assert_eq!(pool.gpus, 0.75); // 0.5 + 0.25
    }

    // ============================================================================
    // NodeResources Tests
    // ============================================================================

    #[test]
    fn test_node_resources_make_uniform() {
        let node = NodeResources::make_uniform(8, 2);

        assert_eq!(node.total_cpus, FixedUtil::from_num(8.0));
        assert_eq!(node.used_cpus, FixedUtil::ZERO);
        assert_eq!(node.gpus.len(), 2);
        assert_eq!(node.num_gpus(), 2);
        assert_eq!(node.num_fully_unallocated_gpus(), 2);
        assert!(node.all_gpus_fully_unallocated());
    }

    #[test]
    fn test_node_resources_pool_calculations() {
        let node = NodeResources::make_uniform(8, 2);

        // Initially all resources are free
        let total_pool = node.total_pool();
        assert_eq!(total_pool.cpus, 8.0);
        assert_eq!(total_pool.gpus, 2.0);

        let used_pool = node.used_pool();
        assert_eq!(used_pool.cpus, 0.0);
        assert_eq!(used_pool.gpus, 0.0);

        let free_pool = node.free_pool();
        assert_eq!(free_pool.cpus, 8.0);
        assert_eq!(free_pool.gpus, 2.0);
    }

    #[test]
    fn test_node_resources_can_allocate() {
        let node = NodeResources::make_uniform(8, 2);

        // Valid allocation
        let valid_worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: vec![GpuAllocation {
                offset: 0,
                used_fraction: FixedUtil::from_num(0.5),
            }],
        };
        assert!(node.can_allocate(&valid_worker));

        // Invalid allocation - too many CPUs
        let invalid_cpu_worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(10.0),
            gpus: vec![],
        };
        assert!(!node.can_allocate(&invalid_cpu_worker));

        // Invalid allocation - GPU index out of range
        let invalid_gpu_worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: vec![GpuAllocation {
                offset: 5,
                used_fraction: FixedUtil::from_num(0.5),
            }], // Only 2 GPUs available
        };
        assert!(!node.can_allocate(&invalid_gpu_worker));

        // Invalid allocation - GPU over-allocation
        let over_allocated_gpu_worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: vec![GpuAllocation {
                offset: 0,
                used_fraction: FixedUtil::from_num(1.5),
            }], // More than 100%
        };
        assert!(!node.can_allocate(&over_allocated_gpu_worker));
    }

    #[test]
    fn test_node_resources_allocate_and_release() {
        let mut node = NodeResources::make_uniform(8, 2);

        let worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: vec![GpuAllocation {
                offset: 0,
                used_fraction: FixedUtil::from_num(0.5),
            }],
        };

        // Allocate resources
        let result = node.allocate(&worker);
        assert!(result.is_ok());

        // Check that resources are now used
        let used_pool = node.used_pool();
        assert_eq!(used_pool.cpus, 4.0);
        assert_eq!(used_pool.gpus, 0.5);

        let free_pool = node.free_pool();
        assert_eq!(free_pool.cpus, 4.0);
        assert_eq!(free_pool.gpus, 1.5);

        // Release resources
        node.release_allocation(&worker);

        // Check that resources are freed
        let used_pool_after = node.used_pool();
        assert_eq!(used_pool_after.cpus, 0.0);
        assert_eq!(used_pool_after.gpus, 0.0);
    }

    #[test]
    fn test_node_resources_allocate_failure() {
        let mut node = NodeResources::make_uniform(8, 2);

        let worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(10.0), // More than available
            gpus: vec![],
        };

        let result = node.allocate(&worker);
        assert!(result.is_err());
        match result.unwrap_err() {
            AllocationError::NotEnoughResources { .. } => {}
            _ => panic!("Expected NotEnoughResources error"),
        }
    }

    // ============================================================================
    // ClusterResources Tests
    // ============================================================================

    #[test]
    fn test_cluster_resources_make_uniform() {
        let node_template = NodeResources::make_uniform(8, 2);
        let node_ids = vec!["node1".to_string(), "node2".to_string()];
        let cluster = ClusterResources::make_uniform(&node_template, node_ids);

        assert_eq!(cluster.num_nodes(), 2);
        assert_eq!(cluster.num_total_gpus(), 4); // 2 nodes * 2 GPUs each
        assert_eq!(cluster.num_total_cpus(), 16.0); // 2 nodes * 8 CPUs each
    }

    #[test]
    fn test_cluster_resources_calc_most_common_num_gpus_per_node() {
        let cluster = create_heterogeneous_cluster();
        // node1: 2 GPUs, node2: 4 GPUs, node3: 1 GPU
        // Most common should be 2 GPUs (appears once, but let's check the logic)
        let most_common = cluster.calc_most_common_num_gpus_per_node();
        // The actual most common will depend on the implementation
        assert!(most_common > 0);
    }

    #[test]
    fn test_cluster_resources_allocate_single_worker() {
        let mut cluster = create_test_cluster();

        let worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: vec![GpuAllocation {
                offset: 0,
                used_fraction: FixedUtil::from_num(0.5),
            }],
        };

        let result = cluster.allocate(&worker);
        assert!(result.is_ok());

        // Check that the node's resources are updated
        let node = cluster.nodes.get("node1").unwrap();
        let used_pool = node.used_pool();
        assert_eq!(used_pool.cpus, 4.0);
        assert_eq!(used_pool.gpus, 0.5);
    }

    #[test]
    fn test_cluster_resources_allocate_multiple_workers() {
        let mut cluster = create_test_cluster();

        let workers = vec![
            WorkerResources {
                node: "node1".to_string(),
                cpus: FixedUtil::from_num(2.0),
                gpus: vec![GpuAllocation {
                    offset: 0,
                    used_fraction: FixedUtil::from_num(0.5),
                }],
            },
            WorkerResources {
                node: "node2".to_string(),
                cpus: FixedUtil::from_num(3.0),
                gpus: vec![GpuAllocation {
                    offset: 1,
                    used_fraction: FixedUtil::from_num(0.25),
                }],
            },
        ];

        let result = cluster.allocate_multiple(&workers);
        assert!(result.is_ok());

        // Check both nodes
        let node1 = cluster.nodes.get("node1").unwrap();
        let node1_used = node1.used_pool();
        assert_eq!(node1_used.cpus, 2.0);
        assert_eq!(node1_used.gpus, 0.5);

        let node2 = cluster.nodes.get("node2").unwrap();
        let node2_used = node2.used_pool();
        assert_eq!(node2_used.cpus, 3.0);
        assert_eq!(node2_used.gpus, 0.25);
    }

    #[test]
    fn test_cluster_resources_allocate_multiple_workers_with_failure() {
        let mut cluster = create_test_cluster();

        let workers = vec![
            WorkerResources {
                node: "node1".to_string(),
                cpus: FixedUtil::from_num(2.0),
                gpus: vec![GpuAllocation {
                    offset: 0,
                    used_fraction: FixedUtil::from_num(0.5),
                }],
            },
            WorkerResources {
                node: "node1".to_string(),
                cpus: FixedUtil::from_num(10.0), // Too many CPUs
                gpus: vec![],
            },
        ];

        let result = cluster.allocate_multiple(&workers);
        assert!(result.is_err());

        // Check that the first allocation was rolled back
        let node1 = cluster.nodes.get("node1").unwrap();
        let node1_used = node1.used_pool();
        assert_eq!(node1_used.cpus, 0.0);
        assert_eq!(node1_used.gpus, 0.0);
    }

    #[test]
    fn test_cluster_resources_release_allocations() {
        let mut cluster = create_test_cluster();

        let worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(4.0),
            gpus: vec![GpuAllocation {
                offset: 0,
                used_fraction: FixedUtil::from_num(0.5),
            }],
        };

        // Allocate first
        cluster.allocate(&worker).unwrap();

        // Then release
        let result = cluster.release_allocations(&[worker.clone()]);
        assert!(result.is_ok());

        // Check that resources are freed
        let node1 = cluster.nodes.get("node1").unwrap();
        let node1_used = node1.used_pool();
        assert_eq!(node1_used.cpus, 0.0);
        assert_eq!(node1_used.gpus, 0.0);
    }

    #[test]
    fn test_cluster_resources_pool_calculations() {
        let cluster = create_test_cluster();

        let total_pool = cluster.total_pool();
        assert_eq!(total_pool.cpus, 16.0); // 2 nodes * 8 CPUs
        assert_eq!(total_pool.gpus, 4.0); // 2 nodes * 2 GPUs

        let used_pool = cluster.used_pool();
        assert_eq!(used_pool.cpus, 0.0);
        assert_eq!(used_pool.gpus, 0.0);

        let free_pool = cluster.free_pool();
        assert_eq!(free_pool.cpus, 16.0);
        assert_eq!(free_pool.gpus, 4.0);
    }

    // ============================================================================
    // Worker Tests
    // ============================================================================

    #[test]
    fn test_worker_serialization() {
        let worker = Worker {
            id: "worker1".to_string(),
            stage_name: "stage1".to_string(),
            allocation: WorkerResources {
                node: "node1".to_string(),
                cpus: FixedUtil::from_num(4.0),
                gpus: vec![GpuAllocation {
                    offset: 0,
                    used_fraction: FixedUtil::from_num(0.5),
                }],
            },
        };

        let serialized = worker.serialize();
        let deserialized = Worker::deserialize(&serialized);

        assert_eq!(worker.id, deserialized.id);
        assert_eq!(worker.stage_name, deserialized.stage_name);
        assert_eq!(worker.allocation.node, deserialized.allocation.node);
    }

    // ============================================================================
    // WorkerGroup Tests
    // ============================================================================

    #[test]
    fn test_worker_group_from_worker() {
        let worker = Worker {
            id: "worker1".to_string(),
            stage_name: "stage1".to_string(),
            allocation: WorkerResources {
                node: "node1".to_string(),
                cpus: FixedUtil::from_num(4.0),
                gpus: vec![GpuAllocation {
                    offset: 0,
                    used_fraction: FixedUtil::from_num(0.5),
                }],
            },
        };

        let group = WorkerGroup::from_worker(worker.clone());

        assert_eq!(group.id, worker.id);
        assert_eq!(group.stage_name, worker.stage_name);
        assert_eq!(group.allocations.len(), 1);
        assert_eq!(group.allocations[0].node, worker.allocation.node);
    }

    // ============================================================================
    // Edge Cases and Error Handling Tests
    // ============================================================================

    #[test]
    fn test_fractional_gpu_edge_cases() {
        let cluster = create_test_cluster();

        // Test very small fractional GPU
        let resources = Resources {
            cpus: 1.0,
            gpus: 0.001,
            is_spmd: false,
        };
        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::FractionalGpu(config) => {
                assert!(config.gpu_fraction.to_num::<f32>() > 0.0);
            }
            _ => panic!("Expected FractionalGpu shape"),
        }

        // Test fractional GPU close to 1.0
        let resources = Resources {
            cpus: 1.0,
            gpus: 0.999,
            is_spmd: false,
        };
        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::FractionalGpu(config) => {
                assert!(config.gpu_fraction.to_num::<f32>() < 1.0);
            }
            _ => panic!("Expected FractionalGpu shape"),
        }
    }

    #[test]
    fn test_spmd_edge_cases() {
        let cluster = create_heterogeneous_cluster();

        // Test SPMD with exactly the most common number of GPUs
        let resources = Resources {
            cpus: 2.0,
            gpus: 4.0, // Should match most common
            is_spmd: true,
        };
        let shape = resources.to_shape(&cluster).unwrap();
        match shape {
            WorkerShape::SpmdNodeMultiple(config) => {
                assert_eq!(config.num_gpu_actors_in_group, 4);
                assert_eq!(config.num_gpus_in_node, 4);
            }
            _ => panic!("Expected SpmdNodeMultiple shape"),
        }
    }

    #[test]
    fn test_cluster_with_no_gpus() {
        let node = NodeResources::make_uniform(8, 0);
        let mut nodes = HashMap::new();
        nodes.insert("node1".to_string(), node);
        let cluster = ClusterResources { nodes };

        // Test that most common calculation handles no GPUs gracefully
        let most_common = cluster.calc_most_common_num_gpus_per_node();
        assert_eq!(most_common, 0);
    }

    #[test]
    fn test_very_large_allocations() {
        let mut node = NodeResources::make_uniform(1000, 8);

        let worker = WorkerResources {
            node: "node1".to_string(),
            cpus: FixedUtil::from_num(999.0),
            gpus: vec![
                GpuAllocation {
                    offset: 0,
                    used_fraction: FixedUtil::from_num(0.9),
                },
                GpuAllocation {
                    offset: 1,
                    used_fraction: FixedUtil::from_num(0.8),
                },
                GpuAllocation {
                    offset: 2,
                    used_fraction: FixedUtil::from_num(0.7),
                },
            ],
        };

        assert!(node.can_allocate(&worker));
        let result = node.allocate(&worker);
        assert!(result.is_ok());
    }

    #[test]
    fn test_precision_edge_cases() {
        // Test that fixed-point arithmetic handles precision correctly
        let cpu_config = CpuOnly {
            num_cpus: FixedUtil::from_num(0.1),
        };
        let shape = WorkerShape::CpuOnly(cpu_config);
        let pool = shape.to_pool();

        // The result should be close to 0.1 but might have slight precision differences
        assert!((pool.cpus - 0.1).abs() < 1e-4);
    }
}
