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
use bincode::{Decode, Encode};
use pyo3::prelude::*;
use std::collections::HashSet;
use thiserror::Error;

use crate::utils::module_builders::ImportablePyModuleBuilder;

// These are the data-carrying variants of our enum
/// A shape which only requires a certain number of CPUs.
///
/// `num_cpus` can be a fraction. In means multiple workers can be allocated to the same cpu.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Encode, Decode)]
pub struct CpuOnly {
    #[pyo3(get, set)]
    pub num_cpus: f32,
}

#[pymethods]
impl CpuOnly {
    #[new]
    fn new(num_cpus: f32) -> Self {
        Self { num_cpus }
    }

    fn __getnewargs__(&self) -> (f32,) {
        (self.num_cpus,)
    }

    fn __repr__(&self) -> String {
        format!("CpuOnly(num_cpus={})", self.num_cpus)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// A shape which only requires CPUs and codec hardware (nvdec/nvenc).
///
/// All of the nvdecs/nvencs will come come from the same gpu.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Encode, Decode)]
pub struct Codec {
    #[pyo3(get, set)]
    pub num_cpus: f32,
    #[pyo3(get, set)]
    pub num_nvdecs: u8,
    #[pyo3(get, set)]
    pub num_nvencs: u8,
}

#[pymethods]
impl Codec {
    #[new]
    fn new(num_cpus: f32, num_nvdecs: u8, num_nvencs: u8) -> Self {
        Self {
            num_cpus,
            num_nvdecs,
            num_nvencs,
        }
    }

    fn __getnewargs__(&self) -> (f32, u8, u8) {
        (self.num_cpus, self.num_nvdecs, self.num_nvencs)
    }

    fn __repr__(&self) -> String {
        format!(
            "Codec(num_cpus={}, num_nvdecs={}, num_nvencs={})",
            self.num_cpus, self.num_nvdecs, self.num_nvencs
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// A shape which requires a fraction of a GPU.
///
/// Can also require cpus, nvdecs and nvencs.
///
/// `num_gpus` must be 0.0 < x < 1.0.
///
/// This enables multiple workers to be allocated on a single gpu.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Encode, Decode)]
pub struct FractionalGpu {
    #[pyo3(get, set)]
    pub num_gpus: f32,
    #[pyo3(get, set)]
    pub num_cpus: f32,
    #[pyo3(get, set)]
    pub num_nvdecs: u8,
    #[pyo3(get, set)]
    pub num_nvencs: u8,
}

#[pymethods]
impl FractionalGpu {
    #[new]
    fn new(num_gpus: f32, num_cpus: f32, num_nvdecs: u8, num_nvencs: u8) -> Self {
        Self {
            num_gpus,
            num_cpus,
            num_nvdecs,
            num_nvencs,
        }
    }

    fn __getnewargs__(&self) -> (f32, f32, u8, u8) {
        (
            self.num_gpus,
            self.num_cpus,
            self.num_nvdecs,
            self.num_nvencs,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "FractionalGpu(num_gpus={}, num_cpus={}, num_nvdecs={}, num_nvencs={})",
            self.num_gpus, self.num_cpus, self.num_nvdecs, self.num_nvencs
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// A shape which requires a whole number GPU(s).
///
/// Can also require cpus, nvdecs and nvencs
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Encode, Decode)]
pub struct WholeNumberedGpu {
    #[pyo3(get, set)]
    pub num_gpus: u8,
    #[pyo3(get, set)]
    pub num_cpus: f32,
    #[pyo3(get, set)]
    pub num_nvdecs_per_gpu: u8,
    #[pyo3(get, set)]
    pub num_nvencs_per_gpu: u8,
}

#[pymethods]
impl WholeNumberedGpu {
    #[new]
    fn new(num_gpus: u8, num_cpus: f32, num_nvdecs_per_gpu: u8, num_nvencs_per_gpu: u8) -> Self {
        Self {
            num_gpus,
            num_cpus,
            num_nvdecs_per_gpu,
            num_nvencs_per_gpu,
        }
    }

    fn __getnewargs__(&self) -> (u8, f32, u8, u8) {
        (
            self.num_gpus,
            self.num_cpus,
            self.num_nvdecs_per_gpu,
            self.num_nvencs_per_gpu,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "WholeNumberedGpu(num_gpus={}, num_cpus={}, num_nvdecs_per_gpu={}, num_nvencs_per_gpu={})",
            self.num_gpus, self.num_cpus, self.num_nvdecs_per_gpu, self.num_nvencs_per_gpu
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// A shape which requires an entire GPU(s), including all of the nvdecs and nvencs.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy, Encode, Decode)]
pub struct EntireGpu {
    #[pyo3(get, set)]
    pub num_gpus: u8,
    #[pyo3(get, set)]
    pub num_cpus: f32,
}

#[pymethods]
impl EntireGpu {
    #[new]
    fn new(num_gpus: u8, num_cpus: f32) -> Self {
        Self { num_gpus, num_cpus }
    }

    fn __getnewargs__(&self) -> (u8, f32) {
        (self.num_gpus, self.num_cpus)
    }

    fn __repr__(&self) -> String {
        format!(
            "EntireGpu(num_gpus={}, num_cpus={})",
            self.num_gpus, self.num_cpus
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
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
#[derive(Debug, Clone, Encode, Decode)]
pub enum WorkerShape {
    CpuOnly(CpuOnly),
    Codec(Codec),
    FractionalGpu(FractionalGpu),
    WholeNumberedGpu(WholeNumberedGpu),
    EntireGpu(EntireGpu),
}

impl WorkerShape {
    pub fn to_pool(
        &self,
        num_nvdecs_per_gpu: u8,
        num_nvencs_per_gpu: u8,
    ) -> Result<PoolOfResources, ShapeError> {
        match self {
            WorkerShape::CpuOnly(cpu_config) => {
                Ok(PoolOfResources::new(cpu_config.num_cpus, 0.0, 0.0, 0.0))
            }
            WorkerShape::Codec(codec_config) => Ok(PoolOfResources::new(
                codec_config.num_cpus,
                0.0,
                codec_config.num_nvdecs.into(),
                codec_config.num_nvencs.into(),
            )),
            WorkerShape::FractionalGpu(fractional_gpu_config) => Ok(PoolOfResources::new(
                fractional_gpu_config.num_cpus,
                fractional_gpu_config.num_gpus,
                fractional_gpu_config.num_nvdecs.into(),
                fractional_gpu_config.num_nvencs.into(),
            )),
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => Ok(PoolOfResources::new(
                whole_numbered_gpu_config.num_cpus,
                whole_numbered_gpu_config.num_gpus.into(),
                (whole_numbered_gpu_config.num_nvdecs_per_gpu as f32) * num_nvdecs_per_gpu as f32,
                (whole_numbered_gpu_config.num_nvencs_per_gpu as f32) * num_nvencs_per_gpu as f32,
            )),
            WorkerShape::EntireGpu(entire_gpu_config) => Ok(PoolOfResources::new(
                entire_gpu_config.num_cpus,
                entire_gpu_config.num_gpus.into(),
                (num_nvdecs_per_gpu as f32) * entire_gpu_config.num_gpus as f32,
                (num_nvencs_per_gpu as f32) * entire_gpu_config.num_gpus as f32,
            )),
        }
    }

    fn get_num_cpus(&self) -> f32 {
        match self {
            WorkerShape::CpuOnly(cpu_config) => cpu_config.num_cpus,
            WorkerShape::Codec(codec_config) => codec_config.num_cpus,
            WorkerShape::FractionalGpu(fractional_gpu_config) => fractional_gpu_config.num_cpus,
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => {
                whole_numbered_gpu_config.num_cpus
            }
            WorkerShape::EntireGpu(entire_gpu_config) => entire_gpu_config.num_cpus,
        }
    }

    fn get_num_gpus(&self) -> f32 {
        match self {
            WorkerShape::CpuOnly(_) => 0.0,
            WorkerShape::Codec(_) => 0.0,
            WorkerShape::FractionalGpu(fractional_gpu_config) => fractional_gpu_config.num_gpus,
            WorkerShape::WholeNumberedGpu(whole_numbered_gpu_config) => {
                whole_numbered_gpu_config.num_gpus.into()
            }
            WorkerShape::EntireGpu(entire_gpu_config) => entire_gpu_config.num_gpus.into(),
        }
    }
}

#[pyclass(name = "WorkerShape")]
#[derive(Debug, Clone)]
pub struct WorkerShapeWrapper {
    pub inner: WorkerShape,
}

impl From<WorkerShape> for WorkerShapeWrapper {
    fn from(shape: WorkerShape) -> Self {
        WorkerShapeWrapper { inner: shape }
    }
}

impl From<WorkerShapeWrapper> for WorkerShape {
    fn from(wrapper: WorkerShapeWrapper) -> Self {
        wrapper.inner
    }
}

#[pymethods]
impl WorkerShapeWrapper {
    /// Only used for pickling
    #[new]
    pub fn new() -> Self {
        WorkerShapeWrapper {
            inner: WorkerShape::CpuOnly(CpuOnly { num_cpus: 0.0 }),
        }
    }

    pub fn to_pool(
        &self,
        num_nvdecs_per_gpu: u8,
        num_nvencs_per_gpu: u8,
    ) -> Result<PoolOfResources, ShapeError> {
        return self.inner.to_pool(num_nvdecs_per_gpu, num_nvencs_per_gpu);
    }

    fn get_num_cpus(&self) -> f32 {
        return self.inner.get_num_cpus();
    }

    fn get_num_gpus(&self) -> f32 {
        return self.inner.get_num_gpus();
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            WorkerShape::CpuOnly(c) => format!("WorkerShape::CpuOnly({})", c.__repr__()),
            WorkerShape::Codec(c) => format!("WorkerShape::Codec({})", c.__repr__()),
            WorkerShape::FractionalGpu(c) => {
                format!("WorkerShape::FractionalGpu({})", c.__repr__())
            }
            WorkerShape::WholeNumberedGpu(c) => {
                format!("WorkerShape::WholeNumberedGpu({})", c.__repr__())
            }
            WorkerShape::EntireGpu(c) => format!("WorkerShape::EntireGpu({})", c.__repr__()),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn __getstate__(&self) -> Vec<u8> {
        bincode::encode_to_vec(&self.inner, bincode::config::standard())
            .expect("Failed to encode WorkerShape")
    }

    /// Called when unpickling the object.
    /// The `state` argument is the bytes returned from `__getstate__`.
    /// We decode it to restore the internal Rust enum.
    pub fn __setstate__(&mut self, state: Vec<u8>) {
        let (decoded, _): (WorkerShape, usize) =
            bincode::decode_from_slice(&state, bincode::config::standard())
                .expect("Failed to decode WorkerShape");
        self.inner = decoded;
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
    #[pyo3(get, set)]
    pub nvdecs: u8,
    #[pyo3(get, set)]
    pub nvencs: u8,
    #[pyo3(get, set)]
    pub entire_gpu: bool,
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
    #[error("NVDEC unavailable on GPU {gpu_index}. Available nvdecs: {available:?}")]
    NvdecUnavailable {
        gpu_index: usize,
        available: Vec<u8>,
    },
    #[error("NVENC unavailable on GPU {gpu_index}. Available nvencs: {available:?}")]
    NvencUnavailable {
        gpu_index: usize,
        available: Vec<u8>,
    },
    #[error("Node '{0}' not found in cluster resources")]
    NodeNotFound(String),
}

impl From<AllocationError> for PyErr {
    fn from(err: AllocationError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

#[pymethods]
impl Resources {
    #[new]
    pub fn new(cpus: f32, gpus: f32, nvdecs: u8, nvencs: u8, entire_gpu: bool) -> Self {
        Self {
            cpus,
            gpus,
            nvdecs,
            nvencs,
            entire_gpu,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Resources(cpus={}, gpus={}, nvdecs={}, nvencs={}, entire_gpu={})",
            self.cpus, self.gpus, self.nvdecs, self.nvencs, self.entire_gpu
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn to_pool(
        &self,
        num_nvdecs_per_gpu: u8,
        num_nvencs_per_gpu: u8,
    ) -> Result<PoolOfResources, ShapeError> {
        self.to_shape()?
            .to_pool(num_nvdecs_per_gpu, num_nvencs_per_gpu)
    }

    pub fn to_shape(&self) -> Result<WorkerShapeWrapper, ShapeError> {
        // Validation
        if self.cpus < 0.0 || self.gpus < 0.0 {
            return Err(ShapeError::NegativeValues(*self));
        }
        if self.cpus == 0.0 && self.gpus == 0.0 && self.nvdecs == 0 && self.nvencs == 0 {
            return Err(ShapeError::ZeroResources(*self));
        }

        // Entire GPU
        if self.entire_gpu {
            if !(self.gpus > 0.0 && self.gpus.abs_diff_eq(&self.gpus.round(), 1e-6)) {
                return Err(ShapeError::EntireGpuNotInteger(*self));
            }
            if self.nvdecs > 0 || self.nvencs > 0 {
                return Err(ShapeError::EntireGpuWithCodecs(*self));
            }
            return Ok(WorkerShapeWrapper {
                inner: WorkerShape::EntireGpu(EntireGpu {
                    num_gpus: self.gpus.round() as u8,
                    num_cpus: self.cpus,
                }),
            });
        }

        // CPU stage
        if self.cpus > 0.0 && self.gpus == 0.0 && self.nvdecs == 0 && self.nvencs == 0 {
            return Ok(WorkerShapeWrapper {
                inner: WorkerShape::CpuOnly(CpuOnly {
                    num_cpus: self.cpus,
                }),
            });
        }

        // Whole numbered GPU
        if self.gpus >= 1.0 - 1e-6 {
            if !self.gpus.abs_diff_eq(&self.gpus.round(), 1e-6) {
                return Err(ShapeError::GpuNotInteger(*self));
            }
            return Ok(WorkerShapeWrapper {
                inner: WorkerShape::WholeNumberedGpu(WholeNumberedGpu {
                    num_gpus: self.gpus.round() as u8,
                    num_cpus: self.cpus,
                    num_nvdecs_per_gpu: self.nvdecs,
                    num_nvencs_per_gpu: self.nvencs,
                }),
            });
        }

        // Codec
        if (self.nvdecs > 0 || self.nvencs > 0) && self.gpus == 0.0 {
            return Ok(WorkerShapeWrapper {
                inner: WorkerShape::Codec(Codec {
                    num_cpus: self.cpus,
                    num_nvdecs: self.nvdecs,
                    num_nvencs: self.nvencs,
                }),
            });
        }

        // Fractional GPU
        if !(self.gpus > 0.0 && self.gpus < 1.0) {
            return Err(ShapeError::FractionalGpuNotValid(*self));
        } else {
            return Ok(WorkerShapeWrapper {
                inner: WorkerShape::FractionalGpu(FractionalGpu {
                    num_gpus: self.gpus,
                    num_cpus: self.cpus,
                    num_nvdecs: self.nvdecs,
                    num_nvencs: self.nvencs,
                }),
            });
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
    /// Number of NVIDIA decoders
    #[pyo3(get, set)]
    pub nvdecs: f32,
    /// Number of NVIDIA encoders
    #[pyo3(get, set)]
    pub nvencs: f32,
}

#[pymethods]
impl PoolOfResources {
    #[new]
    pub fn new(cpus: f32, gpus: f32, nvdecs: f32, nvencs: f32) -> Self {
        Self {
            cpus,
            gpus,
            nvdecs,
            nvencs,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PoolOfResources(cpus={}, gpus={}, nvdecs={}, nvencs={})",
            self.cpus, self.gpus, self.nvdecs, self.nvencs
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn total_num(&self) -> f32 {
        self.cpus + self.gpus + self.nvdecs + self.nvencs
    }

    pub fn multiply_by(&self, factor: f32) -> Self {
        Self {
            cpus: self.cpus * factor,
            gpus: self.gpus * factor,
            nvdecs: self.nvdecs * factor,
            nvencs: self.nvencs * factor,
        }
    }

    pub fn add(&self, other: &PoolOfResources) -> Self {
        Self {
            cpus: self.cpus + other.cpus,
            gpus: self.gpus + other.gpus,
            nvdecs: self.nvdecs + other.nvdecs,
            nvencs: self.nvencs + other.nvencs,
        }
    }

    pub fn sub(&self, other: &PoolOfResources) -> Self {
        Self {
            cpus: self.cpus - other.cpus,
            gpus: self.gpus - other.gpus,
            nvdecs: self.nvdecs - other.nvdecs,
            nvencs: self.nvencs - other.nvencs,
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
            nvdecs: if other.nvdecs != 0.0 {
                self.nvdecs / other.nvdecs
            } else {
                0.0
            },
            nvencs: if other.nvencs != 0.0 {
                self.nvencs / other.nvencs
            } else {
                0.0
            },
        }
    }

    pub fn contains(&self, other: &PoolOfResources) -> bool {
        self.cpus >= other.cpus
            && self.gpus >= other.gpus
            && self.nvdecs >= other.nvdecs
            && self.nvencs >= other.nvencs
    }

    pub fn to_dict(&self) -> std::collections::HashMap<String, f32> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert("cpu".to_string(), self.cpus);
        map.insert("gpu".to_string(), self.gpus);
        map.insert("nvdecs".to_string(), self.nvdecs);
        map.insert("nvencs".to_string(), self.nvencs);
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
    #[pyo3(get, set)]
    pub gpu_fraction: f32,
    // Not exposed as properties to Python; we provide helper methods instead
    nvdecs: HashSet<u8>,
    nvencs: HashSet<u8>,
}

#[pymethods]
impl GpuResources {
    #[new]
    pub fn new(
        index: u8,
        uuid_: uuid::Uuid,
        gpu_fraction: f32,
        nvdecs: Vec<u8>,
        nvencs: Vec<u8>,
    ) -> Self {
        Self {
            index,
            uuid_,
            gpu_fraction,
            nvdecs: nvdecs.into_iter().collect(),
            nvencs: nvencs.into_iter().collect(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuResources(index={}, uuid_={:?}, gpu_fraction={}, nvdecs={:?}, nvencs={:?})",
            self.index, self.uuid_, self.gpu_fraction, self.nvdecs, self.nvencs
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn make_from_num_codecs(
        index: u8,
        uuid_: uuid::Uuid,
        gpu_fraction_available: f32,
        num_nvdecs: u8,
        num_nvencs: u8,
    ) -> Self {
        let nvdecs: HashSet<u8> = (0..num_nvdecs).collect();
        let nvencs: HashSet<u8> = (0..num_nvencs).collect();
        Self {
            index,
            uuid_,
            gpu_fraction: gpu_fraction_available,
            nvdecs,
            nvencs,
        }
    }

    pub fn num_nvdecs(&self) -> usize {
        self.nvdecs.len()
    }

    pub fn num_nvencs(&self) -> usize {
        self.nvencs.len()
    }

    pub fn totals(&self) -> PoolOfResources {
        PoolOfResources {
            cpus: 0.0,
            gpus: self.gpu_fraction,
            nvdecs: self.num_nvdecs() as f32,
            nvencs: self.num_nvencs() as f32,
        }
    }
}

// --------------------
// GPUAllocation
// --------------------
/// Represents the allocation a worker is taking up for a given GPU.
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct GPUAllocation {
    #[pyo3(get, set)]
    pub gpu_index: usize,
    #[pyo3(get, set)]
    pub fraction: f32,
}

#[pymethods]
impl GPUAllocation {
    #[new]
    pub fn new(gpu_index: usize, fraction: f32) -> Self {
        Self {
            gpu_index,
            fraction,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GPUAllocation(gpu_index={}, fraction={})",
            self.gpu_index, self.fraction
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getnewargs__(&self) -> (usize, f32) {
        (self.gpu_index, self.fraction)
    }
}

// --------------------
// CodecAllocation
// --------------------
/// Represents the allocation a worker is taking up for a single hardware accelerated codec (NVDEC/NVENC).
#[pyclass]
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct CodecAllocation {
    #[pyo3(get, set)]
    pub gpu_index: usize,
    #[pyo3(get, set)]
    pub codec_index: usize,
}

#[pymethods]
impl CodecAllocation {
    #[new]
    pub fn new(gpu_index: usize, codec_index: usize) -> Self {
        Self {
            gpu_index,
            codec_index,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CodecAllocation(gpu_index={}, codec_index={})",
            self.gpu_index, self.codec_index
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getnewargs__(&self) -> (usize, usize) {
        (self.gpu_index, self.codec_index)
    }
}

// --------------------
// WorkerResources
// --------------------
/// Represents all the resources allocated to a single worker.
#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct WorkerResources {
    #[pyo3(get, set)]
    pub node: String,
    #[pyo3(get, set)]
    pub cpus: f32,
    #[pyo3(get, set)]
    pub gpus: Vec<GPUAllocation>,
    #[pyo3(get, set)]
    pub nvdecs: Vec<CodecAllocation>,
    #[pyo3(get, set)]
    pub nvencs: Vec<CodecAllocation>,
}

#[pymethods]
impl WorkerResources {
    #[new]
    pub fn new(
        node: String,
        cpus: f32,
        gpus: Option<Vec<GPUAllocation>>,
        nvdecs: Option<Vec<CodecAllocation>>,
        nvencs: Option<Vec<CodecAllocation>>,
    ) -> Self {
        Self {
            node,
            cpus,
            gpus: gpus.unwrap_or_default(),
            nvdecs: nvdecs.unwrap_or_default(),
            nvencs: nvencs.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkerResources(node={}, cpus={}, gpus={:?}, nvdecs={:?}, nvencs={:?})",
            self.node, self.cpus, self.gpus, self.nvdecs, self.nvencs
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getnewargs__(
        &self,
    ) -> (
        String,
        f32,
        Vec<GPUAllocation>,
        Vec<CodecAllocation>,
        Vec<CodecAllocation>,
    ) {
        (
            self.node.clone(),
            self.cpus,
            self.gpus.clone(),
            self.nvdecs.clone(),
            self.nvencs.clone(),
        )
    }

    pub fn validate(&self) {
        assert!(self.cpus >= 0.0);
        for gpu in &self.gpus {
            assert!(gpu.fraction >= 0.0);
        }
    }

    pub fn to_pool(&self) -> PoolOfResources {
        let gpu_sum: f32 = self.gpus.iter().map(|g| g.fraction).sum();
        let nvdec_count: f32 = self.nvdecs.len() as f32;
        let nvenc_count: f32 = self.nvencs.len() as f32;
        PoolOfResources {
            cpus: self.cpus,
            gpus: gpu_sum,
            nvdecs: nvdec_count,
            nvencs: nvenc_count,
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
    #[pyo3(get, set)]
    pub cpus: f32,
    #[pyo3(get, set)]
    pub gpus: Vec<GpuResources>,
    #[pyo3(get, set)]
    pub name: Option<String>,
}

#[pymethods]
impl NodeResources {
    #[new]
    pub fn new(cpus: f32, gpus: Option<Vec<GpuResources>>, name: Option<String>) -> Self {
        Self {
            cpus,
            gpus: gpus.unwrap_or_default(),
            name,
        }
    }

    /// Make a "uniform" node. I.e. all the nodes have the same number of nvdecs and nvencs.
    #[staticmethod]
    pub fn make_uniform(
        num_cpus: u32,
        num_gpus: u32,
        num_nvdecs_per_gpu: u8,
        num_nvencs_per_gpu: u8,
    ) -> Self {
        let mut gpus: Vec<GpuResources> = Vec::with_capacity(num_gpus as usize);
        for i in 0..num_gpus {
            gpus.push(GpuResources::make_from_num_codecs(
                i as u8,
                uuid::Uuid::new_v4(),
                1.0,
                num_nvdecs_per_gpu,
                num_nvencs_per_gpu,
            ));
        }
        Self {
            cpus: num_cpus as f32,
            gpus,
            name: None,
        }
    }

    pub fn totals(&self) -> PoolOfResources {
        let mut out = PoolOfResources {
            cpus: self.cpus,
            gpus: 0.0,
            nvdecs: 0.0,
            nvencs: 0.0,
        };
        for gpu in &self.gpus {
            out = out.add(&gpu.totals());
        }
        out
    }

    pub fn copy_and_allocate(&self, resources: &WorkerResources) -> Result<Self, AllocationError> {
        let mut c = self.clone();
        c.allocate(resources)?;
        Ok(c)
    }

    pub fn allocate(&mut self, resources: &WorkerResources) -> Result<(), AllocationError> {
        self.cpus -= resources.cpus;
        for gpu in &resources.gpus {
            if let Some(device) = self.gpus.get_mut(gpu.gpu_index) {
                device.gpu_fraction -= gpu.fraction;
            } else {
                return Err(AllocationError::GpuIndexOutOfRange {
                    gpu_index: gpu.gpu_index,
                });
            }
        }

        for x in &resources.nvdecs {
            let Some(device) = self.gpus.get_mut(x.gpu_index) else {
                return Err(AllocationError::GpuIndexOutOfRange {
                    gpu_index: x.gpu_index,
                });
            };
            let idx = x.codec_index as u8;
            if !device.nvdecs.contains(&idx) {
                return Err(AllocationError::NvdecUnavailable {
                    gpu_index: x.gpu_index,
                    available: device.nvdecs.iter().copied().collect(),
                });
            }
            device.nvdecs.remove(&idx);
        }

        for x in &resources.nvencs {
            let Some(device) = self.gpus.get_mut(x.gpu_index) else {
                return Err(AllocationError::GpuIndexOutOfRange {
                    gpu_index: x.gpu_index,
                });
            };
            let idx = x.codec_index as u8;
            if !device.nvencs.contains(&idx) {
                return Err(AllocationError::NvencUnavailable {
                    gpu_index: x.gpu_index,
                    available: device.nvencs.iter().copied().collect(),
                });
            }
            device.nvencs.remove(&idx);
        }
        Ok(())
    }

    pub fn release_allocation(&mut self, resources: &WorkerResources) {
        self.cpus += resources.cpus;
        for gpu in &resources.gpus {
            if let Some(device) = self.gpus.get_mut(gpu.gpu_index) {
                device.gpu_fraction += gpu.fraction;
            }
        }

        for x in &resources.nvdecs {
            if let Some(device) = self.gpus.get_mut(x.gpu_index) {
                let idx = x.codec_index as u8;
                device.nvdecs.insert(idx);
            }
        }

        for x in &resources.nvencs {
            if let Some(device) = self.gpus.get_mut(x.gpu_index) {
                let idx = x.codec_index as u8;
                device.nvencs.insert(idx);
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NodeResources(cpus={}, gpus=len({}), name={:?})",
            self.cpus,
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

    #[staticmethod]
    pub fn make_uniform(node_resources: &NodeResources, node_ids: Vec<String>) -> Self {
        let mut node_dict: std::collections::HashMap<String, NodeResources> = Default::default();
        for node_id in node_ids {
            node_dict.insert(node_id.clone(), node_resources.clone());
        }
        Self { nodes: node_dict }
    }

    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[getter]
    pub fn num_gpus(&self) -> usize {
        let mut out: usize = 0;
        for node in self.nodes.values() {
            out += node.gpus.len();
        }
        out
    }

    #[getter]
    pub fn num_cpus(&self) -> f32 {
        self.nodes.values().map(|n| n.cpus).sum()
    }

    pub fn calc_num_nvdecs_per_gpu(&self) -> u8 {
        let mut per_gpu: HashSet<u8> = HashSet::new();
        for node in self.nodes.values() {
            for gpu in &node.gpus {
                per_gpu.insert(gpu.num_nvdecs() as u8);
            }
        }
        if per_gpu.is_empty() {
            0
        } else {
            assert_eq!(per_gpu.len(), 1);
            *per_gpu.iter().next().unwrap()
        }
    }

    pub fn calc_num_nvencs_per_gpu(&self) -> u8 {
        let mut per_gpu: HashSet<u8> = HashSet::new();
        for node in self.nodes.values() {
            for gpu in &node.gpus {
                per_gpu.insert(gpu.num_nvencs() as u8);
            }
        }
        if per_gpu.is_empty() {
            0
        } else {
            assert_eq!(per_gpu.len(), 1);
            *per_gpu.iter().next().unwrap()
        }
    }

    pub fn totals(&self) -> PoolOfResources {
        let mut out = PoolOfResources::default();
        for node in self.nodes.values() {
            out = out.add(&node.totals());
        }
        out
    }

    pub fn is_overallocated(&self) -> bool {
        for node in self.nodes.values() {
            if super::approx_utils::float_lt(node.cpus.into(), 0.0, super::approx_utils::EPSILON) {
                return true;
            }
            for gpu in &node.gpus {
                if super::approx_utils::float_lt(
                    gpu.gpu_fraction.into(),
                    0.0,
                    super::approx_utils::EPSILON,
                ) {
                    return true;
                }
            }
        }
        false
    }

    pub fn make_overallocated_message(&self, totals: &ClusterResources) -> String {
        let mut out: Vec<String> = Vec::new();
        for (node_id, node) in &self.nodes {
            if super::approx_utils::float_lt(node.cpus.into(), 0.0, super::approx_utils::EPSILON) {
                let total_cpus = totals
                    .nodes
                    .get(node_id)
                    .map(|n| n.cpus)
                    .unwrap_or(f32::NAN);
                out.push(format!(
                    "Node {node_id} is overallocated by {:.2} cpus (has capacity for {total_cpus})",
                    -node.cpus
                ));
            }
            for (gpu_idx, gpu) in node.gpus.iter().enumerate() {
                if super::approx_utils::float_lt(
                    gpu.gpu_fraction.into(),
                    0.0,
                    super::approx_utils::EPSILON,
                ) {
                    let total_gpus = totals
                        .nodes
                        .get(node_id)
                        .and_then(|n| n.gpus.get(gpu_idx))
                        .map(|g| g.gpu_fraction)
                        .unwrap_or(f32::NAN);
                    out.push(format!(
                        "Node {node_id} gpu {gpu_idx} is overallocated by {:.2} (has capacity for {total_gpus})",
                        -gpu.gpu_fraction
                    ));
                }
            }
        }
        out.join(", ")
    }

    pub fn copy_and_clear_allocation(
        &self,
        resources: &WorkerResources,
    ) -> Result<Self, AllocationError> {
        let mut c = self.clone();
        c.clear_allocation(resources)?;
        Ok(c)
    }

    pub fn clear_allocation(&mut self, resources: &WorkerResources) -> Result<(), AllocationError> {
        let Some(node) = self.nodes.get_mut(&resources.node) else {
            return Err(AllocationError::NodeNotFound(resources.node.clone()));
        };
        node.cpus += resources.cpus;
        for gpu in &resources.gpus {
            if let Some(device) = node.gpus.get_mut(gpu.gpu_index) {
                device.gpu_fraction += gpu.fraction;
            }
        }

        for x in &resources.nvdecs {
            if let Some(device) = node.gpus.get_mut(x.gpu_index) {
                let idx = x.codec_index as u8;
                device.nvdecs.insert(idx);
            }
        }

        for x in &resources.nvencs {
            if let Some(device) = node.gpus.get_mut(x.gpu_index) {
                let idx = x.codec_index as u8;
                device.nvencs.insert(idx);
            }
        }
        Ok(())
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
#[derive(Debug, PartialEq, Clone)]
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
                cpus: 1.0,
                gpus: vec![],
                nvdecs: vec![],
                nvencs: vec![],
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
        .add_class::<CodecAllocation>()?
        .add_class::<WorkerMetadata>()?
        .add_class::<NodeInfo>()?
        .add_class::<WorkerShapeWrapper>()?
        .add_class::<PoolOfResources>()?
        .add_class::<CpuOnly>()?
        .add_class::<Codec>()?
        .add_class::<FractionalGpu>()?
        .add_class::<WholeNumberedGpu>()?
        .add_class::<EntireGpu>()?
        .add_class::<NodeInfo>()?
        .add_class::<WorkerMetadata>()?
        .finish();
    Ok(())
}