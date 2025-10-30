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

pub mod allocator;
pub mod approx_utils;
pub mod autoscaling_algorithms;
pub mod data_structures;
pub mod fragmentation_allocation_algorithms;
pub mod naiive_worker_allocation;
pub mod resources;

use pyo3::prelude::*;

use crate::utils::module_builders::ImportablePyModuleBuilder;

/// Module initialization
pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let resources_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.resources", m.name().unwrap()))?;
    resources::register_module(py, &resources_module.as_module())?;
    let resources = resources_module.finish();

    let allocator_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.allocator", m.name().unwrap()))?;
    allocator::register_module(py, &allocator_module.as_module())?;
    let allocator = allocator_module.finish();

    let autoscaling_algorithms_module = ImportablePyModuleBuilder::new(
        py,
        &format!("{}.autoscaling_algorithms", m.name().unwrap()),
    )?;
    autoscaling_algorithms::register_module(py, &autoscaling_algorithms_module.as_module())?;
    let autoscaling_algorithms = autoscaling_algorithms_module.finish();

    let data_structures_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.data_structures", m.name().unwrap()))?;
    data_structures::register_module(py, &data_structures_module.as_module())?;
    let data_structures = data_structures_module.finish();

    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_submodule(&resources)?
        .add_submodule(&allocator)?
        .add_submodule(&autoscaling_algorithms)?
        .add_submodule(&data_structures)?
        .finish();
    Ok(())
}
