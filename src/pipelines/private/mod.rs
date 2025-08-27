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

pub mod scheduling;

use pyo3::prelude::*;

use crate::utils::module_builders::ImportablePyModuleBuilder;

/// Module initialization
pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let scheduling_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.scheduling", m.name().unwrap()))?;
    scheduling::register_module(py, &scheduling_module.as_module())?;
    let scheduling = scheduling_module.finish();

    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_submodule(&scheduling)?
        .finish();
    Ok(())
}