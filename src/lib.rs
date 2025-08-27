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

pub mod file_distribution;
pub mod pipelines;
pub mod utils;

use crate::utils::module_builders::ImportablePyModuleBuilder;
use env_logger::Env;
use log::debug;
use pyo3::prelude::*;

#[pyfunction]
pub fn setup_logging() {
    if let Err(e) =
        env_logger::Builder::from_env(Env::default().default_filter_or("warn")).try_init()
    {
        debug!("Failed to initialize logger: {}", e);
    }
}

// https://github.com/gregorycarnegie/autocrop/blob/a0a22aaa86db344efd36bc99643fa7d555a7cf3c/src/lib.rs
#[pymodule]
fn _cosmos_xenna(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let name = m.name()?.extract::<String>()?;
    // Create submodules
    let pipelines_module = ImportablePyModuleBuilder::new(py, &format!("{name}.pipelines"))?;
    pipelines::register_module(py, &pipelines_module.as_module())?;
    let pipelines = pipelines_module.finish();

    let file_distribution_module =
        ImportablePyModuleBuilder::new(py, &format!("{name}.file_distribution"))?;
    file_distribution::register_module(py, &file_distribution_module.as_module())?;
    let file_distribution = file_distribution_module.finish();

    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_submodule(&pipelines)?
        .add_submodule(&file_distribution)?
        .add_function(wrap_pyfunction!(setup_logging, m)?)?
        .finish();

    Ok(())
}