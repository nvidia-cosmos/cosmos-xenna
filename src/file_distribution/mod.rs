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

pub mod assembler;
pub mod common;
pub mod data_plane;
pub mod file_writer;
pub mod models;
pub mod object_store_download;
pub mod p2p_download;
pub mod p2p_server;
pub mod unpacker;

use pyo3::prelude::*;

#[pymodule]
pub fn file_distribution(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(common::get_temp_chunk_path, m)?)?;
    m.add_function(wrap_pyfunction!(common::resolve_path, m)?)?;
    m.add_function(wrap_pyfunction!(common::get_cache_path_for_file, m)?)?;
    m.add_function(wrap_pyfunction!(common::get_cache_path_for_directory, m)?)?;
    m.add_class::<models::ByteRange>()?;
    m.add_class::<models::ObjectAndRange>()?;
    m.add_class::<models::ChunkToDownload>()?;
    m.add_class::<models::DownloadFromNodeOrder>()?;
    m.add_class::<models::Orders>()?;
    m.add_class::<models::ObjectMetadata>()?;
    m.add_class::<models::ObjectNameAndMetadata>()?;
    m.add_class::<data_plane::DataPlane>()?;
    m.add_class::<models::ObjectStoreConfig>()?;
    m.add_class::<models::ObjectStoreByProfile>()?;
    m.add_class::<models::DownloadCatalog>()?;
    m.add_class::<models::NodeStatus>()?;
    m.add_class::<models::ObjectStoreConfigByProfile>()?;
    m.add_class::<models::ObjectToDownload>()?;
    m.add_class::<models::UnpackOptions>()?;
    m.add_class::<models::UnpackMethod>()?;
    m.add_class::<models::CacheInfo>()?;
    m.add_class::<models::NodeMetadata>()?;
    Ok(())
}

use crate::utils::module_builders::ImportablePyModuleBuilder;

/// Module initialization
pub fn register_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let common_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.common", m.name().unwrap()))?;
    common::register_module(py, &common_module.as_module())?;
    let common = common_module.finish();

    let data_plane_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.data_plane", m.name().unwrap()))?;
    data_plane::register_module(py, &data_plane_module.as_module())?;
    let data_plane = data_plane_module.finish();

    let models_module =
        ImportablePyModuleBuilder::new(py, &format!("{}.models", m.name().unwrap()))?;
    models::register_module(py, &models_module.as_module())?;
    let models = models_module.finish();

    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_submodule(&common)?
        .add_submodule(&data_plane)?
        .add_submodule(&models)?
        .finish();
    Ok(())
}
