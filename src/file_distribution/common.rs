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

use pyo3::prelude::*;
use std::path::PathBuf;
use uuid::Uuid;

use crate::file_distribution::models::CacheInfo;
use crate::utils::module_builders::ImportablePyModuleBuilder;

pub const TMP_CHUNK_DIR: &str = "/tmp/chunks";
pub const P2P_DOWNLOAD_TEST_DIR: &str = "/tmp/p2p_download_test";

#[pyfunction]
pub fn resolve_path(original_path: PathBuf, node_id: &str, is_test: bool) -> PathBuf {
    if !is_test {
        return original_path.to_path_buf();
    }

    let storage_root = PathBuf::from(P2P_DOWNLOAD_TEST_DIR).join(node_id);
    let relative_path = original_path.strip_prefix("/").unwrap_or(&original_path);
    storage_root.join(relative_path)
}

#[pyfunction]
pub fn get_temp_chunk_path(chunk_id: Uuid, node_id: &str, is_test: bool) -> PathBuf {
    let temp_dir = resolve_path(PathBuf::from(TMP_CHUNK_DIR), node_id, is_test);
    std::fs::create_dir_all(&temp_dir).unwrap();
    temp_dir.join(format!("{}.chunk", chunk_id))
}

#[pyfunction]
pub fn get_cache_path_for_file(file_path: PathBuf) -> PathBuf {
    CacheInfo::get_cache_path_for_file(file_path)
}

#[pyfunction]
pub fn get_cache_path_for_directory(file_path: PathBuf) -> PathBuf {
    CacheInfo::get_cache_path_for_directory(file_path)
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_function(wrap_pyfunction!(resolve_path, m)?)?
        .add_function(wrap_pyfunction!(get_temp_chunk_path, m)?)?
        .add_function(wrap_pyfunction!(get_cache_path_for_file, m)?)?
        .add_function(wrap_pyfunction!(get_cache_path_for_directory, m)?)?
        .finish();
    Ok(())
}