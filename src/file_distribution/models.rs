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

use object_store::ObjectStore;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use url::Url;
use uuid::Uuid;

use crate::utils::module_builders::ImportablePyModuleBuilder;

#[derive(Error, Debug)]
pub enum UnpackerError {
    #[error("Unsupported unpack method: {0}")]
    UnsupportedUnpackMethod(String),
}

#[derive(Debug)]
pub struct XennaError(object_store::Error);

impl From<object_store::Error> for XennaError {
    fn from(err: object_store::Error) -> Self {
        Self(err)
    }
}

impl From<XennaError> for PyErr {
    fn from(err: XennaError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.0.to_string())
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnpackMethod {
    Auto,
    Tar,
    TarGz,
    Zip,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnpackOptions {
    pub destination: PathBuf,
    pub unpack_method: UnpackMethod,
}

#[pymethods]
impl UnpackOptions {
    #[new]
    fn new(destination: PathBuf, unpack_method: UnpackMethod) -> Self {
        Self {
            destination,
            unpack_method,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectToDownload {
    pub object_id: Uuid,
    pub parent_request_id: Uuid,
    pub profile_name: Option<String>,
    pub uri: String,
    pub destination: PathBuf,
    pub cache_info: CacheInfo,
    pub unpack_options: Option<UnpackOptions>,
}

#[pymethods]
impl ObjectToDownload {
    #[new]
    fn new(
        object_id: Uuid,
        parent_request_id: Uuid,
        profile_name: Option<String>,
        uri: String,
        destination: PathBuf,
        cache_info: CacheInfo,
        unpack_options: Option<UnpackOptions>,
    ) -> Self {
        Self {
            object_id,
            parent_request_id,
            profile_name,
            uri,
            destination,
            cache_info,
            unpack_options,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadCatalog {
    pub objects: HashMap<Uuid, ObjectToDownload>,
    pub chunks: HashMap<Uuid, ChunkToDownload>,
    pub chunks_by_object: HashMap<Uuid, Vec<Uuid>>,
}

#[pymethods]
impl DownloadCatalog {
    #[new]
    fn new(
        objects: Vec<ObjectToDownload>,
        chunks: Vec<ChunkToDownload>,
        chunks_by_object: HashMap<Uuid, Vec<Uuid>>,
    ) -> Self {
        let mut objects_map = HashMap::new();
        for object in objects {
            objects_map.insert(object.object_id, object);
        }
        let mut chunks_map = HashMap::new();
        for chunk in chunks {
            chunks_map.insert(chunk.chunk_id, chunk);
        }
        Self {
            objects: objects_map,
            chunks: chunks_map,
            chunks_by_object,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ByteRange {
    // Starting byte position (inclusive)
    pub start: u64,
    // Ending byte position (exclusive)
    pub end: u64,
}

#[pymethods]
impl ByteRange {
    #[new]
    fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectAndRange {
    pub object_uri: String,
    pub range: Option<ByteRange>,
    pub crc32_checksum: Option<u32>,
}

#[pymethods]
impl ObjectAndRange {
    #[new]
    fn new(object_uri: String, range: Option<ByteRange>, crc32_checksum: Option<u32>) -> Self {
        Self {
            object_uri,
            range,
            crc32_checksum,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkToDownload {
    pub chunk_id: Uuid,
    pub parent_object_id: Uuid,
    pub profile_name: Option<String>,
    pub value: ObjectAndRange,
    pub destination: PathBuf,
    pub size: u64,
}

#[pymethods]
impl ChunkToDownload {
    #[new]
    fn new(
        chunk_id: Uuid,
        parent_object_id: Uuid,
        profile_name: Option<String>,
        value: ObjectAndRange,
        destination: PathBuf,
        size: u64,
    ) -> Self {
        Self {
            chunk_id,
            parent_object_id,
            profile_name,
            value,
            destination,
            size,
        }
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct DownloadFromNodeOrder {
    pub download_chunk: ChunkToDownload,
    pub source_node_id: String,
    pub source_node_address: SocketAddr,
}

impl DownloadFromNodeOrder {
    fn new(
        download_chunk: ChunkToDownload,
        source_node_id: String,
        source_node_address: SocketAddr,
    ) -> Self {
        Self {
            download_chunk,
            source_node_id,
            source_node_address,
        }
    }
}

#[pymethods]
impl DownloadFromNodeOrder {
    #[new]
    fn py_init(
        download_chunk: ChunkToDownload,
        source_node_id: String,
        source_node_ip: String,
        source_node_port: u16,
    ) -> Self {
        Self::new(
            download_chunk,
            source_node_id,
            SocketAddr::new(
                source_node_ip
                    .parse()
                    .expect(&format!("Invalid IP address {}", source_node_ip)),
                source_node_port,
            ),
        )
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Orders {
    pub download_from_s3: Vec<ChunkToDownload>,
    pub download_from_node: Vec<DownloadFromNodeOrder>,
}

#[pymethods]
impl Orders {
    #[new]
    fn new(
        download_from_s3: Vec<ChunkToDownload>,
        download_from_node: Vec<DownloadFromNodeOrder>,
    ) -> Self {
        Self {
            download_from_s3,
            download_from_node,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ObjectMetadata {
    pub size: u64,
    pub last_modified: String,
}

#[pymethods]
impl ObjectMetadata {
    #[new]
    fn new(size: u64, last_modified: String) -> Self {
        Self {
            size,
            last_modified,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ObjectNameAndMetadata {
    pub uri: String,
    pub metadata: ObjectMetadata,
}

#[pymethods]
impl ObjectNameAndMetadata {
    #[new]
    fn new(uri: String, metadata: ObjectMetadata) -> Self {
        Self { uri, metadata }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct NodeStatus {
    pub node_id: String,
    pub downloading_p2p_chunks: HashSet<Uuid>,
    pub downloading_s3_chunks: HashSet<Uuid>,
    pub writing_chunks: HashSet<Uuid>,
    pub available_chunks: HashSet<Uuid>,
    pub completed_or_cached_objects: HashSet<Uuid>,
    pub unneeded_objects: HashSet<Uuid>,
    pub num_active_uploads: usize,
    pub num_active_file_writing_tasks: usize,
    pub num_active_assembling_tasks: usize,
    pub num_active_unpacking_tasks: usize,
}

impl NodeStatus {
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            ..Default::default()
        }
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CacheInfo {
    pub uri: String,
    pub size: u64,
    pub last_modified_unix_micros: u64,
}

impl CacheInfo {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(data: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(data)
    }

    pub fn get_cache_path_for_file(file_path: PathBuf) -> PathBuf {
        let mut cache_path = file_path.as_os_str().to_owned();
        cache_path.push(".s3_cache_info");
        PathBuf::from(cache_path)
    }

    pub fn get_cache_path_for_directory(directory_path: PathBuf) -> PathBuf {
        directory_path.join(".s3_cache_info")
    }
}

#[pymethods]
impl CacheInfo {
    #[new]
    fn new(uri: String, size: u64, last_modified_unix_micros: u64) -> Self {
        Self {
            uri,
            size,
            last_modified_unix_micros: last_modified_unix_micros,
        }
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NodeMetadata {
    pub node_id: String,
    pub address: SocketAddr,
}

#[pymethods]
impl NodeMetadata {
    #[getter]
    fn node_id(&self) -> String {
        self.node_id.clone()
    }

    #[getter]
    fn ip(&self) -> String {
        self.address.ip().to_string()
    }

    #[getter]
    fn port(&self) -> u16 {
        self.address.port()
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RetryConfig {
    pub num_retries: u32,
    pub base_delay_millis: u64,
}

#[pymethods]
impl RetryConfig {
    #[new]
    fn new(num_retries: u32, base_delay_millis: u64) -> Self {
        Self {
            num_retries,
            base_delay_millis,
        }
    }
}

#[pyclass(get_all, set_all)]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ObjectStoreConfig {
    pub uri: String,
    pub config_args: HashMap<String, String>,
}

#[pymethods]
impl ObjectStoreConfig {
    #[new]
    fn new(uri: String, config_args: HashMap<String, String>) -> Self {
        Self { uri, config_args }
    }
}
#[pyclass]
#[derive(Debug, Clone)]
pub struct ObjectStoreConfigByProfile {
    pub profiles: HashMap<Option<String>, ObjectStoreConfig>,
}

#[pymethods]
impl ObjectStoreConfigByProfile {
    #[new]
    fn new(profiles: HashMap<Option<String>, ObjectStoreConfig>) -> Self {
        Self { profiles }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ObjectStoreByProfile {
    pub profiles: HashMap<Option<String>, Arc<dyn ObjectStore>>,
}

fn make_object_store(
    config: ObjectStoreConfig,
) -> Result<Arc<dyn ObjectStore>, object_store::Error> {
    let url = Url::parse(&config.uri).expect("Failed to parse url");
    if url.scheme() == "file" {
        let store =
            object_store::local::LocalFileSystem::new_with_prefix(&PathBuf::from(url.path()))?;
        return Ok(Arc::new(store));
    }

    // Add timeout configurations for better handling of large chunks
    let mut enhanced_config = config.config_args.clone();

    // Set longer timeouts for large chunk downloads if not already specified
    enhanced_config
        .entry("timeout".to_string())
        .or_insert_with(|| "300s".to_string()); // 5 minutes
    enhanced_config
        .entry("connect_timeout".to_string())
        .or_insert_with(|| "60s".to_string()); // 1 minute

    let (store, _) = object_store::parse_url_opts(&url, enhanced_config)?;
    Ok(Arc::new(store))
}

#[pymethods]
impl ObjectStoreByProfile {
    #[new]
    fn new(config_by_profile: ObjectStoreConfigByProfile) -> Result<Self, XennaError> {
        let mut profiles: HashMap<Option<String>, Arc<dyn ObjectStore>> = HashMap::new();
        for (profile_name, config) in config_by_profile.profiles {
            profiles.insert(profile_name.clone(), make_object_store(config)?);
        }
        Ok(Self { profiles })
    }
}

impl ObjectStoreByProfile {
    pub fn get_client(&self, profile_name: Option<&str>) -> Arc<dyn ObjectStore> {
        // Convert Option<&str> to Option<String> for lookup
        let key: Option<String> = profile_name.map(|s| s.to_string());
        self.profiles
            .get(&key)
            .unwrap_or_else(|| {
                panic!(
                    "Profile not found: '{:?}'. Available profiles: {:?}",
                    profile_name,
                    self.profiles.keys()
                )
            })
            .clone()
    }
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<ByteRange>()?
        .add_class::<ObjectAndRange>()?
        .add_class::<ChunkToDownload>()?
        .add_class::<DownloadFromNodeOrder>()?
        .add_class::<Orders>()?
        .add_class::<ObjectMetadata>()?
        .add_class::<ObjectNameAndMetadata>()?
        .add_class::<ObjectStoreConfig>()?
        .add_class::<ObjectStoreByProfile>()?
        .add_class::<DownloadCatalog>()?
        .add_class::<NodeStatus>()?
        .add_class::<ObjectStoreConfigByProfile>()?
        .add_class::<ObjectToDownload>()?
        .add_class::<UnpackOptions>()?
        .add_class::<UnpackMethod>()?
        .add_class::<CacheInfo>()?
        .add_class::<NodeMetadata>()?
        .add_class::<RetryConfig>()?
        .finish();
    Ok(())
}
