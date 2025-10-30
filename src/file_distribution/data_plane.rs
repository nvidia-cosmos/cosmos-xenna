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

//! # P2P Data Plane
//!
//! This module defines the core data plane logic for the Xenna P2P file distribution system.
//! It is responsible for orchestrating the download of file chunks from peer nodes.
//!
//! ## Architecture
//!
//! The `P2pDownloaderWorkerPool` is the central component of this module. It manages a pool of
//! worker threads that are responsible for executing download tasks. This design allows for
//! concurrent chunk downloads, maximizing throughput.
//!
//! Following the architecture outlined in `README.md`, this data plane is designed to be driven
//! by a higher-level control plane (in Python). The control plane submits download tasks, and this
//! module executes them, handling the complexities of network requests, retries, and temporary
//! file management.
//!
//! ## Operations
//!
//! - **Task Execution:** The worker pool receives `P2pDownloadTask` items, which contain all the
//!   necessary information to download a chunk (e.g., chunk ID, peer address, destination).
//!
//! - **Downloading:** Each worker uses a `reqwest` client to make HTTP GET requests to peer P2P
//!   servers. The `download_chunk` function handles the specifics of this request.
//!
//! - **Temporary Storage:** A key design principle is that this module downloads chunks into a
//!   temporary directory, as managed by `get_temp_chunk_path`. It does **not** perform file
//!   assembly. The final assembly of chunks into the target file is the responsibility of a
//!   separate "assembler" component, which is managed by the control plane. This separation of
//!   concerns keeps the data plane focused on high-performance data transfer.
//!
//! - **Retries:** The system includes a retry mechanism with an exponential backoff strategy to
//!   handle transient network errors gracefully.
use crate::file_distribution::assembler::{
    AssemblerError, AssemblerPool, AssemblerTask, TaskStatus as AssemblerTaskStatus,
};
use crate::file_distribution::common::resolve_path;
use crate::file_distribution::file_writer::{
    FileWriterError, FileWriterPool, FileWriterTask, TaskStatus as FileWriterTaskStatus,
};
use crate::file_distribution::object_store_download::{
    DownloadError, ObjectStoreDownloadTask, ObjectStoreDownloader, TaskStatus as OsTaskStatus,
};
use crate::file_distribution::p2p_download::{
    DownloadError as P2pDownloadError, P2pDownloadTask, P2pDownloaderWorkerPool,
    TaskStatus as P2pTaskStatus,
};
use crate::file_distribution::p2p_server::{P2pServer, P2pServerError};
use crate::file_distribution::unpacker::{
    TaskStatus as UnpackerTaskStatus, UnpackerError, UnpackerPool, UnpackerTask,
};

use crate::file_distribution::models::{
    CacheInfo, DownloadCatalog, NodeStatus, ObjectStoreByProfile, Orders, RetryConfig,
};
use crate::utils::module_builders::ImportablePyModuleBuilder;
use log::{debug, warn};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("Orchestrator failed: {0}")]
    Orchestrator(#[from] std::io::Error),
    #[error("Object store download failed: {0}")]
    ObjectStoreDownloadFailed(DownloadError),
    #[error("P2P download failed: {0}")]
    P2pDownloadFailed(P2pDownloadError),
    #[error("File writer failed: {0}")]
    FileWriterFailed(FileWriterError),
    #[error("Assembler failed: {0}")]
    AssemblerFailed(AssemblerError),
    #[error("Unpacker failed: {0}")]
    UnpackerFailed(UnpackerError),
    #[error("P2P server failed: {0}")]
    P2pServerFailed(#[from] P2pServerError),
}

impl std::convert::From<OrchestratorError> for PyErr {
    fn from(err: OrchestratorError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedThings {
    cached_chunks: HashSet<Uuid>,
    objects_no_unpacking: HashSet<Uuid>,
    objects_needed_to_be_unpacked: HashSet<Uuid>,
    objects_already_unpacked: HashSet<Uuid>,
    objects_already_unpacked_and_no_object_present: HashSet<Uuid>,
}

fn find_cached_things_and_remove_invalid(
    download_catalog: &DownloadCatalog,
    node_id: &str,
    is_test: bool,
) -> Result<CachedThings, std::io::Error> {
    let mut cached_objects = HashSet::new();

    for obj in download_catalog.objects.values() {
        let resolved_path = resolve_path(obj.destination.clone(), node_id, is_test);
        let cache_info_path = CacheInfo::get_cache_path_for_file(resolved_path.clone());

        let resolved_path_exists = resolved_path.exists();
        let cache_info_path_exists = cache_info_path.exists();

        if !resolved_path_exists && !cache_info_path_exists {
            continue;
        } else if !resolved_path_exists && cache_info_path_exists {
            warn!(
                "Cache info found for {:?}, but no file exists. Deleting cache info.",
                resolved_path
            );
            std::fs::remove_file(cache_info_path)?;
        } else if resolved_path_exists && !cache_info_path_exists {
            warn!(
                "File found at {:?}, but no cache info. Deleting file.",
                resolved_path
            );
            std::fs::remove_file(resolved_path)?;
        } else {
            // Both exist
            let cache_info_content = std::fs::read_to_string(&cache_info_path)?;
            if let Ok(cache_info) = CacheInfo::from_json(&cache_info_content) {
                let valid_uri = cache_info.uri == obj.cache_info.uri;
                let valid_size = cache_info.size == obj.cache_info.size;
                let valid_last_modified_unix_micros = cache_info.last_modified_unix_micros
                    == obj.cache_info.last_modified_unix_micros;
                if valid_uri && valid_size && valid_last_modified_unix_micros {
                    cached_objects.insert(obj.object_id);
                } else {
                    warn!("Out of date cache for {}. Deleting cached data.", obj.uri);
                    debug!(
                        "Invalid reason: uri: {}, size: {}, last_modified_unix_micros: {}",
                        valid_uri, valid_size, valid_last_modified_unix_micros
                    );
                    warn!("cache_info: {:?}", cache_info);
                    warn!("obj.cache_info: {:?}", obj.cache_info);
                    std::fs::remove_file(&cache_info_path)?;
                    std::fs::remove_file(resolved_path)?;
                }
            } else {
                warn!(
                    "Cache validation failed for {}. Deleting cache info.",
                    obj.uri
                );
                std::fs::remove_file(&cache_info_path)?;
                std::fs::remove_file(resolved_path)?;
            }
        }
    }

    let mut objects_needed_to_be_unpacked = HashSet::new();
    let mut objects_already_unpacked = HashSet::new();
    let mut objects_already_unpacked_and_no_object_present = HashSet::new();

    for obj in download_catalog.objects.values() {
        if let Some(unpack_options) = &obj.unpack_options {
            let unpacked_path = resolve_path(unpack_options.destination.clone(), node_id, is_test);
            let unpacked_cache_info_path =
                CacheInfo::get_cache_path_for_directory(unpacked_path.clone());

            let has_valid_object_cache = cached_objects.contains(&obj.object_id);
            let mut has_valid_unpacked_cache = false;

            let unpacked_path_exists = unpacked_path.exists();
            let unpacked_cache_info_path_exists = unpacked_cache_info_path.exists();

            if unpacked_path_exists && unpacked_cache_info_path_exists {
                let cache_info_content = std::fs::read_to_string(&unpacked_cache_info_path)?;
                if let Ok(cache_info) = CacheInfo::from_json(&cache_info_content) {
                    if cache_info.uri == obj.cache_info.uri
                        && cache_info.size == obj.cache_info.size
                        && cache_info.last_modified_unix_micros
                            == obj.cache_info.last_modified_unix_micros
                    {
                        has_valid_unpacked_cache = true;
                    } else {
                        log::warn!("Out of date cache for {}. Deleting unpacked data.", obj.uri);
                        std::fs::remove_file(&unpacked_cache_info_path)?;
                        std::fs::remove_dir_all(&unpacked_path)?;
                    }
                } else {
                    log::warn!(
                        "Cache validation failed for {}. Deleting unpacked data.",
                        obj.uri
                    );
                    std::fs::remove_file(&unpacked_cache_info_path)?;
                    std::fs::remove_dir_all(&unpacked_path)?;
                }
            } else if !unpacked_path_exists && unpacked_cache_info_path_exists {
                log::warn!(
                    "Cache info found for {:?}, but no file exists. Deleting cache info.",
                    unpacked_path
                );
                std::fs::remove_file(unpacked_cache_info_path)?;
            } else if unpacked_path_exists && !unpacked_cache_info_path_exists {
                log::warn!(
                    "File found at {:?}, but no cache info. Deleting file.",
                    unpacked_path
                );
                std::fs::remove_dir_all(unpacked_path)?;
            }

            if has_valid_object_cache && has_valid_unpacked_cache {
                debug!(
                    "Object {} is already unpacked and has a valid cache.",
                    obj.object_id
                );
                objects_already_unpacked.insert(obj.object_id);
            } else if has_valid_object_cache && !has_valid_unpacked_cache {
                debug!("Object {} is needed to be unpacked.", obj.object_id);
                objects_needed_to_be_unpacked.insert(obj.object_id);
            } else if !has_valid_object_cache && has_valid_unpacked_cache {
                debug!(
                    "Object {} is already unpacked and no object present.",
                    obj.object_id
                );
                objects_already_unpacked_and_no_object_present.insert(obj.object_id);
            }
        }
    }

    let mut cached_chunks = HashSet::new();
    for chunk in download_catalog.chunks.values() {
        if cached_objects.contains(&chunk.parent_object_id) {
            cached_chunks.insert(chunk.chunk_id);
        }
    }

    let objects_no_unpacking = cached_objects
        .difference(&objects_already_unpacked)
        .cloned()
        .collect::<HashSet<_>>()
        .difference(&objects_needed_to_be_unpacked)
        .cloned()
        .collect();

    Ok(CachedThings {
        cached_chunks,
        objects_no_unpacking,
        objects_needed_to_be_unpacked,
        objects_already_unpacked,
        objects_already_unpacked_and_no_object_present,
    })
}

struct AssemblyTracker {
    remaining_chunks_for_assembly: HashMap<Uuid, HashSet<Uuid>>,
}

impl AssemblyTracker {
    fn new(download_catalog: &DownloadCatalog, cached_things: &CachedThings) -> Self {
        let assembled_objects = cached_things
            .objects_no_unpacking
            .iter()
            .chain(cached_things.objects_needed_to_be_unpacked.iter())
            .chain(cached_things.objects_already_unpacked.iter())
            .cloned()
            .collect::<HashSet<_>>();

        let remaining_chunks_for_assembly = download_catalog
            .chunks_by_object
            .iter()
            .filter(|(object_id, _)| !assembled_objects.contains(object_id))
            .map(|(object_id, chunks)| (*object_id, chunks.iter().cloned().collect()))
            .collect();

        Self {
            remaining_chunks_for_assembly,
        }
    }

    /// Takes in newly written chunks and returns objects that are ready to be assembled.
    fn add_written_chunks(
        &mut self,
        download_catalog: &DownloadCatalog,
        newly_written_chunks: &HashSet<Uuid>,
    ) -> HashSet<Uuid> {
        let mut objects_ready_for_assembly = HashSet::new();

        for chunk_id in newly_written_chunks {
            if let Some(chunk) = download_catalog.chunks.get(chunk_id) {
                if let Some(remaining_chunks) = self
                    .remaining_chunks_for_assembly
                    .get_mut(&chunk.parent_object_id)
                {
                    if remaining_chunks.remove(chunk_id) && remaining_chunks.is_empty() {
                        objects_ready_for_assembly.insert(chunk.parent_object_id);
                    }
                }
            } else {
                warn!(
                    "Newly written chunk {} not found in download catalog",
                    chunk_id
                );
            }
        }
        objects_ready_for_assembly
    }

    /// Schedules an object for assembly and returns an assembler task.
    /// It will return None if the object has already been scheduled for assembly.
    fn schedule_assembly_task(
        &mut self,
        download_catalog: &DownloadCatalog,
        object_id: Uuid,
    ) -> Option<AssemblerTask> {
        if self
            .remaining_chunks_for_assembly
            .remove(&object_id)
            .is_some()
        {
            let chunks_needed_for_object =
                download_catalog.chunks_by_object.get(&object_id).unwrap();
            let object_to_download = download_catalog.objects.get(&object_id).unwrap();
            let assembler_task = AssemblerTask {
                object_id,
                chunk_ids: chunks_needed_for_object.clone(),
                destination: object_to_download.destination.clone(),
                cache_info: object_to_download.cache_info.clone(),
            };
            Some(assembler_task)
        } else {
            None
        }
    }
}

/// Orchestrator that manages all file distribution operations.
/// This runs synchronously in the same thread as the control plane's update() call.
pub struct Orchestrator {
    node_id: String,
    download_catalog: DownloadCatalog,
    p2p_server: P2pServer,

    // Worker pools
    object_downloader: ObjectStoreDownloader,
    p2p_downloader: P2pDownloaderWorkerPool,
    file_writer: FileWriterPool,
    assembler: AssemblerPool,
    unpacker: UnpackerPool,

    // Active task tracking
    os_download_active_uuids: HashSet<Uuid>,
    p2p_download_active_uuids: HashSet<Uuid>,
    file_writer_active_uuids: HashSet<Uuid>,
    assembler_active_uuids: HashSet<Uuid>,
    unpacker_active_uuids: HashSet<Uuid>,

    // State tracking
    cached_things: CachedThings,
    assembly_tracker: AssemblyTracker,
    available_chunks: HashSet<Uuid>,
    completed_or_cached_objects: HashSet<Uuid>,
}

impl Orchestrator {
    /// Start the orchestrator and initialize all worker pools.
    pub fn new(
        node_id: String,
        is_test: bool,
        node_parallelism: usize,
        object_store_by_profile: ObjectStoreByProfile,
        p2p_server: P2pServer,
        download_catalog: DownloadCatalog,
        object_store_retry_config: RetryConfig,
        p2p_retry_config: RetryConfig,
    ) -> Result<Self, OrchestratorError> {
        let object_downloader = ObjectStoreDownloader::new(
            object_store_by_profile,
            node_id.clone(),
            is_test,
            object_store_retry_config,
        );
        let p2p_downloader = P2pDownloaderWorkerPool::new(
            node_parallelism,
            node_id.clone(),
            is_test,
            p2p_retry_config,
        );
        let file_writer = FileWriterPool::new(node_parallelism, node_id.clone(), is_test);
        let assembler = AssemblerPool::new(node_parallelism, node_id.clone(), is_test);
        let unpacker = UnpackerPool::new(node_parallelism, node_id.clone(), is_test);

        let cached_things =
            find_cached_things_and_remove_invalid(&download_catalog, &node_id, is_test)?;
        log::info!(
            "Found the following lengths of cached things: cached_chunks: {:?}, objects_no_unpacking: {:?}, objects_needed_to_be_unpacked: {:?}, objects_already_unpacked: {:?}, objects_already_unpacked_and_no_object_present: {:?}",
            cached_things.cached_chunks.len(),
            cached_things.objects_no_unpacking.len(),
            cached_things.objects_needed_to_be_unpacked.len(),
            cached_things.objects_already_unpacked.len(),
            cached_things
                .objects_already_unpacked_and_no_object_present
                .len()
        );
        let assembly_tracker = AssemblyTracker::new(&download_catalog, &cached_things);

        // Schedule unpacking for objects that exist but need to be unpacked
        let startup_unpacking_tasks: Vec<UnpackerTask> = cached_things
            .objects_needed_to_be_unpacked
            .iter()
            .filter_map(|&object_id| {
                let object_to_download = download_catalog.objects.get(&object_id)?;
                let unpack_options = object_to_download.unpack_options.as_ref()?;
                Some(UnpackerTask {
                    object_id,
                    archive_path: resolve_path(
                        object_to_download.destination.clone(),
                        &node_id,
                        is_test,
                    ),
                    unpack_destination: unpack_options.destination.clone(),
                    unpack_method: unpack_options.unpack_method.clone(),
                    cache_info: object_to_download.cache_info.clone(),
                })
            })
            .collect();

        log::info!(
            "Scheduling {} startup unpacking tasks",
            startup_unpacking_tasks.len()
        );

        let mut unpacker_active_uuids = HashSet::new();
        unpacker_active_uuids.extend(startup_unpacking_tasks.iter().map(|t| t.object_id));
        unpacker.add_tasks(startup_unpacking_tasks);

        let available_chunks = cached_things.cached_chunks.clone();
        let mut completed_or_cached_objects = cached_things.objects_already_unpacked.clone();
        completed_or_cached_objects.extend(&cached_things.objects_no_unpacking);
        completed_or_cached_objects.extend(&cached_things.objects_needed_to_be_unpacked);
        completed_or_cached_objects
            .extend(&cached_things.objects_already_unpacked_and_no_object_present);

        Ok(Self {
            node_id,
            download_catalog,
            p2p_server,
            object_downloader,
            p2p_downloader,
            file_writer,
            assembler,
            unpacker,
            os_download_active_uuids: HashSet::new(),
            p2p_download_active_uuids: HashSet::new(),
            file_writer_active_uuids: HashSet::new(),
            assembler_active_uuids: HashSet::new(),
            unpacker_active_uuids,
            cached_things,
            assembly_tracker,
            available_chunks,
            completed_or_cached_objects,
        })
    }

    /// Process one iteration: handle new orders, poll workers, update state.
    /// Returns the current NodeStatus.
    pub fn update(&mut self, orders: Orders) -> Result<NodeStatus, OrchestratorError> {
        log::debug!("Orchestrator update");

        self.p2p_server.check_health()?;

        // Process new orders
        let tasks: Vec<ObjectStoreDownloadTask> = orders
            .download_from_s3
            .into_iter()
            .map(ObjectStoreDownloadTask::from_chunk_to_download)
            .collect();
        self.os_download_active_uuids
            .extend(tasks.iter().map(|t| t.chunk_id.clone()));
        self.object_downloader.add_tasks(tasks);

        let p2p_tasks: Vec<P2pDownloadTask> = orders
            .download_from_node
            .into_iter()
            .map(P2pDownloadTask::from_p2p_download_order)
            .collect();
        self.p2p_download_active_uuids
            .extend(p2p_tasks.iter().map(|t| t.chunk_id.clone()));
        self.p2p_downloader.add_tasks(p2p_tasks);

        let mut newly_written_chunks = HashSet::new();

        // Poll object store downloader and send completed downloads to file writer
        let newly_completed_os_downloads = self.object_downloader.get_task_statuses();
        let mut os_file_writer_tasks = Vec::new();
        for task in newly_completed_os_downloads {
            match task {
                OsTaskStatus::Completed(os_task, data) => {
                    self.os_download_active_uuids.remove(&os_task.chunk_id);
                    let file_writer_task = FileWriterTask {
                        chunk_id: os_task.chunk_id,
                        data,
                    };
                    self.file_writer_active_uuids.insert(os_task.chunk_id);
                    os_file_writer_tasks.push(file_writer_task);
                }
                OsTaskStatus::Failed(_, error) => {
                    warn!("Object store download failed with error: {}", error);
                    return Err(OrchestratorError::ObjectStoreDownloadFailed(error));
                }
            }
        }
        self.file_writer.add_tasks(os_file_writer_tasks);

        // Poll the p2p downloader and send completed downloads to file writer
        let newly_completed_p2p_downloads = self
            .p2p_downloader
            .get_task_statuses()
            .map_err(OrchestratorError::P2pDownloadFailed)?;
        let mut file_writer_tasks = Vec::new();
        for task in newly_completed_p2p_downloads {
            match task {
                P2pTaskStatus::Completed(p2p_task, data) => {
                    self.p2p_download_active_uuids.remove(&p2p_task.chunk_id);
                    let file_writer_task = FileWriterTask {
                        chunk_id: p2p_task.chunk_id,
                        data,
                    };
                    self.file_writer_active_uuids.insert(p2p_task.chunk_id);
                    file_writer_tasks.push(file_writer_task);
                }
                P2pTaskStatus::Failed(task, error) => {
                    self.p2p_download_active_uuids.remove(&task.chunk_id);
                    warn!("P2P download failed with error: {}", error);
                    return Err(OrchestratorError::P2pDownloadFailed(error));
                }
            }
        }
        self.file_writer.add_tasks(file_writer_tasks);

        // Poll the file writer
        let newly_completed_file_writes = self
            .file_writer
            .get_task_statuses()
            .map_err(OrchestratorError::FileWriterFailed)?;
        for task in newly_completed_file_writes {
            match task {
                FileWriterTaskStatus::Completed(task) => {
                    self.file_writer_active_uuids.remove(&task.chunk_id);
                    self.available_chunks.insert(task.chunk_id);
                    newly_written_chunks.insert(task.chunk_id);
                }
                FileWriterTaskStatus::Failed(task, error) => {
                    self.file_writer_active_uuids.remove(&task.chunk_id);
                    warn!("File writer failed with error: {}", error);
                    return Err(OrchestratorError::FileWriterFailed(error));
                }
            }
        }

        // Find tasks to assemble and send them to the assembler
        let objects_ready_for_assembly = self
            .assembly_tracker
            .add_written_chunks(&self.download_catalog, &newly_written_chunks);

        let new_objects_to_assemble: Vec<AssemblerTask> = objects_ready_for_assembly
            .into_iter()
            .filter_map(|object_id| {
                self.assembly_tracker
                    .schedule_assembly_task(&self.download_catalog, object_id)
            })
            .collect();

        self.assembler_active_uuids
            .extend(new_objects_to_assemble.iter().map(|t| t.object_id));
        self.assembler.add_tasks(new_objects_to_assemble);

        // Poll the assembler
        let mut new_objects_to_unpack = Vec::new();
        let newly_completed_assembler_tasks = self.assembler.get_task_statuses();
        for task in newly_completed_assembler_tasks {
            match task {
                AssemblerTaskStatus::Completed(task) => {
                    self.assembler_active_uuids.remove(&task.object_id);
                    let object_to_download =
                        self.download_catalog.objects.get(&task.object_id).unwrap();
                    if let Some(unpack_options) = object_to_download.unpack_options.as_ref() {
                        let unpacker_task = UnpackerTask {
                            object_id: task.object_id,
                            archive_path: task.destination.clone(),
                            unpack_destination: unpack_options.destination.clone(),
                            unpack_method: unpack_options.unpack_method.clone(),
                            cache_info: object_to_download.cache_info.clone(),
                        };
                        new_objects_to_unpack.push(unpacker_task);
                    } else {
                        self.completed_or_cached_objects.insert(task.object_id);
                    }
                }
                AssemblerTaskStatus::Failed(task, error) => {
                    self.assembler_active_uuids.remove(&task.object_id);
                    warn!("Assembler failed with error: {}", error);
                    return Err(OrchestratorError::AssemblerFailed(error));
                }
            }
        }

        // Add tasks to the unpacker
        self.unpacker_active_uuids
            .extend(new_objects_to_unpack.iter().map(|t| t.object_id));
        self.unpacker.add_tasks(new_objects_to_unpack);

        // Poll the unpacker
        let newly_completed_unpacker_tasks = self
            .unpacker
            .get_task_statuses()
            .map_err(OrchestratorError::UnpackerFailed)?;
        for task in newly_completed_unpacker_tasks {
            match task {
                UnpackerTaskStatus::Completed(task) => {
                    self.unpacker_active_uuids.remove(&task.object_id);
                    self.completed_or_cached_objects.insert(task.object_id);
                }
                UnpackerTaskStatus::Failed(task, error) => {
                    self.unpacker_active_uuids.remove(&task.object_id);
                    warn!("Unpacker failed with error: {}", error);
                    return Err(OrchestratorError::UnpackerFailed(error));
                }
            }
        }

        // Build and return the current status
        Ok(NodeStatus {
            node_id: self.node_id.clone(),
            downloading_s3_chunks: self.os_download_active_uuids.clone(),
            downloading_p2p_chunks: self.p2p_download_active_uuids.clone(),
            writing_chunks: self.file_writer_active_uuids.clone(),
            available_chunks: self.available_chunks.clone(),
            completed_or_cached_objects: self.completed_or_cached_objects.clone(),
            unneeded_objects: self
                .cached_things
                .objects_already_unpacked_and_no_object_present
                .clone(),
            num_active_uploads: self.p2p_server.active_uploads(),
            num_active_file_writing_tasks: self.file_writer_active_uuids.len(),
            num_active_assembling_tasks: self.assembler_active_uuids.len(),
            num_active_unpacking_tasks: self.unpacker_active_uuids.len(),
        })
    }
}

#[pyclass]
pub struct DataPlane {
    node_id: String,
    is_test: bool,
    node_parallelism: usize,
    download_catalog: Option<DownloadCatalog>,
    object_store_by_profile: Option<ObjectStoreByProfile>,
    p2p_server: Option<P2pServer>,
    object_store_retry_config: RetryConfig,
    p2p_retry_config: RetryConfig,
    orchestrator: Option<Orchestrator>,
}

#[pymethods]
impl DataPlane {
    #[new]
    pub fn new(
        node_id: &str,
        is_test: bool,
        node_parallelism: usize,
        download_catalog: DownloadCatalog,
        object_store_by_profile: ObjectStoreByProfile,
        object_store_retry_config: RetryConfig,
        p2p_retry_config: RetryConfig,
    ) -> Self {
        // Default to warn level.
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
        Self {
            node_id: node_id.to_string(),
            is_test,
            node_parallelism,
            download_catalog: Some(download_catalog),
            object_store_by_profile: Some(object_store_by_profile),
            p2p_server: None,
            object_store_retry_config,
            p2p_retry_config,
            orchestrator: None,
        }
    }

    pub fn start_p2p_server(&mut self, port: Option<u16>) -> u16 {
        let port = port
            .or_else(|| Some(portpicker::pick_unused_port().unwrap()))
            .unwrap();
        self.p2p_server = Some(P2pServer::new(port, self.node_id.clone(), self.is_test));
        port
    }

    pub fn start(&mut self) -> Result<(), OrchestratorError> {
        if self.orchestrator.is_some() {
            panic!("Orchestrator already started");
        }
        if self.p2p_server.is_none() {
            panic!("P2P server not started");
        }

        let node_id = self.node_id.clone();
        let is_test = self.is_test;
        let node_parallelism = self.node_parallelism;
        let object_store_by_profile = self.object_store_by_profile.take().unwrap();
        let p2p_server = self.p2p_server.take().unwrap();
        let download_catalog = self.download_catalog.take().unwrap();

        let orchestrator = Orchestrator::new(
            node_id,
            is_test,
            node_parallelism,
            object_store_by_profile,
            p2p_server,
            download_catalog,
            self.object_store_retry_config.clone(),
            self.p2p_retry_config.clone(),
        )?;

        self.orchestrator = Some(orchestrator);
        Ok(())
    }

    pub fn update(&mut self, orders: Orders) -> Result<NodeStatus, OrchestratorError> {
        debug!("Dataplane update");

        if self.orchestrator.is_none() {
            panic!("Orchestrator not started");
        }

        self.orchestrator.as_mut().unwrap().update(orders)
    }
}

/// Module initialization
pub fn register_module(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_class::<DataPlane>()?
        .finish();
    Ok(())
}
