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
    CacheInfo, DownloadCatalog, NodeStatus, ObjectStoreByProfile, Orders,
};
use crate::utils::module_builders::ImportablePyModuleBuilder;
use crossbeam_channel::{Receiver, Sender};
use log::{debug, warn};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
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
    #[error("Assembler failed: {0}")]
    AssemblerFailed(AssemblerError),
    #[error("Unpacker failed: {0}")]
    UnpackerFailed(UnpackerError),
    #[error("P2P server failed: {0}")]
    P2pServerFailed(#[from] P2pServerError),
    #[error("Failed to communicate with orchestrator")]
    FailedToCommunicateWithOrchestrator(String),
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
                if cache_info.uri == obj.cache_info.uri
                    && cache_info.size == obj.cache_info.size
                    && cache_info.last_modified_unix_micros
                        == obj.cache_info.last_modified_unix_micros
                {
                    cached_objects.insert(obj.object_id);
                } else {
                    warn!("Out of date cache for {}. Deleting cached data.", obj.uri);
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
                objects_already_unpacked.insert(obj.object_id);
            } else if has_valid_object_cache && !has_valid_unpacked_cache {
                objects_needed_to_be_unpacked.insert(obj.object_id);
            } else if !has_valid_object_cache && has_valid_unpacked_cache {
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

struct AssemblyTracker<'a> {
    download_catalog: &'a DownloadCatalog,
    remaining_chunks_for_assembly: HashMap<Uuid, HashSet<Uuid>>,
}

impl<'a> AssemblyTracker<'a> {
    fn new(download_catalog: &'a DownloadCatalog, cached_things: &CachedThings) -> Self {
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
            download_catalog,
            remaining_chunks_for_assembly,
        }
    }

    /// Takes in newly downloaded chunks and returns objects that are ready to be assembled.
    fn add_downloaded_chunks(&mut self, newly_downloaded_chunks: &HashSet<Uuid>) -> HashSet<Uuid> {
        let mut objects_ready_for_assembly = HashSet::new();

        for chunk_id in newly_downloaded_chunks {
            if let Some(chunk) = self.download_catalog.chunks.get(chunk_id) {
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
                    "Newly downloaded chunk {} not found in download catalog",
                    chunk_id
                );
            }
        }
        objects_ready_for_assembly
    }

    /// Schedules an object for assembly and returns an assembler task.
    /// It will return None if the object has already been scheduled for assembly.
    fn schedule_assembly_task(&mut self, object_id: Uuid) -> Option<AssemblerTask> {
        if self
            .remaining_chunks_for_assembly
            .remove(&object_id)
            .is_some()
        {
            let chunks_needed_for_object = self
                .download_catalog
                .chunks_by_object
                .get(&object_id)
                .unwrap();
            let object_to_download = self.download_catalog.objects.get(&object_id).unwrap();
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

fn orchestrator(
    node_id: String,
    is_test: bool,
    node_parallelism: usize,
    object_store_by_profile: ObjectStoreByProfile,
    p2p_server: P2pServer,
    download_catalog: DownloadCatalog,
    status: Arc<Mutex<NodeStatus>>,
    orders_receiver: Receiver<Orders>,
    shutdown_receiver: Receiver<()>,
) -> Result<(), OrchestratorError> {
    let mut object_downloader =
        ObjectStoreDownloader::new(object_store_by_profile, node_id.clone(), is_test);
    let p2p_downloader = P2pDownloaderWorkerPool::new(node_parallelism, node_id.clone(), is_test);
    let assembler = AssemblerPool::new(node_parallelism, node_id.clone(), is_test);
    let unpacker = UnpackerPool::new(node_parallelism, node_id.clone(), is_test);

    let mut os_download_active_uuids = HashSet::new();
    let mut p2p_download_active_uuids = HashSet::new();
    let mut assembler_active_uuids = HashSet::new();
    let mut unpacker_active_uuids = HashSet::new();

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
    let mut assembly_tracker = AssemblyTracker::new(&download_catalog, &cached_things);

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
    // Add startup unpacking tasks to the unpacker
    unpacker_active_uuids.extend(startup_unpacking_tasks.iter().map(|t| t.object_id));
    unpacker.add_tasks(startup_unpacking_tasks);

    let mut available_chunks = cached_things.cached_chunks;
    let mut completed_or_cached_objects = cached_things.objects_already_unpacked.clone();
    completed_or_cached_objects.extend(&cached_things.objects_no_unpacking);
    completed_or_cached_objects.extend(&cached_things.objects_needed_to_be_unpacked);
    completed_or_cached_objects
        .extend(&cached_things.objects_already_unpacked_and_no_object_present);
    loop {
        // Check for shutdown signal
        if shutdown_receiver.try_recv() != Err(crossbeam_channel::TryRecvError::Empty) {
            return Ok(());
        }

        p2p_server.check_health()?;

        // Check for new orders and assign and track them.
        if let Ok(orders) = orders_receiver.try_recv() {
            let tasks: Vec<ObjectStoreDownloadTask> = orders
                .download_from_s3
                .into_iter()
                .map(ObjectStoreDownloadTask::from_chunk_to_download)
                .collect();
            os_download_active_uuids.extend(tasks.iter().map(|t| t.chunk_id.clone()));
            object_downloader.add_tasks(tasks);
            let p2p_tasks: Vec<P2pDownloadTask> = orders
                .download_from_node
                .into_iter()
                .map(P2pDownloadTask::from_p2p_download_order)
                .collect();
            p2p_download_active_uuids.extend(p2p_tasks.iter().map(|t| t.chunk_id.clone()));
            p2p_downloader.add_tasks(p2p_tasks);
        }

        let mut newly_downloaded_chunks = HashSet::new();
        // Poll object store downloader.
        let newly_completed_os_downloads = object_downloader.get_task_statuses();
        for task in newly_completed_os_downloads {
            match task {
                OsTaskStatus::Completed(task) => {
                    os_download_active_uuids.remove(&task.chunk_id);
                    available_chunks.insert(task.chunk_id);
                    newly_downloaded_chunks.insert(task.chunk_id);
                }
                OsTaskStatus::Failed(_, error) => {
                    return Err(OrchestratorError::ObjectStoreDownloadFailed(error));
                }
            }
        }
        // Poll the p2p downloader.
        let newly_completed_p2p_downloads = p2p_downloader
            .get_task_statuses()
            .map_err(|e| OrchestratorError::P2pDownloadFailed(e))?;
        for task in newly_completed_p2p_downloads {
            match task {
                P2pTaskStatus::Completed(task) => {
                    p2p_download_active_uuids.remove(&task.chunk_id);
                    available_chunks.insert(task.chunk_id);
                    newly_downloaded_chunks.insert(task.chunk_id);
                }
                P2pTaskStatus::Failed(_, error) => {
                    return Err(OrchestratorError::P2pDownloadFailed(error));
                }
            }
        }

        // Find tasks to assemble and send them to the assembler.
        let objects_ready_for_assembly =
            assembly_tracker.add_downloaded_chunks(&newly_downloaded_chunks);

        let new_objects_to_assemble: Vec<AssemblerTask> = objects_ready_for_assembly
            .into_iter()
            .filter_map(|object_id| assembly_tracker.schedule_assembly_task(object_id))
            .collect();

        assembler_active_uuids.extend(new_objects_to_assemble.iter().map(|t| t.object_id));
        assembler.add_tasks(new_objects_to_assemble);

        // Poll the assembler.
        let mut new_objects_to_unpack = Vec::new();
        let newly_completed_assembler_tasks = assembler.get_task_statuses();
        for task in newly_completed_assembler_tasks {
            match task {
                AssemblerTaskStatus::Completed(task) => {
                    assembler_active_uuids.remove(&task.object_id);
                    let object_to_download = download_catalog.objects.get(&task.object_id).unwrap();
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
                        completed_or_cached_objects.insert(task.object_id);
                    }
                }
                AssemblerTaskStatus::Failed(task, error) => {
                    assembler_active_uuids.remove(&task.object_id);
                    return Err(OrchestratorError::AssemblerFailed(error));
                }
            }
        }

        // Add tasks to the unpacker.
        unpacker_active_uuids.extend(new_objects_to_unpack.iter().map(|t| t.object_id));
        unpacker.add_tasks(new_objects_to_unpack);

        // Poll the unpacker.
        let newly_completed_unpacker_tasks = unpacker
            .get_task_statuses()
            .map_err(OrchestratorError::UnpackerFailed)?;
        for task in newly_completed_unpacker_tasks {
            match task {
                UnpackerTaskStatus::Completed(task) => {
                    unpacker_active_uuids.remove(&task.object_id);
                    completed_or_cached_objects.insert(task.object_id);
                }
                UnpackerTaskStatus::Failed(task, error) => {
                    unpacker_active_uuids.remove(&task.object_id);
                    return Err(OrchestratorError::UnpackerFailed(error));
                }
            }
        }

        // Update the status.
        {
            let mut status = status.lock().unwrap();
            status.downloading_s3_chunks = os_download_active_uuids.clone();
            status.downloading_p2p_chunks = p2p_download_active_uuids.clone();
            status.available_chunks = available_chunks.clone();
            status.completed_or_cached_objects = completed_or_cached_objects.clone();
            status.unneeded_objects = cached_things
                .objects_already_unpacked_and_no_object_present
                .clone();
            status.num_active_uploads = p2p_server.active_uploads();
            status.num_active_assembling_tasks = assembler_active_uuids.len();
            status.num_active_unpacking_tasks = unpacker_active_uuids.len();
        }
        // TODO: This should be replaced by a rate limiter.
        thread::sleep(Duration::from_millis(10));
    }
}

#[pyclass]
pub struct DataPlane {
    // --------------------------globally used properties--------------------------
    node_id: String,
    is_test: bool,
    node_parallelism: usize,
    // --------------------------properties that will be sent to the orchestrator--------------------------
    download_catalog: Option<DownloadCatalog>,
    object_store_by_profile: Option<ObjectStoreByProfile>,
    p2p_server: Option<P2pServer>, // This one starts as None until the p2p server is started by the control plane.
    // --------------------------properties that are used to handle the orchestrator--------------------------
    orchestrator_handle: Option<JoinHandle<Result<(), OrchestratorError>>>,
    // The orchestrator will update this status.
    status: Option<Arc<Mutex<NodeStatus>>>,
    // The control plane will send orders to the orchestrator through this channel.
    orders_sender: Option<Sender<Orders>>,
    // The orchestrator will be stopped when this is dropped.
    shutdown_sender: Option<Sender<()>>,
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
    ) -> Self {
        env_logger::builder().try_init().unwrap();
        Self {
            node_id: node_id.to_string(),
            is_test,
            node_parallelism,
            download_catalog: Some(download_catalog),
            object_store_by_profile: Some(object_store_by_profile),
            p2p_server: None,
            orchestrator_handle: None,
            status: None,
            orders_sender: None,
            shutdown_sender: None,
        }
    }

    pub fn start_p2p_server(&mut self, port: Option<u16>) -> u16 {
        let port = port
            .or_else(|| Some(portpicker::pick_unused_port().unwrap()))
            .unwrap();
        self.p2p_server = Some(P2pServer::new(port, self.node_id.clone(), self.is_test));
        port
    }

    pub fn start(&mut self) {
        if self.orchestrator_handle.is_some() {
            panic!("Orchestrator already started");
        }
        if self.p2p_server.is_none() {
            panic!("P2P server not started");
        }
        let (orders_sender, orders_receiver) = crossbeam_channel::unbounded();
        let (shutdown_sender, shutdown_receiver) = crossbeam_channel::unbounded();
        let status = Arc::new(Mutex::new(NodeStatus::new(self.node_id.clone())));

        // Move all necessary fields out of self before spawning the thread
        let node_id = self.node_id.clone();
        let is_test = self.is_test;
        let node_parallelism = self.node_parallelism;
        let object_store_by_profile = self.object_store_by_profile.take().unwrap();
        let p2p_server = self.p2p_server.take().unwrap();
        let download_catalog = self.download_catalog.take().unwrap();
        let status_clone: Arc<Mutex<NodeStatus>> = status.clone();

        let orchestrator_handle = thread::spawn(move || {
            orchestrator(
                node_id,
                is_test,
                node_parallelism,
                object_store_by_profile,
                p2p_server,
                download_catalog,
                status_clone,
                orders_receiver,
                shutdown_receiver,
            )
        });

        self.orders_sender = Some(orders_sender);
        self.shutdown_sender = Some(shutdown_sender);
        self.orchestrator_handle = Some(orchestrator_handle);
        self.status = Some(status);
    }

    /// Helper method to check orchestrator state and return appropriate error if failed
    fn check_orchestrator_state(&mut self) -> Result<(), OrchestratorError> {
        if self.orchestrator_handle.is_none() {
            panic!("Orchestrator not started");
        }

        if let Some(handle) = &self.orchestrator_handle {
            if handle.is_finished() {
                // Orchestrator has finished, check if it was due to an error
                return match self.orchestrator_handle.take().unwrap().join() {
                    Ok(Err(e)) => {
                        debug!("Orchestrator returned an error: {:?}", e);
                        Err(e)
                    }
                    Err(_) => panic!("Orchestrator thread panicked"),
                    _ => panic!("Orchestrator finished unexpectedly without error"),
                };
            }
        }
        Ok(())
    }

    pub fn update(&mut self, orders: Orders) -> Result<NodeStatus, OrchestratorError> {
        // Check orchestrator state first
        self.check_orchestrator_state()?;

        // Send orders - get sender first and handle potential failure
        if self.orders_sender.is_none() {
            self.check_orchestrator_state()?;
            panic!("Orchestrator not started");
        }

        let send_result = self.orders_sender.as_ref().unwrap().send(orders);
        if send_result.is_err() {
            // If sending fails, check if orchestrator failed
            self.check_orchestrator_state()?;
            return Err(OrchestratorError::FailedToCommunicateWithOrchestrator(
                "Failed to send orders to orchestrator".to_string(),
            ));
        }

        // Get status - check if status is available first
        if self.status.is_none() {
            self.check_orchestrator_state()?;
            panic!("Orchestrator not started");
        }

        // Get the status with explicit scope to avoid borrow issues
        let status = {
            let lock_result = self.status.as_ref().unwrap().lock();
            match lock_result {
                Ok(status) => status.clone(),
                Err(_) => {
                    // Drop the lock_result first before checking orchestrator state
                    drop(lock_result);
                    // If lock fails, check if orchestrator failed
                    self.check_orchestrator_state()?;
                    return Err(OrchestratorError::FailedToCommunicateWithOrchestrator(
                        "Failed to lock status".to_string(),
                    ));
                }
            }
        };

        Ok(status)
    }
}

impl Drop for DataPlane {
    fn drop(&mut self) {
        // Taking the sender and dropping it will signal the orchestrator to shut down.
        self.shutdown_sender.take();

        if let Some(handle) = self.orchestrator_handle.take() {
            if let Err(e) = handle.join() {
                eprintln!("Orchestrator thread panicked during shutdown: {:?}", e);
            }
        }
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