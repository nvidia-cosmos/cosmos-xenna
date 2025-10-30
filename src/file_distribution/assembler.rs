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
use crate::file_distribution::common::{get_temp_chunk_path, resolve_path};
use crate::file_distribution::models::CacheInfo;
use crossbeam_channel::Sender;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tempfile::NamedTempFile;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum AssemblerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to create directory '{path}': {source}")]
    CreateDirectoryFailed {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to create temporary file in directory '{path}': {source}")]
    CreateTempFileFailed {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to open chunk file '{path}': {source}")]
    OpenChunkFileFailed {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to write cache file '{path}': {source}")]
    WriteCacheFileFailed {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to remove chunk file '{path}': {source}")]
    RemoveChunkFileFailed {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("Failed to persist temporary file: {0}")]
    Persist(#[from] tempfile::PersistError),
    #[error("Chunk file not found: {0}")]
    ChunkFileNotFound(Uuid),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Destination path has no parent directory")]
    NoParentDir,
    #[error("File size mismatch: actual={actual}, expected={expected}")]
    FileSizeMismatch { actual: u64, expected: u64 },
}

struct ActiveTaskGuard {
    count: Arc<AtomicUsize>,
}

impl ActiveTaskGuard {
    fn new(count: Arc<AtomicUsize>) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        ActiveTaskGuard { count }
    }
}

impl Drop for ActiveTaskGuard {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub struct AssemblerTask {
    pub object_id: Uuid,
    pub chunk_ids: Vec<Uuid>,
    pub destination: PathBuf,
    pub cache_info: CacheInfo,
}

#[derive(Debug)]
pub enum TaskStatus {
    Completed(AssemblerTask),
    Failed(AssemblerTask, AssemblerError),
}

/// A thread pool for executing jobs concurrently.
pub struct AssemblerPool {
    workers: Vec<thread::JoinHandle<()>>,
    task_tx: Option<Sender<AssemblerTask>>,
    active_tasks_count: Arc<AtomicUsize>,
    completed_rx: crossbeam_channel::Receiver<TaskStatus>,
}

fn assemble_chunks(
    chunk_ids: &[Uuid],
    destination: &PathBuf,
    cache_info: &CacheInfo,
    node_id: &str,
    is_test: bool,
) -> Result<(), AssemblerError> {
    log::debug!(
        "Assembling {} chunks for destination: {:?}",
        chunk_ids.len(),
        destination
    );
    let destination = resolve_path(destination.to_path_buf(), node_id, is_test);
    let parent_dir = destination.parent().ok_or(AssemblerError::NoParentDir)?;
    log::debug!("Ensuring parent directory exists: {:?}", parent_dir);
    std::fs::create_dir_all(parent_dir).map_err(|e| AssemblerError::CreateDirectoryFailed {
        path: parent_dir.to_path_buf(),
        source: e,
    })?;

    log::debug!("Creating temporary file in: {:?}", parent_dir);
    let mut temp_file =
        NamedTempFile::new_in(parent_dir).map_err(|e| AssemblerError::CreateTempFileFailed {
            path: parent_dir.to_path_buf(),
            source: e,
        })?;

    for chunk_id in chunk_ids {
        log::debug!("Processing chunk ID: {}", chunk_id);
        let chunk_path = get_temp_chunk_path(*chunk_id, node_id, is_test);
        log::debug!("Chunk file size: {:?}", chunk_path.metadata()?.len());
        if !chunk_path.exists() {
            log::error!("Chunk file not found at path: {:?}", chunk_path);
            return Err(AssemblerError::ChunkFileNotFound(*chunk_id));
        }
        log::debug!("Opening chunk file: {:?}", chunk_path);
        let mut chunk_file =
            File::open(&chunk_path).map_err(|e| AssemblerError::OpenChunkFileFailed {
                path: chunk_path.clone(),
                source: e,
            })?;
        log::debug!("Copying chunk {} to temporary file", chunk_id);
        std::io::copy(&mut chunk_file, &mut temp_file)?;
    }

    // Ensure all data is written to disk before persist
    log::debug!("Flushing temporary file to ensure data is written to disk");
    temp_file.flush()?;
    temp_file.as_file().sync_all()?;

    // Use as_file() to get a reference to the underlying File for metadata
    let file_size = temp_file.as_file().metadata()?.len();
    log::debug!("File size: {}", file_size);
    if file_size != cache_info.size {
        log::error!("File size mismatch: {} != {}", file_size, cache_info.size);
        return Err(AssemblerError::FileSizeMismatch {
            actual: file_size,
            expected: cache_info.size,
        });
    }
    log::debug!("Persisting temporary file to: {:?}", destination);
    temp_file.persist(&destination)?;
    log::debug!("Successfully persisted file to: {:?}", destination);

    let cache_path = CacheInfo::get_cache_path_for_file(destination.clone());
    log::debug!("Writing cache info to: {:?}", cache_path);
    let cache_json = cache_info.to_json()?;
    std::fs::write(&cache_path, cache_json).map_err(|e| AssemblerError::WriteCacheFileFailed {
        path: cache_path.clone(),
        source: e,
    })?;
    log::debug!("Successfully wrote cache info to {:?}.", cache_path);

    // Clean up temporary chunk files
    log::debug!("Cleaning up temporary chunk files.");
    for chunk_id in chunk_ids {
        let chunk_path = get_temp_chunk_path(*chunk_id, node_id, is_test);
        if chunk_path.exists() {
            log::debug!("Removing temporary chunk file: {:?}", chunk_path);
            std::fs::remove_file(&chunk_path).map_err(|e| {
                AssemblerError::RemoveChunkFileFailed {
                    path: chunk_path,
                    source: e,
                }
            })?;
        }
    }
    log::debug!("Finished cleaning up temporary chunk files.");

    log::debug!(
        "Successfully assembled all chunks for destination: {:?}",
        destination
    );
    Ok(())
}

impl AssemblerPool {
    /// Creates a new WorkerPool with a specified number of worker threads.
    ///
    /// # Panics
    /// Panics if `num_workers` is 0.
    pub fn new(num_workers: usize, node_id: String, is_test: bool) -> Self {
        log::debug!(
            "Creating new AssemblerPool with {} workers for node_id '{}'. is_test={}",
            num_workers,
            node_id,
            is_test
        );
        assert!(
            num_workers > 0,
            "AssemblerPool must have at least one worker."
        );

        let (task_tx, task_rx) = crossbeam_channel::unbounded::<AssemblerTask>();
        let (completed_tx, completed_rx) = crossbeam_channel::unbounded::<TaskStatus>();
        let active_tasks_count = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let rx = task_rx.clone();
            let active_count = Arc::clone(&active_tasks_count);
            let completed_tx_cloned = completed_tx.clone();
            let node_id_clone = node_id.to_string();

            let handle = thread::spawn(move || {
                log::debug!("[Worker {}] Starting up.", i);
                for job in rx {
                    log::debug!(
                        "[Worker {}] Got a job for object {}, starting...",
                        i,
                        job.object_id
                    );
                    let _guard = ActiveTaskGuard::new(Arc::clone(&active_count));

                    let object_id = job.object_id;
                    let result = assemble_chunks(
                        &job.chunk_ids,
                        &job.destination,
                        &job.cache_info,
                        &node_id_clone,
                        is_test,
                    );

                    let status = match result {
                        Ok(_) => TaskStatus::Completed(job),
                        Err(e) => {
                            log::error!(
                                "[Worker {}] Assembly failed for object {}: {}",
                                i,
                                object_id,
                                e
                            );
                            TaskStatus::Failed(job, e)
                        }
                    };

                    if let Err(err) = completed_tx_cloned.send(status) {
                        panic!(
                            "Failed to send completion notification for object {}: {}",
                            object_id, err
                        );
                    }

                    log::debug!("[Worker {}] Finished job for object {}.", i, object_id);
                }
                log::debug!("[Worker {}] Shutting down.", i);
            });
            workers.push(handle);
        }

        log::debug!(
            "All {} worker threads spawned and waiting for jobs.",
            num_workers
        );
        AssemblerPool {
            workers,
            task_tx: Some(task_tx),
            active_tasks_count,
            completed_rx,
        }
    }

    /// Adds a collection of tasks (jobs) to the queue.
    ///
    /// The tasks are closures that will be executed by the worker threads.
    pub fn add_tasks(&self, jobs: impl IntoIterator<Item = AssemblerTask>) {
        if let Some(tx) = &self.task_tx {
            for job in jobs {
                log::debug!(
                    "Adding assembly task for object {} to the queue.",
                    job.object_id
                );
                tx.send(job).expect("Failed to send job to a worker.");
            }
        } else {
            log::warn!("Attempted to add tasks after assembler pool shutdown.");
        }
    }

    pub fn get_task_statuses(&self) -> Vec<TaskStatus> {
        log::trace!("Checking for completed task statuses.");
        // Drain any completed tasks without blocking
        let statuses: Vec<TaskStatus> = self.completed_rx.try_iter().collect();
        if !statuses.is_empty() {
            log::debug!("Collected {} new task statuses.", statuses.len());
        }
        statuses
    }

    /// Returns the total number of tasks that are either queued or currently
    /// being processed by a worker.
    pub fn get_num_queued_or_active_tasks(&self) -> usize {
        // Get the number of tasks waiting in the channel's buffer.
        let queued_count = self.task_tx.as_ref().map_or(0, |tx| tx.len());
        // Get the number of tasks being actively processed.
        let active_count = self.active_tasks_count.load(Ordering::Relaxed);
        log::trace!(
            "Queue status: {} queued, {} active.",
            queued_count,
            active_count
        );

        queued_count + active_count
    }
}

// The Drop trait is crucial for automatic and graceful shutdown.
impl Drop for AssemblerPool {
    fn drop(&mut self) {
        log::debug!("--- [Drop] Pool is going out of scope. Shutting down. ---");

        // 1. Drop the sender. This closes the channel, signaling to workers
        //    that no more jobs will be sent.
        if let Some(tx) = self.task_tx.take() {
            drop(tx);
        }

        // 2. Join all worker threads. This waits for them to finish their
        //    current job and exit their loop.
        for handle in self.workers.drain(..) {
            handle.join().expect("Worker thread panicked.");
        }

        log::debug!("--- [Drop] All workers have shut down. ---");
    }
}
