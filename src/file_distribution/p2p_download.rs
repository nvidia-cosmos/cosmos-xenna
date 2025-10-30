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
//! - **In-Memory Downloads:** This module downloads chunks into memory and returns them to the
//!   orchestrator. Disk writing is handled by a separate FileWriterPool to pipeline network I/O
//!   with disk I/O. This prevents download workers from blocking on disk operations and ensures
//!   steady P2P upload availability.
//!
//! - **Retries:** The system includes a retry mechanism with an exponential backoff strategy to
//!   handle transient network errors gracefully.
use crate::file_distribution::models::{DownloadFromNodeOrder, ObjectAndRange, RetryConfig};
use crossbeam_channel::Sender;
use retry::{
    OperationResult,
    delay::{Exponential, jitter},
    retry_with_index,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Request failed with status: {0}")]
    HttpStatus(reqwest::StatusCode),
    #[error("Invalid chunk size. Expected {expected_size} bytes, got {actual_size} bytes.")]
    InvalidChunkSize {
        expected_size: usize,
        actual_size: usize,
    },
    #[error("A worker thread has panicked.")]
    WorkerPanicked,
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
pub struct P2pDownloadTask {
    pub chunk_id: Uuid,
    pub destination: PathBuf,
    pub object_and_range: ObjectAndRange,
    pub addr: SocketAddr,
}

impl P2pDownloadTask {
    pub fn from_p2p_download_order(order: DownloadFromNodeOrder) -> Self {
        Self {
            chunk_id: order.download_chunk.chunk_id,
            destination: order.download_chunk.destination,
            object_and_range: order.download_chunk.value,
            addr: order.source_node_address,
        }
    }
}

#[derive(Debug)]
pub enum TaskStatus {
    Completed(P2pDownloadTask, Vec<u8>), // Task and downloaded data
    Failed(P2pDownloadTask, DownloadError),
}

/// A thread pool for executing jobs concurrently.
pub struct P2pDownloaderWorkerPool {
    workers: Vec<thread::JoinHandle<()>>,
    task_tx: Option<Sender<P2pDownloadTask>>,
    active_tasks_count: Arc<AtomicUsize>,
    completed_rx: crossbeam_channel::Receiver<TaskStatus>,
}

fn download_chunk(
    client: &reqwest::blocking::Client,
    job: &P2pDownloadTask,
) -> Result<Vec<u8>, DownloadError> {
    let mut request = client
        .get(format!(
            "http://{}:{}/chunk/{}",
            job.addr.ip(),
            job.addr.port(),
            job.chunk_id
        ))
        .query(&[("destination", job.destination.to_str().unwrap())]);
    if let Some(ref range) = job.object_and_range.range {
        request = request
            .query(&[("range_start", range.start.to_string())])
            .query(&[("range_end", range.end.to_string())]);
    }

    let response = request.send()?;

    if !response.status().is_success() {
        let status = response.status();
        log::warn!("Request failed with status: {}", status);
        return Err(DownloadError::HttpStatus(status));
    }

    let bytes = response.bytes()?;

    // Validate chunk size if range is specified
    if let Some(range) = job.object_and_range.range.clone() {
        let expected_size = (range.end - range.start) as usize;
        if bytes.len() != expected_size {
            log::error!(
                "Downloaded chunk {} has wrong size. Expected {} bytes, got {} bytes.",
                job.chunk_id,
                expected_size,
                bytes.len()
            );
            return Err(DownloadError::InvalidChunkSize {
                expected_size,
                actual_size: bytes.len(),
            });
        }
    }

    Ok(bytes.to_vec())
}

fn attempt_download(
    current_try: u32,
    chunk_id: Uuid,
    client: &reqwest::blocking::Client,
    job: &P2pDownloadTask,
    max_retries: u32,
) -> OperationResult<Vec<u8>, DownloadError> {
    if current_try > 0 {
        log::info!("Retrying job for chunk {}...", chunk_id);
    }
    match download_chunk(client, job) {
        Ok(data) => OperationResult::Ok(data),
        Err(e) => {
            if let DownloadError::HttpStatus(status) = e {
                if status.is_client_error() {
                    log::debug!(
                        "A client error occurred for chunk {}: {}. This is not retryable.",
                        chunk_id,
                        status
                    );
                    return OperationResult::Err(e);
                }
            }
            log::warn!(
                "Download attempt failed for chunk {}: {:?}. Retry {}/{}, retrying...",
                chunk_id,
                e,
                current_try + 1,
                max_retries
            );
            OperationResult::Retry(e)
        }
    }
}

impl P2pDownloaderWorkerPool {
    /// Creates a new WorkerPool with a specified number of worker threads.
    ///
    /// # Panics
    /// Panics if `num_workers` is 0.
    pub fn new(
        num_workers: usize,
        _node_id: String,
        _is_test: bool,
        retry_config: RetryConfig,
    ) -> Self {
        assert!(
            num_workers > 0,
            "P2pDownloaderWorkerPool must have at least one worker."
        );

        let (task_tx, task_rx) = crossbeam_channel::unbounded::<P2pDownloadTask>();
        let (completed_tx, completed_rx) = crossbeam_channel::unbounded::<TaskStatus>();
        let active_tasks_count = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let rx = task_rx.clone();
            let active_count = Arc::clone(&active_tasks_count);
            let completed_tx_cloned = completed_tx.clone();
            let retry_config_clone = retry_config.clone();

            let handle = thread::spawn(move || {
                let client = reqwest::blocking::Client::builder()
                    .pool_max_idle_per_host(10) // Reuse connections
                    .pool_idle_timeout(std::time::Duration::from_secs(30))
                    .tcp_keepalive(std::time::Duration::from_secs(60))
                    .timeout(std::time::Duration::from_secs(300)) // 5 min timeout for large chunks
                    .build()
                    .expect("Failed to build reqwest client");
                for job in rx {
                    log::debug!(
                        "[Worker {}] Got a job for chunk {}, starting...",
                        i,
                        job.chunk_id
                    );
                    let _guard = ActiveTaskGuard::new(Arc::clone(&active_count));
                    let chunk_id = job.chunk_id;

                    let result = retry_with_index(
                        Exponential::from_millis(retry_config_clone.base_delay_millis)
                            .map(jitter) // Add randomness to prevent thundering herd
                            .take(retry_config_clone.num_retries as usize),
                        |current_try| {
                            attempt_download(
                                current_try as u32,
                                chunk_id,
                                &client,
                                &job,
                                retry_config_clone.num_retries,
                            )
                        },
                    );

                    let status = match result {
                        Ok(data) => TaskStatus::Completed(job, data),
                        Err(e) => TaskStatus::Failed(job, e.error),
                    };

                    if let Err(err) = completed_tx_cloned.send(status) {
                        log::warn!(
                            "Failed to send completion notification for chunk {}: {}",
                            chunk_id,
                            err
                        );
                    }

                    log::debug!("[Worker {}] Finished job for chunk {}.", i, chunk_id);
                }
                log::debug!("[Worker {}] Shutting down.", i);
            });
            workers.push(handle);
        }

        P2pDownloaderWorkerPool {
            workers,
            task_tx: Some(task_tx),
            active_tasks_count,
            completed_rx,
        }
    }

    /// Adds a collection of tasks (jobs) to the queue.
    ///
    /// The tasks are closures that will be executed by the worker threads.
    pub fn add_tasks(&self, jobs: impl IntoIterator<Item = P2pDownloadTask>) {
        if let Some(tx) = &self.task_tx {
            for job in jobs {
                tx.send(job).expect("Failed to send job to a worker.");
            }
        }
    }

    pub fn get_task_statuses(&self) -> Result<Vec<TaskStatus>, DownloadError> {
        if self.workers.iter().any(|handle| handle.is_finished()) {
            return Err(DownloadError::WorkerPanicked);
        }
        // Drain any completed tasks without blocking
        Ok(self.completed_rx.try_iter().collect())
    }

    /// Returns the total number of tasks that are either queued or currently
    /// being processed by a worker.
    pub fn get_num_queued_or_active_tasks(&self) -> usize {
        // Get the number of tasks waiting in the channel's buffer.
        let queued_count = self.task_tx.as_ref().map_or(0, |tx| tx.len());
        // Get the number of tasks being actively processed.
        let active_count = self.active_tasks_count.load(Ordering::Relaxed);

        queued_count + active_count
    }

    /// Returns the total number of tasks that are either queued or currently
    /// being processed by a worker.
    pub fn get_num_active_tasks(&self) -> usize {
        self.active_tasks_count.load(Ordering::Relaxed)
    }
}

// The Drop trait is crucial for automatic and graceful shutdown.
impl Drop for P2pDownloaderWorkerPool {
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
