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

//! # Object Store Downloader
//!
//! This module defines the core data plane logic for downloading files from an object store
//! like S3. It is responsible for orchestrating the download of file chunks.
//!
//! ## Architecture
//!
//! The `ObjectStoreDownloader` is the central component of this module. It is modeled
//! after the `AsyncThreadPool` in `cosmos-s3-utils`. It uses a dedicated manager
//! thread to spawn and manage asynchronous download tasks on a Tokio runtime.
//! This design allows for concurrent chunk downloads while providing a synchronous API
//! to the calling context.
//!
//! ## Operations
//!
//! - **Task Submission:** The downloader receives `ObjectStoreDownloadTask` items via the `add_tasks`
//!   method.
//!
//! - **Task Execution:** For each task, a new asynchronous job is spawned on an internal Tokio runtime.
//!   The number of concurrent jobs can be limited.
//!
//! - **Downloading:** Each job uses an `object_store` client to make requests to the object store.
//!   The `download_chunk_internal` function handles the specifics of this request asynchronously.
//!
//! - **Result Collection:** The manager thread polls for completed tasks, collects their results
//!   (`TaskStatus`), and makes them available through the `get_task_statuses` method.
//!
//! - **Temporary Storage:** This module downloads chunks into a
//!   temporary directory, as managed by `get_temp_chunk_path`. It does **not** perform file
//!   assembly.
//!
//! - **Retries:** The system includes a retry mechanism with an exponential backoff strategy to
//!   handle transient network errors gracefully.

use crate::file_distribution::common::get_temp_chunk_path;
use crate::file_distribution::models::{ChunkToDownload, ObjectAndRange, ObjectStoreByProfile};
use log::{debug, warn};
use object_store::ObjectStore;
use object_store::path::Path;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::NamedTempFile;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("Request failed: {0}")]
    Request(#[from] object_store::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to persist temporary file: {0}")]
    Persist(#[from] tempfile::PersistError),
    #[error("URL parsing error: {0}")]
    Url(#[from] url::ParseError),
    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ObjectStoreDownloadTask {
    pub profile_name: Option<String>,
    pub chunk_id: Uuid,
    pub destination: PathBuf,
    pub object_and_range: ObjectAndRange,
}

impl ObjectStoreDownloadTask {
    pub fn from_chunk_to_download(chunk: ChunkToDownload) -> Self {
        debug!(
            "Creating ObjectStoreDownloadTask from ChunkToDownload: {:?}",
            chunk
        );
        Self {
            profile_name: chunk.profile_name,
            chunk_id: chunk.chunk_id,
            destination: chunk.destination,
            object_and_range: chunk.value,
        }
    }
}

#[derive(Debug)]
pub enum TaskStatus {
    Completed(ObjectStoreDownloadTask),
    Failed(ObjectStoreDownloadTask, DownloadError),
}

pub struct ObjectStoreDownloader {
    runtime: tokio::runtime::Runtime,
    handles: Vec<tokio::task::JoinHandle<TaskStatus>>,
    by_profile: Arc<ObjectStoreByProfile>,
    node_id: String,
    is_test: bool,
}

async fn download_chunk_internal(
    task: &ObjectStoreDownloadTask,
    by_profile: &ObjectStoreByProfile,
    node_id: &str,
    is_test: bool,
) -> Result<(), DownloadError> {
    debug!(
        "Starting download_chunk_internal for chunk_id: {}",
        task.chunk_id
    );
    let profile_name = task.profile_name.as_deref();
    debug!("Using profile: {:?}", profile_name);
    let client = by_profile.get_client(profile_name);
    let object_path = Path::from(task.object_and_range.object_uri.as_ref());

    let bytes = if let Some(range) = &task.object_and_range.range {
        debug!("Downloading range {:?} for object {}", range, object_path);
        client
            .get_range(&object_path, (range.start as usize)..(range.end as usize))
            .await?
    } else {
        debug!("Downloading full object {}", object_path);
        client.get(&object_path).await?.bytes().await?
    };
    debug!(
        "Downloaded {} bytes for chunk_id: {}",
        bytes.len(),
        task.chunk_id
    );
    let temp_path = get_temp_chunk_path(task.chunk_id, node_id, is_test);
    debug!(
        "Writing chunk {} to temporary path: {:?}",
        task.chunk_id, temp_path
    );

    // Create a temporary file in the same directory as the final destination
    // This ensures the atomic rename will work (same filesystem)
    let temp_dir = temp_path.parent().unwrap();
    let mut temp_file = NamedTempFile::new_in(temp_dir)?;

    // Write the data to the temporary file
    temp_file.write_all(&bytes)?;

    // Ensure data is written to disk before persist
    temp_file.flush()?;
    temp_file.as_file().sync_all()?;

    // Atomically move the temporary file to the final destination
    temp_file.persist(&temp_path)?;

    debug!(
        "Successfully wrote chunk {} to temporary path: {:?}",
        task.chunk_id, temp_path
    );

    Ok(())
}

/// Check if an error indicates server-side issues that benefit from aggressive backoff
/// This includes rate limiting (429) and response decoding errors
fn needs_aggressive_backoff(error: &DownloadError) -> bool {
    match error {
        DownloadError::Request(object_store_error) => {
            let error_str = format!("{}", object_store_error);
            error_str.contains("429")
                || error_str.to_lowercase().contains("too many requests")
                || error_str.to_lowercase().contains("slow down")
                || error_str
                    .to_lowercase()
                    .contains("error decoding response body")
        }
        _ => false,
    }
}

async fn run_download_task_async(
    task: ObjectStoreDownloadTask,
    by_profile: Arc<ObjectStoreByProfile>,
    node_id: String,
    is_test: bool,
) -> TaskStatus {
    debug!(
        "Starting run_download_task_async for chunk_id: {}",
        task.chunk_id
    );
    let mut attempts = 0;
    let max_attempts = 6; // Increased for rate limiting scenarios
    let mut delay = tokio::time::Duration::from_millis(200);
    let chunk_id = task.chunk_id;

    loop {
        attempts += 1;
        debug!("Attempt {} for chunk {}", attempts, chunk_id);
        match download_chunk_internal(&task, &by_profile, &node_id, is_test).await {
            Ok(()) => {
                debug!("Successfully downloaded chunk {}", chunk_id);
                return TaskStatus::Completed(task);
            }
            Err(e) => {
                if attempts >= max_attempts {
                    warn!(
                        "Final attempt failed for chunk {}: {}. Giving up.",
                        chunk_id, e
                    );
                    debug!(
                        "Giving up on chunk {} after {} attempts.",
                        chunk_id, max_attempts
                    );
                    return TaskStatus::Failed(task, e);
                }

                // Check if this error needs aggressive backoff
                let needs_aggressive_backoff = needs_aggressive_backoff(&e);

                // Calculate next delay - more aggressive for server-side issues
                let next_delay = if needs_aggressive_backoff {
                    // For server-side issues: start with longer base delay and use larger multiplier
                    let base_delay = tokio::time::Duration::from_millis(1000); // 1 second base
                    let multiplier: u32 = 3; // Triple the delay each time
                    let exponential_factor = multiplier.saturating_pow(attempts - 1);
                    let calculated_delay = base_delay * exponential_factor;

                    // Cap at 30 seconds to avoid extremely long waits
                    std::cmp::min(calculated_delay, tokio::time::Duration::from_secs(30))
                } else {
                    // Standard exponential backoff for other errors
                    delay * 2
                };

                warn!(
                    "Attempt {} failed for chunk {}: {}. {} Retrying in {:?}...",
                    attempts,
                    chunk_id,
                    e,
                    if needs_aggressive_backoff {
                        "(Server issue - aggressive backoff)"
                    } else {
                        ""
                    },
                    next_delay
                );
                debug!(
                    "Will retry chunk {} after delay of {:?} (aggressive backoff: {})",
                    chunk_id, next_delay, needs_aggressive_backoff
                );

                tokio::time::sleep(next_delay).await;
                delay = next_delay;
            }
        }
    }
}

impl ObjectStoreDownloader {
    pub fn new(by_profile: ObjectStoreByProfile, node_id: String, is_test: bool) -> Self {
        debug!(
            "Creating new ObjectStoreDownloader with node_id: {}, is_test: {}",
            node_id, is_test
        );
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        Self {
            runtime,
            handles: Vec::new(),
            by_profile: Arc::new(by_profile),
            node_id,
            is_test,
        }
    }

    pub fn add_tasks(&mut self, tasks: impl IntoIterator<Item = ObjectStoreDownloadTask>) {
        debug!("Adding tasks to ObjectStoreDownloader");
        for task in tasks {
            debug!("Adding task: {:?}", task);
            let future = run_download_task_async(
                task,
                self.by_profile.clone(),
                self.node_id.clone(),
                self.is_test,
            );
            self.handles.push(self.runtime.spawn(future));
        }
        debug!("Current number of handles: {}", self.handles.len());
    }

    pub fn get_task_statuses(&mut self) -> Vec<TaskStatus> {
        debug!(
            "Getting task statuses. Current handle count: {}",
            self.handles.len()
        );
        let (finished, pending): (Vec<_>, Vec<_>) = std::mem::take(&mut self.handles)
            .into_iter()
            .partition(|h| h.is_finished());

        debug!(
            "Found {} finished tasks and {} pending tasks.",
            finished.len(),
            pending.len()
        );

        self.handles = pending;
        let mut results = Vec::with_capacity(finished.len());

        for handle in finished {
            debug!("Processing a finished handle.");
            match self.runtime.block_on(handle) {
                Ok(result) => {
                    debug!("Got result for a task: {:?}", result);
                    results.push(result)
                }
                Err(e) => panic!("Tokio task panicked: {}", e),
            }
        }
        debug!("Returning {} task statuses.", results.len());

        results
    }

    pub fn get_num_queued_or_active_tasks(&self) -> usize {
        let num_tasks = self.handles.len();
        debug!("Getting number of queued or active tasks: {}", num_tasks);
        num_tasks
    }
}