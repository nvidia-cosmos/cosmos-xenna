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
//! - **In-Memory Downloads:** This module downloads chunks into memory and returns them to the
//!   orchestrator. Disk writing is handled by a separate FileWriterPool to pipeline network I/O
//!   with disk I/O. It does **not** perform file assembly.
//!

use crate::file_distribution::models::{
    ChunkToDownload, ObjectAndRange, ObjectStoreByProfile, RetryConfig,
};
use log::{debug, trace, warn};
use object_store::ObjectStore;
use object_store::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio_retry::Retry;
use tokio_retry::strategy::{ExponentialBackoff, jitter};
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("Request failed: {0}")]
    Request(#[from] object_store::Error),
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
    Completed(ObjectStoreDownloadTask, Vec<u8>), // Task and downloaded data
    Failed(ObjectStoreDownloadTask, DownloadError),
}

pub struct ObjectStoreDownloader {
    runtime: tokio::runtime::Runtime,
    handles: Vec<tokio::task::JoinHandle<TaskStatus>>,
    by_profile: Arc<ObjectStoreByProfile>,
    node_id: String,
    is_test: bool,
    retry_config: RetryConfig,
}

async fn download_chunk_internal(
    task: &ObjectStoreDownloadTask,
    by_profile: &ObjectStoreByProfile,
) -> Result<Vec<u8>, DownloadError> {
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
        let result = client.get(&object_path).await?;
        debug!("Got a response for full object {}", object_path);
        result.bytes().await?
    };
    debug!(
        "Downloaded {} bytes for chunk_id: {}",
        bytes.len(),
        task.chunk_id
    );

    Ok(bytes.to_vec())
}

/// Check if this looks like a timeout disguised as a decode error
fn is_likely_timeout_error(error: &DownloadError) -> bool {
    match error {
        DownloadError::Request(object_store_error) => {
            let error_str = format!("{}", object_store_error);
            error_str
                .to_lowercase()
                .contains("error decoding response body")
        }
        _ => false,
    }
}

async fn run_download_task_async(
    task: ObjectStoreDownloadTask,
    by_profile: Arc<ObjectStoreByProfile>,
    _node_id: String,
    _is_test: bool,
    retry_config: RetryConfig,
) -> TaskStatus {
    debug!(
        "Starting run_download_task_async for chunk_id: {}",
        task.chunk_id
    );
    let chunk_id = task.chunk_id;

    let retry_strategy = ExponentialBackoff::from_millis(retry_config.base_delay_millis)
        .map(jitter) // Add randomness to prevent thundering herd
        .take(retry_config.num_retries as usize);

    let result = Retry::spawn(retry_strategy, || async {
        match download_chunk_internal(&task, &by_profile).await {
            Ok(data) => Ok(data),
            Err(e) => {
                // Only retry on the specific "error decoding response body" error
                // Let object_store handle all other retry logic
                if is_likely_timeout_error(&e) {
                    warn!(
                        "Download attempt failed for chunk {} with likely timeout error: {}. \
                        Object: {}, Profile: {:?}, Size: {}B. \
                        Error details: {:?}. \
                        Large chunks may exceed default timeouts. Solutions: \
                        1) Reduce chunk_size_bytes parameter, 2) Configure longer timeouts in ObjectStoreConfig: \
                        config_args={{\"timeout\": \"300s\", \"request_timeout\": \"300s\"}}",
                        chunk_id,
                        e,
                        task.object_and_range.object_uri,
                        task.profile_name,
                        task.object_and_range
                            .range
                            .as_ref()
                            .map_or(0, |r| r.end - r.start),
                        e
                    );
                    Err(e) // Retry
                } else {
                    // Non-retryable error - make it permanent by wrapping in a different error
                    // that we'll handle below
                    debug!(
                        "Download failed for chunk {} with non-retryable error: {}. \
                        Object: {}, Profile: {:?}. \
                        Error details: {:?}",
                        chunk_id,
                        e,
                        task.object_and_range.object_uri,
                        task.profile_name,
                        e
                    );
                    // Return an error that will not be retried
                    Err(e)
                }
            }
        }
    })
    .await;

    match result {
        Ok(data) => {
            debug!("Successfully downloaded chunk {}", chunk_id);
            TaskStatus::Completed(task, data)
        }
        Err(e) => {
            warn!("Download failed for chunk {}: {}", chunk_id, e);
            TaskStatus::Failed(task, e)
        }
    }
}

impl ObjectStoreDownloader {
    pub fn new(
        by_profile: ObjectStoreByProfile,
        node_id: String,
        is_test: bool,
        retry_config: RetryConfig,
    ) -> Self {
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
            retry_config,
        }
    }

    pub fn add_tasks(&mut self, tasks: impl IntoIterator<Item = ObjectStoreDownloadTask>) {
        trace!("Adding tasks to ObjectStoreDownloader");
        for task in tasks {
            trace!("Adding task: {:?}", task);
            let future = run_download_task_async(
                task,
                self.by_profile.clone(),
                self.node_id.clone(),
                self.is_test,
                self.retry_config.clone(),
            );
            self.handles.push(self.runtime.spawn(future));
        }
        trace!("Current number of handles: {}", self.handles.len());
    }

    pub fn get_task_statuses(&mut self) -> Vec<TaskStatus> {
        trace!(
            "Getting task statuses. Current handle count: {}",
            self.handles.len()
        );
        let (finished, pending): (Vec<_>, Vec<_>) = std::mem::take(&mut self.handles)
            .into_iter()
            .partition(|h| h.is_finished());

        trace!(
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
        trace!("Returning {} task statuses.", results.len());

        results
    }

    pub fn get_num_queued_or_active_tasks(&self) -> usize {
        let num_tasks = self.handles.len();
        trace!("Getting number of queued or active tasks: {}", num_tasks);
        num_tasks
    }
}
