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

//! # Unpacker Worker Pool
//!
//! This module is responsible for unpacking archived files.
//! It's designed to work similarly to the assembler, managing a pool of worker
//! threads to perform unpacking tasks concurrently.

use crate::file_distribution::common::resolve_path;
use crate::file_distribution::models::{CacheInfo, UnpackMethod};
use crossbeam_channel::Sender;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tempfile::tempdir_in;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum UnpackerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Command failed: {0}")]
    Command(String),
    #[error("Unsupported unpack method: {0}")]
    UnsupportedUnpackMethod(String),
    #[error("Destination path has no parent directory")]
    NoParentDir,
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Worker thread pool unhealthy: {0} out of {1} workers have died")]
    WorkerPoolUnhealthy(usize, usize),
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
pub struct UnpackerTask {
    pub object_id: Uuid,
    pub archive_path: PathBuf,
    pub unpack_destination: PathBuf,
    pub unpack_method: UnpackMethod,
    pub cache_info: CacheInfo,
}

#[derive(Debug)]
pub enum TaskStatus {
    Completed(UnpackerTask),
    Failed(UnpackerTask, UnpackerError),
}

fn unpack_and_cache(
    archive_path: &Path,
    destination: &Path,
    method: &UnpackMethod,
    node_id: &str,
    is_test: bool,
    cache_info: &CacheInfo,
) -> Result<(), UnpackerError> {
    log::debug!(
        "Starting unpack_and_cache: archive_path={:?}, destination={:?}, method={:?}, node_id={}, is_test={}",
        archive_path,
        destination,
        method,
        node_id,
        is_test
    );

    unpack_file(archive_path, destination, method, node_id, is_test)?;

    let resolved_dest = resolve_path(destination.to_path_buf(), node_id, is_test);
    log::debug!("Resolved destination path: {:?}", resolved_dest);

    let cache_path = CacheInfo::get_cache_path_for_directory(resolved_dest);
    log::debug!("Writing cache info to: {:?}", cache_path);

    let cache_json = cache_info.to_json()?;
    std::fs::write(&cache_path, &cache_json)?;

    log::debug!(
        "Successfully completed unpack_and_cache for archive: {:?}",
        archive_path
    );
    Ok(())
}

fn unpack_method_from_path(path: &Path) -> Result<UnpackMethod, UnpackerError> {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    log::debug!("Auto-detecting unpack method for file: {}", file_name);

    let method = if file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz") {
        Ok(UnpackMethod::TarGz)
    } else if file_name.ends_with(".tar") {
        Ok(UnpackMethod::Tar)
    } else if file_name.ends_with(".zip") {
        Ok(UnpackMethod::Zip)
    } else {
        let ext = path.extension().and_then(|s| s.to_str());
        match ext {
            Some(e) => {
                log::warn!("Unsupported file extension: {}", e);
                Err(UnpackerError::UnsupportedUnpackMethod(format!(
                    "Unsupported file extension '{}' for file: {}",
                    e,
                    path.display()
                )))
            }
            None => {
                log::warn!("No file extension found for: {:?}", path);
                Err(UnpackerError::UnsupportedUnpackMethod(format!(
                    "No extension found for file: {}",
                    path.display()
                )))
            }
        }
    };

    if let Ok(ref detected_method) = method {
        log::debug!("Detected unpack method: {:?}", detected_method);
    }

    method
}

fn unpack_file(
    archive_path: &Path,
    destination: &Path,
    method: &UnpackMethod,
    node_id: &str,
    is_test: bool,
) -> Result<(), UnpackerError> {
    log::debug!(
        "Starting unpack_file: archive={:?}, destination={:?}, method={:?}",
        archive_path,
        destination,
        method
    );

    let method_to_use = match method {
        UnpackMethod::Auto => {
            log::debug!("Auto-detecting unpack method for: {:?}", archive_path);
            unpack_method_from_path(&archive_path)
        }
        _ => {
            log::debug!("Using specified unpack method: {:?}", method);
            Ok(method.clone())
        }
    }?;

    let resolved_archive_path = resolve_path(archive_path.to_path_buf(), node_id, is_test);

    let resolved_dest = resolve_path(destination.to_path_buf(), node_id, is_test);
    log::debug!("Resolved destination: {:?}", resolved_dest);

    let parent_dir = resolved_dest.parent().ok_or(UnpackerError::NoParentDir)?;
    log::debug!("Creating parent directory: {:?}", parent_dir);
    std::fs::create_dir_all(parent_dir)?;

    let temp_dir = tempdir_in(parent_dir)?;
    log::debug!("Created temporary directory: {:?}", temp_dir.path());

    let mut command = match method_to_use {
        UnpackMethod::Zip => {
            log::debug!("Preparing unzip command for: {:?}", archive_path);
            let mut cmd = Command::new("unzip");
            cmd.arg("-q")
                .arg(resolved_archive_path)
                .arg("-d")
                .arg(temp_dir.path());
            cmd
        }
        UnpackMethod::Tar => {
            log::debug!("Preparing tar command for: {:?}", archive_path);
            let mut cmd = Command::new("tar");
            cmd.arg("-xf")
                .arg(resolved_archive_path)
                .arg("-C")
                .arg(temp_dir.path());
            cmd
        }
        UnpackMethod::TarGz => {
            log::debug!("Preparing tar -xzf command for: {:?}", archive_path);
            let mut cmd = Command::new("tar");
            cmd.arg("-xzf")
                .arg(resolved_archive_path)
                .arg("-C")
                .arg(temp_dir.path());
            cmd
        }
        UnpackMethod::Auto => {
            // This case should not be reached due to the logic above
            log::error!("Auto-detection failed for: {:?}", archive_path);
            return Err(UnpackerError::UnsupportedUnpackMethod(
                "Could not auto-detect format".to_string(),
            ));
        }
    };

    log::debug!("Executing unpack command: {:?}", command);
    let output = command.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        log::error!("Unpack command failed with stderr: {}", stderr);
        return Err(UnpackerError::Command(stderr));
    }

    log::debug!("Unpack command completed successfully");

    if resolved_dest.exists() {
        log::debug!("Destination already exists, removing: {:?}", resolved_dest);
        if resolved_dest.is_dir() {
            std::fs::remove_dir_all(&resolved_dest)?;
        } else {
            std::fs::remove_file(&resolved_dest)?;
        }
    }

    log::debug!(
        "Moving unpacked content from {:?} to {:?}",
        temp_dir.path(),
        resolved_dest
    );
    // tempfile::TempDir handles drop correctly, but we need to move the content out first.
    std::fs::rename(temp_dir.path(), &resolved_dest)?;

    log::debug!(
        "Successfully unpacked archive {:?} to {:?}",
        archive_path,
        resolved_dest
    );
    Ok(())
}

pub struct UnpackerPool {
    workers: Vec<thread::JoinHandle<()>>,
    task_tx: Option<Sender<UnpackerTask>>,
    completed_rx: crossbeam_channel::Receiver<TaskStatus>,
}

impl UnpackerPool {
    pub fn new(num_workers: usize, node_id: String, is_test: bool) -> Self {
        assert!(
            num_workers > 0,
            "UnpackerPool must have at least one worker."
        );

        let (task_tx, task_rx) = crossbeam_channel::unbounded::<UnpackerTask>();
        let (completed_tx, completed_rx) = crossbeam_channel::unbounded::<TaskStatus>();
        let active_tasks_count = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_workers);
        let node_id = node_id.to_string();

        for i in 0..num_workers {
            let rx = task_rx.clone();
            let active_count = Arc::clone(&active_tasks_count);
            let completed_tx_cloned = completed_tx.clone();
            let node_id_clone = node_id.clone();
            let is_test_clone = is_test;

            let handle = thread::spawn(move || {
                for job in rx {
                    log::debug!(
                        "[Unpacker Worker {}] Got a job for object {}, starting...",
                        i,
                        job.object_id
                    );
                    let _guard = ActiveTaskGuard::new(Arc::clone(&active_count));

                    let object_id = job.object_id;
                    let result = unpack_and_cache(
                        &job.archive_path,
                        &job.unpack_destination,
                        &job.unpack_method,
                        &node_id_clone,
                        is_test_clone,
                        &job.cache_info,
                    );

                    let status = match result {
                        Ok(_) => {
                            log::debug!(
                                "[Unpacker Worker {}] Successfully completed task for object {}",
                                i,
                                object_id
                            );
                            TaskStatus::Completed(job)
                        }
                        Err(e) => {
                            log::error!(
                                "[Unpacker Worker {}] Task failed for object {}: {:?}",
                                i,
                                object_id,
                                e
                            );
                            TaskStatus::Failed(job, e)
                        }
                    };

                    if let Err(err) = completed_tx_cloned.send(status) {
                        log::error!(
                            "[Unpacker Worker {}] Failed to send completion notification for object {}: {}. Shutting down worker.",
                            i,
                            object_id,
                            err
                        );
                        break; // Exit the worker loop gracefully
                    }

                    log::debug!(
                        "[Unpacker Worker {}] Finished job for object {}.",
                        i,
                        object_id
                    );
                }
                log::debug!("[Unpacker Worker {}] Shutting down.", i);
            });
            workers.push(handle);
        }

        log::debug!(
            "Created UnpackerPool with {} workers for node_id: {}, is_test: {}",
            num_workers,
            node_id,
            is_test
        );

        UnpackerPool {
            workers,
            task_tx: Some(task_tx),
            completed_rx,
        }
    }

    pub fn add_tasks(&self, jobs: impl IntoIterator<Item = UnpackerTask>) {
        if let Some(tx) = &self.task_tx {
            let mut task_count = 0;
            for job in jobs {
                log::debug!("Adding unpacker task for object: {}", job.object_id);
                tx.send(job).expect("Failed to send job to a worker.");
                task_count += 1;
            }
            log::debug!("Added {} unpacker tasks to the queue", task_count);
        } else {
            log::warn!("Attempted to add tasks to a shut down UnpackerPool");
        }
    }

    pub fn get_task_statuses(&self) -> Result<Vec<TaskStatus>, UnpackerError> {
        // Check worker health first - if any worker has died, return an error
        let mut dead_workers = 0;
        for (i, worker) in self.workers.iter().enumerate() {
            if worker.is_finished() {
                dead_workers += 1;
                log::error!("Unpacker Worker {} has died unexpectedly!", i);
            }
        }

        if dead_workers > 0 {
            log::error!(
                "{} out of {} unpacker workers have died",
                dead_workers,
                self.workers.len()
            );
            return Err(UnpackerError::WorkerPoolUnhealthy(
                dead_workers,
                self.workers.len(),
            ));
        }

        Ok(self.completed_rx.try_iter().collect())
    }
}

impl Drop for UnpackerPool {
    fn drop(&mut self) {
        log::debug!("--- [Drop] UnpackerPool is going out of scope. Shutting down. ---");
        if let Some(tx) = self.task_tx.take() {
            drop(tx);
        }
        for handle in self.workers.drain(..) {
            handle.join().expect("Unpacker worker thread panicked.");
        }
        log::debug!("--- [Drop] All unpacker workers have shut down. ---");
    }
}
