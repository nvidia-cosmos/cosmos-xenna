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

//! # File Writer Pool
//!
//! This module implements a thread pool for asynchronous file writing operations.
//! It's designed to pipeline network downloads with disk I/O, preventing download
//! workers from being blocked on disk operations.
//!
//! ## Purpose
//!
//! By separating file writing into its own thread pool, we can:
//! - Keep download workers continuously downloading without blocking on disk I/O
//! - Smooth out P2P upload availability (chunks become available more steadily)
//! - Better utilize both network and disk bandwidth
//!
//! ## Architecture
//!
//! The pool receives `FileWriterTask` items containing in-memory chunk data and
//! writes them to disk atomically using temporary files. This ensures data integrity
//! even if the process crashes during writing.

use crate::file_distribution::common::get_temp_chunk_path;
use crossbeam_channel::Sender;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tempfile::NamedTempFile;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum FileWriterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to persist temporary file: {0}")]
    Persist(#[from] tempfile::PersistError),
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
pub struct FileWriterTask {
    pub chunk_id: Uuid,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub enum TaskStatus {
    Completed(FileWriterTask),
    Failed(FileWriterTask, FileWriterError),
}

/// A thread pool for writing files to disk asynchronously.
pub struct FileWriterPool {
    workers: Vec<thread::JoinHandle<()>>,
    task_tx: Option<Sender<FileWriterTask>>,
    active_tasks_count: Arc<AtomicUsize>,
    completed_rx: crossbeam_channel::Receiver<TaskStatus>,
}

fn write_chunk_to_disk(
    task: &FileWriterTask,
    node_id: &str,
    is_test: bool,
) -> Result<(), FileWriterError> {
    let chunk_destination = get_temp_chunk_path(task.chunk_id, node_id, is_test);

    // Create a temporary file in the same directory as the final destination
    // This ensures the atomic rename will work (same filesystem)
    let temp_dir = chunk_destination.parent().unwrap();
    let mut temp_file = NamedTempFile::new_in(temp_dir)?;

    // Write the data
    temp_file.write_all(&task.data)?;

    // Ensure data is written to disk before persist
    temp_file.flush()?;
    temp_file.as_file().sync_all()?;

    // Atomically move the temporary file to the final destination
    temp_file.persist(&chunk_destination)?;

    Ok(())
}

impl FileWriterPool {
    /// Creates a new FileWriterPool with a specified number of worker threads.
    ///
    /// # Panics
    /// Panics if `num_workers` is 0.
    pub fn new(num_workers: usize, node_id: String, is_test: bool) -> Self {
        assert!(
            num_workers > 0,
            "FileWriterPool must have at least one worker."
        );

        let (task_tx, task_rx) = crossbeam_channel::unbounded::<FileWriterTask>();
        let (completed_tx, completed_rx) = crossbeam_channel::unbounded::<TaskStatus>();
        let active_tasks_count = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            let rx = task_rx.clone();
            let active_count = Arc::clone(&active_tasks_count);
            let completed_tx_cloned = completed_tx.clone();
            let node_id_clone = node_id.clone();

            let handle = thread::spawn(move || {
                for task in rx {
                    log::debug!(
                        "[FileWriter {}] Writing chunk {} to disk...",
                        i,
                        task.chunk_id
                    );
                    let _guard = ActiveTaskGuard::new(Arc::clone(&active_count));
                    let chunk_id = task.chunk_id;

                    let result = write_chunk_to_disk(&task, &node_id_clone, is_test);

                    let status = match result {
                        Ok(_) => {
                            log::debug!("[FileWriter {}] Successfully wrote chunk {}", i, chunk_id);
                            TaskStatus::Completed(task)
                        }
                        Err(e) => {
                            log::warn!(
                                "[FileWriter {}] Failed to write chunk {}: {}",
                                i,
                                chunk_id,
                                e
                            );
                            TaskStatus::Failed(task, e)
                        }
                    };

                    if let Err(err) = completed_tx_cloned.send(status) {
                        log::warn!(
                            "Failed to send completion notification for chunk {}: {}",
                            chunk_id,
                            err
                        );
                    }

                    log::debug!("[FileWriter {}] Finished writing chunk {}.", i, chunk_id);
                }
                log::debug!("[FileWriter {}] Shutting down.", i);
            });
            workers.push(handle);
        }

        FileWriterPool {
            workers,
            task_tx: Some(task_tx),
            active_tasks_count,
            completed_rx,
        }
    }

    /// Adds a collection of tasks to the queue.
    pub fn add_tasks(&self, tasks: impl IntoIterator<Item = FileWriterTask>) {
        if let Some(tx) = &self.task_tx {
            for task in tasks {
                tx.send(task).expect("Failed to send task to file writer.");
            }
        }
    }

    /// Gets completed task statuses without blocking.
    pub fn get_task_statuses(&self) -> Result<Vec<TaskStatus>, FileWriterError> {
        if self.workers.iter().any(|handle| handle.is_finished()) {
            return Err(FileWriterError::WorkerPanicked);
        }
        // Drain any completed tasks without blocking
        Ok(self.completed_rx.try_iter().collect())
    }

    /// Returns the total number of tasks that are either queued or currently
    /// being processed by a worker.
    pub fn get_num_queued_or_active_tasks(&self) -> usize {
        let queued_count = self.task_tx.as_ref().map_or(0, |tx| tx.len());
        let active_count = self.active_tasks_count.load(Ordering::Relaxed);
        queued_count + active_count
    }

    /// Returns the number of tasks currently being processed.
    pub fn get_num_active_tasks(&self) -> usize {
        self.active_tasks_count.load(Ordering::Relaxed)
    }
}

impl Drop for FileWriterPool {
    fn drop(&mut self) {
        log::debug!("--- [Drop] FileWriterPool is going out of scope. Shutting down. ---");

        // Drop the sender to close the channel
        if let Some(tx) = self.task_tx.take() {
            drop(tx);
        }

        // Join all worker threads
        for handle in self.workers.drain(..) {
            handle.join().expect("File writer thread panicked.");
        }

        log::debug!("--- [Drop] All file writer workers have shut down. ---");
    }
}

