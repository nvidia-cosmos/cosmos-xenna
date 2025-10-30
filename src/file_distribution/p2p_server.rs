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

//! # P2P Server
//!
//! This module implements the peer-to-peer (P2P) server component of the Xenna data plane.
//! Its primary responsibility is to handle the exchange of file chunks between worker nodes in the cluster.
//!
//! ## Architecture
//!
//! The server is built using the `axum` web framework and runs within a `tokio` asynchronous runtime.
//! It exposes a simple HTTP API for other nodes to download and upload chunks. This approach was
//! chosen for its simplicity, debuggability, and standard-compliance over a custom RPC protocol.
//!
//! As outlined in the `README.md`, this server is a core part of the Rust-based "Data Plane,"
//! which is designed to offload all heavy data transfer tasks from the Python-based "Control Plane."
//!
//! ## Operations
//!
//! - **Serving Chunks:** When a peer requests a chunk (`GET /chunk/{chunk_id}`), the server first looks
//!   for the chunk in a temporary storage location (where newly downloaded chunks are kept). If not
//!   found, it checks the final destination path, allowing it to serve data that might already have been
//!   assembled. This allows flexibility in serving data.
//!
//! - **Receiving Chunks:** The server can accept chunks from peers via a `POST` request. This is used
//!   for seeding chunks across the network, where one node might push a chunk it has to another.
//!   Received chunks are always written to a temporary directory.
use super::common::{get_temp_chunk_path, resolve_path};
use axum::{
    Json, Router,
    extract::{Path, Query},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use log::debug;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde::Deserialize;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::{net::SocketAddr, path::PathBuf};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::runtime::Runtime;
use tokio::sync::broadcast;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum P2pServerError {
    #[error("Failed to send shutdown signal: {0}")]
    SendShutdownSignal(#[from] broadcast::error::SendError<()>),

    #[error("Server thread panicked")]
    ThreadPanic,

    #[error("Health check failed: {0}")]
    HealthCheck(#[from] reqwest::Error),
}

impl From<P2pServerError> for PyErr {
    fn from(err: P2pServerError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

static ACTIVE_UPLOADS: AtomicUsize = AtomicUsize::new(0);

struct ActiveUploadGuard;

impl ActiveUploadGuard {
    fn new() -> Self {
        ACTIVE_UPLOADS.fetch_add(1, Ordering::Relaxed);
        ActiveUploadGuard
    }
}

impl Drop for ActiveUploadGuard {
    fn drop(&mut self) {
        ACTIVE_UPLOADS.fetch_sub(1, Ordering::Relaxed);
    }
}

// Custom body wrapper that holds the guard during streaming
// This is used so that we can count the number of active uploads.
// If we didn't care about this, we could just use axum::body::Body::from(data)
struct StreamingBodyWithGuard {
    data: Vec<u8>,
    _guard: ActiveUploadGuard,
}

impl StreamingBodyWithGuard {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            _guard: ActiveUploadGuard::new(),
        }
    }

    fn into_body(self) -> axum::body::Body {
        // Create a stream that yields the data in chunks and holds the guard
        const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks for streaming
        let data = std::sync::Arc::new(self.data);
        let _guard = self._guard;

        let stream = futures::stream::unfold(
            (data, 0, Some(_guard)),
            |(data, offset, guard)| async move {
                if offset >= data.len() {
                    // Guard will be dropped here when stream ends
                    None
                } else {
                    let end = std::cmp::min(offset + CHUNK_SIZE, data.len());
                    let chunk = data[offset..end].to_vec();
                    Some((Ok::<_, std::io::Error>(chunk), (data, end, guard)))
                }
            },
        );

        axum::body::Body::from_stream(stream)
    }
}

#[derive(Clone)]
struct ServerConfig {
    node_id: String,
    is_test: bool,
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "healthy" })))
}

#[derive(Deserialize)]
struct ChunkParams {
    destination: PathBuf,
    range_start: Option<u64>,
    range_end: Option<u64>,
}

async fn chunk(
    Path(chunk_id): Path<Uuid>,
    Query(params): Query<ChunkParams>,
    axum::extract::State(config): axum::extract::State<Arc<ServerConfig>>,
) -> Result<Response, (StatusCode, String)> {
    debug!("Request received for chunk {}", chunk_id);

    let temp_chunk_path = get_temp_chunk_path(chunk_id, &config.node_id, config.is_test);
    debug!(
        "Checking for temporary chunk file {}",
        temp_chunk_path.display()
    );
    if temp_chunk_path.exists() {
        debug!("Temporary chunk file found {}", temp_chunk_path.display());
        let content = match tokio::fs::read(&temp_chunk_path).await {
            Ok(content) => content,
            Err(e) => {
                debug!(
                    "Failed to read chunk file {}: {}",
                    temp_chunk_path.display(),
                    e
                );
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to read chunk file {:?}: {}", temp_chunk_path, e),
                ));
            }
        };
        debug!("Successfully read temporary chunk file, sending content");
        let streaming_body = StreamingBodyWithGuard::new(content);
        return Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(streaming_body.into_body())
            .unwrap());
    }
    debug!("Temporary chunk file not found");

    let final_path = resolve_path(params.destination.clone(), &config.node_id, config.is_test);
    debug!(
        "Checking for final destination file {}",
        final_path.display()
    );
    if final_path.exists() {
        debug!("Final destination file found {}", final_path.display());
        let content = if let (Some(start), Some(end)) = (params.range_start, params.range_end) {
            debug!("Reading file range from {} to {}", start, end);
            if start > end {
                debug!("range_start cannot be greater than range_end");
                return Err((
                    StatusCode::BAD_REQUEST,
                    "range_start cannot be greater than range_end".to_string(),
                ));
            }
            let mut file = match tokio::fs::File::open(&final_path).await {
                Ok(file) => file,
                Err(e) => {
                    debug!("Failed to open final file {}: {}", final_path.display(), e);
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to open final file {:?}: {}", final_path, e),
                    ));
                }
            };
            if let Err(e) = file.seek(std::io::SeekFrom::Start(start)).await {
                debug!("Failed to seek in file {}: {}", final_path.display(), e);
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to seek in file {:?}: {}", final_path, e),
                ));
            }
            let len = (end - start) as usize;
            let mut buffer = vec![0; len];
            if let Err(e) = file.read_exact(&mut buffer).await {
                debug!(
                    "Failed to read range from file {}: {}",
                    final_path.display(),
                    e
                );
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to read range from file {:?}: {}", final_path, e),
                ));
            }
            buffer
        } else if params.range_start.is_some() || params.range_end.is_some() {
            debug!("Both range_start and range_end must be provided for a range request");
            return Err((
                StatusCode::BAD_REQUEST,
                "Both range_start and range_end must be provided for a range request".to_string(),
            ));
        } else {
            debug!("Reading full file {}", final_path.display());
            match tokio::fs::read(&final_path).await {
                Ok(content) => content,
                Err(e) => {
                    debug!("Failed to read file {}: {}", final_path.display(), e);
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read file {:?}: {}", final_path, e),
                    ));
                }
            }
        };
        debug!("Successfully read file, sending content");
        let streaming_body = StreamingBodyWithGuard::new(content);
        return Ok(Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(streaming_body.into_body())
            .unwrap());
    }
    debug!("Chunk not found");

    Err((
        StatusCode::NOT_FOUND,
        format!("Chunk {} not found", chunk_id),
    ))
}

async fn write_chunk(
    Path(chunk_id): Path<Uuid>,
    axum::extract::State(config): axum::extract::State<Arc<ServerConfig>>,
    body: axum::body::Body,
) -> Result<StatusCode, (StatusCode, String)> {
    let temp_chunk_path = get_temp_chunk_path(chunk_id, &config.node_id, config.is_test);
    debug!(
        "Writing chunk {} to {}",
        chunk_id,
        temp_chunk_path.display()
    );

    if let Some(parent) = temp_chunk_path.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            debug!("Failed to create directory {:?}: {}", parent, e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create directory {:?}: {}", parent, e),
            ));
        }
    }

    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(e) => {
            debug!("Failed to read request body: {}", e);
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Failed to read request body: {}", e),
            ));
        }
    };

    if let Err(e) = tokio::fs::write(&temp_chunk_path, &body_bytes).await {
        debug!(
            "Failed to write chunk file {}: {}",
            temp_chunk_path.display(),
            e
        );
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to write chunk file {:?}: {}", temp_chunk_path, e),
        ));
    }

    debug!(
        "Successfully wrote chunk {} to {}",
        chunk_id,
        temp_chunk_path.display()
    );
    Ok(StatusCode::OK)
}

fn app(config: Arc<ServerConfig>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/chunk/:chunk_id", get(chunk).post(write_chunk))
        .with_state(config)
}

async fn server_main(
    mut shutdown_rx: broadcast::Receiver<()>,
    listener: tokio::net::TcpListener,
    config: Arc<ServerConfig>,
) {
    let app = app(config);

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_rx.recv().await.ok();
        })
        .await
        .unwrap();
}

#[pyclass]
pub struct P2pServer {
    shutdown_tx: broadcast::Sender<()>,
    server_handle: Option<thread::JoinHandle<()>>,
    addr: SocketAddr,
}

impl P2pServer {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    fn _shutdown_internal(&mut self) -> Result<(), P2pServerError> {
        if self.server_handle.is_none() {
            return Ok(());
        }

        self.shutdown_tx.send(())?;
        if let Some(handle) = self.server_handle.take() {
            if handle.join().is_err() {
                return Err(P2pServerError::ThreadPanic);
            }
        }
        Ok(())
    }
}

#[pymethods]
impl P2pServer {
    #[new]
    #[pyo3(signature = (port, node_id, is_test=false))]
    pub fn new(port: u16, node_id: String, is_test: bool) -> Self {
        // Reset the counter to prevent accumulation from previous instances
        ACTIVE_UPLOADS.store(0, Ordering::Relaxed);

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let shutdown_tx_clone = shutdown_tx.clone();

        let addr = if is_test {
            SocketAddr::from(([127, 0, 0, 1], port))
        } else {
            SocketAddr::from(([0, 0, 0, 0], port))
        };
        let server_config = Arc::new(ServerConfig { node_id, is_test });

        let server_handle = thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(listener) => listener,
                    Err(e) => {
                        eprintln!("Failed to bind to {}: {}", addr, e);
                        return;
                    }
                };
                println!("P2P server listening on {}", addr);
                server_main(shutdown_rx, listener, server_config).await;
            });
        });

        P2pServer {
            shutdown_tx: shutdown_tx_clone,
            server_handle: Some(server_handle),
            addr,
        }
    }

    pub fn check_health(&self) -> Result<(), P2pServerError> {
        let url = format!("http://{}/health", self.addr);
        reqwest::blocking::get(&url)?
            .error_for_status()
            .map(|_| ())
            .map_err(P2pServerError::HealthCheck)
    }

    pub fn active_uploads(&self) -> usize {
        ACTIVE_UPLOADS.load(Ordering::Relaxed)
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        self._shutdown_internal().map_err(PyErr::from)
    }
}

impl Drop for P2pServer {
    fn drop(&mut self) {
        if let Err(e) = self._shutdown_internal() {
            eprintln!("Error shutting down P2P server during drop: {}", e);
        }
    }
}
