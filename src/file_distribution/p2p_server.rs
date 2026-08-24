// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//!   The chunk's destination and byte range are resolved from this node's own download catalog rather
//!   than from the request, so a peer cannot steer the handler at a path outside the download job.
//!
//! - **Receiving Chunks:** The server can accept chunks from peers via a `POST` request. This is used
//!   for seeding chunks across the network, where one node might push a chunk it has to another.
//!   Received chunks are always written to a temporary directory. Only chunks listed in this node's
//!   download catalog are accepted, and the body must be exactly the size the catalog declares.
use super::common::{get_temp_chunk_path, resolve_path};
use super::models::DownloadCatalog;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, Query},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use log::debug;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
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

    #[error("Failed to bind to {addr}: {source}")]
    Bind {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },

    #[error("Server thread exited before signaling readiness")]
    StartupAborted,

    #[error("Download catalog is invalid: chunk {chunk_id} has range start {start} > end {end}")]
    InvalidCatalogRange {
        chunk_id: Uuid,
        start: u64,
        end: u64,
    },
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
        let data: Bytes = self.data.into();
        let _guard = self._guard;

        let stream = futures::stream::unfold(
            (data, 0usize, Some(_guard)),
            |(data, offset, guard)| async move {
                if offset >= data.len() {
                    // Guard will be dropped here when stream ends
                    None
                } else {
                    let end = std::cmp::min(offset + CHUNK_SIZE, data.len());
                    let chunk = data.slice(offset..end);
                    Some((Ok::<_, std::io::Error>(chunk), (data, end, guard)))
                }
            },
        );

        axum::body::Body::from_stream(stream)
    }
}

/// Authoritative, locally-derived facts about a chunk this node may serve.
///
/// Built from the node's `DownloadCatalog`, never from request input. The catalog
/// already pins a chunk's destination, byte range, and size, so the server has no
/// reason to trust a peer's claims about any of them.
#[derive(Debug, Clone)]
struct ServableChunk {
    destination: PathBuf,
    range: Option<std::ops::Range<u64>>,
    size: u64,
}

/// The set of chunks a node may serve or accept, validated once at construction.
///
/// Building this is the only way to get a `P2pServer`, so a catalog defect is
/// reported before any listener opens rather than as a per-request error.
#[derive(Debug, Clone, Default)]
pub struct ServableChunks(HashMap<Uuid, ServableChunk>);

impl ServableChunks {
    pub fn from_catalog(catalog: &DownloadCatalog) -> Result<Self, P2pServerError> {
        catalog
            .chunks
            .iter()
            .map(|(chunk_id, chunk)| {
                let range = chunk.value.range.as_ref().map(|r| r.start..r.end);
                // An inverted range is a defect in the catalog we were handed, not
                // something a peer can cause or a retry can fix.
                if let Some(range) = &range
                    && range.start > range.end
                {
                    return Err(P2pServerError::InvalidCatalogRange {
                        chunk_id: *chunk_id,
                        start: range.start,
                        end: range.end,
                    });
                }
                Ok((
                    *chunk_id,
                    ServableChunk {
                        destination: chunk.destination.clone(),
                        range,
                        size: chunk.size,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>, _>>()
            .map(Self)
    }

    fn get(&self, chunk_id: &Uuid) -> Option<&ServableChunk> {
        self.0.get(chunk_id)
    }
}

#[derive(Clone)]
struct ServerConfig {
    node_id: String,
    is_test: bool,
    /// Chunks this node is permitted to serve, keyed by chunk id.
    servable_chunks: Arc<ServableChunks>,
}

async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "status": "healthy" })))
}

/// Query parameters accepted for wire compatibility with older peers.
///
/// All of these are redundant with the download catalog and are deliberately
/// ignored; see `chunk`.
#[derive(Deserialize)]
struct ChunkParams {
    #[allow(dead_code)]
    destination: Option<PathBuf>,
    #[allow(dead_code)]
    range_start: Option<u64>,
    #[allow(dead_code)]
    range_end: Option<u64>,
}

async fn chunk(
    Path(chunk_id): Path<Uuid>,
    Query(_params): Query<ChunkParams>,
    axum::extract::State(config): axum::extract::State<Arc<ServerConfig>>,
) -> Result<Response, (StatusCode, String)> {
    debug!("Request received for chunk {}", chunk_id);

    // The destination and byte range are taken from the local catalog, not from the
    // request. A peer that asks for a chunk this node was never told to download gets
    // nothing, so a caller cannot steer this handler at an arbitrary path.
    let servable = config.servable_chunks.get(&chunk_id).ok_or_else(|| {
        debug!("Chunk {} is not in this node's download catalog", chunk_id);
        (
            StatusCode::NOT_FOUND,
            format!("Chunk {} not found", chunk_id),
        )
    })?;

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

    let final_path = resolve_path(
        servable.destination.clone(),
        &config.node_id,
        config.is_test,
    );
    debug!(
        "Checking for final destination file {}",
        final_path.display()
    );
    if final_path.exists() {
        debug!("Final destination file found {}", final_path.display());
        let content = if let Some(ref range) = servable.range {
            let (start, end) = (range.start, range.end);
            debug!("Reading file range from {} to {}", start, end);
            // Unreachable: `P2pServer::new` rejects inverted catalog ranges at startup.
            // Kept because the `end - start` below would underflow if that ever regressed.
            if start > end {
                debug!("catalog range start is greater than range end");
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "catalog range start cannot be greater than range end".to_string(),
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
    // Only chunks this node was told to download may be seeded to it. This bounds both
    // the set of writable paths and the accepted body size to what the catalog declares.
    let servable = config.servable_chunks.get(&chunk_id).ok_or_else(|| {
        debug!("Refusing write for chunk {} (not in catalog)", chunk_id);
        (
            StatusCode::NOT_FOUND,
            format!("Chunk {} not found", chunk_id),
        )
    })?;

    let temp_chunk_path = get_temp_chunk_path(chunk_id, &config.node_id, config.is_test);
    debug!(
        "Writing chunk {} to {}",
        chunk_id,
        temp_chunk_path.display()
    );

    if let Some(parent) = temp_chunk_path.parent()
        && let Err(e) = tokio::fs::create_dir_all(parent).await
    {
        debug!("Failed to create directory {:?}: {}", parent, e);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create directory {:?}: {}", parent, e),
        ));
    }

    // Cap the buffered body at the size the catalog declares for this chunk. An
    // unbounded limit here lets any caller that can reach the port exhaust node memory.
    let expected_size = usize::try_from(servable.size).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Chunk {} size does not fit in memory", chunk_id),
        )
    })?;
    let body_bytes = match axum::body::to_bytes(body, expected_size).await {
        Ok(bytes) => bytes,
        Err(e) => {
            debug!("Failed to read request body: {}", e);
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Failed to read request body: {}", e),
            ));
        }
    };

    // A short body would otherwise be written and then served to peers as if it were
    // the whole chunk, since GET prefers the temporary file over the assembled one.
    if body_bytes.len() != expected_size {
        debug!(
            "Refusing truncated write for chunk {}: got {} bytes, expected {}",
            chunk_id,
            body_bytes.len(),
            expected_size
        );
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Chunk {} must be exactly {} bytes, got {}",
                chunk_id,
                expected_size,
                body_bytes.len()
            ),
        ));
    }

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
        if let Some(handle) = self.server_handle.take()
            && handle.join().is_err()
        {
            return Err(P2pServerError::ThreadPanic);
        }
        Ok(())
    }
}

impl P2pServer {
    pub fn new(
        port: u16,
        node_id: String,
        servable_chunks: Arc<ServableChunks>,
        is_test: bool,
    ) -> Result<Self, P2pServerError> {
        // Reset the counter to prevent accumulation from previous instances
        ACTIVE_UPLOADS.store(0, Ordering::Relaxed);

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let shutdown_tx_clone = shutdown_tx.clone();

        let addr = if is_test {
            SocketAddr::from(([127, 0, 0, 1], port))
        } else {
            SocketAddr::from(([0, 0, 0, 0], port))
        };
        let server_config = Arc::new(ServerConfig {
            node_id,
            is_test,
            servable_chunks,
        });

        // Signal readiness back to the caller once the TCP listener is bound.
        // This prevents a race where the server isn't yet accepting connections
        // when the first health check arrives.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), std::io::Error>>();

        let server_handle = thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(listener) => listener,
                    Err(e) => {
                        eprintln!("Failed to bind to {}: {}", addr, e);
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                println!("P2P server listening on {}", addr);
                if ready_tx.send(Ok(())).is_err() {
                    // Caller already gave up; don't bother serving.
                    return;
                }
                server_main(shutdown_rx, listener, server_config).await;
            });
        });

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(P2pServer {
                shutdown_tx: shutdown_tx_clone,
                server_handle: Some(server_handle),
                addr,
            }),
            Ok(Err(source)) => Err(P2pServerError::Bind { addr, source }),
            Err(_) => Err(P2pServerError::StartupAborted),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file_distribution::models::{ByteRange, ChunkToDownload, ObjectAndRange};
    use std::collections::HashMap;

    fn catalog_with_chunk(chunk_id: Uuid, destination: PathBuf, size: u64) -> DownloadCatalog {
        let mut chunks = HashMap::new();
        chunks.insert(
            chunk_id,
            ChunkToDownload {
                chunk_id,
                parent_object_id: Uuid::new_v4(),
                profile_name: None,
                value: ObjectAndRange {
                    object_uri: "s3://bucket/key".to_string(),
                    range: None,
                    crc32_checksum: None,
                },
                destination,
                size,
            },
        );
        DownloadCatalog {
            objects: HashMap::new(),
            chunks,
            chunks_by_object: HashMap::new(),
        }
    }

    fn start_server(node_id: &str, catalog: &DownloadCatalog) -> P2pServer {
        let port = portpicker::pick_unused_port().unwrap();
        let servable = Arc::new(ServableChunks::from_catalog(catalog).unwrap());
        P2pServer::new(port, node_id.to_string(), servable, true).unwrap()
    }

    /// A chunk id that is not in the catalog must never cause a file read, no matter
    /// what path the caller names. This is the arbitrary-file-read regression guard.
    #[test]
    fn unknown_chunk_id_does_not_read_caller_supplied_path() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let secret_dir = tempfile::tempdir().unwrap();
        let secret = secret_dir.path().join("secret.txt");
        std::fs::write(&secret, b"topsecret").unwrap();

        // Empty catalog: the node was never told to download anything.
        let catalog = DownloadCatalog {
            objects: HashMap::new(),
            chunks: HashMap::new(),
            chunks_by_object: HashMap::new(),
        };
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), Uuid::new_v4());
        let response = reqwest::blocking::Client::new()
            .get(&url)
            .query(&[("destination", secret.to_str().unwrap())])
            .send()
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.text().unwrap();
        assert!(
            !body.contains("topsecret"),
            "server leaked caller-supplied file contents: {body}"
        );
    }

    /// For a chunk that *is* in the catalog, the destination comes from the catalog.
    /// A malicious `destination` query parameter must be ignored, not honored.
    #[test]
    fn catalog_destination_wins_over_caller_supplied_destination() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let chunk_id = Uuid::new_v4();

        // In test mode `resolve_path` sandboxes under P2P_DOWNLOAD_TEST_DIR/<node_id>,
        // so materialize the catalog file where the handler will look for it.
        let catalog_destination = PathBuf::from("/artifacts/model.bin");
        let resolved = resolve_path(catalog_destination.clone(), &node_id, true);
        std::fs::create_dir_all(resolved.parent().unwrap()).unwrap();
        std::fs::write(&resolved, b"legitimate-chunk").unwrap();

        let secret_dir = tempfile::tempdir().unwrap();
        let secret = secret_dir.path().join("secret.txt");
        std::fs::write(&secret, b"topsecret").unwrap();

        let catalog = catalog_with_chunk(chunk_id, catalog_destination, 16);
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), chunk_id);
        let response = reqwest::blocking::Client::new()
            .get(&url)
            .query(&[("destination", secret.to_str().unwrap())])
            .send()
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.bytes().unwrap().as_ref(), b"legitimate-chunk");
    }

    /// Writes are restricted to catalog chunks, so an unknown id cannot be used to
    /// buffer an unbounded body in memory.
    #[test]
    fn write_rejects_chunk_absent_from_catalog() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let catalog = DownloadCatalog {
            objects: HashMap::new(),
            chunks: HashMap::new(),
            chunks_by_object: HashMap::new(),
        };
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), Uuid::new_v4());
        let response = reqwest::blocking::Client::new()
            .post(&url)
            .body(vec![0u8; 1024])
            .send()
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// A body larger than the size the catalog declares for the chunk is refused
    /// rather than buffered.
    #[test]
    fn write_rejects_body_larger_than_catalog_chunk_size() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let chunk_id = Uuid::new_v4();
        let declared_size = 64u64;
        let catalog =
            catalog_with_chunk(chunk_id, PathBuf::from("/artifacts/x.bin"), declared_size);
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), chunk_id);
        let response = reqwest::blocking::Client::new()
            .post(&url)
            .body(vec![0u8; (declared_size as usize) * 100])
            .send()
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    /// A short body must not be written: GET prefers the temporary chunk file, so a
    /// truncated write would be served to peers as though it were the whole chunk.
    #[test]
    fn write_rejects_truncated_body() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let chunk_id = Uuid::new_v4();
        let declared_size = 64u64;
        let catalog =
            catalog_with_chunk(chunk_id, PathBuf::from("/artifacts/x.bin"), declared_size);
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), chunk_id);
        for short_len in [0usize, 1, declared_size as usize - 1] {
            let response = reqwest::blocking::Client::new()
                .post(&url)
                .body(vec![0u8; short_len])
                .send()
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "a {short_len}-byte body must be rejected"
            );
        }

        // The temporary chunk file must not exist after the rejected writes.
        let temp_path = get_temp_chunk_path(chunk_id, &node_id, true);
        assert!(
            !temp_path.exists(),
            "a rejected write left a partial chunk at {}",
            temp_path.display()
        );
    }

    /// An exactly-sized body is still accepted.
    #[test]
    fn write_accepts_exactly_sized_body() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let chunk_id = Uuid::new_v4();
        let declared_size = 64u64;
        let catalog =
            catalog_with_chunk(chunk_id, PathBuf::from("/artifacts/x.bin"), declared_size);
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), chunk_id);
        let response = reqwest::blocking::Client::new()
            .post(&url)
            .body(vec![7u8; declared_size as usize])
            .send()
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let temp_path = get_temp_chunk_path(chunk_id, &node_id, true);
        assert_eq!(std::fs::read(&temp_path).unwrap(), vec![7u8; 64]);
    }

    /// An inverted catalog range is rejected up front, rather than surfacing as a
    /// per-request error that no retry could resolve.
    #[test]
    fn building_servable_chunks_rejects_inverted_range() {
        let chunk_id = Uuid::new_v4();
        let mut catalog = catalog_with_chunk(chunk_id, PathBuf::from("/artifacts/x.bin"), 4);
        catalog.chunks.get_mut(&chunk_id).unwrap().value.range =
            Some(ByteRange { start: 9, end: 3 });

        match ServableChunks::from_catalog(&catalog) {
            Ok(_) => panic!("an inverted catalog range must be rejected"),
            Err(P2pServerError::InvalidCatalogRange {
                chunk_id: id,
                start: 9,
                end: 3,
            }) => assert_eq!(id, chunk_id),
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    /// The range served comes from the catalog, so a caller cannot use range
    /// parameters to pull arbitrary offsets out of a file.
    #[test]
    fn catalog_range_is_authoritative() {
        let node_id = format!("test-node-{}", Uuid::new_v4());
        let chunk_id = Uuid::new_v4();

        let catalog_destination = PathBuf::from("/artifacts/ranged.bin");
        let resolved = resolve_path(catalog_destination.clone(), &node_id, true);
        std::fs::create_dir_all(resolved.parent().unwrap()).unwrap();
        std::fs::write(&resolved, b"0123456789").unwrap();

        let mut catalog = catalog_with_chunk(chunk_id, catalog_destination, 4);
        catalog.chunks.get_mut(&chunk_id).unwrap().value.range =
            Some(ByteRange { start: 2, end: 6 });
        let server = start_server(&node_id, &catalog);

        let url = format!("http://{}/chunk/{}", server.addr(), chunk_id);
        let response = reqwest::blocking::Client::new()
            .get(&url)
            .query(&[("range_start", "0"), ("range_end", "10")])
            .send()
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        // Catalog range 2..6, not the caller's 0..10.
        assert_eq!(response.bytes().unwrap().as_ref(), b"2345");
    }
}
