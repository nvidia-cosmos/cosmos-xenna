# Distributed File Distribution System

The Xenna file distribution system provides efficient, peer-to-peer (P2P) downloading of artifacts across Ray clusters. It's designed to eliminate redundant downloads when multiple nodes need the same files (model weights, python environments, etc.) by downloading each chunk only once and sharing it between nodes.

## Overview

When running pipelines on multi-node clusters, you often need the same large files (model weights, conda environments) available on every node. Traditional approaches would download these files independently to each node, leading to:

- **Redundant object store bandwidth usage** (expensive egress costs)
- **Slower overall download times** (rate limiting)
- **Network congestion** (multiple nodes hitting the same endpoints)

The file distribution system solves this by:

1. **Chunking large files** into smaller pieces (default 100MB chunks)
2. **Downloading each chunk only once** from S3 across the entire cluster
3. **Sharing chunks between nodes** via fast peer-to-peer HTTP transfers
4. **Intelligently scheduling** downloads to optimize for speed
5. **Caching results** with metadata validation to avoid re-downloads

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
│  • Collects download_requests from all stages                  │
│  • Calls download_distributed() before pipeline starts         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Distributed Download System                     │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Node A    │    │   Node B    │    │   Node C    │        │
│  │             │    │             │    │             │        │
│  │ NodeWorker  │◄──►│ NodeWorker  │◄──►│ NodeWorker  │        │
│  │ P2P Server  │    │ P2P Server  │    │ P2P Server  │        │
│  │ Downloads   │    │ Downloads   │    │ Downloads   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         └───────────────────┼───────────────────┘              │
│                             │                                  │
│                    ┌─────────────┐                             │
│                    │  Scheduler  │                             │
│                    │ • Rarest-   │                             │
│                    │   first     │                             │
│                    │ • Load      │                             │
│                    │   balancing │                             │
│                    └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Object Store (S3)                         │
│  • Each chunk downloaded only once across entire cluster       │
│  • Supports multiple authentication profiles                   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **NodeWorker** (Ray Actor)

- Runs on each cluster node as a Ray actor
- Downloads chunks from S3 and assembles complete files
- Runs HTTP server to serve chunks to peer nodes
- Handles file unpacking (TAR, GZ, ZIP) when requested
- Manages local caching with metadata validation

#### 2. **Scheduler** (Centralized Coordinator)

- Implements BitTorrent-inspired "rarest-first" algorithm
- Prioritizes downloading rare chunks to improve swarm health
- Load balances across nodes to prevent bottlenecks
- Optimizes for both speed and S3 bandwidth costs
- Respects per-node resource limits (upload/download slots)

#### 3. **Rust Data Plane** (High Performance Core)

- Handles actual file I/O operations in Rust for performance
- Manages chunk assembly, file validation, and unpacking
- Provides HTTP server for P2P transfers
- Implements robust error handling and retry logic

## Pipeline Integration

### Stage-Level Download Requests

Stages can specify files they need by implementing the `download_requests` property:

```python
from cosmos_xenna import file_distribution
from cosmos_xenna.pipelines import v1 as pipelines_v1
import pathlib

class MyModelStage(pipelines_v1.Stage):
    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        return [
            # Download a single model file
            file_distribution.DownloadRequest(
                value=file_distribution.ObjectDownloadRequest(
                    uri="models/my-model/pytorch_model.bin",
                    destination=pathlib.Path("/tmp/models/pytorch_model.bin"),
                    cache=True  # Enable caching
                )
            ),
            
            # Download and extract an archive
            file_distribution.DownloadRequest(
                value=file_distribution.ObjectDownloadRequest(
                    uri="datasets/my-dataset.tar.gz", 
                    destination=pathlib.Path("/tmp/datasets/my-dataset.tar.gz"),
                    unpack_options=file_distribution.UnpackOptions(
                        destination=pathlib.Path("/tmp/datasets/extracted/"),
                        unpack_method=file_distribution.UnpackMethod.Auto
                    )
                )
            ),
            
            # Download all files under a prefix
            file_distribution.DownloadRequest(
                value=file_distribution.PrefixDownloadRequest(
                    uri="tokenizers/my-tokenizer/",
                    destination=pathlib.Path("/tmp/tokenizers/")
                )
            )
        ]
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # Files are guaranteed to be available locally at this point
        self.model = torch.load("/tmp/models/pytorch_model.bin")
```

### Pipeline Configuration

Configure the download system when running pipelines:

```python
from cosmos_xenna.pipelines.private import pipelines, specs
from cosmos_xenna import file_distribution

# Configure object store authentication
download_config = pipelines.DistributedDownloadConfig(
    object_store_config=file_distribution.ObjectStoreConfig.make_for_s3(
        bucket="my-models-bucket",
        access_key_id="...",
        secret_access_key="...",
        region="us-west-2"
    ),
    chunk_size_bytes=50 * 1024 * 1024,  # 50MB chunks
    node_parallelism=20,  # Max concurrent downloads per node
    object_store_parallelism=1000,  # Max concurrent S3 connections
    verbose=True
)

# Run pipeline with distributed downloads
pipeline_spec = specs.PipelineSpec(
    input_data=my_input_data,
    stages=[MyModelStage(), MyProcessingStage()]
)

results = pipelines.run_pipeline(
    pipeline_spec=pipeline_spec,
    distibuted_download_config=download_config
)
```

## Advanced Features

### Multi-Profile Authentication

Support different credentials for different object stores:

```python
# Configure multiple S3 profiles
config_by_profile = file_distribution.ObjectStoreConfigByProfile(
    profiles={
        "models": file_distribution.ObjectStoreConfig.make_for_s3(
            bucket="company-models", 
            access_key_id="...", 
            secret_access_key="..."
        ),
        "datasets": file_distribution.ObjectStoreConfig.make_for_s3(
            bucket="public-datasets",
            access_key_id="...",
            secret_access_key="..."
        )
    }
)

# Use different profiles in download requests
@property
def download_requests(self) -> list[file_distribution.DownloadRequest]:
    return [
        file_distribution.DownloadRequest(
            value=file_distribution.ObjectDownloadRequest(
                profile_name="models",  # Use models profile
                uri="llama2/pytorch_model.bin",
                destination=pathlib.Path("/tmp/model.bin")
            )
        ),
        file_distribution.DownloadRequest(
            value=file_distribution.ObjectDownloadRequest(
                profile_name="datasets",  # Use datasets profile  
                uri="imagenet/train.tar",
                destination=pathlib.Path("/tmp/train.tar")
            )
        )
    ]
```

### Intelligent Caching

On subsequent runs, the system:

1. Checks if cached file exists
2. Validates size and timestamp against S3 metadata
3. Skips download if cache is valid
4. Re-downloads if cache is stale or corrupted

### Archive Extraction

Automatic extraction of common archive formats:

```python
file_distribution.DownloadRequest(
    value=file_distribution.ObjectDownloadRequest(
        uri="models/bert-base.tar.gz",
        destination=pathlib.Path("/tmp/bert-base.tar.gz"),
        unpack_options=file_distribution.UnpackOptions(
            destination=pathlib.Path("/tmp/bert-base/"),
            unpack_method=file_distribution.UnpackMethod.Auto  # Auto-detect format
        )
    )
)

# Supported formats:
# - TAR (.tar)
# - TAR.GZ (.tar.gz, .tgz) 
# - ZIP (.zip)
# - Auto-detection based on file extension
```
