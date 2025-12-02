# Cosmos-xenna

## Introduction

Cosmos-xenna is a Python library for building and running distributed data pipelines using Ray. It
has a heavy focus on pipelines which are a series of inference steps using AI models. For example, a
pipeline which downloads an image, runs a VLM on it to produce a caption, and then runs an embedding model
to produce a text embedding and uploads the resulting data.

Cosmos-xenna simplifies the development of distributed AI pipelines by providing:

- A simple interface
- Autoscaling/autobalancing of stages
- Stateful actors which allow the user to load/download weights before running processing
- Large model support via SPMD mode

## What is Xenna?

Xenna is a distributed pipeline orchestrator that manages multi-stage data processing workflows across Ray clusters. Think of it as a production-ready framework for running AI inference at scale.

**The Core Concept**: You define a linear pipeline as a series of **stages**, where each stage transforms data and passes it to the next stage. Xenna handles all the complexity of:

- **Resource Management**: Automatically allocating CPUs and GPUs to stage workers
- **Autoscaling**: Dynamically adjusting the number of workers per stage to balance throughput
- **Backpressure**: Managing memory by controlling how much data queues between stages
- **Monitoring**: Providing real-time visibility into pipeline performance

**Execution Modes**: Xenna supports three execution modes:

1. **Streaming Mode** (recommended): All stages run concurrently. Data flows through the pipeline continuously, with Xenna automatically balancing worker counts to maximize throughput. This mode minimizes memory usage and is ideal for processing large datasets.

2. **Batch Mode**: Each stage completes fully before the next stage begins. Simpler but requires materializing all intermediate data in memory.

3. **Serving Mode**: This is for online serving scenario, where input data arrives in real time. It behaves very similar to *streaming mode* except that it takes a pair of input (source) & output (sink) queues to poll input & push processed data, instead of taking a pre-populated list of input data and optionally returns output data in the end.

**Architecture**: Xenna is built on top of Ray and is heavily inspired by Ray Data. It creates Ray Actors for each stage worker and runs a main orchestrator loop that constantly polls actors and moves data between stages. The streaming autoscaler tries to ensure that the throughput of all stages is equal and maximized, using complex bin-packing algorithms to optimize GPU utilization.

Most of the code is written in Python, although some of the autoscaling and artifact distribution logic is written in Rust for speed.

## How Pipelines Work

A Xenna pipeline is defined by implementing `Stage` classes and connecting them in a `PipelineSpec`. Here's a concrete example from an end-to-end video captioning pipeline:

```python
import cosmos_xenna.pipelines.v1 as pipelines_v1

# Stage 1: Download videos from S3
class DownloadStage(pipelines_v1.Stage):
    @property
    def stage_batch_size(self) -> int:
        return 10  # Process 10 videos at a time
    
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # Initialize S3 client (runs once per worker)
        self._s3_client = make_s3_client()
    
    def process_data(self, samples: list[Sample]) -> list[Sample]:
        # Download video bytes from S3 (runs for each batch)
        for sample in samples:
            sample.buffer = self._s3_client.download(sample.url)
        return samples

# Stage 2: Run VLM to caption videos
class CaptionStage(pipelines_v1.Stage):
    @property
    def stage_batch_size(self) -> int:
        return 5  # Larger batches would OOM the GPU
    
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=1.0, cpus=1.0)
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # Load model weights (runs once per worker)
        self._model = load_vlm_model()
    
    def process_data(self, samples: list[Sample]) -> list[Sample]:
        # Run inference (runs for each batch)
        for sample in samples:
            sample.caption = self._model.generate(sample.buffer)
        return samples

# Stage 3: Upload captions to S3 and write metadata to Postgres
class WriterStage(pipelines_v1.Stage):
    @property
    def stage_batch_size(self) -> int:
        return 10
    
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._s3_client = make_s3_client()
        self._db = make_postgres_connection()
    
    def process_data(self, samples: list[Sample]) -> list[Sample]:
        # Upload captions and write database records
        for sample in samples:
            self._s3_client.upload(sample.caption, sample.caption_url)
            self._db.insert(sample.metadata)
        return []

# Connect the stages and run
pipeline_spec = pipelines_v1.PipelineSpec(
    input_data=video_samples,  # List of 1000 video samples
    stages=[
        pipelines_v1.StageSpec(DownloadStage(), num_workers_per_node=15),
        CaptionStage(),  # Xenna autoscales this stage
        pipelines_v1.StageSpec(WriterStage(), num_workers_per_node=15),
    ]
)

pipelines_v1.run_pipeline(pipeline_spec)
```

**How it Works**:

1. **Stage Definition**: Each stage implements `setup()` (one-time initialization per worker) and `process_data()` (called for each batch). Stages declare their resource requirements (`gpus`, `cpus`) and batch size (`stage_batch_size`).

2. **Worker Allocation**: Xenna creates Ray Actors as workers for each stage. You can manually specify worker counts (`num_workers`, `num_workers_per_node`) or let Xenna autoscale based on measured throughput.

3. **Data Flow**: In streaming mode, data flows continuously: samples → Stage 1 workers → Stage 2 workers → Stage 3 workers. Xenna manages queues between stages and implements backpressure to prevent memory overflow.

4. **Autoscaling**: The streaming autoscaler measures each stage's throughput (samples/second) and adjusts worker counts to balance the pipeline. For example, if the caption stage is slower, Xenna allocates more workers to it while reducing download workers.

5. **Resource Isolation**: Each stage's workers run independently. CPU-heavy stages (download/upload) don't block GPU stages. Xenna optimally packs workers onto cluster nodes to maximize utilization.

See `examples/simple_vlm_inference.py` for a complete example.

## Pipeline Monitoring

Xenna provides real-time monitoring through detailed status reports logged periodically during execution. These reports give you comprehensive visibility into your pipeline's performance:

```text
Pipeline Stats:
Pipeline duration: 10.019294619560242 minutes
Number of initial input samples: 1000
Number of input samples remaining: 80
Streaming pipeline main loop rate: 96.13111905299391
  Auto Scaling Apply : 0.000001 seconds
  Pool Update        : 0.002647 seconds
  Auto Scaling Submit: 0.000001 seconds
  Monitor Update     : 0.000010 seconds
  Add Tasks          : 0.000051 seconds
  Sleep              : 0.007226 seconds
  Total              : 0.009937 seconds

Cluster Resources:
╒══════════════════════════╤═════════╤═════════════╕
│ Resource                 │   Total │   Available │
╞══════════════════════════╪═════════╪═════════════╡
│ CPUs                     │  384    │      382    │
├──────────────────────────┼─────────┼─────────────┤
│ GPUs                     │   16    │       16    │
├──────────────────────────┼─────────┼─────────────┤
│ Memory (GB)              │ 2093.92 │     2093.92 │
├──────────────────────────┼─────────┼─────────────┤
│ Object Store Memory (GB) │ 2089.07 │     2075.46 │
╘══════════════════════════╧═════════╧═════════════╛

Resource Usage by Stage:
╒═══════════════════════════╤═════════╤═══════════════╤═══════════════╤════════════════════╤══════════════════════════╕
│ Stage                     │   CPU % │   Memory (GB) │   Actor Count │   CPU % per worker │   Memory (GB) per worker │
╞═══════════════════════════╪═════════╪═══════════════╪═══════════════╪════════════════════╪══════════════════════════╡
│ Stage 00 - _DownloadStage │    48   │         12.8  │            30 │               1.6  │                     0.43 │
├───────────────────────────┼─────────┼───────────────┼───────────────┼────────────────────┼──────────────────────────┤
│ Stage 01 - _CaptionStage  │  2403.2 │        114.32 │            16 │             150.2  │                     7.14 │
├───────────────────────────┼─────────┼───────────────┼───────────────┼────────────────────┼──────────────────────────┤
│ Stage 02 - _WriterStage   │    51.2 │          4.24 │            30 │               1.71 │                     0.14 │
╘═══════════════════════════╧═════════╧═══════════════╧═════════════╧════════════════════╧══════════════════════════╛

Stage state:
╒═══════════════════════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╤═════════════╤═════════════════╤══════════════╤═══════════════╤════════════╤═════════════╤═════════════════╕
│ Stage                     │   Actors: │   Actors: │   Actors: │   Actors: │   Actors: │      Tasks: │          Tasks: │       Queue: │        Queue: │     Slots: │      Slots: │          Speed: │
│                           │    Target │   Pending │     Ready │   Running │      Idle │   Completed │   Returned None │   Input Size │   Output Size │   Num Used │   Num Empty │   Tasks/actor/s │
╞═══════════════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════════════╪═════════════════╪══════════════╪═══════════════╪════════════╪═════════════╪═════════════════╡
│ Stage 00 - _DownloadStage │         0 │         0 │        30 │         0 │        30 │          92 │               0 │            0 │           600 │          0 │          60 │        0.367088 │
├───────────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┼─────────────┼─────────────────┼──────────────┼───────────────┼────────────┼─────────────┼─────────────────┤
│ Stage 01 - _CaptionStage  │         0 │         0 │        16 │        16 │         0 │           0 │               0 │            0 │             0 │         32 │           0 │                 │
├───────────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┼─────────────┼─────────────────┼──────────────┼───────────────┼────────────┼─────────────┼─────────────────┤
│ Stage 02 - _WriterStage   │         0 │         0 │        30 │         0 │        30 │           0 │               0 │            0 │             0 │          0 │          60 │                 │
╘═══════════════════════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╧═════════════╧═════════════════╧══════════════╧═══════════════╧════════════╧═════════════╧═════════════════╛
```

**What the Reports Show**:

- **Pipeline Stats**: Overall progress, throughput, and orchestrator loop timing
- **Cluster Resources**: Total vs. available CPUs/GPUs/memory across the cluster
- **Resource Usage by Stage**: CPU/GPU/memory consumption per stage and per worker
- **Stage State**: Detailed per-stage metrics including:
  - **Actor counts**: How many workers are in each state (pending/ready/running/idle)
  - **Task progress**: Completed tasks and queue sizes
  - **Throughput**: Measured speed (tasks/actor/second) used for autoscaling decisions
  - **Slots**: Task scheduling slots (used vs. empty) showing worker utilization

These reports help you diagnose bottlenecks, optimize resource allocation, and ensure your pipeline is running efficiently. For example, if you see one stage with many idle workers and another stage fully saturated, Xenna's autoscaler will rebalance workers between stages.

## SPMD support

Xenna supports stages which run in "SPMD" (Single Program Multiple Data) mode (aka torchrun-like mode). SPMD
just means that multiple instances of the same class are started and they all work together to do inference.
In xenna, this is turned on by setting `is_spmd=True` in a stage's `required_resources`.

### What is SPMD mode?

SPMD mode enables distributed inference across multiple GPUs/nodes for models that require tensor parallelism
or other distributed execution strategies. This is essential for:

- **Large models** that don't fit on a single GPU (e.g., 200B+ parameter models)
- **Frameworks requiring multi-GPU coordination** like vLLM for distributed inference
- **High-throughput inference** that benefits from tensor parallelism

When `is_spmd=True`, Xenna creates a coordinated **worker group** instead of independent actors. All actors
in the group work together as a single inference unit, similar to how `torchrun` coordinates distributed training.

### How SPMD is implemented

Xenna's SPMD implementation follows torchrun conventions:

1. **Worker Groups**: Instead of creating independent actors, Xenna creates a group of coordinated actors
   - One actor is created per GPU
   - All actors in a group process the same task simultaneously
   - Only the primary actor (rank 0) returns results

2. **Environment Variables**: Each actor automatically receives distributed execution environment variables:
   - `RANK`: Global rank across all GPUs (0 to world_size-1)
   - `WORLD_SIZE`: Total number of GPUs in the worker group
   - `LOCAL_RANK`: GPU index on the current node
   - `MASTER_ADDR`: IP address of the primary node
   - `MASTER_PORT`: Port for NCCL communication

3. **CUDA Management**: Unlike regular stages, SPMD stages do **not** modify `CUDA_VISIBLE_DEVICES`.
   This follows torchrun behavior and allows distributed frameworks to manage GPU visibility internally.

4. **NCCL Rendezvous**: Xenna automatically:
   - Selects the primary node from worker group allocations
   - Finds an available port for NCCL communication
   - Registers ports cluster-wide to prevent conflicts
   - Configures all actors with the rendezvous parameters

5. **Scheduling**: SPMD worker groups are scheduled atomically - all actors must be placed before any can start.

### Data flow in SPMD mode

**All ranks receive identical input data**: When a task is scheduled to an SPMD worker group, Xenna broadcasts
the same input data to all actors in the group. Each rank processes the exact same inputs simultaneously,
coordinating through the distributed framework (e.g., vLLM, PyTorch distributed) to produce results.

**Multiple worker groups run in parallel**: Just like regular stages can have multiple independent workers,
SPMD stages can have multiple independent worker groups processing different tasks in parallel. Each worker
group is completely independent:

- **Worker Group 1** (GPUs 0-7): Processing task batch A
- **Worker Group 2** (GPUs 8-15): Processing task batch B  
- **Worker Group 3** (GPUs 16-23): Processing task batch C

This allows Xenna to scale SPMD inference across large clusters while maintaining high GPU utilization.
For example, if you have 64 GPUs and your model needs 8 GPUs (tensor_parallel_size=8), Xenna can create
8 independent worker groups, each processing different batches of data in parallel.

### Example: Large model inference with vLLM

Here's a complete example using vLLM for distributed inference with InternVL 3.5:

```python
import cosmos_xenna.pipelines.v1 as pipelines_v1
import vllm
from transformers import AutoTokenizer
from cosmos_internal_data_utils.utils import model_utils

class LargeModelStage(pipelines_v1.Stage):
    def __init__(self, model_name: str, tensor_parallel_size: int):
        self._model_name = model_name
        self._tp_size = tensor_parallel_size
    
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        # Enable SPMD mode for multi-GPU inference
        return pipelines_v1.Resources(
            gpus=self._tp_size,
            cpus=1.0,
            is_spmd=True  # This enables SPMD coordination
        )
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Setup runs on each actor in the worker group."""
        # Get distributed execution parameters
        dist_params = worker_metadata.distributed_execution_params
        print(f"Setting up rank {dist_params.rank}/{dist_params.world_size}")
        
        # Initialize vLLM with tensor parallelism
        self._llm = vllm.LLM(
            model=self._model_name,
            tensor_parallel_size=self._tp_size,
            # Use "external_launcher" - Xenna provides the distributed env vars
            distributed_executor_backend="external_launcher",
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True
        )
    
    def process_data(self, prompts: list[str]) -> list[str]:
        """Process data - only rank 0 returns results."""
        # All ranks participate in inference, but vLLM handles coordination
        sampling_params = vllm.SamplingParams(max_tokens=512)
        outputs = self._llm.generate(prompts, sampling_params)
        
        # vLLM only returns results from rank 0
        return [output.outputs[0].text for output in outputs]

# Use the stage in a pipeline
stages = [
    pipelines_v1.StageSpec(
        LargeModelStage(
            model_name="OpenGVLab/InternVL3_5-241B-A28B",
            tensor_parallel_size=16  # Use 16 GPUs
        )
    )
]

pipeline_spec = pipelines_v1.PipelineSpec(
    input_data=my_prompts,
    stages=stages
)
pipelines_v1.run_pipeline(pipeline_spec)
```

### Key differences from regular stages

| Aspect | Regular Stage | SPMD Stage |
|--------|--------------|------------|
| **GPU allocation** | Fractional or whole GPUs | Integer number of GPUs |
| **Actor creation** | One actor per stage instance | One actor per GPU (coordinated group) |
| **Task distribution** | Each actor gets different tasks | All ranks in a group get the same task |
| **Parallelism** | Multiple independent actors | Multiple independent worker groups |
| **CUDA_VISIBLE_DEVICES** | Set by Xenna for isolation | Not modified (framework manages) |
| **Environment vars** | Standard Ray environment | Includes RANK, WORLD_SIZE, LOCAL_RANK, etc. |
| **Result handling** | Each actor returns results | Only rank 0 returns results |
| **Scheduling** | Actors scheduled independently | Worker group scheduled atomically |

## P2P artifact downloading

Xenna includes a peer-to-peer (P2P) file distribution system for efficiently downloading large artifacts
(model weights, datasets, etc.) across multi-node Ray clusters. Instead of each node independently
downloading files from object storage, the system downloads each chunk only once across the entire cluster
and shares it between nodes via fast peer-to-peer HTTP transfers.

**Key Benefits:**

- **Reduced object store bandwidth** - Each chunk is downloaded from S3 only once across the entire cluster
- **Faster downloads** - P2P transfers between nodes are typically faster than repeated S3 downloads
- **Lower costs** - Minimizes expensive S3 egress costs on large clusters
- **Intelligent caching** - Files are cached locally with metadata validation to avoid re-downloads

For detailed documentation on the architecture, chunking strategies, and advanced features, see
[`file_distribution/README.md`](cosmos_xenna/file_distribution/README.md).

### Using with Pipelines

The most common way to use the artifact distribution system is by specifying download requirements in your
pipeline stages. Stages can declare what files they need via the `download_requests` property, and Xenna
will automatically download them before the pipeline starts.

```python
import pathlib
import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna import file_distribution

class MyModelStage(pipelines_v1.Stage):
    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        """Specify files this stage needs."""
        return [
            # Download a single file
            file_distribution.DownloadRequest(
                value=file_distribution.ObjectDownloadRequest(
                    profile_name="my-s3-profile",
                    uri="models/my-model/pytorch_model.bin",
                    destination=pathlib.Path("/tmp/models/pytorch_model.bin"),
                    cache=True  # Enable caching for future runs
                )
            ),
            
            # Download and extract an archive
            file_distribution.DownloadRequest(
                value=file_distribution.ObjectDownloadRequest(
                    profile_name="my-s3-profile",
                    uri="models/my-model.tar.gz",
                    destination=pathlib.Path("/tmp/models/my-model.tar.gz"),
                    unpack_options=file_distribution.UnpackOptions(
                        unpack_destination=pathlib.Path("/tmp/models/my-model/"),
                        unpack_method=file_distribution.UnpackMethod.AUTO
                    )
                )
            ),
            
            # Download all files under a prefix (directory)
            file_distribution.DownloadRequest(
                value=file_distribution.PrefixDownloadRequest(
                    profile_name="my-s3-profile",
                    uri="tokenizers/my-tokenizer/",
                    destination=pathlib.Path("/tmp/tokenizers/")
                )
            )
        ]
    
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=1.0, cpus=1.0)
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # Files are guaranteed to be available at this point
        import torch
        self.model = torch.load("/tmp/models/pytorch_model.bin")
    
    def process_data(self, inputs: list[str]) -> list[str]:
        # Your inference code here
        return inputs
```

When running the pipeline, provide a `DistributedDownloadConfig` to configure the download system:

```python
# Configure S3 authentication and download behavior
download_config = pipelines_v1.DistributedDownloadConfig(
    object_store_config=file_distribution.ObjectStoreConfigByProfile(
        profiles={
            "my-s3-profile": file_distribution.ObjectStoreConfig.make_for_s3(
                bucket="my-bucket",
                access_key_id="...",
                secret_access_key="...",
                region="us-west-2"
            )
        }
    ),
    chunk_size_bytes=100 * 1024 * 1024,  # 100MB chunks
    node_parallelism=20,  # Max concurrent downloads per node
    object_store_parallelism=1000,  # Max concurrent S3 connections
    verbose=True
)

# Run the pipeline with distributed downloads
pipeline_spec = pipelines_v1.PipelineSpec(
    input_data=my_data,
    stages=[MyModelStage()]
)

pipelines_v1.run_pipeline(
    pipeline_spec=pipeline_spec,
    distributed_download_config=download_config
)
```

Xenna will automatically collect all `download_requests` from all stages, coordinate the distributed download
before any stage starts, and ensure files are available on all nodes that need them.

### Standalone Usage

You can also use the distributed download system independently, outside of a pipeline context:

```python
from cosmos_xenna import file_distribution
import pathlib

# Configure object store access
object_store_config = file_distribution.ObjectStoreConfigByProfile(
    profiles={
        "my-profile": file_distribution.ObjectStoreConfig.make_for_s3(
            bucket="my-bucket",
            access_key_id="...",
            secret_access_key="...",
            region="us-west-2"
        )
    }
)

# Specify what to download
download_requests = [
    file_distribution.DownloadRequest(
        value=file_distribution.ObjectDownloadRequest(
            profile_name="my-profile",
            uri="path/to/large-file.tar",
            destination=pathlib.Path("/tmp/large-file.tar")
        )
    )
]

# Run the distributed download
file_distribution.download_distributed(
    object_store_config=object_store_config,
    download_requests=download_requests,
    chunk_size_bytes=100 * 1024 * 1024,
    node_parallelism=20,
    object_store_parallelism=1000,
    verbose=True
)
```

This is useful for pre-downloading files before running multiple pipelines, or for custom workflows that
need efficient multi-node downloads.

### Multi-Profile Authentication

You can configure different S3 credentials for different buckets or object stores:

```python
object_store_config = file_distribution.ObjectStoreConfigByProfile(
    profiles={
        "models": file_distribution.ObjectStoreConfig.make_for_s3(
            bucket="company-models-bucket",
            access_key_id="...",
            secret_access_key="..."
        ),
        "datasets": file_distribution.ObjectStoreConfig.make_for_s3(
            bucket="public-datasets-bucket",
            access_key_id="...",
            secret_access_key="..."
        )
    }
)

# Reference profiles in download requests
file_distribution.DownloadRequest(
    value=file_distribution.ObjectDownloadRequest(
        profile_name="models",  # Use the "models" profile
        uri="llama2/pytorch_model.bin",
        destination=pathlib.Path("/tmp/model.bin")
    )
)
```

See [`file_distribution/README.md`](cosmos_xenna/file_distribution/README.md) for more details on caching,
archive extraction, and the underlying architecture.

## Installing

```bash
pip install cosmos-xenna[gpu]
```

## Quick Start

For detailed examples, check out the `examples/` directory.

## Ray cluster requirements

Cosmos-xenna needs a few environment variables to be set before starting Ray clusters. These are set by
Xenna when we start clusters locally, but if using an already existing cluster, they will need to be set in
the processes initializing the cluster.

```bash
# Needed to get debug info from as many actors as possible. By default, Ray only allows 10k
# actors to be listed. However, on large clusters, we may have more than 10k actors.
RAY_MAX_LIMIT_FROM_API_SERVER=40000
RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000
```

## Gotchas

- `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="0"` needs to be set on the cluster. Without this, ray will overwite
  the cuda environment variables assigned by Xenna.
- Because Xenna has it's own allocation mechanisms, each Xenna pipelines needs its own Ray cluster. If
  multiple Xenna pipelines are run on a single cluster, or if other Ray based pipelines are running on
  a single cluster, resources will be over-subscribed.
- Batch mode can easily run out of memory. This is because intermediate data is stored in memory and any non-trivial
  pipeline will have a large number of items to run over.
- Streaming mode can also run out of memory if tasks are too large. By default, Xenna restricts each stage to
  queuing tasks 2x the number of workers on its input. However, if each task is large, this can still queue up
  enough things to run out of memory.

## Development

### Setup development environment

We use UV for development. To get started, [install UV](https://docs.astral.sh/uv/#installation), and
run `uv sync` in this directory.

This will create a virtual environment at `.venv` based on the current lock file and will include all
of the dependencies from core, dev, GPU, and examples.

### Running commands

Use UV to run all commands. For example, to run the example pipeline, use:

```bash
uv run examples/simple_vlm_inference.py 
```

This will auto-sync dependencies if needed and execute the command in the UV-managed virtualenv.

### VSCode integration

We provide recommended extensions and default settings for yotta via the .vscode/ folder. With these
settings, VSCode should automatically format your code and raise linting/typing issues. VSCode will
try to fix some minor linting issues on save.

### Linting

We use Ruff and PyRight for static analysis. Using the default VSCode settings and recommended extensions,
these should auto-run in VSCode. They can be run manually with:

```bash
uv run run_presubmit.py default
```

### Adding dependencies

To add packages to the core dependencies, use `uv add some-package-name`

To add packages to dev use `uv add --dev some-package-name`

To add packages to other groups use `uv add --group some-group some-package-name`

## License and Contact

This project will download and install additional third-party open source software projects. Review the
license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
