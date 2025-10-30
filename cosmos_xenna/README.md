# Ray Pipelines

## Introduction

[Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) is a machine learning oriented orchestration framework. It has many components including data pipelines, training and hyper parameter tuning. Historically, we used Ray-Data to run our pipelines, but this had a lot of issues. Today, we only use Ray Core and have written a simple version of Ray-Data to run our pipelines.

This directory contains this simplified version of Ray-Data.

## Key Features

- **Distributed File Distribution**: Efficient P2P downloading of artifacts (model weights, datasets) to all cluster nodes before pipeline execution. See [file_distribution/README.md](file_distribution/README.md) for details.
- **Streaming and Batch Execution**: Support for both streaming (concurrent stages) and batch (sequential stages) execution modes.
- **Intelligent Resource Management**: Automatic worker allocation and load balancing across CPU and GPU resources.

## Examples

See `/pipelines/examples`.

## Batch vs Streaming

Pipelines can be run in either Batch mode or Streaming mode. Batch is like traditional ETL pipelines where all the data for a given stage is processed before moving on to the next stage. Ray also has a streaming mode where all of the stages are run concurrently. As an example, consider the following pipeline:

1. Run BLIP2 to caption each image
2. Run T5 to create an embedding for each caption

If we were running in batch mode, we would break this pipeline into two stages. The first stage would run BLIP2 on *all* of the images. When all of the images were captionioned, we would then run T5 on *all* the results from the first stage. The advantage of this mode is that we do not need to tell Ray how many workers to assign to each stage. Ray will simply assign all the available workers to stage 1 and then when that is complete it will do the same for stage 2. This mode has the disadvantage that we generally need to materialize (i.e. write to memory or disk) all the data between each stage. This can be a problem for our pipelines as we are often working with datasets which are larger than our cluster storage. We typically end up using PBSS to store these intermediate steps which adds complexity and possibly slows down the pipeline. It can also be hard to effectively utilize heterogenous resources when using this batched execution. For example, we should probably be downloading/uploading data at the same time as we are running inference. In this batched execution mode, it is up to the user to write streaming uploaders/downloaders within their application code to handle this. This also applies to CPU processing. If we also need to run CPU intensive processes, it's up to the user to set up streaming between the CPU and GPU in their application code.

Instead, if we were running this pipeline in streaming mode, Ray would set up a certain number of workers to run stage 1 and a certain number to run stage 2. As soon as *any* results were finished for stage 1, Ray would then feed the results to stage 2. The advantage of this mode is that we do not need to materialize all of the data between stages. This minimizes the potentially slow and expensive transfers to/from PBSS. It is also easier to utilize heterogeneuous workers. For example, imagine we changed our pipeline above to explicitly include PBSS uploads and downloads:

1. Download images from PBSS
2. Run BLIP2 to caption each image
3. Run T5 to create an embedding for each caption
4. Upload the captions/embeddings to PBSS

If we run this pipeline with Ray, it will run all of the stages concurrently. The user no longer needs to worry about uploading/downloading the data concurrently in their application code.

Streaming has the disadvantage of needing to tell Ray how many workers to set up for each stage. If we only have a single stage (or if the stages are using different resources. e.g. stage 1 uses CPU and stage 2 uses GPU), this is trivial. If there are multiple stages sharing the same resources (e.g. all running on the GPU), this can be difficult. Ideally, we create enough workers for each stage such that each stage processes items at the same rate. Additionally, streaming execution needs to ensure that too much data does not get materialized between the stages. I.E. we need to keep the amount of data in memory/on disk below the cluster limits. We have built our own auto-scheduler for sizing stages. However, the user can also manually specify the number of workers to run. Additionally, we carefully manage back-pressure in our pipelines such that the max (global) memory usage to save outputs should be `num_stages * num_workers_per_stage * average_size_of_outputs * 2`.

## Running Ray Pipelines in Yotta

Conceptually, our team's pipelines usually pretty simple. We usually have a large number of items to annotate and we need to run a number of models over these items to annotate them. We do not typically need complex ETL operations like group-bys or joins. This means we only need a relatively simple pipeline scheduler. It also means we can abstract our pipelines at a high level.

### The `run_pipeline` abstraction

To make developing these pipelines easier, we have written a small abstraction layer to run our pipelines. This abstraction layer assumes the pipelines are "linear". This means that the pipelines look like [Stage 1 -> Stage 2 -> Stage 3 -> Stage 4]. I.E. The input of every stage is the output of the previous stage. As of now, this is a good fit for all of our pipelines. The input of the first stage is specified by the user when the job is started. Because of this, it needs to be small enough to fit into memory for a single node. Usually this input will be pointer to objects in S3/PBSS.

This abstraction layer is contained in the `yotta.ray_utils` module. `ray_utils.run_pipeline` takes in a `PipelineSpec` which itself takes in an optional `ExecutionMode`.

`ExecutionMode` tells Ray whether to run the mode in batch or streaming mode. If left unspecified, it will try to use smart defaults. It will use streaming *unless* it is being run locally and only one GPU is available. `PipelineSpec` contains the input data which will be passed to the first stage of the pipeline as well as specifications for all of the stages. These stages are specified via a `Stage` instance or a `StageSpec` instance.

A `Stage` instance tells the Ray how to set up each worker in a given stage. Each stage must declare a `process_data` method and can optionally declare a `setup` method. `setup` will be called once per worker in this stage (or if a worker needs to be restarted). `process_data` will be called once per task. `Stage` can also specify a number of other things. This include the model the stage is running, the conda env the stage needs to run under, the number of gpus needed per worker and the number of cpus per worker. These are all optional fields. If a model is specified and these methods are not filled in, the model's information will be used to fill these in. Additionally, a processing speed can be specified for each stage. At present, we use this to determine the number of workers for each stage. This is covered more later.

We can alternatively specify a `StageSpec` instance for each stage. This instance contains a `Stage` instance as well as some less frequently used arguments. Using a `StageSpec` the user can manually specify a number of workers for the stage.

When running in streaming execution mode, ray needs to know how many workers to use for each stage. This number of workers can be specified manually using the `StageSpec.num_workers` and `StageSpec.num_workers_per_node` fields. Otherwise, Ray will attempt to auto-balance the number of workers per stage. It does this by splitting the stages into seperate cpu and gpu pools. For each pool, it will keep track of each stage's per-worker processing speed and will allocate a number of workers inversely proportional to the processing speed. Theoretically, this should mean that each stage's overall processing speed should be roughly equal. This should keep the pipeline from bottlenecking at any given stage.

Note that each stage is either considered a "cpu stage" or a "gpu stage". If the stage's num_gpus_per_worker is non-zero, we consider it a GPU stage. CPU stages and GPU stages are allocated seperately as they are seperate resource pools. Typically, cpu stages are download/upload stages and everything else is a gpu stage.

Historically, this sort of auto-scaling is typically not needed for cpu stages. Most of our pipelines are GPU-bound. The CPU stages just need to be fast enough to keep the gpus busy. This means that we often just create a fixed number of workers for these stages per node. See `StageSpec.num_workers_per_node` for more info.

### Local pipeline debugging

We also support a `BATCH_DEBUG` execution mode. This mode simply runs the pipeline as a series of for loops. This can be useful for debugging the pipeline as it allows you to set breakpoints. However, this mode *cannot* run anything which requires a specific conda environment (which includes all our current models) due to limitations in single-process python programs.

## Distributed File Distribution

Xenna includes a sophisticated distributed file distribution system that efficiently downloads artifacts (model weights, datasets, conda environments, etc.) to all cluster nodes before pipeline execution. This eliminates redundant downloads and reduces costs.

### Key Benefits

- **Cost Efficient**: Each file chunk downloaded from S3 only once across the entire cluster
- **Fast**: Parallel downloads with intelligent peer-to-peer sharing between nodes  
- **Smart Caching**: Automatic validation and reuse of previously downloaded files
- **Easy Integration**: Stages simply specify what they need via `download_requests` property

### Quick Example

```python
class MyModelStage(pipelines_v1.Stage):
    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        return [
            file_distribution.DownloadRequest(
                value=file_distribution.ObjectDownloadRequest(
                    uri="models/bert-base/pytorch_model.bin",
                    destination=pathlib.Path("/tmp/model.bin")
                )
            )
        ]
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # File is guaranteed to be available locally
        self.model = torch.load("/tmp/model.bin")
```

### Documentation

For complete documentation, examples, and performance tuning guidance, see:

- [file_distribution/README.md](file_distribution/README.md) - Comprehensive documentation
- [file_distribution/examples.py](file_distribution/examples.py) - Practical usage examples

## SPMD (Single Program, Multiple Data) Execution

Xenna supports SPMD execution for distributed inference scenarios where models need to run across multiple GPUs or nodes. This is particularly important for large models that require tensor parallelism or frameworks that only support multi-GPU inference through distributed coordination.

### When to Use SPMD

SPMD mode is required for:

1. **Large Models**: Models too large for a single GPU (e.g., 241B parameter models)
2. **Multi-Node Inference**: When you need to spread inference across multiple nodes
3. **Framework Requirements**: Some frameworks (like vLLM for multi-node) require SPMD coordination
4. **Tensor Parallelism**: Models that split computation across multiple GPUs

### How SPMD Works

SPMD emulates the **torchrun** distributed execution model:

- **One Actor Per GPU**: Creates exactly one Ray actor for each GPU in the distributed group
- **Coordinated Worker Group**: All actors work together as a single logical unit
- **Environment Variables**: Automatically sets up distributed training environment variables:
  - `MASTER_ADDR`: IP address of the rank=0 node
  - `MASTER_PORT`: Automatically selected available port
  - `RANK`: Global rank of each actor (0, 1, 2, ...)
  - `WORLD_SIZE`: Total number of actors in the group
  - `LOCAL_RANK`: Rank within each node (0, 1, 2, ...)
- **CUDA Visibility**: `CUDA_VISIBLE_DEVICES` is **not restricted** (follows torchrun behavior)

### Example: vLLM Multi-Node Inference

```python
from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna import file_distribution

class LargeModelInferenceStage(pipelines_v1.Stage):
    def __init__(self, model_name: str, tensor_parallel_size: int):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
    
    @property
    def required_resources(self) -> pipelines_v1.Resources:
        # Request SPMD execution across multiple GPUs
        return pipelines_v1.Resources(
            cpus=1.0, 
            gpus=self.tensor_parallel_size,  # e.g., 8 GPUs total
            is_spmd=True  # Enable SPMD coordination
        )
    
    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # Environment variables are automatically set by Xenna:
        # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK
        
        import vllm
        self.llm = vllm.LLM(
            model=f"/tmp/{self.model_name}/",
            tensor_parallel_size=self.tensor_parallel_size,
            distributed_executor_backend="external_launcher",  # Key: use external coordination
            trust_remote_code=True,
            gpu_memory_utilization=0.95
        )
    
    def process_data(self, prompts: list[str]) -> list[str]:
        # All actors run the same code, vLLM handles tensor parallelism internally
        outputs = self.llm.generate(prompts)
        return [output.outputs[0].text for output in outputs]
```

### Technical Details

**Master Address/Port Selection**:

- **Master Address**: Uses `ray.util.get_node_ip_address()` from the node where rank=0 resides.
- **Master Port**: Automatically selects an unused port using `portpicker.pick_unused_port()`
- **Port Registry**: Tracks used ports across the cluster to avoid conflicts

**Actor Coordination**:

- Each actor gets a unique `rank` (0 to world_size-1)
- `local_rank` represents position within each node (resets per node)
- All actors share the same NCCL rendezvous parameters for communication
- Worker group acts as a single logical unit for task scheduling

**Environment Setup**:
Unlike regular stages, SPMD stages preserve the full GPU environment, allowing distributed frameworks to manage GPU visibility and communication internally.

### Documentation

For detailed SPMD documentation and examples, see:

- [pipelines/SPMD.md](pipelines/SPMD.md) - Comprehensive SPMD guide with vLLM examples
