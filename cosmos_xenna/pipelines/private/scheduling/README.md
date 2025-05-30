# Ray Pipeline Resource Scheduling and Allocation

This system implements an intelligent resource scheduler and allocator for multi-stage streaming pipelines running on Ray clusters. It addresses one of the fundamental challenges in distributed computing: how to efficiently allocate and manage heterogeneous computing resources (CPUs, GPUs, hardware codecs) across multiple pipeline stages while maintaining optimal performance.

## The Challenge

Modern ML pipelines often involve multiple stages with diverse resource requirements. For example, a video processing pipeline might include:

- Data loading (CPU-intensive)
- Video decoding (requires hardware decoders)
- ML model inference (GPU-intensive)
- Video encoding (requires hardware encoders)

Traditional scheduling approaches often lead to:

- Resource fragmentation
- Underutilization of specialized hardware
- Bottlenecks from improper worker allocation
- Complex manual configuration requirements

## Our Solution

We've developed a system that automatically handles these challenges through:

### Key Features

- **Adaptive Worker Allocation**: Automatically scales workers based on real-time performance measurements, ensuring resources are allocated where they're needed most
- **Smart Resource Packing**: Uses a fragmentation-aware approach to optimize placement of workers across nodes
- **Multi-Stage Pipeline Support**: Handles complex pipelines where each stage can have different resource requirements
- **Mixed Resource Types**: Sophisticated management of:
  - CPUs (supports fractional allocation for maximum utilization)
  - GPUs (whole, fractional, or entire GPU allocation)
  - NVIDIA Hardware Decoders (NVDEC)
  - NVIDIA Hardware Encoders (NVENC)
- **Automatic Slot Calculation**: Prevents work starvation through intelligent task slot management
- **Performance-Based Scaling**: Adapts worker counts based on measured throughput
- **Flexible Configuration**: Supports both automatic scaling and manually specified worker counts

## System Architecture

The scheduler consists of three main components working together:

### 1. Resource Shapes

The shape system provides a flexible way to define resource requirements for different worker types. It abstracts the complexity of resource specification into clear, type-safe definitions.

```python
# Example: Mixed resource requirements
resources = Resources(
    cpus=1.0,      # Use 1 CPU core
    gpus=0.5,      # Use half a GPU
    nvdecs=1,      # Need one hardware decoder
    nvencs=1       # Need one hardware encoder
)
```

Shape Types:

- `CPU_ONLY`: For CPU-bound tasks
- `CODEC`: For tasks needing hardware accelerated video coding
- `FRACTIONAL_GPU`: For tasks that can share GPUs
- `WHOLE_NUMBERED_GPU`: For tasks needing one or more complete GPUs
- `ENTIRE_GPU`: For tasks requiring exclusive GPU access including all codecs

See `yotta.ray_utils._specs.Stage.required_resources` for much more info.

### 2. Allocation System

The allocation system implements a sophisticated approach to resource management using Fragmentation Gradient Descent (FGD). This algorithm:

1. Tracks detailed resource state across the cluster
2. Calculates fragmentation metrics for potential allocations
3. Places workers to minimize resource fragmentation
4. Considers multi-dimensional resource constraints
5. Enables efficient packing of different worker types

### 3. Autoscaling Algorithm

Our autoscaler uses a multi-phase approach to optimize worker allocation:

#### Phase 1: Speed Estimation

- Maintains rolling window estimates of processing speeds for each stage
- Uses these measurements to identify bottlenecks and optimization opportunities

#### Phase 2: Resource Allocation

The allocation process follows a carefully ordered sequence:

1. **Manual Allocation**
   - Handles stages with explicitly specified worker counts
   - Ensures exact requirements are met
   - Fails fast if manual requirements cannot be satisfied

2. **Minimum Workers**
   - Ensures every stage has at least one worker
   - Maintains basic pipeline flow
   - Sets foundation for performance-based scaling

3. **Throughput Optimization**
   - Identifies the current pipeline bottleneck
   - Attempts to improve slowest stage's performance
   - Can reallocate resources from over-provisioned stages
   - Uses FGD to optimize worker placement decisions

4. **Over-allocation**
   - Provides headroom for throughput spikes
   - Controlled by configurable target parameter
   - Attempts to reuse recently removed workers
   - Maintains balanced performance across stages
