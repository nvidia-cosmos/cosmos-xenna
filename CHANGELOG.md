# Changelog

## Latest

## [0.4.0]

### Released

- 2026-04-28

### Added

- Added **continuous mode** — a new execution model where stages opt in via the `ContinuousInterface` mixin and implement `run_continuous(input_queue, output_queue, stop_event)`. The runtime feeds deserialized tasks through an `asyncio.Queue` and collects results asynchronously, enabling stages that pipeline overlapping work (e.g. long-running inference engines) to sustain higher GPU utilization than the classic batch-per-call model.
- Added graceful actor shutdown: `_attempt_graceful_shutdown()` cooperatively drains in-flight work before `ray.kill()`, bounded by a configurable grace period (default 30 s). Continuous-mode stages benefit most — batch-mode stages return immediately.
- Added `validate_positive_int` and `validate_optional_positive_int` attrs validators in `cosmos_xenna.utils.attrs_utils`; applied to `slots_per_actor` on `StageSpec`, `PipelineConfig`, and `stage.Params` to fail fast on invalid values.
- Added abandoned-task warning in `ActorPool.stop()` — logs per-origin-node counts for tasks still queued at shutdown.

### Fixed

- Enforced `strict=True` on `zip()` calls in `streaming.py` and `actor_pool.py` to catch length mismatches between pools and stage-done flags.

### Changed

- **BREAKING**: Minimum Python version raised from **3.9** to **3.12**. The `requires-python` field and Pyright `pythonVersion` now both target `>=3.12`. Consumers on older Python versions must upgrade before adopting this release.
- `_ReadyActor.kill()` now accepts a `graceful` flag (default `True`); `_try_delete_ready_actor` passes `graceful=False` since re-queued tasks make drained results useless.

## [0.3.0]

### Released

- 2026-04-23

### Added

- Added a GitHub Actions CI workflow for Rust and Python tests and lints; P2P bind failures now surface as a typed `RuntimeError` instead of a panic.

### Fixed

- Prevented GPU stage starvation at drain-tail by clamping autoscaler scale-downs to keep enough workers for in-flight and queued tasks per active stage.
- Caught the autoscaler TOCTOU race in `WorkerAllocator.add_worker()` as a typed `AllocationError` instead of crashing the pipeline.
- Tolerated `None` gpustat fields on DGX Spark GB10 so `NodeResourceMonitor`'s metrics loop keeps running on unified-memory GPUs.
- Made resource-shortage errors actionable (per-stage / worker-count remediation first, BATCH hint scoped to STREAMING, mode name in prefix, CPU / GPU units on requires/available).
- Used `math.floor` for CPU-count truncation and clamped the result to `>= 0` to guard against misconfigured `cpu_allocation_percentage`.
- Moved the Ray cluster startup log to after initialization completes.
- Refactored the Rust autoscaler FGD (Fragmented GPU Distribution) algorithm to use a pure scoring overlay instead of in-place mutation, improving correctness and testability.

### Changed

- Registered `L1` and `CPU` pytest markers and silenced the `TestS3Object` collection warning under external CI runners.

## [0.2.3]

### Released

- 2026-04-14

### Fixed

- Fixed the Xenna autoscaler which will return a raised Python exception on allocation failure instead of a `panic!()`.

## [0.2.2]

### Released

- 2026-04-13

### Fixed

- Implemented zero-copy Bytes streaming, removed some allocations and several buffer copies
- Lower verbosity on Ray orphan reap messages
- Fixed lint and cargo-audit warnings, remove unused crates

## [0.2.1]

### Released

- 2026-03-11

### Fixed

- Fixed leaking child processes from stage actors by snapshotting the PID tree before `ray.kill()` and reaping survivors via a pinned follow-up task (opt-in via `XENNA_KILL_ACTOR_SURVIVORS=1`).
- Fixed O(n²) process tree construction in `ProcessTree.make` and hardened it against missing/invalid psutil fields.

## [0.2.0]

### Released

- 2026-03-03

### Added

- Added support for OpenTelemetry distributed tracing via optional `XENNA_RAY_TRACING_HOOK` during Ray initialization.
- Added configurable S3/object-store retry settings in `ObjectStoreConfig.make_for_s3` (`max_retries`, `retry_timeout`, `init_backoff`, `max_backoff`).
- Added autoscaling smoke tests for fragmentation and large-model allocation scenarios.

### Fixed

- Updated Xenna to use `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` for compatibility with upstream Ray behavior.
- Fixed GPU allocation source-of-truth by capping detected GPUs to Ray-reported GPU count per node.

## [0.1.8]

### Released

- 2025-12-01

### Added

- Added a SERVING mode that extends STREAMING mode for online-serving use case.

### Fixed

- Updated Xenna to use Ray-reported CPU resources.

## [0.1.7]

### Released

- 2025-10-30

### Added

- Implemented an SPMD mode, which allows users to run multi-gpu and multi-node inference similarly to torchrun.
- Implemented P2P artifact downloads. This will allow users to efficiently download artifacts before the job starts.
- Improved task status polling for better performance at large scale.

### Fixed

- Xenna should be much less thread hungry than it was before.

## [0.1.6]

### Released

- 2025-09-25

### Fixed

- Fixed a bug in autoscaler in case of dynamic split.

## [0.1.5]

### Released

- 2025-09-15

### Fixed

- Fixed a bug when autoscaler tries to allocate workers for finished stages.

## [0.1.4]

### Released

- 2025-09-05

### Added

- Refactored the autoscaling code to reduce clones for better performance.

## [0.1.3]

### Released

- 2025-08-27

### Added

- Implemented autoscaling algorithm in Rust for better performance and scalability.
- Added metrics for the main loop of streaming executor.

## [0.1.2]

### Released

- 2025-08-19

### Added

- Add workflow to publish packages to PyPI.

### Fixed

- Fixed bug on queue-size stats when back-pressure kicking in.
- Fixed a possible hang when having a fan-in stage with large stage_batch_size.

## [0.1.1]

### Released

- 2025-08-14

### Added

- Add `over_provision_factor` to `StageSpec` to influence stage worker allocation by autoscaler.
- Allow `StageSpec.num_workers_per_node` to be `float` for greater flexibility.
- Add support to respect `CUDA_VISIBLE_DEVICES` if environment variable `XENNA_RESPECT_CUDA_VISIBLE_DEVICES` is set.

## [0.1.0]

### Released

- 2025-06-11

### Added

- Initial version
