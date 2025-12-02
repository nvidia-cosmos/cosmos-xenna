# Changelog

## Latest

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
