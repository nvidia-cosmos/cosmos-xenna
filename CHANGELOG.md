# Changelog


## Latest

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