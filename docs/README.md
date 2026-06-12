# Cosmos-Xenna Documentation

Long-form documentation for the Xenna pipeline runtime. Source-file docstrings
carry the field-by-field contract of each type; documents here explain the
ideas and behavior behind them.

## Index

- [Saturation-Aware Scheduler](scheduler/saturation-aware/README.md): the
  `SATURATION_AWARE` streaming autoscaler. The folder holds an entry-point
  README plus topic notes:
  - [01: Capacity model](scheduler/saturation-aware/01-capacity-model.md)
  - [02: Bottleneck selection](scheduler/saturation-aware/02-bottleneck-selection.md)
  - [03: Growth and sizing](scheduler/saturation-aware/03-growth-and-sizing.md)
  - [04: Cold-start ramp](scheduler/saturation-aware/04-cold-start-ramp.md)
  - [05: Scale-down floor](scheduler/saturation-aware/05-scale-down-floor.md)
  - [Operator tuning guide](scheduler/saturation-aware/tuning.md)

## Layout

Topics are organized by area, one sub-folder per area, with a further
sub-folder per implementation when more than one exists:

    docs/
    └── <topic>/                e.g. scheduler/
        └── <implementation>/   e.g. saturation-aware/
