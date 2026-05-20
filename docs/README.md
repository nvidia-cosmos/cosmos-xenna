# Cosmos-Xenna Documentation

This folder hosts long-form documentation for the Xenna pipeline
runtime. Source-file docstrings carry the field-by-field contract
of every public type; documents in this folder explain the
**design decisions** behind those contracts.

## Index

- [Saturation-Aware Scheduler](scheduler/saturation-aware/README.md)
  — feature-by-feature decision rationale for the streaming-mode
  saturation-aware autoscaler.

## Layout

Topics are organised by area, with one sub-folder per
independently-documentable surface area inside Xenna and a further
sub-folder per specific implementation when more than one exists:

    docs/
    └── <topic>/                e.g. scheduler/
        └── <implementation>/   e.g. saturation-aware/

Today only `scheduler/saturation-aware/` is populated; sibling
folders may land later when other Xenna topics gain documentation.
