# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dispatcher tests for ``Autoscaler._algorithm`` selection.

Pins the contract that ``StreamingSpecificSpec.scheduler`` selects the
right algorithm class at ``Autoscaler.__init__`` and that the
``FRAGMENTATION_BASED`` default is preserved (no behaviour change for
existing pipelines).

Two explicit guarantees:

  1. Default ``StreamingSpecificSpec`` selects ``FragmentationBasedAutoscaler``.
  2. ``SchedulerKind.SATURATION_AWARE`` selects ``SaturationAwareScheduler``.

Tests target ``_make_scheduler_algorithm`` directly to keep the unit
boundary tight and avoid spinning up the full ``Autoscaler`` graph.
The factory consumes a ``PipelineSpec`` so it can collect
``StageSpec.saturation_aware`` overrides itself; these unit tests
construct a minimal ``PipelineSpec`` (no stages, no input data) via
the ``_pipeline_spec`` helper since the dispatch decision only depends
on ``config.mode_specific``.
"""

from cosmos_xenna.pipelines.private import autoscaling_algorithms, specs
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.streaming import _make_scheduler_algorithm


def _pipeline_spec(mode_specific: specs.StreamingSpecificSpec | None = None) -> specs.PipelineSpec:
    """Build a minimal ``PipelineSpec`` suitable for dispatcher unit tests.

    The dispatcher only reads ``config.mode_specific`` for the dispatch
    decision and ``stages`` to collect saturation-aware overrides. The
    helper provides empty ``input_data`` and ``stages`` so each test
    stays focused on the ``mode_specific`` field it varies. Defaults
    to a default ``StreamingSpecificSpec``.
    """
    if mode_specific is None:
        mode_specific = specs.StreamingSpecificSpec()
    return specs.PipelineSpec(
        input_data=[],
        stages=[],
        config=specs.PipelineConfig(mode_specific=mode_specific),
    )


class TestSchedulerDispatch:
    """Per-``SchedulerKind`` instantiation contract."""

    def test_default_selects_fragmentation_based(self) -> None:
        """Default spec preserves the FRAGMENTATION_BASED scheduler -- zero behaviour change."""
        algo = _make_scheduler_algorithm(_pipeline_spec())
        assert isinstance(algo, autoscaling_algorithms.FragmentationBasedAutoscaler)

    def test_explicit_fragmentation_based_routes_to_rust_solver(self) -> None:
        """Explicit FRAGMENTATION_BASED selection routes to the Rust-backed class."""
        spec = specs.StreamingSpecificSpec(scheduler=specs.SchedulerKind.FRAGMENTATION_BASED)
        algo = _make_scheduler_algorithm(_pipeline_spec(spec))
        assert isinstance(algo, autoscaling_algorithms.FragmentationBasedAutoscaler)

    def test_saturation_aware_selects_python_scheduler(self) -> None:
        """SATURATION_AWARE selection routes to the new pure-Python class."""
        spec = specs.StreamingSpecificSpec(scheduler=specs.SchedulerKind.SATURATION_AWARE)
        algo = _make_scheduler_algorithm(_pipeline_spec(spec))
        assert isinstance(algo, SaturationAwareScheduler)

    def test_saturation_aware_passes_config_through(self) -> None:
        """The scheduler receives the ``saturation_aware`` config from the spec."""
        sat_cfg = specs.SaturationAwareConfig(interval_s=15.0)
        spec = specs.StreamingSpecificSpec(
            scheduler=specs.SchedulerKind.SATURATION_AWARE,
            saturation_aware=sat_cfg,
        )
        algo = _make_scheduler_algorithm(_pipeline_spec(spec))
        assert isinstance(algo, SaturationAwareScheduler)
        # _config is the documented attribute name; not part of the public
        # API but pinned here so a rename surfaces during code review.
        assert algo._config is sat_cfg
