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

"""MFI-pressure helpers and pipeline integration.

The classifier consumes a smoothed pressure scalar that the per-stage
pipeline derives from three primitives:

*   ``compute_pressure`` -- the pure ``utilisation * normalized_backlog``
    multiplier with the cold-start cap.
*   ``compute_backlog_time`` -- the Little's Law gauge value (seconds).
*   ``_resolve_pressure_signal`` -- glues both together, refreshes the
    per-stage EWMA, and emits the Prometheus gauges.

These tests pin each primitive's contract independently and then verify
that ``run_per_stage_pipeline`` threads the resulting pressure into
``classify()`` so an EWMA refresh changes the classifier's verdict.
"""

import math

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import _resolve_auto_thresholds
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import run_per_stage_pipeline
from cosmos_xenna.pipelines.private.scheduling_py.pressure import (
    BACKLOG_CAP,
    compute_backlog_time,
    compute_pressure,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Pipeline-test fixture mirroring the documented pressure defaults."""
    return SaturationAwareStageConfig(
        saturation_threshold=0.15,
        activation_threshold=0.05,
        target_backlog_seconds=30.0,
        pressure_smoothing_level=0.20,
        pressure_critical_threshold=2.0,
        pressure_saturation_threshold=1.0,
        pressure_normal_threshold=0.3,
    )


def _fresh_state(cfg: SaturationAwareStageConfig, name: str = "TestStage") -> _StageRuntimeState:
    """Build a runtime state with thresholds pre-resolved (mirrors test_saturation_aware_pipeline.py)."""
    resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
    return _StageRuntimeState(stage_name=name, resolved_thresholds=resolved)


class TestComputePressureEmptyQueue:
    """Empty queue collapses pressure to zero regardless of the other inputs."""

    def test_empty_queue_pins_pressure_to_zero(self) -> None:
        """``input_queue_depth=0`` short-circuits the helper to ``0.0``."""
        result = compute_pressure(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=0,
            observed_throughput=10.0,
            target_backlog_seconds=30.0,
        )
        assert result == 0.0

    def test_empty_queue_with_zero_throughput_still_zero(self) -> None:
        """``queue=0`` short-circuits BEFORE the cold-start branch."""
        result = compute_pressure(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=0,
            observed_throughput=0.0,
            target_backlog_seconds=30.0,
        )
        assert result == 0.0


class TestComputePressureColdStart:
    """Zero / negative throughput with non-empty queue maps to the bounded cap branch."""

    def test_zero_throughput_with_queue_uses_backlog_cap(self) -> None:
        """``observed_throughput == 0.0`` and ``queue > 0`` yields ``utilisation * cap``."""
        result = compute_pressure(
            slots_empty_ratio_ewma=0.10,
            input_queue_depth=100,
            observed_throughput=0.0,
            target_backlog_seconds=30.0,
        )
        # utilisation = 1 - 0.10 = 0.90; normalized_backlog = cap (3.0)
        assert math.isclose(result, 0.90 * BACKLOG_CAP, rel_tol=1e-9)

    def test_negative_throughput_uses_backlog_cap(self) -> None:
        """Negative throughput is treated as the cold-start branch (``<= 0`` test)."""
        result = compute_pressure(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            observed_throughput=-1.0,
            target_backlog_seconds=30.0,
        )
        assert math.isclose(result, BACKLOG_CAP, rel_tol=1e-9)

    def test_cold_start_respects_explicit_backlog_cap_override(self) -> None:
        """The ``backlog_cap`` parameter is honoured by the cold-start branch."""
        result = compute_pressure(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=10,
            observed_throughput=0.0,
            target_backlog_seconds=30.0,
            backlog_cap=5.0,
        )
        assert math.isclose(result, 1.0 * 5.0, rel_tol=1e-9)


class TestComputePressureSteadyState:
    """The active branch returns ``utilisation * min(W_q / target, cap)``."""

    def test_balanced_workload_under_target_returns_proportional_pressure(self) -> None:
        """Queue drains in 10s vs 30s target -> ``normalized_backlog == 1/3``."""
        # W_q = 100 / 10 = 10s; normalized = 10 / 30 = 0.333
        # utilisation = 1 - 0.20 = 0.80
        # pressure = 0.80 * 0.333 = 0.267
        result = compute_pressure(
            slots_empty_ratio_ewma=0.20,
            input_queue_depth=100,
            observed_throughput=10.0,
            target_backlog_seconds=30.0,
        )
        assert math.isclose(result, 0.80 * (10.0 / 30.0), rel_tol=1e-9)

    def test_above_target_pressure_is_clamped_to_cap(self) -> None:
        """``W_q == 200s`` with ``target=30s`` clamps normalized_backlog to ``cap``."""
        # W_q = 200 / 1 = 200s; normalized = 200 / 30 = 6.67 -> clamped to 3.0
        # utilisation = 1 - 0.0 = 1.0; pressure = 1.0 * 3.0 = 3.0
        result = compute_pressure(
            slots_empty_ratio_ewma=0.0,
            input_queue_depth=200,
            observed_throughput=1.0,
            target_backlog_seconds=30.0,
        )
        assert math.isclose(result, BACKLOG_CAP, rel_tol=1e-9)

    def test_zero_utilisation_collapses_pressure(self) -> None:
        """Even with a huge backlog, idle slots (utilisation=0) keep pressure at 0."""
        result = compute_pressure(
            slots_empty_ratio_ewma=1.0,
            input_queue_depth=1000,
            observed_throughput=0.001,  # 1000 / 0.001 = 1_000_000s -> clamped
            target_backlog_seconds=30.0,
        )
        assert result == 0.0


class TestComputeBacklogTime:
    """The gauge helper returns Little's Law value bounded by ``target * cap``."""

    def test_empty_queue_returns_zero(self) -> None:
        """Empty queue -> ``0.0`` so the Grafana gauge does not show stale data."""
        result = compute_backlog_time(
            input_queue_depth=0,
            observed_throughput=10.0,
            target_backlog_seconds=30.0,
        )
        assert result == 0.0

    def test_zero_throughput_with_queue_returns_target_times_cap(self) -> None:
        """Cold-start gauge is bounded so dashboards never display ``+inf``."""
        result = compute_backlog_time(
            input_queue_depth=10,
            observed_throughput=0.0,
            target_backlog_seconds=30.0,
        )
        assert math.isclose(result, 30.0 * BACKLOG_CAP, rel_tol=1e-9)

    def test_steady_state_returns_littles_law_value(self) -> None:
        """The active branch returns ``queue / throughput`` directly (no clamp)."""
        result = compute_backlog_time(
            input_queue_depth=60,
            observed_throughput=10.0,
            target_backlog_seconds=30.0,
        )
        assert math.isclose(result, 6.0, rel_tol=1e-9)


class TestResolvePressureSignalIntegration:
    """``run_per_stage_pipeline`` smooths pressure into ``stage_state.pressure_ewma``."""

    def test_first_cycle_initialises_pressure_ewma_with_raw_value(self, cfg: SaturationAwareStageConfig) -> None:
        """``update_ewma(None, raw, alpha)`` returns ``raw`` on the first cycle."""
        state = _fresh_state(cfg)
        assert state.pressure_ewma is None

        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=8,
            num_empty_slots=2,
            input_queue_depth=100,
            current_workers=1,
            config=cfg,
            observed_throughput_sample=1.0,
            pipeline_name="test",
        )

        # First-cycle EWMA == raw pressure.
        # slots_empty_ratio_ewma = 2/(2+8) = 0.20 -> utilisation = 0.80
        # W_q = 100 / 1 = 100s; normalized = 100/30 = 3.33 -> clamp to 3.0
        # raw pressure = 0.80 * 3.0 = 2.40
        assert state.pressure_ewma is not None
        assert math.isclose(state.pressure_ewma, 0.80 * BACKLOG_CAP, rel_tol=1e-9)

    def test_subsequent_cycle_smooths_with_alpha(self, cfg: SaturationAwareStageConfig) -> None:
        """Second cycle blends ``alpha * raw + (1 - alpha) * prior``."""
        state = _fresh_state(cfg)
        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=8,
            num_empty_slots=2,
            input_queue_depth=100,
            current_workers=1,
            config=cfg,
            observed_throughput_sample=1.0,
            pipeline_name="test",
        )
        first_pressure = state.pressure_ewma
        assert first_pressure is not None

        # Second cycle: queue empties -> raw pressure goes to 0
        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=8,
            num_empty_slots=2,
            input_queue_depth=0,
            current_workers=1,
            config=cfg,
            observed_throughput_sample=1.0,
            pipeline_name="test",
        )
        # alpha = 0.20; blended = 0.20 * 0.0 + 0.80 * first_pressure
        expected = 0.20 * 0.0 + 0.80 * first_pressure
        assert state.pressure_ewma is not None
        assert math.isclose(state.pressure_ewma, expected, rel_tol=1e-9)

    def test_zero_total_slots_skips_pressure_update(self, cfg: SaturationAwareStageConfig) -> None:
        """Zero ready actors and no prior EWMA: ``_resolve_classifier_signal`` returns None;
        the pressure helper is intentionally not called so the next valid cycle does not
        smooth in stale data."""
        state = _fresh_state(cfg)
        # Both classifier and pressure EWMAs unset.
        assert state.slots_empty_ratio_ewma is None
        assert state.pressure_ewma is None

        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
            observed_throughput_sample=2.0,
            pipeline_name="test",
        )

        assert delta == 0
        # Classifier short-circuited -> pressure helper was not called.
        assert state.pressure_ewma is None


class TestPipelinePressureThreading:
    """A high pressure sample makes the slot-pin SATURATED branch fire scale-up."""

    def test_high_pressure_with_slot_pin_yields_saturated_classification(self, cfg: SaturationAwareStageConfig) -> None:
        """End-to-end: slot pin in SATURATED band + high pressure -> classifier returns SATURATED."""
        state = _fresh_state(cfg)
        # 9 used / 1 empty -> ratio 0.10 (in SATURATED band, 0.05 < 0.10 < 0.15)
        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=9,
            num_empty_slots=1,
            input_queue_depth=200,
            current_workers=1,
            config=cfg,
            observed_throughput_sample=1.0,
            pipeline_name="test",
        )
        assert state.classifier_state == StageState.SATURATED

    def test_low_pressure_demotes_slot_pin_to_normal(self, cfg: SaturationAwareStageConfig) -> None:
        """Slot pin in SATURATED band BUT queue drains faster than target -> NORMAL via demotion."""
        state = _fresh_state(cfg)
        # 9 used / 1 empty -> ratio 0.10 (SATURATED band by slot ratio alone)
        # W_q = 5 / 100 = 0.05s; normalized_backlog = 0.05 / 30 = 0.00167 (well below 1.0)
        # pressure = 0.90 * 0.00167 = 0.0015 (below pressure_saturation_threshold = 1.0)
        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=9,
            num_empty_slots=1,
            input_queue_depth=5,
            current_workers=1,
            config=cfg,
            observed_throughput_sample=100.0,
            pipeline_name="test",
        )
        assert state.classifier_state == StageState.NORMAL
