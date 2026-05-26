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

"""Pure-helper tests for the saturation-aware scheduler state module.

Covers ``compute_slots_empty_ratio`` and ``update_ewma``. Both are
pure functions called by the scheduler each cycle; correctness here
underpins every classifier verdict downstream.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    GrowthMode,
    StageRuntimeState,
    StageState,
    compute_slots_empty_ratio,
    update_ewma,
)


class TestStageStateEnum:
    """Pin the enum values that operators see in metrics and logs."""

    def test_string_values_match_metric_label_convention(self) -> None:
        """Values are upper-snake-case so they can be used as Prometheus label values."""
        assert StageState.NORMAL.value == "NORMAL"
        assert StageState.SATURATED.value == "SATURATED"
        assert StageState.SATURATED_CRITICAL.value == "SATURATED_CRITICAL"
        assert StageState.OVER_PROVISIONED.value == "OVER_PROVISIONED"


class TestGrowthModeEnum:
    """Pin the growth-mode enum values that operators see in metrics and logs."""

    def test_string_values_match_metric_label_convention(self) -> None:
        """Values are upper-snake-case so they can be used as Prometheus label values."""
        assert GrowthMode.ACQUIRING.value == "ACQUIRING"
        assert GrowthMode.TRACKING.value == "TRACKING"
        assert GrowthMode.HOLD.value == "HOLD"


class TestStageRuntimeStateDefaults:
    """Initial state shape -- the scheduler relies on these defaults at cycle 1."""

    def test_default_classifier_state_is_normal(self) -> None:
        """First cycle's hysteresis logic needs a defined baseline."""
        state = StageRuntimeState(stage_name="A")
        assert state.classifier.state is StageState.NORMAL
        assert state.classifier.streak == 0

    def test_default_growth_mode_is_acquiring(self) -> None:
        """New stages start in ACQUIRING (slow-start regime)."""
        state = StageRuntimeState(stage_name="A")
        assert state.growth.mode is GrowthMode.ACQUIRING
        assert state.growth.streak == 0

    def test_default_ewma_is_unset(self) -> None:
        """The cold-start sentinel lets ``update_ewma`` skip the warmup tax."""
        state = StageRuntimeState(stage_name="A")
        assert state.classifier.slots_empty_ratio_ewma is None
        assert state.classifier.last_valid_slots_empty_ratio_ewma is None


class TestComputeSlotsEmptyRatio:
    """Pin the per-cycle saturation signal that drives the classifier."""

    def test_all_busy_returns_zero(self) -> None:
        """Zero free slots -> ratio 0.0 (the SATURATED_CRITICAL signal)."""
        assert compute_slots_empty_ratio(num_used_slots=8, num_empty_slots=0) == 0.0

    def test_all_free_returns_one(self) -> None:
        """All slots free -> ratio 1.0 (the OVER_PROVISIONED signal)."""
        assert compute_slots_empty_ratio(num_used_slots=0, num_empty_slots=8) == 1.0

    def test_half_busy_returns_one_half(self) -> None:
        """Exact ratio computation, no rounding shenanigans."""
        assert compute_slots_empty_ratio(num_used_slots=4, num_empty_slots=4) == 0.5

    def test_no_actors_returns_zero(self) -> None:
        """No actors -> ratio 0.0; the worker-floor step independently grows out of this."""
        assert compute_slots_empty_ratio(num_used_slots=0, num_empty_slots=0) == 0.0

    def test_negative_used_slots_is_rejected(self) -> None:
        """Defensive validation -- negative slot counts are a programming bug."""
        with pytest.raises(ValueError, match="num_used_slots must be >= 0"):
            compute_slots_empty_ratio(num_used_slots=-1, num_empty_slots=0)

    def test_negative_empty_slots_is_rejected(self) -> None:
        """Defensive validation -- negative slot counts are a programming bug."""
        with pytest.raises(ValueError, match="num_empty_slots must be >= 0"):
            compute_slots_empty_ratio(num_used_slots=0, num_empty_slots=-1)


class TestUpdateEwma:
    """Pin the smoothing applied to the per-cycle signal."""

    def test_cold_start_returns_sample_directly(self) -> None:
        """First cycle sees the live signal, not a zero-anchored half-step."""
        assert update_ewma(prev_ewma=None, sample=0.42, alpha=0.20) == 0.42

    def test_alpha_one_returns_sample_no_smoothing(self) -> None:
        """alpha=1.0 disables smoothing entirely."""
        assert update_ewma(prev_ewma=0.5, sample=0.10, alpha=1.0) == 0.10

    def test_low_alpha_blends_heavily_toward_previous(self) -> None:
        """alpha=0.2 -> new_ewma = 0.2 * sample + 0.8 * prev."""
        result = update_ewma(prev_ewma=0.5, sample=0.0, alpha=0.2)
        assert result == pytest.approx(0.4)

    def test_alpha_zero_is_rejected(self) -> None:
        """alpha=0 would freeze the EWMA forever and never see new samples."""
        with pytest.raises(ValueError, match=r"alpha must be in \(0\.0, 1\.0\]"):
            update_ewma(prev_ewma=0.5, sample=0.10, alpha=0.0)

    def test_alpha_above_one_is_rejected(self) -> None:
        """alpha > 1.0 would amplify samples instead of smoothing."""
        with pytest.raises(ValueError, match=r"alpha must be in \(0\.0, 1\.0\]"):
            update_ewma(prev_ewma=0.5, sample=0.10, alpha=1.5)
