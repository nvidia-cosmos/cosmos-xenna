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

"""Boundary and resolve tests for SaturationAwareConfig."""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig


def test_defaults_construct_without_error() -> None:
    """Every field has a working default, so a no-arg config is valid."""
    SaturationAwareConfig()


def test_capacity_headroom_above_one_is_rejected() -> None:
    """capacity_headroom is a fraction in the closed unit interval [0, 1]."""
    with pytest.raises(ValueError):
        SaturationAwareConfig(capacity_headroom=1.1)


def test_capacity_headroom_below_zero_is_rejected() -> None:
    """A negative headroom fraction is meaningless and rejected."""
    with pytest.raises(ValueError):
        SaturationAwareConfig(capacity_headroom=-0.1)


def test_speed_alpha_up_zero_is_rejected() -> None:
    """A speed_alpha_up of 0.0 would freeze the EWMA and is rejected."""
    with pytest.raises(ValueError):
        SaturationAwareConfig(speed_alpha_up=0.0)


def test_speed_alpha_up_above_one_is_rejected() -> None:
    """speed_alpha_up above 1.0 is out of range."""
    with pytest.raises(ValueError):
        SaturationAwareConfig(speed_alpha_up=1.5)


def test_speed_alpha_down_zero_is_rejected() -> None:
    """A speed_alpha_down of 0.0 would freeze the EWMA and is rejected."""
    with pytest.raises(ValueError):
        SaturationAwareConfig(speed_alpha_down=0.0)


def test_averaging_samples_below_trust_threshold_is_rejected() -> None:
    """Averaging depth must be at least the trust threshold."""
    with pytest.raises(ValueError):
        SaturationAwareConfig(speed_estimation_averaging_samples=3, speed_estimation_min_data_points=5)


def test_resolve_returns_defaults_when_none() -> None:
    """resolve(None) yields a default-constructed config."""
    assert SaturationAwareConfig.resolve(None) == SaturationAwareConfig()


def test_resolve_passes_through_explicit_config() -> None:
    """resolve(config) returns the same instance unchanged when one is provided."""
    config = SaturationAwareConfig(interval_s=7.0)
    assert SaturationAwareConfig.resolve(config) is config
