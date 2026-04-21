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

"""Tests for resource_monitor helpers.

Focus: ``_safe_gpu_attr`` must tolerate the two shapes of "missing field" that
``gpustat`` exposes on hardware where NVML returns ``None`` for some queries
(e.g. DGX Spark GB10, which shares unified memory with the CPU and reports
``memory.used`` / ``memory.total`` / ``power.limit`` as ``None``):

1. Property getters that eagerly call ``int(self.entry[key])`` raise
   ``TypeError`` when the entry is ``None``.
2. Other getters return ``None`` directly.
"""

from __future__ import annotations

from cosmos_xenna.ray_utils.resource_monitor import _safe_gpu_attr


class _FakeGpu:
    """Stand-in for a ``gpustat`` GPU object with configurable per-field behavior.

    Mirrors the real library: accessing a field whose NVML value is ``None``
    either raises ``TypeError`` (eager ``int()`` coercion) or returns ``None``.
    """

    def __init__(self, entry: dict[str, object]) -> None:
        self._entry = entry

    @property
    def utilization(self) -> int | None:
        v = self._entry.get("utilization.gpu")
        return int(v) if v is not None else None

    @property
    def memory_used(self) -> int:
        return int(self._entry["memory.used"])  # raises TypeError when None

    @property
    def memory_total(self) -> int:
        return int(self._entry["memory.total"])  # raises TypeError when None

    @property
    def power_draw(self) -> int | None:
        v = self._entry.get("power.draw")
        return int(v) if v is not None else None

    @property
    def power_limit(self) -> int:
        return int(self._entry["power.limit"])  # raises TypeError when None


def test_returns_value_when_present() -> None:
    gpu = _FakeGpu({"memory.used": 2048, "memory.total": 16384, "utilization.gpu": 42})
    assert _safe_gpu_attr(gpu, "memory_used", 0) == 2048
    assert _safe_gpu_attr(gpu, "memory_total", 0) == 16384
    assert _safe_gpu_attr(gpu, "utilization", 0.0) == 42


def test_returns_default_when_property_raises_type_error() -> None:
    # Reproduces the DGX Spark GB10 failure mode.
    gpu = _FakeGpu({"memory.used": None, "memory.total": None, "power.limit": None})
    assert _safe_gpu_attr(gpu, "memory_used", 0) == 0
    assert _safe_gpu_attr(gpu, "memory_total", 0) == 0
    assert _safe_gpu_attr(gpu, "power_limit", None) is None


def test_returns_default_when_property_returns_none() -> None:
    gpu = _FakeGpu({"utilization.gpu": None, "power.draw": None})
    assert _safe_gpu_attr(gpu, "utilization", 0.0) == 0.0
    assert _safe_gpu_attr(gpu, "power_draw", None) is None


def test_returns_default_when_attribute_missing() -> None:
    assert _safe_gpu_attr(object(), "does_not_exist", 7) == 7


def test_preserves_falsy_non_none_values() -> None:
    # Genuine zero readings must not be replaced by the default.
    gpu = _FakeGpu({"memory.used": 0, "utilization.gpu": 0})
    assert _safe_gpu_attr(gpu, "memory_used", 999) == 0
    assert _safe_gpu_attr(gpu, "utilization", 999.0) == 0
