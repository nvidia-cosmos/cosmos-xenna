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

"""Immutability contract for the cycle-scoped value objects in ``outputs.py``.

Each phase publishes its result as a snapshot that downstream phases
read but must never mutate. ``@attrs.frozen`` freezes the attribute
bindings but not the contents of a plain ``dict`` field, so the mapping
fields are wrapped in a read-only ``MappingProxyType`` view at
construction. These tests pin that contract by asserting item assignment
raises ``TypeError``.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_identity import BottleneckIdentity
from cosmos_xenna.pipelines.private.scheduling_py.state.outputs import BottleneckSnapshot, IntentPlan


def _snapshot() -> BottleneckSnapshot:
    """Build a minimal ``BottleneckSnapshot`` with single-entry mappings."""
    return BottleneckSnapshot(
        identity=BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=float("nan"),
            median_d_k=float("nan"),
            heterogeneity_ratio=float("nan"),
        ),
        d_k_now={"a": 1.0},
        effective_capacities={"a": 2},
        channels_per_worker_group={"a": 1},
        balance_score_start=0.0,
    )


@pytest.mark.parametrize("field", ["d_k_now", "effective_capacities", "channels_per_worker_group"])
def test_bottleneck_snapshot_mappings_are_read_only(field: str) -> None:
    """Each ``BottleneckSnapshot`` mapping rejects in-place mutation."""
    mapping = getattr(_snapshot(), field)
    with pytest.raises(TypeError):
        mapping["a"] = 99


def test_intent_plan_deltas_is_read_only() -> None:
    """``IntentPlan.deltas`` rejects in-place mutation."""
    plan = IntentPlan(deltas={"a": 1})
    with pytest.raises(TypeError):
        plan.deltas["a"] = 99  # type: ignore[index]
