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

"""Contract tests for ``donor.planning_context.DonorPlanningContext``.

The planning context bundles cluster-wide per-cycle inputs that
donor policies and the coordinator read while evaluating receivers.
These tests cover the value object construction and update
contracts; the factory (``from_cycle``) is covered in
``test_donor_coordinator``.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.donor.planning_context import DonorPlanningContext
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


def _make_context(**overrides: object) -> DonorPlanningContext:
    base: dict[str, object] = {
        "stage_names": ("A", "B"),
        "stage_configs": {},
        "stage_states": {},
        "stage_floors": {0: 1, 1: 1},
        "worker_ids_by_stage": (("A-w0", "A-w1"), ("B-w0",)),
        "worker_ages": {"A-w0": 5, "A-w1": 3, "B-w0": 1},
        "worker_node_map": {"A-w0": "node-0"},
        "d_k_now": {"A": 1.0, "B": 2.0},
        "effective_capacities": {"A": 4, "B": 4},
        "s_k_ewma": {"A": 0.25, "B": 0.5},
        "slots_per_worker_by_stage": {"A": 1, "B": 1},
        "donor_warmup_exclusions": frozenset({"A-w0"}),
        "cycle_counter": 7,
        "last_donation_cycle": {"A": 3},
        "config": SaturationAwareConfig(),
    }
    base.update(overrides)
    return DonorPlanningContext(**base)  # type: ignore[arg-type]


class TestConstructionContract:
    """Frozen value object carries every cycle-scoped donor input."""

    def test_construction_populates_every_field(self) -> None:
        context = _make_context()
        assert context.stage_names == ("A", "B")
        assert context.d_k_now["B"] == 2.0
        assert context.donor_warmup_exclusions == frozenset({"A-w0"})
        assert context.cycle_counter == 7

    def test_last_donation_cycle_is_a_live_reference(self) -> None:
        ledger: dict[str, int] = {"A": 3}
        context = _make_context(last_donation_cycle=ledger)
        ledger["B"] = 9
        assert context.last_donation_cycle["B"] == 9

    def test_instance_is_frozen(self) -> None:
        context = _make_context()
        with pytest.raises(AttributeError):
            context.cycle_counter = 8  # type: ignore[misc]
