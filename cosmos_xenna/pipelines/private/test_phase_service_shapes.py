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

"""Each ``*Services`` value object carries at most seven fields and exposes no ``ledgers``."""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.services import BottleneckServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.floor.services import FloorServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.grow.services import GrowServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.services import IntentServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.manual.services import ManualServices
from cosmos_xenna.pipelines.private.scheduling_py.phases.shrink.services import ShrinkServices


def _field_names(service_cls: type) -> set[str]:
    return {field.name for field in attrs.fields(service_cls)}


class TestNarrowServiceShapes:
    """Every ``*Services`` class drops the legacy ``ledgers`` field."""

    def test_bottleneck_services_has_no_ledgers_field(self) -> None:
        assert "ledgers" not in _field_names(BottleneckServices)

    def test_floor_services_has_no_ledgers_field(self) -> None:
        assert "ledgers" not in _field_names(FloorServices)

    def test_grow_services_has_no_ledgers_field(self) -> None:
        assert "ledgers" not in _field_names(GrowServices)

    def test_intent_services_has_no_ledgers_field(self) -> None:
        assert "ledgers" not in _field_names(IntentServices)

    def test_manual_services_has_no_ledgers_field(self) -> None:
        assert "ledgers" not in _field_names(ManualServices)

    def test_shrink_services_has_no_ledgers_field(self) -> None:
        assert "ledgers" not in _field_names(ShrinkServices)


class TestFatProtocolFieldBudget:
    """Each ``*Services`` class stays at or below the seven-field budget."""

    MAX_FIELDS = 7

    def test_bottleneck_services_field_count_under_budget(self) -> None:
        assert len(attrs.fields(BottleneckServices)) <= self.MAX_FIELDS

    def test_floor_services_field_count_under_budget(self) -> None:
        assert len(attrs.fields(FloorServices)) <= self.MAX_FIELDS

    def test_grow_services_field_count_under_budget(self) -> None:
        assert len(attrs.fields(GrowServices)) <= self.MAX_FIELDS

    def test_intent_services_field_count_under_budget(self) -> None:
        assert len(attrs.fields(IntentServices)) <= self.MAX_FIELDS

    def test_manual_services_field_count_under_budget(self) -> None:
        assert len(attrs.fields(ManualServices)) <= self.MAX_FIELDS

    def test_shrink_services_field_count_under_budget(self) -> None:
        assert len(attrs.fields(ShrinkServices)) <= self.MAX_FIELDS
