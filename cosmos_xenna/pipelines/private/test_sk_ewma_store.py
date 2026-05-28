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

"""Behavior of :class:`SkEwmaStore`."""

import math

from cosmos_xenna.pipelines.private.scheduling_py.state.sk_ewma_store import SkEwmaStore


class TestSkEwmaStoreUpdate:
    """``SkEwmaStore.update`` blends finite samples and preserves NaN seed on missed samples."""

    def test_first_finite_sample_replaces_nan_seed_without_blending(self) -> None:
        store = SkEwmaStore()
        store.seed_nan(["a"])
        store.update("a", 4.0, alpha=0.5)
        assert store.get("a") == 4.0

    def test_subsequent_finite_sample_blends_with_alpha(self) -> None:
        store = SkEwmaStore()
        store.set("a", 2.0)
        store.update("a", 4.0, alpha=0.25)
        assert store.get("a") == 2.0 * 0.75 + 4.0 * 0.25

    def test_non_finite_sample_preserves_previous_value(self) -> None:
        store = SkEwmaStore()
        store.set("a", 3.0)
        store.update("a", math.nan, alpha=0.5)
        assert store.get("a") == 3.0

    def test_zero_sample_preserves_previous_value(self) -> None:
        store = SkEwmaStore()
        store.set("a", 3.0)
        store.update("a", 0.0, alpha=0.5)
        assert store.get("a") == 3.0

    def test_missed_sample_for_unseen_stage_records_nan_seed(self) -> None:
        store = SkEwmaStore()
        store.update("a", math.nan, alpha=0.5)
        assert math.isnan(store.get("a"))


class TestSkEwmaStoreSeedAndView:
    """``seed_nan`` is idempotent; ``view`` returns a read-only mapping over the live store."""

    def test_seed_nan_does_not_overwrite_existing_value(self) -> None:
        store = SkEwmaStore()
        store.set("a", 2.0)
        store.seed_nan(["a", "b"])
        assert store.get("a") == 2.0
        assert math.isnan(store.get("b"))

    def test_view_reflects_live_mutation(self) -> None:
        store = SkEwmaStore()
        view = store.view()
        store.set("a", 1.5)
        assert view["a"] == 1.5

    def test_view_rejects_mutation(self) -> None:
        store = SkEwmaStore()
        view = store.view()
        try:
            view["a"] = 1.0  # type: ignore[index]
        except TypeError:
            return
        msg = "MappingProxyType must reject assignment"
        raise AssertionError(msg)


class TestSkEwmaStoreResetSeeded:
    """``reset_seeded`` clears every entry and seeds the requested names with NaN."""

    def test_reset_seeded_replaces_all_values_with_nan(self) -> None:
        store = SkEwmaStore()
        store.set("a", 1.0)
        store.set("b", 2.0)
        store.reset_seeded(["a", "b"])
        assert math.isnan(store.get("a"))
        assert math.isnan(store.get("b"))

    def test_reset_seeded_view_survives_reset(self) -> None:
        store = SkEwmaStore()
        store.set("a", 1.0)
        view = store.view()
        store.reset_seeded(["a"])
        assert math.isnan(view["a"])

    def test_reset_seeded_drops_entries_not_in_stage_names(self) -> None:
        store = SkEwmaStore()
        store.set("a", 1.0)
        store.set("b", 2.0)
        store.reset_seeded(["a"])
        assert math.isnan(store.get("a"))
        assert math.isnan(store.get("b"))
        assert "b" not in store.view()
