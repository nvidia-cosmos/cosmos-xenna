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

"""Behavior of :class:`FloorStuckCounters`."""

from cosmos_xenna.pipelines.private.scheduling_py.state.floor_stuck_counters import FloorStuckCounters


class TestIncrementStuck:
    """``increment_stuck`` advances by one and returns the post-increment value."""

    def test_increment_starts_at_one_for_unseen_stage(self) -> None:
        store = FloorStuckCounters()
        assert store.increment_stuck("a") == 1

    def test_increment_advances_by_one(self) -> None:
        store = FloorStuckCounters()
        store.increment_stuck("a")
        store.increment_stuck("a")
        assert store.increment_stuck("a") == 3
        assert store.get("a") == 3


class TestResetFor:
    """``reset_for`` drops the entry; absent stages are a no-op."""

    def test_reset_drops_existing_entry(self) -> None:
        store = FloorStuckCounters()
        store.increment_stuck("a")
        store.reset_for("a")
        assert store.get("a") == 0
        assert "a" not in store.view()

    def test_reset_is_noop_for_unseen_stage(self) -> None:
        store = FloorStuckCounters()
        store.reset_for("a")
        assert store.get("a") == 0


class TestReset:
    """``reset`` clears every entry; views captured beforehand observe the cleared state."""

    def test_reset_clears_all_entries(self) -> None:
        store = FloorStuckCounters()
        store.increment_stuck("a")
        store.increment_stuck("b")
        store.reset()
        assert store.get("a") == 0
        assert store.get("b") == 0

    def test_reset_view_survives_reset(self) -> None:
        store = FloorStuckCounters()
        store.increment_stuck("a")
        view = store.view()
        store.reset()
        assert "a" not in view
