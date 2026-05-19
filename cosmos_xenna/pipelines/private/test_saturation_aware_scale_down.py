# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for saturation-aware scale-down worker selection helpers."""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.scale_down import select_workers_to_remove_oldest_first


class TestSelectWorkersToRemoveOldestFirst:
    """Pin the idle-first, oldest-first Phase D selection contract."""

    def test_idle_workers_precede_busier_older_workers(self) -> None:
        """Idle status outranks age when selecting scale-down victims."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["busy-old", "idle-young", "idle-old", "busy-young"],
            worker_ages={
                "busy-old": 100,
                "idle-young": 1,
                "idle-old": 50,
                "busy-young": 0,
            },
            worker_used_slots={
                "busy-old": 7,
                "idle-young": 0,
                "idle-old": 0,
                "busy-young": 1,
            },
            delete_count=4,
        )

        assert selected == ["idle-old", "idle-young", "busy-old", "busy-young"]

    def test_worker_id_breaks_ties_inside_same_idle_and_age_bucket(self) -> None:
        """Equal idle/age workers use worker id for deterministic ordering."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["w-c", "w-a", "w-b"],
            worker_ages={"w-a": 2, "w-b": 2, "w-c": 2},
            worker_used_slots={"w-a": 0, "w-b": 0, "w-c": 0},
            delete_count=3,
        )

        assert selected == ["w-a", "w-b", "w-c"]

    def test_missing_used_slot_signal_defaults_to_idle(self) -> None:
        """Workers missing from the used-slot map are treated as idle."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["missing-signal", "busy-old"],
            worker_ages={"missing-signal": 0, "busy-old": 99},
            worker_used_slots={"busy-old": 1},
            delete_count=1,
        )

        assert selected == ["missing-signal"]

    def test_omitted_used_slot_map_falls_back_to_age_only_selection(self) -> None:
        """Without per-worker signals, ordering collapses to age-DESC."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["young", "old", "middle"],
            worker_ages={"young": 0, "old": 9, "middle": 4},
            worker_used_slots=None,
            delete_count=3,
        )

        assert selected == ["old", "middle", "young"]

    def test_missing_age_defaults_to_new_worker(self) -> None:
        """Workers missing from the age map sort as newly observed."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["unknown-age", "known-age"],
            worker_ages={"known-age": 3},
            worker_used_slots={"unknown-age": 0, "known-age": 0},
            delete_count=2,
        )

        assert selected == ["known-age", "unknown-age"]

    @pytest.mark.parametrize("delete_count", [0, -1])
    def test_non_positive_delete_count_returns_empty_list(self, delete_count: int) -> None:
        """A non-positive delete request is a no-op."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["w0"],
            worker_ages={"w0": 1},
            worker_used_slots={"w0": 0},
            delete_count=delete_count,
        )

        assert selected == []

    def test_delete_count_is_clamped_to_worker_count(self) -> None:
        """A too-large delete request returns each worker at most once."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["w0", "w1"],
            worker_ages={"w0": 1, "w1": 0},
            worker_used_slots={"w0": 0, "w1": 0},
            delete_count=10,
        )

        assert selected == ["w0", "w1"]

    def test_large_worker_set_stays_deterministic_and_prefers_idle_bucket(self) -> None:
        """A 1,000-worker stage selects deterministically from the idle bucket first."""
        worker_ids = [f"w-{index:04d}" for index in range(1_000)]
        worker_ages = {worker_id: index % 17 for index, worker_id in enumerate(worker_ids)}
        worker_used_slots = {worker_id: index % 3 for index, worker_id in enumerate(worker_ids)}

        first = select_workers_to_remove_oldest_first(
            worker_ids=worker_ids,
            worker_ages=worker_ages,
            worker_used_slots=worker_used_slots,
            delete_count=50,
        )
        second = select_workers_to_remove_oldest_first(
            worker_ids=list(reversed(worker_ids)),
            worker_ages=worker_ages,
            worker_used_slots=worker_used_slots,
            delete_count=50,
        )

        assert first == second
        assert len(first) == 50
        assert all(worker_used_slots[worker_id] == 0 for worker_id in first)
