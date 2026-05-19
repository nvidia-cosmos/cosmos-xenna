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


class TestConsolidationTiebreak:
    """Pin the host_gpu_used_fraction primary sort key contract.

    Workers placed on GPUs with the lowest total used fraction are
    removed first so a fractional shrink can free whole GPUs for
    downstream whole-GPU stages.
    """

    def test_lowest_gpu_fraction_is_selected_first(self) -> None:
        """Across three workers on different GPUs, the lowest-fraction GPU is freed first."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["heavy", "light", "medium"],
            worker_ages={"heavy": 5, "light": 5, "medium": 5},
            worker_used_slots={"heavy": 0, "light": 0, "medium": 0},
            worker_host_gpu_used_fractions={"heavy": 0.9, "light": 0.2, "medium": 0.5},
            delete_count=2,
        )

        assert selected == ["light", "medium"]

    def test_consolidation_outranks_idle_status(self) -> None:
        """A busy worker on a low-fraction GPU is removed before an idle worker on a heavy GPU."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["idle-on-heavy", "busy-on-light"],
            worker_ages={"idle-on-heavy": 10, "busy-on-light": 10},
            worker_used_slots={"idle-on-heavy": 0, "busy-on-light": 7},
            worker_host_gpu_used_fractions={"idle-on-heavy": 0.95, "busy-on-light": 0.10},
            delete_count=1,
        )

        assert selected == ["busy-on-light"]

    def test_consolidation_outranks_age(self) -> None:
        """A young worker on a low-fraction GPU is removed before an old worker on a heavy GPU."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["old-on-heavy", "young-on-light"],
            worker_ages={"old-on-heavy": 100, "young-on-light": 1},
            worker_used_slots={"old-on-heavy": 0, "young-on-light": 0},
            worker_host_gpu_used_fractions={"old-on-heavy": 0.99, "young-on-light": 0.05},
            delete_count=1,
        )

        assert selected == ["young-on-light"]

    def test_equal_fraction_falls_back_to_idle_age_id(self) -> None:
        """Workers with the same GPU fraction tiebreak via the existing idle/age/id keys."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["busy-young", "idle-old", "idle-young", "busy-old"],
            worker_ages={"busy-young": 1, "idle-old": 50, "idle-young": 1, "busy-old": 100},
            worker_used_slots={"busy-young": 1, "idle-old": 0, "idle-young": 0, "busy-old": 5},
            worker_host_gpu_used_fractions={
                "busy-young": 0.5,
                "idle-old": 0.5,
                "idle-young": 0.5,
                "busy-old": 0.5,
            },
            delete_count=4,
        )

        assert selected == ["idle-old", "idle-young", "busy-old", "busy-young"]

    def test_missing_gpu_fraction_defaults_to_zero(self) -> None:
        """Workers absent from the fraction map are treated as fully consolidatable."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["unknown-gpu", "known-heavy"],
            worker_ages={"unknown-gpu": 1, "known-heavy": 1},
            worker_used_slots={"unknown-gpu": 0, "known-heavy": 0},
            worker_host_gpu_used_fractions={"known-heavy": 0.99},
            delete_count=1,
        )

        assert selected == ["unknown-gpu"]

    def test_omitted_gpu_fraction_map_falls_back_to_idle_age_id_ordering(self) -> None:
        """Without per-worker GPU fractions, the helper degrades to the prior 3-key sort."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["young-busy", "old-idle"],
            worker_ages={"young-busy": 1, "old-idle": 9},
            worker_used_slots={"young-busy": 1, "old-idle": 0},
            delete_count=1,
        )

        assert selected == ["old-idle"]

    def test_cpu_only_stage_with_zero_fraction_map_uses_idle_age_id(self) -> None:
        """A CPU-only stage passes a fraction map of all zeros and tiebreaks via idle/age/id."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["young", "old"],
            worker_ages={"young": 1, "old": 9},
            worker_used_slots={"young": 0, "old": 0},
            worker_host_gpu_used_fractions={"young": 0.0, "old": 0.0},
            delete_count=1,
        )

        assert selected == ["old"]

    def test_float_precision_does_not_corrupt_ordering(self) -> None:
        """Sub-epsilon GPU fraction differences still order correctly."""
        selected = select_workers_to_remove_oldest_first(
            worker_ids=["a", "b", "c"],
            worker_ages={"a": 1, "b": 1, "c": 1},
            worker_used_slots={"a": 0, "b": 0, "c": 0},
            worker_host_gpu_used_fractions={"a": 0.5, "b": 0.5 + 1e-9, "c": 0.5 + 2e-9},
            delete_count=2,
        )

        assert selected == ["a", "b"]

    def test_consolidation_stress_stays_deterministic(self) -> None:
        """A 1,000-worker stress run sorts deterministically across input orderings."""
        worker_ids = [f"w-{index:04d}" for index in range(1_000)]
        worker_ages = {worker_id: index % 13 for index, worker_id in enumerate(worker_ids)}
        worker_used_slots = {worker_id: index % 5 for index, worker_id in enumerate(worker_ids)}
        worker_host_gpu_used_fractions = {
            worker_id: round(((index * 37) % 100) / 100.0, 2) for index, worker_id in enumerate(worker_ids)
        }

        first = select_workers_to_remove_oldest_first(
            worker_ids=worker_ids,
            worker_ages=worker_ages,
            worker_used_slots=worker_used_slots,
            worker_host_gpu_used_fractions=worker_host_gpu_used_fractions,
            delete_count=100,
        )
        second = select_workers_to_remove_oldest_first(
            worker_ids=list(reversed(worker_ids)),
            worker_ages=worker_ages,
            worker_used_slots=worker_used_slots,
            worker_host_gpu_used_fractions=worker_host_gpu_used_fractions,
            delete_count=100,
        )

        assert first == second
        assert len(first) == 100
        max_selected_fraction = max(worker_host_gpu_used_fractions[worker_id] for worker_id in first)
        unselected = [worker_id for worker_id in worker_ids if worker_id not in first]
        min_unselected_fraction = min(worker_host_gpu_used_fractions[worker_id] for worker_id in unselected)
        # Every selected worker must have a fraction <= every unselected worker (the consolidation
        # bucket boundary), otherwise the primary sort key was violated.
        assert max_selected_fraction <= min_unselected_fraction
