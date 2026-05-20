# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the donor warmup grace in saturation-aware scheduling.

The donor warmup grace
(``donor_warmup_grace_s``, default 180 s) protects a freshly-ready
worker from being yanked out of its stage before the dispatcher
has had time to fill its queue. The grace gates two consumers:

  * Phase D shrink victim selection
    (:func:`select_workers_to_remove_oldest_first`) skips workers
    whose ready age is below ``donor_warmup_grace_s`` so a
    saturation-driven shrink cannot delete a worker that has not
    yet contributed real measurements.
  * Saturation-mode cross-stage donor selection
    (:func:`find_saturation_donor`) skips the same workers so a
    receiver does not absorb a donor that is still in its own
    warmup window.

Floor-mode donor selection
(:func:`select_youngest_eligible_donor`) deliberately bypasses
the grace because the floor is a hard structural requirement;
deadlocking on warmup-protected donors is worse than killing a
young donor.

This module pins:

  * Both helpers honour an ``excluded_worker_ids`` parameter that
    filters the candidate pool.
  * ``None`` and empty sets preserve the unfiltered legacy
    contract.
  * A stage whose entire worker pool is in the excluded set
    becomes ineligible (donor) or untouched (Phase D).
  * The five-layer anti-flap and donor-floor checks still apply
    independently of the excluded set.
  * The :class:`SaturationAwareScheduler` populates
    ``_donor_warmup_excluded_ids`` every cycle from the live
    ready-first-seen map and threads it through to both helpers.
  * Floor-mode donor selection has no ``excluded_worker_ids``
    parameter and therefore admits warmup workers unchanged.
  * The scheduler-level Phase D shrink log distinguishes the
    warmup-grace branch from floor / fraction caps so operators
    can locate the binding constraint in the log stream.
"""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor import (
    DonorCandidate,
    find_saturation_donor,
    select_youngest_eligible_donor,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.scale_down import select_workers_to_remove_oldest_first
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _stage_runtime_over_provisioned(
    *,
    name: str,
    streak: int = 30,
    growth_mode: GrowthMode = GrowthMode.TRACKING,
) -> _StageRuntimeState:
    """Build a runtime state representing a stable OVER_PROVISIONED stage."""
    state = _StageRuntimeState(stage_name=name)
    state.classifier_state = StageState.OVER_PROVISIONED
    state.classifier_streak = streak
    state.growth_mode = growth_mode
    return state


def _stage_runtime_saturated(name: str) -> _StageRuntimeState:
    """Build a runtime state representing a SATURATED receiver stage."""
    state = _StageRuntimeState(stage_name=name)
    state.classifier_state = StageState.SATURATED
    state.classifier_streak = 30
    return state


class TestPhaseDShrinkVictimFilter:
    """``select_workers_to_remove_oldest_first`` honours the excluded set."""

    def test_excluded_worker_dropped_from_victim_pool(self) -> None:
        """A worker in the excluded set never appears in the returned list."""
        victims = select_workers_to_remove_oldest_first(
            worker_ids=["a-w0", "a-w1", "a-w2"],
            worker_ages={"a-w0": 100, "a-w1": 50, "a-w2": 1},
            delete_count=2,
            excluded_worker_ids=frozenset({"a-w2"}),
        )

        assert "a-w2" not in victims
        assert set(victims) == {"a-w0", "a-w1"}

    def test_full_pool_excluded_returns_empty(self) -> None:
        """A stage whose every worker is excluded yields no victims this cycle."""
        victims = select_workers_to_remove_oldest_first(
            worker_ids=["a-w0", "a-w1"],
            worker_ages={"a-w0": 1, "a-w1": 1},
            delete_count=2,
            excluded_worker_ids=frozenset({"a-w0", "a-w1"}),
        )

        assert victims == []

    def test_none_excluded_set_preserves_legacy_behavior(self) -> None:
        """``excluded_worker_ids=None`` keeps the prior contract."""
        victims = select_workers_to_remove_oldest_first(
            worker_ids=["a-w0", "a-w1", "a-w2"],
            worker_ages={"a-w0": 100, "a-w1": 50, "a-w2": 1},
            delete_count=3,
            excluded_worker_ids=None,
        )

        assert set(victims) == {"a-w0", "a-w1", "a-w2"}

    def test_empty_excluded_set_preserves_legacy_behavior(self) -> None:
        """An empty frozenset is equivalent to ``None``."""
        victims = select_workers_to_remove_oldest_first(
            worker_ids=["a-w0", "a-w1"],
            worker_ages={"a-w0": 100, "a-w1": 50},
            delete_count=2,
            excluded_worker_ids=frozenset(),
        )

        assert set(victims) == {"a-w0", "a-w1"}

    def test_mixed_excluded_picks_only_unexcluded(self) -> None:
        """Mixed pool: only non-excluded workers populate the victim list."""
        victims = select_workers_to_remove_oldest_first(
            worker_ids=["a-w0", "a-w1", "a-w2", "a-w3"],
            worker_ages={"a-w0": 100, "a-w1": 90, "a-w2": 80, "a-w3": 70},
            delete_count=2,
            excluded_worker_ids=frozenset({"a-w0", "a-w1"}),
        )

        assert set(victims) == {"a-w2", "a-w3"}

    def test_excluded_does_not_break_consolidation_sort(self) -> None:
        """Consolidation key still drives ordering among non-excluded workers."""
        # w0 on busy GPU (0.9), w1 on idle GPU (0.1), w2 excluded.
        victims = select_workers_to_remove_oldest_first(
            worker_ids=["a-w0", "a-w1", "a-w2"],
            worker_ages={"a-w0": 50, "a-w1": 50, "a-w2": 1},
            delete_count=1,
            worker_host_gpu_used_fractions={"a-w0": 0.9, "a-w1": 0.1, "a-w2": 0.0},
            excluded_worker_ids=frozenset({"a-w2"}),
        )

        # w1 wins on consolidation (lower GPU fraction).
        assert victims == ["a-w1"]


class TestSaturationDonorWarmupFilter:
    """``find_saturation_donor`` honours the excluded set."""

    @pytest.fixture
    def base_kwargs(self) -> dict[str, object]:
        """Shared call kwargs that pin the receiver / config / state baseline."""
        cfg = SaturationAwareConfig(
            enable_cross_stage_donor=True,
            # Anti-flap and donation-interval validators require >= 1 cycle.
            # Use the minimums so a high ``cycle`` value (10_000) leaves us
            # outside any cooldown window for the purposes of the tests.
            cross_stage_donor_anti_flap_cycles=30,
            cross_stage_donor_min_donation_interval_cycles=30,
            cross_stage_donor_max_per_cycle=1,
            cross_stage_donor_require_over_provisioned=True,
            cross_stage_donor_exclude_hold_state=True,
            donor_must_be_strictly_upstream=True,
            stage_defaults=SaturationAwareStageConfig(
                min_workers=1,
                # Cross-validation requires anti-flap >= longest streak;
                # match the anti-flap of 30.
                over_provisioned_streak_min_cycles=30,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
            ),
        )
        return {
            "receiver_stage_index": 1,
            "receiver_stage_name": "B",
            "stage_names": ["A", "B"],
            "stage_floors": {0: 1, 1: 1},
            "worker_ages": {"a-w0": 50, "a-w1": 1, "a-w2": 1},
            "stage_states": {
                "A": _stage_runtime_over_provisioned(name="A", streak=30),
                "B": _stage_runtime_saturated("B"),
            },
            "config": cfg,
            "stage_configs": {
                "A": cfg.stage_defaults,
                "B": cfg.stage_defaults,
            },
            # Cycle far enough out that no cooldown window applies.
            "cycle": 10_000,
            "last_donation_cycle": {},
            "donations_received_this_cycle": {},
        }

    def test_excluded_donor_skipped(self, base_kwargs: dict[str, object]) -> None:
        """An excluded donor candidate is skipped; the next eligible worker is picked."""
        result = find_saturation_donor(
            worker_ids_by_stage=[["a-w0", "a-w1", "a-w2"], ["b-w0"]],
            excluded_worker_ids=frozenset({"a-w0"}),
            **base_kwargs,
        )

        assert isinstance(result, DonorCandidate)
        assert result.worker_id != "a-w0"
        assert result.worker_id in {"a-w1", "a-w2"}

    def test_full_donor_pool_excluded_returns_none(self, base_kwargs: dict[str, object]) -> None:
        """When every donor stage's workers are excluded, no donor is selected."""
        result = find_saturation_donor(
            worker_ids_by_stage=[["a-w0", "a-w1", "a-w2"], ["b-w0"]],
            excluded_worker_ids=frozenset({"a-w0", "a-w1", "a-w2"}),
            **base_kwargs,
        )

        assert result is None

    def test_none_excluded_set_preserves_legacy_behavior(self, base_kwargs: dict[str, object]) -> None:
        """``excluded_worker_ids=None`` matches the prior unfiltered contract."""
        result = find_saturation_donor(
            worker_ids_by_stage=[["a-w0", "a-w1"], ["b-w0"]],
            excluded_worker_ids=None,
            **base_kwargs,
        )

        # The youngest donor wins (a-w1 has age 1 vs a-w0 age 50).
        assert isinstance(result, DonorCandidate)
        assert result.worker_id == "a-w1"

    def test_excluded_does_not_bypass_floor(self, base_kwargs: dict[str, object]) -> None:
        """A stage at its floor is still ineligible regardless of the excluded set."""
        # Floor of A is 1; A has only one worker -> not eligible to donate.
        result = find_saturation_donor(
            worker_ids_by_stage=[["a-w0"], ["b-w0"]],
            excluded_worker_ids=frozenset(),
            **base_kwargs,
        )

        assert result is None

    def test_excluded_does_not_bypass_master_toggle(self, base_kwargs: dict[str, object]) -> None:
        """``enable_cross_stage_donor=False`` short-circuits before the excluded check."""
        cfg = SaturationAwareConfig(
            enable_cross_stage_donor=False,
            stage_defaults=SaturationAwareStageConfig(
                min_workers=1,
                over_provisioned_streak_min_cycles=30,
                worker_warmup_measurement_grace_s=0.0,
                donor_warmup_grace_s=0.0,
            ),
        )
        kwargs = dict(base_kwargs)
        kwargs["config"] = cfg
        kwargs["stage_configs"] = {"A": cfg.stage_defaults, "B": cfg.stage_defaults}

        result = find_saturation_donor(
            worker_ids_by_stage=[["a-w0", "a-w1"], ["b-w0"]],
            excluded_worker_ids=frozenset(),
            **kwargs,
        )

        assert result is None


class TestFloorDonorSelectionUnfiltered:
    """``select_youngest_eligible_donor`` does NOT honour any excluded set.

    Floor-mode donor selection is unconditional; deadlocking the
    floor on warmup-protected donors is a worse failure mode than
    killing a young donor because the floor is a hard structural
    invariant.
    """

    def test_floor_donor_picks_warmup_worker_when_only_choice(self) -> None:
        """Even a freshly-ready worker is fair game for floor enforcement."""
        result = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["a-w0", "a-w1"], ["b-w0"]],
            worker_ages={"a-w0": 50, "a-w1": 0},  # a-w1 is brand new
        )

        assert isinstance(result, DonorCandidate)
        # The youngest donor wins regardless of "warmup" status.
        assert result.worker_id == "a-w1"

    def test_floor_donor_signature_has_no_excluded_param(self) -> None:
        """Defensive: floor donor's call signature pins the contract.

        The absence of an ``excluded_worker_ids`` parameter is the
        contract: anyone adding one in the future would silently
        change the floor invariant from "always pick the youngest"
        to "pick the youngest non-warmup", which can deadlock the
        floor on a slow-warming stage.
        """
        import inspect

        sig = inspect.signature(select_youngest_eligible_donor)
        assert "excluded_worker_ids" not in sig.parameters


class TestSchedulerDonorWarmupExcludedCache:
    """``SaturationAwareScheduler`` populates ``_donor_warmup_excluded_ids`` per cycle."""

    def _scheduler(self, *, donor_grace_s: float = 60.0) -> SaturationAwareScheduler:
        """Build a one-stage scheduler with the requested donor grace."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=donor_grace_s,
                donor_warmup_grace_s=donor_grace_s,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(used_cpus=0, total_cpus=64, gpus=[], name="node-0"),
            },
        )
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster,
            [
                data_structures.ProblemStage(
                    name="hot",
                    stage_batch_size=1,
                    worker_shape=cpu_shape,
                    requested_num_workers=None,
                    over_provision_factor=None,
                ),
            ],
        )
        scheduler.setup(problem)
        return scheduler

    def _stage_state_with_workers(
        self,
        *,
        name: str,
        worker_ids: list[str],
        slots_per_worker: int = 8,
    ) -> data_structures.ProblemStageState:
        worker_groups = [
            data_structures.ProblemWorkerGroupState.make(
                wid,
                [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                num_used_slots=slots_per_worker,
            )
            for wid in worker_ids
        ]
        total_used = sum(slots_per_worker for _ in worker_ids)
        return data_structures.ProblemStageState(
            stage_name=name,
            workers=worker_groups,
            slots_per_worker=slots_per_worker,
            is_finished=False,
            num_used_slots=total_used,
            num_empty_slots=0,
            input_queue_depth=0,
        )

    def test_first_cycle_workers_all_in_excluded_set(self) -> None:
        """On cycle 1 every worker is fresh and the excluded set covers them all."""
        scheduler = self._scheduler(donor_grace_s=60.0)
        ps = data_structures.ProblemState(
            [self._stage_state_with_workers(name="hot", worker_ids=["hot-w0", "hot-w1", "hot-w2"])]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._donor_warmup_excluded_ids == frozenset({"hot-w0", "hot-w1", "hot-w2"})

    def test_excluded_set_clears_after_grace_elapses(self) -> None:
        """All workers age past ``donor_warmup_grace_s`` -> empty excluded set."""
        scheduler = self._scheduler(donor_grace_s=60.0)
        ps = data_structures.ProblemState([self._stage_state_with_workers(name="hot", worker_ids=["hot-w0", "hot-w1"])])

        scheduler.autoscale(time=0.0, problem_state=ps)
        scheduler.autoscale(time=60.0, problem_state=ps)

        assert scheduler._donor_warmup_excluded_ids == frozenset()

    def test_mixed_age_excluded_set_contains_only_young_workers(self) -> None:
        """Workers added on a later cycle live in the excluded set; survivors do not."""
        scheduler = self._scheduler(donor_grace_s=60.0)
        ps_early = data_structures.ProblemState([self._stage_state_with_workers(name="hot", worker_ids=["hot-w0"])])
        ps_late = data_structures.ProblemState(
            [self._stage_state_with_workers(name="hot", worker_ids=["hot-w0", "hot-w1", "hot-w2"])]
        )

        scheduler.autoscale(time=0.0, problem_state=ps_early)
        scheduler.autoscale(time=70.0, problem_state=ps_late)

        # hot-w0 first_seen=0, age=70 -> mature
        # hot-w1, hot-w2 first_seen=70, age=0 -> warmup
        assert scheduler._donor_warmup_excluded_ids == frozenset({"hot-w1", "hot-w2"})

    def test_grace_zero_means_no_exclusion(self) -> None:
        """``donor_warmup_grace_s=0`` always returns an empty excluded set."""
        scheduler = self._scheduler(donor_grace_s=0.0)
        ps = data_structures.ProblemState([self._stage_state_with_workers(name="hot", worker_ids=["hot-w0", "hot-w1"])])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._donor_warmup_excluded_ids == frozenset()

    def test_setup_resets_excluded_set(self) -> None:
        """A fresh ``setup()`` clears the excluded-set cache."""
        scheduler = self._scheduler(donor_grace_s=60.0)
        ps = data_structures.ProblemState([self._stage_state_with_workers(name="hot", worker_ids=["hot-w0"])])
        scheduler.autoscale(time=0.0, problem_state=ps)
        assert scheduler._donor_warmup_excluded_ids != frozenset()

        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(used_cpus=0, total_cpus=64, gpus=[], name="node-0"),
            },
        )
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        scheduler.setup(
            data_structures.Problem(
                cluster,
                [
                    data_structures.ProblemStage(
                        name="hot",
                        stage_batch_size=1,
                        worker_shape=cpu_shape,
                        requested_num_workers=None,
                        over_provision_factor=None,
                    ),
                ],
            )
        )

        assert scheduler._donor_warmup_excluded_ids == frozenset()


@pytest.mark.parametrize("grace_s", [10.0, 60.0, 180.0, 600.0])
class TestParametricDonorGraceBoundary:
    """Donor grace boundary holds across a range of configured values."""

    def test_just_below_grace_excludes_just_at_grace_admits(self, grace_s: float) -> None:
        """Worker age below ``grace_s`` is in the excluded set; ``>= grace_s`` is not."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=grace_s,
                donor_warmup_grace_s=grace_s,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0"),
            },
        )
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        scheduler.setup(
            data_structures.Problem(
                cluster,
                [
                    data_structures.ProblemStage(
                        name="hot",
                        stage_batch_size=1,
                        worker_shape=cpu_shape,
                        requested_num_workers=None,
                        over_provision_factor=None,
                    ),
                ],
            )
        )
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            "hot-w0",
                            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                            num_used_slots=8,
                        )
                    ],
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=8,
                    num_empty_slots=0,
                    input_queue_depth=0,
                ),
            ]
        )

        scheduler.autoscale(time=0.0, problem_state=ps)
        excluded_below = scheduler._build_donor_warmup_excluded_ids(
            [["hot-w0"]],
            now=grace_s - 1e-9,
        )
        excluded_at = scheduler._build_donor_warmup_excluded_ids(
            [["hot-w0"]],
            now=grace_s,
        )

        assert excluded_below == frozenset({"hot-w0"})
        assert excluded_at == frozenset()


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in test_saturation_aware_caps.py and
    other Phase D log-discipline tests.
    """
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


class TestPhaseDShrinkLogWarmupBranch:
    """``_log_phase_d_shrink_outcome`` distinguishes the warmup-grace branch.

    Pins the operator-facing Phase D shrink accounting:
    when ``select_workers_to_remove_oldest_first`` returns fewer
    victims than ``actual_remove`` because the warmup-grace filter
    excluded otherwise-eligible candidates, the deficit is reported
    as ``donor warmup grace left N removed`` rather than a
    floor / fraction-cap deficit. Logging ``actual_remove``
    regardless of how many workers were actually removed
    mis-attributes the deficit to floor / fraction clamps that did
    not bind.
    """

    def _scheduler_with_warmup(
        self,
        *,
        donor_grace_s: float,
        min_workers: int = 1,
        max_scale_down_fraction_per_cycle: float = 1.0,
    ) -> SaturationAwareScheduler:
        """Build a one-stage scheduler with the requested warmup graces."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(
                setup_phase_quiescence_enabled=False,
                worker_warmup_measurement_grace_s=donor_grace_s,
                donor_warmup_grace_s=donor_grace_s,
                min_workers=min_workers,
                max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(used_cpus=0, total_cpus=64, gpus=[], name="node-0"),
            },
        )
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        scheduler.setup(
            data_structures.Problem(
                cluster,
                [
                    data_structures.ProblemStage(
                        name="hot",
                        stage_batch_size=1,
                        worker_shape=cpu_shape,
                        requested_num_workers=None,
                        over_provision_factor=None,
                    ),
                ],
            )
        )
        return scheduler

    def _state_with_workers(self, *, worker_ids: list[str]) -> data_structures.ProblemState:
        """Build a single-stage ProblemState whose workers all have idle slots."""
        return data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            wid,
                            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                            num_used_slots=0,
                        )
                        for wid in worker_ids
                    ],
                    slots_per_worker=1,
                    is_finished=False,
                    num_used_slots=0,
                    num_empty_slots=len(worker_ids),
                    input_queue_depth=0,
                ),
            ]
        )

    def test_intent_shrink_with_all_workers_in_warmup_logs_grace_branch(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """All-warmup pool + intent shrink -> deficit attributed to warmup grace, not floor/fraction.

        Setup:
          * 4 workers, all freshly-ready (now < grace_s).
          * floor=1 so floor would allow up to 3 removals; fraction
            cap=1.0 so fraction does not bind either.
          * Patched intent ``-2`` -> requested_remove=2, actual_remove=2.
          * ``_donor_warmup_excluded_ids`` covers the entire pool ->
            ``select_workers_to_remove_oldest_first`` returns ``[]`` ->
            effective_remove=0.

        Expected log line:
          ``intent -2 workers; donor warmup grace left 0 removed
          (deficit=2, current=4, warmup_excluded=4).``

        Pre-fix bug: the log would say ``floor cap left 2 removed`` or
        ``intent -2 workers`` with no warmup attribution, even though
        zero workers were actually removed.
        """
        scheduler = self._scheduler_with_warmup(donor_grace_s=60.0, min_workers=1)
        ps = self._state_with_workers(worker_ids=["hot-w0", "hot-w1", "hot-w2", "hot-w3"])

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": -2}):
            scheduler.autoscale(time=0.0, problem_state=ps)

        warmup_logs = [r for r in loguru_caplog.records if "donor warmup grace left" in r.message]
        floor_logs = [r for r in loguru_caplog.records if "floor cap left" in r.message]
        fraction_logs = [r for r in loguru_caplog.records if "per-cycle fraction cap left" in r.message]

        assert len(warmup_logs) == 1, (
            f"expected exactly one warmup-branch log, got {len(warmup_logs)}; "
            f"all phase-d records: {[r.message for r in loguru_caplog.records if 'scale-down' in r.message]}"
        )
        assert not floor_logs, "floor-cap branch should not fire when warmup-grace truncated the pool"
        assert not fraction_logs, "fraction-cap branch should not fire when warmup-grace truncated the pool"

        msg = warmup_logs[0].message
        assert "intent -2 workers" in msg
        assert "donor warmup grace left 0 removed" in msg
        assert "deficit=2" in msg
        assert "current=4" in msg
        assert "warmup_excluded=4" in msg

    def test_partial_warmup_pool_logs_grace_branch_with_partial_removal(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Mixed pool: some workers mature, some warmup, intent shrink partially served.

        Setup:
          * 4 workers; cycle 1 seeds w0/w1 (now=0), cycle 2 adds w2/w3
            (now=70) so on cycle 3 (now=70 again, within w2/w3 grace),
            w0/w1 are mature, w2/w3 are still in warmup.
          * floor=1, no fraction cap, intent ``-3``.
          * actual_remove = min(3, 4-1, 4*1.0) = 3.
          * warmup-grace excludes {w2, w3} -> helper returns
            [w0, w1] -> effective_remove=2.

        Expected log line:
          ``intent -3 workers; donor warmup grace left 2 removed
          (deficit=1, current=4, warmup_excluded=2).``
        """
        scheduler = self._scheduler_with_warmup(donor_grace_s=60.0, min_workers=1)
        ps_early = self._state_with_workers(worker_ids=["hot-w0", "hot-w1"])
        ps_full = self._state_with_workers(worker_ids=["hot-w0", "hot-w1", "hot-w2", "hot-w3"])

        # Cycle 1: seed first-seen for w0, w1 only.
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": 0}):
            scheduler.autoscale(time=0.0, problem_state=ps_early)
        # Cycle 2: w2, w3 join. Their first_seen=70.
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": 0}):
            scheduler.autoscale(time=70.0, problem_state=ps_full)
        loguru_caplog.clear()
        # Cycle 3 at now=70 -> w0, w1 age=70 (mature), w2, w3 age=0 (warmup).
        # Intent -3 -> actual_remove=min(3, 4-1, 4)=3 -> helper excludes {w2, w3}
        # -> victims=[w0, w1] -> effective_remove=2 -> deficit=1.
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": -3}):
            scheduler.autoscale(time=70.0, problem_state=ps_full)

        warmup_logs = [r for r in loguru_caplog.records if "donor warmup grace left" in r.message]
        floor_logs = [r for r in loguru_caplog.records if "floor cap left" in r.message]

        assert len(warmup_logs) == 1, (
            f"expected exactly one warmup-branch log, got {len(warmup_logs)}; "
            f"all phase-d records: {[r.message for r in loguru_caplog.records if 'scale-down' in r.message]}"
        )
        assert not floor_logs, (
            "floor-cap should not fire: floor=1 allowed 3 removals, the deficit came from warmup grace"
        )

        msg = warmup_logs[0].message
        assert "intent -3 workers" in msg
        assert "donor warmup grace left 2 removed" in msg
        assert "deficit=1" in msg
        assert "current=4" in msg
        assert "warmup_excluded=2" in msg

    def test_warmup_branch_does_not_fire_when_pool_is_fully_mature(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A mature-only pool with a floor-bound deficit logs the floor branch, not warmup.

        Verifies the warmup branch is gated on
        ``effective_remove < actual_remove`` and stays silent when
        the helper returned the full requested count. The floor-cap
        branch fires when ``actual_remove < requested_remove``
        (floor binding) regardless of warmup state.
        """
        scheduler = self._scheduler_with_warmup(donor_grace_s=60.0, min_workers=2)
        ps = self._state_with_workers(worker_ids=["hot-w0", "hot-w1", "hot-w2", "hot-w3"])

        # Cycle 1 seeds first-seen at 0.
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": 0}):
            scheduler.autoscale(time=0.0, problem_state=ps)
        loguru_caplog.clear()
        # Cycle 2 at now=120 -> all workers age=120 (mature, > 60s grace).
        # Intent -10 -> requested=10, actual=min(10, 4-2, 4*1.0)=2 -> deficit=8 (floor-bound).
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": -10}):
            scheduler.autoscale(time=120.0, problem_state=ps)

        warmup_logs = [r for r in loguru_caplog.records if "donor warmup grace left" in r.message]
        floor_logs = [r for r in loguru_caplog.records if "floor cap left" in r.message]

        assert not warmup_logs, "warmup branch must not fire when effective_remove == actual_remove"
        assert len(floor_logs) == 1, (
            f"expected exactly one floor-cap log, got {len(floor_logs)}; "
            f"all phase-d records: {[r.message for r in loguru_caplog.records if 'scale-down' in r.message]}"
        )
        msg = floor_logs[0].message
        assert "intent -10 workers" in msg
        assert "floor cap left 2 removed" in msg
        assert "floor=2" in msg

    def test_composite_floor_and_warmup_both_bind_emits_both_branches(
        self,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Floor cap AND warmup grace bind in the same cycle -> both branches log.

        Setup:
          * 6 workers, w0/w1 seeded at now=0 (mature in cycle 2),
            w2..w5 seeded at now=70 (warmup in cycle 2 because the
            grace is 60 s and only 0 s has elapsed since they
            joined).
          * floor=2, no fraction cap (default 1.0), intent ``-10``.
          * Stage 1 binds: actual_remove = min(10, 6 - 2, 6 * 1.0)
            = 4 (floor cap); deficit_1 = 6.
          * Stage 2 binds: helper excludes {w2..w5} -> victims =
            [w0, w1] -> effective_remove = 2; deficit_2 =
            actual_remove - effective_remove = 2.
          * Both clauses MUST fire so the operator sees the full
            attribution; suppressing Stage 2 here mis-attributes the
            warmup-grace deficit to the floor cap.
        """
        scheduler = self._scheduler_with_warmup(donor_grace_s=60.0, min_workers=2)
        ps_early = self._state_with_workers(worker_ids=["hot-w0", "hot-w1"])
        ps_full = self._state_with_workers(worker_ids=["hot-w0", "hot-w1", "hot-w2", "hot-w3", "hot-w4", "hot-w5"])

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": 0}):
            scheduler.autoscale(time=0.0, problem_state=ps_early)
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": 0}):
            scheduler.autoscale(time=70.0, problem_state=ps_full)
        loguru_caplog.clear()
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"hot": -10}):
            scheduler.autoscale(time=70.0, problem_state=ps_full)

        warmup_logs = [r for r in loguru_caplog.records if "donor warmup grace left" in r.message]
        floor_logs = [r for r in loguru_caplog.records if "floor cap left" in r.message]
        fraction_logs = [r for r in loguru_caplog.records if "per-cycle fraction cap left" in r.message]

        assert len(floor_logs) == 1, (
            f"expected exactly one floor-cap log, got {len(floor_logs)}; "
            f"all phase-d records: {[r.message for r in loguru_caplog.records if 'scale-down' in r.message]}"
        )
        assert len(warmup_logs) == 1, (
            f"expected exactly one warmup-branch log, got {len(warmup_logs)}; "
            f"all phase-d records: {[r.message for r in loguru_caplog.records if 'scale-down' in r.message]}"
        )
        assert not fraction_logs, "fraction cap is 1.0 (default); branch must not fire"

        floor_msg = floor_logs[0].message
        assert "intent -10 workers" in floor_msg
        assert "floor cap left 2 removed" in floor_msg
        assert "deficit=6" in floor_msg
        assert "floor=2" in floor_msg

        warmup_msg = warmup_logs[0].message
        assert "intent -10 workers" in warmup_msg
        assert "donor warmup grace left 2 removed" in warmup_msg
        assert "deficit=2" in warmup_msg
        assert "warmup_excluded=4" in warmup_msg
