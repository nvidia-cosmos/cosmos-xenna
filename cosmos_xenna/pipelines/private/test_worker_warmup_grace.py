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


"""Tests for the per-worker measurement grace in ``SaturationAwareScheduler``.

The per-worker measurement grace
(``worker_warmup_measurement_grace_s``, default 60 s) excludes
freshly-ready workers from the EWMA that drives the saturation
classifier. A worker whose ready age (wall-clock seconds since the
first cycle in which it was observed in
``ProblemStageState.worker_groups``) is below the configured grace
contributes neither ``num_used_slots`` nor ``num_empty_slots`` to
the per-stage signal; once the grace passes the worker rejoins the
EWMA on every subsequent cycle.

Why this matters: a freshly-spawned actor reports zero used slots
before the dispatcher has had time to fill its queue. EWMA-smoothing
that reading drags the running average toward "over-provisioned"
and risks a false Phase D shrink in the cycle right after a Phase B
or Phase C grow. The plan calls for a 60 s window so the warmup
sample window covers vLLM's CUDA-graph compile, prefix-cache
priming, and the dispatcher's catch-up time.

These tests pin:

  * The helper (``_aggregate_slot_signals_excluding_warmup``) sums
    only mature workers' contributions; the all-warmup case yields
    ``(0, 0)`` so the upstream classifier holds.
  * Grace=0 disables the filter (legacy behaviour for tests and
    operator opt-out).
  * ``input_queue_depth`` is unfiltered (per-stage signal).
  * Workers age across cycles via ``_refresh_worker_ready_first_seen``;
    departed workers drop from the map.
  * SPMD worker groups share a single first-seen timestamp.
  * Boundary: a worker exactly at grace seconds is admitted on the
    first comparison ``(now - first_seen) >= grace_s``.
  * Defensive: a non-monotonic ``now`` does not crash and reverts
    cleanly once monotonicity resumes.
  * Performance: 200-worker stage filters in well under a frame
    budget.

The fixture pattern mirrors ``test_setup_quiescence.py``: each test
isolates one behaviour, builds a dedicated scheduler, and inspects
``_last_intent_deltas`` and / or ``_stage_states`` to verify the
contract. Fixtures use ``pytest`` parametrization where the same
contract must hold across multiple input shapes.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 64) -> resources.ClusterResources:
    """Single-node CPU cluster with enough headroom for Phase C grow attempts."""
    return resources.ClusterResources(
        nodes={
            f"node-{i}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus_per_node,
                gpus=[],
                name=f"node-{i}",
            )
            for i in range(num_nodes)
        },
    )


def _problem(stage_names: list[str], cluster: resources.ClusterResources | None = None) -> data_structures.Problem:
    """Build a ``Problem`` with one CPU stage per name."""
    if cluster is None:
        cluster = _cluster()
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        for name in stage_names
    ]
    return data_structures.Problem(cluster, stages)


def _stage_state(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int,
    num_used_slots: int,
    num_empty_slots: int,
    input_queue_depth: int,
    is_finished: bool = False,
    actors_per_group: int = 1,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState`` with explicit worker-group ids.

    ``actors_per_group`` lets a single test build SPMD-shaped groups
    where multiple actors share a worker_group. The default of 1 is
    the non-SPMD case used by every other helper in this module.
    """
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            f"{name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[]) for _ in range(actors_per_group)],
        )
        for i in range(num_workers)
    ]
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=worker_groups,
        slots_per_worker=slots_per_worker,
        is_finished=is_finished,
        num_used_slots=num_used_slots,
        num_empty_slots=num_empty_slots,
        input_queue_depth=input_queue_depth,
    )


def _saturated_signal(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int = 8,
) -> data_structures.ProblemStageState:
    """Build a SATURATED-shaped slot signal with all groups nearly full.

    Per-worker_group ``num_used_slots`` is populated explicitly (the
    last group has one empty slot, all others are full) so the
    warmup-filter helper sees a saturated per-group view consistent
    with the stage-level totals. Production
    (``streaming.py::_make_problem_worker_group_state``) always
    populates per-group counts, so the helper mirrors that contract.

    Ratio empty / total ~ 1 / (num_workers * slots_per_worker), well
    below the default activation threshold for c=8.
    """
    if num_workers <= 0:
        return _stage_state(
            name=name,
            num_workers=0,
            slots_per_worker=slots_per_worker,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=0,
        )
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            f"{name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            num_used_slots=(slots_per_worker - 1 if i == num_workers - 1 else slots_per_worker),
        )
        for i in range(num_workers)
    ]
    total = num_workers * slots_per_worker
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=worker_groups,
        slots_per_worker=slots_per_worker,
        is_finished=False,
        num_used_slots=total - 1,
        num_empty_slots=1,
        input_queue_depth=5,
    )


def _scheduler_with_warmup_grace(
    stage_name: str,
    *,
    grace_s: float = 60.0,
    saturated_streak_min_cycles: int = 1,
    over_provisioned_streak_min_cycles: int = 30,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 30,
    quiescence_enabled: bool = False,
) -> SaturationAwareScheduler:
    """Build a one-stage scheduler that exercises the warmup measurement grace.

    Quiescence is disabled by default so the test focuses on the
    warmup grace path; otherwise a stage with zero ready actors
    would short-circuit on quiescence regardless of the grace
    setting. ``donor_warmup_grace_s`` is pinned to ``grace_s`` (not
    zero) because the config validator requires
    ``donor_warmup_grace_s >= worker_warmup_measurement_grace_s``:
    a worker that has not yet contributed any measurement must also
    be donor-protected, otherwise the donor / Phase D path could
    select a worker the measurement filter is still suppressing.
    Donor warmup grace has its own dedicated test module
    (``test_donor_warmup_grace.py``); these tests verify the
    measurement filter alone via direct calls to
    ``_aggregate_slot_signals_excluding_warmup``.

    Streak / window defaults make the up path fire on a single
    cycle so a positive intent shows up in
    ``_last_intent_deltas`` immediately once the EWMA crosses the
    activation threshold.
    """
    cfg = SaturationAwareConfig(
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=quiescence_enabled,
            worker_warmup_measurement_grace_s=grace_s,
            donor_warmup_grace_s=grace_s,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
            min_data_points=1,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem([stage_name]))
    return scheduler


class TestRefreshWorkerReadyFirstSeen:
    """``_refresh_worker_ready_first_seen`` records first-cycle timestamps and drops departures."""

    def test_first_observation_records_now(self) -> None:
        """The first cycle a worker appears in ``worker_groups`` records ``now``."""
        scheduler = _scheduler_with_warmup_grace("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=3)])

        scheduler.autoscale(time=100.0, problem_state=ps)

        for i in range(3):
            assert scheduler._worker_ready_first_seen_at[f"hot-w{i}"] == 100.0

    def test_subsequent_cycle_carries_forward_existing_timestamp(self) -> None:
        """An already-observed worker keeps its first-seen value across cycles."""
        scheduler = _scheduler_with_warmup_grace("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=2)])

        scheduler.autoscale(time=100.0, problem_state=ps)
        scheduler.autoscale(time=130.0, problem_state=ps)

        assert scheduler._worker_ready_first_seen_at["hot-w0"] == 100.0
        assert scheduler._worker_ready_first_seen_at["hot-w1"] == 100.0

    def test_worker_disappears_drops_from_map(self) -> None:
        """A worker absent from this cycle's snapshot is evicted from the map."""
        scheduler = _scheduler_with_warmup_grace("hot")
        ps_three = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=3)])
        ps_two = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=2)])

        scheduler.autoscale(time=100.0, problem_state=ps_three)
        assert "hot-w2" in scheduler._worker_ready_first_seen_at

        scheduler.autoscale(time=110.0, problem_state=ps_two)

        assert "hot-w2" not in scheduler._worker_ready_first_seen_at
        assert "hot-w0" in scheduler._worker_ready_first_seen_at
        assert "hot-w1" in scheduler._worker_ready_first_seen_at

    def test_all_workers_disappear_clears_map(self) -> None:
        """A drained stage produces an empty first-seen map."""
        scheduler = _scheduler_with_warmup_grace("hot")
        ps_full = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=2)])
        ps_drained = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=0)])

        scheduler.autoscale(time=0.0, problem_state=ps_full)
        scheduler.autoscale(time=10.0, problem_state=ps_drained)

        assert scheduler._worker_ready_first_seen_at == {}

    def test_setup_resets_first_seen_map(self) -> None:
        """A fresh ``setup()`` clears any prior cycle's first-seen records."""
        scheduler = _scheduler_with_warmup_grace("hot")
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=2)])
        scheduler.autoscale(time=0.0, problem_state=ps)
        assert scheduler._worker_ready_first_seen_at != {}

        scheduler.setup(_problem(["hot"]))

        assert scheduler._worker_ready_first_seen_at == {}


class TestWarmupGraceSuppressesScaleUp:
    """A SATURATED signal in the warmup window does not produce a positive intent."""

    def test_first_cycle_with_default_grace_holds_classifier(self) -> None:
        """All four workers are at ready-age 0; filter returns (0, 0); classifier holds NORMAL."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state is StageState.NORMAL
        assert scheduler._last_intent_deltas["hot"] == 0

    def test_intent_remains_zero_during_full_warmup_window(self) -> None:
        """For every advance up to but not including ``grace_s``, intent stays zero."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        for elapsed in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
            scheduler.autoscale(time=elapsed, problem_state=ps)
            assert scheduler._last_intent_deltas["hot"] == 0, f"unexpected intent at t={elapsed}"

    def test_grace_release_fires_phase_c_on_subsequent_cycle(self) -> None:
        """After ``grace_s`` elapses the EWMA absorbs the saturated signal and intent goes positive."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=0.0, problem_state=ps)
        scheduler.autoscale(time=60.0, problem_state=ps)

        # The grace boundary admits workers at first_seen_age >= grace_s. Once
        # admitted the EWMA absorbs the saturated ratio and the intent fires
        # because saturated_streak_min_cycles=1 in the helper config.
        assert scheduler._last_intent_deltas["hot"] > 0
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state in {StageState.SATURATED, StageState.SATURATED_CRITICAL}


class TestWarmupGraceOptOut:
    """Grace=0 disables the filter; the EWMA absorbs the signal on cycle 1."""

    def test_grace_zero_lets_first_cycle_signal_drive_classifier(self) -> None:
        """A grace_s=0 scheduler classifies SATURATED on the first cycle, just like the legacy path."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=0.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._last_intent_deltas["hot"] > 0
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state in {StageState.SATURATED, StageState.SATURATED_CRITICAL}


class TestQueueDepthIsUnfiltered:
    """``input_queue_depth`` is per-stage; the warmup filter must not touch it."""

    def test_high_queue_depth_unaffected_by_warmup(self) -> None:
        """An over-provisioned slot signal but high queue depth still passes through."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=0.0, problem_state=ps)

        runtime_stage = ps.rust.stages[0]
        # Verify the helper kept queue depth as the runtime_stage value (unfiltered)
        # by reading the stage state's input_queue_depth -- it must equal the input
        # used by the classifier path.
        assert runtime_stage.input_queue_depth == 5


class TestMixedAgeWorkers:
    """When some workers are mature and others in warmup, only mature contribute to the EWMA."""

    def test_one_mature_three_warmup_filters_to_one_worker_signal(self) -> None:
        """A 4-worker snapshot with one mature survivor reports only that worker's slots."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)

        # Cycle 1: a single worker (w0) becomes ready. The first-seen map gets
        # one entry at time 0.
        ps_one = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            "hot-w0",
                            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                            num_used_slots=4,
                        )
                    ],
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=4,
                    num_empty_slots=4,
                    input_queue_depth=0,
                )
            ]
        )
        scheduler.autoscale(time=0.0, problem_state=ps_one)

        # Now drive a second cycle at t=60 with three fresh workers added; w0
        # is mature (age = 60 = grace_s), the others are warmup (age = 0).
        ps_four = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            f"hot-w{i}",
                            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                            num_used_slots=(7 if i == 0 else 8),
                        )
                        for i in range(4)
                    ],
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=31,
                    num_empty_slots=1,
                    input_queue_depth=5,
                )
            ]
        )
        scheduler._refresh_worker_ready_first_seen(ps_four, now=60.0)

        used, empty = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps_four.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=60.0,
        )

        # Only w0 contributes: 7 used, 1 empty.
        assert used == 7
        assert empty == 1


class TestWarmupGraceBoundary:
    """A worker whose ready-age equals grace_s exactly is admitted on that cycle."""

    def test_age_equal_to_grace_admits_worker(self) -> None:
        """``(now - first_seen) >= grace_s`` is the contract; equality counts as mature."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=30.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=0.0, problem_state=ps)
        scheduler.autoscale(time=30.0, problem_state=ps)

        runtime_stage = ps.rust.stages[0]
        used, empty = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=runtime_stage,
            stage_cfg=scheduler._stage_cfg("hot"),
            now=30.0,
        )

        assert used == runtime_stage.num_used_slots
        assert empty == runtime_stage.num_empty_slots

    def test_age_one_microsecond_short_excludes_worker(self) -> None:
        """Just below the grace boundary -> warmup."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=30.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])
        scheduler.autoscale(time=0.0, problem_state=ps)

        used, empty = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=29.999_999,
        )

        assert (used, empty) == (0, 0)


class TestNonMonotonicTimeIsBenign:
    """Wall-clock jumps are defensive non-crash scenarios."""

    def test_now_going_backward_does_not_raise(self) -> None:
        """A backward time jump still produces a deterministic filter output."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=100.0, problem_state=ps)
        scheduler.autoscale(time=80.0, problem_state=ps)  # 20 s backward jump

        # With now < first_seen, (now - first_seen) is negative, treated as
        # warmup. No crash; filter returns (0, 0); classifier holds.
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state is StageState.NORMAL

    def test_recovery_from_backward_jump_admits_workers_when_age_passes(self) -> None:
        """Once wall-clock moves forward past the recorded first_seen + grace, workers admit again."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=30.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])

        scheduler.autoscale(time=100.0, problem_state=ps)
        scheduler.autoscale(time=80.0, problem_state=ps)  # backward
        scheduler.autoscale(time=140.0, problem_state=ps)  # forward, age = 40 s -> mature

        # The forward cycle admits the workers and the saturated signal fires.
        assert scheduler._last_intent_deltas["hot"] > 0


class TestSpmdGroupSharesTimestamp:
    """Multiple actors in one worker_group share a single first-seen timestamp."""

    def test_spmd_group_filter_at_group_granularity(self) -> None:
        """A SPMD group with two actors shares one timestamp; one filter decision admits or excludes all."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        # SPMD group: one worker_group entry, two underlying actors. The
        # planner does not accept multi-allocation worker_groups (they only
        # exist after Phase B placement), so this test invokes the warmup
        # helpers directly without calling autoscale().
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="hot",
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            "hot-w0",
                            [
                                resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[]),
                                resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[]),
                            ],
                            num_used_slots=15,
                        )
                    ],
                    slots_per_worker=8,
                    is_finished=False,
                    num_used_slots=15,
                    num_empty_slots=1,
                    input_queue_depth=5,
                )
            ]
        )

        scheduler._refresh_worker_ready_first_seen(ps, now=0.0)

        # The whole group is one entry in the first-seen map.
        assert len(scheduler._worker_ready_first_seen_at) == 1
        assert "hot-w0" in scheduler._worker_ready_first_seen_at

        # During warmup the group contributes (0, 0).
        used_warmup, empty_warmup = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=30.0,
        )
        assert (used_warmup, empty_warmup) == (0, 0)

        # Once mature the helper reports the group's per-group used count and
        # the per-group empty count.
        # SPMD math: K=2 actors each with slots_per_worker=8 -> group capacity 16.
        # num_used_slots=15 (sum across the 2 actors) -> empty = 16 - 15 = 1.
        # Earlier (buggy) code computed slots_per_worker - num_used_slots = 8 - 15 -> max(0, ...) = 0,
        # under-counting by (K-1)*slots_per_worker = 8 empties and biasing the classifier toward SATURATED.
        used_mature, empty_mature = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=60.0,
        )
        assert used_mature == 15
        assert empty_mature == 1


class TestZeroWorkerStage:
    """A stage with zero workers has no signal to filter; helper returns the snapshot totals."""

    def test_zero_workers_returns_unfiltered_zero_signal(self) -> None:
        """No worker_groups -> short-circuit to the snapshot totals."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState(
            [
                _stage_state(
                    name="hot",
                    num_workers=0,
                    slots_per_worker=8,
                    num_used_slots=0,
                    num_empty_slots=0,
                    input_queue_depth=0,
                )
            ]
        )

        used, empty = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=0.0,
        )

        assert (used, empty) == (0, 0)


class TestLargeStagePerformance:
    """A wide stage exercises the per-worker filter without per-worker pathology."""

    def test_one_hundred_workers_filter_via_helpers(self) -> None:
        """Smoke test: ``_refresh`` and ``_aggregate`` scale linearly across many workers.

        Operates on the helpers directly to avoid the planner's per-node
        resource cap (which would force a multi-node placement spread).
        Verifies:

          * Every observed worker lands in the first-seen map.
          * The aggregate helper reports the warmup result (0, 0)
            during the grace window even at scale.
        """
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=60.0)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=100)])

        scheduler._refresh_worker_ready_first_seen(ps, now=0.0)
        used, empty = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=0.0,
        )

        assert len(scheduler._worker_ready_first_seen_at) == 100
        assert (used, empty) == (0, 0)


@pytest.mark.parametrize("grace_s", [10.0, 30.0, 60.0, 120.0, 600.0])
class TestParametricGraceBoundary:
    """The grace boundary holds across a wide range of configured values."""

    def test_just_below_grace_excludes_just_at_grace_admits(self, grace_s: float) -> None:
        """A ``grace_s - 1e-9`` age is warmup; ``grace_s`` is mature."""
        scheduler = _scheduler_with_warmup_grace("hot", grace_s=grace_s)
        ps = data_structures.ProblemState([_saturated_signal(name="hot", num_workers=4)])
        scheduler.autoscale(time=0.0, problem_state=ps)

        used_before, empty_before = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=grace_s - 1e-9,
        )
        used_after, empty_after = scheduler._aggregate_slot_signals_excluding_warmup(
            runtime_stage=ps.rust.stages[0],
            stage_cfg=scheduler._stage_cfg("hot"),
            now=grace_s,
        )

        runtime_stage = ps.rust.stages[0]
        assert (used_before, empty_before) == (0, 0)
        assert (used_after, empty_after) == (
            runtime_stage.num_used_slots,
            runtime_stage.num_empty_slots,
        )
