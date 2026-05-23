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


"""Lifecycle integration tests for ``SaturationAwareScheduler``.

Each test drives the same scheduler instance through several
consecutive ``autoscale()`` cycles and pins one cross-feature seam:

::

    cold-start quiescence
            |
            v
    warmup grace + trust gate
            |
            v
    first scale-up                (Phase C)
            |
            v
    stabilization / steady-state
            |
            v
    scale-down                    (Phase D)
            |
            v
    floor protection

"""

from collections.abc import Callable, Iterable

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.state import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 64) -> resources.ClusterResources:
    """Build a CPU-only cluster sized generously so placement never bottlenecks Phase C."""
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
    """Build a ``Problem`` with one CPU stage per name on the given cluster."""
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
    num_pending_actors: int,
    is_finished: bool = False,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState``; per-worker slot occupancy mirrors the stage-level signal."""
    used_per_worker = (num_used_slots // num_workers) if num_workers > 0 else 0
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            f"{name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            num_used_slots=used_per_worker,
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
        num_pending_actors=num_pending_actors,
    )


def _build_scheduler(
    stage_names: list[str],
    *,
    saturated_streak_min_cycles: int = 1,
    over_provisioned_streak_min_cycles: int = 2,
    stabilization_window_cycles_up: int = 1,
    stabilization_window_cycles_down: int = 2,
    quiescence_enabled: bool = True,
    min_data_points: int = 1,
    worker_warmup_measurement_grace_s: float = 0.0,
    donor_warmup_grace_s: float | None = None,
    max_scale_down_fraction_per_cycle: float = 1.0,
    min_workers: int | None = 1,
    cluster: resources.ClusterResources | None = None,
) -> SaturationAwareScheduler:
    """Build a scheduler whose defaults fire Phase D after two over-provisioned cycles.

    Args:
        donor_warmup_grace_s: When ``None``, defaults to
            ``worker_warmup_measurement_grace_s`` so the cross-field
            validator on :class:`SaturationAwareStageConfig` accepts
            the resulting pair.

    """
    effective_donor_grace = worker_warmup_measurement_grace_s if donor_warmup_grace_s is None else donor_warmup_grace_s
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(
            min_workers=min_workers,
            min_data_points=min_data_points,
            saturated_streak_min_cycles=saturated_streak_min_cycles,
            over_provisioned_streak_min_cycles=over_provisioned_streak_min_cycles,
            stabilization_window_cycles_up=stabilization_window_cycles_up,
            stabilization_window_cycles_down=stabilization_window_cycles_down,
            setup_phase_quiescence_enabled=quiescence_enabled,
            worker_warmup_measurement_grace_s=worker_warmup_measurement_grace_s,
            donor_warmup_grace_s=effective_donor_grace,
            max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
        ),
    )
    scheduler = SaturationAwareScheduler(cfg)
    scheduler.setup(_problem(stage_names, cluster=cluster))
    return scheduler


def _run_cycles(
    scheduler: SaturationAwareScheduler,
    state_factory: Callable[[int], data_structures.ProblemState],
    *,
    num_cycles: int,
    cycle_interval_s: float = 10.0,
    start_time_s: float = 0.0,
) -> list[data_structures.Solution]:
    """Drive ``num_cycles`` cycles spaced ``cycle_interval_s`` apart; return every ``Solution``."""
    solutions: list[data_structures.Solution] = []
    for cycle_idx in range(num_cycles):
        ps = state_factory(cycle_idx)
        sol = scheduler.autoscale(time=start_time_s + cycle_idx * cycle_interval_s, problem_state=ps)
        solutions.append(sol)
    return solutions


def _new_workers_per_stage(solutions: Iterable[data_structures.Solution], stage_index: int) -> list[int]:
    """Per-cycle list of ``len(new_workers)`` for one stage index."""
    return [len(sol.stages[stage_index].new_workers) for sol in solutions]


def _deleted_workers_per_stage(solutions: Iterable[data_structures.Solution], stage_index: int) -> list[int]:
    """Per-cycle list of ``len(deleted_workers)`` for one stage index."""
    return [len(sol.stages[stage_index].deleted_workers) for sol in solutions]


def _saturated(
    name: str,
    *,
    num_workers: int,
    slots_per_worker: int = 8,
    num_pending_actors: int = 0,
) -> data_structures.ProblemStageState:
    """Build a SATURATED_CRITICAL-shaped slot signal for the stage."""
    total = num_workers * slots_per_worker
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=total,
        num_empty_slots=0,
        input_queue_depth=5,
        num_pending_actors=num_pending_actors,
    )


def _over_provisioned(
    name: str,
    *,
    num_workers: int,
    slots_per_worker: int = 8,
    num_pending_actors: int = 0,
) -> data_structures.ProblemStageState:
    """Build an OVER_PROVISIONED-shaped slot signal: mostly idle with a non-empty queue."""
    total = num_workers * slots_per_worker
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=1,
        num_empty_slots=max(total - 1, 0),
        input_queue_depth=8,
        num_pending_actors=num_pending_actors,
    )


def _normal(
    name: str,
    *,
    num_workers: int,
    slots_per_worker: int = 8,
    num_pending_actors: int = 0,
) -> data_structures.ProblemStageState:
    """Build a NORMAL-shaped slot signal at 50 % occupancy with a small input queue."""
    total = num_workers * slots_per_worker
    used = total // 2
    return _stage_state(
        name=name,
        num_workers=num_workers,
        slots_per_worker=slots_per_worker,
        num_used_slots=used,
        num_empty_slots=total - used,
        input_queue_depth=2,
        num_pending_actors=num_pending_actors,
    )


class TestColdStartQuiescenceReleases:
    """Cold-start gate releases when at least one pending actor becomes ready."""

    def test_cold_start_quiescence_releases_when_actor_becomes_ready(self) -> None:
        """``pending>0, ready==0`` skips the pipeline; first READY actor re-engages it.

        Across a 3-cycle run the same scheduler instance must traverse
        the cold-start gate (no intent recorded) and then route the
        live signal through the classifier on the cycle the gate
        releases.
        """
        scheduler = _build_scheduler(["hot"])

        def state_factory(cycle: int) -> data_structures.ProblemState:
            if cycle == 0:
                # Pure cold-start: pending actors, zero ready.
                return data_structures.ProblemState(
                    [
                        _stage_state(
                            name="hot",
                            num_workers=0,
                            slots_per_worker=8,
                            num_used_slots=0,
                            num_empty_slots=0,
                            input_queue_depth=0,
                            num_pending_actors=2,
                        ),
                    ]
                )
            # Cycle 1+: pending drained, ready actors carry a SATURATED signal.
            return data_structures.ProblemState([_saturated("hot", num_workers=2)])

        solutions = _run_cycles(scheduler, state_factory, num_cycles=3)

        # Cycle 0: gate skipped the pipeline so no intent entry was recorded.
        # The Phase B floor still adds a worker, but the intent dict is empty.
        # Reading after cycle 2 verifies the cycle-1 transition did not leak
        # a stale cycle-0 intent forward.
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state in {
            StageState.SATURATED,
            StageState.SATURATED_CRITICAL,
        }, "post-release classifier must observe the SATURATED signal once gate releases"
        # After release, the classifier streak must have advanced past zero
        # because cycles 1 and 2 each fed it a non-empty signal.
        assert runtime.classifier_streak >= 1
        # No cycle should have rejected the structural Solution shape; every
        # cycle must produce a stage entry for "hot".
        assert all(len(sol.stages) == 1 for sol in solutions)


class TestWarmupGraceExcludesFromTrustGate:
    """Warmup grace blocks fresh ready signal from advancing the trust gate."""

    def test_warmup_grace_excludes_freshly_ready_signal_from_trust_gate(self) -> None:
        """``valid_signal_samples`` ticks only after every worker exits warmup.

        While every ready actor is still inside
        ``worker_warmup_measurement_grace_s`` the trust gate must
        stay at zero samples. Once wall-clock time advances past
        the grace, the same signal must be admitted and the gate
        must open within ``min_data_points`` cycles.
        """
        grace_s = 60.0
        min_data_points = 3
        scheduler = _build_scheduler(
            ["hot"],
            worker_warmup_measurement_grace_s=grace_s,
            min_data_points=min_data_points,
        )

        # Stable per-worker layout across cycles. Same worker ids each cycle
        # so the per-worker first-seen timestamp pins ``cycle 0 == time 0``.
        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState([_saturated("hot", num_workers=2)])

        # Three cycles inside the warmup window (t in {0, 10, 20} <= grace=60s).
        warmup_solutions = _run_cycles(scheduler, state_factory, num_cycles=3, cycle_interval_s=10.0)

        runtime_in_warmup = scheduler._stage_states["hot"]
        assert runtime_in_warmup.valid_signal_samples == 0, (
            f"warmup workers must not advance valid_signal_samples while inside grace, "
            f"got {runtime_in_warmup.valid_signal_samples}"
        )
        # While the trust gate is closed, Phase C cannot grow even on a
        # SATURATED signal -- Phase B floor adds may still happen, but Phase C
        # is gated by the trust counter. Verify the recommendation history
        # stayed empty across all three warmup cycles (no per-stage pipeline
        # ran, so no recommendation was recorded).
        history_in_warmup = scheduler._recommendation_histories["hot"]
        assert len(history_in_warmup._buffer) == 0, (
            "no per-stage pipeline iterations should have recorded a recommendation "
            "while every worker was still in warmup grace"
        )

        # Three more cycles after warmup. At time=70s the first cycle's
        # workers are post-warmup (t-first_seen = 70-0 > 60); subsequent
        # cycles must tick the valid-signal counter monotonically toward
        # min_data_points.
        post_warmup_solutions = _run_cycles(
            scheduler,
            state_factory,
            num_cycles=min_data_points,
            cycle_interval_s=10.0,
            start_time_s=70.0,
        )

        runtime_post = scheduler._stage_states["hot"]
        assert runtime_post.valid_signal_samples >= min_data_points, (
            f"valid_signal_samples must reach min_data_points={min_data_points} after the "
            f"warmup grace expires, got {runtime_post.valid_signal_samples}"
        )
        # Both phases of the run produced structurally valid solutions.
        assert all(len(sol.stages) == 1 for sol in warmup_solutions + post_warmup_solutions)


class TestSteadyState30CyclesNoOscillation:
    """30 consecutive NORMAL cycles produce no oscillation after the floor settles."""

    def test_steady_state_30_cycles_no_oscillation(self) -> None:
        """30 cycles of NORMAL signal yield zero Phase D deletes; no post-floor adds.

        A 30-cycle horizon catches slow oscillation drift that a
        few-cycle test would miss (e.g. the recommendation history
        admitting a stray shrink vote after enough quiet cycles, or
        the EWMA drifting out of NORMAL).
        """
        scheduler = _build_scheduler(["hot"])

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            # Constant NORMAL signal across all 30 cycles. The worker
            # count we report is constant too; the scheduler's
            # ``Solution`` does not feed back into our state factory,
            # so the steady-state assertion checks "did the scheduler
            # ever want to change?" rather than simulating placement
            # convergence.
            return data_structures.ProblemState([_normal("hot", num_workers=4)])

        solutions = _run_cycles(scheduler, state_factory, num_cycles=30)

        # Cycle 0 may produce Phase B floor adds even though the stage
        # is already above the implicit floor=1; verify that NO cycle
        # produces Phase D deletes and NO cycle past the initial floor
        # cycle produces a Phase C add.
        total_deletes = sum(_deleted_workers_per_stage(solutions, stage_index=0))
        assert total_deletes == 0, (
            f"steady NORMAL signal must not trigger Phase D shrink across 30 cycles, got {total_deletes} deletes"
        )
        # Cycles 1..29: zero new_workers (Phase B floor enforcement is allowed on cycle 0
        # only because the initial workers list in problem_state and the floor decision
        # produce a one-time settling step).
        post_settle_adds = sum(len(sol.stages[0].new_workers) for sol in solutions[1:])
        assert post_settle_adds == 0, (
            f"steady NORMAL signal must not trigger Phase C grow after the initial floor "
            f"cycle; got {post_settle_adds} adds across cycles 1..29"
        )
        # The classifier must stay in NORMAL across the whole horizon -- a
        # SATURATED or OVER_PROVISIONED transition mid-run would mean the
        # signal we believed was NORMAL drifted out of band.
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state is StageState.NORMAL


class TestDemandSpikeStabilizationHoldsNewLevel:
    """A demand spike fires Phase C grow at least once; stabilization avoids spam."""

    def test_demand_spike_drives_phase_c_then_stabilization_holds_new_level(self) -> None:
        """Sustained SATURATED signal fires Phase C at least once across the spike.

        Drives a stable SATURATED signal across enough cycles to
        satisfy ``saturated_streak_min_cycles`` plus
        ``stabilization_window_cycles_up``. Per-cycle grow magnitude
        is owned by the Phase C unit tests; the integration check
        is the seam between the classifier streak ramp and the
        recommendation-history gate.
        """
        scheduler = _build_scheduler(
            ["hot"],
            saturated_streak_min_cycles=2,
            stabilization_window_cycles_up=1,
            over_provisioned_streak_min_cycles=3,
            stabilization_window_cycles_down=3,
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            return data_structures.ProblemState([_saturated("hot", num_workers=2)])

        # Drive 5 cycles of saturated signal; that is generously above
        # streak (2) + window_up (1) so the classifier and recommendation
        # history both ripen.
        solutions = _run_cycles(scheduler, state_factory, num_cycles=5)

        total_adds = sum(_new_workers_per_stage(solutions, stage_index=0))
        assert total_adds >= 1, (
            f"sustained SATURATED signal must fire Phase C grow at least once across "
            f"{len(solutions)} cycles, got {total_adds} adds"
        )
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state in {
            StageState.SATURATED,
            StageState.SATURATED_CRITICAL,
        }, "classifier must stay in the saturated band while signal sustains"
        # Every per-stage solution must keep its identity (no shape drift
        # across cycles); the streaming layer fuses these into the same
        # actor pool keyed by stage index.
        assert all(len(sol.stages) == 1 for sol in solutions)


class TestDemandDropStabilizationHoldsFloor:
    """A demand drop fires Phase D shrink at least once and stops at the floor."""

    def test_demand_drop_drives_phase_d_then_stabilization_holds_floor(self) -> None:
        """Sustained OVER_PROVISIONED signal fires Phase D; floor blocks further shrink.

        Starts at 5 workers with ``min_workers=1`` and drives
        sustained OVER_PROVISIONED signal long enough to ripen the
        asymmetric stabilization window. Across the run Phase D
        fires at least once and no per-cycle delete request
        exceeds ``live - floor``.
        """
        scheduler = _build_scheduler(
            ["hot"],
            saturated_streak_min_cycles=1,
            over_provisioned_streak_min_cycles=2,
            stabilization_window_cycles_up=1,
            stabilization_window_cycles_down=2,
            min_workers=1,
        )

        def state_factory(_cycle: int) -> data_structures.ProblemState:
            # 5 idle workers; floor=1; expected to scale down toward floor.
            return data_structures.ProblemState([_over_provisioned("hot", num_workers=5)])

        solutions = _run_cycles(scheduler, state_factory, num_cycles=6)

        total_deletes = sum(_deleted_workers_per_stage(solutions, stage_index=0))
        assert total_deletes >= 1, (
            f"sustained OVER_PROVISIONED signal must fire Phase D shrink at least once "
            f"across {len(solutions)} cycles, got {total_deletes} deletes"
        )
        # Floor protection: across the entire run, the per-cycle delete
        # request must not exceed (live - floor). Live count is the worker
        # count we feed in per cycle (constant 5 here); floor is 1.
        live_per_cycle = 5
        floor = 1
        per_cycle_deletes = [len(sol.stages[0].deleted_workers) for sol in solutions]
        assert all(d <= live_per_cycle - floor for d in per_cycle_deletes), (
            f"Phase D must not request deletes that would cross floor={floor} "
            f"with live={live_per_cycle}: {per_cycle_deletes}"
        )
        runtime = scheduler._stage_states["hot"]
        assert runtime.classifier_state is StageState.OVER_PROVISIONED
