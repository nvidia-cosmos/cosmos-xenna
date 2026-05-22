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


"""Multi-cycle integration tests for Phase C bail-and-recover.

End-to-end tests that drive multiple ``autoscale()`` cycles in
sequence, mirroring the production trace where the donor-retry-failed
path bails one cycle and the next cycle either recovers (cluster has
room, counter resets to 0), continues stuck (cluster still has no
placement, counter increments by exactly +1), or repeats the bail
indefinitely (counter stays at the prior value across many cycles).

Pin the contract:

  1. A bail cycle does NOT mutate the bailed stage's stuck-plan
     counter, regardless of the cycle's prior counter value.
  2. A subsequent recovery cycle resets the counter to 0 when intent
     is fully satisfied.
  3. A subsequent stuck cycle increments the counter by exactly +1
     when intent is partially satisfied with no donor available.
  4. Repeated bails preserve the counter at its prior value across
     every cycle without tripping the strict-+1-or-0 invariant.
"""

from typing import Any
from unittest.mock import patch

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster() -> resources.ClusterResources:
    """Two-node CPU cluster with headroom for Phase C grow on the recovery cycles."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0"),
            "node-1": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-1"),
        },
    )


def _scheduler() -> SaturationAwareScheduler:
    """Single-stage scheduler primed for Phase C grow with absorption enabled."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        skip_cycle_on_allocation_error=True,
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            min_workers=1,
        ),
    )
    cluster = _cluster()
    shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    problem = data_structures.Problem(
        cluster,
        [
            data_structures.ProblemStage(
                name="stage",
                stage_batch_size=1,
                worker_shape=shape,
                requested_num_workers=None,
                over_provision_factor=None,
            ),
        ],
    )
    scheduler = SaturationAwareScheduler(cfg, pipeline_name="test-pipe")
    scheduler.setup(problem)
    return scheduler


def _problem_state_with_one_worker() -> data_structures.ProblemState:
    """Single-stage ``ProblemState`` with one busy worker so Phase C wants more."""
    worker = data_structures.ProblemWorkerGroupState.make(
        "stage-w0",
        [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        num_used_slots=1,
    )
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(
                stage_name="stage",
                workers=[worker],
                slots_per_worker=1,
                is_finished=False,
                num_used_slots=1,
                num_empty_slots=0,
                input_queue_depth=0,
                num_pending_actors=0,
            ),
        ],
    )


def _drive_donor_retry_failed_cycle(
    scheduler: SaturationAwareScheduler,
    ps: data_structures.ProblemState,
    *,
    cycle_time: float,
) -> None:
    """Run one autoscale cycle that bails via the donor-retry-failed synthetic-absorption branch."""
    try_add_returns: list[Any] = [None, None]
    iter_returns = iter(try_add_returns)
    with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
        with patch(
            "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
            side_effect=lambda *_a, **_k: next(iter_returns),
        ):
            with patch.object(scheduler, "_attempt_cross_stage_donation", return_value="stage"):
                scheduler.autoscale(time=cycle_time, problem_state=ps)


def _drive_clean_grow_cycle(
    scheduler: SaturationAwareScheduler,
    ps: data_structures.ProblemState,
    *,
    cycle_time: float,
) -> None:
    """Run one autoscale cycle that fully satisfies a positive intent (counter resets to 0)."""
    with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
        scheduler.autoscale(time=cycle_time, problem_state=ps)


def _drive_clean_break_cycle(
    scheduler: SaturationAwareScheduler,
    ps: data_structures.ProblemState,
    *,
    cycle_time: float,
) -> None:
    """Run one autoscale cycle where Phase C cleanly breaks (no donor, no allocation error; counter +1)."""
    with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
        with patch(
            "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
            return_value=None,
        ):
            with patch.object(scheduler, "_attempt_cross_stage_donation", return_value=None):
                scheduler.autoscale(time=cycle_time, problem_state=ps)


class TestDonorRetryFailedThenRecover:
    """Cycle 1 bails on donor-retry-failed; cycle 2 grows fully and resets the counter to 0."""

    def test_bail_then_full_grow_resets_counter(self) -> None:
        """Counter trajectory: 13 -> 13 (bail) -> 0 (recovery)."""
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        _drive_donor_retry_failed_cycle(scheduler, ps, cycle_time=0.0)
        assert scheduler._stuck_plan_counters["stage"] == 13, "bail cycle must NOT mutate the counter"

        _drive_clean_grow_cycle(scheduler, ps, cycle_time=1.0)
        assert scheduler._stuck_plan_counters["stage"] == 0, "recovery cycle (intent satisfied) resets the counter"


class TestDonorRetryFailedThenContinueStuck:
    """Cycle 1 bails on donor-retry-failed; cycle 2 cleanly cannot place, counter advances by +1."""

    def test_bail_then_clean_break_increments_counter(self) -> None:
        """Counter trajectory: 13 -> 13 (bail) -> 14 (cluster exhausted, no donor)."""
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        _drive_donor_retry_failed_cycle(scheduler, ps, cycle_time=0.0)
        assert scheduler._stuck_plan_counters["stage"] == 13

        _drive_clean_break_cycle(scheduler, ps, cycle_time=1.0)
        assert scheduler._stuck_plan_counters["stage"] == 14, (
            "the funnel-touching cycle (intent unsatisfied) increments by +1"
        )


class TestRepeatedDonorRetryFailedBails:
    """Five consecutive donor-retry-failed bail cycles preserve the counter unchanged."""

    def test_five_consecutive_bails_preserve_counter(self) -> None:
        """Counter stays at 13 across 5 cycles; the strict invariant never fires."""
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        for cycle in range(5):
            _drive_donor_retry_failed_cycle(scheduler, ps, cycle_time=float(cycle))
            assert scheduler._stuck_plan_counters["stage"] == 13, (
                f"counter must remain at 13 across all bail cycles; cycle {cycle} drifted"
            )
