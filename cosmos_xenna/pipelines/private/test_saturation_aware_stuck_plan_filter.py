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


"""Caller-side tests for the value-diff filter on the stuck-plan invariant.

``SaturationAwareScheduler._run_phase_c_grow`` has three early-return
branches that bypass the per-stage ``_set_stuck_plan_counter`` call at
the bottom of the per-stage loop:

  * pre-donor allocation-failure absorption (the first
    ``_try_add_worker_with_defense`` raised ``AllocationError``),
  * post-donor allocation-failure absorption (the second
    ``_try_add_worker_with_defense`` after a successful donation
    raised ``AllocationError``),
  * donor-retry-failed synthetic absorption (both ``try_add_worker``
    calls returned ``None``; the freed donor placement did not match
    the receiver shape).

All three leave the bailed stage's counter at its prior cycle's value.
The post-Phase-D invariant call site filters ``curr_counters`` to
non-finished stages whose value differs from the cycle-start snapshot
``prev_stuck_plan_counters``; the helper itself stays strict (only
``curr == 0`` or ``curr == prev + 1`` are valid).

Pin the contract:

  1. Each early-return path leaves the bailed stage's counter unchanged
     and does NOT raise the strict-+1-or-0 invariant.
  2. A stage Phase C touched with an illegal increment (``curr != prev``
     and not a valid transition) STILL raises -- the value-diff filter
     does not weaken the helper's strict rule.
  3. ``is_finished`` exclusion still wins: a finished stage carrying a
     stuck counter from a prior cycle never reaches the invariant.
"""

from typing import Any
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.errors import SchedulerInvariantError
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster() -> resources.ClusterResources:
    """Two-node CPU cluster matching the ``test_saturation_aware_allocation_error`` fixtures."""
    return resources.ClusterResources(
        nodes={
            "node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0"),
            "node-1": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-1"),
        },
    )


def _scheduler(*, skip_on_error: bool = True) -> SaturationAwareScheduler:
    """Single-stage scheduler primed for Phase C grow with absorption enabled."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        skip_cycle_on_allocation_error=skip_on_error,
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


class TestEarlyReturnPathsPreserveCounter:
    """The three Phase C bail paths leave the bailed stage's counter at its prior value."""

    def test_donor_retry_failed_synthetic_absorption_preserves_counter(self) -> None:
        """Both ``try_add`` calls return ``None``, donor succeeds in between.

        The synthetic ``RuntimeError("donor-retry-failed: ...")`` is built
        and routed through ``_absorb_allocation_failure``; the function
        returns from the donor-retry-failed branch BEFORE the per-stage
        counter setter at the bottom of the loop runs, so the bailed
        stage keeps its prior counter and ``curr == prev`` filters the
        no-op transition out of the invariant assertion.
        """
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        try_add_returns = iter([None, None])
        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=lambda *_a, **_k: next(try_add_returns),
            ):
                with patch.object(scheduler, "_attempt_cross_stage_donation", return_value="stage"):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._stuck_plan_counters["stage"] == 13, "the bailed stage's counter must stay at its prior value"

    def test_pre_donor_allocation_failure_preserves_counter(self) -> None:
        """First ``try_add`` raises ``AllocationError``; loop returns from the pre-donor branch.

        ``_run_phase_c_grow`` checks ``self._phase_c_allocation_failure``
        immediately after the first ``_try_add_worker_with_defense``
        call and returns when the absorb path set the flag, before the
        per-stage counter setter runs.
        """
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=resources.AllocationError("synthetic placement failure"),
            ):
                scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._stuck_plan_counters["stage"] == 13

    def test_post_donor_allocation_failure_preserves_counter(self) -> None:
        """First ``try_add`` returns None, donor succeeds, second ``try_add`` raises ``AllocationError``.

        The post-donor allocation-failure branch checks the
        ``_phase_c_allocation_failure`` flag set by the absorb path on
        the second call and returns before the per-stage counter setter
        runs, so the bailed stage keeps its prior counter.
        """
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        try_add_calls: list[int] = []

        def _try_add(*_args: Any, **_kwargs: Any) -> data_structures.ProblemWorkerGroupState | None:
            try_add_calls.append(1)
            if len(try_add_calls) == 1:
                return None
            raise resources.AllocationError("post-donor synthetic placement failure")

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            with patch(
                "cosmos_xenna.pipelines.private.data_structures.AutoscalePlanContext.try_add_worker",
                side_effect=_try_add,
            ):
                with patch.object(scheduler, "_attempt_cross_stage_donation", return_value="stage"):
                    scheduler.autoscale(time=0.0, problem_state=ps)

        assert len(try_add_calls) == 2, "both pre- and post-donor try_add calls must have been exercised"
        assert scheduler._stuck_plan_counters["stage"] == 13


class TestStrictInvariantStillFiresOnIllegalIncrement:
    """A stage with an observable counter change must still pass the strict +1/0 helper rule."""

    def test_illegal_increment_raises(self) -> None:
        """Pre-seed counter=13, monkeypatch the funnel to write 18, expect ``SchedulerInvariantError``.

        Pins that the value-diff caller filter does not weaken the
        helper rule. ``curr=18, prev=13`` is in ``changed_counters``
        (curr != prev), reaches the helper, and the strict +1/0 rule
        rejects the +5 jump.
        """
        scheduler = _scheduler()
        ps = _problem_state_with_one_worker()
        scheduler._stuck_plan_counters["stage"] = 13

        original_set = scheduler._set_stuck_plan_counter

        def _illegal_set(stage_name: str, value: int, *, last_intent: int) -> None:
            if stage_name == "stage" and value == 0:
                value = 18
            original_set(stage_name, value, last_intent=last_intent)

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 0}):
            with patch.object(scheduler, "_set_stuck_plan_counter", side_effect=_illegal_set):
                with pytest.raises(SchedulerInvariantError, match=r"stage 'stage' transitioned from 13 to 18"):
                    scheduler.autoscale(time=0.0, problem_state=ps)


class TestFinishedStageExcluded:
    """``is_finished`` is the strongest exclusion: a finished stage never reaches the invariant."""

    def test_finished_stage_with_unchanged_counter_does_not_raise(self) -> None:
        """A finished stage carrying a stuck counter from a prior cycle must not trip the invariant.

        The active-stage filter at the invariant call site excludes
        finished stages regardless of any counter value. Pre-seed
        counter=13, mark the only stage as finished, and assert no
        ``SchedulerInvariantError``.
        """
        scheduler = _scheduler()
        scheduler._stuck_plan_counters["stage"] = 13

        worker = data_structures.ProblemWorkerGroupState.make(
            "stage-w0",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
            num_used_slots=0,
        )
        ps = data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name="stage",
                    workers=[worker],
                    slots_per_worker=1,
                    is_finished=True,
                    num_used_slots=0,
                    num_empty_slots=1,
                    input_queue_depth=0,
                    num_pending_actors=0,
                ),
            ],
        )

        with patch.object(scheduler, "_compute_intent_deltas", return_value={"stage": 1}):
            scheduler.autoscale(time=0.0, problem_state=ps)

        assert scheduler._stuck_plan_counters["stage"] == 13, "finished stages keep their counter"
