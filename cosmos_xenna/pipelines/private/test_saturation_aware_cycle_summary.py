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


"""Tests for the per-cycle DEBUG summary emitted by ``_emit_cycle_summary``."""

import logging
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture."""
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


def _scheduler() -> SaturationAwareScheduler:
    """Single-stage scheduler primed for one autoscale cycle."""
    cfg = SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=SaturationAwareStageConfig(
            setup_phase_quiescence_enabled=False,
            worker_warmup_measurement_grace_s=0.0,
            donor_warmup_grace_s=0.0,
            min_workers=1,
        ),
    )
    cluster = resources.ClusterResources(
        nodes={"node-0": resources.NodeResources(used_cpus=0, total_cpus=8, gpus=[], name="node-0")},
    )
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
    scheduler = SaturationAwareScheduler(cfg, pipeline_name="p")
    scheduler.setup(problem)
    return scheduler


def _problem_state() -> data_structures.ProblemState:
    """Single-stage runtime state with one worker."""
    worker = data_structures.ProblemWorkerGroupState.make(
        "stage-w0",
        [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        num_used_slots=0,
    )
    return data_structures.ProblemState(
        [
            data_structures.ProblemStageState(
                stage_name="stage",
                workers=[worker],
                slots_per_worker=1,
                is_finished=False,
                num_used_slots=0,
                num_empty_slots=1,
                input_queue_depth=0,
                num_pending_actors=0,
            ),
        ],
    )


class TestCycleSummary:
    """Pin the cycle-summary DEBUG contract."""

    def test_cycle_summary_emits_one_debug_line_per_cycle(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Every autoscale cycle emits exactly one summary line at DEBUG."""
        scheduler = _scheduler()
        ps = _problem_state()
        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"stage": 0},
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)

        summary_records = [r for r in loguru_caplog.records if "cycle 1 summary" in r.message]
        assert len(summary_records) == 1
        assert summary_records[0].levelno == logging.DEBUG
        msg = summary_records[0].message
        for needle in (
            "regime=",
            "heterogeneity_streak=",
            "heterogeneity_fired=",
            "manual_allocation_aborted_cycle=",
            "floor_allocation_aborted_cycle=",
            "grow_allocation_aborted_cycle=",
        ):
            assert needle in msg

    def test_cycle_summary_is_below_info_level(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Operators running at the default INFO level do not see the summary."""
        scheduler = _scheduler()
        ps = _problem_state()
        loguru_caplog.set_level(logging.INFO, logger="loguru")
        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value={"stage": 0},
        ):
            scheduler.autoscale(time=0.0, problem_state=ps)

        info_or_higher = [r for r in loguru_caplog.records if r.levelno >= logging.INFO and "summary" in r.message]
        assert info_or_higher == []
