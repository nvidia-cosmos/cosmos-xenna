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


"""Tests for the setup-aware ``max_queued`` cold-start backpressure cap.

When a pool has only pending actors (``num_pending_actors > 0``) and no
ready actors yet, the streaming backpressure loop reduces the per-stage
``max_queued`` to a cold-start floor so the upstream stage does not
pre-queue object-store data ahead of model warmup. Restoration is
automatic on the next loop iteration that observes any ready actors.

The cold-start branch is gated by the per-stage
``setup_aware_max_queued`` flag (``SaturationAwareStageConfig``,
default True) and is bypassed for stages already marked finished so a
draining pool's transient ready/pending counts cannot trip the cap.

The downstream batch size is preserved as a lower bound on the cold-start
cap so a slow upstream stage cannot starve a fast downstream stage during
cold start.
"""

import logging
from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import resources, specs
from cosmos_xenna.pipelines.private.scheduling_py.cluster.streaming_backpressure import (
    _SETUP_AWARE_MAX_QUEUED_FLOOR,
    compute_max_queued,
    resolve_setup_aware_max_queued_enabled,
)


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


class _StageForSetupAwareResolver(specs.Stage[int, int]):
    """Minimal CPU stage used only to build ``StageSpec`` fixtures."""

    @property
    def required_resources(self) -> resources.Resources:
        return resources.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int]:
        return in_data


def _pipeline_spec(
    *,
    scheduler: specs.SchedulerKind,
    stage_default: bool = True,
    per_stage_override: bool | None = None,
    spec_override: bool | None = None,
    omit_mode_specific: bool = False,
    execution_mode: specs.ExecutionMode = specs.ExecutionMode.STREAMING,
) -> specs.PipelineSpec:
    """Build the smallest pipeline spec needed to resolve the per-stage flag."""
    stage = specs.StageSpec(
        _StageForSetupAwareResolver(),
        saturation_aware=(
            specs.SaturationAwareStageConfig(setup_aware_max_queued=spec_override)
            if spec_override is not None
            else None
        ),
    )
    saturation_aware = specs.SaturationAwareConfig(
        stage_defaults=specs.SaturationAwareStageConfig(setup_aware_max_queued=stage_default),
        per_stage_overrides=(
            {
                "Stage 00 - _StageForSetupAwareResolver": specs.SaturationAwareStageConfig(
                    setup_aware_max_queued=per_stage_override
                )
            }
            if per_stage_override is not None
            else {}
        ),
    )
    mode_specific = (
        None
        if omit_mode_specific
        else specs.StreamingSpecificSpec(
            scheduler=scheduler,
            saturation_aware=saturation_aware,
        )
    )
    return specs.PipelineSpec(
        input_data=[1],
        stages=[stage],
        config=specs.PipelineConfig(mode_specific=mode_specific, execution_mode=execution_mode),
    )


class TestSetupAwareMaxQueued:
    """``compute_max_queued`` honours the cold-start branch and its gates."""

    def test_zero_ready_one_pending_reduces_max_queued(self) -> None:
        """Cold-start input lowers the cap when the flag is on; flag off keeps the regular value."""
        cold_start_cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=1,
            slots_per_actor=2,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=1,
            setup_aware_enabled=True,
            is_done=False,
            stage_name="Stage 00 - Foo",
        )
        regular_cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=1,
            slots_per_actor=2,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=1,
            setup_aware_enabled=False,
            is_done=False,
            stage_name="Stage 00 - Foo",
        )

        assert cold_start_cap == _SETUP_AWARE_MAX_QUEUED_FLOOR
        assert regular_cap == 8
        assert cold_start_cap < regular_cap

    def test_some_ready_uses_regular_max_queued(self) -> None:
        """Any ready actor blocks the cold-start branch even when pending actors exist."""
        cap = compute_max_queued(
            num_ready_actors=3,
            num_pending_actors=2,
            slots_per_actor=4,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=1,
            setup_aware_enabled=True,
            is_done=False,
            stage_name="Stage 01 - Bar",
        )

        assert cap == max(int(3 * 4 * 1.0), 8, 1)
        assert cap == 12

    def test_zero_ready_zero_pending_uses_regular(self) -> None:
        """A pool with no actors at all is not in cold start; regular formula applies."""
        cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=0,
            slots_per_actor=2,
            max_queued_multiplier=1.5,
            max_queued_lower_bound=8,
            next_stage_batch_size=4,
            setup_aware_enabled=True,
            is_done=False,
            stage_name="Stage 02 - Baz",
        )

        assert cap == max(0, 8, 4)
        assert cap == 8

    def test_finished_stage_excluded(self) -> None:
        """``is_done=True`` bypasses the cold-start branch even with pending actors present."""
        cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=1,
            slots_per_actor=2,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=1,
            setup_aware_enabled=True,
            is_done=True,
            stage_name="Stage 03 - Qux",
        )

        assert cap == max(0, 8, 1)
        assert cap == 8

    def test_feature_flag_off_uses_regular_formula(self) -> None:
        """The cold-start cap is never used when the per-stage flag is False."""
        cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=4,
            slots_per_actor=2,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=1,
            setup_aware_enabled=False,
            is_done=False,
            stage_name="Stage 04 - Quux",
        )

        assert cap == 8

    def test_cold_start_floor_respects_next_stage_batch_size(self) -> None:
        """``next_stage_batch_size`` floors the cold-start cap so downstream is not starved."""
        next_stage_batch_size = 16
        assert next_stage_batch_size > _SETUP_AWARE_MAX_QUEUED_FLOOR

        cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=1,
            slots_per_actor=2,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=next_stage_batch_size,
            setup_aware_enabled=True,
            is_done=False,
            stage_name="Stage 05 - Corge",
        )

        assert cap == next_stage_batch_size

    def test_cold_start_emits_debug_log(self, loguru_caplog: pytest.LogCaptureFixture) -> None:
        """Operators can correlate cap reductions with stage state via a debug log."""
        cap = compute_max_queued(
            num_ready_actors=0,
            num_pending_actors=2,
            slots_per_actor=2,
            max_queued_multiplier=1.0,
            max_queued_lower_bound=8,
            next_stage_batch_size=1,
            setup_aware_enabled=True,
            is_done=False,
            stage_name="Stage 06 - Grault",
        )

        cold_start_logs = [r for r in loguru_caplog.records if "setup-aware max_queued" in r.message]

        assert cap == _SETUP_AWARE_MAX_QUEUED_FLOOR
        assert len(cold_start_logs) == 1, (
            f"expected exactly one cold-start debug log, got {len(cold_start_logs)}; "
            f"all records: {[r.message for r in loguru_caplog.records]}"
        )
        msg = cold_start_logs[0].message
        assert "num_ready=0" in msg
        assert "num_pending=2" in msg
        assert "cold-start cap reduced from 8 to 1" in msg


class TestResolveSetupAwareMaxQueuedEnabled:
    """Resolver honours scheduler kind and saturation-aware config precedence."""

    def test_fragmentation_scheduler_disables_setup_aware_cap(self) -> None:
        """Fragmentation-based scheduling always uses the regular backpressure path."""
        pipeline_spec = _pipeline_spec(
            scheduler=specs.SchedulerKind.FRAGMENTATION_BASED,
            stage_default=True,
            spec_override=True,
        )

        enabled = resolve_setup_aware_max_queued_enabled(
            pipeline_spec,
            stage_idx=0,
            stage_name="Stage 00 - _StageForSetupAwareResolver",
        )

        assert enabled is False

    def test_missing_streaming_mode_specific_disables_setup_aware_cap(self) -> None:
        """Batch-style specs without streaming config cannot enable the cold-start cap."""
        pipeline_spec = _pipeline_spec(
            scheduler=specs.SchedulerKind.SATURATION_AWARE,
            omit_mode_specific=True,
        )

        enabled = resolve_setup_aware_max_queued_enabled(
            pipeline_spec,
            stage_idx=0,
            stage_name="Stage 00 - _StageForSetupAwareResolver",
        )

        assert enabled is False

    def test_batch_execution_mode_disables_setup_aware_cap(self) -> None:
        """Batch pipelines must not materialize saturation-aware config for backpressure."""
        pipeline_spec = _pipeline_spec(
            scheduler=specs.SchedulerKind.SATURATION_AWARE,
            stage_default=True,
            execution_mode=specs.ExecutionMode.BATCH,
        )

        enabled = resolve_setup_aware_max_queued_enabled(
            pipeline_spec,
            stage_idx=0,
            stage_name="Stage 00 - _StageForSetupAwareResolver",
        )

        assert enabled is False

    def test_per_stage_override_wins_over_default(self) -> None:
        """Named saturation-aware override resolves above the cluster default."""
        pipeline_spec = _pipeline_spec(
            scheduler=specs.SchedulerKind.SATURATION_AWARE,
            stage_default=True,
            per_stage_override=False,
        )

        enabled = resolve_setup_aware_max_queued_enabled(
            pipeline_spec,
            stage_idx=0,
            stage_name="Stage 00 - _StageForSetupAwareResolver",
        )

        assert enabled is False

    def test_stage_spec_override_wins_over_named_override(self) -> None:
        """``StageSpec.saturation_aware`` has highest precedence for the flag."""
        pipeline_spec = _pipeline_spec(
            scheduler=specs.SchedulerKind.SATURATION_AWARE,
            stage_default=True,
            per_stage_override=False,
            spec_override=True,
        )

        enabled = resolve_setup_aware_max_queued_enabled(
            pipeline_spec,
            stage_idx=0,
            stage_name="Stage 00 - _StageForSetupAwareResolver",
        )

        assert enabled is True
