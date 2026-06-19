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

"""Focused unit tests for the cold-start ramp (native-extension-free)."""

from collections.abc import Callable

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.config import SaturationAwareConfig
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.ramp import RampReason, StageRampInput, decide

type RampInputFactory = Callable[..., StageRampInput]


@pytest.fixture
def make_input() -> RampInputFactory:
    """Return a factory for ramp inputs defaulting to a cold untrusted stage."""

    def _make_input(
        *,
        current_workers: int = 0,
        deleted_count: int = 0,
        proposed_post: int = 10,
        sample_count: int = 0,
        pending_work_age_s: float = 0.0,
        has_pending_work: bool = False,
        w_target: int | None = None,
        pipeline_warming: bool = False,
    ) -> StageRampInput:
        return StageRampInput(
            current_workers=current_workers,
            deleted_count=deleted_count,
            proposed_post=proposed_post,
            sample_count=sample_count,
            pending_work_age_s=pending_work_age_s,
            has_pending_work=has_pending_work,
            w_target=w_target,
            pipeline_warming=pipeline_warming,
        )

    return _make_input


def _config() -> SaturationAwareConfig:
    """Build a config with the default trust threshold of 5 samples."""
    return SaturationAwareConfig()


def test_cold_start_with_no_samples_caps_at_one(make_input: RampInputFactory) -> None:
    """Any stage with no completed sample is capped at one worker."""
    decision = decide(make_input(sample_count=0, current_workers=0, proposed_post=11), _config())
    assert decision.cap == 1
    assert decision.keep_new == 1
    assert decision.reason is RampReason.COLD


def test_no_samples_running_stage_trims_all_new_workers(make_input: RampInputFactory) -> None:
    """With one worker already up but no sample or evidence, no further workers are added."""
    decision = decide(make_input(sample_count=0, current_workers=1, proposed_post=11), _config())
    assert decision.cap == 1
    assert decision.keep_new == 0


def test_cold_stage_above_cap_adds_no_new_workers(make_input: RampInputFactory) -> None:
    """A 0-sample stage already above the one-worker cap gains nothing; the ramp never deletes existing workers."""
    decision = decide(make_input(sample_count=0, current_workers=2, proposed_post=11), _config())
    assert decision.cap == 1
    assert decision.keep_new == 0
    assert decision.reason is RampReason.COLD


def test_no_sample_within_window_stays_cold(make_input: RampInputFactory) -> None:
    """A 0-sample stage is held at one worker until a full estimation window elapses."""
    config = SaturationAwareConfig()
    decision = decide(make_input(sample_count=0, pending_work_age_s=config.speed_estimation_window_s - 1.0), config)
    assert decision.cap == 1
    assert decision.reason is RampReason.COLD


def test_no_sample_after_window_with_pending_work_trusts_solver(make_input: RampInputFactory) -> None:
    """A stage with work waiting but no sample after a full window is released to the solver."""
    config = SaturationAwareConfig()
    decision = decide(
        make_input(sample_count=0, pending_work_age_s=config.speed_estimation_window_s, has_pending_work=True), config
    )
    assert decision.cap is None
    assert decision.keep_new is None
    assert decision.reason is RampReason.SLOW_START


def test_no_sample_after_window_without_pending_work_stays_cold(make_input: RampInputFactory) -> None:
    """A starved stage (no work waiting) stays capped past the window, never over-spawned."""
    config = SaturationAwareConfig()
    decision = decide(
        make_input(sample_count=0, pending_work_age_s=config.speed_estimation_window_s, has_pending_work=False), config
    )
    assert decision.cap == 1
    assert decision.reason is RampReason.COLD


def test_pipeline_warming_grows_by_one_on_local_work(make_input: RampInputFactory) -> None:
    """A 0-sample stage with a live worker and its own backlog grows by exactly one."""
    decision = decide(
        make_input(sample_count=0, current_workers=1, proposed_post=13, has_pending_work=True),
        _config(),
    )
    assert decision.cap == 2
    assert decision.keep_new == 1
    assert decision.reason is RampReason.PIPELINE_WARMING


def test_pipeline_warming_requires_pending_work(make_input: RampInputFactory) -> None:
    """With no pending work of its own, a 0-sample stage holds at one worker."""
    decision = decide(
        make_input(
            sample_count=0,
            current_workers=1,
            proposed_post=13,
            has_pending_work=False,
        ),
        _config(),
    )
    assert decision.cap == 1
    assert decision.reason is RampReason.COLD


def test_pipeline_warming_requires_an_existing_worker(make_input: RampInputFactory) -> None:
    """A stage with no live worker clears the first cold cap before pending work can accelerate it."""
    decision = decide(
        make_input(
            sample_count=0,
            current_workers=0,
            proposed_post=13,
            has_pending_work=True,
        ),
        _config(),
    )
    assert decision.cap == 1
    assert decision.reason is RampReason.COLD


def test_pipeline_warming_caps_growth_at_one_per_cycle(make_input: RampInputFactory) -> None:
    """Pipeline-evidence warming adds a single worker even when the solver wants many more."""
    decision = decide(
        make_input(sample_count=0, current_workers=3, proposed_post=15, has_pending_work=True),
        _config(),
    )
    assert decision.cap == 4
    assert decision.keep_new == 1
    assert decision.reason is RampReason.PIPELINE_WARMING


def test_pipeline_warming_keeps_one_net_worker_when_solver_also_deletes(make_input: RampInputFactory) -> None:
    """With deletes proposed alongside additions, the kept-new count still nets exactly one extra worker."""
    decision = decide(
        make_input(
            sample_count=0,
            current_workers=3,
            deleted_count=2,
            proposed_post=10,
            has_pending_work=True,
        ),
        _config(),
    )
    assert decision.cap == 4
    assert decision.keep_new == 3
    assert decision.reason is RampReason.PIPELINE_WARMING


def test_slow_start_takes_precedence_over_pipeline_warming(make_input: RampInputFactory) -> None:
    """Once the window elapses with work waiting, the solver is trusted even with full warming preconditions."""
    config = SaturationAwareConfig()
    decision = decide(
        make_input(
            sample_count=0,
            current_workers=1,
            proposed_post=13,
            pending_work_age_s=config.speed_estimation_window_s,
            has_pending_work=True,
        ),
        config,
    )
    assert decision.cap is None
    assert decision.reason is RampReason.SLOW_START


def test_warming_grows_by_one_with_local_work(make_input: RampInputFactory) -> None:
    """A warming stage with its own backlog grows by exactly one worker per cycle."""
    decision = decide(
        make_input(sample_count=1, current_workers=1, proposed_post=11, has_pending_work=True),
        _config(),
    )
    assert decision.cap == 2
    assert decision.keep_new == 1
    assert decision.reason is RampReason.WARMING


def test_warming_caps_growth_at_one_per_cycle_regardless_of_sample_count(make_input: RampInputFactory) -> None:
    """Warming growth is a fixed one per cycle and never scales with sample count or the solver proposal."""
    decision = decide(
        make_input(sample_count=4, current_workers=1, proposed_post=11, has_pending_work=True),
        _config(),
    )
    assert decision.cap == 2
    assert decision.keep_new == 1
    assert decision.reason is RampReason.WARMING


def test_warming_past_window_is_not_released_to_solver(make_input: RampInputFactory) -> None:
    """The slow-starter release is cold-only: a stage with samples past the window still grows +1, never uncapped."""
    config = SaturationAwareConfig()
    decision = decide(
        make_input(
            sample_count=2,
            current_workers=1,
            proposed_post=11,
            pending_work_age_s=config.speed_estimation_window_s,
            has_pending_work=True,
        ),
        config,
    )
    assert decision.cap == 2
    assert decision.keep_new == 1
    assert decision.reason is RampReason.WARMING


def test_warming_dry_stage_holds_at_current(make_input: RampInputFactory) -> None:
    """A warming stage with no pending work of its own holds at its current size."""
    decision = decide(
        make_input(
            sample_count=2,
            current_workers=3,
            proposed_post=10,
            has_pending_work=False,
        ),
        _config(),
    )
    assert decision.cap == 3
    assert decision.keep_new == 0
    assert decision.reason is RampReason.WARMING


def test_trusted_stage_without_capacity_target_is_uncapped(make_input: RampInputFactory) -> None:
    """A trusted stage with no capacity target (no measured bottleneck) is uncapped."""
    decision = decide(make_input(sample_count=5, current_workers=1, proposed_post=11, w_target=None), _config())
    assert decision.cap is None
    assert decision.keep_new is None
    assert decision.reason is RampReason.UNCAPPED


def test_trusted_stage_is_capped_to_w_target(make_input: RampInputFactory) -> None:
    """A trusted stage's large solver proposal is trimmed to its capacity target."""
    decision = decide(
        make_input(sample_count=5, current_workers=1, deleted_count=0, proposed_post=12, w_target=2),
        _config(),
    )
    assert decision.cap == 2
    assert decision.keep_new == 1
    assert decision.reason is RampReason.CAPPED


def test_trusted_proposal_within_w_target_is_not_trimmed(make_input: RampInputFactory) -> None:
    """A trusted stage proposing at or below its capacity target keeps all new workers."""
    decision = decide(
        make_input(sample_count=5, current_workers=1, proposed_post=4, w_target=5),
        _config(),
    )
    assert decision.cap == 5
    assert decision.keep_new is None
    assert decision.reason is RampReason.CAPPED


def test_trusted_w_target_below_current_never_forces_shrink(make_input: RampInputFactory) -> None:
    """A capacity target below the current size trims growth to zero, never deletes."""
    decision = decide(
        make_input(sample_count=5, current_workers=5, deleted_count=0, proposed_post=5, w_target=2),
        _config(),
    )
    assert decision.cap == 2
    assert decision.keep_new == 0
    assert decision.reason is RampReason.CAPPED


def test_trusted_warming_bounds_growth_to_step(make_input: RampInputFactory) -> None:
    """While the pipeline is warming, a trusted stage grows by at most the warmup step, not to w_target."""
    config = _config()
    decision = decide(
        make_input(sample_count=5, current_workers=2, proposed_post=57, w_target=57, pipeline_warming=True),
        config,
    )
    assert decision.cap == 2 + config.pipeline_warmup_growth_step
    assert decision.keep_new == config.pipeline_warmup_growth_step
    assert decision.reason is RampReason.CAPPED_WARMUP


def test_trusted_warming_uses_w_target_when_below_step(make_input: RampInputFactory) -> None:
    """When w_target is below the warmup step, the bound is w_target, so a low target is honored."""
    decision = decide(
        make_input(sample_count=5, current_workers=2, proposed_post=10, w_target=3, pipeline_warming=True),
        _config(),
    )
    assert decision.cap == 3
    assert decision.keep_new == 1
    assert decision.reason is RampReason.CAPPED_WARMUP


def test_trusted_not_warming_uses_full_w_target(make_input: RampInputFactory) -> None:
    """With the pipeline warm, a trusted stage is sized to its full w_target in one cycle (unchanged behavior)."""
    decision = decide(
        make_input(sample_count=5, current_workers=2, proposed_post=57, w_target=57, pipeline_warming=False),
        _config(),
    )
    assert decision.cap == 57
    assert decision.reason is RampReason.CAPPED


def test_trusted_warming_never_forces_shrink(make_input: RampInputFactory) -> None:
    """A warmup bound below the current size trims growth to zero, never deletes existing workers."""
    decision = decide(
        make_input(
            sample_count=5, current_workers=10, deleted_count=0, proposed_post=10, w_target=2, pipeline_warming=True
        ),
        _config(),
    )
    assert decision.cap == 2
    assert decision.keep_new == 0
    assert decision.reason is RampReason.CAPPED_WARMUP


def test_trusted_warming_without_target_is_uncapped(make_input: RampInputFactory) -> None:
    """The warmup gate has no real target to bound, so a trusted stage with no w_target stays uncapped."""
    decision = decide(
        make_input(sample_count=5, current_workers=1, proposed_post=11, w_target=None, pipeline_warming=True),
        _config(),
    )
    assert decision.cap is None
    assert decision.keep_new is None
    assert decision.reason is RampReason.UNCAPPED


def test_ramp_input_does_not_carry_resource_shape(make_input: RampInputFactory) -> None:
    """One generic policy: the ramp depends on SAT signals, never on a stage's resource shape."""
    ramp_input = make_input()
    assert not hasattr(ramp_input, "gpu_fraction")
    assert not hasattr(ramp_input, "is_gpu")


def test_proposal_within_cap_is_not_trimmed(make_input: RampInputFactory) -> None:
    """A solver proposal already at or below the cap keeps all its new workers."""
    decision = decide(make_input(sample_count=0, current_workers=0, proposed_post=1), _config())
    assert decision.cap == 1
    assert decision.keep_new is None


def test_shrink_is_never_turned_into_growth(make_input: RampInputFactory) -> None:
    """A solver-proposed shrink is left untouched; the ramp never adds workers."""
    decision = decide(make_input(sample_count=1, current_workers=5, deleted_count=2, proposed_post=3), _config())
    assert decision.keep_new is None
