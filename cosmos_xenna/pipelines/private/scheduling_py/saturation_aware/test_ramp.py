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
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.ramp import ColdStartRampPolicy, StageRampInput

type RampInputFactory = Callable[..., StageRampInput]


@pytest.fixture
def make_input() -> RampInputFactory:
    """Return a factory for ramp inputs defaulting to a cold fractional-GPU stage."""

    def _make_input(
        *,
        name: str = "s",
        current_workers: int = 0,
        deleted_count: int = 0,
        proposed_post: int = 10,
        sample_count: int = 0,
        gpu_fraction: float = 0.25,
    ) -> StageRampInput:
        return StageRampInput(
            name=name,
            current_workers=current_workers,
            deleted_count=deleted_count,
            proposed_post=proposed_post,
            sample_count=sample_count,
            gpu_fraction=gpu_fraction,
        )

    return _make_input


def _policy() -> ColdStartRampPolicy:
    """Build a policy with the default trust threshold of 5 samples."""
    return ColdStartRampPolicy(SaturationAwareConfig())


def test_cold_start_with_no_samples_caps_at_one(make_input: RampInputFactory) -> None:
    """A fractional-GPU stage with no completed sample is capped at one worker."""
    decision = _policy().decide(make_input(sample_count=0, current_workers=0, proposed_post=11))
    assert decision.cap == 1
    assert decision.keep_new == 1
    assert decision.reason == "cold"


def test_no_samples_running_stage_trims_all_new_workers(make_input: RampInputFactory) -> None:
    """With one worker already up but no sample yet, no further workers are added."""
    decision = _policy().decide(make_input(sample_count=0, current_workers=1, proposed_post=11))
    assert decision.cap == 1
    assert decision.keep_new == 0


def test_warming_thin_evidence_allows_small_step(make_input: RampInputFactory) -> None:
    """One sample of five unlocks ceil(0.2 * solver_growth) workers."""
    decision = _policy().decide(make_input(sample_count=1, current_workers=1, proposed_post=11))
    assert decision.cap == 3
    assert decision.keep_new == 2
    assert decision.reason == "warming"


def test_warming_more_evidence_allows_larger_step(make_input: RampInputFactory) -> None:
    """More samples unlock a larger fraction of the solver's requested growth."""
    decision = _policy().decide(make_input(sample_count=4, current_workers=1, proposed_post=11))
    assert decision.cap == 9
    assert decision.keep_new == 8


def test_warming_allows_at_least_one_more_worker(make_input: RampInputFactory) -> None:
    """A warming stage may always grow by at least one worker."""
    decision = _policy().decide(make_input(sample_count=1, current_workers=1, proposed_post=2))
    assert decision.cap == 2
    assert decision.keep_new is None


def test_trusted_stage_is_uncapped(make_input: RampInputFactory) -> None:
    """Once samples reach the trust threshold the stage is uncapped."""
    decision = _policy().decide(make_input(sample_count=5, current_workers=1, proposed_post=11))
    assert decision.cap is None
    assert decision.keep_new is None
    assert decision.reason == "uncapped"


def test_whole_gpu_stage_is_exempt(make_input: RampInputFactory) -> None:
    """A whole-GPU stage is never capped, even with no samples."""
    decision = _policy().decide(make_input(sample_count=0, gpu_fraction=1.0, proposed_post=10))
    assert decision.cap is None
    assert decision.reason == "uncapped"


def test_cpu_stage_is_exempt(make_input: RampInputFactory) -> None:
    """A CPU stage (zero GPU fraction) is never capped."""
    decision = _policy().decide(make_input(sample_count=0, gpu_fraction=0.0, proposed_post=10))
    assert decision.cap is None


def test_proposal_within_cap_is_not_trimmed(make_input: RampInputFactory) -> None:
    """A solver proposal already at or below the cap keeps all its new workers."""
    decision = _policy().decide(make_input(sample_count=0, current_workers=0, proposed_post=1))
    assert decision.cap == 1
    assert decision.keep_new is None


def test_shrink_is_never_turned_into_growth(make_input: RampInputFactory) -> None:
    """A solver-proposed shrink is left untouched; the ramp never adds workers."""
    decision = _policy().decide(make_input(sample_count=1, current_workers=5, deleted_count=2, proposed_post=3))
    assert decision.keep_new is None
