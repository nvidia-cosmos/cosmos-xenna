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


"""Classifier-signal-noise EWMA update contract.

The donor signal-trust gate divides ``min(classifier_streak, trust_streak_cap)``
by ``1 + classifier_signal_noise_ewma``. A misbehaving noise tracker
either suppresses legitimate donations (false high noise) or admits
flickering classifiers (false low noise). These tests pin the update
contract from the plan in isolation:

*   Cold-start: the first usable delta seeds the EWMA via ``update_ewma``'s
    None-as-seed semantics; no synthetic warmup value.
*   Skip-on-carry-forward: cycles that re-use the prior EWMA value do not
    contribute artificial zero deltas.
*   Skip-on-cold-start (no fresh sample): cycles with zero slots and no
    prior EWMA leave the noise tracker untouched.
*   Preservation across classifier transitions: noise EWMA carries
    history when the classifier oscillates in and out of OVER_PROVISIONED.
*   Param-None gate: helper-direct fixtures that omit
    ``signal_noise_smoothing_level`` keep the field at its cold-start
    sentinel so unrelated tests stay deterministic.
"""

import math

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stage_decision_pipeline import StageDecisionPipeline
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    ClassifierState,
    StageRuntimeState,
    StageState,
)
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import _resolve_auto_thresholds
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Minimal classifier-and-pressure config for noise-tracker tests."""
    return SaturationAwareStageConfig(
        saturation_threshold=0.15,
        activation_threshold=0.05,
        over_provisioned_threshold=0.50,
        target_backlog_seconds=30.0,
        pressure_smoothing_level=0.20,
        pressure_critical_threshold=2.0,
        pressure_saturation_threshold=1.0,
        pressure_normal_threshold=0.3,
        slots_empty_ratio_smoothing_level=1.0,  # raw sample == EWMA; isolates the noise test
    )


def _fresh_state(cfg: SaturationAwareStageConfig, name: str = "TestStage") -> StageRuntimeState:
    """Per-stage state with thresholds pre-resolved (mirrors the pressure suite)."""
    resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
    return StageRuntimeState(stage_name=name, classifier=ClassifierState(resolved_thresholds=resolved))


class TestColdStart:
    """First fresh sample produces no delta; the EWMA stays at its cold-start sentinel."""

    def test_first_fresh_sample_leaves_noise_ewma_none(self, cfg: SaturationAwareStageConfig) -> None:
        """No prev EWMA means no delta; noise tracker MUST stay at ``None``."""
        state = _fresh_state(cfg)
        assert state.classifier.signal_noise_ewma is None

        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=6,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )

        assert state.classifier.signal_noise_ewma is None
        assert state.classifier.slots_empty_ratio_ewma is not None


class TestSeeding:
    """Second fresh sample seeds the EWMA with the raw delta (no alpha-blending)."""

    def test_second_fresh_sample_seeds_ewma_with_raw_delta(self, cfg: SaturationAwareStageConfig) -> None:
        """``update_ewma(None, delta, alpha)`` returns ``delta`` -- pin that contract here."""
        state = _fresh_state(cfg)

        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=6,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        first_ewma = state.classifier.slots_empty_ratio_ewma
        assert first_ewma is not None

        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=6,
            num_empty_slots=2,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        second_ewma = state.classifier.slots_empty_ratio_ewma
        assert second_ewma is not None

        expected_delta = abs(second_ewma - first_ewma)
        assert state.classifier.signal_noise_ewma is not None
        assert math.isclose(state.classifier.signal_noise_ewma, expected_delta, rel_tol=1e-12)


class TestSmoothing:
    """Third sample blends the new delta with the seed using the configured alpha."""

    def test_third_fresh_sample_applies_alpha_blend(self, cfg: SaturationAwareStageConfig) -> None:
        """``new = alpha * delta + (1 - alpha) * prev_noise``; pin the smoothing math."""
        state = _fresh_state(cfg)
        alpha = 0.20

        StageDecisionPipeline(signal_noise_smoothing_level=alpha).compute_recommendation(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=6,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        e0 = state.classifier.slots_empty_ratio_ewma
        assert e0 is not None

        StageDecisionPipeline(signal_noise_smoothing_level=alpha).compute_recommendation(
            stage_state=state,
            num_used_slots=6,
            num_empty_slots=2,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        e1 = state.classifier.slots_empty_ratio_ewma
        noise_after_seed = state.classifier.signal_noise_ewma
        assert e1 is not None
        assert noise_after_seed is not None

        StageDecisionPipeline(signal_noise_smoothing_level=alpha).compute_recommendation(
            stage_state=state,
            num_used_slots=4,
            num_empty_slots=4,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        e2 = state.classifier.slots_empty_ratio_ewma
        assert e2 is not None

        expected = alpha * abs(e2 - e1) + (1.0 - alpha) * noise_after_seed
        assert state.classifier.signal_noise_ewma is not None
        assert math.isclose(state.classifier.signal_noise_ewma, expected, rel_tol=1e-12)


class TestCarryForwardSkipsUpdate:
    """A zero-actor carry-forward cycle re-uses the prior EWMA and MUST not update noise."""

    def test_zero_slots_with_carry_forward_does_not_update_noise(self, cfg: SaturationAwareStageConfig) -> None:
        """Carry-forward path is a no-new-sample cycle; the noise tracker must hold its value."""
        state = _fresh_state(cfg)

        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=6,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=6,
            num_empty_slots=2,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        seeded_noise = state.classifier.signal_noise_ewma
        assert seeded_noise is not None

        # Carry-forward cycle: zero slots but prior valid EWMA present.
        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=0,
            config=cfg,
        )

        assert state.classifier.signal_noise_ewma == seeded_noise


class TestColdStartZeroSlotsSkipsUpdate:
    """A zero-actor cold-start cycle returns ``None`` from the classifier and MUST not update noise."""

    def test_zero_slots_no_prior_signal_does_not_update_noise(self, cfg: SaturationAwareStageConfig) -> None:
        """Cold-start + zero slots means no fresh sample at all; noise tracker MUST stay None."""
        state = _fresh_state(cfg)
        assert state.classifier.signal_noise_ewma is None
        assert state.classifier.slots_empty_ratio_ewma is None

        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=0,
            config=cfg,
        )

        assert state.classifier.signal_noise_ewma is None
        assert state.classifier.slots_empty_ratio_ewma is None


class TestPreservationAcrossClassifierTransitions:
    """Noise EWMA is preserved when the classifier moves OUT and back IN of OVER_PROVISIONED."""

    def test_noise_ewma_carries_history_across_oop_re_entry(self, cfg: SaturationAwareStageConfig) -> None:
        """Classifier transitions MUST NOT reset ``classifier_signal_noise_ewma``."""
        state = _fresh_state(cfg)

        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=8,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=1,
            num_empty_slots=7,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        seeded_noise = state.classifier.signal_noise_ewma
        assert seeded_noise is not None

        # Transition OUT of OVER_PROVISIONED (raw ratio = 0/8 = 0 -> SATURATED_CRITICAL).
        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=8,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        assert state.classifier.state is not StageState.OVER_PROVISIONED

        # Transition back INTO OVER_PROVISIONED (raw ratio = 6/8 = 0.75).
        StageDecisionPipeline(signal_noise_smoothing_level=0.20).compute_recommendation(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=6,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )

        # Noise EWMA must be strictly different from its seeded value (two
        # extra deltas blended in) but MUST NOT be reset to None or to the
        # raw seed; the streak gate is the only across-cycle reset.
        assert state.classifier.signal_noise_ewma is not None
        assert state.classifier.signal_noise_ewma != seeded_noise


class TestParamGate:
    """Omitting ``signal_noise_smoothing_level`` leaves the field at its cold-start sentinel."""

    def test_helper_direct_call_without_smoothing_level_skips_update(self, cfg: SaturationAwareStageConfig) -> None:
        """Existing helper-direct fixtures that do not pass the kwarg MUST NOT see the field mutate."""
        state = _fresh_state(cfg)

        StageDecisionPipeline().compute_recommendation(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=6,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        StageDecisionPipeline().compute_recommendation(
            stage_state=state,
            num_used_slots=6,
            num_empty_slots=2,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )

        assert state.classifier.signal_noise_ewma is None
