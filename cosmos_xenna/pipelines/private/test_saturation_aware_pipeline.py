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

"""Integration tests for the per-stage decision pipeline.

The seven primitives (``compute_slots_empty_ratio``, ``update_ewma``,
``classify``, ``update_streak``, ``should_fire_action``,
``compute_delta``, ``compute_growth_mode_transition``) each have
dedicated unit tests covering every branch. These tests verify that
``run_per_stage_pipeline`` + ``record_executed_delta`` compose them
correctly: the right primitive fires in the right order, state
mutations are consistent, the zero-actors edge cases route through
the carry-forward logic, and multi-cycle traces produce the expected
sequence of decisions.

The two-step API mirrors the scheduler's behavior at runtime:
``run_per_stage_pipeline`` produces a recommendation, and
``record_executed_delta`` is called with the post-Phase-C/D executed
delta so the growth-mode state machine observes the committed result.
Each test that asserts on ``growth_mode`` / ``growth_streak`` uses
the ``_advance_cycle`` helper, which models the scheduler executing
exactly the recommendation.
"""

import attrs
import pytest

from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import _resolve_auto_thresholds
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import (
    record_executed_delta,
    run_per_stage_pipeline,
)
from cosmos_xenna.pipelines.private.scheduling_py.stabilization import _RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def _advance_cycle(
    stage_state: _StageRuntimeState,
    *,
    num_used_slots: int,
    num_empty_slots: int,
    input_queue_depth: int,
    current_workers: int,
    config: SaturationAwareStageConfig,
    recommendation_history: _RecommendationHistory | None = None,
) -> int:
    """Test helper: run the recommendation, then record the executed delta.

    Mirrors the scheduler's call pattern (``run_per_stage_pipeline``
    in ``_compute_intent_deltas`` plus ``record_executed_delta`` in
    ``_record_post_commit_executed_deltas``) for tests that assert on
    the growth-mode state machine. Tests that only inspect the
    returned delta or other recommendation-side state continue to
    call ``run_per_stage_pipeline`` directly.
    """
    delta = run_per_stage_pipeline(
        stage_state=stage_state,
        num_used_slots=num_used_slots,
        num_empty_slots=num_empty_slots,
        input_queue_depth=input_queue_depth,
        current_workers=current_workers,
        config=config,
        recommendation_history=recommendation_history,
    )
    record_executed_delta(stage_state=stage_state, delta_executed=delta, config=config)
    return delta


@pytest.fixture
def cfg() -> SaturationAwareStageConfig:
    """Per-stage config with explicit threshold overrides anchoring test math."""
    return SaturationAwareStageConfig(saturation_threshold=0.15, activation_threshold=0.05)


def _fresh_state(cfg: SaturationAwareStageConfig, name: str = "TestStage") -> _StageRuntimeState:
    """Build a runtime state with classifier thresholds pre-resolved.

    Production resolves thresholds on the first ``autoscale()`` cycle;
    tests build the state directly so they must populate
    ``resolved_thresholds`` themselves before invoking
    ``run_per_stage_pipeline``. The fixture pins explicit overrides
    so the resolved pair is always 0.15 / 0.05 regardless of
    ``slots_per_actor``.
    """
    resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
    return _StageRuntimeState(stage_name=name, resolved_thresholds=resolved)


class TestThresholdLifecycleGuard:
    """The pipeline refuses direct use before threshold resolution."""

    def test_unresolved_thresholds_raise_contextual_runtime_error(self, cfg: SaturationAwareStageConfig) -> None:
        """Direct callers must populate ``resolved_thresholds`` before running the pipeline."""
        state = _StageRuntimeState(stage_name="TestStage")

        with pytest.raises(RuntimeError, match="TestStage.*has no resolved_thresholds"):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=3,
                num_empty_slots=2,
                input_queue_depth=10,
                current_workers=4,
                config=cfg,
            )


class TestNoActionPath:
    """NORMAL signal -> the pipeline fires no action and updates only the EWMA + streak."""

    def test_normal_signal_returns_zero_and_advances_state(self, cfg: SaturationAwareStageConfig) -> None:
        """Mid-band ratio (strictly between sat=0.15 and op=0.50) -> NORMAL -> delta=0."""
        # ratio = 2 / (3 + 2) = 0.4 -- in the NORMAL band (sat 0.15 < 0.4 < op 0.50).
        state = _fresh_state(cfg)
        delta = _advance_cycle(
            state,
            num_used_slots=3,
            num_empty_slots=2,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        assert delta == 0
        assert state.classifier_state is StageState.NORMAL
        assert state.classifier_streak == 1
        assert state.slots_empty_ratio_ewma == pytest.approx(0.4)
        assert state.last_valid_slots_empty_ratio_ewma == pytest.approx(0.4)
        assert state.growth_mode is GrowthMode.ACQUIRING
        assert state.growth_streak == 1
        assert state.prev_workers == 4


class TestCriticalScaleUpPath:
    """SATURATED_CRITICAL signal -> burst scale-up fires on the very first cycle."""

    def test_zero_ratio_triggers_critical_acquiring_growth(self, cfg: SaturationAwareStageConfig) -> None:
        """All slots full + queue pending -> CRITICAL -> fires immediately at streak=1.

        ACQUIRING + CRITICAL: ceil(0.5 * 4) = 2; cap=4 doesn't bite.
        """
        state = _fresh_state(cfg)
        delta = _advance_cycle(
            state,
            num_used_slots=4,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        assert delta == 2
        assert state.classifier_state is StageState.SATURATED_CRITICAL
        # Growth fires; growth_mode stays in ACQUIRING (no shrink event).
        assert state.growth_mode is GrowthMode.ACQUIRING


class TestSaturatedScaleUpPath:
    """SATURATED signal needs streak >= 2 (default) before the action fires."""

    def test_first_cycle_saturated_does_not_fire(self, cfg: SaturationAwareStageConfig) -> None:
        """SATURATED at streak=1 < min=2 -> fire-gate fails -> delta=0."""
        state = _fresh_state(cfg)
        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=10,
            num_empty_slots=1,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        assert delta == 0
        assert state.classifier_state is StageState.SATURATED
        assert state.classifier_streak == 1

    def test_second_cycle_saturated_fires_growth(self, cfg: SaturationAwareStageConfig) -> None:
        """Two consecutive SATURATED cycles -> streak=2 -> fire.

        ACQUIRING + SATURATED: ceil(0.25 * 4) = 1.
        """
        state = _fresh_state(cfg)
        run_per_stage_pipeline(  # cycle 1: streak goes to 1
            stage_state=state,
            num_used_slots=10,
            num_empty_slots=1,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        delta = run_per_stage_pipeline(  # cycle 2: streak goes to 2 -> fire
            stage_state=state,
            num_used_slots=10,
            num_empty_slots=1,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        assert delta == 1
        assert state.classifier_state is StageState.SATURATED
        assert state.classifier_streak == 2


class TestOverProvisionedShrinkPath:
    """OVER_PROVISIONED needs streak >= 30 (default) before shrink fires."""

    def test_shrink_fires_after_window_and_transitions_mode(self, cfg: SaturationAwareStageConfig) -> None:
        """30 consecutive OVER_PROVISIONED cycles -> shrink fires -> ACQUIRING flips to TRACKING."""
        state = _fresh_state(cfg)
        # Walk through 29 OVER_PROVISIONED cycles -> no fire yet.
        for _ in range(29):
            delta = _advance_cycle(
                state,
                num_used_slots=1,
                num_empty_slots=9,
                input_queue_depth=10,
                current_workers=10,
                config=cfg,
            )
            assert delta == 0
        # 30th cycle: streak hits threshold, shrink fires.
        delta = _advance_cycle(
            state,
            num_used_slots=1,
            num_empty_slots=9,
            input_queue_depth=10,
            current_workers=10,
            config=cfg,
        )
        assert state.classifier_streak == 30
        # 10 workers, fraction=0.05 -> floor(0.5)=0, max(1, 0)=1 -> -1.
        assert delta == -1
        # ACQUIRING + first shrink -> TRACKING.
        assert state.growth_mode is GrowthMode.TRACKING
        assert state.growth_streak == 1


class TestEmptyQueueWithIdleSlotsClassifiesAsOverProvisioned:
    """Free slots + drained queue -> OVER_PROVISIONED.

    Replaces the previous queue==0 -> STARVED short-circuit. The first
    cycle does not fire delta because the OVER_PROVISIONED streak
    threshold has not yet been reached.
    """

    def test_empty_queue_with_idle_slots_classifies_as_over_provisioned(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Free slots + empty queue -> OVER_PROVISIONED; first cycle delta is zero."""
        state = _fresh_state(cfg)
        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=1,
            num_empty_slots=9,
            input_queue_depth=0,
            current_workers=10,
            config=cfg,
        )
        assert delta == 0
        assert state.classifier_state is StageState.OVER_PROVISIONED


class TestZeroActorsColdStart:
    """Zero ready actors + no prior signal -> classifier is held; pipeline returns 0."""

    def test_cold_start_with_zero_actors_returns_zero(self, cfg: SaturationAwareStageConfig) -> None:
        """First cycle with zero actors -> no signal -> hold; the worker-floor step bootstraps elsewhere."""
        state = _fresh_state(cfg)
        delta = _advance_cycle(
            state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=0,
            config=cfg,
        )
        assert delta == 0
        # Classifier untouched; EWMA still uninitialised.
        assert state.classifier_state is StageState.NORMAL
        assert state.classifier_streak == 0
        assert state.slots_empty_ratio_ewma is None
        assert state.last_valid_slots_empty_ratio_ewma is None
        # Growth-mode timer ticks regardless (the scheduler always calls
        # ``record_executed_delta`` for every stage that participated in
        # the cycle, including cold-start no-action stages).
        assert state.growth_mode is GrowthMode.ACQUIRING
        assert state.growth_streak == 1


class TestZeroActorsCarryForward:
    """Zero ready actors + prior valid EWMA -> classifier reads the carry-forward value."""

    def test_carry_forward_preserves_classifier_state_across_zero_actor_moment(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Saturated, then zero actors transient -> classifier still sees SATURATED via carry-forward."""
        state = _fresh_state(cfg)
        run_per_stage_pipeline(  # cycle 1: SATURATED, EWMA caches a low value.
            stage_state=state,
            num_used_slots=10,
            num_empty_slots=1,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        cached_ewma = state.slots_empty_ratio_ewma
        assert cached_ewma is not None
        assert cfg.saturation_threshold is not None
        assert cached_ewma < cfg.saturation_threshold

        run_per_stage_pipeline(  # cycle 2: zero actors momentarily.
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=0,
            config=cfg,
        )
        # EWMA itself is unchanged -- carry-forward, no new sample blended in.
        assert state.slots_empty_ratio_ewma == cached_ewma
        # Classifier must have re-classified using the cached value.
        assert state.classifier_state is StageState.SATURATED


class TestEwmaSmoothing:
    """Repeated samples converge the EWMA monotonically toward the steady-state value."""

    def test_ewma_converges_toward_steady_state(self, cfg: SaturationAwareStageConfig) -> None:
        """Same raw ratio fed every cycle -> EWMA monotonically approaches it."""
        state = _fresh_state(cfg)
        target = 0.50
        for _ in range(20):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=1,
                num_empty_slots=1,
                input_queue_depth=10,
                current_workers=4,
                config=cfg,
            )
        assert state.slots_empty_ratio_ewma == pytest.approx(target, rel=1e-2)


class TestFullLifecycleTrace:
    """Multi-cycle simulation pinning the end-to-end pipeline composition.

    The TRACKING -> HOLD transition is exhaustively covered in isolation
    by the ``compute_growth_mode_transition`` unit tests; here we focus
    on the integration path where the EWMA, classifier, streak, and
    growth-mode all advance together for the first ceiling discovery.
    """

    def test_acquiring_grows_then_first_shrink_transitions_to_tracking(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """Cold-start a stage in OVER_PROVISIONED for the streak window -> first shrink -> TRACKING.

        On a fresh state, the cold-start EWMA path means the very
        first OVER_PROVISIONED sample is taken at face value (no
        warmup tax), so 30 consecutive cycles cleanly accumulate the
        streak and fire on the 30th.
        """
        state = _fresh_state(cfg)
        # 29 cycles of OVER_PROVISIONED with streak < 30 -- no shrink yet.
        for _ in range(29):
            delta = _advance_cycle(
                state,
                num_used_slots=1,
                num_empty_slots=9,
                input_queue_depth=10,
                current_workers=10,
                config=cfg,
            )
            assert delta == 0
            assert state.classifier_state is StageState.OVER_PROVISIONED

        # 30th cycle: streak hits threshold, shrink fires, mode flips.
        delta = _advance_cycle(
            state,
            num_used_slots=1,
            num_empty_slots=9,
            input_queue_depth=10,
            current_workers=10,
            config=cfg,
        )
        assert state.classifier_streak == 30
        assert delta == -1
        # ACQUIRING + first shrink -> TRACKING (ceiling discovered).
        assert state.growth_mode is GrowthMode.TRACKING
        assert state.growth_streak == 1


class TestColdStartZeroActorsDoesNotResetGrowthMode:
    """Cold-start path must still tick the growth-mode timer so the HOLD->TRACKING exit fires."""

    def test_zero_actor_cycles_in_hold_eventually_exit_to_tracking(
        self,
        cfg: SaturationAwareStageConfig,
    ) -> None:
        """A stage parked in HOLD with zero actors must still exit to TRACKING after the window expires."""
        # Construct a state already in HOLD via attrs.evolve to skip the lifecycle.
        state = attrs.evolve(
            _fresh_state(cfg),
            growth_mode=GrowthMode.HOLD,
            growth_streak=cfg.stabilization_window_cycles_down,
            last_valid_slots_empty_ratio_ewma=None,
        )
        # Zero actors + no prior EWMA: cold-start short-circuit, but the
        # growth-mode timer must still tick. The scheduler models this by
        # calling ``record_executed_delta(0)`` for every stage that
        # participated in the cycle, including cold-start no-action stages.
        delta = _advance_cycle(
            state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=0,
            config=cfg,
        )
        assert delta == 0
        assert state.growth_mode is GrowthMode.TRACKING
        assert state.growth_streak == 1
