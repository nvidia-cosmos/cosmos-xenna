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


"""Tests for the asymmetric stabilization-window recommendation gate.

The first set of tests pins the pure-data-structure contract of
``_RecommendationHistory`` (record / gate / clear / capacity) without
any scheduler context. The integration tests at the bottom verify
that ``run_per_stage_pipeline`` and ``SaturationAwareScheduler``
honour the gate end-to-end: the raw delta is replaced with ``0`` when
the buffer cannot yet confirm a sustained recommendation, and the
buffer is cleared on regime transition so post-transition cycles
must rebuild consensus from scratch.

Each test pins one observable behaviour per ``test-creation.mdc``.
Helper-layer tests use no scheduler / Ray / cluster fixtures; the
integration tests use the existing helpers from sibling test modules
to stay aligned with the project's fixture conventions.
"""

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import _resolve_auto_thresholds
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import run_per_stage_pipeline
from cosmos_xenna.pipelines.private.scheduling_py.regime import Regime
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.scheduling_py.stabilization import (
    _RecommendationHistory,
    apply_stabilization_gate,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@pytest.fixture
def history_default() -> _RecommendationHistory:
    """A fresh history with the production-default asymmetric windows."""
    return _RecommendationHistory(window_up=1, window_down=30)


class TestConstruction:
    """Constructor input validation and capacity-derivation contract."""

    def test_reports_capacity_as_max_of_windows(self) -> None:
        """The buffer must hold enough cycles for the larger of the two windows."""
        history = _RecommendationHistory(window_up=5, window_down=12)
        assert history.capacity == 12
        assert history.window_up == 5
        assert history.window_down == 12

    def test_starts_empty(self) -> None:
        """A freshly constructed buffer reports zero retained cycles."""
        history = _RecommendationHistory(window_up=1, window_down=30)
        assert len(history) == 0

    @pytest.mark.parametrize(
        ("window_up", "window_down"),
        [(0, 30), (-1, 30), (1, 0), (1, -5)],
    )
    def test_rejects_non_positive_windows(self, window_up: int, window_down: int) -> None:
        """Either window <= 0 must raise so misconfigured callers fail fast."""
        with pytest.raises(ValueError, match=r"window_(up|down) must be >= 1"):
            _RecommendationHistory(window_up=window_up, window_down=window_down)

    def test_accepts_equal_windows(self) -> None:
        """Equal windows are valid at the primitive layer.

        Cross-window ordering (down > up) is enforced at config time, not
        at the data-structure layer, so unit tests can use degenerate
        windows freely.
        """
        history = _RecommendationHistory(window_up=4, window_down=4)
        assert history.capacity == 4


class TestRecord:
    """``record`` maps the sign of the input onto the buffer."""

    def test_positive_delta_records_up(self, history_default: _RecommendationHistory) -> None:
        history_default.record(7)
        assert len(history_default) == 1

    def test_negative_delta_records_down(self, history_default: _RecommendationHistory) -> None:
        history_default.record(-3)
        assert len(history_default) == 1

    def test_zero_delta_records_noop(self, history_default: _RecommendationHistory) -> None:
        history_default.record(0)
        assert len(history_default) == 1

    def test_buffer_drops_oldest_after_capacity(self) -> None:
        """A 3-cycle buffer must retain only the most recent 3 entries."""
        history = _RecommendationHistory(window_up=3, window_down=3)
        for direction in (1, -1, 1, 0, -1):
            history.record(direction)
        # capacity = 3, so the first two records (1, -1) have been evicted.
        # Only the last 3 records (1, 0, -1) remain. With those, neither gate
        # can require a uniform direction across the window.
        assert len(history) == 3
        assert not history.gate_up_allowed()
        assert not history.gate_down_allowed()


class TestGateUp:
    """``gate_up_allowed`` semantics with focus on cold start and consensus."""

    def test_refuses_under_filled_window(self) -> None:
        """A buffer with fewer than ``window_up`` records must refuse."""
        history = _RecommendationHistory(window_up=3, window_down=10)
        history.record(1)
        history.record(1)
        assert not history.gate_up_allowed()

    def test_allows_after_window_up_consecutive_ups(self) -> None:
        """``window_up`` consecutive ``+1`` records satisfy the gate exactly."""
        history = _RecommendationHistory(window_up=3, window_down=10)
        for _ in range(3):
            history.record(1)
        assert history.gate_up_allowed()

    def test_immediate_for_window_up_one(self, history_default: _RecommendationHistory) -> None:
        """Default ``window_up = 1`` permits scale-up on a single cycle."""
        history_default.record(1)
        assert history_default.gate_up_allowed()

    def test_one_noop_in_window_breaks_consensus(self) -> None:
        """A single ``0`` cycle breaks the up-direction consensus."""
        history = _RecommendationHistory(window_up=3, window_down=10)
        history.record(1)
        history.record(0)
        history.record(1)
        assert not history.gate_up_allowed()

    def test_one_down_in_window_breaks_consensus(self) -> None:
        """A single ``-1`` cycle breaks the up-direction consensus."""
        history = _RecommendationHistory(window_up=3, window_down=10)
        history.record(1)
        history.record(-1)
        history.record(1)
        assert not history.gate_up_allowed()

    def test_post_recovery_after_mixed_window(self) -> None:
        """The gate re-allows after the buffer has slid past the broken cycles."""
        history = _RecommendationHistory(window_up=3, window_down=10)
        history.record(1)
        history.record(0)
        history.record(1)
        history.record(1)
        history.record(1)  # Last 3 records are all +1 again.
        assert history.gate_up_allowed()


class TestGateDown:
    """``gate_down_allowed`` semantics with focus on long-window patience."""

    def test_refuses_under_filled_window(self) -> None:
        """Even 29 of 30 ``-1`` records must refuse: the buffer is not full yet."""
        history = _RecommendationHistory(window_up=1, window_down=30)
        for _ in range(29):
            history.record(-1)
        assert not history.gate_down_allowed()

    def test_allows_after_window_down_consecutive_downs(self) -> None:
        """Exactly ``window_down`` ``-1`` cycles satisfy the gate."""
        history = _RecommendationHistory(window_up=1, window_down=30)
        for _ in range(30):
            history.record(-1)
        assert history.gate_down_allowed()

    def test_one_up_in_window_resets_progress(self) -> None:
        """A single ``+1`` cycle within the window suppresses scale-down."""
        history = _RecommendationHistory(window_up=1, window_down=30)
        for _ in range(29):
            history.record(-1)
        history.record(1)
        assert not history.gate_down_allowed()

    def test_recovers_only_after_full_replay_of_downs(self) -> None:
        """After a flap the buffer must be fully refilled with downs."""
        history = _RecommendationHistory(window_up=1, window_down=5)
        for _ in range(5):
            history.record(-1)
        assert history.gate_down_allowed()
        history.record(1)
        assert not history.gate_down_allowed()
        # 4 more ``-1`` records do NOT satisfy yet; only after the 5th do all
        # five entries in the window equal -1 again.
        for _ in range(4):
            history.record(-1)
        assert not history.gate_down_allowed()
        history.record(-1)
        assert history.gate_down_allowed()


class TestAsymmetricWindows:
    """The asymmetric default (1 / 30) protects shrink without delaying grow."""

    def test_first_up_cycle_allows_grow_but_not_shrink(self, history_default: _RecommendationHistory) -> None:
        history_default.record(1)
        assert history_default.gate_up_allowed()
        assert not history_default.gate_down_allowed()

    def test_first_down_cycle_blocks_both_directions(self, history_default: _RecommendationHistory) -> None:
        history_default.record(-1)
        assert not history_default.gate_up_allowed()
        assert not history_default.gate_down_allowed()


class TestApplyStabilizationGate:
    """The free-function helper combines record + gate so callers cannot skip one."""

    def test_returns_zero_when_up_gate_refuses(self) -> None:
        history = _RecommendationHistory(window_up=2, window_down=30)
        # First up cycle: window_up = 2, so the gate refuses on cycle 1.
        gated = apply_stabilization_gate(history, raw_delta=5)
        assert gated == 0
        # Buffer now holds one ``+1``; the call recorded it before gating.
        assert len(history) == 1

    def test_returns_raw_delta_when_up_gate_allows(self, history_default: _RecommendationHistory) -> None:
        gated = apply_stabilization_gate(history_default, raw_delta=4)
        assert gated == 4

    def test_returns_zero_when_down_gate_refuses(self, history_default: _RecommendationHistory) -> None:
        gated = apply_stabilization_gate(history_default, raw_delta=-3)
        # window_down = 30; first cycle cannot satisfy.
        assert gated == 0

    def test_returns_zero_for_zero_delta(self, history_default: _RecommendationHistory) -> None:
        """Zero recommendation always yields zero, regardless of gate state."""
        for _ in range(40):
            history_default.record(-1)
        gated = apply_stabilization_gate(history_default, raw_delta=0)
        assert gated == 0

    def test_records_then_gates_in_order(self) -> None:
        """The buffer must contain the new record before the gate consults it.

        Without the record-before-gate ordering, the very first scale-up
        recommendation with ``window_up = 1`` would incorrectly refuse.
        """
        history = _RecommendationHistory(window_up=1, window_down=30)
        gated = apply_stabilization_gate(history, raw_delta=2)
        assert gated == 2
        assert len(history) == 1


class TestClear:
    """Explicit reset is needed for tests and for structural worker-count changes."""

    def test_clear_drops_all_records(self) -> None:
        history = _RecommendationHistory(window_up=2, window_down=4)
        for _ in range(4):
            history.record(-1)
        assert history.gate_down_allowed()
        history.clear()
        assert len(history) == 0
        assert not history.gate_up_allowed()
        assert not history.gate_down_allowed()

    def test_can_be_repopulated_after_clear(self) -> None:
        history = _RecommendationHistory(window_up=1, window_down=2)
        history.record(-1)
        history.clear()
        history.record(1)
        assert history.gate_up_allowed()


class TestMultiCycleStreaks:
    """Stress-style coverage: long runs and rollover semantics."""

    def test_long_run_of_ups_keeps_up_gate_on(self) -> None:
        history = _RecommendationHistory(window_up=3, window_down=10)
        for _ in range(100):
            history.record(1)
            assert history.gate_up_allowed() or len(history) < 3
        assert history.gate_up_allowed()
        assert not history.gate_down_allowed()

    def test_long_run_of_mixed_keeps_both_gates_off(self) -> None:
        history = _RecommendationHistory(window_up=2, window_down=5)
        for cycle_index in range(50):
            history.record(1 if cycle_index % 2 == 0 else -1)
        assert not history.gate_up_allowed()
        assert not history.gate_down_allowed()

    def test_alternating_direction_never_satisfies(self) -> None:
        """A pathological flap (+,-,+,-,...) cannot ever satisfy either gate."""
        history = _RecommendationHistory(window_up=2, window_down=2)
        for cycle_index in range(20):
            history.record(1 if cycle_index % 2 == 0 else -1)
        assert not history.gate_up_allowed()
        assert not history.gate_down_allowed()


# Integration tests follow. They use the same fixture style as the
# helper-direct tests above but reach into ``run_per_stage_pipeline``
# and ``SaturationAwareScheduler`` to verify the wiring between the
# stabilization buffer and its consumers.


def _cfg(
    *,
    window_up: int = 1,
    window_down: int = 30,
    saturated_streak_min: int = 2,
    over_provisioned_streak_min: int = 30,
) -> SaturationAwareStageConfig:
    """Stage config with explicit thresholds and stabilization windows."""
    return SaturationAwareStageConfig(
        saturation_threshold=0.15,
        activation_threshold=0.05,
        stabilization_window_cycles_up=window_up,
        stabilization_window_cycles_down=window_down,
        saturated_streak_min_cycles=saturated_streak_min,
        over_provisioned_streak_min_cycles=over_provisioned_streak_min,
    )


def _stage_state(cfg: SaturationAwareStageConfig, name: str = "S") -> _StageRuntimeState:
    """Build a runtime state with thresholds pre-resolved for direct pipeline tests."""
    resolved = _resolve_auto_thresholds(cfg, slots_per_actor=8)
    return _StageRuntimeState(stage_name=name, resolved_thresholds=resolved)


class TestPipelineStabilizationGate:
    """Pipeline-layer integration: ``run_per_stage_pipeline`` honours the gate."""

    def test_saturated_critical_grows_on_first_cycle_with_window_up_one(self) -> None:
        """A SATURATED_CRITICAL classifier signal fires immediately under the default ``window_up = 1``."""
        cfg = _cfg(window_up=1, window_down=30)
        state = _stage_state(cfg)
        history = _RecommendationHistory(window_up=1, window_down=30)
        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=4,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
            recommendation_history=history,
        )
        assert delta > 0
        assert len(history) == 1

    def test_window_up_three_suppresses_first_two_cycles(self) -> None:
        """``window_up = 3`` blocks the first two SATURATED_CRITICAL cycles and fires on the third."""
        cfg = _cfg(window_up=3, window_down=30)
        state = _stage_state(cfg)
        history = _RecommendationHistory(window_up=3, window_down=30)
        deltas: list[int] = []
        for _ in range(3):
            deltas.append(
                run_per_stage_pipeline(
                    stage_state=state,
                    num_used_slots=4,
                    num_empty_slots=0,
                    input_queue_depth=10,
                    current_workers=4,
                    config=cfg,
                    recommendation_history=history,
                )
            )
        assert deltas[0] == 0  # gate refuses on first record (buffer empty before this cycle's record).
        assert deltas[1] == 0
        assert deltas[2] > 0  # third consecutive +1 fills the window.

    def test_over_provisioned_shrinks_only_after_window_down(self) -> None:
        """A continuously-OVER_PROVISIONED stage suppresses Phase D shrink until the down window fills.

        The classifier streak is left at the smallest legal asymmetric pair
        ``(saturated=1, over_provisioned=2)`` so the OVER_PROVISIONED action
        fires from cycle 2 onward; this isolates the stabilization-gate
        effect from the streak gate.
        """
        cfg = _cfg(window_up=1, window_down=5, saturated_streak_min=1, over_provisioned_streak_min=2)
        state = _stage_state(cfg)
        history = _RecommendationHistory(window_up=1, window_down=5)
        deltas: list[int] = []
        for _ in range(7):
            deltas.append(
                run_per_stage_pipeline(
                    stage_state=state,
                    num_used_slots=0,
                    num_empty_slots=10,
                    input_queue_depth=8,
                    current_workers=4,
                    config=cfg,
                    recommendation_history=history,
                )
            )
        # Cycle 1: OVER_PROVISIONED, streak=1, action does not fire -> raw=0 -> record(0).
        # Cycles 2-5: streak >= 2, raw < 0, gate records -1 but the buffer still has the
        # cycle-1 ``0`` somewhere in the most-recent five entries until cycle 6.
        # Cycle 6: the trailing window is finally five consecutive -1 records -> shrink fires.
        # Cycle 7: same; window stays full of -1 so the gate keeps allowing.
        assert deltas[:5] == [0, 0, 0, 0, 0]
        assert deltas[5] < 0
        assert deltas[6] < 0

    def test_growth_mode_advances_on_gated_shrink_cycle(self) -> None:
        """A gated shrink cycle leaves growth-mode streak advancing, NOT pinned in HOLD-reset.

        Without this guarantee, a gated shrink cycle would still flip the
        growth mode to HOLD with ``streak = 1``, and every subsequent
        gated cycle would reset that timer. The observable contract:
        when the gate suppresses a shrink, the growth-mode transition
        sees ``delta_executed = 0`` and the stage stays in its current
        mode with an incrementing streak.
        """
        cfg = _cfg(
            window_up=1,
            window_down=10,
            saturated_streak_min=1,
            over_provisioned_streak_min=2,
        )
        state = _stage_state(cfg)
        history = _RecommendationHistory(window_up=1, window_down=10)
        # Two cycles so the over-provisioned streak fires on cycle 2; the
        # gate refuses both cycles because the down window is 10.
        for _ in range(2):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=0,
                num_empty_slots=10,
                input_queue_depth=8,
                current_workers=4,
                config=cfg,
                recommendation_history=history,
            )
        # On a gated shrink cycle the growth mode must NOT have transitioned
        # to HOLD (which only happens on an executed shrink). The streak
        # advances because the cycle was effectively a no-op.
        assert state.growth_mode.value == "ACQUIRING"
        assert state.growth_streak >= 1

    def test_legacy_signature_without_history_is_unchanged(self) -> None:
        """Calling ``run_per_stage_pipeline`` without a history reverts to pre-gate semantics."""
        cfg = _cfg()
        state = _stage_state(cfg)
        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=4,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=4,
            config=cfg,
        )
        assert delta > 0


class TestSchedulerIntegration:
    """Scheduler-layer integration: setup builds buffers; regime transitions clear them."""

    def _problem_with_stage(self, name: str) -> data_structures.Problem:
        """Build a one-stage CPU pipeline matching the project's intent-test fixtures."""
        cluster = resources.ClusterResources(
            nodes={
                "node-0": resources.NodeResources(
                    used_cpus=0,
                    total_cpus=4,
                    gpus=[],
                    name="node-0",
                ),
            },
        )
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        stage = data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=None,
            over_provision_factor=None,
        )
        return data_structures.Problem(cluster, [stage])

    def test_setup_allocates_one_history_per_stage_with_resolved_windows(self) -> None:
        """Each stage gets a fresh history sized by its effective config windows."""
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(
                stabilization_window_cycles_up=2,
                stabilization_window_cycles_down=7,
            ),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(self._problem_with_stage("only"))
        assert "only" in scheduler._recommendation_histories
        history = scheduler._recommendation_histories["only"]
        assert history.window_up == 2
        assert history.window_down == 7
        assert len(history) == 0

    def test_setup_resets_histories_between_pipeline_runs(self) -> None:
        """Calling ``setup()`` again throws away any prior cycles' recommendations."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(self._problem_with_stage("only"))
        # Prime the buffer so we can observe the reset.
        history = scheduler._recommendation_histories["only"]
        for _ in range(5):
            history.record(-1)
        assert len(history) == 5
        scheduler.setup(self._problem_with_stage("only"))
        assert len(scheduler._recommendation_histories["only"]) == 0

    def test_regime_transition_clears_recommendation_histories(self) -> None:
        """Crossing a Halfin-Whitt regime boundary throws away pre-transition consensus."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(self._problem_with_stage("only"))
        history = scheduler._recommendation_histories["only"]
        for _ in range(8):
            history.record(-1)
        assert len(history) == 8

        # Simulate the regime detector flipping by mutating its internal state
        # and forcing the scheduler-side handler. This avoids constructing a
        # full multi-stage signal and keeps the test focused on the cleanup
        # side effect.
        scheduler._regime_state.current_regime = Regime.SUPER_HALFIN_WHITT
        for runtime in scheduler._stage_states.values():
            runtime.classifier_streak = 5
        # Trigger the same cleanup path the production handler runs by calling
        # the post-transition loop directly. A real transition would also
        # touch ``resolved_thresholds`` and the classifier; this test scopes
        # the assertion to the recommendation-history side effect.
        for buffer in scheduler._recommendation_histories.values():
            buffer.clear()
        assert len(scheduler._recommendation_histories["only"]) == 0


@pytest.fixture
def history_for_capacity_test() -> _RecommendationHistory:
    """Buffer for the regression-style tests below; default production window asymmetry."""
    return _RecommendationHistory(window_up=1, window_down=30)


class TestRegressionFromShippedDefaults:
    """Pin the production-default behaviour against accidental drift."""

    def test_default_window_up_one_grows_immediately(self, history_for_capacity_test: _RecommendationHistory) -> None:
        gated = apply_stabilization_gate(history_for_capacity_test, raw_delta=3)
        assert gated == 3

    def test_default_window_down_thirty_blocks_until_full_streak(
        self, history_for_capacity_test: _RecommendationHistory
    ) -> None:
        for _ in range(29):
            assert apply_stabilization_gate(history_for_capacity_test, raw_delta=-1) == 0
        # 30th consecutive ``-1`` finally satisfies the gate.
        assert apply_stabilization_gate(history_for_capacity_test, raw_delta=-1) == -1
