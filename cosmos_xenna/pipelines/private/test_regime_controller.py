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

"""Contract tests for :class:`RegimeController`.

Verify three observable behaviours: the master toggle short-circuits
every read of ledger state; the per-cycle aggregate signal is fed
into ``update_regime_state`` from ``aggregate_cluster_regime_signal``;
and a successful transition clears every stage's classifier state,
streak, valid-signal counter, resolved thresholds, and the per-stage
``RecommendationHistory``.
"""

from typing import cast
from unittest.mock import MagicMock

import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime import (
    Regime,
    RegimeDetectorState,
    RegimeSignal,
)
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller import RegimeController
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    ClassifierState,
    StageRuntimeState,
    StageState,
)
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import ResolvedThresholds
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig

_UPDATE_PATH = "cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller.update_regime_state"
_AGGREGATE_PATH = (
    "cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller.aggregate_cluster_regime_signal"
)


def _stub_problem_state() -> data_structures.ProblemState:
    """Build a sentinel ``ProblemState``; aggregation is patched in the tests."""
    state = MagicMock(spec=data_structures.ProblemState)
    state.rust = MagicMock()
    state.rust.stages = []
    return cast(data_structures.ProblemState, state)


def _resolved_thresholds() -> ResolvedThresholds:
    """Build a sentinel ``ResolvedThresholds`` to seed the classifier cache."""
    return ResolvedThresholds(
        saturation_threshold=0.5,
        activation_threshold=0.1,
        saturation_aggressiveness=1.0,
        slots_per_actor=4,
        saturation_threshold_was_overridden=False,
        activation_threshold_was_overridden=False,
    )


def _stage_runtime(stage_name: str) -> StageRuntimeState:
    """Build a ``StageRuntimeState`` whose classifier already carries non-default state."""
    return StageRuntimeState(
        stage_name=stage_name,
        classifier=ClassifierState(
            state=StageState.SATURATED,
            streak=5,
            valid_signal_samples=12,
            resolved_thresholds=_resolved_thresholds(),
        ),
    )


def _stub_ledgers(
    *,
    stage_states: dict[str, StageRuntimeState],
    histories: dict[str, RecommendationHistory],
    regime_state: RegimeDetectorState | None = None,
) -> SchedulerLedgers:
    """Build a ``SchedulerLedgers`` exposing only the controller's reads."""
    ledgers = MagicMock(spec=SchedulerLedgers)
    ledgers.stage_states = stage_states
    ledgers.recommendation_histories = histories
    ledgers.regime_state = regime_state or RegimeDetectorState()
    return cast(SchedulerLedgers, ledgers)


def _stub_config(*, enabled: bool = True) -> SaturationAwareConfig:
    """Build a ``SaturationAwareConfig`` with the regime master toggle preset."""
    cfg = MagicMock(spec=SaturationAwareConfig)
    cfg.enable_regime_aware_aggressiveness = enabled
    cfg.regime_transition_streak_cycles = 3
    cfg.super_halfin_whitt_aggressiveness_lift = 0.4
    stage_defaults = MagicMock()
    stage_defaults.saturation_aggressiveness = 1.0
    cfg.stage_defaults = stage_defaults
    return cast(SaturationAwareConfig, cfg)


class TestMasterToggleDisabled:
    """``enable_regime_aware_aggressiveness=False`` makes ``update`` a no-op."""

    def test_disabled_toggle_skips_signal_aggregation_and_state_update(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Neither ``aggregate_cluster_regime_signal`` nor ``update_regime_state`` runs when disabled."""
        aggregate = MagicMock()
        update = MagicMock()
        monkeypatch.setattr(_AGGREGATE_PATH, aggregate)
        monkeypatch.setattr(_UPDATE_PATH, update)

        controller = RegimeController(
            config=_stub_config(enabled=False),
            ledgers=_stub_ledgers(stage_states={}, histories={}),
        )
        controller.update(_stub_problem_state())

        aggregate.assert_not_called()
        update.assert_not_called()


class TestSignalAggregationIsForwardedToUpdate:
    """``aggregate_cluster_regime_signal`` output is passed verbatim to ``update_regime_state``."""

    def test_aggregated_signal_is_forwarded_to_update_regime_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The signal returned by aggregation is the exact value passed to ``update_regime_state``."""
        signal = RegimeSignal(
            total_workers=8,
            cluster_idle_fraction=0.12,
            threshold=0.354,
            signal_available=True,
        )
        monkeypatch.setattr(_AGGREGATE_PATH, MagicMock(return_value=signal))
        update = MagicMock(return_value=False)
        monkeypatch.setattr(_UPDATE_PATH, update)

        controller = RegimeController(
            config=_stub_config(),
            ledgers=_stub_ledgers(stage_states={}, histories={}),
        )
        controller.update(_stub_problem_state())

        update.assert_called_once()
        # Positional signal arg is the second positional after the regime state.
        assert update.call_args.args[1] is signal
        assert update.call_args.kwargs["streak_cycles"] == 3


class TestTransitionClearsClassifierAndHistories:
    """A successful transition wipes per-stage classifier state and recommendation histories."""

    def test_transition_clears_per_stage_classifier_and_history(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every classifier sub-state is reset and every ``RecommendationHistory`` is cleared on transition."""
        stage_states = {"s0": _stage_runtime("s0"), "s1": _stage_runtime("s1")}
        history_s0 = RecommendationHistory(window_up=1, window_down=3)
        history_s0.record(1)
        history_s1 = RecommendationHistory(window_up=1, window_down=3)
        history_s1.record(-1)
        histories = {"s0": history_s0, "s1": history_s1}
        regime_state = RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT, streak=0)
        ledgers = _stub_ledgers(
            stage_states=stage_states,
            histories=histories,
            regime_state=regime_state,
        )

        signal = RegimeSignal(
            total_workers=4,
            cluster_idle_fraction=0.5,
            threshold=0.5,
            signal_available=True,
        )
        monkeypatch.setattr(_AGGREGATE_PATH, MagicMock(return_value=signal))
        monkeypatch.setattr(_UPDATE_PATH, MagicMock(return_value=True))

        controller = RegimeController(config=_stub_config(), ledgers=ledgers)
        controller.update(_stub_problem_state())

        for runtime in stage_states.values():
            assert runtime.classifier.resolved_thresholds is None
            assert runtime.classifier.state is StageState.NORMAL
            assert runtime.classifier.streak == 0
            assert runtime.classifier.valid_signal_samples == 0
        for history in histories.values():
            assert len(history) == 0


class TestNoTransitionLeavesStateIntact:
    """When ``update_regime_state`` reports no transition, per-stage state is untouched."""

    def test_no_transition_does_not_clear_classifier_or_history(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``current_regime`` unchanged means classifier streak and resolved thresholds survive."""
        stage = _stage_runtime("s0")
        history = RecommendationHistory(window_up=1, window_down=3)
        history.record(1)
        ledgers = _stub_ledgers(stage_states={"s0": stage}, histories={"s0": history})

        signal = RegimeSignal(
            total_workers=4,
            cluster_idle_fraction=0.5,
            threshold=0.5,
            signal_available=True,
        )
        monkeypatch.setattr(_AGGREGATE_PATH, MagicMock(return_value=signal))
        monkeypatch.setattr(_UPDATE_PATH, MagicMock(return_value=False))

        controller = RegimeController(config=_stub_config(), ledgers=ledgers)
        controller.update(_stub_problem_state())

        assert stage.classifier.streak == 5
        assert stage.classifier.resolved_thresholds is not None
        assert len(history) == 1


class TestEffectiveAggressiveness:
    """``effective_aggressiveness`` returns ``base`` outside SUPER_HALFIN_WHITT, ``base + lift`` inside it."""

    def test_returns_base_when_toggle_disabled(self) -> None:
        """A disabled toggle returns ``base`` regardless of the regime state."""
        ledgers = _stub_ledgers(
            stage_states={},
            histories={},
            regime_state=RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT),
        )
        controller = RegimeController(config=_stub_config(enabled=False), ledgers=ledgers)
        assert controller.effective_aggressiveness(1.2) == 1.2

    def test_returns_lifted_base_in_super_halfin_whitt(self) -> None:
        """SUPER_HALFIN_WHITT + enabled toggle returns ``base + super_halfin_whitt_aggressiveness_lift``."""
        ledgers = _stub_ledgers(
            stage_states={},
            histories={},
            regime_state=RegimeDetectorState(current_regime=Regime.SUPER_HALFIN_WHITT),
        )
        controller = RegimeController(config=_stub_config(), ledgers=ledgers)
        assert controller.effective_aggressiveness(1.2) == 1.6

    def test_returns_base_in_sub_halfin_whitt(self) -> None:
        """SUB_HALFIN_WHITT + enabled toggle returns ``base`` unchanged."""
        ledgers = _stub_ledgers(
            stage_states={},
            histories={},
            regime_state=RegimeDetectorState(current_regime=Regime.SUB_HALFIN_WHITT),
        )
        controller = RegimeController(config=_stub_config(), ledgers=ledgers)
        assert controller.effective_aggressiveness(1.2) == 1.2
