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

"""Contract tests for :class:`ThresholdResolver`.

Verify the three documented invariants: every stage gets resolved
exactly once until a regime transition invalidates the cache;
``problem_state`` carrying an unknown stage raises ``ValueError``;
and the regime-aware aggressiveness lift is plumbed through
``RegimeController.effective_aggressiveness``.
"""

from typing import cast
from unittest.mock import MagicMock

import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller import RegimeController
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import (
    ClassifierState,
    StageRuntimeState,
)
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import ResolvedThresholds
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver import ThresholdResolver
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


def _stub_stage(stage_name: str, slots_per_worker: int = 4) -> MagicMock:
    """Build a runtime ``problem_state`` stage descriptor."""
    stage = MagicMock()
    stage.stage_name = stage_name
    stage.slots_per_worker = slots_per_worker
    return stage


def _stub_problem_state(stage_names: tuple[str, ...]) -> data_structures.ProblemState:
    """Build a ``ProblemState`` carrying ``stage_names`` in order."""
    state = MagicMock(spec=data_structures.ProblemState)
    state.rust = MagicMock()
    state.rust.stages = [_stub_stage(name) for name in stage_names]
    return cast(data_structures.ProblemState, state)


def _runtime(stage_name: str) -> StageRuntimeState:
    """Build a fresh ``StageRuntimeState`` whose classifier is cold-start."""
    return StageRuntimeState(stage_name=stage_name, classifier=ClassifierState())


def _resolver_with(
    *,
    runtime_map: dict[str, StageRuntimeState],
    pipeline: PipelineModel,
    regime: RegimeController,
) -> ThresholdResolver:
    """Build a ``ThresholdResolver`` that owns ``runtime_map`` through a sentinel ledger."""
    ledgers = MagicMock(spec=SchedulerLedgers)
    ledgers.stage_states = runtime_map
    return ThresholdResolver(
        ledgers=cast(SchedulerLedgers, ledgers),
        regime=regime,
        pipeline=pipeline,
    )


def _stage_config() -> SaturationAwareStageConfig:
    """Build a sentinel per-stage config with a documented base aggressiveness."""
    cfg = MagicMock(spec=SaturationAwareStageConfig)
    cfg.saturation_aggressiveness = 1.0
    return cast(SaturationAwareStageConfig, cfg)


def _resolved(saturation_threshold: float = 0.5) -> ResolvedThresholds:
    """Build a sentinel ``ResolvedThresholds`` carrying the resolution outputs."""
    return ResolvedThresholds(
        saturation_threshold=saturation_threshold,
        activation_threshold=0.1,
        saturation_aggressiveness=1.0,
        slots_per_actor=4,
        saturation_threshold_was_overridden=False,
        activation_threshold_was_overridden=False,
    )


def _pipeline(stage_config: SaturationAwareStageConfig) -> PipelineModel:
    """Build a ``PipelineModel`` whose ``stage_config`` returns ``stage_config``."""
    pipeline = MagicMock(spec=PipelineModel)
    pipeline.stage_config.return_value = stage_config
    return cast(PipelineModel, pipeline)


def _regime(*, lift: float = 0.0) -> RegimeController:
    """Build a ``RegimeController`` whose ``effective_aggressiveness`` adds ``lift`` to the base."""
    regime = MagicMock(spec=RegimeController)
    regime.effective_aggressiveness.side_effect = lambda base: base + lift
    return cast(RegimeController, regime)


class TestFirstCycleResolution:
    """First call resolves every stage and writes back onto the runtime classifier."""

    def test_first_call_resolves_every_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each stage gets a fresh resolution and the result is pinned onto ``classifier.resolved_thresholds``."""
        runtime_map = {name: _runtime(name) for name in ("s0", "s1")}
        target = "cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver._resolve_auto_thresholds"
        calls: list[str] = []

        def _fake(_cfg: object, *, slots_per_actor: int, aggressiveness_override: float) -> ResolvedThresholds:
            del slots_per_actor, aggressiveness_override
            calls.append("resolved")
            return _resolved()

        monkeypatch.setattr(target, _fake)
        resolver = _resolver_with(
            runtime_map=runtime_map,
            pipeline=_pipeline(_stage_config()),
            regime=_regime(),
        )

        resolver.ensure_resolved(_stub_problem_state(("s0", "s1")))

        assert calls == ["resolved", "resolved"]
        assert runtime_map["s0"].classifier.resolved_thresholds is not None
        assert runtime_map["s1"].classifier.resolved_thresholds is not None


class TestCachedReuseAcrossCycles:
    """A second call without a runtime invalidation reuses the cached resolution."""

    def test_second_call_does_not_re_resolve_cached_stages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_resolve_auto_thresholds`` is invoked once per stage across two ``ensure_resolved`` calls."""
        runtime_map = {"s0": _runtime("s0")}
        target = "cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver._resolve_auto_thresholds"
        call_count = 0

        def _fake(_cfg: object, *, slots_per_actor: int, aggressiveness_override: float) -> ResolvedThresholds:
            nonlocal call_count
            del slots_per_actor, aggressiveness_override
            call_count += 1
            return _resolved()

        monkeypatch.setattr(target, _fake)
        resolver = _resolver_with(
            runtime_map=runtime_map,
            pipeline=_pipeline(_stage_config()),
            regime=_regime(),
        )
        problem_state = _stub_problem_state(("s0",))

        resolver.ensure_resolved(problem_state)
        resolver.ensure_resolved(problem_state)

        assert call_count == 1


class TestUnknownStageRaisesValueError:
    """A stage missing from the ledger's stage-state map raises ``ValueError``."""

    def test_unknown_stage_in_problem_state_raises_value_error(self) -> None:
        """``problem_state`` carrying ``ghost`` raises ``ValueError`` listing the known stage names."""
        runtime_map = {"s0": _runtime("s0")}
        resolver = _resolver_with(
            runtime_map=runtime_map,
            pipeline=_pipeline(_stage_config()),
            regime=_regime(),
        )
        with pytest.raises(ValueError, match="ghost.*not found in setup"):
            resolver.ensure_resolved(_stub_problem_state(("ghost",)))


class TestRegimeAggressivenessLift:
    """``RegimeController.effective_aggressiveness`` is plumbed into the resolver call."""

    def test_regime_lift_is_forwarded_as_aggressiveness_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The resolver passes the regime-lifted aggressiveness to ``_resolve_auto_thresholds``."""
        runtime_map = {"s0": _runtime("s0")}
        target = "cosmos_xenna.pipelines.private.scheduling_py.thresholds.threshold_resolver._resolve_auto_thresholds"
        captured: list[float] = []

        def _fake(_cfg: object, *, slots_per_actor: int, aggressiveness_override: float) -> ResolvedThresholds:
            del slots_per_actor
            captured.append(aggressiveness_override)
            return _resolved()

        monkeypatch.setattr(target, _fake)
        resolver = _resolver_with(
            runtime_map=runtime_map,
            pipeline=_pipeline(_stage_config()),
            regime=_regime(lift=0.4),
        )
        resolver.ensure_resolved(_stub_problem_state(("s0",)))

        assert captured == [pytest.approx(1.4)]
