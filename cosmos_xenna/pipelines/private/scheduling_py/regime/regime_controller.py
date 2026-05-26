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

"""Cluster Halfin-Whitt regime controller - ``RegimeController``.

Owns the cluster's regime-detection lifecycle:

- aggregates the cluster's regime-detection inputs from per-stage
  runtime snapshots,
- runs the streak gate (``update_regime_state``),
- on transition clears every stage's
  ``classifier.resolved_thresholds`` /
  ``classifier.state`` / ``classifier.streak`` /
  ``classifier.valid_signal_samples`` and the per-stage
  ``RecommendationHistory`` (so the next cycle re-derives
  thresholds at the new aggressiveness),
- exposes ``effective_aggressiveness`` so the threshold resolver
  reads a single source of truth for the SUPER_HALFIN_WHITT
  aggressiveness lift.

The controller takes constructor references to the cluster
config and the ledger so it does not couple back to
``SaturationAwareScheduler``. It is the only owner of
regime-driven cross-cycle invalidation.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime import (
    EXIT_BAND_MULTIPLIER,
    Regime,
    RegimeSignal,
    compute_regime_signal,
    update_regime_state,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class RegimeController:
    """Cluster regime detector + aggressiveness lift + transition handler.

    Args:
        config: Cluster-wide configuration; read for
            ``enable_regime_aware_aggressiveness``,
            ``regime_transition_streak_cycles``, and the
            ``super_halfin_whitt_aggressiveness_lift``.
        ledgers: Cross-cycle ledger; mutated on regime transition
            to invalidate per-stage classifier state and
            stabilization buffers.

    """

    config: SaturationAwareConfig
    ledgers: SchedulerLedgers

    def effective_aggressiveness(self, base: float) -> float:
        """Apply the SUPER_HALFIN_WHITT aggressiveness lift to ``base``.

        Returns ``base`` unchanged when
        ``enable_regime_aware_aggressiveness`` is False or the
        current regime is not SUPER_HALFIN_WHITT; otherwise
        returns ``base + super_halfin_whitt_aggressiveness_lift``.

        """
        if not self.config.enable_regime_aware_aggressiveness:
            return base
        if self.ledgers.regime_state.current_regime is Regime.SUPER_HALFIN_WHITT:
            return base + self.config.super_halfin_whitt_aggressiveness_lift
        return base

    def update(self, problem_state: data_structures.ProblemState) -> None:
        """Detect cluster regime and re-resolve thresholds on transition.

        On transition: clears every stage's resolved thresholds,
        classifier state / streak / valid_signal_samples, and the
        per-stage ``RecommendationHistory``. The next cycle then
        re-derives thresholds at the new aggressiveness and
        rebuilds trust / consensus gates from scratch. No-op when
        ``enable_regime_aware_aggressiveness`` is False or the
        regime signal is unavailable.

        Args:
            problem_state: Per-cycle runtime snapshot.

        """
        if not self.config.enable_regime_aware_aggressiveness:
            return

        signal = aggregate_cluster_regime_signal(problem_state)
        transitioned = update_regime_state(
            self.ledgers.regime_state,
            signal,
            streak_cycles=self.config.regime_transition_streak_cycles,
            exit_band_multiplier=EXIT_BAND_MULTIPLIER,
        )
        if not transitioned:
            return

        effective = self.effective_aggressiveness(self.config.stage_defaults.saturation_aggressiveness)
        _log_regime_transition(self.ledgers.regime_state.current_regime, signal, effective)
        for runtime in self.ledgers.stage_states.values():
            runtime.classifier.resolved_thresholds = None
            runtime.classifier.state = StageState.NORMAL
            runtime.classifier.streak = 0
            runtime.classifier.valid_signal_samples = 0
        for history in self.ledgers.recommendation_histories.values():
            history.clear()


def _log_regime_transition(new_regime: Regime, signal: RegimeSignal, effective_aggressiveness: float) -> None:
    """Emit one INFO line per regime transition."""
    logger.info(
        f"scheduler regime transition: -> {new_regime.value} "
        f"(total_workers={signal.total_workers}, "
        f"cluster_idle_fraction={signal.cluster_idle_fraction:.4f}, "
        f"threshold={signal.threshold:.4f}, "
        f"effective_aggressiveness={effective_aggressiveness:.4f})"
    )


def aggregate_cluster_regime_signal(problem_state: data_structures.ProblemState) -> RegimeSignal:
    """Aggregate the cluster's regime-detection inputs from per-stage state.

    Sums worker counts, used slots, and empty slots across every
    stage in ``problem_state``, then delegates to
    ``compute_regime_signal``. The returned signal is unavailable
    when any active worker stage still carries the ``0/0``
    no-signal sentinel, or when no stage reports any slot
    occupancy at all.

    """
    total_workers = 0
    total_used = 0
    total_empty = 0
    for stage in problem_state.rust.stages:
        stage_workers = len(stage.worker_groups)
        stage_slots = stage.num_used_slots + stage.num_empty_slots
        if stage_workers > 0 and not stage.is_finished and stage_slots == 0:
            return compute_regime_signal(
                total_workers=total_workers + stage_workers,
                total_used_slots=0,
                total_empty_slots=0,
            )
        total_workers += stage_workers
        total_used += stage.num_used_slots
        total_empty += stage.num_empty_slots
    return compute_regime_signal(
        total_workers=total_workers,
        total_used_slots=total_used,
        total_empty_slots=total_empty,
    )


__all__ = (
    "RegimeController",
    "aggregate_cluster_regime_signal",
)
