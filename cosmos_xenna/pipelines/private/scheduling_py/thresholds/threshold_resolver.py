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

"""Per-stage classifier-threshold resolver - ``ThresholdResolver``.

Owns the lazy resolution of the per-stage classifier thresholds
(saturation, activation, over-provisioned) and the one-shot INFO
log that pins the resolved values onto the run log.

Resolution rules
----------------

- the resolver reads ``slots_per_worker`` (M/M/c concurrency)
  from each stage's runtime snapshot,
- it applies the regime-aware aggressiveness lift via the
  ``RegimeController`` so resolution and regime invalidation use
  a single source of truth,
- it caches the result on
  ``StageRuntimeState.classifier.resolved_thresholds`` -
  mid-run ``slots_per_worker`` changes do not re-resolve; only a
  regime transition (which clears
  ``resolved_thresholds``) triggers re-derivation.

Per-stage configs (including any ``StageSpec``-level overrides)
are sourced from the post-setup ``PipelineModel`` so the
resolver does not need a back reference to the scheduler facade.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.regime.regime_controller import RegimeController
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import (
    ResolvedThresholds,
    _resolve_auto_thresholds,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class ThresholdResolver:
    """Lazy per-stage classifier threshold resolver.

    Args:
        ledgers: Cross-cycle ledger; mutated by writing the
            resolved thresholds back onto
            ``StageRuntimeState.classifier.resolved_thresholds``.
        regime: Regime controller; provides the aggressiveness
            lift applied during resolution.
        pipeline: ``PipelineModel`` captured by ``setup()``.
            Provides the effective per-stage config lookup used
            during resolution.

    """

    ledgers: SchedulerLedgers
    regime: RegimeController
    pipeline: PipelineModel

    def ensure_resolved(self, problem_state: data_structures.ProblemState) -> None:
        """Resolve thresholds for every stage that still has ``resolved_thresholds is None``.

        Iterates the runtime snapshot, looks up the matching
        ledger entry, and resolves once per stage per
        invalidation interval. Logs one INFO line per stage that
        is resolved.

        Raises:
            ValueError: ``problem_state`` carries a stage name
                not present in the ``setup()`` state map.

        """
        for stage in problem_state.rust.stages:
            runtime = self.ledgers.stage_states.get(stage.stage_name)
            if runtime is None:
                msg = (
                    f"problem_state stage {stage.stage_name!r} not found in setup() "
                    f"state map (known: {sorted(self.ledgers.stage_states)}); "
                    "problem and problem_state shapes disagree."
                )
                raise ValueError(msg)
            if runtime.classifier.resolved_thresholds is not None:
                continue
            stage_cfg = self.pipeline.stage_config(stage.stage_name)
            effective_aggressiveness = self.regime.effective_aggressiveness(stage_cfg.saturation_aggressiveness)
            resolved = _resolve_auto_thresholds(
                stage_cfg,
                slots_per_actor=stage.slots_per_worker,
                aggressiveness_override=effective_aggressiveness,
            )
            runtime.classifier.resolved_thresholds = resolved
            _log_resolved_thresholds(stage.stage_name, stage_cfg, resolved)


def _log_resolved_thresholds(
    stage_name: str,
    stage_cfg: SaturationAwareStageConfig,
    resolved: ResolvedThresholds,
) -> None:
    """Emit one INFO line per stage describing the resolved thresholds."""
    sat_source = "manual override" if resolved.saturation_threshold_was_overridden else "auto"
    act_source = "manual override" if resolved.activation_threshold_was_overridden else "auto"
    logger.info(
        f"scheduler resolved auto thresholds for stage {stage_name!r}: "
        f"slots_per_actor={resolved.slots_per_actor}, "
        f"saturation_aggressiveness={resolved.saturation_aggressiveness}, "
        f"saturation_threshold={resolved.saturation_threshold:.6f} ({sat_source}), "
        f"activation_threshold={resolved.activation_threshold:.6f} ({act_source}), "
        f"over_provisioned_threshold={stage_cfg.over_provisioned_threshold} (config; not auto-derived)"
    )


__all__ = ("ThresholdResolver",)
