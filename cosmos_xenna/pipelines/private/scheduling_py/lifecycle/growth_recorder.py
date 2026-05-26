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

"""Post-cycle growth-mode state-machine recorder.

The growth-mode classifier (HOLD / ACQUIRING / TRACKING) must observe
what actually landed at the planner, not the recommendation. Hard
caps, the fractional shrink clamp, and allocation failures can all
push the executed delta below the intent; recording the executed
magnitude (``post_shrink_count - pre_grow_count`` per stage) keeps
the state machine honest across cycles.

``GrowthModeRecorder`` is a domain service - a ``@attrs.frozen``
behaviour bundle holding the pipeline shape, the cross-cycle ledger,
and the ``StageDecisionPipeline`` whose
``record_executed_delta`` method advances the per-stage growth-mode
state. The runner constructs one recorder in ``scheduler.setup()``
and calls :meth:`record` after the post-shrink invariant gate.
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.stage_decision_pipeline import StageDecisionPipeline
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers


@attrs.frozen
class GrowthModeRecorder:
    """Feed post-cycle executed deltas into the growth-mode state machine.

    Stateless ``@attrs.frozen`` domain service constructed once in
    ``scheduler.setup()`` and reused across every cycle. The
    decision pipeline is the same shared instance the intent phase
    uses; the noise-smoothing field is irrelevant for
    ``record_executed_delta`` so the default factory is sufficient.

    Attributes:
        pipeline: Immutable post-setup pipeline shape (provides
            ``stage_config(name)``).
        ledgers: Cross-cycle mutable state container (provides
            ``stage_states[name]``).
        decision_pipeline: Constructor-injected
            :class:`StageDecisionPipeline` whose
            ``record_executed_delta`` advances the per-stage
            growth-mode state.

    """

    pipeline: PipelineModel
    ledgers: SchedulerLedgers
    decision_pipeline: StageDecisionPipeline = attrs.Factory(StageDecisionPipeline)

    def record(self, cycle: AutoscaleCycle) -> None:
        """Feed ``post_shrink - pre_grow`` per stage into the growth-mode state machine.

        Reads:
          * ``cycle.intent.deltas`` for the set of recorded stages.
          * ``cycle.pre_grow_worker_counts`` for the pre-Phase-C
            planner snapshot.
          * ``cycle.planner_worker_counts_by_stage_name()`` for the
            post-Phase-D planner snapshot.

        Stages whose ``problem_state`` runtime state was dropped
        between setup() and now (the ``stage_states`` lookup
        returns ``None``) are silently skipped; this matches the
        prior in-runner behaviour and tolerates shape drift the
        invariant suite has already cleared.

        """
        intent_deltas = cycle.intent.deltas
        pre_grow_worker_counts = cycle.pre_grow_worker_counts
        post_counts = cycle.planner_worker_counts_by_stage_name()
        for stage_name in intent_deltas:
            stage_state = self.ledgers.stage_states.get(stage_name)
            if stage_state is None:
                continue
            stage_cfg = self.pipeline.stage_config(stage_name)
            pre = pre_grow_worker_counts.get(stage_name, 0)
            post = post_counts.get(stage_name, 0)
            self.decision_pipeline.record_executed_delta(
                stage_state=stage_state,
                delta_executed=post - pre,
                config=stage_cfg,
            )


__all__ = ["GrowthModeRecorder"]
