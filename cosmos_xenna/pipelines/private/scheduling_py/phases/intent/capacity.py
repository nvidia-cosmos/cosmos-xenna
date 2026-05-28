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

"""Per-stage capacity, floor, and ceiling calculators.

Three small ``@attrs.frozen`` value objects extracted from
``SaturationAwareScheduler`` so per-stage worker-count
arithmetic lives next to its tests, not buried in the facade:

- ``CapacityModel`` - wraps the M/M/c capacity-target calculation
  used by the intent phase to size each stage from its observed
  throughput and backlog target. Reads the per-stage ``s_k_ewma``
  store directly so the capacity sizing always uses the same
  intrinsic service-time view as the bottleneck phase.

- ``FloorCalculator`` - computes the donor / receiver lower bound
  ``target_min`` per stage from ``min_workers`` and
  ``min_workers_per_node``. Applied uniformly to manual and
  non-manual stages (manual stages may donate down to
  ``min_workers`` because ``requested_num_workers`` is a target,
  not a hard lower bound).

- ``CeilingCalculator`` - computes the per-stage hard worker
  ceiling from ``max_workers`` and ``max_workers_per_node``.
  Returns ``None`` when neither cap is configured.

All three calculators are constructed by
``SaturationAwareScheduler.setup()`` and injected through the
per-phase ``*Services`` value objects: ``CapacityModel`` rides
on :class:`IntentServices`; ``FloorCalculator`` rides on
:class:`FloorServices`, :class:`GrowServices`, and
:class:`ShrinkServices`; ``CeilingCalculator`` rides on the same
grow / shrink services. One instance per scheduler is reused
across cycles.
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure import compute_capacity_target_workers
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.sk_ewma_store import SkEwmaStore
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.pipelines.private.scheduling_py.thresholds.auto_thresholds import derive_utilization_target
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig


@attrs.frozen
class CapacityModel:
    """M/M/c capacity-sized worker-target computation.

    Returns ``None`` when ``D_k`` is unobservable or
    ``resolved_thresholds`` is not yet populated;
    ``compute_delta`` treats ``None`` as the discrete-fallback
    sentinel.

    Args:
        s_k_ewma: Per-stage intrinsic service-time EWMA store
            (the same instance the Bottleneck phase writes via
            :meth:`SkEwmaStore.update`).

    """

    s_k_ewma: SkEwmaStore

    def target_workers(
        self,
        *,
        stage_state: StageRuntimeState,
        stage_cfg: SaturationAwareStageConfig,
        input_queue_depth: int,
        observed_throughput: float,
        slots_per_worker: int,
        stage_name: str,
    ) -> int | None:
        """Return the capacity-sized worker target for one stage."""
        if stage_state.classifier.resolved_thresholds is None:
            return None
        d_k_seconds = self.s_k_ewma.get(stage_name)
        return compute_capacity_target_workers(
            queue_depth=input_queue_depth,
            observed_throughput=observed_throughput,
            d_k_seconds=d_k_seconds,
            slots_per_worker=slots_per_worker,
            target_backlog_seconds=stage_cfg.target_backlog_seconds,
            utilization_target=derive_utilization_target(stage_state.classifier.resolved_thresholds),
        )


@attrs.frozen
class FloorCalculator:
    """Per-stage donor / receiver floor (``target_min``).

    ``target_min = max(min_workers (defaulting to 1),
    min_workers_per_node * num_nodes)``.

    Args:
        pipeline: ``PipelineModel`` captured by
            ``SaturationAwareScheduler.setup()``. Provides the
            frozen pipeline shape and the effective per-stage
            config lookup.

    """

    pipeline: PipelineModel

    def compute(self, num_nodes: int) -> dict[int, int]:
        """Return ``{problem_stage_index: target_min}`` for every stage."""
        floors: dict[int, int] = {}
        for stage_index, problem_stage in enumerate(self.pipeline.problem.rust.stages):
            stage_cfg = self.pipeline.stage_config(problem_stage.name)
            floors[stage_index] = max(
                stage_cfg.min_workers if stage_cfg.min_workers is not None else 1,
                stage_cfg.min_workers_per_node * num_nodes if stage_cfg.min_workers_per_node is not None else 0,
            )
        return floors


@attrs.frozen
class CeilingCalculator:
    """Per-stage hard worker ceiling.

    Returns ``min(max_workers, max_workers_per_node * num_nodes)``
    per stage, or ``None`` when neither cap is configured.
    Phase C drops excess ``try_add_worker`` calls and Phase D
    forces a shrink when ``current > ceiling``.

    Args:
        pipeline: ``PipelineModel`` captured by
            ``SaturationAwareScheduler.setup()``. Provides the
            frozen pipeline shape and the effective per-stage
            config lookup.

    """

    pipeline: PipelineModel

    def compute(self, num_nodes: int) -> dict[int, int | None]:
        """Return ``{problem_stage_index: ceiling_or_none}`` for every stage."""
        ceilings: dict[int, int | None] = {}
        for stage_index, problem_stage in enumerate(self.pipeline.problem.rust.stages):
            stage_cfg = self.pipeline.stage_config(problem_stage.name)
            candidates: list[int] = []
            if stage_cfg.max_workers is not None:
                candidates.append(stage_cfg.max_workers)
            if stage_cfg.max_workers_per_node is not None:
                candidates.append(stage_cfg.max_workers_per_node * num_nodes)
            ceilings[stage_index] = min(candidates) if candidates else None
        return ceilings


__all__ = (
    "CapacityModel",
    "CeilingCalculator",
    "FloorCalculator",
)
