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
from cosmos_xenna.utils import python_log as logger


def _stage_ceiling(stage_cfg: SaturationAwareStageConfig, num_nodes: int) -> int | None:
    """Per-stage hard worker ceiling.

    Returns ``min(max_workers, max_workers_per_node * num_nodes)``, or
    ``None`` when neither cap is configured. Shared by
    :class:`CeilingCalculator` (the authoritative ceiling) and
    :class:`FloorCalculator` (which compares its floor against this value
    to detect a floor > ceiling cross-term misconfiguration) so the two
    can never diverge on the ceiling formula.
    """
    candidates: list[int] = []
    if stage_cfg.max_workers is not None:
        candidates.append(stage_cfg.max_workers)
    if stage_cfg.max_workers_per_node is not None:
        candidates.append(stage_cfg.max_workers_per_node * num_nodes)
    return min(candidates) if candidates else None


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
    # Stage names already warned about a floor > ceiling misconfig.
    # The calculator is reused across cycles, so this debounces the WARN
    # to once per stage rather than once per autoscale interval.
    # Excluded from eq / hash / repr: mutable debounce state, not identity.
    _floor_exceeds_ceiling_warned: set[str] = attrs.field(factory=set, init=False, eq=False, repr=False)

    def compute(self, num_nodes: int) -> dict[int, int]:
        """Return ``{problem_stage_index: target_min}`` for every stage.

        Floor-wins policy: the floor (``min_workers`` /
        ``min_workers_per_node``) is a hard guarantee that takes
        precedence over the softer ``max_workers`` policy cap. When the
        raw floor exceeds the per-stage ceiling the floor is returned
        unchanged (NOT clamped down); Phase D's ``allowed_by_floor``
        bound then holds the stage at / above its floor even though that
        leaves it above the ceiling. Exceeding ``max_workers`` does not
        over-allocate physical GPUs - placement / resource-fit guards
        physical capacity - so honoring the operator's minimum is the
        safer default. A once-per-stage WARN flags the cross-term
        misconfiguration so the operator can fix it.
        """
        floors: dict[int, int] = {}
        for stage_index, problem_stage in enumerate(self.pipeline.problem.rust.stages):
            stage_cfg = self.pipeline.stage_config(problem_stage.name)
            raw_floor = max(
                stage_cfg.min_workers if stage_cfg.min_workers is not None else 1,
                stage_cfg.min_workers_per_node * num_nodes if stage_cfg.min_workers_per_node is not None else 0,
            )
            ceiling = _stage_ceiling(stage_cfg, num_nodes)
            if ceiling is not None and raw_floor > ceiling:
                self._warn_floor_exceeds_ceiling(problem_stage.name, raw_floor, ceiling, num_nodes)
            floors[stage_index] = raw_floor
        return floors

    def _warn_floor_exceeds_ceiling(self, stage_name: str, raw_floor: int, ceiling: int, num_nodes: int) -> None:
        """Emit a once-per-stage WARN that the floor exceeds the ceiling.

        Under the floor-wins policy the floor is not clamped; the message
        tells the operator the stage will run above its cap until the
        cross-term config is corrected.
        """
        if stage_name in self._floor_exceeds_ceiling_warned:
            return
        self._floor_exceeds_ceiling_warned.add(stage_name)
        logger.warning(
            f"saturation-aware floor exceeds ceiling: stage '{stage_name}' computed floor "
            f"{raw_floor} > ceiling {ceiling} at num_nodes={num_nodes}; the floor takes precedence "
            f"so the stage will run above the cap until the config is fixed. Fix min_workers / "
            f"min_workers_per_node or max_workers / max_workers_per_node so the floor cannot exceed the cap."
        )


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
        return {
            stage_index: _stage_ceiling(self.pipeline.stage_config(problem_stage.name), num_nodes)
            for stage_index, problem_stage in enumerate(self.pipeline.problem.rust.stages)
        }


__all__ = (
    "CapacityModel",
    "CeilingCalculator",
    "FloorCalculator",
)
