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

"""Per-cycle scheduler working state.

Phase modules read and mutate one ``AutoscaleCycle`` per
``autoscale()`` call. The cycle carries the Rust placement context,
the live ``problem_state`` snapshot, and the phase-output snapshots
each phase populates. Cross-cycle state (stage states, worker ages,
recommendation histories, last-donation ledger, stuck-plan counters)
lives on :class:`SchedulerLedgers` and is read through the per-phase
``*Services`` value objects - no mirror copy on the cycle.

Phase outputs are declared with ``attrs.field(init=False)``: the
field is not part of ``__init__`` and is unset until the producing
phase assigns it. Reading a snapshot before its producing phase ran
raises ``AttributeError`` - the intentional loud failure that
replaces the prior ``| None = None`` + ``require_*()`` accessor
pattern.
"""

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.state.outputs import BottleneckSnapshot, IntentPlan
from cosmos_xenna.pipelines.private.scheduling_py.state.stage_runtime import StageRuntimeState
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class StageCycleView:
    """One stage's projection of the current cycle.

    Carries identity, runtime state, and the small set of derived
    values phase code reads while making per-stage decisions. The
    runtime state is exposed live (a reference into
    ``SchedulerLedgers.stage_states``); phases may update its
    fields directly.

    """

    stage_index: int
    stage_name: str
    runtime_state: StageRuntimeState
    current_workers: int


@attrs.define
class AutoscaleCycle:
    """One autoscale cycle's working state.

    Phase modules read and mutate this object for cycle-local
    data: the placement context, the live ``problem_state``
    snapshot, the donor warmup exclusions, the precomputed
    memory-pressure flag, and the phase-output snapshots.
    Cross-cycle maps (stage states, worker ages, recommendation
    histories, last-donation cycle, stuck-plan counters) are NOT
    mirrored here - they live exclusively on
    :class:`SchedulerLedgers` and are read through the per-phase
    services.

    Phase-output snapshots and the preflight booleans use
    ``attrs.field(init=False)``; reading them before the
    populating preflight / phase step ran raises
    ``AttributeError`` - intentional loud failure.

    """

    ctx: data_structures.AutoscalePlanContext
    problem_state: data_structures.ProblemState
    time: float
    cycle_counter: int
    pipeline_name: str

    donor_warmup_excluded_ids: frozenset[str] = attrs.field(init=False)
    is_memory_pressure_active: bool = attrs.field(init=False)
    bottleneck: BottleneckSnapshot = attrs.field(init=False)
    intent: IntentPlan = attrs.field(init=False)
    pre_grow_worker_counts: dict[str, int] = attrs.field(init=False)
    pre_shrink_worker_counts: dict[int, int] = attrs.field(init=False)

    def view_for(self, stage_index: int, stage_states: dict[str, StageRuntimeState]) -> StageCycleView:
        """Build a stage-scoped view for the given stage index.

        Args:
            stage_index: Zero-based index in
                ``problem_state.rust.stages``.
            stage_states: The cross-cycle stage runtime state map
                (typically ``services.stage_states`` on the
                phase-specific ``*Services`` value object); the
                returned view exposes the live entry as
                ``runtime_state``.

        Returns:
            A frozen ``StageCycleView`` carrying the stage's
            runtime state and the cycle-start worker count.

        Raises:
            IndexError: When the index is out of bounds for the
                current cycle's ``problem_state``.

        """
        stages = self.problem_state.rust.stages
        if stage_index < 0 or stage_index >= len(stages):
            msg = (
                f"AutoscaleCycle.view_for: stage_index={stage_index} out of range "
                f"for problem_state with {len(stages)} stages."
            )
            raise IndexError(msg)
        stage = stages[stage_index]
        return StageCycleView(
            stage_index=stage_index,
            stage_name=stage.stage_name,
            runtime_state=stage_states[stage.stage_name],
            current_workers=len(stage.worker_groups),
        )

    def planner_worker_counts_by_stage_name(self) -> dict[str, int]:
        """Project the live planner worker count keyed by stage name.

        Reads ``ctx.worker_ids_by_stage()`` once and zips against
        the ``problem_state`` stages so the result reflects every
        mutation made by prior phases in the current cycle. Phase
        boundaries use this snapshot to compare pre / post worker
        counts without redrafting the projection in each phase.

        Returns:
            A fresh dict ``stage_name -> worker_count`` over every
            runtime stage. Mutations to the dict do not affect
            the planner state.

        """
        worker_ids_by_index = self.ctx.worker_ids_by_stage()
        return {
            runtime_stage.stage_name: len(worker_ids)
            for runtime_stage, worker_ids in zip(
                self.problem_state.rust.stages,
                worker_ids_by_index,
                strict=True,
            )
        }

    def planner_worker_counts_by_stage_index(self) -> dict[int, int]:
        """Project the live planner worker count keyed by stage index.

        Same source-of-truth as
        ``planner_worker_counts_by_stage_name`` but indexed by the
        Rust planner's zero-based stage index. Phase D uses this
        view to detect over-ceiling stages without going through
        the name -> index translation.

        Returns:
            A fresh dict ``stage_index -> worker_count`` over
            every runtime stage. Mutations do not affect planner
            state.

        Raises:
            ValueError: The planner's stage-vector count drifted
                from ``problem_state.rust.stages``; mirrors the
                loud ``strict=True`` zip failure of
                ``planner_worker_counts_by_stage_name``.

        """
        worker_ids_by_stage = self.ctx.worker_ids_by_stage()
        num_runtime_stages = len(self.problem_state.rust.stages)
        if len(worker_ids_by_stage) != num_runtime_stages:
            msg = (
                "planner_worker_counts_by_stage_index: planner stage-vector "
                f"count {len(worker_ids_by_stage)} != problem_state stage "
                f"count {num_runtime_stages} (pipeline {self.pipeline_name!r}, "
                f"cycle {self.cycle_counter}). Planner state drifted from "
                "problem_state; this is a scheduler defect."
            )
            raise ValueError(msg)
        return {stage_index: len(worker_ids) for stage_index, worker_ids in enumerate(worker_ids_by_stage)}

    def cycle_logger(self, stage: str = "") -> object:
        """Return a loguru logger pre-bound to cycle/pipeline/stage fields.

        Decision sites use this helper to emit structured log
        lines with a stable schema. ``stage=""`` is the
        cluster-level binding.

        """
        return logger.bind(
            cycle=self.cycle_counter,
            pipeline=self.pipeline_name,
            stage=stage,
        )


__all__ = ["AutoscaleCycle", "StageCycleView"]
