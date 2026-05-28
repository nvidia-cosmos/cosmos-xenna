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

"""Post-cycle observability + invariant checks for ``SaturationAwareScheduler``.

Two collaborating classes that own the work the facade used to
inline into ``_autoscale_body`` after the phase runner returned:

- ``StuckPlanInvariant`` - validates the ``stuck_plan_counters``
  ledger advanced monotonically (+1 or 0 per active stage) for
  this cycle. A non-monotonic step indicates a scheduler defect
  (Phase C double-counted or reset mid-cycle); the check raises
  ``SchedulerInvariantError`` so the run terminates loudly
  instead of silently emitting bad metrics.

- ``PostCycleReporter`` - emits the four post-cycle balance
  signals: ``saturation_aware_bottleneck_score``,
  ``saturation_aware_cluster_heterogeneity_ratio``,
  ``saturation_aware_balance_score``, and the cycle summary
  DEBUG log. Also runs the end-of-cycle balance regression check
  by simulating ``D_k`` from post-Phase-D worker counts (with
  intrinsic ``S_k`` held fixed) and comparing against the
  pre-cycle ``balance_score_start``.

Both classes are ``@attrs.frozen`` value objects bound to the
live scheduler ledger so they avoid a permanent back-reference
to the facade.
"""

import math

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.invariants.checks import check_stuck_plan_monotonicity
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.heterogeneity import compute_heterogeneity_ratio
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.metrics import emit_bottleneck_score
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring import (
    compute_balance_score,
    compute_d_k,
    emit_balance_score,
)
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.pipeline_model import PipelineModel
from cosmos_xenna.pipelines.private.scheduling_py.state.allocation_failure_gate import AllocationFailureGate
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.ledgers import SchedulerLedgers
from cosmos_xenna.utils import python_log as logger


@attrs.frozen
class StuckPlanInvariant:
    """Post-cycle invariant: ``stuck_plan_counters`` advanced by 0 or +1 per active stage.

    Filters to non-finished stages with an observable transition
    so the strict +1/0 helper rule does not flag no-op resets or
    stages that Phase C did not touch this cycle. See
    ``docs/scheduler/saturation-aware/`` for the algorithm.

    Args:
        ledgers: Cross-cycle ledger; read for the current
            ``stuck_plan_counters`` snapshot.

    """

    ledgers: SchedulerLedgers

    def check(
        self,
        *,
        problem_state: data_structures.ProblemState,
        prev_stuck_plan_counters: dict[str, int],
    ) -> None:
        """Validate post-cycle counters against ``prev_stuck_plan_counters``.

        Raises:
            SchedulerInvariantError: A counter moved by anything
                other than 0 or +1 (scheduler defect).

        """
        active_stage_names = {stage.stage_name for stage in problem_state.rust.stages if not stage.is_finished}
        changed_counters = {
            name: curr
            for name, curr in self.ledgers.stuck_plan.view().items()
            if name in active_stage_names and curr != prev_stuck_plan_counters.get(name, 0)
        }
        check_stuck_plan_monotonicity(
            prev_counters=prev_stuck_plan_counters,
            curr_counters=changed_counters,
        )


@attrs.frozen
class PostCycleReporter:
    """Post-cycle balance + bottleneck + heterogeneity observability bundle.

    Emits the four post-cycle gauges, runs the end-of-cycle
    balance regression warn, and emits the per-cycle DEBUG
    summary line. All consumers receive the actor-normalized
    ``cycle.bottleneck.d_k_now`` so gauges, regression check, and
    operator log all agree on the same ``D_k`` view as the
    planner.

    Args:
        pipeline: ``PipelineModel`` captured by ``setup()``;
            provides the cluster-wide config and the stable
            stage-name order used to align
            ``post_cycle_worker_counts`` with the post-Shrink
            ``D_k`` reconstruction.
        ledgers: Cross-cycle ledger; mutated by
            ``compute_heterogeneity_ratio`` (heterogeneity warn
            state), read for ``s_k_ewma``, ``cycle_counter``,
            ``heterogeneity_state``, and ``regime_state``.
        pipeline_name: Pipeline tag used in Prometheus labels and
            WARN logs.
        manual_allocation: Manual-phase allocation-failure gate
            (owned by :class:`ManualGrowExecutor`); read for the
            cycle-summary ``aborted_cycle`` field.
        floor_allocation: Floor-phase allocation-failure gate
            (owned by the floor :class:`DonorBackedAddExecutor`);
            read for the cycle-summary ``aborted_cycle`` field.
        grow_allocation: Grow-phase allocation-failure gate
            (owned by the grow :class:`DonorBackedAddExecutor`);
            read for the cycle-summary ``aborted_cycle`` field.

    """

    pipeline: PipelineModel
    ledgers: SchedulerLedgers
    pipeline_name: str
    manual_allocation: AllocationFailureGate
    floor_allocation: AllocationFailureGate
    grow_allocation: AllocationFailureGate

    def emit(self, cycle: AutoscaleCycle) -> None:
        """Emit balance gauges + regression check + cycle summary."""
        self._emit_post_cycle_balance_metrics(cycle)
        self._emit_cycle_summary()

    def _emit_post_cycle_balance_metrics(self, cycle: AutoscaleCycle) -> None:
        """Emit bottleneck, heterogeneity, balance gauges + regression warn."""
        bottleneck = cycle.bottleneck
        d_k_now = bottleneck.d_k_now
        channels_per_worker_group = bottleneck.channels_per_worker_group
        config = self.pipeline.config
        emit_bottleneck_score(
            d_k_by_stage=d_k_now,
            bottleneck_identity=bottleneck.identity,
            pipeline_name=self.pipeline_name,
            effective_capacities=bottleneck.effective_capacities,
        )
        compute_heterogeneity_ratio(
            d_k_by_stage=d_k_now,
            pipeline_name=self.pipeline_name,
            state=self.ledgers.heterogeneity_state,
            warn_threshold=config.cluster_heterogeneity_warn_threshold,
            warn_streak_cycles=config.cluster_heterogeneity_warn_streak,
        )
        emit_balance_score(d_k_now, pipeline_name=self.pipeline_name)
        post_cycle_worker_counts = cycle.ctx.worker_ids_by_stage()
        d_k_end = {
            stage_name: compute_d_k(
                self.ledgers.s_k_ewma.get(stage_name),
                (len(post_cycle_worker_counts[stage_index]) * channels_per_worker_group.get(stage_name, 0))
                if stage_index < len(post_cycle_worker_counts)
                else 0,
            )
            for stage_index, stage_name in enumerate(self.pipeline.stage_names)
            if stage_name in d_k_now
        }
        self._maybe_warn_balance_regression(
            balance_score_start=bottleneck.balance_score_start,
            balance_score_end=compute_balance_score(d_k_end),
        )

    def _emit_cycle_summary(self) -> None:
        """Emit one structured DEBUG summary line per cycle.

        Includes the three per-phase allocation-failure gates so
        operators can attribute an absorbed ``AllocationError`` to
        the phase that engaged the ``skip_cycle_on_allocation_error``
        tolerance contract: ``manual_allocation`` (Phase A /
        Manual), ``floor_allocation`` (Phase B / Floor), and
        ``grow_allocation`` (Phase C / Grow).
        """
        logger.debug(
            f"saturation-aware cycle {self.ledgers.cycle_counter} summary: "
            f"regime={self.ledgers.regime_state.current_regime.value}, "
            f"heterogeneity_streak={self.ledgers.heterogeneity_state.streak_cycles}, "
            f"heterogeneity_fired={self.ledgers.heterogeneity_state.has_fired}, "
            f"manual_allocation_aborted_cycle={self.manual_allocation.aborted_cycle}, "
            f"floor_allocation_aborted_cycle={self.floor_allocation.aborted_cycle}, "
            f"grow_allocation_aborted_cycle={self.grow_allocation.aborted_cycle}"
        )

    def _maybe_warn_balance_regression(
        self,
        *,
        balance_score_start: float,
        balance_score_end: float,
    ) -> None:
        """Emit one WARN log when end-of-cycle balance regressed beyond tolerance.

        Balance is a secondary objective; the donor commit gate
        already enforces throughput non-regression on every plan,
        so this WARN is informational only. Suppressed at cold
        start (NaN scores).

        Args:
            balance_score_start: Cycle-start balance score (from
                the bottleneck phase).
            balance_score_end: End-of-cycle balance score.

        """
        # NaN comparisons short-circuit to False, so cold-start
        # cycles (fewer than two stages with finite ``D_k``) never
        # fire the WARN even though ``balance_score_start`` and
        # ``balance_score_end`` are both NaN.
        tolerance = self.pipeline.config.cross_stage_donor_balance_regression_tolerance
        if (
            math.isfinite(balance_score_start)
            and math.isfinite(balance_score_end)
            and balance_score_end < balance_score_start - tolerance
        ):
            logger.warning(
                f"[scheduler] pipeline balance regression: "
                f"pipeline={self.pipeline_name!r} "
                f"cycle={self.ledgers.cycle_counter} "
                f"balance_score_start={balance_score_start:.4f} "
                f"balance_score_end={balance_score_end:.4f} "
                f"tolerance={tolerance:.4f}"
            )


__all__ = (
    "PostCycleReporter",
    "StuckPlanInvariant",
)
