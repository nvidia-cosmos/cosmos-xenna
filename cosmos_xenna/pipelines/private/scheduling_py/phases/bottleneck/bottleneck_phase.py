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

"""Bottleneck phase: refresh per-cycle ``D_k``, identify the bottleneck, run engagement log.

Runs between Phase B and the intent phase so Phase C / Phase D
both see a fresh ``D_k`` and bottleneck identity. Pulls one
service-time sample from the ``MeasurementCollector``, smooths
``S_k`` via the cross-cycle EWMA, recomputes ``D_k = S_k / c_k``
per stage, identifies the bottleneck against the configured
heterogeneity threshold, and snapshots the balance score for the
post-Phase-D regression invariant. See
``docs/scheduler/saturation-aware/`` for the algorithm.
"""

import math
from collections.abc import Mapping

import attrs

from cosmos_xenna._cosmos_xenna.pipelines.private.scheduling import (  # type: ignore[import-not-found]
    data_structures as rust_data_structures,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.identity import (
    identify_bottleneck,
    maybe_log_bottleneck_engagement,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring import (
    compute_balance_score,
    compute_d_k,
)
from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.services import BottleneckServices
from cosmos_xenna.pipelines.private.scheduling_py.state.autoscale_cycle import AutoscaleCycle
from cosmos_xenna.pipelines.private.scheduling_py.state.outputs import BottleneckSnapshot


@attrs.frozen
class BottleneckPhase:
    """Per-cycle bottleneck identification.

    Stateless ``@attrs.frozen`` Phase implementation. Runs
    between Phase B and the intent phase; writes the per-cycle
    ``BottleneckSnapshot`` onto ``cycle.bottleneck`` so Phase C,
    Phase D, and the post-cycle observability consumers all
    read the same view.

    """

    def run(self, cycle: AutoscaleCycle, services: BottleneckServices) -> None:
        """Compute the per-cycle bottleneck snapshot and publish it on ``cycle``.

        Updates the cross-cycle ``S_k`` EWMA from this cycle's
        service-time samples, recomputes ``D_k`` and the balance
        score, identifies the bottleneck against the configured
        heterogeneity threshold, and emits the engagement INFO log
        when at least one consumer toggle is enabled. Per-stage
        bottleneck topology is projected on-the-fly from
        ``cycle.bottleneck.identity`` at the consumer call site -
        no per-stage mirror is written here.

        """
        pipeline = services.pipeline
        service_times_s = services.measurements.consume_service_time_samples()
        _update_s_k_ewma(services, service_times_s)
        effective_capacities = {
            stage.stage_name: _effective_ready_capacity(stage) for stage in cycle.problem_state.rust.stages
        }
        # Capture per-stage "channels per worker group" so the
        # end-of-cycle balance regression check can convert post-cycle
        # worker counts to the same channel units ``d_k_now`` was
        # computed in. For stages with at least one start-of-cycle
        # worker_group the ratio is exact (full SPMD K + slots_per_worker
        # baked in); for stages that start at zero the fallback is
        # ``slots_per_worker`` (matches ``_compute_post_plan_d_k``'s
        # donor-gate simplification when group structure is unobservable).
        channels_per_worker_group = {
            stage.stage_name: (
                effective_capacities[stage.stage_name] // len(stage.worker_groups)
                if stage.worker_groups
                else stage.slots_per_worker
            )
            for stage in cycle.problem_state.rust.stages
        }
        d_k_now = {
            name: compute_d_k(services.s_k_ewma.get(name), effective_capacities.get(name, 0))
            for name in pipeline.stage_names
        }
        bottleneck_identity = identify_bottleneck(
            d_k_now,
            heterogeneity_threshold=pipeline.config.bottleneck_heterogeneity_threshold,
        )
        balance_score_start = compute_balance_score(d_k_now)

        # Single bundled snapshot is the source of truth for the rest of
        # the pipeline. Phase C, Phase D, and the post-Phase-D consumers
        # (emit_bottleneck_score, balance regression check, donor commit
        # gate) read via ``cycle.bottleneck``.
        snapshot = BottleneckSnapshot(
            identity=bottleneck_identity,
            d_k_now=d_k_now,
            effective_capacities=effective_capacities,
            channels_per_worker_group=channels_per_worker_group,
            balance_score_start=balance_score_start,
        )
        cycle.bottleneck = snapshot

        # Engagement log is silenced when both decision toggles are off:
        # disabling both decision paths must not introduce a new
        # operator log line. The EWMA state and snapshot are still
        # updated so re-enabling either toggle gets warm data on the
        # first cycle.
        if pipeline.config.enable_bottleneck_priority_growth or pipeline.config.enable_bottleneck_shrink_protection:
            maybe_log_bottleneck_engagement(
                identity=bottleneck_identity,
                state=services.bottleneck_engagement_state,
                persistence_cycles=pipeline.config.bottleneck_engagement_persistence_cycles,
                pipeline_name=services.pipeline_name,
            )


def _update_s_k_ewma(
    services: BottleneckServices,
    service_times_s: Mapping[str, float],
) -> None:
    """Apply one EWMA step to per-stage intrinsic ``S_k`` from this cycle's service times.

    Smooths the intrinsic per-task service time (NOT the
    actor-normalized ``D_k``) so actor-count changes do not leak
    into the smoothed signal - they update the per-cycle ``c_k``
    only and the bottleneck path picks them up on the next cycle.

    Args:
        services: Per-phase services view; mutates
            ``services.s_k_ewma`` in place via the store API.
        service_times_s: Per-stage mean service time;
            ``math.nan`` means no completed tasks this cycle.

    """
    pipeline = services.pipeline
    alpha = pipeline.config.bottleneck_d_k_smoothing_level
    for stage_name in pipeline.stage_names:
        latest = service_times_s.get(stage_name, math.nan)
        services.s_k_ewma.update(stage_name, latest, alpha)


def _effective_ready_capacity(runtime_stage: rust_data_structures.ProblemStageState) -> int:
    """Compute concurrent service channels at one stage.

    Returns ``slots_per_worker * sum(K_g)`` across ready worker
    groups. Counts all ready workers including those still inside
    ``worker_warmup_measurement_grace_s``; the classifier owns
    warmup-trust gating.

    Args:
        runtime_stage: Per-cycle stage snapshot.

    Returns:
        Concurrent channels available now; ``0`` for stages with
        no ready worker groups.

    """
    slots_per_worker = runtime_stage.slots_per_worker
    if slots_per_worker <= 0:
        return 0
    total_allocations = 0
    for worker_group in runtime_stage.worker_groups:
        allocation_count = max(1, len(worker_group.resources))
        total_allocations += allocation_count
    return slots_per_worker * total_allocations


__all__ = ["BottleneckPhase"]
