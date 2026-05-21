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


"""Per-stage backpressure helpers for the streaming run-loop.

``compute_max_queued`` returns the per-stage inflight task cap, with a
cold-start branch (gated by ``setup_aware_max_queued``) that prevents
the upstream stage from pre-queueing object-store data while a pool's
actors are still loading models.
``resolve_setup_aware_max_queued_enabled`` resolves the per-stage flag
through the three-tier precedence chain.
"""

import typing

from loguru import logger as _loguru_logger

from cosmos_xenna.pipelines.private import specs

# Cold-start floor for the per-stage backpressure cap when a pool has
# pending actors still in setup and no ready actors yet. A value of 1
# lets the first actor that completes setup pull one task immediately
# without pre-queueing more work behind sibling actors that are still
# loading models, which would otherwise inflate Ray object-store usage.
_SETUP_AWARE_MAX_QUEUED_FLOOR: typing.Final[int] = 1


def compute_max_queued(
    *,
    num_ready_actors: int,
    num_pending_actors: int,
    slots_per_actor: int,
    max_queued_multiplier: float,
    max_queued_lower_bound: int,
    next_stage_batch_size: int,
    setup_aware_enabled: bool,
    is_done: bool,
    stage_name: str,
) -> int:
    """Return the effective per-stage ``max_queued`` cap for the streaming loop.

    The regular formula caps the inflight task count at
    ``num_ready_actors * slots_per_actor * max_queued_multiplier``,
    floored by ``max_queued_lower_bound`` and ``next_stage_batch_size``.

    When ``setup_aware_enabled`` is True and the pool is in cold start
    (``num_ready_actors == 0`` and ``num_pending_actors > 0`` and the
    stage has not been marked done), the cap is reduced to
    ``max(next_stage_batch_size, _SETUP_AWARE_MAX_QUEUED_FLOOR)``. The
    downstream batch size is preserved so the next stage is never
    starved by upstream backpressure.

    Args:
        num_ready_actors: Actors with completed setup.
        num_pending_actors: Actors still in setup phases.
        slots_per_actor: Slots per ready actor for this stage.
        max_queued_multiplier: ``StreamingSpecificSpec.max_queued_multiplier``.
        max_queued_lower_bound: ``StreamingSpecificSpec.max_queued_lower_bound``.
        next_stage_batch_size: ``stage_batch_size`` of the downstream
            stage; ``-1`` when this is the last stage.
        setup_aware_enabled: Per-stage ``setup_aware_max_queued`` flag.
        is_done: ``True`` when the stage is already finished (cold-start
            branch is bypassed because a teardown pool can transiently
            report mismatched ready/pending counts).
        stage_name: Stage name for the debug log binding.

    Returns:
        Effective ``max_queued`` cap.

    """
    regular_max_queued = max(
        int(num_ready_actors * slots_per_actor * max_queued_multiplier),
        max_queued_lower_bound,
        next_stage_batch_size,
    )
    in_cold_start = setup_aware_enabled and not is_done and num_ready_actors == 0 and num_pending_actors > 0
    if not in_cold_start:
        return regular_max_queued
    cold_start_cap = max(next_stage_batch_size, _SETUP_AWARE_MAX_QUEUED_FLOOR)
    _loguru_logger.bind(stage=stage_name).debug(
        f"setup-aware max_queued: cold-start cap reduced from {regular_max_queued} to {cold_start_cap} "
        f"(num_ready={num_ready_actors}, num_pending={num_pending_actors}, "
        f"next_stage_batch_size={next_stage_batch_size}, floor={_SETUP_AWARE_MAX_QUEUED_FLOOR})"
    )
    return cold_start_cap


def resolve_setup_aware_max_queued_enabled(
    pipeline_spec: specs.PipelineSpec,
    stage_idx: int,
    stage_name: str,
) -> bool:
    """Resolve the effective ``setup_aware_max_queued`` flag for a stage.

    The flag is a per-stage tunable on ``SaturationAwareStageConfig`` and
    therefore inherits the 3-tier precedence chain (``StageSpec.saturation_aware``
    > ``SaturationAwareConfig.per_stage_overrides`` > ``stage_defaults``).
    The cluster-level ``SaturationAwareConfig`` is documented as having no
    effect when the scheduler is ``FRAGMENTATION_BASED``, so the resolver
    returns ``False`` (legacy backpressure path) for that scheduler kind
    and for any caller that runs without ``StreamingSpecificSpec``.
    """
    mode_specific = pipeline_spec.config.mode_specific
    if mode_specific is None:
        return False
    if mode_specific.scheduler is not specs.SchedulerKind.SATURATION_AWARE:
        return False
    stage_spec = pipeline_spec.stages[stage_idx]
    assert isinstance(stage_spec, specs.StageSpec)
    # ``materialized_saturation_aware`` lazily constructs the SA config
    # on the saturation-aware branch only; the field defaults to
    # ``None`` so fragmentation-based pipelines never instantiate it.
    effective_cfg = mode_specific.materialized_saturation_aware().get_effective_stage_config(
        stage_name,
        spec_override=stage_spec.saturation_aware,
    )
    return effective_cfg.setup_aware_max_queued


__all__ = [
    "compute_max_queued",
    "resolve_setup_aware_max_queued_enabled",
]
