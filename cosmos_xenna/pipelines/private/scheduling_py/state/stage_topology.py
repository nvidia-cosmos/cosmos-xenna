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

"""Per-stage projection of the cycle's bottleneck identity.

:class:`StageTopologyContext` is the immutable, per-stage view of
the bottleneck-identification result for the current cycle. It is
projected on-the-fly from ``AutoscaleCycle.bottleneck.identity`` and
the stage's DAG index at the call site - never stored on the
per-stage runtime state, never mirrored into a cross-cycle ledger.

The :func:`project_stage_topology` helper performs the projection
from primitive inputs (stage index, the bottleneck's engaged flag,
the bottleneck stage name, the stable stage-name order) so callers
do not have to repeat the lookup-and-compare logic at every site.
"""

from collections.abc import Sequence

import attrs


@attrs.frozen
class StageTopologyContext:
    """Per-stage projection of the cycle's bottleneck topology.

    Two booleans capture everything a consumer (the per-stage
    decision pipeline, the structured DEBUG / INFO log lines) needs
    to know about how the current stage relates to the cluster-wide
    bottleneck this cycle. The value object is constructed at the
    call site of :meth:`StageDecisionPipeline.compute_recommendation`
    via :func:`project_stage_topology`.

    Attributes:
        engaged: ``True`` when the cluster-wide bottleneck gate is
            engaged this cycle. Mirrors
            :attr:`BottleneckIdentity.engaged`.
        is_upstream_of_bottleneck: ``True`` when the owning stage
            sits strictly upstream (smaller DAG index) of the
            engaged bottleneck stage. Always ``False`` when
            ``engaged`` is ``False``, and always ``False`` for the
            bottleneck stage itself.

    """

    engaged: bool = False
    is_upstream_of_bottleneck: bool = False


def project_stage_topology(
    *,
    stage_index: int,
    bottleneck_engaged: bool,
    bottleneck_stage_name: str | None,
    stage_names: Sequence[str],
) -> StageTopologyContext:
    """Project the cycle-wide bottleneck identity into per-stage topology.

    Returns the default ``StageTopologyContext()`` (disengaged,
    not upstream) when:

    - ``bottleneck_engaged`` is ``False`` (cluster homogeneous);
    - ``bottleneck_stage_name`` is ``None`` (defensive guard for a
      disengaged identity carrying an unexpected stage name);
    - ``bottleneck_stage_name`` is not in ``stage_names`` (stale
      identity after a stage-list change between cycles).

    Otherwise returns ``StageTopologyContext(engaged=True,
    is_upstream_of_bottleneck=stage_index <
    bottleneck_index)``. The bottleneck stage itself observes
    ``is_upstream_of_bottleneck=False`` because ``<`` is strict.

    Args:
        stage_index: DAG index of the consumer stage.
        bottleneck_engaged: Whether the cluster-wide bottleneck
            gate is engaged this cycle.
        bottleneck_stage_name: Name of the engaged bottleneck
            stage; ``None`` when disengaged.
        stage_names: Stable stage-name order (matches
            ``problem.rust.stages``).

    """
    if not bottleneck_engaged or bottleneck_stage_name is None:
        return StageTopologyContext()
    try:
        bottleneck_index = stage_names.index(bottleneck_stage_name)
    except ValueError:
        return StageTopologyContext()
    return StageTopologyContext(
        engaged=True,
        is_upstream_of_bottleneck=stage_index < bottleneck_index,
    )


__all__ = ("StageTopologyContext", "project_stage_topology")
