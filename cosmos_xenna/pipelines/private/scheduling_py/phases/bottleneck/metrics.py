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

"""Per-stage ``D_k`` gauge emission and the once-per-cycle bottleneck INFO log.

Pairs the ``xenna_stage_bottleneck_score`` gauge (per-stage
cardinality) with the operator-facing INFO line that names the
engaged bottleneck and reports the throughput bound. Pure
observability: no scheduler state mutation.
"""

import math
from collections.abc import Mapping

from ray.util.metrics import Gauge

from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_identity import BottleneckIdentity
from cosmos_xenna.utils import python_log as logger

# Module-level Gauge instance. Ray's Gauge holds a Cython metric
# handle that registers with the Ray metrics agent on construction;
# constructing one per call would re-register on every cycle and is
# 2-3 orders of magnitude slower than reusing a single handle. The
# tag keys ``{stage, pipeline}`` are the cardinality envelope: one
# observation per stage per pipeline per cycle.
_BOTTLENECK_GAUGE = Gauge(
    "xenna_stage_bottleneck_score",
    description=(
        "Forced Flow Law per-stage actor-normalized service demand "
        "D_k = V_k * S_k / c_k in seconds-per-channel. For Xenna's "
        "linear DAG V_k = 1 so D_k = S_k / c_k where c_k is the "
        "effective ready capacity (concurrent service channels). The "
        "bottleneck stage is argmax_k D_k; pipeline throughput is "
        "bounded by 1 / max_k D_k. NaN samples indicate cold-start "
        "stages with no completed task in the current cycle, or "
        "stages with zero effective capacity."
    ),
    tag_keys=("stage", "pipeline"),
)


def emit_bottleneck_score(
    *,
    d_k_by_stage: Mapping[str, float],
    bottleneck_identity: BottleneckIdentity,
    pipeline_name: str,
    effective_capacities: Mapping[str, int] | None = None,
) -> None:
    """Emit per-stage bottleneck-score gauges and one INFO log line.

    Observes ``D_k`` for every stage on
    ``xenna_stage_bottleneck_score{stage, pipeline}``; cold-start
    stages observe ``math.nan`` so cardinality is stable. If
    ``bottleneck_identity.engaged`` and at least one stage has a
    finite positive ``D_k``, emits one INFO line naming the
    bottleneck and the throughput bound ``1 / max_k D_k``. Pure
    observability - no scheduler state mutation. Sharing identity
    with ``identify_bottleneck`` keeps the operator-facing log
    aligned with the near-tie selection that drives Phase C / D.
    ``effective_capacities`` (optional) adds a ``capacity`` field
    to the INFO log for operator sanity-checking.

    """
    if not d_k_by_stage:
        return

    finite_scores: dict[str, float] = {}
    for stage_name, d_k in d_k_by_stage.items():
        is_finite_positive = math.isfinite(d_k) and d_k > 0.0
        observed = d_k if is_finite_positive else math.nan
        _BOTTLENECK_GAUGE.set(
            observed,
            tags={"stage": stage_name, "pipeline": pipeline_name},
        )
        if is_finite_positive:
            finite_scores[stage_name] = d_k

    if not finite_scores:
        return

    if bottleneck_identity.stage_name is None or bottleneck_identity.stage_name not in finite_scores:
        return

    bottleneck_name = bottleneck_identity.stage_name
    bottleneck_score = finite_scores[bottleneck_name]
    throughput_bound = 1.0 / bottleneck_score
    if effective_capacities is not None and bottleneck_name in effective_capacities:
        capacity_suffix = f", capacity = {effective_capacities[bottleneck_name]}"
    else:
        capacity_suffix = ""
    logger.info(
        f"bottleneck stage: {bottleneck_name!r} "
        f"(D = {bottleneck_score:.2f}s, "
        f"throughput bound = {throughput_bound:.2f} tasks/s"
        f"{capacity_suffix})"
    )
