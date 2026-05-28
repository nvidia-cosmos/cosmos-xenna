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

"""Forced-Flow-Law per-stage scoring primitives.

Holds the pure ``D_k`` computation and the cluster-level balance
score derived from the per-stage distribution. The balance score
is exposed both as a pure function (used by the donor economic
gate to project post-plan balance) and as a gauge-emitting wrapper
used at the post-cycle observability boundary.

::

    +--------------+         +---------------------+
    | compute_d_k  |  ---->  | compute_balance_    |
    | (per stage)  |         |   score (cluster)   |
    +--------------+         +---------+-----------+
                                       |
                                       v
                             +---------------------+
                             | emit_balance_score  |
                             | (cluster gauge      |
                             |  + scalar return)   |
                             +---------------------+
"""

import math
from collections.abc import Mapping

from ray.util.metrics import Gauge

# Cluster pipeline balance score (1.0 / max(1.0, heterogeneity_ratio)).
#
# Per-cluster cardinality - the balance score is a single scalar
# per cluster, so adding a ``stage`` tag would falsely multiply the
# cardinality without any new operator-relevant signal. Reusing one
# Cython handle across cycles keeps the per-cycle observation cost
# negligible (constructing a Gauge re-registers with the Ray
# metrics agent, which is 2-3 orders of magnitude slower).
_BALANCE_SCORE_GAUGE = Gauge(
    "xenna_scheduler_pipeline_balance_score",
    description=(
        "Cluster pipeline balance score = 1.0 / max(1.0, heterogeneity_ratio). "
        "Score = 1.0 means a perfectly balanced pipeline; scores approaching "
        "0 mean a single bottleneck stage dominates throughput. NaN means "
        "fewer than two stages had a finite D_k this cycle."
    ),
    tag_keys=("pipeline",),
)


def compute_d_k(service_time_s: float, effective_capacity: int) -> float:
    """Compute the actor-normalized Forced-Flow service demand ``D_k``.

    Linear-DAG pipeline - ``V_k = 1`` for every stage, so
    ``D_k = service_time_s / effective_capacity`` in seconds per
    channel. Cold-start contract: returns ``math.nan`` when either
    input fails finite-positive (no service-time sample, or
    no ready channels).

    """
    if not math.isfinite(service_time_s) or service_time_s <= 0.0:
        return math.nan
    if effective_capacity <= 0:
        return math.nan
    return service_time_s / effective_capacity


def compute_balance_score(d_k_by_stage: Mapping[str, float]) -> float:
    """Compute ``1.0 / max(1.0, max(D_k) / min(D_k))`` across stages with finite ``D_k``.

    Filters to finite positive ``D_k`` (cold-start stages
    excluded). Returns the balance score in ``(0.0, 1.0]`` when at
    least two stages qualify; ``math.nan`` otherwise so callers
    can treat NaN as "no verdict".

    """
    finite = [v for v in d_k_by_stage.values() if math.isfinite(v) and v > 0.0]
    if len(finite) < 2:
        return math.nan
    ratio = max(finite) / min(finite)
    return 1.0 / max(1.0, ratio)


def emit_balance_score(d_k_by_stage: Mapping[str, float], *, pipeline_name: str) -> float:
    """Emit ``xenna_scheduler_pipeline_balance_score`` and return the scalar.

    Computes the score via ``compute_balance_score``, observes it
    on ``_BALANCE_SCORE_GAUGE``, and hands the value back so the
    caller can use it for the per-cycle balance regression check
    without a second computation. Returns ``math.nan`` for
    cold-start cycles.

    """
    score = compute_balance_score(d_k_by_stage)
    _BALANCE_SCORE_GAUGE.set(score, tags={"pipeline": pipeline_name})
    return score
