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

"""Cluster-wide heterogeneity ratio gauge + persistence-debounced warn log.

The ratio ``max_k D_k / min_k D_k`` is a single per-cluster scalar
that quantifies how dominated by one bottleneck the pipeline is.
:class:`HeterogeneityWarnState` (in ``state/heterogeneity_state.py``)
debounces the operator-facing INFO log so a brief heterogeneity
spike does not produce log spam, and a sustained heterogeneity
above the configured threshold produces exactly one INFO line until
the cluster recovers.

::

    d_k_by_stage --+--> compute_heterogeneity_ratio --+--> _HETEROGENEITY_RATIO_GAUGE
                   |    (mutates state)               |
                   |                                  +--> conditional INFO log
                   |                                       (via _pick_lex_stable_argmax)
                   |
                   +--> (no other consumers)
"""

import math
from collections.abc import Mapping

from ray.util.metrics import Gauge

from cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.identity import (
    _BOTTLENECK_NEAR_TIE_TOLERANCE,
    _pick_lex_stable_argmax,
)
from cosmos_xenna.pipelines.private.scheduling_py.state.heterogeneity_state import HeterogeneityWarnState
from cosmos_xenna.utils import python_log as logger

# Module-level Gauge for the cluster-wide ratio. ``tag_keys`` is
# ``("pipeline",)`` only - the ratio is a single scalar per cluster,
# so adding a ``stage`` tag would falsely multiply the cardinality
# without any new operator-relevant signal. Reusing one Cython
# handle across cycles keeps the per-cycle observation cost
# negligible (constructing a Gauge re-registers with the Ray
# metrics agent, which is 2-3 orders of magnitude slower).
_HETEROGENEITY_RATIO_GAUGE = Gauge(
    "xenna_scheduler_cluster_heterogeneity_ratio",
    description=(
        "Cluster heterogeneity ratio max(D_k) / min(D_k) across stages "
        "whose Forced-Flow service demand was observed this cycle. "
        "Ratio = 1.0 means a perfectly homogeneous pipeline; ratios "
        "much above 1.0 indicate a single bottleneck stage that "
        "dominates pipeline throughput. NaN samples mean fewer than "
        "two stages had a finite D_k this cycle (cold-start or "
        "all-but-one cold-start)."
    ),
    tag_keys=("pipeline",),
)


def compute_heterogeneity_ratio(
    *,
    d_k_by_stage: Mapping[str, float],
    pipeline_name: str,
    state: HeterogeneityWarnState,
    warn_threshold: float,
    warn_streak_cycles: int,
) -> None:
    """Emit the cluster heterogeneity ratio gauge and an optional INFO log.

    Computes ``ratio = max_k D_k / min_k D_k`` across stages with
    finite positive ``D_k``. Fewer than two finite stages -> ratio
    is ``nan`` and streak resets. Mutates ``state`` in place: at or
    below threshold the streak resets and the ``has_fired`` latch
    clears; above threshold the streak increments and the INFO log
    fires exactly once when it reaches ``warn_streak_cycles``. Pure
    observability.

    Raises:
        ValueError: ``warn_threshold`` non-finite or <= 1.0; or
            ``warn_streak_cycles`` not a positive integer.

    """
    if not math.isfinite(warn_threshold) or warn_threshold <= 1.0:
        msg = f"pipeline {pipeline_name!r}: warn_threshold must be finite and > 1.0, got {warn_threshold!r}"
        raise ValueError(msg)
    if not isinstance(warn_streak_cycles, int) or isinstance(warn_streak_cycles, bool) or warn_streak_cycles < 1:
        msg = f"pipeline {pipeline_name!r}: warn_streak_cycles must be an integer >= 1, got {warn_streak_cycles!r}"
        raise ValueError(msg)

    finite_scores: dict[str, float] = {}
    for stage_name, d_k in d_k_by_stage.items():
        if math.isfinite(d_k) and d_k > 0.0:
            finite_scores[stage_name] = d_k

    if len(finite_scores) < 2:
        # Undefined ratio: emit NaN to keep cardinality stable, reset
        # streak (a cold-start cluster has no heterogeneity verdict).
        # ``has_fired`` is intentionally NOT cleared - only a real
        # drop to or below the threshold re-arms a fresh INFO log.
        _HETEROGENEITY_RATIO_GAUGE.set(
            math.nan,
            tags={"pipeline": pipeline_name},
        )
        state.streak_cycles = 0
        return

    max_d = max(finite_scores.values())
    min_d = min(finite_scores.values())
    ratio = max_d / min_d
    _HETEROGENEITY_RATIO_GAUGE.set(
        ratio,
        tags={"pipeline": pipeline_name},
    )

    if ratio <= warn_threshold:
        state.streak_cycles = 0
        state.has_fired = False
        return

    state.streak_cycles += 1
    if state.streak_cycles < warn_streak_cycles or state.has_fired:
        return

    bottleneck_name = _pick_lex_stable_argmax(
        finite_scores,
        near_tie_tolerance=_BOTTLENECK_NEAR_TIE_TOLERANCE,
    )
    bottleneck_score = finite_scores[bottleneck_name]
    logger.info(
        f"high cluster heterogeneity (ratio={ratio:.1f} for "
        f"{state.streak_cycles} cycles); consider raising "
        f"over_provisioned_streak_min_cycles for stage "
        f"{bottleneck_name!r} (bottleneck D={bottleneck_score:.1f}s) "
        f"to give it more recovery margin"
    )
    state.has_fired = True
