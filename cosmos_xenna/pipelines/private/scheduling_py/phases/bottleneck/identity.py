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

"""Bottleneck identification and engagement-log debouncing functions.

The data types that describe the bottleneck identification result
(:class:`BottleneckIdentity`) and the engagement-streak ledger
(:class:`BottleneckEngagementState`) live in ``state/`` so the
``BottleneckSnapshot`` value object (consumed pipeline-wide) does
not have to import upward into ``phases/``. This module owns the
two pure functions that produce / consume those types:
:func:`identify_bottleneck` and
:func:`maybe_log_bottleneck_engagement`. The per-stage projection
of the bottleneck identity (engaged + upstream booleans) is
computed on-the-fly at the consumer call site as
:class:`StageTopologyContext`; no per-stage mirror is stored.

::

    d_k_by_stage --+--> identify_bottleneck ---> BottleneckIdentity
                   |                                    |
                   |                                    v
                   |                       maybe_log_bottleneck_engagement
                   |                                (with state)
                   |
                   +--> _pick_lex_stable_argmax (shared with heterogeneity)

The lex-stable argmax helper is shared with
``heterogeneity.compute_heterogeneity_ratio`` via
``_BOTTLENECK_NEAR_TIE_TOLERANCE`` so the operator-facing warn log
and the Phase C/D engagement gate always name the same stage on
near-ties.
"""

import math
import statistics
from collections.abc import Mapping

from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_engagement_state import BottleneckEngagementState
from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_identity import BottleneckIdentity
from cosmos_xenna.utils import python_log as logger

_BOTTLENECK_NEAR_TIE_TOLERANCE: float = 0.05
"""Default ``near_tie_tolerance`` for bottleneck stage selection.

Shared by :func:`identify_bottleneck` (the Phase C/D engagement
selector) and ``heterogeneity.compute_heterogeneity_ratio`` (the
operator-facing warn log) so both name the same stage when two or
more ``D_k`` values are within the tolerance band of the leader.
Drift between the two would surface as a warn log that points at
a different stage than the scheduler is acting on.
"""


def _pick_lex_stable_argmax(
    finite_scores: Mapping[str, float],
    *,
    near_tie_tolerance: float,
) -> str:
    """Pick the bottleneck stage with lex-stable near-tie resolution.

    Returns the lexicographically-smallest stage whose ``D_k`` is
    within ``near_tie_tolerance`` of the leader. Shared between
    ``identify_bottleneck`` and
    ``heterogeneity.compute_heterogeneity_ratio`` so Phase C/D
    engagement and the operator log name the same stage on
    near-ties.

    Raises:
        ValueError: ``finite_scores`` is empty or
            ``near_tie_tolerance`` is outside ``[0.0, 1.0)``.

    """
    if not finite_scores:
        msg = "finite_scores must contain at least one entry"
        raise ValueError(msg)
    if not 0.0 <= near_tie_tolerance < 1.0:
        msg = f"near_tie_tolerance must be in [0.0, 1.0), got {near_tie_tolerance}"
        raise ValueError(msg)

    max_d = max(finite_scores.values())
    near_tie_floor = max_d * (1.0 - near_tie_tolerance)
    return min(name for name, d in finite_scores.items() if d >= near_tie_floor)


def identify_bottleneck(
    d_k_by_stage: Mapping[str, float],
    *,
    heterogeneity_threshold: float,
    near_tie_tolerance: float = _BOTTLENECK_NEAR_TIE_TOLERANCE,
) -> BottleneckIdentity:
    """Identify the engaged bottleneck stage from actor-normalized ``D_k``.

    Ratio is ``max / median`` for ``n>=3`` and ``max / min`` for
    ``n=2``. Engagement requires at least two finite-positive
    samples and a ratio at or above ``heterogeneity_threshold``.
    Near-ties (within ``near_tie_tolerance`` of the leader) resolve
    by lexicographic ``stage_name`` for cycle-to-cycle stability.

    Raises:
        ValueError: ``heterogeneity_threshold`` non-finite or <= 1.0;
            ``near_tie_tolerance`` outside ``[0.0, 1.0)``.

    """
    if not math.isfinite(heterogeneity_threshold) or heterogeneity_threshold <= 1.0:
        msg = f"heterogeneity_threshold must be finite and > 1.0, got {heterogeneity_threshold!r}"
        raise ValueError(msg)
    if not 0.0 <= near_tie_tolerance < 1.0:
        msg = f"near_tie_tolerance must be in [0.0, 1.0), got {near_tie_tolerance}"
        raise ValueError(msg)

    finite_scores: dict[str, float] = {}
    for name, d_k in d_k_by_stage.items():
        if math.isfinite(d_k) and d_k > 0.0:
            finite_scores[name] = d_k

    if not finite_scores:
        return BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=math.nan,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

    max_d = max(finite_scores.values())
    if len(finite_scores) < 2:
        return BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=max_d,
            median_d_k=math.nan,
            heterogeneity_ratio=math.nan,
        )

    if len(finite_scores) == 2:
        min_d = min(finite_scores.values())
        median_d = (max_d + min_d) / 2.0
        ratio = max_d / min_d
    else:
        median_d = statistics.median(finite_scores.values())
        ratio = max_d / median_d if median_d > 0.0 else math.nan

    if not math.isfinite(ratio) or ratio < heterogeneity_threshold:
        return BottleneckIdentity(
            engaged=False,
            stage_name=None,
            max_d_k=max_d,
            median_d_k=median_d,
            heterogeneity_ratio=ratio,
        )

    return BottleneckIdentity(
        engaged=True,
        stage_name=_pick_lex_stable_argmax(
            finite_scores,
            near_tie_tolerance=near_tie_tolerance,
        ),
        max_d_k=max_d,
        median_d_k=median_d,
        heterogeneity_ratio=ratio,
    )


def maybe_log_bottleneck_engagement(
    *,
    identity: BottleneckIdentity,
    state: BottleneckEngagementState,
    persistence_cycles: int,
    pipeline_name: str,
) -> None:
    """Emit one INFO log when engagement has flipped persistently.

    Mutates ``state`` to debounce log spam at the gate boundary.

    Args:
        identity: This cycle's bottleneck identification.
        state: Mutable streak ledger owned by the scheduler.
        persistence_cycles: Cycles a candidate state must hold
            before announcement. Must be >= 1.
        pipeline_name: Pipeline identifier for the log line.
    """
    if state.last_announced is None:
        # Cold-start seeding: count only consecutive identical
        # candidate values. A noisy mixed sequence (True, False,
        # True, ...) must not seed ``last_announced`` based on
        # whichever value happens to land on the
        # ``persistence_cycles``'th cycle - the gate's contract is
        # persistence-based debouncing, not just "wait N cycles".
        if state.last_candidate is None or identity.engaged == state.last_candidate:
            state.candidate_streak += 1
        else:
            state.candidate_streak = 1
        state.last_candidate = identity.engaged
        if state.candidate_streak >= persistence_cycles:
            state.last_announced = identity.engaged
            state.candidate_streak = 0
        return

    if identity.engaged == state.last_announced:
        state.candidate_streak = 0
        return

    state.candidate_streak += 1
    if state.candidate_streak < persistence_cycles:
        return

    if identity.engaged:
        logger.info(
            f"bottleneck-priority decisions engaged for pipeline {pipeline_name!r}: "
            f"argmax stage {identity.stage_name!r} (D={identity.max_d_k:.2f}s, "
            f"heterogeneity_ratio={identity.heterogeneity_ratio:.2f})"
        )
    else:
        logger.info(
            f"bottleneck-priority decisions disengaged for pipeline {pipeline_name!r} "
            "(cluster homogeneous or cold-start)"
        )
    state.last_announced = identity.engaged
    state.candidate_streak = 0
