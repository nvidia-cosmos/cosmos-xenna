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

"""Record-then-gate helper for the asymmetric stabilization window.

The per-stage state (the recommendation ring buffer) lives in
``state/recommendation_history.py``; this module owns only the
single record-then-gate helper used by
``StageDecisionPipeline.compute_recommendation``. Encapsulating the
two steps under one entry point means callers cannot accidentally
gate without recording, which would silently leave the buffer one
cycle behind reality.
"""

from cosmos_xenna.pipelines.private.scheduling_py.state.recommendation_history import RecommendationHistory


def apply_stabilization_gate(history: RecommendationHistory, raw_delta: int) -> int:
    """Record ``raw_delta`` into ``history`` and return the gated delta.

    Encapsulates record-then-gate so callers cannot accidentally
    gate without recording (which would silently leave the buffer
    one cycle behind reality).

    Args:
        history: Per-stage history. Mutated by the record step.
        raw_delta: This cycle's unclamped intent.

    Returns:
        ``raw_delta`` when the gate allows; ``0`` when it refuses
        or the recommendation is zero.

    """
    history.record(raw_delta)
    if raw_delta > 0:
        return raw_delta if history.gate_up_allowed() else 0
    if raw_delta < 0:
        return raw_delta if history.gate_down_allowed() else 0
    return 0


__all__ = ["apply_stabilization_gate"]
