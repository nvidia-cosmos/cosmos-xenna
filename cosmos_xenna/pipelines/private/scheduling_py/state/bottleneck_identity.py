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

"""Per-cycle bottleneck identification result value object.

:class:`BottleneckIdentity` is produced by
``phases/bottleneck/identity.py::identify_bottleneck`` once per
cycle and stored on the :class:`BottleneckSnapshot` consumed by
Phase C / Phase D, the donor planning context, the post-cycle
reporter, and the per-stage topology projection
(:func:`project_stage_topology`). The value object lives in
``state/`` because :class:`BottleneckSnapshot` is in ``state/`` and
must import this type at construction time; keeping it under
``phases/bottleneck/`` would force an upward import from ``state/``.
"""

import attrs


@attrs.frozen
class BottleneckIdentity:
    """Per-cycle bottleneck identification.

    Produced by :func:`identify_bottleneck`. Consumers gate on
    ``engaged``; ``stage_name`` is the argmax stage when engaged.

    Attributes:
        engaged: True if at least two finite ``D_k`` samples exist
            and the heterogeneity ratio reaches the threshold.
        stage_name: Bottleneck stage when ``engaged``; otherwise None.
        max_d_k: Maximum finite ``D_k`` this cycle; NaN if none.
        median_d_k: Median (n>=3) or mean (n=2) of finite samples;
            NaN if fewer than two stages contribute.
        heterogeneity_ratio: ``max / median`` for n>=3, ``max / min``
            for n=2; NaN otherwise.
    """

    engaged: bool
    stage_name: str | None
    max_d_k: float
    median_d_k: float
    heterogeneity_ratio: float


__all__ = ["BottleneckIdentity"]
