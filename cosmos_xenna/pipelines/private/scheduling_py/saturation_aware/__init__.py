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

"""Saturation-aware streaming scheduler (pure-Python, built on the fragmentation solver).

A backlog-aware autoscaler that biases the proven fragmentation solver's
inputs and post-processes the solution it returns. It modifies no
fragmentation-scheduler code; it is selected per run via ``SchedulerKind``.

::

    config --> estimator --> sizing --> chain --> activity --> floor
    (knobs)    (speed +      (demand    (chain    (active      (scale-
               num_returns)   multiplier factors)  pipeline     down
                              m)                    stock)       floor)

``scheduler`` composes these around the shared solver each cycle. Only
``config`` is import-cheap (no native extension); the others are imported
by ``scheduler`` and the streaming seam, which already depend on the
native solver.
"""
