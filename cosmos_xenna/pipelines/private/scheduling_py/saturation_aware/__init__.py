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

    estimator --> sizing  --> chain  --> activity --> floor
    (speed +      (demand    (fan-out  (active      (scale-
     num_returns)  multiplier factors)  stock)       down floor)

``scheduler`` composes these around the read-only solver each cycle, using
``shape`` for the static per-stage layout and ``solution_editor`` to apply its
overrides to the returned ``Solution``. Per-cycle runtime signals arrive through
the scheduler-agnostic ``scheduling_py.runtime_signals`` seam. ``config`` and
``shape`` are native-free; ``solution_editor`` and ``scheduler`` depend on the
native solver.
"""
