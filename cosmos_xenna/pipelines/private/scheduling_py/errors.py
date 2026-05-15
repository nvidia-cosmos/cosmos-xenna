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

"""Internal exception hierarchy for the saturation-aware scheduler.

Distinct from ``ValueError`` (operator-supplied configuration errors)
and the allocator's ``AllocationError`` (recoverable cluster races):
exceptions defined here signal programming errors inside the scheduler
itself and should never be raised on operator input.
"""


class SchedulerInvariantError(RuntimeError):
    """Raised when an internal scheduler phase invariant is violated.

    Example invariants the scheduler enforces:

      * Worker-floor enforcement leaves a stage below its hard
        ``min_workers`` floor.
      * The combined per-cycle plan (additions plus deletions)
        exceeds cluster capacity in any single resource class.
      * The classifier transitions through an undefined state edge.

    These conditions indicate a scheduler bug, not an operator error.
    A caught ``SchedulerInvariantError`` should be treated as a
    must-fail signal: log, surface, and refuse to apply the plan.
    """
