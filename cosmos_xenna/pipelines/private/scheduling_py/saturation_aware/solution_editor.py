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

"""Typed editor for post-solve worker-delta overrides on a ``Solution``.

The fragmentation solver returns a ``Solution`` whose native stage list reads
back as a fresh clone, so a mutation only persists when the whole list is
written back. This editor reads the list once, applies trims/caps to the
buffer in stage-index and count terms, and writes it back a single time on
commit - keeping the scheduler free of ``solution.rust`` mutation details.
"""

from typing import Any

import attrs

from cosmos_xenna.pipelines.private import data_structures


@attrs.define
class SolutionEditor:
    """Buffer per-stage worker-delta overrides and commit them in one write.

    Attributes:
        solution: The solver's solution; mutated in place on :meth:`commit`.
    """

    _solution: data_structures.Solution
    # Native StageSolution objects (FFI boundary); read once, written back once.
    _stages: list[Any] = attrs.field(init=False)
    _changed: bool = attrs.field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        """Read the native stage list into the edit buffer."""
        self._stages = list(self._solution.rust.stages)

    @property
    def stage_count(self) -> int:
        """Return the number of stages in the solution."""
        return len(self._stages)

    def proposed_new_workers(self, stage_index: int) -> int:
        """Return the buffered new-worker count for a stage."""
        return len(self._stages[stage_index].new_workers)

    def proposed_deletes(self, stage_index: int) -> int:
        """Return the buffered deletion count for a stage."""
        return len(self._stages[stage_index].deleted_workers)

    def trim_new_workers(self, stage_index: int, keep: int) -> bool:
        """Keep at most ``keep`` new workers for a stage.

        Returns:
            True if the new-worker list shrank, False if it was already within
            ``keep``.
        """
        stage = self._stages[stage_index]
        current = list(stage.new_workers)
        if keep >= len(current):
            return False
        stage.new_workers = current[: max(0, keep)]
        self._changed = True
        return True

    def cap_deletes(self, stage_index: int, max_deletes: int) -> bool:
        """Keep at most ``max_deletes`` deletions for a stage.

        Returns:
            True if the deletion list shrank, False if it was already within
            ``max_deletes``.
        """
        stage = self._stages[stage_index]
        current = list(stage.deleted_workers)
        if max_deletes >= len(current):
            return False
        stage.deleted_workers = current[: max(0, max_deletes)]
        self._changed = True
        return True

    def commit(self) -> None:
        """Write the edit buffer back to the native solution once, if it changed."""
        if self._changed:
            self._solution.rust.stages = self._stages
            self._changed = False
