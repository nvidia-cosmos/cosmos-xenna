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

"""Unit tests for the post-solve SolutionEditor (native-extension-free).

The editor is exercised against a duck-typed fake that mimics the
``solution.rust.stages`` surface, so the trim/cap/commit logic is verified
without building the native fragmentation solver.
"""

from collections.abc import Callable
from typing import cast

import pytest

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.solution_editor import SolutionEditor


class _FakeStage:
    def __init__(self, new_workers: list[str], deleted_workers: list[str]) -> None:
        self.new_workers = list(new_workers)
        self.deleted_workers = list(deleted_workers)


class _FakeRust:
    def __init__(self, stages: list[_FakeStage]) -> None:
        self._stages = stages
        self.write_count = 0

    @property
    def stages(self) -> list[_FakeStage]:
        return self._stages

    @stages.setter
    def stages(self, value: list[_FakeStage]) -> None:
        self._stages = value
        self.write_count += 1


class _FakeSolution:
    def __init__(self, stages: list[_FakeStage]) -> None:
        self.rust = _FakeRust(stages)


EditorFactory = Callable[[list[_FakeStage]], tuple[SolutionEditor, _FakeSolution]]


@pytest.fixture
def make_editor() -> EditorFactory:
    """Return a factory building a SolutionEditor over a fresh fake solution."""

    def _factory(stages: list[_FakeStage]) -> tuple[SolutionEditor, _FakeSolution]:
        solution = _FakeSolution(stages)
        return SolutionEditor(cast(data_structures.Solution, solution)), solution

    return _factory


def test_counts_reflect_buffered_state(make_editor: EditorFactory) -> None:
    editor, _ = make_editor([_FakeStage(["a", "b"], ["x"])])
    assert editor.stage_count == 1
    assert editor.proposed_new_workers(0) == 2
    assert editor.proposed_deletes(0) == 1


def test_trim_new_workers_shrinks_and_reports_change(make_editor: EditorFactory) -> None:
    editor, solution = make_editor([_FakeStage(["a", "b", "c"], [])])
    assert editor.trim_new_workers(0, 1) is True
    editor.commit()
    assert solution.rust.stages[0].new_workers == ["a"]


def test_trim_new_workers_within_cap_is_noop(make_editor: EditorFactory) -> None:
    editor, solution = make_editor([_FakeStage(["a"], [])])
    assert editor.trim_new_workers(0, 5) is False
    editor.commit()
    assert solution.rust.stages[0].new_workers == ["a"]


def test_cap_deletes_shrinks_and_reports_change(make_editor: EditorFactory) -> None:
    editor, solution = make_editor([_FakeStage([], ["x", "y", "z"])])
    assert editor.cap_deletes(0, 1) is True
    editor.commit()
    assert solution.rust.stages[0].deleted_workers == ["x"]


def test_commit_without_changes_does_not_rewrite_stage_list(make_editor: EditorFactory) -> None:
    editor, solution = make_editor([_FakeStage(["a"], ["x"])])
    original = solution.rust.stages
    editor.commit()
    assert solution.rust.stages is original
    assert solution.rust.write_count == 0


def test_commit_is_idempotent_after_a_change(make_editor: EditorFactory) -> None:
    editor, solution = make_editor([_FakeStage(["a", "b"], [])])
    editor.trim_new_workers(0, 1)
    editor.commit()
    editor.commit()
    assert solution.rust.write_count == 1
