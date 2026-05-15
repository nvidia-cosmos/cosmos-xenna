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

"""Tests for ``ProblemStageState`` slot-signal fields.

``ProblemStageState`` carries three optional runtime signals:

  * ``num_used_slots``     -- occupied task slots across all workers
  * ``num_empty_slots``    -- free task slots across all workers
  * ``input_queue_depth``  -- pre-batch tasks queued upstream

These tests pin three observable contracts at the data layer:

  1. The four-positional-argument constructor remains unchanged
     (existing call sites keep compiling and produce zeroed signals).
  2. Each new keyword-only argument round-trips through the Python
     wrapper into the underlying Rust struct.
  3. The Rust struct exposes the new fields as ``get_all`` / ``set_all``
     attributes so consumers can either read them after construction
     or mutate them when the streaming layer collects fresh signals.
"""

from cosmos_xenna.pipelines.private import data_structures


def _make_state(**kwargs: object) -> data_structures.ProblemStageState:
    """Build a minimal ``ProblemStageState``; kwargs override the defaults.

    Keeps each test free of unrelated setup so a single behaviour is
    being verified.
    """
    base: dict[str, object] = {
        "stage_name": "stage-x",
        "workers": [],
        "slots_per_worker": 2,
        "is_finished": False,
    }
    base.update(kwargs)
    return data_structures.ProblemStageState(**base)  # type: ignore[arg-type]


class TestPositionalArgumentBackwardCompatibility:
    """Existing four-arg call sites must keep compiling and produce zero signals."""

    def test_four_positional_args_still_construct(self) -> None:
        """The pre-existing positional signature continues to work."""
        state = data_structures.ProblemStageState("s", [], 2, False)
        assert state.rust.stage_name == "s"

    def test_signals_default_to_zero_when_omitted(self) -> None:
        """Omitted slot-signal kwargs map to a "no signal" sentinel of 0."""
        state = data_structures.ProblemStageState("s", [], 2, False)
        assert state.rust.num_used_slots == 0
        assert state.rust.num_empty_slots == 0
        assert state.rust.input_queue_depth == 0


class TestNumUsedSlotsRoundTrip:
    """``num_used_slots`` round-trips from the Python wrapper to the Rust struct."""

    def test_constructor_kwarg_is_visible_via_rust(self) -> None:
        """A populated Python kwarg shows up unchanged on the Rust object."""
        state = _make_state(num_used_slots=7)
        assert state.rust.num_used_slots == 7


class TestNumEmptySlotsRoundTrip:
    """``num_empty_slots`` round-trips from the Python wrapper to the Rust struct."""

    def test_constructor_kwarg_is_visible_via_rust(self) -> None:
        """A populated Python kwarg shows up unchanged on the Rust object."""
        state = _make_state(num_empty_slots=3)
        assert state.rust.num_empty_slots == 3


class TestInputQueueDepthRoundTrip:
    """``input_queue_depth`` round-trips from the Python wrapper to the Rust struct."""

    def test_constructor_kwarg_is_visible_via_rust(self) -> None:
        """A populated Python kwarg shows up unchanged on the Rust object."""
        state = _make_state(input_queue_depth=128)
        assert state.rust.input_queue_depth == 128


class TestExistingFieldsUnchanged:
    """Adding new fields must not perturb the existing four fields."""

    def test_stage_name_unchanged(self) -> None:
        """``stage_name`` is preserved when the new kwargs are also supplied."""
        state = _make_state(stage_name="captioning", num_used_slots=4)
        assert state.rust.stage_name == "captioning"

    def test_slots_per_worker_unchanged(self) -> None:
        """``slots_per_worker`` is preserved when the new kwargs are also supplied."""
        state = _make_state(slots_per_worker=8, num_empty_slots=4)
        assert state.rust.slots_per_worker == 8

    def test_is_finished_unchanged(self) -> None:
        """``is_finished`` is preserved when the new kwargs are also supplied."""
        state = _make_state(is_finished=True, input_queue_depth=2)
        assert state.rust.is_finished is True


class TestPostConstructionMutation:
    """``set_all`` lets streaming.py refresh signals each cycle without rebuilding the wrapper."""

    def test_num_used_slots_is_settable_after_construction(self) -> None:
        """Streaming will refresh ``num_used_slots`` each autoscale cycle."""
        state = _make_state()
        state.rust.num_used_slots = 5
        assert state.rust.num_used_slots == 5

    def test_num_empty_slots_is_settable_after_construction(self) -> None:
        """Streaming will refresh ``num_empty_slots`` each autoscale cycle."""
        state = _make_state()
        state.rust.num_empty_slots = 6
        assert state.rust.num_empty_slots == 6

    def test_input_queue_depth_is_settable_after_construction(self) -> None:
        """Streaming will refresh ``input_queue_depth`` each autoscale cycle."""
        state = _make_state()
        state.rust.input_queue_depth = 999
        assert state.rust.input_queue_depth == 999


class TestSignalsThroughProblemState:
    """``ProblemState`` carries multi-stage signals end-to-end."""

    def test_signals_propagate_through_multi_stage_problem_state(self) -> None:
        """Per-stage signals are preserved when stages are wrapped in a ``ProblemState``.

        Catches regressions where ``ProblemState`` rebuilds its inner
        Rust list but loses the slot-signal fields on the way.
        """
        s_a = _make_state(stage_name="s-a", num_used_slots=1, num_empty_slots=1, input_queue_depth=10)
        s_b = _make_state(stage_name="s-b", num_used_slots=4, num_empty_slots=0, input_queue_depth=0)
        ps = data_structures.ProblemState([s_a, s_b])
        signals = [(s.num_used_slots, s.num_empty_slots, s.input_queue_depth) for s in ps.rust.stages]
        assert signals == [(1, 1, 10), (4, 0, 0)]
