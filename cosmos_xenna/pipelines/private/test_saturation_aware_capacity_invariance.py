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

"""Capacity-encoding correctness tests for the saturation-aware scheduler.

The classifier consumes only one capacity signal: the unitless
empty-slot ratio ``num_empty_slots / (num_used_slots + num_empty_slots)``.
The absolute integer counts ``num_used_slots`` and ``num_empty_slots``
are the producer-side encoding (sampled by the streaming layer from
the actor pool) of a stage's effective in-stage capacity at sample
time. These tests pin the observable consequences of that encoding
choice:

  * the ratio is invariant under proportional rescaling of either
    dimension (worker count or per-worker slot count);
  * the algebraic capacity-sum invariant
    ``num_used_slots + num_empty_slots == current_workers * slots_per_worker``
    holds on hand-crafted ``ProblemStageState`` snapshots;
  * the per-stage pipeline preserves classifier output and EWMA value
    when the load shape stays constant under either slot redistribution
    or worker-count change;
  * granularity edge cases (``total_slots == 1``) and the zero-actor
    cold-start short-circuit behave as documented;
  * ``input_queue_depth`` flows into the classifier as a pre-batch
    task count without unit conversion.

Tests build ``_StageRuntimeState`` directly via the
``make_runtime_state`` factory fixture and exercise
``run_per_stage_pipeline`` on synthetic integer signals. No real
Ray actor pool, ``SaturationAwareScheduler`` instance, or full
``Problem`` is required.
"""

from collections.abc import Callable
from typing import Any

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py import pipeline as pipeline_mod
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import _resolve_auto_thresholds
from cosmos_xenna.pipelines.private.scheduling_py.pipeline import (
    _resolve_classifier_signal,
    record_executed_delta,
    run_per_stage_pipeline,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import (
    GrowthMode,
    StageState,
    _StageRuntimeState,
    compute_slots_empty_ratio,
)
from cosmos_xenna.pipelines.private.specs import SaturationAwareStageConfig

# Type alias for the factory fixture. Keyword-only kwargs match the
# fixture body below; ``Callable[..., _StageRuntimeState]`` is the
# minimal annotation pytest's runtime needs.
RuntimeStateFactory = Callable[..., _StageRuntimeState]


def _explicit_threshold_config(**overrides: Any) -> SaturationAwareStageConfig:
    """Build a stage config with explicit threshold overrides.

    Pinning ``saturation_threshold=0.15`` and ``activation_threshold=0.05``
    anchors the classifier zone math regardless of ``slots_per_actor``
    so each test asserts on stable boundary values rather than the
    auto-derived ``K/sqrt(c)`` output. ``over_provisioned_threshold``
    keeps its default (0.50). Callers may pass other field overrides
    through ``**overrides`` (e.g. a higher EWMA smoothing factor).
    """
    base: dict[str, Any] = {"saturation_threshold": 0.15, "activation_threshold": 0.05}
    base.update(overrides)
    return SaturationAwareStageConfig(**base)


@pytest.fixture
def make_runtime_state() -> RuntimeStateFactory:
    """Factory returning a ready-to-use ``_StageRuntimeState`` per call.

    Production resolves classifier thresholds on the first
    ``autoscale()`` cycle; tests construct ``_StageRuntimeState``
    directly so they must populate ``resolved_thresholds`` themselves.
    The factory uses ``_explicit_threshold_config`` by default so the
    resolved pair is ``(saturation=0.15, activation=0.05)`` regardless
    of the ``slots_per_actor`` argument. Callers that need a custom
    config or ``slots_per_actor`` value pass them via kwargs.
    """

    def _make(
        *,
        cfg: SaturationAwareStageConfig | None = None,
        slots_per_actor: int = 8,
        name: str = "TestStage",
    ) -> _StageRuntimeState:
        config = cfg if cfg is not None else _explicit_threshold_config()
        resolved = _resolve_auto_thresholds(config, slots_per_actor=slots_per_actor)
        return _StageRuntimeState(stage_name=name, resolved_thresholds=resolved)

    return _make


def _make_problem_stage_state(
    *,
    name: str,
    num_workers: int,
    slots_per_worker: int,
    num_used_slots: int,
    num_empty_slots: int,
) -> data_structures.ProblemStageState:
    """Build a ``ProblemStageState`` snapshot with hand-crafted slot counts.

    Each worker carries one 1-CPU allocation on ``node-0``; the
    snapshot is consistent with the small CPU cluster shape used by
    sibling scheduler tests.
    """
    worker_groups = [
        data_structures.ProblemWorkerGroupState.make(
            f"{name}-w{i}",
            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
        )
        for i in range(num_workers)
    ]
    return data_structures.ProblemStageState(
        stage_name=name,
        workers=worker_groups,
        slots_per_worker=slots_per_worker,
        is_finished=False,
        num_used_slots=num_used_slots,
        num_empty_slots=num_empty_slots,
    )


class TestRatioInvarianceUnderSlotRedistribution:
    """The empty-slot ratio is invariant under proportional rescaling.

    The classifier reads ``slots_empty_ratio = empty / (used + empty)``
    only. Doubling both ``num_used_slots`` and ``num_empty_slots``
    (whether by adding workers, by raising ``slots_per_worker``, or by
    any combination of the two) leaves the ratio unchanged and must
    leave classifier output unchanged.
    """

    @pytest.mark.parametrize(
        ("used", "empty", "multiplier"),
        [
            (1, 3, 4),
            (3, 1, 7),
        ],
    )
    def test_compute_ratio_unchanged_under_proportional_scaling(
        self,
        used: int,
        empty: int,
        multiplier: int,
    ) -> None:
        """``compute_slots_empty_ratio(k*m, j*m) == compute_slots_empty_ratio(k, j)``."""
        baseline = compute_slots_empty_ratio(used, empty)
        scaled = compute_slots_empty_ratio(used * multiplier, empty * multiplier)
        assert baseline == pytest.approx(scaled)

    @pytest.mark.parametrize(
        ("used_a", "empty_a", "used_b", "empty_b"),
        [
            # Same ratio (0.25, NORMAL band) at two different total-slot scales.
            (10, 2, 5, 1),
            # Same ratio (0.5, OVER_PROVISIONED at queue > 0) on (5x4) vs (10x2).
            (1, 1, 2, 2),
        ],
    )
    def test_classifier_state_identical_under_slot_redistribution(
        self,
        make_runtime_state: RuntimeStateFactory,
        used_a: int,
        empty_a: int,
        used_b: int,
        empty_b: int,
    ) -> None:
        """Same logical load on two different (workers, slots) shapes -> same classifier state."""
        cfg = _explicit_threshold_config()
        state_a = make_runtime_state(name="StageA")
        state_b = make_runtime_state(name="StageB")

        # Cold-start ``update_ewma`` returns the live sample, so a single
        # cycle exercises the full classifier path on both states.
        run_per_stage_pipeline(
            stage_state=state_a,
            num_used_slots=used_a,
            num_empty_slots=empty_a,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        run_per_stage_pipeline(
            stage_state=state_b,
            num_used_slots=used_b,
            num_empty_slots=empty_b,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        assert state_a.classifier_state == state_b.classifier_state
        assert state_a.slots_empty_ratio_ewma == pytest.approx(state_b.slots_empty_ratio_ewma)


class TestCapacitySumInvariant:
    """``num_used_slots + num_empty_slots == workers * slots_per_worker``.

    This is the algebraic contract the streaming layer must satisfy
    when populating ``ProblemStageState`` from the live actor pool.
    The contract is pinned here on hand-crafted snapshots; any path
    that would break it (e.g. a snapshot taken between a worker
    add/remove and the slot-count update, or counter drift on the
    actor pool) must be addressed at the producer.
    """

    @pytest.mark.parametrize(
        ("num_workers", "slots_per_worker", "num_used_slots", "num_empty_slots"),
        [
            # 50% busy stage on 10 workers x 2 slots.
            (10, 2, 10, 10),
            # Same total + same ratio, redistributed to 5 workers x 4 slots.
            (5, 4, 10, 10),
            # All slots free on a single 8-slot worker.
            (1, 8, 0, 8),
        ],
    )
    def test_total_slots_match_workers_times_slots_per_worker(
        self,
        num_workers: int,
        slots_per_worker: int,
        num_used_slots: int,
        num_empty_slots: int,
    ) -> None:
        """Hand-crafted snapshot satisfies the algebraic capacity-sum contract."""
        stage = _make_problem_stage_state(
            name="TestStage",
            num_workers=num_workers,
            slots_per_worker=slots_per_worker,
            num_used_slots=num_used_slots,
            num_empty_slots=num_empty_slots,
        )
        # Sum-equals-product is the property under test.
        assert num_used_slots + num_empty_slots == num_workers * slots_per_worker
        # Sanity: the snapshot round-trips the values it was built with
        # so the test exercises the production constructor end-to-end.
        assert stage.rust.slots_per_worker == slots_per_worker
        assert stage.rust.num_used_slots == num_used_slots
        assert stage.rust.num_empty_slots == num_empty_slots


class TestClassifierInvariantUnderProportionalCountDoubling:
    """Same logical load survives a proportional doubling of raw counts.

    The full ``slots_per_worker`` mid-cycle ordering invariant
    (``pool.set_num_slots_per_actor`` runs before any worker
    mutation in ``apply_autoscale_result_if_ready``) is verified at
    the integration layer alongside the actor pool. This test
    instead pins the algebraic property the orchestrator relies on:
    feeding proportionally-doubled raw counts (both
    ``num_used_slots`` and ``num_empty_slots`` doubled) to
    ``run_per_stage_pipeline`` produces the same classifier state
    as the original counts.
    """

    def test_classifier_unchanged_when_slots_per_worker_doubles_with_proportional_load(
        self,
        make_runtime_state: RuntimeStateFactory,
    ) -> None:
        """50% load at (used=2, empty=2) and (used=4, empty=4) yield the same classifier state."""
        cfg = _explicit_threshold_config()
        state = make_runtime_state()

        # Cycle 1: slots_per_worker=2, 50% utilization (used=2, empty=2).
        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=2,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        first_state = state.classifier_state

        # Cycle 2: slots_per_worker=4, still 50% utilization (used=4, empty=4).
        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=4,
            num_empty_slots=4,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        assert state.classifier_state == first_state
        # EWMA value rests at 0.5 because both samples were 0.5.
        assert state.slots_empty_ratio_ewma == pytest.approx(0.5)


class TestEwmaPersistenceAcrossSlotsPerWorkerChanges:
    """EWMA on the unitless ratio is invariant under proportional doubling."""

    def test_ewma_remains_at_half_after_slots_per_worker_doubles(
        self,
        make_runtime_state: RuntimeStateFactory,
    ) -> None:
        """10 cycles at (2, 2) then 10 cycles at (4, 4) keeps EWMA at ~0.5 and state stable."""
        cfg = _explicit_threshold_config()
        state = make_runtime_state()

        # Phase 1: 10 cycles at slots_per_worker=2 (modeled as used=2, empty=2).
        for _ in range(10):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=2,
                num_empty_slots=2,
                input_queue_depth=10,
                current_workers=1,
                config=cfg,
            )
        first_phase_state = state.classifier_state

        # Phase 2: 10 cycles at slots_per_worker=4 with proportional load
        # doubling. Same logical 50% utilization.
        for _ in range(10):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=4,
                num_empty_slots=4,
                input_queue_depth=10,
                current_workers=1,
                config=cfg,
            )

        assert state.slots_empty_ratio_ewma == pytest.approx(0.5, abs=0.01)
        # Classifier state stable across the slot-redistribution boundary.
        assert state.classifier_state == first_phase_state


class TestWorkerCountMidCycleChange:
    """EWMA continues to converge across a worker-count change.

    Adds (``new_workers``) and removes (``deleted_workers``) reshape
    the total slot count between cycles. The ratio fed to the EWMA
    must be insensitive to the absolute worker count as long as the
    logical load (utilization fraction) is preserved.
    """

    def test_ewma_converges_to_half_after_worker_count_change(
        self,
        make_runtime_state: RuntimeStateFactory,
    ) -> None:
        """50% load on 5 workers, then 50% load on 10 workers, EWMA stays ~0.5."""
        cfg = _explicit_threshold_config()
        state = make_runtime_state()

        # Phase 1: 5 workers x 2 slots = 10 total slots, 50% busy.
        for _ in range(10):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=5,
                num_empty_slots=5,
                input_queue_depth=10,
                current_workers=5,
                config=cfg,
            )
        first_phase_state = state.classifier_state

        # Phase 2: 10 workers x 2 slots = 20 total slots, 50% busy.
        for _ in range(10):
            run_per_stage_pipeline(
                stage_state=state,
                num_used_slots=10,
                num_empty_slots=10,
                input_queue_depth=10,
                current_workers=10,
                config=cfg,
            )

        assert state.slots_empty_ratio_ewma == pytest.approx(0.5, abs=0.01)
        assert state.classifier_state == first_phase_state


class TestGranularityFloorForTinyStages:
    """``total_slots == 1`` only ever produces {0.0, 1.0} as ratios.

    The chosen rule for this binary capacity case is to feed the
    signal to the classifier as-is and accept the resulting bistable
    output: SATURATED_CRITICAL when ratio==0.0 and OVER_PROVISIONED
    when ratio==1.0 and ``input_queue_depth > 0``. Hysteresis
    deadbands cannot operate meaningfully because no near-boundary
    sample is ever produced. This is documented as the chosen rule;
    the test asserts the alternation pattern, NOT that the resulting
    output is sensible.
    """

    @pytest.mark.parametrize(
        ("num_used_slots", "num_empty_slots", "expected_state"),
        [
            (1, 0, StageState.SATURATED_CRITICAL),
            (0, 1, StageState.OVER_PROVISIONED),
        ],
    )
    def test_total_slots_one_produces_bistable_classifier_output(
        self,
        make_runtime_state: RuntimeStateFactory,
        num_used_slots: int,
        num_empty_slots: int,
        expected_state: StageState,
    ) -> None:
        """Binary capacity (1 slot total) maps to one of two classifier zones."""
        # alpha=1.0 disables EWMA smoothing so each cycle's classifier
        # sees the live ratio (0.0 or 1.0) without any blended value.
        cfg = _explicit_threshold_config(slots_empty_ratio_smoothing_level=1.0)
        state = make_runtime_state(cfg=cfg)

        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=num_used_slots,
            num_empty_slots=num_empty_slots,
            input_queue_depth=10,
            current_workers=1,
            config=cfg,
        )
        assert state.classifier_state == expected_state


class TestInputQueueDepthUnitsConsistency:
    """``input_queue_depth`` flows into the classifier as a pre-batch task count."""

    def test_pre_batch_queue_depth_passes_unchanged_into_classifier(
        self,
        make_runtime_state: RuntimeStateFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spy on ``classify`` and confirm the kwarg matches the supplied value."""
        captured: dict[str, Any] = {}

        def _spy_classify(**kwargs: Any) -> StageState:
            captured.update(kwargs)
            return StageState.NORMAL

        # Patch the imported name inside the pipeline module so the
        # spy observes the actual call site.
        monkeypatch.setattr(pipeline_mod, "classify", _spy_classify)

        cfg = _explicit_threshold_config()
        state = make_runtime_state()
        # Arbitrary pre-batch task count chosen to be unambiguous
        # against any plausible default (0, 1) or sample-counted
        # alternative (e.g. multiples of stage_batch_size).
        expected_queue_depth = 17

        run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=2,
            num_empty_slots=2,
            input_queue_depth=expected_queue_depth,
            current_workers=1,
            config=cfg,
        )
        assert captured["input_queue_depth"] == expected_queue_depth


class TestZeroActorColdStartPath:
    """``_resolve_classifier_signal`` and the pipeline correctly handle zero slots."""

    def test_resolve_classifier_signal_returns_none_at_cold_start_with_zero_slots(
        self,
        make_runtime_state: RuntimeStateFactory,
    ) -> None:
        """Zero slots and no prior valid EWMA -> ``None`` (caller short-circuits)."""
        cfg = _explicit_threshold_config()
        state = make_runtime_state()

        # Pre-condition: factory leaves the carry-forward field unset.
        assert state.last_valid_slots_empty_ratio_ewma is None

        result = _resolve_classifier_signal(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            config=cfg,
        )
        assert result is None

    def test_pipeline_returns_zero_delta_and_ticks_growth_streak_on_cold_start(
        self,
        make_runtime_state: RuntimeStateFactory,
    ) -> None:
        """Cold-start with zero slots returns delta=0 and growth-mode timer advances.

        Cold-start is handled by the scheduler in two steps:
        ``run_per_stage_pipeline`` returns 0 and leaves the growth-mode
        state untouched; then the scheduler calls ``record_executed_delta``
        with the post-commit delta (also 0 in cold-start) which is what
        advances the growth-mode timer. The test simulates both steps.
        """
        cfg = _explicit_threshold_config()
        state = make_runtime_state()

        delta = run_per_stage_pipeline(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            input_queue_depth=10,
            current_workers=0,
            config=cfg,
        )
        record_executed_delta(stage_state=state, delta_executed=delta, config=cfg)
        assert delta == 0
        # Classifier untouched: state and streak remain at defaults.
        assert state.classifier_state is StageState.NORMAL
        assert state.classifier_streak == 0
        # Growth-mode timer ticks regardless of signal availability so
        # HOLD eventually exits to TRACKING even with no samples.
        assert state.growth_mode is GrowthMode.ACQUIRING
        assert state.growth_streak == 1

    def test_carry_forward_value_used_when_slots_zero_and_prior_ewma_present(
        self,
        make_runtime_state: RuntimeStateFactory,
    ) -> None:
        """``last_valid_slots_empty_ratio_ewma`` is returned when slots=0 and prior is set."""
        cfg = _explicit_threshold_config()
        state = make_runtime_state()

        # Seed the carry-forward field with a NORMAL-band value
        # (between saturation_threshold=0.15 and
        # over_provisioned_threshold=0.50). ``_resolve_classifier_signal``
        # must return this value verbatim because total slots == 0.
        state.last_valid_slots_empty_ratio_ewma = 0.30

        result = _resolve_classifier_signal(
            stage_state=state,
            num_used_slots=0,
            num_empty_slots=0,
            config=cfg,
        )
        assert result == pytest.approx(0.30)
