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


"""Tests for the hard-worker-cap final clamp.

The cap is the per-stage upper bound on workers, computed as
``min(max_workers, max_workers_per_node * num_nodes)``. It composes
with the existing per-stage floor (``min_workers``,
``min_workers_per_node``) so the full sandwich is

    floor <= effective_workers <= ceiling

Phase C clamps every ``try_add_worker`` call to ``ceiling - current``
and emits an INFO log when the request is bounded. Phase D forces a
shrink toward the ceiling when ``current > ceiling`` (e.g. an
operator just lowered the cap below the running worker count),
respecting the configured floor and per-cycle fraction clamp. Manual
stages and finished stages are skipped.

Test fixtures use pytest factory-fixture composition so each test
gets a fresh scheduler / problem / problem_state instance with
explicit dependencies. Module-level helpers ``_cluster`` and
``_make_config`` remain pure constructors (no scheduler state) and
are imported by the fixtures below.
"""

import logging
import sys
from collections.abc import Callable, Iterator
from unittest.mock import patch

import pytest
from loguru import logger as loguru_logger

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _test_stage_config(**overrides: object) -> SaturationAwareStageConfig:
    """Build a :class:`SaturationAwareStageConfig` with both warmup graces disabled.

    Cap tests focus on hard-cap clamps and the floor / ceiling
    sandwich, all orthogonal to the donor warmup grace. Pinning
    both grace fields to ``0.0`` keeps Phase D shrink and the
    saturation-mode donor unfiltered so the cap clamps remain the
    single binding constraint under test.
    """
    return SaturationAwareStageConfig(
        worker_warmup_measurement_grace_s=0.0,
        donor_warmup_grace_s=0.0,
        **overrides,  # type: ignore[arg-type]
    )


# Type aliases that document the factory-fixture call shapes.
SchedulerFactory = Callable[..., tuple[SaturationAwareScheduler, data_structures.Problem]]
ProblemStateFactory = Callable[..., data_structures.ProblemState]
AutoscaleFactory = Callable[
    [SaturationAwareScheduler, data_structures.ProblemState, dict[str, int]],
    data_structures.Solution,
]


def _cluster(*, total_cpus: int = 16, num_nodes: int = 1) -> resources.ClusterResources:
    """Build a CPU-only cluster with the requested topology."""
    return resources.ClusterResources(
        nodes={
            f"node-{index}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus,
                gpus=[],
                name=f"node-{index}",
            )
            for index in range(num_nodes)
        },
    )


def _make_config(
    *,
    max_workers: int | None = None,
    max_workers_per_node: int | None = None,
    min_workers: int = 1,
    min_workers_per_node: int | None = None,
    max_scale_down_fraction_per_cycle: float = 1.0,
) -> SaturationAwareConfig:
    """Build a default config with optional cap and floor configuration."""
    return SaturationAwareConfig(
        floor_stuck_grace_cycles=0,
        stage_defaults=_test_stage_config(
            min_workers=min_workers,
            min_workers_per_node=min_workers_per_node,
            max_workers=max_workers,
            max_workers_per_node=max_workers_per_node,
            max_scale_down_fraction_per_cycle=max_scale_down_fraction_per_cycle,
        ),
    )


# Pytest fixtures: shared logging bridge + factory fixtures.


@pytest.fixture
def loguru_caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Bridge loguru records into pytest's stdlib-based ``caplog`` fixture.

    Mirrors the pattern used in the sibling Phase C / Phase D test files.
    """
    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").log(msg.record["level"].no, msg.record["message"]),
        format="{message}",
    )
    caplog.set_level(logging.DEBUG, logger="loguru")
    try:
        yield caplog
    finally:
        loguru_logger.remove(handler_id)


@pytest.fixture
def make_scheduler() -> SchedulerFactory:
    """Build a fresh setup-completed scheduler with the requested topology and config.

    Returns a closure ``factory(stage_specs, *, cfg=None, num_nodes=1,
    total_cpus=16, stage_spec_overrides=None)`` so each test creates
    exactly the number of schedulers it needs, with full control over
    the cluster shape, stage configuration, and any
    ``StageSpec.saturation_aware`` overrides that streaming would
    inject through the constructor. Each invocation returns a
    brand-new ``SaturationAwareScheduler`` so internal state never
    leaks across tests or across multiple calls within one test.
    """

    def _factory(
        stage_specs: list[tuple[str, int | None]],
        *,
        cfg: SaturationAwareConfig | None = None,
        num_nodes: int = 1,
        total_cpus: int = 16,
        stage_spec_overrides: dict[str, SaturationAwareStageConfig] | None = None,
    ) -> tuple[SaturationAwareScheduler, data_structures.Problem]:
        if cfg is None:
            cfg = _make_config()
        cluster = _cluster(num_nodes=num_nodes, total_cpus=total_cpus)
        cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
        problem = data_structures.Problem(
            cluster,
            [
                data_structures.ProblemStage(
                    name=name,
                    stage_batch_size=1,
                    worker_shape=cpu_shape,
                    requested_num_workers=requested,
                    over_provision_factor=None,
                )
                for name, requested in stage_specs
            ],
        )
        scheduler = SaturationAwareScheduler(cfg, stage_spec_overrides=stage_spec_overrides)
        scheduler.setup(problem)
        return scheduler, problem

    return _factory


@pytest.fixture
def make_state() -> ProblemStateFactory:
    """Build a ``ProblemState`` from ``(stage_name, worker_ids, is_finished)`` rows.

    Returns a closure so each test composes the exact stage-state
    shape it needs. The returned ``ProblemState`` is a plain data
    container with no shared mutable state, so the factory pattern
    is purely a notational convenience here.
    """

    def _factory(
        stage_specs: list[tuple[str, list[str], bool]],
    ) -> data_structures.ProblemState:
        return data_structures.ProblemState(
            [
                data_structures.ProblemStageState(
                    stage_name=name,
                    workers=[
                        data_structures.ProblemWorkerGroupState.make(
                            worker_id,
                            [resources.WorkerResourcesInternal(node="node-0", cpus=1.0, gpus=[])],
                            num_used_slots=0,
                        )
                        for worker_id in worker_ids
                    ],
                    slots_per_worker=1,
                    is_finished=finished,
                )
                for name, worker_ids, finished in stage_specs
            ],
        )

    return _factory


@pytest.fixture
def autoscale_with_intents() -> AutoscaleFactory:
    """Run autoscale with the given intent deltas patched into the scheduler.

    The factory is stateless: each invocation patches
    ``intent_phase.compute`` for exactly one ``autoscale`` call and
    cleans up on exit. Tests that need multiple cycles call the
    factory once per cycle.
    """

    def _factory(
        scheduler: SaturationAwareScheduler,
        state: data_structures.ProblemState,
        intents: dict[str, int],
    ) -> data_structures.Solution:
        with patch(
            "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.intent_phase.IntentPhase._compute_intent_deltas",
            return_value=dict(intents),
        ):
            return scheduler.autoscale(time=0.0, problem_state=state)

    return _factory


class TestComputeStageCeilings:
    """Pin the per-stage ceiling resolution against config and topology.

    The resolver returns ``min(max_workers, max_workers_per_node *
    num_nodes)`` per stage, or ``None`` when neither cap is set.
    These tests exercise the resolver directly so config-resolution
    bugs surface independently of Phase C / Phase D wiring.
    """

    def test_no_cap_configured_returns_none_per_stage(self, make_scheduler: SchedulerFactory) -> None:
        """The default configuration leaves every stage uncapped."""
        scheduler, _ = make_scheduler([("A", None), ("B", None)])

        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=1)

        assert ceilings == {0: None, 1: None}

    def test_only_max_workers_returns_that_value(self, make_scheduler: SchedulerFactory) -> None:
        """``max_workers`` alone sets the ceiling at that integer."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))

        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=1)

        assert ceilings == {0: 4}

    def test_only_max_workers_per_node_multiplies_by_num_nodes(self, make_scheduler: SchedulerFactory) -> None:
        """``max_workers_per_node`` is multiplied by the cluster node count."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers_per_node=2), num_nodes=4)

        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=4)

        assert ceilings == {0: 8}

    def test_both_caps_picks_the_smaller(self, make_scheduler: SchedulerFactory) -> None:
        """The effective ceiling is the minimum of both caps when both are set."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=10, max_workers_per_node=1),
            num_nodes=4,
        )

        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=4)
        # max_workers=10, max_workers_per_node*num_nodes=1*4=4. min(10, 4) = 4.
        assert ceilings == {0: 4}

    def test_stage_spec_override_wins_at_runtime(self, make_scheduler: SchedulerFactory) -> None:
        """The runtime resolver honors ``StageSpec.saturation_aware`` above named overrides.

        ``stage_spec_overrides`` is injected through the
        ``SaturationAwareScheduler`` constructor (mirroring the wiring
        ``streaming.Autoscaler`` performs via
        ``_make_scheduler_algorithm``) and must outrank both
        ``per_stage_overrides`` and ``stage_defaults`` when the
        runtime resolver computes ceilings.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(max_workers=10),
            per_stage_overrides={"A": _test_stage_config(max_workers=8)},
        )
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=cfg,
            stage_spec_overrides={"A": _test_stage_config(max_workers=2)},
        )

        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=1)

        assert ceilings == {0: 2}

    def test_stage_spec_override_participates_in_anti_flap_validation(
        self,
        make_scheduler: SchedulerFactory,
    ) -> None:
        """Runtime overrides cannot weaken the cross-stage donor anti-flap invariant.

        With constructor injection the cross-validation runs eagerly
        inside ``SaturationAwareScheduler.__init__``, so a
        misconfigured pipeline fails fast at build time rather than
        mid-``autoscale()``. The fixture surfaces the constructor's
        ``ValueError`` directly through ``make_scheduler``.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            cross_stage_donor_anti_flap_cycles=3,
            stage_defaults=_test_stage_config(over_provisioned_streak_min_cycles=3),
        )

        with pytest.raises(ValueError, match=r"cross_stage_donor_anti_flap_cycles \(3\).*4"):
            make_scheduler(
                [("A", None)],
                cfg=cfg,
                stage_spec_overrides={"A": _test_stage_config(over_provisioned_streak_min_cycles=4)},
            )

    def test_max_workers_dominates_when_smaller_than_per_node_total(self, make_scheduler: SchedulerFactory) -> None:
        """``max_workers`` wins when smaller than the per-node total."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=2, max_workers_per_node=4),
            num_nodes=4,
        )

        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=4)
        # max_workers=2, max_workers_per_node*num_nodes=4*4=16. min(2, 16) = 2.
        assert ceilings == {0: 2}

    def test_called_before_setup_raises_runtime_error(self) -> None:
        """The helper refuses to run before ``setup()`` populates the pipeline."""
        scheduler = SaturationAwareScheduler(_make_config(max_workers=4))

        with pytest.raises(RuntimeError, match="runner read before setup"):
            _ = scheduler.runner


class TestFloorExceedsCeilingWarnsButNotClamped:
    """Pin floor-wins: a cross-term floor > ceiling is warned, never clamped.

    ``min_workers`` / ``min_workers_per_node`` and ``max_workers`` /
    ``max_workers_per_node`` mix different terms with the live node
    count, so ``__attrs_post_init__`` (same-term checks only) cannot
    catch e.g. ``min_workers=10`` with ``max_workers_per_node=2`` on one
    node, which yields ``floor=10 > ceiling=2``. The floor is a hard
    guarantee that wins over the softer ``max_workers`` policy cap:
    ``FloorCalculator`` returns the raw floor unchanged and emits a
    once-per-stage WARN. Phase D's ``allowed_by_floor`` bound then holds
    the stage at / above its floor (the end-to-end consequence is pinned
    by ``TestPhaseDCapForcedShrink.test_per_node_floor_blocks_shrink_
    even_when_cap_demands_it``, which fails if the clamp is reintroduced).
    """

    def test_floor_exceeding_ceiling_is_not_clamped(self, make_scheduler: SchedulerFactory) -> None:
        """``min_workers=10`` with a per-node ceiling of 2 keeps the floor at 10, not the cap."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(min_workers=10, max_workers_per_node=2),
            num_nodes=1,
        )

        floors = scheduler.runner.grow_services.floors.compute(num_nodes=1)
        ceilings = scheduler.runner.grow_services.ceilings.compute(num_nodes=1)

        # raw floor = max(10, 0) = 10; ceiling = min(2 * 1) = 2; floor wins -> 10 (unclamped).
        assert ceilings == {0: 2}
        assert floors == {0: 10}

    def test_floor_equal_to_ceiling_is_not_warned(
        self,
        make_scheduler: SchedulerFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The ``floor == ceiling`` boundary passes through unchanged and stays silent."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(min_workers=2, max_workers=2),
            num_nodes=1,
        )

        floors = scheduler.runner.grow_services.floors.compute(num_nodes=1)

        assert floors == {0: 2}
        warns = [record for record in loguru_caplog.records if "floor exceeds ceiling" in record.message]
        assert warns == [], "floor == ceiling is well-configured; no WARN expected"

    def test_floor_exceeds_ceiling_warns_once_naming_stage(
        self,
        make_scheduler: SchedulerFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The misconfig warns exactly once per stage (debounced), naming the stage, floor, and ceiling."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(min_workers=10, max_workers_per_node=2),
            num_nodes=1,
        )

        # Compute several times: the WARN must not thrash every cycle.
        for _ in range(3):
            scheduler.runner.grow_services.floors.compute(num_nodes=1)

        warns = [record for record in loguru_caplog.records if "floor exceeds ceiling" in record.message]
        assert len(warns) == 1, f"WARN must be debounced to once per stage, got {len(warns)}"
        msg = warns[0].message
        assert "'A'" in msg
        assert "floor 10" in msg
        assert "ceiling 2" in msg
        # Pin the floor-wins semantics: the stage runs above the cap (not clamped down).
        assert "takes precedence" in msg


class TestStageSpecOverrideConstructorContract:
    """Pin the constructor contract for ``stage_spec_overrides``.

    The constructor docstring promises that ``None`` and an empty
    mapping behave identically (no overrides installed, no validation
    performed). Pinning this with a focused unit test prevents a
    future ``is not None`` rewrite from silently breaking the
    empty-input fast path or from invoking
    ``validate_effective_stage_configs`` on an empty tuple in the hot
    construction path.
    """

    def test_none_and_empty_mapping_install_no_overrides(self) -> None:
        """``None`` and ``{}`` both leave ``_stage_spec_overrides`` empty."""
        cfg = _make_config()

        scheduler_none = SaturationAwareScheduler(cfg)
        scheduler_default = SaturationAwareScheduler(cfg, stage_spec_overrides=None)
        scheduler_empty = SaturationAwareScheduler(cfg, stage_spec_overrides={})

        assert scheduler_none._stage_spec_overrides == {}
        assert scheduler_default._stage_spec_overrides == {}
        assert scheduler_empty._stage_spec_overrides == {}

    def test_non_empty_mapping_is_copied_into_scheduler(self) -> None:
        """A non-empty override map is stored as a copy, not by reference."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(),
        )
        overrides = {"A": _test_stage_config(max_workers=2)}

        scheduler = SaturationAwareScheduler(cfg, stage_spec_overrides=overrides)

        assert scheduler._stage_spec_overrides == overrides
        # Mutating the caller's map after construction must not leak
        # into the scheduler; the constructor copies via ``dict(...)``.
        overrides["A"] = _test_stage_config(max_workers=99)
        assert scheduler._stage_spec_overrides["A"].max_workers == 2

    def test_stored_overrides_reject_post_construction_mutation(self) -> None:
        """The stored override view raises ``TypeError`` on direct mutation.

        Pinned because the override map is documented as immutable
        post-construction. The constructor wraps the copied dict in
        ``types.MappingProxyType`` so the runtime closure does not rely
        on underscore-prefix convention alone; this test guards against
        a future refactor that swaps the proxy back for a bare dict.
        Both the populated and empty-input paths produce a read-only
        view, so the assertion fires on both.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(),
        )

        scheduler_with_overrides = SaturationAwareScheduler(
            cfg,
            stage_spec_overrides={"A": _test_stage_config(max_workers=2)},
        )
        scheduler_empty = SaturationAwareScheduler(cfg)

        with pytest.raises(TypeError):
            scheduler_with_overrides._stage_spec_overrides["A"] = _test_stage_config(max_workers=99)  # type: ignore[index]
        with pytest.raises(TypeError):
            scheduler_empty._stage_spec_overrides["A"] = _test_stage_config(max_workers=99)  # type: ignore[index]


class TestPhaseCCapClamp:
    """Pin the contract that Phase C never grows past the configured cap.

    The Phase C path is the canonical scale-up driver. Capping
    ``try_add_worker`` calls is the central design contract of the
    hard worker cap; these tests pin every direction the clamp is
    expected to bind.

    Most tests seed each stage with one worker so Phase B's floor
    enforcement is a no-op and the asserted ``new_workers`` count
    purely reflects Phase C's response to the injected intent.
    """

    def test_intent_above_headroom_clamps_to_headroom(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Intent of +10 with headroom 3 results in exactly 3 grows (Phase C)."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        # Seed at floor: Phase B no-op; only Phase C contributes to ``new_workers``.
        # Headroom = 4 - 1 = 3, so intent (10) is clamped to 3.
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 10})

        assert len(solution.stages[0].new_workers) == 3

    def test_intent_at_headroom_grows_exactly_to_cap(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Intent equal to remaining headroom grows the full intent without clamping."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        # Headroom = 4 - 1 = 3; intent = 3; exact-boundary case (no INFO log).
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 3})

        assert len(solution.stages[0].new_workers) == 3

    def test_intent_below_headroom_grows_full_intent(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Intent below the cap is unaffected by the clamp."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=10))
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 3})

        assert len(solution.stages[0].new_workers) == 3

    def test_current_already_at_cap_zero_grow(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """A stage already at its cap drops every positive intent to zero grow."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 5})

        assert solution.stages[0].new_workers == []

    def test_cap_clamp_to_zero_headroom_resets_stuck_counter(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Hard cap clamping a positive intent to zero headroom clears the stuck counter.

        Pins the per-stage loop reset branch in
        ``phases.phase_c.run``: when ``intent_phase.compute`` produces
        a positive intent but the hard worker cap leaves zero headroom
        (``current >= ceiling``), the clamp routes through the inner
        ``if intent <= 0`` reset path and the per-stage stuck counter
        clears. Without this reset, a stage that hits its operator-set
        cap would carry a stale counter into the watchdog and
        eventually promote a misleading "stuck plan" INFO line.
        """
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        # Seed at cap (4 workers) so the headroom is zero and the cap clamp fires.
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])
        # Seed a prior-stuck history without rigging the cluster.
        scheduler.ledgers.stuck_plan.set("A", 7)

        autoscale_with_intents(scheduler, state, {"A": 5})

        assert scheduler.ledgers.stuck_plan.get_counter("A") == 0, (
            "the hard-cap zero-headroom branch must reset the stuck counter so the watchdog "
            "does not promote cap-bound stages to 'stuck plan' INFO lines."
        )

    def test_current_above_cap_phase_c_does_not_shrink(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Phase C never shrinks; over-cap state is left to Phase D's forced-shrink path."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 5})

        # Phase C never grows past the cap; Phase D handles the existing
        # excess via its forced-shrink path (covered in the next class).
        assert solution.stages[0].new_workers == []

    def test_per_node_cap_on_multi_node_cluster(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """``max_workers_per_node=1`` on a 4-node cluster grows up to 4 total (Phase C)."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers_per_node=1),
            num_nodes=4,
        )
        # Seed at floor=1; Phase C clamps intent (100) to (4 - 1) = 3.
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 100})

        assert len(solution.stages[0].new_workers) == 3

    def test_no_cap_grows_full_intent(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Without any cap the Phase C grow is bounded only by cluster placement."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config())
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 5})

        assert len(solution.stages[0].new_workers) == 5

    def test_clamp_emits_single_info_log_per_bound_stage(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The cap-bound message names the stage, intent, headroom, current, and ceiling."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", ["A-w0", "A-w1"], False)])

        autoscale_with_intents(scheduler, state, {"A": 10})

        cap_logs = [record for record in loguru_caplog.records if "hard worker cap" in record.message]
        assert len(cap_logs) == 1
        assert "intent +10" in cap_logs[0].message
        assert "current=2" in cap_logs[0].message
        assert "ceiling=4" in cap_logs[0].message

    def test_intent_within_headroom_emits_no_cap_log(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When intent fits within the headroom the cap stays silent.

        Pins that the cap INFO log is operator-actionable: it fires
        only when the cap actually clamps a request. Spurious logs on
        unbound intents would erode the signal value of the message.
        """
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=10))
        state = make_state([("A", ["A-w0", "A-w1"], False)])

        autoscale_with_intents(scheduler, state, {"A": 3})

        cap_logs = [record for record in loguru_caplog.records if "hard worker cap" in record.message]
        assert cap_logs == []

    def test_massive_intent_still_bounded_by_cap(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """A maliciously large intent does not bypass the cap."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": sys.maxsize})

        # Headroom = 4 - 1 = 3; clamped to 3 regardless of intent magnitude.
        assert len(solution.stages[0].new_workers) == 3

    def test_finished_stage_with_cap_skipped_in_phase_c(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Finished stages are skipped before the cap clamp evaluates intent."""
        scheduler, _ = make_scheduler(
            [("A", None), ("B", None)],
            cfg=_make_config(max_workers=4),
        )
        state = make_state([("A", ["A-w0"], False), ("B", [], True)])

        solution = autoscale_with_intents(scheduler, state, {"A": 2, "B": 5})

        assert len(solution.stages[0].new_workers) == 2
        assert solution.stages[1].new_workers == []

    def test_per_stage_caps_are_independent(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Each stage's cap is resolved independently of other stages."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=_test_stage_config(
                min_workers=1,
                max_scale_down_fraction_per_cycle=1.0,
            ),
            per_stage_overrides={
                "A": _test_stage_config(min_workers=1, max_workers=2),
                "B": _test_stage_config(min_workers=1, max_workers=6),
            },
        )
        scheduler, _ = make_scheduler([("A", None), ("B", None)], cfg=cfg)
        state = make_state([("A", ["A-w0"], False), ("B", ["B-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 10, "B": 10})

        # A: headroom = 2 - 1 = 1; clamped from 10 to 1.
        # B: headroom = 6 - 1 = 5; clamped from 10 to 5.
        assert len(solution.stages[0].new_workers) == 1
        assert len(solution.stages[1].new_workers) == 5


class TestPhaseDCapForcedShrink:
    """Pin the contract that Phase D shrinks toward the cap when current exceeds it.

    A cap is a "hard" cap. Operators may lower it mid-run; the
    scheduler must converge the worker count toward the new ceiling
    on subsequent cycles, respecting the configured floor and the
    per-cycle fraction clamp.
    """

    def test_current_above_cap_with_zero_intent_forces_shrink_to_cap(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """``current=8, cap=4, intent=0`` deletes 4 to reach the cap."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        assert len(solution.stages[0].deleted_workers) == 4

    def test_current_above_cap_with_negative_intent_takes_max_of_drivers(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """When intent shrink is smaller than cap excess, the cap dominates."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": -2})

        # max(|-2|, 8 - 4) = max(2, 4) = 4; cap dominates the smaller intent.
        assert len(solution.stages[0].deleted_workers) == 4

    def test_negative_intent_above_cap_excess_dominates_the_cap(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """When intent shrink exceeds cap excess, intent wins."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": -6})

        # max(|-6|, 8 - 4) = max(6, 4) = 6; intent dominates the smaller cap excess.
        assert len(solution.stages[0].deleted_workers) == 6

    def test_current_above_cap_with_positive_intent_forces_shrink_via_cap(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """A positive (grow) intent does not block Phase D's cap-driven shrink.

        The classifier may still request growth on a stage that already
        exceeds the cap (e.g. immediately after the operator lowered
        ``max_workers`` while saturation signals are stale). Phase C
        clamps the positive intent to zero (no headroom), and Phase D's
        forced-shrink driver pulls the count back to the cap on the
        same cycle. The ``max(-intent if intent < 0 else 0, ceiling_excess)``
        composition collapses to ``max(0, 4) = 4`` here.
        """
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 5})

        # Phase C grows nothing (current >= ceiling); Phase D shrinks to the cap.
        assert solution.stages[0].new_workers == []
        assert len(solution.stages[0].deleted_workers) == 4

    def test_fraction_cap_binds_below_cap_excess(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """The per-cycle fraction clamp slows convergence toward the hard cap."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=4, max_scale_down_fraction_per_cycle=0.25),
        )
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        # fraction_cap = max(1, floor(8 * 0.25)) = 2. min(4 cap-excess, 2 fraction) = 2.
        assert len(solution.stages[0].deleted_workers) == 2

    def test_per_node_floor_blocks_shrink_even_when_cap_demands_it(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """The configured floor wins over a stricter cap; never crosses ``min_workers``.

        Configured via ``min_workers_per_node`` (which is not constrained
        against ``max_workers`` by cross-field validation, unlike plain
        ``min_workers``). This pathological config models an operator who
        lowered the per-stage cap below the per-node floor mid-run.
        """
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=4, min_workers_per_node=6),
        )
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        # allowed_by_floor = 8 - max(1, 6 * 1) = 2; cap_excess = 8 - 4 = 4.
        # min(4, 2) = 2. The floor never lets the stage drop below 6.
        assert len(solution.stages[0].deleted_workers) == 2

    def test_cap_driven_shrink_fully_blocked_by_floor_logs_conflict(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A floor that blocks every cap-driven deletion still emits operator context."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=4, min_workers_per_node=8),
        )
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        assert solution.stages[0].deleted_workers == []
        deficit_logs = [record for record in loguru_caplog.records if "floor cap left 0 removed" in record.message]
        assert len(deficit_logs) == 1
        msg = deficit_logs[0].message
        assert "hard worker cap overflow requested 4 workers" in msg
        assert "deficit=4" in msg
        assert "current=8" in msg
        assert "floor=8" in msg
        assert "ceiling=4" in msg

    def test_current_at_cap_zero_shrink(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """``current == ceiling`` and zero intent leaves the stage untouched."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        assert solution.stages[0].deleted_workers == []

    def test_current_below_cap_with_positive_intent_no_shrink(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """A positive (grow) intent below the cap does not trigger Phase D."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", ["A-w0"], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 2})

        assert solution.stages[0].deleted_workers == []

    def test_no_cap_uses_pre_2ix_phase_d_behavior(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Without a cap, Phase D shrinks only on negative intent (no regression)."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config())
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])

        # Zero intent + no cap = no shrink: nothing drives a removal request.
        solution_zero = autoscale_with_intents(scheduler, state, {"A": 0})
        assert solution_zero.stages[0].deleted_workers == []

        # Negative intent + no cap = shrink driven by intent only.
        solution_neg = autoscale_with_intents(scheduler, state, {"A": -1})
        assert len(solution_neg.stages[0].deleted_workers) == 1

    def test_manual_stage_with_cap_is_not_shrunk_by_phase_d(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Manual stages remain operator-driven; the cap is not enforced via Phase D."""
        scheduler, _ = make_scheduler([("A", 8)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        assert solution.stages[0].deleted_workers == []

    def test_finished_stage_with_cap_excess_is_not_shrunk(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Finished stages bypass the forced-shrink path."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], True)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        assert solution.stages[0].deleted_workers == []

    def test_cap_driven_shrink_emits_distinct_info_log(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A cap-driven shrink without a deficit produces the cap-overflow log line."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        autoscale_with_intents(scheduler, state, {"A": 0})

        cap_logs = [record for record in loguru_caplog.records if "hard worker cap overflow" in record.message]
        assert len(cap_logs) == 1
        assert "ceiling=4" in cap_logs[0].message
        assert "current=8" in cap_logs[0].message
        assert "intent=0" in cap_logs[0].message

    def test_cap_driven_shrink_with_floor_deficit_attributes_to_cap_overflow(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A floor-bound deficit on a cap-driven request is logged as cap overflow.

        Before the log-discipline fix, a positive (or zero) classifier
        intent combined with a floor-blocked cap-overflow shrink was
        misleadingly logged as ``intent -X workers; floor cap left Y
        removed``, which suggested the classifier had requested the
        shrink. The driver was the operator-configured cap, not the
        classifier; the message must say so.
        """
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=4, min_workers_per_node=6),
        )
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        autoscale_with_intents(scheduler, state, {"A": 0})

        deficit_logs = [record for record in loguru_caplog.records if "floor cap left" in record.message]
        assert len(deficit_logs) == 1
        msg = deficit_logs[0].message
        assert "hard worker cap overflow requested 4 workers" in msg
        assert "intent -" not in msg
        assert "floor cap left 2 removed" in msg
        assert "ceiling=4" in msg

    def test_cap_driven_shrink_with_fraction_deficit_attributes_to_cap_overflow(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A fraction-bound deficit on a cap-driven request is logged as cap overflow.

        Same correction as the floor-deficit case: a fraction-clamped
        cap-overflow shrink must be attributed to the cap, not to a
        classifier ``intent -X`` that the classifier never produced.
        """
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers=4, max_scale_down_fraction_per_cycle=0.25),
        )
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        autoscale_with_intents(scheduler, state, {"A": 0})

        deficit_logs = [record for record in loguru_caplog.records if "per-cycle fraction cap left" in record.message]
        assert len(deficit_logs) == 1
        msg = deficit_logs[0].message
        assert "hard worker cap overflow requested 4 workers" in msg
        assert "intent -" not in msg
        assert "per-cycle fraction cap left 2 removed" in msg
        assert "ceiling=4" in msg

    def test_intent_driven_floor_deficit_keeps_intent_phrasing(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Pure classifier-intent shrinks (no cap overflow) keep the ``intent -X`` phrasing.

        Pins that the cap-attribution log distinction only rewrites
        cap-driven deficit logs. Pure intent-driven shrinks (no cap;
        classifier wants more deletes than the floor allows) preserve
        the ``intent -X workers; floor cap left Y removed`` phrasing
        so operators grepping for classifier-driven scale-down see a
        stable message.
        """
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(min_workers=2))
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])

        autoscale_with_intents(scheduler, state, {"A": -10})

        deficit_logs = [record for record in loguru_caplog.records if "floor cap left" in record.message]
        assert len(deficit_logs) == 1
        msg = deficit_logs[0].message
        assert "intent -10 workers" in msg
        assert "hard worker cap overflow" not in msg

    def test_floor_cap_equals_fraction_cap_logs_floor_branch(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When the floor and fraction clamps tie, the deficit log names the floor branch.

        Pins the operator-facing tie-break: the binding clamp is
        reported as ``floor cap`` whenever ``allowed_by_floor ==
        fraction_cap``. The branch chosen names the operator-set
        ``min_workers`` floor, which is the more actionable cause.
        """
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(min_workers=5, max_scale_down_fraction_per_cycle=0.375),
        )
        # current=8, allowed_by_floor = 8 - 5 = 3,
        # fraction_cap = max(1, floor(8 * 0.375)) = 3 (tie),
        # requested_remove = 5 (large negative intent),
        # actual_remove = min(5, 3, 3) = 3, deficit = 2.
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        autoscale_with_intents(scheduler, state, {"A": -5})

        floor_logs = [record for record in loguru_caplog.records if "floor cap left" in record.message]
        fraction_logs = [record for record in loguru_caplog.records if "per-cycle fraction cap left" in record.message]
        assert len(floor_logs) == 1
        assert fraction_logs == []
        msg = floor_logs[0].message
        assert "floor cap left 3 removed" in msg
        assert "deficit=2" in msg
        assert "floor=5" in msg

    def test_per_node_cap_on_multi_node_cluster_forces_shrink(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """``max_workers_per_node=1`` with 4 nodes shrinks 8 workers to 4."""
        scheduler, _ = make_scheduler(
            [("A", None)],
            cfg=_make_config(max_workers_per_node=1),
            num_nodes=4,
        )
        state = make_state([("A", [f"A-w{i}" for i in range(8)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": 0})

        assert len(solution.stages[0].deleted_workers) == 4


class TestCapMultiCycleStability:
    """Pin that the cap holds across long runs of saturated demand.

    A saturated stage capped at four workers grows up to the cap, then
    subsequent cycles with current already at the cap drop every
    positive intent to zero. These tests pin both base case and step
    case.
    """

    def test_first_cycle_with_zero_workers_grows_to_cap(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Base case: cycle 1 grows from 0 to ceiling (Phase B + Phase C combined)."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": sys.maxsize})

        # Phase B grows from 0 to floor (1); Phase C clamps the remaining intent
        # to (cap - current_after_B) = 3, landing exactly at the cap.
        assert len(solution.stages[0].new_workers) == 4

    def test_subsequent_cycle_at_cap_does_not_grow(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
    ) -> None:
        """Step case: cycle 2 with workers already at cap grows zero."""
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])

        solution = autoscale_with_intents(scheduler, state, {"A": sys.maxsize})

        assert solution.stages[0].new_workers == []

    def test_repeated_cap_clamp_does_not_thrash_the_intent_log(
        self,
        make_scheduler: SchedulerFactory,
        make_state: ProblemStateFactory,
        autoscale_with_intents: AutoscaleFactory,
        loguru_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Re-running autoscale at-cap with positive intent emits one log per cycle.

        The cap-bound INFO log fires whenever a positive intent EXCEEDS
        the remaining headroom. Once at the cap, every cycle's intent
        produces exactly one log line (deterministic, no duplicates).
        """
        scheduler, _ = make_scheduler([("A", None)], cfg=_make_config(max_workers=4))
        state = make_state([("A", [f"A-w{i}" for i in range(4)], False)])

        for _ in range(3):
            autoscale_with_intents(scheduler, state, {"A": 5})

        cap_logs = [record for record in loguru_caplog.records if "hard worker cap" in record.message]
        assert len(cap_logs) == 3
