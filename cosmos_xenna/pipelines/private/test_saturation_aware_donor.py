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

"""Behaviour tests for cross-stage donor selection in floor mode.

Two layers are pinned:

  1. The pure helper ``select_youngest_eligible_donor`` -- floor
     preservation, upstream preference, youngest-first ordering, and
     the deterministic worker-id tiebreaker.
  2. End-to-end donor fallback through
     ``SaturationAwareScheduler.autoscale``: cluster-full bootstrap
     succeeds when a donor is available, raises with operator
     context when every other stage is at its own floor, and ignores
     classifier-state / HOLD / per-donor budgets (none of which are
     applied in floor mode).
"""

from typing import cast
from unittest.mock import patch

import pytest

from cosmos_xenna.pipelines.private import data_structures, resources
from cosmos_xenna.pipelines.private.scheduling_py.donor import (
    DonorCandidate,
    select_youngest_eligible_donor,
)
from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import SaturationAwareScheduler
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


def _cluster(*, num_nodes: int = 1, total_cpus_per_node: int = 16) -> resources.ClusterResources:
    """Multi-node cluster sized to fit each test fixture exactly."""
    return resources.ClusterResources(
        nodes={
            f"node-{i}": resources.NodeResources(
                used_cpus=0,
                total_cpus=total_cpus_per_node,
                gpus=[],
                name=f"node-{i}",
            )
            for i in range(num_nodes)
        },
    )


def _problem(
    stage_specs: list[tuple[str, int | None]],
    *,
    num_nodes: int = 1,
    total_cpus_per_node: int = 16,
) -> data_structures.Problem:
    """Build a ``Problem`` whose stages carry the given ``requested_num_workers``."""
    cluster = _cluster(num_nodes=num_nodes, total_cpus_per_node=total_cpus_per_node)
    cpu_shape = resources.Resources(cpus=1.0).to_worker_shape(cluster)
    stages = [
        data_structures.ProblemStage(
            name=name,
            stage_batch_size=1,
            worker_shape=cpu_shape,
            requested_num_workers=requested,
            over_provision_factor=None,
        )
        for name, requested in stage_specs
    ]
    return data_structures.Problem(cluster, stages)


def _make_workers(
    stage_name: str,
    count: int,
    *,
    num_nodes: int = 1,
) -> list[data_structures.ProblemWorkerGroupState]:
    """Build ``count`` 1-CPU workers spread round-robin across nodes."""
    return [
        data_structures.ProblemWorkerGroupState.make(
            f"{stage_name}-w{i}",
            [resources.WorkerResourcesInternal(node=f"node-{i % num_nodes}", cpus=1.0, gpus=[])],
        )
        for i in range(count)
    ]


def _problem_state(
    stage_specs: list[tuple[str, int, int, bool]],
    *,
    num_nodes: int = 1,
) -> data_structures.ProblemState:
    """Build a ``ProblemState`` from ``(name, num_workers, slots_per_worker, is_finished)``."""
    states = [
        data_structures.ProblemStageState(
            stage_name=name,
            workers=_make_workers(name, num_workers, num_nodes=num_nodes),
            slots_per_worker=slots,
            is_finished=finished,
        )
        for name, num_workers, slots, finished in stage_specs
    ]
    return data_structures.ProblemState(states)


class TestSelectYoungestEligibleDonor:
    """Pure helper: ``(age ASC, worker_id ASC)`` with floor preservation and upstream preference."""

    def test_no_eligible_donor_returns_none(self) -> None:
        """Every other stage at its own floor -> no donor available."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: 2, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={},
        )
        assert donor is None

    def test_floor_preservation_blocks_donor_at_exact_floor(self) -> None:
        """A donor with current == floor cannot drop further; stage A skipped."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: 3, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1", "A-w2"], ["B-w0"]],
            worker_ages={},
        )
        # A is at exact floor; B is the receiver. Nothing else qualifies.
        assert donor is None

    def test_upstream_donor_preferred_over_downstream_when_both_eligible(self) -> None:
        """Stage 0 is upstream of stage 2 (the receiver); stage 4 is downstream.

        Both have spare workers; the helper returns the upstream donor.
        """
        donor = select_youngest_eligible_donor(
            receiver_stage_index=2,
            stage_floors={0: 1, 1: 99, 2: 1, 4: 1},
            worker_ids_by_stage=[
                ["upstream-w0", "upstream-w1"],
                [],
                ["receiver-w0"],
                [],
                ["downstream-w0", "downstream-w1"],
            ],
            worker_ages={},
        )
        assert donor is not None
        assert donor.stage_index == 0

    def test_downstream_donor_used_when_no_upstream_eligible(self) -> None:
        """Receiver is stage 0 (no upstream exists); downstream stage donates."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=0,
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0"], ["B-w0", "B-w1"]],
            worker_ages={},
        )
        assert donor is not None
        assert donor.stage_index == 1
        assert donor.worker_id == "B-w0"

    def test_youngest_age_wins_within_eligible_pool(self) -> None:
        """Multiple eligible upstream candidates; the youngest is selected."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=2,
            stage_floors={0: 1, 1: 1, 2: 1},
            worker_ids_by_stage=[
                ["A-w0", "A-w1"],
                ["B-w0", "B-w1"],
                [],
            ],
            worker_ages={"A-w0": 10, "A-w1": 7, "B-w0": 2, "B-w1": 9},
        )
        assert donor is not None
        assert donor.worker_id == "B-w0"
        assert donor.age == 2

    def test_age_tie_falls_back_to_worker_id_order(self) -> None:
        """Two candidates with the same age fall back to ``worker_id`` ASC."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=2,
            stage_floors={0: 1, 1: 1, 2: 1},
            worker_ids_by_stage=[["B-w1", "A-w0"], [], []],
            worker_ages={},
        )
        assert donor is not None
        assert donor.worker_id == "A-w0"

    def test_missing_floor_defaults_to_one(self) -> None:
        """A stage missing from ``stage_floors`` is assumed to have floor=1."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={},  # Nothing set
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={},
        )
        assert donor is not None
        assert donor.stage_index == 0

    def test_returns_donor_candidate_with_full_metadata(self) -> None:
        """The returned ``DonorCandidate`` exposes stage_index, worker_id, and age."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: 1, 1: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0"]],
            worker_ages={"A-w0": 5, "A-w1": 3},
        )
        assert donor == DonorCandidate(stage_index=0, worker_id="A-w1", age=3)


class TestPhaseBDonorFallback:
    """End-to-end donor fallback through ``SaturationAwareScheduler.autoscale``."""

    def test_cluster_full_bootstrap_donates_then_receiver_grows(self) -> None:
        """A receiver at zero workers reaches its floor by donating from a peer.

        Cluster has room for 4 1-CPU workers; stage A holds all 4
        while its own floor is 1. Stage B starts at 0 and reaches its
        floor by receiving one donated worker.
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False), ("B", 0, 1, False)]),
        )
        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1

    def test_manual_request_is_soft_for_floor_donor(self) -> None:
        """Floor bootstrap may donate a manual stage down to its configured floor."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", 4), ("B", None)], total_cpus_per_node=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False), ("B", 0, 1, False)]),
        )

        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1

    def test_scheduler_persists_worker_ages_for_youngest_donor(self) -> None:
        """End-to-end donor selection uses cross-cycle ages, not worker-id order."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None), ("C", None)], total_cpus_per_node=4))

        scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 2, 1, False), ("B", 1, 1, False), ("C", 1, 1, False)]),
        )
        scheduler.autoscale(
            time=1.0,
            problem_state=_problem_state([("A", 3, 1, False), ("B", 1, 1, False), ("C", 0, 1, True)]),
        )
        solution = scheduler.autoscale(
            time=2.0,
            problem_state=_problem_state([("A", 3, 1, False), ("B", 0, 1, False), ("C", 1, 1, False)]),
        )

        assert [worker.id for worker in solution.stages[0].deleted_workers] == ["A-w2"]
        assert len(solution.stages[1].new_workers) == 1

    def test_donor_floor_preservation_blocks_donation(self) -> None:
        """When every other stage is at its own floor, donor fallback fails.

        Cluster has room for 2 1-CPU workers; both held by stage A
        whose ``min_workers=2`` (its own floor). Stage B needs 1 worker
        but A cannot donate without dropping below its floor. Grace=0
        forces an immediate raise on the first failed cycle so the
        contract-pinned RuntimeError is observable here.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"A": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=2))
        with pytest.raises(RuntimeError, match="no eligible cross-stage donor"):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 2, 1, False), ("B", 0, 1, False)]),
            )

    def test_truly_infeasible_min_workers_raises_immediately(self) -> None:
        """A receiver whose floor exceeds total cluster capacity raises.

        Cluster has room for 2 1-CPU workers; stage A is the only stage and
        its ``min_workers=5`` cannot fit. No donor candidates exist. Grace=0
        bypasses the stuck-window so the immediate-raise contract is visible.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=5),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=2))
        with pytest.raises(RuntimeError, match="target_min=5"):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 0, 1, False)]),
            )

    def test_no_donor_message_lists_no_eligible_clause(self) -> None:
        """The error message names the no-eligible-donor branch explicitly."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"A": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=2))
        with pytest.raises(RuntimeError, match=r"no eligible cross-stage donor"):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 2, 1, False), ("B", 0, 1, False)]),
            )

    def test_no_donor_failure_warns_during_grace_then_raises(self) -> None:
        """A no-donor floor miss is tolerated only inside the grace window."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=1,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"A": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=2))
        state = _problem_state([("A", 2, 1, False), ("B", 0, 1, False)])

        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning") as warning:
            solution = scheduler.autoscale(time=0.0, problem_state=state)

        assert solution.stages[0].deleted_workers == []
        assert solution.stages[1].new_workers == []
        assert any("minimum-worker floor stuck (1/1 grace cycles)" in call.args[0] for call in warning.call_args_list)

        with pytest.raises(RuntimeError, match=r"no eligible cross-stage donor"):
            scheduler.autoscale(time=1.0, problem_state=state)

    def test_partial_floor_progress_resets_stuck_counter(self) -> None:
        """Direct growth resets the stuck counter even when the floor remains short."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=1,
            stage_defaults=SaturationAwareStageConfig(min_workers=5),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=4))

        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning") as warning:
            solution = scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 0, 1, False)]),
            )

        assert len(solution.stages[0].new_workers) == 4
        assert scheduler._floor_stuck_counters == {}
        assert any("minimum-worker floor partially satisfied" in call.args[0] for call in warning.call_args_list)

        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning") as warning:
            scheduler.autoscale(
                time=1.0,
                problem_state=_problem_state([("A", 4, 1, False)]),
            )

        assert any("minimum-worker floor stuck (1/1 grace cycles)" in call.args[0] for call in warning.call_args_list)
        with pytest.raises(RuntimeError, match=r"target_min=5"):
            scheduler.autoscale(
                time=2.0,
                problem_state=_problem_state([("A", 4, 1, False)]),
            )

    def test_floor_mode_uses_floor_only_for_donor_eligibility(self) -> None:
        """Floor mode does not apply saturation-mode anti-flap filters.

        Stage A is non-manual with extra capacity; nothing about its
        runtime state would block donation in floor mode. Stage B is
        also non-manual and needs 1 worker. A donates to B (no
        anti-flap layers apply in floor mode).
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=4))
        # A starts at 4 (above its implicit floor of 1); B starts at 0.
        # Cluster is full; only donation rescues B.
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 4, 1, False), ("B", 0, 1, False)]),
        )
        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1

    def test_upstream_donor_preferred_in_full_cluster_bootstrap(self) -> None:
        """When both upstream and downstream stages can donate, upstream wins.

        Cluster has 4 CPUs. Stage A (upstream, non-manual, current=2, floor=1)
        and stage C (downstream, non-manual, current=2, floor=1) both have spare
        workers. Stage B (the receiver, current=0, floor=1) sits between them.
        The donation comes from A.
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None), ("C", None)], total_cpus_per_node=4))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state(
                [("A", 2, 1, False), ("B", 0, 1, False), ("C", 2, 1, False)],
            ),
        )
        # Receiver B grows by 1; donor was upstream A (one of A's workers removed).
        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1
        assert solution.stages[2].deleted_workers == []


class TestSelectYoungestEligibleDonorAdversarial:
    """Pathological / boundary inputs to the pure helper."""

    def test_zero_stages_returns_none(self) -> None:
        """An empty problem yields no candidates."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=0,
            stage_floors={},
            worker_ids_by_stage=[],
            worker_ages={},
        )
        assert donor is None

    def test_receiver_stage_index_out_of_range_does_not_pick_self(self) -> None:
        """If the receiver index is out of range, no stage is excluded; donor still picks correctly."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=99,  # Out of range; the helper does not validate this.
            stage_floors={0: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"]],
            worker_ages={},
        )
        # The helper does not raise on out-of-range receiver; it treats every stage as eligible.
        # This is documented behaviour: the caller is expected to pass a valid receiver index.
        assert donor is not None
        assert donor.stage_index == 0

    def test_floor_zero_allows_drain_to_empty(self) -> None:
        """A floor of 0 allows the donor stage to be fully drained.

        The helper does not enforce a positive floor; it is the caller's
        responsibility to set ``stage_floors`` to safe values.
        """
        donor = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: 0, 1: 1},
            worker_ids_by_stage=[["A-w0"], []],
            worker_ages={},
        )
        # 1 - 1 = 0 >= 0, eligible.
        assert donor is not None
        assert donor.stage_index == 0

    def test_negative_floor_treated_as_floor_zero(self) -> None:
        """Negative floors are not clamped; the comparison still works correctly."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=1,
            stage_floors={0: -5, 1: 1},
            worker_ids_by_stage=[["A-w0"], []],
            worker_ages={},
        )
        # 1 - 1 = 0 >= -5, eligible.
        assert donor is not None

    def test_huge_age_values_sort_correctly(self) -> None:
        """Workers with very large ages still sort correctly against younger ones."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=2,
            stage_floors={0: 1, 1: 1, 2: 1},
            worker_ids_by_stage=[["A-w0", "A-w1"], ["B-w0", "B-w1"], []],
            worker_ages={"A-w0": 10**18, "A-w1": 10**18, "B-w0": 1, "B-w1": 5},
        )
        assert donor is not None
        assert donor.worker_id == "B-w0"
        assert donor.age == 1

    def test_many_stages_many_workers_picks_globally_youngest(self) -> None:
        """A stress fixture with 50 stages and 20 workers each picks the globally youngest worker."""
        worker_ids_by_stage = [[f"S{s}-w{w}" for w in range(20)] for s in range(50)]
        worker_ages = {f"S{s}-w{w}": (s * 20 + w + 1) for s in range(50) for w in range(20)}
        # Inject one worker with age 0 in stage 25.
        worker_ages["S25-w0"] = 0
        # Receiver is stage 49 (last); stage 25 is upstream; floor=1 everywhere.
        donor = select_youngest_eligible_donor(
            receiver_stage_index=49,
            stage_floors={s: 1 for s in range(50)},
            worker_ids_by_stage=worker_ids_by_stage,
            worker_ages=worker_ages,
        )
        assert donor is not None
        assert donor.worker_id == "S25-w0"
        assert donor.age == 0

    def test_empty_inner_lists_skip_candidates(self) -> None:
        """A stage with no workers cannot be a donor regardless of floor."""
        donor = select_youngest_eligible_donor(
            receiver_stage_index=2,
            stage_floors={0: 0, 1: 0, 2: 1},
            worker_ids_by_stage=[[], [], []],
            worker_ages={},
        )
        # All empty -> no candidates.
        assert donor is None


class TestPhaseBDonorFallbackAdversarial:
    """Adversarial scheduler-integration tests for the donor fallback path."""

    def test_repeated_donations_in_one_iteration_succeed(self) -> None:
        """A receiver needing two workers from the same donor can take both within one cycle.

        Cluster has 8 CPUs; stage A holds 8 (floor=1, can donate down to 1);
        stage B (receiver, floor=3) starts at 0. The loop should iterate
        three times: try_add fails -> donate from A -> retry succeeds.
        Repeats until B has 3 workers.
        """
        cfg = SaturationAwareConfig(
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"B": SaturationAwareStageConfig(min_workers=3)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=8))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state([("A", 8, 1, False), ("B", 0, 1, False)]),
        )
        assert len(solution.stages[0].deleted_workers) == 3
        assert len(solution.stages[1].new_workers) == 3

    def test_donor_floor_blocks_after_repeated_donations(self) -> None:
        """Repeated donations stop when the donor reaches its own floor.

        Cluster has 4 CPUs; stage A holds 4 (floor=1; can give 3);
        stage B (receiver, floor=5) starts at 0. The loop donates 3 times
        until A is at its floor, then has no donor and raises.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"B": SaturationAwareStageConfig(min_workers=5)},
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=4))
        with pytest.raises(RuntimeError, match=r"no eligible cross-stage donor"):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 4, 1, False), ("B", 0, 1, False)]),
            )

    def test_donation_sequence_two_receivers(self) -> None:
        """Two non-manual receivers in one cycle each pull from the same donor.

        Cluster has 5 CPUs; stage A (donor) holds 5; stages B and C
        (receivers) both start at 0 with floor=1. Each receiver pulls
        exactly one donation; A ends at 3.
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None), ("C", None)], total_cpus_per_node=5))
        solution = scheduler.autoscale(
            time=0.0,
            problem_state=_problem_state(
                [("A", 5, 1, False), ("B", 0, 1, False), ("C", 0, 1, False)],
            ),
        )
        assert len(solution.stages[0].deleted_workers) == 2
        assert len(solution.stages[1].new_workers) == 1
        assert len(solution.stages[2].new_workers) == 1

    def test_donation_logs_at_info_level_with_structured_format(self) -> None:
        """A successful donation emits exactly one INFO line with the expected fields."""
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=4))
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.info") as info:
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 4, 1, False), ("B", 0, 1, False)]),
            )
        # The donor INFO log fires exactly once for the single donation; pre-resolve
        # logs (per-stage threshold INFO) also fire, so filter for the donor line.
        donor_calls = [
            call
            for call in info.call_args_list
            if "cross-stage minimum-floor donor accepted" in (call.args[0] if call.args else "")
        ]
        assert len(donor_calls) == 1
        msg = donor_calls[0].args[0]
        assert "donor_stage_index=0" in msg
        assert "donor_age=" in msg

    def test_no_eligible_donor_message_includes_remediation_hint(self) -> None:
        """The no-donor error message tells the operator how to recover."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=0,
            stage_defaults=SaturationAwareStageConfig(min_workers=10),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=2))
        with pytest.raises(RuntimeError, match=r"Reduce min_workers / min_workers_per_node"):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 0, 1, False)]),
            )

    def test_retry_failure_names_donor_without_logging_success(self) -> None:
        """A failed post-donation retry raises immediately, even when grace is enabled."""
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("donor", None), ("receiver", None)]))

        class _RetryExhaustedContext:
            """Fake planner where donor removal works but receiver placement still fails."""

            def __init__(self) -> None:
                self._worker_ids_by_stage = [["donor-w0", "donor-w1"], []]

            def worker_ids_by_stage(self) -> list[list[str]]:
                return self._worker_ids_by_stage

            def worker_ages(self) -> dict[str, int]:
                return {"donor-w0": 0, "donor-w1": 10}

            def try_add_worker(self, stage_index: int) -> data_structures.ProblemWorkerGroupState | None:
                del stage_index
                return None

            def try_remove_worker(self, stage_index: int, worker_id: str) -> bool:
                self._worker_ids_by_stage[stage_index].remove(worker_id)
                return True

        expected = r"post-donation retry returned no placement .*donor_stage_index=0, donor_worker_id='donor-w0'"
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.info") as info:
            with pytest.raises(RuntimeError, match=expected):
                scheduler._run_phase_b_floor(
                    cast(data_structures.AutoscalePlanContext, _RetryExhaustedContext()),
                    _problem_state([("donor", 2, 1, False), ("receiver", 0, 1, False)]),
                )

        donor_success_logs = [
            call
            for call in info.call_args_list
            if "cross-stage minimum-floor donor accepted" in (call.args[0] if call.args else "")
        ]
        assert donor_success_logs == []

    def test_finished_donor_stage_not_floor_enforced_but_still_donatable(self) -> None:
        """A finished stage is skipped by floor enforcement but its workers can still donate.

        Whether finished stages are eligible donors is a contract decision: today they ARE
        eligible because their floor still applies (their workers are not actively processing
        but they remain registered). This test pins that contract.
        """
        cfg = SaturationAwareConfig(stage_defaults=SaturationAwareStageConfig(min_workers=1))
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=2))
        solution = scheduler.autoscale(
            time=0.0,
            # A is finished but has 2 workers; B needs 1 worker; cluster full.
            problem_state=_problem_state([("A", 2, 1, True), ("B", 0, 1, False)]),
        )
        # B receives one donated worker from A even though A is finished.
        assert len(solution.stages[0].deleted_workers) == 1
        assert len(solution.stages[1].new_workers) == 1


class TestPersistWorkerAgesDefensive:
    """The cross-cycle worker-age persistence path tolerates partial maps."""

    def test_age_persistence_falls_back_to_zero_for_missing_id(self) -> None:
        """A worker id present in worker_ids_by_stage but absent from worker_ages defaults to age 0.

        This pins the defensive `.get(worker_id, 0)` in _persist_worker_ages
        so that a Rust-side invariant break does not raise KeyError mid-cycle.
        """
        scheduler = SaturationAwareScheduler(SaturationAwareConfig())
        scheduler.setup(_problem([("A", None)]))

        class _PartialContext:
            """Mock context whose worker_ages is missing one of the live ids."""

            def worker_ids_by_stage(self) -> list[list[str]]:
                return [["A-w0", "A-w1"]]

            def worker_ages(self) -> dict[str, int]:
                return {"A-w0": 5}  # A-w1 missing

        scheduler._persist_worker_ages(cast(data_structures.AutoscalePlanContext, _PartialContext()))
        assert scheduler._worker_ages == {"A-w0": 5, "A-w1": 0}


class TestFloorStuckGraceWindow:
    """Per-stage stuck-cycle counter and the ``floor_stuck_grace_cycles`` knob."""

    @staticmethod
    def _build_unsatisfiable_scheduler(*, grace: int) -> SaturationAwareScheduler:
        """Build a scheduler whose stage A floor cannot be satisfied this cycle.

        ``min_workers=5`` on a 2-CPU cluster forces every floor-enforcement
        attempt to fail (no donor exists; cluster placement exhausted).
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=grace,
            stage_defaults=SaturationAwareStageConfig(min_workers=5),
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=2))
        return scheduler

    @staticmethod
    def _problem_state_unsatisfiable() -> data_structures.ProblemState:
        return _problem_state([("A", 2, 1, False)])

    def test_grace_zero_raises_on_first_failed_cycle(self) -> None:
        """``floor_stuck_grace_cycles=0`` restores the legacy immediate-raise contract."""
        scheduler = self._build_unsatisfiable_scheduler(grace=0)
        with pytest.raises(RuntimeError, match=r"target_min=5"):
            scheduler.autoscale(time=0.0, problem_state=self._problem_state_unsatisfiable())

    def test_grace_window_swallows_first_failures_and_warns(self) -> None:
        """A stuck stage warns each cycle and does not raise while inside the grace window."""
        scheduler = self._build_unsatisfiable_scheduler(grace=3)
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning") as warning:
            for _ in range(3):
                scheduler.autoscale(time=0.0, problem_state=self._problem_state_unsatisfiable())
        # Counter increments once per stuck cycle.
        assert scheduler._floor_stuck_counters == {"A": 3}
        stuck_warnings = [
            call
            for call in warning.call_args_list
            if "minimum-worker floor stuck" in (call.args[0] if call.args else "")
        ]
        assert len(stuck_warnings) == 3

    def test_grace_window_raises_on_grace_plus_one_cycle(self) -> None:
        """The ``RuntimeError`` fires on the ``grace + 1``-th consecutive failed cycle."""
        scheduler = self._build_unsatisfiable_scheduler(grace=3)
        # First 3 cycles inside the grace window: warn but do not raise.
        for _ in range(3):
            scheduler.autoscale(time=0.0, problem_state=self._problem_state_unsatisfiable())
        # The 4th cycle (counter would become 4 > 3) raises.
        with pytest.raises(RuntimeError, match=r"target_min=5"):
            scheduler.autoscale(time=0.0, problem_state=self._problem_state_unsatisfiable())

    def test_counter_resets_on_successful_donation(self) -> None:
        """A successful add (direct or via donor) resets the per-stage stuck counter.

        Build a scheduler that fails for a few cycles (cluster full + no donor),
        then introduce a donor stage and run a final cycle that succeeds via
        donation. The counter must clear to zero.
        """
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=10,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={
                "A": SaturationAwareStageConfig(min_workers=2),
                "B": SaturationAwareStageConfig(min_workers=2),
            },
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=2))
        cfg_b_pinned = _problem_state([("A", 0, 1, False), ("B", 2, 1, False)])

        # Cycle 1: A stuck (A floor=2; B floor=2; B at floor; no donor).
        scheduler.autoscale(time=0.0, problem_state=cfg_b_pinned)
        assert scheduler._floor_stuck_counters.get("A", 0) == 1

        # Cycle 2: same -- counter increments.
        scheduler.autoscale(time=0.0, problem_state=cfg_b_pinned)
        assert scheduler._floor_stuck_counters.get("A", 0) == 2

        # Now relax B's floor so it can donate; A's floor still 2; cycle 3 succeeds.
        cfg_relaxed = SaturationAwareConfig(
            floor_stuck_grace_cycles=10,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={"A": SaturationAwareStageConfig(min_workers=2)},
        )
        scheduler_relaxed = SaturationAwareScheduler(cfg_relaxed)
        scheduler_relaxed.setup(_problem([("A", None), ("B", None)], total_cpus_per_node=2))
        scheduler_relaxed._floor_stuck_counters = {"A": 5}  # Pre-loaded counter.
        # Cycle 3: A floor=2; B floor=1; B has 2 workers -> can donate one.
        scheduler_relaxed.autoscale(time=0.0, problem_state=cfg_b_pinned)
        # After the successful donation, the counter MUST reset.
        assert scheduler_relaxed._floor_stuck_counters.get("A", 0) == 0

    def test_per_stage_counters_are_independent(self) -> None:
        """A stuck receiver's counter stays isolated from skipped stages."""
        cfg = SaturationAwareConfig(
            floor_stuck_grace_cycles=10,
            stage_defaults=SaturationAwareStageConfig(min_workers=1),
            per_stage_overrides={
                "A": SaturationAwareStageConfig(min_workers=10),  # always stuck
            },
        )
        scheduler = SaturationAwareScheduler(cfg)
        scheduler.setup(_problem([("A", None), ("B", 0)], total_cpus_per_node=4))
        # A stays stuck (floor 10 on a full 4-CPU cluster); B is manual and skipped.
        for _ in range(3):
            scheduler.autoscale(
                time=0.0,
                problem_state=_problem_state([("A", 4, 1, False), ("B", 0, 1, False)]),
            )
        assert scheduler._floor_stuck_counters == {"A": 3}
        # B's counter never appears in the dict because manual stages are skipped by Phase B.
        assert "B" not in scheduler._floor_stuck_counters

    def test_setup_clears_counter_on_re_initialization(self) -> None:
        """Calling ``setup`` again clears any stuck counters from the prior run."""
        scheduler = self._build_unsatisfiable_scheduler(grace=10)
        scheduler.autoscale(time=0.0, problem_state=self._problem_state_unsatisfiable())
        assert scheduler._floor_stuck_counters == {"A": 1}
        # Re-setup with a fresh problem -- counters should reset.
        scheduler.setup(_problem([("A", None)], total_cpus_per_node=2))
        assert scheduler._floor_stuck_counters == {}

    def test_warning_message_includes_counter_progress(self) -> None:
        """The WARNING message reports current/total grace cycles for operator visibility."""
        scheduler = self._build_unsatisfiable_scheduler(grace=5)
        with patch("cosmos_xenna.pipelines.private.scheduling_py.saturation_aware.logger.warning") as warning:
            scheduler.autoscale(time=0.0, problem_state=self._problem_state_unsatisfiable())
        stuck_warning = next(
            (call for call in warning.call_args_list if "minimum-worker floor stuck" in call.args[0]),
            None,
        )
        assert stuck_warning is not None
        msg = stuck_warning.args[0]
        assert "(1/5 grace cycles)" in msg
        assert "target_min=5" in msg
        assert "no eligible cross-stage donor" in msg
