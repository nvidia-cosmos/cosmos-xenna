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

"""Cross-stage donor selection.

Two modes ship from this module.

Floor mode (``select_youngest_eligible_donor``) is used by Phase B
floor enforcement when the cluster is full and the receiver cannot
reach its minimum-worker floor through fresh placement. It picks the
youngest worker from any non-receiver stage that can spare one
without violating its own floor; upstream donors are preferred when
any are eligible.

Saturation mode (``find_saturation_donor``) is used by Phase C
saturation-driven scale-up when the cluster is full and the receiver
wants to grow because its classifier signals SATURATED. Selection is
more conservative because operator pressure is weaker than in the
floor case; five anti-flap layers reject donors that would
oscillate.

The non-negotiable donor-floor preservation rule applies in both
modes: a stage whose live worker count minus one would drop below
its floor is filtered out, preventing a single donation from
cascading into another stage's bootstrap.
"""

import operator

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.state import GrowthMode, StageState, _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@attrs.frozen
class DonorWorker:
    """One worker removal in a multi-donor plan.

    Carries the planner indices needed to translate the symbolic
    choice into ``ctx.probe_add_after_removals`` and
    ``ctx.remove_workers_atomically`` arguments, plus the
    deterministic age tiebreaker the resource-fit search shares
    with the floor selector.

    Attributes:
        stage_index: Position of the donor stage in problem order.
        worker_id: Planner-assigned worker identifier on that stage.
        age: Worker age in autoscale cycles. Older workers are
            preferred during deterministic tie-breaking; freshly
            observed workers default to ``0``.

    """

    stage_index: int
    worker_id: str
    age: int


@attrs.frozen
class DonorPlan:
    """A non-empty group of donor worker removals targeting one receiver.

    Phase B floor and Phase C saturation paths both consume this
    type. Single-worker plans are valid (and the only shape the
    selection helpers emit until the multi-donor resource-fit
    search in 3c.4 lands); larger plans are the structural runway
    for that follow-up. Distinct donor stages in ``removals`` each
    have their anti-flap timestamps advanced on commit so a single
    multi-stage donation does not silently allow one stage to
    repeatedly donate within the anti-flap window.

    Attributes:
        removals: Workers to remove, in commit order. The tuple
            MUST be non-empty: ``None`` represents "no donation",
            an empty ``DonorPlan`` is never constructed.
        receiver_stage_index: Position of the receiver stage in
            problem order; carried alongside ``removals`` so the
            planner probe / commit path does not need to re-derive
            it from caller-side state.

    """

    removals: tuple[DonorWorker, ...]
    receiver_stage_index: int

    def __attrs_post_init__(self) -> None:
        if not self.removals:
            msg = "DonorPlan.removals must contain at least one DonorWorker; pass None to signal no donation"
            raise ValueError(msg)


def select_youngest_eligible_donor(
    *,
    receiver_stage_index: int,
    stage_floors: dict[int, int],
    worker_ids_by_stage: list[list[str]],
    worker_ages: dict[str, int],
) -> DonorPlan | None:
    """Pick the youngest eligible donor across stages, with upstream preference.

    Eligibility rules:

      - The donor stage must differ from ``receiver_stage_index``.
      - The donor stage's live worker count minus one must be at
        least the donor's own floor (``stage_floors.get(idx, 1)``);
        prevents cascading rescue.
      - Upstream donors (stage index strictly less than
        ``receiver_stage_index``) are preferred. When no upstream
        donor is eligible, candidates from any non-receiver stage
        are considered.

    Among the remaining candidates, ``(age ASC, worker_id ASC)``
    selects the youngest worker; the ``worker_id`` tiebreaker keeps
    the choice deterministic when ages are uniform. The chosen
    worker is wrapped in a single-worker ``DonorPlan`` so floor
    enforcement and the saturation planner share one return type.

    Args:
        receiver_stage_index: Index of the stage that needs the
            extra worker.
        stage_floors: Per-stage donor floors. A missing entry
            defaults to ``1`` (the implicit one-worker floor).
        worker_ids_by_stage: Per-stage live worker ids in problem
            order. Each inner list is the snapshot of workers that
            stage currently holds in the planner's working state.
        worker_ages: Cluster-wide worker ages keyed by worker id.
            Missing entries default to age 0 (treated as freshly
            observed).

    Returns:
        A single-worker ``DonorPlan`` or ``None`` when no stage can
        donate without violating its own floor.

    """
    eligible_stages = [
        stage_index
        for stage_index, workers in enumerate(worker_ids_by_stage)
        if stage_index != receiver_stage_index and len(workers) - 1 >= stage_floors.get(stage_index, 1)
    ]
    if not eligible_stages:
        return None

    upstream = [s for s in eligible_stages if s < receiver_stage_index]
    pool = upstream if upstream else eligible_stages

    candidates = [
        DonorWorker(
            stage_index=stage_index,
            worker_id=wid,
            age=worker_ages.get(wid, 0),
        )
        for stage_index in pool
        for wid in worker_ids_by_stage[stage_index]
    ]
    if not candidates:
        return None

    chosen = min(candidates, key=operator.attrgetter("age", "worker_id"))
    return DonorPlan(removals=(chosen,), receiver_stage_index=receiver_stage_index)


def find_saturation_donor(
    *,
    receiver_stage_index: int,
    receiver_stage_name: str,
    stage_names: list[str],
    stage_floors: dict[int, int],
    worker_ids_by_stage: list[list[str]],
    worker_ages: dict[str, int],
    stage_states: dict[str, _StageRuntimeState],
    config: SaturationAwareConfig,
    stage_configs: dict[str, SaturationAwareStageConfig],
    cycle: int,
    last_donation_cycle: dict[str, int],
    excluded_worker_ids: frozenset[str] | None = None,
) -> DonorPlan | None:
    """Pick a donor for saturation-driven Phase C growth.

    The non-negotiable donor-floor rule from
    :func:`select_youngest_eligible_donor` applies unchanged. On top
    of it, three layers of anti-flap protection still apply:

      1. Donor classifier must be ``OVER_PROVISIONED`` with at least
         ``stage_cfg.over_provisioned_streak_min_cycles`` full
         streak (gated by
         ``config.cross_stage_donor_require_over_provisioned``).
      2. Donor growth mode must not be ``HOLD`` (gated by
         ``config.cross_stage_donor_exclude_hold_state``).
      3. Receiver must not have donated within the last
         ``config.cross_stage_donor_anti_flap_cycles`` (prevents
         donate-then-receive ping-pong).

    The master toggle ``config.enable_cross_stage_donor`` short-
    circuits the whole helper to ``None`` when disabled. The
    ``config.donor_must_be_strictly_upstream`` flag restricts donors
    to stages with strictly smaller DAG depth when True. Per-cycle
    receiver absorption is naturally bounded by the receiver's
    Phase C intent (capped by ``aggressive_growth_max_per_cycle``);
    a separate cross-stage cap is redundant. Donor-side cooldown is
    subsumed by the OVER_PROVISIONED + streak gate above.

    Args:
        receiver_stage_index: Index of the stage that needs the
            extra worker.
        receiver_stage_name: Name of the receiver stage. Used for
            the receiver-side anti-flap lookup.
        stage_names: Stage names in problem order; the i-th entry
            is the name of the stage at index ``i``.
        stage_floors: Per-stage donor floors. A missing entry
            defaults to ``1``.
        worker_ids_by_stage: Per-stage live worker ids in problem
            order. Each inner list is the snapshot the planner
            currently holds for that stage.
        worker_ages: Cluster-wide worker ages keyed by worker id.
            Missing entries default to ``0``.
        stage_states: Per-stage runtime state keyed by stage name.
            Drives the classifier and growth-mode checks.
        config: Cluster-wide configuration.
        stage_configs: Per-stage effective configs keyed by stage
            name. Drives the streak threshold.
        cycle: Current monotonic cycle number, against which the
            anti-flap cooldown is evaluated.
        last_donation_cycle: Per-stage record of the cycle at which
            each stage most recently donated. Read only by the
            receiver-was-recent-donor anti-flap gate.
        excluded_worker_ids: Optional set of worker ids to drop
            from the candidate pool before donor selection.
            Saturation-aware callers populate this with the donor
            warmup grace set so freshly-warmed workers are not
            yanked off their stage before they have had a chance to
            absorb load. The donor stage itself remains eligible if
            any of its other workers are mature; donor-stage
            elimination only happens when every one of the stage's
            workers is in warmup. ``None`` or an empty set leaves
            the candidate pool unfiltered.

    Returns:
        A single-worker ``DonorPlan`` or ``None`` when the master
        toggle is disabled, the receiver is itself in cooldown, no
        donor stage passes every filter, or every potential donor
        worker is in ``excluded_worker_ids``. The multi-donor
        resource-fit search lands in a follow-up change; this
        helper currently emits one-worker plans only.

    Raises:
        ValueError: If ``stage_names`` and ``worker_ids_by_stage``
            disagree in length, or if ``receiver_stage_index`` is
            outside ``[0, len(worker_ids_by_stage))``. Either
            condition would otherwise surface as an ``IndexError``
            mid-loop or as silently wrong donor selection far from
            the misalignment site.

    """
    if len(stage_names) != len(worker_ids_by_stage):
        msg = (
            "stage_names and worker_ids_by_stage must align in length, "
            f"got len(stage_names)={len(stage_names)} vs "
            f"len(worker_ids_by_stage)={len(worker_ids_by_stage)}"
        )
        raise ValueError(msg)
    if not 0 <= receiver_stage_index < len(worker_ids_by_stage):
        msg = (
            f"receiver_stage_index={receiver_stage_index} is out of bounds for "
            f"len(worker_ids_by_stage)={len(worker_ids_by_stage)}"
        )
        raise ValueError(msg)

    if not config.enable_cross_stage_donor:
        return None

    receiver_anti_flap_cycle = last_donation_cycle.get(receiver_stage_name)
    if (
        receiver_anti_flap_cycle is not None
        and cycle - receiver_anti_flap_cycle < config.cross_stage_donor_anti_flap_cycles
    ):
        return None

    eligible_stages: list[int] = []
    for donor_index, donor_workers in enumerate(worker_ids_by_stage):
        if donor_index == receiver_stage_index:
            continue
        if config.donor_must_be_strictly_upstream and donor_index >= receiver_stage_index:
            continue
        if len(donor_workers) - 1 < stage_floors.get(donor_index, 1):
            continue

        donor_name = stage_names[donor_index]
        donor_state = stage_states.get(donor_name)
        if donor_state is None:
            continue
        if config.cross_stage_donor_require_over_provisioned:
            donor_cfg = stage_configs.get(donor_name)
            if donor_cfg is None:
                continue
            if donor_state.classifier_state is not StageState.OVER_PROVISIONED:
                continue
            if donor_state.classifier_streak < donor_cfg.over_provisioned_streak_min_cycles:
                continue
        if config.cross_stage_donor_exclude_hold_state and donor_state.growth_mode is GrowthMode.HOLD:
            continue

        eligible_stages.append(donor_index)

    if not eligible_stages:
        return None

    excluded = excluded_worker_ids or frozenset()
    candidates = [
        DonorWorker(
            stage_index=donor_index,
            worker_id=wid,
            age=worker_ages.get(wid, 0),
        )
        for donor_index in eligible_stages
        for wid in worker_ids_by_stage[donor_index]
        if wid not in excluded
    ]
    if not candidates:
        return None

    chosen = min(candidates, key=operator.attrgetter("age", "worker_id"))
    return DonorPlan(removals=(chosen,), receiver_stage_index=receiver_stage_index)
