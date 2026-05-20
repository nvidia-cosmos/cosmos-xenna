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
class DonorCandidate:
    """A worker selected for donation to a receiver stage."""

    stage_index: int
    worker_id: str
    age: int


def select_youngest_eligible_donor(
    *,
    receiver_stage_index: int,
    stage_floors: dict[int, int],
    worker_ids_by_stage: list[list[str]],
    worker_ages: dict[str, int],
) -> DonorCandidate | None:
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
    the choice deterministic when ages are uniform.

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
        The selected ``DonorCandidate`` or ``None`` when no stage can
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
        DonorCandidate(
            stage_index=stage_index,
            worker_id=wid,
            age=worker_ages.get(wid, 0),
        )
        for stage_index in pool
        for wid in worker_ids_by_stage[stage_index]
    ]
    if not candidates:
        return None

    return min(candidates, key=operator.attrgetter("age", "worker_id"))


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
    donations_received_this_cycle: dict[str, int],
    excluded_worker_ids: frozenset[str] | None = None,
) -> DonorCandidate | None:
    """Pick a donor for saturation-driven Phase C growth, with five anti-flap layers.

    The non-negotiable donor-floor rule from
    :func:`select_youngest_eligible_donor` applies unchanged. On top
    of it, five layers reject donors that would oscillate:

      1. Donor classifier must be ``OVER_PROVISIONED`` with at least
         ``stage_cfg.over_provisioned_streak_min_cycles`` full
         streak (gated by
         ``config.cross_stage_donor_require_over_provisioned``).
      2. Donor growth mode must not be ``HOLD`` (gated by
         ``config.cross_stage_donor_exclude_hold_state``).
      3. Receiver must not have donated within the last
         ``config.cross_stage_donor_anti_flap_cycles`` (prevents
         donate-then-receive ping-pong).
      4. Receiver must not have already absorbed
         ``config.cross_stage_donor_max_per_cycle`` donations this
         cycle.
      5. Donor must not have donated within the last
         ``config.cross_stage_donor_min_donation_interval_cycles``
         (donor-side cooldown between consecutive donations).

    The master toggle ``config.enable_cross_stage_donor`` short-
    circuits the whole helper to ``None`` when disabled. The
    ``config.donor_must_be_strictly_upstream`` flag (separate from
    the five anti-flap layers, but applied here) restricts donors to
    stages with strictly smaller DAG depth when True.

    Args:
        receiver_stage_index: Index of the stage that needs the
            extra worker.
        receiver_stage_name: Name of the receiver stage. Used for
            the layer 3 / 4 lookups against the per-stage cycle
            dicts.
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
            Drives the layer 1 / 2 classifier and growth-mode
            checks.
        config: Cluster-wide configuration. Carries the master
            toggle, the upstream-only flag, and all five anti-flap
            cycle counts.
        stage_configs: Per-stage effective configs keyed by stage
            name. Drives the layer 1 streak threshold.
        cycle: Current monotonic cycle number, against which
            cross-cycle cooldowns are evaluated.
        last_donation_cycle: Per-stage record of the cycle at which
            each stage most recently donated.
        donations_received_this_cycle: Per-stage receiver counter,
            reset at the top of every autoscale cycle.
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
        The selected ``DonorCandidate`` or ``None`` when the master
        toggle is disabled, the receiver is itself in cooldown, the
        receiver has hit its per-cycle absorption cap, no donor
        stage passes every filter, or every potential donor worker
        is in ``excluded_worker_ids``.

    """
    if not config.enable_cross_stage_donor:
        return None

    receiver_anti_flap_cycle = last_donation_cycle.get(receiver_stage_name)
    if (
        receiver_anti_flap_cycle is not None
        and cycle - receiver_anti_flap_cycle < config.cross_stage_donor_anti_flap_cycles
    ):
        return None

    if donations_received_this_cycle.get(receiver_stage_name, 0) >= config.cross_stage_donor_max_per_cycle:
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
        donor_last_donation = last_donation_cycle.get(donor_name)
        if (
            donor_last_donation is not None
            and cycle - donor_last_donation < config.cross_stage_donor_min_donation_interval_cycles
        ):
            continue

        eligible_stages.append(donor_index)

    if not eligible_stages:
        return None

    excluded = excluded_worker_ids or frozenset()
    candidates = [
        DonorCandidate(
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

    return min(candidates, key=operator.attrgetter("age", "worker_id"))
