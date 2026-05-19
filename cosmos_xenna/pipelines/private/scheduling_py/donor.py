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

Picks the youngest worker from a non-receiver stage that can spare
one without violating its own minimum-worker floor. Upstream stages
(lower index in the problem's stage order) are preferred when any
are eligible; otherwise any non-receiver stage may donate. The
selector is mode-agnostic: callers control behaviour purely through
the ``stage_floors`` argument.

The non-negotiable constraint is donor-floor preservation: a stage
whose live worker count minus one would drop below its floor is
filtered out, preventing a single donation from cascading into
another stage's bootstrap.
"""

import operator

import attrs


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
