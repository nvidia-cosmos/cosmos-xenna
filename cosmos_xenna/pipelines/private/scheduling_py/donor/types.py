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

"""Shared donor-selection value types.

Holds the data types consumed across the donor subsystem:
``DonorWorker`` / ``DonorPlan`` describe what the donor path
intends to commit; ``RejectReason`` enumerates the gates that may
reject a candidate plan; ``GateResult`` carries the economic gate
verdict; ``DonorCommitOutcome`` carries the probe + atomic-remove
transaction result; ``DonorAcquireResult`` is the
``DonorCoordinator.acquire`` return type.

The types live in their own module so consumers can import them
without pulling in the selection, economics, or coordinator
implementation.
"""

import enum

import attrs


class RejectReason(enum.StrEnum):
    """Enumerated reasons for rejecting a donor plan.

    String-valued so reasons can be emitted directly into
    structured DEBUG log lines without ``.value`` boilerplate.
    Each reason maps to exactly one gate check inside the donor
    pipeline; expanding the enum is the contract for adding a
    new gate. Placeholder placement reasons (``worker_not_found`` /
    ``release_failed`` / ``no_placement``) live on
    ``data_structures.PlacementProbeResult.reject_reason`` and are
    surfaced separately in ``placement_reject_reason``.

    """

    MASTER_TOGGLE_OFF = "master_toggle_off"
    RECEIVER_ANTI_FLAP = "receiver_anti_flap"
    NO_CANDIDATES = "no_candidates"
    SIGNAL_TRUST = "signal_trust"
    RESOURCE_FIT = "resource_fit"
    SPREAD_BELOW_THRESHOLD = "spread_below_threshold"
    THROUGHPUT_REGRESSION = "throughput_regression"
    DONOR_FLIP_GUARD = "donor_flip_guard"
    BALANCE_REGRESSION = "balance_regression"


@attrs.frozen
class DonorWorker:
    """One worker removal in a multi-donor plan.

    Carries the planner indices needed for
    ``probe_add_after_removals`` and ``remove_workers_atomically``,
    plus the deterministic age tiebreaker shared by the resource-
    fit search and the floor selector
    (``(age ASC, worker_id ASC, stage_index ASC)``).

    """

    stage_index: int
    worker_id: str
    age: int


@attrs.frozen
class DonorPlan:
    """A non-empty group of donor worker removals targeting one receiver.

    Consumed by Phase B floor and Phase C saturation paths.
    Single-worker plans are valid; multi-worker plans emerge from
    ``ResourceFitPlanner`` when no single donor's release fits.
    Distinct donor stages each advance their anti-flap timestamp
    on commit. ``None`` represents "no donation" - an empty
    ``DonorPlan`` is never constructed (raises on construction
    via the field validator).

    """

    removals: tuple[DonorWorker, ...] = attrs.field(validator=attrs.validators.min_len(1))
    receiver_stage_index: int


@attrs.frozen
class GateResult:
    """Outcome of the donor-plan economic gate evaluation.

    ``accepted`` is the headline verdict; ``reject_reason`` names
    the failing gate when ``accepted`` is False. The remaining
    scoring fields capture the metrics the structured log lines
    surface so the rejection / commit log carries the inputs that
    drove the decision.
    """

    accepted: bool
    reject_reason: RejectReason | None
    spread: float
    donor_cost: float
    receiver_value: float
    throughput_before: float
    throughput_after: float
    max_d_before: float
    max_d_after: float
    balance_before: float
    balance_after: float
    signal_trust_per_donor: dict[str, float]


@attrs.frozen
class DonorCommitOutcome:
    """Result of a probe-and-commit transaction on a donor plan.

    Three terminal states; ``probe_failed`` and
    ``atomic_remove_failed`` are mutually exclusive when
    ``committed`` is ``False``. ``placement_reject_reason``
    carries the planner's textual reason
    (``worker_not_found`` / ``release_failed`` / ``no_placement``)
    on ``probe_failed``; empty string otherwise.

    """

    committed: bool
    probe_failed: bool
    atomic_remove_failed: bool
    placement_reject_reason: str


@attrs.frozen
class DonorAcquireResult:
    """Final outcome of one ``DonorCoordinator.acquire`` call.

    Replaces the bare ``DonorPlan | None`` return so callers can
    distinguish "no eligible donor" (soft fallback) from
    "probe re-validation failed after selection" (Phase B treats
    as operator-actionable; Phase C as recoverable defensive
    event). Callers MUST inspect ``reject_reason`` when ``plan``
    is ``None`` - the coordinator does not raise on probe
    failure.

    Attributes:
        plan: The committed donor plan, or ``None`` when any
            gate rejected.
        attempted_plan: The plan produced by the resource-fit
            search regardless of whether it eventually
            committed. Populated on every branch that reaches
            the search (so a gate or commit-time rejection
            still surfaces the plan operators can inspect);
            ``None`` only when the policy short-circuited before
            the search (master toggle off, no eligible donors,
            empty candidate pool).
        reject_reason: Named reject gate; ``None`` only on the
            success branch (when ``plan is not None``).
        placement_reject_reason: Planner-supplied textual reason
            (``worker_not_found`` / ``release_failed`` /
            ``no_placement``) when ``reject_reason ==
            RESOURCE_FIT`` and the failure originated from the
            commit-time probe. Empty otherwise.
        gate_result: The economic gate's metrics (success or
            reject); ``None`` for policies that have no gate
            (floor mode) and for early-exit branches that ran no
            gate.

    """

    plan: DonorPlan | None
    attempted_plan: DonorPlan | None
    reject_reason: RejectReason | None
    placement_reject_reason: str
    gate_result: GateResult | None

    @property
    def committed(self) -> bool:
        """``True`` when the coordinator committed a donor plan."""
        return self.plan is not None

    @property
    def probe_failed_at_commit(self) -> bool:
        """``True`` when commit-time probe re-validation rejected the plan.

        Distinct from ``RESOURCE_FIT`` returned during the
        selection-time resource-fit search:
        ``placement_reject_reason`` is the discriminator - it is
        only populated by the commit-time transaction.
        """
        return self.reject_reason is RejectReason.RESOURCE_FIT and bool(self.placement_reject_reason)


__all__ = (
    "DonorAcquireResult",
    "DonorCommitOutcome",
    "DonorPlan",
    "DonorWorker",
    "GateResult",
    "RejectReason",
)
