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

"""Per-stage intrinsic service-time EWMA store.

Owns the smoothed ``S_k`` per stage that the capacity sizer divides
by slots / workers and the donor planner consumes when scoring
candidates. Writes go through :meth:`SkEwmaStore.update` (one EWMA
step) or :meth:`SkEwmaStore.set` (direct overwrite for seeding); reads
go through :meth:`SkEwmaStore.get` (``math.nan`` when the stage was
never sampled) or :meth:`SkEwmaStore.view` (read-only mapping)::

    .update --+
    .set    --+--> SkEwmaStore --+--> .get
                                 +--> .view
"""

import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import attrs


@attrs.define
class SkEwmaStore:
    """Per-stage EWMA-smoothed intrinsic service time ``S_k``.

    Stage names map to the latest smoothed ``S_k`` value (``math.nan``
    for stages that have never received a finite sample). The backing
    mapping is encapsulated so callers cannot mutate it directly; all
    state changes flow through the methods on this class so the seed
    / blend semantics stay in one place.
    """

    _values: dict[str, float] = attrs.Factory(dict)

    def update(self, stage_name: str, sample: float, alpha: float) -> None:
        """Apply one EWMA step for ``stage_name`` with smoothing ``alpha``.

        Non-finite or non-positive samples are treated as missed
        measurements: the previous value is preserved, and the entry
        is NaN-seeded if absent. The first finite sample replaces a
        NaN seed directly (no blend would slow convergence); subsequent
        finite samples blend ``alpha * sample + (1 - alpha) * prev``.
        """
        prev = self._values.get(stage_name, math.nan)
        if not math.isfinite(sample) or sample <= 0.0:
            self._values.setdefault(stage_name, math.nan)
            return
        self._values[stage_name] = sample if not math.isfinite(prev) else prev * (1.0 - alpha) + sample * alpha

    def set(self, stage_name: str, value: float) -> None:
        """Overwrite ``stage_name`` with ``value`` without an EWMA blend.

        Direct setter for callers that need a known starting value
        rather than a smoothed sample. :meth:`update` is the normal
        write path; :meth:`set` skips the blend step and accepts
        any ``float`` (including ``math.nan``) verbatim.
        """
        self._values[stage_name] = value

    def set_many(self, samples: Mapping[str, float]) -> None:
        """Overwrite each ``stage_name -> value`` pair without an EWMA blend.

        Bulk variant of :meth:`set` for callers that already hold a
        mapping. Iteration order follows ``samples``; entries are
        applied independently so a partial failure cannot leave the
        store in a torn state.
        """
        for name, value in samples.items():
            self._values[name] = value

    def get(self, stage_name: str) -> float:
        """Return the current EWMA value for ``stage_name`` (NaN if absent)."""
        return self._values.get(stage_name, math.nan)

    def seed_nan(self, stage_names: Iterable[str]) -> None:
        """Seed every named stage with NaN; existing values are preserved.

        The Bottleneck phase relies on the seed so the first finite
        per-stage sample replaces (does not blend with) the cold-start
        zero.
        """
        for name in stage_names:
            self._values.setdefault(name, math.nan)

    def reset_seeded(self, stage_names: Iterable[str]) -> None:
        """Clear every entry and NaN-seed ``stage_names`` in place.

        The clear + seed is atomic over the backing storage so callers
        that hold a :meth:`view` see exactly two states: pre-reset
        and post-reset. Object identity is preserved across the call
        so references captured before re-setup remain valid.
        """
        self._values.clear()
        for name in stage_names:
            self._values[name] = math.nan

    def view(self) -> Mapping[str, float]:
        """Return a read-only mapping view over the store.

        Read-only callers should hold the view rather than copies so
        the latest writes are visible without re-fetching; the
        underlying ``MappingProxyType`` rejects mutation, making
        accidental writes impossible.
        """
        return MappingProxyType(self._values)


__all__ = ("SkEwmaStore",)
