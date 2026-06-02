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

"""Warmup-delete grace for the saturation-aware scheduler.

Records when each worker was first observed and protects recently-seen
workers from deletion, so an expensive stage that just spun up a worker
(e.g. a vLLM actor mid model-load) is not torn down on the next cycle.
This is the scheduler's own copy of the post-Ready grace idea; it does
not depend on the shared apply-path grace.
"""

from collections.abc import Iterable, Sequence

import attrs


@attrs.define
class WarmupGrace:
    """Per-worker first-seen tracker that vetoes deletion of young workers.

    Call :meth:`observe` once per cycle with the live worker ids before
    consulting :meth:`allowed_deletions`.
    """

    _first_seen_s: dict[str, float] = attrs.field(factory=dict)

    def observe(self, worker_ids: Iterable[str], now: float) -> None:
        """Record first-seen timestamps for new workers and forget gone ones."""
        live = set(worker_ids)
        for worker_id in live:
            self._first_seen_s.setdefault(worker_id, now)
        for worker_id in list(self._first_seen_s):
            if worker_id not in live:
                del self._first_seen_s[worker_id]

    def allowed_deletions(self, delete_ids: Sequence[str], now: float, grace_s: float) -> list[str]:
        """Return the subset of ``delete_ids`` old enough to delete.

        A worker is protected (dropped from the result) when it was first
        observed less than ``grace_s`` ago. Unknown ids are not protected.
        """
        return [worker_id for worker_id in delete_ids if not self._is_young(worker_id, now, grace_s)]

    def _is_young(self, worker_id: str, now: float, grace_s: float) -> bool:
        first_seen = self._first_seen_s.get(worker_id)
        return first_seen is not None and (now - first_seen) < grace_s
