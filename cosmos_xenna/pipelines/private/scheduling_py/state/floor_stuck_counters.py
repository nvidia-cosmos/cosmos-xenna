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

"""Per-stage floor-stuck cycle counter."""

from collections.abc import Mapping
from types import MappingProxyType

import attrs


@attrs.define
class FloorStuckCounters:
    """Per-stage counter tracking consecutive floor-stuck cycles.

    The Floor phase increments the counter on a stuck cycle and
    clears it when forward progress lands. Downstream gating turns a
    sufficiently long stuck streak into a long-running-operator WARN
    or a floor-unmet ``RuntimeError``. The backing storage is
    encapsulated; all state changes flow through the methods on this
    class.
    """

    _counts: dict[str, int] = attrs.Factory(dict)

    def increment_stuck(self, stage_name: str) -> int:
        """Increment ``stage_name``'s stuck count and return the new value."""
        self._counts[stage_name] = self._counts.get(stage_name, 0) + 1
        return self._counts[stage_name]

    def reset_for(self, stage_name: str) -> None:
        """Clear the stuck streak for ``stage_name`` (no-op if absent)."""
        self._counts.pop(stage_name, None)

    def get(self, stage_name: str) -> int:
        """Return the current stuck count for ``stage_name`` (zero if absent)."""
        return self._counts.get(stage_name, 0)

    def view(self) -> Mapping[str, int]:
        """Return a read-only mapping view over the counter dict."""
        return MappingProxyType(self._counts)

    def reset(self) -> None:
        """Clear every entry in place.

        Object identity of the underlying mapping is preserved so
        references captured before re-setup remain valid.
        """
        self._counts.clear()


__all__ = ("FloorStuckCounters",)
