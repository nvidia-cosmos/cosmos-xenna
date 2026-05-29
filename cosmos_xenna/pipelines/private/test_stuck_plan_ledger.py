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

"""Write-path guard for ``StuckPlanLedger.record``.

The per-stage stuck counter is monotonic -- the Grow phase only resets
it to ``0`` or advances it by ``+1``. ``record`` is the single write
path, so it rejects a negative value before it can be stored or
forwarded to the detector's structured log / Prometheus metrics.
"""

import pytest

from cosmos_xenna.pipelines.private.scheduling_py.state.stuck_plan_ledger import StuckPlanLedger


def test_record_rejects_negative_counter_without_mutation() -> None:
    """A negative counter raises ``ValueError`` and leaves the ledger untouched."""
    ledger = StuckPlanLedger()

    with pytest.raises(ValueError, match="negative stuck-cycle counter"):
        ledger.record("stage_a", -1, last_intent=2, threshold_cycles=18, pipeline_name="p")

    # The guard fires before ``self._counters[stage_name] = value`` and
    # before ``self.detector.update(...)``, so the stage was never
    # recorded and the detector was never notified.
    assert "stage_a" not in ledger.view()
    assert ledger.get_counter("stage_a") == 0
