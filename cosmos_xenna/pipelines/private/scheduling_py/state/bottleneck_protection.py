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

"""Once-per-streak INFO log for Shrink bottleneck-protection events.

The Shrink phase protects an engaged bottleneck stage from negative-
intent shrink so transient idle (upstream stall, brief slot drop,
model reload) does not pay the full
``worker_warmup_measurement_grace_s`` ramp on re-grow. An operator
log line is emitted ONCE per protection streak so a steady-state
workload with a stuck bottleneck stage sees a single INFO record per
event instead of one per cycle.

State machine::

    not protected ---first cycle in protection---> protected (LOG)
                                                       |
                                                       v
                                              still protected (silent)
                                                       |
                                                       v
                                              drops out (re-arms)
                                                       |
                                                       v
                                              re-enters --> LOG again

The state is owned by :class:`BottleneckProtectionLogger` which the
Shrink phase reads at the top of every run (``previous_cycle``) and
replaces at the tail (``replace_snapshot(currently_protected)``).
"""

import attrs

from cosmos_xenna.pipelines.private.scheduling_py.state.bottleneck_identity import BottleneckIdentity
from cosmos_xenna.utils import python_log as logger


@attrs.define
class BottleneckProtectionLogger:
    """Once-per-streak INFO logger for Shrink bottleneck protection.

    Holds the set of stages that were protected on the previous
    cycle so the INFO log fires only when a stage newly enters
    protection. Any stage that drops out of the new set re-arms a
    fresh INFO log on its next entry.

    Attributes:
        previous_cycle: Set of stage names that were protected on the
            previous Shrink cycle. Mutated at the tail of every
            Shrink run via :meth:`replace_snapshot`.

    """

    previous_cycle: set[str] = attrs.Factory(set)

    def maybe_log(
        self,
        *,
        stage_name: str,
        intent: int,
        bottleneck_meta: BottleneckIdentity,
    ) -> None:
        """Emit the once-per-streak INFO if the stage newly entered protection.

        The Shrink phase calls this for every stage that was held by
        bottleneck protection on the current cycle. The log fires
        only on the transition cycle (``stage_name not in
        previous_cycle``); subsequent cycles that keep the stage
        protected stay silent.

        Args:
            stage_name: The stage being protected this cycle.
            intent: The per-stage signed intent that triggered the
                protection (always negative because protection only
                applies to intent < 0).
            bottleneck_meta: Identified bottleneck snapshot for the
                cycle; used for the structured log fields
                (``max_d_k`` and ``heterogeneity_ratio``).

        """
        if stage_name in self.previous_cycle:
            return
        logger.info(
            f"phase D bottleneck shrink protected: stage {stage_name!r} "
            f"intent={intent} but D_k={bottleneck_meta.max_d_k:.2f}s is "
            f"argmax (ratio={bottleneck_meta.heterogeneity_ratio:.2f}); "
            "skipping shrink to preserve throughput across transient idle"
        )

    def replace_snapshot(self, protected: set[str]) -> None:
        """Replace the previous-cycle snapshot at the tail of every Shrink run.

        A stage that drops out of the new set will re-arm a fresh
        INFO log on its next entry, while stages that remain protected
        stay silent.

        Args:
            protected: Stages held by bottleneck protection during
                the just-finished Shrink cycle. A shallow copy is
                stored so later caller mutations do not affect
                ``previous_cycle``.

        """
        if protected is None:
            self.previous_cycle = set()
        else:
            self.previous_cycle = protected.copy()


__all__ = ["BottleneckProtectionLogger"]
