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

"""Rebuildable fragmentation-solver problem for the saturation-aware scheduler.

The solver pins a stage's worker count (``requested_num_workers``) as a hard
constraint it satisfies before any balancing, so a pinned stage that cannot fit
on a saturated cluster makes the whole solve raise. This template captures the
static per-stage solver inputs once so the scheduler can rebuild the problem
with per-stage request overrides - for example holding a pinned stage at its
current worker count when the cluster cannot grow it.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Self

import attrs

from cosmos_xenna.pipelines.private import data_structures, resources, specs


@attrs.frozen
class SolverStageTemplate:
    """Static solver inputs for one stage, with a swappable worker request.

    Attributes:
        name: Canonical stage name.
        batch_size: Input items consumed per processed batch.
        worker_shape: Per-worker resource shape resolved against the cluster.
        requested_num_workers: Operator-pinned worker count, or ``None`` when
            the stage autoscales.
        over_provision_factor: Optional solver over-provision factor.
    """

    name: str
    batch_size: int
    worker_shape: resources.WorkerShape
    requested_num_workers: int | None
    over_provision_factor: float | None


@attrs.frozen
class SolverProblemTemplate:
    """Rebuild a fragmentation ``Problem`` with per-stage request overrides.

    Attributes:
        cluster: Cluster resources the solver allocates against.
        stages: Per-stage static solver inputs, in pipeline order.
    """

    cluster: resources.ClusterResources
    stages: tuple[SolverStageTemplate, ...]

    @classmethod
    def from_stage_specs(
        cls, stage_specs: Sequence[specs.StageSpec[Any, Any]], cluster: resources.ClusterResources
    ) -> Self:
        """Capture each stage spec's solver inputs against ``cluster``."""
        num_nodes = len(cluster.nodes)
        return cls(
            cluster=cluster,
            stages=tuple(
                SolverStageTemplate(
                    name=spec.name(index),
                    batch_size=spec.stage.stage_batch_size,
                    worker_shape=spec.stage.required_resources.to_worker_shape(cluster),
                    requested_num_workers=spec.resolved_num_workers(num_nodes),
                    over_provision_factor=spec.over_provision_factor,
                )
                for index, spec in enumerate(stage_specs)
            ),
        )

    def build(self, requested_overrides: Mapping[str, int] | None = None) -> data_structures.Problem:
        """Build a ``Problem``, replacing requested counts named in ``requested_overrides``.

        Args:
            requested_overrides: Per-stage-name worker counts that replace the
                captured ``requested_num_workers``. Unnamed stages keep theirs.

        Returns:
            A fresh ``Problem`` for the fragmentation solver.
        """
        overrides = requested_overrides or {}
        return data_structures.Problem(
            self.cluster,
            [
                data_structures.ProblemStage(
                    stage.name,
                    stage.batch_size,
                    stage.worker_shape,
                    requested_num_workers=overrides.get(stage.name, stage.requested_num_workers),
                    over_provision_factor=stage.over_provision_factor,
                )
                for stage in self.stages
            ],
        )
