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

"""Post-setup pipeline shape - ``PipelineModel``.

Frozen value object that captures the static pipeline shape the
scheduler observes via ``setup(problem)``. Built once and shared
by every helper that previously took a ``Callable[[], Problem | None]``
or ``Callable[[], Sequence[str]]`` resolver closure.

The model centralises three facts that the per-cycle code needs:

- The frozen pipeline ``Problem`` (stage list, worker shapes,
  cluster shape) - exposed as ``problem``.
- The stable stage-name order matching ``problem.rust.stages`` --
  exposed as ``stage_names`` (a ``tuple[str, ...]``).
- The effective per-stage config lookup that merges
  ``SaturationAwareConfig.per_stage_overrides`` with optional
  ``StageSpec``-level overrides - exposed as ``stage_config(name)``.

Because the model is constructed inside ``setup()`` after the
problem is captured, every helper that takes the model can assume
the pipeline is fully described: there is no ``problem is None``
guard, no fallback path. Setup-order checks live solely in the
scheduler facade.

Builders for the model live in this module; helpers consume the
model through narrow ``@attrs.frozen`` constructors.
"""

from collections.abc import Mapping

import attrs

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig


@attrs.frozen
class PipelineModel:
    """Post-``setup()`` pipeline shape exposed to per-cycle helpers.

    Owns the immutable view of the pipeline that the scheduler
    captured at ``setup`` time. Helpers that previously accepted
    resolver closures (``Callable[[], Problem | None]``,
    ``Callable[[], Sequence[str]]``, ``Callable[[str], StageConfig]``)
    now accept this value object directly.

    The model is **immutable** and **non-optional** by construction:
    the scheduler facade builds it inside ``setup()`` once the
    problem is captured, then hands the same instance to every
    helper. Helpers never see ``None`` and never have to reason
    about facade-level setup state.

    Attributes:
        problem: Frozen pipeline ``Problem`` captured by
            ``SaturationAwareScheduler.setup()``.
        stage_names: Stable stage-name order matching
            ``problem.rust.stages``. Used by post-cycle reporters
            and warmup-grace projections.
        config: Cluster-wide ``SaturationAwareConfig``. Held so
            ``stage_config`` can compose cluster defaults with
            spec overrides without a back reference to the
            facade.
        stage_spec_overrides: Read-only mapping of stage-name to
            ``StageSpec``-level override. Outranks the named
            entries on ``config.per_stage_overrides`` and the
            cluster defaults. Empty when no spec-level overrides
            were configured.

    """

    problem: data_structures.Problem
    stage_names: tuple[str, ...]
    config: SaturationAwareConfig
    stage_spec_overrides: Mapping[str, SaturationAwareStageConfig]

    @classmethod
    def from_problem(
        cls,
        *,
        problem: data_structures.Problem,
        config: SaturationAwareConfig,
        stage_spec_overrides: Mapping[str, SaturationAwareStageConfig],
    ) -> "PipelineModel":
        """Build a model from a captured ``Problem`` and the scheduler's config.

        Derives ``stage_names`` from ``problem.rust.stages`` so the
        order is canonical and immutable for the lifetime of the
        cycle.

        Args:
            problem: Pipeline ``Problem`` captured by ``setup()``.
            config: Cluster-wide configuration.
            stage_spec_overrides: ``StageSpec`` overrides forwarded
                from the streaming spec.

        """
        return cls(
            problem=problem,
            stage_names=tuple(stage.name for stage in problem.rust.stages),
            config=config,
            stage_spec_overrides=stage_spec_overrides,
        )

    def stage_config(self, stage_name: str) -> SaturationAwareStageConfig:
        """Return the effective per-stage config for ``stage_name``.

        Composes the spec-level override (when present) with the
        cluster-wide ``SaturationAwareConfig`` defaults and the
        named entry in ``config.per_stage_overrides``. The composition
        is delegated to ``SaturationAwareConfig.get_effective_stage_config``
        so the rule lives in one place.

        Args:
            stage_name: Stage name as it appears in
                ``problem.rust.stages``.

        Returns:
            The resolved ``SaturationAwareStageConfig``.

        """
        return self.config.get_effective_stage_config(
            stage_name=stage_name,
            spec_override=self.stage_spec_overrides.get(stage_name),
        )


__all__ = ("PipelineModel",)
