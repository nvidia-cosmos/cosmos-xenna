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

"""Saturation-aware scheduler - the public class.

The class implements the same public API as
``FragmentationBasedAutoscaler`` (``setup``,
``update_with_measurements``, ``autoscale``) so the two can be
swapped at ``Autoscaler.__init__`` based on
``StreamingSpecificSpec.scheduler``.

The pure-function primitives (classifier, decisions, growth-mode
state machine, per-stage pipeline) are wired into the per-stage
state map built at ``setup()`` time. ``autoscale()`` returns valid
``Solution`` objects with one ``StageSolution`` per pipeline stage.

The per-cycle pipeline integration with real slot signals (used /
empty slots from ``ActorPool``, input queue depth from upstream
queue) requires injecting that data through the ``ProblemState``
boundary; today ``ProblemState`` carries only stage names, worker
groups, slot counts, and the ``is_finished`` flag. Until the signal
injection lands, ``autoscale()`` returns a no-op ``Solution`` per
stage (no worker adds, no removals; existing slot counts preserved)
so the dispatcher works end-to-end without crashing.
"""

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.state import _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig


class SaturationAwareScheduler:
    """Pure-Python saturation-aware scheduler.

    Selected via ``StreamingSpecificSpec.scheduler == SchedulerKind.SATURATION_AWARE``.
    Replaces the Rust-backed ``FragmentationBasedAutoscaler`` at the
    algorithm layer; ``Autoscaler``-level concerns (deletion clamps,
    threading, measurement aggregation) remain owned by ``streaming.py``.

    Attributes:
        _config: Cluster-wide configuration. Per-stage configs are
            resolved on demand via
            ``SaturationAwareConfig.get_effective_stage_config``.
        _problem: Pipeline structure captured at ``setup()`` time. Held
            for use by the per-cycle decision pipeline (DAG depth,
            fixed stage indices); ``None`` until ``setup()`` is called.
        _stage_states: Per-stage runtime state, keyed by stage name.
            Populated at ``setup()``; mutated each cycle by the
            per-stage pipeline.
        _stage_names: Stage names in pipeline (DAG) order, captured at
            ``setup()``. Used to iterate stages deterministically.

    """

    def __init__(self, config: SaturationAwareConfig) -> None:
        """Initialize the scheduler.

        Args:
            config: Cluster-wide ``SaturationAwareConfig``. Stored by
                reference; per-stage configs are resolved lazily via
                ``config.get_effective_stage_config``.

        """
        self._config = config
        self._problem: data_structures.Problem | None = None
        self._stage_states: dict[str, _StageRuntimeState] = {}
        self._stage_names: list[str] = []

    def setup(self, problem: data_structures.Problem) -> None:
        """Capture the static pipeline shape and initialize per-stage state.

        Args:
            problem: The frozen pipeline ``Problem`` - stage names, DAG
                edges, resource requirements. The Python wrapper does
                not expose stages directly; reach through ``.rust`` to
                iterate the pyclass-exposed ``stages`` field, which is
                the canonical source of stage names ahead of any
                runtime state arriving in ``autoscale()``.

        """
        self._problem = problem
        self._stage_names = [stage.name for stage in problem.rust.stages]
        self._stage_states = {name: _StageRuntimeState(stage_name=name) for name in self._stage_names}

    def update_with_measurements(
        self,
        time: float,
        measurements: data_structures.Measurements,
    ) -> None:
        """Ingest the latest measurement batch.

        The fragmentation-based scheduler uses this to feed its
        windowed throughput estimator. The saturation-aware scheduler
        is signal-driven from ``ProblemState.actor_pool_state``
        directly, so per-task measurements are not required for its
        own decisions. They are still accepted and discarded so the
        call sites in ``streaming.py`` need no special-case branch.

        Args:
            time: Current wall-clock time in seconds.
            measurements: Per-stage measurement batch since the previous
                cycle.

        """
        del time, measurements

    def autoscale(
        self,
        time: float,
        problem_state: data_structures.ProblemState,
    ) -> data_structures.Solution:
        """Compute the autoscale plan for the current cycle.

        Walks each stage in ``problem_state`` and emits one
        ``StageSolution`` per stage carrying the existing
        ``slots_per_worker`` and empty add / delete lists. The result
        is a no-op ``Solution`` that preserves the current cluster
        shape; callers (``Autoscaler.apply_autoscale_result_if_ready``)
        see no scaling work to apply.

        Args:
            time: Current wall-clock time in seconds.
            problem_state: Current pipeline state - per-stage actor
                pool snapshots, queue depths, and resource usage.

        Returns:
            A ``Solution`` with one ``StageSolution`` per stage in
            ``problem_state``, in the same order. Each
            ``StageSolution`` has empty ``new_workers`` and
            ``deleted_workers`` lists and the existing
            ``slots_per_worker``.

        """
        del time
        stages = [
            data_structures.StageSolution.make(slots_per_worker=stage.slots_per_worker)
            for stage in problem_state.rust.stages
        ]
        return data_structures.Solution.make(stages=stages)
