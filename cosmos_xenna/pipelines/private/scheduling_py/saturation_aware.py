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

Implements the same public API as ``FragmentationBasedAutoscaler``
(``setup``, ``update_with_measurements``, ``autoscale``) so the two
can be swapped at ``Autoscaler.__init__`` via
``StreamingSpecificSpec.scheduler``.

``setup()`` seeds an empty per-stage ``_StageRuntimeState`` map.
Classifier thresholds are resolved lazily on the first
``autoscale()`` cycle: ``ProblemStageState.slots_per_worker``
supplies the M/M/c concurrency the resolver needs, and the
formula's ``K/sqrt(c)`` derivation is unstable until that value is
present. Once resolved, the values live on
``_StageRuntimeState.resolved_thresholds`` for the lifetime of the
run and are never re-derived (re-derivation would invalidate the
EWMA / streak history tuned to the original threshold band).

``autoscale()`` constructs an ``AutoscalePlanContext`` per cycle so
per-stage decision logic can stage worker adds and removes against
a working cluster snapshot; the staged plan is frozen into a
``Solution`` via ``ctx.into_solution()``.
"""

from cosmos_xenna.pipelines.private import data_structures
from cosmos_xenna.pipelines.private.scheduling_py.auto_thresholds import (
    ResolvedThresholds,
    _resolve_auto_thresholds,
)
from cosmos_xenna.pipelines.private.scheduling_py.regime import (
    Regime,
    RegimeDetectorState,
    RegimeSignal,
    compute_regime_signal,
    update_regime_state,
)
from cosmos_xenna.pipelines.private.scheduling_py.state import _StageRuntimeState
from cosmos_xenna.pipelines.private.specs import SaturationAwareConfig, SaturationAwareStageConfig
from cosmos_xenna.utils import python_log as logger


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
        _regime_state: Cross-cycle hysteresis state for the
            Halfin-Whitt regime detector. Used to lift the effective
            ``saturation_aggressiveness`` when the cluster is packed
            close to capacity. Defaults to ``SUB_HALFIN_WHITT`` so
            cold-start cycles use the base aggressiveness.

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
        self._regime_state: RegimeDetectorState = RegimeDetectorState()

    def setup(self, problem: data_structures.Problem) -> None:
        """Capture the pipeline shape and seed empty per-stage state.

        Args:
            problem: The frozen pipeline ``Problem``. The Python
                wrapper does not expose stages directly; iterating
                ``.rust.stages`` gives stage names ahead of any
                runtime state arriving in ``autoscale()``.

        """
        self._problem = problem
        self._stage_names = [stage.name for stage in problem.rust.stages]
        self._stage_states = {name: _StageRuntimeState(stage_name=name) for name in self._stage_names}

    def _ensure_thresholds_resolved(self, problem_state: data_structures.ProblemState) -> None:
        """Lazily resolve per-stage classifier thresholds on the first cycle.

        Reads ``slots_per_worker`` from each stage's runtime
        ``ProblemStageState`` (the M/M/c concurrency), feeds it to the
        resolver with the regime-aware effective aggressiveness, stores
        the result on ``_StageRuntimeState``, and emits one INFO log
        line per stage. Stages whose ``resolved_thresholds`` is already
        populated short-circuit; mid-run ``slots_per_worker`` changes
        do not re-resolve. A regime transition (handled by
        ``_update_regime_aware_aggressiveness``) drops every stage's
        resolved thresholds so this method re-derives them with the
        new effective aggressiveness on the same cycle.

        Raises:
            ValueError: If ``problem_state`` carries a stage name not
                present in the ``setup()`` state map (shape mismatch).

        """
        for stage in problem_state.rust.stages:
            runtime = self._stage_states.get(stage.stage_name)
            if runtime is None:
                msg = (
                    f"problem_state stage {stage.stage_name!r} not found in setup() "
                    f"state map (known: {sorted(self._stage_states)}); "
                    "problem and problem_state shapes disagree."
                )
                raise ValueError(msg)
            if runtime.resolved_thresholds is not None:
                continue
            stage_cfg = self._config.get_effective_stage_config(
                stage_name=stage.stage_name,
                spec_override=None,
            )
            effective_aggressiveness = self._effective_aggressiveness(stage_cfg.saturation_aggressiveness)
            resolved = _resolve_auto_thresholds(
                stage_cfg,
                slots_per_actor=stage.slots_per_worker,
                aggressiveness_override=effective_aggressiveness,
            )
            runtime.resolved_thresholds = resolved
            self._log_resolved_thresholds(stage.stage_name, stage_cfg, resolved)

    def _effective_aggressiveness(self, base: float) -> float:
        """Apply the super-Halfin-Whitt lift to the base aggressiveness if active.

        Returns ``base`` unchanged when the cluster is in the
        sub-Halfin-Whitt regime or when the regime-aware lift is
        disabled. Returns ``base + super_halfin_whitt_aggressiveness_lift``
        when the cluster is in super-Halfin-Whitt and the lift is
        enabled.

        """
        if not self._config.enable_regime_aware_aggressiveness:
            return base
        if self._regime_state.current_regime is Regime.SUPER_HALFIN_WHITT:
            return base + self._config.super_halfin_whitt_aggressiveness_lift
        return base

    @staticmethod
    def _log_regime_transition(new_regime: Regime, signal: RegimeSignal) -> None:
        """Emit one INFO line per regime transition."""
        logger.info(
            f"scheduler regime transition: -> {new_regime.value} "
            f"(total_workers={signal.total_workers}, "
            f"cluster_idle_fraction={signal.cluster_idle_fraction:.4f}, "
            f"threshold={signal.threshold:.4f})"
        )

    @staticmethod
    def _log_resolved_thresholds(
        stage_name: str,
        stage_cfg: SaturationAwareStageConfig,
        resolved: ResolvedThresholds,
    ) -> None:
        """Emit one INFO line per stage describing the resolved thresholds."""
        sat_source = "manual override" if resolved.saturation_threshold_was_overridden else "auto"
        act_source = "manual override" if resolved.activation_threshold_was_overridden else "auto"
        logger.info(
            f"scheduler resolved auto thresholds for stage {stage_name!r}: "
            f"slots_per_actor={resolved.slots_per_actor}, "
            f"saturation_aggressiveness={resolved.saturation_aggressiveness}, "
            f"saturation_threshold={resolved.saturation_threshold:.6f} ({sat_source}), "
            f"activation_threshold={resolved.activation_threshold:.6f} ({act_source}), "
            f"over_provisioned_threshold={stage_cfg.over_provisioned_threshold} (config; not auto-derived)"
        )

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

        Args:
            time: Current wall-clock time in seconds.
            problem_state: Current per-stage runtime snapshot.

        Returns:
            A ``Solution`` with one ``StageSolution`` per stage in
            ``problem_state``, in the same order, carrying the existing
            ``slots_per_worker`` and any worker adds / removes staged
            on the cycle's ``AutoscalePlanContext``.

        Raises:
            RuntimeError: ``setup()`` was not called.
            ValueError: ``problem`` and ``problem_state`` disagree on
                stage names or count.

        """
        del time
        if self._problem is None:
            msg = "SaturationAwareScheduler.autoscale() called before setup()"
            raise RuntimeError(msg)

        self._update_regime_aware_aggressiveness(problem_state)
        self._ensure_thresholds_resolved(problem_state)
        ctx = data_structures.AutoscalePlanContext.from_problem_state(self._problem, problem_state)
        return ctx.into_solution()

    def _update_regime_aware_aggressiveness(self, problem_state: data_structures.ProblemState) -> None:
        """Detect the cluster's Halfin-Whitt regime and re-resolve thresholds on transition.

        Computes the per-cycle regime signal, applies hysteresis via
        ``update_regime_state``, and -- on a regime transition -- drops
        every stage's ``resolved_thresholds`` so the next call to
        ``_ensure_thresholds_resolved`` re-derives them with the
        appropriate effective aggressiveness. Cycles whose signal is
        unavailable (no stage populates ``num_used_slots`` /
        ``num_empty_slots`` yet) leave the regime state and resolved
        thresholds untouched. Respects the
        ``enable_regime_aware_aggressiveness`` flag.

        """
        if not self._config.enable_regime_aware_aggressiveness:
            return

        signal = _aggregate_cluster_regime_signal(problem_state)
        transitioned = update_regime_state(
            self._regime_state,
            signal,
            streak_cycles=self._config.regime_transition_streak_cycles,
            exit_band_multiplier=1.5,
        )
        if not transitioned:
            return

        self._log_regime_transition(self._regime_state.current_regime, signal)
        for runtime in self._stage_states.values():
            runtime.resolved_thresholds = None


def _aggregate_cluster_regime_signal(problem_state: data_structures.ProblemState) -> RegimeSignal:
    """Aggregate the cluster's regime-detection inputs from per-stage state.

    Sums worker counts, used slots, and empty slots across every stage
    in ``problem_state``, then delegates to ``compute_regime_signal``.
    Production wiring of ``num_used_slots`` / ``num_empty_slots`` on
    ``ProblemStageState`` is the trigger that flips
    ``signal_available`` from ``False`` to ``True``; until that wiring
    lands the regime detector observes the no-signal default and
    leaves the regime state untouched.
    """
    total_workers = 0
    total_used = 0
    total_empty = 0
    for stage in problem_state.rust.stages:
        total_workers += len(stage.worker_groups)
        total_used += stage.num_used_slots
        total_empty += stage.num_empty_slots
    return compute_regime_signal(
        total_workers=total_workers,
        total_used_slots=total_used,
        total_empty_slots=total_empty,
    )
