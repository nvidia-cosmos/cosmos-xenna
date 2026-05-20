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


"""End-to-end smoke test for the streaming-mode autoscaler dispatcher.

Pins the cross-cutting contract that a small CPU-only Xenna pipeline can
complete under either ``StreamingSpecificSpec.scheduler`` value:

* ``SchedulerKind.FRAGMENTATION_BASED``: the Rust-backed autoscaler that
  drives placement from cluster fragmentation signals.
* ``SchedulerKind.SATURATION_AWARE``: the pure-Python signal-driven
  autoscaler that drives placement from per-stage saturation classifiers.

The test exercises real Ray actor placement, the per-cycle autoscale
calculation thread, and the streaming apply loop. It is intentionally
GPU-free and resource-frugal so a CI runner with two CPUs can complete
it without timing out.

Universal invariants (asserted for both schedulers):

* Every input task reaches the last stage (no dropped tasks).
* Every stage materialises at least one worker (proves the autoscaler
  applied a placement decision).
* Wall-clock duration stays under a generous backstop so a real deadlock
  fails the test instead of hanging the CI runner.

Saturation-aware-only invariants (asserted only when
``scheduler is SchedulerKind.SATURATION_AWARE``):

* Peak simultaneous worker count per stage stays at or below the
  operator-configured ``max_workers`` cap. The cap is a
  simultaneous-count cap, so lifetime distinct PIDs would
  over-count whenever Phase D shrinks a stage and Phase C later
  re-grows it (even when the cap is honoured at every instant).
  The test records each ``process_data`` active span and computes the
  maximum number of distinct PIDs with overlapping spans.
* Observed active worker count per stage stays at or above the
  operator-configured ``min_workers`` floor while that stage is
  processing work. Lifetime setup count would not catch a stage that
  shrank below the floor after first materialising a worker.

Resource scaling follows the
``cosmos_xenna.utils.ci.is_running_in_cicd`` pattern used throughout
``cosmos_xenna/pipelines/v1/``: smaller per-worker CPU shares and a
smaller task count when running in CI, larger values for local
developer hosts.

Pipeline shape:

      +----------+    +-----------+    +-----------+
      | producer | -> |  compute  | -> | consumer  | -> results
      |  cpu=c   |    |  cpu=c    |    |  cpu=c    |
      +----------+    +-----------+    +-----------+
      stage 0          stage 1          stage 2
"""

import os
import time
from collections.abc import Iterator
from typing import Any, cast

import pytest
import ray

from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.pipelines.private import resources, specs
from cosmos_xenna.utils.ci import is_running_in_cicd

_TRACKER_ACTOR_NAME = "saturation_aware_integration_tracker"

_STAGE_NAMES: tuple[str, ...] = ("producer", "compute", "consumer")

_MAX_WORKERS_PER_STAGE = 2
_MIN_WORKERS_PER_STAGE = 1

_WALL_CLOCK_TIMEOUT_S = 120.0


@pytest.fixture(autouse=True)
def _ensure_clean_ray() -> Iterator[None]:
    """Guarantee a fresh Ray cluster per test in this module.

    Shutting down before AND after each test makes the order of
    tests irrelevant and contains the blast radius of any future
    test that forgets its own cleanup. Mirrors the autouse pattern in
    ``test_autoscaling.py`` so each parameterised run gets an
    independent Ray context.
    """
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def _init_ray_for_test(monkeypatch: pytest.MonkeyPatch, num_cpus: int) -> None:
    """Pre-initialise Ray with the dashboard-agent env vars the pipeline expects.

    The streaming pipeline queries the Ray state API with ``limit=40000``.
    That request is rejected by the dashboard agent unless
    ``RAY_MAX_LIMIT_FROM_API_SERVER`` and ``RAY_MAX_LIMIT_FROM_DATA_SOURCE``
    are set before the dashboard agent starts. Because the named tracker
    actor created later in the test would otherwise lazy-init Ray with the
    default limits, we set the env vars first and then bring Ray up with the
    desired CPU shape; the pipeline's later ``ignore_reinit_error=True``
    ``ray.init`` becomes a no-op against the cluster we just stood up.

    ``monkeypatch.setenv`` reverts the env vars at the end of the test so
    they do not leak into sibling tests run in the same pytest process.

    Args:
        monkeypatch: pytest fixture used to scope environment changes to
            the current test.
        num_cpus: Synthetic core count to expose to the Ray scheduler.
    """
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("RAY_MAX_LIMIT_FROM_API_SERVER", "40000")
    monkeypatch.setenv("RAY_MAX_LIMIT_FROM_DATA_SOURCE", "40000")
    ray.init(
        num_cpus=num_cpus,
        num_gpus=0,
        include_dashboard=True,
        ignore_reinit_error=True,
    )


@ray.remote(num_cpus=0)
class _WorkerTracker:
    """Collects per-stage worker liveness signals from stage actors.

    Two records are maintained per stage. Each Ray worker actor lives
    in its own OS process, so PIDs uniquely identify workers within a
    single test run.

    * ``_setups`` -- set of PIDs that called ``setup()`` at any point
      in the run. Drives the universal "the autoscaler placed at least
      one worker for this stage" check. Robust against worker churn:
      a worker that was created and later removed still contributes
      its PID to this set, which is the desired semantic for that
      check.
    * ``_active_spans`` -- list of ``(pid, start_ts, end_ts)`` tuples
      emitted on every ``process_data`` call. Drives
      :meth:`observed_active_bounds`, which counts distinct PIDs with
      overlapping processing spans. This pins the operator-configured
      worker band during real work without relying on lifetime PID
      counts.

    ``num_cpus=0`` keeps the tracker out of the autoscaler's resource
    budget so a CI host with a small core count still has all of its
    CPUs available to the pipeline stages.
    """

    def __init__(self) -> None:
        self._setups: dict[str, set[int]] = {}
        self._active_spans: dict[str, list[tuple[int, float, float]]] = {}

    def register_setup(self, stage_name: str, pid: int) -> None:
        """Record that a worker for ``stage_name`` called ``setup()``."""
        self._setups.setdefault(stage_name, set()).add(pid)

    def record_active_span(self, stage_name: str, pid: int, start_ts: float, end_ts: float) -> None:
        """Record the wall-clock interval where one worker processed one item."""
        self._active_spans.setdefault(stage_name, []).append((pid, start_ts, end_ts))

    def setup_counts(self) -> dict[str, int]:
        """Distinct PIDs that called ``setup()`` per stage (lifetime count).

        Returns:
            Mapping from stage name to lifetime distinct setup-PID count.
        """
        return {k: len(v) for k, v in self._setups.items()}

    def observed_active_bounds(self) -> dict[str, tuple[int, int]]:
        """Minimum and peak distinct PIDs actively processing at span starts.

        For each stage anchors a concurrency snapshot at every recorded
        ``process_data`` start and counts distinct PIDs whose active
        spans include that timestamp. Anchoring at starts catches cap
        breaches exactly when new overlapping work begins while keeping
        the floor check scoped to windows where the stage is doing work.

        Returns:
            Mapping from stage name to ``(min_observed, peak_observed)``.
        """
        result: dict[str, tuple[int, int]] = {}
        for stage_name, spans in self._active_spans.items():
            if not spans:
                result[stage_name] = (0, 0)
                continue
            counts = [
                len({pid for pid, start_ts, end_ts in spans if start_ts <= anchor_ts < end_ts})
                for _, anchor_ts, _ in spans
            ]
            result[stage_name] = (min(counts), max(counts))
        return result


class _RecordingStage(pipelines_v1.Stage[int, int]):
    """Lightweight CPU-only stage that records its worker liveness.

    The stage performs a deterministic fan-out-of-one transform
    (``[x * 2 for x in batch]``) and sleeps for a configurable duration
    per ``process_data`` call to simulate work without burning CPU.

    On every ``setup`` it registers ``(stage_name, os.getpid())`` with
    the named tracker actor (``register_setup``) so the test can assert
    that the autoscaler placed at least one worker for this stage.

    On every ``process_data`` call it records the active processing
    span (``record_active_span``) carrying the same PID plus start/end
    timestamps. The call is awaited so the test observes a causally
    complete timeline after ``run_pipeline`` returns.
    """

    def __init__(self, stage_name: str, cpus: float, process_dur_s: float, tracker_name: str) -> None:
        self._stage_name = stage_name
        self._cpus = float(cpus)
        self._process_dur_s = float(process_dur_s)
        self._tracker_name = tracker_name
        self._tracker_handle: ray.actor.ActorHandle[Any] | None = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cpus, gpus=0.0)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        self._tracker_handle = ray.get_actor(self._tracker_name)
        ray.get(self._tracker_handle.register_setup.remote(self._stage_name, os.getpid()))

    def process_data(self, in_data: list[int]) -> list[int]:
        # ``setup`` is guaranteed to run before the first ``process_data``
        # call, so ``_tracker_handle`` is always populated here.
        assert self._tracker_handle is not None, "process_data invoked before setup()"
        start_ts = time.time()
        if self._process_dur_s > 0.0:
            time.sleep(self._process_dur_s)
        tracker_handle = cast(Any, self._tracker_handle)
        ray.get(
            tracker_handle.record_active_span.remote(
                self._stage_name,
                os.getpid(),
                start_ts,
                time.time(),
            )
        )
        return [x * 2 for x in in_data]


def _make_saturation_aware_stage_config() -> specs.SaturationAwareStageConfig:
    """Build a stage config pinning only ``min_workers`` and ``max_workers``."""
    return specs.SaturationAwareStageConfig(
        min_workers=_MIN_WORKERS_PER_STAGE,
        max_workers=_MAX_WORKERS_PER_STAGE,
    )


def _build_pipeline_spec(
    scheduler: specs.SchedulerKind,
    *,
    num_tasks: int,
    cpus_per_stage: float,
    process_dur_s: float,
    tracker_name: str,
) -> pipelines_v1.PipelineSpec:
    """Construct the parameterised pipeline spec for one scheduler kind."""
    sat_aware_config = specs.SaturationAwareConfig(stage_defaults=_make_saturation_aware_stage_config())
    stage_specs = [
        pipelines_v1.StageSpec(_RecordingStage(stage_name, cpus_per_stage, process_dur_s, tracker_name))
        for stage_name in _STAGE_NAMES
    ]
    return pipelines_v1.PipelineSpec(
        input_data=list(range(num_tasks)),
        stages=stage_specs,
        config=pipelines_v1.PipelineConfig(
            logging_interval_s=5,
            return_last_stage_outputs=True,
            mode_specific=pipelines_v1.StreamingSpecificSpec(
                autoscale_interval_s=1.0,
                scheduler=scheduler,
                saturation_aware=sat_aware_config,
            ),
        ),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "scheduler",
    [
        specs.SchedulerKind.FRAGMENTATION_BASED,
        specs.SchedulerKind.SATURATION_AWARE,
    ],
    ids=["fragmentation_based", "saturation_aware"],
)
def test_pipeline_runs_to_completion_under_each_scheduler(
    scheduler: specs.SchedulerKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a small CPU-only pipeline through both autoscaler implementations.

    Universal assertions apply to both ``SchedulerKind`` values: every
    input task reaches the last stage, every stage materialises at least
    one worker, and wall-clock time stays under ``_WALL_CLOCK_TIMEOUT_S``.

    Under ``SchedulerKind.SATURATION_AWARE`` the test additionally
    checks both ends of the operator band on
    ``SaturationAwareConfig.stage_defaults``:

    * Peak observed active worker count per stage stays at or below
      ``max_workers``.
    * Minimum observed active worker count per stage stays at or above
      ``min_workers`` while that stage is processing work.
    """
    num_tasks = 30 if is_running_in_cicd() else 60
    cpus_per_stage = 0.25 if is_running_in_cicd() else 0.5
    process_dur_s = 0.3
    num_cpus = 4 if is_running_in_cicd() else 8

    _init_ray_for_test(monkeypatch, num_cpus=num_cpus)

    tracker = _WorkerTracker.options(name=_TRACKER_ACTOR_NAME).remote()  # type: ignore[attr-defined]
    try:
        spec = _build_pipeline_spec(
            scheduler,
            num_tasks=num_tasks,
            cpus_per_stage=cpus_per_stage,
            process_dur_s=process_dur_s,
            tracker_name=_TRACKER_ACTOR_NAME,
        )

        start = time.monotonic()
        results = pipelines_v1.run_pipeline(spec)
        elapsed = time.monotonic() - start

        setup_counts: dict[str, int] = ray.get(tracker.setup_counts.remote())  # type: ignore[attr-defined]
        active_bounds: dict[str, tuple[int, int]] = ray.get(  # type: ignore[attr-defined]
            tracker.observed_active_bounds.remote()
        )

        assert results is not None, (
            f"run_pipeline returned None despite return_last_stage_outputs=True; scheduler={scheduler.value}"
        )
        assert len(results) == num_tasks, (
            f"Dropped tasks: expected {num_tasks}, got {len(results)}; "
            f"scheduler={scheduler.value}, setup_counts={setup_counts}"
        )
        assert sorted(results) == [x * 8 for x in range(num_tasks)], (
            f"Output values diverged from the expected x*8 transform; scheduler={scheduler.value}"
        )

        for stage_name in _STAGE_NAMES:
            assert setup_counts.get(stage_name, 0) >= 1, (
                f"Stage '{stage_name}' never materialised a worker; "
                f"scheduler={scheduler.value}, setup_counts={setup_counts}"
            )

        assert elapsed < _WALL_CLOCK_TIMEOUT_S, (
            f"Pipeline took {elapsed:.1f}s (limit {_WALL_CLOCK_TIMEOUT_S:.1f}s); "
            f"likely deadlocked; scheduler={scheduler.value}, setup_counts={setup_counts}"
        )

        if scheduler is specs.SchedulerKind.SATURATION_AWARE:
            for stage_name in _STAGE_NAMES:
                # The operator-configured cap is a simultaneous-count cap,
                # so the assertion uses peak observed active workers rather
                # than lifetime distinct PIDs, which would over-count under
                # worker churn even when the cap is honoured.
                observed_min, peak = active_bounds.get(stage_name, (0, 0))
                assert peak <= _MAX_WORKERS_PER_STAGE, (
                    f"Stage '{stage_name}' exceeded the operator-configured max_workers "
                    f"({_MAX_WORKERS_PER_STAGE}); active_peak={peak}, "
                    f"setup_counts={setup_counts}, active_bounds={active_bounds}"
                )
                assert observed_min >= _MIN_WORKERS_PER_STAGE, (
                    f"Stage '{stage_name}' fell below the operator-configured min_workers "
                    f"({_MIN_WORKERS_PER_STAGE}) while processing work; "
                    f"setup_counts={setup_counts}, active_bounds={active_bounds}"
                )
    finally:
        ray.kill(tracker)
