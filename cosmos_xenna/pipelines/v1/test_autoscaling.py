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


"""
Simple ray example which uses ray to download, slightly modify and upload a directory of tars.

See the "Running a multinode Ray job" of pipelines/examples/README.md for more info.
"""

import os
import time
import uuid
from typing import Iterator, Optional

import pytest
import ray

import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.utils.ci import is_running_in_cicd


@pytest.fixture(autouse=True)
def _ensure_clean_ray() -> Iterator[None]:
    """Guarantee a fresh Ray cluster per test in this module.

    Shutting down before AND after each test makes the order of
    tests irrelevant and contains the blast radius of any future
    test that forgets its own cleanup.
    """
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


class _ProcessStage(pipelines_v1.Stage):
    def __init__(self, setup_dur: float, process_dur: float) -> None:
        self._setup_dur = float(setup_dur)
        self._process_dur = float(process_dur)

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        # Scale per-stage CPU requirements for CI jobs (5 stages => 2.5 total).
        cpus = 0.5 if is_running_in_cicd() else 1.0
        return pipelines_v1.Resources(cpus=cpus, gpus=0.0)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        time.sleep(self._setup_dur)

    def process_data(self, task: list[float]) -> list[float]:
        time.sleep(self._process_dur)
        return [x * 2 for x in task]


@pytest.mark.slow
def test_autoscaling() -> None:
    tasks = range(100)
    # We make a "spec" which tells our code how to run our pipeline. This spec is very simple. It is just a list of
    # objects we want to run over and a single stage to run over the objects.
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(_ProcessStage(1, 0), num_workers_per_node=1),
            _ProcessStage(3, 0.1),
            _ProcessStage(1, 1),
            _ProcessStage(5, 0.3),
            pipelines_v1.StageSpec(_ProcessStage(1, 0), num_workers_per_node=1),
        ],
        config=pipelines_v1.PipelineConfig(
            logging_interval_s=5,
            mode_specific=pipelines_v1.StreamingSpecificSpec(
                autoscale_interval_s=1,
                autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.DEBUG,
            ),
        ),
    )
    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    finally:
        ray.shutdown()


class _FixedCpuStage(pipelines_v1.Stage):
    """Stage with explicit CPU/GPU shape for autoscaler regression tests.

    When `tracker_name` is provided, the stage's `setup` hook
    registers `(name, os.getpid())` with the named Ray actor at
    that name - tests then query the tracker after the pipeline
    completes to count distinct worker PIDs per stage.
    """

    def __init__(
        self,
        name: str,
        cpus: float,
        gpus: float = 0.0,
        setup_dur: float = 0.0,
        process_dur: float = 0.0,
        tracker_name: Optional[str] = None,
    ) -> None:
        self._name = name
        self._cpus = cpus
        self._gpus = gpus
        self._setup_dur = float(setup_dur)
        self._process_dur = float(process_dur)
        self._tracker_name = tracker_name

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cpus, gpus=self._gpus)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        if self._tracker_name is not None:
            tracker = ray.get_actor(self._tracker_name)
            ray.get(tracker.register.remote(self._name, os.getpid()))
        time.sleep(self._setup_dur)

    def process_data(self, task: list[float]) -> list[float]:
        time.sleep(self._process_dur)
        return [x * 2 for x in task]


@ray.remote(num_cpus=0)
class _WorkerTracker:
    """Track unique ``(stage_name, pid)`` pairs registered by stage setups.

    Each Ray worker actor runs in its own process, so distinct PIDs identify
    distinct workers. A starved stage materialises only the Phase 2 floor
    worker; a count > 1 for a given stage therefore proves Phase 3
    preemption (or Phase 4 headroom) grew that stage beyond the floor.

    ``num_cpus=0`` keeps the tracker out of the autoscaler's budget so a
    test that claims a 100-CPU cluster sees all 100 CPUs available to
    stage workers.
    """

    def __init__(self) -> None:
        self._pids: dict[str, set[int]] = {}

    def register(self, stage_name: str, pid: int) -> None:
        self._pids.setdefault(stage_name, set()).add(pid)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._pids.items()}


def _init_ray_for_autoscale_test(monkeypatch: pytest.MonkeyPatch, num_cpus: int, num_gpus: int) -> None:
    """Pre-initialize Ray with the same options the pipeline uses.

    ``run_pipeline`` eventually calls
    ``cluster.init_or_connect_to_cluster`` which sets
    ``RAY_MAX_LIMIT_FROM_API_SERVER`` / ``RAY_MAX_LIMIT_FROM_DATA_SOURCE``
    to 40000 and then calls
    ``ray.init(include_dashboard=True, ignore_reinit_error=True, ...)``.
    When we need to override ``num_cpus`` / ``num_gpus`` for a contention
    scenario we must call ``ray.init`` first with those counts, but we
    must also set the same env vars **before** ``ray.init`` -- the
    dashboard agent reads them at startup. If set after, the agent is
    already running with smaller defaults and the pipeline's later
    state-API queries (``limit=40000``) are rejected with HTTP 500 /
    ``RayStateApiException``. The ``include_dashboard=True`` flag here
    mirrors the pipeline's init so the dashboard agent is started
    (otherwise the later pipeline init is a no-op under
    ``ignore_reinit_error=True``).

    ``monkeypatch.setenv`` is used so the env vars are reverted at the
    end of the test instead of leaking into sibling tests run in the
    same pytest process.
    """
    # Authoritative assignment via monkeypatch: a smaller inherited value
    # for RAY_MAX_LIMIT_FROM_API_SERVER silently reproduces the exact HTTP
    # 500 / RayStateApiException this helper exists to prevent.
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("RAY_MAX_LIMIT_FROM_API_SERVER", "40000")
    monkeypatch.setenv("RAY_MAX_LIMIT_FROM_DATA_SOURCE", "40000")
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        include_dashboard=True,
        ignore_reinit_error=True,
    )


@pytest.mark.slow
def test_autoscaler_does_not_starve_downstream_under_cpu_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Phase 2 floor: downstream keeps >= 1 worker under upstream load.

    Scenario:

    - 100-CPU cluster, ``num_gpus=0``.
    - Two stages, ``cpus=5`` each (max 20 workers would fit across both).
    - Upstream is the per-task bottleneck (``process_dur=0.5``) with many
      inputs, so the Rust autoscaler's throughput estimate drives it to
      grow upstream aggressively.
    - Downstream is fast per-task (``process_dur=0.05``) but cannot run
      without a worker. If Phase 2's floor does not fire, the pipeline
      deadlocks.

    Two assertions, in order of diagnostic value:

    1. ``counts['downstream'] >= 1`` via a ``_WorkerTracker``
       named actor - direct state evidence that downstream actually
       instantiated a worker. This is the canonical Phase 2 oracle.
    2. A generous wall-clock backstop (``elapsed < 90s``) so that an
       actual deadlock terminates the test instead of hanging.
    """
    _init_ray_for_autoscale_test(monkeypatch, num_cpus=100, num_gpus=0)
    try:
        tracker = _WorkerTracker.options(name="floor_tracker").remote()  # type: ignore[attr-defined]
        try:
            spec = pipelines_v1.PipelineSpec(
                input_data=list(range(50)),
                stages=[
                    _FixedCpuStage("upstream", cpus=5, process_dur=0.5, tracker_name="floor_tracker"),
                    _FixedCpuStage("downstream", cpus=5, process_dur=0.05, tracker_name="floor_tracker"),
                ],
                config=pipelines_v1.PipelineConfig(
                    logging_interval_s=2,
                    mode_specific=pipelines_v1.StreamingSpecificSpec(
                        autoscale_interval_s=1,
                        autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.DEBUG,
                    ),
                ),
            )
            start = time.monotonic()
            pipelines_v1.run_pipeline(spec)
            elapsed = time.monotonic() - start
            counts: dict[str, int] = ray.get(tracker.counts.remote())  # type: ignore[attr-defined]
            assert counts.get("downstream", 0) >= 1, (
                f"Phase 2 floor failed: downstream never instantiated a worker; counts={counts}"
            )
            assert elapsed < 90.0, f"Pipeline took {elapsed:.1f}s; likely deadlocked; counts={counts}"
        finally:
            ray.kill(tracker)
    finally:
        ray.shutdown()


@pytest.mark.slow
def test_autoscaler_preempts_upstream_for_slow_downstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Phase 3 preemption: donors give up workers for a slower downstream.

    Preemption contract: when the downstream stage is the cluster
    bottleneck, the Rust max-min loop identifies it as the slowest
    stage and pulls workers from upstream, which qualifies as a
    donor because ``current_workers > 1`` and
    ``throughput_if_one_removed > min_throughput``.

    Scenario:

    - 100-CPU cluster, ``num_gpus=0``.
    - Two stages, ``cpus=5`` each.
    - Upstream is fast per-task (``process_dur=0.05``) - intrinsically
      needs few workers.
    - Downstream is slow per-task (``process_dur=0.5``) - the cluster
      bottleneck. The Rust max-min loop must identify downstream as the
      slowest stage and pull workers from upstream (valid donor because
      ``current_workers > 1`` and ``throughput_if_one_removed >
      min_throughput``).

    After the run, we query the ``_WorkerTracker`` actor for the set of
    distinct PIDs that registered themselves. ``downstream > 1`` proves
    at least one preemption cycle fired (or Phase 4 headroom grew
    downstream, which is the same contract from the user's POV -
    downstream did not stay stuck at the Phase 2 floor).
    """
    _init_ray_for_autoscale_test(monkeypatch, num_cpus=100, num_gpus=0)
    try:
        tracker = _WorkerTracker.options(name="autoscale_tracker").remote()  # type: ignore[attr-defined]
        try:
            spec = pipelines_v1.PipelineSpec(
                input_data=list(range(80)),
                stages=[
                    _FixedCpuStage("upstream", cpus=5, process_dur=0.05, tracker_name="autoscale_tracker"),
                    _FixedCpuStage("downstream", cpus=5, process_dur=0.5, tracker_name="autoscale_tracker"),
                ],
                config=pipelines_v1.PipelineConfig(
                    logging_interval_s=2,
                    mode_specific=pipelines_v1.StreamingSpecificSpec(
                        autoscale_interval_s=1,
                        autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.DEBUG,
                    ),
                ),
            )
            pipelines_v1.run_pipeline(spec)
            counts: dict[str, int] = ray.get(tracker.counts.remote())  # type: ignore[attr-defined]
            assert counts.get("downstream", 0) > 1, (
                f"Phase 3 preemption failed: downstream stuck at "
                f"{counts.get('downstream', 0)} unique worker(s); counts={counts}"
            )
        finally:
            ray.kill(tracker)
    finally:
        ray.shutdown()


def _install_fake_gpus(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Patch NVML-based GPU discovery so tests can allocate synthetic GPUs.

    ``cosmos_xenna.pipelines.private.resources.get_local_gpu_info`` is
    the sole producer of node GPU inventory consumed by
    ``make_cluster_resources_for_ray_cluster`` (via the
    ``@ray.remote`` helper ``_get_node_info_from_current_node``).
    Patching it yields deterministic fake GPUs without requiring
    NVIDIA hardware / NVML and lets a CPU-only dev host exercise the
    ``cpu=1, gpu=1`` tail stage that reproduces the production
    ``AllocationError`` cascade.

    Cross-process caveat: on a *single-node* local Ray cluster the
    remote ``_get_node_info_from_current_node`` task is scheduled on
    the same node that started Ray (the driver host) and Ray re-imports
    ``resources`` in the worker process, picking up the patched module
    attribute. We verify the patch actually took effect by calling
    ``get_local_gpu_info()`` back through the patched module reference.
    On a multi-node cluster this trick would not propagate, but
    multi-node clusters by construction have real GPUs, so the patch is
    unnecessary there.
    """
    fake_gpus = [resources.GpuInfo(index=i, name=f"FakeGPU-{i}", uuid_=uuid.uuid4()) for i in range(count)]
    monkeypatch.setattr(resources, "get_local_gpu_info", lambda: list(fake_gpus))
    # Sanity-check the patch reaches the symbol consumed by Ray's worker
    # import path. A regression here (e.g. the producer moves modules)
    # would otherwise surface as a hard-to-diagnose "GPU stage starved"
    # failure inside the test body.
    observed = resources.get_local_gpu_info()
    assert len(observed) == count, (
        f"_install_fake_gpus: expected {count} fake GPUs, got {len(observed)}; "
        "patch likely missed the symbol consumed by ``_get_node_info_from_current_node``."
    )


@pytest.mark.slow
def test_autoscaler_chain_with_gpu_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce the AllocationError cascade.

    Layout mirrors the failing production pipeline:

    - Stage 0 ``producer``:  cpu=1, fast      (a lot of input for stage 1)
    - Stage 1 ``transcode``: cpu=5, slow
    - Stage 2 ``prep``:      cpu=3, medium
    - Stage 3 ``gpu_tail``:  cpu=1, gpu=1

    Two strict assertions:

    1. Pipeline completes within a wall-clock upper bound. In the
       production failure, the GPU stage never instantiated and Slurm
       killed the job on inactivity timeout -- here pytest's per-test
       timeout would fire instead.
    2. ``counts['gpu_tail'] >= 1`` -- the GPU stage actually materialized
       at least one worker. This is the precise negation of the
       production cascade, which was "CPU companion could not be
       placed, GPU stage never created, GPU idle".
    """
    _install_fake_gpus(monkeypatch, count=4)
    _init_ray_for_autoscale_test(monkeypatch, num_cpus=100, num_gpus=4)
    try:
        tracker = _WorkerTracker.options(name="chain_tracker").remote()  # type: ignore[attr-defined]
        try:
            spec = pipelines_v1.PipelineSpec(
                input_data=list(range(60)),
                stages=[
                    _FixedCpuStage("producer", cpus=1, process_dur=0.05, tracker_name="chain_tracker"),
                    _FixedCpuStage("transcode", cpus=5, process_dur=0.4, tracker_name="chain_tracker"),
                    _FixedCpuStage("prep", cpus=3, process_dur=0.2, tracker_name="chain_tracker"),
                    _FixedCpuStage(
                        "gpu_tail",
                        cpus=1,
                        gpus=1,
                        process_dur=0.3,
                        tracker_name="chain_tracker",
                    ),
                ],
                config=pipelines_v1.PipelineConfig(
                    logging_interval_s=2,
                    mode_specific=pipelines_v1.StreamingSpecificSpec(
                        autoscale_interval_s=1,
                        autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.DEBUG,
                    ),
                ),
            )
            start = time.monotonic()
            pipelines_v1.run_pipeline(spec)
            elapsed = time.monotonic() - start
            counts: dict[str, int] = ray.get(tracker.counts.remote())  # type: ignore[attr-defined]
            assert elapsed < 180.0, f"Chain pipeline took {elapsed:.1f}s; possible deadlock; counts={counts}"
            assert counts.get("gpu_tail", 0) >= 1, f"GPU tail starved (production cascade reproduced): counts={counts}"
        finally:
            ray.kill(tracker)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_autoscaling()
