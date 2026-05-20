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


from __future__ import annotations

import abc
import copy
import enum
import multiprocessing
import typing
from typing import Any, Generic, Optional, Sequence

import attrs

from cosmos_xenna import file_distribution
from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.pipelines.private.continuous_wrapped_stage import ContinuousWrappedStage
from cosmos_xenna.ray_utils import runtime_envs, stage
from cosmos_xenna.ray_utils.continuous_stage import ContinuousInterface
from cosmos_xenna.utils import approx, attrs_utils
from cosmos_xenna.utils.verbosity import VerbosityLevel

# Aliased to avoid shadowing the ``stage`` module import inside the
# ``StageAndParams`` class body, where the field name ``stage`` makes
# the bare module reference ambiguous to type checkers.
StageParams = stage.Params

T = typing.TypeVar("T")
V = typing.TypeVar("V")


_DEFAULT_SLOTS_PER_ACTOR = 2
_DEFAULT_LOG_INTERVAL_S = 60


class ExecutionMode(enum.Enum):
    """
    Enumeration for the mode of execution, defining how data processing should be handled.

    See README.md for more info.
    """

    # All stages are processing data concurrently. This is usually what we want to run when running
    # production workloads on the cloud
    STREAMING = 0
    # Stages are processed sequentially and all the data is materialized between stages. Generally,
    # we cannot run this for production workloads as we generally do not have enough local storage to
    # materialize intermediate data products. Additionally, for many pipelines, streaming mode can be
    # more efficient at processing.
    BATCH = 1
    # Online serving mode where stages run continuously. Similar to STREAMING but designed for online
    # serving scenarios where requests arrive dynamically via an input source queue and results are
    # pushed to output a sink queue. Workers remain active indefinitely waiting for new requests.
    SERVING = 2


class Stage(abc.ABC, Generic[T, V]):
    """Abstract base class representing a processing stage in a Ray data processing pipeline.

    This class serves as a foundation for building stages in yotta pipelines.
    Each stage can perform specific data transformations, with flexible resource allocation for GPU or CPU processing.

    Resource Allocation Rules:
    - Exactly one of num_gpus_per_worker or num_cpus_per_worker must be non-None
    - GPU stages automatically get allocated 1 CPU in addition to their GPU allocation
    - Both GPU and CPU allocations can be fractional (e.g., 0.5 GPU or 0.5 CPU)
    - For fractional GPU allocations, multiple workers can share the same GPU

    Worker Assignment Behavior:
    - CPU workers: Assignment is only relevant at the node level. Workers assigned to different CPUs
      on the same node functionally behave the same way
    - GPU workers: Ray manages CUDA environment variables to ensure each worker only sees its
      assigned GPU(s). For CPU-only stages, CUDA variables will point to no GPUs

    Environment Management:
    - Each stage can run in a separate conda environment
    - The environment can be specified either by:
      1. Implementing the conda_env_name property
      2. Using the environment specified by the model (if a model is set)

    See README.md for more information.
    """

    @property
    def stage_batch_size(self) -> int:
        """The number of samples to process at a time.

        This is used to determine how many samples to process at a time.
        """
        return 1

    @property
    @abc.abstractmethod
    def required_resources(self) -> resources.Resources:
        """The new way to specify resources required for a stage.

        Return a ray_utils.Resources object which represents the size/shape of each worker in this stage.
        If None, inherit from the model's required resources.

        This `Resources` class provides an intuitive interface for specifying resource requirements
        that get translated into more detailed internal worker shapes. Here's how the
        resource specifications map to different worker shapes and their allocation
        behaviors:

        1. CPU-Only Shape:
            - Set cpus > 0
            - Leave gpus = 0
            Example: Resources(cpus=2.0)
            Allocation behavior:
                - Only allocated CPU cores, no GPU resources
                - Multiple workers can share the same CPU cores through fractional allocation
                - Never allocated to GPU resources even if available
                - Ray/Yotta does not actually keep track of what particular cores are assigned to particular workers.
                  Instead, for each node, the cpus are treated as a big pool.

        2. Fractional GPU Shape (sharing GPUs):
            - Set gpus to value between 0 and 1 exclusive
            - Optionally set cpus
            Example: Resources(cpus=1.0, gpus=0.5)
            Allocation behavior:
                - Gets allocated fraction of a single GPU's compute capacity
                - Multiple workers can share same GPU up to 100% total utilization

        3. Whole Numbered GPU Shape:
            - Set gpus to integer ≥ 1
            - Optionally set cpus
            Example: Resources(cpus=1.0, gpus=2)
            Allocation behavior:
                - Gets allocated requested number of whole GPUs
                - Each GPU is allocated exclusively (not shared)
                - System optimizes GPU selection to minimize fragmentation

        4. SPMD Shape (Distributed Inference):
            - Set gpus to integer ≥ 1
            - Set is_spmd=True
            Example: Resources(cpus=1.0, gpus=8, is_spmd=True)
            Allocation behavior:
                - Allocated 8 GPUs (prefers to allocate within a single node)
                - Creates one actor per GPU
                - Emulates torchrun behavior by setting the proper environment variables
                - CUDA_VISIBLE_DEVICES is NOT modified (follows torchrun convention)
                - Required for large models needing tensor parallelism or multi-node inference

        SPMD Use Cases:
            - Large models requiring tensor parallelism (e.g., 241B parameter models)
            - vLLM multi-node inference (use distributed_executor_backend="external_launcher")
            - PyTorch models that only support multi-GPU through distributed coordination
            - Any inference workload requiring coordination across multiple GPUs/nodes

        Resource Allocation Strategy:
        The system uses a fragmentation-aware allocation strategy that:
        - Minimizes resource fragmentation across the cluster
        - Tries to keep related resources (GPU compute) together
        - Prefers allocations that maintain flexibility for future requests
        - Can reuse recently freed allocations to prevent thrashing
        - Balances load across available nodes while respecting constraints
        """
        pass

    @property
    def env_info(self) -> runtime_envs.RuntimeEnv | None:
        """Returns the name of the Conda environment for this stage.

        Can be overwritten by subclasses if needed.

        If this is None, we run this stage in the yotta-core env.

        If a model is present, we use that model's conda env name.
        """
        return None

    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        """Returns a list of download requests for this stage.

        This property allows stages to specify artifacts (model weights, conda environments, etc.) that need to be
        downloaded to all cluster nodes before pipeline execution begins. The distributed P2P download
        system will efficiently download these files once and share them between nodes.

        Before starting the pipeline, Xenna will use xenna.file_distribution.download_distributed to download
        all requests from all stages to all nodes. This ensures that required files are available locally
        on each node, eliminating the need for repeated downloads during pipeline execution.

        Returns:
            A list of DownloadRequest objects specifying files to download. Each request can be:
            - ObjectDownloadRequest: Download a single file (with optional unpacking)
            - PrefixDownloadRequest: Download all files under a prefix/directory

        Example:
            ```python
            @property
            def download_requests(self) -> list[file_distribution.DownloadRequest]:
                return [
                    # Download a model file
                    file_distribution.DownloadRequest(
                        value=file_distribution.ObjectDownloadRequest(
                            uri="models/pytorch_model.bin",
                            destination=pathlib.Path("/tmp/model.bin"),
                        )
                    ),
                    # Download and extract an archive
                    file_distribution.DownloadRequest(
                        value=file_distribution.ObjectDownloadRequest(
                            uri="datasets/data.tar.gz",
                            destination=pathlib.Path("/tmp/data.tar.gz"),
                            unpack_options=file_distribution.UnpackOptions(
                                destination=pathlib.Path("/tmp/extracted/"),
                            ),
                        )
                    ),
                    # Download all files under a prefix
                    file_distribution.DownloadRequest(
                        value=file_distribution.PrefixDownloadRequest(
                            uri="tokenizer/", destination=pathlib.Path("/tmp/tokenizer/")
                        )
                    ),
                ]
            ```

        Note:
            - Files are guaranteed to be available locally when setup_on_node(), setup() and process_data() are called
            - The system uses the Rust object_store crate for high-performance storage access
            - URIs in download requests must be RELATIVE to the ObjectStoreConfig base URI
            - The system uses intelligent caching to avoid re-downloading unchanged files
            - Large files are automatically chunked and distributed via P2P for efficiency
            - Authentication is handled via DistributedDownloadConfig passed to run_pipeline()

        See:
            - cosmos_xenna/file_distribution/README.md for detailed documentation
            - DistributedDownloadConfig for authentication and performance tuning
        """
        return []

    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        """Sets up a worker in this stage.

        Can be overwritten by subclasses if needed.

        This is called on every newly created worker in this stage. Typically, this would be used to load a model into
        gpu or create an S3 client.
        """
        pass

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        """Sets up a worker in this stage.

        Can be overwritten by subclasses if needed.

        This is called on every newly created worker in this stage. Typically, this would be used to load a model into
        gpu or create an S3 client.

        By default, if a model is present, this will call the model's setup function.
        """
        pass

    def destroy(self) -> None:
        """Release per-worker resources before the actor exits.

        Called by ``StageWorker.shutdown`` while the worker still has its conda env, GPU
        attachments, and Python interpreter intact - before ``ray.kill()`` SIGKILLs the
        actor. Stages that own external resources (vLLM EngineCore subprocesses, torch
        CUDA contexts, native handles) should override this to free them synchronously
        so that ``ray.kill()`` never has to terminate them out from under the driver,
        which would leak GPU memory as ghost CUDA contexts.

        Default is a no-op for stages with no such ownership. Implementations should be
        bounded in time and tolerate being called even when ``setup`` did not complete.
        Exceptions are caught by the caller and logged; they will not block teardown.
        """
        pass

    @abc.abstractmethod
    def process_data(self, in_data: list[T]) -> list[V] | None:
        """Processes the input data.

        This method must be implemented by subclasses to define specific data processing logic.

        Args:
            in_data: Input data to be processed.

        Returns:
            The result of processing the input data. This can be any pickleble type. This can also be None. If this is
            None, the data will not be passed to the next stage. This is useful if you want to ignore this piece of
            data for the rest of the pipeline. For example, if the input data was invalid.
        """
        pass


def validate_stage(stage: Stage[Any, Any], cluster_resources: resources.ClusterResources) -> None:
    stage.required_resources.to_worker_shape(cluster_resources)


@attrs.define
class StageSpec(typing.Generic[T, V]):
    """Specification for a pipeline stage.

    This class defines the configuration for a pipeline stage, including the worker to be used and various optional
    parameters.
    """

    stage: Stage[T, V]
    # Hard-coded number of workers to use for this stage. If this and num_workers are both None, we let the scheduling
    # algorithm decide.
    num_workers: int | None = None
    # Hard-coded number of workers per node to use for this stage. If this and num_workers are both None, we let the
    # scheduling algorithm decide.
    num_workers_per_node: float | None = None

    # The following parameters correspond to parameters in PipelineSpec.
    # For this stage, if these values are None, we use the values set in PipelineSpec. Otherwise, these parameters
    # take precedent. See PipelineSpec for documentation on the parameters.
    num_setup_attempts_python: int | None = None
    num_run_attempts_python: int | None = None
    ignore_failures: bool | None = None
    reset_workers_on_failure: bool | None = None
    slots_per_actor: int | None = attrs.field(default=None, validator=attrs_utils.validate_optional_positive_int)
    worker_max_lifetime_m: int | None = None
    worker_restart_interval_m: int | None = None
    max_setup_failure_percentage: float | None = None

    # Over-provision factor for this stage. It is applied to the measured processing
    # speed of the stage to influence the worker allocation. Honoured only when
    # ``StreamingSpecificSpec.scheduler == SchedulerKind.FRAGMENTATION_BASED``;
    # the saturation-aware scheduler ignores this field.
    over_provision_factor: float | None = None

    # Per-stage override for the saturation-aware scheduler. When ``None`` the
    # scheduler resolver falls through to ``SaturationAwareConfig.per_stage_overrides``
    # and then ``SaturationAwareConfig.stage_defaults``. Honoured only when
    # ``StreamingSpecificSpec.scheduler == SchedulerKind.SATURATION_AWARE``.
    saturation_aware: "SaturationAwareStageConfig | None" = None

    def name(self, index: int | None = None) -> str:
        if index is None:
            return str(type(self.stage).__name__)
        else:
            return f"Stage {index:02d} - {type(self.stage).__name__}"

    def validate(self, cluster_resources: resources.ClusterResources) -> None:
        if self.num_workers is not None and self.num_workers_per_node is not None:
            raise ValueError(
                "Expected only one of self.num_workers and self.num_workers_per_node to be non-None. "
                f"However, got {self.num_workers=} and {self.num_workers_per_node=}"
            )
        validate_stage(self.stage, cluster_resources)

    def override_with_pipeline_params(self, p: PipelineConfig) -> StageSpec:
        """Maybe override some fields using the global params.

        The StageSpec and PipelineSpec share some params we want to override the stage with the global params if the
        stage params are None.
        """
        c = copy.deepcopy(self)

        def _override_if_none(attr_name: str):  # noqa: ANN202
            if getattr(c, attr_name) is None:
                setattr(c, attr_name, getattr(p, attr_name))

        _override_if_none("num_setup_attempts_python")
        _override_if_none("num_run_attempts_python")
        _override_if_none("ignore_failures")
        _override_if_none("reset_workers_on_failure")
        _override_if_none("slots_per_actor")
        _override_if_none("worker_max_lifetime_m")
        _override_if_none("worker_restart_interval_m")
        _override_if_none("max_setup_failure_percentage")
        return c


class SchedulerKind(str, enum.Enum):
    """Streaming-mode autoscaler implementation.

    ``FRAGMENTATION_BASED`` (default) selects the Rust-backed
    ``FragmentationBasedAutoscaler``. ``SATURATION_AWARE`` selects the
    pure-Python saturation-aware scheduler. The flag is read once at
    ``Autoscaler.__init__`` and frozen for the lifetime of the run.
    """

    FRAGMENTATION_BASED = "fragmentation_based"
    SATURATION_AWARE = "saturation_aware"


@attrs.define
class SaturationAwareStageConfig:
    """Per-stage tunables for the saturation-aware scheduler.

    Each stage may use a different instance, allowing workloads with
    materially different signal profiles to coexist on the same cluster
    (e.g. a stage with multi-minute model warmup alongside a stage that
    warms in seconds).

    Default values apply unless overridden via:

    1. ``StageSpec.saturation_aware`` (programmatic, per stage)
    2. ``SaturationAwareConfig.per_stage_overrides[stage_name]`` (cluster-level keyed by name)
    3. ``SaturationAwareConfig.stage_defaults`` (cluster-wide default)

    Detailed tuning guidance per workload class lives in
    ``cosmos-xenna/docs/scheduler/saturation-aware/tuning.md``.
    """

    # Minimum cycles with at least one ready actor before the classifier is
    # trusted for that stage. Acts as a count-based data-sufficiency gate.
    min_data_points: int = attrs.field(default=5, validator=attrs_utils.validate_positive_int)

    # Operator-facing aggressiveness knob. The Halfin-Whitt parameter (beta)
    # in the K/sqrt(c) Erlang-C M/M/c knee formula. Higher values make every
    # stage's saturation threshold trigger sooner; the default 0.30 is the
    # canonical balanced QED value. Power users tune this single primary knob;
    # the explicit threshold overrides below stay available for stage-level
    # pinning. Range: [0.10, 0.60]. Tuning guidance lives in
    # ``cosmos-xenna/docs/scheduler/saturation-aware/08-auto-derived-thresholds.md``.
    saturation_aggressiveness: float = attrs.field(
        default=0.30,
        validator=attrs.validators.and_(attrs.validators.ge(0.10), attrs.validators.le(0.60)),
    )
    # Slots-empty fraction below which a stage is classified SATURATED
    # (sustained -> ordinary scale-up). ``None`` (the default) auto-derives
    # on the first ``autoscale()`` cycle from ``saturation_aggressiveness``
    # and the stage's runtime ``slots_per_worker`` via the K/sqrt(c) formula
    # clamped to ``[auto_threshold_min, auto_threshold_max]``. An explicit
    # numeric value pins the threshold for this stage and bypasses the
    # formula. Cross-field: the resolved triple must satisfy
    # activation < saturation < over_provisioned_threshold (enforced at
    # resolution time).
    #
    # Migration note: prior to the K/sqrt(c) auto-derivation the field
    # defaulted to 0.15 (and ``activation_threshold`` to 0.05). The new
    # ``None`` / auto default produces different values per
    # ``slots_per_actor``; e.g. at the project default ``slots_per_actor=2``
    # the auto value is ~0.21 (saturation) / ~0.07 (activation), ~40% more
    # aggressive than the legacy default. Pin both fields to ``0.15`` and
    # ``0.05`` to preserve the legacy behaviour exactly.
    saturation_threshold: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0))),
    )
    # Slots-empty fraction below which a stage is classified SATURATED_CRITICAL
    # (burst response). ``None`` (the default) auto-derives on the first
    # ``autoscale()`` cycle as ``saturation * activation_to_saturation_ratio``.
    # An explicit numeric value pins the threshold for this stage. Cross-field:
    # the resolved value must be strictly less than the resolved
    # ``saturation_threshold`` (enforced at resolution time).
    #
    # Migration note: see ``saturation_threshold`` above. Pin both
    # ``saturation_threshold`` and this field to the legacy 0.15 / 0.05 to
    # preserve pre-auto-derivation behaviour exactly.
    activation_threshold: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0))),
    )
    # Lower clamp on the auto-derived ``saturation_threshold``. Acts as a
    # safety floor: even at very large ``c`` the formula cannot pin the
    # threshold below this value, preventing the classifier from firing on
    # single-slot transients in 100+ slot stages.
    auto_threshold_min: float = attrs.field(
        default=0.02,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(0.10)),
    )
    # Upper clamp on the auto-derived ``saturation_threshold``. Acts as a
    # safety ceiling: even at ``c=1`` the formula cannot exceed this value.
    # Must be strictly less than ``over_provisioned_threshold`` so the
    # auto-derived saturation never collides with the over-provisioned
    # zone (cross-field validator below).
    auto_threshold_max: float = attrs.field(
        default=0.45,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0)),
    )
    # Fraction of the resolved saturation threshold at which the
    # SATURATED_CRITICAL zone begins; consumed only when
    # ``activation_threshold`` is auto-derived. Default 0.33 reproduces the
    # legacy ``0.05 / 0.15 = 0.33`` ratio.
    activation_to_saturation_ratio: float = attrs.field(
        default=0.33,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.lt(1.0)),
    )
    # Slots-empty fraction above which a stage is classified OVER_PROVISIONED
    # (sustained -> scale-down). Not auto-derived: the over-provisioned zone
    # sits in the flat tail of the M/M/c response-time curve and is largely
    # c-insensitive, so a fixed default works for almost any c. Cross-field:
    # must be > the resolved saturation threshold (validated at resolution
    # time).
    over_provisioned_threshold: float = attrs.field(
        default=0.50,
        validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0)),
    )

    # Fractional band around saturation_threshold within which the classifier
    # holds the previous state to prevent edge oscillation.
    saturation_deadband_pct: float = attrs.field(
        default=0.15,
        validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0)),
    )
    # Fractional band around over_provisioned_threshold; conventionally larger
    # than the saturation-side band so scale-down requires stronger evidence.
    over_provisioned_deadband_pct: float = attrs.field(
        default=0.30,
        validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(1.0)),
    )

    # Cycles a stage must remain SATURATED before scale-up is applied.
    saturated_streak_min_cycles: int = attrs.field(default=2, validator=attrs_utils.validate_positive_int)
    # Cycles in SATURATED_CRITICAL before burst delta is applied.
    saturated_critical_streak_min_cycles: int = attrs.field(default=1, validator=attrs_utils.validate_positive_int)
    # Cycles a stage must remain OVER_PROVISIONED before scale-down is applied.
    # Cross-field: must dominate saturated_streak_min_cycles for asymmetric stabilization.
    over_provisioned_streak_min_cycles: int = attrs.field(default=30, validator=attrs_utils.validate_positive_int)
    # Cycles a stage must remain STARVED before logging the upstream-bottleneck warning.
    starved_streak_min_cycles: int = attrs.field(default=6, validator=attrs_utils.validate_positive_int)

    # When True, per-stage growth mode (ACQUIRING / TRACKING / HOLD) shapes
    # the per-cycle delta. When False, growth uses fixed TRACKING values.
    enable_growth_mode_state_machine: bool = True
    # ACQUIRING-mode multiplicative growth factor on SATURATED_CRITICAL:
    # delta = ceil(factor * current_workers), capped by aggressive_growth_max_per_cycle.
    acquiring_critical_growth_factor: float = attrs.field(default=0.5, validator=attrs.validators.gt(0.0))
    # ACQUIRING-mode multiplicative growth factor on SATURATED.
    acquiring_saturated_growth_factor: float = attrs.field(default=0.25, validator=attrs.validators.gt(0.0))
    # TRACKING-mode absolute growth count on SATURATED_CRITICAL.
    tracking_critical_growth_count: int = attrs.field(default=2, validator=attrs_utils.validate_positive_int)
    # TRACKING-mode absolute growth count on SATURATED.
    tracking_saturated_growth_count: int = attrs.field(default=1, validator=attrs_utils.validate_positive_int)
    # HOLD-mode growth count on SATURATED_CRITICAL (HOLD allows only burst response).
    hold_critical_growth_count: int = attrs.field(default=1, validator=attrs.validators.ge(0))
    # HOLD-mode growth count on SATURATED (typically zero so HOLD blocks
    # all non-critical growth during post-shrink stabilization).
    hold_saturated_growth_count: int = attrs.field(default=0, validator=attrs.validators.ge(0))
    # Hard cap on the per-cycle additions any growth decision may produce.
    aggressive_growth_max_per_cycle: int = attrs.field(default=4, validator=attrs_utils.validate_positive_int)

    # Weight on the NEW sample in the EWMA recursion:
    #   smoothed = level * new + (1 - level) * prior
    # Higher values are more responsive (less smoothing);
    # lower values smooth more heavily.
    # Default 0.20 gives ~3-cycle half-life.
    # 0.0 is rejected (would freeze the smoothed value forever).
    slots_empty_ratio_smoothing_level: float = attrs.field(
        default=0.20,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0)),
    )

    # Recommendation history depth for the scale-up direction.
    stabilization_window_cycles_up: int = attrs.field(default=1, validator=attrs_utils.validate_positive_int)
    # Recommendation history depth for the scale-down direction.
    # Cross-field: must be > stabilization_window_cycles_up for asymmetric stabilization.
    stabilization_window_cycles_down: int = attrs.field(default=30, validator=attrs_utils.validate_positive_int)

    # Maximum fraction of a stage's current actors that may be deleted in a
    # single cycle. Prevents cliff scale-downs that starve downstream stages.
    max_scale_down_fraction_per_cycle: float = attrs.field(
        default=0.05,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0)),
    )

    # Setup-phase quiescence gate. When True (default) and a stage has any
    # pending actors at the start of an autoscale cycle, the saturation-aware
    # scheduler suppresses scale-up decisions for that stage until at least
    # one of the pending actors becomes ready. Cold-start (pending > 0 and
    # ready == 0) skips the entire intent pipeline so the classifier streak
    # and stabilization-window buffer are not polluted by zero-signal cycles;
    # hot-pending (pending > 0 and ready > 0) clamps positive intents to 0
    # so Phase C does not pile additional adds on top of an in-flight setup.
    # Phase D scale-down and Phase B floor are unaffected.
    setup_phase_quiescence_enabled: bool = True
    # Per-worker measurement grace: after a worker first appears in the
    # ready snapshot, its slot samples are excluded from per-stage averages
    # until this many seconds elapse. This keeps freshly-ready actors from
    # biasing the EWMA while the dispatcher is still filling their queues.
    worker_warmup_measurement_grace_s: float = attrs.field(default=60.0, validator=attrs.validators.ge(0.0))
    # Donor warmup grace: after a worker first appears in the ready
    # snapshot, it is excluded from Phase D and saturation-donor victim
    # pools until this many seconds elapse.
    donor_warmup_grace_s: float = attrs.field(default=180.0, validator=attrs.validators.ge(0.0))

    # Optional cluster-wide minimum workers for this stage. ``None`` means the
    # implicit one-worker floor (1) applies. Useful for pre-warming stages with
    # long model load. When the cluster cannot satisfy the floor during the
    # worker-floor enforcement step, the scheduler raises ``RuntimeError``
    # (fail-fast on infeasible config).
    min_workers: int | None = attrs.field(default=None, validator=attrs_utils.validate_optional_positive_int)
    # Optional per-node minimum workers for this stage. ``None`` means no
    # per-node floor. Composes with ``min_workers``: effective floor is
    # ``max(min_workers, min_workers_per_node * num_nodes)`` when both are set.
    min_workers_per_node: int | None = attrs.field(default=None, validator=attrs_utils.validate_optional_positive_int)
    # Optional cluster-wide cap on the number of workers for this stage.
    # ``None`` means unbounded (cluster capacity is the ceiling). Cross-field:
    # must be >= min_workers when both are set.
    max_workers: int | None = attrs.field(default=None, validator=attrs_utils.validate_optional_positive_int)
    # Optional per-node cap on workers of this stage. ``None`` means
    # unbounded by this knob. Cross-field: must be >= min_workers_per_node.
    max_workers_per_node: int | None = attrs.field(default=None, validator=attrs_utils.validate_optional_positive_int)

    # Setup-aware queue cap: lower the upstream ``max_queued`` cap while the
    # stage is in cold start (pending actors exist, no ready actors yet).
    setup_aware_max_queued: bool = True

    def __attrs_post_init__(self) -> None:
        """Validate cross-field invariants only.

        Single-field constraints (positive integers, fractions in [0, 1], ...)
        are enforced by ``attrs.field(validator=...)`` on each field above.
        This method handles invariants that span two or more fields.

        Raises:
            ValueError: When two or more fields are set to mutually
                inconsistent values.
        """
        # Threshold ordering - classifier zones must not overlap. Only
        # validated here when both auto-derivable thresholds are explicit
        # (the resolver enforces the same invariant on the resolved values
        # when one or both are auto).
        if self.saturation_threshold is not None and self.activation_threshold is not None:
            if not (self.activation_threshold < self.saturation_threshold < self.over_provisioned_threshold):
                msg = (
                    f"Threshold ordering violated: "
                    f"activation_threshold={self.activation_threshold} must be < "
                    f"saturation_threshold={self.saturation_threshold} must be < "
                    f"over_provisioned_threshold={self.over_provisioned_threshold}"
                )
                raise ValueError(msg)
        elif self.saturation_threshold is not None:
            # Saturation pinned, activation auto. The pinned saturation must
            # at least leave room for the over-provisioned ordering.
            if not (self.saturation_threshold < self.over_provisioned_threshold):
                msg = (
                    f"Threshold ordering violated: "
                    f"saturation_threshold={self.saturation_threshold} must be < "
                    f"over_provisioned_threshold={self.over_provisioned_threshold}"
                )
                raise ValueError(msg)
        elif self.activation_threshold is not None:
            # Activation pinned, saturation auto. The pinned activation must
            # not already exceed the over-provisioned threshold (defensive --
            # the resolver also re-checks this against the resolved saturation).
            if not (self.activation_threshold < self.over_provisioned_threshold):
                msg = (
                    f"Threshold ordering violated: "
                    f"activation_threshold={self.activation_threshold} must be < "
                    f"over_provisioned_threshold={self.over_provisioned_threshold}"
                )
                raise ValueError(msg)

        # Auto-threshold clamps - the auto-derived saturation threshold
        # must be drawn from a non-empty interval; the upper clamp must
        # leave room for the lower one.
        if not (self.auto_threshold_min < self.auto_threshold_max):
            msg = (
                f"auto_threshold_min ({self.auto_threshold_min}) must be < "
                f"auto_threshold_max ({self.auto_threshold_max})"
            )
            raise ValueError(msg)

        # Auto-threshold ceiling vs over-provisioned floor - any auto-derived
        # saturation that hits the upper clamp must still be strictly below
        # the over-provisioned threshold; otherwise the resolver's zone
        # ordering fails when a regime-aware lift pushes the formula output
        # into the clamp at small slots_per_actor.
        if not (self.auto_threshold_max < self.over_provisioned_threshold):
            msg = (
                f"auto_threshold_max ({self.auto_threshold_max}) must be < "
                f"over_provisioned_threshold ({self.over_provisioned_threshold}) "
                "so the auto-derived saturation never collides with the "
                "over-provisioned zone."
            )
            raise ValueError(msg)

        # Slow-start grace ordering - donor grace must cover worker grace.
        if self.donor_warmup_grace_s < self.worker_warmup_measurement_grace_s:
            msg = (
                f"donor_warmup_grace_s ({self.donor_warmup_grace_s}) must be >= "
                f"worker_warmup_measurement_grace_s ({self.worker_warmup_measurement_grace_s})"
            )
            raise ValueError(msg)

        # Streak ordering - shrink streak must dominate growth streak.
        if self.over_provisioned_streak_min_cycles <= self.saturated_streak_min_cycles:
            msg = (
                f"over_provisioned_streak_min_cycles ({self.over_provisioned_streak_min_cycles}) must be "
                f"strictly > saturated_streak_min_cycles ({self.saturated_streak_min_cycles}) "
                f"(asymmetric stabilization)"
            )
            raise ValueError(msg)

        # Stabilization windows - down dominates up.
        if self.stabilization_window_cycles_down <= self.stabilization_window_cycles_up:
            msg = (
                f"stabilization_window_cycles_down ({self.stabilization_window_cycles_down}) must be > "
                f"stabilization_window_cycles_up ({self.stabilization_window_cycles_up}) "
                f"(asymmetric stabilization)"
            )
            raise ValueError(msg)

        # Min <= max relations (when both sides set).
        if self.min_workers is not None and self.max_workers is not None and self.min_workers > self.max_workers:
            msg = f"min_workers ({self.min_workers}) must be <= max_workers ({self.max_workers})"
            raise ValueError(msg)
        if (
            self.min_workers_per_node is not None
            and self.max_workers_per_node is not None
            and self.min_workers_per_node > self.max_workers_per_node
        ):
            msg = (
                f"min_workers_per_node ({self.min_workers_per_node}) must be <= "
                f"max_workers_per_node ({self.max_workers_per_node})"
            )
            raise ValueError(msg)


@attrs.define
class SaturationAwareConfig:
    """Cluster-level (global) tunables for the saturation-aware scheduler.

    Holds cluster-wide configuration plus the per-stage default and explicit
    per-stage override registry. Stage-local tunables live on
    ``SaturationAwareStageConfig``; see that class for resolution order.
    Detailed tuning guidance per workload class lives in
    ``cosmos-xenna/docs/scheduler/saturation-aware/tuning.md``.
    """

    # Cycle period for the autoscaler control loop, in seconds. Effective
    # response time is ``interval_s * streak_min_cycles``.
    interval_s: float = attrs.field(default=10.0, validator=attrs.validators.gt(0.0))

    # When True, detect the cluster's Halfin-Whitt regime per cycle and lift
    # ``saturation_aggressiveness`` by ``super_halfin_whitt_aggressiveness_lift``
    # whenever the cluster sits in the super-Halfin-Whitt regime.
    # The lift makes scale-up faster without changing scale-down.
    # Set False to pin the aggressiveness at its base value and skip
    # regime tracking entirely -
    # useful for diagnosis or queueing-theory-purity A/B comparisons.
    enable_regime_aware_aggressiveness: bool = attrs.field(
        default=True,
        validator=attrs.validators.instance_of(bool),
    )
    # Additive lift applied to ``saturation_aggressiveness`` when the
    # cluster is in super-Halfin-Whitt. Default 0.15 shifts the canonical
    # 0.30 base to 0.45 (the auto-derived ``saturation_threshold``
    # increases proportionally, so the classifier fires SATURATED earlier
    # in the empty-slot ratio band). Range ``[0.0, 0.5]``; setting to
    # 0.0 disables the numeric lift but still runs regime tracking.
    super_halfin_whitt_aggressiveness_lift: float = attrs.field(
        default=0.15,
        validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.le(0.5)),
    )
    # Consecutive cycles required to commit a regime transition (in either
    # direction). Hysteresis prevents a noisy ``cluster_idle_fraction``
    # oscillating around the regime threshold from flapping the effective
    # aggressiveness.
    regime_transition_streak_cycles: int = attrs.field(default=3, validator=attrs_utils.validate_positive_int)

    # When True, growth attempts are sorted by DAG depth descending
    # (downstream stages first). Reflects the invariant that pipeline
    # throughput is bounded by the tail stage.
    enable_dag_priority_growth: bool = True
    # When True, on allocation failure the scheduler may take a worker from
    # an upstream stage to feed the receiver (cluster-full rebalance).
    enable_cross_stage_donor: bool = True
    # When True, a donor must have strictly smaller DAG depth than the
    # receiver (never take from a downstream stage).
    donor_must_be_strictly_upstream: bool = True

    # When True, donor candidates must be in OVER_PROVISIONED state with a
    # full streak (not merely non-SATURATED). Strongest anti-flap layer.
    cross_stage_donor_require_over_provisioned: bool = True
    # When True, donors in HOLD growth mode are ineligible.
    cross_stage_donor_exclude_hold_state: bool = True
    # Cycles a stage must wait after donating before it can receive via
    # cross-stage logic. Cross-field: must dominate the longest stage's
    # over_provisioned_streak_min_cycles.
    cross_stage_donor_anti_flap_cycles: int = attrs.field(default=30, validator=attrs_utils.validate_positive_int)
    # Maximum cross-stage donations a single receiver may absorb in one cycle.
    cross_stage_donor_max_per_cycle: int = attrs.field(default=1, validator=attrs_utils.validate_positive_int)
    # Cycles a donor must wait between consecutive donations.
    cross_stage_donor_min_donation_interval_cycles: int = attrs.field(
        default=30, validator=attrs_utils.validate_positive_int
    )
    # Number of consecutive cycles the minimum-worker floor enforcement may
    # fail without receiver progress (cluster placement exhausted AND no
    # eligible cross-stage donor) before raising ``RuntimeError`` and failing
    # the pipeline. Default 6 autoscale cycles; wall-clock duration depends on
    # the scheduler loop cadence.
    # Set to 0 to disable the grace window: floor enforcement raises on the
    # first failed no-donor allocation. Counter resets on any successful add
    # or donation for the receiver stage. Post-donation retry failures raise
    # immediately because the donor removal cannot be rolled back safely.
    floor_stuck_grace_cycles: int = attrs.field(default=6, validator=attrs.validators.ge(0))

    # Fraction of ``interval_s`` above which a cycle's wall-clock duration
    # triggers a WARN log via ``loop_watchdog`` (cluster-wide). Range
    # ``(0.0, 1.0]``.
    cycle_time_warn_threshold: float = attrs.field(
        default=0.5,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0)),
    )

    # When True, ``MemoryPressureMonitor`` polls Ray's cluster object-store
    # memory and freezes Phase C scale-up when used fraction exceeds the
    # critical threshold below.
    enable_memory_pressure_gate: bool = True
    # Fraction of cluster object-store memory above which the gate freezes
    # Phase C scale-up.
    memory_pressure_critical_threshold: float = attrs.field(
        default=0.85,
        validator=attrs.validators.and_(attrs.validators.gt(0.0), attrs.validators.le(1.0)),
    )
    # Polling interval (seconds) for the Ray cluster memory query.
    memory_pressure_polling_interval_s: float = attrs.field(default=5.0, validator=attrs.validators.gt(0.0))

    # When True, an absorbed Phase C ``try_add_worker`` exception logs the
    # per-GPU fragmentation snapshot + bumps the failure counter and skips
    # the rest of Phase C for that cycle; when False, the exception
    # propagates so the run-loop crashes loudly.
    skip_cycle_on_allocation_error: bool = True
    # Consecutive Phase C cycles a stage must stay stuck before the
    # detector promotes the per-cycle WARN to a one-shot INFO.
    stuck_plan_detection_cycles: int = attrs.field(default=18, validator=attrs_utils.validate_positive_int)

    # Ratio of ``max(D_k) / min(D_k)`` (Forced-Flow service demand)
    # above which the cluster is considered heterogeneous enough that
    # the dominant bottleneck stage may benefit from a longer
    # ``over_provisioned_streak_min_cycles`` to absorb measurement
    # noise. Default 5.0 -- a homogeneous pipeline has a ratio of 1.0,
    # so the threshold must be strictly greater than 1.0. Values
    # between 3.0 and 10.0 are typical: lower values fire the tuning
    # log on mildly skewed pipelines (CPU prep + GPU caption);
    # higher values quiet the log unless one stage is an order of
    # magnitude slower than the rest.
    cluster_heterogeneity_warn_threshold: float = attrs.field(
        default=5.0,
        validator=attrs.validators.gt(1.0),
    )
    # Number of consecutive autoscale cycles the heterogeneity ratio
    # must stay above
    # ``cluster_heterogeneity_warn_threshold`` before exactly one
    # INFO tuning-recommendation log fires. Default 30 cycles ~= 5 min
    # at the default ``interval_s = 10.0``. The streak gate prevents
    # alert fatigue on transient ratio spikes (cold-start, brief
    # measurement noise) while sustained heterogeneity still
    # surfaces in operator log scrapes within minutes. After firing
    # once, the log re-arms only after the ratio drops back to or
    # below the threshold and climbs above it again.
    cluster_heterogeneity_warn_streak: int = attrs.field(
        default=30,
        validator=attrs_utils.validate_positive_int,
    )

    # Default ``SaturationAwareStageConfig`` applied to every stage that has
    # no more specific override.
    stage_defaults: SaturationAwareStageConfig = attrs.field(factory=SaturationAwareStageConfig)
    # Optional explicit per-stage overrides keyed by stage name. Each value is
    # a full ``SaturationAwareStageConfig``; copy ``stage_defaults`` and tweak
    # the fields you want to change. Lower precedence than
    # ``StageSpec.saturation_aware``.
    per_stage_overrides: dict[str, SaturationAwareStageConfig] = attrs.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        """Validate cluster-wide cross-field invariants for the configs known here.

        Single-field constraints are enforced by ``attrs.field(validator=...)``
        on each field above. This method validates invariants that span the
        cluster guardrails and the per-stage configs visible at construction
        time: ``stage_defaults`` and ``per_stage_overrides``. The
        higher-precedence ``StageSpec.saturation_aware`` overrides live on
        ``StageSpec``, not on this cluster config, so they are not yet
        available; ``SaturationAwareScheduler.__init__`` re-runs
        ``validate_effective_stage_configs(spec_overrides=...)`` once those
        runtime overrides are collected, applying the same invariants to the
        full three-tier set.

        Raises:
            ValueError: When two or more fields are set to mutually
                inconsistent values.
        """
        self.validate_effective_stage_configs()

    def validate_effective_stage_configs(
        self,
        spec_overrides: Sequence[SaturationAwareStageConfig] = (),
    ) -> None:
        """Validate cross-field invariants across every effective stage config.

        Called twice in the pipeline-build lifecycle:

        1. Eagerly from ``__attrs_post_init__`` with the default empty
           ``spec_overrides``, when only ``stage_defaults`` and
           ``per_stage_overrides`` are knowable. Catches misconfigured
           cluster configs at the constructor where the bad value was
           written.
        2. From ``SaturationAwareScheduler.__init__`` with the collected
           ``StageSpec.saturation_aware`` overrides, after the pipeline
           spec is fully assembled. Re-validates the same invariants
           against the full three-tier set so a higher-precedence
           override cannot silently weaken a cluster guardrail.

        The validated invariants are monotone in the set of configs (adding
        more configs to ``all_stage_configs`` can only raise the longest
        observed streak), so call 2 strictly extends call 1; both calls are
        required because they fire at different times.

        Args:
            spec_overrides: Highest-precedence stage configs collected from
                ``StageSpec.saturation_aware``. Empty (the default) when the
                cluster config is being constructed standalone, since those
                overrides are not yet collected at that point.

        Raises:
            ValueError: When cluster-wide guardrails are weaker than any stage
                config they must dominate.
        """
        all_stage_configs = [self.stage_defaults, *self.per_stage_overrides.values(), *spec_overrides]
        longest_shrink_streak = max(cfg.over_provisioned_streak_min_cycles for cfg in all_stage_configs)
        if self.cross_stage_donor_anti_flap_cycles < longest_shrink_streak:
            msg = (
                f"cross_stage_donor_anti_flap_cycles ({self.cross_stage_donor_anti_flap_cycles}) must be "
                f">= the longest over_provisioned_streak_min_cycles across all stage configs "
                f"({longest_shrink_streak}); otherwise the anti-flap safeguard is weaker than the "
                f"streak it must dominate."
            )
            raise ValueError(msg)

    def get_effective_stage_config(
        self,
        stage_name: str,
        spec_override: SaturationAwareStageConfig | None = None,
    ) -> SaturationAwareStageConfig:
        """Resolve the effective stage config using three-tier precedence.

        Args:
            stage_name: Name of the stage being looked up.
            spec_override: Override read from ``StageSpec.saturation_aware``,
                or ``None`` if the stage spec did not set one.

        Returns:
            The resolved ``SaturationAwareStageConfig`` for the stage.
            Lookup precedence: ``spec_override`` -> ``per_stage_overrides`` ->
            ``stage_defaults``.

        """
        if spec_override is not None:
            return spec_override
        if stage_name in self.per_stage_overrides:
            return self.per_stage_overrides[stage_name]
        return self.stage_defaults


@attrs.define
class StreamingSpecificSpec:
    # How often to run the stage auto-scaler.
    autoscale_interval_s: float = 60 * 3.0
    # Window size with which the auto-scaler estimates the processing speed of each stage.
    # Making it larger makes the estimate more stable, but also less responsive to changes.
    autoscale_speed_estimation_window_duration_s: float = 60 * 3.0
    # Minimum number of data points to keep even if they are outside the window.
    autoscale_speed_estimation_min_data_points: int = 5
    # In streaming mode, when the numeber of max queued tasks exceeds `num_actors * num_slots_per_actor`,
    # i.e. when there is no empty slot, Xenna applies a back-pressure to upstream stages to
    # prevent memory and storage from blowout. The 2 parameters below can help tune that behavior.
    # - This multiplier is applied as `num_actors * num_slots_per_actor * max_queued_multiplier`,
    #   i.e. when you have enough system memory, increase this value to (e.g.) 1.5 is typically beneficial.
    max_queued_multiplier: float = 1.0
    # - When certain stage is super fast and hence scaled down to e.g. just 1 actor, then the
    #   max_queued will be very small (e.g. 2 if slots_per_actor=2). This can make the pipeline
    #   unstable that performance fluctuation can cause downstream stages to get starved.
    #   So this parameter sets a lower bound on max_queued to prevent that.
    max_queued_lower_bound: int = 8
    # Add verbosity level for the autoscaler
    autoscaler_verbosity_level: VerbosityLevel = VerbosityLevel.NONE
    executor_verbosity_level: VerbosityLevel = VerbosityLevel.INFO
    # Backlog-aware scale-down guard.
    #
    # Default ``False`` - Rust-proposed worker deletions pass through unchanged.
    #
    # Set ``True`` to have the autoscaler clamp deletions so that the surviving workers
    # on each pool can still drain the queued backlog (upstream queue + this pool's own queue)
    # at the pre-scaling ``slots_per_actor``. This protects downstream stages from the drain-tail
    # starvation pattern where source completion triggers aggressive CPU-stage scale-down before
    # queued work has drained.
    enable_backlog_aware_scaledown: bool = False
    # Grace period applied to *newly Ready* worker groups before the autoscaler is
    # allowed to delete them. Worker groups still in setup are always protected
    # (independent of this value); the grace extends that protection for a short
    # window after they reach Ready.
    #
    # The motivation is preventing the "thrash" pattern where the Rust autoscaler
    # spins up a worker, the worker finishes an expensive setup (e.g. vLLM model
    # load + ``torch.compile`` for a large LLM, ~60-90 s), and the *very next*
    # autoscaler tick decides the queue is momentarily empty and tears the worker
    # down. The setup cost is then wasted and, for GPU stages, the SIGKILL-on-
    # destroy fallback raises the risk of orphaned CUDA contexts.
    #
    # Set to ``0.0`` to disable the post-Ready grace (pending-actor protection
    # still applies). End-of-stage teardown bypasses the grace via the
    # ``stages_is_dones`` flag so final drain is never delayed.
    scale_down_grace_after_ready_s: float = 60.0

    # Selects the autoscaler implementation. ``FRAGMENTATION_BASED`` (default)
    # uses the Rust-backed solver; ``SATURATION_AWARE`` uses the pure-Python
    # saturation-aware scheduler. The flag is read once at
    # ``Autoscaler.__init__`` and frozen for the lifetime of the run.
    scheduler: SchedulerKind = SchedulerKind.FRAGMENTATION_BASED
    # Configuration for the saturation-aware scheduler. Has no effect when
    # ``scheduler == SchedulerKind.FRAGMENTATION_BASED``.
    saturation_aware: SaturationAwareConfig = attrs.field(factory=SaturationAwareConfig)


@attrs.define
class DashboardSpec:
    port: int = 8080

    def get_ip(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@attrs.define
class PipelineConfig:
    # Execution mode to run the pipeline under. See ExecutionMode and README.md
    execution_mode: ExecutionMode = ExecutionMode.STREAMING
    # Number of attempts to try to call Stage.setup(). If this is > 1, we will log any exceptions and try the specified
    # number of times.
    num_setup_attempts_python: int = 1
    # Number of attempts to try to call Stage.process_data() per task. If this is > 1, we will log any exceptions and
    # try the specified number of times.
    num_run_attempts_python: int = 1
    # Sometimes, setup() can fail sporatically. This is often due to Lustre jankiness. It can be helpful to ignore these
    # failures and retry. If this is non-None, we will retry and then fail the pipeline if a stage fails to setup more
    # than this percentage. For example, if this value is 50 and we try to start 10 actors and 6 of them fail, we will
    # fail the pipeline. If 4 of them fail, we will continue running the pipeline.
    max_setup_failure_percentage: float | None = None
    # If true, any failures in "process_data" will be ignored. If this is false, failures will crash the pipeline.
    # Be careful with this, it can be helpful for catching rare errors, but can also cause a pipeline to run continually
    # in a very broken state. If you choose to set this to True, make sure to examine the logs of the pipeline to
    # check the health.
    ignore_failures: bool = False
    # If true, reset workers when a failure occurs. This can be helpful if you have some class of errors which break the
    # the GPU and only a reset worker can clear it.
    # NOTE: For now, this is only enbled if ignore_failures is set to True.
    reset_workers_on_failure: bool = False
    # Number of tasks to request concurrently per actor. This is an internal detail for streaming pipelines. We request
    # ray to process multiple tasks per worker (default 2). This forces Ray to pre-fetch data and should make it so we
    # are very unlikely to be blocked on IO.
    slots_per_actor: int = attrs.field(default=_DEFAULT_SLOTS_PER_ACTOR, validator=attrs_utils.validate_positive_int)
    # When work stealing is enabled, Xenna will steal queued tasks from busy actors and give them to idle actors.
    # Without this, Xenna can leave some nodes idle when the number of tasks supplied to the pipeline are less than
    # num_actors * num_slots_per_actor (typically == 2); i.e. when there are very few tasks.
    # Ideally, this would always be turned on. However, right now, work stealing can be slow for large jobs.
    enable_work_stealing: bool = False
    # When polling for completed tasks, Xenna groups inflight tasks into chunks and call ray.wait once per chunk.
    max_tasks_to_poll_per_chunk: int = 8
    # Maxmum lifetime in minutes for stage workers before getting terminated and restarted.
    worker_max_lifetime_m: int = 0
    # Interval in minutes between two over-lifetime restart within a stage's actor pool.
    worker_restart_interval_m: int = 1
    # How long to wait between loging pipeline status. Default is every 60 seconds.
    logging_interval_s: float = _DEFAULT_LOG_INTERVAL_S
    # If true, failed tasks will return Nones. This means that the task will not be retried.
    # Be careful with this, this may be the incorrect thing to do for your pipeline.
    failures_return_nones: bool = False
    # If true, the outputs of the last stage will be retained and returned from by `run_pipeline`, otherwise they will
    # be discarded and `run_pipeline` will return None. Retaining this data can be useful if you want to further process
    # it. However, users can also very easily forget that they are doing this and run our of memory.
    return_last_stage_outputs: bool = False
    # Logging verbosity control
    actor_pool_verbosity_level: VerbosityLevel = VerbosityLevel.INFO
    monitoring_verbosity_level: VerbosityLevel = VerbosityLevel.INFO
    # Mode specific parameters
    mode_specific: StreamingSpecificSpec | None = None
    # Whether to log the layout of the ray workers.
    # This can be useful for debugging scheduling/allocation, but is very verbose.
    log_worker_allocation_layout: bool = False
    # The percentage of CPU resources to allocate to the pipeline. This is used to leave some CPU resources for the
    # node manager and other internal ray processes.
    cpu_allocation_percentage: float = 0.95
    # If true, clear the CUDA_VISIBLE_DEVICES environment variable on CPU actors.
    # Otherwise, CUDA_VISIBLE_DEVICES will be set as they are on the node.
    # This is needed to turn off sometimes for libraries which require GPU access on import.
    clear_cuda_visible_devices_on_cpu_actors: bool = True


@attrs.define
class JobInfo:
    """Info about the pipeline job.

    This info can be used to tag reported pipeline metrics.
    """

    pipeline_type: str
    pipeline_version: str
    pipeline_mode: str


@attrs.define
class ServingQueues:
    source: multiprocessing.Queue
    sink: multiprocessing.Queue

    def __deepcopy__(self, memo: dict) -> "ServingQueues":
        # do a shallow copy
        return ServingQueues(self.source, self.sink)


@attrs.define
class PipelineSpec:
    """Specification for a simplified ray pipeline.

    This class encapsulates the configuration for the entire pipeline, including
    the input data and the sequence of stages.

    See ray_utils/README.md for more info.
    """

    # offline processing with pre-populated input data
    # TODO: Can we support a generator here?
    input_data: Sequence[Any]
    stages: Sequence[StageSpec | Stage]
    config: PipelineConfig = attrs.field(factory=PipelineConfig)
    job_info: Optional[JobInfo] = None

    # online serving with input queue to poll for new requests and output queue to push results
    serving_queues: ServingQueues | None = None

    def _format_stage_spec(self, stage_spec: StageSpec) -> str:
        stage = stage_spec.stage
        stage_info = f"   class_name: {type(stage).__name__}\n"
        stage_info += f"   required_resources: {stage.required_resources}\n"

        for field in attrs.fields(StageSpec):
            if field.name != "stage":
                stage_info += f"      {field.name}: {getattr(stage_spec, field.name)}\n"

        return stage_info

    def __str__(self) -> str:
        info = "PipelineSpec:\n"

        for field in attrs.fields(PipelineSpec):
            if field.name not in ["input_data", "stages"]:
                info += f"  {field.name}: {getattr(self, field.name)}\n"

        for i, stage_spec in enumerate(self.stages):
            assert isinstance(stage_spec, StageSpec)
            info += f"  Stage {i}:\n"
            info += self._format_stage_spec(stage_spec)

        return info


class WrappedStage(stage.Interface):
    def __init__(self, stage: Stage):
        self._stage = stage

    def setup_on_node(self, node_info: resources.NodeInfo, worker_metadata: resources.WorkerMetadata) -> None:
        self._stage.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: resources.WorkerMetadata) -> None:
        self._stage.setup(worker_metadata)

    def process_data(self, data: Any) -> Any:
        return self._stage.process_data(data)

    def destroy(self) -> None:
        self._stage.destroy()


@attrs.define
class StageAndParams:
    stage: stage.Interface
    params: StageParams


def make_actor_pool_stage_from_stage_spec(
    pipeline_config: PipelineConfig, spec: StageSpec, stage_idx: int, cluster_resources: resources.ClusterResources
) -> StageAndParams:
    assert spec.slots_per_actor is not None
    assert spec.worker_max_lifetime_m is not None
    assert spec.worker_restart_interval_m is not None
    assert spec.num_setup_attempts_python is not None
    assert spec.num_run_attempts_python is not None
    assert spec.ignore_failures is not None
    assert spec.reset_workers_on_failure is not None

    if approx.float_gt(spec.stage.required_resources.gpus, 0.0):
        modify_cuda_visible_devices_env_var = True
    else:
        # This is a little confusing. If the stage requires no GPUs, we don't want to modify the CUDA_VISIBLE_DEVICES.
        # This means that (assuming the node has gpus) the stage will have the same CUDA_VISIBLE_DEVICES as the rest of
        # the node.
        modify_cuda_visible_devices_env_var = pipeline_config.clear_cuda_visible_devices_on_cpu_actors

    wrapped_stage: stage.Interface
    if isinstance(spec.stage, ContinuousInterface):
        wrapped_stage = ContinuousWrappedStage(spec.stage)
    else:
        wrapped_stage = WrappedStage(spec.stage)

    return StageAndParams(
        wrapped_stage,
        stage.Params(
            shape=spec.stage.required_resources.to_worker_shape(cluster_resources),
            stage_batch_size=spec.stage.stage_batch_size,
            slots_per_actor=spec.slots_per_actor,
            worker_max_lifetime_m=spec.worker_max_lifetime_m,
            worker_restart_interval_m=spec.worker_restart_interval_m,
            name=spec.name(stage_idx),
            num_node_setup_retries=1,  # TODO: Make this configurable
            num_setup_retries=spec.num_setup_attempts_python,
            num_run_retries=spec.num_run_attempts_python,
            ignore_failures=spec.ignore_failures,
            restart_workers_on_failure=spec.reset_workers_on_failure,
            runtime_env=spec.stage.env_info if spec.stage.env_info is not None else runtime_envs.RuntimeEnv(),
            logging_context=None,  # TODO: Make this configurable
            max_setup_failure_percentage=spec.max_setup_failure_percentage,
            modify_cuda_visible_devices_env_var=modify_cuda_visible_devices_env_var,
        ),
    )
