# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Data structures used to represent allocated/available resources on a cluster/node/gpu.

Many of the classes in this module are "shapes". A shape is a fully specified resource requirement for something.
Shapes are meant to specified by users on a per-stage basis.
"""

from __future__ import annotations

import abc
import copy
import enum
import os
import pprint
from typing import Optional, Union

import attrs
import ray
from loguru import logger

from cosmos_xenna.utils import approx

try:
    import pynvml

    HAS_NVML = True
except ImportError:
    pynvml = None
    HAS_NVML = False


CICD_ENV_VAR = "IS_RUNNING_IN_CICD"


class AllocationError(Exception):
    pass


@attrs.define
class Resources:
    """A user friendly way to specify the resources required for something.

    This class provides an intuitive interface for specifying resource requirements
    that get translated into more detailed internal worker shapes.

    See `yotta.ray_utils._specs.Stage.required_resources` for much more info.
    """

    cpus: float = 0.0
    gpus: Union[float, int] = 0
    nvdecs: int = 0
    nvencs: int = 0
    entire_gpu: bool = False

    def to_dict(self) -> dict[str, float]:
        return {"cpu": self.cpus, "gpu": self.gpus, "nvdecs": self.nvdecs, "nvencs": self.nvencs}

    # TODO: test
    def to_shape(self) -> WorkerShape:
        # Validation
        if self.cpus < 0.0 or self.gpus < 0.0 or self.nvdecs < 0 or self.nvencs < 0:
            raise ValueError(f"Invalid shape = {self}. Some values were negative.")
        if self.cpus == 0.0 and self.gpus == 0.0 and self.nvdecs == 0 and self.nvencs == 0:
            raise ValueError(
                f"Invalid shape = {self}. Expected at least one value to be nonzero, but all values were zero."
            )

        # Entire GPU
        if self.entire_gpu:
            if not (self.gpus > 0.0 or isinstance(self.gpus, int) or approx.is_almost_whole(self.gpus)):
                raise ValueError(
                    f"Invalid shape = {self}. If entire_gpu is set to True, "
                    f"self.gpus needs to be an integer (e.g. 1, 2, 3, 3.0)."
                )
            if self.nvdecs > 0 or self.nvencs > 0:
                raise ValueError(
                    f"Invalid shape = {self}. If self.entire_gpu is True, nvdecs and nvencs can not be explictly "
                    "asked for."
                )
            return WorkerShape.make(EntireGpu(int(self.gpus), self.cpus))

        # CPU stage
        if self.cpus != 0.0 and self.gpus == 0.0 and self.nvdecs == 0 and self.nvencs == 0:
            return WorkerShape.make(CpuOnly(self.cpus))

        # Whole numbered GPU
        if approx.float_gte(self.gpus, 1.0):
            if not (isinstance(self.gpus, int) or approx.is_almost_whole(self.gpus)):
                raise ValueError(
                    f"Invalid shape = {self}. If self.gpus is greater than 1, "
                    f"self.gpus needs to be an integer (e.g. 1, 2, 3, 3.0)."
                )
            return WorkerShape.make(WholeNumberedGpu(int(self.gpus), self.cpus, self.nvdecs, self.nvencs))

        # Codec
        if (self.nvdecs > 0 or self.nvencs > 0) and self.gpus == 0.0:
            return WorkerShape.make(Codec(self.cpus, self.nvdecs, self.nvencs))

        # Fractional GPU
        if self.gpus < 1.0:
            return WorkerShape.make(FractionalGpu(self.gpus, self.cpus, self.nvdecs, self.nvencs))

        raise ValueError(f"Unexpected value: {self}.")

    def to_pool_of_resources(self, num_nvdecs_per_gpu: int, num_nvencs_per_gpu: int) -> PoolOfResources:
        return self.to_shape().to_pool_of_resources(num_nvdecs_per_gpu, num_nvencs_per_gpu)


class ShapeInterface(abc.ABC):
    """Interface for a shape.

    This is useful when an algorithm doesn't care about the shape specifics, but only the number of resources
    required by the shape.
    """

    @abc.abstractmethod
    def validate(self) -> None:
        pass

    @abc.abstractmethod
    def get_num_cpus(self) -> float:
        pass

    @abc.abstractmethod
    def get_num_gpus(self) -> Union[int, float]:
        pass

    @abc.abstractmethod
    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        pass

    @abc.abstractmethod
    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        pass


@attrs.define
class CpuOnly(ShapeInterface):
    """A shape which only requires a certain number of CPUs.

    `num_cpus` can be a fraction. In means multiple workers can be allocated to the same cpu.
    """

    num_cpus: float

    def validate(self) -> None:
        assert 0.0 < self.num_cpus

    def get_num_cpus(self) -> float:
        return self.num_cpus

    def get_num_gpus(self) -> Union[int, float]:
        return 0

    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        return 0

    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        return 0


@attrs.define
class Codec:
    """A shape which only requires CPUs and codec hardware (nvdec/nvenc).

    All of the nvdecs/nvencs will come come from the same gpu.
    """

    num_cpus: float = 1.0
    num_nvdecs: int = 0
    num_nvencs: int = 0

    def validate(self) -> None:
        assert 0.0 <= self.num_cpus
        assert 0 <= self.num_nvdecs
        assert 0 <= self.num_nvencs
        assert 0 < self.num_nvdecs or 0 < self.num_nvencs

    def get_num_cpus(self) -> float:
        return self.num_cpus

    def get_num_gpus(self) -> Union[int, float]:
        return 0

    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        return self.num_nvdecs

    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        return self.num_nvencs


@attrs.define
class FractionalGpu:
    """A shape which requires a fraction of a GPU.

    Can also require cpus, nvdecs and nvencs.

    `num_gpus` must be 0.0 < x < 1.0.

    This enables multiple workers to be allocated on a single gpu.
    """

    num_gpus: float
    num_cpus: float = 1.0
    num_nvdecs: int = 0
    num_nvencs: int = 0

    def validate(self) -> None:
        assert 0.0 <= self.num_cpus
        assert 0.0 < self.num_gpus < 1.0
        assert 0 <= self.num_nvdecs
        assert 0 <= self.num_nvencs

    def get_num_cpus(self) -> float:
        return self.num_cpus

    def get_num_gpus(self) -> Union[int, float]:
        return self.num_gpus

    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        return self.num_nvdecs

    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        return self.num_nvencs


@attrs.define
class WholeNumberedGpu:
    """A shape which requires a whole number GPU(s).

    Can also require cpus, nvdecs and nvencs
    """

    num_gpus: int
    num_cpus: float = 1.0
    num_nvdecs_per_gpu: int = 0
    num_nvencs_per_gpu: int = 0

    def validate(self) -> None:
        assert 0.0 <= self.num_cpus
        assert 0 < self.num_gpus
        assert 0 <= self.num_nvdecs_per_gpu
        assert 0 <= self.num_nvencs_per_gpu

    def get_num_cpus(self) -> float:
        return self.num_cpus

    def get_num_gpus(self) -> Union[int, float]:
        return self.num_gpus

    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        return self.num_nvdecs_per_gpu * self.num_gpus

    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        return self.num_nvencs_per_gpu * self.num_gpus


@attrs.define
class EntireGpu:
    """A shape which requires an entire GPU(s), including all of the nvdecs and nvencs."""

    num_gpus: int
    num_cpus: float = 1.0

    def validate(self) -> None:
        assert 0.0 <= self.num_cpus
        assert 0 < self.num_gpus

    def get_num_cpus(self) -> float:
        return self.num_cpus

    def get_num_gpus(self) -> Union[int, float]:
        return self.num_gpus

    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        return self.num_gpus * num_nvdecs_per_gpu

    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        return self.num_gpus * num_nvencs_per_gpu


class WorkerShapeType(enum.Enum):
    """An enum which corresponds to all of the shape types."""

    CPU_ONLY = 0
    CODEC = 1
    FRACTIONAL_GPU = 2
    WHOLE_NUMBERED_GPU = 3
    ENTIRE_GPU = 4


WorkerShapeTypes = Union[CpuOnly, Codec, FractionalGpu, WholeNumberedGpu, EntireGpu]


@attrs.define
class WorkerShape(ShapeInterface):
    """A class representing the shape of compute resources for a worker.

    This class encapsulates different types of compute resource configurations and
    provides methods to query and manipulate these configurations. It supports
    various resource types including CPU-only, codec, and different GPU
    configurations.

    Attributes:
        type: The type of worker shape configuration (WorkerShapeType).
        data: The specific configuration data for the worker shape.

    Example:
        ```python
        cpu_config = CpuOnly(num_cpus=4)
        worker = WorkerShape.make(cpu_config)
        num_cpus = worker.get_num_cpus()
        ```
    """

    type: WorkerShapeType
    data: WorkerShapeTypes

    def num_cpus(self) -> Union[int, float]:
        """Returns the number of CPUs in the configuration.

        Returns:
            The number of CPUs as either an integer or float.
        """
        return self.data.num_cpus

    @classmethod
    def make(cls, data: Union[CpuOnly, Codec, FractionalGpu, WholeNumberedGpu, EntireGpu]) -> WorkerShape:
        """Creates a new WorkerShape instance from the provided configuration data.

        Args:
            data: The configuration data for the worker shape. Can be one of:
                CpuOnly, Codec, FractionalGpu, WholeNumberedGpu, or EntireGpu.

        Returns:
            A new WorkerShape instance.

        Raises:
            ValueError: If the provided data type is not supported.
        """
        if isinstance(data, CpuOnly):
            shape_type = WorkerShapeType.CPU_ONLY
        elif isinstance(data, Codec):
            shape_type = WorkerShapeType.CODEC
        elif isinstance(data, FractionalGpu):
            shape_type = WorkerShapeType.FRACTIONAL_GPU
        elif isinstance(data, WholeNumberedGpu):
            shape_type = WorkerShapeType.WHOLE_NUMBERED_GPU
        elif isinstance(data, EntireGpu):
            shape_type = WorkerShapeType.ENTIRE_GPU
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return cls(type=shape_type, data=data)

    def cpu_only(self) -> CpuOnly:
        """Returns the CPU-only configuration if this is a CPU-only worker.

        Returns:
            The CpuOnly configuration data.

        Raises:
            AssertionError: If this is not a CPU-only worker.
        """
        assert self.type == WorkerShapeType.CPU_ONLY
        return self.data  # type: ignore

    def codec(self) -> Codec:
        """Returns the codec configuration if this is a codec worker.

        Returns:
            The Codec configuration data.

        Raises:
            AssertionError: If this is not a codec worker.
        """
        assert self.type == WorkerShapeType.CODEC
        return self.data  # type: ignore

    def fractional_gpu(self) -> FractionalGpu:
        """Returns the fractional GPU configuration if this uses fractional GPUs.

        Returns:
            The FractionalGpu configuration data.

        Raises:
            AssertionError: If this is not a fractional GPU worker.
        """
        assert self.type == WorkerShapeType.FRACTIONAL_GPU
        return self.data  # type: ignore

    def whole_numbered_gpu(self) -> WholeNumberedGpu:
        """Returns the whole numbered GPU configuration if this uses whole GPUs.

        Returns:
            The WholeNumberedGpu configuration data.

        Raises:
            AssertionError: If this is not a whole numbered GPU worker.
        """
        assert self.type == WorkerShapeType.WHOLE_NUMBERED_GPU
        return self.data  # type: ignore

    def entire_gpu(self) -> EntireGpu:
        """Returns the entire GPU configuration if this uses entire GPUs.

        Returns:
            The EntireGpu configuration data.

        Raises:
            AssertionError: If this is not an entire GPU worker.
        """
        assert self.type == WorkerShapeType.ENTIRE_GPU
        return self.data  # type: ignore

    def validate(self) -> None:
        """Validates the worker shape configuration.

        Ensures the shape type is valid and the underlying data is valid.

        Raises:
            AssertionError: If the configuration is invalid.
        """
        assert self.type in {e for e in WorkerShapeType}
        self.data.validate()

    def get_num_cpus(self) -> float:
        """Gets the number of CPUs in the configuration.

        Returns:
            The number of CPUs as a float.
        """
        return self.data.get_num_cpus()

    def get_num_gpus(self) -> Union[int, float]:
        """Gets the number of GPUs in the configuration.

        Returns:
            The number of GPUs as either an integer or float.
        """
        return self.data.get_num_gpus()

    def get_num_nvdecs(self, num_nvdecs_per_gpu: int) -> Union[int, float]:
        """Gets the number of NVIDIA decoders in the configuration.

        Args:
            num_nvdecs_per_gpu: The number of NVIDIA decoders per GPU.

        Returns:
            The total number of NVIDIA decoders as either an integer or float.
        """
        return self.data.get_num_nvdecs(num_nvdecs_per_gpu)

    def get_num_nvencs(self, num_nvencs_per_gpu: int) -> Union[int, float]:
        """Gets the number of NVIDIA encoders in the configuration.

        Args:
            num_nvencs_per_gpu: The number of NVIDIA encoders per GPU.

        Returns:
            The total number of NVIDIA encoders as either an integer or float.
        """
        return self.data.get_num_nvencs(num_nvencs_per_gpu)

    def to_pool_of_resources(self, num_nvdecs_per_gpu: int, num_nvencs_per_gpu: int) -> PoolOfResources:
        """Converts the worker shape to a pool of resources.

        Args:
            num_nvdecs_per_gpu: The number of NVIDIA decoders per GPU.
            num_nvencs_per_gpu: The number of NVIDIA encoders per GPU.

        Returns:
            A PoolOfResources instance representing the total available resources.
        """
        return PoolOfResources(
            self.get_num_cpus(),
            self.get_num_gpus(),
            self.get_num_nvdecs(num_nvdecs_per_gpu),
            self.get_num_nvencs(num_nvencs_per_gpu),
        )


@attrs.define
class PoolOfResources:
    """
    Represents the resources required by a worker or available on a node.

    This is a way of reporting resources which doesn't keep track of the nuances around node/gpu boundaries. It can
    be useful for user facing reporting and some simple allocation algorithms.
    """

    cpus: float = 0.0  # Number of CPUs (can be fractional)
    gpus: float = 0.0  # Number of GPUs (can be fractional)
    nvdecs: float = 0.0  # Number of NVIDIA decoders
    nvencs: float = 0.0  # Number of NVIDIA encoders

    def total_num(self) -> float:
        return self.cpus + self.gpus + self.nvdecs + self.nvencs

    def mutiply_by(self, factor: float) -> PoolOfResources:
        return PoolOfResources(
            cpus=self.cpus * factor, gpus=self.gpus * factor, nvdecs=self.nvdecs * factor, nvencs=self.nvencs * factor
        )

    def __add__(self, other: PoolOfResources) -> PoolOfResources:
        return PoolOfResources(
            cpus=self.cpus + other.cpus,
            gpus=self.gpus + other.gpus,
            nvdecs=self.nvdecs + other.nvdecs,
            nvencs=self.nvencs + other.nvencs,
        )

    def __sub__(self, other: PoolOfResources) -> PoolOfResources:
        return PoolOfResources(
            cpus=self.cpus - other.cpus,
            gpus=self.gpus - other.gpus,
            nvdecs=self.nvdecs - other.nvdecs,
            nvencs=self.nvencs - other.nvencs,
        )

    def __truediv__(self, other: PoolOfResources) -> PoolOfResources:
        return PoolOfResources(
            cpus=self.cpus / other.cpus if other.cpus else 0.0,
            gpus=self.gpus / other.gpus if other.gpus else 0.0,
            nvdecs=self.nvdecs / other.nvdecs if other.nvdecs else 0.0,
            nvencs=self.nvencs / other.nvencs if other.nvencs else 0.0,
        )

    def contains(self, other: PoolOfResources) -> bool:
        return (
            self.cpus >= other.cpus
            and self.gpus >= other.gpus
            and self.nvdecs >= other.nvdecs
            and self.nvencs >= other.nvencs
        )

    def to_dict(self) -> dict[str, Union[float, int]]:
        """
        Convert the Resources object to a dictionary.
        Useful for serialization or easy access to resource values.
        """
        return {"cpu": self.cpus, "gpu": self.gpus, "nvdecs": self.nvdecs, "nvencs": self.nvencs}


@attrs.define
class GpuResources:
    """Represents the state of allocation for a single GPU."""

    gpu_fraction: Union[float, int] = 1
    nvdecs: set[int] = attrs.field(factory=set)
    nvencs: set[int] = attrs.field(factory=set)

    @classmethod
    def make_from_num_codecs(
        cls, gpu_fraction_available: Union[float, int] = 1, num_nvdecs: int = 0, num_nvencs: int = 0
    ) -> GpuResources:
        return GpuResources(gpu_fraction_available, set(range(num_nvdecs)), set(range(num_nvencs)))

    @property
    def num_nvdecs(self) -> int:
        return len(self.nvdecs)

    @property
    def num_nvencs(self) -> int:
        return len(self.nvencs)

    def totals(self) -> PoolOfResources:
        return PoolOfResources(0.0, self.gpu_fraction, self.num_nvdecs, self.num_nvencs)


@attrs.define
class GPUAllocation:
    """Represents the allocation a worker is taking up for a given GPU."""

    gpu_index: int
    fraction: float


@attrs.define
class CodecAllocation:
    """Represents the allocation a worker is taking up for a single hardware accelerated codec (NVDEC/NVENC)."""

    gpu_index: int
    codec_index: int


@attrs.define
class WorkerResources:
    """Represents all the resources allocated to a single worker."""

    node: str
    cpus: float
    gpus: list[GPUAllocation] = attrs.field(factory=list)
    nvdecs: list[CodecAllocation] = attrs.field(factory=list)
    nvencs: list[CodecAllocation] = attrs.field(factory=list)

    def validate(self) -> None:
        assert self.cpus >= 0.0
        for gpu in self.gpus:
            assert gpu.fraction >= 0.0

    def to_pool(self) -> PoolOfResources:
        return PoolOfResources(
            self.cpus,
            sum([x.fraction for x in self.gpus], 0.0),
            sum([1 for _ in self.nvdecs], 0.0),
            sum([1 for _ in self.nvencs], 0.0),
        )


@attrs.define
class NodeResources:
    """
    Represents all the resources available on a single node in a cluster.
    """

    cpus: Union[float, int]
    gpus: list[GpuResources] = attrs.field(factory=list)
    name: Optional[str] = None

    @classmethod
    def make_uniform(
        cls, num_cpus: int = 0, num_gpus: int = 0, num_nvdecs_per_gpu: int = 0, num_nvencs_per_gpu: int = 0
    ) -> NodeResources:
        """Make a "uniform" node. I.e. all the nodes have the same number of nvdecs and nvencs."""
        out = NodeResources(num_cpus, gpus=[])
        for _ in range(num_gpus):
            out.gpus.append(GpuResources(1, set(range(num_nvdecs_per_gpu)), set(range(num_nvencs_per_gpu))))
        return out

    def totals(self) -> PoolOfResources:
        out = PoolOfResources(cpus=self.cpus)
        for gpu in self.gpus:
            out += gpu.totals()
        return out

    def copy_and_allocate(self, resources: WorkerResources) -> NodeResources:
        c = copy.deepcopy(self)
        c.allocate(resources)
        return c

    def allocate(self, resources: WorkerResources) -> None:
        self.cpus -= resources.cpus
        for gpu in resources.gpus:
            self.gpus[gpu.gpu_index].gpu_fraction -= gpu.fraction

        for x in resources.nvdecs:
            if x.codec_index not in self.gpus[x.gpu_index].nvdecs:
                raise AllocationError(
                    f"Asked to allocatocate {resources}, but GPU {x.gpu_index} only has the following "
                    f"nvdecs available: {list(self.gpus[x.gpu_index].nvdecs)}"
                )
            self.gpus[x.gpu_index].nvdecs.remove(x.codec_index)

        for x in resources.nvencs:
            if x.codec_index not in self.gpus[x.gpu_index].nvencs:
                raise AllocationError(
                    f"Asked to allocatocate {resources}, but GPU {x.gpu_index} only has the following "
                    f"nvencs available: {list(self.gpus[x.gpu_index].nvencs)}"
                )
            self.gpus[x.gpu_index].nvencs.remove(x.codec_index)

    def release_allocation(self, resources: WorkerResources) -> None:
        self.cpus += resources.cpus
        for gpu in resources.gpus:
            self.gpus[gpu.gpu_index].gpu_fraction += gpu.fraction

        for x in resources.nvdecs:
            assert x.codec_index not in self.gpus[x.gpu_index].nvdecs
            self.gpus[x.gpu_index].nvdecs.add(x.codec_index)

        for x in resources.nvencs:
            assert x.codec_index not in self.gpus[x.gpu_index].nvencs
            self.gpus[x.gpu_index].nvencs.add(x.codec_index)

    def __str__(self) -> str:
        return pprint.pformat(attrs.asdict(self))


def _make_gpu_resources_from_gpu_name(gpu_name: str) -> GpuResources:
    """This is a hack which determines the number of nvdec/nvencs per gpu based on the GPU name.

    Ideally, we'd have a better source for this data, but we couldn't find a good one.
    """
    if "H100" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=7)
    elif "A100" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=7)
    elif "L40" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=3, num_nvencs=3)
    elif "L4" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=4, num_nvencs=2)
    elif "RTX 6000" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=3, num_nvencs=3)
    elif "RTX A6000" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=3, num_nvencs=3)
    elif "NVIDIA" in gpu_name:
        return GpuResources.make_from_num_codecs(num_nvdecs=0, num_nvencs=0)
    else:
        raise ValueError(
            f"Unknown gpu type: {gpu_name}. Likely it needs to be added to "
            "cosmos_xenna.ray_utils.cluster._make_gpu_resources_from_gpu_name"
        )


@attrs.define
class GpuInfo:
    index: int
    name: str
    uuid: str


def _get_local_gpu_info() -> list[GpuInfo]:
    """Uses pynvml to get information about GPUs on the local node."""
    gpus = []
    if not HAS_NVML:
        logger.warning("pynvml is not installed. Assuming no GPUs.")
        return []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            # pynvml returns bytes, decode to string
            gpus.append(GpuInfo(index=i, name=str(name), uuid=str(uuid)))
    except pynvml.NVMLError as e:
        logger.warning(f"Could not initialize NVML or get GPU info: {e}. Assuming no GPUs.")
        # Return empty list if NVML fails (e.g., no NVIDIA driver)
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            # Ignore shutdown errors if initialization failed
            pass

    return gpus


def _make_gpu_resources_from_current_node() -> Optional[list[GpuResources]]:
    """Look at the current node and determine the resources available per gpu.

    There is no good way to find number of nvdecs/nvdecs available. So we do this hack where we look at
    """
    logger.info("Determining number of nvdecs/nvencs per gpu in this cluster.")
    gpus = _get_local_gpu_info()

    if not gpus:
        logger.info("No gpus found. Returning None.")
        return None
    
    # HACK: when running in CI/CD, we ignore 'NVIDIA DGX Display' gpus
    # This hack is incomplete. We also need to make sure the cuda env vars are set correctly.
    if CICD_ENV_VAR in os.environ:
        logger.info("Running in CI/CD. Ignoring 'NVIDIA DGX Display' gpus")
        gpus = [x for x in gpus if "NVIDIA DGX Display" not in x.name]

    # Respect CUDA_VISIBLE_DEVICES
    visible_device = []
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_device = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        visible_device = [x.index for x in gpus]

    unique_names = set([str(x.name) for x in gpus])
    if len(unique_names) != 1:
        raise ValueError(f"Running on a node with multiple gpu types: {unique_names}. This is not supported as of now.")

    outs = []
    for gpu in gpus:
        logger.info(f"Gpu with name {gpu.name} found. Looking up nvdecs and nvencs...")
        out = _make_gpu_resources_from_gpu_name(gpu.name)
        logger.info(f"Found the following gpu resources: {out}")

        if not (gpu.index in visible_device or gpu.uuid in visible_device):
            out.gpu_fraction = 0
            logger.warning(f"Gpu with index {gpu.index} and uuid {gpu.uuid} not in CUDA_VISIBLE_DEVICES. 
                        Not allocating any resources for this gpu. But Xenna will still use the nvdecs/nvencs for this gpu.")
        outs.append(out)
    return outs


@attrs.define
class ClusterResources:
    """
    Represents the total resources available in the entire cluster.
    """

    nodes: dict[str, NodeResources]  # dict of all nodes in the cluster

    @classmethod
    def make_uniform(cls, node_resources: NodeResources, node_ids: set[str]) -> ClusterResources:
        node_dict = {}
        for node_id in node_ids:
            node_dict[node_id] = copy.deepcopy(node_resources)
        return ClusterResources(node_dict)

    @classmethod
    def make_for_ray_cluster(
        cls,
        cpu_allocation_percentage: float = 1.0,
        nodes: Optional[list] = None,
    ) -> ClusterResources:
        """
        Make a ClusterResources object for a ray cluster.

        If nodes is None, calls ray.nodes() to get a list of connected nodes.

        ray.nodes() returns something which looks like this:
        [
            {
                "NodeID": "xx",
                "Alive": true,
                "NodeManagerAddress": "xx",
                "NodeManagerHostname": "xx",
                "NodeManagerPort": 11,
                "ObjectManagerPort": 11,
                "ObjectStoreSocketName": "/tmp/ray/session_2024-08-23_09-07-26_009842_799459/sockets/plasma_store",
                "RayletSocketName": "/tmp/ray/session_2024-08-23_09-07-26_009842_799459/sockets/raylet",
                "MetricsExportPort": 11,
                "NodeName": "xx",
                "RuntimeEnvAgentPort": 11,
                "alive": true,
                "Resources": {
                    "GPU": 1.0,
                    "accelerator_type:RTX": 1.0,
                    "memory": 11,
                    "node:__internal_head__": 1.0,
                    "object_store_memory": 11,
                    "node:xx": 1.0,
                    "CPU":11
                },
                "Labels": {
                    "ray.io/node_id": "xx"
                }
            },
            ...
        ]

        We will use this node info to collect the number of CPUS and GPUs for each node. We also rely on a
        user-provided "resources_per_gpu" parameter. This parameter tells use how many NVDECs/NVENCs are on each
        GPU. Ideally, which is something Ray does not give us.
        """
        if nodes is None:
            nodes = ray.nodes()

        out = ClusterResources({})
        for node in nodes:
            node_id = node["NodeID"]
            reported_resources = node["Resources"]
            if "GPU" not in reported_resources:
                gpus = []
            else:
                gpus = _make_gpu_resources_from_current_node()
                if gpus is None:
                    gpus = []
            out.nodes[str(node_id)] = NodeResources(
                cpus=int(reported_resources["CPU"] * cpu_allocation_percentage),
                gpus=gpus,  # type: ignore
                name=str(node_id),
            )
        return out

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_gpus(self) -> int:
        out = 0
        for node in self.nodes.values():
            for _ in node.gpus:
                out += 1
        return out

    def calc_num_nvdecs_per_gpu(self) -> int:
        per_gpu: set[int] = set()
        for node in self.nodes.values():
            for gpu in node.gpus:
                per_gpu.add(gpu.num_nvdecs)

        if not per_gpu:
            return 0
        else:
            assert len(per_gpu) == 1
            return next(iter(per_gpu))

    def calc_num_nvencs_per_gpu(self) -> int:
        per_gpu: set[int] = set()
        for node in self.nodes.values():
            for gpu in node.gpus:
                per_gpu.add(gpu.num_nvencs)

        if not per_gpu:
            return 0
        else:
            assert len(per_gpu) == 1
            return next(iter(per_gpu))

    def totals(self) -> PoolOfResources:
        out = PoolOfResources()
        for node in self.nodes.values():
            out += node.totals()
        return out

    def is_overallocated(self) -> bool:
        over_allocated = False
        for node in self.nodes.values():
            if over_allocated:
                break
            if node.cpus < 0.0:
                over_allocated = True
                break
            for gpu in node.gpus:
                if gpu.gpu_fraction < 0.0 or gpu.num_nvdecs < 0.0 or gpu.num_nvencs < 0.0:
                    over_allocated = True
                    break
        return over_allocated

    def copy_and_clear_allocation(self, resources: WorkerResources) -> ClusterResources:
        c = copy.deepcopy(self)
        c.clear_allocation(resources)
        return c

    # TODO: Maybe need to add asserts?
    def clear_allocation(self, resources: WorkerResources) -> None:
        node = self.nodes[resources.node]
        node.cpus += resources.cpus
        for gpu in resources.gpus:
            node.gpus[gpu.gpu_index].gpu_fraction += gpu.fraction

        for x in resources.nvdecs:
            node.gpus[x.gpu_index].nvdecs.add(x.codec_index)

        for x in resources.nvencs:
            node.gpus[x.gpu_index].nvencs.add(x.codec_index)

    def __str__(self) -> str:
        return pprint.pformat(attrs.asdict(self))


@attrs.define
class Worker:
    """An allocated worker"""

    id: str
    stage_name: str
    allocation: WorkerResources

    def __hash__(self) -> int:
        return hash(self.id)


@attrs.define
class WorkerMetadata:
    worker_id: str
    allocation: WorkerResources


@attrs.define
class NodeInfo:
    node_id: str
