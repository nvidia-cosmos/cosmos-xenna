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
Resource Monitoring Utilities

This module provides utilities to monitor and log system resources, including
memory, network IO, CPU utilization, and disk usage. This can be helpful when running
jobs on NGC. By default, run_envelope kicks off a resource monitor per node.
"""

from __future__ import annotations

import tempfile

try:
    import gpustat

    HAS_PYNVML = True
except Exception:  # noqa: BLE001
    HAS_PYNVML = False

import datetime
import os
import pathlib
import resource
import threading
import time
from typing import Any, Optional

import attrs
import jinja2
import psutil

from cosmos_xenna.utils import python_log as logger
from cosmos_xenna.utils import timing

# Template string for system monitoring display.
_MONITOR_TEMPLATE_STR = """
- System Monitor [{{time_string}}]
# Mem: {{d.memory_used_percentage | round(1)}}% ({{d.memory_used_gbyte | round(1)}}/{{d.memory_total_gbyte | round(1)}} GB)
# Disk: {{d.tmp_disk_used_percentage | round(1)}}% ({{d.tmp_disk_used_gbyte | round(1)}}/{{d.tmp_disk_total_gbyte | round(1)}} GB)
# Net: Up {{d.network_up_mibps | round(3)}} MiB/s - Down {{d.network_down_mibps | round(3)}} MiB/s
# GPUs:
{{gpu_string}}
- CPUs: {{cpu_string}}
"""  # noqa: E501
_MONITOR_TEMPLATE_TEMPLATE = jinja2.Template(_MONITOR_TEMPLATE_STR)

_GB2B = 1024 * 1024 * 1024
_MB2B = 1024 * 1024
_GB2MB = 1024


@attrs.define
class MemoryLimitInfo:
    total_gb: float
    cgroup_gb: Optional[float] = None
    rss_limit_gb: Optional[float] = None


def get_memory_limit_info() -> MemoryLimitInfo:
    """Get the memory limit (in bytes) for this system.

    Takes the minimum value from the following locations:

    - Total system host memory
    - Cgroups limit (if set)
    - RSS rlimit (if set)
    """

    out = MemoryLimitInfo(psutil.virtual_memory().total / float(_GB2B))

    # Check cgroups if available
    cgroup_limits = []
    for path in [
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroups v1 hard limit
        "/sys/fs/cgroup/memory/memory.soft_limit_in_bytes",  # cgroups v1 soft limit
        "/sys/fs/cgroup/memory.max",  # cgroups v2 hard limit
        "/sys/fs/cgroup/memory.high",  # cgroups v2 soft limit
        "/sys/fs/cgroup/memory.low",  # cgroups v2 softest limit
    ]:
        try:
            with open(path, encoding="utf-8") as f:
                cgroups_limit = int(f.read())
                cgroup_limits.append(cgroups_limit)
        except Exception:  # noqa: BLE001
            pass
    if cgroup_limits:
        out.cgroup_gb = min(cgroup_limits) / float(_GB2B)

    try:
        hard_limit = resource.getrlimit(resource.RLIMIT_RSS)[1]
        if hard_limit != 0:
            out.rss_limit_gb = hard_limit  # / float(_GB2B)
    except OSError:
        pass

    return out


@attrs.define
class GPUData:
    """Dataclass representing individual GPU data."""

    utilization: float
    memory_used: float
    memory_total: float
    power_usage_percentage: int | None
    power_limit_w: int | None


@attrs.define
class SystemData:
    """Dataclass representing the system's current resource state."""

    # Metrics related to various system components.
    time: float
    memory_used_bytes: int
    memory_total_bytes: int
    network_down_bytes_per_second: float
    network_up_bytes_per_second: float
    cpu_utilization_percentages: list[float]
    tmp_disk_used_bytes: int
    tmp_disk_total_bytes: int

    # GPU related metrics
    gpus: Optional[list[GPUData]] = None

    # Computed properties to provide metrics in user-friendly units.
    @property
    def memory_used_gbyte(self) -> float:
        return float(self.memory_used_bytes) / _GB2B

    @property
    def memory_total_gbyte(self) -> float:
        return float(self.memory_total_bytes) / _GB2B

    @property
    def memory_used_percentage(self) -> float:
        if self.memory_total_bytes == 0:
            return 0.0
        return 100.0 * self.memory_used_bytes / self.memory_total_bytes

    @property
    def network_down_gbps(self) -> float:
        return 8 * float(self.network_down_bytes_per_second) / _GB2B

    @property
    def network_up_gbps(self) -> float:
        return 8 * float(self.network_up_bytes_per_second) / _GB2B

    @property
    def network_down_mibps(self) -> float:
        """Network down speed in MiB/s."""
        return float(self.network_down_bytes_per_second) / _MB2B

    @property
    def network_up_mibps(self) -> float:
        """Network up speed in MiB/s."""
        return float(self.network_up_bytes_per_second) / _MB2B

    @property
    def tmp_disk_used_gbyte(self) -> float:
        return float(self.tmp_disk_used_bytes) / _GB2B

    @property
    def tmp_disk_total_gbyte(self) -> float:
        return float(self.tmp_disk_total_bytes) / _GB2B

    @property
    def tmp_disk_used_percentage(self) -> float:
        if self.tmp_disk_total_bytes == 0:
            return 0.0
        return 100.0 * self.tmp_disk_used_bytes / self.tmp_disk_total_bytes

    @property
    def gpu_summary(self) -> str:
        """Provides a summary of GPU utilizations, memory usage, and power usage."""
        if self.gpus is None:
            return "No GPUs detected"

        gpu_summary_strings = []
        for idx, gpu in enumerate(self.gpus):
            mem_used = float(gpu.memory_used) / _GB2MB
            mem_total = float(gpu.memory_total) / _GB2MB
            power_str = (
                f"{gpu.power_usage_percentage:.2f}W power" if gpu.power_usage_percentage is not None else "N/A power"
            )
            gpu_summary_strings.append(
                f"GPU{idx}: {gpu.utilization:.2f}% util, {mem_used:.2f}/{mem_total:.2f} GB memory used, {power_str}"
            )
        return "\n".join(gpu_summary_strings)

    def _cpu_summary(self) -> str:
        """Provides a summary of CPU utilization percentages using standard Python."""
        cpus = sorted(self.cpu_utilization_percentages)
        n = len(cpus)

        if n == 0:
            return "No CPU data available"

        avg_utilization = sum(cpus) / n
        min_utilization = cpus[0]
        max_utilization = cpus[-1]

        # Calculate percentiles without numpy
        q25_index = int(0.25 * (n + 1)) - 1
        median_index = int(0.50 * (n + 1)) - 1
        q75_index = int(0.75 * (n + 1)) - 1

        # Adjust index for 0-based indexing and handle edge cases
        q25_index = max(0, min(q25_index, n - 1))
        median_index = max(0, min(median_index, n - 1))
        q75_index = max(0, min(q75_index, n - 1))

        # Simple percentile calculation (nearest rank)
        # More sophisticated interpolation could be used if needed
        q25_utilization = cpus[q25_index]
        median_utilization = cpus[median_index]  # This handles even/odd n correctly for median
        q75_utilization = cpus[q75_index]

        cores_above_5_percent = sum(1 for c in cpus if c > 5)  # Count cores with utilization > 5%

        return (
            f"avg={avg_utilization:.2f}%, min={min_utilization:.2f}%, "
            f"max={max_utilization:.2f}%, 25th={q25_utilization:.2f}%, "
            f"median={median_utilization:.2f}%, 75th={q75_utilization:.2f}%, "
            f"cores >5%={cores_above_5_percent}"
        )

    def __str__(self) -> str:
        """Provides a formatted string representation using the jinja2 template."""
        return _MONITOR_TEMPLATE_TEMPLATE.render(
            time_string=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            d=self,
            cpu_string=self._cpu_summary(),
            gpu_string=self.gpu_summary,
        )


def get_filesystem_usage(path: pathlib.Path) -> tuple[int, int]:
    """Retrieve the filesystem usage in bytes for a given path."""
    statvfs = os.statvfs(path)
    total_blocks = statvfs.f_blocks
    free_blocks = statvfs.f_bfree
    block_size = statvfs.f_frsize
    used_blocks = total_blocks - free_blocks
    used_bytes = used_blocks * block_size
    total_bytes = total_blocks * block_size
    return used_bytes, total_bytes


def _safe_gpu_attr(gpu: object, name: str, default: Any) -> Any:
    """Read a ``gpustat`` GPU attribute, returning ``default`` when unavailable.

    ``gpustat`` eagerly coerces some NVML fields with ``int(...)`` inside
    ``@property`` getters, which raises ``TypeError`` when NVML returns ``None``
    (e.g. ``memory.used`` / ``memory.total`` / ``power.limit`` on DGX Spark's
    GB10 integrated GPU, which shares unified memory with the CPU). Other
    getters return ``None`` directly. Collapse both cases to ``default`` so a
    single missing field does not break the whole metrics loop.
    """
    try:
        value = getattr(gpu, name)
    except (TypeError, KeyError, AttributeError):
        return default
    return default if value is None else value


class ResourceMonitor:
    """Class to fetch real-time metrics of system resources."""

    @staticmethod
    def _get_io_counts() -> tuple[int, int]:
        counters = psutil.net_io_counters()
        return counters.bytes_recv, counters.bytes_sent

    def __init__(self, get_gpu_stats: bool | None = None) -> None:
        if get_gpu_stats is None:
            get_gpu_stats = HAS_PYNVML
        self._old_bytes_recv, self._old_bytes_sent = ResourceMonitor._get_io_counts()
        self._old_time = time.time()
        self._tmp_path = pathlib.Path(tempfile.gettempdir())
        self._get_gpu_stats = bool(get_gpu_stats)

    def update(self) -> SystemData:
        """Update and return the current system metrics."""

        new_time = time.time()
        new_bytes_recv, new_bytes_sent = ResourceMonitor._get_io_counts()
        elapsed_time = new_time - self._old_time
        recv_bytes_per_second = float(new_bytes_recv - self._old_bytes_recv) / elapsed_time
        sent_bytes_per_second = float(new_bytes_sent - self._old_bytes_sent) / elapsed_time

        cpu_percents = sorted(psutil.cpu_percent(percpu=True), reverse=True)
        mem_data = psutil.virtual_memory()
        # Sometimes (I think during SLURM shutdown), we get errors that _tmp_path does not exist.
        # This can pollute the logs a bit, so we just return zeros if we hit that condiditon.
        if self._tmp_path.exists():
            tmp_used, tmp_total = get_filesystem_usage(self._tmp_path)
        else:
            tmp_used, tmp_total = 0, 0
        gpus = None
        if self._get_gpu_stats:
            try:
                gpu_stats = gpustat.GPUStatCollection.new_query()
            except Exception:  # noqa: BLE001
                gpus = None
            else:
                gpus = [
                    GPUData(
                        utilization=_safe_gpu_attr(gpu, "utilization", 0.0),
                        memory_used=_safe_gpu_attr(gpu, "memory_used", 0),
                        memory_total=_safe_gpu_attr(gpu, "memory_total", 0),
                        power_usage_percentage=_safe_gpu_attr(gpu, "power_draw", None),
                        power_limit_w=_safe_gpu_attr(gpu, "power_limit", None),
                    )
                    for gpu in gpu_stats
                ]

        out = SystemData(
            new_time,
            mem_data.used,
            mem_data.total,
            recv_bytes_per_second,
            sent_bytes_per_second,
            cpu_percents,
            tmp_used,
            tmp_total,
            gpus,
        )
        self._old_bytes_recv = new_bytes_recv
        self._old_bytes_sent = new_bytes_sent
        self._old_time = new_time
        return out


class ResourceMonitorThread(threading.Thread):
    """Threaded class to continuously monitor and log system resources."""

    def __init__(self, logging_rate_hz: float = 0.1, send_to_server_rate_hz: float = 0.1) -> None:
        super(ResourceMonitorThread, self).__init__()
        self._stop_event = threading.Event()
        self._logging_rate_hz = float(logging_rate_hz)
        self._send_to_server_rate_hz = float(send_to_server_rate_hz)

    def stop(self) -> None:
        self._stop_event.set()

    def __enter__(self) -> ResourceMonitorThread:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:  # noqa: ANN001
        self.stop()
        self.join()
        return False

    def run(self) -> None:
        monitor = ResourceMonitor()
        limiter = timing.RateLimiter(max(self._logging_rate_hz, self._send_to_server_rate_hz))
        logging_limited_doer = timing.RateLimitedCaller(self._logging_rate_hz)

        def log(data: SystemData) -> None:
            logger.info(data)

        while not self._stop_event.is_set():
            limiter.sleep()
            data = monitor.update()
            logging_limited_doer.maybe_do(log, data)


@attrs.define
class ProcessInfo:
    """Information about a linux process."""

    pid: int
    name: str
    memory_usage_rss: int
    memory_usage_shared: int
    cpu_utilization: float  # percentage
    # TODO: Not implemented yet. This is a bit tricky to calculate.
    gpu_utilization: Optional[float] = None  # percentage, if applicable
    children: list[ProcessInfo] = attrs.field(factory=list)

    def __str__(self, indent: str = "") -> str:
        gpu_str = f", GPU: {self.gpu_utilization:5.2f}%" if self.gpu_utilization is not None else ""
        result = (
            f"{indent}{self.name} (PID: {self.pid})\n"
            f"{indent}  CPU: {self.cpu_utilization:5.2f}%, "
            f"Mem: {self.memory_usage_rss / (1024 * 1024):6.2f} MB{gpu_str}\n"
        )
        for child in self.children:
            result += child.__str__(indent + "  ")
        return result

    def ray_memory_utilization(self) -> float:
        """Calculating ray memory usage is a bit tricky because of how it uses shared memory.

        See this page:
        https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/debug-memory.html

        Specifically, this passage:
        Calculate per process memory usage by RSS - SHR because SHR is for Ray object store as explained above. The
        total memory usage is typically SHR (object store memory usage, 30% of memory) +
        sum(RSS - SHR from each ray proc) + sum(RSS - SHR from system components. e.g., raylet, GCS. Usually small).
        """
        return (self.memory_usage_rss - self.memory_usage_shared) + sum(
            child.total_memory_usage() for child in self.children
        )

    def total_cpu_utilization(self) -> float:
        """Calculate total CPU utilization including all child processes."""
        return self.cpu_utilization + sum(child.total_cpu_utilization() for child in self.children)

    def total_memory_usage(self) -> int:
        """Calculate total memory usage including all child processes."""
        return self.memory_usage_rss + sum(child.total_memory_usage() for child in self.children)


@attrs.define
class ProcessTree:
    """A tree of process information.

    The root node is a dummy node. It actually doesn't represent any processes. It's just a node from which real
    processes are branched.
    """

    root: ProcessInfo

    def total_cpu_util(self) -> float:
        def sum_cpu_util(node: ProcessInfo) -> float:
            return node.cpu_utilization + sum(sum_cpu_util(child) for child in node.children)

        return sum_cpu_util(self.root)

    def __str__(self) -> str:
        return f"Process Tree (Total CPU: {self.total_cpu_util():6.2f}%)\n{'=' * 40}\n{self.root.__str__()}"

    @classmethod
    def make(cls) -> ProcessTree:
        # NOTE: For some reason, pytype thinks "proc" does not have an "info" field, so we need to add a bunch of
        # ignores.
        # Fetch all process information in a single call
        processes = list(psutil.process_iter(["pid", "name", "memory_info", "cpu_percent", "ppid"]))

        # Create a mapping of pid to ProcessInfo
        process_dict: dict[int, ProcessInfo] = {}
        pid_to_ppid: dict[int, int] = {}
        skipped_invalid_pid = 0

        for proc in processes:
            try:
                info = proc.info  # type: ignore[attr-defined]
                pid = info.get("pid") if isinstance(info, dict) else None
                if not isinstance(pid, int):
                    skipped_invalid_pid += 1
                    continue
                memory_info = info.get("memory_info") if isinstance(info, dict) else None
                ppid = info.get("ppid") if isinstance(info, dict) else 0

                pid_to_ppid[pid] = ppid if isinstance(ppid, int) else 0
                process_dict[pid] = ProcessInfo(
                    pid=pid,
                    name=str(info.get("name", "")) if isinstance(info, dict) else "",
                    memory_usage_rss=getattr(memory_info, "rss", 0),
                    memory_usage_shared=getattr(memory_info, "shared", 0),
                    cpu_utilization=float(info.get("cpu_percent", 0.0)) if isinstance(info, dict) else 0.0,
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError):
                continue

        if skipped_invalid_pid > 0:
            logger.warning(
                "ProcessTree.make: skipped "
                f"{skipped_invalid_pid} process entry(ies) with missing/invalid pid out of {len(processes)}"
            )

        # Build the process tree
        root = ProcessInfo(pid=0, name="ROOT", memory_usage_rss=0, memory_usage_shared=0, cpu_utilization=0)
        for pid, proc_info in process_dict.items():
            ppid = pid_to_ppid.get(pid, 0)
            if ppid == 0 or ppid == 1 or ppid not in process_dict:
                root.children.append(proc_info)
            else:
                process_dict[ppid].children.append(proc_info)

        # TODO: Add GPU utilization if available
        return ProcessTree(root=root)
