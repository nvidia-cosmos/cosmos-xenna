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


"""Defense-in-depth helpers for Phase C allocation failures."""

from typing import Any

from ray.util.metrics import Counter

from cosmos_xenna.utils import python_log as logger

ALLOCATION_FAILURES_METRIC = "xenna_scheduler_allocation_failures_total"

_ALLOCATION_FAILURES_COUNTER = Counter(
    ALLOCATION_FAILURES_METRIC,
    description=(
        "Per-stage Phase C allocation failures absorbed by the "
        "skip_cycle_on_allocation_error gate. Increments once per "
        "raised AllocationError (or unexpected Exception) before the "
        "scheduler skips the cycle."
    ),
    tag_keys=("stage", "pipeline"),
)


def emit_allocation_failure(
    *,
    stage_name: str,
    pipeline_name: str,
    cluster_resources: Any,
    exc: BaseException,
) -> None:
    """Log a per-GPU fragmentation snapshot at ERROR and bump the failure counter.

    Snapshot formatting is wrapped so a malformed
    ``cluster_resources`` shape cannot mask the absorbed ``exc``;
    placeholders are substituted and the formatter exception is
    appended to the ERROR record.

    Args:
        stage_name: Stage whose ``try_add_worker`` raised.
        pipeline_name: ``pipeline`` Prometheus tag value.
        cluster_resources: ``resources.ClusterResources`` or its underlying
            rust object; both expose ``nodes`` and per-node ``gpus``.
        exc: The absorbed exception.
    """
    _ALLOCATION_FAILURES_COUNTER.inc(tags={"stage": stage_name, "pipeline": pipeline_name})

    gpu_snapshot: list[dict[str, object]] | str
    cpu_snapshot: list[dict[str, object]] | str
    formatter_note = ""
    try:
        gpu_snapshot = _format_gpu_fragmentation(cluster_resources)
        cpu_snapshot = _format_cpu_fragmentation(cluster_resources)
    except Exception as format_exc:  # noqa: BLE001 -- diagnostic path must not mask the absorbed exception
        gpu_snapshot = "<unavailable: formatting error>"
        cpu_snapshot = "<unavailable: formatting error>"
        formatter_note = f". snapshot formatter raised {type(format_exc).__name__}: {format_exc!r}"

    logger.error(
        f"saturation-aware allocation failure: stage {stage_name!r} "
        f"raised {type(exc).__name__}: {exc!r}. "
        f"Per-GPU fragmentation snapshot: {gpu_snapshot}. "
        f"Per-node CPU snapshot: {cpu_snapshot}{formatter_note}"
    )


def _format_gpu_fragmentation(cluster_resources: Any) -> list[dict[str, object]]:
    """Return ``(node, gpu_index, used_fraction, free_fraction)`` for every GPU.

    Stable-ordered by ``(node_id, gpu.index)`` so two snapshots of the
    same cluster compare as identical strings. The per-GPU
    ``used_fraction`` is resolved via :func:`_gpu_used_fraction` so the
    snapshot works against both Python ``GpuResources`` and the Rust
    binding without runtime type checks leaking into the caller.
    """
    rows: list[dict[str, object]] = []
    for node_id in sorted(cluster_resources.nodes):
        node = cluster_resources.nodes[node_id]
        for gpu in sorted(node.gpus, key=lambda g: g.index):
            used = _gpu_used_fraction(gpu)
            rows.append(
                {
                    "node": node_id,
                    "gpu_index": gpu.index,
                    "used_fraction": round(used, 4),
                    "free_fraction": round(max(0.0, 1.0 - used), 4),
                }
            )
    return rows


def _gpu_used_fraction(gpu: Any) -> float:
    """Return ``used_fraction`` for both Python and Rust ``GpuResources``.

    The Python attrs ``GpuResources`` exposes ``used_fraction`` as a
    plain attribute. The Rust binding does not publish that field
    directly; ``used_pool().gpus`` is the supported accessor. This
    helper picks the right path so the snapshot works against either
    object without runtime type checks leaking into the caller.
    """
    if hasattr(gpu, "used_fraction"):
        return float(gpu.used_fraction)
    return float(gpu.used_pool().gpus)


def _node_cpu_totals(node: Any) -> tuple[float, float]:
    """Return ``(used_cpus, total_cpus)`` for both Python and Rust ``NodeResources``.

    The Python attrs ``NodeResources`` exposes ``used_cpus`` /
    ``total_cpus`` as plain attributes. The Rust binding does not
    publish those fields directly; ``used_pool()`` / ``total_pool()``
    are the supported accessors. This helper picks the right path so
    the snapshot works against either object without runtime type
    checks leaking into the caller.
    """
    if hasattr(node, "total_cpus"):
        return float(node.used_cpus), float(node.total_cpus)
    return float(node.used_pool().cpus), float(node.total_pool().cpus)


def _format_cpu_fragmentation(cluster_resources: Any) -> list[dict[str, object]]:
    """Return ``(node, cpu_used, cpu_total, cpu_free_fraction)`` for every node.

    Adds a CPU view alongside the per-GPU snapshot so CPU-only stages
    whose ``try_add_worker`` fails surface in the log with actionable
    capacity data. Stable-ordered by ``node_id`` to match the GPU
    snapshot's deterministic-string contract.
    """
    rows: list[dict[str, object]] = []
    for node_id in sorted(cluster_resources.nodes):
        node = cluster_resources.nodes[node_id]
        used, total = _node_cpu_totals(node)
        free_fraction = (total - used) / total if total > 0.0 else 0.0
        rows.append(
            {
                "node": node_id,
                "cpu_used": round(used, 4),
                "cpu_total": round(total, 4),
                "cpu_free_fraction": round(max(0.0, free_fraction), 4),
            }
        )
    return rows


__all__ = [
    "ALLOCATION_FAILURES_METRIC",
    "emit_allocation_failure",
]
