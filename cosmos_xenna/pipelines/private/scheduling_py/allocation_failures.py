# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

    Args:
        stage_name: Stage whose ``try_add_worker`` raised.
        pipeline_name: ``pipeline`` Prometheus tag value.
        cluster_resources: ``resources.ClusterResources`` or its underlying
            rust object; both expose ``nodes`` and per-node ``gpus``.
        exc: The absorbed exception.
    """
    _ALLOCATION_FAILURES_COUNTER.inc(tags={"stage": stage_name, "pipeline": pipeline_name})
    snapshot = _format_gpu_fragmentation(cluster_resources)
    logger.error(
        f"saturation-aware allocation failure: stage {stage_name!r} "
        f"raised {type(exc).__name__}: {exc!r}. "
        f"Per-GPU fragmentation snapshot: {snapshot}"
    )


def _format_gpu_fragmentation(cluster_resources: Any) -> list[dict[str, object]]:
    """Return ``(node, gpu_index, used_fraction, free_fraction)`` for every GPU.

    Stable-ordered by ``(node_id, gpu.index)`` so two snapshots of the
    same cluster compare as identical strings.
    """
    rows: list[dict[str, object]] = []
    for node_id in sorted(cluster_resources.nodes):
        node = cluster_resources.nodes[node_id]
        for gpu in sorted(node.gpus, key=lambda g: g.index):
            rows.append(
                {
                    "node": node_id,
                    "gpu_index": gpu.index,
                    "used_fraction": round(gpu.used_fraction, 4),
                    "free_fraction": round(max(0.0, 1.0 - gpu.used_fraction), 4),
                }
            )
    return rows


__all__ = [
    "ALLOCATION_FAILURES_METRIC",
    "emit_allocation_failure",
]
