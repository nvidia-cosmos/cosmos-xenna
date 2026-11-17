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


import uuid
from unittest import mock

import pytest

from cosmos_xenna.pipelines.private import resources
from cosmos_xenna.pipelines.private.resources import parse_visible_cuda_devices


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ("", []),
        (None, None),
        ("0,2", [0, 2]),
        (" 1 ", [1]),
    ],
)
def test_parse_indices(tokens, expected):
    assert parse_visible_cuda_devices(tokens) == expected


def test_parse_full_uuid_with_prefix():
    u = uuid.UUID("00000000-0000-0000-0000-000000000002")
    tokens = f"GPU-{u}"
    out = parse_visible_cuda_devices(tokens)
    assert out == [u]


def test_parse_full_uuid_without_prefix():
    u = uuid.UUID("00000000-0000-0000-0000-000000000003")
    out = parse_visible_cuda_devices(str(u))
    assert out == [u]


@pytest.mark.parametrize("prefix_len", [4, 8, 12, 16])
def test_parse_short_uuid_prefix_with_prefix(prefix_len):
    u = uuid.UUID("33333333-3333-3333-3333-333333333333")
    short = str(u)[:prefix_len]
    out = parse_visible_cuda_devices(f"GPU-{short}")
    # parser returns normalized compact form (no hyphens, lowercase)
    assert out == [str(u)[:prefix_len]]


@pytest.mark.parametrize("prefix_len", [4, 8, 12, 16])
def test_parse_short_uuid_prefix_without_prefix(prefix_len):
    u = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    short = str(u)[:prefix_len]
    out = parse_visible_cuda_devices(short)
    assert out == [str(u)[:prefix_len]]


def test_parse_short_uuid_prefix_compact_form():
    u = uuid.UUID("12345678-90ab-cdef-1234-567890abcdef")
    compact_prefix = str(u)[:10]
    out = parse_visible_cuda_devices(compact_prefix)
    assert out == [compact_prefix]


def test_parse_multiple_mixed_tokens():
    u = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    # index, full UUID, and short prefix
    tokens = f"0,GPU-{u},ccc"
    out = parse_visible_cuda_devices(tokens)
    assert out == [0, u, "ccc"]


def test_parse_invalid_token_raises():
    with pytest.raises(ValueError):
        parse_visible_cuda_devices("GPU-")


# Ray checks that node IDs are 28-byte hex strings.
_HEAD_ID = "aa" * 28
_WORKER_ID = "bb" * 28


def _ray_node(node_id: str, ray_resources: dict[str, float]) -> dict:
    """A ``ray.nodes()`` entry. Like Ray, leave out resources the node has none of."""
    return {"NodeID": node_id, "Alive": True, "NodeManagerHostname": node_id, "Resources": ray_resources}


def _make_cluster_resources(
    nodes: list[dict], machine_cpus: int, machine_gpus: int
) -> tuple[resources.ClusterResources, list[str]]:
    """Run ``make_cluster_resources_for_ray_cluster`` without Ray.

    The per-node probe is replaced by one that reports ``machine_cpus`` CPUs and ``machine_gpus`` GPUs for every
    node, as ``os.cpu_count()`` and NVML would on a real machine. Returns the cluster and the probed node IDs.
    """
    probed: list[str] = []

    def options(scheduling_strategy, **kwargs):
        probed.append(scheduling_strategy.node_id)
        return mock.MagicMock(remote=mock.MagicMock(return_value=scheduling_strategy.node_id))

    def fake_get(node_ids):
        gpus = [resources.GpuInfo(index=i, name="gpu", uuid_=uuid.uuid4()) for i in range(machine_gpus)]
        return [resources.ResourceInfoFromNode(node_id=x, cpus=machine_cpus, gpus=gpus) for x in node_ids]

    probe = mock.MagicMock(options=options)
    with (
        mock.patch.object(resources, "_get_node_info_from_current_node", probe),
        mock.patch.object(resources.ray, "get", fake_get),
    ):
        cluster = resources.make_cluster_resources_for_ray_cluster(nodes=nodes)
    return cluster, probed


def test_node_without_cpus_or_gpus_in_ray_is_skipped():
    # A head node started with `ray start --head --num-cpus=0 --num-gpus=0`, as on Anyscale.
    head = _ray_node(_HEAD_ID, {"memory": 1.0, "node:__internal_head__": 1.0})
    worker = _ray_node(_WORKER_ID, {"CPU": 4.0, "GPU": 1.0, "memory": 1.0})

    cluster, probed = _make_cluster_resources([head, worker], machine_cpus=64, machine_gpus=1)

    assert probed == [_WORKER_ID]
    assert list(cluster.nodes) == [_WORKER_ID]
    assert cluster.nodes[_WORKER_ID].total_cpus == 4
    assert len(cluster.nodes[_WORKER_ID].gpus) == 1


def test_node_without_cpus_in_ray_has_no_cpus():
    # A missing CPU entry means the node has no CPUs, not "use the CPU count of the machine".
    node = _ray_node(_WORKER_ID, {"GPU": 2.0, "memory": 1.0})

    cluster, _ = _make_cluster_resources([node], machine_cpus=64, machine_gpus=2)

    assert cluster.nodes[_WORKER_ID].total_cpus == 0
    assert len(cluster.nodes[_WORKER_ID].gpus) == 2


def test_cpu_count_keeps_zero_reported_by_ray():
    from_ray = resources.ResourceInfoFromRay(node_id="n", cpus=0.0, gpus=None)
    from_node = resources.ResourceInfoFromNode(node_id="n", cpus=64, gpus=[])
    assert resources._get_cpu_count(from_ray, from_node) == 0


def test_cpu_count_falls_back_to_node_when_ray_does_not_report():
    from_ray = resources.ResourceInfoFromRay(node_id="n", cpus=None, gpus=None)
    from_node = resources.ResourceInfoFromNode(node_id="n", cpus=64, gpus=[])
    assert resources._get_cpu_count(from_ray, from_node) == 64
