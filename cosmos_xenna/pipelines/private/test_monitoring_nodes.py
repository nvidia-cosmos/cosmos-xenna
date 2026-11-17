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

"""Tests for which nodes ``RayResourceMonitor`` starts a ``NodeResourceMonitor`` on."""

from unittest import mock

import pytest

import cosmos_xenna.pipelines.private.monitoring as monitoring

# Ray checks that node IDs are 28-byte hex strings.
_HEAD_ID = "aa" * 28
_GPU_ONLY_ID = "bb" * 28
_WORKER_ID = "cc" * 28
_DEAD_ID = "dd" * 28


@pytest.mark.L1
@pytest.mark.CPU
def test_ray_monitor_skips_nodes_without_cpus() -> None:
    """A node with no CPUs in Ray (e.g. a `--num-cpus=0` head) can never run the one-CPU monitor actor."""
    nodes = [
        {"NodeID": _HEAD_ID, "Alive": True, "Resources": {"memory": 1.0, "node:__internal_head__": 1.0}},
        {"NodeID": _GPU_ONLY_ID, "Alive": True, "Resources": {"GPU": 1.0}},
        {"NodeID": _WORKER_ID, "Alive": True, "Resources": {"CPU": 4.0}},
        {"NodeID": _DEAD_ID, "Alive": False, "Resources": {"CPU": 4.0}},
    ]
    with (
        mock.patch.object(monitoring.ray, "nodes", return_value=nodes),
        mock.patch.object(monitoring, "NodeResourceMonitor") as monitor_cls,
    ):
        ray_monitor = monitoring.RayResourceMonitor()

    assert ray_monitor._node_ids == [_WORKER_ID]
    assert monitor_cls.options.call_count == 1
    assert monitor_cls.options.call_args.kwargs["scheduling_strategy"].node_id == _WORKER_ID
