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

from cosmos_xenna.pipelines.private.scheduling_py.saturation_aware import grace

_GRACE_S = 60.0


def test_just_observed_worker_is_protected() -> None:
    g = grace.WarmupGrace()
    g.observe(["w0"], now=100.0)
    assert g.allowed_deletions(["w0"], now=100.0, grace_s=_GRACE_S) == []


def test_worker_older_than_grace_is_deletable() -> None:
    g = grace.WarmupGrace()
    g.observe(["w0"], now=30.0)
    assert g.allowed_deletions(["w0"], now=100.0, grace_s=_GRACE_S) == ["w0"]


def test_grace_protects_only_young_workers_in_a_mixed_list() -> None:
    g = grace.WarmupGrace()
    g.observe(["old"], now=10.0)
    g.observe(["old", "young"], now=70.0)  # 'old' keeps first-seen 10.0; 'young' first seen 70.0
    # now=100: old age=90 (>= 60, deletable); young age=30 (< 60, protected)
    assert g.allowed_deletions(["old", "young"], now=100.0, grace_s=_GRACE_S) == ["old"]


def test_unknown_worker_is_deletable() -> None:
    g = grace.WarmupGrace()
    g.observe(["w0"], now=100.0)
    assert g.allowed_deletions(["never_seen"], now=100.0, grace_s=_GRACE_S) == ["never_seen"]


def test_observe_forgets_workers_no_longer_present() -> None:
    g = grace.WarmupGrace()
    g.observe(["w0"], now=10.0)
    g.observe(["w1"], now=10.0)  # w0 no longer live -> forgotten, so it is no longer protected
    assert g.allowed_deletions(["w0"], now=12.0, grace_s=_GRACE_S) == ["w0"]
