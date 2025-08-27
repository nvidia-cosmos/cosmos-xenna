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

import uuid

import pytest

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
