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

import pytest  # noqa: F401

from cosmos_xenna.utils import grouping


def test_split_by_chunk_size():
    # Test basic chunking.
    assert list(grouping.split_by_chunk_size(range(10), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    # Test dropping incomplete chunks.
    assert list(grouping.split_by_chunk_size(range(10), 3, drop_incomplete_chunk=True)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]

    # Test chunk size larger than iterable length.
    assert list(grouping.split_by_chunk_size(range(3), 10)) == [[0, 1, 2]]

    # Test empty iterable.
    assert list(grouping.split_by_chunk_size([], 3)) == []


def test_split_into_n_chunks():
    # Test basic chunking.
    assert list(grouping.split_into_n_chunks(range(10), 3)) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Test number of chunks larger than iterable length.
    assert list(grouping.split_into_n_chunks(range(3), 10)) == [[0], [1], [2]]

    # Test splitting into equal-sized chunks.
    assert list(grouping.split_into_n_chunks(range(9), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    # Test empty iterable.
    assert list(grouping.split_into_n_chunks([], 3)) == []


def test_pairwise():
    # Test basic pairing.
    assert list(grouping.pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    # Test empty iterable.
    assert list(grouping.pairwise([])) == []

    # Test iterable with a single item.
    assert list(grouping.pairwise([1])) == []

    # Test iterable with two items.
    assert list(grouping.pairwise([1, 2])) == [(1, 2)]
