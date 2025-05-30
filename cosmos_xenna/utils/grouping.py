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

"""
Utility Functions for grouping iterables.

This module provides a collection of utility functions designed to assist with common tasks related to manipulating
and transforming iterables in Python.

These utilities are generic and work with any iterable types. They're particularly useful for data processing tasks,
batching operations, and other scenarios where dividing data into specific groupings is necessary.

Note:
    While these utilities are designed for flexibility and ease-of-use,
    they may not be optimized for extremely large datasets or performance-critical applications.
"""

import itertools
import typing
from collections.abc import Generator, Iterable

T = typing.TypeVar("T")


def split_by_chunk_size(
    iterable: Iterable[T],
    chunk_size: int,
    drop_incomplete_chunk: bool = False,
) -> Generator[list[T], None, None]:
    """
    Splits an iterable into chunks of the specified size.

    Args:
    - iterable (Iterable[T]): The input iterable to be split.
    - chunk_size (int): Size of each chunk.
    - drop_incomplete_chunk (bool, optional): If True, drops the last chunk if its size is less than the
      specified chunk size. Defaults to False.

    Yields:
    - Generator[list[T], None, None]: Chunks of the input iterable.
    """
    out = []
    cur_count = 0
    for value in iterable:
        out.append(value)
        cur_count += 1
        if cur_count >= chunk_size:
            yield out
            out = []
            cur_count = 0
    if out and not drop_incomplete_chunk:
        yield out


def split_into_n_chunks(iterable: Iterable[T], num_chunks: int) -> Generator[list[T], None, None]:
    """
    Splits an iterable into a specified number of chunks.

    Args:
    - iterable (Iterable[T]): The input iterable to be split.
    - num_chunks (int): The desired number of chunks.

    Yields:
    - Generator[list[T], None, None]: Chunks of the input iterable.
    """
    it = list(iterable)
    if len(it) <= num_chunks:
        yield from [[x] for x in it]
        return
    d, r = divmod(len(it), num_chunks)
    for i in range(num_chunks):
        si = (d + 1) * min(r, i) + d * (0 if i < r else i - r)
        yield it[si : si + (d + 1 if i < r else d)]


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """
    Returns pairs of consecutive items from the input iterable.

    Args:
    - iterable (Iterable[T]): The input iterable.

    Returns:
    - Iterable[tuple[T, T]]: Pairs of consecutive items.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


K = typing.TypeVar("K")
V1 = typing.TypeVar("V1")
V2 = typing.TypeVar("V2")


class DictZipError(Exception):
    """Custom exception raised when the input dictionaries do not have identical keys."""

    pass


def dict_zip(d1: dict[K, V1], d2: dict[K, V2]) -> Generator[tuple[K, V1, V2], None, None]:
    """
    Zip two dictionaries together, yielding tuples of (key, value1, value2).

    This function creates a generator that iterates through the keys of two input dictionaries
    and yields tuples containing the key and the corresponding values from both dictionaries.
    It ensures that both dictionaries have exactly the same keys before proceeding.

    Args:
        d1 (Dict[K, V1]): The first dictionary to zip.
        d2 (Dict[K, V2]): The second dictionary to zip.

    Yields:
        tuple[K, V1, V2]: A tuple containing:
            - K: The key (of type K) present in both dictionaries.
            - V1: The value (of type V1) associated with the key in the first dictionary.
            - V2: The value (of type V2) associated with the key in the second dictionary.

    Raises:
        DictZipError: If the keys of the two input dictionaries are not identical.

    Example:
        >>> dict1 = {"a": 1, "b": 2, "c": 3}
        >>> dict2 = {"a": "one", "b": "two", "c": "three"}
        >>> for key, val1, val2 in dict_zip(dict1, dict2):
        ...     print(f"{key}: {val1}, {val2}")
        a: 1, one
        b: 2, two
        c: 3, three

    Notes:
        - This function uses sets to compare dictionary keys, so the order of keys
          in the input dictionaries does not affect the comparison.
        - The function is lazy and generates values on-the-fly, making it memory-efficient
          for large dictionaries.
        - Type annotations allow for generic usage with different key and value types.
    """
    if set(d1.keys()) != set(d2.keys()):
        raise DictZipError("The keys of the two dictionaries are not exactly equal")

    for key in d1.keys():
        yield key, d1[key], d2[key]
