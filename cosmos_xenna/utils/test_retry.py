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

import time

import pytest

from cosmos_xenna.utils import retry


def test_successful_execution():
    result = retry.do_with_retries(lambda: "success")
    assert result == "success"


def test_retry_on_specific_exception():
    attempts = [0]

    def test_func():
        if attempts[0] < 2:
            attempts[0] += 1
            raise ValueError("Temporary error")
        return "success"

    result = retry.do_with_retries(test_func, exceptions_to_retry=[ValueError], max_attempts=3, max_wait_time_s=0.0)
    assert result == "success"
    assert attempts[0] == 2


def test_retry_limit():
    attempts = [0]

    def test_func():
        attempts[0] += 1
        raise ValueError("Temporary error")

    with pytest.raises(ValueError):
        retry.do_with_retries(test_func, exceptions_to_retry=[ValueError], max_attempts=3, max_wait_time_s=0.0)
    assert attempts[0] == 3


def test_backoff_and_max_wait_time():
    start_time = time.time()
    attempts = [0]

    def test_func():
        attempts[0] += 1
        if attempts[0] < 4:
            raise ValueError("Temporary error")
        return "success"

    retry.do_with_retries(
        test_func, exceptions_to_retry=[ValueError], max_attempts=4, backoff_factor=1, max_wait_time_s=0.01
    )
    elapsed_time = time.time() - start_time
    assert 0.03 <= elapsed_time


def test_different_exceptions():
    attempts = [0]

    def test_func():
        attempts[0] += 1
        if attempts[0] == 1:
            raise ValueError("Temporary error")
        elif attempts[0] == 2:
            raise IndexError("Temporary error")
        return "success"

    result = retry.do_with_retries(
        test_func,
        exceptions_to_retry=[ValueError, IndexError],
        max_attempts=3,
        max_wait_time_s=0.0,
    )
    assert result == "success"
    assert attempts[0] == 3


def test_no_retry_for_unspecified_exceptions():
    attempts = [0]

    def test_func():
        attempts[0] += 1
        raise KeyError("Permanent error")

    with pytest.raises(KeyError):
        retry.do_with_retries(test_func, exceptions_to_retry=[ValueError], max_attempts=3, max_wait_time_s=0.0)
    assert attempts[0] == 1


def test_retry_for_generic_exceptions():
    attempts = [0]

    def test_func():
        attempts[0] += 1
        if attempts[0] < 4:
            raise ValueError("Temporary error")
        return "success"

    retry.do_with_retries(test_func, max_attempts=4, max_wait_time_s=0.0)
    assert attempts[0] == 4
