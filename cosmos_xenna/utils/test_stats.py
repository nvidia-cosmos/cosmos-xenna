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

import math

import pytest

from cosmos_xenna.utils.stats import binom_cdf


def test_standard_cases():
    """Tests standard, typical inputs against known results."""
    # Fair coin, 10 flips, P(X<=5)
    assert binom_cdf(k=5, n=10, p=0.5) == pytest.approx(0.623046875, abs=1e-5)
    # Biased coin (p=0.7), 20 flips, P(X<=15)
    assert binom_cdf(k=15, n=20, p=0.7) == pytest.approx(0.7624949477, abs=1e-5)
    # Different case: n=8, p=0.2, k=2 => P(X<=2)
    # P(X=0) = C(8,0)*0.2^0*0.8^8 = 0.16777216
    # P(X=1) = C(8,1)*0.2^1*0.8^7 = 8*0.2*0.2097152 = 0.33554432
    # P(X=2) = C(8,2)*0.2^2*0.8^6 = 28*0.04*0.262144 = 0.29360128
    # Sum = 0.16777216 + 0.33554432 + 0.29360128 = 0.79691776
    assert binom_cdf(k=2, n=8, p=0.2) == pytest.approx(0.79691776, abs=1e-5)


def test_edge_k():
    """Tests edge cases for k."""
    # k < 0
    assert binom_cdf(k=-1, n=10, p=0.5) == 0.0
    # k = 0
    assert binom_cdf(k=0, n=10, p=0.5) == pytest.approx(math.pow(0.5, 10))  # P(X=0)
    # k = n
    assert binom_cdf(k=10, n=10, p=0.5) == 1.0
    # k > n
    assert binom_cdf(k=11, n=10, p=0.5) == 1.0
    # k = n-1
    # P(X<=n-1) = 1 - P(X=n) = 1 - C(n,n)*p^n*(1-p)^0 = 1 - p^n
    assert binom_cdf(k=9, n=10, p=0.5) == pytest.approx(1.0 - math.pow(0.5, 10))


def test_edge_p():
    """Tests edge cases for p."""
    # p = 0
    assert binom_cdf(k=0, n=10, p=0) == 1.0  # P(X<=0) is P(X=0) which is 1
    assert binom_cdf(k=1, n=10, p=0) == 1.0  # P(X<=1) is P(X=0)+P(X=1)=1+0=1
    assert binom_cdf(k=-1, n=10, p=0) == 0.0  # k<0 case

    # p = 1
    assert binom_cdf(k=9, n=10, p=1) == 0.0  # P(X<=9) is 0 (only X=10 possible)
    assert binom_cdf(k=10, n=10, p=1) == 1.0  # P(X<=10) is 1
    assert binom_cdf(k=11, n=10, p=1) == 1.0  # k>n case


def test_float_k():
    """Tests that float values for k are handled correctly (floored)."""
    # k=5.3 should be same as k=5
    assert binom_cdf(k=5.3, n=10, p=0.5) == binom_cdf(k=5, n=10, p=0.5)
    # k=5.9 should be same as k=5
    assert binom_cdf(k=5.9, n=10, p=0.5) == binom_cdf(k=5, n=10, p=0.5)
    # k=0.1 should be same as k=0
    assert binom_cdf(k=0.1, n=10, p=0.5) == binom_cdf(k=0, n=10, p=0.5)
    # k=-0.1 should be same as k=-1 (result 0)
    assert binom_cdf(k=-0.1, n=10, p=0.5) == 0.0
    assert binom_cdf(k=-0.9, n=10, p=0.5) == 0.0


def test_invalid_inputs():
    """Tests that invalid inputs raise the appropriate errors."""
    # Invalid n
    with pytest.raises(ValueError, match=r"'n' must be a non-negative integer."):
        binom_cdf(k=5, n=-1, p=0.5)
    with pytest.raises(TypeError, match=r"'n' must be an integer."):
        binom_cdf(k=5, n=10.5, p=0.5)  # type: ignore

    # Invalid p
    with pytest.raises(ValueError, match=r"Probability 'p' must be between 0 and 1."):
        binom_cdf(k=5, n=10, p=-0.1)
    with pytest.raises(ValueError, match=r"Probability 'p' must be between 0 and 1."):
        binom_cdf(k=5, n=10, p=1.1)

    # Invalid k type
    with pytest.raises(TypeError, match=r"'k' must be a numeric value"):
        binom_cdf(k="abc", n=10, p=0.5)  # type: ignore
    with pytest.raises(TypeError, match=r"'k' must be a numeric value"):
        binom_cdf(k=None, n=10, p=0.5)  # type: ignore


def test_larger_n():
    """Tests a case with a larger n."""
    # Compare with scipy.stats.binom.cdf(25, 50, 0.5) -> 0.5561446391097192
    assert binom_cdf(k=25, n=50, p=0.5) == pytest.approx(0.5561446, abs=1e-4)
    # Compare with scipy.stats.binom.cdf(10, 30, 0.2) -> 0.9744794114544817
    assert binom_cdf(k=10, n=30, p=0.2) == pytest.approx(0.9744794, abs=1e-4)


def test_zero_trials():
    """Tests the case where n=0."""
    # If n=0, only k=0 is possible, P(X=0)=1.
    assert binom_cdf(k=0, n=0, p=0.5) == 1.0
    assert binom_cdf(k=-1, n=0, p=0.5) == 0.0
    assert binom_cdf(k=1, n=0, p=0.5) == 1.0  # k>=n edge case
    assert binom_cdf(k=0, n=0, p=0) == 1.0
    assert binom_cdf(k=0, n=0, p=1) == 1.0
