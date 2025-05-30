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
from typing import Union


def binom_cdf(k: Union[float, int], n: int, p: float) -> float:
    """
    Calculates the binomial cumulative distribution function (CDF) from scratch.

    This function computes the probability P(X <= k), where X is a random
    variable following the binomial distribution B(n, p). It does not require
    the scipy library.

    Args:
      k: The upper limit for the number of successes (float or int).
         If float, it will be floored to the nearest integer towards zero.
         The function calculates P(X <= floor(k)).
      n: The total number of trials (integer, n >= 0).
      p: The probability of success on a single trial (float, 0 <= p <= 1).

    Returns:
      The cumulative probability P(X <= floor(k)) as a float.

    Raises:
      ValueError: If n is negative, or p is not between 0 and 1.
      TypeError: If n is not an integer or k is not numeric.
    """
    # --- Input Validation ---
    if not isinstance(n, int):
        raise TypeError("'n' must be an integer.")
    if n < 0:
        raise ValueError("'n' must be a non-negative integer.")
    if not (0 <= p <= 1):
        raise ValueError("Probability 'p' must be between 0 and 1.")

    # Floor k if it's a float, mimicking scipy.stats.binom.cdf behavior
    if not isinstance(k, (int, float)):
        raise TypeError("'k' must be a numeric value (int or float).")
    k = math.floor(k)  # Convert k to integer for the loop range

    # --- Edge Case Handling ---
    if k < 0:
        # If k is negative, the cumulative probability is 0
        return 0.0
    if k >= n:
        # If k is n or more, the probability includes all possible outcomes (0 to n)
        # The CDF is 1, unless p=0 or p=1 leads to specific outcomes.
        # However, the loop below correctly sums to 1 in these cases too,
        # but returning 1.0 directly is more efficient.
        return 1.0

    # Handle p=0 and p=1 specifically for efficiency and to avoid 0^0 issues
    if p == 0:
        # If p=0, the only possible outcome is 0 successes (P(X=0)=1).
        # CDF(k) = P(X<=k) = 1 if k >= 0, else 0.
        # Since we handled k<0 above, k must be >= 0 here.
        return 1.0
    if p == 1:
        # If p=1, the only possible outcome is n successes (P(X=n)=1).
        # CDF(k) = P(X<=k) = 1 if k >= n, else 0.
        # Since we handled k>=n above, k must be < n here.
        return 0.0

    # --- Calculation ---
    cumulative_prob = 0.0
    for i in range(k + 1):  # Sum probabilities from i=0 up to k (inclusive)
        # Calculate the binomial coefficient C(n, i) = n! / (i! * (n-i)!)
        # Using math.comb (available in Python 3.8+) is efficient and accurate
        try:
            comb = math.comb(n, i)
        except AttributeError:
            # Fallback combination calculation if math.comb is not available (Python < 3.8)
            # Calculates C(n, i) iteratively to avoid large factorials
            if i < 0 or i > n:
                comb = 0  # Should not happen given loop range and k checks
            elif i == 0 or i == n:
                comb = 1
            elif i > n // 2:
                i = n - i  # Optimization: C(n, k) == C(n, n-k)  # noqa: PLW2901

            # Calculate C(n,i) = product_{j=1 to i} (n - j + 1) / j
            if i == 0:  # Handles the case after potential optimization i = n-i
                comb = 1
            else:
                # Use floating point division for intermediate steps
                res = 1.0
                for j in range(i):
                    res = res * (n - j) / (j + 1)
                # Binomial coefficient is always an integer, round to handle potential float inaccuracies
                comb = round(res)

        # Calculate the probability term: C(n, i) * p^i * (1-p)^(n-i)
        # Using math.pow can sometimes offer better precision for edge cases like 0^0
        try:
            prob_term = comb * math.pow(p, i) * math.pow(1 - p, n - i)
        except ValueError:
            # Handle potential domain errors from math.pow, e.g., 0**negative
            # This shouldn't typically happen with valid p, n, i
            prob_term = 0.0  # Assign 0 probability if calculation fails

        # Add the probability of exactly i successes to the cumulative total
        cumulative_prob += prob_term

    # Clamp the result to [0, 1] to correct any minor floating-point inaccuracies
    return max(0.0, min(1.0, cumulative_prob))
