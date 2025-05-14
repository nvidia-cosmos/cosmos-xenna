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

from typing import Union

EPSILON = 1e-6


def float_eq(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """Compare two floats with epsilon tolerance."""
    return abs(a - b) < epsilon


def float_gt(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """Compare if a is greater than b with epsilon tolerance."""
    return (a - b) > epsilon


def float_lt(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """Compare if a is less than b with epsilon tolerance."""
    return (b - a) > epsilon


def float_gte(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """Compare if a is greater than or equal to b with epsilon tolerance."""
    return not float_lt(a, b, epsilon)


def float_lte(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """Compare if a is less than or equal to b with epsilon tolerance."""
    return not float_gt(a, b, epsilon)


def is_almost_whole(x: Union[float, int], epsilon: float = EPSILON) -> bool:
    """
    Checks if a number is close to a whole number.

    Args:
        x: The number to check.
        tolerance: The allowed difference from a whole number.

    Returns:
        True if the number is close to a whole number, False otherwise.
    """
    return abs(x - round(x)) < epsilon
