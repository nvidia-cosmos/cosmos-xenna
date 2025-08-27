// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub const EPSILON: f64 = 1e-6;

/// Compare two floats with epsilon tolerance.
pub fn float_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

/// Compare if a is greater than b with epsilon tolerance.
pub fn float_gt(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b) > epsilon
}

/// Compare if a is less than b with epsilon tolerance.
pub fn float_lt(a: f64, b: f64, epsilon: f64) -> bool {
    (b - a) > epsilon
}

/// Compare if a is greater than or equal to b with epsilon tolerance.
pub fn float_gte(a: f64, b: f64, epsilon: f64) -> bool {
    !float_lt(a, b, epsilon)
}

/// Compare if a is less than or equal to b with epsilon tolerance.
pub fn float_lte(a: f64, b: f64, epsilon: f64) -> bool {
    !float_gt(a, b, epsilon)
}

/// Checks if a number is close to a whole number.
///
/// # Arguments
///
/// * `x`: The number to check.
/// * `epsilon`: The allowed difference from a whole number.
///
/// # Returns
///
/// True if the number is close to a whole number, False otherwise.
pub fn is_almost_whole(x: f64, epsilon: f64) -> bool {
    (x - x.round()).abs() < epsilon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_eq() {
        assert!(float_eq(1.0, 1.0, EPSILON));
        assert!(float_eq(1.0, 1.0 + EPSILON / 2.0, EPSILON));
        assert!(!float_eq(1.0, 1.0 + EPSILON * 2.0, EPSILON));
        assert!(!float_eq(1.0, 1.0 - EPSILON * 2.0, EPSILON));
        assert!(float_eq(0.1 + 0.2, 0.3, EPSILON));
    }

    #[test]
    fn test_float_gt() {
        assert!(float_gt(1.0, 0.0, EPSILON));
        assert!(float_gt(1.0, 1.0 - EPSILON * 2.0, EPSILON));
        assert!(!float_gt(1.0, 1.0, EPSILON));
        assert!(!float_gt(1.0, 1.0 - EPSILON / 2.0, EPSILON));
    }

    #[test]
    fn test_float_lt() {
        assert!(float_lt(0.0, 1.0, EPSILON));
        assert!(float_lt(1.0, 1.0 + EPSILON * 2.0, EPSILON));
        assert!(!float_lt(1.0, 1.0, EPSILON));
        assert!(!float_lt(1.0, 1.0 + EPSILON / 2.0, EPSILON));
    }

    #[test]
    fn test_float_gte() {
        assert!(float_gte(1.0, 0.0, EPSILON));
        assert!(float_gte(1.0, 1.0, EPSILON));
        assert!(float_gte(1.0, 1.0 - EPSILON / 2.0, EPSILON));
        assert!(float_gte(1.0, 1.0 + EPSILON / 2.0, EPSILON));
        assert!(!float_gte(1.0, 1.0 + EPSILON * 2.0, EPSILON));
    }

    #[test]
    fn test_float_lte() {
        assert!(float_lte(0.0, 1.0, EPSILON));
        assert!(float_lte(1.0, 1.0, EPSILON));
        assert!(float_lte(1.0, 1.0 + EPSILON / 2.0, EPSILON));
        assert!(float_lte(1.0, 1.0 - EPSILON / 2.0, EPSILON));
        assert!(!float_lte(1.0, 1.0 - EPSILON * 2.0, EPSILON));
    }

    #[test]
    fn test_is_almost_whole() {
        assert!(is_almost_whole(1.0, EPSILON));
        assert!(!is_almost_whole(1.001, EPSILON));
        assert!(is_almost_whole(0.9999999, EPSILON));
        assert!(is_almost_whole(2.0, EPSILON));
        assert!(is_almost_whole(1.99999999, EPSILON));
        assert!(is_almost_whole(1.0 + EPSILON / 2.0, EPSILON));
        assert!(!is_almost_whole(1.0 + EPSILON * 2.0, EPSILON));
    }
}
