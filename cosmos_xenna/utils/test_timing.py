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

from cosmos_xenna.utils import timing


def test_rate_limiter():
    # Given: A rate limiter set to 1 Hz
    limiter = timing.RateLimiter(1)

    # When: We record the time, use the limiter and then record the time again
    start_time = time.time()
    limiter.sleep()
    middle_time = time.time()
    limiter.sleep()
    end_time = time.time()

    # Then: The first call should not sleep, but the second call should sleep roughly 1 second
    assert middle_time - start_time < 0.1  # The first call should not introduce a noticeable delay
    assert 0.9 <= end_time - middle_time <= 1.1  # The second call should sleep around 1 second


def mock_function(*args, **kwargs):
    return True


def test_rate_limited_caller():
    # Given: A rate-limited caller set to 1 Hz
    caller = timing.RateLimitedCaller(1)

    # When: We attempt to call the function twice in quick succession
    first_call = caller.maybe_do(mock_function)
    second_call = caller.maybe_do(mock_function)

    # Then: The first call should execute, but the second call should be skipped
    assert first_call is True
    assert second_call is None

    # When: We wait for more than 1 second and attempt to call the function again
    time.sleep(1.1)
    third_call = caller.maybe_do(mock_function)

    # Then: The function should execute again
    assert third_call is True


def test_rate_limited_caller_with_zero_hz():
    caller = timing.RateLimitedCaller(0)

    first_call = caller.maybe_do(mock_function)
    second_call = caller.maybe_do(mock_function)

    assert first_call is None
    assert second_call is None


# === RateEstimatorDuration Tests ===


def test_rate_estimator_duration_basic():
    # Given: An estimator looking back 10 seconds
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=10)
    current_time = time.time()

    # When: We add two events with durations 0.5s and 0.7s
    estimator.update(duration=0.5, current_time=current_time)
    # Check rate with one event
    expected_rate_1 = 1.0 / 0.5
    assert estimator.get_rate(current_time=current_time + 0.05) == pytest.approx(expected_rate_1)
    assert estimator.maybe_get_rate(current_time=current_time + 0.05) == pytest.approx(expected_rate_1)

    estimator.update(duration=0.7, current_time=current_time + 0.1)  # Add second event

    # Then: The rate should be the inverse of the average duration (0.6s)
    expected_rate_2 = 1.0 / 0.6
    assert estimator.get_rate(current_time=current_time + 0.2) == pytest.approx(expected_rate_2)
    assert estimator.maybe_get_rate(current_time=current_time + 0.2) == pytest.approx(expected_rate_2)


def test_rate_estimator_duration_min_events_respected():
    # Given: An estimator looking back 1 second, requiring min 2 events
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=1, min_num_events=2)
    current_time = time.time()

    # When: We add one event
    estimator.update(duration=0.5, current_time=current_time)

    # Then: Rate should be based on the single event
    expected_rate_1 = 1.0 / 0.5
    assert estimator.get_rate(current_time=current_time + 0.1) == pytest.approx(expected_rate_1)
    assert estimator.maybe_get_rate(current_time=current_time + 0.1) == pytest.approx(expected_rate_1)

    # When: We add a second event far apart in time (older than the window)
    estimator.update(duration=0.7, current_time=current_time + 2)  # 2s > 1s window

    # Then: Even though the first event is > 1s old, it should be kept because min_num_events=2
    # Rate should be based on the two events (avg duration 0.6s)
    expected_rate_2_events = 1.0 / 0.6
    assert estimator.get_rate(current_time=current_time + 2.1) == pytest.approx(expected_rate_2_events)
    assert estimator.maybe_get_rate(current_time=current_time + 2.1) == pytest.approx(expected_rate_2_events)

    # When: We add a third event, also outside the original window relative to the first
    estimator.update(duration=0.9, current_time=current_time + 4)  # 4s > 1s window

    # Then: The first event (0.5s) should be dropped because it's too old and we have 3 events > min_num_events=2
    # Rate should be based on the remaining 2 events (0.7, 0.9), avg = 0.8s
    expected_rate_3_events = 1.0 / 0.8
    assert estimator.get_rate(current_time=current_time + 4.1) == pytest.approx(expected_rate_3_events)
    assert estimator.maybe_get_rate(current_time=current_time + 4.1) == pytest.approx(expected_rate_3_events)


def test_rate_estimator_duration_min_events_trumps_window():
    # Given: An estimator looking back 1 second, requiring min 3 events
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=1, min_num_events=3)
    current_time = time.time()

    # When: We add 4 events, with the first one falling outside the 1s window relative to the last
    estimator.update(duration=0.2, current_time=current_time)  # t=0
    estimator.update(duration=0.3, current_time=current_time + 0.5)  # t=0.5
    estimator.update(duration=0.4, current_time=current_time + 1.0)  # t=1.0
    estimator.update(duration=0.5, current_time=current_time + 1.5)  # t=1.5 (first event now >1s old)

    # Then: Because len(deque)=4 > min_num_events=3, the oldest event (duration 0.2) should be dropped
    # because it is outside the 1s window.
    # The rate should be based on the remaining 3 events (0.3, 0.4, 0.5), avg = 0.4
    expected_rate = 1.0 / 0.4
    # Check slightly after the last event
    assert estimator.get_rate(current_time=current_time + 1.6) == pytest.approx(expected_rate)
    assert estimator.maybe_get_rate(current_time=current_time + 1.6) == pytest.approx(expected_rate)


def test_rate_estimator_duration_no_events():
    # Given: An estimator
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=10)

    # Then: With no events, rate is 0 / None
    assert estimator.get_rate() == 0.0
    assert estimator.maybe_get_rate() is None


def test_rate_estimator_duration_single_event():
    # Given: An estimator
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=10)
    current_time = time.time()

    # When: One event is added
    estimator.update(duration=0.5, current_time=current_time)

    # Then: With only one event, rate is 1 / duration
    expected_rate = 1.0 / 0.5
    assert estimator.get_rate(current_time=current_time + 0.1) == pytest.approx(expected_rate)
    assert estimator.maybe_get_rate(current_time=current_time + 0.1) == pytest.approx(expected_rate)


def test_rate_estimator_duration_min_events_validation():
    # Then: Initializing with min_num_events < 1 should raise ValueError
    with pytest.raises(ValueError, match="min_num_events must be at least 1"):
        timing.RateEstimatorDuration(previous_duration_to_look_s=10, min_num_events=0)


def test_rate_estimator_duration_zero_duration():
    # Given: An estimator
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=10)
    current_time = time.time()

    # When: We add one event with zero duration
    estimator.update(duration=0.0, current_time=current_time)

    # Then: Rate should be 0.0 / None
    assert estimator.get_rate(current_time=current_time + 0.1) == 0.0
    assert estimator.maybe_get_rate(current_time=current_time + 0.1) is None

    # When: We add a second event with zero duration
    estimator.update(duration=0.0, current_time=current_time + 0.1)

    # Then: get_rate should still return 0.0, maybe_get_rate should return None
    assert estimator.get_rate(current_time=current_time + 0.2) == 0.0
    assert estimator.maybe_get_rate(current_time=current_time + 0.2) is None

    # When: We add a non-zero duration event
    estimator.update(duration=0.5, current_time=current_time + 0.3)

    # Then: Rate is based on average (0.0 + 0.0 + 0.5) / 3 = 0.5 / 3
    expected_rate = 1.0 / (0.5 / 3.0)
    assert estimator.get_rate(current_time=current_time + 0.4) == pytest.approx(expected_rate)
    assert estimator.maybe_get_rate(current_time=current_time + 0.4) == pytest.approx(expected_rate)


def test_rate_estimator_duration_basic_no_min_events():
    # Given: An estimator looking back 10 seconds, min_num_events=None (default)
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=10)
    current_time = time.time()

    # When: We add two events with durations 0.5s and 0.7s
    estimator.update(duration=0.5, current_time=current_time)
    # Check rate with one event
    expected_rate_1 = 1.0 / 0.5
    assert estimator.get_rate(current_time=current_time + 0.05) == pytest.approx(expected_rate_1)
    assert estimator.maybe_get_rate(current_time=current_time + 0.05) == pytest.approx(expected_rate_1)

    estimator.update(duration=0.7, current_time=current_time + 0.1)  # Add second event

    # Then: The rate should be the inverse of the average duration (0.6s)
    expected_rate_2 = 1.0 / 0.6
    assert estimator.get_rate(current_time=current_time + 0.2) == pytest.approx(expected_rate_2)
    assert estimator.maybe_get_rate(current_time=current_time + 0.2) == pytest.approx(expected_rate_2)


def test_rate_estimator_duration_no_min_events_removes_old():
    # Given: An estimator looking back 1 second, min_num_events=None
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=1)
    current_time = time.time()

    # When: We add two events, the second making the first older than the window
    estimator.update(duration=0.5, current_time=current_time)  # t=0
    estimator.update(duration=0.7, current_time=current_time + 1.1)  # t=1.1, first event is now 1.1s old

    # Then: The first event should be removed because it's > 1s old and min_num_events is None
    # Rate should be based only on the second event (duration 0.7s)
    expected_rate = 1.0 / 0.7
    assert estimator.get_rate(current_time=current_time + 1.2) == pytest.approx(expected_rate)
    assert estimator.maybe_get_rate(current_time=current_time + 1.2) == pytest.approx(expected_rate)
    # Check internal state if possible (or infer from rate)
    assert len(estimator._timestamps_and_durations) == 1


def test_rate_estimator_duration_min_events_set_keeps_old():
    # Given: An estimator looking back 1 second, but requiring min 1 event
    estimator = timing.RateEstimatorDuration(previous_duration_to_look_s=1, min_num_events=1)
    current_time = time.time()

    # When: We add one event
    estimator.update(duration=0.5, current_time=current_time)

    # When: We check the rate far in the future (event is older than window)
    future_time = current_time + 5
    # Then: The event should *still* be there because min_num_events=1
    expected_rate_1 = 1.0 / 0.5
    assert estimator.get_rate(current_time=future_time) == pytest.approx(expected_rate_1)
    assert estimator.maybe_get_rate(current_time=future_time) == pytest.approx(expected_rate_1)
    assert len(estimator._timestamps_and_durations) == 1
