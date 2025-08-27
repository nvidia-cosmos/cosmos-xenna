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
Module for Rate Limiting and Periodic Function Calls

This module provides tools for rate limiting and making periodic function calls.
"""

import collections
import contextlib
import statistics
import time
import typing
from typing import Optional

from cosmos_xenna.utils import python_log as logger

T = typing.TypeVar("T")


class RateLimiter:
    """
    RateLimiter enforces a maximum call frequency.

    This class is used to ensure a piece of code doesn't execute more frequently than a specified rate.
    """

    def __init__(self, freq_hz: float) -> None:
        """
        Initialize the RateLimiter.

        Args:
            freq_hz (float): Desired frequency in hertz (i.e., calls per second).
        """
        self._freq = float(freq_hz)
        self._last_time = 0

    def sleep(self) -> None:
        """
        Sleeps the program if needed to maintain the desired rate.

        If called more frequently than the desired frequency, this method will
        pause the execution to maintain the rate.
        """
        # If this is the first call, just update the timestamp and return.
        if not self._last_time:
            self._last_time = time.time()
            return

        # Calculate required sleep duration to maintain desired rate.
        time_to_sleep = self._last_time + 1.0 / self._freq - time.time()
        # If sleep time is positive, sleep for the required duration.
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        # Update the last called time.
        self._last_time = time.time()


class RateLimitedCaller:
    """
    RateLimitedCaller allows functions to be called at a restricted rate.

    This class provides a mechanism to invoke a function at no more than a specified frequency.
    If the function is requested to be called more frequently, it will be skipped.
    """

    def __init__(self, freq_hz: float) -> None:
        """
        Initialize the RateLimitedCaller.

        Args:
            freq_hz (float): Desired call frequency in hertz (i.e., calls per second).
        """
        self._freq_hz = float(freq_hz)
        self._next_time = 0

    def maybe_do(self, func: typing.Callable[..., T], *args, **kwargs) -> Optional[T]:
        """
        Possibly execute the given function based on the desired rate.

        If the function is due to be called (based on the desired frequency), it's executed.
        Otherwise, the function is skipped.

        Args:
            func (Callable): Function to be executed.
            *args: Positional arguments for `func`.
            **kwargs: Keyword arguments for `func`.

        Returns:
            Optional[T]: The result of the function if executed, otherwise None.
        """
        current_time = time.time()

        # If the current time is beyond the next scheduled time, execute the function.
        if self._freq_hz > 0 and current_time >= self._next_time:
            self._next_time = current_time + 1.0 / self._freq_hz
            return func(*args, **kwargs)
        else:
            # If not due, skip the function call.
            return None


class RateLimitChecker:
    """
    RateLimitedCaller checks if a function can be called based on a restricted rate.

    This class provides a mechanism to check if a function can be invoked at a specified frequency.
    It returns True if the function can be called without hitting the rate limit, and False otherwise.
    """

    def __init__(self, freq_hz: float) -> None:
        """
        Initialize the RateLimitedCaller.

        Args:
            freq_hz (float): Desired check frequency in hertz (i.e., checks per second).
        """
        self._freq_hz = float(freq_hz)
        self._next_time = 0

    def can_call(self, check_only: bool = False) -> bool:
        """
        Check if the function can be executed based on the desired rate.

        args:
            check_only: If True, only returns if the timer is ready for another call. Will not actually move the timer.

        Returns:
            bool: True if the function can be called, otherwise False.
        """
        current_time = time.time()

        # If the current time is beyond the next scheduled time, the function can be called.
        if current_time >= self._next_time:
            if not check_only:
                self._next_time = current_time + 1.0 / self._freq_hz
            return True
        else:
            # If not due, function cannot be called.
            return False


class RateEstimator:
    def __init__(self, previous_duration_to_look_s: float):
        """
        Initializes the RateEstimator object.

        :param x_seconds: The window size in seconds for the rolling average calculation.
        """
        self._previous_duration_to_look_s = float(previous_duration_to_look_s)
        self._timestamps = collections.deque()

    def _remove_old(self, current_time: float) -> None:
        while self._timestamps and current_time - self._timestamps[0] > self._previous_duration_to_look_s:
            self._timestamps.popleft()

    def update(self) -> None:
        """
        Updates the internal state with a new event timestamp, while removing
        outdated timestamps.
        """
        current_time = time.time()
        # Append the current timestamp
        self._timestamps.append(current_time)
        self._remove_old(current_time)

    def get_rate(self) -> float:
        """
        Calculates and returns the current rate in Hertz, based on the rolling
        average of events in the last x_seconds.

        :return: The calculated rate in Hertz.
        """
        self._remove_old(time.time())
        if not self._timestamps or len(self._timestamps) < 2:
            return 0  # Not enough data to calculate rate

        # Calculate the difference in time between the first and last timestamps in the window
        time_diff = self._timestamps[-1] - self._timestamps[0]

        if time_diff == 0:
            return float(len(self._timestamps))  # Prevent division by zero; all events happened at the same time

        # The rate is the number of events divided by the total time in seconds
        rate = (len(self._timestamps) - 1) / time_diff
        return rate


class RateEstimatorDuration:
    def __init__(self, previous_duration_to_look_s: float, min_num_events: Optional[int] = None):
        """
        Initializes the DurationRateEstimator object.

        Calculates the rate based on the average duration of events within the
        last `previous_duration_to_look_s` seconds.

        If `min_num_events` is provided (must be >= 1), it will always keep at least
        that many events in its history, even if they fall outside the time window,
        to ensure a rate can be calculated for infrequent events.

        If `min_num_events` is None (the default), events older than
        `previous_duration_to_look_s` are always removed, regardless of how many
        events remain.

        Args:
            previous_duration_to_look_s: The window size in seconds to primarily
                consider for the rolling average of durations.
            min_num_events: If not None, the minimum number of event durations to
                keep, even if they are older than `previous_duration_to_look_s`.
                Must be at least 1 if provided.
        """
        if min_num_events is not None and min_num_events < 1:
            raise ValueError("min_num_events must be at least 1 if provided.")
        self._previous_duration_to_look_s = float(previous_duration_to_look_s)
        self._min_num_events = min_num_events
        self._timestamps_and_durations: collections.deque[tuple[float, float]] = collections.deque()

    def _remove_old(self, current_time: float) -> None:
        """Removes old events based on time window and optionally min_num_events."""
        while self._timestamps_and_durations:  # Check if deque is non-empty first
            # Check if the oldest event is outside the time window
            is_too_old = current_time - self._timestamps_and_durations[0][0] > self._previous_duration_to_look_s

            # If min_num_events is set, check if we are above the minimum count
            can_remove_based_on_count = (
                self._min_num_events is None or len(self._timestamps_and_durations) > self._min_num_events
            )

            # Remove if it's too old AND we are allowed to remove based on count (or count isn't enforced)
            if is_too_old and can_remove_based_on_count:
                self._timestamps_and_durations.popleft()
            else:
                # If the oldest isn't removable (either too new or protected by min_count), stop checking
                break

    def update(self, duration: float, current_time: Optional[float] = None) -> None:
        """
        Updates the estimator with a new duration and timestamp.

        Removes old entries based on the configured time window and `min_num_events` policy.

        Args:
            duration: Duration of the event in seconds.
            current_time: Optional timestamp for the event; defaults to `time.time()`.
        """
        if current_time is None:
            current_time = time.time()
        # Add the new duration
        self._timestamps_and_durations.append((current_time, duration))
        self._remove_old(current_time)

    def get_rate(self, current_time: Optional[float] = None) -> float:
        """
        Calculates and returns the average rate in events per second.

        The rate is based on the average duration of the events currently stored,
        respecting the time window and optional `min_num_events` criteria.

        Args:
            current_time: Optional timestamp to use as 'now'; defaults to `time.time()`.

        Returns:
            The average rate in events per second (Hz). Returns 0 if no events are
            stored or if the average duration is zero.
        """
        if current_time is None:
            current_time = time.time()
        self._remove_old(current_time)
        # Need at least 1 event to calculate a rate based on average duration.
        if not self._timestamps_and_durations:
            return 0.0  # No data to calculate rate

        # Calculate the average duration of events within the window/min_events
        average_duration = statistics.mean([x[1] for x in self._timestamps_and_durations])
        # Rate is the inverse of the average duration. Handle potential division by zero.
        return 1.0 / average_duration if average_duration > 0 else 0.0

    def maybe_get_rate(self, current_time: Optional[float] = None) -> Optional[float]:
        """
        Calculates and returns the average rate, or None if insufficient data.

        Similar to `get_rate`, but returns None if no events are stored or if the
        average duration is zero.

        Args:
            current_time: Optional timestamp to use as 'now'; defaults to `time.time()`.

        Returns:
            The average rate in events per second (Hz), or None if no events are
            stored or the average duration is zero.
        """
        if current_time is None:
            current_time = time.time()
        self._remove_old(current_time)
        # Need at least 1 event to calculate a rate.
        if not self._timestamps_and_durations:
            return None  # No data to calculate rate

        # Calculate the average duration of events within the window/min_events
        average_duration = statistics.mean([x[1] for x in self._timestamps_and_durations])
        # Rate is the inverse of the average duration. Handle potential division by zero.
        return 1.0 / average_duration if average_duration > 0 else None


@contextlib.contextmanager
def time_operation(operation_name: str) -> typing.Iterator[None]:
    """Context manager which logs the time take for an operation."""
    start_time = time.time()
    logger.info(f"Running operation = '{operation_name}'...")
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{operation_name} took {duration:.2f} seconds.")
