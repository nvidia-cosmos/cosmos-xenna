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

use std::{
    thread::sleep,
    time::{Duration, Instant},
};

pub struct RateLimiter {
    min_duration: Duration,
    last_tick: Instant,
}

impl RateLimiter {
    pub fn new(min_duration: Duration) -> Self {
        RateLimiter {
            min_duration,
            last_tick: Instant::now(),
        }
    }

    pub fn wait(&mut self) {
        let elapsed = self.last_tick.elapsed();
        if elapsed < self.min_duration {
            sleep(self.min_duration - elapsed);
        }
        self.last_tick = Instant::now();
    }
}

pub struct RateLimitedDoer {
    last_triggered: Option<Instant>,
    duration: Duration,
}

impl RateLimitedDoer {
    pub fn new(duration: Duration) -> Self {
        RateLimitedDoer {
            last_triggered: None,
            duration,
        }
    }

    pub fn maybe_do<F: FnOnce()>(&mut self, action: F) {
        let now = Instant::now();
        if self.last_triggered.is_none()
            || now.duration_since(self.last_triggered.unwrap()) >= self.duration
        {
            action();
            self.last_triggered = Some(now);
        }
    }
}