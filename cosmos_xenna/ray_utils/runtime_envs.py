# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import copy
import os
from typing import Optional

import attrs
import ray.runtime_env

# Logging env vars forwarded to every actor so worker processes configure loguru
# (via cosmos_xenna.utils.python_log) identically to the driver. Ray actors are
# separate processes started by the raylet; they inherit the raylet's environment,
# but forwarding these explicitly guarantees consistent structured logging even if
# the raylet was started without them.
_FORWARDED_LOG_ENV_VARS = ("PYTHON_LOG", "PYTHON_LOG_FORMAT", "CURATOR_RUN_ID")


@attrs.define
class CondaEnv:
    name: str


@attrs.define
class RuntimeEnv:
    """A typed wrapper around the ray runtime environment class.

    We use this for clarity when setting up the runtime environment for a pipeline.
    """

    conda: Optional[CondaEnv] = None
    extra_env_vars: dict[str, str] = attrs.field(factory=dict)

    def to_ray_runtime_env(self) -> ray.runtime_env.RuntimeEnv:
        env_vars = copy.deepcopy(self.extra_env_vars)
        # Only forward logging toggles/identity to workers when structured logging is
        # active (PYTHON_LOG_FORMAT=json). In the default text mode this is a no-op, so
        # non-JSON cosmos-xenna consumers see no change to actor environments. Never
        # override anything a stage explicitly set in extra_env_vars.
        #
        # Resolve the mode from the effective value the worker would see: a caller that
        # sets PYTHON_LOG_FORMAT via extra_env_vars still triggers forwarding, even if
        # the driver process env itself does not carry it.
        log_format = env_vars.get("PYTHON_LOG_FORMAT", os.environ.get("PYTHON_LOG_FORMAT", ""))
        if log_format.strip().lower() == "json":
            for name in _FORWARDED_LOG_ENV_VARS:
                value = os.environ.get(name)
                if value is not None and name not in env_vars:
                    env_vars[name] = value

        if self.conda:
            return ray.runtime_env.RuntimeEnv(env_vars=env_vars, conda=self.conda.name)
        return ray.runtime_env.RuntimeEnv(env_vars=env_vars)

    def format(self) -> str:
        out = []
        if self.conda:
            out.append(f"conda: {self.conda.name}")
        if self.extra_env_vars:
            # Don't show key values as they may be secrets
            out.append(f"extra_env_vars: {', '.join(self.extra_env_vars.keys())}")
        return "\n".join(out)
