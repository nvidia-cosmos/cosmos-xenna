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

import copy
from typing import Optional

import attrs
import ray.runtime_env


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
        kwargs = {}
        kwargs["env_vars"] = copy.deepcopy(self.extra_env_vars)

        if self.conda:
            kwargs["conda"] = self.conda.name

        return ray.runtime_env.RuntimeEnv(**kwargs)

    def format(self) -> str:
        out = []
        if self.conda:
            out.append(f"conda: {self.conda.name}")
        if self.extra_env_vars:
            # Don't show key values as they may be secrets
            out.append(f"extra_env_vars: {', '.join(self.extra_env_vars.keys())}")
        return "\n".join(out)
