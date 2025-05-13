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
