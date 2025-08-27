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

from __future__ import annotations

import os
import uuid


def get_gpu_ids_from_cuda_env_vars() -> list[int | uuid.UUID]:
    """Get the number of GPUs from the CUDA_VISIBLE_DEVICES environment variable."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    out: list[int | uuid.UUID] = []
    if cuda_visible_devices:
        for id in cuda_visible_devices.split(","):
            if id.strip():
                try:
                    out.append(uuid.UUID(id.lower().strip("gpu-")))
                except ValueError:
                    out.append(int(id))
    # If the variable is not set or empty, it implies all GPUs are visible,
    # but this function specifically gets the count *from* the env var.
    # Return 0 or consider raising an error/warning if needed.
    return out


def get_num_gpus() -> int:
    # TODO: This is a hack. We should use some other method to get the number of GPUs.
    return len(get_gpu_ids_from_cuda_env_vars())
