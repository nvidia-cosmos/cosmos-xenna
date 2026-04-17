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

"""Helpers for detecting whether code is running in CI.

The CI workflow sets ``IS_RUNNING_IN_CICD=1`` on jobs that execute tests. Tests
can use :func:`is_running_in_cicd` to scale down resource requirements so they
fit on small hosted runners while keeping more realistic values when run
locally by developers.
"""

import os


def is_running_in_cicd() -> bool:
    """Return ``True`` if the ``IS_RUNNING_IN_CICD`` env var is set to a truthy value."""
    return os.environ.get("IS_RUNNING_IN_CICD", "").strip() not in ("", "0", "false", "False")
