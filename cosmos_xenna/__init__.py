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


import os

# Ray otherwise wraps every worker launch in another ``uv run`` when the driver
# is running under uv, which can cause worker startup to hang. Users can opt
# back into Ray's uv runtime-environment integration by setting this before
# importing cosmos_xenna.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
