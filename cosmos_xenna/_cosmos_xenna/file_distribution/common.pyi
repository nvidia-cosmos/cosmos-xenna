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

from pathlib import Path
from uuid import UUID

def resolve_path(original_path: Path, node_id: str, is_test: bool) -> Path: ...
def get_temp_chunk_path(chunk_id: UUID, node_id: str, is_test: bool) -> Path: ...
def get_cache_path_for_file(file_path: Path) -> Path: ...
def get_cache_path_for_directory(file_path: Path) -> Path: ...
