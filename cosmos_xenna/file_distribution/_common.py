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

import pathlib

from cosmos_xenna._cosmos_xenna.file_distribution import common  # type: ignore


def resolve_path(original_path: pathlib.Path, node_id: str, is_test: bool) -> pathlib.Path:
    return common.resolve_path(original_path, node_id, is_test)


def get_cache_path_for_file(file_path: pathlib.Path) -> pathlib.Path:
    return common.get_cache_path_for_file(file_path)


def get_cache_path_for_directory(directory_path: pathlib.Path) -> pathlib.Path:
    return common.get_cache_path_for_directory(directory_path)
