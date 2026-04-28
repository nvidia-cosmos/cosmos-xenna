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


import hashlib
import json
import pprint
from typing import Any

import attrs


def hash_attrs_object(obj: Any) -> str:
    json_string = json.dumps(attrs.asdict(obj), sort_keys=True).encode("utf-8")
    hash_object = hashlib.sha256(json_string)
    return hash_object.hexdigest()


def format_attrs_object(obj: Any) -> str:
    return pprint.pformat(attrs.asdict(obj), indent=2)


def format_attrs_list(obj: list[Any]) -> str:
    return "\n".join([format_attrs_object(x) for x in obj])


def validate_positive_int(_: Any, attribute: "attrs.Attribute[int]", value: int) -> None:
    """Reject integer values < 1 (require >= 1)."""
    if value < 1:
        msg = f"{attribute.name} must be >= 1, got {value}"
        raise ValueError(msg)


def validate_optional_positive_int(_: Any, attribute: "attrs.Attribute[int | None]", value: int | None) -> None:
    """Reject integer values < 1 (require >= 1 or None)."""
    if value is None:
        return
    if value < 1:
        msg = f"{attribute.name} must be >= 1 (or None), got {value}"
        raise ValueError(msg)
