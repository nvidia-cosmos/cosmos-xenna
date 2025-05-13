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
