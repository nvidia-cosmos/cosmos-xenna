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

import os
import subprocess
import sys
from pathlib import Path


def _pkg_root() -> str:
    # Path to directory that contains the top-level package 'cosmos_xenna'
    # test file: .../packages/cosmos-xenna/cosmos_xenna/utils/test_python_log.py
    # package root needed on PYTHONPATH: .../packages/cosmos-xenna
    return str(Path(__file__).resolve().parents[2])


def _filter_nonempty(items):
    return [x for x in items if x]


def _run_code_and_capture_stderr(pycode: str, env_overlay: dict, extra_paths: list[str]) -> str:
    env = os.environ.copy()
    env.update(env_overlay)
    pythonpath_parts = [*_filter_nonempty(env.get("PYTHONPATH", "").split(os.pathsep)), *_filter_nonempty(extra_paths)]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    proc = subprocess.run(
        [sys.executable, "-c", pycode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    return proc.stderr


def _write(tmp_path: Path, rel: str, content: str) -> Path:
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return dest


def test_subprocess_default_info_allows_info_and_above(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/mod.py",
        (
            "from cosmos_xenna.utils import python_log as L\n\n"
            "def run():\n"
            "    L.debug('DBG')\n"
            "    L.info('INF')\n"
            "    L.error('ERR')\n"
        ),
    )

    code = "import pkg.mod as m; m.run()"
    err = _run_code_and_capture_stderr(code, {"PYTHON_LOG": "info"}, [str(tmp_path), _pkg_root()])
    assert "INF" in err
    assert "ERR" in err
    assert "DBG" not in err


def test_subprocess_off_disables_all(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/mod.py",
        ("from cosmos_xenna.utils import python_log as L\n\ndef run():\n    L.critical('CRIT')\n"),
    )

    code = "import pkg.mod as m; m.run()"
    err = _run_code_and_capture_stderr(code, {"PYTHON_LOG": "off"}, [str(tmp_path), _pkg_root()])
    assert err.strip() == ""


def test_subprocess_specific_rule_overrides_default(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/db.py",
        (
            "from cosmos_xenna.utils import python_log as L\n\n"
            "def run():\n"
            "    L.info('DB_INF')\n"
            "    L.warning('DB_WARN')\n"
        ),
    )

    code = "import pkg.db as m; m.run()"
    err = _run_code_and_capture_stderr(code, {"PYTHON_LOG": "info,pkg.db=warning"}, [str(tmp_path), _pkg_root()])
    assert "DB_WARN" in err
    assert "DB_INF" not in err


def test_subprocess_most_specific_pattern_wins(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/api/__init__.py", "")
    _write(tmp_path, "pkg/api/v1/__init__.py", "")
    _write(
        tmp_path,
        "pkg/api/v1/users.py",
        (
            "from cosmos_xenna.utils import python_log as L\n\n"
            "def run():\n"
            "    L.warning('USR_WARN')\n"
            "    L.error('USR_ERR')\n"
        ),
    )
    _write(
        tmp_path,
        "pkg/other.py",
        ("from cosmos_xenna.utils import python_log as L\n\ndef run():\n    L.debug('OTH_DBG')\n"),
    )

    code_users = "import pkg.api.v1.users as m; m.run()"
    err_users = _run_code_and_capture_stderr(
        code_users,
        {"PYTHON_LOG": "pkg.*=debug,pkg.api.v1.users=error"},
        [str(tmp_path), _pkg_root()],
    )
    assert "USR_ERR" in err_users
    assert "USR_WARN" not in err_users

    code_other = "import pkg.other as m; m.run()"
    err_other = _run_code_and_capture_stderr(
        code_other,
        {"PYTHON_LOG": "pkg.*=debug,pkg.api.v1.users=error"},
        [str(tmp_path), _pkg_root()],
    )
    assert "OTH_DBG" in err_other
