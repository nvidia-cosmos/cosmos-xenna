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


import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_text_mode_leaves_record_extra_empty():
    # In text mode the module must NOT bind identity fields or stamp seq, so other
    # cosmos-xenna consumers keep the pre-structured-logging behavior (empty extra).
    code = (
        "import json, sys\n"
        "from cosmos_xenna.utils import python_log as L\n"
        "from loguru import logger\n"
        "seen = {}\n"
        "def _cap(msg):\n"
        "    seen.clear()\n"
        "    seen.update(msg.record['extra'])\n"
        "logger.add(_cap, level='TRACE', format='{message}')\n"
        "L.info('X')\n"
        "print('EXTRAKEYS' + json.dumps(sorted(seen)), file=sys.stderr)\n"
    )
    err = _run_code_and_capture_stderr(code, {"PYTHON_LOG": "info"}, [_pkg_root()])
    markers = [line for line in err.splitlines() if line.startswith("EXTRAKEYS")]
    assert markers, err
    assert json.loads(markers[0][len("EXTRAKEYS") :]) == []


# ---------- Structured (JSON) logging tests ----------

# In json mode python_log installs its own fallback JSON handler on the stdlib root
# logger, so (unlike the earlier design) the tests do NOT need to install a Ray-like
# root handler themselves -- doing so would double-emit. Kept as an empty string so
# the test bodies below read the same as when a preamble was required.
_JSON_ROOT_PREAMBLE = ""


def _json_lines(err: str) -> list[dict]:
    return [json.loads(line) for line in err.splitlines() if line.strip().startswith("{")]


def test_json_mode_emits_valid_json_with_identity_fields():
    code = (
        _JSON_ROOT_PREAMBLE + "from cosmos_xenna.utils import python_log as L\nL.info('JSONINFO')\nL.info('SECOND')\n"
    )
    err = _run_code_and_capture_stderr(
        code,
        {
            "PYTHON_LOG": "info",
            "PYTHON_LOG_FORMAT": "json",
            "POD_NAME": "cosmos-curator-3",
            "CURATOR_RUN_ID": "run-xyz",
        },
        [_pkg_root()],
    )
    objs = _json_lines(err)
    assert [o["message"] for o in objs] == ["JSONINFO", "SECOND"]
    first = objs[0]
    assert first["pod"] == "cosmos-curator-3"
    assert first["replica"] == "3"
    assert first["run_id"] == "run-xyz"
    assert isinstance(first["pid"], int)
    # seq is gap-free and monotonic across emitted records.
    seqs = [o["seq"] for o in objs]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_json_mode_off_still_silences():
    code = _JSON_ROOT_PREAMBLE + "from cosmos_xenna.utils import python_log as L\nL.critical('NOPE')\n"
    err = _run_code_and_capture_stderr(
        code,
        {"PYTHON_LOG": "off", "PYTHON_LOG_FORMAT": "json"},
        [_pkg_root()],
    )
    assert _json_lines(err) == []


def test_json_mode_pod_falls_back_to_slurm_nodename():
    # Off k8s POD_NAME is absent; pod must fall back to SLURMD_NODENAME so SLURM nodes
    # stay distinguishable in pre-Ray/worker logs (empty POD_NAME is treated as unset).
    # replica stays empty off-k8s: it is a k8s StatefulSet ordinal sourced only from
    # POD_NAME, so a numeric-suffixed node name must not be misread as a replica.
    code = _JSON_ROOT_PREAMBLE + "from cosmos_xenna.utils import python_log as L\nL.info('NOPOD')\n"
    err = _run_code_and_capture_stderr(
        code,
        {
            "PYTHON_LOG": "info",
            "PYTHON_LOG_FORMAT": "json",
            "POD_NAME": "",
            "SLURMD_NODENAME": "slurm-node-4",
            "CURATOR_RUN_ID": "",
        },
        [_pkg_root()],
    )
    obj = _json_lines(err)[0]
    assert obj["pod"] == "slurm-node-4"
    assert obj["replica"] == ""
    assert obj["run_id"] == ""


def test_json_mode_preserves_patched_tag():
    # make_tagged_logger() prefixes messages via logger.patch(); assert the bridge
    # preserves that prefix (and identity extras) in JSON mode.
    code = (
        _JSON_ROOT_PREAMBLE
        + "from cosmos_xenna.utils import python_log as L\n"
        + "from loguru import logger\n"
        + "def _pre(record):\n"
        + "    record['message'] = '[TAG] ' + record['message']\n"
        + "tagged = logger.patch(_pre)\n"
        + "tagged.warning('HELLO')\n"
    )
    err = _run_code_and_capture_stderr(
        code,
        {"PYTHON_LOG": "info", "PYTHON_LOG_FORMAT": "json", "POD_NAME": "pod-7"},
        [_pkg_root()],
    )
    obj = _json_lines(err)[0]
    assert obj["message"] == "[TAG] HELLO"
    assert obj["pod"] == "pod-7"
    assert obj["replica"] == "7"
    assert isinstance(obj["seq"], int)


def test_json_mode_emits_without_external_root_handler():
    # Regression: before the fallback handler existed, json-mode records were dropped
    # (or emitted as plain text via stdlib lastResort) when nothing had configured the
    # root logger. Importing python_log alone must now yield JSON with identity fields.
    code = "from cosmos_xenna.utils import python_log as L\nL.info('NO_EXTERNAL_HANDLER')\n"
    err = _run_code_and_capture_stderr(
        code,
        {"PYTHON_LOG": "info", "PYTHON_LOG_FORMAT": "json", "POD_NAME": "pod-9"},
        [_pkg_root()],
    )
    obj = _json_lines(err)[0]
    assert obj["message"] == "NO_EXTERNAL_HANDLER"
    assert obj["pod"] == "pod-9"
    assert obj["replica"] == "9"
    assert isinstance(obj["seq"], int)


def test_json_mode_defers_to_external_root_handler():
    # Regression: a Ray worker configures loguru (adding the fallback) while Ray has
    # ALSO installed its own root handler. The fallback must defer at emit time so the
    # record is not written twice. Simulated here with a non-JSON external handler.
    code = (
        "import logging, sys\n"
        "_h = logging.StreamHandler(sys.stderr)\n"
        "_h.setFormatter(logging.Formatter('EXT:%(message)s'))\n"
        "logging.getLogger().addHandler(_h)\n"
        "from cosmos_xenna.utils import python_log as L\n"
        "L.info('DEFERTEST')\n"
    )
    err = _run_code_and_capture_stderr(
        code,
        {"PYTHON_LOG": "info", "PYTHON_LOG_FORMAT": "json", "POD_NAME": "pod-1"},
        [_pkg_root()],
    )
    # Rendered exactly once, by the external handler; the fallback emits nothing.
    assert err.count("DEFERTEST") == 1
    assert "EXT:DEFERTEST" in err
    assert _json_lines(err) == []


def test_json_mode_emits_when_external_handler_rejects_record():
    # Regression: the fallback previously deferred whenever ANY external root handler
    # existed, silently dropping records the external handler would not emit. With a
    # WARNING-level external handler the INFO record must still be emitted as JSON by
    # the fallback, while the WARNING record is emitted exactly once by the external
    # handler (fallback defers only for records another handler would actually emit).
    code = (
        "import logging, sys\n"
        "_h = logging.StreamHandler(sys.stderr)\n"
        "_h.setLevel(logging.WARNING)\n"
        "_h.setFormatter(logging.Formatter('EXT:%(message)s'))\n"
        "logging.getLogger().addHandler(_h)\n"
        "from cosmos_xenna.utils import python_log as L\n"
        "L.info('SHOULD_BE_JSON_FALLBACK')\n"
        "L.warning('SHOULD_BE_EXTERNAL')\n"
    )
    err = _run_code_and_capture_stderr(
        code,
        {"PYTHON_LOG": "info", "PYTHON_LOG_FORMAT": "json", "POD_NAME": "pod-2"},
        [_pkg_root()],
    )
    objs = _json_lines(err)
    assert [o["message"] for o in objs] == ["SHOULD_BE_JSON_FALLBACK"]
    assert objs[0]["pod"] == "pod-2"
    assert "EXT:SHOULD_BE_JSON_FALLBACK" not in err
    assert err.count("EXT:SHOULD_BE_EXTERNAL") == 1
    assert err.count("SHOULD_BE_EXTERNAL") == 1


@pytest.mark.parametrize(
    ("python_log", "expected"),
    [
        ("", "INFO"),
        ("warning", "WARNING"),
        ("info,my.module=debug", "DEBUG"),
        ("warning,my.module=trace", "DEBUG"),
        ("trace,my.module=off", "DEBUG"),
        ("off,my.module=warning", "WARNING"),
        ("off,my.module=off,other.*=off", "CRITICAL"),
    ],
)
def test_ray_log_level_allows_most_verbose_enabled_threshold(monkeypatch, python_log, expected):
    from cosmos_xenna.utils import python_log as L

    monkeypatch.delenv("PYTHON_LOG_RAY_LEVEL", raising=False)
    monkeypatch.setenv("PYTHON_LOG", python_log)

    assert L.ray_json_log_level() == expected


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (" error ", "ERROR"),
        ("warn", "WARNING"),
        ("fatal", "CRITICAL"),
        ("trace", "DEBUG"),
        ("off", "CRITICAL"),
    ],
)
def test_ray_log_level_normalizes_valid_override(monkeypatch, override, expected):
    from cosmos_xenna.utils import python_log as L

    monkeypatch.setenv("PYTHON_LOG", "warning,my.module=trace")
    monkeypatch.setenv("PYTHON_LOG_RAY_LEVEL", override)

    assert L.ray_json_log_level() == expected


@pytest.mark.parametrize("override", ["", " \t ", "verbose"])
def test_ray_log_level_ignores_blank_or_invalid_override(monkeypatch, override):
    from cosmos_xenna.utils import python_log as L

    monkeypatch.setenv("PYTHON_LOG", "error,my.module=debug")
    monkeypatch.setenv("PYTHON_LOG_RAY_LEVEL", override)

    assert L.ray_json_log_level() == "DEBUG"


def test_ray_handoff_removes_fallback_handler(monkeypatch):
    # When Ray's LoggingConfig will own the root logger, apply_ray_logging_config must
    # remove the fallback handler (so records are not emitted twice) and set the
    # logging_config kwarg while leaving log_to_driver at the caller's value (True) so
    # worker logs stay visible on the driver, matching text-mode behavior.
    import logging
    import types

    from cosmos_xenna.utils import python_log as L

    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.delenv("PYTHON_LOG_TO_DRIVER", raising=False)
    monkeypatch.delenv("RAY_BACKEND_LOG_JSON", raising=False)
    try:
        L.ensure_configured(force=True)
        root = logging.getLogger()
        assert any(isinstance(h, L._FallbackJsonHandler) for h in root.handlers)

        fake_ray = types.ModuleType("ray")

        class _LoggingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_ray.LoggingConfig = _LoggingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ray", fake_ray)

        kwargs: dict = {}
        log_to_driver = L.apply_ray_logging_config(kwargs, log_to_driver=True)

        assert log_to_driver is True
        assert isinstance(kwargs["logging_config"], _LoggingConfig)
        assert kwargs["logging_config"].kwargs["encoding"] == "JSON"
        # Logger name is surfaced on every Ray record for schema parity.
        assert kwargs["logging_config"].kwargs["additional_log_standard_attrs"] == ["name"]
        # Ray's C++ backend log format is left to the operator; we must not set it.
        assert "RAY_BACKEND_LOG_JSON" not in os.environ
        assert not any(isinstance(h, L._FallbackJsonHandler) for h in root.handlers)
    finally:
        monkeypatch.setenv("PYTHON_LOG_FORMAT", "text")
        L.ensure_configured(force=True)


def test_ray_handoff_leaves_backend_log_json_untouched(monkeypatch):
    # RAY_BACKEND_LOG_JSON is operator-controlled: apply_ray_logging_config must never
    # set, clear, or override whatever value (or absence) the operator provided.
    import types

    from cosmos_xenna.utils import python_log as L

    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.setenv("RAY_BACKEND_LOG_JSON", "0")
    try:
        L.ensure_configured(force=True)

        fake_ray = types.ModuleType("ray")

        class _LoggingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_ray.LoggingConfig = _LoggingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ray", fake_ray)

        L.apply_ray_logging_config({}, log_to_driver=True)
        assert os.environ["RAY_BACKEND_LOG_JSON"] == "0"
    finally:
        monkeypatch.setenv("PYTHON_LOG_FORMAT", "text")
        L.ensure_configured(force=True)


def test_ray_handoff_tolerates_old_logging_config(monkeypatch):
    # Older Ray without additional_log_standard_attrs must still get a JSON config.
    import types

    from cosmos_xenna.utils import python_log as L

    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.delenv("PYTHON_LOG_TO_DRIVER", raising=False)
    try:
        L.ensure_configured(force=True)

        fake_ray = types.ModuleType("ray")

        class _OldLoggingConfig:
            def __init__(self, *, encoding, log_level):
                self.kwargs = {"encoding": encoding, "log_level": log_level}

        fake_ray.LoggingConfig = _OldLoggingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ray", fake_ray)

        kwargs: dict = {}
        log_to_driver = L.apply_ray_logging_config(kwargs, log_to_driver=True)

        assert log_to_driver is True
        assert isinstance(kwargs["logging_config"], _OldLoggingConfig)
        assert kwargs["logging_config"].kwargs["encoding"] == "JSON"
        assert "additional_log_standard_attrs" not in kwargs["logging_config"].kwargs
    finally:
        monkeypatch.setenv("PYTHON_LOG_FORMAT", "text")
        L.ensure_configured(force=True)


def test_ray_handoff_respects_log_to_driver_override(monkeypatch):
    # PYTHON_LOG_TO_DRIVER=false opts a JSON-mode deployment out of driver forwarding so
    # the driver stream stays pure JSON (worker logs shipped from per-node Ray files).
    import types

    from cosmos_xenna.utils import python_log as L

    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.setenv("PYTHON_LOG_TO_DRIVER", "false")
    try:
        L.ensure_configured(force=True)

        fake_ray = types.ModuleType("ray")

        class _LoggingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_ray.LoggingConfig = _LoggingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ray", fake_ray)

        log_to_driver = L.apply_ray_logging_config({}, log_to_driver=True)
        assert log_to_driver is False
    finally:
        monkeypatch.setenv("PYTHON_LOG_FORMAT", "text")
        L.ensure_configured(force=True)
