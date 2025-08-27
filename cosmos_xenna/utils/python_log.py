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

"""
Loguru configuration that respects a RUST_LOG-like environment variable named
PYTHON_LOG.

This module centralizes Loguru setup so applications can control logging with a
single environment variable and consistent semantics. It also re-exports the most
common logger methods for convenience.

Usage:
  - Import once early in your program (import side-effects will initialize):
      import cosmos_xenna.utils.python_log as python_log
  - Or call ensure_configured() explicitly if you prefer to control init timing.
  - Use the re-exported methods directly:
      python_log.info("hello")

Environment variable: PYTHON_LOG
  - Comma-separated directives.
  - Each directive is either "<level>" (global default) or "<pattern>=<level>".
  - <pattern> supports fnmatch globs (* and ?), matched against the dotted module path
    (e.g. "package.sub.module").
  - Levels: trace, debug, info, warning, error, critical, off.
  - Most-specific matching rule wins (longest matching pattern string).

Examples:
  PYTHON_LOG=info
  PYTHON_LOG=debug,myapp.db=warning
  PYTHON_LOG=myapp.*=trace,sqlalchemy.engine=warning,*=info
  PYTHON_LOG=off  # disables all logs
"""

import os
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, List, Mapping, Optional, Sequence

from loguru import logger as _logger


@dataclass(frozen=True)
class _Rule:
    """A single pattern-level directive from PYTHON_LOG."""

    pattern: str
    level_name: str  # canonical, e.g. "DEBUG" or "OFF"


@dataclass(frozen=True)
class _LogConfig:
    """Parsed configuration from PYTHON_LOG: default level and per-pattern rules."""

    default_level_name: str
    rules: Sequence[_Rule]


@dataclass
class _RuntimeState:
    """Mutable runtime state for this module's initialization."""

    configured: bool = False


# initialize on import
_STATE = _RuntimeState(configured=False)

# ---------- Implementation ----------

_LEVEL_ALIASES = {
    "trace": "TRACE",
    "debug": "DEBUG",
    "info": "INFO",
    "warn": "WARNING",
    "warning": "WARNING",
    "error": "ERROR",
    "critical": "CRITICAL",
    "fatal": "CRITICAL",
    "off": "OFF",
}

_DEFAULT_LEVEL = "INFO"


def _parse_env(env: Optional[str]) -> _LogConfig:
    """
    Parse the PYTHON_LOG value into (default_level, rules).

    Returns:
      - default_level: One of the normalized level names (e.g. "INFO"), used when
        no pattern-specific rule matches. If "OFF", logging is disabled globally.
      - rules: A list of (pattern, level) where level is normalized to a Loguru
        level name (e.g. "DEBUG"). Patterns use fnmatch-style globs matched
        against dotted module paths.
    """
    if not env:
        return _LogConfig(default_level_name=_DEFAULT_LEVEL, rules=())
    default_level: Optional[str] = None
    rules: List[_Rule] = []
    for raw in env.split(","):
        part = raw.strip()
        if not part:
            continue
        if "=" in part:
            pat, lvl = part.split("=", 1)
            lvl_norm = _normalize_level(lvl.strip())
            if lvl_norm:
                rules.append(_Rule(pattern=pat.strip(), level_name=lvl_norm))
        else:
            lvl_norm = _normalize_level(part)
            if lvl_norm:
                default_level = lvl_norm
    return _LogConfig(default_level_name=default_level or _DEFAULT_LEVEL, rules=tuple(rules))


def _normalize_level(lvl: str) -> Optional[str]:
    """Return the canonical Loguru level name for a user-provided alias."""
    return _LEVEL_ALIASES.get(lvl.lower())


def _module_path_from_record(record: Mapping[str, Any]) -> str:
    """
    Derive a best-effort dotted module path from a Loguru record.

    Preference order:
      1) Convert record["file"].path by stripping a sys.path prefix and replacing
         path separators with dots, then removing the file extension.
      2) Fall back to record["module"] (leaf module name).

    This dotted path is used for pattern matching against PYTHON_LOG rules.
    """
    leaf = record["module"]
    try:
        file_path = record["file"].path
    except (KeyError, AttributeError):
        return leaf
    file_path = os.path.normpath(file_path)
    for base in sys.path:
        if not base or not isinstance(base, str):
            continue
        base_norm = os.path.normpath(base)
        if file_path.startswith(base_norm):
            rel = file_path[len(base_norm) :].lstrip(os.sep)
            rel_no_ext, _ = os.path.splitext(rel)
            dotted = rel_no_ext.replace(os.sep, ".")
            return dotted.lstrip(".")
    return leaf


def _make_filter(config: _LogConfig) -> Callable[[Mapping[str, Any]], bool]:
    """
    Build and return a filter callable suitable for a Loguru sink.

    The filter determines, for each record, whether it should be emitted based on
    the most specific matching rule (longest matching pattern) or the default
    threshold when no pattern matches.
    """
    level_no = {name: _logger.level(name).no for name in _LEVEL_ALIASES.values() if name != "OFF"}
    default_no = None if config.default_level_name == "OFF" else level_no[config.default_level_name]
    compiled = [(r.pattern, None if r.level_name == "OFF" else level_no[r.level_name]) for r in config.rules]

    def select_threshold(modpath: str) -> Optional[int]:
        best = (-1, None)
        for pat, no in compiled:
            if fnmatch(modpath, pat):
                plen = len(pat)
                if plen > best[0]:
                    best = (plen, no)
        return best[1] if best[0] >= 0 else default_no

    def _filter(record: Mapping[str, Any]) -> bool:
        mod = _module_path_from_record(record)
        thr = select_threshold(mod)
        if thr is None:
            return False
        return record["level"].no >= thr

    return _filter


def _configure_from_env() -> None:
    """
    Configure Loguru according to the PYTHON_LOG environment variable.

    Steps:
      - Remove any existing sinks to avoid duplicate emission when reloading.
      - Parse PYTHON_LOG into a default level and pattern rules.
      - Build a filter that enforces thresholds per module path.
      - Add a single stderr sink at TRACE; filtering is handled by our filter.
    """
    env = os.getenv("PYTHON_LOG", "").strip()
    _logger.remove()
    config = _parse_env(env)
    filt = _make_filter(config)
    _logger.add(sys.stderr, level="TRACE", filter=filt, backtrace=False, diagnose=False)


def ensure_configured(force: bool = False) -> None:
    """
    Idempotent initialization from PYTHON_LOG (RUST_LOG-like semantics).

    Call this once at startup to configure Loguru sinks/filters according to
    the PYTHON_LOG environment variable. Importing this module also triggers
    configuration automatically, so explicit calls are optional.

    - When force is False (default), repeated calls are no-ops.
    - When force is True, configuration is rebuilt from the current env.
    """
    global _STATE  # noqa: PLW0602
    if _STATE.configured and not force:
        return
    _configure_from_env()
    _STATE.configured = True


def reload_from_env() -> None:
    """
    Re-read PYTHON_LOG and reconfigure sinks/filters.

    This is equivalent to calling ensure_configured(force=True). Useful if the
    environment variable changed during runtime and you want to apply new rules.
    """
    ensure_configured(force=True)


ensure_configured()


# ---------- Re-export the logger methods ----------
trace = _logger.trace
debug = _logger.debug
info = _logger.info
success = _logger.success
warning = _logger.warning
error = _logger.error
critical = _logger.critical
exception = _logger.exception
log = _logger.log
