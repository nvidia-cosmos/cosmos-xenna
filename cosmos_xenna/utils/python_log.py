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

Structured (JSON) logging: PYTHON_LOG_FORMAT
  - "text" (default) keeps the human-readable stderr sink (unchanged behavior).
  - "json" (case-insensitive) replaces the stderr sink with a loguru->stdlib
    bridge so Ray's structured logging can pick up xenna/curator logs. Every
    record is enriched with static per-process identity fields (pod, replica,
    pid, run_id) plus a gap-free per-process ``seq`` tiebreaker.
  - In json mode a fallback JSON handler is installed on the stdlib root logger so
    records still render as JSON before ``ray.init`` configures the root logger (or
    when Ray is not used at all). It is removed at the Ray hand-off to avoid dupes.
"""

import itertools
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Sequence

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

# ---------- Structured (JSON) logging ----------

# Per-process, gap-free counter stamped onto every EMITTED record (see _make_filter).
# next() on itertools.count is atomic under the GIL, so it is safe across threads.
_SEQ_COUNTER = itertools.count()

# loguru defines TRACE=5 and SUCCESS=25 which stdlib logging does not know about.
_TRACE_LEVEL_NO = 5
_SUCCESS_LEVEL_NO = 25

# stdlib LogRecord attribute names we must not clobber when flattening loguru
# extras onto a forwarded record. Derived from a blank record so it tracks the
# running Python version, plus the two names makeRecord() reserves.
_RESERVED_LOGRECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def _wants_json(value: Optional[str] = None) -> bool:
    """Return True when PYTHON_LOG_FORMAT selects JSON output (case-insensitive)."""
    raw = os.getenv("PYTHON_LOG_FORMAT", "") if value is None else value
    return raw.strip().lower() == "json"


def _replica_from_pod_name(pod: str) -> str:
    """Return the trailing ``-N`` ordinal of a pod name, or "" when absent."""
    _, sep, tail = pod.rpartition("-")
    return tail if sep and tail.isdigit() else ""


def _node_identity() -> str:
    """Best-effort identifier for the emitting node/instance across platforms.

    Prefers the k8s ``POD_NAME`` (downward API), then SLURM's ``SLURMD_NODENAME``,
    then the container/host name. This keeps the ``pod`` field populated and
    distinguishable off-k8s (SLURM/NVCF/local), where ``POD_NAME`` is absent.
    """
    return os.getenv("POD_NAME") or os.getenv("SLURMD_NODENAME") or socket.gethostname() or ""


def _identity_extra() -> dict[str, Any]:
    """Static per-process identity fields bound to every record via logger.configure()."""
    pod = _node_identity()
    return {
        "pod": pod,
        # replica = k8s StatefulSet ordinal; sourced strictly from POD_NAME so it is
        # populated only on k8s/NVCF and stays "" off-k8s (SLURM/local) instead of
        # misreading a node name like pool0-0218 as a replica ordinal.
        "replica": _replica_from_pod_name(os.getenv("POD_NAME", "")),
        "pid": os.getpid(),
        "run_id": os.getenv("CURATOR_RUN_ID", ""),
        "seq": 0,  # placeholder; replaced per emitted record by the sink filter
    }


def _install_stdlib_level_names() -> None:
    """Teach stdlib logging loguru's custom level names so JSON output stays readable."""
    logging.addLevelName(_TRACE_LEVEL_NO, "TRACE")
    logging.addLevelName(_SUCCESS_LEVEL_NO, "SUCCESS")


class _LoguruToStdlibBridge(logging.Handler):
    """Loguru sink that re-emits records into the stdlib logging system.

    In JSON mode we do not want loguru to render its own serialized line; instead
    every loguru record is forwarded to ``logging.getLogger(record.name)`` so that
    Ray's root JSON handler renders it, enriched with Ray's job/worker context.

    Loguru's built-in ``StandardSink`` invokes this handler with a stdlib
    ``LogRecord`` whose ``record.extra`` holds the loguru ``extra`` mapping
    (pod/replica/pid/run_id/seq plus anything bound via ``logger.bind()``) and
    whose ``exc_info`` preserves any exception. We flatten those extras onto the
    record (guarding reserved attribute names) so they surface as structured
    fields, then route through the named logger so the record propagates to
    whatever handler Ray installed on the root logger.
    """

    def emit(self, record: logging.LogRecord) -> None:
        extra = record.__dict__.pop("extra", None)
        if isinstance(extra, dict):
            # Flatten loguru extras to top-level attributes (the conventional stdlib
            # `extra=` shape) so downstream JSON formatters surface them as fields.
            for key, value in extra.items():
                attr = key if key not in _RESERVED_LOGRECORD_ATTRS else f"extra_{key}"
                record.__dict__.setdefault(attr, value)
        logging.getLogger(record.name).handle(record)


class _FlatJsonFormatter(logging.Formatter):
    """Render a stdlib ``LogRecord`` as a single flat JSON object.

    Used by the fallback root handler so that structured logs still emit as JSON
    when no external handler (e.g. Ray's) has configured the root logger. The shape
    intentionally mirrors Ray's structured-logging output (flat, top-level fields)
    rather than the nested loguru-``serialize`` shape, so a given process emits a
    consistent schema before and after ``ray.init`` takes over the root logger. Any
    non-standard record attributes (loguru extras flattened by the bridge, plus
    anything attached via stdlib ``extra=``) are surfaced as top-level fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "levelname": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "process": record.process,
            "asctime": self.formatTime(record),
            "timestamp_ns": int(record.created * 1_000_000_000),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOGRECORD_ATTRS and key not in data:
                data[key] = value
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


class _FallbackJsonHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Last-resort stderr handler installed on the root logger in JSON mode.

    It guarantees that bridged loguru records render as JSON even before ``ray.init``
    configures the root logger (without it, stdlib's ``lastResort`` would drop INFO
    and emit WARNING+ as unstructured text). It is a distinct subclass so it can be
    identified and removed at the Ray hand-off.

    ``emit`` defers to another root handler only when that handler would actually
    emit the record -- i.e. it is not another ``_FallbackJsonHandler``, the record's
    level meets the handler's own threshold (``NOTSET`` accepts everything), and no
    attached filter rejects it. This keeps the fallback a genuine last resort: on
    Ray workers where loguru is configured while Ray has already installed its own
    root handler, Ray owns the record so the fallback stays silent; but when an
    external handler is present yet would drop the record (e.g. its level is above
    ``record.levelno``), the fallback still writes JSON so records are never
    silently dropped.
    """

    def emit(self, record: logging.LogRecord) -> None:
        root = logging.getLogger()
        for handler in root.handlers:
            if handler is self or isinstance(handler, _FallbackJsonHandler):
                continue
            if record.levelno < handler.level:
                continue
            if not handler.filter(record):
                continue
            return
        super().emit(record)


def _remove_fallback_root_handler() -> None:
    """Remove any previously installed fallback JSON handler from the root logger."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, _FallbackJsonHandler):
            root.removeHandler(handler)


def _install_fallback_root_handler() -> None:
    """Install (idempotently) the fallback JSON handler on the stdlib root logger.

    The handler level is left at NOTSET so it emits every record the loguru sink
    filter already decided to forward; we deliberately do not change the root
    logger's level to avoid side effects on unrelated stdlib loggers.
    """
    _remove_fallback_root_handler()
    handler = _FallbackJsonHandler(sys.stderr)
    handler.setFormatter(_FlatJsonFormatter())
    logging.getLogger().addHandler(handler)


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


def _make_filter(config: _LogConfig, *, stamp_seq: bool) -> Callable[[Mapping[str, Any]], bool]:
    """
    Build and return a filter callable suitable for a Loguru sink.

    The filter determines, for each record, whether it should be emitted based on
    the most specific matching rule (longest matching pattern) or the default
    threshold when no pattern matches.

    When ``stamp_seq`` is True (JSON mode only), the monotonic ``seq`` tiebreaker is
    stamped onto passing records. In text mode it is left off so the record ``extra``
    stays empty, keeping non-JSON cosmos-xenna consumers byte-for-byte unchanged.
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
        if record["level"].no < thr:
            return False
        if stamp_seq:
            # Stamp the monotonic seq only on records that actually pass the level
            # filter, so it stays gap-free and acts as a timestamp-collision tiebreaker.
            record["extra"]["seq"] = next(_SEQ_COUNTER)
        return True

    return _filter


def _configure_from_env() -> None:
    """
    Configure Loguru according to the PYTHON_LOG environment variable.

    Steps:
      - Remove any existing sinks to avoid duplicate emission when reloading.
      - Parse PYTHON_LOG into a default level and pattern rules.
      - In text mode (default): add the human-readable stderr sink (unchanged).
      - In json mode (PYTHON_LOG_FORMAT=json): bind static per-process identity fields
        (pod/replica/pid/run_id/seq), add the loguru->stdlib bridge sink so Ray's
        structured logging renders the records as JSON, and install a fallback JSON
        handler on the root logger so records still emit as JSON before Ray has
        configured the root logger (or when Ray is absent entirely). The fallback is
        removed at the Ray hand-off (see apply_ray_logging_config) to avoid dupes.
    """
    env = os.getenv("PYTHON_LOG", "").strip()
    json_mode = _wants_json()
    _logger.remove()
    # Drop any fallback handler from a prior configuration so re-config is clean and
    # text mode never leaves a stray JSON handler behind.
    _remove_fallback_root_handler()
    # Only bind identity extras in JSON mode. In text mode reset to an empty extra so
    # behavior matches the pre-structured-logging default for other xenna consumers.
    # Bind via extra (not patcher) so logger.patch() chains such as make_tagged_logger()
    # keep working while these fields ride on every record.
    _logger.configure(extra=_identity_extra() if json_mode else {})
    config = _parse_env(env)
    filt = _make_filter(config, stamp_seq=json_mode)
    if json_mode:
        _install_stdlib_level_names()
        # format="{message}" keeps the raw (tag-prefixed) message so the downstream
        # JSON encoder wraps it cleanly instead of a pre-rendered loguru line.
        _logger.add(
            _LoguruToStdlibBridge(),
            level="TRACE",
            filter=filt,
            format="{message}",
            backtrace=False,
            diagnose=False,
        )
        _install_fallback_root_handler()
    else:
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


# ---------- Ray structured-logging integration ----------


def wants_json_logs() -> bool:
    """Return True when PYTHON_LOG_FORMAT selects structured (JSON) logging."""
    return _wants_json()


def ray_json_log_level() -> str:
    """Return the stdlib level name Ray should use for its JSON root logger.

    A valid ``PYTHON_LOG_RAY_LEVEL`` overrides ``PYTHON_LOG`` using the same aliases
    as other level directives. Blank or unrecognized overrides are ignored, matching
    ``PYTHON_LOG`` parsing. Otherwise, the most verbose enabled default or per-module
    level is selected. Ray/stdlib have no TRACE or OFF level, so those map to DEBUG
    and CRITICAL respectively. The loguru sink filter remains the authoritative
    per-module gate; this only prevents Ray's root threshold from discarding a record
    that already passed that filter.

    PYTHON_LOG input     Canonical level     Ray JSON level
    -------------------------------------------------------
    trace                TRACE               DEBUG
    debug                DEBUG               DEBUG
    info / unset         INFO                INFO
    warn / warning       WARNING             WARNING
    error                ERROR               ERROR
    critical / fatal     CRITICAL            CRITICAL
    off                  OFF                 CRITICAL*

    * Loguru filters all matching records; CRITICAL is only the safe Ray threshold.

    Derived-level examples:

    PYTHON_LOG                         Derived Ray level
    ----------------------------------------------------
    info                               INFO
    info,my.module=debug               DEBUG
    warning,my.module=trace            DEBUG
    off,my.module=warning              WARNING
    off,my.module=off                  CRITICAL

    * PYTHON_LOG_RAY_LEVEL overrides the derived level. Blank or invalid overrides are ignored.
    """
    override = os.getenv("PYTHON_LOG_RAY_LEVEL")
    selected_level_name = _normalize_level(override.strip()) if override else None
    if selected_level_name is None:
        config = _parse_env(os.getenv("PYTHON_LOG", "").strip())
        enabled_level_names = [
            level_name
            for level_name in (config.default_level_name, *(rule.level_name for rule in config.rules))
            if level_name != "OFF"
        ]
        if not enabled_level_names:
            selected_level_name = "OFF"
        else:
            selected_level_name = min(enabled_level_names, key=lambda name: _logger.level(name).no)
    return {"TRACE": "DEBUG", "OFF": "CRITICAL"}.get(selected_level_name, selected_level_name)


def apply_ray_logging_config(ray_init_kwargs: MutableMapping[str, Any], *, log_to_driver: bool) -> bool:
    """Gate Ray structured logging on PYTHON_LOG_FORMAT; return effective log_to_driver.

    Text mode is a no-op. JSON mode sets ``ray.LoggingConfig(encoding="JSON", ...)`` and
    removes the fallback root handler so records are not emitted twice. Older Ray without
    ``LoggingConfig`` logs a warning and leaves the fallback in place.

    ``log_to_driver`` is returned unchanged unless ``PYTHON_LOG_TO_DRIVER`` overrides it:
      - ``true``  -> workers forward logs to the driver (default; matches text mode).
      - ``false`` -> no worker log forwarding; the driver stream stays strict JSON
        (Ray prefixes forwarded worker lines with ``(Actor pid=…)``, which is not valid JSON).

    ``RAY_BACKEND_LOG_JSON`` (Ray's C++ backend log format) is left to the operator.
    """
    if not wants_json_logs():
        return log_to_driver
    import ray  # local import keeps python_log usable without Ray installed

    logging_config_cls = getattr(ray, "LoggingConfig", None)
    if logging_config_cls is None:
        _logger.warning(
            "PYTHON_LOG_FORMAT=json is set but this Ray version has no ray.LoggingConfig; "
            "continuing with the fallback JSON root handler (no Ray log context)."
        )
        return log_to_driver
    # Ray is about to configure the root logger; drop our fallback to avoid duplicates.
    _remove_fallback_root_handler()
    logging_config_kwargs: dict[str, Any] = {"encoding": "JSON", "log_level": ray_json_log_level()}
    # ``additional_log_standard_attrs=["name"]`` surfaces the logger name on every Ray
    # record so Ray output matches the fallback/launcher schema and tolerate older Ray
    # that lacks the parameter.
    try:
        ray_init_kwargs["logging_config"] = logging_config_cls(
            additional_log_standard_attrs=["name"], **logging_config_kwargs
        )
    except TypeError:
        ray_init_kwargs["logging_config"] = logging_config_cls(**logging_config_kwargs)
    override = os.getenv("PYTHON_LOG_TO_DRIVER")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return log_to_driver


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
