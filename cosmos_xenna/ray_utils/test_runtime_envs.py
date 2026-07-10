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

"""Tests for RuntimeEnv logging-env forwarding, which is gated on PYTHON_LOG_FORMAT."""

import pytest

from cosmos_xenna.ray_utils.runtime_envs import RuntimeEnv


def _env_vars(runtime_env) -> dict:
    return dict(runtime_env.get("env_vars") or {})


def test_text_mode_forwards_no_logging_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # In the default text mode, no logging vars are forwarded to actors, so non-JSON
    # cosmos-xenna consumers see an unchanged actor environment.
    monkeypatch.delenv("PYTHON_LOG_FORMAT", raising=False)
    monkeypatch.setenv("PYTHON_LOG", "debug")
    monkeypatch.setenv("CURATOR_RUN_ID", "r1")
    monkeypatch.setenv("POD_NAME", "pod-1")
    env = _env_vars(RuntimeEnv(extra_env_vars={"FOO": "bar"}).to_ray_runtime_env())
    assert env == {"FOO": "bar"}


def test_json_mode_forwards_logging_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # In JSON mode the toggle + run id are forwarded so workers log identically.
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.setenv("PYTHON_LOG", "debug")
    monkeypatch.setenv("CURATOR_RUN_ID", "r1")
    env = _env_vars(RuntimeEnv(extra_env_vars={"FOO": "bar"}).to_ray_runtime_env())
    assert env["FOO"] == "bar"
    assert env["PYTHON_LOG"] == "debug"
    assert env["PYTHON_LOG_FORMAT"] == "json"
    assert env["CURATOR_RUN_ID"] == "r1"


def test_json_mode_does_not_forward_pod_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # POD_NAME is per-pod identity supplied by each pod (k8s downward API); forwarding
    # the driver's value would misattribute worker logs on multi-node clusters.
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.setenv("POD_NAME", "driver-pod-0")
    env = _env_vars(RuntimeEnv().to_ray_runtime_env())
    assert "POD_NAME" not in env


def test_json_mode_via_extra_env_vars_triggers_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    # The mode is resolved from the effective value the worker sees: setting
    # PYTHON_LOG_FORMAT via extra_env_vars alone (absent from the process env) still
    # forwards the other logging vars from the environment.
    monkeypatch.delenv("PYTHON_LOG_FORMAT", raising=False)
    monkeypatch.setenv("PYTHON_LOG", "debug")
    monkeypatch.setenv("CURATOR_RUN_ID", "r1")
    env = _env_vars(RuntimeEnv(extra_env_vars={"PYTHON_LOG_FORMAT": "json"}).to_ray_runtime_env())
    assert env["PYTHON_LOG_FORMAT"] == "json"
    assert env["PYTHON_LOG"] == "debug"
    assert env["CURATOR_RUN_ID"] == "r1"


def test_json_mode_does_not_override_explicit_extra_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # A value a stage set explicitly in extra_env_vars must win over the forwarded one.
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.setenv("PYTHON_LOG", "debug")
    env = _env_vars(RuntimeEnv(extra_env_vars={"PYTHON_LOG": "trace"}).to_ray_runtime_env())
    assert env["PYTHON_LOG"] == "trace"


def test_json_mode_skips_unset_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only vars actually present in the environment are forwarded.
    monkeypatch.setenv("PYTHON_LOG_FORMAT", "json")
    monkeypatch.delenv("PYTHON_LOG", raising=False)
    monkeypatch.delenv("CURATOR_RUN_ID", raising=False)
    env = _env_vars(RuntimeEnv().to_ray_runtime_env())
    assert env == {"PYTHON_LOG_FORMAT": "json"}
