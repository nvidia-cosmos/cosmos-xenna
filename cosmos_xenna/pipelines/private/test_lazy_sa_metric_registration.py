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


"""Pin lazy registration of saturation-aware Prometheus metrics.

Importing ``scheduling_py`` (or ``streaming``) must NOT load the
saturation-aware modules, which would register their Histograms /
Gauges as a side effect even for fragmentation-based pipelines that
never observe them.
"""

import json
import subprocess
import sys
import textwrap

import pytest

# Modules whose import would register saturation-aware Prometheus metric
# series. Each constructs its own Histogram/Gauge at module load time,
# so absence from ``sys.modules`` proves no metric was registered.
#
# ``pressure`` is included because it is pulled in transitively from
# ``SaturationAwareStageConfig.__attrs_post_init__`` (needs the
# ``BACKLOG_CAP`` constant for cross-field validation). Constructing a
# default ``StreamingSpecificSpec()`` must NOT trigger this chain on the
# fragmentation-based path; the field defaults to ``None`` and is
# materialized lazily via ``materialized_saturation_aware()`` in the
# saturation-aware branch only.
_SA_METRIC_MODULES = (
    "cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware",
    "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring",
    "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.heterogeneity",
    "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.metrics",
    "cosmos_xenna.pipelines.private.scheduling_py.cluster.memory_pressure",
    "cosmos_xenna.pipelines.private.scheduling_py.lifecycle.loop_watchdog",
    "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure",
)


def _run_in_subprocess(script: str) -> str:
    """Execute ``script`` in a fresh interpreter and return its stripped stdout.

    A fresh subprocess is required because earlier tests in the suite
    have almost certainly already loaded the saturation-aware modules
    in the parent process.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class TestLazySaturationAwareMetricRegistration:
    """Pin that fragmentation-based imports do not load SA metric modules."""

    def test_import_scheduling_py_package_does_not_load_sa_modules(self) -> None:
        """``import scheduling_py`` only loads ``errors`` -- no SA metric module is in ``sys.modules``."""
        script = """
            import sys
            import cosmos_xenna.pipelines.private.scheduling_py  # noqa: F401
            sa_modules = [
                "cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.heterogeneity",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.metrics",
                "cosmos_xenna.pipelines.private.scheduling_py.cluster.memory_pressure",
                "cosmos_xenna.pipelines.private.scheduling_py.lifecycle.loop_watchdog",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure",
            ]
            loaded = [m for m in sa_modules if m in sys.modules]
            print(",".join(loaded))
        """
        loaded = _run_in_subprocess(script)
        assert loaded == "", (
            f"Expected no SA metric modules in sys.modules after importing scheduling_py; "
            f"got: {loaded!r}. Eager imports in __init__ defeat the lazy-registration guarantee."
        )

    def test_import_streaming_does_not_load_sa_modules(self) -> None:
        """``import streaming`` reaches the SA branch only via deferred import; SA modules stay unloaded."""
        script = """
            import sys
            import cosmos_xenna.pipelines.private.streaming  # noqa: F401
            sa_modules = [
                "cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.heterogeneity",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.metrics",
                "cosmos_xenna.pipelines.private.scheduling_py.cluster.memory_pressure",
                "cosmos_xenna.pipelines.private.scheduling_py.lifecycle.loop_watchdog",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure",
            ]
            loaded = [m for m in sa_modules if m in sys.modules]
            print(",".join(loaded))
        """
        loaded = _run_in_subprocess(script)
        assert loaded == "", (
            f"Expected no SA metric modules in sys.modules after importing streaming; "
            f"got: {loaded!r}. The deferred import in _make_scheduler_algorithm has been broken."
        )

    def test_default_streaming_spec_does_not_load_sa_modules(self) -> None:
        """Constructing the default ``StreamingSpecificSpec()`` (FRAGMENTATION_BASED) leaves SA metric modules unloaded.

        The ``saturation_aware`` field defaults to ``None`` so the
        ``SaturationAwareConfig`` factory chain (which transitively
        imports ``pressure`` for ``BACKLOG_CAP``) never fires on the
        fragmentation-based path. Materialization happens lazily inside
        ``materialized_saturation_aware()`` only when the SA branches in
        ``streaming.py`` / ``streaming_backpressure.py`` request the
        config.

        The subprocess emits a single ``json.dumps(...)`` blob so the
        parent test parses with ``json.loads`` rather than ad-hoc string
        splitting; an unrelated stray ``print`` from any imported module
        surfaces as a clean ``json.JSONDecodeError`` instead of a
        misleading ``ValueError`` from a key/value split.
        """
        script = """
            import json
            import sys
            import cosmos_xenna.pipelines.private.specs as specs
            spec = specs.StreamingSpecificSpec()
            sa_modules = [
                "cosmos_xenna.pipelines.private.scheduling_py.scheduler.saturation_aware",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.scoring",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.heterogeneity",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.bottleneck.metrics",
                "cosmos_xenna.pipelines.private.scheduling_py.cluster.memory_pressure",
                "cosmos_xenna.pipelines.private.scheduling_py.lifecycle.loop_watchdog",
                "cosmos_xenna.pipelines.private.scheduling_py.phases.intent.pressure",
            ]
            print(json.dumps({
                "loaded": [m for m in sa_modules if m in sys.modules],
                "sa_none": spec.saturation_aware is None,
                "scheduler": spec.scheduler.value,
            }))
        """
        data = json.loads(_run_in_subprocess(script))
        assert data["loaded"] == [], (
            f"Expected no SA metric modules in sys.modules after constructing default "
            f"StreamingSpecificSpec(); got: {data['loaded']!r}. The default factory must keep "
            f"SaturationAwareConfig (and its pressure import) un-instantiated."
        )
        assert data["sa_none"] is True, data["sa_none"]
        assert data["scheduler"] == "fragmentation_based", data["scheduler"]

    @pytest.mark.parametrize("module_name", _SA_METRIC_MODULES)
    def test_explicit_sa_module_import_does_load_it(self, module_name: str) -> None:
        """Sanity check: an explicit import of an SA module DOES load it."""
        script = f"""
            import sys
            import {module_name}  # noqa: F401
            print({module_name!r} in sys.modules)
        """
        result = _run_in_subprocess(script)
        assert result == "True", f"Explicit import of {module_name!r} should load it; got {result!r}"
