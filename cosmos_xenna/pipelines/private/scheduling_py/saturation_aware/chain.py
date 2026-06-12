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

"""Pipeline chain-factor math for the saturation-aware scheduler.

Mirrors the fragmentation solver's ``num_input_samples_per_sample``: the
cumulative fan-out from the source to each stage. Converting per-stage
depths by these factors expresses queued or active work in common
source-item units, so a stage can see the whole upstream stock that will
fan out to it (not only its own immediate input depth).
"""

from collections.abc import Sequence

# Smallest chain factor treated as a usable measurement. A chain factor below
# this (a near-total fan-in, or a corrupted/degenerate measurement) makes the
# reciprocal ``1 / factor`` explode, which previously inflated source-normalized
# stock and capacity to non-physical values (observed ``1e8+`` cap_src / stock).
# Both source normalizers (here and ``capacity._source_capacities``) treat a
# sub-threshold factor as unusable (contributes ``0.0``) so an impossible value
# can never enter the bottleneck min or the floor's stock reasoning. A legitimate
# heavy fan-in down to ~``1e6:1`` still passes.
MIN_CHAIN_FACTOR = 1e-6


def chain_factors(num_returns_per_batch: Sequence[float], stage_batch_sizes: Sequence[int]) -> list[float]:
    """Return cumulative fan-out ``k[i] = product over j < i of f[j]``.

    ``f[j] = num_returns_per_batch[j] / stage_batch_sizes[j]`` is stage
    ``j``'s output items per input item, so ``k[i]`` is the number of
    stage-``i`` input items produced per source item. ``k[0]`` is 1.

    Args:
        num_returns_per_batch: Per-stage returns produced per processed batch.
        stage_batch_sizes: Per-stage input items consumed per batch (each > 0).

    Returns:
        Per-stage cumulative fan-out factors, one per stage.

    Raises:
        ValueError: If the two sequences differ in length or a batch size
            is not positive, or a per-batch return count is negative.
    """
    if len(num_returns_per_batch) != len(stage_batch_sizes):
        raise ValueError(
            f"length mismatch: num_returns={len(num_returns_per_batch)} batch_sizes={len(stage_batch_sizes)}"
        )
    for i, num_returns in enumerate(num_returns_per_batch):
        if num_returns < 0.0:
            raise ValueError(f"num_returns_per_batch[{i}] must be >= 0, got {num_returns}")
    factors: list[float] = []
    acc = 1.0
    for i, batch in enumerate(stage_batch_sizes):
        if batch <= 0:
            raise ValueError(f"stage_batch_sizes[{i}] must be > 0, got {batch}")
        if i == 0:
            factors.append(1.0)
            continue
        acc *= num_returns_per_batch[i - 1] / stage_batch_sizes[i - 1]
        factors.append(acc)
    return factors


def source_stock_threshold(batch_size: int, chain_factor: float) -> float:
    """Return one batch worth of source-item stock for a stage.

    A stage is considered to have real work when its whole-chain at-or-upstream
    stock (in source-item units) reaches one batch's worth of source items,
    ``batch_size / chain_factor``. A ``chain_factor`` below
    :data:`MIN_CHAIN_FACTOR` (a fully dropping upstream stage, or a degenerate /
    corrupted factor whose reciprocal would explode) collapses the threshold to
    ``0.0``, matching :func:`whole_chain_stock`, which omits the same factors
    from the stock sum. This defines the one-batch "has work" boundary in
    source-item units for the scale-down release gate (``floor.compute_floors``).
    The growth gate enforces the same one-batch concept directly on local pending
    depth (``_Cycle.has_local_input``), so growth and release agree on the line
    without both routing through this helper.

    Args:
        batch_size: Stage input items consumed per batch (``> 0`` in practice).
        chain_factor: Stage's cumulative fan-out from :func:`chain_factors`.

    Returns:
        The source-unit stock at which the stage has at least one batch of work,
        or ``0.0`` when ``chain_factor`` is below :data:`MIN_CHAIN_FACTOR`.
    """
    return batch_size / chain_factor if chain_factor >= MIN_CHAIN_FACTOR else 0.0


def whole_chain_stock(queue_depths: Sequence[float], chain: Sequence[float]) -> list[float]:
    """Return per-stage at-or-upstream pending stock, in source-item units.

    ``stock[k] = sum over u <= k of D[u] / k[u]``, where ``D[u]`` is the
    stage-``u`` depth (queued or active, in stage-``u`` input samples) and
    ``k[u]`` the chain factor. A chain factor below :data:`MIN_CHAIN_FACTOR`
    (a fully dropping upstream stage, or a degenerate/corrupted factor whose
    reciprocal would explode) contributes nothing, since no work it admits can
    be expressed in source units.

    Args:
        queue_depths: Per-stage depths, in stage-input samples.
        chain: Per-stage chain factors from :func:`chain_factors`.

    Returns:
        Per-stage cumulative upstream-or-own stock, in source-item units.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(queue_depths) != len(chain):
        raise ValueError(f"length mismatch: queue_depths={len(queue_depths)} chain={len(chain)}")
    stock: list[float] = []
    running = 0.0
    for queue_depth, factor in zip(queue_depths, chain, strict=True):
        non_negative_depth = max(queue_depth, 0.0)
        if factor >= MIN_CHAIN_FACTOR:
            running += non_negative_depth / factor
        stock.append(running)
    return stock
