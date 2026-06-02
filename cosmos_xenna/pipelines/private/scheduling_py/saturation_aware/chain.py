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
queue depths by these factors expresses all pending work in common
source-item units, so a stage can see the whole upstream stock that will
fan out to it (not only its own immediate input queue).
"""

from collections.abc import Sequence


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
            is not positive.
    """
    if len(num_returns_per_batch) != len(stage_batch_sizes):
        raise ValueError(
            f"length mismatch: num_returns={len(num_returns_per_batch)} batch_sizes={len(stage_batch_sizes)}"
        )
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


def whole_chain_stock(queue_depths: Sequence[float], chain: Sequence[float]) -> list[float]:
    """Return per-stage at-or-upstream pending stock, in source-item units.

    ``stock[k] = sum over u <= k of Q[u] / k[u]``, where ``Q[u]`` is the
    stage-``u`` input queue depth (in stage-``u`` input samples) and
    ``k[u]`` the chain factor. A non-positive ``k[u]`` (a fully dropping
    upstream stage) contributes nothing, since no work flows past it.

    Args:
        queue_depths: Per-stage input queue depths, in stage-input samples.
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
        if factor > 0.0:
            running += queue_depth / factor
        stock.append(running)
    return stock
