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

import loguru
import ray.runtime_context
from loguru import logger

from cosmos_xenna.pipelines.private import resources

API_LIMIT = 40000


def make_cluster_resources_from_ray_nodes(
    cpu_allocation_percentage: float = 1.0,
) -> resources.ClusterResources:
    """Make a ClusterResources object from the current ray nodes."""
    return resources.make_cluster_resources_for_ray_cluster(cpu_allocation_percentage)


def logger_custom_serializer(obj: "loguru.Logger") -> None:  # String literal is needed for python 3.9.
    return None


def logger_custom_deserializer(obj: None) -> "loguru.Logger":  # String literal is needed for python 3.9.
    # Initialize a default logger
    return logger


def init_or_connect_to_cluster(
    log_to_driver: bool = True,
) -> ray.runtime_context.RuntimeContext:
    """Initializes a new local Ray cluster or connects to an existing one.

    This function serves as a central point for managing Ray cluster connections.
    - If `existing_cluster` is True, it attempts to connect to a pre-existing Ray cluster
      using `ray.init(address="auto")`. This is typically used in environments where
      a Ray cluster is managed externally (e.g., by a job scheduler like Slurm with ngc-ray).
    - If `existing_cluster` is False (default), it initializes a new local Ray cluster
      using `ray.init()`. This starts a head node on the local machine, along with
      a dashboard for monitoring. The dashboard will be accessible via the URL
      printed to the logs.

    Args:
        log_to_driver: If True (default), logs from Ray workers will be forwarded to the
            driver process (the process that called this function). This is useful for
            debugging but can potentially cause performance overhead in large-scale jobs
            due to the volume of logs. Set to False to disable log forwarding from workers.
        existing_cluster: If True, connect to an existing cluster identified by `ray.init(address="auto")`.
            If False (default), start a new local Ray cluster.

    Returns:
        ray.runtime_context.RuntimeContext: The Ray RuntimeContext object, which provides
            information about the connected Ray cluster, including the dashboard URL.
    """

    # We need to set this env var to avoid ray from setting CUDA_VISIBLE_DEVICES.
    # We set these manually in Xenna because we allocate the gpus manually instead of relying on ray's mechanisms.
    # This will *only* get picked up from here if the cluster is started from this script. In the case of previously
    # existing clusters, this needs to be set in the processes that set up the cluster.
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "0"
    # These need to be set to allow listing debug info about more than 10k actors.
    os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = str(API_LIMIT)
    os.environ["RAY_MAX_LIMIT_FROM_DATA_SOURCE"] = str(API_LIMIT)
    # User can turn on metrics export via env var XENNA_RAY_METRICS_PORT
    ray_metrics_port = os.getenv("XENNA_RAY_METRICS_PORT", None)

    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__, serializer=logger_custom_serializer, deserializer=logger_custom_deserializer
    )

    context = ray.init(
        include_dashboard=True,
        ignore_reinit_error=True,
        log_to_driver=log_to_driver,
        _metrics_export_port=ray_metrics_port,
    )
    logger.info(f"Ray dashboard url: {context.dashboard_url}")
    return context
