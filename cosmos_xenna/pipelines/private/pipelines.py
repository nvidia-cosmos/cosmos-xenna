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

from __future__ import annotations

import copy
import inspect
from typing import Optional

from cosmos_xenna._cosmos_xenna import setup_logging
from cosmos_xenna.pipelines.private import batch, specs, streaming
from cosmos_xenna.ray_utils import cluster
from cosmos_xenna.utils import python_log as logger
from cosmos_xenna.utils.verbosity import VerbosityLevel


def _validate_method_signature(
    instance_class: type,
    instance_method_name: str,
    base_class: type,
    base_method_name: str,
) -> None:
    """
    Validates that if a method is overridden in instance_class, its signature matches
    the signature of the corresponding method in base_class.
    """
    instance_method_attr = getattr(instance_class, instance_method_name, None)
    base_method_attr = getattr(base_class, base_method_name, None)

    if instance_method_attr is None:
        logger.warning(
            f"Method {instance_method_name} not found in {instance_class.__name__}. Cannot validate signature."
        )
        return

    if base_method_attr is None:
        # This should not happen if base_class and base_method_name are correct
        logger.error(f"Base method {base_method_name} not found in {base_class.__name__}. Cannot validate.")
        return

    # Check if the method is overridden.
    # This means the function object in the subclass is different from the base class's.
    if instance_method_attr is base_method_attr:
        return  # Not overridden, signature is implicitly correct

    # If overridden, inspect signatures
    try:
        # Inspect the method as defined on the class
        instance_sig = inspect.signature(instance_method_attr)
        base_sig = inspect.signature(base_method_attr)
    except (ValueError, TypeError):
        logger.warning(
            f"Could not inspect signature for method {instance_method_name} on {instance_class.__name__} "
            f"or {base_method_name} on {base_class.__name__} (it might not be a Python function)."
        )
        return

    instance_params = list(instance_sig.parameters.values())
    base_params = list(base_sig.parameters.values())

    if len(instance_params) != len(base_params):
        raise TypeError(
            f"Method '{instance_method_name}' in stage '{instance_class.__name__}' "
            f"has an incorrect number of parameters. "
            f"Expected {len(base_params)} (from {base_class.__name__}.{base_method_name}), got {len(instance_params)}. "
            f"Expected signature: {base_sig}, Actual signature: {instance_sig}."
        )


def run_pipeline(
    pipeline_spec: specs.PipelineSpec,
) -> Optional[list]:
    """Entry point for calling a pipeline.

    Depending on the environment and the pipeline spec, this can call a STREAMING, BATCH or BATCH_DEBUG pipeline.

    Before we start a pipeline, we do the following:

    - Download any required model weights
    - (if not a debug pipeline) Connect to a ray cluster if running on the cloud, otherwise start a local ray cluster
      and connect to it
    - Run the pipeline
    - (if pipeline_spec.return_last_stage_outputs is True) Return the results from the last stage.

    See yotta/ray_utils/README.md for more info on running pipelines.

    Args:
        pipeline_spec: The pipeline to run
        execution_mode: (Deprecated, see pipeline_spec.execution_mode)

    Returns:
        (If pipeline_spec.config.return_last_stage_outputs is true) The list of items from the last stage in the
        pipeline. NOTE: These are pulled down to the host machine. You probably do not what to return anything
        heavy-weight here.
    """
    # Setup logging in rust.
    setup_logging()
    # Convert the stages field into StageSpecs if needed.
    pipeline_spec = copy.deepcopy(pipeline_spec)
    pipeline_spec.stages = [x if isinstance(x, specs.StageSpec) else specs.StageSpec(x) for x in pipeline_spec.stages]
    # Validate the stages:
    for stage_spec_item in pipeline_spec.stages:
        assert isinstance(stage_spec_item, specs.StageSpec)
        stage_spec_item.validate()

    for stage_spec_item in pipeline_spec.stages:
        assert isinstance(stage_spec_item, specs.StageSpec)
        actual_stage_instance = stage_spec_item.stage
        stage_class = type(actual_stage_instance)

        # Validate methods from specs.Stage
        _validate_method_signature(stage_class, "setup", specs.Stage, "setup")
        _validate_method_signature(stage_class, "setup_on_node", specs.Stage, "setup_on_node")
        _validate_method_signature(stage_class, "process_data", specs.Stage, "process_data")

    # Override stage level params with global params if needed
    for idx in range(len(pipeline_spec.stages)):
        pipeline_spec.stages[idx] = pipeline_spec.stages[idx].override_with_pipeline_params(pipeline_spec.config)
    if not pipeline_spec.input_data:
        logger.warning(
            "No input data specified for the pipeline. Skipping running the pipeline and return an empty list."
        )
        return []
    stage_names = [x.name(i) for i, x in enumerate(pipeline_spec.stages)]
    assert len(stage_names) == len(set(stage_names)), f"Expected stage names to be unique, but got: {stage_names}"

    if pipeline_spec.config.monitoring_verbosity_level >= VerbosityLevel.INFO:
        logger.info(pipeline_spec)

    logger.info("Initialized Ray cluster.")
    cluster.init_or_connect_to_cluster()

    cluster_resources = cluster.make_cluster_resources_from_ray_nodes(
        cpu_allocation_percentage=pipeline_spec.config.cpu_allocation_percentage
    )
    logger.info(f"Cluster resources: {cluster_resources}")
    logger.info(f"Created/connected to cluster with resources: {cluster_resources.totals()}")
    if pipeline_spec.config.execution_mode == specs.ExecutionMode.STREAMING:
        if pipeline_spec.config.mode_specific is None:
            pipeline_spec.config.mode_specific = specs.StreamingSpecificSpec()
        return streaming.run_pipeline(pipeline_spec, cluster_resources)
    elif pipeline_spec.config.execution_mode == specs.ExecutionMode.BATCH:
        return batch.run_pipeline(pipeline_spec, cluster_resources)
    else:
        raise ValueError(f"unknown execution mode={pipeline_spec.config.execution_mode}")
