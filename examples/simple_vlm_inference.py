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
An example Ray pipeline for LLM inference using hardcoded image URLs.

This pipeline demonstrates:
- Downloading images directly from URLs using `requests`.
- Using a Hugging Face transformer model for inference (e.g., image captioning).
- Logging results to the console.
- Structuring a simplified multi-stage pipeline with Ray.
"""

import io
import random
from typing import Optional

import attrs
import huggingface_hub
import requests
from PIL import Image
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.processing_auto import AutoProcessor

from cosmos_xenna.pipelines import v1 as pipelines_v1
from cosmos_xenna.utils import python_log as logger

# Switch to Phi-3.5 model
MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
MODEL_REVISION = "refs/pr/38"
LIMIT = 20  # Limit the number of samples to download/process
BATCH_SIZE = 4  # Adjust based on GPU memory

_IMAGE_URLS = [
    "https://images.metmuseum.org/CRDImages/es/original/DP169011.jpg",
    "https://openaccess-cdn.clevelandart.org/1987.58/1987.58_web.jpg",
    "https://live.staticflickr.com/8708/16959810206_ae01fdb5ff_b.jpg",
]


@attrs.define
class _Sample:
    """A single sample processed by the pipeline."""

    image_url: str  # Image URL from the dataset
    image_bytes: Optional[bytes] = None
    generated_caption: Optional[str] = None


@attrs.define
class _TaskData:
    """Represents a batch of samples to be processed by a pipeline stage."""

    samples: list[_Sample]


class _DownloadStage(pipelines_v1.Stage):
    """Stage for downloading images from URLs."""

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0, gpus=0.0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        logger.info("Initializing download stage...")
        # Optional: Initialize a requests session for potential performance improvements
        self.session = requests.Session()
        self._cache: dict[str, bytes] = {}  # Initialize cache

    def process_data(self, samples: list[_Sample]) -> list[_Sample]:
        """Download images from URLs, using a cache."""
        logger.debug(f"Downloading images for task with {len(samples)} samples.")
        for sample in samples:
            if sample.image_url in self._cache:
                sample.image_bytes = self._cache[sample.image_url]
                logger.debug(f"Cache hit for {sample.image_url}")
                continue  # Skip download if cached

            logger.debug(f"Cache miss for {sample.image_url}. Downloading...")
            response = self.session.get(sample.image_url, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise an exception for bad status codes
            image_bytes = response.content
            sample.image_bytes = image_bytes
            # Only store in cache if download was successful and got bytes
            self._cache[sample.image_url] = image_bytes
            logger.debug(f"Successfully downloaded and cached image from {sample.image_url}")

        # Filter out samples that failed to download before returning
        return [s for s in samples if s.image_bytes is not None]


class _InferenceStage(pipelines_v1.Stage):
    """Stage for running inference using a Hugging Face model."""

    def __init__(self, model_name: str, stage_batch_size: int) -> None:
        self._model_name = model_name
        self._stage_batch_size = stage_batch_size

    @property
    def stage_batch_size(self) -> int:
        return self._stage_batch_size

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=2.0, gpus=1.0)

    def setup_on_node(self, node_info: pipelines_v1.NodeInfo, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        # Cache this model on the node. You can assume that this only gets called once per node.
        # So, if there are 8 actors running on a node, this will only get called once, on one of them.
        logger.info(f"Ensuring model {self._model_name} artifacts are cached on node {node_info.node_id}...")
        # Use snapshot_download to download all files without loading the model into memory.
        # This assumes the model and processor files are within the same repository.
        huggingface_hub.snapshot_download(
            repo_id=self._model_name,
            revision=MODEL_REVISION,
            local_files_only=False,  # Download if not cached
            resume_download=True,  # Resume interrupted downloads
            # trust_remote_code is handled by from_pretrained in the worker setup
        )
        logger.info(f"Model {self._model_name} artifacts are cached or downloading on node {node_info.node_id}.")

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Load the Hugging Face model and processor from the cache."""
        logger.info(f"Loading model {self._model_name} from cache on worker...")
        # Load Phi-3.5 model from cache only.
        # Assumes setup_per_node has already downloaded it.
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            _attn_implementation="eager",
            local_files_only=True,  # Fail if not cached
        )
        self.model.to("cuda")
        self.processor = AutoProcessor.from_pretrained(
            self._model_name,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,  # Fail if not cached
        )
        logger.info(f"Model {self._model_name} and processor loaded successfully from cache.")

    def process_data(self, samples: list[_Sample]) -> list[_Sample]:
        """Run batched inference on the images using Phi-3.5 Vision."""
        if not self.model or not self.processor:
            raise RuntimeError("Model and processor not loaded. Setup must be called first.")

        for sample in samples:
            if sample.image_bytes is None:
                logger.warning(f"Skipping sample with missing image bytes (URL: {sample.image_url})")
                continue

            # Decode image bytes into PIL Image
            pil_image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
            images_for_processor = [pil_image]

            messages = [
                {"role": "user", "content": "<|image_1|>\nCaption this image."},
            ]

            prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(prompt, images_for_processor, return_tensors="pt").to("cuda")

            generation_args = {
                "max_new_tokens": 1000,
                "temperature": 0.0,
                "do_sample": False,
            }

            generate_ids = self.model.generate(
                **inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args
            )

            # remove input tokens
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            sample.generated_caption = response

        for sample in samples:
            sample.image_bytes = None
        return samples


def _get_samples(limit: int) -> list[_Sample]:
    """Create tasks by randomly selecting from hardcoded image URLs."""
    logger.info(f"Creating tasks using hardcoded image URLs, limit={limit}...")

    samples = []
    for _ in range(limit):
        url = random.choice(_IMAGE_URLS)
        samples.append(_Sample(image_url=url, image_bytes=None))

    logger.info(f"Finished creating {len(samples)} samples.")
    return samples


def main() -> None:
    # Create tasks by sampling from hardcoded URLs
    tasks = _get_samples(LIMIT)

    if not tasks:
        logger.error("No tasks generated. Check _IMAGE_URLS and LIMIT. Exiting.")
        return

    # Define the simplified pipeline structure
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=tasks,
        stages=[
            pipelines_v1.StageSpec(_DownloadStage()),
            pipelines_v1.StageSpec(_InferenceStage(MODEL_NAME, BATCH_SIZE)),
        ],
        config=pipelines_v1.PipelineConfig(
            log_worker_allocation_layout=True,
            return_last_stage_outputs=True,
        ),
    )

    logger.info("\nStarting Ray pipeline...")
    logger.info(pipeline_spec)
    outputs = pipelines_v1.run_pipeline(pipeline_spec)
    logger.info("\nPipeline finished. Got the following outputs:")
    for output in outputs:
        logger.info(output)


if __name__ == "__main__":
    main()
