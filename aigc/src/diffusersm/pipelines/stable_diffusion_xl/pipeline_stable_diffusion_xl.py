# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch 
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ...image_processor import PipelineImageInput, VaeImageProcessor


from ...loaders import (
    StableDiffusionXLLoraLoaderMixin,
)

from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.attention_processor import (
    AttnProcessor2_0,
)

from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    logging,
    scale_lora_layers,
)
from ..pipeline_utils import (
    DiffusionPipeline,
    StableDiffusionMixin,
)
from .pipeline_output import StableDiffusionXLPipelineOutput


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

class StableDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    StableDiffusionXLLoraLoaderMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.
    """
    
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor"
    ]
    
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negativate_add_time_ids",
    ]
    
    def __init__(
        self,
        vae: AutoencoderKL,
        
    ):
        super().__init__()
        self.register_modules(
            
        )
        