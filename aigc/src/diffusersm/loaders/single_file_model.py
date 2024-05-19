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
import re
from contextlib import nullcontext
from typing import Optional

from ..utils import logging

logger = logging.get_logger(__name__)


class FromOriginalModelMixin:
    """
    Load pretrained weights saved in the `.ckpt` or `.safetensors` format into a model.
    """
    
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path_or_dict: Optional[str] = None, **kwargs):
        r"""
        Instantiate a model from pretrained weights saved in the original `.ckpt` or `.safetensors` format. The model
        is set in evaluation mode (`model.eval()`) by default.
        """
        class_name = cls.__name__
        
        config = kwargs.pop("config", None)
        original_config = kwargs.pop("original_config", None)

        if config is not None and original_config is not None:
            raise ValueError(
                "`from_single_file` cannot accept both `config` and `original_config` arguments. Please provide only one of these arguments"
            )
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        subfolder = kwargs.pop("subfolder", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        
        pass