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
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch

from ..utils import (set_weights_and_activate_adapters,
                     set_adapter_layers,
                     delete_adapter_layers
                     )

USE_PEFT_BACKEND = True

class UNet2DConditionLoadersMixin:
    """
    Load LoRA layers into a [`UNet2DCondtionModel`].
    """
    def load_attn_process(self, pretrained_model_name_or_path_or_dict:Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `torch.nn.Module` class.
        
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
                
        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "../xx_dir/stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.unet.load_attn_procs(
            "../xx_dir/jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)  # _LOW_CPU_MEM_USAGE_DEFAULT
        
        _pipeline = kwargs.pop("_pipeline", None)
        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True
        
        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = None  # TODO, parser the pretrained_model_name_or_path_or_dict
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except  IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = None # TODO parser the pretrained_model_name_or_path_or_dict
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict
        
        # fill attn processors
        lora_layers_list = []
        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys()) and not USE_PEFT_BACKEND
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())
        
        if USE_PEFT_BACKEND:
            # In that case we have nothing to do as loading the adapter weights is already handled above by `set_peft_model_state_dict`
            # on the Unet
            pass
        elif is_lora:
            raise ValueError(
                "LoRACompatibleConv and etc has not support here, you should use peft"
            )
        elif is_custom_diffusion:
            raise ValueError(
                "CustomDiffusionAttnProcessor and etc has not support here, you should use peft"
            )
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )
        
    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))
        
    def _fuse_lora_apply(self, module, adapter_names=None):
        if not USE_PEFT_BACKEND:
            raise ValueError(
                    "The `adapter_names` argument is not supported in your environment. Please switch"
                    " to PEFT backend to use this argument by installing latest PEFT and transformers."
                    " `pip install -U peft transformers`"
                )
        else:
            from peft.tuners.tuners_utils import BaseTunerLayer
            merge_kwargs = {"safe_merge": self._safe_fusing}
            
            if isinstance(module, BaseTunerLayer):
                if self.lora_scale != 1.0:
                    module.scale_layer(self.lora_scale)
                # For BC with previous PEFT versions, we need to check the signature
                # of the `merge` method to see if it supports the `adapter_names` argument.
                supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
                if "adapter_names" in supported_merge_kwargs:
                    merge_kwargs["adapter_names"] = adapter_names
                elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                    raise ValueError(
                        "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                        " to the latest version of PEFT. `pip install -U peft`"
                    )
                module.merge(**merge_kwargs)
            
    def unfuse_lora(self):
        self.apply(self._unfuse_lora_apply)
        
    def _unfuse_lora_apply(self, module):
        if not USE_PEFT_BACKEND:
            raise ValueError(
                    "The `adapter_names` argument is not supported in your environment. Please switch"
                    " to PEFT backend to use this argument by installing latest PEFT and transformers."
                    " `pip install -U peft transformers`"
            )
        else:
            from peft.tuners.tuners_utils import BaseTunerLayer
            
            if isinstance(module, BaseTunerLayer):
                module.unmerge()
    
    def set_adapters(self, 
                        adapter_names: Union[List[str], str],
                        weights: Optional[Union[List[float], float]]=None,):
        """ 
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.
                
        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")
        
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
        
        if weights is None:
            weights = [1.0] * len(adapter_names)
        elif isinstance(weights, float):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )
        set_weights_and_activate_adapters(self, adapter_names, weights)
        
    def disable_lora(self):
        """
        Disable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)
        
    def enable_lora(self):
        """ 
        Enable the UNet's active LoRA layers.
        
        Example as disable_lora
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)
        
    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the UNet.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.
        
        ```
        pipeline.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
            
        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)
            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)