import os
import torch
import itertools
import accelerate
import safetensors
from torch import nn
from functools import partial
from collections import OrderedDict
from ..utils import (CONFIG_NAME, logging,
                     WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME)
from ..utils import SAFETENSORS_FILE_EXTENSION
from typing import Union, Any, Callable, List, Optional, Tuple


logger = logging.get_logeer(__name__)

_LOW_CPU_MEM_USAGE_DEFAULT = True

def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

def get_parameter_device(parameter: nn.Module) -> torch.device:
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5
        # should be remove, because we only install torch > 2.0
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device

def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        raise "install torch version > 2.0" 
    
    
def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try: 
        file_extension = os.path.basename(checkpoint_file).split('.')[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safetensors.torch.load_file(checkpoint_file, device='cpu')
        else:
            return torch.load(checkpoint_file, map_location='cpu')
    except Exception as e:
        try: 
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError): 
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )

class ModelMixin(nn.Module):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self):
        super().__init__()
    
    def __getattr__(self, name: str) -> Any:
        """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129 We need to overwrite
        __getattr__ here in addition so that we don't trigger `torch.nn.Module`'s __getattr__':
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. 
            Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'unet.config.{name}'."
            print(deprecation_message)
            return self._internal_dict[name]

        # call PyTorch's https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        return super().__getattr__(name)
    
    @property
    def is_gradient_checkpointing(self) -> bool:
        """ 
        Whether gradient checkpointing is activated for this model or not.
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules)
    
    def enable_gradient_checkpointing(self) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self) -> None:
        """
        Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[Callable] = None) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)
            for child in module.children():
                fn_recursive_set_mem_eff(child)
        
        for module in self.children():
            if isinstance(module, nn.Module):
                fn_recursive_set_mem_eff(module)
                
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None) -> None:
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
        inference. Speed up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import torch
        >>> from diffusers import UNet2DConditionModel
        >>> from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

        >>> model = UNet2DConditionModel.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", torch_dtype=torch.float16
        ... )
        >>> model = model.to("cuda")
        >>> model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)
        
    def disable_xformers_memory_efficient_attention(self) -> None:
        r"""
        Disable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).
        """
        self.set_use_memory_efficient_attention_xformers(False)
        
    def save_pretrained(
        self, 
        save_directory: Union[str, os.PathLike],
        is_main_process: bool= True,
        save_function: Optional[Callable] = None,
        variant: Optional[str] = None,
        safe_serialization: bool = True,
        **kwargs,
        ):
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Only save the model itself if we are using distributed training
        model_to_save = self
        
        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)
            
        # Save the model
        state_dict = model_to_save.state_dict()
        
        weight_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weight_name = _add_variant(weight_name, variant)
        
        # Save the model
        if safe_serialization:
            safetensors.torch.save_file(state_dict, os.path.join(save_directory, weight_name), metadata={"format": "pt"})
        else:
            torch.save(state_dict, os.path.join(save_directory, weight_name))
        
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
                
        Example:

        ```py
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        
        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True
            
        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path
        
        # load config
        config, unused_kwargs = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            local_files_only=local_files_only,
            subfolder=subfolder,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            **kwargs,
        )
        
        # load model
        model_file = None
        if use_safetensors:
                try:
                    model_file = None # TODO 
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    pass
        if model_file is None:
            model_file = None # TODO 
        
        if low_cpu_mem_usage:
            # Instantiate model with empty weights
            with accelerate.init_empty_weights():
                model = cls.from_config(config, **unused_kwargs)
            
            # if device_map is None, load the state dict and move the params from meta device to the cpu
            if device_map is None:
                param_device = "cpu"
                pass
            else:
                # else let accelerate handle loading and dispatching.
                # Load weights and dispatch according to the device_map
                # by default the device_map is None and the weights are loaded on the CPU
                try:
                    accelerate.load_checkpoint_and_dispatch(
                        model,
                        model_file,
                        device_map,
                        max_memory=max_memory,
                        offload_folder=offload_folder,
                        offload_state_dict=offload_state_dict,
                        dtype=torch_dtype,
                    )
                except AttributeError as e:
                    # When using accelerate loading, we do not have the ability to load the state
                    # dict and rename the weight names manually. Additionally, accelerate skips
                    # torch loading conventions and directly writes into `module.{_buffers, _parameters}`
                    # (which look like they should be private variables?), so we can't use the standard hooks
                    # to rename parameters on load. We need to mimic the original weight names so the correct
                    # attributes are available. After we have loaded the weights, we convert the deprecated
                    # names to the new non-deprecated names. Then we _greatly encourage_ the user to convert
                    # the weights so we don't have to do this again.
                    raise e
        
            loading_info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
        else:
            model = cls.from_config(config, **unused_kwargs)

            state_dict = load_state_dict(model_file, variant=variant)

            model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                model,
                state_dict,
                model_file,
                pretrained_model_name_or_path,
            )

            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
        
        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}.")
        elif torch_dtype is not None:
            model = model.to(torch_dtype)
        
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        
        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info
        else:
            return model
        
    @classmethod
    def _load_pretrained_model(cls, 
                               model, 
                               state_dict: OrderedDict,
                               pretrained_model_name_or_path: Union[str, os.PathLike],
                               ignore_mismatched_sizes: bool = False,
                               ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = list(state_dict.keys())

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        pass
    
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (trainable or non-embedding) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters.
            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embedding parameters.

        Returns:
            `int`: The number of parameters.

        Example:

        ```py
        from diffusers import UNet2DConditionModel

        model_id = "runwayml/stable-diffusion-v1-5"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        unet.num_parameters(only_trainable=True)
        859520964
        ```
        """
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules()
                if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)