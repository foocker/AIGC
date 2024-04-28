import contextlib
import copy
import random
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from .models import UNet2DConditionModel

from .utils import (
    convert_all_state_dict_to_peft, 
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
)

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_snr(noise_scheduler, timesteps):
    pass


def unet_lora_state_dict(unet: UNet2DConditionModel) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    pass


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    pass


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """
    pass
