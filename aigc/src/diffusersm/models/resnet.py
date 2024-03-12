import torch 
from torch import nn
from typing import Union, Tuple

from .downsampling import (  # noqa
    Downsample1D,
    )
from .upsampling import (  # noqa
    Upsample1D,
)
from .activations import get_activation

def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    elif len(tensor.shape) == 3:
        # tensor[:, :, None, :] why 3 -> 4 ?by unet_rl.py
        return tensor
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")
    
class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 n_groups: int = 8,
                 activation: str = "mish"
                 ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.act_func = get_activation(activation)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out_temp = self.conv(inputs)
        out_temp_1d = rearrange_dims(out_temp) # b, c h
        out_temp = self.group_norm(out_temp_1d)
        out_temp_1d = rearrange_dims(out_temp)
        output = self.act_func(out_temp_1d)
        
        return output

class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels :int,
                 embed_dim: int,
                 kernel_size: Union[int, Tuple[int, int]] = 5,
                 activation: str = "mish",
                 ):
        super().__init__()
        self.conv_in = Conv1dBlock(in_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)
        
        self.time_emb_act = get_activation(activation)
        self.time_emb = nn.Linear(embed_dim, out_channels)
        
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        
    def forward(self, inputs:torch.Tensor, t:torch.Tensor)-> torch.Tensor:
        """Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)
        