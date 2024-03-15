import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional


class Upsample1D(nn.Module):
    """ A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """
    def __init__(self,
                 channels: int,
                 out_channels: Optional[int] = None,
                 use_conv: bool = False,
                 use_conv_transpose: bool = False,
                 name: str = "conv"
                 ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(channels, self.out_channels, 3, padding=1)
            
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)
        outputs = F.interpolate(inputs, scale_factor=2., mode="nearest")
        if self.use_conv:
            outputs = self.conv(outputs)
        return outputs
        