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
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..activations import get_activation

from ..resnet import Downsample1D, ResidualTemporalBlock1D, Upsample1D, rearrange_dims

class DownResnetBlock1D(nn.Module):
    """ 
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 num_layers: int = 1,
                 conv_shortcut: bool = False,
                 temb_channels: int = 32,
                 groups: int = 32,
                 groups_out: Optional[int] = None,
                 non_linearity: Optional[str] = None,
                 time_embedding_norm: str = "default",
                 output_scale_factor: float = 1.0,
                 add_downsample: bool = True
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor
        
        if groups_out is None:
            groups_out = groups
        
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]
        
        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels))
        
        self.resnets = nn.ModuleList(resnets)
        
        if non_linearity is None:
            self.non_linearity = None
        else:
            self.non_linearity = get_activation(non_linearity)
        
        self.downsample = None 
        if add_downsample:
            self.downsample = Downsample1D(in_channels, out_channels, use_conv=True, padding=1)
        
    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        output_states = ()
        
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)
        
        output_states += (hidden_states, )
        
        if self.non_linearity is not None:
            hidden_states = self.non_linearity(hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        
        return hidden_states, output_states

class DownBlock1D(nn.Module):
    """ 
    """
    def __init__():
        super().__init__()
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
    
class AttnDownBlock1D(nn.Module):
    """ 
    """
    def __init__():
        super().__init__()
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
    
class DownBlock1DSkip(nn.Module):
    """ 
    """
    def __init__():
        super().__init__()
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


DownBlockType = Union[DownResnetBlock1D, DownBlock1D, AttnDownBlock1D, DownBlock1DSkip]
MidBlockType = Union[MidResTemporaBlock1D, ValueFunctionMidBlock1D, UnetMidBlock1D]
UpBlockType = Union[UpResnetBlock1D, UpBlock1D, AttnUpBlock1D, UpBlock1DNoSkip]
OutBlockType = Union[OutConvBlock1D, OutValueFunctionBlock1D]


def get_down_block(down_block_type: str,
                   num_layers: int,
                   in_channels: int,
                   out_channels: int,
                   temb_channels: int,
                   add_downsample: bool,
                   ) -> DownBlockType:
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D()
    elif down_block_type == "DownBlock1D":
        return DownBlock1D()
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D()
    elif down_block_type == "DownBlock1DSkip":
        return DownBlock1DSkip()
    else:
        raise ValueError(f"{down_block_type} does not exit." )
        

def get_mid_block():
    pass

def get_up_block():
    pass

def get_out_block():
    pass