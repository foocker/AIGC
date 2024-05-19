# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import get_activation


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states
    
class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x