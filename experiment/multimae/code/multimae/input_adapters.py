# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .multimae_utils import build_1d_sincos_posemb, pair, trunc_normal_


class PatchedInputAdapter(nn.Module):
    """Adapter for 1D sequence inputs, like battery data.
    Creates tokens from patches over the sequence.

    :param num_channels: Number of input channels of the sequence
    :param stride_level: Stride level compared to the full-sized sequence.
        E.g. 4 for 1/4th the size of the sequence.
    :param patch_size_full: Int of the patch size over the full sequence length.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param input_length: Default sequence length. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 learnable_pos_emb: bool = False,
                 input_length: int = 256):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = patch_size_full
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.input_length = input_length
        self.num_patches = self.input_length // patch_size_full

        # Actual patch length, taking into account stride of input
        self.P_L = max(1, self.patch_size_full // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        l_posemb = self.input_length // (self.stride_level * self.P_L)
        
        if self.sincos_pos_emb:
            self.pos_emb = build_1d_sincos_posemb(length=l_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, l_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Sequence -> tokens projection
        self.proj = nn.Conv1d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=self.P_L, stride=self.P_L
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming sequence to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input sequence tensor [B, C, L]
        """
        B, C, L = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (L % self.P_L == 0), f'Sequence length {L} must be divisible by patch size {self.P_L}'
        N_L = L // self.P_L # Number of patches

        # Create patches [B, C, L] -> [B, N_L, C] (wait, proj output is [B, D, N_L])
        # self.proj(x) -> [B, dim_tokens, N_L]
        # rearrange to [B, N_L, dim_tokens]
        x_patch = rearrange(self.proj(x), 'b d nl -> b nl d')

        # Create positional embedding
        # pos_emb is [1, dim_tokens, l_posemb]
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_L), mode='linear', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nl -> b nl d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x
