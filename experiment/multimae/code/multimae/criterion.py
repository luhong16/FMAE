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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedMSELoss(nn.Module):
    """MSE loss with masking for 1D sequences
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nl):
        p = self.scale_factor
        # imgs: [B, C, L]
        # L = nl * p
        x = rearrange(imgs, "b c (nl p) -> b nl (p c)", nl=nl, p=p)
        return x

    def unpatchify(self, x, nl):
        p = self.scale_factor
        imgs = rearrange(x, "b nl (p c) -> b c (nl p)", nl=nl, p=p)
        return imgs

    def forward(self, input, target, mask=None):
        # input: [B, C, L]
        L = input.shape[-1]
        nl = L // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nl)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nl)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            # mask: [B, nl]
            # We want to upsample to [B, L]
            '''
            print(f"input:{input[:1].shape, input[:1]}")
            print(f"target:{target[:1].shape, target[:1]}")
            print(f"original mask:{mask[:1].shape, mask[:1]}")
            '''
            mask = F.interpolate(mask.unsqueeze(1).float(), size=L, mode='nearest').squeeze(1)
            # print(f"original loss:{loss[:1].shape, loss[:1]}, mask:{mask[:1].shape, mask[:1]}")
            loss = loss.mean(dim=1)  # B, C, L -> B, L
            loss = loss * mask
            # Compute mean per sample
            '''
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            # loss = loss.nanmean() # Account for zero masks
            valid_mask = ~torch.isnan(loss)
            if valid_mask.sum() > 0:
                loss = loss[valid_mask].mean()
            '''
            numer = loss.flatten(start_dim=1).sum(dim=1)
            denom = mask.flatten(start_dim=1).sum(dim=1)
            
            # Avoid division by zero to prevent NaN generation
            valid_sample = denom > 0
            denom = torch.where(valid_sample, denom, torch.ones_like(denom))
            loss = numer / denom
            
            if valid_sample.sum() > 0:
                loss = loss[valid_sample].mean()
            else:
                loss = torch.tensor(0.0).to(loss.device)
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss
