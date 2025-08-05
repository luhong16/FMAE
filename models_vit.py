# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import Block 
from timm.models.layers import trunc_normal_

from util.pos_embed import get_1d_sincos_pos_embed
class PatchEmbed(nn.Module):
    """ 1D EV battery to Patch Embedding
    """
    def __init__(
            self,
            snippet_size=128,
            patch_size=8,
            in_chans=6,
            embed_dim=768,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()
        self.snippet_size = snippet_size
        self.patch_size = patch_size
        assert snippet_size % patch_size == 0
        self.num_patches = snippet_size // patch_size

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        assert(L == self.snippet_size, f"Input snippet size ({L}) doesn't match model ({self.snippet_size}).")
        x = self.proj(x)
        x = x.transpose(1, 2)  # BCL' -> BL'C
        x = self.norm(x)
        return x
    
class SqueezeLayer(nn.Module):
    def __init__(self, dim=None):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, self.dim)
        
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_snippet=0, global_pool=False, dec_kwargs=None, **kwargs):
        
        embed_dim = kwargs['embed_dim']
        if "pos_embed_dim" in kwargs:
            pos_embed_dim = kwargs['pos_embed_dim']
            self.pos_embed_type = 'concat' # concat on embeddings 
            kwargs['embed_dim'] += pos_embed_dim
        else:
            pos_embed_dim = embed_dim
            self.pos_embed_type = 'add'

        kwargs.pop('pos_embed_dim', None)

        self.downstream = kwargs['downstream']
        del kwargs['downstream']
        super(VisionTransformer, self).__init__(**kwargs)

        norm_layer = kwargs['norm_layer']
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(kwargs['embed_dim'])

            # del self.norm  # remove the original norm
        
        self.patch_embed = PatchEmbed(kwargs['img_size'], kwargs['patch_size'], kwargs['in_chans'], embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_snippet = num_snippet
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, pos_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.pos_embed.requires_grad_(False)
        self.mask_channel_token = nn.Parameter(torch.zeros(1, kwargs['in_chans'], embed_dim)) #, requires_grad=False # other lab data requires_grad=True
        print("sin cos")

        self.dec_kwargs = dec_kwargs
        # decoder for SFT
        if dec_kwargs is not None:
            decoder_embed_dim = dec_kwargs['decoder_embed_dim']
            self.decoder_embed = nn.Linear(kwargs['embed_dim'], decoder_embed_dim, bias=True)
            
            if "decoder_pos_embed_dim" in dec_kwargs:
                decoder_pos_embed_dim = dec_kwargs["decoder_pos_embed_dim"]
                self.decoder_pos_embed_type = 'concat' # concat on embeddings 
                decoder_embed_dim += decoder_pos_embed_dim
            else:
                decoder_pos_embed_dim = decoder_embed_dim
                self.decoder_pos_embed_type = 'add'
            
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_pos_embed_dim), requires_grad=False)  # fixed sin-cos embedding
                
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, dec_kwargs['decoder_num_heads'], kwargs['mlp_ratio'], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(dec_kwargs['decoder_depth'])])

            self.decoder_norm = norm_layer(decoder_embed_dim)

        if self.num_snippet > 0:
            
            if self.downstream == 'RUL': #False: #
                if dec_kwargs is not None:
                    self.head = nn.Linear((kwargs['embed_dim'] + decoder_embed_dim) * (self.num_snippet-1), kwargs['num_classes'])
                else:
                    self.head = nn.Linear(kwargs['embed_dim'] * (self.num_snippet-1), kwargs['num_classes'])

            else:
                self.head = nn.Linear(kwargs['embed_dim'], kwargs['num_classes'])
        

        self.initialize_weights()


    def process_RUL_multi_feature(self, x_all):
        
        if self.dec_kwargs is not None:
            x = x_all.clone()
            x = self.norm(x)
            dec_feat = self.multi_snippet_forward_decoder(x)

        if self.dec_kwargs is not None:
            bsz, _, decoder_embed_dim = dec_feat.shape
            dec_feat = dec_feat[:, 1:, :].reshape(bsz, self.num_snippet, self.patch_embed.num_patches, decoder_embed_dim).mean(dim=2)
            dec_feat = self.decoder_norm(dec_feat)

        bsz, _, embed_dim = x_all.shape
        x_all = x_all[:, 1:, :].reshape(bsz, self.num_snippet, self.patch_embed.num_patches, embed_dim).mean(dim=2)
        x_all = self.fc_norm(x_all)

        if self.dec_kwargs is not None:
            x_all = torch.cat((x_all, dec_feat), dim=2)

        _, _, hidden_dim = x_all.shape
        x_diff_term = x_all[:, 1:, :] -  x_all[:, :-1, :]
        outcome = x_diff_term.reshape(bsz, (self.num_snippet-1) * hidden_dim)

        return outcome

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.mask_channel_token, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)

        if self.num_snippet > 0:
            
            if isinstance(self.head, nn.Sequential):
                for layer in self.head:
                    if hasattr(layer, 'weight'):
                        trunc_normal_(layer.weight, std=.02)
            else:
                trunc_normal_(self.head.weight, std=.02)
                nn.init.constant_(self.head.bias, 0)

    def multi_snippet_forward_features(self, x, mask_channel=None):
        
        for i in range(len(x)):

            if mask_channel is not None:
                x[i] = x[i] * (1 - mask_channel)
            x[i] = self.patch_embed(x[i])
            
            if mask_channel is not None:
                x[i] = x[i] + (mask_channel * self.mask_channel_token).sum(dim=1).unsqueeze(1)
            
            if self.pos_embed_type == 'add':
                x[i] = x[i] + self.pos_embed[:, 1:, :]
            else:
                x[i] = torch.cat([x[i], self.pos_embed[:, 1:, :].repeat(x[i].shape[0], 1, 1)], dim=2)


        if self.pos_embed_type == 'add':
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = torch.cat([self.cls_token, self.pos_embed[:, :1, :]], dim=2)
            
        cls_tokens = cls_token.expand(x[i].shape[0], -1, -1)
        x_all = torch.cat([cls_tokens]+x, dim=1)
        
        for blk in self.blocks:
            x_all = blk(x_all)

        if self.global_pool:
            
            if self.downstream == 'RUL': 
                outcome = self.process_RUL_multi_feature(x_all)
            else:
                bsz, _, embed_dim = x_all.shape
                x_all = x_all[:, 1:, :].reshape(bsz, self.num_snippet, self.patch_embed.num_patches, embed_dim).mean(dim=2)
                x_all = self.fc_norm(x_all)
                outcome = x_all.mean(dim=1)
            
        else:
            x_all = self.norm(x_all)
            outcome = x_all[:, 0]

        return outcome
        
    def forward_features(self, x, mask_channel=None):
        B = x.shape[0]

        if mask_channel is not None:
            # here mask_channel has shape (N,C,1), thus no need to use .unsqueeze(-1) like models_mae
            x = x * (1 - mask_channel)

        x = self.patch_embed(x)

        if mask_channel is not None:
            # here mask_channel has shape (N,C,1), thus no need to use .unsqueeze(-1) like models_mae
            x = x + (mask_channel * self.mask_channel_token).sum(dim=1).unsqueeze(1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed_type == 'add':
            x = x + self.pos_embed
        else:
            x = torch.cat([x, self.pos_embed.repeat(x.shape[0], 1, 1)], dim=2)
            
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def multi_snippet_forward_decoder(self, x_all):
        
        x_all = self.decoder_embed(x_all)
        cls_feature = x_all[:, :1, :]
        x_list = []
        L = (x_all.shape[1] - 1) // self.num_snippet
        x_feature_list = []
        lst = 1
        for i in range(self.num_snippet):
            
            x = x_all[:, lst: lst+L, :]
            lst += L

            if self.pos_embed_type == 'add':
                x = x + self.decoder_pos_embed[:, 1:, :]
            else:
                x = torch.cat([x, self.decoder_pos_embed[:, 1:, :].repeat(x.shape[0], 1, 1)], dim=2)
            x_feature_list.append(x)

        if self.pos_embed_type == 'add':
            cls_feature = cls_feature + self.decoder_pos_embed[:, :1, :]
        else:
            cls_feature = torch.cat([cls_feature, self.decoder_pos_embed[:, :1, :].repeat(cls_feature.shape[0], 1, 1)], dim=2)
            
        x = torch.cat([cls_feature]+x_feature_list, dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        return x
        
    def forward(self, x, mask_channel=None):
        if self.num_snippet > 0:
            x = self.multi_snippet_forward_features(x, mask_channel)
        else:
            x = self.forward_features(x, mask_channel)
        x = self.head(x)
        return x

    
    def SFT_forward_encoder(self, x, unavaliable_channel):
        
        B = x.shape[0]

        if unavaliable_channel is not None:
            # here unavaliable_channel has shape (N,C)
            x = x * (1 - unavaliable_channel.unsqueeze(-1))

        x = self.patch_embed(x)

        if unavaliable_channel is not None:
            x = x + (unavaliable_channel.unsqueeze(-1) * self.mask_channel_token).sum(dim=1).unsqueeze(1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed_type == 'add':
            x = x + self.pos_embed
        else:
            x = torch.cat([x, self.pos_embed.repeat(x.shape[0], 1, 1)], dim=2)

        for blk in self.blocks:
            x = blk(x)

        return x


    def SFT_forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        if self.decoder_pos_embed_type == 'add':
            x = x + self.decoder_pos_embed
        else:
            x = torch.cat([x, self.decoder_pos_embed.repeat(x.shape[0], 1, 1)], dim=2)
                
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def SFT_forward_loss(self, snippets, pred, unavaliable_channel):
        """
        snippets: [N, C, L]
        pred: [N, L', p * C]
        unavaliable_channel: [N, C], 0 is avaliable, 1 is unavaliable, some channels of lab data are unavaliable
        """
        chan = snippets.shape[1]
        l = self.patch_embed.num_patches
        p = self.patch_embed.patch_size
        assert pred.shape[1] == l

        target = snippets.reshape(shape=(snippets.shape[0], chan, l, p)).transpose(1, 2)
        pred = pred.reshape(shape=(pred.shape[0], l, chan, p))
        loss = (pred - target) ** 2
        total_mask = (1 - unavaliable_channel).unsqueeze(1).unsqueeze(3).repeat(1,l,1,p)
        loss_batch = (loss * total_mask).sum(dim=(1,2,3)) / total_mask.sum(dim=(1,2,3))
        loss = loss_batch.mean()

        return loss, loss_batch

    def SFT_forward(self, input, target, unavaliable_channel=None):
        latent = self.SFT_forward_encoder(input, unavaliable_channel)
        pred = self.SFT_forward_decoder(latent)  # [N, L, p*p*3]
        loss, loss_batch = self.SFT_forward_loss(target, pred, unavaliable_channel)
        return loss

def vit_base_patch16_depth1(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patch16(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patch16_depth3(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=192, depth=3, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_tiny_patch16_depth3_SFT(in_chans, **kwargs):
    dec_kwargs = {"decoder_embed_dim": 128, 
                "decoder_depth": 2, 
                "decoder_num_heads": 4}
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=192, depth=3, num_heads=3, mlp_ratio=4, qkv_bias=True,
        dec_kwargs=dec_kwargs, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_half_patch16(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=96, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def vit_half_patch16_with_decoder(in_chans, **kwargs):
    dec_kwargs = {"decoder_embed_dim": 64, 
                "decoder_depth": 4, 
                "decoder_num_heads": 4, 
                "decoder_pos_embed_dim": 8}
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=96, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True,
        dec_kwargs=dec_kwargs,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def vit_base_patch16(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch32(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=32, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch32_halfembed(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=32, in_chans=in_chans, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch32_quarterembed(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=32, in_chans=in_chans, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch32_tinyembed(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=32, in_chans=in_chans, embed_dim=96, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=16, in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(in_chans, **kwargs):
    model = VisionTransformer(
        patch_size=14, in_chans=in_chans, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model