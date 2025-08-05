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

from timm.models.vision_transformer import Block #PatchEmbed, 

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
        """
        mask_channels: (C). 0 is keep, 1 is mask (same with mask patch)
        """
        B, C, L = x.shape
        assert(L == self.snippet_size, f"Input snippet size ({L}) doesn't match model ({self.snippet_size}).")
        x = self.proj(x)
        x = x.transpose(1, 2)  # BC'L' -> BL'C'
        x = self.norm(x)

        return x
    
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=128, patch_size=8, in_chans=6, out_chans=None,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_snippet=0, **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        print("kwargs", kwargs)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.mask_snippet_num = kwargs["mask_snippet_num"]
        if "decoder_channel_id" in kwargs:
            self.decoder_channel_id = kwargs["decoder_channel_id"] 
        self.decoder_patch_embed = PatchEmbed(img_size, patch_size, len(self.decoder_channel_id), decoder_embed_dim)
        self.decoder_pad_type = kwargs["decoder_pad_type"]    #  mask_token or soc_current_mileage_embed
        num_patches = self.patch_embed.num_patches
        self.mask_channel_token = nn.Parameter(torch.zeros(1, in_chans, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if "pos_embed_dim" in kwargs:
            pos_embed_dim = kwargs["pos_embed_dim"]
            self.pos_embed_type = 'concat' # concat on embeddings 
            embed_dim += pos_embed_dim
        else:
            pos_embed_dim = embed_dim
            self.pos_embed_type = 'add'


        self.num_snippet = num_snippet
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, pos_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        if "decoder_pos_embed_dim" in kwargs:
            decoder_pos_embed_dim = kwargs["decoder_pos_embed_dim"]
            self.decoder_pos_embed_type = 'concat' # concat on embeddings 
            decoder_embed_dim += decoder_pos_embed_dim
        else:
            decoder_pos_embed_dim = decoder_embed_dim
            self.decoder_pos_embed_type = 'add'
        
        self.decoder_type = kwargs["decoder_type"]  #combine or separate
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_pos_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if out_chans is None:
            out_chans = in_chans

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * out_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        d_w = self.decoder_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(d_w.view([d_w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_channel_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_channel(self, x, mask_channel_ratio, unavaliable_channel):
        
        N, C, L = x.shape
        len_keep = int(C * (1 - mask_channel_ratio))

        noise = torch.rand(N, C, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, C], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # or
        mask = 1 - (1 - mask) * (1 - unavaliable_channel)

        x_masked = x * (1 - mask).unsqueeze(-1)
        #print("MASK", mask, mask.shape, x_masked.shape) #DEBUG
        return x_masked, mask

    def multi_snippet_forward_encoder(self, x, mask_patch_ratio, mask_channel_ratio, unavaliable_channel):

        # x should be list of snippets
        x[0], mask_channel = self.random_masking_channel(x[0], mask_channel_ratio, unavaliable_channel)
        for i in range(1, len(x)):
            x[i] = x[i] * (1 - mask_channel).unsqueeze(-1)
            
        masks = []
        ids_restores = []
        for i in range(len(x)):
            x[i] = self.patch_embed(x[i])
            x[i] = x[i] + (mask_channel.unsqueeze(-1) * self.mask_channel_token).sum(dim=1).unsqueeze(1)
            if self.pos_embed_type == 'add':
                x[i] = x[i] + self.pos_embed[:, 1:, :]
            else:
                x[i] = torch.cat([x[i], self.pos_embed[:, 1:, :].repeat(x[i].shape[0], 1, 1)], dim=2)
            x[i], _mask, _ids_restore = self.random_masking(x[i], mask_patch_ratio if i >= self.mask_snippet_num else 1.0)
            masks.append(_mask)
            ids_restores.append(_ids_restore)

        if self.pos_embed_type == 'add':
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = torch.cat([self.cls_token, self.pos_embed[:, :1, :]], dim=2)
            
        cls_tokens = cls_token.expand(x[i].shape[0], -1, -1)
        x_all = torch.cat([cls_tokens]+x, dim=1)

        # print('x_all', x_all.shape, cls_tokens.shape, self.pos_embed.shape)

        # apply Transformer blocks
        for blk in self.blocks:
            x_all = blk(x_all)
        x_all = self.norm(x_all)

        return x_all, masks, mask_channel, ids_restores


    def multi_snippet_forward_decoder(self, x_all, ids_restores, input_x, mask):
        
        x_all = self.decoder_embed(x_all)
        # print('decoder_x', x_all.shape)
        cls_feature = x_all[:, :1, :]
        x_list = []
        # 1 is CLS
        # assert (x_all.shape[1] - 1) % self.num_snippet == 0
        L = (x_all.shape[1] - 1) // self.num_snippet
        x_feature_list = []
        lst = 1
        for i in range(len(ids_restores)):
            
            ids_restore = ids_restores[i]
            if i >= self.mask_snippet_num:
                x = x_all[:, lst: lst+L, :]
                lst += L
            else:
                x = x_all[:, lst: lst, :]
            # append mask tokens to sequence
            if self.decoder_pad_type == "mask_token":
                mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                x = torch.cat([x, mask_tokens], dim=1)  # no cls token
                x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            else:
                x = torch.cat([x, torch.zeros(x.shape[0], ids_restore.shape[1] - x.shape[1], x.shape[2], device=x.device)], dim=1)
                x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
                mask_tokens_all = self.decoder_patch_embed(input_x[i][:, self.decoder_channel_id, :])
                x = x + mask_tokens_all * mask[i].unsqueeze(-1)

            if self.decoder_type == 'separate':
                x = torch.cat([cls_feature, x], dim=1)  # append cls token

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

                # remove cls & sep token
                x = x[:, 1:, :]
                x_list.append(x)
            else:
                if self.pos_embed_type == 'add':
                    x = x + self.decoder_pos_embed[:, 1:, :]
                else:
                    x = torch.cat([x, self.decoder_pos_embed[:, 1:, :].repeat(x.shape[0], 1, 1)], dim=2)
                x_feature_list.append(x)

        if self.decoder_type == 'combine':
            if self.pos_embed_type == 'add':
                cls_feature = cls_feature + self.decoder_pos_embed[:, :1, :]
            else:
                cls_feature = torch.cat([cls_feature, self.decoder_pos_embed[:, :1, :].repeat(cls_feature.shape[0], 1, 1)], dim=2)
                
            x = torch.cat([cls_feature]+x_feature_list, dim=1)

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)
            
            for i in range(self.num_snippet):
                x_list.append(x[:, 1+i*self.patch_embed.num_patches: 1+(i+1)*self.patch_embed.num_patches, :])

        return x_list


    def forward_encoder(self, x, mask_patch_ratio, mask_channel_ratio, unavaliable_channel):
        
        # mask channel 
        x, mask_channel = self.random_masking_channel(x, mask_channel_ratio, unavaliable_channel)
        
        # embed patches
        x = self.patch_embed(x)

        # add mask channel token
        x = x + (mask_channel.unsqueeze(-1) * self.mask_channel_token).sum(dim=1).unsqueeze(1)

        # add pos embed w/o cls token
        
        if self.pos_embed_type == 'add':
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = torch.cat([x, self.pos_embed[:, 1:, :].repeat(x.shape[0], 1, 1)], dim=2)
            
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_patch_ratio)
        
        # append cls token
        
        if self.pos_embed_type == 'add':
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        else:
            cls_token = torch.cat([self.cls_token, self.pos_embed[:, :1, :]], dim=2)

        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, mask_channel, ids_restore


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

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

    def forward_loss(self, snippets, pred, mask, mask_channel, unavaliable_channel):
        """
        snippets: [N, C, L]
        pred: [N, L', p * C]
        mask: [N, L'], 0 is keep, 1 is remove, 
        mask_channel: [N, C], 0 is keep, 1 is remove, 
        unavaliable_channel: [N, C], 0 is avaliable, 1 is unavaliable, some channels of lab data are unavaliable
        """
        chan = snippets.shape[1]
        l = self.patch_embed.num_patches
        p = self.patch_embed.patch_size
        assert pred.shape[1] == l

        target = snippets.reshape(shape=(snippets.shape[0], chan, l, p)).transpose(1, 2)
        pred = pred.reshape(shape=(pred.shape[0], l, chan, p))
        if self.norm_pix_loss:
            # in original code, target has shape same as pred
            mean = target.mean(dim=(-1,-2), keepdim=True)
            var = target.var(dim=(-1,-2), keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        
        total_mask = (1 - (1 - mask.unsqueeze(-1).unsqueeze(-1)) * (1 - mask_channel.unsqueeze(1).unsqueeze(-1))).repeat(1,1,1,p)
        total_mask = total_mask * (1 - unavaliable_channel).unsqueeze(1).unsqueeze(3)
        
        loss_batch = (loss * total_mask).sum(dim=(1,2,3)) / total_mask.sum(dim=(1,2,3))
        loss = loss_batch.mean()

        return loss, loss_batch

    def multi_snippet_forward_loss(self, snippets, preds, masks, mask_channel, unavaliable_channel):

        losses = []
        losses_batch = []
        for (snippet, pred, mask) in zip(snippets, preds, masks):
            loss, loss_batch = self.forward_loss(snippet, pred, mask, mask_channel, unavaliable_channel)
            # print("loss", loss, loss_batch)
            losses.append(loss)
            losses_batch.append(loss_batch)
        return torch.sum(torch.stack(losses), dim=0), torch.sum(torch.stack(losses_batch), dim=0)
    
    def forward(self, input, target, mask_patch_ratio=0.75, mask_channel_ratio=0, unavaliable_channel=None):
        if self.num_snippet > 0:
            # print('input', input[0][0, :, 0])
            input_x = input.copy()
            latent, mask, mask_channel, ids_restore = self.multi_snippet_forward_encoder(input, mask_patch_ratio, mask_channel_ratio, unavaliable_channel)
            
            # print('inputx', input[0][0, :, 0], input_x[0][0, :, 0], target[0][0, :, 0])
            pred = self.multi_snippet_forward_decoder(latent, ids_restore, input_x, mask)  
            loss, loss_batch = self.multi_snippet_forward_loss(target, pred, mask, mask_channel, unavaliable_channel)
        else:    
            latent, mask, mask_channel, ids_restore = self.forward_encoder(input, mask_patch_ratio, mask_channel_ratio, unavaliable_channel)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss, loss_batch = self.forward_loss(target, pred, mask, mask_channel, unavaliable_channel)
        return loss, pred, mask, loss_batch

def mae_vit_tiny_patch16_depth1(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=1, num_heads=3,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny_patch16_depth3(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=3, num_heads=3,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_tiny_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_half_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=96, depth=6, num_heads=3,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def mae_vit_base_patch16_dec512d1b_depth1(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=1, num_heads=12,
        decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch32_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch32_dec512d8b_quarterembed(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=32, embed_dim=192, depth=12, num_heads=6,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch32_dec512d8b_tinyembed(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=32, embed_dim=96, depth=12, num_heads=3,
        decoder_embed_dim=64, decoder_depth=8, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch32_dec512d8b_halfembed(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=32, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def debug(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=6, depth=2, num_heads=2,
        decoder_embed_dim=12, decoder_depth=2, decoder_num_heads=3,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
debug = debug
mae_vit_base_patch8 = mae_vit_base_patch8_dec512d8b
mae_vit_base_patch16_depth1 = mae_vit_base_patch16_dec512d1b_depth1
mae_vit_tiny_patch16_depth3 = mae_vit_tiny_patch16_depth3
mae_vit_tiny_patch16 = mae_vit_tiny_patch16
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch32 = mae_vit_base_patch32_dec512d8b
mae_vit_base_patch32_halfembed = mae_vit_base_patch32_dec512d8b_halfembed
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
