import torch
import torch.nn as nn
from LiPM.iTransformer import iBlockRaw, iTransformer, MultiHeadAttention, FeedForward, RMSNorm
import numpy as np
EPS = 1e-6


class BatteryNet(nn.Module):
    def __init__(self, config):
        super(BatteryNet, self).__init__()
        self.backbone = iTransformer(config)
        if 'pretrain' in config.task:
            self.config = config

            self.inter_head = Seq2SeqHead(config)
            self.Q_head = Seq2SeqHead(config, use_Q=True)

            self.mse_loss = nn.MSELoss()

            self.patch_len = config.patch_len

            self.cur_idx_inter = 0 
            self.idx_pool = np.arange(config.max_iter)  
            np.random.shuffle(self.idx_pool)
    
    def get_emb(self, seq, t, tm, pm):
        self.backbone.eval()
        with torch.no_grad():
            emb = self.backbone(seq, t, tm, pm)
        return emb.detach()

    def get_mask(self, x, pm, tm):
        _patch_rand = torch.rand(x.shape[0], x.shape[1])  
        _channel_rand = torch.rand(x.shape[0], x.shape[1], x.shape[3])  
        mask_patch = (_patch_rand < self.config.patch_ratio).long().to(x.device)
        mask_channel = (_channel_rand < self.config.channel_ratio).long().to(x.device)
        forward_mask = pm & ~mask_patch  
        loss_mask_patch = pm.unsqueeze(-1) & (mask_patch.unsqueeze(-1) | mask_channel)
        loss_mask = loss_mask_patch.unsqueeze(2) & tm.unsqueeze(3)
        masked_x = x * (~loss_mask)
        return forward_mask, loss_mask, masked_x


    def cal_MAE_loss(self, pred, y, loss_mask, t_mask):
        shape = y.shape
        pred = pred.view(shape[0], -1, shape[-1])
        y = y.view(shape[0], -1, shape[-1])
        loss_mask = loss_mask.view(shape[0], -1, shape[-1])
        t_mask = t_mask.reshape(shape[0], -1)
        loss = torch.mul(pred - y, loss_mask).pow(2).mean()
        mae_loss = torch.mul(torch.abs(pred - y), loss_mask).mean()

        return loss, mae_loss.detach()


    def cal_Q_loss(self, pred, y, t_mask):
        shape = y.shape
        pred = pred.view(shape[0], -1, shape[-1])
        y = y.view(shape[0], -1, shape[-1])
        t_mask = t_mask.reshape(shape[0], -1)
        loss = torch.mul(pred - y, t_mask.unsqueeze(-1)).pow(2).sum() / (t_mask.sum()*pred.shape[-1] + EPS) 
        mae_loss = torch.mul(torch.abs(pred - y), t_mask.unsqueeze(-1)).sum() / (t_mask.sum()*pred.shape[-1] + EPS)

        return loss, mae_loss.detach()

    def forward(self, batch):
        vc, QdQc, t, tm, pm = batch
        forward_mask, loss_mask, masked_vc = self.get_mask(vc, pm, tm[:, :, 1:])
        emb_mask = self.backbone(masked_vc, t, tm, forward_mask)
        emb_full = self.backbone(vc, t, tm, pm)
        t = t[:, :, 1:]
        tm = tm[:, :, 1:]
        pred_vc = self.inter_head(emb_mask, t, tm, forward_mask)
        pred_Q = self.Q_head(emb_full, t, tm, pm)
        loss_MAE, MAE_mae = self.cal_MAE_loss(pred_vc, vc, loss_mask, tm)
        loss_Q, Q_mae = self.cal_Q_loss(pred_Q, QdQc, tm)
        loss = self.config.weight_MAE*loss_MAE + self.config.weight_Q*loss_Q
        loss = loss / (self.config.weight_MAE + self.config.weight_Q)  

        loss_dict = {'total': loss.item(), 
                     'MMAE_mse': loss_MAE.item(), 'MMAE_mae': MAE_mae.item(), 
                     'CIR_mse': loss_Q.item(), 'CIR_mae': Q_mae.item()}
        return loss, loss_dict


class Seq2SeqHead(nn.Module):
    def __init__(self, config, use_Q=False, n_var=2):
        super(Seq2SeqHead, self).__init__()
        self.kvs_merge = nn.Linear(config.emb_dim, config.down_dim)
        if use_Q:
            self.Q = nn.Parameter(torch.zeros((config.patch_num, config.down_dim)))
        else:
            self.Q = None
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(InterBlock(config))

        self.norm = nn.Sequential(
            RMSNorm(config.down_dim),
            nn.Linear(config.down_dim, config.down_dim, bias=False)
            )
        self.lin_out = nn.Linear(config.down_dim, config.patch_len*n_var)
        self.out_layers = nn.ModuleList()
        for i in range(2):
            self.out_layers.append(iBlockRaw(config, 2))

        self.n_var = n_var

    def forward(self, emb, t, t_m, patch_mask=None):
        kv = self.kvs_merge(emb)
        Q = self.Q.expand(emb.shape[0], -1, -1) if self.Q is not None else kv

        patch_mask_mat = patch_mask.unsqueeze(-1) * patch_mask.unsqueeze(-2)
        for layer in self.layers:
            Q = layer(Q, kv if self.Q is not None else Q, patch_mask_mat)
        
        h = self.norm(Q)  
        
        out = self.lin_out(h).view(h.shape[0], h.shape[1], -1, self.n_var)   
        mask_mat = t_m.unsqueeze(-1) * t_m.unsqueeze(-2)
        for layer in self.out_layers:
            out = layer(out, t, mask_mat)

        return out  


class InterBlock(nn.Module):
    def __init__(self, config) -> None:
        super(InterBlock, self).__init__()
        self.self_attn = MultiHeadAttention(config.down_n_head, config.down_dim//config.down_n_head, config.down_dim//config.down_n_head, config.down_dim)
        self.inter_attn = MultiHeadAttention(config.down_n_head, config.down_dim//config.down_n_head, config.down_dim//config.down_n_head, config.down_dim)
        self.feed_forward = FeedForward(config.down_dim, config.down_dim*4)
        self.kv_norm = RMSNorm(config.down_dim)
        self.q_norm = RMSNorm(config.down_dim)
        self.ffn_norm = RMSNorm(config.down_dim)

    def forward(self, q, kv, mask=None):
        _q = self.q_norm(q)
        out_q = q + self.self_attn(_q, _q, _q, mask)
        norm_q = self.q_norm(out_q)
        norm_kv = self.kv_norm(kv)
        h = out_q + self.inter_attn(norm_q, norm_kv, norm_kv, mask)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out

