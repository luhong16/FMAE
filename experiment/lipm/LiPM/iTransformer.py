import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

EPS = 1e-6
INF = 1e9


def is_nan(ts: torch.Tensor, info):
    if torch.isnan(ts.sum()):
        raise Exception(f'nan, {info}')
    

class ScaleDotProduct(nn.Module):
    def __init__(self, scale=1) -> None:
        super(ScaleDotProduct, self).__init__()
        self.scale = scale
    
    def forward(self, Q, K, V, mask=None):
        attn = torch.matmul(Q / self.scale, K.transpose(2, 3))  
        if mask is not None:
            attn = attn.masked_fill(mask==0, -INF)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  

        return out, attn
    

class RoPE(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super(RoPE, self).__init__()
        self.freqs_cis = self.precompute_freqs_cis(d_model, max_len)

    def precompute_freqs_cis(self, d_model, max_len, theta=1000.0):
        freqs = 1.0/(theta ** (torch.arange(0, d_model, 2)[: (d_model//2)].float()/d_model))
       
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs).float()  

        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 

        return freqs_cis  
    
    def broadcast(self, freqs_cis, x):
        dim_n = len(x.shape)
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])  
        shape = [d if i == 1 or i == dim_n - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape).to(x.device)
    
    def forward(self, x):
        with autocast(enabled=False):
            x_= x.reshape(*x.shape[:-1], -1, 2).float()

            x_ = torch.view_as_complex(x_)
            freqs_cis = self.broadcast(self.freqs_cis[: x.shape[1]], x_)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return x_out.type_as(x)
    

class IrregularRoPE(nn.Module):
    def __init__(self, d_model) -> None:
        super(IrregularRoPE, self).__init__()
        self.thetas = self.precompute_thetas(d_model)

    def precompute_thetas(self, d_model, theta=1000.0):
        thetas = 1.0/(theta ** (torch.arange(0, d_model, 2)[: (d_model//2)].float()/d_model))
        return thetas

    def broadcast(self, x, t):
        assert self.thetas.shape[0] == (x.shape[-1])  
        shape = [d if i!=2 else 1 for i, d in enumerate(x.shape)]

        freqs_cis = self.compute_freqs_cis(t, shape)

        return freqs_cis.to(x.device)

    def compute_freqs_cis(self, t, shape):
        f_shape = list(t.shape)
        f_shape.append(self.thetas.shape[0])
        freqs = torch.outer(t.flatten(), self.thetas.to(t.device)).float().reshape(*f_shape) 
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
        return freqs_cis.view(*shape)

    def forward(self, x, t):
        with autocast(enabled=False):
            x_= x.reshape(*x.shape[:-1], -1, 2).float()

            x_ = torch.view_as_complex(x_)
            freqs_cis = self.broadcast(x_, t)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
        return x_out.type_as(x)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(config.n_head, config.q_k_dim, config.v_dim, config.d_model)
        self.feed_forward = FeedForward(config.d_model, config.d_model * 4, config.dp)
        if config.norm == 'rsm':
            self.attention_norm = RMSNorm(config.d_model)
            self.ffn_norm = RMSNorm(config.d_model)
        else:
            self.attention_norm = nn.LayerNorm(config.d_model)
            self.ffn_norm = nn.LayerNorm(config.d_model)
        self.pre_norm = config.pre_norm
        self.dropout = nn.Dropout(config.dp)

    def forward(self, x, mask=None, k=None, v=None):
        if self.pre_norm:
            q = self.attention_norm(x)
        else:
            q = x

        if k is None:
            k, v, = q, q
        h = x + self.dropout(self.attention(q, k, v, mask))

        if not self.pre_norm:
            h = self.attention_norm(h)
        
        if self.pre_norm:
            _h = self.ffn_norm(h)
        else:
            _h = h
        
        out = h + self.feed_forward(_h)

        if not self.pre_norm:
            out = self.ffn_norm(out)
        return out


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) 

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, q_k_dim, v_dim, d_model):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.q_k_dim = q_k_dim 
        self.v_dim = v_dim

        self.Q_weight = nn.Linear(d_model, self.n_head * self.q_k_dim, bias=False)
        self.K_weight = nn.Linear(d_model, self.n_head * self.q_k_dim, bias=False)
        self.V_weight = nn.Linear(d_model, self.n_head * self.v_dim, bias=False)
        
        self.out_weight = nn.Linear(self.n_head * self.v_dim, d_model, bias=False)

        self.attention = ScaleDotProduct(self.q_k_dim**0.5)
        self.pe = RoPE(q_k_dim)

    def forward(self, q, k, v, mask=None):
        batch_n, q_l, k_v_l = q.shape[0], q.shape[1], k.shape[1]

        Q = self.Q_weight(q).view(batch_n, q_l, self.n_head, self.q_k_dim)
        K = self.K_weight(k).view(batch_n, k_v_l, self.n_head, self.q_k_dim)
        V = self.V_weight(v).view(batch_n, k_v_l, self.n_head, self.v_dim)

        Q, K = self.pe(Q), self.pe(K)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) 
            
        out_Q, attn = self.attention(Q, K, V, mask)
        out_Q = out_Q.transpose(1, 2).contiguous().view(batch_n, q_l, -1)
        out_Q = self.out_weight(out_Q)

        return out_Q
    

class iBlock(nn.Module):
    def __init__(self, config):
        super(iBlock, self).__init__()
        self.n_head = 8
        self.q_k_dim = 32
        self.v_dim = 32

        self.Q_weight = nn.Linear(config.n_var, self.n_head * self.q_k_dim, bias=False)
        self.K_weight = nn.Linear(config.n_var, self.n_head * self.q_k_dim, bias=False)
        self.V_weight = nn.Linear(config.n_var, self.n_head * self.v_dim, bias=False)

        self.out_weight = nn.Linear(self.n_head * self.v_dim, config.d_model, bias=False)

        self.attention = ScaleDotProduct(self.q_k_dim**0.5)
        self.pe = IrregularRoPE(self.q_k_dim)

        self.ffn_norm = nn.LayerNorm(self.n_head * self.v_dim)
        self.ffn = FeedForward(self.n_head * self.v_dim, 4*self.n_head * self.v_dim)
        self.emb_token = nn.Parameter(torch.zeros((1, 1, 1, 2)))
        
    def forward(self, x, t, mask):
        batch_n, pn, pl = x.shape[0], x.shape[1], x.shape[2]
        emb_token = self.emb_token.expand(batch_n, pn, 1, -1)
        _x = torch.cat((emb_token, x), dim=2)
        Q = self.Q_weight(_x).view(batch_n*pn, pl+1, self.n_head, self.q_k_dim)
        K = self.K_weight(_x).view(batch_n*pn, pl+1, self.n_head, self.q_k_dim)
        V = self.V_weight(_x).view(batch_n*pn, pl+1, self.n_head, self.v_dim)

        t = t.view(batch_n*pn, -1)
        Q, K = self.pe(Q, t), self.pe(K, t) 
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        if mask is not None:
            if len(mask.shape) == 3:
                mask_mat = mask.unsqueeze(-1)*mask.unsqueeze(-2)
            else:
                mask_mat = mask
            mask_mat = mask_mat.view(batch_n*pn, pl+1, pl+1)
            mask_mat = mask_mat.unsqueeze(1) 
        _Q, _ = self.attention(Q, K, V, mask_mat)
        last_Q = _Q[:, :, 0, :]  
        last_Q = last_Q.contiguous().view(batch_n, pn, -1)  
        last_Q = last_Q + self.ffn(last_Q)
        out = self.out_weight(self.ffn_norm(last_Q))  

        return out 


class iBlockRaw(nn.Module):
    def __init__(self, config, n_var=1):
        super(iBlockRaw, self).__init__()
        self.n_head = 4
        self.q_k_dim = 4 * n_var
        self.v_dim = 4 * n_var
        self.n_var = n_var

        self.Q_weight = nn.Linear(n_var, self.n_head * self.q_k_dim, bias=False)
        self.K_weight = nn.Linear(n_var, self.n_head * self.q_k_dim, bias=False)
        self.V_weight = nn.Linear(n_var, self.n_head * self.v_dim, bias=False)

        self.out_weight = nn.Linear(self.n_head * self.v_dim, n_var, bias=False)
        self.ffn = FeedForward(self.n_head * self.v_dim, self.n_head * self.v_dim * 4)

        self.attention = ScaleDotProduct(self.q_k_dim**0.5)
        self.pe = IrregularRoPE(self.q_k_dim)

    def forward(self, x, t, mask, attn=False):
        if x.shape[-1] != self.n_var:
            if self.n_var == 1:
                x = x.unsqueeze(-1)
            else:
                raise Exception(f'n_var is {self.n_var}, but shape of x is {x.shape}')
        batch_n, pn, pl = x.shape[0], x.shape[1], x.shape[2]
        Q = self.Q_weight(x).view(batch_n*pn, pl, self.n_head, self.q_k_dim)
        K = self.K_weight(x).view(batch_n*pn, pl, self.n_head, self.q_k_dim)
        V = self.V_weight(x).view(batch_n*pn, pl, self.n_head, self.v_dim)

        t = t.view(batch_n*pn, -1)
        Q, K = self.pe(Q, t), self.pe(K, t) 
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if mask is not None:
            if len(mask.shape) == 3:
                mask_a = mask.unsqueeze(-1)
                mask_b = mask.unsqueeze(-2)
                mask_mat = mask_a*mask_b
            else:
                mask_mat = mask
            mask_mat = mask_mat.view(batch_n*pn, pl, pl)
            mask_mat = mask_mat.unsqueeze(1)  
        _Q, _attn = self.attention(Q, K, V, mask_mat) 
        _Q = _Q.transpose(1, 2).contiguous().view(batch_n, pn, pl, -1)  
        out_Q = _Q + self.ffn(_Q)
        out = self.out_weight(out_Q)
        if attn:
            return out, _attn, _Q, out_Q, Q, K, V
        return out       


class iTransformer(nn.Module):
    def __init__(self, config) -> None:
        super(iTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(TransformerBlock(config))
        self.to_embedding = iBlock(config)
        self.output = nn.Linear(config.d_model, config.emb_dim, bias=False)

    def forward(self, x: torch.Tensor, t, t_mask=None, patch_mask=None):
        h = self.to_embedding(x.contiguous(), t, t_mask)
        patch_mask_mat = patch_mask.unsqueeze(-1)*patch_mask.unsqueeze(-2)
        for layer in self.layers:
            h = layer(h, patch_mask_mat)

        out = self.output(h) 
        return out