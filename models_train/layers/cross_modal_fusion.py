import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dims,
                 k_dims=None,
                 v_dims=None,
                 h_dims=None,
                 o_dims=None,
                 heads=8,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)

        self.reset_parameters()

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.m):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.q(q).transpose(0, 1).contiguous()
        k = self.k(k).transpose(0, 1).contiguous()
        v = self.v(v).transpose(0, 1).contiguous()

        b = q.size(1) * self._heads

        q = q.view(-1, b, self._head_dims).transpose(0, 1)
        k = k.view(-1, b, self._head_dims).transpose(0, 1)
        v = v.view(-1, b, self._head_dims).transpose(0, 1)

        att = torch.bmm(q, k.transpose(1, 2)) / self._head_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            mask = mask.repeat_interleave(self._heads, dim=0)
            att += mask

        att = att.softmax(-1)

        if self.drop1 is not None:
            att = self.drop1(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)

        if self.drop2 is not None:
            m = self.drop2(m)

        return m


class FFN(nn.Module):
    def __init__(self, num_input, p=0.1, ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(num_input, num_input * ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(num_input * ratio, num_input)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, p=dropout, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, p=dropout)

    def forward(self, x, mask=None):
        x_res = x
        x = self.norm1(x)
        x = x_res + self.attn(x, mask=mask)

        x_res2 = x
        x = self.norm2(x)
        x = x_res2 + self.ffn(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, num_hidden, dropout_attn=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(num_hidden)
        self.norm_kv = nn.LayerNorm(num_hidden)
        self.cross_attn = MultiHeadAttention(num_hidden, p=dropout_attn)

        self.norm_out_video = nn.LayerNorm(num_hidden)
        self.ffn_video = FFN(num_hidden, p=dropout_attn, ratio=4)

        self.norm_out_text = nn.LayerNorm(num_hidden)
        self.ffn_text = FFN(num_hidden, p=dropout_attn, ratio=4)

    def forward(self, video, text, mask=None):
        q = self.norm_q(video)
        kv = self.norm_kv(text)

        fused = self.cross_attn(q, k=kv, v=kv, mask=mask)
        video_out = video + fused

        x = self.norm_out_video(video_out)
        video_out = video_out + self.ffn_video(x)

        text_out = text + self.ffn_text(self.norm_out_text(text))

        return video_out, text_out
    

class CrossModalFusion(nn.Module):
    MAX_LEN = 1024

    def __init__(self, video_dim, text_dim, hidden_size=512,dropout=0.1, n_layers=3, heads=8):
        super(CrossModalFusion, self).__init__()
        self.proj_video = nn.Linear(video_dim, hidden_size)
        self.proj_text = nn.Linear(text_dim, hidden_size)

        self.pos_embed_video = nn.Parameter(torch.zeros(1, self.MAX_LEN, hidden_size))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, self.MAX_LEN, hidden_size))
        nn.init.trunc_normal_(self.pos_embed_video, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_text, std=0.02)

        # Self-attention for text (MHSA)
        self.text_self_attn = SelfAttentionBlock(hidden_size, heads=heads, dropout=dropout)

        # Cross-modal attention layers
        self.multiway_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_size,dropout_attn=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, video_feats, text_feats, mask_text=None):
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(0)
        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0)

        if video_feats.size(0) != text_feats.size(0):
            raise ValueError("Mismatched batch sizes between video and text features.")

        v_proj = self.proj_video(video_feats)
        t_proj = self.proj_text(text_feats)

        v_proj = v_proj + self.pos_embed_video[:, :v_proj.size(1), :]
        t_proj = t_proj + self.pos_embed_text[:, :t_proj.size(1), :]

        # Apply text self-attention (MHSA)
        t_proj = self.text_self_attn(t_proj, mask=mask_text)

        # Cross-attention layers
        for layer in self.multiway_layers:
            v_proj, t_proj = layer(v_proj, t_proj, mask=mask_text)

        return v_proj, t_proj

