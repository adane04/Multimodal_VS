import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        B_q, N_q, D_q = q.size()
        B_k, N_k, D_k = k.size()
        B_v, N_v, D_v = v.size()
        H = self.num_heads
        d_k = self.d_k

        def shape(x, B, N, D):
            return x.view(B, N, H, D // H).transpose(1, 2)  # [B, H, N, d_k]

        q = shape(self.q_linear(q), B_q, N_q, D_q)
        k = shape(self.k_linear(k), B_k, N_k, D_k)
        v = shape(self.v_linear(v), B_v, N_v, D_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, N_q, N_k]

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # [B, H, N_q, d_k]
        context = context.transpose(1, 2).contiguous().view(B_q, N_q, D_q)
        return self.out(context)

class CrossModalFusion(nn.Module):
    def __init__(self, video_dim, text_dim, hidden_size=512, num_heads=8, dropout=0.1):
        super(CrossModalFusion, self).__init__()
        self.proj_video = nn.Linear(video_dim, hidden_size)
        self.proj_text = nn.Linear(text_dim, hidden_size)
        self.attn_fusion = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, video_feats, text_feats, mask_text=None):
        """
        Inputs:
            video_feats: [B, N_video, D_video]
            text_feats: [B, N_text, D_text]
            mask_text: [B, N_text] (optional)
        Returns:
            language_guided_video: [B, N_video, hidden_size]
            text_proj: [B, N_text, hidden_size] (optional: can be returned for loss/etc.)
        """
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(0)
        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0)

        B_v, N_video, _ = video_feats.shape
        B_t, N_text, _ = text_feats.shape
        if B_v != B_t:
            raise ValueError(f"Batch size mismatch: video_feats ({B_v}) vs text_feats ({B_t})")

        # Project to shared space
        v_proj = self.proj_video(video_feats)  # Q: [B, N_video, hidden]
        t_proj = self.proj_text(text_feats)    # K/V: [B, N_text, hidden]

        # video attends to language
        attended_video = self.attn_fusion(v_proj, t_proj, t_proj, mask=mask_text)
        output = self.norm(v_proj + attended_video)

        return output, t_proj  # 


if __name__ == '__main__':
    pass
 # example usage
 # fusion_module = CrossModalFusion(video_dim=1024, text_dim=768, hidden_size=512, num_heads=8)

 # fused_video, fused_text = fusion_module(video_feats, text_feats, mask_fused)
