import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from models_train.layers.bvae import VAE
from models_train.layers.cross_modal_fusion import CrossModalFusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Parameter reset for initializing weights
def reset_parameters(named_parameters):
    for name, p in named_parameters:
        if "weight" in name:
            nn.init.xavier_normal_(p)
        if "bias" in name:
            nn.init.constant_(p, 0.0)

def scaled_dot_product_attention(q, k, v, dk, attention_mask=None, dropout=0.2, training=True):

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)
    

    if attention_mask is not None:
        scores = scores + ((1 - attention_mask) * -1e5)  # Masking to ignore certain positions (padding or future tokens)
    
    #  get attention probabilities
    scores = F.softmax(scores, dim=-1)
    
    #  dropout for regularization)
    scores = F.dropout(scores, p=dropout, training=training)
    
    return scores, torch.matmul(scores, v)

# Multi-Headed  
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, attention_size, num_heads=8, input_size=None, dropout=0.1):
        '''
        Multi-Headed Dot-product Self-Attention Module.
        '''
        super(MultiHeadSelfAttention, self).__init__()
        if input_size is None:
            input_size = attention_size
        
        self.dk = input_size // num_heads  # Head dimension
        self.num_heads = num_heads
        
        # Linear layer to project input into query, key, and value (Q, K, V)
        self.kqv = nn.Linear(input_size, 3 * attention_size, bias=False)  
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        self.head_projection = nn.Linear(attention_size, attention_size)
        
        self.outt = nn.Sequential(nn.Linear(attention_size, 1), nn.Sigmoid())  # Output layer (scalar score)
        
        # Reset parameters
        reset_parameters(self.named_parameters())

    def forward(self, x, attention_mask=None):
        '''
        Args:
            x: [seq_len, batch_size, input_size] Input tensor (e.g., feature map from CNN)
            attention_mask: Optional attention mask (to ignore certain positions, like padding)
        Returns:
            out: Attention-weighted output values
            scores: Attention scores
        '''
        if attention_mask is not None and len(attention_mask.size()) == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  #  mask is shaped
        
        k, q, v = self.kqv(x).chunk(3, dim=-1)  # (seq_len, batch_size, attention_size)

        k = k.view(k.shape[0], self.num_heads, -1, self.dk)  # [seq_len, num_heads, batch_size, head_dim]
        q = q.view(q.shape[0], self.num_heads, -1, self.dk)
        v = v.view(v.shape[0], self.num_heads, -1, self.dk)

        scores, out = scaled_dot_product_attention(q, k, v, self.dk, attention_mask=attention_mask, dropout=self.dropout_layer.p, training=self.training)
        
        scores = self.dropout_layer(scores)
        
        out = out.view(out.shape[0], -1)  # [seq_len, attention_size]
        out = self.head_projection(out)

        out = self.outt(out)  # Output score for each frame
        return out, scores

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size)  # Attention over fused features
        self.vae = VAE(input_size, hidden_size, num_layers)
        self.fusion_module = CrossModalFusion(video_dim=512, text_dim=512, hidden_size=hidden_size, heads=8)

    def forward(self, video_feats, text_feats, mask_fused=None, uniform=False):
        """
        Args:
            video_feats: Tensor of shape [B, N_video, 1024]
            text_feats: Tensor of shape [B, N_text, 768]
            mask_fused: Optional mask of shape [B, N_video + N_text]
            uniform: If True, skip attention and use uniform weighting
        Returns:
            scores: Attention weights over input sequence
            h_mu: VAE latent mean
            h_log_variance: VAE latent log-variance
            decoded_features: Reconstructed features from VAE
        """
        print(f"Input shapes - Video: {video_feats.shape}, Text: {text_feats.shape}")  # Debug
    
        # Fuse the modalities
        if video_feats.dim() == 2: 
            video_feats = video_feats.unsqueeze(0)  # [1, N_video, D]
        if text_feats.dim() == 2:
             text_feats = text_feats.unsqueeze(0)  # [1, N_text, D]

        fused_video, fused_text = self.fusion_module(video_feats, text_feats, mask_fused) #[B, N_video, D]

        
        # use text-guided video
        fused = fused_video.permute(1, 0, 2) # rearranges tensor dimensions:  from [B, N_video, D] --> [N_video, B, D]

        if not uniform:
            scores, weighted_features = self.attn(fused)  # [seq_len, B, D]
            weighted_features = fused * scores.view(-1, 1, 1)  # Elementwise weighting
        else:
            scores = None
            weighted_features = fused

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)
        return scores, h_mu, h_log_variance, decoded_features,fused_video, fused_text



if __name__ == '__main__':
    pass
