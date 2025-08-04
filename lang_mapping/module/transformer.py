import torch
import torch.nn as nn
import torch.nn.functional as F
from lang_mapping.utils.utils import rotary_pe_3d  
import math
import xformers.ops as xops

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def generate_subsequent_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) 
    mask = mask.bool()  # True/False
    mask = mask.masked_fill(mask, float('-inf')) 
    return mask

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_heads=8, 
        dim_feedforward=1024, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,
        coords_src: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, _ = src.shape
        
        q = self.W_q(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(src).view(B, S, self.n_heads, self.head_dim)

        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        
        attn = xops.memory_efficient_attention(q, k, v, p=self.dropout_attn.p if self.training else 0.0)
        attn = attn.reshape(B, S, self.d_model)
        
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        out = self.norm2(src2 + self.dropout_ff(ff))
        
        return out

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=input_dim,
                n_heads=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
                
        self.apply(init_weights_kaiming)
               
    def forward(
        self,
        visual_token: torch.Tensor,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        for layer in self.layers:
            visual_token = layer(src=visual_token, coords_src=coords)
 
        return visual_token

class XformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.self_attn_proj = nn.Linear(d_model, 3 * d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        
        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_kv_proj = nn.Linear(d_model, 2 * d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_attn1 = nn.Dropout(dropout)
        self.dropout_attn2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, tgt, memory, tgt_mask_bias=None):
        B, T, D = tgt.shape
        
        # Masked Self-Attention
        q, k, v = self.self_attn_proj(tgt).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, D // self.nhead)
        k = k.view(B, T, self.nhead, D // self.nhead)
        v = v.view(B, T, self.nhead, D // self.nhead)
        
        self_attn_out = xops.memory_efficient_attention(q, k, v, attn_bias=tgt_mask_bias)
        self_attn_out = self_attn_out.view(B, T, D)
        
        tgt = self.norm1(tgt + self.dropout_attn1(self.self_attn_out(self_attn_out)))

        # Cross-Attention
        q = self.cross_attn_q_proj(tgt)
        k, v = self.cross_attn_kv_proj(memory).chunk(2, dim=-1)

        _, S, _ = memory.shape

        q = q.view(B, T, self.nhead, D // self.nhead)
        k = k.view(B, S, self.nhead, D // self.nhead)
        v = v.view(B, S, self.nhead, D // self.nhead)

        cross_attn_out = xops.memory_efficient_attention(q, k, v)
        cross_attn_out = cross_attn_out.view(B, T, D)
        
        tgt2 = self.norm2(tgt + self.dropout_attn2(self.cross_attn_out(cross_attn_out)))

        # FFN
        ff_out = self.linear2(self.activation(self.linear1(tgt2)))
        tgt3 = self.norm3(tgt2 + self.dropout_ff(ff_out))
        
        return tgt3


class ActionTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        action_dim: int,
        action_pred_horizon: int = 16,
    ):
        super().__init__()
        
        self.query_embed = nn.Embedding(action_pred_horizon, d_model)
        
        self.layers = nn.ModuleList([
            XformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])
        
        self.action_head = nn.Linear(d_model, action_dim)
        self.action_pred_horizon = action_pred_horizon
        self.causal_attn_bias = xops.LowerTriangularMask()
        
    def forward(self, memory, text_emb, state) -> torch.Tensor:
        B, _, d_model = memory.shape
  
        memory = torch.cat([state, text_emb, memory], dim=1)
        
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        decoder_out = query_pos
        for layer in self.layers:
            decoder_out = layer(
                tgt=decoder_out,
                memory=memory,
                tgt_mask_bias=self.causal_attn_bias
            )
        
        action_out = self.action_head(decoder_out)
        return action_out
