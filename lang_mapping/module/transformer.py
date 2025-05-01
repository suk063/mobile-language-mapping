import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import rotary_pe_3d  
import math

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class PerceiverAttentionLayer(nn.Module):
    def __init__(
        self, 
        dim: int = 256, 
        nhead: int = 8, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        assert dim % nhead == 0, "dim must be divisible by nhead"

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # Output projection after attention
        self.out_proj = nn.Linear(dim, dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim),
        )

        # LayerNorm layers
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords_q: torch.Tensor = None,
        coords_kv: torch.Tensor = None
    ) -> torch.Tensor:
        B, Q_len, _ = q.shape
        _, KV_len, _ = k.shape

        # Linear projection
        q_proj = self.W_q(q).view(B, Q_len, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, Q_len, head_dim]
        k_proj = self.W_k(k).view(B, KV_len, self.nhead, self.head_dim).transpose(1, 2) # [B, nhead, KV_len, head_dim]
        v_proj = self.W_v(v).view(B, KV_len, self.nhead, self.head_dim).transpose(1, 2) # [B, nhead, KV_len, head_dim]

        # Apply Rotary Positional Embedding if provided
        if coords_q is not None:
            q_proj = rotary_pe_3d(q_proj, coords_q)
        if coords_kv is not None:
            k_proj = rotary_pe_3d(k_proj, coords_kv)

        # 3) Scaled dot-product attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_proj)

        # Combine heads back into single tensor
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q_len, self.dim)

        # 5) Residual + Norm
        out = self.out_proj(attn_output)
        x = self.ln1(q + self.dropout_attn(out))

        # 6) Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout_ff(ffn_out))

        return x
 
class GlobalPerceiver(nn.Module):
    """
    - Query: state_projected -> shape [B, 1, hidden_dim]
    - Key/Value: derived from valid_feats -> implicit_decoder -> [B, N, hidden_dim]
    - coords_q: head_translation -> [B, 1, 3]
    - coords_kv: valid_coords -> [B, N, 3]
    """
    def __init__(
        self,
        input_dim: int = 240,
        nhead: int = 8,
        num_layers: int = 2,
        hidden_dim: int = 1024,
        out_dim: int = 240,
        num_learnable_tokens: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nhead = nhead

        # Learnable tokens (2 tokens = hand, head)
        self.num_learnable_tokens = num_learnable_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, self.num_learnable_tokens, input_dim))
        nn.init.xavier_uniform_(self.global_tokens)

        # Perceiver cross-attn layers
        self.layers = nn.ModuleList([
            PerceiverAttentionLayer(dim=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
            for _ in range(num_layers)
        ])

        # projection
        self.out_proj = nn.Linear(input_dim, out_dim)
        self.apply(init_weights_kaiming)

    def forward(
        self,
        state,                     # [B, hidden_dim]
        hand_translation_t: torch.Tensor,  # [B, 3]
        head_translation_t: torch.Tensor,  # [B, 3]
        valid_coords: torch.Tensor,          # [B, N, 3]
        valid_feats_projected: torch.Tensor            # [B, N, feat_dim]
    ) -> torch.Tensor:
        """
        Args:
            hand_translation_all: [B, 3]
            head_translation_all: [B, 3]
            valid_coords:         [B, N, 3]
            valid_feats:          [B, N, feat_dim]
        Returns:
            out: [B, 16, out_dim]
        """
        B2, N, _ = valid_feats_projected.shape

        # (1) state token
        state_token = state.unsqueeze(1)  # [B,1,hidden_dim]
        coords_state = torch.zeros(B2, 1, 3, device=state.device)

        # (2) learnable tokens
        global_tokens = self.global_tokens.repeat(B2, 1, 1)  # [B,num_learnable_tokens,input_dim]
        hand_coords = hand_translation_t.unsqueeze(1).repeat(1, self.num_learnable_tokens // 2, 1)  # [B,8,3]
        head_coords = head_translation_t.unsqueeze(1).repeat(1, self.num_learnable_tokens // 2, 1)  # [B,8,3]
        
        coords_learned = torch.cat([hand_coords, head_coords], dim=1)  # [B,2,3]

        # Combine them: total Q_len=3
        q = torch.cat([state_token, global_tokens], dim=1)      
        coords_q = torch.cat([coords_state, coords_learned], dim=1)

        # K, V
        k = valid_feats_projected
        v = valid_feats_projected
        coords_kv = valid_coords
        
        # (4) Pass through cross-attention layers
        x = q
        for layer in self.layers:
            x = layer(
                q=x,
                k=k,
                v=v,
                coords_q=coords_q,
                coords_kv=coords_kv
            )
        # x shape: [B,3,hidden_dim]
        # The first token (index=0) is the state token; we only want the 2 learned tokens
        # => [B,2,hidden_dim], then apply out_proj
        out_tokens = x[:, 1:, :]
        out = self.out_proj(out_tokens)  # [B, 16,out_dim]
        return out 

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projection
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,        # (B, Lq, d_model)
        key: torch.Tensor,          # (B, Lk, d_model)
        value: torch.Tensor,        # (B, Lk, d_model)
        coords_query: torch.Tensor = None,  # (B, Lq, 3) or None
        coords_key: torch.Tensor = None  # (B, Lk, 3) or None
    ) -> torch.Tensor:

        B, Lq, _ = query.shape
        _, Lk, _ = key.shape

        # 1) Q, K, V projection
        q = self.W_q(query)  # (B, Lq, d_model)
        k = self.W_k(key)    # (B, Lk, d_model)
        v = self.W_v(value)  # (B, Lk, d_model)
        
        # 2) reshape => (B, n_heads, Lq(or Lk), head_dim)
        q = q.view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, Lq, head_dim)
        k = k.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, Lk, head_dim)
        v = v.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, Lk, head_dim)
        
         # 3) keyRoPE
        if coords_query is not None:  
            q = rotary_pe_3d(q, coords_query)
        if coords_key is not None:
            k = rotary_pe_3d(k, coords_key)
        
        # 4) Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, n_heads, Lq, Lk)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # (B, n_heads, Lq, head_dim)
        
        # 5)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.out_proj(out)
        return out

def generate_subsequent_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) 
    mask = mask.bool()  # True/False
    mask = mask.masked_fill(mask, float('-inf')) 
    return mask

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def make_casual_mask(seq_len, len_m1, device):
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    causal_mask[:len_m1, len_m1:] = True

    return causal_mask 

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
        
        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,             # (B, S, d_model)
        coords_src: torch.Tensor = None,  # (B, S, 3) or None
        causal_mask=None,
    ) -> torch.Tensor:
        # src shape: (B, S, d_model)
        B, S, _ = src.shape
        
        # 1) Q, K, V projections
        q = self.W_q(src)  # (B, S, d_model)
        k = self.W_k(src)
        v = self.W_v(src)
        
        # 2) Reshape and transpose for multi-head
        # => (B, n_heads, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3) Apply RoPE if coords_src is provided
        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
            # v is often unchanged in RoPE
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
        attn = torch.matmul(F.softmax(scores, -1), v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.d_model)
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        
        return self.norm2(src2 + self.dropout_ff(ff))

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=240,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        proj_dim=16,
    ):
        super().__init__()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=input_dim,
                n_heads=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj= nn.Linear(input_dim, proj_dim)
        
        self.apply(init_weights_kaiming)
               
    def forward(
        self,
        hand_token_t: torch.Tensor,  # [B, N, input_dim]
        head_token_t: torch.Tensor,  # [B, N, input_dim] 
        hand_token_m1: torch.Tensor,  # [B, N, input_dim]
        head_token_m1: torch.Tensor,  # [B, N, input_dim]
        coords_hand_t: torch.Tensor = None,
        coords_head_t: torch.Tensor = None, 
        coords_hand_m1: torch.Tensor = None,
        coords_head_m1: torch.Tensor = None,
        state_t: torch.Tensor = None,  # [B, input_dim] or None
        state_m1: torch.Tensor = None,  # [B, input_dim] or None
    ) -> torch.Tensor:
        B, N, D = hand_token_t.shape
        
        tokens = []
        coords_list = []
        coords_src = None
        
        if state_m1 is not None:
            state_token_m1 = state_m1.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token_m1)
            coords_list.append(torch.zeros(B, 1, 3, device=state_t.device))
        
        tokens.append(hand_token_m1)     # [B, N, D]
        tokens.append(head_token_m1) 
        
        if coords_hand_m1 is not None:
            coords_list.append(coords_hand_m1)
            coords_list.append(coords_head_m1)
        
        if state_t is not None:
            state_token_t = state_t.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token_t)
            coords_list.append(torch.zeros(B, 1, 3, device=state_t.device))

        tokens.append(hand_token_t)     # [B, N, D]
        tokens.append(head_token_t) 
        
        if coords_hand_t is not None:
            coords_list.append(coords_hand_t)
            coords_list.append(coords_head_t)
        
        src = torch.cat(tokens, dim=1)
        # DEBUG: (Woojeh)
        if coords_hand_t is not None and len(coords_list) > 0:
            coords_src = torch.cat(coords_list, dim=1)  # (B, S, 3)
        
        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(
                src=src,
                coords_src=coords_src,
            )

        start_idx = 1 + 512 # 1 for state, 512 for m1, 1 for state

        return src[:, start_idx:, :]

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
        
        self.query_embed = nn.Embedding(action_pred_horizon, d_model)  # [3, d_model]
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.action_head = nn.Linear(d_model, action_dim)
        self.action_pred_horizon = action_pred_horizon
        self.casual_mask = generate_subsequent_mask(self.action_pred_horizon+1)
        
    def forward(self, memory, state) -> torch.Tensor:

        # state # [B, 1, d_model]

        B, N, d_model = memory.shape
  
        # memory = memory.view(B, fs*N, d_model)             # [B, 2*N, d_model]
        memory = memory.permute(1, 0, 2).contiguous()     # [N, B, d_model]
        
        query_pos = self.query_embed.weight                # [3, d_model]
        query_pos = query_pos.unsqueeze(1).repeat(1, B, 1) # [3, B, d_model]
        
        state = state.permute(1, 0, 2).contiguous()
        tgt = torch.cat([state, query_pos], dim=0)
        
        decoder_out = self.decoder(
            tgt=tgt,    # [T, B, d_model]
            memory=memory,      # [N, B, d_model]
            tgt_mask=self.casual_mask
        ) 
        
        decoder_out = decoder_out.permute(1, 0, 2)         # [B, 4, d_model]
        action_out = self.action_head(decoder_out)         # [B, 4, action_dim]
        return action_out[:, 1:, :]


# class LocalSelfAttentionFusion(nn.Module):
#     """
#     Fuses two feature vectors via self-attention with a residual connection.
#     Each voxel is treated independently as a 2-token sequence.
#     """
#     def __init__(self, feat_dim=120, num_heads=8):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
#         self.layernorm = nn.LayerNorm(feat_dim)

#     def forward(self, feat1, feat2):
#         """
#         Args:
#             feat1, feat2: (B, N, feat_dim).
#               - In typical usage here, N=1 and feat_dim=384.
#         Returns:
#             (B, N, feat_dim): fused feature after 2-token self-attention with residual connection.
#         """

#         # Stack feat1 and feat2 into a 2-token sequence: (B, N, 2, feat_dim)
#         x_stacked = torch.stack([feat1, feat2], dim=2)  # shape: (B, N, 2, feat_dim)
#         B, N, T, D = x_stacked.shape  # T=2 (token sequence length is 2)
#         x = x_stacked.view(B*N, T, D)  # Reshape to (B*N, 2, D) for MHA (batch_first=True)

#         # Self-attention on the 2 tokens
#         attn_output, _ = self.mha(x, x, x) # MHA output shape: (B*N, 2, D)

#         # --- Add Residual Connection ---
#         x = x + attn_output # Residual connection

#         # --- Apply Layer Normalization ---
#         y = self.layernorm(x)  # LayerNorm output shape: (B*N, 2, D)

#         # --- Select Output Token ---
#         # fused = y.mean(dim=1).view(B, N, D) # Alternative: Average the 2 output tokens.
#         fused = y[:, 0, :].view(B, N, D) # Take the feature of the first token.

#         return fused

class LocalSelfAttentionFusion(nn.Module):
    """
    Fuse two per-voxel feature vectors with a minimal Transformer encoder block
    and project the result to `output_dim` (default: 240).
    """
    def __init__(
        self,
        feat_dim: int = 768,
        num_heads: int = 8,
        ffn_multiplier: int = 2,
        dropout: float = 0.1,
        output_dim: int = 368,
    ):
        super().__init__()

        # ─── Self-attention sub-layer ──────────────────────────────
        self.ln_1   = nn.LayerNorm(feat_dim)
        self.mha    = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_1 = nn.Dropout(dropout)

        # ─── Feed-forward sub-layer ───────────────────────────────
        self.ln_2   = nn.LayerNorm(feat_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(feat_dim, ffn_multiplier * feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * feat_dim, feat_dim),
        )
        self.drop_2 = nn.Dropout(dropout)

        # ─── Final projection ─────────────────────────────────────
        self.output_proj = nn.Linear(feat_dim, output_dim)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        B, N, _ = feat1.shape            # (B, N, D)

        # (B, N, 2, D) → (B·N, 2, D)
        x = torch.stack([feat1, feat2], dim=2).flatten(0, 1)

        # ── Self-attention ─────────────────────────────
        qkv = self.ln_1(x)     # one LN call
        y, _ = self.mha(qkv, qkv, qkv)
        x = x + self.drop_1(y)

        # ── FFN ───────────────────────────────────────
        y = self.ffn(self.ln_2(x))
        x = x + self.drop_2(y)

        # first token → (B·N, D) → (B, N, D)
        fused = x[:, 0, :].contiguous().view(B, N, -1)

        # projection to output_dim
        return self.output_proj(fused)