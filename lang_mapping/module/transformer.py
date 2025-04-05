import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import rotary_pe_3d  
from ..utils import positional_encoding

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
class TransformerCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Define multi-head attention layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,        # [B, S, d_model]
        coords_src: torch.Tensor = None,
        text: torch.Tensor = None,# [B, T, d_model] or None
    ) -> torch.Tensor:
        
        # Self-Attention (use rotary encoding if coords_src is given)
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_rot = src
        else:
            q_rot = k_rot = v_rot = src

        attn_out, _ = self.self_attn(q_rot, k_rot, v_rot)
        src = self.norm1(src + self.dropout1(attn_out))

        # Cross-Attention with text
        if text is not None:
            attn_out_text, _ = self.cross_attn_text(query=src, key=text, value=text)
            src = self.norm2(src + self.dropout2(attn_out_text))

        # FeedForward
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = self.norm_ff(src + self.dropout_ff(ff_out))
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=256, num_layers=2, num_heads=8, output_dim=1024, num_token=512):
        super().__init__()
        
        self.modality_embed_state = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_text = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_learnable = nn.Parameter(torch.randn(1, 1, input_dim))
        self.modality_embed_3d = nn.Parameter(torch.randn(1, 1, input_dim))
        
        self.layers = nn.ModuleList([
            TransformerCrossAttentionLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * num_token, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim)
        )
        
        self.apply(init_weights_kaiming)
        
    def forward(
        self,
        hand: torch.Tensor,        # [B, N, input_dim]
        head: torch.Tensor,        # [B, N, input_dim]
        coords_hand: torch.Tensor = None,
        coords_head: torch.Tensor = None,
        state: torch.Tensor = None,          # [B, input_dim] or None
        text_embeddings: torch.Tensor = None,# [B, input_dim] or None
        global_token: torch.Tensor = None  # [B, input_dim] or None
    ) -> torch.Tensor:
        B, N, D = hand.shape
        # -------------------------------------------------------------------
        # 1) Construct the initial src tokens (self-attention input)
        # -------------------------------------------------------------------
        tokens = []
        coords_list = []
        # If we have state token
        if state is not None:
            state_token = state.unsqueeze(1) + self.modality_embed_state  # [B, 1, input_dim]
            tokens.append(state_token)
            coords_list.append(torch.zeros(B, 1, 3, device=state.device))
        # If we have text token
        if text_embeddings is not None:
            text_token = text_embeddings.unsqueeze(1) + self.modality_embed_text # [B, 1, input_dim]
            tokens.append(text_token)
            coords_list.append(torch.zeros(B, 1, 3, device=state.device))
        # If we have global_token token
        if global_token is not None:     
            Bg, Mg, Dg = global_token.shape
            global_token = global_token + self.modality_embed_learnable
            tokens.append(global_token)
            coords_list.append(torch.zeros(Bg, Mg, 3, device=state.device))
        # Now add hand + head tokens
        hand = hand + self.modality_embed_3d
        head = head + self.modality_embed_3d
        
        tokens.append(hand)  # [B, N, input_dim]
        tokens.append(head)  # [B, N, input_dim]
        # Build coords if provided
        if coords_hand is not None and coords_head is not None:
            coords_list.append(coords_hand)
            coords_list.append(coords_head)
            coords_src = torch.cat(coords_list, dim=1)  # [B, S+2N, 3]
        else:
            coords_src = None
        # Concatenate all tokens along the sequence dimension
        src = torch.cat(tokens, dim=1)  # shape: [B, S+2N, input_dim]
        # -------------------------------------------------------------------
        # 2) Pass through stacked TransformerCrossAttentionLayers
        # -------------------------------------------------------------------
        text_ = text_embeddings.unsqueeze(1) if text_embeddings is not None else None

        for layer in self.layers:
            src = layer(
                src=src,
                coords_src=coords_src,
                text=text_
            )
        # -------------------------------------------------------------------
        # 3) Post-fusion MLP
        # -------------------------------------------------------------------
        num_special = 0
        if state is not None:
            num_special += 1
        if text_embeddings is not None:
            num_special += 1
        # if global_token is not None:
        #     num_special += Mg
        
        fused_tokens = src[:, num_special:, :]   # shape: [B, 2N, input_dim]
        data = fused_tokens.reshape(B, -1)       # flatten the remaining
        out = self.post_fusion_mlp(data)         # [B, output_dim]
        return out
    
class PerceiverAttentionLayer(nn.Module):
    """
    Inputs:
      - q: [B, Q, dim]
      - k: [B, N, dim]
      - v: [B, N, dim]
      - coords_q:  [B, Q, 3] (positional coords for q)
      - coords_kv: [B, N, 3] (positional coords for k and v)
    """
    def __init__(self, dim: int, nhead: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords_q: torch.Tensor = None,
        coords_kv: torch.Tensor = None
    ) -> torch.Tensor:
        # (1) Apply rotary PE to Query if coords_q is provided
        if coords_q is not None:
            q = rotary_pe_3d(q, coords_q)

        # (2) Apply rotary PE to Key if coords_kv is provided
        if coords_kv is not None:
            k = rotary_pe_3d(k, coords_kv)
            # v remains unchanged (by design), but could also apply PE if desired

        # (3) Multi-head cross-attention
        attn_out, _ = self.attn(q, k, v)  # [B, Q, dim]

        # (4) Residual + LayerNorm
        x = self.ln1(q + attn_out)        # [B, Q, dim]

        # (5) Position-wise feedforward
        ffn_out = self.ffn(x)            # [B, Q, dim]

        # (6) Residual + LayerNorm
        x = self.ln2(x + ffn_out)        # [B, Q, dim]
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
        hidden_dim: int = 120,
        nhead: int = 8,
        num_layers: int = 4,
        out_dim: int = 120,
        voxel_proj: nn.Module = None,
        num_learnable_tokens: int = 16,
    ):
        super().__init__()
        
        self.modality_embed_state = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.modality_embed_learnable = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.hidden_dim = hidden_dim
        self.voxel_proj = voxel_proj

        # Build a stack of cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverAttentionLayer(dim=hidden_dim, nhead=nhead)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        self.num_learnable_tokens = num_learnable_tokens
        self.learnable_tokens = nn.Parameter(
            torch.zeros(1, num_learnable_tokens, hidden_dim)
        )
        
        self.apply(init_weights_kaiming)

    def forward(
        self,
        state: torch.Tensor,            # [B, state_dim]
        valid_coords: torch.Tensor,     # [B, N, 3]
        valid_feats: torch.Tensor       # [B, N, voxel_feature_dim]
    ) -> torch.Tensor:
        """
        Args:
            state:           [B, state_dim]
            head_translation [B, 3]
            valid_coords:    [B, N, 3]
            valid_feats:     [B, N, voxel_feature_dim]
        """
        B, N, _ = valid_feats.shape

        # (1) Query = [ state_token + learnable_tokens ]
        state_token = state.unsqueeze(1)  # [B, 1, hidden_dim]
        state_token = state_token + self.modality_embed_state
        
        learnable_tokens = (
            self.learnable_tokens + self.modality_embed_learnable
        ).repeat(B, 1, 1)  # [B, num_learnable_tokens, hidden_dim]
        q = torch.cat([state_token, learnable_tokens], dim=1)  # [B, 1 + num_learnable_tokens, hidden_dim]
        
        coords_state = torch.zeros(B, 1, 3, device=q.device) 
        coords_learnable = torch.zeros(B, self.num_learnable_tokens, 3, device=q.device)  # [B, num_learnable_tokens, 3]
        coords_q = torch.cat([coords_state, coords_learnable], dim=1) 
        
        feats_flat = valid_feats.reshape(B * N, -1)
        coords_flat = valid_coords.reshape(B * N, 3)
        
        k_flat, _ = self.voxel_proj(
            feats_flat, coords_flat, return_intermediate=True
        )

        k = k_flat.view(B, N, self.hidden_dim)
        v = k
        
        coords_kv = valid_coords

        # (5) Pass through Perceiver cross-attention layers
        x = q
        for layer in self.layers:
            x = layer(
                q=x,
                k=k,
                v=v,
                coords_q=coords_q,
                coords_kv=coords_kv
            )

        # (6) Final projection
        out = self.out_proj(x[:, 1:, :])  # [B, num_learnable_tokens, out_dim]

        return out
