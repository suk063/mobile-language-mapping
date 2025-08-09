import torch
import torch.nn as nn
import torch.nn.functional as F
from lang_mapping.utils.utils import rotary_pe_3d
import xformers.ops as xops
from pytorch3d.ops import knn_points
from typing import Optional

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
        dropout=0.1,
        use_xformers: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_xformers = use_xformers

        
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
        key_padding_mask: Optional[torch.Tensor] = None,
        coords_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = src.shape
        
        q = self.W_q(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(src).view(B, S, self.n_heads, self.head_dim)

        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
        
        if self.use_xformers:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            
            attn_bias = None
            if key_padding_mask is not None:
                # xformers >=0.0.23 removed `KeyPaddingMask`. Build a dense bias tensor instead.
                seq_len = key_padding_mask.size(1)
                mask = key_padding_mask[:, None, None, :].to(q.dtype)
                attn_bias = mask.expand(-1, self.n_heads, seq_len, -1) * (-1e9)
            else:
                attn_bias = None
                
            attn = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.dropout_attn.p if self.training else 0.0,
            )  # (B, S, H, D)    

        else:
            # PyTorch's scaled_dot_product_attention expects (B, n_heads, S, head_dim)
            v = v.transpose(1, 2).contiguous()
            # Build an attention mask from the key\_padding\_mask (True → ignore)
            attn_mask = None
            if key_padding_mask is not None:
                # expected shape: (B, 1, 1, K) broadcastable to (B, H, Q, K)
                attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_attn.p if self.training else 0.0,
            )
            attn = attn.transpose(1, 2).contiguous() # (B, S, H, D)
        
        # Collapse heads ---------------------------------------------------
        attn = attn.reshape(B, S, self.d_model).contiguous()

        # Residual & FF -----------------------------------------------------
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
                dim_feedforward=hidden_dim,
                use_xformers=True
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
        
    def forward(self, visual_token, state, text_emb=None, global_tok=None) -> torch.Tensor:
        B, _, d_model = visual_token.shape
        tokens = torch.cat([visual_token, state], dim=1)
        if text_emb is not None:
            tokens = torch.cat([tokens, text_emb], dim=1)
        if global_tok is not None:
            tokens = torch.cat([tokens, global_tok], dim=1)
       
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        decoder_out = query_pos
        for layer in self.layers:
            decoder_out = layer(
                tgt=decoder_out,
                memory=tokens,
                tgt_mask_bias=self.causal_attn_bias
            )
        
        action_out = self.action_head(decoder_out)
        return action_out


class LocalFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        radius: float = 0.1,
        k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.radius, self.k = radius, k
        self.attn = TransformerLayer(
            d_model=dim,
            n_heads=n_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            use_xformers=False
        )

    # ----------------------------------------------------------
    # Find neighbor indices within <radius>; pad with query itself
    # ----------------------------------------------------------
    def _neigh_indices(
        self,
        q_xyz: torch.Tensor,           # (B, N, 3)  – query coordinates
        kv_xyz: torch.Tensor,          # (B, L, 3)  – scene coordinates
        kv_pad: Optional[torch.Tensor] # (B, L) bool – True → padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        idx     : (B, N, k) long  – neighbor indices (query-padded)
        invalid : (B, N, k) bool  – True → padding slot
        """
        B, N, _ = q_xyz.shape
        k = self.k
        radius = self.radius

        if kv_pad is not None:
            kv_xyz_masked = kv_xyz.clone()
            far_val = 1e9
            kv_xyz_masked[kv_pad] = far_val
        else:
            kv_xyz_masked = kv_xyz

        # KNN
        knn = knn_points(q_xyz, kv_xyz_masked, K=k, return_nn=False)
        dists, idx_topk = knn.dists, knn.idx     # (B, N, k), (B, N, k)
        invalid = dists > (radius * radius)
        idx = torch.where(invalid, torch.zeros_like(idx_topk), idx_topk)

        return idx, invalid


    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(
        self,
        q_xyz:   torch.Tensor,                # (B, N, 3)
        q_feat:  torch.Tensor,                # (B, N, C)
        kv_xyz:  torch.Tensor,                # (B, L, 3)
        kv_feat: torch.Tensor,                # (B, L, C)
        kv_pad:  Optional[torch.Tensor] = None  # (B, L) bool
    ) -> torch.Tensor:
        B, N, C = q_feat.shape
        idx, invalid = self._neigh_indices(q_xyz, kv_xyz, kv_pad)  # (B, N, k)

        # Debug        
        # num_valid = (~invalid).sum()
        # print(f"Number of valid neighbors: {num_valid.item()}")
        
        # gather neighbor coordinates / features
        batch = torch.arange(B, device=q_feat.device).view(B, 1, 1)
        neigh_xyz  = kv_xyz[batch.expand_as(idx), idx]             # (B, N, k, 3)
        neigh_feat = kv_feat[batch.expand_as(idx), idx]            # (B, N, k, C)
        
        # replace padding slots with the query point itself
        q_xyz_expanded = q_xyz.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, 3)
        q_feat_expanded = q_feat.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, C)
        neigh_xyz[invalid] = q_xyz_expanded[invalid]
        neigh_feat[invalid] = q_feat_expanded[invalid]

        # concatenate query token with neighbor tokens
        tokens = torch.cat([q_feat.unsqueeze(2), neigh_feat], dim=2)  # (B, N, k+1, C)
        # token_xyz = torch.cat([q_xyz.unsqueeze(2), neigh_xyz], dim=2)  # (B, N, k+1, 3)
        
        # key-padding mask for attention (True → ignore)
        key_padding_mask = torch.cat(
            [torch.zeros_like(invalid[..., :1]), invalid], dim=-1
        ).view(B * N, self.k + 1)

        # reshape to (B*N, S, C) for the transformer layer
        BM = B * N
        fused = self.attn(
            tokens.view(BM, self.k + 1, C).contiguous(),
            key_padding_mask=key_padding_mask,
        )  # (BM, k+1, C)

        # return only the query position (index 0 within each group)
        fused_q = fused[:, 0, :].view(B, N, C)
        
        return fused_q