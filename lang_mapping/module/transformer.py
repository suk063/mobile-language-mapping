import torch
import torch.nn as nn
import torch.nn.functional as F
from lang_mapping.utils.utils import rotary_pe_3d
import xformers.ops as xops

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from torch_geometric.nn import MLP, PointTransformerConv
from torch_geometric.nn.pool import radius
from torch_geometric.utils import to_dense_batch


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
        max_seq_len: int = 1024,
    ):
        super().__init__()
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
        
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
               
    def forward(
        self,
        visual_token: torch.Tensor,
        state: torch.Tensor,
        text_emb: torch.Tensor,
        global_feat: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        use_pe: bool = True,
    ) -> torch.Tensor:
        
        if global_feat is not None:
            tokens = torch.cat([text_emb, state, global_feat, visual_token], dim=1)
        else:
            tokens = torch.cat([text_emb, state, visual_token], dim=1)
            
        if use_pe:
            tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        
        coords_full = None
        if coords is not None:
            B = tokens.size(0)
            prefix = 2 + (1 if global_feat is not None else 0) 
            zeros = torch.zeros(B, prefix, 3, device=coords.device, dtype=coords.dtype)
            coords_full = torch.cat([zeros, coords], dim=1)

        for layer in self.layers:
            tokens = layer(src=tokens, coords_src=coords_full)
 
        return tokens


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

    def forward(self, tgt, memory, tgt_mask_bias=None, memory_key_padding_mask=None):
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

        cross_attn_bias = None
        if memory_key_padding_mask is not None:
            # Build dense bias tensor for xformers
            mask = memory_key_padding_mask[:, None, None, :].to(q.dtype)
            cross_attn_bias = mask.expand(-1, self.nhead, T, -1) * (-1e9)

        cross_attn_out = xops.memory_efficient_attention(q, k, v, attn_bias=cross_attn_bias)
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
        transf_input_dim: int,
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

        self.memory_proj = nn.Linear(transf_input_dim, d_model)
        self.apply(init_weights_kaiming)

        self.causal_attn_bias = xops.LowerTriangularMask()
        
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: (B, S_mem, transf_input_dim)
        returns: (B, T=action_pred_horizon, action_dim)
        """
        B = memory.size(0)

        # project encoder memory to d_model
        tokens = self.memory_proj(memory)  # (B, S_mem, d_model)

        # decoder queries
        decoder_out = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, T, d_model)

        for layer in self.layers:
            decoder_out = layer(
                tgt=decoder_out,
                memory=tokens,
                tgt_mask_bias=self.causal_attn_bias,  # causal on target only
            )

        return self.action_head(decoder_out)  # (B, T, action_dim)    

class ZeroPos(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), self.out_dim)

class LocalFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        radius: float = 0.2,
        k: int = 8,
        dropout: float = 0.1,
        use_rel_pos: bool = False
    ):
        super().__init__()
        self.radius, self.k = radius, k

        self.convs = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()

        for _ in range(num_layers):
            # PointTransformerConv for local feature aggregation.
            # It will update q_feat based on nearby kv_feat.
            pos_nn = (MLP([3, dim, dim], plain_last=False, batch_norm=False) if use_rel_pos else ZeroPos(dim))
            attn_nn = MLP([dim, dim], plain_last=False, batch_norm=False) # Maps q - k + pos_emb

            self.convs.append(
                PointTransformerConv(
                    in_channels=dim,
                    out_channels=dim,
                    pos_nn=pos_nn,
                    attn_nn=attn_nn,
                    add_self_loops=False  # This is a bipartite graph
                )
            )
            self.norm1s.append(nn.LayerNorm(dim))

            # Feed-forward network
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * ff_mult, dim),
                    nn.Dropout(dropout)
                )
            )
            self.norm2s.append(nn.LayerNorm(dim))


    def forward(
        self,
        q_xyz:   torch.Tensor,                # (B, N, 3)
        q_feat:  torch.Tensor,                # (B, N, C)
        kv_xyz:  torch.Tensor,                # (B, L, 3)
        kv_feat: torch.Tensor,                # (B, L, C)
        kv_pad:  Optional[torch.Tensor] = None  # (B, L) bool
    ) -> torch.Tensor:
        B, N, C = q_feat.shape
        L = kv_xyz.shape[1]

        # 1. Convert dense tensors to PyG format (flat vectors + batch indices)
        q_xyz_flat = q_xyz.reshape(-1, 3)
        out = q_feat.reshape(-1, C)
        q_batch = torch.arange(B, device=q_xyz.device).repeat_interleave(N)

        if kv_pad is not None:
            kv_mask = ~kv_pad
            kv_xyz_flat = kv_xyz[kv_mask]
            kv_feat_flat = kv_feat[kv_mask]
            kv_batch_full = torch.arange(B, device=kv_xyz.device).unsqueeze(1).expand(B, L)
            kv_batch = kv_batch_full[kv_mask]
        else:
            kv_xyz_flat = kv_xyz.reshape(-1, 3)
            kv_feat_flat = kv_feat.reshape(-1, C)
            kv_batch = torch.arange(B, device=kv_xyz.device).repeat_interleave(L)

        # 2. Find neighbors from kv for each q point
        target_idx, source_idx = radius(x=kv_xyz_flat, y=q_xyz_flat, r=self.radius,
                          batch_x=kv_batch, batch_y=q_batch, max_num_neighbors=self.k)

        edge_index = torch.stack([source_idx, target_idx], dim=0)

        # 3. Apply layers of PointTransformerConv for bipartite cross-attention
        for i in range(len(self.convs)):
            updated_q_feat = self.convs[i](
                x=(kv_feat_flat, out),
                pos=(kv_xyz_flat, q_xyz_flat),
                edge_index=edge_index
            )

            # Residual connection, FFN, and normalization
            out = self.norm1s[i](out + updated_q_feat)
            out2 = self.ffns[i](out)
            out = self.norm2s[i](out + out2)

        # 5. Convert back to dense tensor (B, N, C)
        final_feat, _ = to_dense_batch(out, q_batch, batch_size=B, max_num_nodes=N)

        return final_feat