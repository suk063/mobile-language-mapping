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
        hand_translation_all: torch.Tensor,  # [B, 3]
        head_translation_all: torch.Tensor,  # [B, 3]
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
        hand_coords = hand_translation_all.unsqueeze(1).repeat(1, self.num_learnable_tokens // 2, 1)  # [B,8,3]
        head_coords = head_translation_all.unsqueeze(1).repeat(1, self.num_learnable_tokens // 2, 1)  # [B,8,3]
        
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
        
        # Text cross-attention
        self.cross_attn_text = CrossAttentionLayer(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
        # Perceiver cross-attention
        self.cross_attn_perceiver = CrossAttentionLayer(d_model, n_heads, dropout)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,             # (B, S, d_model)
        coords_src: torch.Tensor = None,  # (B, S, 3) or None
        text: torch.Tensor = None,         # (B, T, d_model) or None
        perceiver: torch.Tensor = None,    # (B, P, d_model) or None
        coords_cam: torch.Tensor = None    # (B, P, 3) or None
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
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn_output)))
        
        if text is not None:    
            B_txt, T, _ = text.shape
            text_coords = torch.zeros(B_txt, T, 3, device=text.device)
            
            attn_text = self.cross_attn_text(
                query=src2,
                key=text,
                value=text, 
                coords_query=coords_src,  
                coords_key=text_coords  
            )
            src2 = self.norm3(src2 + self.dropout3(attn_text))
        
        if perceiver is not None:
            attn_perceiver = self.cross_attn_perceiver(
                query=src2,
                key=perceiver,
                value=perceiver,
                coords_query=coords_src,    
                coords_key=coords_cam  
            )
            src2 = self.norm4(src2 + self.dropout4(attn_perceiver))
        
        # Feed froward network
        ff_out = self.linear2(self.activation(self.linear1(src2)))
        out2 = self.norm2(src2 + self.dropout_ff(ff_out))
        
        return out2
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=240,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        output_dim=128,
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
        
        # Post-fusion MLP
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * (256 * 2), 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )
        
        self.apply(init_weights_kaiming)
        
    def forward(
        self,
        hand_token: torch.Tensor,  # [B, N, input_dim]
        head_token: torch.Tensor,  # [B, N, input_dim] 
        coords_hand: torch.Tensor = None,
        coords_head: torch.Tensor = None,
        state: torch.Tensor = None,  # [B, input_dim] or None
        text_embeddings: torch.Tensor = None,  # [B, input_dim] or None
        perceiver_out_all: torch.Tensor = None,
        hand_translation_all: torch.Tensor = None,
        head_translation_all: torch.Tensor = None,
    ) -> torch.Tensor:
        B2, N, D = hand_token.shape
        
        tokens = []
        coords_list = []
        
        if state is not None:
            state_token = state.unsqueeze(1)  # [B, 1, D]
            tokens.append(state_token)
            coords_list.append(torch.zeros(B2, 1, 3, device=state.device))
        
        text_token = None
        if text_embeddings is not None:
            text_token = text_embeddings.unsqueeze(1)  # [B, 1, D]
            tokens.append(text_token)
            coords_list.append(torch.zeros(B2, 1, 3, device=state.device))

        coords_cam = None
        if perceiver_out_all is not None:
            M = perceiver_out_all.shape[1]
            tokens.append(perceiver_out_all)    # [B, M, D]
            
            hand_translation_all = hand_translation_all.unsqueeze(1).repeat(1, M // 2, 1)  # [B,8,3]
            head_translation_all = head_translation_all.unsqueeze(1).repeat(1, M // 2, 1)  # [B,8,3]
            
            coords_cam = torch.cat([hand_translation_all, head_translation_all], dim=1)         # [B,16,3]
            coords_list.append(coords_cam)  # [B, 2, 3]

        tokens.append(hand_token)     # [B, N, D]
        tokens.append(head_token) 
        src = torch.cat(tokens, dim=1)
        
        coords_src = None
        if coords_hand is not None:
            coords_list.append(coords_hand)
            coords_list.append(coords_head)
            coords_src = torch.cat(coords_list, dim=1) # [B, S+N, 3]
        
        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(
                src=src,
                coords_src=coords_src,
                text=text_token,
                perceiver=perceiver_out_all,
                coords_cam=coords_cam,
            )

        start_idx = 0
        if state is not None:
            start_idx += 1
        if text_embeddings is not None:
            start_idx += 1
        if perceiver_out_all is not None:
            start_idx += M

        # fused_tokens = src[:, start_idx:, :]  # [B, (N + ?), input_dim]
        
        fused_tokens = self.output_proj(src[:, start_idx:, :])
        
        data = fused_tokens.reshape(B2, -1)
        out = self.post_fusion_mlp(data)  # [B, output_dim]
        return out

class LocalSelfAttentionFusion(nn.Module):
    """
    Fuses two feature vectors via self-attention.
    Each voxel is treated independently as a 2-token sequence.
    """
    def __init__(self, feat_dim=120, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(feat_dim)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1, feat2: (B, N, feat_dim). 
              - In typical usage here, N=1 and feat_dim=384.
        Returns:
            (B, N, feat_dim): fused feature after 2-token self-attention.
        """
        
        # Stack feat1 and feat2 into a 2-token sequence: (B, N, 2, feat_dim)
        x = torch.stack([feat1, feat2], dim=2)  # shape: (B, N, 2, feat_dim)
        B, N, T, D = x.shape  # T=2
        x = x.view(B*N, T, D)  # (B*N, 2, D)

        # Self-attention on the 2 tokens
        y, _ = self.mha(x, x, x)
        y = self.layernorm(y)  # (B*N, 2, D)

        # Average the 2 tokens to get a single feature vector
        fused = y.mean(dim=1).view(B, N, D)
        # fused = y[:, 0, :].view(B, N, D)
        
        return fused