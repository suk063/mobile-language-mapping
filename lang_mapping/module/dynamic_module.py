import torch
import torch.nn as nn

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

class LocalCrossAttentionFusion(nn.Module):
    """
    feat1 -> Query
    feat2 -> Key/Value
    """
    def __init__(self, feat_dim=120, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(feat_dim)

    def forward(self, feat1, feat2):
        """
        feat1: (B, N, feat_dim)
        feat2: (B, N, feat_dim)
        Cross-attention -> (B, N, feat_dim)
        """
        B, N, D = feat1.shape

        fused, _ = self.cross_attn(feat1, feat2, feat2)  # (B, N, D)

        # LayerNorm
        fused = self.layernorm(fused)  # (B, N, D)

        return fused
    

class LocalSelfAttentionFusionMulti(nn.Module):
    """
    Fuses multiple features (e.g. [voxel_feat_t, voxel_feat_tp1, voxel_feat_tm1]) via self-attention.
    """
    def __init__(self, feat_dim=120, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(feat_dim)

    def forward(self, feats_list):
        """
        Args:
            feats_list: list of [B, N, D], 예: [feat_t, feat_tp1, feat_tm1]
        Returns:
            fused: [B, N, D]
        """
        x = torch.stack(feats_list, dim=2)  # [B, N, 3, D]
        B, N, T, D = x.shape

        # (B*N, T, D) reshape
        x = x.view(B*N, T, D)
        y, _ = self.mha(x, x, x)
        y = self.layernorm(y)  # [B*N, T, D]

        y0 = y[:, 0, :]        # [B*N, D]
        fused = y0.view(B, N, D)  # [B, N, D]

        return fused
    
        
class FlowTimeAggregator(nn.Module):
    """
    feats + positional_encoding(flow) → MLP → residual(feats + mlp_out)
    """
    def __init__(self, feat_dim=768,  hidden_dim=768, L=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.L = L

        # flow용 PE 차원 = 2 * L * 3
        self.flow_pe_dim = 2 * self.L * 3
        self.mlp_in_dim = self.feat_dim + self.flow_pe_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim)
        )
    
    def positional_encoding_3d(self, flow_3d):
        # flow_3d: [N, 3]
        # output: [N, 2*L*3]
        N = flow_3d.shape[0]
        pe_list = []
        freqs = 2.0 ** torch.arange(self.L, device=flow_3d.device)
        for i in range(3):
            v = flow_3d[:, i]  # shape: [N]
            for f in freqs:
                pe_list.append(torch.sin(f*v))
                pe_list.append(torch.cos(f*v))
        pe = torch.stack(pe_list, dim=1)
        return pe

    def forward(self, feats, flow_3d):
        """
        feats: [N, feat_dim]
        flow_3d: [N, 3]
        out: feats + mlp(feats||PE(flow))
        """
        pe_flow = self.positiona_encoding_3d(flow_3d)  # [N, 2*L*3]
        x = torch.cat([feats, pe_flow], dim=-1)         # [N, feat_dim + 2*L*3]
        mlp_out = self.mlp(x)                           # [N, feat_dim]
        out = feats + mlp_out                           # residual
        return out
    