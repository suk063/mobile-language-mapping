import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import rotary_pe_3d  
from .utils import positional_encoding

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes a layer's weights orthogonally.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class TransformerCrossAttentionLayer(nn.Module):
    """
    Consists of:
    1) Self-Attention (with optional rotary PE)
    2) Cross-Attention (Q=src, K=V=text)
    3) FeedForward
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, text, coords_src=None):
        # Self-Attention (rotary PE applied if coords_src is given)
        if coords_src is not None:
            q_rot = rotary_pe_3d(src, coords_src)
            k_rot = rotary_pe_3d(src, coords_src)
            v_ = src
        else:
            q_rot = k_rot = v_ = src

        src2, _ = self.self_attn(query=q_rot, key=k_rot, value=v_)
        src = self.norm1(src + self.dropout1(src2))

        # Cross-Attention (Q=src, K=V=text)
        src2, _ = self.cross_attn(query=src, key=text, value=text)
        src = self.norm2(src + self.dropout2(src2))

        # FeedForward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm3(src + self.dropout3(src2))
        return src

class TransformerEncoder(nn.Module):
    """
    Stacks multiple TransformerCrossAttentionLayers to fuse
    state, hand, head, and text embeddings. Produces a final output.
    """
    def __init__(self, input_dim=120, hidden_dim=256, num_layers=2, num_heads=8, output_dim=1024):
        super().__init__()
        self.state_projection = nn.Linear(42, input_dim)
        self.layers = nn.ModuleList([
            TransformerCrossAttentionLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)
        ])
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 * 256, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)
        )

    def forward(self, hand, head, coords_hand, coords_head, state, text_embeddings):
        """
        hand, head: [B, N, input_dim]
        coords_hand, coords_head: [B, N, 3]
        state: [B, 42]
        text_embeddings: [B, T, input_dim]
        """
        B, N, D = hand.shape

        # Project state into a single token
        state_token = self.state_projection(state).unsqueeze(1)
        coords_state = torch.zeros(B, 1, 3, device=state.device)

        # For simplicity, assume text_embeddings is [B, 1, input_dim]
        # If not, adjust accordingly
        text_embeddings = text_embeddings.unsqueeze(1)

        # Concatenate state, hand, head
        src = torch.cat([state_token, hand, head], dim=1)
        coords_src = torch.cat([coords_state, coords_hand, coords_head], dim=1)

        # Pass through Transformer layers
        for layer in self.layers:
            src = layer(src=src, text=text_embeddings, coords_src=coords_src)

        # Post-fusion MLP
        data = src[:, 1:, :].reshape(B, -1)
        out = self.post_fusion_mlp(data)
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
    
class ActionMLP(nn.Module):
    """
    A feed-forward MLP for producing action outputs from a latent state vector.
    """
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: list = [2048, 1024, 512],
        final_std: float = 0.01 * np.sqrt(2)
    ):
        """
        Args:
            input_dim (int): Dimension of the input (e.g., token + state embedding).
            action_dim (int): Number of action outputs.
            hidden_dims (list): Dimensions of the hidden layers.
            final_std (float): Std for the final layer initialization.
        """
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(layer_init(nn.Linear(prev_dim, hdim)))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hdim

        # Final layer
        final_linear = layer_init(
            nn.Linear(prev_dim, action_dim),
            std=final_std
        )
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the action outputs."""
        return self.net(x)

class ImplicitDecoder(nn.Module):
    """
    A simple MLP to decode a 3D coordinate (with positional encoding) and its corresponding
    voxel feature into another feature vector (default 768-D).
    """
    def __init__(self, voxel_feature_dim=768, hidden_dim=768, output_dim=768, L=10):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.pe_dim = 2 * self.L * 3  # 2*L for sine/cosine, times 3 for x, y, z

        self.input_dim = self.voxel_feature_dim + self.pe_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim + self.pe_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)

        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4 = nn.LayerNorm(self.hidden_dim)

        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, voxel_features, coords_3d, return_intermediate=False):
        """
        Args:
            voxel_features (Tensor): [N, voxel_feature_dim=768].
            coords_3d (Tensor): [N, 3].
            return_intermediate (bool)
        Returns:
            Tensor: [N, output_dim=768]
        """
        pe = positional_encoding(coords_3d, L=self.L)  # [N, pe_dim]

        # 1) fc1
        x = torch.cat([voxel_features, pe], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)), inplace=True)

        x1 = x

        # 2) fc2
        x = F.relu(self.ln2(self.fc2(x1)), inplace=True)

        # 3) fc3 
        x = torch.cat([x, pe], dim=-1)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)
        
        # 4) fc4
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)

        # 5) fc5 
        out = self.fc5(x)

        if return_intermediate:
            return x1, out
        else:
            return out
        
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
        pe_flow = self.positional_encoding_3d(flow_3d)  # [N, 2*L*3]
        x = torch.cat([feats, pe_flow], dim=-1)         # [N, feat_dim + 2*L*3]
        mlp_out = self.mlp(x)                           # [N, feat_dim]
        out = feats + mlp_out                           # residual
        return out
    
class LoRALinear(nn.Module):
    def __init__(self, 
                 linear_layer: nn.Linear,
                 rank: int = 4,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.original_weight = linear_layer.weight
        self.original_bias = linear_layer.bias
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank)))
        
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.normal_(self.lora_B, std=0.02)

        self.original_weight.requires_grad_(False)
        if self.original_bias is not None:
            self.original_bias.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        result = torch.matmul(x, self.original_weight.T)
        if self.original_bias is not None:
            result = result + self.original_bias

        lora_out = torch.matmul(self.dropout(x), self.lora_A.T)  # (batch, rank)
        lora_out = torch.matmul(lora_out, self.lora_B.T)         # (batch, out_features)
        result += self.scaling * lora_out

        return result