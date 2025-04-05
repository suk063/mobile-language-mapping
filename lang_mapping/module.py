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

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    
class ConcatMLPFusion(nn.Module):
    """
    (feat1, feat2, coords_3d) -> [concat + sinusoidal PE] -> MLP -> fused
    """
    def __init__(self, feat_dim=120, L=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.L = L
        
        self.pos_enc_dim = 2 * self.L * 3

        in_dim = feat_dim * 2 + self.pos_enc_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, feat1, feat2, coords_3d):
        """
        Args:
            feat1, feat2: (B * N, feat_dim)
            coords_3d:    (B * N, 3)  
        
        Returns:
            fused: (B * N, feat_dim) 
        """

        x = torch.cat([feat1, feat2], dim=-1)

        pe = positional_encoding(coords_3d, L=self.L)  # (B*N, 2*L*3)

        # feature + positional_encoding을 concat
        x = torch.cat([x, pe], dim=-1)  # (B, N, feat_dim*2 + pos_enc_dim)

        # MLP 
        fused = self.mlp(x)  # (B, N, feat_dim)
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
        
        self.apply(init_weights_kaiming)

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

        # 2) fc2
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)

        # 3) fc3 
        x = torch.cat([x, pe], dim=-1)
        x1 = self.fc3(x)
        x = F.relu(self.ln3(x1), inplace=True)
        
        # 4) fc4
        x = F.relu(self.ln4(self.fc4(x)), inplace=True)

        # 5) fc5 
        out = self.fc5(x)

        if return_intermediate:
            return x1, out
        else:
            return out

class StateProj(nn.Module):
    def __init__(self, state_dim=42, output_dim=120):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, output_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, state):
        out = self.mlp(state)
        return out

class VoxelProj(nn.Module):
    def __init__(self, voxel_feature_dim=120, L=6):
        super().__init__()
        self.L = L
        self.pos_enc_dim = 2 * self.L * 3

        self.mlp = nn.Sequential(
            nn.Linear(voxel_feature_dim + self.pos_enc_dim, voxel_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(voxel_feature_dim),
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, voxel_feat, coords_3d):
        """
        Args:
            voxel_feat: (N, voxel_feature_dim)
            coords_3d:  (N, 3)
        Returns:
            projected:  (N, voxel_feature_dim)
        """
        pe = positional_encoding(coords_3d, L=self.L)  # (N, 2*L*3)
        x = torch.cat([voxel_feat, pe], dim=-1)
        out = self.mlp(x)
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

class ImplicitDecoderDINO(nn.Module):
    def __init__(self, voxel_feature_dim=768, hidden_dim=768, output_dim_clip=768, output_dim_dino=1024, L=10):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.pe_dim = 2 * self.L * 3  # positional encoding dimension

        # voxel_feat + positional_encoding
        self.input_dim = self.voxel_feature_dim + self.pe_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim + self.pe_dim, self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)

        # ----------- CLIP BRANCH -----------
        self.fc4_clip = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4_clip = nn.LayerNorm(self.hidden_dim)
        self.fc5_clip = nn.Linear(self.hidden_dim, output_dim_clip)

        # ----------- DINO BRANCH -----------
        self.fc4_dino = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln4_dino = nn.LayerNorm(self.hidden_dim)
        self.fc5_dino = nn.Linear(self.hidden_dim, output_dim_dino)

    def forward(self, voxel_features, coords_3d):
        """
        Args:
            voxel_features (Tensor): [N, voxel_feature_dim].
            coords_3d (Tensor): [N, 3].
        Returns:
            clip_out: (N, output_dim_clip=768)
            dino_out: (N, output_dim_dino=1024)
        """
        pe = positional_encoding(coords_3d, L=self.L)  # [N, pe_dim]

        # 1) fc1
        x = torch.cat([voxel_features, pe], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)), inplace=True)

        # 2) fc2
        x = F.relu(self.ln2(self.fc2(x)), inplace=True)

        # 3) fc3
        x = torch.cat([x, pe], dim=-1)
        x = F.relu(self.ln3(self.fc3(x)), inplace=True)

        # CLIP branch
        x_clip = F.relu(self.ln4_clip(self.fc4_clip(x)), inplace=True)
        clip_out = self.fc5_clip(x_clip)

        # DINO branch
        x_dino = F.relu(self.ln4_dino(self.fc4_dino(x)), inplace=True)
        dino_out = self.fc5_dino(x_dino)

        return clip_out, dino_out
    
class ConcatMLPFusionDINO(nn.Module):
    """
    (feat_clip, feat_voxel, feat_dino, coords_3d) -> [concat + sinusoidal PE] -> MLP -> fused
    """
    def __init__(self, feat_dim=120, L=10, num_feats=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.L = L
        self.pos_enc_dim = 2 * self.L * 3

        in_dim = feat_dim * num_feats + self.pos_enc_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, feat_voxel, feat_clip, feat_dino, coords_3d):
        """
        Args:
            feat_clip:  (B*N, feat_dim)
            feat_voxel: (B*N, feat_dim)
            feat_dino:  (B*N, feat_dim)
            coords_3d:  (B*N, 3)
        Returns:
            fused: (B*N, feat_dim)
        """
        pe = positional_encoding(coords_3d, L=self.L)  # (B*N, 2*L*3)
        x = torch.cat([feat_voxel, feat_clip, feat_dino, pe], dim=-1)
        fused = self.mlp(x)
        return fused