import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import positional_encoding

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
    def __init__(self, feat_dim=768, clip_embedding_dim=768, output_dim=384, L=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.L = L
        
        if L == 0:
            self.pos_enc_dim = 0
        else:
            self.pos_enc_dim = 2 * self.L * 3

        in_dim = feat_dim + clip_embedding_dim + self.pos_enc_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        
        self.output_proj = nn.Linear(feat_dim, output_dim)

    def forward(self, feat1, feat2, coords_3d=None):
        """
        Args:
            feat1: (B * N, feat_dim) 
            feat2: (B * N, clip_embedding_dim)
            coords_3d:    (B * N, 3)  
        
        Returns:
            fused: (B * N, feat_dim) 
        """

        x = torch.cat([feat1, feat2], dim=-1)

        if coords_3d is not None:
            pe = positional_encoding(coords_3d, L=self.L)  # (B*N, 2*L*3)
            x = torch.cat([x, pe], dim=-1)

        # MLP 
        fused = self.mlp(x) + feat1 # (B, N, feat_dim)
        return self.output_proj(fused)
    
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
    def __init__(
        self,
        voxel_feature_dim=768,
        hidden_dim=768,
        output_dim=768,
        L=0,
        pe_type='none'
    ):
        super().__init__()
        self.voxel_feature_dim = voxel_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.L = L
        self.pe_type = pe_type
        
        if self.pe_type == 'sinusoidal':
            self.pe_dim = 2 * self.L * 3  
        elif self.pe_type == 'concat':
            self.pe_dim = 3 
        elif self.pe_type == 'none':
            self.pe_dim = 0
        else:
            raise ValueError(f"Unknown pe_type: {self.pe_type}. Use 'sinusoidal' or 'concat'.")

        self.input_dim = self.voxel_feature_dim + self.pe_dim

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

    def forward(self, voxel_features, coords_3d=None):
        if self.pe_type == 'sinusoidal':
            pe = positional_encoding(coords_3d, L=self.L)  # [N, 2 * L * 3]
        elif self.pe_type == 'concat':
            pe = coords_3d  # [N, 3]
        elif self.pe_type == 'none':
            pe = torch.zeros((voxel_features.shape[0], 0), device=voxel_features.device)  # [N, 0]
        
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

        return out

class StateProj(nn.Module):
    def __init__(self, state_dim=42, output_dim=768):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim)
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, state):
        out = self.mlp(state)
        return out

class VoxelProj(nn.Module):
    def __init__(self, voxel_feature_dim=768):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
            nn.ReLU(),
            nn.LayerNorm(voxel_feature_dim),
            nn.Linear(voxel_feature_dim, voxel_feature_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, voxel_feat):
        """
        Args:
            voxel_feat: (N, voxel_feature_dim)
        Returns:
            projected:  (N, voxel_feature_dim)
        """
        out = self.mlp(voxel_feat)
        return out

class DimReducer(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, L=10):
        super().__init__()
        self.L = L
        self.pos_enc_dim = 2 * self.L * 3

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + self.pos_enc_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, x, coords_3d=None):
        """
        Args:
            voxel_feat: (N, voxel_feature_dim)
            coords_3d:  (N, 3)
        Returns:
            projected:  (N, voxel_feature_dim)
        """
        if coords_3d is not None:
            pe = positional_encoding(coords_3d, L=self.L)  # (N, 2*L*3)
            x = torch.cat([x, pe], dim=-1)
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