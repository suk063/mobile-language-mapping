import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lang_mapping.utils.utils import positional_encoding

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

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


class DimReducer(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, x):
        out = self.mlp(x)
        return out
