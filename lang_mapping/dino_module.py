import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import positional_encoding

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