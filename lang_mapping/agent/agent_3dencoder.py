import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Local imports
from ..module import *
from ..utils import get_3d_coordinates, get_visual_features, transform

import open_clip

class DP3Encoder(nn.Module):
    """
    DP3 Encoder: 3-layer MLP + LayerNorm + Max-Pooling + Projection
    - Input  : (B, N, in_dim) 
    - Output : (B, out_dim) 
    """
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, N, in_dim)
        return: (B, out_dim)
        """
        feats = self.mlp(pts)                 # (B, N, out_dim)
        feats_pooled = feats.max(dim=1)[0]    # (B, out_dim)
        return feats_pooled

class Agent_3dencoder(nn.Module):
    def __init__(
        self,
        sample_obs,
        single_act_shape,
        state_mlp_dim: int = 1024,
        device: str = "cuda",
        camera_intrinsics: tuple = (71.9144, 71.9144, 112, 112),
    ):
        """
        Maintains a voxel-hash representation for 3D scenes and uses a CLIP-based
        feature extractor plus an implicit decoder for mapping.
        """
        super().__init__()

        self.device = device
        self.epoch = 0

        # Prepare state dimension
        state_obs: torch.Tensor = sample_obs["state"]
        state_dim = state_obs.shape[1]

        # MLP for raw state
        self.state_mlp = nn.Linear(state_dim, state_mlp_dim).to(self.device)
        self.dp3_encoder = DP3Encoder(in_dim=3, hidden_dim=128, out_dim=512).to(self.device)

        # Action MLP
        action_dim = np.prod(single_act_shape)
        self.action_mlp = ActionMLP(
            input_dim=state_mlp_dim * 2,
            action_dim=action_dim
        ).to(self.device)

        # Camera intrinsics
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics

    def forward_policy(self, observations, object_labels, step_nums):
        pixels: Dict[str, torch.Tensor] = observations["pixels"]
        state: torch.Tensor = observations["state"]

        # Depth resizing
        hand_depth = pixels["fetch_hand_depth"] / 1000.0
        head_depth = pixels["fetch_head_depth"] / 1000.0
        if hand_depth.dim() == 5:
            b2, fs2, d2, h2, w2 = hand_depth.shape
            hand_depth = hand_depth.view(b2, fs2 * d2, h2, w2)
            head_depth = head_depth.view(b2, fs2 * d2, h2, w2)
            hand_depth = F.interpolate(hand_depth, (16, 16), mode="nearest")
            head_depth = F.interpolate(head_depth, (16, 16), mode="nearest")
            
        else:
            b2, _, _, _ = hand_depth.shape

        hand_pose = pixels["fetch_hand_pose"]
        head_pose = pixels["fetch_head_pose"]

        hand_visfeat = head_visfeat = torch.randn(b2, 768, 16, 16).cuda()

        # Compute 3D coords
        hand_coords_world, _ = get_3d_coordinates(
            hand_visfeat, hand_depth, hand_pose, self.fx, self.fy, self.cx, self.cy
        )
        head_coords_world, _ = get_3d_coordinates(
            head_visfeat, head_depth, head_pose, self.fx, self.fy, self.cx, self.cy
        )

        B_, C_, Hf, Wf = hand_coords_world.shape
        N = Hf * Wf

        hand_coords_world_flat = hand_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)
        head_coords_world_flat = head_coords_world.permute(0, 2, 3, 1).reshape(B_ * N, 3)

        # Prepare coords for transformer
        batch_hand_coords = hand_coords_world_flat.view(B_, N, 3)
        batch_head_coords = head_coords_world_flat.view(B_, N, 3)

        hand_dp3_feat = self.dp3_encoder(batch_hand_coords) 
        head_dp3_feat = self.dp3_encoder(batch_head_coords) 

        visual_token = torch.cat([hand_dp3_feat, head_dp3_feat], dim=1)

        # Final action
        state_token = self.state_mlp(state)
        inp = torch.cat([visual_token, state_token], dim=1)
        action_pred = self.action_mlp(inp)

        return action_pred
